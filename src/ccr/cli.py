"""CLI entry point for ccr."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path
import shutil
import subprocess
import time

import click

from .config import (
    load_config,
    get_profile,
    build_proxy_env,
    DirectProfile,
    ProxyProfile,
    CONFIG_FILE,
    CONFIG_DIR,
)
from .server_state import (
    save_server_info,
    is_server_running,
    get_server_port,
    stop_server,
    list_running_servers,
)
from .debug_log import DebugTarget, DEFAULT_KEEP


def _strip_debug_flags(args: list[str]) -> tuple[DebugTarget | None, int, list[str]]:
    """Extract ccr --debug / --debug-keep from a raw arg list (run subcommand).

    These are ccr flags, not claude args, so they must be consumed before the
    remaining args are forwarded to claude. Mirrors the existing --happy handling.
    Returns (debug_target, keep, remaining_args).
    """
    remaining: list[str] = []
    debug_value: str | None = None
    keep = DEFAULT_KEEP
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--debug":
            debug_value = "all"
            i += 1
            continue
        if a.startswith("--debug="):
            debug_value = a.split("=", 1)[1]
            i += 1
            continue
        if a == "--debug-keep" and i + 1 < len(args):
            try:
                keep = int(args[i + 1])
            except ValueError:
                pass
            i += 2
            continue
        if a.startswith("--debug-keep="):
            try:
                keep = int(a.split("=", 1)[1])
            except ValueError:
                pass
            i += 1
            continue
        remaining.append(a)
        i += 1
    return DebugTarget.parse(debug_value), keep, remaining


class CCRGroup(click.Group):
    """Route unknown commands to the 'run' subcommand as profile names."""

    def resolve_command(self, ctx, args):
        cmd_name = args[0] if args else None
        if cmd_name and cmd_name not in self.commands and not cmd_name.startswith("-") and not cmd_name.startswith("_"):
            return "run", self.commands["run"], args
        return super().resolve_command(ctx, args)

    def format_usage(self, ctx, formatter):
        formatter.write("Usage: ccr <profile> [claude-args...]\n")
        formatter.write("       ccr <command> [options]\n")


@click.group(cls=CCRGroup)
def main():
    """CCR - Claude Code Router. Switch providers and proxy local models."""


@main.command(hidden=True, context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.argument("profile_name")
@click.argument("claude_args", nargs=-1, type=click.UNPROCESSED)
def run(profile_name: str, claude_args: tuple[str, ...]):
    """Launch claude with a profile."""
    args = list(claude_args)
    use_happy = "--happy" in args
    if use_happy:
        args.remove("--happy")

    debug, debug_keep, args = _strip_debug_flags(args)

    profile = get_profile(profile_name)

    if isinstance(profile, DirectProfile):
        _run_direct(profile, args, use_happy)
    else:
        _run_proxy(profile, args, use_happy, debug=debug, debug_keep=debug_keep)


@main.command()
@click.argument("profile_name")
@click.option("--debug", "debug", default=None,
              type=click.Choice(["codex", "anthropic", "all"], case_sensitive=False),
              is_flag=False, flag_value="all",
              help="Record all raw request/upstream bytes to ~/.ccr/debug/<profile>/. "
                   "Optional value: codex, anthropic, or all (default).")
@click.option("--debug-keep", "debug_keep", default=DEFAULT_KEEP, show_default=True, type=int,
              help="Number of recent request recordings to keep per profile.")
def start(profile_name: str, debug: str | None, debug_keep: int):
    """Start a persistent proxy server for a profile.

    Only works with proxy-type profiles (e.g. glm5).
    The server runs in the background until 'ccr stop' is called.
    """
    profile = get_profile(profile_name)

    if isinstance(profile, DirectProfile):
        click.echo(f"Profile '{profile_name}' is a direct profile and does not need a server.", err=True)
        sys.exit(1)

    if is_server_running(profile_name):
        port = get_server_port(profile_name)
        click.echo(f"Proxy server for '{profile_name}' is already running on port {port}.")
        return

    # Spawn background process: ccr serve <profile> [--debug ...]
    ccr_path = shutil.which("ccr")
    if not ccr_path:
        ccr_path = sys.executable.replace("python", "ccr")

    serve_argv = [ccr_path, "serve", profile_name]
    if debug is not None:
        serve_argv += ["--debug", debug, "--debug-keep", str(debug_keep)]

    # Capture stderr to surface startup errors (e.g. port in use)
    stderr_pipe = subprocess.PIPE
    proc = subprocess.Popen(
        serve_argv,
        stdout=subprocess.DEVNULL,
        stderr=stderr_pipe,
        start_new_session=True,
    )

    # Wait for server to become ready
    import urllib.request
    import urllib.error

    for _ in range(50):  # up to ~5 seconds
        time.sleep(0.1)
        port = get_server_port(profile_name)
        if port is None:
            # Check if the process died
            if proc.poll() is not None:
                stderr_out = proc.stderr.read().decode(errors="replace").strip() if proc.stderr else ""
                msg = f"Failed to start proxy server for '{profile_name}'."
                if stderr_out:
                    msg += f"\n{stderr_out}"
                click.echo(msg, err=True)
                sys.exit(1)
            continue
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            if profile.codex_port:
                click.echo(
                    f"Proxy server for '{profile_name}' started: "
                    f"anthropic on 127.0.0.1:{port}, codex on 127.0.0.1:{profile.codex_port} -> {profile.api_url}"
                )
            else:
                click.echo(f"Proxy server for '{profile_name}' started on 127.0.0.1:{port} -> {profile.api_url}")
            if debug is not None:
                click.echo(f"  debug: recording {debug} -> ~/.ccr/debug/{profile_name}/ (keep {debug_keep})")
            return
        except (urllib.error.URLError, OSError):
            continue

    click.echo(f"Timeout waiting for proxy server for '{profile_name}' to start.", err=True)
    sys.exit(1)


@main.command()
@click.argument("profile_name", required=False)
def stop(profile_name: str | None):
    """Stop a persistent proxy server.

    If no profile is specified, stops all running servers.
    """
    if profile_name:
        if not is_server_running(profile_name):
            click.echo(f"No running server for '{profile_name}'.")
            return
        stop_server(profile_name)
        click.echo(f"Proxy server for '{profile_name}' stopped.")
    else:
        servers = list_running_servers()
        if not servers:
            click.echo("No running proxy servers.")
            return
        for name in servers:
            stop_server(name)
            click.echo(f"Proxy server for '{name}' stopped.")


@main.command()
@click.argument("profile_name")
def restart(profile_name: str):
    """Restart a persistent proxy server."""
    profile = get_profile(profile_name)

    if isinstance(profile, DirectProfile):
        click.echo(f"Profile '{profile_name}' is a direct profile and does not need a server.", err=True)
        sys.exit(1)

    if is_server_running(profile_name):
        stop_server(profile_name)
        click.echo(f"Proxy server for '{profile_name}' stopped.")

    # Reuse start logic
    ctx = click.get_current_context()
    ctx.invoke(start, profile_name=profile_name)


@main.command()
def status():
    """Show status of persistent proxy servers."""
    servers = list_running_servers()
    if not servers:
        click.echo("No running proxy servers.")
        return

    profiles = load_config()

    max_name = max(len(n) for n in servers)
    click.echo(f"{'PROFILE':<{max_name+2}} {'PID':<8} {'PORT':<8} UPSTREAM")
    click.echo(f"{'─'*(max_name+2)} {'─'*8} {'─'*8} {'─'*40}")

    for name, (pid, port) in sorted(servers.items()):
        profile = profiles.get(name)
        upstream = profile.api_url if isinstance(profile, ProxyProfile) else "?"
        click.echo(f"{name:<{max_name+2}} {pid:<8} {port:<8} {upstream}")


@main.command("list")
def list_profiles():
    """List all configured profiles."""
    profiles = load_config()
    if not profiles:
        click.echo("No profiles configured.")
        return

    max_name = max(len(n) for n in profiles)
    max_type = max(len(p.type) for p in profiles.values())

    click.echo(f"{'PROFILE':<{max_name+2}} {'TYPE':<{max_type+2}} DETAIL")
    click.echo(f"{'─'*(max_name+2)} {'─'*(max_type+2)} {'─'*40}")

    for name, profile in profiles.items():
        if isinstance(profile, DirectProfile):
            base_url = profile.env.get("ANTHROPIC_BASE_URL", "")
            model = profile.env.get("ANTHROPIC_MODEL", "")
            detail = model if model else base_url
        else:
            detail = f"{profile.api_url} -> {profile.model}"

        click.echo(f"{name:<{max_name+2}} {profile.type:<{max_type+2}} {detail}")


@main.command("serve", hidden=True)
@click.argument("profile_name")
@click.option("--debug", "debug", default=None,
              type=click.Choice(["codex", "anthropic", "all"], case_sensitive=False),
              is_flag=False, flag_value="all",
              help="Record raw request/upstream bytes to ~/.ccr/debug/<profile>/.")
@click.option("--debug-keep", "debug_keep", default=DEFAULT_KEEP, show_default=True, type=int,
              help="Number of recent request recordings to keep per profile.")
def serve(profile_name: str, debug: str | None, debug_keep: int):
    """Internal: run a persistent proxy server. Called by 'ccr start'."""
    profile = get_profile(profile_name)

    if isinstance(profile, DirectProfile):
        click.echo(f"Cannot serve a direct profile.", err=True)
        sys.exit(1)

    debug_target = DebugTarget.parse(debug)
    asyncio.run(_serve_async(profile_name, profile, debug=debug_target, debug_keep=debug_keep))


@main.command()
@click.argument("profile_name")
def activate(profile_name: str):
    """Print environment variables for a profile.

    Usage: eval "$(ccr activate opus46)"
    """
    profile = get_profile(profile_name)

    if isinstance(profile, DirectProfile):
        env = dict(profile.env)
    else:
        # Check if a persistent server is running
        port = get_server_port(profile_name)
        if port is not None:
            env = build_proxy_env(port, max_output_tokens=profile.max_output_tokens, max_context_tokens=profile.max_context_tokens, autocompact_pct=profile.autocompact_pct)
        else:
            click.echo("# Proxy profile - no persistent server running.", err=True)
            click.echo("# Use 'ccr start <profile>' first, or 'ccr <profile>' for ephemeral proxy.", err=True)
            env = build_proxy_env(port=0, max_output_tokens=profile.max_output_tokens, max_context_tokens=profile.max_context_tokens, autocompact_pct=profile.autocompact_pct)
            env["_CCR_NOTE"] = f"proxy_target={profile.api_url}"

    for key, value in sorted(env.items()):
        click.echo(f"export {key}={_shell_quote(value)}")



@main.command("remote-sync")
@click.argument("ssh_server")
@click.option("--source", "source_file", default=None, help="Local config file to sync (default: ~/.ccr/config.yaml.ssh)")
@click.option("--dest", "dest_path", default=None, help="Remote destination path (default: ~/.ccr/config.yaml)")
def remote_sync(ssh_server: str, source_file: str | None, dest_path: str | None):
    """Copy local SSH config to a remote server via scp.

    By default syncs ~/.ccr/config.yaml.ssh to <ssh_server>:~/.ccr/config.yaml.

    Example: ccr remote-sync sg_g8
    """
    src = Path(source_file) if source_file else CONFIG_DIR / "config.yaml.ssh"
    dst = dest_path or "~/.ccr/config.yaml"

    if not src.exists():
        click.echo(f"Source file not found: {src}", err=True)
        sys.exit(1)

    # Ensure remote ~/.ccr directory exists
    mkdir_cmd = ["ssh", ssh_server, "mkdir", "-p", "~/.ccr"]
    click.echo(f"Ensuring remote directory exists on {ssh_server}...")
    ret = subprocess.call(mkdir_cmd)
    if ret != 0:
        click.echo(f"Failed to create remote directory on {ssh_server} (exit {ret}).", err=True)
        sys.exit(1)

    # scp the file
    scp_cmd = ["scp", str(src), f"{ssh_server}:{dst}"]
    click.echo(f"Syncing {src} -> {ssh_server}:{dst} ...")
    ret = subprocess.call(scp_cmd)
    if ret != 0:
        click.echo(f"scp failed (exit {ret}).", err=True)
        sys.exit(1)

    click.echo(f"Done. {src} synced to {ssh_server}:{dst}")


async def _serve_async(
    profile_name: str,
    profile: ProxyProfile,
    debug: DebugTarget | None = None,
    debug_keep: int = DEFAULT_KEEP,
):
    from .proxy import ProxyServer
    from .codex_proxy import CodexProxyServer

    server = ProxyServer(
        api_url=profile.api_url,
        api_key=profile.api_key,
        model=profile.model,
        port=profile.proxy_port,
        max_output_tokens=profile.max_output_tokens,
        debug=debug, debug_keep=debug_keep, profile_name=profile_name,
    )
    actual_port = await server.start()

    codex_server: CodexProxyServer | None = None
    codex_port: int | None = None
    if profile.codex_port:
        codex_server = CodexProxyServer(
            api_url=profile.api_url,
            api_key=profile.api_key,
            model=profile.model,
            port=profile.codex_port,
            debug=debug, debug_keep=debug_keep, profile_name=profile_name,
        )
        codex_port = await codex_server.start()

    save_server_info(profile_name, os.getpid(), actual_port)

    if codex_port:
        click.echo(
            f"Serving '{profile_name}' on 127.0.0.1:{actual_port} (anthropic) "
            f"+ 127.0.0.1:{codex_port} (codex) -> {profile.api_url}",
            err=True,
        )
    else:
        click.echo(f"Serving '{profile_name}' on 127.0.0.1:{actual_port} -> {profile.api_url}", err=True)
    if debug is not None:
        targets = []
        if debug.codex:
            targets.append("codex")
        if debug.anthropic:
            targets.append("anthropic")
        click.echo(f"  debug: recording {'+'.join(targets)} -> ~/.ccr/debug/{profile_name}/ (keep {debug_keep})", err=True)

    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    def _sig_handler():
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _sig_handler)

    try:
        await stop_event.wait()
    finally:
        from .server_state import remove_server_info
        await server.stop()
        if codex_server is not None:
            await codex_server.stop()
        remove_server_info(profile_name)
        click.echo(f"Proxy server for '{profile_name}' stopped.", err=True)


def _run_direct(profile: DirectProfile, claude_args: list[str], use_happy: bool = False):
    env = {**os.environ, **profile.env}
    if use_happy:
        happy_path = _find_happy()
        args = [happy_path] + claude_args
        os.execvpe(happy_path, args, env)
    else:
        claude_path = _find_claude()
        args = [claude_path] + claude_args
        os.execvpe(claude_path, args, env)


def _run_proxy(
    profile: ProxyProfile, claude_args: list[str], use_happy: bool = False,
    debug: DebugTarget | None = None, debug_keep: int = DEFAULT_KEEP,
):
    # Check if a persistent server is already running for this profile
    existing_port = get_server_port(profile.name)
    if existing_port is not None:
        if debug is not None:
            click.echo(
                f"Note: reusing already-running '{profile.name}' server; --debug only applies to a "
                f"freshly-started server. Run `ccr stop {profile.name}` then `ccr {profile.name} --debug` "
                f"to enable recording.",
                err=True,
            )
        _run_proxy_with_port(profile, existing_port, claude_args, use_happy, persistent=True)
    else:
        # Start ephemeral proxy (original behavior)
        asyncio.run(_run_proxy_async(profile, claude_args, use_happy, debug=debug, debug_keep=debug_keep))


def _run_proxy_with_port(
    profile: ProxyProfile,
    port: int,
    claude_args: list[str],
    use_happy: bool = False,
    persistent: bool = False,
):
    """Launch claude connecting to an already-running proxy on the given port."""
    proxy_env = build_proxy_env(port, max_output_tokens=profile.max_output_tokens, max_context_tokens=profile.max_context_tokens, autocompact_pct=profile.autocompact_pct)
    env = {**os.environ, **proxy_env}

    if use_happy:
        exe = _find_happy()
        args = [exe] + claude_args
    else:
        exe = _find_claude()
        args = [exe] + claude_args

    os.execvpe(exe, args, env)


async def _run_proxy_async(
    profile: ProxyProfile, claude_args: list[str], use_happy: bool = False,
    debug: DebugTarget | None = None, debug_keep: int = DEFAULT_KEEP,
):
    from .proxy import run_proxy_until_done
    from .codex_proxy import run_codex_proxy_until_done
    from .server_state import save_server_info, remove_server_info

    server, port = await run_proxy_until_done(
        api_url=profile.api_url,
        api_key=profile.api_key,
        model=profile.model,
        port=profile.proxy_port,
        max_output_tokens=profile.max_output_tokens,
        debug=debug, debug_keep=debug_keep, profile_name=profile.name,
    )

    codex_server = None
    if profile.codex_port:
        codex_server, _ = await run_codex_proxy_until_done(
            api_url=profile.api_url,
            api_key=profile.api_key,
            model=profile.model,
            port=profile.codex_port,
            debug=debug, debug_keep=debug_keep, profile_name=profile.name,
        )

    # Register so future `ccr <profile>` can reuse this server
    save_server_info(profile.name, os.getpid(), port)

    click.echo(f"Proxy started on 127.0.0.1:{port} -> {profile.api_url}", err=True)
    if debug is not None:
        targets = []
        if debug.codex:
            targets.append("codex")
        if debug.anthropic:
            targets.append("anthropic")
        click.echo(f"  debug: recording {'+'.join(targets)} -> ~/.ccr/debug/{profile.name}/ (keep {debug_keep})", err=True)

    proxy_env = build_proxy_env(port, max_output_tokens=profile.max_output_tokens, max_context_tokens=profile.max_context_tokens, autocompact_pct=profile.autocompact_pct)
    env = {**os.environ, **proxy_env}

    if use_happy:
        happy_path = _find_happy()
        args = [happy_path] + claude_args
    else:
        claude_path = _find_claude()
        args = [claude_path] + claude_args

    proc = await asyncio.create_subprocess_exec(*args, env=env)

    try:
        await proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        await proc.wait()
    finally:
        await server.stop()
        if codex_server is not None:
            await codex_server.stop()
        remove_server_info(profile.name)
        click.echo("Proxy stopped.", err=True)

    sys.exit(proc.returncode or 0)


def _find_claude() -> str:
    path = os.environ.get("CLAUDE_PATH") or shutil.which("claude")
    if not path:
        click.echo("Claude Code not found. Install: npm install -g @anthropic-ai/claude-code", err=True)
        sys.exit(1)
    return path


def _find_happy() -> str:
    path = os.environ.get("HAPPY_PATH") or shutil.which("happy")
    if not path:
        click.echo("Happy CLI not found. Install: npm install -g happy", err=True)
        sys.exit(1)
    return path


def _shell_quote(value: str) -> str:
    if not value or any(c in value for c in " \t\n'\"$`\\!#&|;(){}[]<>?*~"):
        return "'" + value.replace("'", "'\\''") + "'"
    return value
