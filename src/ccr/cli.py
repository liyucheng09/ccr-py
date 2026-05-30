"""CLI entry point for ccr."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import shutil

import click

from .config import (
    load_config,
    get_profile,
    build_proxy_env,
    DirectProfile,
    ProxyProfile,
    CONFIG_FILE,
)


class CCRGroup(click.Group):
    """Route unknown commands to the 'run' subcommand as profile names."""

    def resolve_command(self, ctx, args):
        cmd_name = args[0] if args else None
        if cmd_name and cmd_name not in self.commands and not cmd_name.startswith("-"):
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

    profile = get_profile(profile_name)

    if isinstance(profile, DirectProfile):
        _run_direct(profile, args, use_happy)
    else:
        _run_proxy(profile, args, use_happy)


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
            detail = f"{profile.api_url} → {profile.model}"

        click.echo(f"{name:<{max_name+2}} {profile.type:<{max_type+2}} {detail}")


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
        click.echo("# Proxy profile - environment shown with placeholder port.", err=True)
        click.echo("# Use 'ccr <profile>' to auto-start proxy with real port.", err=True)
        env = build_proxy_env(port=0, max_output_tokens=profile.max_output_tokens, max_context_tokens=profile.max_context_tokens, autocompact_pct=profile.autocompact_pct)
        env["_CCR_NOTE"] = f"proxy_target={profile.api_url}"

    for key, value in sorted(env.items()):
        click.echo(f"export {key}={_shell_quote(value)}")


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


def _run_proxy(profile: ProxyProfile, claude_args: list[str], use_happy: bool = False):
    asyncio.run(_run_proxy_async(profile, claude_args, use_happy))


async def _run_proxy_async(profile: ProxyProfile, claude_args: list[str], use_happy: bool = False):
    from .proxy import run_proxy_until_done

    server, port = await run_proxy_until_done(
        api_url=profile.api_url,
        api_key=profile.api_key,
        model=profile.model,
        port=profile.proxy_port,
        max_output_tokens=profile.max_output_tokens,
    )

    click.echo(f"Proxy started on 127.0.0.1:{port} → {profile.api_url}", err=True)

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
