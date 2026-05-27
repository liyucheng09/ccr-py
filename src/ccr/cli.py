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
    profile = get_profile(profile_name)

    if isinstance(profile, DirectProfile):
        _run_direct(profile, list(claude_args))
    else:
        _run_proxy(profile, list(claude_args))


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
        env = build_proxy_env(port=0)
        env["_CCR_NOTE"] = f"proxy_target={profile.api_url}"

    for key, value in sorted(env.items()):
        click.echo(f"export {key}={_shell_quote(value)}")


def _run_direct(profile: DirectProfile, claude_args: list[str]):
    claude_path = _find_claude()
    env = {**os.environ, **profile.env}
    args = [claude_path] + claude_args
    os.execvpe(claude_path, args, env)


def _run_proxy(profile: ProxyProfile, claude_args: list[str]):
    asyncio.run(_run_proxy_async(profile, claude_args))


async def _run_proxy_async(profile: ProxyProfile, claude_args: list[str]):
    from .proxy import run_proxy_until_done

    claude_path = _find_claude()

    server, port = await run_proxy_until_done(
        api_url=profile.api_url,
        api_key=profile.api_key,
        model=profile.model,
        port=profile.proxy_port,
    )

    click.echo(f"Proxy started on 127.0.0.1:{port} → {profile.api_url}", err=True)

    proxy_env = build_proxy_env(port)
    env = {**os.environ, **proxy_env}
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


def _shell_quote(value: str) -> str:
    if not value or any(c in value for c in " \t\n'\"$`\\!#&|;(){}[]<>?*~"):
        return "'" + value.replace("'", "'\\''") + "'"
    return value
