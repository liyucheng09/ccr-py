from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path.home() / ".ccr"
CONFIG_FILE = CONFIG_DIR / "config.yaml"


@dataclass
class DirectProfile:
    name: str
    env: dict[str, str] = field(default_factory=dict)

    @property
    def type(self) -> str:
        return "direct"


@dataclass
class ProxyProfile:
    name: str
    api_url: str
    api_key: str = "dummy"
    model: str = ""
    proxy_port: int = 0
    max_output_tokens: int | None = None
    max_context_tokens: int | None = None
    autocompact_pct: int | None = None

    @property
    def type(self) -> str:
        return "proxy"


Profile = DirectProfile | ProxyProfile


def _parse_profile(name: str, data: dict[str, Any]) -> Profile:
    ptype = data.get("type", "direct")
    if ptype == "direct":
        return DirectProfile(name=name, env=dict(data.get("env", {})))
    if ptype == "proxy":
        raw_max_out = data.get("max_output_tokens")
        raw_max_ctx = data.get("max_context_tokens")
        raw_ac_pct = data.get("autocompact_pct")
        return ProxyProfile(
            name=name,
            api_url=data["api_url"],
            api_key=data.get("api_key", "dummy"),
            model=data.get("model", ""),
            proxy_port=data.get("proxy_port", 0),
            max_output_tokens=int(raw_max_out) if raw_max_out is not None else None,
            max_context_tokens=int(raw_max_ctx) if raw_max_ctx is not None else None,
            autocompact_pct=int(raw_ac_pct) if raw_ac_pct is not None else None,
        )
    print(f"Unknown profile type '{ptype}' for '{name}'", file=sys.stderr)
    sys.exit(1)


def load_config() -> dict[str, Profile]:
    if not CONFIG_FILE.exists():
        print(f"Config not found: {CONFIG_FILE}", file=sys.stderr)
        print(f"Create it with your profiles. See: ccr --help", file=sys.stderr)
        sys.exit(1)

    with open(CONFIG_FILE) as f:
        raw = yaml.safe_load(f)

    profiles_raw = raw.get("profiles", {})
    if not profiles_raw:
        print(f"No profiles defined in {CONFIG_FILE}", file=sys.stderr)
        sys.exit(1)

    profiles: dict[str, Profile] = {}
    for name, data in profiles_raw.items():
        profiles[name] = _parse_profile(name, data)

    return profiles


def get_profile(name: str) -> Profile:
    profiles = load_config()
    if name not in profiles:
        print(f"Profile '{name}' not found. Available: {', '.join(profiles)}", file=sys.stderr)
        sys.exit(1)
    return profiles[name]


def env_for_profile(profile: Profile) -> dict[str, str]:
    if isinstance(profile, DirectProfile):
        return dict(profile.env)

    # ProxyProfile: will be filled in by cli.py after proxy starts
    raise RuntimeError("Call build_proxy_env() for proxy profiles")


def build_proxy_env(port: int, max_output_tokens: int | None = None, max_context_tokens: int | None = None, autocompact_pct: int | None = None) -> dict[str, str]:
    env = {
        "ANTHROPIC_AUTH_TOKEN": "ccr-proxy",
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
        "NO_PROXY": "127.0.0.1",
        "DISABLE_TELEMETRY": "true",
        "DISABLE_COST_WARNINGS": "true",
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    }
    if max_output_tokens is not None:
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(max_output_tokens)
    if max_context_tokens is not None:
        env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(max_context_tokens)
    if autocompact_pct is not None:
        env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] = str(autocompact_pct)
    return env
