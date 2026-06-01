"""Manage persistent proxy server state (PID, port files)."""

from __future__ import annotations

import os
import signal
from pathlib import Path

SERVERS_DIR = Path.home() / ".ccr" / "servers"


def _ensure_dir() -> None:
    SERVERS_DIR.mkdir(parents=True, exist_ok=True)


def save_server_info(profile: str, pid: int, port: int) -> None:
    _ensure_dir()
    (SERVERS_DIR / f"{profile}.pid").write_text(str(pid))
    (SERVERS_DIR / f"{profile}.port").write_text(str(port))


def read_server_info(profile: str) -> tuple[int, int] | None:
    pid_path = SERVERS_DIR / f"{profile}.pid"
    port_path = SERVERS_DIR / f"{profile}.port"
    if not pid_path.exists() or not port_path.exists():
        return None
    try:
        pid = int(pid_path.read_text().strip())
        port = int(port_path.read_text().strip())
        return pid, port
    except (ValueError, OSError):
        return None


def remove_server_info(profile: str) -> None:
    for suffix in (".pid", ".port"):
        p = SERVERS_DIR / f"{profile}{suffix}"
        try:
            p.unlink()
        except OSError:
            pass


def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def is_server_running(profile: str) -> bool:
    info = read_server_info(profile)
    if info is None:
        return False
    pid, _ = info
    alive = is_pid_alive(pid)
    if not alive:
        remove_server_info(profile)
    return alive


def get_server_port(profile: str) -> int | None:
    """Return port if server is running, else None."""
    info = read_server_info(profile)
    if info is None:
        return None
    pid, port = info
    if not is_pid_alive(pid):
        remove_server_info(profile)
        return None
    return port


def stop_server(profile: str) -> bool:
    """Send SIGTERM to the server. Returns True if process was found."""
    info = read_server_info(profile)
    if info is None:
        return False
    pid, _ = info
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    remove_server_info(profile)
    return True


def list_running_servers() -> dict[str, tuple[int, int]]:
    """Return {profile: (pid, port)} for all running servers."""
    if not SERVERS_DIR.exists():
        return {}
    result: dict[str, tuple[int, int]] = {}
    for pid_file in SERVERS_DIR.glob("*.pid"):
        profile = pid_file.stem
        info = read_server_info(profile)
        if info is None:
            continue
        pid, port = info
        if is_pid_alive(pid):
            result[profile] = (pid, port)
        else:
            remove_server_info(profile)
    return result
