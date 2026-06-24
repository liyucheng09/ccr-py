"""Long-lived debug recording for ccr proxies.

When enabled via `--debug`, every request gets its own directory under
``~/.ccr/debug/<profile>/`` capturing the full raw byte stream so a hung
stream (e.g. an upstream that stops emitting mid-reasoning) can be diagnosed
after the fact: the per-chunk timestamps in ``upstream.stream`` show exactly
when/where the upstream went silent.

Non-debug mode is zero-cost: :func:`make_recorder` returns ``None`` and the
hot path in the proxies short-circuits on a ``None`` guard.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEBUG_ROOT = Path.home() / ".ccr" / "debug"
DEFAULT_KEEP = 100


@dataclass
class DebugTarget:
    """Which proxy paths should be recorded."""
    codex: bool = False
    anthropic: bool = False

    @classmethod
    def parse(cls, value: str | None) -> "DebugTarget | None":
        """Parse a --debug argument. ``None``/empty → both. Unknown → None (off)."""
        if value is None or value == "" or value == "all":
            return cls(codex=True, anthropic=True)
        if value == "codex":
            return cls(codex=True, anthropic=False)
        if value == "anthropic":
            return cls(codex=False, anthropic=True)
        return None


class RequestRecorder:
    """Records one request's raw bytes + key events to a per-request directory.

    Not thread-safe; intended to be used within a single asyncio request task.
    Writes are synchronous and small-per-chunk; this is debug-only so we
    prioritize diagnosability over throughput.
    """

    def __init__(
        self,
        base_dir: Path,
        profile: str,
        path_tag: str,
        keep: int,
        target_kind: str,
    ):
        self._base_dir = base_dir
        self._profile = profile
        self._path_tag = path_tag
        self._keep = keep
        self._target_kind = target_kind
        self._dir: Path | None = None
        self._stream_file = None
        self._events_file = None
        self._start_mono = time.monotonic()
        self._start_wall = time.time()
        self._seq = 0
        self._closed = False

    def start(self) -> None:
        profile_dir = self._base_dir / self._profile
        profile_dir.mkdir(parents=True, exist_ok=True)
        self._seq = self._next_seq(profile_dir)
        ts = int(self._start_wall)
        name = f"{self._seq:04d}_{ts}_{self._path_tag}"
        self._dir = profile_dir / name
        self._dir.mkdir(parents=True, exist_ok=True)
        self._stream_file = open(self._dir / "upstream.stream", "ab", buffering=0)
        self._events_file = open(self._dir / "events.jsonl", "a", buffering=1)
        self._write_meta()
        self._rotate(profile_dir)
        self.event("recorder_start")

    def _next_seq(self, profile_dir: Path) -> int:
        existing = [int(p.name.split("_", 1)[0]) for p in profile_dir.iterdir()
                    if p.is_dir() and p.name[:4].isdigit()]
        return (max(existing) + 1) if existing else 1

    def _write_meta(self) -> None:
        assert self._dir is not None
        meta = {
            "profile": self._profile,
            "target_kind": self._target_kind,
            "path_tag": self._path_tag,
            "start_wall_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self._start_wall)),
            "start_wall_epoch": self._start_wall,
            "seq": self._seq,
        }
        (self._dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    def _rotate(self, profile_dir: Path) -> None:
        """Keep only the most recent ``keep`` request directories."""
        if self._keep <= 0:
            return
        dirs = sorted(
            (p for p in profile_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()),
            key=lambda p: int(p.name.split("_", 1)[0]),
        )
        excess = len(dirs) - self._keep
        for d in dirs[:max(0, excess)]:
            _rm_tree(d)

    def _t_ms(self) -> int:
        return int((time.monotonic() - self._start_mono) * 1000)

    def log_request_body(self, body: Any) -> None:
        """Write the client request body as request.json."""
        if self._dir is None:
            return
        try:
            text = json.dumps(body, ensure_ascii=False, indent=2, default=str)
        except (TypeError, ValueError):
            text = str(body)
        (self._dir / "request.json").write_text(text, encoding="utf-8")
        self.event("request_logged")

    def log_request_raw(self, data: bytes) -> None:
        """Write raw (non-JSON) request bytes as request.raw."""
        if self._dir is None:
            return
        (self._dir / "request.raw").write_bytes(data)
        self.event("request_logged")

    def log_upstream_chunk(self, data: bytes) -> None:
        """Append an upstream response chunk with a relative-timestamp prefix.

        Format: ``[+1234ms len=4096]\n<raw bytes>`` — the offset is monotonic
        milliseconds since recorder start, so a stall shows up as a long gap
        with no further chunks.
        """
        if self._stream_file is None or self._closed:
            return
        header = f"[+{self._t_ms()}ms len={len(data)}]\n".encode("ascii")
        try:
            self._stream_file.write(header)
            self._stream_file.write(data)
        except OSError:
            pass

    def event(self, name: str, **fields: Any) -> None:
        """Append a structured event to events.jsonl."""
        if self._events_file is None or self._closed:
            return
        entry = {"t_ms": self._t_ms(), "event": name, **fields}
        try:
            self._events_file.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass

    def close(self, status: str, **extra: Any) -> None:
        if self._closed:
            return
        # Write the close event BEFORE flipping _closed, since event() short-
        # circuits on _closed.
        self.event("close", status=status, **extra)
        self._closed = True
        for f in (self._stream_file, self._events_file):
            if f is not None:
                try:
                    f.flush()
                    f.close()
                except OSError:
                    pass
        self._stream_file = None
        self._events_file = None


def _rm_tree(path: Path) -> None:
    try:
        for child in path.iterdir():
            if child.is_dir():
                _rm_tree(child)
            else:
                child.unlink()
        path.rmdir()
    except OSError:
        pass


def make_recorder(
    debug: DebugTarget | None,
    profile: str,
    path_tag: str,
    keep: int,
    target_kind: str,
) -> RequestRecorder | None:
    """Build a recorder for ``target_kind`` (``'codex'`` or ``'anthropic'``).

    Returns ``None`` when debug is off or this target isn't enabled, so callers
    can write ``if rec := make_recorder(...): rec.log_upstream_chunk(...)`` and
    pay nothing on the hot path when debugging is disabled.
    """
    if debug is None:
        return None
    if target_kind == "codex" and not debug.codex:
        return None
    if target_kind == "anthropic" and not debug.anthropic:
        return None
    rec = RequestRecorder(DEBUG_ROOT, profile, path_tag, keep, target_kind)
    rec.start()
    return rec
