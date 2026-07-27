"""Structured logging: a shared rich console + a machine-parseable JSONL event log.

``get_logger`` / ``configure_logging`` are the always-on console entry points (unchanged API; now backed by
a rich handler when a TTY is present). For a run, ``attach_run_logging(run_dir)`` adds two file handlers:
``run.log`` (full human-readable) and ``events.jsonl`` (ONE JSON object per line — ts, level, logger, event
and any structured fields) so any tool — incl. ``scripts/monitor.py`` — can tail/parse a run incrementally
and catch anything that is even slightly off. ``log_event`` emits a structured event to console + both files.
The module-level :data:`CONSOLE` is shared with ``src.utils.monitoring`` so live progress bars and log lines
interleave cleanly on one rich console.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

__all__ = ["CONSOLE", "configure_logging", "get_logger", "attach_run_logging", "log_event"]

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_configured = False
_run_attached: set[str] = set()

#: Shared rich Console (stderr) — None until first use; used by the log handler AND the live progress
#: display so bars + log lines never trample each other. Falls back to plain stderr when rich is absent.
try:
    from rich.console import Console

    CONSOLE: Any = Console(stderr=True)
except Exception:  # pragma: no cover - rich is a declared dep; degrade to plain logging
    CONSOLE = None


def _jsonable(x: Any) -> Any:
    """Best-effort JSON coercion (numpy scalars/arrays, paths) so logging never crashes a run."""
    try:
        import numpy as np

        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
    except Exception:  # pragma: no cover
        pass
    return str(x)


class _JsonlFormatter(logging.Formatter):
    """One JSON object per line: ts, level, logger, event, msg + any structured ``fields``."""

    def format(self, record: logging.LogRecord) -> str:
        obj: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", None),
            "msg": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            obj.update(fields)
        if record.exc_info:
            obj["exc"] = self.formatException(record.exc_info)
        return json.dumps(obj, default=_jsonable)


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure root logging once (idempotent). Rich console when a TTY is present, else plain stderr."""
    global _configured
    if _configured:
        logging.getLogger().setLevel(level)
        return
    handler: logging.Handler
    try:
        if CONSOLE is not None and sys.stderr.isatty():
            from rich.logging import RichHandler

            handler = RichHandler(console=CONSOLE, rich_tracebacks=True, show_path=False, markup=False)
            handler.setFormatter(logging.Formatter("%(name)s | %(message)s", datefmt="%H:%M:%S"))
        else:
            raise ImportError
    except Exception:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    _configured = True


def attach_run_logging(run_dir: str | Path, level: int = logging.INFO) -> dict[str, Path]:
    """Add per-run file handlers — ``run.log`` (human) + ``events.jsonl`` (structured). Idempotent per dir."""
    if not _configured:
        configure_logging()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = {"log": run_dir / "run.log", "events": run_dir / "events.jsonl"}
    key = str(run_dir.resolve())
    if key in _run_attached:
        return paths
    root = logging.getLogger()
    txt = logging.FileHandler(paths["log"], encoding="utf-8")
    txt.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    txt.setLevel(level)
    root.addHandler(txt)
    jsonl = logging.FileHandler(paths["events"], encoding="utf-8")
    jsonl.setFormatter(_JsonlFormatter())
    jsonl.setLevel(level)
    root.addHandler(jsonl)
    _run_attached.add(key)
    return paths


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring logging on first use."""
    if not _configured:
        configure_logging()
    return logging.getLogger(name)


def _short(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    s = str(v)
    return s if len(s) <= 80 else s[:77] + "..."


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    """Emit a structured event: a readable console/text line + a JSONL record carrying ``fields``.

    e.g. ``log_event(log, "candidate_done", arm="distributional", cand=3, fitness=0.012, secs=131.4)``.
    """
    parts = " ".join(f"{k}={_short(v)}" for k, v in fields.items())
    logger.log(level, "%s %s", event, parts, extra={"event": event, "fields": fields})
