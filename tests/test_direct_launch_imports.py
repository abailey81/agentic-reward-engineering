"""Regression guard for the `from scripts.X` import-bug CLASS (audit 2026-07-24).

`from scripts.X import Y` raises ModuleNotFoundError("No module named 'scripts'") when a script is
launched DIRECTLY (`python scripts/foo.py`): Python puts scripts/ (NOT the repo root) on sys.path[0],
so `scripts` is not an importable package. It resolves under pytest / `-m` / a Myriad `-m` node run
(repo root on sys.path) — which is exactly why the whole class is INVISIBLE to the normal test suite
and bit six sites this session (seal write/verify, capture_env x2, cost_sweep, cluster summary).

The ONLY faithful guard is a SUBPROCESS launch with the repo root kept OFF PYTHONPATH, reproducing a
real `python scripts/foo.py`. `src` is an installed package (pyproject `packages=["src"]`) so `from
src.X` still resolves; only `from scripts.X` would fail on the unfixed code.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _direct_launch(*args: str, timeout: int = 180) -> str:
    """Run `python scripts/<args>` as a real direct launch (repo root NOT on PYTHONPATH); return
    combined stdout+stderr. The script may fail for OTHER reasons (empty archive) — we assert only
    that the import-bug class did not fire."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)  # faithful direct launch: only the script's dir on sys.path[0]
    proc = subprocess.run(
        [sys.executable, str(_REPO / "scripts" / args[0]), *args[1:]],
        capture_output=True, text=True, cwd=str(_REPO), env=env, timeout=timeout,
    )
    return proc.stdout + proc.stderr


def test_cost_sweep_direct_launch_no_scripts_import_crash(tmp_path: Path) -> None:
    """F2 (CONFIRMED live crash pre-fix): `python scripts/cost_sweep.py --root <dir>` hard-crashed
    on `from scripts.analyze_campaign import` before any output. The empty dir makes it fail on
    'no records' instead — but never with the import error."""
    out = _direct_launch("cost_sweep.py", "--root", str(tmp_path))
    assert "No module named 'scripts'" not in out, out[-2000:]
