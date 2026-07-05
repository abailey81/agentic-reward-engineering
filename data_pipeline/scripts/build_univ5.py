#!/usr/bin/env python3
"""Build the extended panels: univ5 (headline, zero-fill) + univ5s (corrected Shumway band-end).

ADR-051 P1: consumes the frozen vault + the x26 extension artifacts (rf_trd_x26*/rf_mcapm_x26*
prefix-collected automatically; the spliced pit_membership_200411_202606 is the LATEST staged
entry; fred_macro_x26 is the LATEST macro frame) under the 2026-06-30 config period. Two builds:

  * ``_univ5``  — the extended HEADLINE panel (univ3 conventions: no surcharge; the env's
    liquidate_to_cash zero-fill handles dead names downstream).
  * ``_univ5s`` — the corrected Shumway variant: OBSERVED-terminal recovery keeps each dead
    name's realised terminal (vendor_terminal_kept) and surcharges ONLY terminal-less names —
    the band-end that replaces the M&A-contaminated univ4.

Write-once: a re-run after a partial failure will FileExistsError on the frozen names — that is
deliberate (investigate, clean up the partial suffix explicitly, never silently overwrite).

Run:  $env:PYTHONPATH="<repo>\\data_pipeline"; & <repo>\\.venv-lseg\\Scripts\\python.exe data_pipeline\\scripts\\build_univ5.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "data_pipeline"))

from src.data.build_universe import build_universe  # noqa: E402


def main() -> int:
    t0 = time.perf_counter()
    print("[build] univ5 (headline, no surcharge) ...", flush=True)
    out5 = build_universe(suffix="_univ5", run_id="build_univ5")
    print(f"[build] univ5 DONE in {time.perf_counter() - t0:.0f}s:", flush=True)
    for k, v in out5.items():
        print(f"    {k}: {v}", flush=True)

    t1 = time.perf_counter()
    print("[build] univ5s (Shumway + observed-terminal recovery) ...", flush=True)
    out5s = build_universe(suffix="_univ5s", run_id="build_univ5s", apply_delisting=True)
    print(f"[build] univ5s DONE in {time.perf_counter() - t1:.0f}s:", flush=True)
    for k, v in out5s.items():
        print(f"    {k}: {v}", flush=True)
    print("[build] ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
