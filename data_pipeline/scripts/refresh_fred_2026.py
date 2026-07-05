#!/usr/bin/env python3
"""Refresh the FRED macro series through the 2026-06-30 cutoff (ADR-051 P1 rebuild).

The frozen ``fred_macro.csv`` ends 2025-12-31; the univ5 build needs VIX (and the rf/term
series) through the extension, else the 2026 sessions carry NaN VIX features. FRED is public
data (keyless fredgraph route), so the refresh is a plain re-pull of the SAME four series over
the FULL span, frozen under the versioned name ``fred_macro_x26.csv`` (write-once vault; the
build now consumes the LATEST fred_macro* entry). Full-span (not append) so one artifact is
self-consistent — FRED occasionally revises history, and a single provenance-stamped frame
beats a seam.

Run:  $env:PYTHONPATH="<repo>\\data_pipeline"; & <repo>\\.venv-lseg\\Scripts\\python.exe data_pipeline\\scripts\\refresh_fred_2026.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "data_pipeline"))

from src.data import acquire  # noqa: E402
from src.data.vault import freeze, manifest_entries  # noqa: E402

SERIES = ["VIXCLS", "DGS3MO", "DGS10", "T10Y2Y"]
START, END = "2005-01-01", "2026-06-30"
NAME = "fred_macro_x26.csv"


def main() -> int:
    if NAME in {e["name"] for e in manifest_entries()}:
        print(f"[fred-x26] {NAME} already frozen (write-once) — nothing to do")
        return 0
    df = acquire.fetch_fred({"series": SERIES, "start": START, "end": END})
    if df.empty or df["VIXCLS"].dropna().empty:
        raise RuntimeError("FRED refresh returned an empty frame — do not freeze")
    last_vix = df["VIXCLS"].dropna().index.max()
    print(f"[fred-x26] pulled {df.shape[0]} rows x {df.shape[1]} series; last VIX print: {last_vix.date()}")
    if str(last_vix.date()) < "2026-06-27":
        raise RuntimeError(f"VIX ends {last_vix.date()} — does not reach the 2026-06-30 cutoff week")
    art = freeze(df, "raw", NAME,
                 {"vendor": "FRED", "stage": "refresh_fred_2026",
                  "query": "fredgraph.csv keyless", "series": SERIES,
                  "params": {"start": START, "end": END},
                  "reason": "ADR-051 extension: frozen fred_macro.csv ends 2025-12-31"},
                 run_id="universe_refinitiv_x26")
    print(f"[fred-x26] froze {art.relpath} ({art.rows} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
