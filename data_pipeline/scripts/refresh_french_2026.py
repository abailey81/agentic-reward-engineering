#!/usr/bin/env python3
"""Refresh the Ken French factor files toward the 2026-06-30 cutoff (ADR-051 follow-up).

The on-disk ``french_F-F_Research_Data_Factors_daily.csv`` / momentum file end 2026-04-30, but the
Split-C sealed test runs to 2026-06-30 — a ~2-month attribution-coverage gap (flagged in
attribution spec, 2026-07-02). The Ken French library is PUBLIC but publishes with a
lag, so this refresh pulls whatever is currently available, freezes it under versioned names
(``french_*_x26.csv``), and REPORTS the achieved end date. If upstream still ends short of the
cutoff, the attribution leg runs on the covered window with the residual gap DISCLOSED (the
attribution is report-only; it never gates the m=6 family).

Run:  $env:PYTHONPATH="<repo>\\data_pipeline"; & <repo>\\.venv-lseg\\Scripts\\python.exe data_pipeline\\scripts\\refresh_french_2026.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "data_pipeline"))

from src.data.vault import freeze, manifest_entries  # noqa: E402

DATASETS = {
    "F-F_Research_Data_Factors_daily": "french_ff3_daily_x26.csv",
    "F-F_Momentum_Factor_daily": "french_mom_daily_x26.csv",
}
CUTOFF = "2026-06-30"
_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"


def _fetch_french_direct(dataset: str) -> pd.DataFrame:
    """Direct zip download + manual parse of the fixed-format Ken French daily CSVs.

    Bypasses pandas_datareader, whose famafrench parser breaks on modern pandas (str-vs-float
    TypeError inside nanops, observed 2026-07-02). Format: header junk lines, then rows
    ``YYYYMMDD, col1, col2, ...`` in PERCENT; the daily files end with a blank line (+ possibly
    a copyright footer). Values -> decimal (/100), matching the on-disk french_* convention.
    """
    import io
    import urllib.request
    import zipfile

    import pandas as pd

    url = f"{_BASE}{dataset}_CSV.zip"
    with urllib.request.urlopen(url, timeout=120) as r:
        blob = r.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        text = zf.read(csv_name).decode("utf-8", errors="replace")
    rows: list[list[str]] = []
    header: list[str] | None = None
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if header is None:
            # the (only) header row starts with an empty first cell followed by factor names
            if parts and parts[0] == "" and len(parts) > 1 and any(p for p in parts[1:]):
                header = ["date"] + [p for p in parts[1:]]
            continue
        if parts and len(parts[0]) == 8 and parts[0].isdigit():
            rows.append(parts[: len(header)])
    if header is None or not rows:
        raise RuntimeError(f"{dataset}: could not parse the CSV layout")
    df = pd.DataFrame(rows, columns=header)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date")
    df = df.apply(pd.to_numeric, errors="coerce") / 100.0  # percent -> decimal
    return df[df.index >= "2005-01-01"]


def main() -> int:
    existing = {e["name"] for e in manifest_entries()}
    for dataset, name in DATASETS.items():
        if name in existing:
            print(f"[french-x26] {name} already frozen (write-once) - skipping")
            continue
        df = _fetch_french_direct(dataset)
        if df.empty:
            raise RuntimeError(f"{dataset}: empty frame - do not freeze")
        # pandas-datareader returns a PeriodIndex for some famafrench sets; normalize to timestamps.
        try:
            idx_max = df.index.max()
            end = str(getattr(idx_max, "date", lambda: idx_max)())
        except Exception:  # noqa: BLE001 - report-only formatting
            end = str(df.index.max())
        print(f"[french-x26] {dataset}: {df.shape[0]} rows, ends {end} "
              f"({'REACHES' if end >= CUTOFF else 'SHORT OF'} the {CUTOFF} cutoff)")
        art = freeze(df, "raw", name,
                     {"vendor": "KenFrench", "stage": "refresh_french_2026",
                      "query": f"famafrench:{dataset}", "params": {"start": "2005-01-01"},
                      "achieved_end": end,
                      "reason": "ADR-051 Split-C test end 2026-06-30; prior CSVs end 2026-04-30"},
                     run_id="universe_refinitiv_x26")
        print(f"[french-x26] froze {art.relpath} ({art.rows} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
