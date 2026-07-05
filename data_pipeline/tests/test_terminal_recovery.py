"""OBSERVED-terminal recovery for the Shumway delisting correction (ADR-051).

The reason mnemonics do not resolve under this entitlement (probed 2026-07-01), so the
corrected surcharge band-end (univ5s) rests on recovering each dead name's REALISED terminal
return from the daily series itself: recovered -> the corrector keeps it (vendor_terminal_kept,
no surcharge); not recoverable (stale series, prints long before the delist month) -> stays
``None`` and the conservative -30/-55% surcharge applies. These tests pin the PURE helper
(:func:`build_universe._recover_terminal_from_returns`) hermetically — no vault, no manifest.

Run isolated (repo-root conftest binds the ENGINE's src; here src is data_pipeline's):
    PYTHONPATH=data_pipeline python -m pytest data_pipeline/tests/test_terminal_recovery.py --noconftest
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_DP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_DP))
for _m in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
    del sys.modules[_m]

from src.data.build_universe import _recover_terminal_from_returns  # noqa: E402


def _frame(n: int = 120) -> pd.DataFrame:
    idx = pd.bdate_range("2008-04-01", periods=n)
    df = pd.DataFrame(0.001, index=idx, columns=["DEAD.N^I08", "STALE.O^I08", "ALIVE.N"])
    return df


def test_recovers_last_valid_return_near_the_delist_date() -> None:
    df = _frame()
    # the dead name prints through session 100 with a crash terminal, NaN afterwards
    df.iloc[101:, df.columns.get_loc("DEAD.N^I08")] = np.nan
    df.iloc[100, df.columns.get_loc("DEAD.N^I08")] = -0.449
    delist = df.index[110]  # derived (suffix month-end snap) 10 sessions after the last print
    rec = _recover_terminal_from_returns("DEAD.N^I08", delist, df)
    assert rec is not None
    value, date = rec
    assert value == -0.449
    assert date == df.index[100]


def test_stale_series_is_not_recovered() -> None:
    df = _frame()
    # prints stop 80 sessions before the derived delist date -> stale, no recovery (surcharge applies)
    df.iloc[31:, df.columns.get_loc("STALE.O^I08")] = np.nan
    delist = df.index[110]
    assert _recover_terminal_from_returns("STALE.O^I08", delist, df) is None


def test_prints_after_the_delist_date_block_recovery() -> None:
    df = _frame()  # ALIVE prints through the whole frame — a delist date mid-frame is suspect
    assert _recover_terminal_from_returns("ALIVE.N", df.index[50], df) is None


def test_missing_column_and_all_nan_are_none() -> None:
    df = _frame()
    assert _recover_terminal_from_returns("GHOST.N^A01", df.index[-1], df) is None
    df["EMPTY.N^I08"] = np.nan
    assert _recover_terminal_from_returns("EMPTY.N^I08", df.index[-1], df) is None


def test_gap_boundary_is_inclusive_at_max_gap() -> None:
    df = _frame()
    col = df.columns.get_loc("DEAD.N^I08")
    df.iloc[61:, col] = np.nan
    df.iloc[60, col] = -0.2
    # exactly max_gap_sessions (45) between last print (60) and delist (105) -> accepted
    assert _recover_terminal_from_returns("DEAD.N^I08", df.index[105], df) is not None
    # one session beyond -> rejected
    assert _recover_terminal_from_returns("DEAD.N^I08", df.index[106], df) is None
