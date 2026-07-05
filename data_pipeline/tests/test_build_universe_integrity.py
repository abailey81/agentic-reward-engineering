"""Integrity-screen wiring on the RESEARCH panel (data_pipeline/src/data/build_universe.py).

The datasheet claims the Refinitiv research panel carries Ince-Porter + split-artifact integrity
screens; before the 2026-06-25 data-integrity fix, ``build_universe`` never called ``integrity.*``
(the screens ran ONLY on the yfinance pilot path in ``cli.cmd_build``). These behaviour tests pin:

  * the new POSITIVE forward-split signature (``integrity.forward_split_artifact_flags``) catches a
    +200% one-day total return (the unadjusted 3:1 split artifact, e.g. JCI Oct-2007) and clears it
    when a vendor split is recorded;
  * ``build_universe.screen_research_returns`` is FLAG-ONLY (returns are returned byte-unchanged),
    assembles the flag report, and — critically — does NOT flag every ~0.01 return as sub-$1 when no
    price panel is supplied (the bug a naive price==return wiring would introduce);
  * a real artifact (a +200% split spike) surfaces as a forward-split suspect.

Run isolated (the repo-root conftest binds the ENGINE's ``src``; here ``src`` is data_pipeline's):
    PYTHONPATH=data_pipeline python -m pytest data_pipeline/tests/test_build_universe_integrity.py --noconftest
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make ``src`` resolve to data_pipeline's package (build_universe uses ``from . import integrity``).
_DP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_DP))
for _m in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
    del sys.modules[_m]

from src.data import build_universe as BU  # noqa: E402
from src.data import integrity  # noqa: E402


# --------------------------------------------------------------------------- #
# forward_split_artifact_flags: +100/+200/+300% TR signatures (positive split)  #
# --------------------------------------------------------------------------- #
def _ret_series() -> pd.Series:
    idx = pd.bdate_range("2007-01-02", periods=8)
    # a clean +200% (3:1 split) day amid normal returns
    return pd.Series([0.01, -0.02, 2.0, 0.012, 0.0, -0.005, 0.008, 0.003], index=idx)


def test_forward_split_flags_a_plus_200pct_day() -> None:
    out = integrity.forward_split_artifact_flags(_ret_series(), None)
    assert (out["return"] == 2.0).any()
    flagged = out[np.isclose(out["return"], 2.0)].iloc[0]
    assert flagged["flag"] == "unadjusted_fwd_split_suspect"
    assert not bool(flagged["vendor_split_recorded"])


def test_forward_split_cleared_when_vendor_split_recorded() -> None:
    s = _ret_series()
    splits = pd.Series(np.nan, index=s.index)
    splits.loc[s.index[2]] = 3.0  # vendor records the 3:1 on the +200% day
    out = integrity.forward_split_artifact_flags(s, splits)
    flagged = out[np.isclose(out["return"], 2.0)].iloc[0]
    assert bool(flagged["vendor_split_recorded"])
    assert flagged["flag"] == "split_recorded_ok"


def test_forward_split_ignores_ordinary_returns() -> None:
    idx = pd.bdate_range("2020-01-02", periods=5)
    s = pd.Series([0.03, -0.04, 0.10, -0.10, 0.05], index=idx)  # nothing near +100/200/300%
    assert integrity.forward_split_artifact_flags(s, None).empty


# --------------------------------------------------------------------------- #
# screen_research_returns: flag-only, no false sub-$1 storm, finds the artifact #
# --------------------------------------------------------------------------- #
def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2007-01-02", periods=60)
    df = pd.DataFrame(rng.standard_normal((60, 3)) * 0.01, index=idx, columns=["A", "B", "C"])
    df.iloc[30, df.columns.get_loc("A")] = 2.0        # +200% forward-split artifact on A
    df.iloc[40, df.columns.get_loc("B")] = -0.50      # -50% price-side split signature on B
    return df


def test_screen_is_flag_only_returns_unchanged() -> None:
    df = _panel()
    before = df.copy()
    returns_out, report, summary = BU.screen_research_returns(df, prices=None, splits=None)
    pd.testing.assert_frame_equal(returns_out, before)   # NOT mutated (R4 / PRIME DIRECTIVE)
    assert summary["policy"].startswith("flag-only")


def test_screen_no_false_sub_dollar_storm_without_prices() -> None:
    """With NO price panel the $1-prior-price screen must match NOTHING (a naive price==return
    wiring would read every ~0.01 return as sub-$1 and flag the whole panel)."""
    _, _, summary = BU.screen_research_returns(_panel(), prices=None, splits=None)
    assert summary["n_sub_dollar_prior"] == 0


def test_screen_sub_dollar_fires_only_on_real_sub_dollar_prices() -> None:
    """With a real (unadjusted) price panel, the $1 screen flags exactly the post-sub-$1-price
    returns and nothing else."""
    df = _panel()
    px = pd.DataFrame(50.0, index=df.index, columns=df.columns)   # all well above $1
    px.iloc[10, px.columns.get_loc("C")] = 0.40                   # C trades < $1 at t=10
    _, report, summary = BU.screen_research_returns(df, prices=px, splits=None)
    sub = report[report["reason"] == "sub_dollar_prior"] if "reason" in report.columns else report.iloc[0:0]
    # the only sub-$1 prior is C at t=10, so the return at t=11 (the next session) is the lone flag
    assert summary["n_sub_dollar_prior"] == 1
    assert set(sub["name"]) == {"C"}


def test_screen_finds_forward_and_price_side_split_artifacts() -> None:
    _, report, summary = BU.screen_research_returns(_panel(), prices=None, splits=None)
    assert summary["n_unadjusted_fwd_split_suspect"] == 1     # A's +200%
    assert summary["n_unadjusted_split_suspect"] == 1         # B's -50%
    reasons = set(report["reason"]) if "reason" in report.columns else set()
    assert {"unadjusted_fwd_split_suspect", "unadjusted_split_suspect"} <= reasons
