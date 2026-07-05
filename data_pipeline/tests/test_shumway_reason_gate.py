"""Reason-gated Shumway delisting surcharge (data_pipeline/src/data/membership.py).

The frozen Refinitiv vault carries NO delisting reason (verified: rf_meta_* has only date/exchange/
sector, and the date is empty for all 333 dead RICs), so the −30/−55 surcharge is applied
UNCONDITIONALLY — including to premium M&A whose true terminal was positive (DELL buyout, TWX→AT&T,
ABMD→J&J, ...). The reason-gating plumbing is wired so that the moment the reason is re-pulled
once the delisting reason is re-pulled, M&A names keep their real last return and only genuine performance
terminations are surcharged. These tests pin BOTH:

  * the gate is a strict NO-OP when the reason is absent (univ4 stays byte-identical — no fabrication);
  * once a reason IS present, M&A is withheld from the surcharge and performance/unknown is surcharged.

Run isolated (repo-root conftest binds the ENGINE's src; here src is data_pipeline's):
    PYTHONPATH=data_pipeline python -m pytest data_pipeline/tests/test_shumway_reason_gate.py --noconftest
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

from src.data.membership import apply_shumway_corrections, classify_delist_reason  # noqa: E402


def _panel() -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=10)
    df = pd.DataFrame(0.01, index=idx, columns=["MNA", "FAIL"])
    # both names "die" at the last session (no NaN tail needed for this unit; last_valid = row 9)
    return df


def test_classify_reason_buckets() -> None:
    assert classify_delist_reason("Acquired by Intel") == "mna"
    assert classify_delist_reason("Merger") == "mna"
    assert classify_delist_reason("Taken private") == "mna"
    assert classify_delist_reason("Chapter 11 bankruptcy") == "performance"
    assert classify_delist_reason("Failed listing standards / compliance") == "performance"
    assert classify_delist_reason(None) == "unknown"
    assert classify_delist_reason("") == "unknown"
    assert classify_delist_reason("some unmapped phrase") == "unknown"


def test_gate_is_noop_when_reason_absent() -> None:
    """Reason absent (the current vault): BOTH names are surcharged exactly as before — the gate
    introduces NO change, so univ4 stays byte-identical and nothing is fabricated."""
    df = _panel()
    last = df.index[-1]
    delisted = {
        "MNA": {"date": last, "exchange": "nasdaq", "vendor_terminal_return": None, "reason": None},
        "FAIL": {"date": last, "exchange": "nyse_amex", "vendor_terminal_return": None, "reason": None},
    }
    out, log = apply_shumway_corrections(df, delisted)
    actions = set(log["action"])
    assert actions == {"shumway_correction_applied"}            # BOTH surcharged (reason unknown -> surcharge)
    # nasdaq -55%, nyse_amex -30%, multiplicative onto the +0.01 last return
    assert out.at[last, "MNA"] == np.float64((1.01) * (1 - 0.55) - 1.0)
    assert out.at[last, "FAIL"] == np.float64((1.01) * (1 - 0.30) - 1.0)
    assert set(log["reason_class"]) == {"unknown"}


def test_gate_withholds_surcharge_for_mna_once_reason_present() -> None:
    """With a reason: the M&A name keeps its real last return (NO surcharge); the failure is surcharged."""
    df = _panel()
    last = df.index[-1]
    delisted = {
        "MNA": {"date": last, "exchange": "nasdaq", "vendor_terminal_return": None,
                "reason": "Acquired by Intel (merger)"},
        "FAIL": {"date": last, "exchange": "nyse_amex", "vendor_terminal_return": None,
                 "reason": "Chapter 11 bankruptcy"},
    }
    out, log = apply_shumway_corrections(df, delisted)
    by_ric = {r["ric"]: r for _, r in log.iterrows()}
    # M&A: untouched real return, action records the withholding
    assert out.at[last, "MNA"] == np.float64(0.01)
    assert by_ric["MNA"]["action"] == "mna_keep_vendor_terminal"
    # FAIL: surcharged -30%, classified performance
    assert out.at[last, "FAIL"] == np.float64((1.01) * (1 - 0.30) - 1.0)
    assert by_ric["FAIL"]["action"] == "shumway_correction_applied"
    assert by_ric["FAIL"]["reason_class"] == "performance"


def test_vendor_terminal_still_preferred_over_reason_gate() -> None:
    """A present vendor terminal return short-circuits BEFORE the reason gate (kept as-is)."""
    df = _panel()
    last = df.index[-1]
    delisted = {"MNA": {"date": last, "exchange": "nasdaq",
                        "vendor_terminal_return": 0.42, "reason": "Merger"}}
    out, log = apply_shumway_corrections(df, delisted)
    assert log.iloc[0]["action"] == "vendor_terminal_kept"
    assert out.at[last, "MNA"] == np.float64(0.01)   # frame not overwritten
