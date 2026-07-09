"""Tests for the Panel data-contract (src/data/validation.py).

Constructs panels that satisfy the structural Panel.__post_init__ (shape + finiteness) but violate a
SEMANTIC / leakage invariant, and checks validate_panel flags exactly that. Also checks the real
synthetic panel passes the contract clean.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.data.panel import Panel
from src.data.synthetic import make_synthetic_panel
from src.data.validation import (
    PanelContractError,
    PanelOverlapDiff,
    panel_overlap_diff,
    validate_panel,
)


def _clean(t: int = 20, n: int = 4) -> Panel:
    rng = np.random.default_rng(0)
    returns = rng.standard_normal((t, n)) * 0.01
    vix = np.abs(rng.standard_normal(t)) * 4.0 + 15.0
    dates = np.arange(np.datetime64("2020-01-01"), np.datetime64("2020-01-01") + np.timedelta64(t, "D"))
    asset_ids = np.arange(n)
    return Panel(returns=returns, vix=vix, dates=dates, asset_ids=asset_ids)


def test_clean_panel_passes() -> None:
    assert validate_panel(_clean(), strict=False) == []
    validate_panel(_clean(), strict=True)  # must not raise


def test_synthetic_panel_satisfies_contract() -> None:
    """The shipped synthetic panel (used everywhere in tests) must satisfy the data contract."""
    p = make_synthetic_panel(n_assets=6, n_days=160, seed=3)
    assert validate_panel(p, strict=False) == []


def test_real_gold_panel_satisfies_contract_if_present() -> None:
    """Integration: the REAL licensed gold panel must satisfy the contract (calibration guard). Skips
    cleanly where the licensed data is absent (e.g. CI / a no-licence runner)."""
    try:
        from src.data.loaders import load_gold_panel
    except Exception as e:  # pragma: no cover - import guard
        pytest.skip(f"loaders unavailable: {e}")
    try:
        result = load_gold_panel(phase="development")
    except FileNotFoundError:
        pytest.skip("gold panel not present in this environment")
    panel = getattr(result, "panel", result)
    validate_panel(panel, strict=True)  # must not raise on real delisting-bearing data


def test_unsorted_or_duplicate_dates_flagged_as_leakage() -> None:
    p = _clean()
    dates = p.dates.copy()
    dates[5] = dates[4]  # duplicate session -> non-positive gap -> leakage risk
    bad = dataclasses.replace(p, dates=dates)
    issues = validate_panel(bad, strict=False)
    assert any("strictly increasing" in s for s in issues)
    with pytest.raises(PanelContractError):
        validate_panel(bad, strict=True)


def test_return_below_minus_one_flagged_but_exact_total_loss_allowed() -> None:
    p = _clean()
    r = p.returns.copy()
    r[0, 0] = -1.5  # losing MORE than 100% is impossible -> flagged
    assert any("< -1.0" in s for s in validate_panel(dataclasses.replace(p, returns=r), strict=False))
    # exactly -1.0 (delisting/bankruptcy total loss; the project's delisting band includes -100%) is ALLOWED
    r2 = p.returns.copy()
    r2[0, 0] = -1.0
    assert not any("-1.0" in s for s in validate_panel(dataclasses.replace(p, returns=r2), strict=False))


def test_implausible_return_magnitude_flagged() -> None:
    p = _clean()
    r = p.returns.copy()
    r[3, 2] = 25.0  # +2500% in a day -> split/units artifact
    issues = validate_panel(dataclasses.replace(p, returns=r), strict=False, max_abs_return=10.0)
    assert any("implausible daily move" in s for s in issues)


def test_all_zero_column_flagged() -> None:
    p = _clean()
    r = p.returns.copy()
    r[:, 1] = 0.0  # a dead / gap-filled asset
    issues = validate_panel(dataclasses.replace(p, returns=r), strict=False)
    assert any("all-zero return column" in s for s in issues)


def test_negative_vix_flagged() -> None:
    p = _clean()
    vix = p.vix.copy()
    vix[2] = -1.0
    issues = validate_panel(dataclasses.replace(p, vix=vix), strict=False)
    assert any("volatility index cannot be negative" in s for s in issues)


def test_duplicate_and_noninteger_asset_ids_flagged() -> None:
    p = _clean()
    dup = dataclasses.replace(p, asset_ids=np.array([0, 1, 1, 2]))
    assert any("not unique" in s for s in validate_panel(dup, strict=False))
    nonint = dataclasses.replace(p, asset_ids=np.array([0.0, 1.0, 2.0, 3.0]))
    assert any("integer" in s for s in validate_panel(nonint, strict=False))


def test_negative_market_caps_flagged() -> None:
    p = _clean()
    mc = np.abs(np.random.default_rng(1).standard_normal((p.T, p.N))) * 1e6
    mc[0, 0] = -5.0
    issues = validate_panel(dataclasses.replace(p, market_caps=mc), strict=False)
    assert any("market_caps < 0" in s or "capitalisation cannot be negative" in s for s in issues)


def test_strict_aggregates_all_violations_in_message() -> None:
    p = _clean()
    r = p.returns.copy()
    r[0, 0] = -2.0
    vix = p.vix.copy()
    vix[0] = -1.0
    bad = dataclasses.replace(p, returns=r, vix=vix)
    with pytest.raises(PanelContractError) as ei:
        validate_panel(bad, strict=True)
    msg = str(ei.value)
    assert "< -1.0" in msg and "negative" in msg  # both violations reported together


# --------------------------------------------------------------------------- #
# C4 — panel_overlap_diff: byte-diff a candidate returns panel vs the frozen   #
# --------------------------------------------------------------------------- #
def _returns_frame(seed: int = 0, *, cols=("A.N", "B.N", "C.N"), n: int = 8):
    import pandas as pd

    rng = np.random.default_rng(seed)
    idx = pd.date_range("2005-01-03", periods=n, freq="D")
    data = {c: rng.standard_normal(n) * 0.01 for c in cols}
    return pd.DataFrame(data, index=idx)


def test_overlap_diff_identical_frames() -> None:
    """An identical candidate reports zero changed cells over the full overlap."""
    ref = _returns_frame(0)
    diff = panel_overlap_diff(ref.copy(), ref)
    assert isinstance(diff, PanelOverlapDiff)
    assert diff.identical_over_overlap
    assert diff.n_changed_cells == 0
    assert diff.max_abs_delta == 0.0
    assert diff.n_overlap_rows == len(ref) and diff.n_overlap_cols == ref.shape[1]


def test_overlap_diff_detects_changed_cells() -> None:
    """A perturbed cell is counted, with the (date, ric, ref->cand) example + max |delta| reported."""
    ref = _returns_frame(0)
    cand = ref.copy()
    cand.iloc[2, 1] = ref.iloc[2, 1] + 0.5  # one changed cell, delta 0.5
    diff = panel_overlap_diff(cand, ref)
    assert diff.n_changed_cells == 1
    assert not diff.identical_over_overlap
    assert abs(diff.max_abs_delta - 0.5) < 1e-9
    dt, ric, a, b = diff.changed_examples[0]
    assert ric == ref.columns[1]
    assert abs(b - a - 0.5) < 1e-9


def test_overlap_diff_nan_aware() -> None:
    """Two ALIGNED NaNs are EQUAL; a NaN-vs-number cell IS a change."""
    ref = _returns_frame(0)
    cand = ref.copy()
    ref.iloc[0, 0] = np.nan
    cand.iloc[0, 0] = np.nan  # aligned NaN -> not a change
    cand.iloc[1, 0] = np.nan  # NaN vs a real number -> a change
    diff = panel_overlap_diff(cand, ref)
    assert diff.n_changed_cells == 1
    dt, ric, a, b = diff.changed_examples[0]
    assert ric == ref.columns[0] and np.isnan(b) and not np.isnan(a)


def test_overlap_diff_reports_schema_drift_without_counting_it() -> None:
    """Added/retired names + calendar drift are reported separately, not as changed overlap cells."""
    ref = _returns_frame(0, cols=("A.N", "B.N", "C.N"), n=8)
    # Candidate drops C.N, adds D.N (schema drift), and adds one extra trailing session (calendar drift).
    cand = _returns_frame(0, cols=("A.N", "B.N", "D.N"), n=9)
    # Make the SHARED A.N/B.N over the shared 8 rows identical to ref so overlap is clean.
    cand.iloc[:8, cand.columns.get_loc("A.N")] = ref["A.N"].to_numpy()
    cand.iloc[:8, cand.columns.get_loc("B.N")] = ref["B.N"].to_numpy()
    diff = panel_overlap_diff(cand, ref)
    assert diff.identical_over_overlap  # shared A.N/B.N over shared rows unchanged
    assert diff.ref_only_cols == ["C.N"]
    assert diff.cand_only_cols == ["D.N"]
    assert diff.cand_only_rows == 1 and diff.ref_only_rows == 0
    assert "changed cells: 0" in diff.summary()


def test_overlap_diff_reads_parquet_paths(tmp_path) -> None:
    """Accepts parquet PATHS (the verify_gold CLI path), not just in-memory frames."""
    ref = _returns_frame(0)
    cand = ref.copy()
    cand.iloc[0, 0] = ref.iloc[0, 0] + 1.0
    rp = tmp_path / "returns_panel_ref.parquet"
    cp = tmp_path / "returns_panel_cand.parquet"
    ref.to_parquet(rp)
    cand.to_parquet(cp)
    diff = panel_overlap_diff(cp, rp)
    assert diff.n_changed_cells == 1 and abs(diff.max_abs_delta - 1.0) < 1e-9
