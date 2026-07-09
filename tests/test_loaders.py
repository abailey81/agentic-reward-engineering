"""Tests for the real-gold loader (``src/data/loaders.py``).

These run against the frozen gold panel in ``data/gold/`` and are skipped if it is absent (so CI without
the licensed data still passes). They check the two invariants the loader exists to guarantee:
finiteness (the env's contract) and anonymisation (the contamination defence).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.loaders import gold_suffix, load_gold_panel
from src.data.panel import Panel

# Suffix-aware skip guard: point at the ACTIVE panel the loader actually reads (univ5 post-Split-C),
# so the suite neither runs against a missing panel nor skips when the active one is present.
_GOLD = Path(__file__).resolve().parents[1] / "data" / "gold" / f"returns_panel_{gold_suffix()}.parquet"
pytestmark = pytest.mark.skipif(not _GOLD.exists(), reason="frozen gold panel not present (licensed data)")


def test_dev_panel_loads_finite_and_shaped() -> None:
    res = load_gold_panel("development")
    p = res.panel
    assert isinstance(p, Panel)
    assert p.returns.shape[1] == 30  # top-30 universe
    assert p.returns.shape[0] == p.vix.shape[0] == p.dates.shape[0]  # aligned T
    assert np.isfinite(p.returns).all()  # env rejects non-finite — the loader's job
    assert np.isfinite(p.vix).all()


def test_panel_is_anonymised() -> None:
    """No RIC/ticker/string identity may live in the Panel (N3 contamination defence)."""
    res = load_gold_panel("development")
    assert res.panel.asset_ids.dtype.kind in "iu"  # integer ids only
    assert list(res.panel.asset_ids) == list(range(30))
    # the RIC map is provenance-only and lives OUTSIDE the panel
    assert len(res.ric_by_id) == 30
    assert all(isinstance(r, str) for r in res.ric_by_id.values())


def test_liquidate_to_cash_makes_dead_names_finite() -> None:
    """Wachovia (WB.N^A09, delists 2009) is in the dev top-30; default policy must absorb its NaNs."""
    res = load_gold_panel("development", on_missing="liquidate_to_cash")
    assert np.isfinite(res.panel.returns).all()
    assert "WB.N^A09" in res.ric_by_id.values()  # the dead name is RETAINED, not survivorship-dropped


def test_error_policy_flags_delisting_nans() -> None:
    """on_missing='error' must refuse the dev window (it contains mid-window delistings)."""
    with pytest.raises(ValueError, match="NaN"):
        load_gold_panel("development", on_missing="error")


def test_ffill_then_zero_zeros_dead_tail_not_fabricates() -> None:
    """ffill_then_zero must ZERO a delisted name's post-delisting tail, NOT repeat its last return
    (final-audit #20). Wachovia (WB.N^A09) delists 2009 but the dev window runs to 2016 (SPLIT C), so
    its final sessions are dead: under the FIXED policy that tail is 0.0; a bare .ffill() would have
    carried its last traded return forward, fabricating compounding wealth on a dead name."""
    res = load_gold_panel("development", on_missing="ffill_then_zero")
    assert np.isfinite(res.panel.returns).all()
    wb_id = [i for i, ric in res.ric_by_id.items() if ric == "WB.N^A09"]
    assert wb_id, "WB.N^A09 (delists 2009) expected in the dev top-30"
    col = res.panel.returns[:, wb_id[0]]
    assert col[-1] == 0.0  # the 2014 final session is in the dead tail -> zeroed, not a carried return


def test_walk_forward_window_start_selects_pit_universe() -> None:
    """The loader can select a POINT-IN-TIME walk-forward top-30 by window_start — the #13 building
    block for a PIT-universe robustness re-evaluation of the sealed test leg. The 2020 PIT cohort
    (the Split-C test start) differs from the held 2005 development cohort (the composition bias the
    robustness check addresses), and the PIT panel loads finite + prelagged over the 2020-2026 span."""
    from pathlib import Path

    from src.data.loaders import _GOLD_DIR, _read, _window_rics

    top30 = _read("top30_selection", Path(_GOLD_DIR))
    dev, _ = _window_rics(top30, "development")
    pit, start = _window_rics(top30, "walk_forward", "2020-01-02")
    assert str(start)[:10] == "2020-01-02"
    assert set(dev) != set(pit)  # the 2005 cohort and the 2020 PIT universe are NOT identical
    res = load_gold_panel(phase="walk_forward", window_start="2020-01-02", end="2026-06-30")
    assert np.isfinite(res.panel.returns).all()
    assert res.panel.vix_prelagged is True
    with pytest.raises(KeyError, match="window_start"):
        _window_rics(top30, "walk_forward", "1999-01-01")  # absent window -> fail loud


def test_walk_forward_requires_explicit_end() -> None:
    with pytest.raises(ValueError, match="explicit `end`"):
        load_gold_panel("walk_forward")


def test_gold_vix_is_in_points_and_regimes_do_not_collapse() -> None:
    """The loader normalizes the gold's FRACTIONAL vix (FRED VIXCLS/100) to conventional POINTS.

    Regression for the vix-units bug (2026-06-19): the gold stores vix as ~0.09..0.83; the
    VIX-threshold regime rule (config/regimes.yaml calm<15 / stress>25) then labelled EVERY date
    calm, collapsing the independent-regime count to 1 and silently zeroing the regime-stratified
    analysis that bounds H2 power (audit B-6). The loader now rescales to points (~9..83), so all
    three regimes appear and there are MANY independent episodes over the 2005-2016 train window.
    """
    from src.regimes.definition import independent_regime_count, label_regimes
    from src.utils.config import load_config

    p = load_gold_panel("development").panel
    # Points, not fractions: a real VIX never sits below ~5 and spikes well past the 25 stress line.
    assert float(np.median(p.vix)) > 5.0
    assert float(p.vix.max()) > 25.0
    assert float(p.vix.min()) > 1.0  # decisively not the ~0.09 fractional convention

    labels = label_regimes(p, load_config("regimes"))
    assert set(np.unique(labels)) == {0, 1, 2}  # calm + normal + stress all realised
    assert independent_regime_count(labels) > 1  # the collapse-to-1 bug cannot recur


def test_seed_leading_vix_uses_past_not_future_session() -> None:
    """A leading VIX NaN in a window is seeded from the genuine PRIOR session — never bfill'd from a
    LATER in-window session (the data-1 prelag leakage fix). bfill fires ONLY when no prior data exists
    (the global-first boundary cell, which is irreducible). Tests the extracted _seed_leading_vix rule
    directly, since no real gold window has both a leading NaN AND genuine prior data."""
    from src.data.loaders import _seed_leading_vix

    idx = pd.to_datetime(["2018-01-02", "2018-01-03", "2018-01-04", "2018-01-05"])
    cash = pd.DataFrame({"vix": [0.11, 0.12, np.nan, 0.14]}, index=idx)

    # Window starts 2018-01-04 -> leading NaN, but 2018-01-03 (=0.12) precedes it: seed from the PAST.
    start = pd.Timestamp("2018-01-04")
    out = _seed_leading_vix(cash.loc[start:, "vix"].copy(), cash, start)
    assert out.iloc[0] == 0.12  # the genuine t-1 close, NOT the future in-window 0.14
    assert not out.isna().any()

    # Window starts at the GLOBAL-FIRST session with a leading NaN and NO prior data -> bfill (irreducible).
    start0 = idx[0]
    vix0 = cash.loc[start0:, "vix"].copy()
    vix0.iloc[0] = np.nan
    out0 = _seed_leading_vix(vix0, cash, start0)
    assert out0.iloc[0] == 0.12  # bfilled from the next in-window value ONLY because nothing precedes start0
    assert not out0.isna().any()
