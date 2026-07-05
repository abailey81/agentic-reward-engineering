"""Tests for market-regime labelling (FINAL_PLAN F.12, audit B-6)."""

from __future__ import annotations


import numpy as np
import pytest

from src.data.panel import Panel
from src.regimes.definition import independent_regime_count, label_regimes
from src.utils.config import load_config


def _panel_with_vix(vix: np.ndarray) -> Panel:
    t = vix.size
    returns = np.zeros((t, 2))
    dates = np.arange("2010-01-04", t, dtype="datetime64[D]")
    return Panel(returns=returns, vix=vix, dates=dates, asset_ids=np.arange(2, dtype=np.int64))


def test_labels_cover_panel(synthetic_panel: Panel) -> None:
    """Every row of the panel receives a regime label in {0,1,2}."""
    cfg = load_config("regimes")
    labels = label_regimes(synthetic_panel, cfg)
    assert labels.shape == (synthetic_panel.T,)
    assert set(np.unique(labels)).issubset({0, 1, 2})


def test_vix_thresholds_produce_expected_labels() -> None:
    """A VIX series crossing the thresholds yields calm/normal/stress = 0/1/2."""
    cfg = load_config("regimes")  # calm=15, stress=25
    vix = np.array([10.0, 15.0, 20.0, 25.0, 30.0])
    panel = _panel_with_vix(vix)
    labels = label_regimes(panel, cfg)
    # <15 -> calm(0); [15,25] -> normal(1); >25 -> stress(2)
    np.testing.assert_array_equal(labels, np.array([0, 1, 1, 1, 2]))


def test_independent_regime_count_counts_contiguous_blocks() -> None:
    """independent_regime_count counts run-length blocks, not distinct labels."""
    labels = np.array([0, 0, 1, 1, 1, 2, 1, 1, 0])  # blocks: 0,1,2,1,0 -> 5
    assert independent_regime_count(labels) == 5
    assert independent_regime_count(np.array([1, 1, 1])) == 1
    assert independent_regime_count(np.array([])) == 0
    assert independent_regime_count(np.array([0, 1, 0, 1])) == 4


def test_unimplemented_variants_raise() -> None:
    """nber/hmm variants raise NotImplementedError; vix_threshold does not."""
    cfg = load_config("regimes")
    panel = _panel_with_vix(np.array([10.0, 20.0, 30.0]))
    for variant in ("nber_dates", "hmm_2state", "hmm_3state"):
        bad = dict(cfg)
        bad["definition"] = variant
        with pytest.raises(NotImplementedError):
            label_regimes(panel, bad)


def test_independent_regime_count_reproducible(synthetic_panel: Panel) -> None:
    """The independent-regime count is reproducible for a fixed config + panel."""
    cfg = load_config("regimes")
    a = independent_regime_count(label_regimes(synthetic_panel, cfg))
    b = independent_regime_count(label_regimes(synthetic_panel, cfg))
    assert a == b
    assert isinstance(a, int)
