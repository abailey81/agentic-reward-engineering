"""Numerical-robustness guards for the inference stack (P0-1, audit 2026-06-19).

Pins the fixes for the degenerate-input failure modes the audit found:
- a NEAR-constant return series must NOT score ``deflated_sharpe == 1.0`` (the bug
  let a flat reward win candidate selection, because ``np.full(750, 1e-3).std()``
  is ``~2e-19`` not exactly ``0`` so the old exact ``sd == 0`` guard never fired);
- non-finite (NaN/inf) returns from a pathological reward must be stripped, not
  propagated to a NaN statistic.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.inference.bootstrap import cvar, sharpe_ratio
from src.inference.deflated_sharpe import deflated_sharpe_ratio
from src.selection.fitness import held_out_fitness


@pytest.mark.parametrize("c", [0.0, 0.001, 0.01, -0.02, 1.5])
def test_constant_series_is_not_maximally_significant(c: float) -> None:
    """A constant/near-constant series must not score DSR == 1.0 (the P0-1 bug)."""
    r = np.full(750, c, dtype=float)
    dsr = deflated_sharpe_ratio(r, n_trials=50)
    assert 0.0 <= dsr <= 1.0
    assert dsr < 0.5, f"flat series scored DSR={dsr} (should be < 0.5; was ==1.0 pre-fix)"
    # Sharpe of a flat series is exactly 0 (no excess return per unit of risk).
    assert sharpe_ratio(r) == 0.0
    # The selection signal must not reward a flat candidate.
    assert held_out_fitness(r, n_trials=50) < 0.5


def test_nan_inf_do_not_poison_stats() -> None:
    """Non-finite returns must be stripped, never propagated to a NaN statistic."""
    bad = np.array([0.01, np.nan, -0.02, np.inf, 0.005, -0.011, -np.inf], dtype=float)
    assert np.isfinite(sharpe_ratio(bad))
    assert np.isfinite(cvar(bad, 0.05))
    assert np.isfinite(deflated_sharpe_ratio(bad, n_trials=10))
    assert np.isfinite(held_out_fitness(bad, n_trials=10))


def test_cvar_monotone_in_alpha() -> None:
    """CVaR(alpha) is non-increasing as alpha shrinks (a deeper tail is no better)."""
    rng = np.random.default_rng(0)
    r = rng.standard_normal(2000) * 0.01
    vals = [cvar(r, a) for a in (0.25, 0.10, 0.05, 0.01)]
    assert all(vals[i + 1] <= vals[i] + 1e-12 for i in range(len(vals) - 1))
