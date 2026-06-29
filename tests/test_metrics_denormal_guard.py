"""Regression: profit_factor / gain_loss_ratio must honour the 'finite sentinels, never crash' contract
even under a DENORMAL-magnitude loss.

Surfaced by the deep backtest-metrics property sweep (2026-06-28): the two ratios previously guarded their
divisor with a bare ``loss.size`` / ``loss.sum() != 0`` test rather than the module-wide ``> _EPS`` magnitude
convention, so a denormal loss (|loss| ~ 1e-310) slipped past the guard and overflowed the ratio to +inf.
``src/backtest/metrics.py`` now guards by magnitude; these tests lock that fix.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backtest.metrics import compute_metrics  # noqa: E402


def test_denormal_loss_does_not_overflow_profit_ratios_to_inf() -> None:
    """A series of real gains plus a single denormal-magnitude loss yields FINITE ratios.

    Pre-fix this overflowed: wins.sum()/abs(loss.sum()) with abs(loss.sum()) ~ 1e-320 -> +inf. With the
    ``> _EPS`` magnitude guard the negligible loss is treated as "no material loss", so both ratios take the
    documented all-wins sentinel (1e9) rather than +inf.
    """
    returns = np.array([0.01, 0.02, 0.015, 0.03, -1e-320], dtype=float)  # last entry is a denormal loss
    m = compute_metrics(returns)
    assert np.isfinite(m["profit_factor"]), f"profit_factor overflowed: {m['profit_factor']!r}"
    assert np.isfinite(m["gain_loss_ratio"]), f"gain_loss_ratio overflowed: {m['gain_loss_ratio']!r}"
    # Negligible (sub-_EPS) loss + real wins == the all-wins sentinel, consistent with omega/no-loss handling.
    assert m["profit_factor"] == 1e9
    assert m["gain_loss_ratio"] == 1e9


def test_material_losses_still_give_a_finite_non_sentinel_ratio() -> None:
    """A normal mixed series still produces the genuine finite ratio (the guard didn't break the happy path)."""
    returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01], dtype=float)
    m = compute_metrics(returns)
    for key in ("profit_factor", "gain_loss_ratio"):
        assert np.isfinite(m[key])
        assert 0.0 < m[key] < 1e9, f"{key} should be a genuine finite ratio, got {m[key]!r}"


def test_no_wins_series_is_zero_not_sentinel() -> None:
    """An all-losses series keeps the documented 0.0 (reserved for the no-wins case), still finite."""
    returns = np.array([-0.01, -0.02, -0.005], dtype=float)
    m = compute_metrics(returns)
    assert m["profit_factor"] == 0.0
    assert m["gain_loss_ratio"] == 0.0


def test_exact_zero_return_periods_are_neither_wins_nor_losses() -> None:
    """Inserting exact-zero-return periods must NOT change the win/loss-split metrics.

    Mutation-probe gap (2026-06-28): the win/loss split is ``r > 0`` / ``r < 0``; a boundary mutation to
    ``>= 0`` / ``<= 0`` puts a zero period into BOTH wins and losses, but no prior fixture contained an
    EXACT 0.0 return, so the mutant survived. A flat (exactly 0.0) period is neither a gain nor a loss, so
    `gain_loss_ratio` (a mean over wins / mean over losses) must be invariant to inserting zeros — this
    metamorphic identity fails under the boundary mutation, killing it.
    """
    base = np.array([0.02, -0.01, 0.03], dtype=float)
    with_zeros = np.array([0.02, 0.0, -0.01, 0.0, 0.03], dtype=float)
    mb, mz = compute_metrics(base), compute_metrics(with_zeros)
    assert mz["gain_loss_ratio"] == mb["gain_loss_ratio"]  # zeros excluded from the loss-mean denominator
    assert mz["profit_factor"] == mb["profit_factor"]      # and from both sums
    # And a zero period is not counted as a winning day.
    assert mz["hit_rate"] == 2.0 / 5.0
