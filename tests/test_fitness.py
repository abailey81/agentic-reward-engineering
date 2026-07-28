"""Tests for src/selection/fitness.py — reward-independent selection (F.4, B-3)."""

from __future__ import annotations

import numpy as np
import pytest

from src.selection.fitness import held_out_fitness


def test_fitness_raises_on_training_split_input() -> None:
    """Passing a non-validation split raises (selection lives on validation, B-2/B-3)."""
    rng = np.random.default_rng(0)
    returns = 0.0005 + 0.01 * rng.standard_normal(1000)
    with pytest.raises(ValueError):
        held_out_fitness(returns, n_trials=10, split="train")


def test_fitness_finite_on_validation_returns() -> None:
    """held_out_fitness returns a finite scalar on a validation-returns fixture."""
    rng = np.random.default_rng(1)
    returns = 0.0005 + 0.01 * rng.standard_normal(1000)
    val = held_out_fitness(returns, n_trials=10, split="val")
    assert np.isfinite(val)
    assert isinstance(val, float)


def test_fitness_depends_only_on_returns() -> None:
    """Fitness is a deterministic function of returns alone (no reward value, B-3).

    Re-computing on the same validation returns is byte-identical regardless of
    any external reward scale, because no reward value ever enters the function.
    """
    rng = np.random.default_rng(2)
    returns = 0.0005 + 0.01 * rng.standard_normal(1500)
    first = held_out_fitness(returns, n_trials=20, split="val")
    second = held_out_fitness(returns.copy(), n_trials=20, split="val")
    assert first == second


def test_fitness_penalty_lowers_when_cvar_worse() -> None:
    """lam>0 lowers fitness, and a worse (more negative) CVaR lowers it more."""
    rng = np.random.default_rng(3)
    returns = 0.0005 + 0.01 * rng.standard_normal(2000)

    base = held_out_fitness(returns, n_trials=10, split="val", lam=0.0)
    penalized = held_out_fitness(returns, n_trials=10, split="val", lam=5.0)
    assert penalized < base

    # A series with a fatter left tail (worse CVaR) is penalized more heavily.
    worse = returns.copy()
    worse[:50] -= 0.2  # inject severe losses
    pen_worse = held_out_fitness(worse, n_trials=10, split="val", lam=5.0)
    pen_base = held_out_fitness(returns, n_trials=10, split="val", lam=5.0)
    # The penalty term grows with |CVaR|, so the worse-tail series loses more
    # fitness from the penalty than the base series does.
    from src.inference.bootstrap import cvar
    from src.inference.deflated_sharpe import deflated_sharpe_ratio

    dsr_worse = deflated_sharpe_ratio(worse, 10)
    dsr_base = deflated_sharpe_ratio(returns, 10)
    pen_term_worse = dsr_worse - pen_worse
    pen_term_base = dsr_base - pen_base
    assert pen_term_worse > pen_term_base
    assert abs(cvar(worse, 0.05)) > abs(cvar(returns, 0.05))


def test_fitness_penalty_no_nan_on_empty_series() -> None:
    """lam>0 on an EMPTY validation series must not propagate NaN through the CVaR penalty (Rank 18).

    cvar([]) is non-finite; ``lam*abs(nan)`` would poison fitness with NaN (silently mis-ranking the
    candidate). The guard treats a non-finite CVaR as a zero penalty, so the result mirrors the
    DSR-only (lam=0) value exactly (NaN==NaN compared via np.isnan, finite compared directly).
    """
    empty = np.array([], dtype=float)
    pen = held_out_fitness(empty, n_trials=10, split="val", lam=5.0)
    base = held_out_fitness(empty, n_trials=10, split="val", lam=0.0)
    # No penalty was applied: penalized == base. Either both finite-equal or both NaN.
    assert (np.isnan(pen) and np.isnan(base)) or pen == base


def test_fitness_penalty_no_nan_on_all_nonfinite_series() -> None:
    """lam>0 on an all-non-finite series must not turn a finite/-inf DSR into NaN via the penalty."""
    bad = np.full(64, np.nan)
    pen = held_out_fitness(bad, n_trials=10, split="val", lam=3.0)
    base = held_out_fitness(bad, n_trials=10, split="val", lam=0.0)
    # The penalty contributes 0.0 (CVaR is non-finite), so penalized matches the unpenalized value.
    assert (np.isnan(pen) and np.isnan(base)) or pen == base


def test_fitness_preserves_var_sr_param() -> None:
    """The recent ``var_sr`` kwarg (Rank 16) is still accepted and forwarded (signature preserved)."""
    rng = np.random.default_rng(7)
    returns = 0.0005 + 0.01 * rng.standard_normal(800)
    # Passing an explicit cross-trial Sharpe variance must not raise and must stay finite.
    out = held_out_fitness(returns, n_trials=10, split="val", var_sr=0.25)
    assert np.isfinite(out)
