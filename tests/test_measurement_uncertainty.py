"""Strict tests for the WS5 uncertainty-propagation additions to ReturnDistribution (2026-06-28):
block-bootstrap CVaR confidence intervals, the bootstrap bias estimate, the reliability tier, and the
stationary-block index sampler. These are ADDITIVE + DETERMINISTIC and must NOT change the existing
``tail_stats``/``cvar`` values (backward-compat is asserted explicitly).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.feedback.measurement import ReturnDistribution  # noqa: E402

_CVAR_ATOL = 1e-9


def _series(n: int = 760, seed: int = 7) -> np.ndarray:
    """A realistic, mildly heavy-tailed, time-ordered daily-return series."""
    rng = np.random.default_rng(seed)
    return rng.standard_t(df=5, size=n) * 0.01  # ~1% daily scale, fat tails


# --------------------------------------------------------------------------- #
# Backward-compatibility: the additions must not perturb the point estimators  #
# --------------------------------------------------------------------------- #
def test_additions_do_not_change_point_estimates() -> None:
    """tail_stats + cvar are byte-identical to a fit that never calls the new uncertainty methods."""
    r = _series()
    a = ReturnDistribution().fit(r)
    b = ReturnDistribution().fit(r)
    _ = a.cvar_ci(0.05, n_boot=64, seed=1)  # exercise the new path on `a`
    _ = a.cvar_uncertainty_report(n_boot=64, seed=1)
    assert a.tail_stats() == b.tail_stats()
    for lvl in (0.25, 0.10, 0.05, 0.01):
        assert a.cvar(lvl) == pytest.approx(b.cvar(lvl), abs=0.0)
    assert a._raw is not None and a._raw.shape == (r[np.isfinite(r)].size,)


# --------------------------------------------------------------------------- #
# Reliability tier / exceedance count                                          #
# --------------------------------------------------------------------------- #
def test_exceedance_count_is_ceil_alpha_t() -> None:
    r = _series(n=750)
    d = ReturnDistribution().fit(r)
    assert d.exceedance_count(0.05) == math.ceil(0.05 * d.T)
    assert d.exceedance_count(0.01) == math.ceil(0.01 * d.T)


def test_reliability_tiers_at_boundaries() -> None:
    d = ReturnDistribution().fit(_series(n=750))  # T=750: 0.05->38 (high), 0.01->8 (medium)
    assert d.reliability(0.05) == "high"
    assert d.reliability(0.01) == "medium"
    dsmall = ReturnDistribution().fit(_series(n=120, seed=3))  # 0.01->2 exceedances -> low
    assert dsmall.reliability(0.01) == "low"


# --------------------------------------------------------------------------- #
# Stationary block indices                                                     #
# --------------------------------------------------------------------------- #
def test_block_indices_valid_and_deterministic() -> None:
    t = 200
    i1 = ReturnDistribution._stationary_block_indices(t, 8.0, np.random.default_rng(0))
    i2 = ReturnDistribution._stationary_block_indices(t, 8.0, np.random.default_rng(0))
    assert i1.shape == (t,) and i2.shape == (t,)
    assert i1.min() >= 0 and i1.max() < t
    assert np.array_equal(i1, i2)  # deterministic given an equally-seeded rng


def test_block_length_controls_run_structure() -> None:
    """Larger expected_block ⇒ longer consecutive (mod-t) runs than a near-IID (block≈1) sampler."""
    t = 2000

    def mean_run(idx: np.ndarray) -> float:
        runs, cur = [], 1
        for k in range(1, len(idx)):
            if idx[k] == (idx[k - 1] + 1) % t:
                cur += 1
            else:
                runs.append(cur)
                cur = 1
        runs.append(cur)
        return float(np.mean(runs))

    short = mean_run(ReturnDistribution._stationary_block_indices(t, 1.0, np.random.default_rng(1)))
    long = mean_run(ReturnDistribution._stationary_block_indices(t, 50.0, np.random.default_rng(1)))
    assert long > short


# --------------------------------------------------------------------------- #
# Bootstrap CVaR CI                                                            #
# --------------------------------------------------------------------------- #
def test_cvar_ci_deterministic_given_seed() -> None:
    d = ReturnDistribution().fit(_series())
    a = d.cvar_ci(0.05, n_boot=128, seed=42)
    b = d.cvar_ci(0.05, n_boot=128, seed=42)
    assert a == b  # exact
    c = d.cvar_ci(0.05, n_boot=128, seed=43)
    assert a != c  # a different seed gives a different interval


def test_cvar_ci_is_ordered_finite_and_brackets_point() -> None:
    d = ReturnDistribution().fit(_series())
    point = d.cvar(0.05)
    lo, hi = d.cvar_ci(0.05, n_boot=400, ci=0.90, seed=0)
    assert math.isfinite(lo) and math.isfinite(hi)
    assert lo <= hi
    # The point estimate should sit within a generous bracket of the 90% CI (not a hard coverage claim).
    assert lo - 5e-3 <= point <= hi + 5e-3


def test_cvar_ci_degenerate_on_constant_series() -> None:
    """A constant-return series has zero tail dispersion ⇒ the block bootstrap CI collapses to the point."""
    d = ReturnDistribution().fit(np.full(500, -0.001))
    lo, hi = d.cvar_ci(0.05, n_boot=64, seed=0)
    assert lo == pytest.approx(hi, abs=_CVAR_ATOL)
    assert lo == pytest.approx(d.cvar(0.05), abs=1e-6)


def test_cvar_ci_does_not_mutate_self() -> None:
    d = ReturnDistribution().fit(_series())
    before = (d.xi, d.beta, d.u, d.exceed_frac, d.T)
    d.cvar_ci(0.05, n_boot=64, seed=0)
    d.cvar_bias(0.05, n_boot=64, seed=0)
    assert (d.xi, d.beta, d.u, d.exceed_frac, d.T) == before


# --------------------------------------------------------------------------- #
# Bias estimate + uncertainty report                                           #
# --------------------------------------------------------------------------- #
def test_cvar_bias_finite_and_deterministic() -> None:
    d = ReturnDistribution().fit(_series())
    x = d.cvar_bias(0.05, n_boot=200, seed=5)
    y = d.cvar_bias(0.05, n_boot=200, seed=5)
    assert math.isfinite(x) and x == y


def test_uncertainty_report_shape_and_determinism() -> None:
    d = ReturnDistribution().fit(_series())
    r1 = d.cvar_uncertainty_report(levels=(0.05,), n_boot=128, seed=0)
    r2 = d.cvar_uncertainty_report(levels=(0.05,), n_boot=128, seed=0)
    assert r1 == r2
    cell = r1["cvar_05"]
    assert set(cell) == {"point", "ci_lo", "ci_hi", "bias", "n_exceedances", "reliability"}
    assert cell["ci_lo"] <= cell["ci_hi"]
    assert cell["reliability"] in {"high", "medium", "low"}
    assert cell["n_exceedances"] == float(d.exceedance_count(0.05))


def test_requires_fit_before_bootstrap() -> None:
    d = ReturnDistribution()
    with pytest.raises(RuntimeError):
        d.cvar_ci(0.05)


# --------------------------------------------------------------------------- #
# Mutation-probe gap closers (2026-06-28): EVT-routing boundary + left_tail_mass direction/k.            #
# --------------------------------------------------------------------------- #
def test_cvar_auto_routes_empirical_above_cutoff_and_not_below() -> None:
    """The EVT_ALPHA_CUTOFF routing: alpha=0.10 (> 0.05 cutoff) MUST use the empirical estimator, while
    alpha=0.05 routes through the EVT branch. Widening the cutoff (a mutation) is caught here."""
    d = ReturnDistribution().fit(_series(seed=11))
    # alpha=0.10 is above the cutoff ⇒ auto == empirical (byte-identical).
    assert d.cvar(0.10, method="auto") == pytest.approx(d.cvar(0.10, method="empirical"), abs=0.0)
    # alpha=0.05 is at/below the cutoff ⇒ auto takes the EVT branch (not the empirical one) when in-region.
    assert d.cvar(0.05, method="auto") == pytest.approx(d.cvar(0.05, method="evt"), abs=0.0)


def test_left_tail_mass_direction_and_multiplier() -> None:
    """left_tail_mass = P(return < -k·std), k=2 — a SMALL left-tail probability for a ~symmetric sample, and
    measurably larger for a left-skewed one. Catches (a) flipping the comparison (→ ~0.98) and (b) zeroing k
    (→ P(return<0) ≈ 0.5)."""
    rng = np.random.default_rng(0)
    sym = rng.standard_normal(2000) * 0.01
    ltm_sym = ReturnDistribution().fit(sym).tail_stats()["left_tail_mass"]
    assert 0.0 < ltm_sym < 0.15  # ~P(N<-2σ)≈0.023; a flip→~0.98 or k=0→~0.5 both blow this bound
    # Left-skew (inject a few deep negatives) ⇒ strictly more mass below -2σ than the symmetric sample.
    skewed = sym.copy()
    skewed[:60] = -0.08
    ltm_skew = ReturnDistribution().fit(skewed).tail_stats()["left_tail_mass"]
    assert ltm_skew > ltm_sym
