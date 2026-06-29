"""Property-based + metamorphic tests for the inference primitives and the reward contract.

Where the rest of the suite checks *examples* (this input -> that output), this module checks
*laws* — algebraic identities that must hold for ALL admissible inputs, generated adversarially by
Hypothesis. Three reasons this matters for the dissertation specifically:

1. **The CVaR estimator is the headline tail statistic**, and the theory (Chapter 3) leans on CVaR
   being a *coherent* risk measure. Artzner, Delbaen, Eber & Heath (1999) give **FOUR** axioms:
   translation equivariance, positive homogeneity, monotonicity, and **SUB-ADDITIVITY**. Here we
   *prove the implementation satisfies those axioms* over thousands of generated samples — i.e. the
   code realises the object the theory assumes, not merely a function that happens to agree on a few
   fixtures.

   NOTE: Corrected 2026-07-26 (deep review): this paragraph previously enumerated only the first
   THREE axioms, and sub-additivity was untested anywhere in ``tests/``. That is the one axiom the
   whole argument turns on — it is precisely what CVaR has and VaR lacks, and it is the reason
   Chapter 3 can use CVaR as a coherent risk measure at all. A measure-theoretic examiner checks it
   first. ``test_cvar_subadditivity`` below now covers it.
2. **Metamorphic relations** (scale-invariance of Sharpe, sqrt-time annualisation, sign reflection)
   catch whole classes of bug that example tests miss, with no need for a known ground-truth output.
3. **Reproducibility**: the Hypothesis profile is *derandomised* (tests/conftest.py), so any failing
   example is a stable, replayable counter-example — consistent with the project's determinism contract.

Hypothesis is an optional dev dependency; the module skips cleanly if it is absent.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("hypothesis")

from hypothesis import assume, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from hypothesis.extra.numpy import arrays  # noqa: E402

from src.inference.bootstrap import (  # noqa: E402
    cvar,
    iqm,
    paired_seed_difference_test,
    sharpe_difference_test,
    sharpe_ratio,
    stationary_bootstrap_indices,
)
from src.reward.contract import validate_signature

# --------------------------------------------------------------------------- #
# Strategies                                                                   #
# --------------------------------------------------------------------------- #
#: Finite, bounded return arrays. Magnitude capped well inside float64's exact-integer range so the
#: algebraic identities below are not contaminated by catastrophic-cancellation float error.
_FINITE = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False, width=64)
_POS_SCALE = st.floats(min_value=1e-2, max_value=1e2, allow_nan=False, allow_infinity=False)
_ANY_SCALE = st.floats(min_value=-1e2, max_value=1e2, allow_nan=False, allow_infinity=False)
_SHIFT = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)
_ALPHA = st.floats(min_value=1e-3, max_value=1.0, allow_nan=False, allow_infinity=False)


def _returns(min_size: int = 1, max_size: int = 400):
    """A 1-D float64 array strategy of finite returns."""
    return arrays(dtype=np.float64, shape=st.integers(min_size, max_size), elements=_FINITE)


def _nondegenerate(r: np.ndarray, *, min_std: float = 1e-2) -> bool:
    """True when ``r`` is clear of the Sharpe near-constant guard region."""
    r = np.asarray(r, dtype=float)
    return bool(np.ptp(r) > 0.0 and float(r.std(ddof=0)) > min_std)


# --------------------------------------------------------------------------- #
# CVaR — the coherent-risk axioms (ties directly to the Chapter-3 theory)      #
# --------------------------------------------------------------------------- #
class TestCVaRCoherenceAxioms:
    @settings(max_examples=200)
    @given(
        x=arrays(np.float64, 60, elements=st.floats(-0.2, 0.2, allow_nan=False, allow_infinity=False)),
        y=arrays(np.float64, 60, elements=st.floats(-0.2, 0.2, allow_nan=False, allow_infinity=False)),
        alpha=st.sampled_from([0.05, 0.10, 0.25, 0.50]),
    )
    def test_cvar_subadditivity(self, x: np.ndarray, y: np.ndarray, alpha: float) -> None:
        """SUB-ADDITIVITY — the fourth Artzner axiom, and the one the whole theory turns on.

        Diversification may not increase risk: for a risk measure rho, rho(X+Y) <= rho(X) + rho(Y).
        This is exactly what CVaR has and VaR lacks (VaR fails sub-additivity
        specifically), and it is why Chapter 3 may treat CVaR as coherent.

        SIGN CONVENTION: ``bootstrap.cvar`` returns the SIGNED lower-tail mean of RETURNS (more
        negative = worse), so the risk measure is rho(X) = -cvar(X) and sub-additivity becomes

            cvar(X + Y) >= cvar(X) + cvar(Y).

        WHY IT HOLDS FOR THIS ESTIMATOR, exactly and not just asymptotically: ``cvar`` is the mean of
        the worst k = ceil(alpha*T) observations, i.e. (1/k)*min over k-subsets S of sum_{i in S} z_i.
        Let S* be the minimising subset for X+Y. Then
            k*cvar(X+Y) = sum_{S*}(x+y) = sum_{S*}x + sum_{S*}y >= min_S sum_S x + min_S sum_S y,
        because S* is merely *a* feasible k-subset for each term. Equality holds iff one subset is
        worst for both. So the property is exact for equal-length finite samples, and a tiny
        floating-point tolerance is all that is needed.

        Added 2026-07-26 (deep review): the module previously enumerated only three of the
        four axioms and never tested this one.
        """
        assume(np.all(np.isfinite(x)) and np.all(np.isfinite(y)))
        lhs = cvar(x + y, alpha)
        rhs = cvar(x, alpha) + cvar(y, alpha)
        assert lhs >= rhs - 1e-12, (
            f"CVaR sub-additivity violated at alpha={alpha}: cvar(X+Y)={lhs!r} < "
            f"cvar(X)+cvar(Y)={rhs!r} — the estimator would not be a coherent risk measure"
        )

    @given(r=_returns(min_size=1))
    def test_alpha_one_equals_mean(self, r: np.ndarray) -> None:
        """CVaR at alpha=1 averages the WHOLE sample -> it is exactly the mean."""
        assert math.isclose(cvar(r, 1.0), float(r.mean()), rel_tol=1e-9, abs_tol=1e-12)

    @given(r=_returns(min_size=1), a1=_ALPHA, a2=_ALPHA)
    def test_monotone_nondecreasing_in_alpha(self, r: np.ndarray, a1: float, a2: float) -> None:
        """A wider tail (larger alpha) can only RAISE the tail mean: CVaR is non-decreasing in alpha."""
        lo, hi = sorted((a1, a2))
        assert cvar(r, lo) <= cvar(r, hi) + 1e-9

    @given(r=_returns(min_size=1), alpha=_ALPHA, c=_SHIFT)
    def test_translation_equivariance(self, r: np.ndarray, alpha: float, c: float) -> None:
        """Coherence axiom: CVaR(r + c) == CVaR(r) + c (shifting every return shifts the tail mean)."""
        assert np.isclose(cvar(r + c, alpha), cvar(r, alpha) + c, rtol=1e-6, atol=1e-6)

    @given(r=_returns(min_size=1), alpha=_ALPHA, k=_POS_SCALE)
    def test_positive_homogeneity(self, r: np.ndarray, alpha: float, k: float) -> None:
        """Coherence axiom: CVaR(k*r) == k*CVaR(r) for k >= 0 (positive scaling of returns)."""
        assert np.isclose(cvar(k * r, alpha), k * cvar(r, alpha), rtol=1e-6, atol=1e-9)

    @given(r=_returns(min_size=1), alpha=_ALPHA)
    def test_bounded_between_min_and_mean(self, r: np.ndarray, alpha: float) -> None:
        """The worst-k mean lies between the global minimum and the global mean."""
        assert float(r.min()) - 1e-9 <= cvar(r, alpha) <= float(r.mean()) + 1e-9

    @given(r=_returns(min_size=1), delta=arrays(np.float64, st.integers(1, 1), elements=_FINITE))
    def test_monotonicity_under_dominance(self, r: np.ndarray, delta: np.ndarray) -> None:
        """Coherence axiom: if a <= b elementwise then CVaR(a) <= CVaR(b) (dominated returns -> worse tail)."""
        d = np.abs(np.resize(delta, r.shape))  # b = a + |delta| >= a elementwise
        for alpha in (0.05, 0.25, 1.0):
            assert cvar(r, alpha) <= cvar(r + d, alpha) + 1e-9

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.0 + 1e-9, 2.0, float("nan"), float("inf")])
    def test_alpha_out_of_range_raises(self, bad: float) -> None:
        with pytest.raises(ValueError):
            cvar(np.array([0.1, -0.2, 0.3]), bad)


# --------------------------------------------------------------------------- #
# Sharpe ratio — metamorphic relations                                        #
# --------------------------------------------------------------------------- #
class TestSharpeMetamorphic:
    @given(r=_returns(min_size=2), k=_POS_SCALE)
    def test_scale_invariance(self, r: np.ndarray, k: float) -> None:
        """Sharpe = mean/std is invariant to a positive rescale of returns (both scale by k)."""
        assume(_nondegenerate(r))
        assert np.isclose(sharpe_ratio(k * r), sharpe_ratio(r), rtol=1e-6, atol=1e-9)

    @given(r=_returns(min_size=2))
    def test_sign_reflection(self, r: np.ndarray) -> None:
        """Negating returns negates the Sharpe ratio."""
        assume(_nondegenerate(r))
        assert np.isclose(sharpe_ratio(-r), -sharpe_ratio(r), rtol=1e-6, atol=1e-9)

    @given(r=_returns(min_size=2), k=st.integers(1, 365))
    def test_annualisation_scales_as_sqrt(self, r: np.ndarray, k: int) -> None:
        """The annualisation factor enters as sqrt(periods_per_year)."""
        assume(_nondegenerate(r))
        base = sharpe_ratio(r, periods_per_year=1)
        assert np.isclose(sharpe_ratio(r, periods_per_year=k), base * math.sqrt(k), rtol=1e-6, atol=1e-9)

    @given(c=_SHIFT, n=st.integers(2, 200))
    def test_constant_series_is_zero(self, c: float, n: int) -> None:
        """A (near-)constant series has zero dispersion -> the guard returns exactly 0.0."""
        assert sharpe_ratio(np.full(n, c)) == 0.0

    @given(r=_returns(min_size=2), c=st.floats(1e-3, 1e2, allow_nan=False, allow_infinity=False))
    def test_adding_positive_drift_does_not_decrease_sharpe(self, r: np.ndarray, c: float) -> None:
        """Adding a positive constant lifts the mean while leaving the std unchanged -> Sharpe rises."""
        assume(_nondegenerate(r))
        assert sharpe_ratio(r + c) >= sharpe_ratio(r) - 1e-9


# --------------------------------------------------------------------------- #
# IQM — robust-location equivariances                                         #
# --------------------------------------------------------------------------- #
class TestIQMProperties:
    @given(x=arrays(np.float64, st.integers(1, 3), elements=_FINITE))
    def test_small_sample_falls_back_to_mean(self, x: np.ndarray) -> None:
        """Fewer than 4 values -> the interquartile trim is ill-defined, so IQM == plain mean."""
        assert math.isclose(iqm(x), float(x.mean()), rel_tol=1e-9, abs_tol=1e-12)

    @given(x=_returns(min_size=1))
    def test_bounded_by_min_and_max(self, x: np.ndarray) -> None:
        assert float(x.min()) - 1e-9 <= iqm(x) <= float(x.max()) + 1e-9

    @given(x=_returns(min_size=1), perm_seed=st.integers(0, 2**32 - 1))
    def test_permutation_invariance(self, x: np.ndarray, perm_seed: int) -> None:
        """IQM is a function of the multiset of values, not their order."""
        shuffled = np.random.default_rng(perm_seed).permutation(x)
        assert math.isclose(iqm(x), iqm(shuffled), rel_tol=1e-9, abs_tol=1e-9)

    @given(x=_returns(min_size=1), c=_SHIFT)
    def test_translation_equivariance(self, x: np.ndarray, c: float) -> None:
        assert np.isclose(iqm(x + c), iqm(x) + c, rtol=1e-6, atol=1e-6)

    @given(x=_returns(min_size=1), k=_ANY_SCALE)
    def test_scale_equivariance(self, x: np.ndarray, k: float) -> None:
        """IQM(k*x) == k*IQM(x) for ANY k (the central trim window is symmetric, so a sign flip is fine)."""
        assert np.isclose(iqm(k * x), k * iqm(x), rtol=1e-6, atol=1e-9)


# --------------------------------------------------------------------------- #
# Stationary bootstrap index path                                             #
# --------------------------------------------------------------------------- #
class TestStationaryBootstrapIndices:
    @given(n=st.integers(1, 500), p=st.floats(1e-3, 1.0, allow_nan=False, allow_infinity=False))
    def test_shape_and_range(self, n: int, p: float) -> None:
        idx = stationary_bootstrap_indices(n, p, np.random.default_rng(0))
        assert idx.shape == (n,)
        assert int(idx.min()) >= 0 and int(idx.max()) < n

    @given(n=st.integers(1, 200), p=st.floats(1e-3, 1.0, allow_nan=False, allow_infinity=False), s=st.integers(0, 10**6))
    def test_deterministic_under_fixed_seed(self, n: int, p: float, s: int) -> None:
        a = stationary_bootstrap_indices(n, p, np.random.default_rng(s))
        b = stationary_bootstrap_indices(n, p, np.random.default_rng(s))
        assert np.array_equal(a, b)

    @pytest.mark.parametrize("n,p", [(0, 0.5), (-1, 0.5), (10, 0.0), (10, -0.1), (10, 1.0001)])
    def test_invalid_args_raise(self, n: int, p: float) -> None:
        with pytest.raises(ValueError):
            stationary_bootstrap_indices(n, p, np.random.default_rng(0))


# --------------------------------------------------------------------------- #
# Paired / two-sample difference tests — invariants of the inference contract  #
# --------------------------------------------------------------------------- #
class TestDifferenceTestInvariants:
    @given(
        ab=arrays(np.float64, st.integers(2, 60), elements=_FINITE).flatmap(
            lambda a: st.tuples(st.just(a), arrays(np.float64, a.shape, elements=_FINITE))
        )
    )
    @settings(max_examples=60)
    def test_effect_is_antisymmetric(self, ab: tuple[np.ndarray, np.ndarray]) -> None:
        """The point effect is statistic(a)-statistic(b); swapping arms negates it EXACTLY (no RNG needed)."""
        a, b = ab
        rng = np.random.default_rng(0)
        eff_ab = paired_seed_difference_test(a, b, n_boot=50, rng=np.random.default_rng(0))["effect"]
        eff_ba = paired_seed_difference_test(b, a, n_boot=50, rng=rng)["effect"]
        assert np.isclose(eff_ab, -eff_ba, rtol=1e-9, atol=1e-12)

    @given(a=arrays(np.float64, st.integers(2, 80), elements=_FINITE))
    @settings(max_examples=60)
    def test_identical_arms_have_zero_effect(self, a: np.ndarray) -> None:
        """Comparing an arm to itself gives an exactly-zero point effect."""
        res = paired_seed_difference_test(a, a.copy(), n_boot=50, rng=np.random.default_rng(1))
        assert res["effect"] == 0.0

    @given(
        ab=arrays(np.float64, st.integers(8, 60), elements=_FINITE).flatmap(
            lambda a: st.tuples(st.just(a), arrays(np.float64, a.shape, elements=_FINITE))
        )
    )
    @settings(max_examples=40)
    def test_pvalue_is_a_probability(self, ab: tuple[np.ndarray, np.ndarray]) -> None:
        """Every reported p-value is a finite-resolution probability in [1/(n_boot+1), 1]."""
        a, b = ab
        n_boot = 200
        for res in (
            paired_seed_difference_test(a, b, n_boot=n_boot, rng=np.random.default_rng(2)),
            sharpe_difference_test(a, b, n_boot=n_boot, rng=np.random.default_rng(2)),
        ):
            assert 1.0 / (n_boot + 1) - 1e-12 <= res["pvalue"] <= 1.0

    @given(a=arrays(np.float64, st.integers(8, 50), elements=_FINITE), s=st.integers(0, 10**6))
    @settings(max_examples=30)
    def test_pvalue_deterministic_under_fixed_seed(self, a: np.ndarray, s: int) -> None:
        """Same data + same seed -> identical p-value (reproducible inference)."""
        b = a[::-1].copy()
        r1 = sharpe_difference_test(a, b, n_boot=100, rng=np.random.default_rng(s))
        r2 = sharpe_difference_test(a, b, n_boot=100, rng=np.random.default_rng(s))
        assert r1["pvalue"] == r2["pvalue"] and r1["stat"] == r2["stat"]

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            paired_seed_difference_test(np.zeros(5), np.zeros(6))
        with pytest.raises(ValueError):
            sharpe_difference_test(np.zeros(5), np.zeros(6))

    def test_one_sided_p_is_direct_and_directional(self) -> None:
        """R64: the paired test returns a DIRECT upper-tail one-sided p (``P(boot - obs >= obs)``), not
        the symmetry-assuming ``p_two / 2``. Properties: it is a probability; a strong positive effect
        drives it small and at/below the two-sided p; a strong negative effect drives it above 0.5 (so a
        wrong-direction estimate cannot reject H1: a>b); and on a ~symmetric (mean) bootstrap it reduces
        to ~ ``p_two / 2`` — the fix only DIFFERS where the bootstrap is skewed (the CVaR leg)."""
        n_boot = 4000
        # independent per-arm draws so the paired bootstrap of the mean-difference is non-degenerate
        a_pos = np.random.default_rng(11).normal(2.0, 1.0, size=40)
        b_pos = np.random.default_rng(12).normal(0.0, 1.0, size=40)
        r_pos = paired_seed_difference_test(a_pos, b_pos, statistic=np.mean, n_boot=n_boot,
                                            rng=np.random.default_rng(1))
        assert "pvalue_one_sided_greater" in r_pos
        p1 = r_pos["pvalue_one_sided_greater"]
        assert 1.0 / (n_boot + 1) - 1e-12 <= p1 <= 1.0          # a probability
        assert r_pos["effect"] > 0 and p1 <= r_pos["pvalue"] + 1e-9 and p1 < 0.05  # small, <= two-sided
        # wrong direction (effect < 0): upper-tail p must be large
        r_neg = paired_seed_difference_test(b_pos, a_pos, statistic=np.mean, n_boot=n_boot,
                                            rng=np.random.default_rng(2))
        assert r_neg["effect"] < 0 and r_neg["pvalue_one_sided_greater"] > 0.5
        # ~symmetric (mean) case reduces to ~ p_two/2 (the two agree within MC noise; they only DIVERGE
        # on a skewed bootstrap — the CVaR leg — which is the entire point of the R64 fix).
        a_sym = np.random.default_rng(13).normal(0.45, 1.0, size=40)
        b_sym = np.random.default_rng(14).normal(0.0, 1.0, size=40)
        r_sym = paired_seed_difference_test(a_sym, b_sym, statistic=np.mean, n_boot=n_boot,
                                            rng=np.random.default_rng(3))
        assert r_sym["pvalue_one_sided_greater"] <= r_sym["pvalue"] + 1e-9
        assert r_sym["pvalue_one_sided_greater"] == pytest.approx(r_sym["pvalue"] / 2.0, abs=0.01)


# --------------------------------------------------------------------------- #
# Reward contract signature validator — adversarial signature space            #
# --------------------------------------------------------------------------- #
class TestRewardContractSignature:
    def test_accepts_the_exact_contract(self) -> None:
        def reward(weights, returns, prev_weights, port_ret, info):  # noqa: ANN001, ARG001
            return 0.0, {}, None

        assert validate_signature(reward) is True

    @pytest.mark.parametrize(
        "fn",
        [
            lambda weights, returns, prev_weights, port_ret: None,  # too few params
            lambda weights, returns, prev_weights, port_ret, info, extra: None,  # too many
            lambda w, returns, prev_weights, port_ret, info: None,  # wrong first name
            lambda returns, weights, prev_weights, port_ret, info: None,  # wrong order
            lambda *args: None,  # var-positional
            lambda **kwargs: None,  # var-keyword
            42,  # not callable
            "reward",  # not callable
        ],
    )
    def test_rejects_non_conforming(self, fn: object) -> None:
        assert validate_signature(fn) is False

    def test_rejects_keyword_only_params(self) -> None:
        def reward(weights, returns, prev_weights, port_ret, *, info):  # noqa: ANN001, ARG001
            return 0.0, {}, None

        assert validate_signature(reward) is False
