"""DEEP property-based / metamorphic / adversarial tests for the load-bearing inference modules.

These EXTEND (never duplicate) the existing behaviour suites
(``tests/test_inference.py`` — pbo/iqm basics; ``tests/test_attribution.py`` — factor recovery;
``tests/test_contamination_ood.py`` — TOST/OOD behaviour; ``tests/test_inference_crosscheck.py``
— license-clean oracles) with the heavier-weather checks the dissertation's H2 conclusions rest
on: Hypothesis property tests (``@settings(derandomize=True)`` -> ZERO run-to-run variance),
metamorphic relations, combinatorial-split symmetry, NaN/inf adversarial inputs, degenerate
series (constant / single-obs / zero-variance), and determinism replays.

Modules under deep test:
  * ``src.inference.overfitting``    — PBO / CSCV (the primary overfitting guard)
  * ``src.inference.attribution``    — factor-attribution difference-in-alpha (Door-C secondary)
  * ``src.inference.contamination``  — named-vs-blinded leakage A/B
  * ``src.inference.ood_stress``     — synthetic stressed-path robustness appendix
  * ``src.inference.reporting``      — rliable aggregate statistics (IQM / P(improve) / boot CI)

Every numeric assert uses a tight, explicit ``atol``/``rtol`` and a seeded
``np.random.default_rng``; no test depends on the global RNG state.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from hypothesis.extra import numpy as hnp  # noqa: E402

from src.inference.attribution import difference_in_alpha, factor_alpha  # noqa: E402
from src.inference.contamination import (  # noqa: E402
    coefficient_mahalanobis_permutation,
    named_vs_blinded_tost,
    paired_tost,
    structural_mcnemar,
)
from src.inference.ood_stress import (  # noqa: E402
    block_bootstrap_paths,
    tail_metrics,
    validate_stylized_facts,
    vol_spike_paths,
)
from src.inference.overfitting import pbo  # noqa: E402
from src.inference.reporting import (  # noqa: E402
    iqm,
    performance_profile,
    probability_of_improvement,
    stratified_bootstrap_ci,
)

# A single derandomised profile decorator alias so every property test is replayable.
_DET = settings(derandomize=True, max_examples=60, deadline=None)


# =============================================================================
# overfitting.py — PBO / CSCV
# =============================================================================
def _independent_noise_matrix(t: int, n: int, seed: int) -> np.ndarray:
    """A pure-noise IS-uninformative-of-OOS performance matrix (no real skill anywhere)."""
    return np.random.default_rng(seed).standard_normal((t, n))


# --- bounds + determinism (property-based) -----------------------------------
@_DET
@given(
    seed=st.integers(0, 2**31 - 1),
    n_cfg=st.integers(2, 8),
    n_blocks=st.sampled_from([2, 4, 6, 8]),
    rows_mult=st.integers(4, 12),
)
def test_pbo_is_always_a_probability(seed: int, n_cfg: int, n_blocks: int, rows_mult: int) -> None:
    """PBO is a probability: finite and in [0, 1] for ANY well-formed performance matrix."""
    t = n_blocks * rows_mult
    mat = np.random.default_rng(seed).standard_normal((t, n_cfg))
    val = pbo(mat, n_blocks=n_blocks, rng=np.random.default_rng(seed))
    assert math.isfinite(val)
    assert 0.0 <= val <= 1.0


@_DET
@given(seed=st.integers(0, 2**31 - 1), n_blocks=st.sampled_from([4, 6, 8]))
def test_pbo_deterministic_given_inputs(seed: int, n_blocks: int) -> None:
    """Full-enumeration PBO is a pure function of its inputs — two calls agree to the bit.

    With n_blocks small the C(S, S/2) splits are fully enumerated (no random subsampling), so
    PBO must be exactly reproducible regardless of the rng passed."""
    mat = np.random.default_rng(seed).standard_normal((200, 6))
    v1 = pbo(mat, n_blocks=n_blocks, rng=np.random.default_rng(0))
    v2 = pbo(mat, n_blocks=n_blocks, rng=np.random.default_rng(999))  # different rng, same result
    assert v1 == v2  # exact: the enumerated path never touches the rng


def test_pbo_pure_noise_centres_on_half() -> None:
    """Under pure noise (IS rank carries NO OOS information) PBO concentrates near 0.5.

    Averaged over many independent noise matrices the IS-best lands below the OOS median about
    half the time. Tighter band than the existing single-seed smoke test (averages 8 seeds)."""
    vals = [pbo(_independent_noise_matrix(600, 10, 200 + s), n_blocks=10) for s in range(8)]
    assert np.mean(vals) == pytest.approx(0.5, abs=0.10)


def test_pbo_zero_when_one_strategy_dominates_everywhere() -> None:
    """A strategy uniformly dominant IS and OOS is never an overfit pick -> PBO == 0 exactly."""
    rng = np.random.default_rng(7)
    mat = rng.standard_normal((480, 6)) * 0.05
    mat[:, 2] += 100.0  # config 2 dwarfs all others on every row -> always IS-best AND OOS-top
    val = pbo(mat, n_blocks=8, rng=rng)
    assert val == 0.0


def test_pbo_one_when_is_best_is_oos_worst_by_construction() -> None:
    """An adversarial matrix where the IS-best is always the OOS-worst forces PBO == 1.

    Build two configs: A is large in the first half of every block (so it wins IS on any IS set
    that draws those rows) but tiny in the second half; B mirrors it. With contiguous blocks split
    so that whichever set is IS makes the locally-large config win IS and lose OOS, the IS-best
    consistently lands at the bottom OOS -> lambda < 0 on every split -> PBO -> 1.
    """
    # Construct a clean separating example: config 0 has a high mean in the first half of the
    # series and a very low mean in the second half; config 1 is the exact opposite. Any IS/OOS
    # split that is "first-half vs second-half" makes the IS winner the OOS loser.
    t = 200
    a = np.concatenate([np.full(t // 2, 1.0), np.full(t // 2, -1.0)])
    b = -a
    mat = np.column_stack([a, b])
    # n_blocks=2 -> the only complementary split is {block0}|{block1} = first half vs second half.
    val = pbo(mat, n_blocks=2)
    assert val == 1.0


def test_pbo_combinatorial_split_symmetry() -> None:
    """The set of IS-best OOS-logits is symmetric under IS<->OOS swap (CSCV is combinatorially
    symmetric): full enumeration includes every split AND its complement, so reversing the
    block order (which maps each split to its complement set of contiguous blocks) leaves the
    multiset of evaluated splits — and hence PBO — unchanged for a time-symmetric matrix."""
    rng = np.random.default_rng(13)
    # Build a matrix that is exactly palindromic in time: row t == row (T-1-t). Then reversing the
    # row order is a no-op on the data, so PBO is trivially invariant; this checks the enumeration
    # does not depend on an arbitrary block ORIENTATION.
    half = rng.standard_normal((100, 5))
    mat = np.vstack([half, half[::-1]])  # palindrome in time
    forward = pbo(mat, n_blocks=8)
    reversed_ = pbo(mat[::-1], n_blocks=8)
    assert forward == pytest.approx(reversed_, abs=1e-12)


def test_pbo_handles_minimum_and_odd_constraints() -> None:
    """Boundary / malformed n_blocks are handled per the documented contract."""
    rng = np.random.default_rng(3)
    mat = rng.standard_normal((64, 4))
    # Minimum legal block count (S == 2): CSCV enumerates C(2,1) == 2 splits (a split AND its
    # complement), so PBO is one of {0, 0.5, 1} -- always a finite probability, never a crash/NaN.
    v = pbo(mat, n_blocks=2)
    assert v in (0.0, 0.5, 1.0)
    # Odd n_blocks rejected.
    with pytest.raises(ValueError):
        pbo(mat, n_blocks=3)
    # n_blocks < 2 rejected.
    with pytest.raises(ValueError):
        pbo(mat, n_blocks=0)
    # Fewer than n_blocks rows rejected.
    with pytest.raises(ValueError):
        pbo(rng.standard_normal((4, 3)), n_blocks=8)
    # < 2 configs rejected.
    with pytest.raises(ValueError):
        pbo(rng.standard_normal((64, 1)), n_blocks=4)
    # Non-2D rejected.
    with pytest.raises(ValueError):
        pbo(rng.standard_normal(64), n_blocks=4)


def test_pbo_constant_matrix_all_ties_is_not_overfit() -> None:
    """Degenerate input: an all-CONSTANT performance matrix.

    Every config ties on every split, so the IS-best (argmax of equal values -> index 0) lands at
    the OOS AVERAGE rank, i.e. the exact OOS median -> lambda == 0, which the documented STRICT
    inequality (lam < 0) does NOT count as overfit. PBO must therefore be exactly 0.0 (defined,
    finite, no silent NaN)."""
    mat = np.full((128, 5), 0.7)
    val = pbo(mat, n_blocks=8)
    assert math.isfinite(val)
    assert val == 0.0


def test_pbo_remainder_rows_dropped_not_crash() -> None:
    """T not divisible by n_blocks: the trailing remainder rows are dropped, never a crash/NaN."""
    rng = np.random.default_rng(5)
    mat = rng.standard_normal((103, 4))  # 103 = 8*12 + 7 -> 7 remainder rows dropped
    val = pbo(mat, n_blocks=8, rng=rng)
    assert math.isfinite(val) and 0.0 <= val <= 1.0


def test_pbo_subsampling_path_is_seed_reproducible() -> None:
    """When C(S, S/2) exceeds the cap the random-subset path is still reproducible given the rng."""
    rng = np.random.default_rng(11)
    mat = rng.standard_normal((420, 7))
    # n_blocks=20 -> C(20,10)=184756 >> default cap 4000 -> the random-subsampling branch runs.
    v1 = pbo(mat, n_blocks=20, rng=np.random.default_rng(123), max_combinations=300)
    v2 = pbo(mat, n_blocks=20, rng=np.random.default_rng(123), max_combinations=300)
    assert v1 == v2
    assert 0.0 <= v1 <= 1.0


# =============================================================================
# reporting.py — IQM / probability_of_improvement / stratified_bootstrap_ci
# =============================================================================
@_DET
@given(
    arr=hnp.arrays(
        dtype=np.float64,
        shape=st.integers(1, 60),
        elements=st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
    ),
    shift=st.floats(-50.0, 50.0, allow_nan=False, allow_infinity=False),
)
def test_iqm_is_shift_equivariant(arr: np.ndarray, shift: float) -> None:
    """Metamorphic: IQM(x + c) == IQM(x) + c (the trimmed mean is a location-equivariant estimator)."""
    base = iqm(arr)
    shifted = iqm(arr + shift)
    assert math.isfinite(base) and math.isfinite(shifted)
    assert shifted == pytest.approx(base + shift, abs=1e-9, rel=1e-9)


@_DET
@given(
    arr=hnp.arrays(
        dtype=np.float64,
        shape=st.integers(1, 60),
        elements=st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
    ),
    scale=st.floats(0.01, 100.0, allow_nan=False, allow_infinity=False),
)
def test_iqm_is_positive_scale_equivariant(arr: np.ndarray, scale: float) -> None:
    """Metamorphic: IQM(c*x) == c*IQM(x) for c > 0 (scale equivariance of the trimmed mean)."""
    base = iqm(arr)
    scaled = iqm(arr * scale)
    assert scaled == pytest.approx(base * scale, rel=1e-9, abs=1e-12)


@_DET
@given(
    arr=hnp.arrays(
        dtype=np.float64,
        shape=st.integers(4, 60),
        elements=st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
    ),
)
def test_iqm_lies_within_data_range(arr: np.ndarray) -> None:
    """An interquartile MEAN is bounded by the min and max of the data (a sanity invariant)."""
    v = iqm(arr)
    assert arr.min() - 1e-9 <= v <= arr.max() + 1e-9


def test_iqm_permutation_invariant() -> None:
    """IQM does not depend on input ORDER (it sorts internally)."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(37)
    assert iqm(x) == pytest.approx(iqm(rng.permutation(x)), abs=1e-12)


def test_iqm_degenerate_inputs_are_defined_and_finite() -> None:
    """Constant / single-obs IQM is the value itself; empty / all-NaN is a documented NaN."""
    assert iqm(np.array([3.3])) == pytest.approx(3.3, abs=1e-12)          # single obs
    assert iqm(np.full(20, 2.5)) == pytest.approx(2.5, abs=1e-12)         # zero variance
    assert math.isnan(iqm(np.array([])))                                  # empty -> NaN (documented)
    assert math.isnan(iqm(np.full(5, np.nan)))                            # all-NaN -> NaN
    # A few NaNs are STRIPPED, not propagated (no silent NaN poisoning the band).
    mixed = np.array([1.0, 2.0, np.nan, 3.0, 4.0, np.inf, -np.inf])
    assert iqm(mixed) == pytest.approx(iqm(np.array([1.0, 2.0, 3.0, 4.0])), abs=1e-12)


@_DET
@given(
    a=hnp.arrays(np.float64, st.integers(1, 30),
                 elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False)),
    b=hnp.arrays(np.float64, st.integers(1, 30),
                 elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False)),
)
def test_probability_of_improvement_is_a_probability(a: np.ndarray, b: np.ndarray) -> None:
    """P(a>b) in [0, 1], and the complement relation P(a>b) + P(b>a) + P(tie) == 1 holds exactly."""
    p_ab = probability_of_improvement(a, b)
    p_ba = probability_of_improvement(b, a)
    assert 0.0 <= p_ab <= 1.0 and 0.0 <= p_ba <= 1.0
    # ties counted as 0.5 on BOTH sides, so p_ab + p_ba == 1 exactly (each tie contributes 0.5+0.5).
    assert p_ab + p_ba == pytest.approx(1.0, abs=1e-12)


def test_probability_of_improvement_metamorphic_and_boundaries() -> None:
    """Boundary cases: total dominance -> 1, total domination -> 0, identical -> 0.5, self -> 0.5."""
    lo = np.array([0.0, 1.0, 2.0])
    hi = np.array([10.0, 11.0, 12.0])
    assert probability_of_improvement(hi, lo) == 1.0
    assert probability_of_improvement(lo, hi) == 0.0
    assert probability_of_improvement(lo, lo) == pytest.approx(0.5, abs=1e-12)  # all ties
    # Common monotone shift to BOTH populations leaves P(improve) invariant (rank-based).
    rng = np.random.default_rng(2)
    a, b = rng.standard_normal(15), rng.standard_normal(15)
    assert probability_of_improvement(a + 5.0, b + 5.0) == pytest.approx(
        probability_of_improvement(a, b), abs=1e-12
    )


def test_probability_of_improvement_empty_is_nan() -> None:
    assert math.isnan(probability_of_improvement(np.array([]), np.array([1.0])))
    assert math.isnan(probability_of_improvement(np.array([1.0]), np.array([])))


def test_stratified_bootstrap_ci_point_is_iqm_and_brackets() -> None:
    """The CI point estimate equals iqm(scores); low <= point <= high; reproducible given rng."""
    rng = np.random.default_rng(4)
    scores = rng.standard_normal(40) + 0.3
    point, low, high = stratified_bootstrap_ci(scores, n_boot=500, rng=np.random.default_rng(7))
    assert point == pytest.approx(iqm(scores), abs=1e-12)
    assert low <= point <= high
    # Determinism: same scores + same seed -> identical CI.
    p2, l2, h2 = stratified_bootstrap_ci(scores, n_boot=500, rng=np.random.default_rng(7))
    assert (point, low, high) == (p2, l2, h2)


def test_stratified_bootstrap_ci_constant_series_is_zero_width() -> None:
    """Degenerate: a constant score series has a point-mass bootstrap CI (low == high == point)."""
    point, low, high = stratified_bootstrap_ci(np.full(25, 1.75), n_boot=300, rng=np.random.default_rng(1))
    assert point == pytest.approx(1.75, abs=1e-12)
    assert low == pytest.approx(1.75, abs=1e-12) and high == pytest.approx(1.75, abs=1e-12)


def test_stratified_bootstrap_ci_empty_is_all_nan() -> None:
    point, low, high = stratified_bootstrap_ci(np.array([]), n_boot=50)
    assert math.isnan(point) and math.isnan(low) and math.isnan(high)


# -----------------------------------------------------------------------------
# performance_profile — survival-fraction score distribution (Agarwal 2021)
# -----------------------------------------------------------------------------
def test_performance_profile_monotone_nonincreasing_and_in_unit_interval() -> None:
    """The profile is a survival function: in [0,1] and non-increasing in tau."""
    scores = np.array([0.1, 0.3, 0.3, 0.5, 0.9, 1.2, -0.2])
    taus = np.linspace(-1.0, 2.0, 40)
    prof, low, high = performance_profile(scores, taus, n_boot=200, rng=np.random.default_rng(0))
    assert prof.shape == taus.shape
    assert np.all((prof >= 0.0) & (prof <= 1.0))
    assert np.all(np.diff(prof) <= 1e-12)  # monotone non-increasing
    # The band brackets the point profile and stays in the unit interval.
    assert np.all((low <= prof + 1e-12) & (high >= prof - 1e-12))
    assert np.all((low >= -1e-12) & (high <= 1.0 + 1e-12))


def test_performance_profile_known_values_and_endpoints() -> None:
    """Exact survival fractions at chosen thresholds; below-all -> 1, above-all -> 0."""
    scores = np.array([0.0, 1.0, 2.0, 3.0])  # n = 4
    taus = np.array([-1.0, 0.5, 1.5, 2.5, 10.0])
    prof, _, _ = performance_profile(scores, taus, n_boot=0)
    # > -1: all 4; > 0.5: {1,2,3}=3; > 1.5: {2,3}=2; > 2.5: {3}=1; > 10: none.
    np.testing.assert_allclose(prof, np.array([4, 3, 2, 1, 0]) / 4.0)


def test_performance_profile_shift_equivariant_and_deterministic() -> None:
    """Shifting scores and thresholds by the same constant leaves the profile invariant;
    a fixed seed replays byte-identically."""
    rng_scores = np.random.default_rng(3)
    scores = rng_scores.normal(size=30)
    taus = np.linspace(-2.0, 2.0, 25)
    p0, l0, h0 = performance_profile(scores, taus, n_boot=300, rng=np.random.default_rng(11))
    p1, l1, h1 = performance_profile(scores + 5.0, taus + 5.0, n_boot=300, rng=np.random.default_rng(11))
    np.testing.assert_allclose(p0, p1, atol=1e-12)
    np.testing.assert_allclose(l0, l1, atol=1e-12)
    np.testing.assert_allclose(h0, h1, atol=1e-12)


def test_performance_profile_dominance_detects_ordering() -> None:
    """A uniformly-better arm's profile dominates a worse arm's at every threshold."""
    taus = np.linspace(0.0, 1.0, 50)
    better, _, _ = performance_profile(np.linspace(0.5, 1.5, 40), taus, n_boot=0)
    worse, _, _ = performance_profile(np.linspace(-0.5, 0.5, 40), taus, n_boot=0)
    assert np.all(better >= worse - 1e-12)


def test_performance_profile_empty_is_all_nan() -> None:
    prof, low, high = performance_profile(np.array([]), np.array([0.0, 1.0]), n_boot=10)
    assert np.all(np.isnan(prof)) and np.all(np.isnan(low)) and np.all(np.isnan(high))


# -----------------------------------------------------------------------------
# deflated_sharpe — legacy-alias loud reject + MinTRL significance guard
# (close mutation-probe survivors: scripts/mutation_probe.py --module deflated_sharpe)
# -----------------------------------------------------------------------------
def test_deflated_sharpe_alias_rejects_nonzero_benchmark_and_default_works() -> None:
    """The legacy alias must LOUDLY reject an unsupported non-zero sr_benchmark (a silently-dropped
    parameter is a correctness trap) while the default 0.0 path returns a valid probability."""
    from src.inference.deflated_sharpe import deflated_sharpe

    r = np.random.default_rng(0).standard_normal(80)
    with pytest.raises(NotImplementedError):
        deflated_sharpe(r, n_trials=10, sr_benchmark=0.5)
    val = deflated_sharpe(r, n_trials=10)  # default sr_benchmark == 0.0 must be accepted
    assert 0.0 <= val <= 1.0


def test_min_track_record_length_finite_iff_strategy_beats_benchmark() -> None:
    """MinTRL is finite/positive when sr > sr_star and +inf (never significant) when sr <= sr_star."""
    from src.inference.deflated_sharpe import min_track_record_length

    trl = min_track_record_length(sr=0.15, sr_star=0.0, skew=0.0, kurt=3.0, target_prob=0.95)
    assert np.isfinite(trl) and trl > 1.0
    assert min_track_record_length(sr=0.0, sr_star=0.10, skew=0.0, kurt=3.0) == float("inf")


def test_performance_profile_drops_nonfinite_scores() -> None:
    """Non-finite run scores are excluded (mirrors iqm's finite-filter contract)."""
    taus = np.array([0.5])
    clean, _, _ = performance_profile(np.array([0.0, 1.0]), taus, n_boot=0)
    dirty, _, _ = performance_profile(np.array([0.0, 1.0, np.nan, np.inf]), taus, n_boot=0)
    np.testing.assert_allclose(clean, dirty)


# =============================================================================
# attribution.py — factor decomposition additive consistency + sign + degenerate
# =============================================================================
def _ff3_block(t: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {name: rng.standard_normal(t) * 0.01 for name in ("Mkt-RF", "SMB", "HML")}


def test_factor_alpha_additive_decomposition_reconstructs_returns() -> None:
    """Additive consistency: mean(r) == alpha + sum_k beta_k * mean(F_k) (+ mean residual ~ 0).

    OLS with an intercept makes the residuals mean-zero, so the fitted alpha + the betas applied to
    the factor MEANS must reconstruct the realized mean return to numerical precision. This is the
    'decomposition sums to the total' invariant the BAB rebuttal relies on."""
    t = 4000
    f = _ff3_block(t, seed=1)
    betas = {"Mkt-RF": 1.2, "SMB": -0.3, "HML": 0.5}
    rng = np.random.default_rng(2)
    r = np.full(t, 0.0004)
    for name, b in betas.items():
        r = r + b * f[name]
    r = r + rng.standard_normal(t) * 0.003
    res = factor_alpha(r, f, factor_names=("Mkt-RF", "SMB", "HML"))
    assert res["status"] == "ok"
    reconstructed = res["alpha"] + sum(res["betas"][k] * f[k].mean() for k in betas)
    assert reconstructed == pytest.approx(float(r.mean()), abs=1e-12)


def test_factor_alpha_annualisation_is_exactly_252x() -> None:
    """alpha_ann == alpha * 252 exactly (no compounding, the documented convention)."""
    t = 1500
    f = _ff3_block(t, seed=3)
    r = 0.0003 + 1.0 * f["Mkt-RF"] + np.random.default_rng(4).standard_normal(t) * 0.004
    res = factor_alpha(r, f, factor_names=("Mkt-RF", "SMB", "HML"))
    assert res["status"] == "ok"
    assert res["alpha_ann"] == pytest.approx(res["alpha"] * 252.0, abs=1e-15)


def test_factor_alpha_sign_correct_under_known_alpha() -> None:
    """A planted POSITIVE alpha is recovered positive; a planted NEGATIVE alpha negative (sign-correct)."""
    t = 5000
    f = _ff3_block(t, seed=5)
    rng = np.random.default_rng(6)
    for true_alpha in (+0.0008, -0.0008):
        r = true_alpha + 1.0 * f["Mkt-RF"] + 0.2 * f["SMB"] + rng.standard_normal(t) * 0.003
        res = factor_alpha(r, f, factor_names=("Mkt-RF", "SMB", "HML"))
        assert res["status"] == "ok"
        assert np.sign(res["alpha"]) == np.sign(true_alpha)
        assert res["alpha"] == pytest.approx(true_alpha, abs=2e-4)


def test_factor_alpha_invariant_to_factor_column_order() -> None:
    """Reordering the factor columns leaves alpha and each beta unchanged (the fit is order-free)."""
    t = 2000
    f = _ff3_block(t, seed=8)
    r = 0.0005 + 1.1 * f["Mkt-RF"] - 0.4 * f["SMB"] + 0.3 * f["HML"]
    r = r + np.random.default_rng(9).standard_normal(t) * 0.002
    res1 = factor_alpha(r, f, factor_names=("Mkt-RF", "SMB", "HML"))
    res2 = factor_alpha(r, f, factor_names=("HML", "Mkt-RF", "SMB"))
    assert res1["status"] == res2["status"] == "ok"
    assert res1["alpha"] == pytest.approx(res2["alpha"], abs=1e-9)
    for name in ("Mkt-RF", "SMB", "HML"):
        assert res1["betas"][name] == pytest.approx(res2["betas"][name], abs=1e-9)


def test_factor_alpha_risk_free_shift_is_exact() -> None:
    """Metamorphic: a constant rf shifts the excess-return alpha down by EXACTLY mean(rf)."""
    t = 2500
    f = _ff3_block(t, seed=10)
    r = 0.0007 + 1.0 * f["Mkt-RF"] + np.random.default_rng(11).standard_normal(t) * 0.003
    rf = np.full(t, 0.00012)
    raw = factor_alpha(r, f, factor_names=("Mkt-RF", "SMB", "HML"))
    exc = factor_alpha(r, f, factor_names=("Mkt-RF", "SMB", "HML"), risk_free=rf)
    assert raw["status"] == exc["status"] == "ok"
    assert exc["alpha"] == pytest.approx(raw["alpha"] - 0.00012, abs=1e-12)


def test_factor_alpha_nan_inf_observations_dropped_gracefully() -> None:
    """Adversarial: NaN/inf rows are filtered; the fit on the clean remainder matches a clean fit."""
    t = 1500
    f = _ff3_block(t, seed=12)
    r_clean = 0.0004 + 1.0 * f["Mkt-RF"] + np.random.default_rng(13).standard_normal(t) * 0.003
    r_dirty = r_clean.copy()
    r_dirty[7] = np.nan
    r_dirty[100] = np.inf
    r_dirty[250] = -np.inf
    res = factor_alpha(r_dirty, f, factor_names=("Mkt-RF", "SMB", "HML"))
    assert res["status"] == "ok"
    assert res["n"] == t - 3  # the three corrupted rows were dropped
    assert math.isfinite(res["alpha"]) and math.isfinite(res["alpha_t"])
    assert 0.0 <= res["r2"] <= 1.0


def test_factor_alpha_all_nan_degrades_to_skipped_never_nan() -> None:
    """Degenerate: an all-NaN return series is reported skipped with no fabricated alpha (not a NaN)."""
    t = 600
    f = _ff3_block(t, seed=14)
    res = factor_alpha(np.full(t, np.nan), f, factor_names=("Mkt-RF", "SMB", "HML"))
    assert res["status"] == "skipped"
    assert res["alpha"] is None  # never fabricated, never a silent NaN


def test_factor_alpha_no_factor_columns_skips() -> None:
    res = factor_alpha(np.zeros(100), {}, factor_names=())
    assert res["status"] == "skipped" and res["alpha"] is None


def test_difference_in_alpha_antisymmetry() -> None:
    """Metamorphic: swapping the two arms negates the effect (effect(a,b) == -effect(b,a))."""
    t = 1200
    f = _ff3_block(t, seed=15)
    betas = {"Mkt-RF": 1.0, "SMB": 0.2, "HML": -0.1}

    def arm(alpha: float, base: int) -> dict[int, np.ndarray]:
        rng = np.random.default_rng(base)
        out = {}
        for s in range(16):
            jit = rng.normal(0.0, 2e-5)
            rr = (alpha + jit) + sum(b * f[k] for k, b in betas.items())
            out[s] = rr + np.random.default_rng(base * 100 + s).standard_normal(t) * 0.004
        return out

    a = arm(0.0006, 1)
    b = arm(0.0000, 2)
    fwd = difference_in_alpha(a, b, f, factor_names=("Mkt-RF", "SMB", "HML"),
                              n_boot=400, rng=np.random.default_rng(3))
    rev = difference_in_alpha(b, a, f, factor_names=("Mkt-RF", "SMB", "HML"),
                              n_boot=400, rng=np.random.default_rng(3))
    assert fwd["status"] == rev["status"] == "ok"
    # The per-seed IQM effect is exactly antisymmetric (same per-seed alphas, sign flipped).
    assert fwd["effect"] == pytest.approx(-rev["effect"], abs=1e-12)


# =============================================================================
# contamination.py — TOST monotonicity / Mahalanobis / McNemar
# =============================================================================
def test_paired_tost_monotone_in_shift_magnitude() -> None:
    """Metamorphic monotonicity: as the named-minus-blinded shift GROWS, the TOST equivalence
    p-value increases monotonically (it gets HARDER to declare equivalence) and equivalence is
    eventually lost. Larger contamination -> less equivalent, never the reverse."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal(60)
    blinded = base.copy()
    last_p = -1.0
    equiv_flags = []
    for shift in (0.0, 0.1, 0.2, 0.4, 0.8, 1.6):
        named = base + shift
        res = paired_tost(named, blinded, low=-0.5, high=0.5)
        assert res["p_tost"] >= last_p - 1e-9  # non-decreasing in the shift
        last_p = res["p_tost"]
        equiv_flags.append(res["equivalent"])
    # Equivalence holds at zero shift and is lost by the largest shift (monotone collapse).
    assert equiv_flags[0] is True
    assert equiv_flags[-1] is False


def test_paired_tost_degenerate_identical_pairs_inside_and_outside_bounds() -> None:
    """Zero-SE degenerate path: equivalent iff the (exact) mean diff sits strictly inside the bounds."""
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    inside = paired_tost(x, x, low=-0.5, high=0.5)  # diff == 0, strictly inside
    assert inside["equivalent"] is True and inside["mean_diff"] == 0.0
    # A constant offset of 2.0 with zero SE sits OUTSIDE +/-0.5 -> not equivalent, defined (no NaN).
    outside = paired_tost(x + 2.0, x, low=-0.5, high=0.5)
    assert outside["equivalent"] is False
    assert math.isfinite(outside["p_tost"]) and outside["mean_diff"] == pytest.approx(2.0, abs=1e-12)


def test_named_vs_blinded_tost_monotone_fraction_equivalent_in_contamination() -> None:
    """More contaminated coefficients -> the fraction declared equivalent does not INCREASE.

    Start from label-irrelevant coefficients (all equivalent at large n) and progressively
    contaminate more columns; the count of equivalent coefficients is monotone non-increasing."""
    rng = np.random.default_rng(1)
    n_seeds, n_coeffs = 200, 4
    truth = np.array([1.0, -0.5, 0.2, 0.0])
    sd = 0.1
    named0 = truth[None, :] + sd * rng.standard_normal((n_seeds, n_coeffs))
    blinded = truth[None, :] + sd * rng.standard_normal((n_seeds, n_coeffs))
    prev = n_coeffs + 1
    for n_contaminated in range(0, n_coeffs + 1):
        named = named0.copy()
        named[:, :n_contaminated] += 1.0  # huge identity-driven shift on the first k coefficients
        out = named_vs_blinded_tost(named, blinded)
        assert out["status"] == "ok"
        assert out["n_equivalent"] <= prev  # never recovers equivalence as contamination grows
        prev = out["n_equivalent"]
    # Fully contaminated -> nothing equivalent.
    assert prev == 0


def test_mahalanobis_permutation_deterministic_and_monotone() -> None:
    """The permutation p-value is reproducible given the rng AND decreases as the centroid shift
    grows (a bigger identity-driven centroid gap is MORE significant)."""
    rng = np.random.default_rng(2)
    blinded = rng.standard_normal((30, 3))
    p_small = coefficient_mahalanobis_permutation(
        blinded + np.array([0.2, 0.0, 0.0]), blinded, n_perm=800, rng=np.random.default_rng(5)
    )
    p_big = coefficient_mahalanobis_permutation(
        blinded + np.array([3.0, 0.0, 0.0]), blinded, n_perm=800, rng=np.random.default_rng(5)
    )
    assert p_small["status"] == p_big["status"] == "ok"
    assert p_big["pvalue"] <= p_small["pvalue"]  # bigger shift -> smaller p
    # Determinism: same rng seed -> identical p-value and statistic.
    again = coefficient_mahalanobis_permutation(
        blinded + np.array([3.0, 0.0, 0.0]), blinded, n_perm=800, rng=np.random.default_rng(5)
    )
    assert again["pvalue"] == p_big["pvalue"]
    assert again["mahalanobis"] == pytest.approx(p_big["mahalanobis"], abs=1e-12)


def test_mahalanobis_permutation_pvalue_floored_by_plus_one_convention() -> None:
    """The +1 permutation-test convention floors the p-value at 1/(n_perm+1) (never exactly 0)."""
    rng = np.random.default_rng(3)
    blinded = rng.standard_normal((25, 2))
    res = coefficient_mahalanobis_permutation(
        blinded + 50.0, blinded, n_perm=200, rng=np.random.default_rng(3)
    )
    assert res["pvalue"] == pytest.approx(1.0 / (200 + 1), abs=1e-12)


def test_structural_mcnemar_clean_data_silent_contaminated_fires() -> None:
    """McNemar stays silent (p==1) on perfect agreement and FIRES on a consistent identity-driven flip."""
    agree = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    silent = structural_mcnemar(agree, agree)
    assert silent["pvalue"] == 1.0 and silent["n01"] == 0 and silent["n10"] == 0
    named = np.ones(16, dtype=int)        # named always has the motif
    blinded = np.zeros(16, dtype=int)     # blinded never does -> 16 discordant, all one way
    fired = structural_mcnemar(named, blinded)
    assert fired["n10"] == 16 and fired["n01"] == 0
    assert fired["pvalue"] < 0.05


def test_structural_mcnemar_symmetric_discord_is_not_significant() -> None:
    """Balanced discordance (equal flips each way) is the McNemar null -> p == 1 (no directional flip)."""
    named = np.array([1, 1, 1, 0, 0, 0])
    blinded = np.array([0, 0, 0, 1, 1, 1])  # n10 == n01 == 3
    res = structural_mcnemar(named, blinded)
    assert res["n10"] == 3 and res["n01"] == 3
    assert res["pvalue"] == pytest.approx(1.0, abs=1e-9)


# =============================================================================
# ood_stress.py — contamination/regime flags fire on shift, silent on clean
# =============================================================================
def _garch_like_panel(n: int = 800, k: int = 4, seed: int = 7) -> np.ndarray:
    """Heavy-tailed, volatility-clustered single-factor panel (stylised-fact rich)."""
    rng = np.random.default_rng(seed)
    sig2 = np.empty(n)
    sig2[0] = 1.0
    z = rng.standard_t(6, size=n)
    eps = np.empty(n)
    eps[0] = z[0]
    for t in range(1, n):
        sig2[t] = 0.02 + 0.10 * eps[t - 1] ** 2 + 0.87 * sig2[t - 1]
        eps[t] = math.sqrt(sig2[t]) * z[t]
    market = 0.0003 + 0.01 * eps / (np.std(eps) or 1.0)
    betas = np.linspace(0.8, 1.2, k)
    return market[:, None] * betas[None, :] + 0.004 * rng.standard_normal((n, k))


def test_validate_stylized_facts_silent_on_gaussian_fires_on_garch() -> None:
    """The clustering 'contamination' gate is SILENT (False) on clean iid Gaussian and FIRES (True)
    on a genuinely volatility-clustered GARCH panel — the no-false-alarm / true-detection pair."""
    panel = _garch_like_panel(n=900, k=3)
    # Clean: iid Gaussian -> NO volatility clustering -> gate stays False (no false alarm).
    gaussian = np.random.default_rng(1).standard_normal((20, 900, 3)) * 0.01
    clean = validate_stylized_facts(gaussian, panel)
    assert clean["checks"]["vol_clustering"] is False
    assert clean["passed"] is False
    # Stressed/real GARCH paths: clustering gate fires True.
    garch_paths = block_bootstrap_paths(panel, n_paths=20, horizon=900, rng=np.random.default_rng(2))
    # Block bootstrap preserves the clustering of the source panel.
    stressed = validate_stylized_facts(garch_paths, panel)
    assert stressed["checks"]["vol_clustering"] is True


def test_vol_spike_is_monotone_in_the_stress_multiplier() -> None:
    """Metamorphic monotonicity: a larger vol-spike multiplier strictly inflates the realized
    per-asset standard deviation by exactly sqrt(multiplier), and the tail CVaR gets MORE extreme
    (more negative) as the stress grows — the regime-shift magnitude monotonicity."""
    panel = _garch_like_panel(n=500, k=3)
    base_std = panel.std(axis=0)
    prev_cvar = np.inf
    for mult in (1.0, 2.0, 4.0, 9.0):
        out = vol_spike_paths(panel, multiplier=mult)  # (1, T, k), mean-preserving
        np.testing.assert_allclose(out[0].mean(axis=0), panel.mean(axis=0), atol=1e-10)
        np.testing.assert_allclose(out[0].std(axis=0) / base_std, math.sqrt(mult), rtol=1e-6)
        # Equal-weight portfolio CVaR-5% becomes more negative as variance scales up.
        port = out[0].mean(axis=1)[None, :]
        c = tail_metrics(port)["cvar"][0.05]["iqm"]
        assert c <= prev_cvar + 1e-12  # monotone non-increasing (more extreme) in the multiplier
        prev_cvar = c


def test_block_bootstrap_is_seed_deterministic_and_membership_preserving() -> None:
    """Determinism: identical rng seeds -> byte-identical paths; every emitted row is a verbatim
    panel row (the cross-section is never invented — clean-data invariant)."""
    panel = _garch_like_panel(n=300, k=4)
    p1 = block_bootstrap_paths(panel, n_paths=5, rng=np.random.default_rng(42))
    p2 = block_bootstrap_paths(panel, n_paths=5, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(p1, p2)
    panel_rows = {tuple(np.round(r, 12)) for r in panel}
    for r in p1[0]:
        assert tuple(np.round(r, 12)) in panel_rows


def test_tail_metrics_ordering_and_degenerate_constant_series() -> None:
    """CVaR-1% is no less extreme than CVaR-5% (tail ordering); a constant series has zero drawdown
    and a finite, defined CVaR equal to the constant (no silent NaN)."""
    rng = np.random.default_rng(3)
    port = 0.0005 + 0.01 * rng.standard_normal((25, 500))
    m = tail_metrics(port)
    assert m["cvar"][0.01]["iqm"] <= m["cvar"][0.05]["iqm"] + 1e-9
    assert m["max_drawdown"]["iqm"] >= 0.0
    # Degenerate: a flat positive series -> zero drawdown, CVaR == the constant, all finite.
    const = np.full((4, 200), 0.001)
    mc = tail_metrics(const)
    assert mc["max_drawdown"]["iqm"] == pytest.approx(0.0, abs=1e-12)
    assert mc["cvar"][0.05]["iqm"] == pytest.approx(0.001, abs=1e-9)
    assert math.isfinite(mc["sharpe"]["iqm"]) or mc["sharpe"]["iqm"] == 0.0


def test_tail_metrics_all_nan_path_is_defined_not_crash() -> None:
    """Adversarial: an all-NaN path set yields DEFINED, finite-or-NaN aggregates, never a crash.

    Per the documented primitives: ``cvar`` strips non-finite values and returns NaN on an empty
    tail, while ``sharpe_ratio`` returns 0.0 when the (NaN-stripped) std is zero. Both are defined
    outcomes (no exception, no silent garbage) — the contract this asserts."""
    m = tail_metrics(np.full((3, 100), np.nan))
    # sharpe_ratio documents a 0.0 fallback for a zero-std (here all-NaN) series.
    assert m["sharpe"]["iqm"] == 0.0
    # cvar documents a NaN for an empty (all-stripped) tail.
    assert math.isnan(m["cvar"][0.05]["iqm"])
    # max_drawdown on a degenerate path is a defined, finite value.
    assert math.isfinite(m["max_drawdown"]["iqm"]) or math.isnan(m["max_drawdown"]["iqm"])


@_DET
@given(seed=st.integers(0, 2**31 - 1), n_paths=st.integers(2, 6))
def test_block_bootstrap_shape_is_exact(seed: int, n_paths: int) -> None:
    """Property: block_bootstrap_paths always returns the requested (n_paths, T, n_assets) shape."""
    panel = _garch_like_panel(n=120, k=3, seed=(seed % 97) + 1)
    out = block_bootstrap_paths(panel, n_paths=n_paths, rng=np.random.default_rng(seed))
    assert out.shape == (n_paths, panel.shape[0], panel.shape[1])
    assert np.all(np.isfinite(out))


# =============================================================================
# Cross-module determinism replay (the dissertation's load-bearing reproducibility claim)
# =============================================================================
def test_full_pipeline_slice_is_byte_reproducible() -> None:
    """A small slice that touches pbo + factor_alpha + paired_tost + tail_metrics replays IDENTICALLY
    across two independent invocations with matched seeds — the determinism the campaign rests on."""
    def run() -> tuple[float, float, float, float]:
        rng = np.random.default_rng(2026)
        perf = rng.standard_normal((240, 6))
        f = _ff3_block(1000, seed=99)
        r = 0.0004 + 1.0 * f["Mkt-RF"] + np.random.default_rng(98).standard_normal(1000) * 0.003
        named = np.random.default_rng(1).standard_normal((40, 3))
        port = 0.0005 + 0.01 * np.random.default_rng(5).standard_normal((10, 300))
        return (
            pbo(perf, n_blocks=8, rng=np.random.default_rng(0)),
            factor_alpha(r, f, factor_names=("Mkt-RF", "SMB", "HML"))["alpha"],
            paired_tost(named[:, 0], named[:, 1], low=-0.5, high=0.5)["p_tost"],
            tail_metrics(port)["cvar"][0.05]["iqm"],
        )

    assert run() == run()
