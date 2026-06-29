"""Behaviour tests for the selection-aware inference stack (FINAL_PLAN F.11)."""

from __future__ import annotations


import math

import numpy as np
import pytest
from scipy.stats import norm

from src.inference.bootstrap import (
    cvar,
    cvar_difference_test,
    null_calibration,
    paired_seed_difference_test,
    sharpe_difference_test,
    sharpe_ratio,
    stationary_bootstrap_indices,
)
from src.inference.deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)
from src.inference.es_backtest import dm_size_power_calibration, hln_factor
from src.inference.multiple_testing import benjamini_hochberg, romano_wolf
from src.inference.overfitting import pbo
from src.inference.reporting import (
    iqm,
    probability_of_improvement,
    stratified_bootstrap_ci,
)


# ---------------------------------------------------------------------------
# bootstrap.py
# ---------------------------------------------------------------------------
def _mean_run_length(idx: np.ndarray) -> float:
    """Mean length of consecutive +1 (mod n) runs in an index path."""
    n = idx.size
    runs = []
    cur = 1
    for t in range(1, n):
        if idx[t] == (idx[t - 1] + 1) % n:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    return float(np.mean(runs))


def test_bootstrap_indices_length_and_range(rng) -> None:
    n = 500
    idx = stationary_bootstrap_indices(n, p=0.1, rng=rng)
    assert idx.shape == (n,)
    assert idx.min() >= 0 and idx.max() < n
    assert idx.dtype.kind == "i"


def test_bootstrap_block_structure_grows_as_p_shrinks(rng) -> None:
    """p~1 => ~iid (short runs); small p => long blocks."""
    n = 2000
    rl_iid = _mean_run_length(stationary_bootstrap_indices(n, p=0.99, rng=rng))
    rl_small = _mean_run_length(stationary_bootstrap_indices(n, p=0.05, rng=rng))
    assert rl_iid < 1.5  # near-iid: almost every step restarts
    assert rl_small > rl_iid
    # Expected block length is ~1/p; 1/0.05 = 20, allow generous tolerance.
    assert rl_small > 5.0


def test_sharpe_ratio_sign_and_scale() -> None:
    # Constant series has zero std => Sharpe defined as 0.0.
    assert sharpe_ratio(np.full(1000, 0.01)) == 0.0
    rng = np.random.default_rng(0)
    good = 0.001 + 0.01 * rng.standard_normal(50_000)
    bad = -0.001 + 0.01 * rng.standard_normal(50_000)
    assert sharpe_ratio(good) > 0
    assert sharpe_ratio(bad) < 0
    # Annualization scales by sqrt(252).
    daily = sharpe_ratio(good, periods_per_year=1)
    annual = sharpe_ratio(good, periods_per_year=252)
    assert annual == pytest.approx(daily * math.sqrt(252), rel=1e-9)


def test_cvar_monotone_in_alpha(normal_returns) -> None:
    c1 = cvar(normal_returns, 0.01)
    c5 = cvar(normal_returns, 0.05)
    c10 = cvar(normal_returns, 0.10)
    # All negative (lower tail of N(0,1)).
    assert c1 < 0 and c5 < 0 and c10 < 0
    # Smaller alpha => more extreme => more negative.
    assert c1 < c5 < c10
    # Closed-form ES of N(0,1) at 5%: -phi(z)/alpha ~ -2.063.
    assert c5 == pytest.approx(-2.063, abs=0.05)


def _equal_pair_sampler(rng):
    a = 0.0005 + 0.01 * rng.standard_normal(800)
    b = 0.0005 + 0.01 * rng.standard_normal(800)
    return a, b


def test_sharpe_test_null_calibration() -> None:
    rng = np.random.default_rng(7)

    def test_fn(a, b):
        return sharpe_difference_test(a, b, n_boot=600, rng=rng)

    res = null_calibration(test_fn, _equal_pair_sampler, n_reps=160, rng=rng)
    assert res["rejection_rate"] <= 0.15
    assert 0.35 <= res["mean_pvalue"] <= 0.65


def test_paired_seed_difference_one_sided_null_calibration() -> None:
    """R64 / audit-2026-07-02: certify the ONE-SIDED size of the ACTUAL headline rule.

    The co-primary H2-Tail leg rejects on ``pvalue_one_sided_greater`` (the direct upper-tail p, valid
    under a skewed CVaR bootstrap), NOT the two-sided ``pvalue`` — and ONLY ``paired_seed_difference_test``
    exposes it (``sharpe``/``cvar_difference_test`` do not, so the two calibration tests above exercise
    the two-sided branch ONLY, never ``rejection_rate_one_sided``). The module docstring flags exactly
    this as the size that "can drift" under skew. Under a true exchangeable null the one-sided rule must
    be correctly sized (~alpha) with a mean p ~0.5; this certifies the R64 branch the tail IUT depends on.
    """
    rng = np.random.default_rng(20260709)

    def sampler(r):
        return r.standard_normal(24), r.standard_normal(24)  # exchangeable null: a, b iid same dist

    def test_fn(a, b):
        return paired_seed_difference_test(a, b, n_boot=600, rng=rng)

    res = null_calibration(test_fn, sampler, n_reps=300, rng=rng)
    # The R64 one-sided branch actually fired (the keys the two-sided-only calibration tests never see).
    for key in ("rejection_rate_one_sided", "mean_pvalue_one_sided", "pvalues_one_sided"):
        assert key in res
    assert np.asarray(res["pvalues_one_sided"]).shape == (300,)
    assert np.all(np.isfinite(res["pvalues_one_sided"]))
    # The ONE-SIDED rule (the tail IUT's actual decision statistic) is correctly sized under the null:
    # ~alpha (measured 0.06), does not drift high, and is not vacuously never-rejecting.
    assert 0.01 <= res["rejection_rate_one_sided"] <= 0.12
    assert 0.40 <= res["mean_pvalue_one_sided"] <= 0.60  # ~Uniform(0,1) mean 0.5 (measured 0.5003)
    # The two-sided branch on the SAME headline test stays well-sized too.
    assert res["rejection_rate"] <= 0.12
    assert 0.40 <= res["mean_pvalue"] <= 0.60


def test_cvar_test_null_calibration() -> None:
    rng = np.random.default_rng(11)

    def test_fn(a, b):
        return cvar_difference_test(a, b, alpha=0.05, n_boot=600, rng=rng)

    res = null_calibration(test_fn, _equal_pair_sampler, n_reps=160, rng=rng)
    assert res["rejection_rate"] <= 0.15
    assert 0.35 <= res["mean_pvalue"] <= 0.65


def test_sharpe_test_returns_keys(rng) -> None:
    a = 0.001 + 0.01 * rng.standard_normal(500)
    b = 0.0005 + 0.01 * rng.standard_normal(500)
    res = sharpe_difference_test(a, b, n_boot=500, rng=rng)
    assert set(res) == {"stat", "pvalue", "ci_low", "ci_high"}
    assert 0.0 <= res["pvalue"] <= 1.0
    assert res["ci_low"] <= res["ci_high"]


# ---------------------------------------------------------------------------
# overfitting.py (PBO / CSCV)
# ---------------------------------------------------------------------------
def test_pbo_in_unit_interval() -> None:
    rng = np.random.default_rng(1)
    mat = rng.standard_normal((400, 10))
    val = pbo(mat, n_blocks=10, rng=rng)
    assert 0.0 <= val <= 1.0


def test_pbo_zero_when_one_config_dominates() -> None:
    rng = np.random.default_rng(2)
    t, n = 400, 8
    mat = rng.standard_normal((t, n)) * 0.1
    # Config 0 has a large mean everywhere -> best IS and best OOS always.
    mat[:, 0] += 5.0
    val = pbo(mat, n_blocks=10, rng=rng)
    assert val == pytest.approx(0.0, abs=0.02)


def test_pbo_near_half_on_pure_noise() -> None:
    vals = []
    for seed in range(5):
        rng = np.random.default_rng(100 + seed)
        mat = rng.standard_normal((500, 12))
        vals.append(pbo(mat, n_blocks=12, rng=rng))
    assert np.mean(vals) == pytest.approx(0.5, abs=0.15)


# ---------------------------------------------------------------------------
# deflated_sharpe.py
# ---------------------------------------------------------------------------
def test_psr_matches_hand_computed_normal() -> None:
    # Normal: skew 0, kurt 3 => denom var = 1 + (3-1)/4 * sr^2.
    sr, n = 0.1, 101
    denom = math.sqrt(1.0 + (3.0 - 1.0) / 4.0 * sr * sr)
    expected = float(norm.cdf(sr * math.sqrt(n - 1) / denom))
    got = probabilistic_sharpe_ratio(sr, 0.0, n, skew=0.0, kurt=3.0)
    assert got == pytest.approx(expected, abs=1e-10)
    assert 0.0 <= got <= 1.0


def test_dsr_in_unit_interval_and_monotone_in_trials() -> None:
    rng = np.random.default_rng(3)
    returns = 0.0008 + 0.01 * rng.standard_normal(2000)
    d_few = deflated_sharpe_ratio(returns, n_trials=2)
    d_many = deflated_sharpe_ratio(returns, n_trials=500)
    assert 0.0 <= d_few <= 1.0 and 0.0 <= d_many <= 1.0
    # More trials => higher SR_star => lower DSR.
    assert d_many <= d_few


def test_expected_max_sharpe_increases_with_trials() -> None:
    e10 = expected_max_sharpe(1.0, 10)
    e1000 = expected_max_sharpe(1.0, 1000)
    assert e1000 > e10 > 0.0


def test_dsr_single_trial_does_not_saturate_regression_r65() -> None:
    """R65 / DEEP_AUDIT 2026-06-28: the ``n_trials <= 1`` guard in ``expected_max_sharpe``.

    Before the fix, ``n_trials == 1`` gave ``norm.ppf(1 - 1/1) = ppf(0) = -inf`` so ``sr_star = -inf``
    and ``deflated_sharpe_ratio(x, n_trials=1) == 1.0`` for EVERY series -- even a strongly NEGATIVE-Sharpe
    one. That silently saturated the H1/T0 benchmark-floor gate (every un-searched benchmark DSR read 1.0,
    so a winner could never clear the floor). The guard makes a single trial (no multiplicity) benchmark
    against ``sr_star = 0``, i.e. the DSR reduces to the PSR against zero.
    """
    # The guard clauses return exactly 0.0 (documented contract): N <= 1, and a non-positive var_sr.
    assert expected_max_sharpe(1.0, 1) == 0.0
    assert expected_max_sharpe(1.0, 0) == 0.0
    assert expected_max_sharpe(0.0, 100) == 0.0
    assert expected_max_sharpe(-1.0, 100) == 0.0

    rng = np.random.default_rng(65)
    neg = -0.02 + 0.01 * rng.standard_normal(500)  # strongly negative per-period Sharpe (~ -2)
    pos = 0.02 + 0.01 * rng.standard_normal(500)  # strongly positive per-period Sharpe (~ +2)
    # The bug's signature: a losing series must NOT saturate to 1.0 -- with sr_star=0 it is < 0.5.
    dsr_neg = deflated_sharpe_ratio(neg, n_trials=1)
    dsr_pos = deflated_sharpe_ratio(pos, n_trials=1)
    assert dsr_neg < 0.5 < dsr_pos
    assert dsr_neg < 0.99  # pre-fix this was exactly 1.0 for the loser
    # And the documented reduction is EXACT: DSR(., n_trials=1) == PSR against sr_star=0 on the same moments.
    r = pos.astype(float)
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    z = (r - mu) / sd
    sr, skew, kurt = mu / sd, float(np.mean(z**3)), float(np.mean(z**4))
    assert dsr_pos == pytest.approx(
        probabilistic_sharpe_ratio(sr, 0.0, r.size, skew, kurt), abs=1e-12
    )


def _dsr_from_inputs(sr, skew, kurt, n, var_sr, n_trials):
    """Hand-rolled DSR: deflate ``sr`` by E[max SR] under ``var_sr`` / ``n_trials``.

    Mirrors ``deflated_sharpe_ratio`` arithmetic on RAW (sr, skew, kurt, n) moments so the
    golden value is computed independently of any return-series moment estimation.
    """
    g = 0.5772156649015329
    e = math.e
    sr_star = math.sqrt(var_sr) * (
        (1.0 - g) * norm.ppf(1.0 - 1.0 / n_trials)
        + g * norm.ppf(1.0 - 1.0 / (n_trials * e))
    )
    denom = math.sqrt(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr)
    return float(norm.cdf((sr - sr_star) * math.sqrt(n - 1) / denom))


def test_dsr_canonical_var_sr_differs_from_within_series_proxy() -> None:
    """Rank 16: canonical cross-trial ``var_sr`` DSR DIFFERS from the ``var_sr=None`` proxy.

    On a HETEROGENEOUS (dispersed-skill) candidate population the empirical cross-trial
    Sharpe dispersion is a DIFFERENT quantity from the single-series sampling-variance proxy,
    so the deflated Sharpe of one selected series changes when the correct cross-trial
    variance is supplied. The golden value is hand-computed from the series moments.
    """
    # A near-Gaussian per-period validation series for the WINNER (skew ~ 0, kurt ~ 3) so the
    # moment arithmetic is transparent. Fixed seed -> deterministic moments.
    rng = np.random.default_rng(1234)
    winner = 0.0008 + 0.01 * rng.standard_normal(750)

    # A dispersed-skill candidate population: per-candidate per-period Sharpes spread WIDE
    # (some genuinely skilled, some adverse) -> cross-trial variance >> the within-series proxy.
    cand_sharpes = np.array([-0.15, -0.09, -0.04, 0.0, 0.05, 0.09, 0.14, 0.22])
    var_sr_cross = float(np.var(cand_sharpes, ddof=1))
    n_trials = cand_sharpes.size

    # Recover the winner's per-period moments exactly as the implementation does.
    r = winner
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    sr = mu / sd
    z = (r - mu) / sd
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4))
    n = r.size

    # The within-series proxy variance the var_sr=None branch uses.
    var_sr_proxy = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr) / (n - 1)

    dsr_canonical = deflated_sharpe_ratio(winner, n_trials, var_sr=var_sr_cross)
    dsr_proxy = deflated_sharpe_ratio(winner, n_trials, var_sr=None)

    # Golden hand-computed values (independent of the function's internal moment estimation).
    golden_canonical = _dsr_from_inputs(sr, skew, kurt, n, var_sr_cross, n_trials)
    golden_proxy = _dsr_from_inputs(sr, skew, kurt, n, var_sr_proxy, n_trials)
    assert dsr_canonical == pytest.approx(golden_canonical, abs=1e-10)
    assert dsr_proxy == pytest.approx(golden_proxy, abs=1e-10)

    # The cross-trial dispersion is materially larger than the within-series proxy here, so
    # the canonical DSR is STRICTLY LOWER (more deflation) -- the two genuinely differ.
    assert var_sr_cross > 5.0 * var_sr_proxy
    assert dsr_canonical < dsr_proxy
    assert abs(dsr_canonical - dsr_proxy) > 1e-3


def test_dsr_canonical_coincides_with_proxy_under_homogeneous_null() -> None:
    """Rank 16: the two DSRs COINCIDE when cross-trial var == the within-series proxy var.

    Under the homogeneous zero-skill null the canonical cross-trial Sharpe dispersion equals
    the single-series sampling variance, so passing that proxy variance EXPLICITLY reproduces
    the ``var_sr=None`` value exactly (the proxy is the canonical input in that regime).
    """
    rng = np.random.default_rng(77)
    series = 0.01 * rng.standard_normal(1000)  # ~zero-Sharpe homogeneous null

    sr, sd = float(series.mean() / series.std(ddof=1)), 1.0  # noqa: F841 (sd implicit below)
    z = (series - series.mean()) / series.std(ddof=1)
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4))
    n = series.size
    var_sr_proxy = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr) / (n - 1)

    dsr_proxy = deflated_sharpe_ratio(series, n_trials=8, var_sr=None)
    dsr_with_proxy_var = deflated_sharpe_ratio(series, n_trials=8, var_sr=var_sr_proxy)
    # Supplying the canonical input that the homogeneous null implies == the var_sr=None path.
    assert dsr_with_proxy_var == pytest.approx(dsr_proxy, abs=1e-12)


# ---------------------------------------------------------------------------
# multiple_testing.py
# ---------------------------------------------------------------------------
def test_bh_controls_fdr_on_all_null() -> None:
    rng = np.random.default_rng(4)
    rates = []
    for _ in range(40):
        pvals = rng.random(50)  # all-null => Uniform(0,1)
        rej = benjamini_hochberg(pvals, q=0.1)
        rates.append(rej.mean())
    # Under all-null, expected number of rejections is tiny.
    assert np.mean(rates) <= 0.1


def test_bh_rejects_tiny_pvalue_in_nulls() -> None:
    rng = np.random.default_rng(5)
    pvals = rng.random(50)
    pvals[7] = 1e-8
    rej = benjamini_hochberg(pvals, q=0.1)
    assert rej[7]


def test_bh_monotone_in_q() -> None:
    rng = np.random.default_rng(6)
    pvals = np.sort(rng.random(40))
    pvals[:5] = np.array([0.001, 0.004, 0.01, 0.02, 0.03])
    r_low = benjamini_hochberg(pvals, q=0.05)
    r_high = benjamini_hochberg(pvals, q=0.25)
    # Larger q rejects a superset.
    assert np.all(r_high[r_low])


def test_romano_wolf_rejects_strong_signal() -> None:
    rng = np.random.default_rng(8)
    m = 6
    boot = rng.standard_normal((2000, m))  # centred null draws
    stats = np.array([5.0, 0.1, 0.2, 0.0, 0.3, 0.1])  # hypothesis 0 is strong
    rej = romano_wolf(stats, boot, alpha=0.05)
    assert rej[0]
    assert not rej[3]


def test_romano_wolf_controls_fwer_under_complete_null() -> None:
    """Romano-Wolf's DEFINING guarantee (docstring: "strong control of the family-wise error rate"):
    under the COMPLETE null (every hypothesis true) the probability of ANY false rejection is <= alpha.

    Monte-Carlo calibration: the observed statistics AND the bootstrap null are drawn from the SAME
    distribution (one-sided, larger = more evidence), so the max-statistic critical value should admit a
    false rejection at rate ~alpha and never materially above it. The structural tests
    (``..._stops_at_first_non_rejection``, ``..._is_order_aware_stepdown``) prove the stepdown mechanics;
    this proves those mechanics actually deliver the error guarantee -- the one property the whole method
    exists to provide, previously never empirically verified. Deterministic seed (replayable; no
    wall-clock / global RNG).
    """
    def _fwer(m: int, alpha: float, n_reps: int = 1500, n_boot: int = 1500) -> float:
        rng = np.random.default_rng(20260709)
        hits = 0
        for _ in range(n_reps):
            stats = rng.standard_normal(m)
            boot = rng.standard_normal((n_boot, m))
            if romano_wolf(stats, boot, alpha=alpha).any():
                hits += 1
        return hits / n_reps

    fwer_05 = _fwer(4, 0.05)  # measured 0.0433 at this seed
    fwer_10 = _fwer(4, 0.10)  # measured 0.0940 at this seed
    # Strong control: at or below nominal, never materially above it (the guarantee, with MC/quantile slack).
    assert fwer_05 <= 0.065
    assert fwer_10 <= 0.125
    # ...and genuinely calibrated, not a vacuously-never-rejecting implementation.
    assert fwer_05 >= 0.025
    assert fwer_10 >= 0.060


# ---------------------------------------------------------------------------
# reporting.py
# ---------------------------------------------------------------------------
def test_iqm_matches_fixture() -> None:
    data = np.arange(1, 101, dtype=float)  # 1..100
    assert iqm(data) == pytest.approx(50.5, abs=1e-9)


def test_iqm_robust_to_outliers() -> None:
    base = np.arange(1, 101, dtype=float)
    contaminated = np.concatenate([base, [1e9, 1e9, -1e9, -1e9]])
    assert iqm(contaminated) == pytest.approx(iqm(base), abs=2.0)


def test_probability_of_improvement_fixture() -> None:
    a = np.array([10.0, 11.0, 12.0])
    b = np.array([1.0, 2.0, 3.0])
    assert probability_of_improvement(a, b) == pytest.approx(1.0)
    assert probability_of_improvement(b, a) == pytest.approx(0.0)
    # Equal populations => 0.5 (all ties).
    c = np.array([5.0, 5.0])
    assert probability_of_improvement(c, c) == pytest.approx(0.5)


def test_stratified_bootstrap_ci_brackets_iqm(rng) -> None:
    scores = 1.0 + 0.5 * rng.standard_normal(200)
    point, low, high = stratified_bootstrap_ci(scores, n_boot=1000, ci=0.95, rng=rng)
    assert point == pytest.approx(iqm(scores))
    assert low < high
    assert low <= point <= high


# ===========================================================================
# STRICT statistical-property tests (these stats GATE the H2 conclusions).
# Added 2026-06-27. Each docstring cites the property being certified.
# ===========================================================================


# ---------------------------------------------------------------------------
# 1. PROBABILISTIC / DEFLATED SHARPE — deflated_sharpe.py
# ---------------------------------------------------------------------------
def test_psr_increases_with_track_record_length() -> None:
    """PROPERTY: for a fixed positive Sharpe, PSR is strictly increasing in n.

    PSR = Phi((sr - sr*) * sqrt(n-1) / sqrt(denom)); the sqrt(n-1) factor makes the
    z-argument grow with the track-record length, so a longer record is more
    confident the true Sharpe beats the benchmark (Bailey & Lopez de Prado 2012).
    """
    sr, sr_star = 0.1, 0.0
    psr = [
        probabilistic_sharpe_ratio(sr, sr_star, n, skew=0.0, kurt=3.0)
        for n in (50, 200, 1000, 5000)
    ]
    # Strictly increasing in n, and all a valid probability.
    assert all(0.0 <= v <= 1.0 for v in psr)
    assert all(psr[i] < psr[i + 1] for i in range(len(psr) - 1))


def test_psr_lower_for_heavy_tails_than_normal() -> None:
    """PROPERTY: non-normality (high kurtosis) LOWERS PSR vs a normal sample.

    The PSR denominator sqrt(1 - skew*sr + (kurt-1)/4 * sr^2) grows with kurtosis,
    inflating the Sharpe-estimator uncertainty, so for identical (sr, sr*, n, skew=0)
    a leptokurtic series (kurt=9) has a SMALLER z-argument and a LOWER PSR than a
    Gaussian one (kurt=3). Holds for sr>0 where the (kurt-1)/4*sr^2 term dominates.
    """
    sr, sr_star, n = 0.15, 0.0, 1000
    psr_normal = probabilistic_sharpe_ratio(sr, sr_star, n, skew=0.0, kurt=3.0)
    psr_heavy = probabilistic_sharpe_ratio(sr, sr_star, n, skew=0.0, kurt=9.0)
    assert psr_heavy < psr_normal
    assert 0.0 <= psr_heavy <= 1.0 and 0.0 <= psr_normal <= 1.0


def test_psr_degenerate_denominator_finite_and_sign_correct() -> None:
    """PROPERTY: a non-positive PSR denominator returns a finite, sign-correct value.

    When 1 - skew*sr + (kurt-1)/4*sr^2 <= 0 (a pathological moment combination) the
    implementation falls back to a sign-based decision rather than dividing by a
    non-positive variance: PSR -> 1.0 when sr > sr* and 0.0 otherwise — never NaN/inf.
    """
    # Force denom_var <= 0 via a large negative kurtosis-driven term + skew.
    sr = 2.0
    # denom = 1 - skew*sr + (kurt-1)/4*sr^2; pick kurt very small and skew large positive.
    psr_better = probabilistic_sharpe_ratio(sr, 0.0, 500, skew=5.0, kurt=0.0)
    psr_worse = probabilistic_sharpe_ratio(-sr, 0.0, 500, skew=-5.0, kurt=0.0)
    assert math.isfinite(psr_better) and math.isfinite(psr_worse)
    assert psr_better == 1.0  # sr > sr_star
    assert psr_worse == 0.0  # sr < sr_star


def test_dsr_never_exceeds_psr_against_zero() -> None:
    """PROPERTY: DSR <= PSR(sr*=0) — deflating by trials never RAISES confidence.

    The DSR is a PSR benchmarked against sr* = E[max SR] >= 0 (the expected maximum
    Sharpe under n_trials zero-skill trials), whereas the undeflated PSR benchmarks
    against 0. Since PSR is monotone decreasing in sr* and E[max SR] >= 0, the
    deflated value can only be <= the undeflated one (Bailey & Lopez de Prado 2014).
    """
    rng = np.random.default_rng(2026)
    returns = 0.0008 + 0.01 * rng.standard_normal(2000)
    sr, skew, kurt = (
        float(returns.mean() / returns.std(ddof=1)),
        0.0,
        0.0,
    )
    z = (returns - returns.mean()) / returns.std(ddof=1)
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4))
    n = returns.size
    psr_zero = probabilistic_sharpe_ratio(sr, 0.0, n, skew, kurt)
    for n_trials in (2, 10, 100, 1000):
        dsr = deflated_sharpe_ratio(returns, n_trials=n_trials)
        assert dsr <= psr_zero + 1e-12


# ---------------------------------------------------------------------------
# 2. STATIONARY BOOTSTRAP — bootstrap.py
# ---------------------------------------------------------------------------
def test_bootstrap_indices_deterministic_given_seed() -> None:
    """PROPERTY: stationary_bootstrap_indices is a pure function of (n, p, seed).

    Reproducibility underpins the whole pre-registered study: two generators seeded
    identically must yield byte-identical index paths, and a different seed must differ.
    """
    path_a = stationary_bootstrap_indices(300, 0.1, np.random.default_rng(42))
    path_b = stationary_bootstrap_indices(300, 0.1, np.random.default_rng(42))
    path_c = stationary_bootstrap_indices(300, 0.1, np.random.default_rng(43))
    assert np.array_equal(path_a, path_b)
    assert not np.array_equal(path_a, path_c)


def test_bootstrap_mean_block_length_matches_one_over_p() -> None:
    """PROPERTY: empirical mean block length ~ 1/p (Politis-Romano geometric blocks).

    Block lengths are Geometric(p) with expectation 1/p; averaged over many draws the
    observed mean run length must sit within +-20% of 1/p. (For p=0.1 -> ~10.)
    """
    n, p = 4000, 0.1
    rng = np.random.default_rng(7)
    run_lengths = [
        _mean_run_length(stationary_bootstrap_indices(n, p, rng)) for _ in range(40)
    ]
    mean_rl = float(np.mean(run_lengths))
    target = 1.0 / p
    assert 0.8 * target <= mean_rl <= 1.2 * target


def test_bootstrap_indices_valid_range_small_n() -> None:
    """PROPERTY: every resampled index lies in [0, n) including the wrap-around step.

    The continue-block branch advances (cur + 1) % n, so a small n exercises the
    modular wrap; no index may fall outside the valid range.
    """
    rng = np.random.default_rng(9)
    for n in (1, 2, 5, 17):
        idx = stationary_bootstrap_indices(n, 0.05, rng)
        assert idx.min() >= 0 and idx.max() < n
        assert idx.shape == (n,)


# ---------------------------------------------------------------------------
# 3. DIFFERENCE TESTS (sharpe / cvar) — bootstrap.py
# ---------------------------------------------------------------------------
def test_sharpe_diff_pvalue_uniform_under_null(equal_sharpe_pair) -> None:
    """PROPERTY: under H0 (equal Sharpe) the p-value distribution is ~Uniform(0,1).

    A correctly sized test yields uniform p-values under the null. Uses the
    equal_sharpe_pair fixture's distribution; a lenient KS test (p > 0.01) catches
    gross miscalibration without flaking on Monte-Carlo noise.
    """
    from scipy.stats import kstest

    a0, b0 = equal_sharpe_pair  # establishes the null distribution shape
    n_obs = a0.size
    pvals = []
    for s in range(120):
        gen = np.random.default_rng(1000 + s)
        a = 0.0005 + 0.01 * gen.standard_normal(n_obs)
        b = 0.0005 + 0.01 * gen.standard_normal(n_obs)
        res = sharpe_difference_test(a, b, n_boot=400, rng=gen)
        pvals.append(res["pvalue"])
    ks = kstest(pvals, "uniform")
    assert ks.pvalue > 0.01  # do not reject uniformity (lenient)


def test_sharpe_diff_detects_mean_separation() -> None:
    """PROPERTY: a real Sharpe gap is detected (p < 0.05) and sign tracks the winner.

    With a >> b in mean (same vol, long series) the test must reject H0 and the
    reported effect must be positive when a is the better arm (and negative when
    the arms are swapped) — the statistic SIGN tracks which arm is better.
    """
    rng = np.random.default_rng(2024)
    a = 0.0020 + 0.01 * rng.standard_normal(3000)  # high Sharpe
    b = 0.0001 + 0.01 * rng.standard_normal(3000)  # near-zero Sharpe
    res_ab = sharpe_difference_test(a, b, n_boot=1500, rng=rng)
    res_ba = sharpe_difference_test(b, a, n_boot=1500, rng=rng)
    assert res_ab["pvalue"] < 0.05
    # SR(a) > SR(b): observed diff (and hence stat) positive for (a,b), negative for (b,a).
    assert res_ab["stat"] > 0.0
    assert res_ba["stat"] < 0.0


def test_cvar_diff_detects_tail_separation_and_sign() -> None:
    """PROPERTY: a real tail-loss gap is detected and the stat sign tracks the safer arm.

    Arm a has a far heavier lower tail (more negative CVaR) than b, so CVaR(a)-CVaR(b)
    < 0; the test must reject H0 and the statistic sign must flip when arms are swapped.
    """
    rng = np.random.default_rng(303)
    # a: heavy negative tail (Student-t, df=3); b: light Gaussian tail.
    a = 0.01 * rng.standard_t(3, size=3000)
    b = 0.01 * rng.standard_normal(3000)
    res_ab = cvar_difference_test(a, b, alpha=0.05, n_boot=1500, rng=rng)
    res_ba = cvar_difference_test(b, a, alpha=0.05, n_boot=1500, rng=rng)
    assert res_ab["pvalue"] < 0.05
    # CVaR(a) is more negative => diff < 0 => stat < 0 for (a,b), > 0 for (b,a).
    assert res_ab["stat"] < 0.0
    assert res_ba["stat"] > 0.0


# ---------------------------------------------------------------------------
# 4. ES BACKTEST / DM — es_backtest.py
# ---------------------------------------------------------------------------
def test_hln_factor_shrinks_and_tends_to_one() -> None:
    """PROPERTY: the HLN factor is <1 (shrinks |DM stat|) and -> 1 as T grows.

    Harvey-Leybourne-Newbold (1997): the small-sample rescaling
    sqrt((T+1-2h+h(h-1)/T)/T) is < 1 for h=1, T>1 (more conservative statistic),
    and converges to 1 as T -> infinity (the asymptotic DM regime is recovered).
    """
    # h=1: factor = sqrt((T+1-2)/T) = sqrt((T-1)/T) < 1 for all T>1, increasing in T.
    f_small = hln_factor(10, h=1)
    f_mid = hln_factor(250, h=1)
    f_large = hln_factor(1_000_000, h=1)
    assert 0.0 < f_small < f_mid < f_large < 1.0
    assert f_large == pytest.approx(1.0, abs=1e-3)


def test_dm_hln_factor_shrinks_statistic_relative_to_normal() -> None:
    """PROPERTY: the HLN-corrected DM statistic has smaller magnitude than the raw DM.

    Since dm_stat_hln = dm_stat * hln_factor(T,h) and the factor is <1, the corrected
    statistic is shrunk toward zero (more conservative) — verified on a concrete
    score differential via the wired dm_hln_test.
    """
    from src.inference.es_backtest import dm_hln_test

    rng = np.random.default_rng(55)
    d = 0.3 + rng.standard_normal(40)  # non-trivial mean, modest T
    res = dm_hln_test(d, h=1)
    assert abs(res["dm_stat_hln"]) < abs(res["dm_stat"])
    assert res["dm_stat_hln"] == pytest.approx(res["dm_stat"] * res["hln_factor"], rel=1e-9)


def test_dm_empirical_size_within_tolerance_under_null() -> None:
    """PROPERTY: under H0 (zero-mean differential) empirical size <= nominal + tol.

    A well-calibrated DM test rejects at ~alpha under the null. The HLN-corrected
    size should sit at or below the nominal level (it is the conservative variant),
    while the uncorrected normal test may over-reject slightly at small T. Lenient
    tolerance avoids Monte-Carlo flakiness while catching gross mis-sizing.
    """
    res = dm_size_power_calibration(
        t=60, alpha=0.05, effect=0.5, n_reps=600, seed=4321
    )
    # HLN is the conservative headline variant: size at or below nominal + small tol.
    assert res["size_hln"] <= 0.05 + 0.03
    # The uncorrected normal variant may over-reject but not wildly.
    assert res["size_normal"] <= 0.05 + 0.06
    # Power at a real effect must exceed size (the test can detect a true gap).
    assert res["power_hln"] > res["size_hln"]


# ---------------------------------------------------------------------------
# 5. MULTIPLE TESTING — multiple_testing.py
# ---------------------------------------------------------------------------
def test_bh_rejection_set_is_clean_threshold() -> None:
    """PROPERTY: the BH rejection set is a clean prefix in sorted-p order (no gaps).

    Benjamini-Hochberg is a step-up procedure: it rejects exactly the k smallest
    p-values for some cutoff k. So when reordered by raw p-value the rejection mask
    must be a contiguous block of Trues followed by Falses — never an interior gap
    (a rejected hypothesis with a SMALLER raw p than an accepted one).
    """
    rng = np.random.default_rng(606)
    pvals = np.sort(rng.random(60))
    pvals[:8] = np.array([1e-5, 1e-4, 1e-3, 0.002, 0.005, 0.01, 0.02, 0.03])
    rng2 = np.random.default_rng(7)
    perm = rng2.permutation(pvals.size)
    shuffled = pvals[perm]
    rej = benjamini_hochberg(shuffled, q=0.1)
    # Reorder rejections back into ascending-p order; must be [True...True False...False].
    order = np.argsort(shuffled, kind="mergesort")
    rej_sorted = rej[order]
    # No accepted hypothesis may precede (have smaller p than) a rejected one.
    first_false = np.argmin(rej_sorted) if not rej_sorted.all() else rej_sorted.size
    assert rej_sorted[:first_false].all()  # clean prefix of Trues
    assert not rej_sorted[first_false:].any()  # nothing rejected after the cutoff


def test_bh_trivial_and_all_equal_no_nan() -> None:
    """PROPERTY: m=1 and all-equal-p inputs return a clean boolean mask, never NaN.

    Edge cases the family-wise machinery must survive: a single hypothesis, and a
    degenerate family of identical p-values (the step-up threshold must still be
    well-defined). The output is a boolean array, so it can never be NaN, but it
    must also make the correct accept/reject call.
    """
    # m=1: a tiny p is rejected; a large p is not.
    assert benjamini_hochberg(np.array([1e-6]), q=0.1).tolist() == [True]
    assert benjamini_hochberg(np.array([0.9]), q=0.1).tolist() == [False]
    # All-equal small p-values: all rejected (each p <= k/m * q at the top rank).
    rej_small = benjamini_hochberg(np.full(10, 0.001), q=0.1)
    assert rej_small.dtype == bool and rej_small.all()
    # All-equal large p-values: none rejected.
    rej_large = benjamini_hochberg(np.full(10, 0.8), q=0.1)
    assert rej_large.dtype == bool and not rej_large.any()


def test_romano_wolf_is_order_aware_stepdown() -> None:
    """PROPERTY: Romano-Wolf is a step-down procedure ordered by observed statistic.

    The stepdown rejects from the largest statistic downward and STOPS at the first
    non-rejection, so the rejection set is order-coherent: it can never reject a
    hypothesis with a smaller statistic while failing to reject one with a larger
    statistic. Verified on a graded signal where only the top few exceed the null.
    """
    rng = np.random.default_rng(909)
    m = 8
    boot = rng.standard_normal((4000, m))  # centred null draws, unit scale
    # Graded statistics: two strong, the rest null-sized.
    stats_obs = np.array([6.0, 4.5, 0.4, 0.2, 0.1, 0.3, 0.0, 0.15])
    rej = romano_wolf(stats_obs, boot, alpha=0.05)
    assert rej[0] and rej[1]  # the two strong signals are rejected
    # Order-coherence: no rejected stat may be smaller than an un-rejected stat.
    rejected_stats = stats_obs[rej]
    accepted_stats = stats_obs[~rej]
    if rejected_stats.size and accepted_stats.size:
        assert rejected_stats.min() > accepted_stats.max()


def test_romano_wolf_trivial_m1_no_error() -> None:
    """PROPERTY: a single-hypothesis (m=1) family is handled cleanly (no NaN/raise).

    With m=1 the max-statistic null is just that hypothesis's bootstrap column; a
    statistic above the 1-alpha quantile rejects, below does not.
    """
    rng = np.random.default_rng(11)
    boot = rng.standard_normal((3000, 1))
    assert romano_wolf(np.array([5.0]), boot, alpha=0.05).tolist() == [True]
    assert romano_wolf(np.array([0.0]), boot, alpha=0.05).tolist() == [False]


def test_no_inference_docstring_claims_a_CERTIFIED_size() -> None:
    """The bootstrap tests' size is MEASURED (and anti-conservative), never "certified".

    Regression for a real residual-overstatement defect (deep code-review loop 77, 2026-07-26).
    ``bootstrap.py``'s module docstring already recorded that the word "certified" "was an
    overstatement … and at the call sites below", because the measured size at production settings is
    **0.0573 two-sided / 0.0613 one-sided** against nominal 0.05 — mildly ANTI-conservative. But that
    loop-5 correction was only added centrally: four call sites in ``bootstrap.py`` (including
    ``paired_seed_difference_test``, the headline test) and one in ``es_backtest.py`` still asserted an
    unqualified certification. The ``es_backtest.py`` site was the worst — a different module, so a
    reader there had no correction anywhere in view.

    A size claim is the kind of thing a referee checks, so this asserts the prose directly: any
    "certif*" wording in these modules must sit next to the measured qualifier."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "inference"
    offenders = []
    for mod in ("bootstrap.py", "es_backtest.py"):
        # whitespace COLLAPSED so a line-wrapped claim cannot slip through (the loop-76 lesson: a
        # naive substring test missed a real defect whose phrase straddled a newline)
        text = re.sub(r"\s+", " ", (root / mod).read_text(encoding="utf-8")).lower()
        for m in re.finditer(r"size (?:is |was )?(?:\w+ ){0,2}certified", text):
            window = text[max(0, m.start() - 200): m.end() + 200]
            if "measured" not in window and "overstatement" not in window:
                offenders.append(f"{mod}: ...{text[m.start() - 60:m.end() + 60]}...")
    assert not offenders, (
        "these claim a CERTIFIED size with no measured qualifier in view; the measured size is "
        f"0.0573/0.0613 vs nominal 0.05 (anti-conservative): {offenders}"
    )
