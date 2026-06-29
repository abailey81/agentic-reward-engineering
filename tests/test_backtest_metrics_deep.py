"""DEEP mathematical-property tests for ``src.backtest.metrics`` (block B11).

These are *identity* tests for the performance/risk layer that underlies the dissertation's
backtest reporting: closed-form analytic matches, metamorphic transformation laws (scale / shift /
permute / reverse), and boundary / adversarial degeneracy. They complement (do NOT duplicate)
``tests/test_backtest_metrics.py``, which covers hand series, reuse-consistency, key presence, and
basic ranges.

Everything below tests ONLY the real public surface of the module
(``compute_metrics``, ``drawdown_series``, ``regime_conditional_metrics``, ``tearsheet_markdown``)
and the audited primitives it reuses (``src.inference.bootstrap.{sharpe_ratio, cvar}``).

Implementation facts pinned here (verified against the source, 2026-06):
  * ``compute_metrics["sharpe"]`` reuses ``_sharpe`` which uses the POPULATION std (ddof=0).
  * ``compute_metrics["ann_volatility"]`` uses the SAMPLE std (ddof=1).
  * ``cvar(r, a)`` is the mean of the worst ``ceil(a*T)`` returns.
  * ``var_hist_{tag}`` is ``-np.quantile(r, a)`` (positive loss magnitude).
  * ``max_drawdown`` is the POSITIVE magnitude (>= 0); CAGR/Calmar/Sortino degrade to a finite 0.0.
  * Omega uses a FIXED tau = 0 MAR (rf-invariant); no-downside -> 1e9 sentinel.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

hyp = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from hypothesis.extra import numpy as hnp  # noqa: E402

from src.backtest.metrics import (  # noqa: E402
    PERIODS_PER_YEAR,
    compute_metrics,
    drawdown_series,
    regime_conditional_metrics,
    tearsheet_markdown,
)
from src.inference.bootstrap import cvar as _cvar  # noqa: E402
from src.inference.bootstrap import sharpe_ratio as _sharpe  # noqa: E402

# A daily-scale return strategy that stays numerically benign: bounded, finite, away from -100%
# (so wealth never hits the ruin branch unless a test asks for it). width_min keeps |r| modest.
_RET = hnp.arrays(
    dtype=np.float64,
    shape=hnp.array_shapes(min_dims=1, max_dims=1, min_side=2, max_side=64),
    elements=st.floats(min_value=-0.25, max_value=0.25, allow_nan=False, allow_infinity=False),
)


# =========================================================================== #
# Sharpe — analytic, scale-invariance, sign, annualisation                    #
# =========================================================================== #
def test_sharpe_closed_form_constant_vol_two_point_series() -> None:
    # A balanced two-value series {+a, -a} repeated: mean=0 -> Sharpe=0 exactly (guarded const-tail).
    r = np.tile([0.02, -0.02], 50)
    assert compute_metrics(r)["sharpe"] == pytest.approx(0.0, abs=1e-12)


def test_sharpe_matches_population_std_definition() -> None:
    # PIN the ddof: compute_metrics reuses _sharpe (ddof=0), NOT ann_return/ann_vol (which is ddof=1).
    rng = np.random.default_rng(101)
    r = rng.standard_normal(500) * 0.01 + 0.0007
    mu, sd0 = r.mean(), r.std(ddof=0)
    expected = mu / sd0 * math.sqrt(PERIODS_PER_YEAR)
    assert compute_metrics(r)["sharpe"] == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_sharpe_is_NOT_ann_return_over_ann_vol_ddof_mismatch() -> None:
    # Documents the deliberate ddof asymmetry: sharpe (ddof=0) != ann_return_arith/ann_volatility (ddof=1)
    # for finite n -> they must differ by the (n-1)/n Bessel factor in the std, never accidentally align.
    rng = np.random.default_rng(7)
    r = rng.standard_normal(40) * 0.01 + 0.001
    m = compute_metrics(r)
    naive = m["ann_return_arith"] / m["ann_volatility"]
    # sharpe uses population std (smaller) -> magnitude strictly larger than the sample-std ratio.
    assert abs(m["sharpe"]) > abs(naive)
    ratio = abs(m["sharpe"]) / abs(naive)
    assert ratio == pytest.approx(math.sqrt(40 / 39), rel=1e-9)


# Scale factor kept modest so k*r remains a sane return series (k*r > -1, compounding stays finite);
# the scale-equivariance identity itself is exact regardless, this just keeps the full-suite call benign.
_POS_SCALE = st.floats(min_value=0.1, max_value=3.0, allow_nan=False, allow_infinity=False)


@settings(derandomize=True, max_examples=150)
@given(_RET, _POS_SCALE)
def test_sharpe_scale_equivariant_under_positive_scale(r: np.ndarray, k: float) -> None:
    # Sharpe = mean/std is scale-INVARIANT under r -> k*r (k>0): both numerator and denominator scale by k.
    base = compute_metrics(r)["sharpe"]
    scaled = compute_metrics(k * r)["sharpe"]
    assert scaled == pytest.approx(base, rel=1e-9, abs=1e-9)


@settings(derandomize=True, max_examples=150)
@given(_RET)
def test_sharpe_sign_tracks_mean_excess(r: np.ndarray) -> None:
    if np.ptp(r) == 0.0:  # constant -> guarded to 0
        assert compute_metrics(r)["sharpe"] == 0.0
        return
    s = compute_metrics(r)["sharpe"]
    mu = float(r.mean())
    if abs(mu) < 1e-9:
        return  # near-zero mean: sign is numerically ambiguous, skip
    assert math.copysign(1.0, s) == math.copysign(1.0, mu)


@settings(derandomize=True, max_examples=120)
@given(_RET, st.integers(min_value=12, max_value=504))
def test_sharpe_annualisation_factor_is_sqrt_ppy(r: np.ndarray, ppy: int) -> None:
    if np.ptp(r) == 0.0:
        return
    s1 = compute_metrics(r, periods_per_year=1)["sharpe"]
    sp = compute_metrics(r, periods_per_year=ppy)["sharpe"]
    assert sp == pytest.approx(s1 * math.sqrt(ppy), rel=1e-9, abs=1e-12)


@settings(derandomize=True, max_examples=120)
@given(_RET)
def test_sharpe_permutation_invariant(r: np.ndarray) -> None:
    perm = np.random.default_rng(0).permutation(r)
    assert compute_metrics(perm)["sharpe"] == pytest.approx(compute_metrics(r)["sharpe"], rel=1e-9, abs=1e-12)


# =========================================================================== #
# Sortino & downside deviation                                                #
# =========================================================================== #
def test_sortino_closed_form_known_series() -> None:
    # r = [+0.02, -0.01] repeated. rf=0, target=0.
    #   mean = 0.005 ; downside dev = sqrt(mean(min(r,0)^2)) = sqrt(((0)^2+(0.01)^2)/2) = 0.01/sqrt(2)
    #   sortino = mean*ppy / (dd*sqrt(ppy)) = 0.005*252 / (0.01/sqrt(2)*sqrt(252))
    r = np.tile([0.02, -0.01], 100)
    ppy = 252
    dd_dev = math.sqrt(((0.0) ** 2 + (0.01) ** 2) / 2.0) * math.sqrt(ppy)
    expected = (r.mean() * ppy) / dd_dev
    assert compute_metrics(r)["sortino"] == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_sortino_ge_sharpe_when_downside_dev_le_total_vol() -> None:
    # For a positive-mean, right-skewed series the downside deviation (ddof-0 RMS of shortfalls below 0)
    # is <= the total annualised vol, so Sortino >= Sharpe. Use rf=0 so the numerators match exactly.
    # Build a series with genuine (but small) downside: occasional small losses, frequent larger gains.
    rng = np.random.default_rng(11)
    gains = rng.uniform(0.001, 0.03, size=900)
    losses = -rng.uniform(0.0005, 0.005, size=100)  # smaller-magnitude losses -> downside dev < total vol
    r = np.concatenate([gains, losses])
    rng.shuffle(r)
    m = compute_metrics(r)
    assert m["downside_deviation_ann"] > 0.0  # genuine downside present
    assert m["sortino"] >= m["sharpe"] - 1e-9


@settings(derandomize=True, max_examples=120)
@given(_RET, _POS_SCALE)
def test_sortino_scale_invariant(r: np.ndarray, k: float) -> None:
    # mean and downside-deviation both scale by k -> ratio invariant (rf=0).
    base = compute_metrics(r)["sortino"]
    scaled = compute_metrics(k * r)["sortino"]
    assert scaled == pytest.approx(base, rel=1e-9, abs=1e-9)


def test_downside_deviation_zero_for_all_gains() -> None:
    r = np.full(50, 0.01)
    assert compute_metrics(r)["downside_deviation_ann"] == pytest.approx(0.0, abs=1e-12)
    assert compute_metrics(r)["sortino"] == 0.0  # documented finite sentinel (dd_dev <= eps)


# =========================================================================== #
# Max drawdown — sign convention, monotone, hand series, scale invariance     #
# =========================================================================== #
@settings(derandomize=True, max_examples=200)
@given(_RET)
def test_max_drawdown_in_unit_interval(r: np.ndarray) -> None:
    mdd = drawdown_series(r)["max_drawdown"]
    assert 0.0 <= mdd <= 1.0 + 1e-12  # POSITIVE magnitude convention, bounded by total ruin


def test_max_drawdown_zero_for_monotone_increasing_equity() -> None:
    dd = drawdown_series(np.full(30, 0.005))
    assert dd["max_drawdown"] == 0.0
    assert dd["avg_drawdown"] == 0.0
    assert dd["time_under_water"] == 0.0
    assert dd["max_drawdown_duration"] == 0


def test_max_drawdown_hand_built_peak_trough() -> None:
    # Wealth: 1 -> 1.2 (peak) -> 0.6 -> 0.66. Deepest trough 0.6 vs peak 1.2 -> DD = 0.6/1.2 - 1 = -0.5.
    r = np.array([0.20, -0.50, 0.10])
    dd = drawdown_series(r)
    assert dd["max_drawdown"] == pytest.approx(0.50, abs=1e-12)


@settings(derandomize=True, max_examples=120)
@given(_RET)
def test_max_drawdown_invariant_to_trailing_zero_returns(r: np.ndarray) -> None:
    # Appending 0% returns extends the wealth curve flat at its final level: it adds no new peak and no
    # new trough, so the running-peak-relative max drawdown is exactly preserved. (A LEADING zero is NOT
    # invariant here: drawdown is measured from the first realised wealth point, not from unit capital,
    # so a leading 0.0 injects a fresh peak above a series that declines from the start — a documented
    # convention of drawdown_series, deliberately not asserted.)
    base = drawdown_series(r)["max_drawdown"]
    padded = drawdown_series(np.concatenate([r, [0.0, 0.0]]))["max_drawdown"]
    assert padded == pytest.approx(base, abs=1e-12)


def test_max_drawdown_invariant_to_initial_capital_via_closed_form() -> None:
    # The DEFINING scale-invariance: drawdown is a ratio from the running peak, so multiplying the whole
    # wealth curve by any c>0 cancels. drawdown_series fixes initial capital at 1.0, so we verify the
    # property against a hand wealth curve scaled by c and the module's own peak-relative formula.
    r = np.array([0.20, -0.50, 0.10, 0.30, -0.40])
    dd = drawdown_series(r)
    wealth = np.cumprod(1.0 + r)
    for c in (0.01, 1.0, 137.0):
        w = c * wealth
        peak = np.maximum.accumulate(w)
        expected_mdd = float(-(w / peak - 1.0).min())  # scale c cancels in the ratio
        assert dd["max_drawdown"] == pytest.approx(expected_mdd, abs=1e-12)


def test_drawdown_total_ruin_is_minus_one_not_nan() -> None:
    # A -100% period drives wealth to 0; max_drawdown must be 1.0 (full magnitude), finite, no NaN.
    r = np.array([0.05, -1.0, 0.10])
    dd = drawdown_series(r)
    assert dd["max_drawdown"] == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.isfinite(dd["drawdown"]))


def test_drawdown_empty_series_zeroed() -> None:
    dd = drawdown_series([])
    assert dd["max_drawdown"] == 0.0
    assert dd["wealth"].size == 0
    assert dd["drawdown"].size == 0
    assert dd["max_drawdown_duration"] == 0


# =========================================================================== #
# Calmar — definition, div-by-zero fallback                                   #
# =========================================================================== #
def test_calmar_equals_cagr_over_maxdd() -> None:
    rng = np.random.default_rng(21)
    r = rng.standard_normal(750) * 0.01 + 0.0005
    m = compute_metrics(r)
    # The fixture must actually reach the regime under test (deep review 2026-07-26, #73 pattern): this
    # test's ONLY assertion sits under the guard, so a fixture that produced no drawdown would assert
    # nothing and still pass. The zero-drawdown branch is covered by the sibling test below. MEASURED for
    # seed 21: max_drawdown = 0.2517.
    assert m["max_drawdown"] > 1e-12, "fixture drifted: no drawdown, so this test would assert nothing"
    assert m["calmar"] == pytest.approx(m["cagr"] / m["max_drawdown"], rel=1e-12, abs=1e-12)


def test_calmar_zero_when_no_drawdown() -> None:
    # Monotone gains -> max_drawdown == 0 -> documented finite 0.0 fallback (no div-by-zero NaN/inf).
    m = compute_metrics(np.full(40, 0.01))
    assert m["max_drawdown"] == 0.0
    assert m["calmar"] == 0.0
    assert np.isfinite(m["calmar"])


# =========================================================================== #
# CVaR / VaR — closed form, monotonicity in alpha, ordering, permutation      #
# =========================================================================== #
def test_cvar_closed_form_known_sample() -> None:
    # 20 sorted values; worst ceil(0.10*20)=2 are the two smallest -> their mean.
    r = np.linspace(-0.10, 0.09, 20)  # -0.10, -0.09, ... step 0.01
    m = compute_metrics(r, var_levels=(0.10,))
    expected = (r[0] + r[1]) / 2.0
    assert m["cvar_10"] == pytest.approx(expected, abs=1e-12)
    assert m["cvar_10"] == pytest.approx(_cvar(r, 0.10), abs=1e-12)


def test_var_hist_is_negative_quantile() -> None:
    rng = np.random.default_rng(31)
    r = rng.standard_normal(5000) * 0.01
    m = compute_metrics(r, var_levels=(0.05,))
    assert m["var_hist_05"] == pytest.approx(-np.quantile(r, 0.05), abs=1e-12)
    assert m["var_hist_05"] > 0.0  # loss magnitude reported positive


def test_cvar_more_extreme_than_var_at_same_level() -> None:
    # CVaR (mean of the worst tail) is at least as bad as VaR (the tail boundary) -> cvar <= -var_hist.
    rng = np.random.default_rng(32)
    r = rng.standard_normal(10000) * 0.01
    m = compute_metrics(r, var_levels=(0.05,))
    assert m["cvar_05"] <= -m["var_hist_05"] + 1e-9


def test_cvar_monotone_in_alpha() -> None:
    # A smaller alpha selects a more extreme (more negative / >= as bad) tail mean.
    rng = np.random.default_rng(33)
    r = rng.standard_normal(20000) * 0.01
    m = compute_metrics(r, var_levels=(0.01, 0.05, 0.10))
    assert m["cvar_01"] <= m["cvar_05"] + 1e-12
    assert m["cvar_05"] <= m["cvar_10"] + 1e-12


@settings(derandomize=True, max_examples=120)
@given(_RET)
def test_cvar_permutation_invariant(r: np.ndarray) -> None:
    perm = np.random.default_rng(1).permutation(r)
    a = compute_metrics(r, var_levels=(0.05,))["cvar_05"]
    b = compute_metrics(perm, var_levels=(0.05,))["cvar_05"]
    assert b == pytest.approx(a, rel=1e-9, abs=1e-12)


@settings(derandomize=True, max_examples=120)
@given(_RET, _POS_SCALE)
def test_cvar_positive_homogeneous(r: np.ndarray, k: float) -> None:
    # Coherence: CVaR is positively homogeneous -> CVaR(k*r) = k*CVaR(r) for k>0.
    base = compute_metrics(r, var_levels=(0.05,))["cvar_05"]
    scaled = compute_metrics(k * r, var_levels=(0.05,))["cvar_05"]
    assert scaled == pytest.approx(k * base, rel=1e-9, abs=1e-12)


@settings(derandomize=True, max_examples=120)
@given(_RET, st.floats(min_value=-0.05, max_value=0.05, allow_nan=False, allow_infinity=False))
def test_cvar_translation_equivariant(r: np.ndarray, c: float) -> None:
    # Coherence: CVaR(r + c) = CVaR(r) + c (adding a constant shifts the tail mean by c).
    base = compute_metrics(r, var_levels=(0.05,))["cvar_05"]
    shifted = compute_metrics(r + c, var_levels=(0.05,))["cvar_05"]
    assert shifted == pytest.approx(base + c, rel=1e-9, abs=1e-9)


# =========================================================================== #
# Volatility & return — annualisation, scale, shift                           #
# =========================================================================== #
def test_ann_volatility_closed_form_sample_std() -> None:
    rng = np.random.default_rng(41)
    r = rng.standard_normal(300) * 0.01
    m = compute_metrics(r)
    assert m["ann_volatility"] == pytest.approx(r.std(ddof=1) * math.sqrt(252), rel=1e-12, abs=1e-12)


@settings(derandomize=True, max_examples=120)
@given(_RET, _POS_SCALE)
def test_ann_volatility_scales_linearly(r: np.ndarray, k: float) -> None:
    base = compute_metrics(r)["ann_volatility"]
    assert compute_metrics(k * r)["ann_volatility"] == pytest.approx(k * base, rel=1e-9, abs=1e-12)


@settings(derandomize=True, max_examples=120)
@given(_RET, st.floats(min_value=-0.05, max_value=0.05, allow_nan=False, allow_infinity=False))
def test_ann_volatility_shift_invariant(r: np.ndarray, c: float) -> None:
    # Adding a constant to every return leaves the (sample) volatility unchanged.
    base = compute_metrics(r)["ann_volatility"]
    assert compute_metrics(r + c)["ann_volatility"] == pytest.approx(base, rel=1e-9, abs=1e-12)


def test_ann_return_arith_is_mean_times_ppy() -> None:
    rng = np.random.default_rng(42)
    r = rng.standard_normal(252) * 0.01 + 0.0003
    assert compute_metrics(r)["ann_return_arith"] == pytest.approx(r.mean() * 252, rel=1e-12)


def test_total_return_is_compounded_product() -> None:
    r = np.array([0.10, -0.05, 0.02])
    expected = (1.10 * 0.95 * 1.02) - 1.0
    assert compute_metrics(r)["total_return"] == pytest.approx(expected, rel=1e-12)


def test_cagr_geometric_closed_form() -> None:
    # Constant 0.01 over 252 periods, ppy=252 -> CAGR = (1.01^252)^(252/252) - 1 = 1.01^252 - 1.
    r = np.full(252, 0.01)
    assert compute_metrics(r)["cagr"] == pytest.approx(1.01**252 - 1.0, rel=1e-12)


def test_cagr_total_ruin_reports_minus_one() -> None:
    r = np.array([0.05, -1.0, 0.20])  # growth product = 0 -> documented -1 (no complex root)
    assert compute_metrics(r)["cagr"] == -1.0


# =========================================================================== #
# Turnover & cost drag                                                        #
# =========================================================================== #
def test_turnover_ann_and_cost_drag_closed_form() -> None:
    rng = np.random.default_rng(51)
    tov = np.abs(rng.standard_normal(252)) * 0.05
    r = rng.standard_normal(252) * 0.01
    m = compute_metrics(r, turnover=tov, cost_bps=12.0)
    assert m["turnover_ann"] == pytest.approx(tov.mean() * 252, rel=1e-12)
    assert m["cost_drag_ann"] == pytest.approx(tov.mean() * 12.0 * 1e-4 * 252, rel=1e-12)


def test_turnover_nonnegative_and_zero_for_no_trading() -> None:
    r = np.random.default_rng(52).standard_normal(100) * 0.01
    m = compute_metrics(r, turnover=np.zeros(100), cost_bps=10.0)
    assert m["turnover_ann"] == 0.0
    assert m["cost_drag_ann"] == 0.0


def test_no_turnover_keys_when_turnover_absent() -> None:
    m = compute_metrics(np.random.default_rng(53).standard_normal(50) * 0.01)
    assert "turnover_ann" not in m
    assert "cost_drag_ann" not in m


# =========================================================================== #
# Omega / profit factor / gain-loss — shape ratios & sentinels                #
# =========================================================================== #
def test_omega_reciprocal_under_negation() -> None:
    # Omega(r) about tau=0 = sum(gains)/sum(losses). Negating r swaps gains<->losses -> reciprocal.
    rng = np.random.default_rng(61)
    r = rng.standard_normal(500) * 0.01 + 0.0001
    o_pos = compute_metrics(r)["omega"]
    o_neg = compute_metrics(-r)["omega"]
    assert o_pos * o_neg == pytest.approx(1.0, rel=1e-9)


def test_omega_scale_invariant() -> None:
    rng = np.random.default_rng(62)
    r = rng.standard_normal(400) * 0.01 + 0.0002
    assert compute_metrics(3.7 * r)["omega"] == pytest.approx(compute_metrics(r)["omega"], rel=1e-9)


def test_omega_no_downside_uses_finite_sentinel() -> None:
    m = compute_metrics(np.full(20, 0.01))
    assert m["omega"] == 1e9  # documented large finite sentinel, never +inf
    assert np.isfinite(m["omega"])


def test_profit_factor_and_gain_loss_no_loss_sentinel() -> None:
    m = compute_metrics(np.full(15, 0.02))
    assert m["profit_factor"] == 1e9
    assert m["gain_loss_ratio"] == 1e9


def test_profit_factor_closed_form() -> None:
    r = np.array([0.03, 0.01, -0.02, -0.02])
    # wins.sum()=0.04, |loss.sum()|=0.04 -> profit_factor = 1.0
    assert compute_metrics(r)["profit_factor"] == pytest.approx(1.0, rel=1e-12)


# =========================================================================== #
# Hit rate / pct_positive / best & worst period                              #
# =========================================================================== #
def test_hit_rate_closed_form_and_bounds() -> None:
    r = np.array([0.01, -0.01, 0.0, 0.02, -0.03])  # 2 of 5 strictly positive
    m = compute_metrics(r)
    assert m["hit_rate"] == pytest.approx(2 / 5, abs=1e-12)
    assert m["pct_positive"] == pytest.approx(2 / 5, abs=1e-12)
    assert 0.0 <= m["hit_rate"] <= 1.0


def test_best_and_worst_period_are_extrema() -> None:
    rng = np.random.default_rng(71)
    r = rng.standard_normal(300) * 0.01
    m = compute_metrics(r)
    assert m["best_period"] == pytest.approx(r.max(), abs=1e-12)
    assert m["worst_period"] == pytest.approx(r.min(), abs=1e-12)


# =========================================================================== #
# Benchmark-relative — beta, alpha, IR, tracking error                        #
# =========================================================================== #
def test_beta_one_and_zero_alpha_against_self() -> None:
    rng = np.random.default_rng(81)
    b = rng.standard_normal(1000) * 0.01
    m = compute_metrics(b, benchmark=b)
    assert m["beta"] == pytest.approx(1.0, abs=1e-9)
    assert m["alpha_ann"] == pytest.approx(0.0, abs=1e-9)
    assert m["tracking_error_ann"] == pytest.approx(0.0, abs=1e-9)
    assert m["information_ratio"] == 0.0  # zero active risk -> documented 0 fallback


def test_alpha_recovers_known_constant() -> None:
    rng = np.random.default_rng(82)
    bench = rng.standard_normal(3000) * 0.01
    daily_alpha = 0.0004
    r = 1.0 * bench + daily_alpha  # beta=1, constant alpha
    m = compute_metrics(r, benchmark=bench)
    assert m["beta"] == pytest.approx(1.0, abs=1e-6)
    assert m["alpha_ann"] == pytest.approx(daily_alpha * 252, abs=1e-4)


def test_tracking_error_zero_when_identical_and_positive_otherwise() -> None:
    rng = np.random.default_rng(83)
    bench = rng.standard_normal(1000) * 0.01
    r = bench + rng.standard_normal(1000) * 0.005
    assert compute_metrics(r, benchmark=bench)["tracking_error_ann"] > 0.0


# =========================================================================== #
# Metamorphic full-suite reversal: time-reversing the series                  #
# =========================================================================== #
@settings(derandomize=True, max_examples=80)
@given(_RET)
def test_time_reversal_preserves_distribution_metrics(r: np.ndarray) -> None:
    # Order-independent (distributional) metrics are invariant to reversing the series; path-dependent
    # ones (drawdown) need not be. Assert the distributional set is preserved exactly.
    fwd = compute_metrics(r)
    rev = compute_metrics(r[::-1])
    for key in ("sharpe", "ann_volatility", "ann_return_arith", "cvar_05", "var_hist_05", "hit_rate",
                "omega", "skew", "excess_kurtosis", "best_period", "worst_period", "total_return"):
        assert rev[key] == pytest.approx(fwd[key], rel=1e-9, abs=1e-12), key


# =========================================================================== #
# Boundary / adversarial inputs                                               #
# =========================================================================== #
def test_empty_series_returns_only_n_periods() -> None:
    m = compute_metrics([])
    assert m == {"n_periods": 0.0}


def test_single_element_series_is_finite() -> None:
    m = compute_metrics([0.01])
    assert m["n_periods"] == 1.0
    # n=1 -> sample std 0 -> ann_volatility 0, sharpe guarded to 0, all finite.
    assert all(np.isfinite(v) for v in m.values() if isinstance(v, float))
    assert m["ann_volatility"] == 0.0
    assert m["sharpe"] == 0.0


def test_all_zero_series_zero_variance_policy() -> None:
    m = compute_metrics(np.zeros(100))
    assert m["sharpe"] == 0.0  # zero-variance -> documented finite 0, not NaN/inf
    assert m["ann_volatility"] == 0.0
    assert m["sortino"] == 0.0
    assert m["skew"] == 0.0  # 0/0 shape guarded to 0
    assert m["excess_kurtosis"] == 0.0
    assert m["omega"] == 1.0  # no gains, no losses -> 1.0 sentinel
    assert all(np.isfinite(v) for v in m.values() if isinstance(v, float))


def test_constant_nonzero_series_sharpe_guarded() -> None:
    m = compute_metrics(np.full(50, 0.003))
    assert m["sharpe"] == 0.0  # ptp==0 guard fires despite nonzero mean
    assert np.isfinite(m["sharpe"])


def test_all_negative_series_metrics_finite_and_signed() -> None:
    m = compute_metrics(np.full(40, -0.01))
    assert m["total_return"] < 0.0
    assert m["cagr"] < 0.0
    assert m["max_drawdown"] > 0.0
    assert m["hit_rate"] == 0.0
    assert m["omega"] == pytest.approx(0.0, abs=1e-12)  # no wins -> 0.0
    assert all(np.isfinite(v) for v in m.values() if isinstance(v, float))


def test_nan_inf_stripped_before_compute() -> None:
    r = np.array([0.01, np.nan, -0.02, np.inf, -np.inf, 0.03])
    m = compute_metrics(r)
    assert m["n_periods"] == 3.0  # nan, +inf, -inf all dropped -> 3 finite survive
    # the metrics must equal those of the finite subset (rf=0)
    finite = np.array([0.01, -0.02, 0.03])
    assert m["sharpe"] == pytest.approx(_sharpe(finite), rel=1e-12, abs=1e-12)
    assert all(np.isfinite(v) for v in m.values() if isinstance(v, float))


@settings(derandomize=True, max_examples=200)
@given(_RET)
def test_no_metric_is_nonfinite_on_benign_inputs(r: np.ndarray) -> None:
    # The module's contract: a (benign, bounded) input never yields a NaN/inf metric.
    m = compute_metrics(r, var_levels=(0.01, 0.05, 0.10))
    bad = {k: v for k, v in m.items() if isinstance(v, float) and not np.isfinite(v)}
    assert not bad, bad


# =========================================================================== #
# Regime-conditional invariants                                               #
# =========================================================================== #
def test_regime_all_group_matches_full_series_metrics() -> None:
    rng = np.random.default_rng(91)
    r = rng.standard_normal(400) * 0.01 + 0.0002
    reg = np.where(np.arange(400) % 2 == 0, "calm", "stress")
    rc = regime_conditional_metrics(r, reg, metrics=("sharpe", "cagr", "max_drawdown"))
    full = compute_metrics(r)
    assert rc["all"]["sharpe"] == pytest.approx(full["sharpe"], rel=1e-12)
    assert rc["all"]["cagr"] == pytest.approx(full["cagr"], rel=1e-12)
    assert rc["all"]["n_periods"] == 400.0


def test_regime_subgroup_periods_partition_total() -> None:
    rng = np.random.default_rng(92)
    r = rng.standard_normal(300) * 0.01
    reg = np.where(np.arange(300) % 3 == 0, "stress", "calm")
    rc = regime_conditional_metrics(r, reg)
    assert rc["calm"]["n_periods"] + rc["stress"]["n_periods"] == 300.0


def test_regime_length_mismatch_fails_loud() -> None:
    with pytest.raises(ValueError):
        regime_conditional_metrics(np.zeros(10), np.array(["a"] * 9))


def test_regime_benchmark_misalignment_fails_loud() -> None:
    with pytest.raises(ValueError):
        regime_conditional_metrics(
            np.zeros(10), np.array(["a"] * 10), benchmark=np.zeros(9)
        )


# =========================================================================== #
# Tearsheet rendering                                                         #
# =========================================================================== #
def test_tearsheet_renders_em_dash_for_missing_and_nonfinite() -> None:
    # A metric dict missing a key still renders (em-dash), and a +inf-equivalent is dashed.
    a = compute_metrics(np.random.default_rng(1).standard_normal(200) * 0.01)
    md = tearsheet_markdown({"arm": a}, title="T")
    assert "## T" in md
    assert "| Sharpe |" in md
    assert md.strip().splitlines()[-1].endswith("|")  # well-formed final row


def test_tearsheet_pct_and_num_formatting() -> None:
    a = {"cagr": 0.1234, "sharpe": 1.5, "max_drawdown": 0.05}
    md = tearsheet_markdown({"arm": a})
    assert "12.34%" in md  # cagr formatted as pct
    assert "1.500" in md   # sharpe formatted as num (3dp)
