"""Tests for the comparative ES backtest (Fissler-Ziegel joint elicitability + Nolde-Ziegel DM test).

The load-bearing test is STRICT CONSISTENCY: the FZ0 score must be minimized at the true (VaR, ES).
If that holds, the scoring function is implemented correctly regardless of sign-convention worries.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from src.inference.es_backtest import (
    comparative_es_backtest,
    dm_hln_test,
    dm_size_power_calibration,
    fz0_loss,
    hln_factor,
    var_es_estimates,
)

ALPHA = 0.05
# Analytic standard-normal lower-tail (VaR, ES) at alpha=5%.
TRUE_VAR = float(stats.norm.ppf(ALPHA))                       # ~ -1.6449
TRUE_ES = float(-stats.norm.pdf(TRUE_VAR) / ALPHA)            # ~ -2.0627


def test_fz0_strictly_consistent_minimized_at_truth(normal_returns: np.ndarray) -> None:
    """E[FZ0] is minimized at the true (VaR, ES): every perturbation raises the mean score."""
    base = fz0_loss(normal_returns, TRUE_VAR, TRUE_ES, ALPHA).mean()
    perturbations = [
        (TRUE_VAR * 0.8, TRUE_ES),
        (TRUE_VAR * 1.2, TRUE_ES),
        (TRUE_VAR, TRUE_ES * 0.8),
        (TRUE_VAR, TRUE_ES * 1.2),
        (TRUE_VAR * 0.9, TRUE_ES * 1.1),
    ]
    for v, e in perturbations:
        assert fz0_loss(normal_returns, v, e, ALPHA).mean() > base, f"({v:.3f},{e:.3f}) not worse"


def test_var_es_estimates_match_normal_closed_form(normal_returns: np.ndarray) -> None:
    v, e = var_es_estimates(normal_returns, ALPHA)
    assert v == pytest.approx(TRUE_VAR, abs=0.03)
    assert e == pytest.approx(TRUE_ES, abs=0.05)


def test_fz0_rejects_nonnegative_es(normal_returns: np.ndarray) -> None:
    with pytest.raises(ValueError, match="negative"):
        fz0_loss(normal_returns, TRUE_VAR, 0.0, ALPHA)
    with pytest.raises(ValueError, match="alpha"):
        fz0_loss(normal_returns, TRUE_VAR, TRUE_ES, 1.5)


def test_comparative_backtest_prefers_the_better_forecast(rng: np.random.Generator) -> None:
    """A correctly-specified (VaR, ES) forecast beats one that badly underestimates the tail."""
    realized = rng.standard_normal(5_000)
    good = (TRUE_VAR, TRUE_ES)
    bad = (TRUE_VAR, TRUE_ES * 0.5)  # ES magnitude halved -> underestimates tail loss
    res = comparative_es_backtest(realized, good, bad, alpha=ALPHA, n_boot=500, rng=rng)
    assert res["better"] == "model1"
    assert res["mean_score_diff"] < 0.0
    assert res["pvalue"] < 0.05  # clear misspecification, ample data -> rejects equal accuracy


def test_comparative_backtest_tie_for_identical_forecasts(rng: np.random.Generator) -> None:
    realized = rng.standard_normal(1_000)
    f = (TRUE_VAR, TRUE_ES)
    res = comparative_es_backtest(realized, f, f, alpha=ALPHA, n_boot=200, rng=rng)
    assert res["mean_score_diff"] == 0.0
    assert res["better"] == "tie"


# ---------------------------------------------------------------------------
# HLN small-sample DM correction + size/power calibration (audit V13)
# ---------------------------------------------------------------------------


def test_hln_factor_matches_closed_form() -> None:
    """hln_factor(T, h) == sqrt[(T + 1 - 2h + h(h-1)/T) / T] on known inputs (HLN 1997)."""
    # h = 1: factor reduces to sqrt((T-1)/T).
    assert hln_factor(10, 1) == pytest.approx(np.sqrt(9.0 / 10.0))
    assert hln_factor(100, 1) == pytest.approx(np.sqrt(99.0 / 100.0))
    # h = 2: sqrt[(T + 1 - 4 + 2/T) / T].
    t = 50
    expected = np.sqrt((t + 1 - 4 + 2.0 / t) / t)
    assert hln_factor(t, 2) == pytest.approx(expected)
    # Always shrinks toward zero in the relevant T >> h regime.
    assert 0.0 < hln_factor(20, 1) < 1.0


def test_hln_factor_rejects_bad_args() -> None:
    with pytest.raises(ValueError, match="t must be"):
        hln_factor(0, 1)
    with pytest.raises(ValueError, match="h must be"):
        hln_factor(10, 0)


def test_dm_hln_corrected_pvalue_is_more_conservative_at_small_t(rng: np.random.Generator) -> None:
    """At small T the HLN-corrected p-value >= the uncorrected one (factor < 1 AND heavier t tail)."""
    for t in (15, 30, 60):
        # A modest-signal differential so neither p-value is pinned at 0 or 1.
        d = 0.3 + rng.standard_normal(t)
        res = dm_hln_test(d, h=1)
        assert res["pvalue_hln"] >= res["pvalue_normal"] - 1e-12
        assert abs(res["dm_stat_hln"]) <= abs(res["dm_stat"]) + 1e-12  # rescaled toward zero


def test_dm_hln_h1_uses_plain_sample_variance(rng: np.random.Generator) -> None:
    """For h=1 the Newey-West variance has no autocovariance terms: DM == mean/ (s/sqrt(T))."""
    d = 0.2 + rng.standard_normal(200)
    res = dm_hln_test(d, h=1)
    # gamma0 = population variance (1/n); SE of mean = sqrt(gamma0 / n).
    se = np.sqrt((d - d.mean()).var(ddof=0) / d.size)
    assert res["dm_stat"] == pytest.approx(d.mean() / se, rel=1e-9)


def test_dm_hln_degenerate_inputs_are_safe() -> None:
    res = dm_hln_test(np.array([0.5]), h=1)  # n < 2
    assert res["pvalue_normal"] == 1.0 and res["pvalue_hln"] == 1.0
    res = dm_hln_test(np.full(50, 0.7), h=1)  # zero-variance differential
    assert res["pvalue_normal"] == 1.0 and res["pvalue_hln"] == 1.0


def test_comparative_backtest_exposes_dm_hln_keys(rng: np.random.Generator) -> None:
    realized = rng.standard_normal(400)
    good = (TRUE_VAR, TRUE_ES)
    bad = (TRUE_VAR, TRUE_ES * 0.5)
    res = comparative_es_backtest(realized, good, bad, alpha=ALPHA, n_boot=200, rng=rng)
    for key in ("dm_stat", "dm_stat_hln", "pvalue_dm_normal", "pvalue_dm_hln"):
        assert key in res
    # Same conservativeness ordering flows through the wrapper.
    assert res["pvalue_dm_hln"] >= res["pvalue_dm_normal"] - 1e-12


def test_dm_calibration_size_near_nominal_under_null() -> None:
    """MC size of the DM-HLN test under H0 sits near the nominal 5%; HLN is not anti-conservative.

    Cheap rep count for the suite; deterministic seed (no wall-clock RNG).
    """
    cal = dm_size_power_calibration(t=120, alpha=0.05, n_reps=600, seed=2024)
    # Empirical size in a sane band around nominal (MC noise at 600 reps ~ +/-0.018 s.e.).
    assert 0.02 <= cal["size_hln"] <= 0.11
    # The HLN correction must not be MORE liberal than the uncorrected normal test.
    assert cal["size_hln"] <= cal["size_normal"] + 1e-9
    # The test has real power against a clear alternative.
    assert cal["power_hln"] > 0.5


def test_dm_calibration_is_deterministic() -> None:
    """Same seed => identical size/power (replay guarantee; no global/time RNG)."""
    a = dm_size_power_calibration(t=80, n_reps=300, seed=7)
    b = dm_size_power_calibration(t=80, n_reps=300, seed=7)
    assert a == b


# ---------------------------------------------------------------------------
# Edge / degrade / validation paths (coverage completion)
# ---------------------------------------------------------------------------


def test_var_es_estimates_rejects_bad_alpha() -> None:
    # alpha out of (0,1) -> ValueError (line 94)
    with pytest.raises(ValueError, match="alpha"):
        var_es_estimates(np.array([-1.0, 0.0, 1.0]), 0.0)
    with pytest.raises(ValueError, match="alpha"):
        var_es_estimates(np.array([-1.0, 0.0, 1.0]), 1.0)


def test_var_es_estimates_all_nonfinite_returns_nan() -> None:
    # every value non-finite -> stripped to empty -> (nan, nan) (line 97-98)
    v, e = var_es_estimates(np.array([np.nan, np.inf, -np.inf]), ALPHA)
    assert np.isnan(v) and np.isnan(e)


def test_var_es_estimates_strips_nan_before_quantile(rng: np.random.Generator) -> None:
    # A single NaN must not poison the VaR/ES; result matches the finite-only series.
    clean = rng.standard_normal(2_000)
    poisoned = np.concatenate([clean, [np.nan, np.nan]])
    vc, ec = var_es_estimates(clean, ALPHA)
    vp, ep = var_es_estimates(poisoned, ALPHA)
    assert vp == pytest.approx(vc) and ep == pytest.approx(ec)


def test_hac_variance_multistep_includes_autocovariance(rng: np.random.Generator) -> None:
    """For h>1 the Newey-West variance adds Bartlett-weighted autocovariance terms (lines 150-154).

    On a strongly positively autocorrelated differential the h>1 long-run variance must EXCEED the
    plain h=1 sample variance of the mean (positive autocovariances inflate the LRV).
    """
    # AR(1) with rho ~ 0.7 so gamma_1 > 0.
    t = 400
    eps = rng.standard_normal(t)
    d = np.empty(t)
    d[0] = eps[0]
    for k in range(1, t):
        d[k] = 0.7 * d[k - 1] + eps[k]
    r1 = dm_hln_test(d, h=1)
    r5 = dm_hln_test(d, h=5)
    # Same mean(d); larger LRV under h=5 -> |DM| statistic SHRINKS relative to h=1.
    assert abs(r5["dm_stat"]) < abs(r1["dm_stat"])


def test_calibration_rejects_bad_args() -> None:
    with pytest.raises(ValueError, match="t must be"):
        dm_size_power_calibration(t=1)               # line 279
    with pytest.raises(ValueError, match="alpha"):
        dm_size_power_calibration(t=50, alpha=1.5)   # line 281
    with pytest.raises(ValueError, match="ar1"):
        dm_size_power_calibration(t=50, ar1=1.0)     # line 283


def test_calibration_ar1_branch_runs_and_is_deterministic() -> None:
    # ar1 != 0 exercises the AR(1) simulation branch (lines 296-299); still deterministic + sane.
    a = dm_size_power_calibration(t=60, ar1=0.5, n_reps=200, seed=11)
    b = dm_size_power_calibration(t=60, ar1=0.5, n_reps=200, seed=11)
    assert a == b
    assert 0.0 <= a["size_hln"] <= 1.0 and a["power_hln"] >= a["size_hln"] - 0.2


def test_comparative_backtest_default_rng_when_none() -> None:
    # rng=None takes the default_rng() branch (line 370); still returns a well-formed result.
    realized = np.linspace(-3.0, 3.0, 300)
    good = (TRUE_VAR, TRUE_ES)
    bad = (TRUE_VAR, TRUE_ES * 0.5)
    res = comparative_es_backtest(realized, good, bad, alpha=ALPHA, n_boot=100)
    assert set(res) >= {"mean_score_diff", "stat", "pvalue", "better"}
    assert 0.0 <= res["pvalue"] <= 1.0


def test_loss_diff_tail_index_flags_heavy_tails_B552() -> None:
    """B.5.2 (2026-07-06): the FZ0 loss-differential Hill tail-index is computed and flags a
    heavy-tailed differential (Pareto alpha=1.5 -> estimate near 1.5, flag True); the block is
    always present with the k convention."""
    import numpy as np

    from src.inference.es_backtest import comparative_es_backtest

    rng = np.random.default_rng(3)
    # realized returns with a genuine lower tail; two distinct (VaR, ES) forecasts (es < 0)
    r = rng.standard_t(df=3, size=1500) * 0.01
    out = comparative_es_backtest(r, (-0.02, -0.03), (-0.025, -0.04), alpha=0.05,
                                  n_boot=200, rng=np.random.default_rng(0))
    blk = out["loss_diff_tail"]
    assert set(blk) == {"hill_alpha", "k", "heavy_tailed_for_dm_companion"}
    assert blk["k"] == max(10, int(0.05 * 1500))
    assert np.isfinite(blk["hill_alpha"]) and blk["hill_alpha"] > 0
    # The machine-readable flag (B.5.2's actual deliverable) must FIRE on this heavy-tailed differential.
    # Previously the docstring promised "flag True" but nothing asserted it -- so an inverted/broken flag
    # would have passed. hill_alpha ~1.4 << 4 => the DM-HLN companion is correctly marked down-weighted.
    assert blk["hill_alpha"] < 4.0
    assert blk["heavy_tailed_for_dm_companion"] is True


def test_loss_diff_tail_index_not_flagged_for_light_tailed_differential() -> None:
    """B.5.2 companion (the previously untested FALSE branch): a THIN-tailed, NON-degenerate loss
    differential must read ``heavy_tailed_for_dm_companion is False`` -- not True, and not the ``None``
    default. Bounded (uniform) returns + a broad alpha so most observations are hits => the differential
    varies everywhere (non-degenerate) and inherits the bounded, light tail (hill_alpha >> 4), so the
    DM-HLN companion is NOT down-weighted. This pins the flag's discrimination, not just its presence."""
    r = (np.random.default_rng(5).random(4000) - 0.5) * 0.04  # bounded support => very thin tails
    out = comparative_es_backtest(r, (0.0, -0.01), (0.003, -0.02), alpha=0.5,
                                  n_boot=80, rng=np.random.default_rng(0))
    blk = out["loss_diff_tail"]
    assert np.isfinite(blk["hill_alpha"]) and blk["hill_alpha"] > 4.0
    assert blk["heavy_tailed_for_dm_companion"] is False
