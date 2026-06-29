"""Tests for the backtesting performance/risk suite (block B11).

Hand-computed values where possible, invariances, reuse-consistency with the audited inference primitives,
and degenerate-input safety. Every metric is checked, not just smoke-imported.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.backtest.metrics import (
    compute_metrics,
    drawdown_series,
    regime_conditional_metrics,
    tearsheet_markdown,
)
from src.inference.bootstrap import cvar as _cvar
from src.inference.bootstrap import sharpe_ratio as _sharpe


# --------------------------------------------------------------------------- #
# drawdown analytics                                                           #
# --------------------------------------------------------------------------- #
def test_drawdown_on_hand_series() -> None:
    # +10%, -50% (peak->trough), +0% : max drawdown is 50% from the post-gain peak.
    r = np.array([0.10, -0.50, 0.0])
    dd = drawdown_series(r)
    assert dd["max_drawdown"] == pytest.approx(0.50, abs=1e-9)
    assert dd["max_drawdown_duration"] == 2  # two periods below the peak
    assert 0.0 < dd["time_under_water"] <= 1.0


def test_no_drawdown_for_monotone_gains() -> None:
    dd = drawdown_series(np.full(20, 0.01))
    assert dd["max_drawdown"] == 0.0
    assert dd["max_drawdown_duration"] == 0
    assert dd["time_under_water"] == 0.0


# --------------------------------------------------------------------------- #
# reuse-consistency with the audited inference primitives (DRY)                #
# --------------------------------------------------------------------------- #
def test_sharpe_and_cvar_match_inference_primitives() -> None:
    r = np.random.default_rng(0).standard_normal(750) * 0.012 + 0.0004
    m = compute_metrics(r)
    assert m["sharpe"] == pytest.approx(_sharpe(r), abs=1e-9)          # rf=0 -> identical
    assert m["cvar_05"] == pytest.approx(_cvar(r, 0.05), abs=1e-12)
    assert m["cvar_01"] == pytest.approx(_cvar(r, 0.01), abs=1e-12)


def test_risk_free_reduces_sharpe() -> None:
    # A higher risk-free rate lowers the (excess-return) Sharpe of a positive-mean noisy series.
    r = np.random.default_rng(1).standard_normal(252) * 0.01 + 0.001
    assert compute_metrics(r, risk_free=0.0)["sharpe"] > compute_metrics(r, risk_free=0.0008)["sharpe"]


def test_starr_ratio_is_excess_mean_over_cvar() -> None:
    """STARR (tail-adjusted Sharpe) = per-period excess-return mean / |CVaR| at each level; report-only,
    deterministic, positive for a positive-mean series, and zero-guarded on a degenerate (no-loss) tail."""
    r = np.random.default_rng(7).standard_normal(750) * 0.012 + 0.0006
    m = compute_metrics(r, risk_free=0.0)
    for tag, a in (("05", 0.05), ("01", 0.01)):
        cvar = m[f"cvar_{tag}"]
        assert cvar < 0.0  # losses are negative
        assert m[f"starr_{tag}"] == pytest.approx(float(np.mean(r)) / abs(cvar), rel=1e-9)
        # sign-consistency: STARR sign tracks the (excess) mean-return sign (rf=0 -> excess==r).
        assert np.sign(m[f"starr_{tag}"]) == np.sign(float(np.mean(r)))
    # A drift-DOMINATED series (drift >> std/sqrt(n)) has a reliably positive tail-adjusted return.
    strong = np.random.default_rng(7).standard_normal(750) * 0.005 + 0.003
    assert compute_metrics(strong, risk_free=0.0)["starr_05"] > 0.0
    # Degenerate: an all-zero series has CVaR == 0 -> the |CVaR| > _EPS guard returns 0.0 (never +inf/NaN).
    assert compute_metrics(np.zeros(60))["starr_05"] == 0.0
    # A constant positive series has CVaR == the constant, so STARR = mean/|CVaR| = 1.0 (well-defined, finite).
    assert compute_metrics(np.full(60, 0.001))["starr_05"] == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# named-metric correctness + ranges                                            #
# --------------------------------------------------------------------------- #
def test_full_suite_keys_and_ranges() -> None:
    rng = np.random.default_rng(2)
    r = rng.standard_normal(1000) * 0.01 + 0.0006
    bench = rng.standard_normal(1000) * 0.011 + 0.0003
    tov = np.abs(rng.standard_normal(1000)) * 0.05
    m = compute_metrics(r, benchmark=bench, turnover=tov, cost_bps=10, n_trials=40, var_sr=0.4)
    # every advertised family is present
    for k in (
        "cagr", "ann_volatility", "sharpe", "sortino", "calmar", "omega", "martin_ratio",
        "max_drawdown", "ulcer_index", "pain_index", "cvar_05", "var_hist_05",
        "tail_ratio", "skew", "excess_kurtosis", "hit_rate", "gain_loss_ratio", "profit_factor",
        "turnover_ann", "cost_drag_ann", "information_ratio", "tracking_error_ann", "beta",
        "alpha_ann", "psr",
    ):
        assert k in m, k
        assert np.isfinite(m[k]), k
    assert 0.0 <= m["psr"] <= 1.0
    assert 0.0 <= m["hit_rate"] <= 1.0
    assert m["max_drawdown"] >= 0.0
    assert m["turnover_ann"] == pytest.approx(tov.mean() * 252, rel=1e-9)


def test_omega_is_one_at_symmetric_threshold() -> None:
    # A symmetric zero-mean series has equal gains/losses about 0 -> Omega ~ 1.
    r = np.array([0.01, -0.01, 0.02, -0.02, 0.015, -0.015])
    assert compute_metrics(r)["omega"] == pytest.approx(1.0, abs=1e-9)


def test_omega_is_rf_invariant_but_sharpe_is_not() -> None:
    # Omega is about a FIXED 0 MAR, so a per-period risk-free series must NOT shift it; Sharpe/Sortino
    # legitimately use rf as the excess baseline, so they MUST move. (Guards the array-rf generalisation.)
    r = np.random.default_rng(7).standard_normal(500) * 0.01 + 0.0003
    rf = np.full(r.size, 8e-5)  # ~2%/yr daily T-bill
    assert compute_metrics(r, risk_free=0.0)["omega"] == pytest.approx(
        compute_metrics(r, risk_free=rf)["omega"], rel=1e-12
    )
    assert compute_metrics(r, risk_free=0.0)["sharpe"] != pytest.approx(
        compute_metrics(r, risk_free=rf)["sharpe"], rel=1e-9
    )


def test_tail_ratio_direction() -> None:
    # Right-skewed (big gains, small losses) -> tail ratio > 1.
    r = np.concatenate([np.full(95, -0.001), np.full(5, 0.05)])
    assert compute_metrics(r)["tail_ratio"] > 1.0


def test_beta_recovers_known_loading() -> None:
    rng = np.random.default_rng(3)
    bench = rng.standard_normal(2000) * 0.01
    r = 1.5 * bench + rng.standard_normal(2000) * 1e-4  # beta ~ 1.5
    assert compute_metrics(r, benchmark=bench)["beta"] == pytest.approx(1.5, abs=0.05)


# --------------------------------------------------------------------------- #
# degenerate inputs are safe (never crash)                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("series", [[], [0.01], np.zeros(50), np.full(30, -1.0)])
def test_degenerate_inputs_safe(series) -> None:
    m = compute_metrics(series)
    assert all(np.isfinite(v) for v in m.values() if isinstance(v, float))


def test_non_finite_inputs_stripped() -> None:
    r = np.array([0.01, np.nan, -0.02, np.inf, 0.0])
    m = compute_metrics(r)
    assert m["n_periods"] == 3.0  # nan + inf dropped


# --------------------------------------------------------------------------- #
# regime-conditional + tearsheet                                               #
# --------------------------------------------------------------------------- #
def test_regime_conditional_groups_and_all() -> None:
    rng = np.random.default_rng(4)
    r = rng.standard_normal(600) * 0.01
    reg = np.where(np.arange(600) % 3 == 0, "stress", "calm")
    rc = regime_conditional_metrics(r, reg)
    assert set(rc) == {"all", "calm", "stress"}
    assert rc["all"]["n_periods"] == 600.0
    assert rc["calm"]["n_periods"] + rc["stress"]["n_periods"] == 600.0


def test_tearsheet_renders_comparison() -> None:
    rng = np.random.default_rng(5)
    a = compute_metrics(rng.standard_normal(500) * 0.01 + 0.0005)
    b = compute_metrics(rng.standard_normal(500) * 0.01)
    md = tearsheet_markdown({"distributional": a, "scalar": b}, title="H2")
    assert "## H2" in md
    assert "Sharpe" in md and "distributional" in md and "scalar" in md
    assert "CVaR 5%" in md and "Max drawdown" in md
