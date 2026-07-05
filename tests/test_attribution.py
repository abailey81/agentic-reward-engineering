"""Behaviour tests for the factor-attribution Door-C secondary (``src.inference.attribution``).

Synthetic-return tests with KNOWN ground truth (the testing rule — invariances/bounds,
not smoke):

  * a known alpha + known factor loading is RECOVERED by ``factor_alpha`` (point estimate);
  * a zero-alpha series (returns = pure factor exposure + mean-zero noise) gives ``alpha ~ 0`` with
    ``|t| `` small (no spurious alpha);
  * the Newey-West (1994) lag equals ``floor(4*(T/100)^(2/9))`` and the statsmodels HAC covariance equals
    a hand-rolled Newey-West covariance (the estimator is correct);
  * HAC SEs are SANE — under strong positive residual autocorrelation the HAC alpha-SE EXCEEDS the
    iid OLS SE (HAC widens, never silently narrows);
  * ``difference_in_alpha`` finds a positive, significant gap when arm A has the higher per-seed
    alpha and a null (non-significant) gap when the two arms share the same alpha;
  * the ``campaign_attribution`` driver DEGRADES GRACEFULLY: no factor data -> skipped/available
    False; a rung whose columns are absent -> that cell skipped (FF5/BAB/QMJ before a pull); a
    missing comparator arm -> that contrast skipped; and the result keys are DISJOINT from the frozen
    H2 family (so ``assert_realized_family_matches_frozen`` can never mis-match it).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.inference.attribution import (
    FACTOR_LADDER,
    _newey_west_cov,
    campaign_attribution,
    difference_in_alpha,
    factor_alpha,
    load_factor_panel,
    newey_west_hac_lag,
)


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _factor_block(t: int, *, seed: int = 0, k: int = 3) -> dict[str, np.ndarray]:
    """A small FF3-shaped factor block of daily decimals (mean-zero-ish, realistic scale)."""
    rng = np.random.default_rng(seed)
    names = ["Mkt-RF", "SMB", "HML"][:k]
    return {name: rng.standard_normal(t) * 0.01 for name in names}


def _make_series(
    factors: dict[str, np.ndarray],
    betas: dict[str, float],
    alpha: float,
    *,
    noise: float,
    seed: int,
) -> np.ndarray:
    """``r_t = alpha + sum_k beta_k f_{k,t} + eps_t`` with iid normal noise (rf = 0)."""
    rng = np.random.default_rng(seed)
    t = len(next(iter(factors.values())))
    r = np.full(t, float(alpha))
    for name, b in betas.items():
        r = r + b * np.asarray(factors[name], dtype=float)
    return r + rng.standard_normal(t) * noise


# --------------------------------------------------------------------------- #
# Newey-West (1994) auto-lag rule + Newey-West (1987) estimator correctness     #
# --------------------------------------------------------------------------- #
def test_newey_west_lag_matches_closed_form() -> None:
    import math

    for n in (50, 100, 252, 756, 2000, 2520):
        assert newey_west_hac_lag(n) == int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    # T=100 -> 4*(1)^(2/9) = 4 exactly (the canonical anchor of the rule).
    assert newey_west_hac_lag(100) == 4
    assert newey_west_hac_lag(0) == 0 and newey_west_hac_lag(-5) == 0


def test_statsmodels_hac_equals_hand_rolled_newey_west() -> None:
    # The hand-rolled Bartlett-kernel Newey-West covariance must equal statsmodels' HAC meat
    # (use_correction=False) bit-for-bit up to numerical tolerance — certifies the estimator.
    sm = pytest.importorskip("statsmodels.api")
    t = 600
    f = _factor_block(t, seed=3)
    y = _make_series(f, {"Mkt-RF": 1.1, "SMB": -0.3, "HML": 0.4}, alpha=0.0003, noise=0.004, seed=9)
    x = np.column_stack([np.ones(t)] + [f[name] for name in ("Mkt-RF", "SMB", "HML")])
    lag = newey_west_hac_lag(t)

    res = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": False})
    beta = np.asarray(res.params)
    resid = y - x @ beta
    hand = _newey_west_cov(x, resid, lag)
    assert np.allclose(np.asarray(res.cov_params()), hand, rtol=1e-8, atol=1e-14)


def test_factor_alpha_fallback_matches_statsmodels_when_statsmodels_absent(monkeypatch) -> None:
    """factor_alpha's defensive fallback (the ``except`` at attribution.py:289) BECOMES the primary path
    on a statsmodels-free install, so its answer must equal the statsmodels path's. The test above
    certifies ``_newey_west_cov`` in ISOLATION; this certifies it is correctly WIRED as the fallback:
    force ``import statsmodels.api`` to raise (None in sys.modules) and assert the hand-rolled path
    returns ``status="ok"`` with alpha / t / SE / R^2 / betas identical to the statsmodels path.
    """
    import sys

    pytest.importorskip("statsmodels.api")  # need the reference path to compare the fallback against
    t = 600
    f = _factor_block(t, seed=3)
    y = _make_series(f, {"Mkt-RF": 1.1, "SMB": -0.3, "HML": 0.4}, alpha=0.0003, noise=0.004, seed=9)
    names = ("Mkt-RF", "SMB", "HML")
    ref = factor_alpha(y, f, factor_names=names)                 # statsmodels HAC path
    monkeypatch.setitem(sys.modules, "statsmodels.api", None)    # -> `import statsmodels.api` raises
    fb = factor_alpha(y, f, factor_names=names)                  # forced hand-rolled OLS+NW fallback
    assert ref["status"] == "ok" and fb["status"] == "ok"
    assert fb["hac_lag"] == ref["hac_lag"] and fb["n"] == ref["n"]
    assert fb["alpha"] == pytest.approx(ref["alpha"], rel=1e-6, abs=1e-12)
    assert fb["alpha_t"] == pytest.approx(ref["alpha_t"], rel=1e-5)
    assert fb["alpha_se"] == pytest.approx(ref["alpha_se"], rel=1e-5)
    assert fb["r2"] == pytest.approx(ref["r2"], rel=1e-6)
    for name in names:
        assert fb["betas"][name] == pytest.approx(ref["betas"][name], rel=1e-6)
        assert fb["betas_t"][name] == pytest.approx(ref["betas_t"][name], rel=1e-5)


# --------------------------------------------------------------------------- #
# factor_alpha — recovery + zero-alpha + HAC sanity                            #
# --------------------------------------------------------------------------- #
def test_known_alpha_and_betas_recovered() -> None:
    t = 3000
    f = _factor_block(t, seed=1)
    true_alpha = 0.0005  # ~12.6%/yr
    true_betas = {"Mkt-RF": 1.2, "SMB": -0.4, "HML": 0.6}
    y = _make_series(f, true_betas, true_alpha, noise=0.003, seed=2)

    res = factor_alpha(y, f, factor_names=("Mkt-RF", "SMB", "HML"))
    assert res["status"] == "ok"
    assert res["alpha"] == pytest.approx(true_alpha, abs=2e-4)  # within sampling error at noise=0.3%/day
    for name, b in true_betas.items():
        assert res["betas"][name] == pytest.approx(b, abs=0.03)
    assert res["alpha_ann"] == pytest.approx(res["alpha"] * 252.0)
    assert res["hac_lag"] == newey_west_hac_lag(t)
    assert 0.0 <= res["r2"] <= 1.0
    # A real alpha at this scale/length should be detectable.
    assert res["alpha_t"] > 2.0 and res["alpha_p"] < 0.05


def test_zero_alpha_gives_tstat_near_zero() -> None:
    # Returns are PURE factor exposure + mean-zero noise -> the intercept is genuinely zero.
    t = 3000
    f = _factor_block(t, seed=5)
    y = _make_series(f, {"Mkt-RF": 1.0, "SMB": 0.2, "HML": -0.1}, alpha=0.0, noise=0.004, seed=6)
    res = factor_alpha(y, f, factor_names=("Mkt-RF", "SMB", "HML"))
    assert res["status"] == "ok"
    assert abs(res["alpha"]) < 2e-4
    assert abs(res["alpha_t"]) < 2.0  # no spurious alpha
    assert res["alpha_p"] > 0.05


def test_hac_se_exceeds_iid_under_positive_autocorrelation() -> None:
    # Build residuals with strong positive AR(1) serial correlation; HAC must widen the alpha SE
    # relative to the iid (lag=0 / White) SE — the whole point of using Newey-West on daily P&L.
    rng = np.random.default_rng(11)
    t = 1500
    f = _factor_block(t, seed=8)
    # AR(1) residual with phi=0.6 (positive autocorrelation).
    eps = np.zeros(t)
    innov = rng.standard_normal(t) * 0.004
    phi = 0.6
    for i in range(1, t):
        eps[i] = phi * eps[i - 1] + innov[i]
    y = _make_series(f, {"Mkt-RF": 1.0, "SMB": 0.1, "HML": 0.1}, alpha=0.0002, noise=0.0, seed=0) + eps

    iid = factor_alpha(y, f, factor_names=("Mkt-RF", "SMB", "HML"), hac_lag=0)
    hac = factor_alpha(y, f, factor_names=("Mkt-RF", "SMB", "HML"))  # Newey-West (1994) lag
    assert iid["status"] == "ok" and hac["status"] == "ok"
    assert hac["hac_lag"] >= 1
    # Positive serial correlation inflates the long-run variance -> HAC SE strictly larger.
    assert hac["alpha_se"] > iid["alpha_se"]


def test_excess_return_lhs_subtracts_risk_free() -> None:
    # With a constant rf, the excess-return intercept must drop by exactly that rf vs the raw intercept.
    t = 2000
    f = _factor_block(t, seed=12)
    y = _make_series(f, {"Mkt-RF": 1.0, "SMB": 0.0, "HML": 0.0}, alpha=0.0006, noise=0.003, seed=13)
    rf = np.full(t, 0.0001)
    raw = factor_alpha(y, f, factor_names=("Mkt-RF", "SMB", "HML"))
    exc = factor_alpha(y, f, factor_names=("Mkt-RF", "SMB", "HML"), risk_free=rf)
    assert raw["status"] == "ok" and exc["status"] == "ok"
    assert exc["alpha"] == pytest.approx(raw["alpha"] - 0.0001, abs=1e-9)


# --------------------------------------------------------------------------- #
# factor_alpha — graceful degradation                                          #
# --------------------------------------------------------------------------- #
def test_factor_alpha_skips_short_series() -> None:
    f = {"Mkt-RF": np.array([0.01, -0.02, 0.0]), "SMB": np.array([0.0, 0.01, -0.01])}
    res = factor_alpha(np.array([0.01, 0.0, -0.01]), f, factor_names=("Mkt-RF", "SMB"))
    assert res["status"] == "skipped" and "too few" in res["reason"]
    assert res["alpha"] is None  # never fabricated


def test_factor_alpha_skips_rank_deficient_factor() -> None:
    t = 500
    # A constant (all-equal) factor column is collinear with the intercept -> rank-deficient.
    f = {"Mkt-RF": np.linspace(-0.01, 0.01, t), "CONST": np.full(t, 0.005)}
    y = np.random.default_rng(0).standard_normal(t) * 0.01
    res = factor_alpha(y, f, factor_names=("Mkt-RF", "CONST"))
    assert res["status"] == "skipped" and "rank-deficient" in res["reason"]


# --------------------------------------------------------------------------- #
# difference_in_alpha — the headline per-seed paired test                      #
# --------------------------------------------------------------------------- #
def _seeded_arm(
    factors: dict[str, np.ndarray], betas: dict[str, float], alpha: float, *, n_seeds: int, base_seed: int
) -> dict[int, np.ndarray]:
    """One realized series per seed at a common alpha + a small per-seed alpha jitter (across-seed var)."""
    rng = np.random.default_rng(base_seed)
    out: dict[int, np.ndarray] = {}
    for s in range(n_seeds):
        a_s = alpha + rng.normal(0.0, 3e-5)  # training-RNG across-seed dispersion in alpha
        out[s] = _make_series(factors, betas, a_s, noise=0.004, seed=base_seed * 100 + s)
    return out


def test_difference_in_alpha_detects_positive_gap() -> None:
    t = 1200
    f = _factor_block(t, seed=21)
    betas = {"Mkt-RF": 1.0, "SMB": 0.2, "HML": -0.1}
    a = _seeded_arm(f, betas, alpha=0.0006, n_seeds=20, base_seed=1)  # distributional: higher alpha
    b = _seeded_arm(f, betas, alpha=0.0000, n_seeds=20, base_seed=2)  # scalar: zero alpha
    res = difference_in_alpha(
        a, b, f, factor_names=("Mkt-RF", "SMB", "HML"), n_boot=800, rng=np.random.default_rng(3)
    )
    assert res["status"] == "ok"
    assert res["n_seeds"] == 20
    assert res["effect"] > 0.0  # distributional alpha higher
    assert res["alpha_a_iqm"] > res["alpha_b_iqm"]
    assert res["pvalue"] < 0.05  # a ~15%/yr alpha gap over 20 seeds is detectable


def test_difference_in_alpha_null_when_alphas_equal() -> None:
    t = 1200
    f = _factor_block(t, seed=31)
    betas = {"Mkt-RF": 1.0, "SMB": 0.1, "HML": 0.1}
    a = _seeded_arm(f, betas, alpha=0.0002, n_seeds=20, base_seed=11)
    b = _seeded_arm(f, betas, alpha=0.0002, n_seeds=20, base_seed=12)  # SAME alpha
    res = difference_in_alpha(
        a, b, f, factor_names=("Mkt-RF", "SMB", "HML"), n_boot=800, rng=np.random.default_rng(13)
    )
    assert res["status"] == "ok"
    assert res["pvalue"] > 0.05  # no real difference -> not significant
    assert abs(res["effect"]) < 1e-3


def test_difference_in_alpha_skips_missing_columns_and_few_seeds() -> None:
    t = 600
    f = _factor_block(t, seed=41)
    a = _seeded_arm(f, {"Mkt-RF": 1.0}, alpha=0.0003, n_seeds=5, base_seed=21)
    b = _seeded_arm(f, {"Mkt-RF": 1.0}, alpha=0.0, n_seeds=5, base_seed=22)
    # A column absent from the panel -> skipped, never fabricated.
    miss = difference_in_alpha(a, b, f, factor_names=("Mkt-RF", "RMW"), n_boot=200)
    assert miss["status"] == "skipped" and "RMW" in miss["reason"]
    # Only one shared seed -> cannot run the across-seed bootstrap.
    a1 = {0: a[0]}
    b1 = {0: b[0]}
    few = difference_in_alpha(a1, b1, f, factor_names=("Mkt-RF",), n_boot=200)
    assert few["status"] == "skipped" and few["n_seeds"] == 1


# --------------------------------------------------------------------------- #
# campaign_attribution — driver, graceful degradation, disjoint keys           #
# --------------------------------------------------------------------------- #
def _records(arm: str, by_seed: dict[int, np.ndarray]) -> list[dict]:
    return [
        {
            "run_id": f"{arm}-s{s}",
            "arm": arm,
            "seed": s,
            "frozen": True,
            "metrics": {"test_returns": [float(x) for x in vec]},
            "test_returns": [float(x) for x in vec],
        }
        for s, vec in by_seed.items()
    ]


def test_campaign_attribution_runs_constructable_rungs_and_skips_pull_rungs() -> None:
    t = 1000
    f = _factor_block(t, seed=51)  # Mkt-RF/SMB/HML only -> capm+ff3 runnable; carhart4+ need Mom; ff5+ need pulls
    betas = {"Mkt-RF": 1.0, "SMB": 0.2, "HML": -0.1}
    recs: list[dict] = []
    recs += _records("distributional", _seeded_arm(f, betas, 0.0006, n_seeds=15, base_seed=1))
    recs += _records("scalar", _seeded_arm(f, betas, 0.0, n_seeds=15, base_seed=2))
    recs += _records("placebo", _seeded_arm(f, betas, 0.0001, n_seeds=15, base_seed=3))
    recs += _records("scalar_cvar5", _seeded_arm(f, betas, 0.0001, n_seeds=15, base_seed=4))

    out = campaign_attribution(recs, f, n_boot=500, rng=np.random.default_rng(0))
    assert out["status"] == "ok"
    assert out["family"] == "factor_attribution_difference_in_alpha"
    cells = {(c["contrast"], c["rung"]): c for c in out["cells"]}
    # CAPM + FF3 have all columns present -> ok; carhart4 (needs Mom) + ff5/ff6/bab/qmj -> skipped.
    assert cells[("distributional>scalar", "capm")]["status"] == "ok"
    assert cells[("distributional>scalar", "ff3")]["status"] == "ok"
    assert cells[("distributional>scalar", "carhart4")]["status"] == "skipped"
    assert "Mom" in cells[("distributional>scalar", "carhart4")]["reason"]
    assert cells[("distributional>scalar", "ff6_bab_qmj")]["status"] == "skipped"
    # The runnable family is the union of (3 contrasts x 2 runnable rungs) = 6 cells.
    assert out["n_family"] == 6
    # The distributional>scalar FF3 cell recovers a positive alpha gap.
    assert cells[("distributional>scalar", "ff3")]["alpha_diff"] > 0.0
    assert cells[("distributional>scalar", "ff3")]["direction_ok"] is True


def test_campaign_attribution_no_factor_data_degrades_to_skipped() -> None:
    t = 400
    f = _factor_block(t, seed=61)
    recs = _records("distributional", _seeded_arm(f, {"Mkt-RF": 1.0}, 0.0005, n_seeds=10, base_seed=1))
    recs += _records("scalar", _seeded_arm(f, {"Mkt-RF": 1.0}, 0.0, n_seeds=10, base_seed=2))
    out = campaign_attribution(recs, None, n_boot=200)  # no factors at all
    assert out["status"] == "skipped" and out["available"] is False and out["cells"] == []
    out2 = campaign_attribution(recs, {}, n_boot=200)  # empty factor dict
    assert out2["status"] == "skipped" and out2["available"] is False


def test_campaign_attribution_skips_contrast_with_missing_arm() -> None:
    t = 800
    f = _factor_block(t, seed=71)
    # Only the distributional arm present -> every contrast is missing its comparator.
    recs = _records("distributional", _seeded_arm(f, {"Mkt-RF": 1.0}, 0.0005, n_seeds=10, base_seed=1))
    out = campaign_attribution(recs, f, n_boot=200)
    assert out["status"] == "skipped"  # nothing runnable
    assert any("scalar" in s.get("contrast", "") for s in out["skipped"])


def test_result_keys_are_disjoint_from_frozen_h2_family() -> None:
    # The Door-C secondary must use keys that the frozen-family guard cannot mistake for an H2 member
    # (which keys on arm_a/arm_b/metric/level). The attribution cells must NOT carry metric/level.
    t = 800
    f = _factor_block(t, seed=81)
    recs = _records("distributional", _seeded_arm(f, {"Mkt-RF": 1.0, "SMB": 0.1, "HML": 0.0}, 0.0005, n_seeds=8, base_seed=1))
    recs += _records("scalar", _seeded_arm(f, {"Mkt-RF": 1.0, "SMB": 0.1, "HML": 0.0}, 0.0, n_seeds=8, base_seed=2))
    out = campaign_attribution(recs, f, n_boot=300)
    assert out["family"] == "factor_attribution_difference_in_alpha"
    for c in out["cells"]:
        assert "metric" not in c and "level" not in c
        assert "rung" in c and "factor" in c  # the disjoint discriminators
        # A frozen-family member tuple would be (arm_a, arm_b, metric, level); ours never completes it.


def test_ladder_is_the_canonical_capm_to_qmj_progression() -> None:
    names = [name for name, _ in FACTOR_LADDER]
    assert names == ["capm", "ff3", "carhart4", "ff5", "ff6", "ff6_bab", "ff6_bab_qmj"]
    cols = dict(FACTOR_LADDER)
    assert cols["capm"] == ("Mkt-RF",)
    assert cols["ff3"] == ("Mkt-RF", "SMB", "HML")
    assert cols["carhart4"][-1] == "Mom"
    assert "BAB" in cols["ff6_bab"] and "QMJ" in cols["ff6_bab_qmj"]


# --------------------------------------------------------------------------- #
# load_factor_panel — on-disk wiring + graceful degradation                    #
# --------------------------------------------------------------------------- #
def test_load_factor_panel_absent_files_degrade(tmp_path) -> None:
    dates = np.array(["2020-01-02", "2020-01-03", "2020-01-06"], dtype="datetime64[D]")
    out = load_factor_panel(dates, raw_dir=tmp_path)  # empty dir
    assert out["available"] is False and out["factors"] == {}
    # The pull-needed columns are still reported so a caller knows what is missing.
    assert set(out["needs_pull"]) == {"RMW", "CMA", "BAB", "QMJ"}


def test_load_factor_panel_injected_provider_supplies_columns() -> None:
    dates = np.array(["2020-01-02", "2020-01-03", "2020-01-06"], dtype="datetime64[D]")

    def provider(d):
        n = len(d)
        return {"BAB": np.full(n, 0.001), "QMJ": np.full(n, -0.0005)}

    out = load_factor_panel(dates, raw_dir="/nonexistent", factor_provider=provider)
    assert "BAB" in out["factors"] and "QMJ" in out["factors"]
    assert out["available"] is True
    assert "BAB" not in out["needs_pull"] and "QMJ" not in out["needs_pull"]
