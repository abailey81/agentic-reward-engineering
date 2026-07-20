"""Behaviour tests for the causal-chain mediation analysis (src/inference/mediation.py).

Deterministic (seeded bootstrap, numpy only). Checks: a constructed full-mediation chain is recovered with
a CI excluding zero; the responsiveness-null chain (a≈0) yields indirect≈0 and mediated=False (the chain is
severed at link 1 — the headline framing); and the degrade/validation paths.
"""

from __future__ import annotations

import numpy as np

import pytest

from src.inference.mediation import PROP_MEDIATED_C_SE_K, _slopes, mediation_analysis

SEED = 20260701


def test_recovers_a_full_mediation_chain() -> None:
    rng = np.random.default_rng(SEED)
    n = 200
    x = rng.normal(size=n)
    m = 0.8 * x + 0.3 * rng.normal(size=n)          # X strongly drives M (path a)
    y = 0.9 * m + 0.2 * rng.normal(size=n)          # M drives Y (path b); no direct X->Y
    res = mediation_analysis(x, m, y, n_boot=1000, rng=np.random.default_rng(1))
    assert res["status"] == "ok"
    assert res["a"] > 0.5 and res["b"] > 0.5        # both transmission links present
    assert res["indirect"] > 0.3                     # a*b clearly positive
    assert res["mediated"] is True                   # CI excludes 0
    assert res["ci_low"] > 0.0
    assert abs(res["c_direct"]) < abs(res["indirect"])  # effect runs through M, not around it


def test_null_responsiveness_severs_the_chain_at_link_one() -> None:
    # The predicted NULL: the fed signal does NOT move the code (a ~ 0). Then a*b ~ 0 for any b.
    rng = np.random.default_rng(SEED)
    n = 200
    x = rng.normal(size=n)
    m = rng.normal(size=n)                            # M independent of X -> path a ~ 0
    y = 0.9 * m + 0.2 * rng.normal(size=n)          # M still drives Y (path b real)
    res = mediation_analysis(x, m, y, n_boot=1000, rng=np.random.default_rng(2))
    assert res["status"] == "ok"
    assert abs(res["a"]) < 0.2                        # responsiveness null
    assert abs(res["indirect"]) < 0.15               # mediated effect collapses
    assert res["mediated"] is False                  # CI includes 0: chain broken at link 1
    assert res["ci_low"] < 0.0 < res["ci_high"]


def test_is_deterministic_under_fixed_seed() -> None:
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=60)
    m = 0.5 * x + rng.normal(size=60)
    y = 0.5 * m + rng.normal(size=60)
    a = mediation_analysis(x, m, y, n_boot=500, rng=np.random.default_rng(7))
    b = mediation_analysis(x, m, y, n_boot=500, rng=np.random.default_rng(7))
    assert a["indirect"] == b["indirect"] and a["ci_low"] == b["ci_low"] and a["ci_high"] == b["ci_high"]


def test_indirect_equals_c_minus_cprime_in_linear_ols() -> None:
    # Algebraic identity for single-mediator OLS: a*b == c_total - c_direct.
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=120)
    m = 0.6 * x + rng.normal(size=120)
    y = 0.4 * x + 0.7 * m + rng.normal(size=120)
    res = mediation_analysis(x, m, y, n_boot=200, rng=np.random.default_rng(3))
    assert res["indirect"] == np.float64(res["c_total"] - res["c_direct"]).item() or abs(
        res["indirect"] - (res["c_total"] - res["c_direct"])
    ) < 1e-9


def test_degrade_and_validation_paths() -> None:
    assert mediation_analysis([1.0, 2.0], [1.0, 2.0], [1.0, 2.0])["status"] == "no_data"  # n < 3
    assert mediation_analysis([1, 2, 3], [1, 2], [1, 2, 3])["status"] == "no_data"        # length mismatch
    # zero-variance treatment -> degenerate
    assert mediation_analysis([1, 1, 1, 1], [1, 2, 3, 4], [1, 2, 3, 4])["status"] == "no_data"
    # zero-variance mediator -> degenerate (the second half of the guard on line 102)
    assert mediation_analysis([1, 2, 3, 4], [2, 2, 2, 2], [1, 2, 3, 4])["status"] == "no_data"


def test_slopes_raises_on_rank_deficient_design() -> None:
    # A constant predictor column makes the design rank-deficient -> LinAlgError (line 52).
    design = np.column_stack([np.ones(4), np.array([3.0, 3.0, 3.0, 3.0])])
    with pytest.raises(np.linalg.LinAlgError, match="rank-deficient"):
        _slopes(design, np.array([1.0, 2.0, 3.0, 4.0]))


def test_prop_mediated_undefined_when_total_effect_is_near_zero() -> None:
    """P1 STABILITY GUARD (predicted-null regime): a near-zero total effect c must NOT yield a large or
    sign-flipped prop_mediated. Construct X ⟂ Y (c_total ≈ 0) while M carries real X→M and M→Y links, so the
    unstable small/small ratio would otherwise explode. The guard must mark prop_mediated NaN + flag it."""
    rng = np.random.default_rng(SEED)
    n = 200
    x = rng.normal(size=n)
    m = 0.8 * x + 0.3 * rng.normal(size=n)              # path a real
    # Y depends on M but with a compensating -0.8*x direct term so the TOTAL X->Y effect c ≈ 0.
    y = 0.9 * m - 0.72 * x + 0.2 * rng.normal(size=n)   # c_total = a*b + c_direct ≈ 0.72 - 0.72 ≈ 0
    res = mediation_analysis(x, m, y, n_boot=1500, rng=np.random.default_rng(11))
    assert res["status"] == "ok"
    assert abs(res["c_total"]) < 0.15                    # total effect really is near zero
    # the guard fires: ratio is undefined (NaN) rather than a misleading number
    assert res["prop_mediated_undefined"] is True
    assert np.isnan(res["prop_mediated"])
    # the guard's evidence is surfaced: the c_total bootstrap CI includes 0 (or |c| within k*SE)
    ci_includes_zero = res["c_total_ci_low"] <= 0.0 <= res["c_total_ci_high"]
    within_noise = abs(res["c_total"]) < PROP_MEDIATED_C_SE_K * res["c_total_boot_se"]
    assert ci_includes_zero or within_noise


def test_prop_mediated_defined_when_total_effect_is_clearly_nonzero() -> None:
    """The guard must NOT over-fire: with a strong, CI-separated total effect, prop_mediated is a finite
    number in a sensible range (full mediation -> close to 1)."""
    rng = np.random.default_rng(SEED)
    n = 200
    x = rng.normal(size=n)
    m = 0.8 * x + 0.3 * rng.normal(size=n)
    y = 0.9 * m + 0.2 * rng.normal(size=n)               # strong total effect through M
    res = mediation_analysis(x, m, y, n_boot=1500, rng=np.random.default_rng(12))
    assert res["status"] == "ok"
    assert res["prop_mediated_undefined"] is False
    assert np.isfinite(res["prop_mediated"])
    assert 0.5 < res["prop_mediated"] < 1.5              # near-full mediation, no explosion
    # c_total CI must exclude 0 for the ratio to be reported
    assert res["c_total_ci_low"] > 0.0 or res["c_total_ci_high"] < 0.0


def test_default_rng_used_when_none() -> None:
    # rng=None -> default_rng(0) branch (line 105); deterministic across two default calls.
    x = np.linspace(-1.0, 1.0, 50)
    m = 0.5 * x + 0.1 * np.sin(np.arange(50))
    y = 0.5 * m + 0.1 * np.cos(np.arange(50))
    a = mediation_analysis(x, m, y, n_boot=300)
    b = mediation_analysis(x, m, y, n_boot=300)
    assert a["status"] == "ok"
    assert a["indirect"] == b["indirect"] and a["ci_low"] == b["ci_low"]


def test_unstandardized_path_reports_raw_units() -> None:
    # standardize=False skips the z-scoring block (branch 107->110) and reports raw-unit effects.
    rng = np.random.default_rng(SEED)
    n = 200
    x = rng.normal(size=n)
    m = 0.8 * x + 0.3 * rng.normal(size=n)
    y = 0.9 * m + 0.2 * rng.normal(size=n)
    res = mediation_analysis(x, m, y, n_boot=500, standardize=False, rng=np.random.default_rng(9))
    assert res["status"] == "ok"
    assert res["standardized"] is False
    assert res["mediated"] is True
    # a*b identity still holds in raw OLS units
    assert abs(res["indirect"] - (res["c_total"] - res["c_direct"])) < 1e-9


def test_bootstrap_linalgerror_is_swallowed_to_nan() -> None:
    # A binary treatment with a rare level makes some case-resamples draw only one distinct x
    # value -> a rank-deficient bootstrap design -> LinAlgError caught, that replicate set nan
    # (lines 126-127). The analysis still returns ok with a finite CI from the valid replicates.
    n = 12
    x = np.zeros(n)
    x[0] = 1.0  # single "1"; many bootstraps omit it -> constant x -> rank-deficient
    m = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    y = 0.5 * m + np.array([0.1, -0.1, 0.2, -0.2, 0.1, -0.1, 0.2, -0.2, 0.1, -0.1, 0.2, -0.2])
    res = mediation_analysis(x, m, y, n_boot=400, rng=np.random.default_rng(123))
    assert res["status"] == "ok"
    # some replicates were dropped as invalid -> fewer valid than requested
    assert res["n_boot_valid"] < 400
    assert np.isfinite(res["ci_low"]) and np.isfinite(res["ci_high"])


def test_mediated_is_gated_on_the_valid_boot_fraction() -> None:
    """``mediated`` must require a RELIABLE bootstrap, not merely a non-empty one (2026-07-26 review).

    ``_effects`` raises ``LinAlgError`` on a rank-deficient resample and that replicate is dropped, so
    ``valid.size`` alone only excluded the ZERO case — a ``mediated=True`` could rest on a handful of
    survivors. This mirrors the P7b gate already applied in the sibling instruments
    (``responsiveness.responsiveness``, which DEFINES ``MIN_BOOT_VALID_FRACTION``, and
    ``information_gap``), which share the identical degeneracy mode.
    """
    from src.inference.responsiveness import MIN_BOOT_VALID_FRACTION

    rng = np.random.default_rng(7)
    n, n_boot = 200, 300
    x = rng.standard_normal(n)
    m = 0.8 * x + 0.3 * rng.standard_normal(n)
    y = 0.7 * m + 0.2 * rng.standard_normal(n)
    res = mediation_analysis(x, m, y, n_boot=n_boot, rng=rng)

    # The flag is reported and is exactly the documented threshold rule.
    assert res["ci_reliable"] == (res["n_boot_valid"] >= MIN_BOOT_VALID_FRACTION * n_boot)
    # A healthy design keeps every replicate, so the gate passes and mediation is still detected.
    assert res["ci_reliable"] is True
    assert res["mediated"] is True
    # THE INVARIANT: a positive verdict can never outrun the reliability gate.
    assert not (res["mediated"] and not res["ci_reliable"])
