"""Property tests for the v2 cross-model synthesis (R80/R82 frozen spec)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inference.cross_model import (  # noqa: E402
    capability_regression,
    generation_indexed_responsiveness,
    leg_family_bh,
    pair_did,
    permutation_test,
    sign_count,
)


def _legs(rng, n_legs=6, n_seeds=30, shift=0.0, common_shock=0.0):
    """Simulated leg results; optional per-seed COMMON shock induces cross-leg dependence."""
    shock = common_shock * rng.standard_normal(n_seeds)
    return {
        f"leg{i}": {
            "cvar_diff_per_seed": shift + shock + 0.01 * rng.standard_normal(n_seeds),
            "t0_floor_pass": True,
        }
        for i in range(n_legs)
    }


# --------------------------------------------------------------------------- #
# sign_count                                                                    #
# --------------------------------------------------------------------------- #
def test_sign_count_excludes_failed_floor_legs():
    rng = np.random.default_rng(0)
    legs = _legs(rng, n_legs=4, shift=0.02)
    legs["legX"] = {"cvar_diff_per_seed": np.full(30, 9.9), "t0_floor_pass": False}
    out = sign_count(legs)
    assert out["n_legs_included"] == 4                       # the failed leg never votes
    assert out["excluded_failed_floor"] == ["legX"]
    assert out["n_dist_safer"] == 4


def test_sign_count_direction_convention():
    """Positive mean diff (signed-return CVaR) = dist SAFER; negative = dist worse."""
    legs = {
        "safe": {"cvar_diff_per_seed": np.full(30, +0.01), "t0_floor_pass": True},
        "bad": {"cvar_diff_per_seed": np.full(30, -0.01), "t0_floor_pass": True},
    }
    out = sign_count(legs)
    assert out["per_leg_sign"] == {"safe": 1, "bad": -1}


# --------------------------------------------------------------------------- #
# permutation_test                                                              #
# --------------------------------------------------------------------------- #
def test_permutation_deterministic_under_seed():
    rng = np.random.default_rng(1)
    legs = _legs(rng, shift=0.005)
    a = permutation_test(legs, n_reps=2000, seed=7)
    b = permutation_test(legs, n_reps=2000, seed=7)
    assert a == b


def test_permutation_null_is_calibrated():
    """Under a true null (zero shift, independent noise) the p-value is not extreme."""
    ps = []
    for s in range(5):
        rng = np.random.default_rng(100 + s)
        legs = _legs(rng, shift=0.0)
        ps.append(permutation_test(legs, n_reps=2000, seed=s)["p_value"])
    assert all(0.005 <= p <= 0.995 for p in ps)


def test_permutation_detects_a_strong_signal():
    rng = np.random.default_rng(2)
    legs = _legs(rng, shift=0.05)  # strong consistent dist-safer effect across 6 legs
    out = permutation_test(legs, n_reps=4000, seed=3)
    assert out["observed_sign_count"] == 6
    assert out["observed_statistic"] == pytest.approx(0.05, abs=0.01)  # the pooled mean
    assert out["p_value"] < 0.01


def test_permutation_is_conservative_under_shared_shocks():
    """THE dependence property (registered rationale): with strong common per-seed shocks and a
    negligible tilt, the joint-flip null absorbs the shock variance — the tiny tilt earns a LARGE
    p (no false positive), while unanimity in the descriptive COUNT is routine under this null
    (null_count_q95 high) — exactly why the count is descriptive-only and the pooled mean is the
    test statistic."""
    rng = np.random.default_rng(3)
    legs = _legs(rng, n_legs=6, shift=0.0, common_shock=0.05)  # dependence, no true effect
    for leg in legs.values():
        leg["cvar_diff_per_seed"] = leg["cvar_diff_per_seed"] + 0.001  # negligible common tilt
    out = permutation_test(legs, n_reps=4000, seed=4)
    assert out["p_value"] > 0.05                              # dependence-aware: no false positive
    assert out["null_count_q95"] >= 5                         # count-unanimity routine -> count is descriptive


def test_permutation_requires_seed_alignment():
    legs = {
        "a": {"cvar_diff_per_seed": np.zeros(30), "t0_floor_pass": True},
        "b": {"cvar_diff_per_seed": np.zeros(29), "t0_floor_pass": True},
    }
    with pytest.raises(ValueError, match="seed-aligned"):
        permutation_test(legs, n_reps=100, seed=0)


# --------------------------------------------------------------------------- #
# pair_did                                                                      #
# --------------------------------------------------------------------------- #
def test_pair_did_recovers_injected_interaction():
    rng = np.random.default_rng(5)
    n = 30
    base = rng.standard_normal(n) * 0.01
    top_scalar = base + rng.standard_normal(n) * 0.002
    top_dist = top_scalar + 0.03                    # content effect at the top rung
    bottom_scalar = base + rng.standard_normal(n) * 0.002
    bottom_dist = bottom_scalar + 0.01              # weaker at the bottom -> DiD = +0.02
    out = pair_did(top_dist, top_scalar, bottom_dist, bottom_scalar, n_boot=4000, seed=6)
    assert out["ci_low"] <= 0.02 <= out["ci_high"]
    assert out["estimate"] == pytest.approx(0.02, abs=0.005)


def test_pair_did_seed_pairing_cancels_common_shocks():
    """Seed-paired bootstrap: common seed-level shocks cancel, so the CI is far narrower than the
    shock scale — the pairing property the estimator is registered for."""
    rng = np.random.default_rng(7)
    n = 30
    shock = rng.standard_normal(n) * 1.0            # HUGE common per-seed shock
    top_scalar = shock + rng.standard_normal(n) * 0.001
    top_dist = top_scalar + 0.02
    bottom_scalar = shock + rng.standard_normal(n) * 0.001
    bottom_dist = bottom_scalar + 0.02              # DiD = 0 exactly
    out = pair_did(top_dist, top_scalar, bottom_dist, bottom_scalar, n_boot=2000, seed=8)
    assert abs(out["estimate"]) < 0.005
    assert (out["ci_high"] - out["ci_low"]) < 0.02  # pairing killed the unit-scale shock


def test_pair_did_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="seed-aligned"):
        pair_did(np.zeros(30), np.zeros(30), np.zeros(29), np.zeros(29))


# --------------------------------------------------------------------------- #
# the remaining registered instruments                                          #
# --------------------------------------------------------------------------- #
def test_leg_family_bh_delegates():
    out = leg_family_bh({"a": 0.001, "b": 0.9}, q=0.05)
    assert out["rejected"]["a"] is True and out["rejected"]["b"] is False


def test_generation_indexed_responsiveness_shape_and_trend():
    rng = np.random.default_rng(9)
    cands = []
    for gen in range(1, 7):
        x = rng.standard_normal(25)
        noise = rng.standard_normal(25)
        strength = gen / 6.0                        # responsiveness grows across generations
        m = strength * x + (1 - strength) * noise
        cands += [{"generation": gen, "x": float(xi), "m": float(mi)} for xi, mi in zip(x, m)]
    out = generation_indexed_responsiveness(cands, n_boot=200, seed=10)
    assert out["generations"] == [1, 2, 3, 4, 5, 6]
    assert all(out["per_generation"][g]["n"] == 25 for g in out["generations"])
    assert out["trend_spearman"] is not None and out["trend_spearman"] > 0.5


def test_capability_regression_anchors():
    pts = [{"label": f"m{i}", "responsiveness": i * 0.1, "external": float(i), "m2": float(i)}
           for i in range(6)]
    ext = capability_regression(pts, anchor="external")
    assert ext["rho"] == pytest.approx(1.0) and ext["primary_anchor"] == "external"
    with pytest.raises(ValueError, match="anchor"):
        capability_regression(pts, anchor="bogus")
    assert capability_regression(pts[:2], anchor="m2")["rho"] is None  # <3 points -> honest None


# ---- pooled_bound (R86 — the registered bounded-effect statement) ----------------------------------- #
def test_pooled_bound_covers_truth_and_recovers_shift() -> None:
    from src.inference.cross_model import pooled_bound

    rng = np.random.default_rng(7)
    res = pooled_bound(_legs(rng, shift=0.004), n_boot=2000)
    assert res["ci_low"] < 0.004 < res["ci_high"]          # the injected effect sits inside the CI
    assert res["ci_level"] == 0.90 and res["n_legs"] == 6


def test_pooled_bound_dependence_honest_no_fake_root_k_shrink() -> None:
    """k perfectly correlated legs must yield (about) ONE leg's CI width, never a sqrt(k) shrink."""
    from src.inference.cross_model import pooled_bound

    rng = np.random.default_rng(11)
    one = rng.standard_normal(30) * 0.01
    dup = {f"leg{i}": {"cvar_diff_per_seed": one.copy(), "t0_floor_pass": True} for i in range(9)}
    single = {"only": {"cvar_diff_per_seed": one.copy(), "t0_floor_pass": True}}
    w_dup = pooled_bound(dup, n_boot=3000)
    w_one = pooled_bound(single, n_boot=3000)
    width_dup = w_dup["ci_high"] - w_dup["ci_low"]
    width_one = w_one["ci_high"] - w_one["ci_low"]
    assert width_dup == pytest.approx(width_one, rel=0.15)  # identical, not /3


def test_pooled_bound_relative_scale_and_empty_inclusion() -> None:
    from src.inference.cross_model import pooled_bound

    rng = np.random.default_rng(3)
    legs = _legs(rng, n_legs=4, shift=0.001)
    for leg in legs.values():
        leg["cvar_b_per_seed"] = np.full(30, -0.02)         # scalar-arm pooled CVaR level = -0.02
    res = pooled_bound(legs, n_boot=1000)
    assert res["scalar_arm_pooled_cvar"] == pytest.approx(-0.02)
    assert res["relative_to_scalar_cvar"]["estimate"] == pytest.approx(res["estimate"] / 0.02)
    # No included legs -> honest None, never a fabricated bound.
    none_in = {"a": {"cvar_diff_per_seed": np.ones(5), "t0_floor_pass": False}}
    assert pooled_bound(none_in)["estimate"] is None
