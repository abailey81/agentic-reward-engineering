"""Behaviour tests for the σ_D/ρ pilot extractor (the n_seeds freeze-blocker)."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.sigma_seed_pilot import (
    estimate_seed_count,
    k80_for_rho,
    paired_seed_stats,
    run_pilot,
)


def _crn_pair(n: int, sigma: float, rho: float, seed: int = 0) -> tuple[dict[int, float], dict[int, float]]:
    """Two per-seed score dicts with per-arm SD ``sigma`` and pairing correlation ``rho``."""
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    a = sigma * z1
    b = sigma * (rho * z1 + np.sqrt(max(0.0, 1.0 - rho * rho)) * z2)
    return {i: float(a[i]) for i in range(n)}, {i: float(b[i]) for i in range(n)}


# --------------------------------------------------------------------------- #
# paired_seed_stats                                                            #
# --------------------------------------------------------------------------- #
def test_recovers_sigma_and_rho_on_large_sample():
    a, b = _crn_pair(4000, sigma=0.25, rho=0.6, seed=1)
    s = paired_seed_stats(a, b)
    assert s["status"] == "ok"
    assert s["n_shared"] == 4000
    assert s["sigma_a"] == pytest.approx(0.25, abs=0.02)
    assert s["sigma_b"] == pytest.approx(0.25, abs=0.02)
    assert s["sigma_seed"] == pytest.approx(0.25, abs=0.02)
    assert s["rho"] == pytest.approx(0.6, abs=0.03)
    # CRN identity σ_D² = σ_a² + σ_b² − 2ρσ_aσ_b must hold (descriptive check).
    assert s["crn_identity_ok"] is True


def test_pairing_reduces_diff_variance_only_when_rho_positive():
    # ρ>0: σ_D < √(σ_a²+σ_b²) (pairing helps). ρ<0: σ_D > it (pairing HURTS).
    a_pos, b_pos = _crn_pair(4000, 0.3, rho=0.7, seed=2)
    a_neg, b_neg = _crn_pair(4000, 0.3, rho=-0.5, seed=3)
    sp, sn = paired_seed_stats(a_pos, b_pos), paired_seed_stats(a_neg, b_neg)
    indep = np.sqrt(0.3**2 + 0.3**2)
    assert sp["sigma_d"] < indep
    assert sn["sigma_d"] > indep


def test_uses_only_shared_seeds():
    a = {0: 1.0, 1: 2.0, 2: 3.0, 5: 9.0}
    b = {1: 2.1, 2: 2.9, 3: 0.0}  # shared = {1, 2}
    s = paired_seed_stats(a, b)
    assert s["n_shared"] == 2
    assert s["shared_seeds"] == [1, 2]


def test_degrades_gracefully_below_two_shared_seeds():
    s = paired_seed_stats({0: 1.0}, {0: 1.1})
    assert s["status"] == "skipped"
    assert s["sigma_seed"] is None
    assert "shared" in s["reason"]


def test_drops_nonfinite_pairs():
    a = {0: 1.0, 1: float("nan"), 2: 3.0, 3: 4.0}
    b = {0: 1.1, 1: 2.0, 2: float("inf"), 3: 4.2}
    s = paired_seed_stats(a, b)
    assert s["n_shared"] == 2  # seeds 0 and 3 only
    assert s["shared_seeds"] == [0, 3]


# --------------------------------------------------------------------------- #
# k80_for_rho                                                                  #
# --------------------------------------------------------------------------- #
def test_k80_matches_doc_grid_and_is_monotone():
    assert k80_for_rho(0.0) == pytest.approx(0.72)
    assert k80_for_rho(0.7) == pytest.approx(0.44)
    assert k80_for_rho(0.3) == pytest.approx(0.61)
    # monotone non-increasing in ρ over [0, 0.7]
    vals = [k80_for_rho(r) for r in (0.0, 0.2, 0.4, 0.6, 0.7)]
    assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))


def test_k80_floors_for_negative_rho_and_none():
    assert k80_for_rho(-0.5) == pytest.approx(0.72)  # no pairing benefit -> ρ=0 floor
    assert k80_for_rho(None) == pytest.approx(0.72)


# --------------------------------------------------------------------------- #
# estimate_seed_count (SESOI-derived decision)                                 #
# --------------------------------------------------------------------------- #
def test_small_sigma_seed_is_equivalence_achievable_at_floor():
    d = estimate_seed_count(0.05, rho=0.0)
    assert d["achievable_equivalence"] is True
    assert d["recommended_n"] == 30
    assert d["headline_framing"] == "practical_equivalence"


def test_large_sigma_seed_is_not_achievable():
    # The directional proxy 0.36 maps to MDE ~0.18 DSR >> 0.05 SESOI at every feasible n.
    d = estimate_seed_count(0.36, rho=0.0)
    assert d["achievable_equivalence"] is False
    assert d["recommended_n"] is None
    assert d["headline_framing"] == "bounded_effect_inconclusive"


def test_ladder_mde_decreases_with_n_and_recommends_smallest_passing():
    d = estimate_seed_count(0.12, rho=0.0)
    dsr = [row["mde_dsr"] for row in d["ladder"]]
    assert all(dsr[i] > dsr[i + 1] for i in range(len(dsr) - 1))  # MDE shrinks with n
    if d["achievable_equivalence"]:
        # recommended n is the first grid point meeting the SESOI
        first = next(row["n"] for row in d["ladder"] if row["meets_sesoi"])
        assert d["recommended_n"] == first


def test_negative_rho_is_flagged():
    d = estimate_seed_count(0.1, rho=-0.3)
    assert d["rho_negative_warning"] is True


# --------------------------------------------------------------------------- #
# run_pilot driver (graceful on empty / records-only)                          #
# --------------------------------------------------------------------------- #
def test_run_pilot_skips_on_empty_records():
    res = run_pilot([], arm_a="differential_sharpe", arm_b="return_minus_cvar")
    assert res["sigma_seed_pilot"] is False
    assert res["per_statistic"]["sharpe"]["decision"]["status"] == "skipped"


def test_run_pilot_measures_from_synthetic_records():
    # Two fixed-reward arms, 12 shared seeds, distinct per-step test return series per seed.
    rng = np.random.default_rng(7)
    records = []
    for arm, mu in (("differential_sharpe", 0.0006), ("return_minus_cvar", 0.0005)):
        for seed in range(12):
            r = rng.normal(mu, 0.01, size=300) + 0.002 * (seed - 6) / 6.0  # seed-correlated shift
            records.append({"arm": arm, "seed": seed, "metrics": {"test_returns": r.tolist()}})
    res = run_pilot(records, arm_a="differential_sharpe", arm_b="return_minus_cvar")
    assert res["sigma_seed_pilot"] is True
    sh = res["per_statistic"]["sharpe"]["stats"]
    assert sh["status"] == "ok"
    assert sh["n_shared"] == 12
    assert sh["sigma_seed"] is not None and sh["sigma_seed"] > 0
    # both legs present
    assert "cvar_05" in res["per_statistic"]
