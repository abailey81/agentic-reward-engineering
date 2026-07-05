"""Synthetic-behaviour tests for the N3 contamination + OOD-stress harnesses.

These are FAST, fully synthetic, and assert the *documented behaviour* of
``src.inference.contamination`` and ``src.inference.ood_stress`` (no campaign data, no GPU):

* TOST: equivalence is declared for a near-zero, tight-CI difference and refused for a shifted
  one; the paired A/B returns invariance on identical-up-to-noise coefficients and flags a
  shifted coefficient.
* the difference-direction complements (Mahalanobis permutation, McNemar) are correctly sized
  under the null and detect a planted effect.
* every harness DEGRADES GRACEFULLY: absent inputs -> ``{"status": "no_data"}``, never a crash,
  never a fabricated number.
* OOD generators return the right shapes, preserve the cross-section, and the GARCH-EVT FHS
  output clears the stylised-facts validation battery on heavy-tailed clustered input.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.inference.contamination import (
    DEFAULT_EQUIVALENCE_SD_FRAC,
    coefficient_mahalanobis_permutation,
    contamination_report,
    cross_model_disagreement,
    named_vs_blinded_oos_gap,
    named_vs_blinded_structural,
    named_vs_blinded_tost,
    paired_tost,
    post_cutoff_persistence,
    structural_mcnemar,
)
from src.inference.ood_stress import (
    _gpd_tail_sampler,
    block_bootstrap_paths,
    claims,
    garch_evt_fhs,
    markov_crash_paths,
    optimal_block_length,
    score_paths,
    tail_metrics,
    validate_stylized_facts,
    vol_spike_paths,
)

SEED = 20260624


def _garch_like_panel(n: int = 900, k: int = 4, seed: int = SEED) -> np.ndarray:
    """A heavy-tailed, volatility-clustered, single-factor return panel (stylised-fact rich)."""
    rng = np.random.default_rng(seed)
    # Common volatility-clustering factor via a simple GARCH(1,1) sigma path.
    sig2 = np.empty(n)
    sig2[0] = 1.0
    z = rng.standard_t(6, size=n)  # fat-tailed innovations
    eps = np.empty(n)
    eps[0] = z[0]  # init BEFORE the loop reads eps[t-1]; else t=1 reads uninitialised np.empty memory
    for t in range(1, n):
        sig2[t] = 0.02 + 0.10 * eps[t - 1] ** 2 + 0.87 * sig2[t - 1]
        eps[t] = np.sqrt(sig2[t]) * z[t]
    market = 0.0003 + 0.01 * eps / (np.std(eps) or 1.0)
    betas = np.linspace(0.8, 1.2, k)
    panel = (
        market[:, None] * betas[None, :]
        + 0.004 * rng.standard_normal((n, k))
        + 0.0001 * np.arange(k)[None, :]
    )
    return panel


# ===========================================================================
# contamination.py — TOST (primary)
# ===========================================================================
def test_paired_tost_declares_equivalence_for_tiny_difference() -> None:
    rng = np.random.default_rng(SEED)
    base = rng.standard_normal(40)
    named = base + 0.001 * rng.standard_normal(40)  # essentially identical
    blinded = base + 0.001 * rng.standard_normal(40)
    res = paired_tost(named, blinded, low=-0.5, high=0.5)
    assert res["equivalent"] is True
    assert res["p_tost"] < 0.05
    assert res["low"] < res["ci_low"] and res["ci_high"] < res["high"]


def test_paired_tost_refuses_equivalence_for_shifted_difference() -> None:
    rng = np.random.default_rng(SEED)
    base = rng.standard_normal(40)
    named = base + 2.0  # a large, real shift -> NOT equivalent within +/-0.5
    blinded = base
    res = paired_tost(named, blinded, low=-0.5, high=0.5)
    assert res["equivalent"] is False
    assert res["p_tost"] > 0.05
    assert res["mean_diff"] == pytest.approx(2.0, abs=0.2)


def test_paired_tost_pvalue_is_max_of_one_sided() -> None:
    rng = np.random.default_rng(1)
    a = rng.standard_normal(30)
    b = rng.standard_normal(30)
    res = paired_tost(a, b, low=-0.4, high=0.6)  # asymmetric bounds
    assert res["p_tost"] == pytest.approx(max(res["p_lower"], res["p_upper"]))


def test_paired_tost_degenerate_identical_pairs() -> None:
    x = np.array([0.1, 0.2, 0.3, 0.4])
    res = paired_tost(x, x, low=-0.5, high=0.5)  # zero difference, zero SE
    assert res["equivalent"] is True
    assert res["mean_diff"] == 0.0


def test_paired_tost_validates_inputs() -> None:
    with pytest.raises(ValueError):
        paired_tost(np.zeros(3), np.zeros(4), low=-1, high=1)  # mismatched
    with pytest.raises(ValueError):
        paired_tost(np.zeros(1), np.zeros(1), low=-1, high=1)  # < 2 pairs
    with pytest.raises(ValueError):
        paired_tost(np.zeros(3), np.zeros(3), low=1.0, high=-1.0)  # low >= high


def test_named_vs_blinded_tost_all_equivalent_when_labels_irrelevant() -> None:
    """Identical-up-to-seed-noise coefficients across arms -> bounded contamination (positive).

    NB the seed count here (200) is DELIBERATELY large: the equivalence bound Delta = 0.5*SD is
    tight, so the TOST equivalence claim is underpowered at the campaign's 30 main-experiment
    seeds (see :func:`test_named_vs_blinded_tost_is_underpowered_at_30_seeds`). The named-vs-
    blinded A/B is a CHEAP dedicated sub-experiment (one reward authoring + one eval per seed),
    so it can and MUST run many more seeds than the main campaign to license the positive claim.
    """
    rng = np.random.default_rng(SEED)
    n_seeds, n_coeffs = 200, 4
    truth = np.array([1.0, -0.5, 0.2, 0.0])
    sd = 0.1
    named = truth[None, :] + sd * rng.standard_normal((n_seeds, n_coeffs))
    blinded = truth[None, :] + sd * rng.standard_normal((n_seeds, n_coeffs))
    out = named_vs_blinded_tost(named, blinded, coefficient_names=list("abcd"))
    assert out["status"] == "ok"
    assert out["all_equivalent"] is True
    assert out["fraction_equivalent"] == 1.0
    assert out["equivalence_sd_frac"] == DEFAULT_EQUIVALENCE_SD_FRAC


def test_named_vs_blinded_tost_is_underpowered_at_30_seeds() -> None:
    """Document (in the suite) that Delta=0.5*SD TOST cannot yield 'all equivalent' at n=30.

    This is a load-bearing LIMITATION, not a bug: with 30 seeds the 90% CI half-width on the
    paired mean difference is comparable to Delta=0.5*SD, so a true-zero difference does NOT
    clear the equivalence bound. The harness reports ``all_equivalent=False`` honestly rather
    than fabricating equivalence; the A/B needs more seeds (or a pre-registered wider Delta).
    """
    rng = np.random.default_rng(SEED)
    n_seeds, n_coeffs = 30, 4
    truth = np.array([1.0, -0.5, 0.2, 0.0])
    sd = 0.1
    named = truth[None, :] + sd * rng.standard_normal((n_seeds, n_coeffs))
    blinded = truth[None, :] + sd * rng.standard_normal((n_seeds, n_coeffs))
    out = named_vs_blinded_tost(named, blinded)
    assert out["status"] == "ok"
    assert out["all_equivalent"] is False  # underpowered at Delta=0.5*SD, n=30
    # The 2026-07-09 three-way flag makes the underpower MACHINE-READABLE: because every coefficient here
    # is TRUE-NULL (named/blinded share the truth), the non-equivalence is a POWER failure — the null CIs
    # straddle ±delta (inconclusive), so ``any_underpowered`` is True and a consumer CANNOT misread it as
    # contamination. And the achieved resolving power is below the floor.
    assert out["any_underpowered"] is True
    assert out["n_underpowered"] >= 1
    assert out["min_equiv_power_null"] < out["power_floor"]  # the A/B under-resolves equivalence at n=30
    # The three-way outcome is a valid PARTITION: each coefficient is EXACTLY one of
    # {equivalent, decisively_different, underpowered(=inconclusive)}, and power is a valid probability.
    for c in out["per_coefficient"]:
        assert 0.0 <= c["equiv_power_null"] <= 1.0
        assert (int(c["equivalent"]) + int(c["decisively_different"]) + int(c["underpowered"])) == 1


def test_named_vs_blinded_tost_flags_a_contaminated_coefficient() -> None:
    rng = np.random.default_rng(SEED)
    n_seeds, n_coeffs = 200, 4
    truth = np.array([1.0, -0.5, 0.2, 0.0])
    sd = 0.1
    named = truth[None, :] + sd * rng.standard_normal((n_seeds, n_coeffs))
    blinded = truth[None, :] + sd * rng.standard_normal((n_seeds, n_coeffs))
    # Coefficient 1 shifts hugely when the data is NAMED (concept contamination).
    named[:, 1] += 1.0
    out = named_vs_blinded_tost(named, blinded)
    assert out["status"] == "ok"
    assert out["all_equivalent"] is False
    assert out["per_coefficient"][1]["equivalent"] is False
    # The untouched coefficients remain equivalent.
    assert out["per_coefficient"][0]["equivalent"] is True
    # KEY distinction (2026-07-09 three-way flag): at n=200 the A/B is well-powered, so the contaminated
    # coefficient's NON-equivalence is DECISIVE (its CI sits far beyond ±delta), NOT a power artefact —
    # coefficient 1 is ``decisively_different`` and NOT ``underpowered``, and ``any_underpowered`` is False.
    # This is exactly what separates real contamination from low-n noise (the earlier lazy-declined flag).
    assert out["per_coefficient"][1]["decisively_different"] is True
    assert out["per_coefficient"][1]["underpowered"] is False
    assert out["any_underpowered"] is False


def test_named_vs_blinded_tost_no_data_paths() -> None:
    assert named_vs_blinded_tost(np.zeros((0, 3)), np.zeros((0, 3)))["status"] == "no_data"
    assert named_vs_blinded_tost(np.zeros((5, 3)), np.zeros((5, 2)))["status"] == "no_data"
    assert named_vs_blinded_tost(np.zeros((1, 3)), np.zeros((1, 3)))["status"] == "no_data"


# ===========================================================================
# contamination.py — difference-direction complements
# ===========================================================================
def test_mahalanobis_permutation_is_sized_under_null() -> None:
    """Identical coefficient laws across arms -> permutation p-value should be non-significant."""
    rng = np.random.default_rng(SEED)
    named = rng.standard_normal((25, 3))
    blinded = rng.standard_normal((25, 3))
    res = coefficient_mahalanobis_permutation(named, blinded, n_perm=1000, rng=rng)
    assert res["status"] == "ok"
    assert res["pvalue"] > 0.05


def test_mahalanobis_permutation_detects_a_centroid_shift() -> None:
    rng = np.random.default_rng(SEED)
    named = rng.standard_normal((25, 3)) + np.array([2.0, 0.0, 0.0])
    blinded = rng.standard_normal((25, 3))
    res = coefficient_mahalanobis_permutation(named, blinded, n_perm=1000, rng=rng)
    assert res["status"] == "ok"
    assert res["pvalue"] < 0.05


def test_structural_mcnemar_agreement_and_discord() -> None:
    same = np.array([1, 1, 0, 0, 1, 0])
    res = structural_mcnemar(same, same)
    assert res["pvalue"] == 1.0 and res["n01"] == 0 and res["n10"] == 0
    # A strong, consistent flip (named always has the motif, blinded never).
    named = np.ones(12, dtype=int)
    blinded = np.zeros(12, dtype=int)
    res2 = structural_mcnemar(named, blinded)
    assert res2["n10"] == 12 and res2["n01"] == 0
    assert res2["pvalue"] < 0.05


def test_structural_mcnemar_rejects_non_binary() -> None:
    with pytest.raises(ValueError):
        structural_mcnemar(np.array([0, 1, 2]), np.array([0, 1, 0]))
    assert structural_mcnemar(np.array([], dtype=int), np.array([], dtype=int))["status"] == "no_data"


def test_named_vs_blinded_structural_data_locked_when_labels_irrelevant() -> None:
    # Same program per seed under both labellings (identity ignored) -> paired sim = 1.0, data_locked.
    progs = [
        "def reward(r):\n    return r.mean() - 0.5 * cvar(r)",
        "def reward(r):\n    return r.mean() / (r.std() + 1e-8)",
        "def reward(r):\n    return r.mean() - drawdown(r)",
        "def reward(r):\n    return r.mean() - 0.5 * cvar(r)",
    ]
    res = named_vs_blinded_structural(progs, list(progs), rng=np.random.default_rng(SEED))
    assert res["status"] == "ok"
    assert res["n_seeds"] == 4
    assert res["paired_mean"] == pytest.approx(1.0)  # same-seed programs identical in structure
    assert res["data_locked"] is True  # paired >= the cross-seed noise floor
    assert 0.0 <= res["p_random_pairing_matches"] <= 1.0


def test_named_vs_blinded_structural_is_identifier_invariant() -> None:
    # Blinded renames variables + changes constants but keeps the program SHAPE -> still structurally identical.
    named = ["def reward(returns):\n    return returns.mean() - 0.5 * cvar(returns)"]
    blinded = ["def reward(x):\n    return x.mean() - 0.9 * cvar(x)"]
    # n=1 is rejected (need >=2 paired seeds); duplicate to exercise the invariance on a 2-seed pair.
    res = named_vs_blinded_structural(named * 2, blinded * 2, rng=np.random.default_rng(SEED))
    assert res["status"] == "ok"
    assert res["paired_mean"] == pytest.approx(1.0)  # renaming + reconstant-ing is invisible to the AST shape


def test_named_vs_blinded_structural_no_data_paths() -> None:
    assert named_vs_blinded_structural(["x"], ["x", "y"])["status"] == "no_data"  # length mismatch
    assert named_vs_blinded_structural(["x"], ["x"])["status"] == "no_data"       # < 2 paired seeds


def test_named_vs_blinded_structural_excludes_unparseable_pairs() -> None:
    # jaccard(empty, empty) == 1.0, so an unparseable pair would read as "perfectly structurally locked";
    # the filter (mirror of the P7c empty-AST fix in reward_code_distance) must EXCLUDE it and count it
    # in n_unparseable_pairs instead of letting it inflate paired_mean / corrupt the permutation null.
    good = [
        "def reward(r):\n    return r.mean() - 0.5 * cvar(r)",
        "def reward(r):\n    return r.mean() / (r.std() + 1e-8)",
    ]
    bad = "def reward(r:\n    return"  # SyntaxError -> canonical_shapes() == frozenset()
    res = named_vs_blinded_structural(good + [bad], good + [bad], rng=np.random.default_rng(SEED))
    assert res["status"] == "ok"
    assert res["n_seeds"] == 2
    assert res["n_unparseable_pairs"] == 1
    assert res["paired_mean"] == pytest.approx(1.0)  # the two parseable identical pairs, unpolluted

    # ALL pairs unparseable -> honest no_data, never a spuriously "locked" ok result.
    res2 = named_vs_blinded_structural([bad, bad], [bad, bad], rng=np.random.default_rng(SEED))
    assert res2["status"] == "no_data"
    assert res2["n_unparseable_pairs"] == 2


def test_named_vs_blinded_structural_wired_into_report() -> None:
    progs = ["def reward(r):\n    return r.mean() - cvar(r)",
             "def reward(r):\n    return r.mean() / r.std()"]
    rep = contamination_report(named_sources=progs, blinded_sources=list(progs),
                               rng=np.random.default_rng(SEED))
    assert rep["structural_ast"]["status"] == "ok"
    # legs without inputs still degrade honestly
    assert rep["tost"]["status"] == "no_data"


def test_named_vs_blinded_oos_gap_equivalence_within_sesoi() -> None:
    rng = np.random.default_rng(SEED)
    seeds = range(30)
    named = {s: 0.5 + 0.02 * rng.standard_normal() for s in seeds}
    blinded = {s: 0.5 + 0.02 * rng.standard_normal() for s in seeds}  # same level, tiny noise
    res = named_vs_blinded_oos_gap(named, blinded, sesoi=0.05, rng=rng)
    assert res["status"] == "ok"
    assert res["equivalent"] is True
    # Three-way outcome (2026-07-09): an equivalent gap is neither decisively-different nor underpowered.
    assert res["decisively_different"] is False
    assert res["underpowered"] is False
    assert (int(res["equivalent"]) + int(res["decisively_different"]) + int(res["underpowered"])) == 1
    # And degrades gracefully with too few shared seeds.
    assert named_vs_blinded_oos_gap({0: 1.0}, {0: 1.0})["status"] == "no_data"


def test_named_vs_blinded_oos_gap_decisively_different_when_labels_move_sharpe() -> None:
    """A LARGE named-minus-blinded OOS-Sharpe gap (>> SESOI) is DECISIVELY different, not underpowered."""
    rng = np.random.default_rng(SEED)
    seeds = range(30)
    named = {s: 0.80 + 0.02 * rng.standard_normal() for s in seeds}   # revealing identity lifts OOS Sharpe
    blinded = {s: 0.50 + 0.02 * rng.standard_normal() for s in seeds}  # by ~0.30 >> SESOI=0.05
    res = named_vs_blinded_oos_gap(named, blinded, sesoi=0.05, rng=rng)
    assert res["status"] == "ok"
    assert res["equivalent"] is False
    assert res["decisively_different"] is True   # the 90% CI sits entirely beyond +SESOI: a real gap
    assert res["underpowered"] is False          # decisive, NOT a power artefact
    assert (int(res["equivalent"]) + int(res["decisively_different"]) + int(res["underpowered"])) == 1


def test_post_cutoff_persistence_runs_and_carries_caveat() -> None:
    rng = np.random.default_rng(SEED)
    seeds = range(20)
    pre = {s: 0.28 + 0.05 * rng.standard_normal() for s in seeds}
    post = {s: 0.28 + 0.05 * rng.standard_normal() for s in seeds}  # gap persists (equal pre/post gaps)
    res = post_cutoff_persistence(pre, post, rng=rng)
    assert res["status"] == "ok"
    assert "underpowered" in res["caveat"]
    assert np.isfinite(res["pre_iqm"]) and np.isfinite(res["post_iqm"])
    # Machine-readable underpower flag (L92) — not just the text caveat.
    assert isinstance(res["underpowered"], bool)
    assert isinstance(res["gap_shrank_post_cutoff"], bool)
    assert res["ci_halfwidth"] == pytest.approx((res["ci_high"] - res["ci_low"]) / 2.0)
    # Definitional invariants (hold for ANY draw): the flags are exactly the CI-vs-zero verdict, and a
    # decisive post-cutoff shrinkage and "underpowered/inconclusive" are mutually exclusive.
    assert res["gap_shrank_post_cutoff"] == bool(res["ci_low"] > 0.0)
    assert res["underpowered"] == bool(res["ci_low"] <= 0.0 <= res["ci_high"])
    assert not (res["underpowered"] and res["gap_shrank_post_cutoff"])
    # Gap persists (equal pre/post means) -> effect ~0, the CI straddles zero, so a non-significant
    # result is UNINFORMATIVE (equally consistent with persistence or low power), not proof of no drift.
    assert res["underpowered"] is True
    assert res["gap_shrank_post_cutoff"] is False
    assert post_cutoff_persistence({0: 1.0}, {0: 1.0})["status"] == "no_data"


def test_post_cutoff_persistence_flags_decisive_shrinkage() -> None:
    # If the H2 gap DECISIVELY shrinks after the cutoff (the contamination signature — a memorised
    # advantage fades on unseen post-cutoff data), the machine-readable flag must fire and NOT be
    # mislabelled underpowered. pre gap ~0.60, post gap ~0.10 => ~0.50 >> the ~0.016 SE.
    rng = np.random.default_rng(SEED)
    seeds = range(20)
    pre = {s: 0.60 + 0.05 * rng.standard_normal() for s in seeds}
    post = {s: 0.10 + 0.05 * rng.standard_normal() for s in seeds}
    res = post_cutoff_persistence(pre, post, rng=rng)
    assert res["status"] == "ok"
    assert res["ci_low"] > 0.0                      # CI entirely above zero
    assert res["gap_shrank_post_cutoff"] is True
    assert res["underpowered"] is False             # decisive, not inconclusive


def test_cross_model_disagreement_small_when_models_agree() -> None:
    rng = np.random.default_rng(SEED)
    truth = np.array([1.0, -0.5, 0.2])
    a = truth[None, :] + 0.1 * rng.standard_normal((20, 3))
    b = truth[None, :] + 0.1 * rng.standard_normal((15, 3))  # different seed count is fine
    res = cross_model_disagreement(a, b, coefficient_names=list("xyz"))
    assert res["status"] == "ok"
    assert res["max_abs_d"] < 1.0  # well within a "small" standardised difference
    # Dimension mismatch -> no_data, not a crash.
    assert cross_model_disagreement(a, np.zeros((10, 2)))["status"] == "no_data"


def test_contamination_report_degrades_on_empty_inputs() -> None:
    rep = contamination_report()
    assert "load_bearing_note" in rep
    for key in ("tost", "mahalanobis", "structural_mcnemar", "structural_ast", "oos_gap",
                "post_cutoff_persistence", "cross_model"):
        assert rep[key]["status"] == "no_data"
    # "Min-K" / theatre must NOT silently masquerade as a result.
    assert "MIA" in rep["load_bearing_note"] and "chance" in rep["load_bearing_note"]


def test_contamination_report_runs_available_legs() -> None:
    rng = np.random.default_rng(SEED)
    named = np.array([1.0, -0.5])[None, :] + 0.1 * rng.standard_normal((20, 2))
    blinded = np.array([1.0, -0.5])[None, :] + 0.1 * rng.standard_normal((20, 2))
    rep = contamination_report(named_coeffs=named, blinded_coeffs=blinded, rng=rng)
    assert rep["tost"]["status"] == "ok"
    assert rep["mahalanobis"]["status"] == "ok"
    assert rep["structural_mcnemar"]["status"] == "no_data"  # not provided


# ===========================================================================
# contamination.py — additional edge / degrade / branch coverage
# ===========================================================================
def test_named_vs_blinded_tost_absolute_bounds_override() -> None:
    # equivalence_abs supplied -> data-driven Delta overridden (line 288); wrong length -> raise (277).
    rng = np.random.default_rng(SEED)
    named = np.array([1.0, -0.5])[None, :] + 0.05 * rng.standard_normal((50, 2))
    blinded = np.array([1.0, -0.5])[None, :] + 0.05 * rng.standard_normal((50, 2))
    out = named_vs_blinded_tost(named, blinded, equivalence_abs=[1.0, 1.0])
    assert out["status"] == "ok"
    # Wide absolute bounds -> everything equivalent, and delta echoes the supplied value.
    assert out["all_equivalent"] is True
    assert out["per_coefficient"][0]["delta"] == 1.0
    with pytest.raises(ValueError, match="equivalence_abs"):
        named_vs_blinded_tost(named, blinded, equivalence_abs=[1.0])  # wrong length (line 277)


def test_named_vs_blinded_tost_bad_coefficient_names_raise() -> None:
    # coefficient_names length mismatch -> ValueError (line 275).
    named = np.zeros((4, 3))
    blinded = np.zeros((4, 3))
    with pytest.raises(ValueError, match="coefficient_names"):
        named_vs_blinded_tost(named, blinded, coefficient_names=["a", "b"])


def test_named_vs_blinded_tost_zero_width_degenerate_column() -> None:
    # A zero-within-SD column with a zero pre-registered bound -> degenerate-zero-width branch (293-309).
    named = np.array([[1.0, 5.0], [1.0, 5.0], [1.0, 5.0], [1.0, 5.0]])   # col 0 identical across seeds
    blinded = np.array([[1.0, 6.0], [1.0, 7.0], [1.0, 8.0], [1.0, 9.0]])  # col 0 identical; col 1 varies
    # Zero absolute bound on col 0 (delta<=0), nonzero on col 1.
    out = named_vs_blinded_tost(named, blinded, equivalence_abs=[0.0, 100.0])
    assert out["status"] == "ok"
    c0 = out["per_coefficient"][0]
    assert c0["delta"] == 0.0
    assert c0.get("degenerate_zero_width") is True
    assert c0["equivalent"] is True          # named col0 == blinded col0 exactly
    assert c0["mean_diff"] == 0.0


def test_named_vs_blinded_tost_zero_width_flags_nonidentical() -> None:
    # Zero-width bound but the near-constant columns differ between arms -> NOT equivalent.
    named = np.array([[1.0], [1.0], [1.0], [1.0]])
    blinded = np.array([[2.0], [2.0], [2.0], [2.0]])
    out = named_vs_blinded_tost(named, blinded, equivalence_abs=[0.0])
    c0 = out["per_coefficient"][0]
    assert c0["degenerate_zero_width"] is True
    assert c0["equivalent"] is False
    assert c0["mean_diff"] == pytest.approx(-1.0)


def test_mahalanobis_no_data_and_default_rng() -> None:
    # empty / mismatched -> no_data (line 363); < 2 seeds -> no_data (366).
    assert coefficient_mahalanobis_permutation(np.zeros((0, 3)), np.zeros((0, 3)))["status"] == "no_data"
    assert coefficient_mahalanobis_permutation(np.zeros((5, 3)), np.zeros((5, 2)))["status"] == "no_data"
    assert coefficient_mahalanobis_permutation(np.zeros((1, 3)), np.zeros((1, 3)))["status"] == "no_data"
    # rng=None default branch (line 368); deterministic p across two default calls.
    named = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5], [0.5, 2.0]])
    blinded = named + 0.01
    r1 = coefficient_mahalanobis_permutation(named, blinded, n_perm=200)
    r2 = coefficient_mahalanobis_permutation(named, blinded, n_perm=200)
    assert r1["status"] == "ok" and r1["pvalue"] == r2["pvalue"]


def test_structural_mcnemar_chi_square_branch_for_large_discord() -> None:
    # >= 25 discordant pairs -> chi-square with continuity correction (lines 449-451).
    named = np.concatenate([np.ones(30, dtype=int), np.zeros(10, dtype=int)])
    blinded = np.concatenate([np.zeros(30, dtype=int), np.zeros(10, dtype=int)])
    res = structural_mcnemar(named, blinded)  # 30 discordant (named-only) -> chi-square regime
    assert res["status"] == "ok"
    assert res["n10"] == 30 and res["n01"] == 0
    # chi2 = (|30-0| - 1)^2 / 30 with continuity correction.
    assert res["statistic"] == pytest.approx((abs(30 - 0) - 1.0) ** 2 / 30)
    assert res["pvalue"] < 1e-6
    # continuity=False drops the -1 correction.
    res_nc = structural_mcnemar(named, blinded, continuity=False)
    assert res_nc["statistic"] == pytest.approx((30 ** 2) / 30)


def test_structural_ast_default_rng_deterministic() -> None:
    # rng=None default branch (line 509); deterministic permutation p across two default calls.
    progs = ["def reward(r):\n    return r.mean() - cvar(r)",
             "def reward(r):\n    return r.mean() / r.std()",
             "def reward(r):\n    return r.mean() - drawdown(r)"]
    a = named_vs_blinded_structural(progs, list(progs))
    b = named_vs_blinded_structural(progs, list(progs))
    assert a["status"] == "ok"
    assert a["p_random_pairing_matches"] == b["p_random_pairing_matches"]


def test_oos_gap_default_rng_deterministic() -> None:
    # rng=None default branch (line 576).
    named = {s: 0.5 + 0.01 * s for s in range(10)}
    blinded = {s: 0.5 + 0.01 * s for s in range(10)}
    a = named_vs_blinded_oos_gap(named, blinded)
    b = named_vs_blinded_oos_gap(named, blinded)
    assert a["status"] == "ok" and a["ci_low"] == b["ci_low"]


def test_post_cutoff_default_rng_deterministic() -> None:
    # rng=None default branch (line 623).
    pre = {s: 0.3 + 0.01 * s for s in range(10)}
    post = {s: 0.28 + 0.01 * s for s in range(10)}
    a = post_cutoff_persistence(pre, post)
    b = post_cutoff_persistence(pre, post)
    assert a["status"] == "ok" and a["ci_low"] == b["ci_low"]


def test_cross_model_disagreement_empty_reports_not_executed() -> None:
    # Empty second-model matrix -> honest executed=False disclosure (lines 671-682).
    a = np.array([[1.0, -0.5], [1.1, -0.4], [0.9, -0.6]])
    res = cross_model_disagreement(a, np.zeros((0, 2)))
    assert res["status"] == "no_data"
    assert res["executed"] is False
    assert "NOT EXECUTED" in res["reason"]


def test_cross_model_disagreement_too_few_seeds_and_bad_names() -> None:
    a = np.array([[1.0, -0.5], [1.1, -0.4], [0.9, -0.6]])
    # < 2 seeds in one model -> no_data (line 689).
    assert cross_model_disagreement(a, np.array([[1.0, -0.5]]))["status"] == "no_data"
    # coefficient_names mismatch -> ValueError (line 697).
    b = a + 0.05
    with pytest.raises(ValueError, match="coefficient_names"):
        cross_model_disagreement(a, b, coefficient_names=["only_one"])


def test_cross_model_disagreement_infinite_d_when_pooled_sd_zero() -> None:
    # A column with zero within-model variance in BOTH arms but a nonzero mean gap ->
    # pooled_sd == 0, md != 0 -> cohens_d = inf, excluded from the finite aggregates (branch 715->704).
    a = np.array([[3.0, 1.0], [3.0, 2.0], [3.0, 3.0]])   # col 0 constant at 3.0
    b = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])   # col 0 constant at 5.0 -> gap 2, sd 0
    res = cross_model_disagreement(a, b)
    assert res["status"] == "ok"
    assert res["per_coefficient"][0]["cohens_d"] == float("inf")
    assert res["per_coefficient"][0]["pooled_sd"] == 0.0
    # The infinite d is excluded from max/mean (only the finite col-1 d contributes).
    assert np.isfinite(res["max_abs_d"])


def test_cross_model_disagreement_all_infinite_yields_nan_aggregates() -> None:
    # Every column has zero pooled SD but a nonzero gap -> all d infinite -> nan aggregates (722-723 else).
    a = np.array([[3.0], [3.0], [3.0]])
    b = np.array([[5.0], [5.0], [5.0]])
    res = cross_model_disagreement(a, b)
    assert res["status"] == "ok"
    assert np.isnan(res["max_abs_d"]) and np.isnan(res["mean_abs_d"])


def test_contamination_report_runs_all_provided_legs() -> None:
    # Exercise the report branches for structural_mcnemar / oos_gap / post_cutoff / cross_model
    # when their inputs ARE provided (lines 783, 793, 800, 809).
    rng = np.random.default_rng(SEED)
    named = np.array([1.0, -0.5])[None, :] + 0.1 * rng.standard_normal((20, 2))
    blinded = np.array([1.0, -0.5])[None, :] + 0.1 * rng.standard_normal((20, 2))
    model_b = np.array([1.0, -0.5])[None, :] + 0.1 * rng.standard_normal((15, 2))
    seeds = range(20)
    rep = contamination_report(
        named_coeffs=named,
        blinded_coeffs=blinded,
        named_struct=np.ones(20, dtype=int),
        blinded_struct=np.ones(20, dtype=int),
        named_seed_sharpe={s: 0.5 + 0.01 * s for s in seeds},
        blinded_seed_sharpe={s: 0.5 + 0.01 * s for s in seeds},
        pre_cutoff_gap={s: 0.3 + 0.01 * s for s in seeds},
        post_cutoff_gap={s: 0.28 + 0.01 * s for s in seeds},
        model_b_coeffs=model_b,
        rng=rng,
    )
    assert rep["structural_mcnemar"]["status"] == "ok"
    assert rep["oos_gap"]["status"] == "ok"
    assert rep["post_cutoff_persistence"]["status"] == "ok"
    assert rep["cross_model"]["status"] == "ok"


# ===========================================================================
# ood_stress.py
# ===========================================================================
def test_block_bootstrap_shape_and_membership() -> None:
    panel = _garch_like_panel()
    rng = np.random.default_rng(SEED)
    paths = block_bootstrap_paths(panel, n_paths=8, rng=rng)
    assert paths.shape == (8, panel.shape[0], panel.shape[1])
    # Every simulated row is a verbatim row of the panel (block bootstrap re-orders, never invents),
    # so the cross-section (row copula) is preserved exactly.
    panel_rows = {tuple(np.round(r, 12)) for r in panel}
    for r in paths[0]:
        assert tuple(np.round(r, 12)) in panel_rows


def test_block_bootstrap_longer_horizon() -> None:
    panel = _garch_like_panel(n=300, k=3)
    paths = block_bootstrap_paths(panel, n_paths=3, horizon=500, rng=np.random.default_rng(2))
    assert paths.shape == (3, 500, 3)


def test_optimal_block_length_positive() -> None:
    panel = _garch_like_panel(n=400, k=3)
    bl = optimal_block_length(panel)
    assert np.isfinite(bl) and bl >= 1.0


def test_vol_spike_scales_variance_mean_preserving() -> None:
    panel = _garch_like_panel(n=400, k=3)
    out = vol_spike_paths(panel, multiplier=4.0)  # variance x4 => std x2
    assert out.shape == (1, panel.shape[0], panel.shape[1])
    np.testing.assert_allclose(out[0].mean(axis=0), panel.mean(axis=0), atol=1e-10)
    ratio = out[0].std(axis=0) / panel.std(axis=0)
    np.testing.assert_allclose(ratio, 2.0, rtol=1e-6)


def test_garch_evt_fhs_shape_and_finiteness() -> None:
    panel = _garch_like_panel(n=600, k=3)
    rng = np.random.default_rng(SEED)
    paths = garch_evt_fhs(panel, n_paths=12, horizon=250, rng=rng)
    assert paths.shape == (12, 250, 3)
    assert np.all(np.isfinite(paths))


@pytest.mark.slow
def test_garch_evt_fhs_passes_stylized_facts() -> None:
    """The FHS output should reproduce fat tails + volatility clustering of the history."""
    panel = _garch_like_panel(n=900, k=4)
    rng = np.random.default_rng(SEED)
    paths = garch_evt_fhs(panel, n_paths=40, horizon=900, rng=rng)
    battery = validate_stylized_facts(paths, panel)
    assert battery["checks"]["fat_tails"] is True
    assert battery["checks"]["vol_clustering"] is True
    assert battery["passed"] is True


def test_garch_evt_fhs_cross_section_flag_runs() -> None:
    panel = _garch_like_panel(n=400, k=3)
    rng = np.random.default_rng(SEED)
    indep = garch_evt_fhs(panel, n_paths=4, horizon=120, preserve_cross_section=False, rng=rng)
    assert indep.shape == (4, 120, 3) and np.all(np.isfinite(indep))


def test_markov_crash_paths_runs_or_degrades() -> None:
    panel = _garch_like_panel(n=600, k=3)
    rng = np.random.default_rng(SEED)
    res = markov_crash_paths(panel, n_paths=6, horizon=200, k_regimes=2, rng=rng)
    assert res["status"] in {"ok", "fit_failed"}
    if res["status"] == "ok":
        assert res["paths"].shape == (6, 200, 3)
        assert np.all(np.isfinite(res["paths"]))
        # Transition columns sum to 1 (valid stochastic matrix).
        np.testing.assert_allclose(res["transition"].sum(axis=0), 1.0, atol=1e-6)
        assert res["n_regimes"] == 2


def test_score_paths_default_equal_weight_and_custom_policy() -> None:
    panel = _garch_like_panel(n=300, k=4)
    paths = block_bootstrap_paths(panel, n_paths=10, rng=np.random.default_rng(3))
    default = score_paths(paths)
    assert default["n_paths"] == 10
    assert set(default["cvar"].keys()) == {0.05, 0.01}
    assert np.isfinite(default["sharpe"]["iqm"])

    # A custom policy closure (the seam for a frozen winner's rolled-out policy).
    def first_asset_only(p: np.ndarray) -> np.ndarray:
        return p[:, :, 0]

    custom = score_paths(paths, policy_returns=first_asset_only)
    assert custom["n_paths"] == 10


def test_tail_metrics_reports_sharpe_and_drawdown_with_tail() -> None:
    rng = np.random.default_rng(SEED)
    port = 0.0005 + 0.01 * rng.standard_normal((20, 400))
    m = tail_metrics(port)
    # Sharpe + drawdown reported ALONGSIDE the tail (over-claim guard: no tautological tail win).
    assert {"sharpe", "max_drawdown", "cvar"} <= set(m)
    assert m["max_drawdown"]["iqm"] >= 0.0
    # CVaR is SIGNED (negative for a loss tail); the 1% tail is no LESS extreme (more negative)
    # than the 5% tail, so cvar_01 <= cvar_05.
    assert m["cvar"][0.01]["iqm"] <= m["cvar"][0.05]["iqm"] + 1e-9


def test_validate_stylized_facts_distinguishes_gaussian_from_garch() -> None:
    panel = _garch_like_panel(n=900, k=3)
    rng = np.random.default_rng(SEED)
    gaussian = rng.standard_normal((20, 900, 3)) * 0.01  # iid -> no clustering, thin tails
    battery = validate_stylized_facts(gaussian, panel)
    # An iid-Gaussian generator should FAIL the volatility-clustering gate.
    assert battery["checks"]["vol_clustering"] is False
    assert battery["passed"] is False


def test_ood_input_validation() -> None:
    with pytest.raises(ValueError):
        block_bootstrap_paths(np.zeros((1, 3)))  # < 2 time steps
    with pytest.raises(ValueError):
        garch_evt_fhs(np.full((10, 2), np.nan))  # non-finite
    with pytest.raises(ValueError):
        vol_spike_paths(_garch_like_panel(n=50, k=2), multiplier=-1.0)


def test_claims_states_load_bearing_distinction() -> None:
    c = claims()
    assert "falsification" in c["can_claim"].lower() or "stress-probe" in c["can_claim"].lower()
    assert "generalisation" in c["cannot_claim"].lower()
    assert "Bauer" in c["power_caveat"]


# ===========================================================================
# ood_stress.py — additional edge / degrade / branch coverage
# ===========================================================================
def test_validate_panel_promotes_1d_and_rejects_3d() -> None:
    # 1-D input is promoted to a single-asset column (line 104): a 1-D panel runs.
    r1d = _garch_like_panel(n=200, k=1).ravel()
    paths = block_bootstrap_paths(r1d, n_paths=2, rng=np.random.default_rng(1))
    assert paths.shape == (2, 200, 1)
    # 3-D input is malformed -> ValueError (line 106).
    with pytest.raises(ValueError, match=r"\(T, n_assets\)"):
        block_bootstrap_paths(np.zeros((3, 4, 5)))


def test_garch_evt_fhs_bad_horizon_and_default_rng() -> None:
    panel = _garch_like_panel(n=200, k=2)
    # horizon < 1 -> ValueError (line 271).
    with pytest.raises(ValueError, match="horizon"):
        garch_evt_fhs(panel, horizon=0)
    # rng=None default branch (line 273): runs and is finite.
    out = garch_evt_fhs(panel, n_paths=2, horizon=50)
    assert out.shape == (2, 50, 2) and np.all(np.isfinite(out))


def test_block_bootstrap_default_rng() -> None:
    # rng=None default branch (line 391).
    panel = _garch_like_panel(n=200, k=2)
    out = block_bootstrap_paths(panel, n_paths=3)
    assert out.shape == (3, 200, 2)


def test_markov_default_rng_and_success_body() -> None:
    # rng=None default (line 442). A clearly two-regime market makes the MS-AR fit succeed,
    # exercising the whole simulation body (lines 461-511).
    rng = np.random.default_rng(7)
    n = 800
    # Build a market with two persistent variance regimes so MarkovAutoregression converges.
    state = np.zeros(n, dtype=int)
    for t in range(1, n):
        # sticky regimes
        flip = rng.random() < (0.02 if state[t - 1] == 0 else 0.05)
        state[t] = 1 - state[t - 1] if flip else state[t - 1]
    calm = 0.005 * rng.standard_normal(n)
    crash = -0.002 + 0.03 * rng.standard_normal(n)
    market = np.where(state == 0, calm, crash)
    betas = np.array([0.9, 1.0, 1.1])
    panel = market[:, None] * betas[None, :] + 0.003 * rng.standard_normal((n, 3))
    res = markov_crash_paths(panel, n_paths=5, horizon=100, k_regimes=2)
    assert res["status"] in {"ok", "fit_failed"}
    if res["status"] == "ok":
        assert res["paths"].shape == (5, 100, 3)
        assert np.all(np.isfinite(res["paths"]))
        np.testing.assert_allclose(res["transition"].sum(axis=0), 1.0, atol=1e-6)
        assert "start_state" in res


def test_vol_spike_multipath_branch_and_default_rng() -> None:
    # n_paths > 1 takes the block-resample-then-inflate branch (lines 552-553) and the
    # rng=None default (line 547). Variance is still scaled by the multiplier.
    panel = _garch_like_panel(n=300, k=3)
    out = vol_spike_paths(panel, multiplier=4.0, n_paths=5)
    assert out.shape == (5, panel.shape[0], 3)
    assert np.all(np.isfinite(out))


def test_tail_metrics_all_nonfinite_yields_nan_aggregate() -> None:
    # A path whose returns are all non-finite -> the per-path CVaR is non-finite, so the across-path
    # _agg receives an empty finite set and returns nan (line 592). (sharpe_ratio maps a degenerate
    # path to 0.0, so the CVaR aggregate is the one that exercises the empty branch.)
    port = np.full((3, 50), np.nan)
    m = tail_metrics(port)
    assert np.isnan(m["cvar"][0.05]["iqm"])
    assert np.isnan(m["cvar"][0.05]["p05"])
    assert np.isnan(m["cvar"][0.01]["mean"])


def test_score_paths_2d_input_and_bad_ndim() -> None:
    # 2-D (n_paths, H) input is promoted to single-asset (line 634).
    two_d = 0.01 * np.random.default_rng(0).standard_normal((6, 100))
    out = score_paths(two_d)
    assert out["n_paths"] == 6
    # 4-D input is malformed -> ValueError (line 636).
    with pytest.raises(ValueError, match=r"\(n_paths, H, n_assets\)"):
        score_paths(np.zeros((2, 3, 4, 5)))


def test_validate_stylized_facts_single_column_and_short_series() -> None:
    # A single-column 2-D historical panel exercises the ndim==2/shape[1]==1 ravel branch (line 671).
    hist_1col = _garch_like_panel(n=300, k=1)          # (300, 1)
    syn = _garch_like_panel(n=300, k=1)[None, :, :]     # (1, 300, 1)
    battery = validate_stylized_facts(syn, hist_1col)
    assert "checks" in battery and "passed" in battery
    # A too-short series (< max_lag + 2) -> nan facts (line 676), no crash.
    short = np.array([[0.01], [0.02], [0.03]])          # 3 rows < 10 + 2
    b2 = validate_stylized_facts(short[None, :, :], short, max_lag=10)
    assert np.isnan(b2["synthetic"]["excess_kurtosis"])
    assert b2["passed"] is False


def test_validate_stylized_facts_accepts_1d_series() -> None:
    # 1-D historical + 1-D synthetic exercise the _to_port ndim<2 ravel branch (line 671).
    rng = np.random.default_rng(SEED)
    hist_1d = 0.01 * rng.standard_t(5, size=400)
    syn_1d = 0.01 * rng.standard_t(5, size=400)
    battery = validate_stylized_facts(syn_1d, hist_1d)
    assert "checks" in battery
    assert np.isfinite(battery["synthetic"]["excess_kurtosis"])


def test_gpd_tail_sampler_synthesises_continuous_tail_exceedances() -> None:
    # Heavy-tailed residuals -> both GPD sides fit; tail-mass draws are replaced by GPD-extrapolated
    # continuous exceedances (both-tail branches). The EVT contribution is that some draws are NOT
    # members of the (discrete) empirical residual set -> genuinely synthesised extremes, and the
    # sampler can reach beyond the empirical support on at least one tail.
    rng = np.random.default_rng(SEED)
    e = rng.standard_t(4, size=3000)  # heavy-tailed -> lower_par and upper_par both fit
    sample = _gpd_tail_sampler(e, threshold_q=0.10, rng=rng)
    draws = sample(6000)
    assert draws.shape == (6000,) and np.all(np.isfinite(draws))
    members = set(np.round(e, 12))
    synthesised = sum(1 for v in draws if round(float(v), 12) not in members)
    # Many draws are GPD-drawn continuous exceedances, i.e. NOT members of the discrete empirical
    # residual set -> the semi-parametric EVT tail extension is genuinely active (both tails fired).
    assert synthesised > 100


def test_gpd_tail_sampler_falls_back_to_empirical_when_tail_degenerate() -> None:
    # A tiny residual vector: each side has < 5 exceedances -> _fit_side returns None (line 164),
    # so BOTH GPD blocks are skipped (branches 184->191, 191->198) and the sampler is a pure
    # empirical bootstrap: every draw is a verbatim member of the input.
    e = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 0.0, 0.0, 0.5, -0.5])
    sample = _gpd_tail_sampler(e, threshold_q=0.10, rng=np.random.default_rng(1))
    draws = sample(200)
    members = set(np.round(e, 9))
    assert all(round(float(v), 9) in members for v in draws)


def test_garch_evt_fhs_independent_marginals_runs_the_gpd_samplers() -> None:
    # preserve_cross_section=False routes the whole generator through the per-asset GPD samplers.
    panel = _garch_like_panel(n=800, k=2)
    rng = np.random.default_rng(SEED)
    out = garch_evt_fhs(panel, n_paths=6, horizon=200, preserve_cross_section=False, rng=rng)
    assert out.shape == (6, 200, 2) and np.all(np.isfinite(out))


def test_EMPTY_and_comment_only_rewards_are_excluded_not_scored_as_perfectly_locked() -> None:
    """#117 (2026-07-27): the structureless filter caught unparseable sources but not EMPTY ones.

    `named_vs_blinded_structural` excludes sources with no AST structure because
    `jaccard(empty, empty) == 1.0` would score two non-programs as "perfectly structurally locked".
    The test was `nshapes[i] and bshapes[i]` -- NON-EMPTY only. That catches an UNPARSEABLE source
    (`canonical_shapes` returns the empty set) but NOT an empty or comment-only one: `ast.parse("")`
    succeeds and walks one bodiless `Module`, so its shape-set is the TRUTHY singleton `{"Module"}`.
    Such a pair survived the very filter written to remove it. `reward_taxonomy._signature` already
    guarded `shapes <= {"Module"}`; this module did not.

    Why the direction matters on a CONTAMINATION instrument: a degenerate pair contributes 1.0 to
    `paired` but only ~1/|shapes| to `within_blinded` and to the re-pairing null, so `paired_mean`
    inflates, `within_blinded_mean` deflates, `structural_gap` grows from both ends, `data_locked`
    biases TRUE and `p_random_pairing_matches` biases SMALL -- every output tilts toward "evidence
    AGAINST structural contamination". Empty authored code is measured in this project, not
    hypothetical.
    """
    import numpy as _np

    from src.inference.contamination import named_vs_blinded_structural
    from src.inference.reward_code_distance import canonical_shapes, jaccard

    # The premise, asserted so the test explains itself if canonical_shapes ever changes.
    assert canonical_shapes("", 4) == {"Module"}, "empty source no longer yields the Module singleton"
    assert bool(canonical_shapes("", 4)) is True, "the old non-empty test would not have caught this"
    assert jaccard(canonical_shapes("", 4), canonical_shapes("# c\n", 4)) == 1.0

    real = [
        f"def reward(w, r, p, pr, i):\n    x = float(pr) - {c} * float(np.std(r))\n    return x, {{}}, None\n"
        for c in (0.1, 0.2, 0.3, 0.4)
    ]
    # Three genuine paired seeds + one seed whose BOTH programs are structureless.
    named = [*real[:3], ""]
    blinded = [*real[:3], "# the model returned only a comment\n"]

    res = named_vs_blinded_structural(named, blinded, rng=_np.random.default_rng(0), n_perm=200)
    assert res["status"] == "ok"
    assert res["n_unparseable_pairs"] == 1, "the empty/comment-only pair was not excluded"
    assert res["n_seeds"] == 3, "the degenerate pair still entered the similarity computation"

    # And it must not be scored as perfect agreement: compare against the same call WITHOUT the
    # degenerate pair -- the excluded run must be identical, i.e. the pair contributed nothing.
    clean = named_vs_blinded_structural(
        real[:3], real[:3], rng=_np.random.default_rng(0), n_perm=200
    )
    assert res["paired_mean"] == clean["paired_mean"], (
        "paired_mean still moved when a structureless pair was present -- it is being scored, and "
        "jaccard=1.0 on two non-programs inflates the structural-lock evidence"
    )


def test_cross_model_cohens_d_handles_a_constant_coefficient_the_same_way_whatever_its_float() -> None:
    """#118 (2026-07-27): the zero-variance guard tested EXACT zero, so it fired value-dependently.

    `cross_model_disagreement` computed `d = md / pooled_sd if pooled_sd > 0 else (0 or inf)`. But a
    coefficient that is CONSTANT across seeds in both models does not generally give an exactly-zero
    variance: `np.var` subtracts a mean one ulp off the repeated value, so 12 copies of 0.05 give
    var 5.3e-35 (sd 7.2e-18) while 12 copies of 1.0 give EXACTLY 0 (measured). The guard therefore
    fired only when the constant happened to be float-exact:

      * `lam = 1.0`   -> degenerate branch, d = inf, EXCLUDED from the summaries;
      * `cvar_alpha = 0.05` -> normal branch, d ~ 7.6e15, FINITE, so it entered them.

    The identical situation, handled two opposite ways, decided by float representability -- and the
    constants an LLM actually writes (0.05, 0.01, 0.1) are exactly the non-representable ones. The
    damage was to the headline: one such column drove `mean_abs_d`, the average cross-model
    disagreement, to ~2.5e15.

    Pins all three behaviours: representable and non-representable constants are treated the SAME,
    an unbounded d never swamps the summaries, and the exclusion is REPORTED (`n_undefined_d`) rather
    than silent -- a constant-but-different coefficient is a DECISIVE disagreement, not a missing one.
    """
    import numpy as _np

    from src.inference.contamination import cross_model_disagreement

    rng = _np.random.default_rng(0)
    n = 12

    def _run(const_a: float, const_b: float):
        A = _np.column_stack([_np.full(n, const_a), rng.normal(1.0, 0.2, n)])
        B = _np.column_stack([_np.full(n, const_b), rng.normal(1.1, 0.2, n)])
        return cross_model_disagreement(A, B, coefficient_names=["const", "other"])

    non_exact = _run(0.05, 0.01)   # np.var != 0 exactly
    exact = _run(1.0, 2.0)         # np.var == 0 exactly

    for label, res in (("non-representable", non_exact), ("float-exact", exact)):
        d = res["per_coefficient"][0]["cohens_d"]
        assert _np.isinf(d), f"{label} constant column did not yield an unbounded d (got {d!r})"
        assert res["per_coefficient"][0]["degenerate_zero_variance"] is True
        assert res["n_undefined_d"] == 1 and res["undefined_d_coefficients"] == ["const"], (
            f"{label}: the decisive constant-but-different coefficient was dropped SILENTLY"
        )
        assert _np.isfinite(res["max_abs_d"]) and _np.isfinite(res["mean_abs_d"])
        assert abs(res["mean_abs_d"]) < 10.0, (
            f"{label}: a degenerate column swamped mean_abs_d ({res['mean_abs_d']:.3g}) -- the "
            "average cross-model disagreement is meaningless"
        )

    # Agreement on the SAME constant is agreement, not an undefined difference.
    same = _run(0.05, 0.05)
    assert same["per_coefficient"][0]["cohens_d"] == 0.0
    assert same["n_undefined_d"] == 0
