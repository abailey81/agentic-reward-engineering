"""Tests for the campaign power analysis (FINAL_PLAN B-5; viva Q21).

Behaviour checks (all fast — tiny n_sims/n_boot, deterministic seeds):
  * the fixed import resolves (the stub's ``from src.regimes import detect`` is gone);
  * ``independent_regime_count`` returns a sane N on a regime fixture;
  * the MC power routine returns power in [0, 1] with the right MONOTONICITY:
      - power increases with the effect size,
      - the MDE increases with sigma,
      - the MDE decreases with more regimes / more seeds (larger n_eff);
  * the selection-aware alpha penalty shrinks alpha and is the identity at m = 1;
  * the TOST equivalence logic flags equivalence / non-equivalence correctly.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

# Make ``scripts`` importable as a namespace package (mirrors tests/test_cost_sweep.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.panel import Panel  # noqa: E402
from src.regimes.definition import independent_regime_count, label_regimes  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def test_import_resolves() -> None:
    """The module imports and exposes its public API (the broken import is fixed)."""
    mod = importlib.import_module("scripts.power_analysis")
    for name in (
        "PowerConfig",
        "simulate_power",
        "minimum_detectable_effect",
        "selection_aware_alpha",
        "tost_equivalence",
        "regime_count_on_panel",
    ):
        assert hasattr(mod, name), name
    # The stub's bad symbol must not be present.
    import src.regimes as regimes_pkg

    assert not hasattr(regimes_pkg, "detect")


def test_assurance_seed_ladder_reproduces_the_seed_decision():
    """B-A3: the χ² upper-CI assurance sizing must reproduce the seed-decision doc EXACTLY, so the
    grade-securing tier ladder rests on verified numbers, not a re-derivation that could drift."""
    pa = importlib.import_module("scripts.power_analysis")
    assert pa.assurance_seed_count(0.80)["n"] == 279
    assert pa.assurance_seed_count(0.90)["n"] == 340
    assert pa.assurance_seed_count(0.95)["n"] == 403
    assert pa.assurance_seed_count(0.99)["n"] == 568  # the new 99% tier
    ns = [pa.assurance_seed_count(c)["n"] for c in (0.80, 0.90, 0.95, 0.99)]
    assert ns == sorted(ns)  # monotone in confidence
    # E1 ladder: 30 core / 100 σ-precision / 189 point-estimate power / 279=80% / 340=90% / 403=95% / 568=99%
    assert pa.ASSURANCE_TIER_BOUNDS == (30, 100, 189, 279, 340, 403, 568)
    # σ_up at 90% is 0.495 (NOT 0.449 — that is the 80% bound; the seed-decision doc mislabels it)
    assert abs(pa.assurance_seed_count(0.90)["sigma_up"] - 0.495) < 0.002
    assert abs(pa.assurance_seed_count(0.80)["sigma_up"] - 0.449) < 0.002
    with pytest.raises(ValueError, match="confidence"):
        pa.assurance_seed_count(1.5)


def test_recommend_assurance_target_is_throughput_and_deadline_aware():
    """GRADE SECURITY: pick the HIGHEST assurance tier whose uniform-n sweep fits the calendar at the
    MEASURED throughput; fall back to the n=30 distinction floor when even 90% will not fit. The stop
    is exogenous (throughput + calendar, never the effect), so every recommended tier is a valid
    single-look design."""
    pa = importlib.import_module("scripts.power_analysis")

    # PIN the sweep width: this test exercises the SELECTION LOGIC, not the roster size. Letting it
    # inherit the config-derived default would make a legitimate roster change (R108 took arms 7 -> 9)
    # fail an unrelated behaviour test, and would silently assert a config value this test does not own.
    units = 12

    hi = pa.recommend_assurance_target(40.0, 20.0, sweep_units=units)  # ample speed + time -> 99%
    assert hi["recommended_confidence"] == 0.99 and hi["recommended_n"] == 568
    assert hi["floor_only"] is False

    lo = pa.recommend_assurance_target(10.0, 10.0, sweep_units=units)  # thin -> bank the floor only
    assert lo["floor_only"] is True and lo["recommended_n"] == 30
    assert lo["recommended_confidence"] is None

    mid = pa.recommend_assurance_target(20.0, 15.0, sweep_units=units)  # 95% fits, 99% does not
    assert mid["recommended_confidence"] == 0.95 and mid["recommended_n"] == 403

    prev = 0  # monotone in time: more days never lowers the deadline-safe target
    for d in (8, 12, 16, 24, 40):
        n = pa.recommend_assurance_target(20.0, float(d), sweep_units=units)["recommended_n"]
        assert n >= prev
        prev = n

    # a wider sweep at fixed speed+calendar can only LOWER (never raise) the reachable tier
    assert (
        pa.recommend_assurance_target(20.0, 15.0, sweep_units=2 * units)["recommended_n"]
        <= mid["recommended_n"]
    )

    with pytest.raises(ValueError):
        pa.recommend_assurance_target(0.0, 10.0)
    with pytest.raises(ValueError):
        pa.recommend_assurance_target(20.0, -1.0)
    with pytest.raises(ValueError, match="sweep_units"):
        pa.recommend_assurance_target(20.0, 15.0, sweep_units=0)


def test_assurance_sweep_units_is_derived_from_config_not_hardcoded():
    """The sweep width is arms + H1 baselines + the H3 winner, READ FROM CONFIG.

    Regression for a stale planning constant (deep review 2026-07-26): it was the literal 12 ("7 arms
    + 4 H1 + H3") while BOTH inputs had since changed (roster 7 -> 9, H1 canon 4 -> 11), so the true
    width was 21 — the recommender was 43% optimistic about what fits the calendar, silently."""
    pa = importlib.import_module("scripts.power_analysis")
    from src.utils.config import load_config

    n_arms = len(load_config("campaign")["arms"])
    n_h1 = len(load_config("preregistration")["h1_baselines"])
    assert pa.assurance_sweep_units() == n_arms + n_h1 + 1

    # and the resolved value is RECORDED in the result, so a plan can be audited after the fact
    out = pa.recommend_assurance_target(20.0, 15.0)
    assert out["sweep_units"] == pa.assurance_sweep_units()


def _cfg(**kw):
    from scripts.power_analysis import PowerConfig

    # NB: n_boot is chosen so the p-value floor 1/(n_boot+1) sits BELOW the
    # selection-aware alpha_eff (~0.0085 at m=6); with a too-small n_boot the
    # minimum achievable p-value exceeds alpha_eff and power is identically 0.
    base = dict(
        n_regimes=6,
        seeds=5,
        folds=1,
        sigma_dsr=0.30,
        target_power=0.80,
        alpha=0.05,
        n_comparisons=6,
        n_sims=100,
        n_boot=249,
        seed=0,
    )
    base.update(kw)
    return PowerConfig(**base)


# --------------------------------------------------------------------------- #
# Regime count
# --------------------------------------------------------------------------- #


def test_independent_regime_count_sane_on_fixture(synthetic_panel: Panel) -> None:
    """N from the regime API is a positive, single-to-low-double-digit integer."""
    from scripts.power_analysis import regime_count_on_panel

    cfg = load_config("regimes")
    n = regime_count_on_panel(synthetic_panel, cfg)
    direct = independent_regime_count(label_regimes(synthetic_panel, cfg))
    assert n == direct
    assert isinstance(n, int)
    assert n >= 1
    assert n <= synthetic_panel.T  # cannot exceed the number of dates


def test_n_eff_is_product() -> None:
    """n_eff = seeds x folds x regimes."""
    cfg = _cfg(n_regimes=7, seeds=5, folds=2)
    assert cfg.n_eff == 7 * 5 * 2


# --------------------------------------------------------------------------- #
# Selection-aware alpha
# --------------------------------------------------------------------------- #


def test_selection_aware_alpha_penalty() -> None:
    from scripts.power_analysis import selection_aware_alpha

    assert selection_aware_alpha(0.05, 1) == pytest.approx(0.05)
    a6 = selection_aware_alpha(0.05, 6)
    assert a6 < 0.05
    # Sidak sits just above the Bonferroni floor alpha/m.
    assert 0.05 / 6 <= a6 < 0.05
    # More comparisons => stricter per-test alpha.
    assert selection_aware_alpha(0.05, 12) < a6


# --------------------------------------------------------------------------- #
# Power routine: range + monotonicity
# --------------------------------------------------------------------------- #


def test_power_in_unit_interval() -> None:
    from scripts.power_analysis import simulate_power

    for effect in (0.0, 0.3, 1.0):
        p = simulate_power(effect, _cfg())
        assert 0.0 <= p <= 1.0


def test_power_increases_with_effect() -> None:
    """A larger true effect yields at least as much power (monotone in effect)."""
    from scripts.power_analysis import simulate_power

    cfg = _cfg(n_sims=150)
    p_small = simulate_power(0.10, cfg)
    p_large = simulate_power(1.20, cfg)
    assert p_large > p_small
    assert p_large > 0.5  # a 4-sigma gap on n_eff=30 cells is easily detected


def test_null_rejection_near_alpha_eff() -> None:
    """Under H0 (effect=0) the rejection rate is small (~ selection-aware alpha)."""
    from scripts.power_analysis import selection_aware_alpha, simulate_power

    cfg = _cfg(n_sims=300)
    p0 = simulate_power(0.0, cfg)
    a_eff = selection_aware_alpha(cfg.alpha, cfg.n_comparisons)
    # Generous MC-noise band around the (small) selection-aware alpha.
    assert p0 <= a_eff + 0.06


def test_mde_increases_with_sigma() -> None:
    """Noisier DSR (larger sigma) => a larger minimum detectable effect."""
    from scripts.power_analysis import minimum_detectable_effect

    lo = minimum_detectable_effect(_cfg(sigma_dsr=0.20), effect_points=13)
    hi = minimum_detectable_effect(_cfg(sigma_dsr=0.60), effect_points=13)
    assert lo["reached"] and hi["reached"]
    assert hi["mde"] > lo["mde"]


def test_mde_decreases_with_more_regimes() -> None:
    """More independent regimes (larger n_eff) => a smaller MDE."""
    from scripts.power_analysis import minimum_detectable_effect

    few = minimum_detectable_effect(_cfg(n_regimes=4), effect_points=13)
    many = minimum_detectable_effect(_cfg(n_regimes=12), effect_points=13)
    assert few["reached"] and many["reached"]
    assert many["mde"] < few["mde"]


def test_mde_decreases_with_more_seeds() -> None:
    """More seeds (larger n_eff) => a smaller MDE."""
    from scripts.power_analysis import minimum_detectable_effect

    few = minimum_detectable_effect(_cfg(seeds=2), effect_points=13)
    many = minimum_detectable_effect(_cfg(seeds=10), effect_points=13)
    assert few["reached"] and many["reached"]
    assert many["mde"] < few["mde"]


# --------------------------------------------------------------------------- #
# TOST equivalence
# --------------------------------------------------------------------------- #


def test_tost_flags_equivalence_for_overlapping_arms() -> None:
    """Two arms drawn from the SAME distribution are equivalent within a wide margin."""
    from scripts.power_analysis import tost_equivalence

    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 0.10, size=40)
    b = rng.normal(0.0, 0.10, size=40)
    res = tost_equivalence(a, b, margin=0.30, n_boot=400, rng=rng)
    assert res.equivalent is True
    assert -res.margin < res.estimate < res.margin


def test_tost_rejects_equivalence_for_separated_arms() -> None:
    """A clear mean gap larger than the margin is NOT flagged equivalent."""
    from scripts.power_analysis import tost_equivalence

    rng = np.random.default_rng(1)
    a = rng.normal(1.0, 0.10, size=40)  # mean 1.0
    b = rng.normal(0.0, 0.10, size=40)  # mean 0.0  => gap ~1.0 >> margin
    res = tost_equivalence(a, b, margin=0.20, n_boot=400, rng=rng)
    assert res.equivalent is False


def test_tost_margin_must_be_positive() -> None:
    from scripts.power_analysis import tost_equivalence

    with pytest.raises(ValueError):
        tost_equivalence(np.zeros(5), np.zeros(5), margin=0.0)


def test_tost_paired_mode_uses_shared_seed_index_and_narrows_ci() -> None:
    """P2: paired=True must resample a SHARED index (CRN pairing). For strongly positively-correlated arms
    (the common-random-number regime) the PAIRED difference CI is NARROWER than the INDEPENDENT one, because
    the shared-seed covariance cancels. The estimate (a difference of full-sample means) is identical."""
    from scripts.power_analysis import tost_equivalence

    rng = np.random.default_rng(7)
    base = rng.normal(0.0, 0.5, size=40)          # shared per-seed signal (CRN)
    a = base + rng.normal(0.0, 0.02, size=40)
    b = base + rng.normal(0.0, 0.02, size=40) + 0.01  # tiny offset, highly correlated with a

    paired = tost_equivalence(a, b, margin=0.30, n_boot=4000, rng=np.random.default_rng(1), paired=True)
    indep = tost_equivalence(a, b, margin=0.30, n_boot=4000, rng=np.random.default_rng(1), paired=False)

    # same point estimate (full-sample means), different CI construction
    assert abs(paired.estimate - indep.estimate) < 1e-12
    paired_width = paired.ci_high - paired.ci_low
    indep_width = indep.ci_high - indep.ci_low
    assert paired_width < indep_width  # pairing removes the shared-seed variance -> tighter CI
    assert paired.equivalent is True


def test_tost_paired_requires_equal_shapes() -> None:
    from scripts.power_analysis import tost_equivalence

    with pytest.raises(ValueError, match="same shape"):
        tost_equivalence(np.zeros(5), np.zeros(6), margin=0.1, paired=True)


# --------------------------------------------------------------------------- #
# MDE record shape
# --------------------------------------------------------------------------- #


def test_minimum_detectable_effect_record() -> None:
    from scripts.power_analysis import minimum_detectable_effect

    rec = minimum_detectable_effect(_cfg(), effect_points=13)
    assert set(rec) == {"mde", "mde_sigma_units", "grid", "power", "reached", "alpha_eff"}
    powers = np.asarray(rec["power"], dtype=float)
    assert powers.min() >= 0.0 and powers.max() <= 1.0
    assert np.asarray(rec["grid"]).shape == powers.shape


# --------------------------------------------------------------------------- #
# R37 (M4) — one-sided IUT mode vs the conservative Šidák-m sensitivity        #
# --------------------------------------------------------------------------- #
def test_iut_mode_alpha_eff_is_unpenalised_alpha() -> None:
    """In the live one-sided IUT mode the per-test alpha is alpha straight (NO Šidák penalty)."""
    from scripts.power_analysis import minimum_detectable_effect

    rec = minimum_detectable_effect(_cfg(iut_one_sided=True, seeds=10), effect_points=9)
    assert rec["alpha_eff"] == pytest.approx(0.05)
    rec_sidak = minimum_detectable_effect(_cfg(iut_one_sided=False, seeds=10), effect_points=9)
    assert rec_sidak["alpha_eff"] < 0.05  # the Šidák-over-6 penalty


def test_iut_mode_has_more_power_than_sidak_two_sided() -> None:
    """The one-sided IUT leg rejects more often than the two-sided Šidák-α_eff at the same effect.

    Two independent reasons: (i) one-sided halves the p (p_one = p_two/2 in-direction), (ii) the per-test
    alpha is 0.05 not the smaller Šidák-α_eff (~0.0085). Both raise power, so the IUT power dominates.
    """
    from scripts.power_analysis import simulate_power

    # n_boot large enough that the two-sided rule CAN reject below alpha_eff (~0.0085) — so the comparison
    # is meaningful, not a floor artefact.
    base = dict(seeds=12, sigma_dsr=0.30, n_sims=150, n_boot=999, seed=1)
    eff = 0.45
    p_iut = simulate_power(eff, _cfg(iut_one_sided=True, **base))
    p_sidak = simulate_power(eff, _cfg(iut_one_sided=False, **base))
    assert p_iut >= p_sidak


def test_iut_mde_not_larger_than_sidak_mde() -> None:
    """The PRIMARY one-sided IUT MDE is <= the conservative Šidák-m MDE (the doc's 'no larger' claim)."""
    from scripts.power_analysis import minimum_detectable_effect

    base = dict(seeds=12, sigma_dsr=0.30, n_sims=150, n_boot=999, seed=2)
    mde_iut = minimum_detectable_effect(_cfg(iut_one_sided=True, **base), effect_points=21)
    mde_sidak = minimum_detectable_effect(_cfg(iut_one_sided=False, **base), effect_points=21)
    # Both should be reachable at this effect grid / n; the IUT MDE is the smaller (or equal) effect.
    assert mde_iut["reached"]
    if mde_sidak["reached"]:
        assert float(mde_iut["mde"]) <= float(mde_sidak["mde"]) + 1e-9


def test_iut_null_rejection_near_one_sided_alpha() -> None:
    """At effect=0 the one-sided IUT rejects at ~alpha/2 (direction-gated), <= alpha — conservative/sized."""
    from scripts.power_analysis import simulate_power

    p0 = simulate_power(0.0, _cfg(iut_one_sided=True, seeds=12, n_sims=400, n_boot=999, seed=3))
    # Direction-gating + one-sided halving keep the null rejection at or below alpha (here ~0.025).
    assert p0 <= 0.08


def test_render_markdown_reports_both_primary_and_sidak(tmp_path) -> None:
    """The regenerated doc names the one-sided IUT primary AND keeps the Šidák-m=6 sensitivity."""
    from scripts.power_analysis import minimum_detectable_effect, render_markdown
    from dataclasses import replace

    cfg = _cfg(iut_one_sided=True, seeds=10, n_sims=80, n_boot=499)
    mde = minimum_detectable_effect(cfg, effect_points=9)
    mde_sidak = minimum_detectable_effect(replace(cfg, iut_one_sided=False), effect_points=9)
    from scripts.power_analysis import selection_aware_alpha

    md = render_markdown(
        cfg, mde, n_trials=210, rho_table="(table)",
        mde_sidak=mde_sidak, sidak_alpha=selection_aware_alpha(0.05, 6),
    )
    assert "ONE-SIDED IUT" in md
    assert "BH-over-6" in md or "Šidák-m" in md
    assert "live BH" in md or "Romano-Wolf" in md
    # The Šidák-m=6 figure must NOT be deleted — it appears as a sensitivity section.
    assert "sensitivity" in md.lower()


def test_doc_regenerates_via_main(tmp_path, monkeypatch) -> None:
    """`python scripts/power_analysis.py --mode iut_one_sided` writes a CAMPAIGN_power.md with both figures."""
    import sys

    from scripts import power_analysis as PA

    out = tmp_path / "CAMPAIGN_power.md"
    argv = [
        "power_analysis", "--mode", "iut_one_sided", "--n-sims", "60", "--n-boot", "399",
        "--effect-points", "9", "--out", str(out),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    PA.main()
    text = out.read_text(encoding="utf-8")
    assert "Decision rule" in text
    assert "ONE-SIDED IUT" in text
    assert "Šidák" in text  # the retained conservative sensitivity


# --------------------------------------------------------------------------- #
# T2.5 — Sharpe -> validation-DSR MDE reconciliation                           #
# --------------------------------------------------------------------------- #
def test_sharpe_mde_to_dsr_matches_closed_form_ceiling() -> None:
    """The Sharpe->DSR map equals the documented conservative ceiling phi(0)*sqrt(T-1)/sqrt(ppy)*ΔSR."""
    import math

    from scipy.stats import norm

    from scripts.power_analysis import (
        PERIODS_PER_YEAR,
        VALIDATION_TRACK_LENGTH,
        sharpe_mde_to_dsr,
    )

    sharpe_mde = 0.256
    got = sharpe_mde_to_dsr(sharpe_mde)
    expected = (
        float(norm.pdf(0.0))
        * math.sqrt(VALIDATION_TRACK_LENGTH - 1)
        / math.sqrt(PERIODS_PER_YEAR)
        * sharpe_mde
    )
    assert got == pytest.approx(expected, rel=1e-12)
    # At the campaign track length the ceiling is well above the 0.05 SESOI (the documented finding:
    # Sharpe and DSR scales differ), and clearly within (0, 1) as a probability-scale shift.
    assert 0.05 < got < 1.0


def test_sharpe_mde_to_dsr_monotone_and_scaling() -> None:
    """ΔDSR scales linearly in the Sharpe MDE and grows with the track length T (sqrt(T-1))."""
    from scripts.power_analysis import sharpe_mde_to_dsr

    # Linear in the Sharpe MDE.
    assert sharpe_mde_to_dsr(0.4) == pytest.approx(2.0 * sharpe_mde_to_dsr(0.2), rel=1e-12)
    # Increasing in T (more validation sessions -> a Sharpe gap is more "significant" in DSR terms).
    assert sharpe_mde_to_dsr(0.2, track_length=1200) > sharpe_mde_to_dsr(0.2, track_length=400)


def test_sharpe_mde_to_dsr_handles_unreached_mde() -> None:
    """A non-positive / non-finite Sharpe MDE (e.g. an unreached grid) maps to a 0.0 DSR shift, no raise."""
    import math

    from scripts.power_analysis import sharpe_mde_to_dsr

    assert sharpe_mde_to_dsr(0.0) == 0.0
    assert sharpe_mde_to_dsr(-1.0) == 0.0
    assert sharpe_mde_to_dsr(math.nan) == 0.0


def test_sharpe_mde_to_dsr_validates_inputs() -> None:
    """Degenerate track length / annualisation are rejected loudly (not silently mis-mapped)."""
    from scripts.power_analysis import sharpe_mde_to_dsr

    with pytest.raises(ValueError):
        sharpe_mde_to_dsr(0.2, track_length=1)
    with pytest.raises(ValueError):
        sharpe_mde_to_dsr(0.2, periods_per_year=0)


def test_render_markdown_reports_mde_in_dsr_units_and_inconclusive_branch(tmp_path) -> None:
    """The doc states the MDE in DSR units AND spells out the INCONCLUSIVE fallback branch (T2.5)."""
    from dataclasses import replace

    from scripts.power_analysis import (
        minimum_detectable_effect,
        render_markdown,
        selection_aware_alpha,
        sharpe_mde_to_dsr,
    )

    cfg = _cfg(iut_one_sided=True, seeds=10, n_sims=80, n_boot=499)
    mde = minimum_detectable_effect(cfg, effect_points=9)
    mde_sidak = minimum_detectable_effect(replace(cfg, iut_one_sided=False), effect_points=9)
    md = render_markdown(
        cfg, mde, n_trials=210, rho_table="(table)",
        mde_sidak=mde_sidak, sidak_alpha=selection_aware_alpha(0.05, 6),
    )
    # The reconciliation section + the MDE restated in validation-DSR units.
    assert "Sharpe ↔ validation-DSR reconciliation" in md
    assert "validation-DSR" in md
    assert "conservative ceiling" in md
    # The explicit INCONCLUSIVE branch narrative (the bankable-null bound condition).
    assert "INCONCLUSIVE" in md
    assert "in validation-DSR units" in md  # the TOST must be in DSR units to license equivalence
    # The reported DSR number is the closed-form ceiling of the reported Sharpe MDE.
    if mde["reached"]:
        ceil = sharpe_mde_to_dsr(float(mde["mde"]))
        assert f"{ceil:.3f} validation-DSR" in md


def test_main_prints_dsr_reconciliation(tmp_path, monkeypatch, capsys) -> None:
    """`main` prints the MDE in DSR units with the SESOI verdict (console reconciliation, T2.5)."""
    import sys

    from scripts import power_analysis as PA

    out = tmp_path / "CAMPAIGN_power.md"
    argv = [
        "power_analysis", "--mode", "iut_one_sided", "--n-sims", "60", "--n-boot", "399",
        "--effect-points", "9", "--out", str(out),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    PA.main()
    captured = capsys.readouterr().out
    assert "MDE in DSR units" in captured
    assert "validation-DSR" in captured
    # The generated doc carries the reconciliation too.
    text = out.read_text(encoding="utf-8")
    assert "Sharpe ↔ validation-DSR reconciliation" in text
