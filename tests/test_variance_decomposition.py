"""Fast behaviour tests for scripts/variance_decomposition.py (reviewer attack #10).

No torch, no real training. Two kinds of check:

  * **Recovery** — synthetic data with KNOWN variance components is fed to the one-way
    random-effects ANOVA and the σ²_search (between-search) / σ²_seed (within-seed) estimates
    are checked against the planted truth (Monte-Carlo-averaged so the test is not flaky), and
    the balanced divisor n0 == S / the unbalanced n0 formula is checked exactly.
  * **Graceful skip** — the K=1 (single search run), no-residual-d.f., empty-arm, records-only,
    and missing-contrast paths all return ``status="skipped"`` and NEVER raise.

Mirrors the ``tests/test_campaign_inference.py`` synthetic-record fixtures (sys.path insert +
``import variance_decomposition``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import variance_decomposition as VD  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _balanced_groups(
    k: int, s: int, sigma_between: float, sigma_within: float, rng: np.random.Generator
) -> list[np.ndarray]:
    """K groups of S scores from y_ks = a_k + e_ks, a_k~N(0,sigma_between^2), e~N(0,sigma_within^2)."""
    groups = []
    for _ in range(k):
        a = rng.normal(0.0, sigma_between)
        groups.append(a + rng.normal(0.0, sigma_within, size=s))
    return groups


def _test_record(arm: str, seed: int, vec: np.ndarray) -> dict:
    """A minimal per-(arm, seed) frozen-winner TEST record (matches the campaign schema)."""
    tr = [float(x) for x in vec]
    return {
        "run_id": f"{arm}-s{seed}",
        "arm": arm,
        "seed": seed,
        "fold": 0,
        "candidate_id": f"{arm}-winner",
        "generation": 0,
        "reward_source_hash": "deadbeef",
        "feedback_block": "",
        "metrics": {"val_fitness": 0.0, "test_returns": tr, "per_period_pnl": tr},
        "wall_clock": 0.0,
        "env_fingerprint": "test",
        "frozen": True,
        "test_returns": tr,
    }


# --------------------------------------------------------------------------- #
# one_way_random_effects — recovery of known components                         #
# --------------------------------------------------------------------------- #
def test_anova_recovers_known_components_monte_carlo() -> None:
    """Averaged over many planted datasets, the ANOVA estimates hit the true variance components.

    A single ANOVA estimate is noisy (few groups), so we average the estimator over many
    independent draws — the ANOVA/method-of-moments estimators are UNBIASED for both components,
    so the Monte-Carlo means must land near the planted truth.
    """
    rng = np.random.default_rng(0)
    sig_b, sig_w = 0.50, 1.00  # true between=0.25, within=1.00 (variances)
    k, s, reps = 8, 25, 1500
    betw = np.empty(reps)
    within = np.empty(reps)
    for i in range(reps):
        groups = _balanced_groups(k, s, sig_b, sig_w, rng)
        res = VD.one_way_random_effects(groups)
        assert res["status"] == "ok"
        # Use the PRE-truncation raw between-component for the unbiasedness check (truncation at 0
        # biases the mean upward; the raw estimator is the unbiased one).
        betw[i] = res["sigma2_between_raw"]
        within[i] = res["sigma2_within"]
    # Within (σ²_seed) is very tightly estimated (k*s residual d.f.).
    assert within.mean() == pytest.approx(sig_w**2, rel=0.05)
    # Between (σ²_search) has only k-1 d.f. -> looser tolerance, but unbiased.
    assert betw.mean() == pytest.approx(sig_b**2, rel=0.15)


def test_anova_balanced_n0_equals_group_size() -> None:
    """For a balanced design n0 == S exactly, and the EMS identity σ̂²_between=(MSB−MSW)/n0 holds."""
    rng = np.random.default_rng(1)
    groups = _balanced_groups(5, 20, 0.4, 0.9, rng)
    res = VD.one_way_random_effects(groups)
    assert res["n0"] == pytest.approx(20.0, abs=1e-9)
    # Reconstruct the component from the reported mean squares (exact algebra, not a re-estimate).
    expect = max(0.0, (res["ms_between"] - res["ms_within"]) / res["n0"])
    assert res["sigma2_between"] == pytest.approx(expect, abs=1e-12)


def test_anova_unbalanced_n0_matches_closed_form() -> None:
    """n0 = (1/(K-1))*(N - sum n_k^2 / N) for an UNBALANCED design (the Searle/Montgomery divisor)."""
    rng = np.random.default_rng(2)
    sizes = [10, 20, 30]
    groups = [rng.normal(0.0, 1.0, size=n) for n in sizes]
    res = VD.one_way_random_effects(groups)
    n_total = sum(sizes)
    n0_expected = (n_total - sum(n * n for n in sizes) / n_total) / (len(sizes) - 1)
    assert res["n0"] == pytest.approx(n0_expected, abs=1e-9)
    assert res["group_sizes"] == sizes


def test_anova_truncates_negative_between_to_zero() -> None:
    """When the true between variance is 0, MSB<MSW happens by chance -> σ̂²_between truncated to 0."""
    rng = np.random.default_rng(3)
    # sigma_between = 0 -> all groups share one mean; the raw estimate is often negative.
    saw_negative_raw = False
    for _ in range(50):
        groups = _balanced_groups(6, 30, 0.0, 1.0, rng)
        res = VD.one_way_random_effects(groups)
        assert res["sigma2_between"] >= 0.0  # truncated estimate is ALWAYS non-negative
        if res["sigma2_between_raw"] < 0.0:
            saw_negative_raw = True
            assert res["sigma2_between"] == 0.0  # the negative raw was truncated to exactly 0
    assert saw_negative_raw  # the truncation path was actually exercised


# --------------------------------------------------------------------------- #
# one_way_random_effects — graceful degradation                                 #
# --------------------------------------------------------------------------- #
def test_anova_single_group_skips_search_keeps_seed() -> None:
    """K=1: σ²_search unidentified (skipped) but σ²_seed (within) still computed from that group."""
    rng = np.random.default_rng(4)
    res = VD.one_way_random_effects([rng.normal(0.0, 1.0, size=30)])
    assert res["status"] == "skipped"
    assert res["K"] == 1
    assert res["sigma2_between"] is None
    assert res["sigma2_within"] is not None and res["sigma2_within"] > 0.0


def test_anova_all_single_point_groups_skip() -> None:
    """Every group a single seed (N==K): no residual d.f. -> σ²_seed undefined, skipped, no raise."""
    res = VD.one_way_random_effects([np.array([0.1]), np.array([0.2]), np.array([0.3])])
    assert res["status"] == "skipped"
    assert res["sigma2_within"] is None
    assert res["sigma2_between"] is None


def test_anova_empty_input_skips() -> None:
    assert VD.one_way_random_effects([])["status"] == "skipped"
    assert VD.one_way_random_effects([np.array([])])["status"] == "skipped"


# --------------------------------------------------------------------------- #
# market_bootstrap_variance                                                     #
# --------------------------------------------------------------------------- #
def test_market_bootstrap_variance_positive_and_scales_with_path_noise() -> None:
    """σ²_market is a positive variance and a NOISIER test path yields a larger σ²_market."""
    from src.inference.bootstrap import sharpe_ratio

    rng = np.random.default_rng(5)
    quiet = rng.standard_normal(800) * 0.005 + 0.0004
    noisy = rng.standard_normal(800) * 0.030 + 0.0004
    mq = VD.market_bootstrap_variance(
        quiet, statistic=sharpe_ratio, n_boot=400, rng=np.random.default_rng(7)
    )
    mn = VD.market_bootstrap_variance(
        noisy, statistic=sharpe_ratio, n_boot=400, rng=np.random.default_rng(7)
    )
    assert mq["status"] == "ok" and mn["status"] == "ok"
    assert mq["sigma2_market"] > 0.0
    assert np.isfinite(mn["sigma2_market"])
    assert mq["se_market"] == pytest.approx(np.sqrt(mq["sigma2_market"]), rel=1e-9)


def test_market_bootstrap_variance_short_path_skips() -> None:
    from src.inference.bootstrap import sharpe_ratio

    res = VD.market_bootstrap_variance(np.array([0.01]), statistic=sharpe_ratio, n_boot=50)
    assert res["status"] == "skipped" and res["sigma2_market"] is None


# --------------------------------------------------------------------------- #
# decompose — the full budget + verdict                                        #
# --------------------------------------------------------------------------- #
def test_decompose_verdict_gap_exceeds_when_search_noise_tiny() -> None:
    """A clear IQM gap with NEAR-ZERO between-search variance -> verdict 'exceeds √σ²_search'."""
    rng = np.random.default_rng(6)
    # distributional per-seed Sharpes high (~1.0), scalar low (~0.0); tiny between-run variance.
    dist_runs = [rng.normal(1.0, 0.05, 25) + rng.normal(0.0, 0.001) for _ in range(4)]
    scal_runs = [rng.normal(0.0, 0.05, 25) + rng.normal(0.0, 0.001) for _ in range(4)]
    res = VD.decompose({"distributional": dist_runs, "scalar": scal_runs}, n_boot=200)
    assert res["status"] == "ok"
    assert res["verdict"]["gap_exceeds_sqrt_sigma2_search"] is True
    assert res["iqm_gap"] == pytest.approx(1.0, abs=0.15)
    # σ²_seed (within) is the planted ~0.05^2 = 0.0025 for the distributional arm.
    assert res["components"]["distributional"]["seed"]["sigma2"] == pytest.approx(0.0025, rel=0.3)


def test_decompose_verdict_inconclusive_when_search_noise_large() -> None:
    """A small gap swamped by LARGE between-search variance -> verdict does NOT exceed."""
    rng = np.random.default_rng(8)
    # Big reward-draw (between-run) variance: each run's mean jumps by ~1.0; gap is tiny.
    dist_runs = [rng.normal(0.0, 0.05, 25) + rng.normal(0.0, 1.0) for _ in range(5)]
    scal_runs = [rng.normal(0.0, 0.05, 25) + rng.normal(0.0, 1.0) for _ in range(5)]
    res = VD.decompose({"distributional": dist_runs, "scalar": scal_runs}, n_boot=150)
    assert res["status"] == "ok"
    # √σ²_search is ~1.0 while the gap is ~0 -> cannot clear it.
    assert res["verdict"]["gap_exceeds_sqrt_sigma2_search"] is False


def test_decompose_k1_degrades_verdict_skipped() -> None:
    """K=1 per arm -> σ²_search unidentified -> verdict skipped, σ²_seed still present, no crash."""
    rng = np.random.default_rng(9)
    res = VD.decompose(
        {"distributional": [rng.normal(1.0, 0.05, 30)], "scalar": [rng.normal(0.0, 0.05, 30)]},
        n_boot=100,
    )
    assert res["status"] == "skipped"
    assert res["verdict"]["status"] == "skipped"
    assert res["components"]["distributional"]["seed"]["sigma2"] is not None
    assert res["components"]["distributional"]["search"]["status"] == "skipped"


def test_decompose_missing_contrast_arm_skips_not_raises() -> None:
    rng = np.random.default_rng(10)
    res = VD.decompose({"distributional": [rng.normal(1.0, 0.05, 25) for _ in range(3)]}, n_boot=80)
    assert res["status"] == "skipped"
    assert res["verdict"]["status"] == "skipped"


def test_decompose_market_path_threads_into_components() -> None:
    """When market_paths are supplied, σ²_market is populated per arm (block-bootstrap of the path)."""
    rng = np.random.default_rng(11)
    dist_runs = [rng.normal(1.0, 0.05, 20) for _ in range(3)]
    scal_runs = [rng.normal(0.0, 0.05, 20) for _ in range(3)]
    paths = {
        "distributional": rng.standard_normal(400) * 0.01 + 0.0006,
        "scalar": rng.standard_normal(400) * 0.01 + 0.0001,
    }
    res = VD.decompose(
        {"distributional": dist_runs, "scalar": scal_runs},
        market_paths=paths,
        n_boot=200,
    )
    assert res["components"]["distributional"]["market"]["sigma2"] is not None
    assert res["components"]["distributional"]["market"]["sigma2"] > 0.0
    assert res["components"]["scalar"]["market"]["status"] == "ok"


# --------------------------------------------------------------------------- #
# decompose_from_campaign — the records driver (graceful)                       #
# --------------------------------------------------------------------------- #
def test_driver_single_archive_is_k1_skipped_but_reports_seed() -> None:
    """One campaign archive (K=1) -> σ²_search skipped, σ²_seed + σ²_market reported, no crash."""
    rng = np.random.default_rng(12)
    recs = []
    for s in range(12):
        recs.append(_test_record("distributional", s, rng.standard_normal(300) * 0.01 + 0.0006))
        recs.append(_test_record("scalar", s, rng.standard_normal(300) * 0.01 + 0.0001))
    res = VD.decompose_from_campaign([recs], n_boot=120)
    assert res["n_runs"] == 1
    assert res["components"]["distributional"]["search"]["status"] == "skipped"
    assert res["components"]["distributional"]["seed"]["sigma2"] is not None
    # σ²_market built from the seed-median representative path.
    assert res["components"]["distributional"]["market"]["sigma2"] is not None


def test_driver_k3_identifies_search_component() -> None:
    """Three independent search archives -> K=3 -> σ²_search identified (status ok), verdict computed."""
    recs_per_run = []
    for run in range(3):
        rng = np.random.default_rng(100 + run)
        # Each run's winner has a slightly different mean (the reward-draw effect) -> nonzero σ²_search.
        dist_mu = 0.0008 + run * 0.0002
        recs = []
        for s in range(10):
            recs.append(_test_record("distributional", s, rng.standard_normal(300) * 0.01 + dist_mu))
            recs.append(_test_record("scalar", s, rng.standard_normal(300) * 0.01 + 0.0001))
        recs_per_run.append(recs)
    res = VD.decompose_from_campaign(recs_per_run, n_boot=120)
    assert res["n_runs"] == 3
    assert res["components"]["distributional"]["search"]["status"] == "ok"
    assert res["components"]["distributional"]["search"]["K"] == 3
    assert res["verdict"]["status"] == "ok"
    assert isinstance(res["verdict"]["gap_exceeds_sqrt_sigma2_search"], bool)


def test_driver_records_only_no_test_vectors_skips() -> None:
    """Records with NO test_returns (search-only candidates) -> empty score table -> skipped, no crash."""
    search_only = [
        {
            "run_id": f"distributional-c{j}",
            "arm": "distributional",
            "seed": 0,
            "fold": 0,
            "candidate_id": f"c{j}",
            "generation": 0,
            "reward_source_hash": "x",
            "feedback_block": "",
            "metrics": {"val_fitness": 0.1, "val_returns": [0.01, 0.02, 0.03]},
            "wall_clock": 0.0,
            "env_fingerprint": "t",
        }
        for j in range(5)
    ]
    res = VD.decompose_from_campaign([search_only], n_boot=50)
    assert res["status"] == "skipped"
    assert res["components"]["distributional"]["seed"]["sigma2"] is None
    assert res["components"]["distributional"]["search"]["status"] == "skipped"


def test_driver_empty_runs_skips() -> None:
    """No runs at all -> fully skipped, never raises."""
    res = VD.decompose_from_campaign([], n_boot=20)
    assert res["status"] == "skipped"
    assert res["n_runs"] == 0


# --------------------------------------------------------------------------- #
# rendering                                                                    #
# --------------------------------------------------------------------------- #
def test_verdict_markdown_renders_table_and_verdict() -> None:
    rng = np.random.default_rng(13)
    dist_runs = [rng.normal(1.0, 0.05, 20) for _ in range(3)]
    scal_runs = [rng.normal(0.0, 0.05, 20) for _ in range(3)]
    res = VD.decompose({"distributional": dist_runs, "scalar": scal_runs}, n_boot=120)
    md = VD.verdict_markdown(res)
    assert "Variance decomposition" in md
    assert "σ²_seed" in md and "σ²_search" in md and "σ²_market" in md
    assert "Verdict" in md
    assert "distributional>scalar" in md


def test_verdict_markdown_skipped_path_is_safe() -> None:
    """The markdown renderer handles a fully-skipped result without raising / NaN formatting."""
    res = VD.decompose_from_campaign([], n_boot=20)
    md = VD.verdict_markdown(res)
    assert "Verdict" in md and "not computed" in md
