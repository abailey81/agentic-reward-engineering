"""License-clean cross-check oracles for the custom inference (audit 2026-06-19).

Validates the bespoke implementations against independent, OSI-licensed references that are
ALREADY project dependencies (statsmodels BSD-3, arch NCSA/3-clause-BSD-like) -- the "our numbers
reproduce the canonical implementation" robustness footnote the inference audit asked for. mlfinlab is
proprietary and quantstats' PSR is non-canonical (both rejected as oracles); statsmodels' BH and arch's
StationaryBootstrap are the clean ones.

Oracles:
  * Benjamini-Hochberg FDR  <- statsmodels.stats.multitest.multipletests(method='fdr_bh')  [EXACT]
  * stationary block bootstrap  <- arch.bootstrap.StationaryBootstrap / optimal_block_length  [LOOSE]
    Sanity-checks src/inference/bootstrap.py::stationary_bootstrap_indices (the resampler under
    sharpe_difference_test / cvar_difference_test) against arch's canonical Politis-Romano (1994)
    implementation: percentile CI and bootstrap SE of a statistic must agree to loose tolerance.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.inference.bootstrap import stationary_bootstrap_indices
from src.inference.multiple_testing import benjamini_hochberg


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_benjamini_hochberg_matches_statsmodels(seed: int) -> None:
    """Our BH rejection set must equal statsmodels.multipletests(method='fdr_bh') exactly."""
    sm = pytest.importorskip("statsmodels.stats.multitest")
    rng = np.random.default_rng(seed)
    # A mix of small (signal) + uniform (null) p-values across a realistic family size.
    pvals = np.concatenate([rng.uniform(0.0, 0.02, 5), rng.uniform(0.0, 1.0, 20)])
    for q in (0.05, 0.1, 0.2):
        ours = benjamini_hochberg(pvals, q)
        theirs = sm.multipletests(pvals, alpha=q, method="fdr_bh")[0]
        assert np.array_equal(ours, theirs), f"BH mismatch at q={q}, seed={seed}"


def test_benjamini_hochberg_edge_cases() -> None:
    """All-null -> no rejections; all-significant -> all rejected; both agree with statsmodels."""
    sm = pytest.importorskip("statsmodels.stats.multitest")
    p_null = np.array([0.5, 0.6, 0.9, 0.99])
    assert not benjamini_hochberg(p_null, 0.1).any()
    assert np.array_equal(
        benjamini_hochberg(p_null, 0.1), sm.multipletests(p_null, alpha=0.1, method="fdr_bh")[0]
    )
    p_sig = np.array([1e-6, 1e-5, 1e-4])
    assert benjamini_hochberg(p_sig, 0.1).all()


# ---------------------------------------------------------------------------
# Stationary block bootstrap <- arch.bootstrap.StationaryBootstrap (LOOSE oracle)
# ---------------------------------------------------------------------------
def _bespoke_boot_means(x: np.ndarray, p: float, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap distribution of mean(x) via the bespoke stationary resampler.

    Uses the same ``stationary_bootstrap_indices`` path machinery that backs
    ``sharpe_difference_test`` / ``cvar_difference_test``; the mean is the simplest
    statistic on which arch exposes a directly comparable CI/cov.
    """
    n = x.size
    out = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        out[i] = x[stationary_bootstrap_indices(n, p, rng)].mean()
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_stationary_bootstrap_se_matches_arch(seed: int) -> None:
    """Bespoke bootstrap SE of the mean must match arch's StationaryBootstrap.cov (loose)."""
    arch_bs = pytest.importorskip("arch.bootstrap")
    rng = np.random.default_rng(seed)
    # AR(1) so there is genuine serial dependence for the block bootstrap to capture.
    n, phi = 800, 0.4
    eps = 0.01 * rng.standard_normal(n)
    x = np.empty(n)
    x[0] = eps[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    x = x + 0.0005  # small positive drift

    p = 0.1                      # expected block length 1/p = 10, matching the code default
    block_size = 1.0 / p
    n_boot = 3000

    ours = _bespoke_boot_means(x, p, n_boot, np.random.default_rng(seed + 100))
    se_ours = float(ours.std(ddof=1))

    bs = arch_bs.StationaryBootstrap(block_size, x, seed=np.random.default_rng(seed + 200))
    cov = bs.cov(lambda a: np.array([a.mean()]), reps=n_boot)
    se_arch = float(np.sqrt(cov.ravel()[0]))

    # Loose sanity tolerance: two independent stationary-bootstrap implementations on the
    # same series and matched expected block length should agree on the SE within ~30%.
    assert se_ours == pytest.approx(se_arch, rel=0.30), f"SE mismatch: ours={se_ours} arch={se_arch}"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_stationary_bootstrap_ci_matches_arch(seed: int) -> None:
    """Bespoke percentile CI of the mean must bracket-agree with arch's basic-bootstrap CI (loose)."""
    arch_bs = pytest.importorskip("arch.bootstrap")
    rng = np.random.default_rng(seed)
    n, phi = 800, 0.4
    eps = 0.01 * rng.standard_normal(n)
    x = np.empty(n)
    x[0] = eps[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    x = x + 0.0005

    p = 0.1
    block_size = 1.0 / p
    n_boot = 3000

    ours = _bespoke_boot_means(x, p, n_boot, np.random.default_rng(seed + 100))
    lo_ours, hi_ours = np.quantile(ours, [0.025, 0.975])
    half_width = 0.5 * (hi_ours - lo_ours)

    bs = arch_bs.StationaryBootstrap(block_size, x, seed=np.random.default_rng(seed + 200))
    ci = bs.conf_int(lambda a: np.array([a.mean()]), reps=n_boot, method="percentile", size=0.95)
    lo_arch, hi_arch = float(ci[0, 0]), float(ci[1, 0])

    # Endpoints agree to within ~half the bespoke CI half-width (loose: these are random,
    # finite-replication CIs from two different RNG streams).
    tol = 0.5 * half_width
    assert lo_ours == pytest.approx(lo_arch, abs=tol), f"CI low mismatch: {lo_ours} vs {lo_arch}"
    assert hi_ours == pytest.approx(hi_arch, abs=tol), f"CI high mismatch: {hi_ours} vs {hi_arch}"


def test_optimal_block_length_recovers_dependence() -> None:
    """arch.optimal_block_length sanity: a strongly AR(1) series wants a longer block than ~iid noise.

    Anchors the code's fixed p=0.1 (expected block length 10) as a reasonable order of magnitude for
    autocorrelated daily P&L rather than an arbitrary constant.
    """
    arch_bs = pytest.importorskip("arch.bootstrap")
    rng = np.random.default_rng(0)
    n = 2000
    eps = rng.standard_normal(n)
    iid = eps.copy()
    ar = np.empty(n)
    ar[0] = eps[0]
    for t in range(1, n):
        ar[t] = 0.7 * ar[t - 1] + eps[t]

    bl_iid = float(arch_bs.optimal_block_length(iid)["stationary"].iloc[0])
    bl_ar = float(arch_bs.optimal_block_length(ar)["stationary"].iloc[0])
    assert bl_ar > bl_iid  # dependence => longer optimal block
    assert bl_iid > 0.0


# --------------------------------------------------------------------------- #
# rliable oracle: the per-seed (Agarwal et al. 2021) inference primitives      #
# --------------------------------------------------------------------------- #
def test_iqm_matches_rliable_oracle() -> None:
    """Our IQM (bootstrap.iqm AND reporting.iqm) == the canonical rliable.metrics.aggregate_iqm. The
    per-seed rliable inference is the HEADLINE H2 unit, so its central statistic is oracle-validated."""
    pytest.importorskip("rliable")
    from rliable import metrics as rm

    from src.inference.bootstrap import iqm as b_iqm
    from src.inference.reporting import iqm as r_iqm

    rng = np.random.default_rng(7)
    for _ in range(25):
        x = rng.standard_normal(int(rng.integers(8, 250)))
        oracle = float(rm.aggregate_iqm(x.reshape(-1, 1)))
        assert b_iqm(x) == pytest.approx(oracle, abs=1e-9)
        assert r_iqm(x) == pytest.approx(oracle, abs=1e-9)


def test_probability_of_improvement_matches_rliable_oracle() -> None:
    """Our reporting.probability_of_improvement == rliable.metrics.probability_of_improvement (the
    rliable probability-of-improvement used for the per-seed arm contrast)."""
    pytest.importorskip("rliable")
    from rliable import metrics as rm

    from src.inference.reporting import probability_of_improvement as my_poi

    rng = np.random.default_rng(11)
    for _ in range(15):
        a = rng.standard_normal(int(rng.integers(5, 40))) + rng.uniform(-0.3, 0.3)
        b = rng.standard_normal(int(rng.integers(5, 40)))
        oracle = float(rm.probability_of_improvement(a.reshape(-1, 1), b.reshape(-1, 1)))
        assert my_poi(a, b) == pytest.approx(oracle, abs=1e-9)


def test_performance_profile_matches_rliable_oracle() -> None:
    """Our reporting.performance_profile point curve == rliable.library.create_performance_profile
    (the run-score distribution / performance profile of Agarwal et al. 2021)."""
    pytest.importorskip("rliable")
    from rliable import library as rly

    from src.inference.reporting import performance_profile as my_pp

    rng = np.random.default_rng(23)
    for _ in range(15):
        x = rng.standard_normal(int(rng.integers(8, 120)))
        taus = np.linspace(x.min() - 0.5, x.max() + 0.5, 30)
        mine, _, _ = my_pp(x, taus, n_boot=0)
        oracle_dict, _ = rly.create_performance_profile({"a": x.reshape(-1, 1)}, taus)
        np.testing.assert_allclose(mine, np.asarray(oracle_dict["a"]), atol=1e-9)
