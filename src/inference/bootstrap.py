"""Stationary-bootstrap inference for dependent return series (FINAL_PLAN F.11).

Purpose
-------
Provide block-resampling machinery and paired performance-difference tests that
remain valid under the serial correlation present in daily portfolio returns.
These tests back the headline pairwise comparisons between arms (distributional
vs scalar feedback) and feed the multiple-testing family in
``src.inference.multiple_testing``.

Why the stationary bootstrap
----------------------------
Daily P&L is autocorrelated (volatility clustering, momentum/mean-reversion),
so the iid bootstrap and naive t-tests understate sampling error. The
Politis-Romano (1994) stationary bootstrap resamples *blocks* of random,
geometrically distributed length with wrap-around, which preserves the
short-range dependence structure while keeping the resampled series stationary.

Algorithm (stationary bootstrap, index form)
---------------------------------------------
1. Draw a starting index uniformly in ``[0, n)``.
2. With probability ``p`` start a fresh block (draw a new uniform start); with
   probability ``1 - p`` continue the current block by stepping the index
   forward by one (mod ``n`` -- wrap-around).
3. Expected block length is ``1 / p``; block lengths are ``Geometric(p)``.
4. Repeat until ``n`` indices are produced -- that is one bootstrap path.

Difference tests
----------------
``sharpe_difference_test`` implements a *re-centred basic (empirical) stationary
block-bootstrap* test for ``H0: SR(a) - SR(b) = 0``: the two-sided p-value is the
bootstrap probability that the null-recentred difference ``boot - obs`` exceeds the
observed difference ``obs`` in absolute value. The test does **not** studentize:
although a bootstrap standard error ``se`` is computed and reported (as a scaled
``stat = obs / se``), it *cancels exactly* in the decision rule
``|(boot - obs) / se| >= |obs / se|`` (the ``se`` divides both sides), so the
p-value is identical to the un-studentized empirical-bootstrap p-value
``(|boot - obs| >= |obs|).mean()``. This is the *stationary* bootstrap of
Politis & Romano (**1994**, random geometric block lengths), chosen on its own
merits for autocorrelated returns. We deliberately do **not** claim the Ledoit &
Wolf (2008) studentized construction — LW use the *circular* block bootstrap of
Politis & Romano (**1992**, fixed block size) *and* a studentization that does not
cancel; our size is MEASURED empirically by ``null_calibration`` (audit C-7)
rather than by appeal to that result — measured, and mildly anti-conservative;
see the warning block below for the numbers.

``cvar_difference_test`` applies the analogous re-centred stationary-bootstrap
construction to the difference in CVaR (expected shortfall) at level ``alpha``.
No published difference-in-CVaR test was located, so this is a *bespoke*
extension whose size is likewise MEASURED empirically by ``null_calibration``
(audit C-7) rather than by a citation — see the warning block below.

``null_calibration`` repeatedly applies a difference test under a true null and
reports the empirical rejection rate at the 5% level together with the raw
p-values (audit C-7).

NOTE: WHAT "CERTIFIED" DOES AND DOES NOT MEAN — MEASURED 2026-07-26 (deep review).
The word "certified" above and at the call sites below was an overstatement, and is
corrected here rather than left to a referee. The size was measured directly at
PRODUCTION settings (``paired_seed_difference_test``, ``statistic=iqm``, ``n_boot=2000``,
n=30 paired seeds, 6,000 replications under an exchangeable Gaussian null; Monte-Carlo
SE at nominal = 0.0028)::

    two-sided rejection at alpha=0.05 : 0.0573
    one-sided rejection at alpha=0.05 : 0.0613

So the test is **mildly ANTI-conservative — about 2.6 (two-sided) and 4.0 (one-sided)
Monte-Carlo SEs ABOVE nominal — not "correctly sized" and not "<= alpha"**. The drift is
the re-centred basic bootstrap's own approximation error at n=30, not an IQM artefact
(it persists with ``statistic=np.mean`` and at n=100). It is small in absolute terms, but
each IUT leg is decided at alpha=0.05, so the joint IUT inherits roughly the same drift,
and every claim in the register that the simulated null rejection is "conservative" points
the WRONG WAY. Report the measured value; do not claim exactness.

Note also that the fast in-suite guards in ``tests/test_inference.py`` are SMOKE bounds at
n_reps=160 (where the Monte-Carlo SE is about 0.019), not size certifications: a bound of
0.12-0.15 sits several SEs above nominal and cannot realistically fail. They catch a
grossly broken test, which is what a fast test should do — the number above is the
measurement.

FINAL_PLAN refs
---------------
- F.11 (selection-aware inference stack): block bootstrap + paired tests.
- audit C-7: null-calibration certification of the test machinery.
"""

from __future__ import annotations


import math
from typing import Callable

import numpy as np

__all__ = [
    "stationary_bootstrap_indices",
    "sharpe_ratio",
    "cvar",
    "iqm",
    "sharpe_difference_test",
    "cvar_difference_test",
    "paired_seed_difference_test",
    "null_calibration",
]


def iqm(x: np.ndarray) -> float:
    """Interquartile mean (rliable; Agarwal et al. 2021, NeurIPS) of per-seed scores.

    The mean of the middle 50% of the values (trim 25% from each tail) — a robust central tendency
    for an RL evaluation's per-seed scores, insensitive to a few lucky/unlucky training seeds and far
    more reliable than the raw mean across the small seed counts typical of deep-RL studies. Falls
    back to the plain mean for fewer than 4 values, where the interquartile trim is ill-defined.

    Returns NaN only for an all-non-finite / empty input.
    """
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    if a.size < 4:
        return float(a.mean())
    a = np.sort(a)
    lo = a.size // 4  # == int(0.25*n) (matches scipy.stats.trim_mean(x, 0.25))
    return float(a[lo : a.size - lo].mean())


def _iqm_rows(m: np.ndarray) -> np.ndarray:
    """Row-wise :func:`iqm` for an ALL-FINITE 2-D array (the vectorized bootstrap fast path).

    Mirrors :func:`iqm` exactly for finite rows: ``n < 4`` -> plain row mean; else per-row sort and
    mean of the central ``[n//4 : n - n//4]`` band (trim 25% each tail). Callers must guarantee
    finiteness — that makes iqm's per-row finite filter a no-op, which is what keeps this
    bit-identical to calling ``iqm`` on each row (same values, same contiguous-axis reductions).
    """
    n = int(m.shape[1])
    if n == 0:
        return np.full(m.shape[0], np.nan)
    if n < 4:
        return m.mean(axis=1)
    s = np.sort(m, axis=1)
    lo = n // 4
    return s[:, lo : n - lo].mean(axis=1)


def paired_seed_difference_test(
    a: np.ndarray,
    b: np.ndarray,
    *,
    statistic: Callable[[np.ndarray], float] = iqm,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
    return_draws: bool = False,
) -> dict[str, float | np.ndarray]:
    """Re-centred bootstrap test for ``H0: statistic(a) - statistic(b) = 0`` over PAIRED per-seed scores.

    ``a`` and ``b`` are per-seed scores (e.g. each arm's per-seed Sharpe or CVaR) for two arms, PAIRED
    element-wise by the shared training seed. The bootstrap resamples SEED INDICES i.i.d. with
    replacement (seeds are independent), applies the SAME draw to both arms (paired, so the seed-level
    common variance cancels), and recomputes ``statistic(a) - statistic(b)``. This carries the
    ACROSS-SEED (training-RNG) variance — the dominant, relevant uncertainty in a multi-seed RL
    evaluation (rliable; Agarwal et al. 2021) — which a per-period mean-over-seeds DESTROYS (averaging
    N i.i.d.-seed paths shrinks the tested object's variance ~N×, so a per-period bootstrap on the
    averaged series is anti-conservative by ~√N). The two-sided p-value uses the SAME re-centred basic
    empirical-bootstrap convention as :func:`sharpe_difference_test` — the bootstrap probability that
    ``|boot - obs| >= |obs|`` — so :func:`null_calibration` MEASURES its size identically (audit C-7).
    NOTE: Measured, not certified: at production settings this test rejects at **0.0573 two-sided /
    0.0613 one-sided** against nominal 0.05 (mildly ANTI-conservative — see the module docstring for
    the full measurement). Do not describe it as correctly sized or conservative.

    Parameters
    ----------
    a, b : np.ndarray
        Paired per-seed scores (same length; element i is the same training seed for both arms).
    statistic : callable, optional
        Central-tendency functional applied to a score array; default the rliable IQM (:func:`iqm`).
    n_boot : int
        Bootstrap replications.
    rng : numpy.random.Generator, optional
        Seeded for reproducibility.
    return_draws : bool
        Opt-in: also return the raw bootstrap draws of the effect under ``"draws"`` (an
        ``np.ndarray`` — the ONE non-float value). Default off so every existing consumer's
        float-only dict shape is unchanged; use it when a NON-95% interval must be derived from the
        SAME draws (e.g. the TOST-convention 90% equivalence flag in
        ``src.inference.contamination.named_vs_blinded_oos_gap``) instead of re-running the bootstrap.

    Returns
    -------
    dict
        ``{"stat", "pvalue", "pvalue_one_sided_greater", "effect", "ci_low", "ci_high"}`` (plus
        ``"draws"`` iff ``return_draws``) where ``effect`` is the raw ``statistic(a) - statistic(b)``
        (positive = a better); ``pvalue`` is the two-sided re-centred p; ``pvalue_one_sided_greater``
        is the DIRECT upper-tail one-sided p for H1: effect>0 (``P(boot - obs >= obs)``), valid under
        a skewed bootstrap — use this for a one-sided decision rather than ``pvalue/2`` (R64);
        ``stat`` is the effect scaled by the bootstrap SE (reported only; cancels in the decision
        rule); the CI is the 95% percentile interval.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape (paired per-seed scores)")
    n = a.size
    if n < 2:
        raise ValueError("paired_seed_difference_test needs >= 2 paired seeds")
    if rng is None:
        rng = np.random.default_rng(0)  # P24: deterministic fallback (reported paths pass an explicit rng)

    obs = statistic(a) - statistic(b)
    # One (n_boot, n) index matrix (batch-5 M2, 2026-07-03): numpy fills a C-order array element-by-
    # element from the bit generator, so row i consumes EXACTLY the words the old per-iteration
    # ``rng.integers(0, n, size=n)`` drew — the bootstrap draws (hence p-values/CIs) are byte-identical
    # to the pre-vectorization loop on the same seeded rng.
    idx = rng.integers(0, n, size=(int(n_boot), n))
    if statistic is iqm and bool(np.isfinite(a).all() and np.isfinite(b).all()):
        # Vectorized IQM fast path: the pure-Python resample loop (2 iqm calls x n_boot per test) made
        # the power simulator's ~900k invocations of this function a multi-DAY run. With all-finite
        # inputs every resampled row is all-finite, so the row-wise mirror is exact — bit-identical to
        # per-row iqm() (pinned by test_vectorized_iqm_fast_path_is_bit_identical_to_reference_loop).
        # Non-iqm statistics and non-finite inputs take the reference loop below (old behavior).
        boot = _iqm_rows(a[idx]) - _iqm_rows(b[idx])
    else:
        boot = np.empty(int(n_boot), dtype=float)
        for i in range(int(n_boot)):
            boot[i] = statistic(a[idx[i]]) - statistic(b[idx[i]])

    se = boot.std(ddof=1)
    se = se if (np.isfinite(se) and se > 0.0) else float("nan")
    stat = obs / se if np.isfinite(se) else 0.0
    # Re-centred basic empirical bootstrap (se cancels): two-sided p = P(|boot - obs| >= |obs|).
    pvalue = float((np.abs(boot - obs) >= abs(obs)).mean()) if np.isfinite(se) else 1.0
    pvalue = min(1.0, max(pvalue, 1.0 / (n_boot + 1)))  # finite-resolution guard
    # Direct one-sided p for the PRE-SPECIFIED alternative H1: effect > 0 (``a`` better than ``b``) —
    # the upper-tail null probability P(boot - obs >= obs), read straight off the bootstrap draws. This is
    # VALID under ANY skew of the bootstrap (a CVaR/ES difference need not be symmetric): halving the
    # symmetric two-sided p (the old analysis-layer ``p_two/2``) equals this ONLY when ``boot - obs`` is
    # symmetric, and otherwise mis-states the one-sided tail — anti-conservative when the UPPER tail (beyond
    # +obs) is the heavier one, conservative when the lower tail is — decision-flipping at alpha on the
    # co-primary CVaR-5% leg (R64 / DEEP_AUDIT 2026-06-28). The IUT caller still gates on ``effect > 0``, so
    # a wrong-direction estimate (where this tail is large) cannot reject regardless.
    if np.isfinite(se):
        p_one = float(((boot - obs) >= obs).mean())
        p_one = min(1.0, max(p_one, 1.0 / (n_boot + 1)))  # finite-resolution guard
    else:
        p_one = 1.0
    out: dict[str, float | np.ndarray] = {
        "stat": float(stat),
        "pvalue": pvalue,
        "pvalue_one_sided_greater": p_one,
        "effect": float(obs),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
    }
    if return_draws:
        out["draws"] = boot
    return out


def stationary_bootstrap_indices(
    n: int,
    p: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Politis-Romano stationary-bootstrap index path of length ``n``.

    Parameters
    ----------
    n:
        Length of the series / number of indices to generate.
    p:
        Geometric block-restart probability in ``(0, 1]``; expected block
        length is ``1 / p``. With ``p == 1`` every step restarts, giving an iid
        bootstrap; small ``p`` produces long contiguous blocks.
    rng:
        Optional NumPy generator for reproducibility.

    Returns
    -------
    numpy.ndarray
        Integer array of shape ``(n,)`` with values in ``[0, n)``: a single
        stationary-bootstrap resampling path with wrap-around.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not (0.0 < p <= 1.0):
        raise ValueError("p must lie in (0, 1]")
    if rng is None:
        rng = np.random.default_rng(0)  # P24: deterministic fallback (reported paths pass an explicit rng)

    idx = np.empty(n, dtype=np.intp)
    # Pre-draw the restart decisions and candidate fresh starts for speed.
    restart = rng.random(n) < p
    fresh = rng.integers(0, n, size=n)

    cur = int(fresh[0])  # always start a fresh block at position 0
    idx[0] = cur
    for t in range(1, n):
        if restart[t]:
            cur = int(fresh[t])
        else:
            cur = (cur + 1) % n
        idx[t] = cur
    return idx


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio of a per-period return series.

    Parameters
    ----------
    returns:
        One-dimensional array of per-period (e.g. daily) returns.
    periods_per_year:
        Number of periods per year used for annualization (default ``252``).

    Returns
    -------
    float
        ``mean / std * sqrt(periods_per_year)`` using the population (ddof=0)
        standard deviation. Returns ``0.0`` if the standard deviation is zero.

    Notes
    -----
    The ``sqrt(periods_per_year)`` time-aggregation assumes i.i.d. returns; under serial
    correlation it biases the annualised POINT estimate (Lo 2002, *FAJ* — the correct factor is
    ``sqrt(q) / sqrt(q + 2*sum_{k}(q-k)*rho_k)``). This is a known reporting caveat, NOT a flaw in
    the headline test: the H2 inference is a per-seed paired bootstrap over the actual per-period
    return series (``paired_seed_difference_test``), which carries the true sampling variance directly
    and never relies on this annualised scalar. The annualised Sharpe is reported as a descriptive
    point estimate only; a Lo-2002-corrected variant would change reported numbers (pre-reg-relevant).
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]  # strip non-finite from a pathological reward (P0-1)
    if r.size == 0:
        return 0.0
    mu = float(r.mean())
    sd = float(r.std(ddof=0))
    # Relative near-zero guard: a (near-)constant series leaves float residue in sd,
    # so an exact `sd == 0` test silently fails (P0-1, audit 2026-06-19).
    if not np.isfinite(sd) or sd <= 1e-12 * (abs(mu) + 1e-12) or np.ptp(r) == 0.0:
        return 0.0
    return float(mu / sd * math.sqrt(periods_per_year))


def cvar(returns: np.ndarray, alpha: float) -> float:
    """Conditional Value-at-Risk (expected shortfall) of the lower tail.

    Parameters
    ----------
    returns:
        One-dimensional array of returns.
    alpha:
        Tail level in ``(0, 1]`` (e.g. ``0.05`` for the 5% expected shortfall).

    Returns
    -------
    float
        The mean of the worst ``ceil(alpha * T)`` returns (the lower tail).
        For loss-making tails this is negative; smaller ``alpha`` selects a
        more extreme (more negative) tail.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must lie in (0, 1]")
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]  # strip non-finite so a NaN can't poison the tail mean (P0-1)
    t = r.size
    if t == 0:
        return float("nan")
    k = max(1, int(math.ceil(alpha * t)))
    # The k smallest (worst) returns.
    worst = np.partition(r, k - 1)[:k]
    return float(worst.mean())


def _bootstrap_statistic_distribution(
    a: np.ndarray,
    b: np.ndarray,
    stat_fn: Callable[[np.ndarray, np.ndarray], float],
    p: float,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bootstrap distribution of ``stat_fn(a, b)`` under matched index resampling.

    Both series are resampled with the *same* stationary-bootstrap index path so
    that any contemporaneous dependence between ``a`` and ``b`` is preserved.
    """
    n = a.size
    out = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = stationary_bootstrap_indices(n, p, rng)
        out[i] = stat_fn(a[idx], b[idx])
    return out


def sharpe_difference_test(
    a: np.ndarray,
    b: np.ndarray,
    p: float = 0.1,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Re-centred stationary-bootstrap test for ``H0: SR(a) - SR(b) = 0``.

    Parameters
    ----------
    a, b:
        Aligned per-period return series for the two strategies.
    p:
        Stationary-bootstrap block-restart probability.
    n_boot:
        Number of bootstrap replications.
    rng:
        Optional NumPy generator for reproducibility.

    Returns
    -------
    dict
        ``{"stat", "pvalue", "ci_low", "ci_high"}`` where ``stat`` is the
        observed difference scaled by the bootstrap standard error (a reported
        effect-size summary that does **not** drive the decision — see Notes),
        ``pvalue`` is the two-sided bootstrap p-value, and ``ci_low``/``ci_high``
        form a 95% percentile bootstrap CI for the raw Sharpe difference.

    Notes
    -----
    Despite the reported ``stat = obs / se``, this is **not** a studentized test:
    the bootstrap standard error ``se`` cancels in the decision rule
    ``|(boot - obs) / se| >= |obs / se|``, so the two-sided p-value equals the
    un-studentized empirical-bootstrap p-value — the bootstrap probability that
    the null-recentred difference ``boot - obs`` exceeds the observed difference
    ``obs`` in absolute value. It is therefore a re-centred basic (empirical)
    stationary block bootstrap whose size is MEASURED (not certified) by
    ``null_calibration`` (audit C-7) — the measured drift is anti-conservative,
    see the module docstring; it also explains why this is *not* the Ledoit-Wolf
    (2008) studentized construction.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    if rng is None:
        rng = np.random.default_rng(0)  # P24: deterministic fallback (reported paths pass an explicit rng)

    def diff(x: np.ndarray, y: np.ndarray) -> float:
        return sharpe_ratio(x) - sharpe_ratio(y)

    obs = diff(a, b)
    boot = _bootstrap_statistic_distribution(a, b, diff, p, n_boot, rng)

    se = boot.std(ddof=1)
    if se == 0.0 or not np.isfinite(se):
        se = float("nan")

    stat = obs / se if np.isfinite(se) else 0.0
    # Null-recentred bootstrap statistics. NB: `se` cancels here against `stat`
    # (`|(boot-obs)/se| >= |obs/se|` <=> `|boot-obs| >= |obs|`), so this is the
    # un-studentized re-centred empirical bootstrap p-value, not a studentized one.
    centred = (boot - obs) / se if np.isfinite(se) else np.zeros_like(boot)
    if np.isfinite(se):
        pvalue = float((np.abs(centred) >= abs(stat)).mean())
    else:
        pvalue = 1.0
    # Guard against an exact-zero p-value (finite bootstrap resolution).
    pvalue = min(1.0, max(pvalue, 1.0 / (n_boot + 1)))

    ci_low = float(np.quantile(boot, 0.025))
    ci_high = float(np.quantile(boot, 0.975))
    return {"stat": float(stat), "pvalue": pvalue, "ci_low": ci_low, "ci_high": ci_high}


def cvar_difference_test(
    a: np.ndarray,
    b: np.ndarray,
    alpha: float = 0.05,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Two-sample re-centred stationary-bootstrap test for ``H0: CVaR(a) - CVaR(b) = 0``.

    This is the appropriate tool when ``a`` and ``b`` are the *realized* return series of two different
    strategies (e.g. the distributional vs scalar arm) and the question is whether their realized tail
    losses differ — the same re-centred stationary block-bootstrap construction as
    :func:`sharpe_difference_test`, applied to the CVaR functional (the bootstrap SE cancels in the
    p-value, so it is **not** studentized). No *published, named* two-sample difference-in-CVaR test
    exists (deep-research #2/#3), so this bespoke construction's size is MEASURED empirically by
    :func:`null_calibration` (audit C-7) — measured, NOT certified: the drift is anti-conservative
    (module docstring). CVaR/ES is a well-defined estimable functional because the pair
    (VaR, ES) is *jointly* elicitable (Fissler & Ziegel, 2016), even though ES alone is not.

    NOT to be confused with a *comparative backtest*: if instead you are comparing the tail-FORECAST
    accuracy of two risk models on a COMMON realized series, use
    :func:`src.inference.es_backtest.comparative_es_backtest` (the Diebold-Mariano test on the strictly
    consistent FZ0 (VaR, ES) score; Nolde & Ziegel, 2017).

    Power caveat (Bauer, 2025): tail-risk difference tests have low power at the most extreme quantiles
    (``alpha`` = 1%, 2.5%) and short windows — interpret CVaR-1% comparisons with care.

    Parameters
    ----------
    a, b:
        Aligned per-period realized return series for the two strategies.
    alpha:
        Tail level for CVaR (e.g. ``0.05``). The stationary-bootstrap block-restart
        probability is fixed internally at ``0.1`` (expected block length 10).
    n_boot:
        Number of bootstrap replications.
    rng:
        Optional NumPy generator for reproducibility.

    Returns
    -------
    dict
        ``{"stat", "pvalue", "ci_low", "ci_high"}``; semantics mirror
        :func:`sharpe_difference_test` for the CVaR difference.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    if rng is None:
        rng = np.random.default_rng(0)  # P24: deterministic fallback (reported paths pass an explicit rng)

    p = 0.1

    def diff(x: np.ndarray, y: np.ndarray) -> float:
        return cvar(x, alpha) - cvar(y, alpha)

    obs = diff(a, b)
    boot = _bootstrap_statistic_distribution(a, b, diff, p, n_boot, rng)

    se = boot.std(ddof=1)
    if se == 0.0 or not np.isfinite(se):
        se = float("nan")

    stat = obs / se if np.isfinite(se) else 0.0
    centred = (boot - obs) / se if np.isfinite(se) else np.zeros_like(boot)
    if np.isfinite(se):
        pvalue = float((np.abs(centred) >= abs(stat)).mean())
    else:
        pvalue = 1.0
    pvalue = min(1.0, max(pvalue, 1.0 / (n_boot + 1)))

    ci_low = float(np.quantile(boot, 0.025))
    ci_high = float(np.quantile(boot, 0.975))
    return {"stat": float(stat), "pvalue": pvalue, "ci_low": ci_low, "ci_high": ci_high}


def null_calibration(
    test_fn: Callable[[np.ndarray, np.ndarray], dict[str, float]],
    dist_sampler: Callable[[np.random.Generator], tuple[np.ndarray, np.ndarray]],
    n_reps: int = 200,
    rng: np.random.Generator | None = None,
) -> dict[str, object]:
    """MEASURE a difference test's size under a true null (audit C-7).

    NOTE: Named "null_calibration" for history, but it CERTIFIES nothing: it reports an empirical
    rejection rate with Monte-Carlo error. At production settings the headline test measured
    0.0573 two-sided / 0.0613 one-sided against nominal 0.05 — mildly anti-conservative. Report the
    measured value with its MC SE; never restate it as "correctly sized" or "conservative".

    Parameters
    ----------
    test_fn:
        Callable ``(a, b) -> dict`` returning a mapping with a ``"pvalue"`` key.
    dist_sampler:
        Callable ``(rng) -> (a, b)`` drawing a pair of series under the null.
    n_reps:
        Number of null replications.
    rng:
        Optional NumPy generator for reproducibility.

    Returns
    -------
    dict
        ``{"rejection_rate", "mean_pvalue", "pvalues"}`` where
        ``rejection_rate`` is the empirical fraction of p-values below 0.05.
    """
    if rng is None:
        rng = np.random.default_rng(0)  # P24: deterministic fallback (reported paths pass an explicit rng)
    pvalues = np.empty(n_reps, dtype=float)
    # Also certify the ONE-SIDED decision rule when the test exposes it. The headline H2-Tail leg rejects on
    # ``pvalue_one_sided_greater`` (R64 — the DIRECT upper-tail p, valid under a skewed/heavy CVaR bootstrap),
    # NOT the two-sided ``pvalue``. Certifying only the two-sided size would validate a statistic the tail IUT
    # never uses — precisely the skewed regime where a re-centred-bootstrap one-sided size can drift. We collect
    # both; under a true (exchangeable) null the effect is symmetric about 0, so each p is ~Uniform(0,1) and
    # each rejects at ~alpha. (Audit 2026-07-02: close the certificate↔decision-rule misalignment.)
    pvalues_one = np.full(n_reps, np.nan, dtype=float)
    have_one = True
    for i in range(n_reps):
        a, b = dist_sampler(rng)
        res = test_fn(a, b)
        pvalues[i] = float(res["pvalue"])
        p_one = res.get("pvalue_one_sided_greater") if hasattr(res, "get") else None
        if p_one is None:
            have_one = False
        else:
            pvalues_one[i] = float(p_one)
    rejection_rate = float((pvalues < 0.05).mean())
    out: dict[str, object] = {
        "rejection_rate": rejection_rate,
        "mean_pvalue": float(pvalues.mean()),
        "pvalues": pvalues,
    }
    if have_one:
        # Size of the ACTUAL one-sided headline rule (the number to report alongside the two-sided certificate).
        out["rejection_rate_one_sided"] = float((pvalues_one < 0.05).mean())
        out["mean_pvalue_one_sided"] = float(np.nanmean(pvalues_one))
        out["pvalues_one_sided"] = pvalues_one
    return out
