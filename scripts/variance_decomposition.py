"""Variance decomposition of the H2 IQM gap — the "one-lucky-reward" defence (reviewer attack #10).

Purpose (CAMPAIGN_DEEP_RESEARCH_FINDINGS_2026-06-21 §5.8; red-team #10)
----------------------------------------------------------------------
The headline H2 statistic is an **IQM gap** between two arms' per-seed test scores
(``IQM(distributional per-seed Sharpes) − IQM(scalar per-seed Sharpes)``; the rliable unit
of ``scripts/analyze_campaign.py::collect_family_pvalues``). A reviewer's sharpest attack is
"that gap is one *lucky reward* the LLM search happened to draw, not a property of the
distributional-feedback channel." This module answers it by **partitioning the variance of a
single per-seed test score into three nested, identifiable sources** and printing the verdict

    headline:  the IQM gap exceeds √σ²_search

so the effect is a property of the *channel*, not one lucky reward.

The three components (what each is, and where the data comes from)
-----------------------------------------------------------------
For ONE arm, organise the per-seed test scores as a ``(K, S)`` table — ``K`` independent
**re-runs of the reward search** (each yields a possibly-different winner reward) × ``S``
training **seeds** the frozen winner is re-trained at. Then:

* **σ²_seed** — variance across the ``S`` training seeds for a *fixed* winner reward (the
  training-RNG noise). FREE: it is the within-search across-seed dispersion already present in
  the 30 frozen-winner test records the campaign writes. In the one-way random-effects model
  below it is the **within-group (residual) mean square** ``MS_within``.
* **σ²_search** — variance across *independent search re-runs* (the **reward-draw** variance —
  exactly the "one lucky reward" quantity). Needs ``K ≥ 2`` re-run searches per arm (§5.8:
  "re-run the search K=3 per arm for the distributional+scalar pair; ~150 extra trainings,
  fits the 200-GPU-hr cap"). It is the **between-group variance component**
  ``σ̂²_search = (MS_between − MS_within) / n₀`` of the one-way random-effects ANOVA (groups =
  the K search runs). With ``K = 1`` it is **not identified** (zero between-group d.f.) and is
  reported ``status="skipped"`` — σ²_seed and σ²_market are still produced.
* **σ²_market** — sampling noise of the *single realized 2020-2026 test path itself*
  (independent of which reward / seed produced it). Estimated by a **stationary block bootstrap**
  of one representative winner's per-step test return series (``src.inference.bootstrap``,
  Politis-Romano 1994): the variance of the per-path score across bootstrap replications.

The estimator (one-way random-effects ANOVA, method of moments)
---------------------------------------------------------------
Model the per-seed scores ``y_ks = μ + a_k + e_ks`` with the **search-run effect**
``a_k ~ N(0, σ²_search)`` and the **seed residual** ``e_ks ~ N(0, σ²_seed)`` (Searle, Casella &
McCulloch 1992; Montgomery, *DOE*, the random-effects model; PSU STAT-503 §3.5). The
expected mean squares are ``E[MS_between] = σ²_seed + n₀·σ²_search`` and
``E[MS_within] = σ²_seed``, giving the ANOVA (method-of-moments) estimators

    σ̂²_seed   = MS_within
    σ̂²_search = (MS_between − MS_within) / n₀

with the unbalanced-design divisor ``n₀ = (1/(K−1))·(N − Σ_k S_k² / N)`` (``N = Σ_k S_k``),
which reduces to the common per-group count ``S`` when balanced. A **negative** ``σ̂²_search``
(``MS_between < MS_within`` — sampling noise when the true between-search variance is ≈0) is
**truncated to 0**: nonnegative quadratic unbiased estimators do not exist for a variance
component, and simple truncation uniformly improves the raw ANOVA estimate (standard practice).

Why σ²_search is the right yardstick (NOT the headline d.f.)
------------------------------------------------------------
The realized H2 test's degrees of freedom is ``n_seeds = 30`` PAIRED per-seed scores
(``paired_seed_difference_test``; §5.8 line: "neither N_block nor N_hmm is the headline d.f.").
This module does **not** re-test H2. It produces a *descriptive* variance budget: σ²_search is
the scale of the "lucky reward" the attack invokes, and the verdict reports whether the
observed IQM gap is larger than one standard deviation of that reward-draw noise. It is a
report-only appendix in its own declared family (key ``generator``/``component``), disjoint
from the frozen ``arm_a/arm_b/metric/level`` family (so ``assert_realized_family_matches_frozen``
stays green; §2 "Amendment-free additions").

Deliverable contract
--------------------
* Pure estimators (:func:`one_way_random_effects`, :func:`market_bootstrap_variance`,
  :func:`decompose`) — numpy + ``src.inference.bootstrap`` only, no torch, unit-tested.
* A driver (:func:`decompose_from_campaign`) shaped to the campaign records that **degrades
  gracefully** — ``status="skipped"`` with a reason, NEVER a crash — when the 30-seed data is
  absent (records-only / no winner test vectors) or only ``K = 1`` search run exists.
* A K-run input: pass ``--runs run0 run1 run2`` (each an archive root of one full search) to
  populate σ²_search; a single ``--root`` degrades to ``K = 1``.

Sources: Searle-Casella-McCulloch (*Variance Components*, 1992); Montgomery *DOE* (random-
effects one-way model); PSU STAT-503 §3.5; Politis & Romano 1994 (stationary bootstrap);
CAMPAIGN_DEEP_RESEARCH_FINDINGS_2026-06-21 §5.3/§5.8.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np

__all__ = [
    "ARMS_DEFAULT",
    "HEADLINE_CONTRAST",
    "one_way_random_effects",
    "market_bootstrap_variance",
    "decompose",
    "decompose_from_campaign",
    "verdict_markdown",
    "build_parser",
    "main",
]

_LOG = logging.getLogger(__name__)

#: The arms the variance study targets (§5.8: "the distributional+scalar pair"); the driver
#: degrades any arm with no usable data to ``status="skipped"`` rather than failing.
ARMS_DEFAULT: tuple[str, ...] = ("distributional", "scalar")

#: The headline contrast whose IQM gap is defended (PREREGISTRATION §1 H2 core leg).
HEADLINE_CONTRAST: tuple[str, str] = ("distributional", "scalar")


# --------------------------------------------------------------------------- #
# Pure estimators                                                              #
# --------------------------------------------------------------------------- #
def one_way_random_effects(groups: list[np.ndarray]) -> dict[str, Any]:
    """One-way random-effects ANOVA variance components (method of moments) for ``y_ks = μ + a_k + e_ks``.

    ``groups`` is a list of per-group score arrays — group ``k`` (a search re-run) holds the
    ``S_k`` per-seed scores of that run's frozen winner. Returns the between-group component
    ``sigma2_between`` (= the **reward-draw / σ²_search** variance) and the within-group
    residual ``sigma2_within`` (= the across-seed / **σ²_seed** variance), via

        MS_between = Σ_k S_k·(ȳ_k − ȳ)² / (K − 1)
        MS_within  = Σ_k Σ_s (y_ks − ȳ_k)² / (N − K)
        σ̂²_within  = MS_within
        σ̂²_between = max(0, (MS_between − MS_within) / n₀)
        n₀         = (1/(K − 1))·(N − Σ_k S_k² / N)          # = S when balanced

    where ``N = Σ_k S_k`` and ``K`` is the number of groups. The negative-estimate **truncation
    to 0** is the standard fix (no nonnegative quadratic unbiased estimator of a variance
    component exists; simple truncation uniformly improves the raw ANOVA estimate).

    Graceful degradation (never raises on a degenerate layout):
      * ``K < 2``               → ``status="skipped"`` (between-group variance unidentified);
        ``sigma2_within`` is still computed when that single group has ≥ 2 points.
      * ``N − K < 1``           → ``MS_within`` (hence σ²_seed) undefined: every group has a
        single point → ``sigma2_within=None``; ``status="skipped"``.
      * all groups single-point → as above.

    Returns
    -------
    dict
        ``{"sigma2_between", "sigma2_within", "ms_between", "ms_within", "n0", "K",
        "group_sizes", "grand_mean", "status", ["reason"]}``. Variance keys are ``float`` or
        ``None`` (never fabricated); ``status`` ∈ {"ok", "skipped"}.
    """
    arrs = [np.asarray(g, dtype=float).ravel() for g in groups]
    arrs = [a[np.isfinite(a)] for a in arrs]
    arrs = [a for a in arrs if a.size >= 1]
    K = len(arrs)
    sizes = [int(a.size) for a in arrs]
    N = int(sum(sizes))

    base: dict[str, Any] = {
        "sigma2_between": None,
        "sigma2_within": None,
        "ms_between": None,
        "ms_within": None,
        "n0": None,
        "K": K,
        "group_sizes": sizes,
        "grand_mean": None,
    }

    if K == 0 or N == 0:
        base.update(status="skipped", reason="no usable scores")
        return base

    # Grand mean over ALL observations (the unbalanced grand mean, not a mean-of-group-means).
    grand = float(np.concatenate(arrs).mean())
    base["grand_mean"] = grand
    group_means = np.array([float(a.mean()) for a in arrs], dtype=float)

    # Within-group (residual) sum of squares and its mean square (σ²_seed). Needs N - K >= 1.
    ss_within = float(sum(float(((a - a.mean()) ** 2).sum()) for a in arrs))
    df_within = N - K
    ms_within = (ss_within / df_within) if df_within >= 1 else None
    base["ms_within"] = ms_within
    if ms_within is not None:
        base["sigma2_within"] = float(ms_within)

    if K < 2:
        # One search run: the across-seed (σ²_seed) residual is still meaningful, but the
        # between-search (σ²_search) component is not identified (zero between-group d.f.).
        base.update(
            status="skipped",
            reason=f"only K={K} search run(s); σ²_search needs K >= 2 independent re-runs",
        )
        return base

    # Between-group mean square.
    ss_between = float((np.array(sizes, dtype=float) * (group_means - grand) ** 2).sum())
    df_between = K - 1
    ms_between = ss_between / df_between
    base["ms_between"] = float(ms_between)

    # Unbalanced divisor n0 (= common S when balanced). With df_between>=1 and N>0 this is finite.
    sum_sq = float(sum(s * s for s in sizes))
    n0 = (N - sum_sq / N) / df_between
    base["n0"] = float(n0)

    if ms_within is None:
        # Every group is a single point (N == K): no residual d.f., so σ²_seed and the
        # MS_between - MS_within contrast are both undefined — cannot separate the components.
        base.update(
            status="skipped",
            reason=f"no within-group d.f. (N={N}, K={K}); every search run has a single seed",
        )
        return base

    if n0 <= 0.0:
        base.update(status="skipped", reason="degenerate design (n0 <= 0)")
        return base

    # ANOVA between-group variance component, truncated at 0 (standard non-negativity fix).
    sigma2_between_raw = (ms_between - ms_within) / n0
    base["sigma2_between"] = float(max(0.0, sigma2_between_raw))
    base["sigma2_between_raw"] = float(sigma2_between_raw)  # pre-truncation (diagnostic)
    base["status"] = "ok"
    return base


def market_bootstrap_variance(
    test_returns: np.ndarray,
    *,
    statistic: Callable[[np.ndarray], float],
    p: float = 0.1,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """σ²_market: sampling variance of a single test path's score via the stationary block bootstrap.

    Resamples the per-step test return series ``test_returns`` with the Politis-Romano (1994)
    stationary bootstrap (random geometric block length ``1/p``, wrap-around;
    ``src.inference.bootstrap.stationary_bootstrap_indices``), recomputes ``statistic`` on each
    path, and returns the **variance of the statistic across bootstrap replications** — the
    block-bootstrap estimate of the test-path sampling variance under the serial dependence
    (volatility clustering) of daily P&L. This is the market-path noise component: it is
    orthogonal to which reward (σ²_search) or seed (σ²_seed) produced the path.

    Parameters
    ----------
    test_returns : np.ndarray
        One representative frozen-winner per-step test return series (e.g. the seed-median
        path, or any single seed's path) for the arm. Needs >= 2 finite points.
    statistic : callable
        Score functional applied to a return series (e.g. ``sharpe_ratio``); the SAME functional
        whose IQM gap is being defended, so σ²_market is on the score's scale.
    p, n_boot, rng
        Stationary-bootstrap block-restart probability (expected block length ``1/p``),
        replication count, and an optional seeded generator.

    Returns
    -------
    dict
        ``{"sigma2_market", "se_market", "point", "n_steps", "p", "n_boot", "status",
        ["reason"]}``. ``sigma2_market`` is the across-replication variance (``None`` when the
        path is too short — graceful, never raised).
    """
    from src.inference.bootstrap import stationary_bootstrap_indices

    r = np.asarray(test_returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    out: dict[str, Any] = {
        "sigma2_market": None,
        "se_market": None,
        "point": None,
        "n_steps": int(r.size),
        "p": float(p),
        "n_boot": int(n_boot),
    }
    if r.size < 2:
        out.update(status="skipped", reason=f"test path too short ({r.size} finite steps)")
        return out
    if rng is None:
        rng = np.random.default_rng(0)

    out["point"] = float(statistic(r))
    boot = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        idx = stationary_bootstrap_indices(r.size, p, rng)
        boot[i] = float(statistic(r[idx]))
    boot = boot[np.isfinite(boot)]
    if boot.size < 2:
        out.update(status="skipped", reason="bootstrap produced no finite statistics")
        return out
    var = float(np.var(boot, ddof=1))
    out.update(
        sigma2_market=var,
        se_market=float(np.sqrt(var)),
        status="ok",
    )
    return out


def _iqm_gap(
    scores_a: list[np.ndarray] | np.ndarray,
    scores_b: list[np.ndarray] | np.ndarray,
) -> float | None:
    """IQM(pooled a-scores) − IQM(pooled b-scores), or ``None`` if either side is empty.

    The per-seed scores of every search run are POOLED per arm (the IQM gap is a property of the
    arm's score distribution, matching the headline ``collect_family_pvalues`` IQM-over-per-seed
    construction). Returns ``None`` rather than NaN when a side has no scores (graceful).
    """
    from src.inference.bootstrap import iqm

    def _pool(x: list[np.ndarray] | np.ndarray) -> np.ndarray:
        if isinstance(x, np.ndarray):
            arr = np.asarray(x, dtype=float).ravel()
        else:
            parts = [np.asarray(g, dtype=float).ravel() for g in x]
            arr = np.concatenate(parts) if parts else np.empty(0, dtype=float)
        return arr[np.isfinite(arr)]

    a = _pool(scores_a)
    b = _pool(scores_b)
    if a.size == 0 or b.size == 0:
        return None
    return float(iqm(a) - iqm(b))


def decompose(
    arm_run_scores: dict[str, list[np.ndarray]],
    *,
    contrast: tuple[str, str] = HEADLINE_CONTRAST,
    market_paths: dict[str, np.ndarray] | None = None,
    score_statistic: Callable[[np.ndarray], float] | None = None,
    p: float = 0.1,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Full σ²_seed / σ²_search / σ²_market decomposition + the "gap vs √σ²_search" verdict.

    Parameters
    ----------
    arm_run_scores : dict[str, list[np.ndarray]]
        ``{arm: [run0_seed_scores, run1_seed_scores, ...]}`` — per arm, a list over the ``K``
        independent search re-runs, each entry that run's winner's per-seed scores (one score per
        training seed). ``K = 1`` (a single campaign archive) degrades σ²_search to skipped.
    contrast : (str, str)
        The arm pair whose IQM gap is defended (default the H2 core leg ``distributional>scalar``).
    market_paths : dict[str, np.ndarray], optional
        ``{arm: representative per-step test return series}`` for the σ²_market block bootstrap.
        Omitted → σ²_market is reported as skipped (the seed/search split still runs).
    score_statistic : callable, optional
        The score functional (default ``src.inference.bootstrap.sharpe_ratio``); used for
        σ²_market only (``arm_run_scores`` are already reduced per-seed scores).
    p, n_boot, rng
        Stationary-bootstrap knobs + seed for σ²_market.

    Returns
    -------
    dict
        ``{"contrast", "components": {arm: {"seed": ..., "search": ..., "market": ...}},
        "iqm_gap", "sigma2_search", "se_search", "verdict": {...}, "status"}``. The headline
        ``sigma2_search`` is the contrast arm-a's between-search component (the arm whose reward
        is the treatment); ``verdict.gap_exceeds_sqrt_sigma2_search`` is the #10 answer.
    """
    from src.inference.bootstrap import sharpe_ratio

    if score_statistic is None:
        score_statistic = sharpe_ratio
    if rng is None:
        rng = np.random.default_rng(0)
    arm_a, arm_b = contrast

    components: dict[str, dict[str, Any]] = {}
    for arm, runs in arm_run_scores.items():
        anova = one_way_random_effects(runs)
        market = (
            market_bootstrap_variance(
                market_paths[arm],
                statistic=score_statistic,
                p=p,
                n_boot=n_boot,
                rng=np.random.default_rng(int(rng.integers(0, 2**32 - 1))),
            )
            if market_paths and arm in market_paths
            else {"sigma2_market": None, "status": "skipped", "reason": "no test path supplied"}
        )
        components[arm] = {
            "seed": {
                "sigma2": anova.get("sigma2_within"),
                "status": "ok" if anova.get("sigma2_within") is not None else "skipped",
                "ms_within": anova.get("ms_within"),
            },
            "search": {
                "sigma2": anova.get("sigma2_between"),
                "status": anova.get("status"),
                "reason": anova.get("reason"),
                "K": anova.get("K"),
                "n0": anova.get("n0"),
                "ms_between": anova.get("ms_between"),
                "sigma2_raw": anova.get("sigma2_between_raw"),
            },
            "market": {
                "sigma2": market.get("sigma2_market"),
                "se": market.get("se_market"),
                "status": market.get("status"),
                "reason": market.get("reason"),
                "n_steps": market.get("n_steps"),
            },
        }

    # The IQM gap defended (pooled per-seed scores across runs, per arm).
    gap = (
        _iqm_gap(arm_run_scores.get(arm_a, []), arm_run_scores.get(arm_b, []))
        if arm_a in arm_run_scores and arm_b in arm_run_scores
        else None
    )

    # σ²_search yardstick = the TREATMENT arm's (arm_a) between-search component. The "lucky
    # reward" the attack invokes is a draw of arm_a's reward; that is the noise the gap must clear.
    sigma2_search = components.get(arm_a, {}).get("search", {}).get("sigma2")
    se_search = (
        float(np.sqrt(sigma2_search))
        if isinstance(sigma2_search, (int, float)) and sigma2_search is not None
        else None
    )

    verdict: dict[str, Any]
    if gap is None or se_search is None:
        verdict = {
            "gap_exceeds_sqrt_sigma2_search": None,
            "status": "skipped",
            "reason": (
                "IQM gap unavailable (missing contrast arm)"
                if gap is None
                else f"σ²_search not identified for '{arm_a}' "
                f"({components.get(arm_a, {}).get('search', {}).get('reason', 'no data')})"
            ),
        }
        overall = "skipped"
    else:
        # σ²_search == 0 means the ANOVA truncated a (near-)zero/negative raw between-search
        # component to 0 (MS_between <= MS_within): the reward-draw variance is NOT distinguishable
        # from zero at this K. That is the EXPECTED case under σ_seed-dominance (the pilot finding),
        # so we flag it — a bare "ratio = ∞ / EXCEEDS" would over-state the precision of a small-K
        # zero estimate. The gap still trivially clears a zero yardstick; the rendering is honest
        # about WHY (see verdict_markdown).
        sigma2_search_raw = components.get(arm_a, {}).get("search", {}).get("sigma2_raw")
        verdict = {
            "gap_exceeds_sqrt_sigma2_search": bool(abs(gap) > se_search),
            "iqm_gap": float(gap),
            "sqrt_sigma2_search": float(se_search),
            "ratio_gap_over_se": (float(gap / se_search) if se_search > 0 else float("inf")),
            "sigma2_search_zero": bool(se_search == 0.0),
            "sigma2_search_raw": (
                float(sigma2_search_raw)
                if isinstance(sigma2_search_raw, (int, float)) and sigma2_search_raw is not None
                else None
            ),
            "status": "ok",
        }
        overall = "ok"

    return {
        "contrast": f"{arm_a}>{arm_b}",
        "components": components,
        "iqm_gap": gap,
        "sigma2_search": sigma2_search,
        "se_search": se_search,
        "verdict": verdict,
        "status": overall,
    }


# --------------------------------------------------------------------------- #
# The campaign-records driver (degrades gracefully — never crashes)            #
# --------------------------------------------------------------------------- #
def _per_seed_scores_from_records(
    records: list[dict[str, Any]],
    arm: str,
    statistic: Callable[[np.ndarray], float],
) -> dict[int, float]:
    """``{seed: statistic(that seed's frozen-winner test series)}`` for one arm (graceful on gaps).

    Mirrors ``scripts/analyze_campaign.py::_seed_scores`` (the headline inference unit): one
    frozen-winner TEST record per (arm, seed), reduced to a per-seed score. Records lacking a
    usable ``metrics['test_returns']`` vector or a ``seed`` are skipped; an arm with no usable
    record returns ``{}`` (the driver then reports σ²_seed/σ²_search skipped, never crashes).
    """
    out: dict[int, float] = {}
    for r in records:
        if r.get("arm") != arm or r.get("seed") is None:
            continue
        metrics = r.get("metrics") or {}
        tr = metrics.get("test_returns")
        if tr is None:
            tr = r.get("test_returns")
        if tr is None:
            continue
        v = np.asarray(tr, dtype=float).ravel()
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        out[int(r["seed"])] = float(statistic(v))
    return out


def _representative_test_path(
    records: list[dict[str, Any]], arm: str
) -> np.ndarray | None:
    """One representative per-step test return series for ``arm`` (the seed-MEDIAN path, by score).

    For σ²_market we need a SINGLE realized path, not a seed-average (averaging seed paths shrinks
    the path's own variance — the same anti-conservatism the H2 #9/#14 fix removed). We take the
    seed whose Sharpe is the MEDIAN across seeds (a representative, not best/worst, single path),
    aligned to the common leading length. Returns ``None`` when no usable record exists.
    """
    from src.inference.bootstrap import sharpe_ratio

    paths: dict[int, np.ndarray] = {}
    for r in records:
        if r.get("arm") != arm or r.get("seed") is None:
            continue
        metrics = r.get("metrics") or {}
        tr = metrics.get("test_returns")
        if tr is None:
            tr = r.get("test_returns")
        if tr is None:
            continue
        v = np.asarray(tr, dtype=float).ravel()
        v = v[np.isfinite(v)]
        if v.size >= 2:
            paths[int(r["seed"])] = v
    if not paths:
        return None
    seeds = sorted(paths)
    scores = np.array([sharpe_ratio(paths[s]) for s in seeds], dtype=float)
    # Median seed by score: argsort then take the middle (lower-median for an even count).
    order = np.argsort(scores, kind="mergesort")
    median_seed = seeds[int(order[(len(seeds) - 1) // 2])]
    return paths[median_seed]


def _load_runs(roots: Sequence[str | Path]) -> list[list[dict[str, Any]]]:
    """Load one campaign records list per search-run root (via analyze_campaign's loader).

    Each root is the ``output_dir`` of ONE full search (the campaign writes
    ``<leg>/<arm>/<cand>``); ``load_campaign_records`` walks both legs. A root that fails to load
    (absent / unreadable) contributes an EMPTY list rather than aborting the study (graceful).
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from analyze_campaign import load_campaign_records  # type: ignore[import-not-found]

    runs: list[list[dict[str, Any]]] = []
    for root in roots:
        try:
            runs.append(load_campaign_records(root))
        except Exception as exc:  # noqa: BLE001 — a bad run root must not crash the study
            _LOG.warning("variance_decomposition: failed to load run root %r: %s", str(root), exc)
            runs.append([])
    return runs


def decompose_from_campaign(
    run_records: list[list[dict[str, Any]]],
    *,
    arms: tuple[str, ...] = ARMS_DEFAULT,
    contrast: tuple[str, str] = HEADLINE_CONTRAST,
    n_boot: int = 2000,
    p: float = 0.1,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Driver: build the ``(K, S)`` per-seed score table per arm from K campaign archives, decompose.

    ``run_records`` is a list over the ``K`` independent search re-runs; each element is that
    run's loaded campaign records (``analyze_campaign.load_campaign_records`` output). With
    ``K = 1`` (a single campaign archive, the common case) σ²_seed and σ²_market are reported and
    σ²_search degrades to ``status="skipped"`` (NOT a crash). Per-arm Sharpe is the per-seed
    score (the headline unit); the σ²_market path is the seed-median test series of the LAST run.

    Returns the :func:`decompose` dict, plus ``"n_runs"`` and a per-arm ``"seeds_per_run"`` audit.
    """
    from src.inference.bootstrap import sharpe_ratio

    if rng is None:
        rng = np.random.default_rng(0)

    arm_run_scores: dict[str, list[np.ndarray]] = {arm: [] for arm in arms}
    seeds_per_run: dict[str, list[int]] = {arm: [] for arm in arms}
    for recs in run_records:
        for arm in arms:
            scores = _per_seed_scores_from_records(recs, arm, sharpe_ratio)
            # One group per run that actually has >=1 seed score; empty runs contribute nothing
            # (so K counts only the runs with data for that arm — honest group count).
            seeds_per_run[arm].append(len(scores))
            if scores:
                arm_run_scores[arm].append(
                    np.array([scores[s] for s in sorted(scores)], dtype=float)
                )

    # σ²_market path: the seed-median representative path from the LAST run that has one for the arm.
    market_paths: dict[str, np.ndarray] = {}
    for arm in arms:
        for recs in reversed(run_records):
            path = _representative_test_path(recs, arm)
            if path is not None:
                market_paths[arm] = path
                break

    result = decompose(
        arm_run_scores,
        contrast=contrast,
        market_paths=market_paths or None,
        score_statistic=sharpe_ratio,
        p=p,
        n_boot=n_boot,
        rng=rng,
    )
    result["n_runs"] = len(run_records)
    result["seeds_per_run"] = seeds_per_run
    return result


# --------------------------------------------------------------------------- #
# Rendering                                                                    #
# --------------------------------------------------------------------------- #
def _fmt(x: Any, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}"
    return str(x)


def verdict_markdown(result: dict[str, Any]) -> str:
    """Render the variance budget + the "gap vs √σ²_search" verdict as markdown (report-only)."""
    lines = [
        "# Variance decomposition — the one-lucky-reward defence (reviewer attack #10)",
        "",
        "Partition of a single per-seed test score's variance into **σ²_seed** (across the 30 "
        "training seeds, free), **σ²_search** (across K independent search re-runs — the "
        "reward-draw variance), and **σ²_market** (stationary block bootstrap of the realized "
        "test path). One-way random-effects ANOVA (method of moments); σ²_market via "
        "Politis-Romano (1994). Report-only appendix (declared family, disjoint from the frozen "
        "arm×metric family); the headline H2 d.f. remains the 30 paired per-seed scores.",
        "",
        f"K (independent search re-runs supplied): **{result.get('n_runs', '?')}**  "
        f"(σ²_search needs K ≥ 2).",
        "",
        "| arm | σ²_seed | σ²_search | σ²_market | σ²_search status |",
        "|---|---|---|---|---|",
    ]
    for arm, comp in result.get("components", {}).items():
        seed_v = comp.get("seed", {}).get("sigma2")
        search = comp.get("search", {})
        search_v = search.get("sigma2")
        market_v = comp.get("market", {}).get("sigma2")
        sstat = search.get("status", "?")
        if sstat == "skipped" and search.get("reason"):
            sstat = f"skipped ({search['reason']})"
        lines.append(
            f"| {arm} | {_fmt(seed_v)} | {_fmt(search_v)} | {_fmt(market_v)} | {sstat} |"
        )

    v = result.get("verdict", {})
    lines.append("")
    if v.get("status") == "ok":
        gap = v.get("iqm_gap", 0.0)
        se = v.get("sqrt_sigma2_search", 0.0)
        ok = v.get("gap_exceeds_sqrt_sigma2_search")
        ratio = v.get("ratio_gap_over_se")
        if v.get("sigma2_search_zero"):
            # σ²_search truncated to ~0 (between-search MS <= within-search MS): the reward-draw
            # variance is NOT distinguishable from zero at this K. The gap trivially clears a zero
            # yardstick, so report it HONESTLY (not as a crisp "ratio = ∞ / channel-not-luck" win).
            raw = v.get("sigma2_search_raw")
            lines += [
                f"**Verdict ({result.get('contrast', '?')}):** σ²_search is estimated at **~0** "
                f"(raw pre-truncation `{_fmt(raw)}`; between-search MS ≤ within-search MS) — the "
                f"reward-draw noise is **not distinguishable from zero** at this K. The IQM gap "
                f"`{_fmt(gap)}` therefore clears it trivially.",
                "",
                "→ Consistent with SEED variance dominating reward-draw variance (the σ_seed-"
                "dominance finding): the one-lucky-reward attack has ~no variance to stand on. This "
                "is a small-K estimate, so report it as SUPPORTIVE-but-weak (more independent search "
                "re-runs would sharpen σ²_search), NOT a precise margin.",
            ]
        else:
            verdict_word = "EXCEEDS" if ok else "does NOT exceed"
            lines += [
                f"**Verdict ({result.get('contrast', '?')}):** IQM gap "
                f"`{_fmt(gap)}` {verdict_word} `√σ²_search = {_fmt(se)}` "
                f"(gap / √σ²_search = {_fmt(ratio, 2)}).",
                "",
                (
                    "→ The effect is a property of the channel, not one lucky reward."
                    if ok
                    else "→ The gap is within one standard deviation of the reward-draw noise — "
                    "report as INCONCLUSIVE on the one-lucky-reward axis (more search re-runs or a "
                    "larger effect needed)."
                ),
            ]
    else:
        lines.append(
            f"**Verdict:** not computed — {v.get('reason', 'insufficient data')} "
            "(σ²_seed / σ²_market above are still reported where available; supply K ≥ 2 search "
            "re-runs via `--runs` to identify σ²_search)."
        )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Variance decomposition (σ²_seed / σ²_search / σ²_market) — reviewer attack #10.",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help="One archive root per INDEPENDENT search re-run (each the output_dir of a full "
        "search). >= 2 roots identify σ²_search; a single root degrades to K=1 (σ²_search "
        "skipped). If omitted, falls back to --root (K=1).",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Single campaign archive root (K=1 fallback when --runs is omitted; default: "
        "config campaign.output_dir, then the prototype output_dir).",
    )
    parser.add_argument("--n-boot", type=int, default=2000, help="Stationary-bootstrap reps for σ²_market.")
    parser.add_argument("--p", type=float, default=0.1, help="Stationary-bootstrap block-restart prob.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for σ²_market reproducibility.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output dir for variance_decomposition.{md,json} (default: the first run root).",
    )
    return parser


def _resolve_roots(args: argparse.Namespace) -> list[str]:
    """Resolve the list of run roots from --runs / --root / config (graceful, K=1 fallback)."""
    if args.runs:
        return [str(r) for r in args.runs]
    if args.root:
        return [str(args.root)]
    # Config fallback: the campaign output_dir, then the prototype's (single K=1 archive).
    try:
        from src.utils.config import load_config

        campaign_root = str(load_config("campaign").get("output_dir", "outputs/campaign"))
        if Path(campaign_root).is_dir():
            return [campaign_root]
        return [str(load_config("prototype").get("output_dir", "outputs/prototype"))]
    except Exception:  # noqa: BLE001 — no config (isolated run) -> a sane default
        return ["outputs/campaign"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args()
    roots = _resolve_roots(args)

    run_records = _load_runs(roots)
    result = decompose_from_campaign(
        run_records, n_boot=int(args.n_boot), p=float(args.p), rng=np.random.default_rng(int(args.seed))
    )

    out_dir = Path(args.out) if args.out else Path(roots[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    md = verdict_markdown(result)
    (out_dir / "variance_decomposition.md").write_text(md, encoding="utf-8")
    (out_dir / "variance_decomposition.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )

    print(f"[variance_decomposition] K={result['n_runs']} run root(s); wrote {out_dir / 'variance_decomposition.md'}")
    for arm, comp in result.get("components", {}).items():
        print(
            f"  {arm:>14}: σ²_seed={_fmt(comp['seed']['sigma2'])} "
            f"σ²_search={_fmt(comp['search']['sigma2'])} ({comp['search']['status']}) "
            f"σ²_market={_fmt(comp['market']['sigma2'])}"
        )
    v = result.get("verdict", {})
    if v.get("status") == "ok":
        ok = v.get("gap_exceeds_sqrt_sigma2_search")
        print(
            f"  VERDICT {result['contrast']}: IQM gap {_fmt(v['iqm_gap'])} "
            f"{'>' if ok else '<='} √σ²_search {_fmt(v['sqrt_sigma2_search'])} "
            f"-> {'channel, not luck' if ok else 'INCONCLUSIVE'}"
        )
    else:
        print(f"  VERDICT: skipped — {v.get('reason', 'insufficient data')}")


if __name__ == "__main__":
    main()
