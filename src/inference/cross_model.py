"""Cross-model synthesis for the v2 replication legs — REPORT-ONLY (R80/R82, frozen spec).

The replication legs (ten under R95) share the market panel and the CRN seed set BY DESIGN (pairing), so they are NOT
independent replications: a naive binomial sign test (0.5^k) would overstate the evidence. The
registered synthesis therefore has two tiers:

* :func:`sign_count` — the replication count reported DESCRIPTIVELY, on the CVaR-leg (tail)
  distributional-vs-scalar contrast (the Sharpe leg is predicted-tie for every model, so its sign
  is noise by design), with the T0-floor leg-inclusion criterion: a leg whose winners fail the
  naive-benchmark floor in either contrasted arm is an authoring/search FAILURE — reported in the
  reliability table, never counted as a vote.
* :func:`permutation_test` — the inferential tier: for each shared CRN seed, flip the
  (distributional <-> scalar) assignment with p=1/2 SIMULTANEOUSLY across all included legs and
  recompute the sign statistic. Because the flip is per-seed and joint across legs, the
  shared-seed/shared-panel dependence lives INSIDE the null distribution instead of being
  falsely assumed away (10,000 reps, seeded).

Plus the registered instruments: :func:`pair_did` (the family-pair content-effect x capability
interaction on the COMMON floor-30 seed subset, seed-paired bootstrap), :func:`leg_family_bh`
(BH across the report-only leg family — delegates to the existing implementation),
:func:`generation_indexed_responsiveness` (does feedback-use strengthen across the loop's
generations? — reuses the SQ1 estimator), and :func:`capability_regression` (responsiveness vs a
capability anchor; primary = the pre-declared external composite, secondary = the M2 reading
score). Everything here is deterministic under its seed and DISJOINT from the m=6 confirmatory
machinery — nothing gates H1–H4.

WIRED: and this paragraph is the correction of a stale warning (deep review #115, 2026-07-27).
:func:`pooled_bound` is registered as *the* cross-model bounded-effect statement (R86,
``config/preregistration.yaml: synthesis_exactness.pooled_bound``) and **R101 reframed the headline
around it**, so whether it has an executable route to a number is load-bearing. It does:
``scripts/analyze_campaign.py`` imports ``leg_aggregate.cross_model_synthesis`` and CALLS it
(``out["cross_model"] = cross_model_synthesis(root)``), which chains into this module's
:func:`sign_count` / :func:`permutation_test` / :func:`pooled_bound` via the import inside
``cross_model_synthesis``. The end-to-end test exists and also locks the anti-fabrication states:
``tests/test_leg_aggregate.py`` ("the production wiring (row 34)"), including
``test_no_leg_archives_is_MISSING_DATA_not_a_null_effect`` and
``test_a_missing_T0_floor_is_a_MISSING_INPUT_not_a_result``.

NOTE: HISTORY, kept because the correction matters more than the claim: until 2026-07-27 this docstring
opened "NOT YET WIRED … no production caller … do not read this docstring as describing an executed
path". That was TRUE when written (2026-07-26, deep review) and was made FALSE the same day by
the wiring change, which did not update it. The write-time registry row 34 WAS corrected
(it carries a "CLOSED BY ROUTE (a) — WIRED" box, left open rather than ticked because discharging an
obligation is an open design decision) — so the fact was fixed in the register and never followed into the CODE,
the exact inverse of the #107 lesson. The stale text was actively harmful, not merely untidy: row
34's two closure routes are "wire it" or "WITHDRAW the registered claim", so a writer trusting this
docstring could have withdrawn a claim that is in fact executable, after the compute was spent. The
same paragraph also warned of a "latent per-period-vs-annualised floor-unit trap in ``leg_aggregate``
that only bites on wiring" — that trap was fixed in the same 2026-07-26 change (``per_seed_series``
delegates to the canonical ``bootstrap.sharpe_ratio``, removing the annualisation and ddof
discrepancies together).

The R16 precedent the old text invoked still stands as the reason to check: ``h2_conjunction`` was
"implemented and unit-tested but previously unwired, so the documented headline test never actually
ran". A unit-tested module is not a wired one — which is why the wiring above is asserted by an
end-to-end test rather than by this sentence.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from src.inference.multiple_testing import benjamini_hochberg
from src.inference.responsiveness import responsiveness

__all__ = [
    "sign_count",
    "permutation_test",
    "pooled_bound",
    "pair_did",
    "leg_family_bh",
    "generation_indexed_responsiveness",
    "capability_regression",
]


def _included(leg_results: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    """The T0-filtered legs: label -> per-seed CVaR diff array (dist − scalar, seed-aligned)."""
    out: dict[str, np.ndarray] = {}
    for label, res in leg_results.items():
        if bool(res.get("t0_floor_pass", False)):
            arr = np.asarray(res["cvar_diff_per_seed"], dtype=float)
            if arr.ndim != 1 or arr.size == 0:
                raise ValueError(f"leg {label!r}: cvar_diff_per_seed must be a non-empty 1-D array")
            out[label] = arr
    return out


def sign_count(leg_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The DESCRIPTIVE cross-leg replication count on the CVaR-leg contrast (frozen spec).

    Sign convention: the diff is (distributional − scalar) per-seed CVaR; CVaR here is a signed
    return (more negative = worse tail), so a NEGATIVE mean diff means the distributional arm's
    tail is WORSE and a POSITIVE mean diff means distributional is SAFER. The registered
    direction of interest is dist-safer (positive).
    """
    included = _included(leg_results)
    excluded = sorted(set(leg_results) - set(included))
    per_leg_sign = {label: int(np.sign(np.mean(arr))) for label, arr in included.items()}
    return {
        "n_legs_included": len(included),
        "n_dist_safer": sum(1 for s in per_leg_sign.values() if s > 0),
        "n_dist_worse": sum(1 for s in per_leg_sign.values() if s < 0),
        "n_zero": sum(1 for s in per_leg_sign.values() if s == 0),
        "excluded_failed_floor": excluded,
        "per_leg_sign": per_leg_sign,
        "note": "descriptive — legs share the panel and CRN seeds (NOT independent); "
                "inference lives in permutation_test",
    }


def permutation_test(
    leg_results: dict[str, dict[str, Any]],
    n_reps: int = 10_000,
    seed: int = 20260720,
) -> dict[str, Any]:
    """The registered per-seed JOINT-FLIP permutation test (frozen spec verbatim).

    Null: no content effect in any leg. Scheme: one Bernoulli(1/2) flip PER SEED PER REP, applied
    to every included leg's diff at that seed simultaneously — cross-leg dependence induced by the
    shared seeds/panel is thereby preserved under the null. Statistic: **the POOLED MEAN difference
    over the included legs** (the multivariate paired sign-flip test). One-sided p toward the
    registered direction (dist-safer), add-one corrected.

    NOTE: This sentence read "Statistic: the number of included legs whose mean flipped diff is positive"
    until 2026-07-27 (deep review #116). That is the SUPERSEDED statistic: the register was refined
    pre-freeze on 2026-07-21 to the pooled mean precisely because a sign-COUNT is near-powerless here
    (under joint per-seed flips with highly correlated legs, unanimity is routine under the null), and
    the sign count was demoted to the DESCRIPTIVE layer in :func:`sign_count`. The refinement landed
    in ``config/preregistration.yaml`` (``synthesis_exactness.permutation_test``) and in the CODE
    below — which computes ``mat.mean()`` and has always been correct — but not in this docstring,
    which meanwhile claimed to be the "frozen spec verbatim". The stale wording mattered because it
    described the inferential statistic of a registered analysis: a write-up trusting it would have
    reported the method as a sign-count test, i.e. as the exact procedure the register rejects.
    """
    included = _included(leg_results)
    if not included:
        return {"p_value": None, "n_legs": 0, "note": "no legs passed the T0 floor"}
    labels = sorted(included)
    sizes = {included[label].size for label in labels}
    if len(sizes) != 1:
        raise ValueError("all included legs must share the same CRN seed count (seed-aligned)")
    mat = np.vstack([included[label] for label in labels])          # (n_legs, n_seeds)
    n_seeds = mat.shape[1]
    # STATISTIC (registered, refined pre-freeze 2026-07-21): the POOLED MEAN difference — the
    # multivariate paired sign-flip test. The sign COUNT stays the descriptive layer
    # (sign_count above): under joint per-seed flips with highly correlated legs, a count
    # statistic is near-powerless (all legs flip together, so unanimity is routine under the
    # null); the pooled mean retains magnitude information and full power while the identical
    # flip scheme keeps the shared-seed/panel dependence inside the null.
    observed = float(mat.mean())                                    # pooled over legs AND seeds
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(int(n_reps), n_seeds))    # per-seed, shared across legs
    seed_means = mat.mean(axis=0)                                   # (n_seeds,) leg-pooled per seed
    null_stats = flips @ seed_means / n_seeds                       # per-rep pooled means
    p = float((np.sum(null_stats >= observed) + 1) / (int(n_reps) + 1))  # one-sided, add-one
    # The descriptive unanimity count under the same null (for the reporting table).
    per_rep_leg_means = np.einsum("rs,ls->rl", flips, mat) / n_seeds
    null_counts = np.sum(per_rep_leg_means > 0, axis=1)
    return {
        "p_value": p,
        "observed_statistic": observed,
        "observed_sign_count": int(np.sum(mat.mean(axis=1) > 0)),
        "n_legs": len(labels),
        "n_seeds": n_seeds,
        "n_reps": int(n_reps),
        "null_mean": float(null_stats.mean()),
        "null_q95": float(np.quantile(null_stats, 0.95)),
        "null_count_q95": float(np.quantile(null_counts, 0.95)),
        "labels": labels,
    }


def pooled_bound(
    leg_results: dict[str, dict[str, Any]],
    n_boot: int = 10_000,
    seed: int = 20260721,
) -> dict[str, Any]:
    """R86 — the registered BOUNDED-EFFECT statement: a 90% seed-block-bootstrap CI on the pooled
    mean (dist − scalar) CVaR-5% difference over the T0-included legs.

    A large permutation p alone is absence-of-evidence; per-leg TOSTs at the 30-seed floor are
    inconclusive by construction — the pooled 9×30 paired seeds is where the precision lives.
    Dependence honesty: resample SEED indices i.i.d. and apply the SAME draw to every leg (the
    per-seed leg-pooled means carry the cross-leg dependence inside each resample — k perfectly
    correlated legs yield exactly one leg's CI, never a fake √k shrink). Reported in daily-return
    units AND as a fraction of the scalar-arm pooled CVaR level (interpretability; the SESOI is
    DSR-denominated so no CVaR-unit margin exists — this is a bound, not a TOST verdict).
    """
    included = _included(leg_results)
    if not included:
        return {"estimate": None, "note": "no legs passed the T0 floor"}
    labels = sorted(included)
    if len({included[k].size for k in labels}) != 1:
        raise ValueError("all included legs must share the same CRN seed count (seed-aligned)")
    mat = np.vstack([included[k] for k in labels])              # (n_legs, n_seeds)
    seed_means = mat.mean(axis=0)                               # leg-pooled per seed
    n_seeds = seed_means.size
    if n_seeds < 2:
        # row 30b: a 1-seed "CI" is a silent point interval; mirror paired_seed_difference_test.
        raise ValueError(f"pooled_bound needs >=2 CRN seeds (got {n_seeds})")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_seeds, size=(int(n_boot), n_seeds))
    boot = seed_means[idx].mean(axis=1)
    est = float(seed_means.mean())
    lo, hi = float(np.quantile(boot, 0.05)), float(np.quantile(boot, 0.95))
    # Scale context: the scalar arm's pooled CVaR level over the same included legs/seeds.
    scalar_levels = [
        np.asarray(leg_results[k]["cvar_b_per_seed"], dtype=float)
        for k in labels if "cvar_b_per_seed" in leg_results[k]
    ]
    if scalar_levels and len({a.size for a in scalar_levels}) != 1:
        raise ValueError("cvar_b_per_seed lengths differ across included legs (seed alignment broken)")
    scalar_level = float(np.mean(np.vstack(scalar_levels))) if scalar_levels else None
    if scalar_level is not None and not np.isfinite(scalar_level):
        scalar_level = None  # row 30b: NaN would silently poison relative_to_scalar_cvar
    rel = (
        {"estimate": est / abs(scalar_level), "ci_low": lo / abs(scalar_level),
         "ci_high": hi / abs(scalar_level)}
        if scalar_level not in (None, 0.0) else None
    )
    return {
        "estimate": est, "ci_low": lo, "ci_high": hi, "ci_level": 0.90,
        "n_legs": len(labels), "n_seeds": int(n_seeds), "n_boot": int(n_boot),
        "labels": labels,
        "scalar_arm_pooled_cvar": scalar_level,
        "relative_to_scalar_cvar": rel,
    }


def pair_did(
    top_dist: np.ndarray,
    top_scalar: np.ndarray,
    bottom_dist: np.ndarray,
    bottom_scalar: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 20260720,
) -> dict[str, Any]:
    """The family-pair difference-in-differences on the COMMON floor-30 CRN seed subset.

    estimate = mean[(top_dist − top_scalar) − (bottom_dist − bottom_scalar)] — the content-effect
    × capability interaction with the vendor held constant. The bootstrap resamples SEED INDICES
    i.i.d. and applies the SAME draw to all four arrays (seed-paired: common seed-level shocks
    cancel inside each resample exactly as in the estimator).
    """
    arrs = [np.asarray(a, dtype=float) for a in (top_dist, top_scalar, bottom_dist, bottom_scalar)]
    n = arrs[0].size
    if any(a.ndim != 1 or a.size != n for a in arrs):
        raise ValueError("all four per-seed arrays must be 1-D and seed-aligned (common subset)")
    if n < 2:
        raise ValueError(f"pair_did needs >=2 common seeds (got {n})")
    per_seed = (arrs[0] - arrs[1]) - (arrs[2] - arrs[3])
    estimate = float(per_seed.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    boot = per_seed[idx].mean(axis=1)
    return {
        "estimate": estimate,
        "ci_low": float(np.quantile(boot, 0.05)),
        "ci_high": float(np.quantile(boot, 0.95)),   # 90% CI, mirroring the TOST convention
        "n_seeds": int(n),
        "n_boot": int(n_boot),
        "per_seed": per_seed.tolist(),
    }


def leg_family_bh(pvalues: dict[str, float], q: float = 0.05) -> dict[str, Any]:
    """BH across the report-only leg family (delegates to the frozen implementation)."""
    labels = sorted(pvalues)
    rejected = benjamini_hochberg([pvalues[k] for k in labels], q=q)
    return {"labels": labels, "rejected": dict(zip(labels, [bool(r) for r in rejected])), "q": q}


def generation_indexed_responsiveness(
    candidates: list[dict[str, Any]],
    n_boot: int = 2_000,
    seed: int = 20260720,
) -> dict[str, Any]:
    """SQ1 per generation: does the model LEARN to use the feedback within the loop? (registered)

    Each candidate dict carries ``generation`` (int), ``x`` (the fed-signal summary), and ``m``
    (the authored-code feature). Reuses the frozen SQ1 estimator per generation, then summarises
    the trend as Spearman(rho, generation).
    """
    by_gen: dict[int, list[dict[str, Any]]] = {}
    for c in candidates:
        by_gen.setdefault(int(c["generation"]), []).append(c)
    out: dict[int, dict[str, Any]] = {}
    for gen in sorted(by_gen):
        rows = by_gen[gen]
        x = np.asarray([r["x"] for r in rows], dtype=float)
        m = np.asarray([r["m"] for r in rows], dtype=float)
        out[gen] = responsiveness(x, m, n_boot=n_boot, rng=np.random.default_rng(seed))
        out[gen]["n"] = len(rows)
    gens = sorted(out)
    rhos = [out[g].get("coef") for g in gens]
    trend = None
    valid = [(g, r) for g, r in zip(gens, rhos) if r is not None and np.isfinite(r)]
    if len(valid) >= 3:
        trend = float(stats.spearmanr([g for g, _ in valid], [r for _, r in valid]).statistic)
    return {"per_generation": out, "trend_spearman": trend, "generations": gens}


def capability_regression(
    points: list[dict[str, Any]],
    anchor: str = "external",
) -> dict[str, Any]:
    """Spearman of per-leg responsiveness vs a capability anchor (registered: primary=external).

    Each point: {"label", "responsiveness", "external": float, "m2": float}. ``anchor`` selects
    the x-axis; the output flags which anchor is the registered PRIMARY.
    """
    if anchor not in ("external", "m2"):
        raise ValueError(f"anchor must be 'external' or 'm2', got {anchor!r}")
    pts = [p for p in points if p.get(anchor) is not None and p.get("responsiveness") is not None]
    if len(pts) < 3:
        return {"anchor": anchor, "n": len(pts), "rho": None,
                "note": "fewer than 3 usable points", "primary_anchor": "external"}
    x = np.asarray([float(p[anchor]) for p in pts])
    y = np.asarray([float(p["responsiveness"]) for p in pts])
    res = stats.spearmanr(x, y)
    return {
        "anchor": anchor,
        "primary_anchor": "external",   # frozen: external composite primary, M2 secondary
        "n": len(pts),
        "rho": float(res.statistic),
        "p_value": float(res.pvalue),
        "labels": [str(p.get("label")) for p in pts],
    }
