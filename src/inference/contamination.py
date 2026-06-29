"""N3 contamination harness — the named-vs-blinded A/B (PREREGISTRATION §8, deep-research §5.9).

What this measures, and what it deliberately does NOT
-----------------------------------------------------
The leakage risk for an LLM *reward designer* (NOT a forecaster) is **concept
leakage**: priors that CVaR/drawdown-penalised reward terms work on famous regimes,
plus implicit coefficient-tuning toward crisis dates the model has read about. It is
**not** outcome recall — the reward sees only anonymised arrays (no tickers, no dates;
``src/reward/contract.py`` lines 18-19). So the question is not "did the model memorise
a label" but "does revealing the identity of the data change the reward code it writes,
*holding the data itself fixed*."

The **named-vs-blinded A/B** answers exactly that and is the load-bearing test:

* **Factor 1 — identity revelation.** NAMED (real dates / ticker labels / regime names
  exposed in the prompt) vs BLINDED (the anonymised arrays the live pipeline actually
  feeds).
* **Factor 2 — regime content held FIXED.** Both arms see the *identical* numerical
  arrays; only the surrounding labels differ. Because the data is identical across arms,
  the comparison **escapes the membership-inference (MIA) distribution-shift confound**
  that invalidates almost all post-hoc contamination claims (Meeus et al. 2024, "SoK:
  MIAs on LLMs are Rushing Nowhere", arXiv:2406.17975: post-hoc member/non-member splits
  carry a strong distribution shift; Duan et al. 2024, arXiv:2402.07841: MIAs barely beat
  chance at frontier scale). Here the only thing that varies is the label, so a measured
  difference is *concept use*, not a distribution artefact.

The dependent variable is the **reward-coefficient vector** ``theta`` extracted from each
authored reward (a probe-regression coefficient vector is the primary instrument; a static
AST/structural read corroborates), plus a **structural binary vector** ``s`` (which reward
*motifs* appear: a CVaR term, a drawdown term, a turnover penalty, ...).

**The primary statistic is a paired TOST equivalence test, per coefficient.** For each
coefficient ``k`` we test whether the named-minus-blinded paired mean difference lies inside
``[-Delta_k, +Delta_k]`` with ``Delta_k = equivalence_sd_frac * (within-arm seed SD)``
(Lakens 2017; Schuirmann 1987). Rejecting *both* one-sided nulls licenses the positive claim
**"the coefficient is invariant to identity revelation within +/-Delta"** — i.e.
**bounded contamination stated as a positive result**, not the absence-of-evidence non-result
a bare two-sided p>0.05 would be. (TOST p-value = max of the two one-sided p-values; the 90%
CI lies inside the bounds at the 5% level.)

What is LOAD-BEARING vs THEATRE (state this in the write-up; do not blur it)
---------------------------------------------------------------------------
* LOAD-BEARING: the named-vs-blinded A/B (this module's primary), post-cutoff persistence of
  the H2 gap (:func:`post_cutoff_persistence`), and a cutoff-dated second model
  (:func:`cross_model_disagreement`). These have a realised, comparable target.
* THEATRE / CATEGORY ERROR (deliberately NOT implemented here): Min-K%-prob or
  loss/perplexity attacks "on the reward function". There is no realised target token
  sequence to score — the reward is *generated*, not *recalled* — so a per-token membership
  score has no estimand. MIA on Opus closed-weight logprobs (only top-k logprobs exposed,
  no reference model) barely beats chance (Duan 2024). Structural blinding itself is
  **mixed-effectiveness hygiene** (Glasserman-Lin 2023: anonymisation removes *distraction*,
  not look-ahead; Sarkar-Vafa: masking can be defeated by reconstruction) — *necessary*, but
  its *sufficiency* is the empirical question this A/B answers, not an assumption.

Degrades gracefully
-------------------
Every public function returns a structured ``dict`` and **never fabricates**: when the
campaign artefacts (named/blinded coefficient archives, pre/post-cutoff per-seed records,
a second model's coefficients) are absent, it returns ``{"status": "no_data", ...}`` with the
reason. The wiring contract for the campaign is documented with the arm roster.

References
----------
Lakens (2017) *Equivalence Tests* (TOST); Schuirmann (1987) JPB&P 15:657 (two one-sided
tests); Duan et al. (2024) COLM, arXiv:2402.07841 (MIAs ~ chance); Meeus et al. (2024) "SoK"
arXiv:2406.17975 (MIA distribution-shift confound); Glasserman & Lin (2023) arXiv:2309.17322;
McNemar (1947). Reuses ``src.inference.bootstrap.paired_seed_difference_test`` (rliable;
Agarwal et al. 2021) for the downstream OOS-gap legs.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

__all__ = [
    "DEFAULT_EQUIVALENCE_SD_FRAC",
    "paired_tost",
    "named_vs_blinded_tost",
    "coefficient_mahalanobis_permutation",
    "structural_mcnemar",
    "named_vs_blinded_structural",
    "named_vs_blinded_oos_gap",
    "post_cutoff_persistence",
    "cross_model_disagreement",
    "contamination_report",
]

#: Equivalence half-width as a fraction of the within-arm per-seed SD of a coefficient
#: (deep-research §5.9: ``Delta = 0.5 * within-arm-seed-SD``). Pre-registered; overridable
#: per call. A coefficient whose named-vs-blinded shift is bounded by half its own
#: seed-to-seed noise is "invariant for practical purposes".
DEFAULT_EQUIVALENCE_SD_FRAC: float = 0.5

#: Achieved null-equivalence power below which a per-coefficient TOST is flagged ``underpowered`` (2026-07-09).
#: A NON-equivalent verdict on an underpowered coefficient is UNINFORMATIVE (the design cannot resolve
#: equivalence at this n/Delta), NOT evidence of contamination — so ``all_equivalent=False`` can be read as
#: low power vs a real effect. 0.8 is the conventional power floor.
TOST_UNDERPOWER_FLOOR: float = 0.8


# ---------------------------------------------------------------------------
# Primary: paired TOST equivalence (the "bounded contamination as a positive claim")
# ---------------------------------------------------------------------------
def paired_tost(
    named: np.ndarray,
    blinded: np.ndarray,
    *,
    low: float,
    high: float,
) -> dict[str, float | bool]:
    """One paired TOST (two one-sided t-tests) for equivalence of a paired mean difference.

    Tests ``H01: mean(named - blinded) <= low`` and ``H02: mean(named - blinded) >= high``
    on the paired differences ``d = named - blinded`` (paired element-wise by SEED). The
    equivalence p-value is the **maximum** of the two one-sided p-values; equivalence at level
    ``alpha`` is declared (``equivalent=True``) iff that p-value ``< alpha`` is decided by the
    caller, but the convention is exposed via the 90% CI: ``equivalent`` here means *both*
    one-sided tests reject at 0.05 (i.e. the 90% CI ``[ci_low, ci_high]`` lies strictly inside
    ``(low, high)``), the standard TOST decision (Lakens 2017; Schuirmann 1987).

    Parameters
    ----------
    named, blinded : np.ndarray
        Paired samples (same length; element ``i`` is the same seed). For the A/B these are a
        single coefficient's value across seeds in the NAMED vs BLINDED arm.
    low, high : float
        Equivalence bounds for the mean difference (typically ``-Delta`` and ``+Delta``).

    Returns
    -------
    dict
        ``{"mean_diff", "p_tost", "p_lower", "p_upper", "ci_low", "ci_high", "t_lower",
        "t_upper", "df", "equivalent", "low", "high"}``. ``p_lower`` tests ``H01`` (diff above
        ``low``); ``p_upper`` tests ``H02`` (diff below ``high``). The 90% CI is the
        equi-tailed t interval used in the standard TOST<->CI duality.

    Raises
    ------
    ValueError
        If the inputs differ in length, have ``< 2`` pairs, or ``low >= high``.
    """
    a = np.asarray(named, dtype=float).ravel()
    b = np.asarray(blinded, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError("named and blinded must be paired (same shape)")
    if a.size < 2:
        raise ValueError("paired_tost needs >= 2 pairs")
    if not (low < high):
        raise ValueError("require low < high (an equivalence interval)")

    d = a - b
    n = d.size
    df = n - 1
    mean_diff = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(n)

    if se == 0.0 or not np.isfinite(se):
        # Degenerate: every pair identical (se=0). The difference is exactly mean_diff with no
        # sampling error; it is equivalent iff that point sits strictly inside the bounds.
        equivalent = bool(low < mean_diff < high)
        p_tost = 0.0 if equivalent else 1.0
        return {
            "mean_diff": mean_diff,
            "p_tost": p_tost,
            "p_lower": p_tost,
            "p_upper": p_tost,
            "ci_low": mean_diff,
            "ci_high": mean_diff,
            "t_lower": float("inf"),
            "t_upper": float("inf"),
            "df": float(df),
            "equivalent": equivalent,
            "low": float(low),
            "high": float(high),
        }

    # H01: diff <= low   -> upper-tail one-sided test of (diff - low)/se  (reject => diff > low)
    t_lower = (mean_diff - low) / se
    p_lower = float(stats.t.sf(t_lower, df))
    # H02: diff >= high  -> lower-tail one-sided test of (diff - high)/se (reject => diff < high)
    t_upper = (mean_diff - high) / se
    p_upper = float(stats.t.cdf(t_upper, df))

    p_tost = max(p_lower, p_upper)
    # 90% equi-tailed t CI (the 1-2*alpha interval whose containment in (low, high) is the
    # TOST decision at alpha=0.05).
    tcrit = float(stats.t.ppf(0.95, df))
    ci_low = mean_diff - tcrit * se
    ci_high = mean_diff + tcrit * se
    equivalent = bool(ci_low > low and ci_high < high)

    return {
        "mean_diff": mean_diff,
        "p_tost": p_tost,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "t_lower": float(t_lower),
        "t_upper": float(t_upper),
        "df": float(df),
        "equivalent": equivalent,
        "low": float(low),
        "high": float(high),
    }


def named_vs_blinded_tost(
    named_coeffs: np.ndarray,
    blinded_coeffs: np.ndarray,
    *,
    coefficient_names: Sequence[str] | None = None,
    equivalence_sd_frac: float = DEFAULT_EQUIVALENCE_SD_FRAC,
    equivalence_abs: Sequence[float] | None = None,
) -> dict[str, Any]:
    """PRIMARY A/B statistic: per-coefficient paired TOST, named vs blinded.

    ``named_coeffs`` / ``blinded_coeffs`` are ``(n_seeds, n_coeffs)`` coefficient matrices: row
    = seed (paired across the two arms), column = a reward-coefficient ``theta_k`` (probe-
    regression coefficient primary; AST corroboration). The arms are the SAME data with only
    the identity labels differing, so a coefficient that is **equivalent within +/-Delta_k**
    demonstrates **bounded concept-contamination as a positive claim**.

    The per-coefficient bound is ``Delta_k = equivalence_sd_frac * sd_k`` where ``sd_k`` is the
    pooled within-arm seed SD of coefficient ``k`` (deep-research §5.9: half the seed-to-seed
    noise) — UNLESS an explicit ``equivalence_abs`` vector is supplied (pre-registered absolute
    bounds), which then overrides the data-driven width.

    POWER WARNING (load-bearing limitation, not a bug). With ``Delta = 0.5 * SD`` the equivalence
    bound is TIGHT: at the tier-0 floor (n=30) the 90% CI half-width on the paired
    mean difference is comparable to ``Delta``, so a genuinely null (label-irrelevant) coefficient
    will typically NOT clear the bound — a measured ``all_equivalent = False`` at n=30 is
    underpowered, NOT evidence of contamination. The named-vs-blinded A/B is a CHEAP dedicated
    sub-experiment (one reward authoring + one cheap eval per seed; it does NOT re-run the 50k-step
    main training), so it should run more seeds than the tier-0 floor (empirically ~150-200 seeds
    give >0.95 power for the all-coefficient claim at ``Delta = 0.5 * SD``). Report the per-
    coefficient CIs and the achieved power, and pre-register either the seed count or a wider
    ``Delta`` — do not silently read a low-power non-rejection as contamination.

    Returns
    -------
    dict
        ``{"status": "ok", "n_seeds", "n_coeffs", "per_coefficient": [ {name, delta, sd_within,
        mean_diff, p_tost, ci_low, ci_high, equivalent}, ... ], "n_equivalent", "all_equivalent",
        "fraction_equivalent"}``. Degenerate columns (zero within-arm SD *and* a zero
        pre-registered bound) are reported with ``"delta": 0.0`` and ``equivalent`` decided by an
        exact-zero difference. Returns ``{"status": "no_data", ...}`` if either matrix is empty.

    Notes
    -----
    No multiplicity correction is applied to the *equivalence* decisions here: TOST controls the
    per-test false-equivalence (Type-I-for-equivalence) rate, and the headline framing is
    per-coefficient invariance. A family-level summary (``fraction_equivalent``) is reported; if
    a single omnibus verdict is wanted, use :func:`coefficient_mahalanobis_permutation` (a joint
    DIFFERENCE test, the complementary direction).
    """
    A = np.atleast_2d(np.asarray(named_coeffs, dtype=float))
    B = np.atleast_2d(np.asarray(blinded_coeffs, dtype=float))
    if A.size == 0 or B.size == 0:
        return {"status": "no_data", "reason": "empty coefficient matrix"}
    if A.shape != B.shape:
        return {
            "status": "no_data",
            "reason": f"named {A.shape} and blinded {B.shape} coefficient matrices must match",
        }
    n_seeds, n_coeffs = A.shape
    if n_seeds < 2:
        return {"status": "no_data", "reason": "need >= 2 paired seeds for TOST"}

    names = (
        list(coefficient_names)
        if coefficient_names is not None
        else [f"theta_{k}" for k in range(n_coeffs)]
    )
    if len(names) != n_coeffs:
        raise ValueError("coefficient_names length must equal n_coeffs")
    if equivalence_abs is not None and len(equivalence_abs) != n_coeffs:
        raise ValueError("equivalence_abs length must equal n_coeffs")

    # t critical for the 90% CI (the TOST-convention interval; df = n_seeds - 1), reused for the achieved-
    # power diagnostic below so it matches paired_tost's own interval exactly.
    t_crit = float(stats.t.ppf(0.95, max(1, n_seeds - 1)))

    per_coeff: list[dict[str, Any]] = []
    n_equiv = 0
    for k in range(n_coeffs):
        col_a, col_b = A[:, k], B[:, k]
        # Pooled within-arm seed SD (the seed-to-seed noise of this coefficient).
        sd_within = float(
            math.sqrt(0.5 * (np.var(col_a, ddof=1) + np.var(col_b, ddof=1)))
        )
        if equivalence_abs is not None:
            delta = float(equivalence_abs[k])
        else:
            delta = float(equivalence_sd_frac * sd_within)
        if delta <= 0.0:
            # No usable scale: declare equivalence only on an exactly-identical column.
            diff = float((col_a - col_b).mean())
            equiv = bool(np.allclose(col_a, col_b))
            per_coeff.append(
                {
                    "name": names[k],
                    "delta": 0.0,
                    "sd_within": sd_within,
                    "mean_diff": diff,
                    "p_tost": 0.0 if equiv else 1.0,
                    "ci_low": diff,
                    "ci_high": diff,
                    "ci_halfwidth": 0.0,
                    "equiv_power_null": float("nan"),  # power undefined for a zero-delta exact-match test
                    "equivalent": equiv,
                    "decisively_different": bool(not equiv),  # a non-identical zero-width column IS a real diff
                    "underpowered": False,  # zero-width exact-match case: a difference is real, not underpower
                    "degenerate_zero_width": True,
                }
            )
            n_equiv += int(equiv)
            continue
        res = paired_tost(col_a, col_b, low=-delta, high=delta)
        # THREE-WAY TOST OUTCOME (Lakens 2017) operationalising the module's n=30 underpower caveat (was
        # documented-only; surfaced 2026-07-09). Given the 90% CI [ci_low, ci_high]:
        #   * EQUIVALENT           — CI ⊂ (-delta, delta): bounded difference (a positive claim).
        #   * DECISIVELY_DIFFERENT — CI entirely BEYOND ±delta (ci_low>=delta or ci_high<=-delta): a real
        #     effect LARGER than the SESOI — decisive non-equivalence, NOT a power artefact.
        #   * UNDERPOWERED (=inconclusive) — neither: the CI STRADDLES a ±delta boundary, so the design
        #     cannot resolve equivalence. A non-equivalent verdict here is UNINFORMATIVE (low power at this
        #     n/Delta), NOT contamination — so an analyst cannot misread all_equivalent=False as a positive
        #     contamination finding. This crucially does NOT mis-flag a decisively-contaminated coefficient
        #     (whose CI sits far beyond ±delta) as "underpowered" — that is decisively_different.
        ci_low, ci_high = float(res["ci_low"]), float(res["ci_high"])
        ci_halfwidth = float((ci_high - ci_low) / 2.0)
        equivalent = bool(res["equivalent"])
        decisively_different = bool(ci_low >= delta or ci_high <= -delta)
        underpowered = bool(not equivalent and not decisively_different)
        # Reported companion: P(a TRUE-null coefficient would be declared equivalent at this n/Delta) =
        # 2·Phi(t_crit·(delta/h - 1)) - 1 for half-width h < delta, else 0 — a continuous resolving-power
        # readout (min over coeffs vs TOST_UNDERPOWER_FLOOR summarises whether the A/B can resolve equivalence).
        if ci_halfwidth <= 0.0:
            equiv_power_null = 1.0
        elif ci_halfwidth >= delta:
            equiv_power_null = 0.0
        else:
            equiv_power_null = float(2.0 * stats.norm.cdf(t_crit * (delta / ci_halfwidth - 1.0)) - 1.0)
        per_coeff.append(
            {
                "name": names[k],
                "delta": delta,
                "sd_within": sd_within,
                "mean_diff": res["mean_diff"],
                "p_tost": res["p_tost"],
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ci_halfwidth": ci_halfwidth,
                "equiv_power_null": equiv_power_null,
                "equivalent": equivalent,
                "decisively_different": decisively_different,
                "underpowered": underpowered,
            }
        )
        n_equiv += int(equivalent)

    # Family-level power summary (2026-07-09): so a consumer reading ``all_equivalent=False`` can tell
    # whether the non-equivalence is DECISIVE or merely UNDERPOWERED at this n/Delta. ``any_underpowered``
    # True ⟹ at least one coefficient has achieved null-equivalence power below TOST_UNDERPOWER_FLOOR, so a
    # non-equivalent verdict must NOT be read as a positive contamination finding (the POWER WARNING, now
    # machine-readable). ``min_equiv_power_null`` is the weakest coefficient's power (finite entries only).
    n_underpowered = int(sum(1 for c in per_coeff if c.get("underpowered")))
    _finite_powers = [
        float(c["equiv_power_null"])
        for c in per_coeff
        if isinstance(c.get("equiv_power_null"), float) and np.isfinite(c["equiv_power_null"])
    ]
    return {
        "status": "ok",
        "n_seeds": int(n_seeds),
        "n_coeffs": int(n_coeffs),
        "equivalence_sd_frac": float(equivalence_sd_frac),
        "per_coefficient": per_coeff,
        "n_equivalent": int(n_equiv),
        "all_equivalent": bool(n_equiv == n_coeffs),
        "fraction_equivalent": float(n_equiv / n_coeffs),
        "n_underpowered": n_underpowered,
        "fraction_underpowered": float(n_underpowered / n_coeffs),
        "any_underpowered": bool(n_underpowered > 0),
        "min_equiv_power_null": (min(_finite_powers) if _finite_powers else float("nan")),
        "power_floor": float(TOST_UNDERPOWER_FLOOR),
    }


# ---------------------------------------------------------------------------
# Secondary: a joint DIFFERENCE test (the complementary direction to TOST)
# ---------------------------------------------------------------------------
def coefficient_mahalanobis_permutation(
    named_coeffs: np.ndarray,
    blinded_coeffs: np.ndarray,
    *,
    n_perm: int = 5000,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Permutation test on the Mahalanobis distance between the named & blinded coeff CENTROIDS.

    A *difference*-direction omnibus complement to the per-coefficient TOST: the observed
    statistic is the Mahalanobis distance between the mean coefficient vectors of the two arms,
    using the pooled within-arm covariance (shrunk to the diagonal when singular). Under the
    null of label-irrelevance the NAMED/BLINDED tag is exchangeable *within each seed* (the data
    is identical), so the reference distribution permutes the named/blinded assignment per seed
    (a paired/sign-flip permutation). A small p-value is *evidence of contamination*; a large
    p-value is the absence of evidence (NOT equivalence — that is the TOST's job).

    Returns ``{"status": "ok", "mahalanobis", "pvalue", "n_perm", "n_seeds", "n_coeffs"}`` or a
    ``{"status": "no_data", ...}`` mapping. Reuses only numpy/scipy.
    """
    A = np.atleast_2d(np.asarray(named_coeffs, dtype=float))
    B = np.atleast_2d(np.asarray(blinded_coeffs, dtype=float))
    if A.size == 0 or B.size == 0 or A.shape != B.shape:
        return {"status": "no_data", "reason": "empty or mismatched coefficient matrices"}
    n_seeds, n_coeffs = A.shape
    if n_seeds < 2:
        return {"status": "no_data", "reason": "need >= 2 paired seeds"}
    if rng is None:
        rng = np.random.default_rng(0)

    def _maha(named: np.ndarray, blinded: np.ndarray) -> float:
        diff_mean = named.mean(axis=0) - blinded.mean(axis=0)
        pooled = np.vstack(
            [named - named.mean(axis=0), blinded - blinded.mean(axis=0)]
        )
        cov = np.cov(pooled, rowvar=False)
        cov = np.atleast_2d(cov)
        # Ridge-shrink for a singular/ill-conditioned covariance (few seeds, many coeffs).
        ridge = 1e-8 * float(np.trace(cov)) / max(1, cov.shape[0]) + 1e-12
        cov = cov + ridge * np.eye(cov.shape[0])
        try:
            sol = np.linalg.solve(cov, diff_mean)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(cov, diff_mean, rcond=None)[0]
        return float(math.sqrt(max(0.0, float(diff_mean @ sol))))

    obs = _maha(A, B)
    # Paired sign-flip permutation: per seed, swap (named, blinded) with prob 1/2.
    count = 1  # include the observed (standard +1 permutation-test convention)
    stacked = np.stack([A, B], axis=0)  # (2, n_seeds, n_coeffs)
    for _ in range(n_perm):
        flip = rng.random(n_seeds) < 0.5
        perm_named = np.where(flip[:, None], stacked[1], stacked[0])
        perm_blind = np.where(flip[:, None], stacked[0], stacked[1])
        if _maha(perm_named, perm_blind) >= obs:
            count += 1
    pvalue = count / (n_perm + 1)
    return {
        "status": "ok",
        "mahalanobis": obs,
        "pvalue": float(pvalue),
        "n_perm": int(n_perm),
        "n_seeds": int(n_seeds),
        "n_coeffs": int(n_coeffs),
    }


def structural_mcnemar(
    named_struct: np.ndarray,
    blinded_struct: np.ndarray,
    *,
    continuity: bool = True,
) -> dict[str, Any]:
    """McNemar test on a single binary structural motif across paired (named, blinded) rewards.

    ``named_struct`` / ``blinded_struct`` are 0/1 vectors over seeds: did the authored reward
    contain a given motif (e.g. a CVaR term)? McNemar's test on the discordant pairs (b =
    named-only, c = blinded-only) tests whether identity revelation flips motif *presence*. A
    non-significant result is consistent with structural invariance (corroborating the TOST);
    a significant one flags identity-driven structural change (McNemar 1947).

    Returns ``{"status": "ok", "n01", "n10", "statistic", "pvalue", "n_seeds"}`` (``n01`` =
    named-absent/blinded-present, ``n10`` = named-present/blinded-absent) or ``no_data``. When
    there are no discordant pairs the p-value is 1.0 (perfect structural agreement).
    """
    a = np.asarray(named_struct).ravel().astype(int)
    b = np.asarray(blinded_struct).ravel().astype(int)
    if a.size == 0 or a.shape != b.shape:
        return {"status": "no_data", "reason": "empty or mismatched structural vectors"}
    if not (set(np.unique(a)) <= {0, 1} and set(np.unique(b)) <= {0, 1}):
        raise ValueError("structural vectors must be binary (0/1)")
    n10 = int(np.sum((a == 1) & (b == 0)))  # named-only
    n01 = int(np.sum((a == 0) & (b == 1)))  # blinded-only
    disc = n10 + n01
    if disc == 0:
        return {
            "status": "ok",
            "n01": n01,
            "n10": n10,
            "statistic": 0.0,
            "pvalue": 1.0,
            "n_seeds": int(a.size),
            "note": "no discordant pairs (perfect structural agreement)",
        }
    # Exact binomial McNemar for small discordant counts (robust); chi-square otherwise.
    if disc < 25:
        pvalue = float(min(1.0, 2.0 * stats.binom.cdf(min(n10, n01), disc, 0.5)))
        statistic = float(min(n10, n01))
    else:
        corr = 1.0 if continuity else 0.0
        statistic = float((abs(n10 - n01) - corr) ** 2 / disc)
        pvalue = float(stats.chi2.sf(statistic, 1))
    return {
        "status": "ok",
        "n01": n01,
        "n10": n10,
        "statistic": statistic,
        "pvalue": pvalue,
        "n_seeds": int(a.size),
    }


def named_vs_blinded_structural(
    named_sources: Sequence[str],
    blinded_sources: Sequence[str],
    *,
    depth: int = 4,
    n_perm: int = 2000,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """AST-STRUCTURAL arm of the named-vs-blinded A/B: does identity revelation change the reward CODE STRUCTURE?

    The coefficient TOST and per-motif McNemar above test *what coefficients / which motifs* the reward
    carries; this tests the **whole-program structure**, identifier- and literal-invariantly, by reusing the
    canonical AST-shape similarity (:mod:`src.inference.reward_code_distance` — variable names and constant
    values are discarded, so cosmetic renaming is invisible). It is the structural complement that closes the
    gap the audits flagged: a placebo can echo the tail *tokens* yet still write a different *program*.

    Paired by seed: ``named_sources[i]`` and ``blinded_sources[i]`` are the rewards authored for the SAME seed
    on the SAME numerical arrays under the NAMED vs BLINDED labelling. Two quantities are contrasted:

    * ``paired_mean`` — mean same-seed ``sim(named_i, blinded_i)``: the structural agreement that survives
      revealing identity, holding the data fixed;
    * ``within_blinded_mean`` — mean cross-seed ``sim(blinded_i, blinded_j)`` (``i<j``): the natural
      seed-to-seed structural variation under a FIXED labelling — the noise floor.

    If the label is irrelevant to program structure, revealing identity perturbs structure NO MORE than
    reseeding does, so ``paired_mean >= within_blinded_mean`` (``data_locked``). A re-pairing permutation test
    (random named→blinded bijections) reports ``p_random_pairing_matches`` — the probability a random pairing
    is as structurally tight as the true same-seed pairing; a SMALL value means the same-data programs are
    distinctly tighter than chance, i.e. structure is driven by the shared data, **evidence AGAINST structural
    contamination**. A ``paired_mean`` that collapses to the noise floor (large p) would instead flag
    identity-driven structure. Report-only, DISJOINT from the frozen m=6 family; deterministic (no model/GPU).

    Returns ``{"status": "ok", "n_seeds", "paired_mean", "within_blinded_mean", "structural_gap",
    "data_locked", "p_random_pairing_matches", "depth"}`` or ``{"status": "no_data", ...}``.
    """
    from itertools import combinations

    from src.inference.reward_code_distance import canonical_shapes, has_structure, jaccard

    named = [str(s) for s in named_sources]
    blinded = [str(s) for s in blinded_sources]
    if len(named) != len(blinded):
        return {"status": "no_data", "reason": "named and blinded sources must be paired (same length)"}
    n = len(named)
    if n < 2:
        return {"status": "no_data", "reason": "need >= 2 paired seeds for the structural A/B"}
    if rng is None:
        rng = np.random.default_rng(0)

    nshapes = [canonical_shapes(s, depth) for s in named]
    bshapes = [canonical_shapes(s, depth) for s in blinded]
    # Exclude STRUCTURELESS sources BEFORE any similarity: jaccard(empty, empty) == 1.0, so a pair of
    # them would score as "perfectly structurally locked", inflating paired_mean and corrupting the
    # re-pairing permutation null (mirrors the P7c fix in
    # reward_code_distance.reward_code_structure_report; audit 2026-07-02).
    #
    # NOTE: WIDENED 2026-07-27 (deep review #117). The test was `nshapes[i] and bshapes[i]`, i.e. NON-EMPTY
    # only — which catches an UNPARSEABLE source (`canonical_shapes` returns the empty set) but NOT an
    # EMPTY or comment-only one: `ast.parse("")` succeeds and walks exactly one bodiless `Module`, so
    # its shape-set is the TRUTHY singleton {"Module"} (MEASURED: "" / "# comment" / whitespace all
    # return {'Module'}, and jaccard between two of them is 1.0). Such a pair therefore survived the
    # very filter written to remove it. `reward_taxonomy._signature` already guards `shapes <=
    # {"Module"}`; this module did not — the same defect fixed in one place and not the other.
    #
    # It biased EVERY output toward the reassuring answer, which is why it matters on a CONTAMINATION
    # instrument: a degenerate pair contributes 1.0 to `paired` but only ~1/|shapes| to `within` (vs a
    # real program) and to the re-pairing null — so paired_mean inflates, within_blinded_mean deflates,
    # `structural_gap` grows from both ends, `data_locked` biases TRUE and `p_random_pairing_matches`
    # biases SMALL, i.e. straight toward "evidence AGAINST structural contamination". And empty
    # authored code is MEASURED here, not hypothetical (the 2026-07-25 unpinned-thinking finding had
    # several legs author empty code at the frozen config).
    usable = [i for i in range(n) if has_structure(nshapes[i]) and has_structure(bshapes[i])]
    n_unparseable = n - len(usable)
    if len(usable) < 2:
        return {
            "status": "no_data",
            "reason": (
                f"fewer than 2 usable paired seeds after excluding STRUCTURELESS sources "
                f"({n_unparseable} of {n} pairs unparseable, empty or comment-only)"
            ),
            "n_unparseable_pairs": int(n_unparseable),
        }
    nshapes = [nshapes[i] for i in usable]
    bshapes = [bshapes[i] for i in usable]
    n = len(usable)

    paired = np.array([jaccard(nshapes[i], bshapes[i]) for i in range(n)], dtype=float)
    paired_mean = float(paired.mean())
    within = np.array(
        [jaccard(bshapes[i], bshapes[j]) for i, j in combinations(range(n), 2)], dtype=float
    )
    within_mean = float(within.mean()) if within.size else float("nan")
    gap = paired_mean - within_mean

    # Re-pairing permutation: random bijection named->blinded; how often is it as tight as the true pairing?
    ge = 1  # +1 includes the observed (standard permutation-test convention)
    idx = np.arange(n)
    for _ in range(n_perm):
        perm = rng.permutation(idx)
        pm = float(np.mean([jaccard(nshapes[i], bshapes[perm[i]]) for i in range(n)]))
        if pm >= paired_mean - 1e-12:
            ge += 1
    p_random = ge / (n_perm + 1)

    return {
        "status": "ok",
        "n_seeds": int(n),
        "n_unparseable_pairs": int(n_unparseable),
        "paired_mean": paired_mean,
        "within_blinded_mean": within_mean,
        "structural_gap": float(gap),
        "data_locked": bool(paired_mean >= within_mean),
        "p_random_pairing_matches": float(p_random),
        "depth": int(depth),
    }


def named_vs_blinded_oos_gap(
    named_seed_sharpe: Mapping[int, float],
    blinded_seed_sharpe: Mapping[int, float],
    *,
    sesoi: float = 0.05,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Downstream NAMED-minus-BLINDED OOS-performance gap, gated to a SESOI.

    The most concrete A/B endpoint: train the agent on the reward authored under each labelling,
    then compare the per-seed out-of-sample Sharpe. ``*_seed_sharpe`` are ``{seed: sharpe}`` maps
    (paired by shared seed). Reuses :func:`src.inference.bootstrap.paired_seed_difference_test`
    (rliable IQM + paired seed bootstrap), so the across-seed variance is carried. ``equivalent``
    flags whether the TOST-convention 90% bootstrap CI (the 5/95 quantiles of the SAME paired
    draws; two one-sided 5% tests — Lakens 2017, matching every other equivalence flag in the
    stack) lies inside ``[-sesoi, +sesoi]``.

    NOTE: UNITS — read before quoting this margin (2026-07-26 review). The endpoint here is a **Sharpe**
    difference, but the registered ``sesoi = 0.05`` is in **validation-DSR units**
    (``config/preregistration.yaml: inference.sesoi``), and amendment **R104** derives its Sharpe
    equivalent explicitly: ``sesoi_ann_sharpe_equiv = 0.0756`` (``= 0.05 / k``, ``k = 0.6616``
    DSR per ann-Sharpe). The previous wording "0.05 DSR/Sharpe units" conflated the two units that
    R104 exists to separate. Applying ``0.05`` on a Sharpe scale is therefore a **~34% TIGHTER**
    margin than the registered Sharpe equivalent — i.e. **CONSERVATIVE for an equivalence claim**
    (harder to declare equivalence, never easier), so it cannot manufacture a bounded-gap positive.
    The default is deliberately LEFT at 0.05: loosening a margin to 0.0756 on a report-only
    sub-experiment is a design decision for the pre-registration, not a code change — pass ``sesoi``
    explicitly if the Sharpe-equivalent margin is wanted. This endpoint is a report-only A/B
    sub-experiment; the registered confirmatory instrument is ``named_vs_blinded_structural`` (AST).

    The reported ``ci_low``/``ci_high`` remain the 95% interval;
    ``ci90_low``/``ci90_high`` are also reported so the flag is auditable.

    Returns the bootstrap dict augmented with ``{"status", "n_seeds", "sesoi", "ci90_low",
    "ci90_high", "equivalent"}`` or ``{"status": "no_data", ...}`` when fewer than two seeds are
    shared.
    """
    from src.inference.bootstrap import iqm, paired_seed_difference_test

    common = sorted(set(named_seed_sharpe) & set(blinded_seed_sharpe))
    if len(common) < 2:
        return {
            "status": "no_data",
            "reason": "need >= 2 shared seeds for the paired OOS-gap bootstrap",
            "n_seeds": len(common),
        }
    a = np.array([named_seed_sharpe[s] for s in common], dtype=float)
    b = np.array([blinded_seed_sharpe[s] for s in common], dtype=float)
    if rng is None:
        rng = np.random.default_rng(0)
    res = paired_seed_difference_test(a, b, statistic=iqm, n_boot=n_boot, rng=rng, return_draws=True)
    draws = np.asarray(res.pop("draws"), dtype=float)
    # Equivalence FLAG on the TOST 90% convention (two one-sided 5% tests <=> the 90% CI inside the
    # margin; Lakens 2017): every other equivalence flag in the stack uses 90%, so gating this one on
    # the REPORTED 95% CI was stricter than the shared convention and inconsistent with it. The 90%
    # interval comes from the SAME bootstrap draws (no second bootstrap); ci_low/ci_high stay 95%.
    ci90_low, ci90_high = (float(q) for q in np.quantile(draws, [0.05, 0.95]))
    equivalent = bool(ci90_low > -sesoi and ci90_high < sesoi)
    # THREE-WAY TOST outcome (mirrors named_vs_blinded_tost, 2026-07-09) so a NON-equivalent OOS gap is
    # machine-readably attributable to LOW POWER vs a REAL identity-driven performance gap:
    #   * decisively_different — the 90% bootstrap CI lies ENTIRELY beyond ±SESOI (a real gap > SESOI,
    #     i.e. identity revelation moved out-of-sample Sharpe more than the SESOI — decisive contamination);
    #   * underpowered (=inconclusive) — the CI straddles a ±SESOI boundary, so the A/B cannot resolve
    #     equivalence at this n; a non-equivalent verdict here is UNINFORMATIVE, NOT contamination.
    decisively_different = bool(ci90_low >= sesoi or ci90_high <= -sesoi)
    underpowered = bool(not equivalent and not decisively_different)
    out: dict[str, Any] = {"status": "ok", "n_seeds": len(common), "sesoi": float(sesoi)}
    out.update({k: float(v) for k, v in res.items()})
    out["ci90_low"] = ci90_low
    out["ci90_high"] = ci90_high
    out["ci90_halfwidth"] = float((ci90_high - ci90_low) / 2.0)
    out["equivalent"] = equivalent
    out["decisively_different"] = decisively_different
    out["underpowered"] = underpowered
    return out


# ---------------------------------------------------------------------------
# Load-bearing: post-cutoff persistence of the H2 gap
# ---------------------------------------------------------------------------
def post_cutoff_persistence(
    pre_cutoff: Mapping[int, float],
    post_cutoff: Mapping[int, float],
    *,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Is the H2 (distributional-minus-scalar) gap the SAME before and after the model cutoff?

    The reward designer cannot have memorised *post-cutoff* regimes, so if contamination drove
    H2 the gap should SHRINK after the cutoff. ``pre_cutoff`` / ``post_cutoff`` are ``{seed:
    H2_gap}`` maps — the per-seed distributional-minus-scalar metric difference computed on the
    pre- vs post-cutoff slice of the SAME sealed test leg (the split is pre-registered). A paired
    seed bootstrap (reuse :func:`paired_seed_difference_test`) tests whether the pre-cutoff gap
    exceeds the post-cutoff gap.

    CRITICAL caveat (reported, not hidden): the post-cutoff window is SHORT, so this test is
    **underpowered** — a non-significant pre-vs-post difference is weak evidence of persistence,
    not proof of no contamination. The effect size + CI are the deliverable, not the p-value.

    Returns the bootstrap dict + ``{"status", "n_seeds", "pre_iqm", "post_iqm", "caveat"}`` or
    ``{"status": "no_data", ...}``.
    """
    from src.inference.bootstrap import iqm, paired_seed_difference_test

    common = sorted(set(pre_cutoff) & set(post_cutoff))
    if len(common) < 2:
        return {
            "status": "no_data",
            "reason": "need >= 2 shared seeds across the pre/post-cutoff split",
            "n_seeds": len(common),
        }
    pre = np.array([pre_cutoff[s] for s in common], dtype=float)
    post = np.array([post_cutoff[s] for s in common], dtype=float)
    if rng is None:
        rng = np.random.default_rng(0)
    res = paired_seed_difference_test(pre, post, statistic=iqm, n_boot=n_boot, rng=rng)
    ci_low, ci_high = float(res["ci_low"]), float(res["ci_high"])
    # Machine-readable version of the underpower caveat (2026-07-09; mirrors the L90/L91 contamination legs
    # so "report effect+CI, not p" is ACTIONABLE, not just prose). ``effect`` = pre_iqm - post_iqm: a
    # POSITIVE value means the H2 gap SHRANK after the model cutoff — the contamination signal (a memorised
    # advantage fades on unseen post-cutoff data).
    #   * ``gap_shrank_post_cutoff`` — the 95% CI is entirely POSITIVE (ci_low > 0): decisive shrinkage.
    #   * ``underpowered`` — the CI INCLUDES ZERO: a non-significant result CANNOT be read as persistence
    #     (evidence AGAINST contamination); it is equally consistent with LOW POWER on the short post-cutoff
    #     window, so it is uninformative — exactly what the caveat warns, now machine-readable.
    out: dict[str, Any] = {
        "status": "ok",
        "n_seeds": len(common),
        "pre_iqm": float(iqm(pre)),
        "post_iqm": float(iqm(post)),
        "ci_halfwidth": float((ci_high - ci_low) / 2.0),
        "gap_shrank_post_cutoff": bool(ci_low > 0.0),
        "underpowered": bool(ci_low <= 0.0 <= ci_high),
        "caveat": (
            "post-cutoff window is short; this comparison is underpowered (deep-research §5.9). "
            "Report the effect size + CI, not the p-value alone."
        ),
    }
    out.update({k: float(v) for k, v in res.items()})
    return out


# ---------------------------------------------------------------------------
# Load-bearing: cross-model disagreement (a cutoff-dated 2nd model)
# ---------------------------------------------------------------------------
def cross_model_disagreement(
    model_a_coeffs: np.ndarray,
    model_b_coeffs: np.ndarray,
    *,
    coefficient_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compare reward-coefficient centroids from two DIFFERENT models (different cutoffs).

    ``model_a_coeffs`` / ``model_b_coeffs`` are ``(n_seeds_a, n_coeffs)`` and
    ``(n_seeds_b, n_coeffs)`` coefficient matrices for two models (e.g. Opus 5 vs a
    commit-pinned open-weights model with a different cutoff, ideally a ChronoGPT/DatedGPT
    pre-cutoff model). Seed counts NEED NOT match (the models are independent designers), so this
    is an UNPAIRED two-sample comparison of the coefficient means, reported as a per-coefficient
    standardised mean difference (Cohen's d with pooled SD) plus an L2 centroid distance.

    This is a *robustness triangulation*, NOT a hypothesis test: agreement across models with
    different training cutoffs argues the reward design is driven by the (anonymised) data, not by
    one model's memorised priors. Returns ``{"status": "ok", "centroid_l2", "per_coefficient":
    [...], "max_abs_d", "mean_abs_d"}`` or ``no_data``.

    SCOPE / HONESTY (V10): the confirmatory campaign runs only the single Claude model family
    (Sonnet -> Opus, same vendor). A second open-weights model IS pinned (config/llm.yaml
    ``open_weights_check_model`` = Qwen3-Coder via OpenRouter — R71, pinned 2026-07-02) but the
    secondary panel is a SEPARATE, key- and compute-gated run; until it is RUN this function has no
    ``model_b`` to compare and the leg degrades to ``{"status": "no_data", "executed": False}``.
    Plural "models" here therefore describes the SPECIFIED (PENDING/UNEXECUTED) check, not two model
    families that were actually run.
    """
    A = np.atleast_2d(np.asarray(model_a_coeffs, dtype=float))
    B = np.atleast_2d(np.asarray(model_b_coeffs, dtype=float))
    if A.size == 0 or B.size == 0:
        # HONEST DISCLOSURE (V10): an empty second-model matrix means the cutoff-dated second model was
        # NOT run (the R71 secondary panel — config/llm.yaml open_weights_check_model, pinned to
        # Qwen3-Coder via OpenRouter — has not been executed). Report the omission explicitly rather
        # than as a bare "empty matrix" so it is not misread as a pass.
        return {
            "status": "no_data",
            "executed": False,
            "reason": (
                "cross-model check NOT EXECUTED: a model's coefficient matrix is empty (the R71 "
                "second-model panel — config/llm.yaml open_weights_check_model — was not run)."
            ),
        }
    if A.shape[1] != B.shape[1]:
        return {
            "status": "no_data",
            "reason": f"coefficient dim mismatch: {A.shape[1]} vs {B.shape[1]}",
        }
    if A.shape[0] < 2 or B.shape[0] < 2:
        return {"status": "no_data", "reason": "need >= 2 seeds per model for a pooled SD"}
    n_coeffs = A.shape[1]
    names = (
        list(coefficient_names)
        if coefficient_names is not None
        else [f"theta_{k}" for k in range(n_coeffs)]
    )
    if len(names) != n_coeffs:
        raise ValueError("coefficient_names length must equal n_coeffs")

    mean_a, mean_b = A.mean(axis=0), B.mean(axis=0)
    centroid_l2 = float(np.linalg.norm(mean_a - mean_b))
    per_coeff: list[dict[str, Any]] = []
    abs_ds: list[float] = []
    undefined: list[str] = []
    na, nb = A.shape[0], B.shape[0]
    for k in range(n_coeffs):
        va, vb = float(np.var(A[:, k], ddof=1)), float(np.var(B[:, k], ddof=1))
        pooled_sd = math.sqrt(
            ((na - 1) * va + (nb - 1) * vb) / max(1, (na + nb - 2))
        )
        md = float(mean_a[k] - mean_b[k])
        # SCALE-AWARE degeneracy (deep review #118, 2026-07-27). This tested `pooled_sd > 0`, i.e.
        # EXACT zero — but a coefficient that is CONSTANT across seeds in both models does not
        # generally give an exactly-zero variance: `np.var` subtracts a mean that is one ulp off the
        # repeated value, so 12 copies of 0.05 give var 5.3e-35 (sd 7.2e-18) while 12 copies of 1.0
        # give EXACTLY 0 (MEASURED). The old guard therefore fired only when the constant happened to
        # be exactly representable: `lam=1.0` took the degenerate branch (d = inf, excluded from the
        # summaries below) while `cvar_alpha=0.05` took the NORMAL branch and produced a finite
        # d ~ 7.6e15 that passed `isfinite` and entered them — the identical situation handled two
        # opposite ways, decided by float representability. And the constants an LLM actually emits in
        # reward code (0.05, 0.01, 0.1) are exactly the NON-representable ones, so the fragile path was
        # the likely one. The consequence was a corrupted headline: one such column drove `mean_abs_d`
        # — the average cross-model disagreement — to ~2.5e15 in a MEASURED reproduction.
        scale = max(abs(float(mean_a[k])), abs(float(mean_b[k])), 1.0)
        degenerate = pooled_sd <= 1e-12 * scale
        if degenerate:
            # No within-model variation in EITHER model: the standardised difference is 0 when the two
            # constants agree and genuinely unbounded when they differ. `inf` is the honest value.
            d = 0.0 if abs(md) <= 1e-12 * scale else float("inf")
        else:
            d = float(md / pooled_sd)
        per_coeff.append(
            {"name": names[k], "mean_a": float(mean_a[k]), "mean_b": float(mean_b[k]),
             "mean_diff": md, "cohens_d": d, "pooled_sd": pooled_sd,
             "degenerate_zero_variance": bool(degenerate)}
        )
        if np.isfinite(d):
            abs_ds.append(abs(d))
        else:
            undefined.append(names[k])
    finite = np.asarray(abs_ds, dtype=float)
    return {
        "status": "ok",
        "centroid_l2": centroid_l2,
        "per_coefficient": per_coeff,
        # Summaries are over the FINITE d's, so an unbounded one cannot swamp them — but the omission
        # is REPORTED, never silent: `n_undefined_d` counts coefficients that are constant within both
        # models yet differ BETWEEN them, which is a decisive disagreement, not a missing one. Reading
        # `max_abs_d` without it would understate cross-model divergence (#118).
        "max_abs_d": float(finite.max()) if finite.size else float("nan"),
        "mean_abs_d": float(finite.mean()) if finite.size else float("nan"),
        "n_undefined_d": len(undefined),
        "undefined_d_coefficients": undefined,
        "n_seeds_a": int(na),
        "n_seeds_b": int(nb),
    }


# ---------------------------------------------------------------------------
# One-call orchestrator (degrades gracefully on partial data)
# ---------------------------------------------------------------------------
def contamination_report(
    *,
    named_coeffs: np.ndarray | None = None,
    blinded_coeffs: np.ndarray | None = None,
    coefficient_names: Sequence[str] | None = None,
    named_struct: np.ndarray | None = None,
    blinded_struct: np.ndarray | None = None,
    named_sources: Sequence[str] | None = None,
    blinded_sources: Sequence[str] | None = None,
    named_seed_sharpe: Mapping[int, float] | None = None,
    blinded_seed_sharpe: Mapping[int, float] | None = None,
    pre_cutoff_gap: Mapping[int, float] | None = None,
    post_cutoff_gap: Mapping[int, float] | None = None,
    model_b_coeffs: np.ndarray | None = None,
    equivalence_sd_frac: float = DEFAULT_EQUIVALENCE_SD_FRAC,
    sesoi: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Run every available N3 leg; each block degrades to ``{"status": "no_data"}`` if unfed.

    Returns a dict with keys ``{"tost", "mahalanobis", "structural_mcnemar", "structural_ast", "oos_gap",
    "post_cutoff_persistence", "cross_model"}`` plus a top-level ``"load_bearing_note"`` that
    re-states the load-bearing-vs-theatre distinction. NEVER fabricates: a leg whose inputs are
    absent is reported as ``no_data`` with the reason. This is the single entry point the campaign
    wiring calls.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    out: dict[str, Any] = {
        "load_bearing_note": (
            "LOAD-BEARING: named-vs-blinded TOST (primary) + post-cutoff persistence + a "
            "cutoff-dated 2nd model. THEATRE (not implemented, category error): Min-K%/MIA on "
            "'the reward function' — no realised target token; MIA on closed logprobs ~ chance "
            "(Duan 2024). Structural blinding is necessary hygiene; its sufficiency is the "
            "empirical question this A/B answers (Meeus 2024; Glasserman-Lin 2023)."
        )
    }

    if named_coeffs is not None and blinded_coeffs is not None:
        out["tost"] = named_vs_blinded_tost(
            named_coeffs, blinded_coeffs,
            coefficient_names=coefficient_names, equivalence_sd_frac=equivalence_sd_frac,
        )
        out["mahalanobis"] = coefficient_mahalanobis_permutation(
            named_coeffs, blinded_coeffs, rng=rng
        )
    else:
        out["tost"] = {"status": "no_data", "reason": "named/blinded coefficients not provided"}
        out["mahalanobis"] = {"status": "no_data", "reason": "named/blinded coefficients not provided"}

    if named_struct is not None and blinded_struct is not None:
        out["structural_mcnemar"] = structural_mcnemar(named_struct, blinded_struct)
    else:
        out["structural_mcnemar"] = {"status": "no_data", "reason": "structural vectors not provided"}

    if named_sources is not None and blinded_sources is not None:
        out["structural_ast"] = named_vs_blinded_structural(named_sources, blinded_sources, rng=rng)
    else:
        out["structural_ast"] = {"status": "no_data", "reason": "named/blinded reward sources not provided"}

    if named_seed_sharpe is not None and blinded_seed_sharpe is not None:
        out["oos_gap"] = named_vs_blinded_oos_gap(
            named_seed_sharpe, blinded_seed_sharpe, sesoi=sesoi, rng=rng
        )
    else:
        out["oos_gap"] = {"status": "no_data", "reason": "named/blinded per-seed Sharpe not provided"}

    if pre_cutoff_gap is not None and post_cutoff_gap is not None:
        out["post_cutoff_persistence"] = post_cutoff_persistence(
            pre_cutoff_gap, post_cutoff_gap, rng=rng
        )
    else:
        out["post_cutoff_persistence"] = {
            "status": "no_data", "reason": "pre/post-cutoff per-seed gaps not provided",
        }

    if named_coeffs is not None and model_b_coeffs is not None:
        out["cross_model"] = cross_model_disagreement(
            named_coeffs, model_b_coeffs, coefficient_names=coefficient_names
        )
    else:
        # HONEST DISCLOSURE (V10), not a silent skip: the cutoff-dated second-model check is a frozen
        # PREREGISTRATION §8 protocol item whose secondary panel (R71 — config/llm.yaml
        # ``open_weights_check_model``, pinned to Qwen3-Coder via OpenRouter, 2026-07-02) has NOT been
        # executed. Only the single Claude model family (Sonnet -> Opus) was run, so this leg has no
        # second designer to compare against. ``status`` stays "no_data" (it carries no result), but the
        # reason states the omission explicitly so an analyst cannot misread the absence as a passed check.
        out["cross_model"] = {
            "status": "no_data",
            "executed": False,
            "reason": (
                "cross-model check NOT EXECUTED: the R71 open-weights second-model panel "
                "(config/llm.yaml open_weights_check_model, pinned to Qwen3-Coder via OpenRouter) has "
                "not been run; only the single Claude family (Sonnet -> Opus) was run. Frozen "
                "PREREGISTRATION §8 item; the secondary panel is a separate, key- and compute-gated run."
            ),
        }

    return out
