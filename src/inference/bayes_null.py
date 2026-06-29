"""Bayesian evidence FOR the null — a report-only complement to the frozen TOST equivalence test (R67).

A pre-registered methodology UPGRADE, not a new hypothesis: it expresses the SAME headline null question
(is the distributional-vs-scalar effect practically zero?) in a Bayesian frame, yielding POSITIVE evidence
*for* practical equivalence (a Bayes factor ``BF01 > 1``, a credible interval inside the ROPE) rather than
only the frequentist "failed to reject". This directly answers the standard examiner attack on a null —
"is it informative, or merely underpowered?" — which TOST alone cannot (Lakens et al. 2020; Dienes 2014).

Scope discipline (why this is safe on a frozen, pre-registered design):
- It is **report-only** and **disjoint** from the frozen m=6 testing family — it never gates a hypothesis.
- It reads only the SAME paired per-seed difference scores the headline IUT consumes; it adds NO new data,
  NO new estimand, NO new comparison.
- It is **pure scipy/numpy** (no PyMC/Stan/NumPyro/arviz), DETERMINISTIC, and respects the numpy<2 pins.
- THE ONE RESEARCHER DEGREE OF FREEDOM — the Cauchy prior scale ``r`` and the ROPE half-width — MUST be
  PINNED in ``PREREGISTRATION.md`` BEFORE the freeze. An un-pinned prior is a garden-of-forking-path: by
  Bartlett's / the Jeffreys-Lindley paradox, ``BF01 -> inf`` as ``r -> inf`` regardless of the data, so a
  vague prior can MANUFACTURE null evidence. We therefore (a) fix ``r = DEFAULT_R`` (the BayesFactor
  "medium" default) as the confirmatory headline, (b) reuse the already-frozen SESOI as the ROPE, and
  (c) ALWAYS report a robustness curve over ``R_GRID`` — the null is "supported" only if it survives the
  band. TOST itself has NO prior and is immune to this; the BF is the complement, not the replacement.
- NOTE: **GAP, disclosed 2026-07-26 (deep review): the pin is PROSE-ONLY and therefore OUTSIDE the
  freeze gate.** ``r``, ``R_GRID`` and ``bf_threshold`` are pinned in ``PREREGISTRATION.md`` as required
  above, but ``config/preregistration.yaml`` carries NO corresponding key (verified: zero ``jzs`` /
  ``prior`` / ``bf_threshold`` entries in the machine mirror). ``scripts/freeze.py``'s prose<->YAML
  assertions can only check fields that exist on BOTH sides, so this particular "MUST be PINNED"
  requirement is enforced by convention, not by the gate — unlike the SESOI, the equivalence margin or
  the m=6 family, all of which ARE machine-mirrored and asserted. The hash still binds the prose (the
  ``PREREGISTRATION.md`` bytes are hashed), so the VALUE cannot change silently post-freeze; what is
  missing is the cross-file agreement check that would catch the CODE drifting from the prose. Closing
  it properly means mirroring these three values into the YAML and adding an assertion — a small,
  clearly-scoped change, deliberately not made unilaterally here because it adds a registered field.

References
---------
- Rouder, Speckman, Sun, Morey & Iverson (2009), *Psychon. Bull. Rev.* 16(2):225-237 — JZS default BF.
- Wagenmakers (2007), *Psychon. Bull. Rev.* 14(5):779-804 — BIC approximation (prior-free cross-check).
- Kruschke (2018), *AMPPS* 1(2):270-280 — ROPE + HDI decision (use a 90% HDI to mirror TOST's 90% CI).
- Lakens, McLatchie, Isager, Scheel & Dienes (2020) — BF and TOST reported as complements for a null.
- Campbell & Gustafson (2024), *Psychol. Methods* — at matched error, TOST / HDI-ROPE / interval-null BF
  reverse-engineer one another; agreement across the two machineries = a severely tested null.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import integrate, stats

__all__ = [
    "DEFAULT_R",
    "R_GRID",
    "jzs_bf10",
    "bic_bf01",
    "bf01_robustness_curve",
    "hdi",
    "bayesian_null_report",
]

#: Cauchy prior scale on the standardized effect (BayesFactor "medium" default = sqrt(2)/2).
DEFAULT_R: float = math.sqrt(2.0) / 2.0
#: Pre-declared robustness band over the prior scale ("medium"/"wide"/"ultrawide" + a tighter 0.5).
R_GRID: tuple[float, ...] = (0.5, DEFAULT_R, 1.0, math.sqrt(2.0))


def _paired_t(diffs: np.ndarray) -> tuple[float, int, float, float]:
    """(t, n, mean, sd) of the one-sample/paired difference scores; sd is the SAMPLE sd (ddof=1).

    Uses the relative near-zero guard the rest of the inference stack uses (a near-constant difference
    series leaves float residue in the sd, so an exact ``sd == 0`` test silently fails)."""
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    n = int(d.size)
    if n < 2:
        return float("nan"), n, float("nan"), float("nan")
    mu = float(d.mean())
    sd = float(d.std(ddof=1))
    if not np.isfinite(sd) or sd <= 1e-12 * (abs(mu) + 1e-12) or np.ptp(d) == 0.0:
        # Degenerate (constant) differences: an exact point null if mu==0, else an unbounded effect.
        t = 0.0 if abs(mu) <= 1e-12 else math.copysign(float("inf"), mu)
        return t, n, mu, 0.0
    return mu / (sd / math.sqrt(n)), n, mu, sd


def jzs_bf10(t: float, n: int, r: float = DEFAULT_R) -> float:
    """JZS default Bayes factor ``BF10`` (evidence for H1 over H0) for a one-sample/paired t (Rouder 2009).

    Cauchy(scale ``r``) prior on the standardized effect, Jeffreys prior on the variance; the marginal is a
    1-D integral over the variance hyperparameter ``g`` (computed with :func:`scipy.integrate.quad`)."""
    if not math.isfinite(t):
        return float("inf")  # an unbounded observed effect -> decisive evidence for H1
    if abs(t) > 10.0:
        # Far outside the null regime the JZS BF10 is monotone increasing in |t| (already decisive,
        # BF10 >> 100 at |t| = 10 for any r in R_GRID), but the quad integrand UNDERFLOWS at extreme
        # |t|: num -> 0 => bf10 = 0 => bf01 = inf => the downstream verdict degrades to 'inconclusive'
        # (or 1/bf10 raises ZeroDivisionError in bf01_robustness_curve) for a decisively NON-null
        # effect — the WRONG direction. Short-circuit to decisive evidence AGAINST the null instead
        # (bf01 = 1/inf = 0.0 -> the evidence_against_null branch in bayesian_null_report).
        return float("inf")
    nu = n - 1
    r2 = r * r

    def integrand(g: float) -> float:
        a = 1.0 + n * g * r2
        return (
            a**-0.5
            * (1.0 + t * t / (a * nu)) ** (-(nu + 1) / 2.0)
            * (2.0 * math.pi) ** -0.5
            * g**-1.5
            * math.exp(-1.0 / (2.0 * g))
        )

    num, _ = integrate.quad(integrand, 0.0, np.inf, limit=200)
    den = (1.0 + t * t / nu) ** (-(nu + 1) / 2.0)
    return float(num / den)


def bic_bf01(t: float, n: int) -> float:
    """Prior-free BIC approximation to ``BF01`` (evidence for H0), Wagenmakers (2007): a closed-form
    cross-check on :func:`jzs_bf10` that uses NO prior — agreement of the two is a free robustness check."""
    if not math.isfinite(t):
        return 0.0
    return float(math.sqrt(n) * (1.0 + t * t / (n - 1)) ** (-n / 2.0))


def bf01_robustness_curve(t: float, n: int, r_grid: tuple[float, ...] = R_GRID) -> dict[float, float]:
    """``BF01`` across the pre-declared prior-scale band — the disclosed sensitivity that neutralises the
    prior-as-forking-path objection (the null is supported only if it survives the whole band)."""
    return {float(r): float(1.0 / jzs_bf10(t, n, r)) for r in r_grid}


def hdi(draws: np.ndarray, mass: float = 0.90) -> tuple[float, float]:
    """Highest-density interval of ``draws`` at credible ``mass`` (narrowest-window estimator; Kruschke).

    Default ``mass=0.90`` to MIRROR TOST's 90% CI (two one-sided 5% tests), so the HDI+ROPE decision lines
    up with the frequentist equivalence decision rather than being gratuitously more conservative."""
    sx = np.sort(np.asarray(draws, dtype=float))
    sx = sx[np.isfinite(sx)]
    m = sx.size
    if m == 0:
        return float("nan"), float("nan")
    k = max(1, int(math.floor(mass * m)))
    if k >= m:
        return float(sx[0]), float(sx[-1])
    widths = sx[k:] - sx[: m - k]
    i = int(np.argmin(widths))
    return float(sx[i]), float(sx[i + k])


def bayesian_null_report(
    diffs: np.ndarray,
    sesoi: float,
    *,
    r: float = DEFAULT_R,
    bootstrap_draws: np.ndarray | None = None,
    mass: float = 0.90,
    bf_threshold: float = 3.0,
) -> dict[str, Any]:
    """Report-only Bayesian evidence-for-the-null complement to the frozen TOST (R67).

    Parameters
    ----------
    diffs : np.ndarray
        The paired per-seed difference scores the headline IUT consumes (e.g. per-seed Sharpe or CVaR
        difference, distributional − comparator). Same input, different inferential lens.
    sesoi : float
        The frozen smallest effect size of interest (the ROPE half-width = the equivalence margin). The
        ROPE is ``[-sesoi, +sesoi]`` — reusing the already-frozen SESOI keeps this calibration-free.
    r : float
        Cauchy prior scale (PIN in the pre-registration; default = ``DEFAULT_R``).
    bootstrap_draws : np.ndarray, optional
        Existing bootstrap draws of the effect (e.g. the paired-seed bootstrap distribution). If given, the
        HDI is computed on them (Bayesian-bootstrap-shaped, noninformative-prior approximation; Rubin 1981);
        otherwise the exact conjugate-Student-t posterior interval is used.
    mass : float
        Credible mass for the HDI / posterior interval (0.90 mirrors TOST's 90% CI).
    bf_threshold : float
        BF01 needed to call the evidence "for equivalence" at the headline prior (3 = Jeffreys "moderate").

    Returns
    -------
    dict
        ``{t, n, effect, bf01, bf01_bic, bf01_robustness, posterior:{mean, ci_low, ci_high, rope_mass},
        hdi_low, hdi_high, hdi_in_rope, robust_for_null, verdict}``. ``robust_for_null`` is True iff BF01
        ≥ ``bf_threshold`` across the ENTIRE ``R_GRID`` (the disclosed prior-robustness gate).
    """
    t, n, mu, sd = _paired_t(diffs)
    if n < 2:
        return {"status": "insufficient", "n": n}

    bf10 = jzs_bf10(t, n, r)
    bf01 = float(1.0 / bf10) if bf10 > 0 else float("inf")
    curve = bf01_robustness_curve(t, n)

    # Conjugate-t posterior on the mean difference under the reference prior p(mu, sigma^2) ∝ 1/sigma^2:
    #   mu | data ~ t_{n-1}(loc = mean, scale = sd/sqrt(n)); the credible interval equals the frequentist
    # t CI under this prior, so the Bayesian posterior is added at ZERO modelling risk.
    if sd > 0.0:
        post = stats.t(df=n - 1, loc=mu, scale=sd / math.sqrt(n))
        lo, hi = float(post.ppf((1 - mass) / 2)), float(post.ppf(1 - (1 - mass) / 2))
        rope_mass = float(post.cdf(sesoi) - post.cdf(-sesoi))
    else:  # degenerate constant differences
        lo = hi = mu
        rope_mass = 1.0 if abs(mu) <= sesoi else 0.0

    if bootstrap_draws is not None and np.asarray(bootstrap_draws).size:
        h_lo, h_hi = hdi(np.asarray(bootstrap_draws, dtype=float), mass=mass)
    else:
        h_lo, h_hi = lo, hi
    hdi_in_rope = bool(np.isfinite(h_lo) and np.isfinite(h_hi) and -sesoi <= h_lo and h_hi <= sesoi)

    robust_for_null = bool(all(v >= bf_threshold for v in curve.values()))
    if robust_for_null and hdi_in_rope:
        verdict = "evidence_for_practical_equivalence"
    elif bf01 < 1.0 / bf_threshold:
        verdict = "evidence_against_null"
    else:
        verdict = "inconclusive"

    return {
        "status": "ok",
        "t": float(t),
        "n": int(n),
        "effect": float(mu),
        "prior_r": float(r),
        "bf01": bf01,
        "bf01_bic": bic_bf01(t, n),
        "bf01_robustness": curve,
        "posterior": {"mean": float(mu), "ci_low": lo, "ci_high": hi, "rope_mass": rope_mass},
        "hdi_low": h_lo,
        "hdi_high": h_hi,
        "rope": [float(-sesoi), float(sesoi)],
        "hdi_in_rope": hdi_in_rope,
        "robust_for_null": robust_for_null,
        "verdict": verdict,
    }
