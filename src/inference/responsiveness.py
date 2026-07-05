"""Responsiveness + the numeracy-bottleneck (legible-format) differential — report-only mechanism (ADR-039).

SQ1 of the mechanism chain asks *responsiveness*: does the authored reward CODE track the fed tail signal at
all? This module measures that as the association between a fed-signal summary ``X`` and an authored-code
feature ``M`` over candidates, with a bootstrap CI (Spearman rank by default — robust to the heavy-tailed
generation-to-generation deltas and to the exact numeric scale).

The headline reframe (ADR-039) is the **numeracy bottleneck**: frontier LLMs cannot reliably verbalise or
compare *close small floats* (50–70% accuracy; arXiv:2602.07812; NUMCoT, arXiv:2406.02864), and the fed CVaR
values (e.g. −0.0577 vs −0.0582) sit squarely in that failure regime. If the channel is silent because the
numbers are *illegible* rather than because tail information is *useless*, then presenting the SAME tail
information in a more legible format (basis points, rank/decile framing, or a rescaled contrast) should RAISE
responsiveness. :func:`legible_format_responsiveness_differential` tests exactly that contrast — a positive,
CI-separated differential is a citable mechanism for the predicted null *and* a concrete scaling hypothesis
(legibility, not capacity, is the lever). This is report-only, DISJOINT from the frozen ``m=6`` family, and is
**not a new confirmatory arm** — it is a secondary ablation over an alternative feedback *rendering* of the
identical tail content. Deterministic (seeded bootstrap; numpy + scipy, already dependencies).

References: arXiv:2602.07812 (close-float comparison failure); NUMCoT arXiv:2406.02864; Spearman (1904);
Efron & Tibshirani (1993) bootstrap CIs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

__all__ = ["responsiveness", "legible_format_responsiveness_differential"]

#: Minimum fraction of the requested bootstrap replicates that must be NON-degenerate for the percentile CI
#: to be trusted. The authored-code feature M is often INTEGER-VALUED (a count of tail-shaped constructs), so
#: many case-resamples collapse to a constant column and yield a NaN coefficient that is dropped; if too few
#: valid replicates survive, the percentile CI is unreliable and ``ci_reliable`` is flagged False (P7b).
MIN_BOOT_VALID_FRACTION: float = 0.5


def _coef(x: np.ndarray, m: np.ndarray, method: str) -> float:
    """Responsiveness coefficient of ``m`` on ``x``: Spearman rho, or the standardised OLS slope (= Pearson r)."""
    if method == "spearman":
        if x.size < 2 or np.ptp(x) == 0 or np.ptp(m) == 0:
            return float("nan")
        return float(stats.spearmanr(x, m).statistic)
    if method == "slope":
        sx, sm = x.std(ddof=0), m.std(ddof=0)
        if sx == 0 or sm == 0:
            return float("nan")
        return float(np.cov(x, m, ddof=0)[0, 1] / (sx * sm))  # standardised slope == Pearson r
    raise ValueError(f"unknown method {method!r} (use 'spearman' or 'slope')")


def _bootstrap_coef_raw(
    x: np.ndarray, m: np.ndarray, method: str, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """Per-replicate bootstrap coefficients, LENGTH-PRESERVING (degenerate resamples -> NaN, NOT dropped).

    Keeping the array at full ``n_boot`` length preserves the replicate INDEX, which the two-condition
    differential relies on to pair by bootstrap replicate rather than by compacted position."""
    n = x.size
    out = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        out[i] = _coef(x[idx], m[idx], method)
    return out


def _bootstrap_coef(
    x: np.ndarray, m: np.ndarray, method: str, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """Finite bootstrap coefficients (degenerate resamples dropped) — the single-condition percentile CI."""
    out = _bootstrap_coef_raw(x, m, method, n_boot, rng)
    return out[np.isfinite(out)]


def responsiveness(
    x: np.ndarray,
    m: np.ndarray,
    *,
    method: str = "spearman",
    n_boot: int = 5000,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Responsiveness of the authored-code feature ``M`` to the fed signal ``X`` (SQ1), with a bootstrap CI.

    ``x`` / ``m`` are equal-length paired arrays (one row per candidate). ``method`` is ``"spearman"`` (rank
    correlation; default, scale-robust) or ``"slope"`` (standardised OLS slope = Pearson r). Returns
    ``{"status": "ok", "coef", "ci_low", "ci_high", "responsive", "n", "method", "n_boot_valid",
    "ci_reliable"}`` where ``responsive`` is True iff the 95% percentile CI excludes 0 AND the CI is reliable.
    ``ci_reliable`` is False when fewer than ``MIN_BOOT_VALID_FRACTION`` of the requested resamples were
    non-degenerate — integer-valued ``m`` (a construct count) makes many case-resamples collapse to a
    constant column, dropping their coefficient; a percentile CI from too few survivors cannot be trusted, so
    ``responsive`` is forced False in that case (P7b). ``{"status": "no_data", ...}`` for < 3 rows or a
    degenerate (constant) input.
    """
    x = np.asarray(x, dtype=float).ravel()
    m = np.asarray(m, dtype=float).ravel()
    if x.size != m.size:
        return {"status": "no_data", "reason": "x and m must be equal-length paired arrays"}
    if x.size < 3:
        return {"status": "no_data", "reason": "need >= 3 paired rows"}
    # Non-finite is checked BEFORE constancy (deep review 2026-07-26, #71). ``np.ptp`` of an array holding
    # a NaN is NaN, and ``NaN == 0`` is False — so the degeneracy guard below could not fire on exactly the
    # degenerate input it exists to reject. MEASURED: an all-NaN ``x`` returned ``status="ok"`` with
    # ``coef=NaN``, and a single NaN returned ``status="ok"`` with ``coef=NaN`` plus a percentile CI built
    # from only the ~34% of resamples that happened to dodge the NaN row — a real-looking interval around an
    # unusable point estimate. Callers gate on ``status``, so advertising success here is the harm; report
    # no_data instead. Deliberately NOT "drop the non-finite rows": choosing a filtering policy would change
    # what this registered SQ1 statistic is computed over, which is a design decision, not a bug fix.
    if not (np.isfinite(x).all() and np.isfinite(m).all()):
        return {"status": "no_data", "reason": "non-finite value in x or m"}
    if np.ptp(x) == 0 or np.ptp(m) == 0:
        return {"status": "no_data", "reason": "degenerate (constant) x or m"}
    if rng is None:
        rng = np.random.default_rng(0)

    coef = _coef(x, m, method)
    boots = _bootstrap_coef(x, m, method, n_boot, rng)
    if boots.size:
        ci_low, ci_high = (float(v) for v in np.percentile(boots, [2.5, 97.5]))
    else:
        ci_low = ci_high = float("nan")
    # P7b: gate the percentile CI on the valid-boot FRACTION. Integer-valued m (a construct count) makes many
    # resamples degenerate (constant column -> NaN coef, dropped); a CI from too few survivors is unreliable.
    ci_reliable = bool(boots.size >= MIN_BOOT_VALID_FRACTION * int(n_boot))
    responsive = bool(ci_reliable and boots.size and (ci_low > 0.0 or ci_high < 0.0))
    return {
        "status": "ok",
        "coef": float(coef),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "responsive": responsive,
        "n": int(x.size),
        "method": method,
        "n_boot_valid": int(boots.size),
        "ci_reliable": ci_reliable,
    }


def legible_format_responsiveness_differential(
    x_legible: np.ndarray,
    m_legible: np.ndarray,
    x_raw: np.ndarray,
    m_raw: np.ndarray,
    *,
    method: str = "spearman",
    n_boot: int = 5000,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Legible-minus-raw responsiveness gap — the numeracy-bottleneck test.

    Two INDEPENDENT conditions present the SAME tail content rendered differently: ``*_legible`` (e.g. basis
    points / rank framing) vs ``*_raw`` (the close-small-float vector −0.0577…). Each is a candidate-level
    ``(X, M)`` pair set. The statistic is ``coef_legible − coef_raw``; its CI comes from independently
    resampling the two conditions and differencing the bootstrap coefficients (unpaired; Efron–Tibshirani).
    A positive, zero-excluding differential (``legibility_helps``) means rendering the numbers legibly raises
    responsiveness — i.e. the channel's silence is a **legibility** bottleneck, not evidence that tail
    information is useless (the citable mechanism + scaling hypothesis behind the predicted null).

    Returns ``{"status": "ok", "coef_legible", "coef_raw", "differential", "ci_low", "ci_high",
    "legibility_helps", "legible", "raw"}`` (the two nested ``responsiveness`` dicts) or ``no_data`` if either
    condition is degenerate.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    leg = responsiveness(x_legible, m_legible, method=method, n_boot=n_boot, rng=rng)
    raw = responsiveness(x_raw, m_raw, method=method, n_boot=n_boot, rng=rng)
    if leg["status"] != "ok" or raw["status"] != "ok":
        return {
            "status": "no_data",
            "reason": "a condition was degenerate or too small",
            "legible": leg,
            "raw": raw,
        }

    xl = np.asarray(x_legible, dtype=float).ravel()
    ml = np.asarray(m_legible, dtype=float).ravel()
    xr = np.asarray(x_raw, dtype=float).ravel()
    mr = np.asarray(m_raw, dtype=float).ravel()
    # LENGTH-PRESERVING per-replicate coefficients (degenerate resamples -> NaN, NOT compacted). Pair the two
    # INDEPENDENT conditions by REPLICATE INDEX i, then drop any pair where either condition's resample was
    # degenerate. The previous code compacted each array separately (dropping NaNs) and then differenced
    # ``bl[:k] - br[:k]`` — that silently paired replicate i of the legible condition with a DIFFERENT
    # replicate of the raw condition whenever the two had a different number of dropped resamples, corrupting
    # the differential's bootstrap distribution. Index-pairing + a joint finite mask fixes it (P7a).
    bl = _bootstrap_coef_raw(xl, ml, method, n_boot, rng)
    br = _bootstrap_coef_raw(xr, mr, method, n_boot, rng)
    both_finite = np.isfinite(bl) & np.isfinite(br)
    if not both_finite.any():
        return {"status": "no_data", "reason": "empty bootstrap", "legible": leg, "raw": raw}
    diff = bl[both_finite] - br[both_finite]  # independent conditions, paired by REPLICATE index
    n_boot_valid = int(diff.size)
    ci_low, ci_high = (float(v) for v in np.percentile(diff, [2.5, 97.5]))
    differential = float(leg["coef"] - raw["coef"])
    # P7b, propagated (deep review 2026-07-26, #72). ``responsiveness`` gates its own verdict on the
    # valid-replicate fraction (``responsive = ci_reliable and ...``); this differential asserted
    # ``legibility_helps`` with NO such gate, even though it had both conditions' ``ci_reliable`` flags in
    # hand. MEASURED: with one non-finite row in the legible condition, this returned ``status="ok"``,
    # ``differential=NaN`` and a ``legibility_helps`` verdict computed from 678/2000 SELF-SELECTED
    # replicates — the ones that happened to dodge the bad row — while ``leg["ci_reliable"]`` was already
    # False. The differential's CI inherits both conditions' degenerate-resample problem, so it inherits
    # their reliability gate too, and reports the flag so a reader can see it.
    ci_reliable = bool(
        leg["ci_reliable"] and raw["ci_reliable"]
        and n_boot_valid >= MIN_BOOT_VALID_FRACTION * int(n_boot)
    )
    return {
        "status": "ok",
        "coef_legible": float(leg["coef"]),
        "coef_raw": float(raw["coef"]),
        "differential": differential,
        "ci_low": ci_low,
        "ci_high": ci_high,
        # one-sided sense: legibility RAISES responsiveness — but never claimed off an untrustworthy CI
        "legibility_helps": bool(ci_reliable and ci_low > 0.0),
        "ci_reliable": ci_reliable,
        "method": method,
        "n_boot_valid": n_boot_valid,
        "legible": leg,
        "raw": raw,
    }
