"""Causal-chain mediation for the mechanism story — report-only (grade-strategy mechanism kernel).

The headline reframe casts the result as a three-link causal chain and asks a null to LOCATE where it
breaks::

    fed tail signal  --(a)-->  authored reward CODE  --(b)-->  realised tail outcome
          \\___________________________(c')__________________________/

This module estimates that mediation directly, per candidate. With
``X`` = a scalar summary of the fed tail signal (e.g. the magnitude of the CVaR feedback, or its
generation-to-generation change), ``M`` = a scalar feature of the *authored reward program* (e.g. the
fitted CVaR-term coefficient, or an AST tail-construct count — the same instruments the contamination /
reward-code-distance modules use), and ``Y`` = the realised out-of-sample tail outcome (e.g. CVaR-5%),
it returns the standard linear-mediation decomposition (Baron & Kenny 1986; Preacher & Hayes 2008):

* **path a** — regress ``M`` on ``X`` (does the fed signal move the code? — *responsiveness*);
* **path b** — coefficient on ``M`` in ``Y ~ X + M`` (does the code move the outcome? — *transmission*);
* **total effect c**, **direct effect c'**, and the **indirect (mediated) effect a·b** (= ``c − c'`` in
  the linear/OLS case), with a **nonparametric bootstrap percentile CI on a·b** (Preacher–Hayes).

Why this earns the mechanism headline under a NULL. If the fed signal does not change the program
(``a ≈ 0`` — the responsiveness null this work predicts), then ``a·b ≈ 0`` for ANY ``b``: the chain is
severed at the **first** link, and the equivalence in ``Y`` is *explained*, not merely observed. The
module makes that locatable and quantified rather than asserted.

Honesty (stated in the write-up, not hidden). Observational mediation is **associational**: the
indirect-effect estimate has a causal interpretation only under sequential ignorability (no unmeasured
confounding of the X→M, X→Y and M→Y relations; Imai, Keele & Tingey 2010). Here X, M, Y are all read off
the same trained candidate, so M is endogenous to the agent it steers (see ``measurement.py``: the fed
tail is the trained policy's own realised returns) — we therefore report this as a **descriptive
decomposition of the mechanism**, DISJOINT from the frozen ``m=6`` testing family; it never gates a
hypothesis. Deterministic (numpy only; seeded bootstrap), so it replays byte-identically from the archive.

References: Baron & Kenny (1986) JPSP 51:1173; Preacher & Hayes (2008) Behav. Res. Methods 40:879
(bootstrap of the indirect effect); Imai, Keele & Tingley (2010) Psych. Methods 15:309 (causal mediation
assumptions); MacKinnon et al. (2002) (product-of-coefficients).
"""

from __future__ import annotations

from typing import Any

import numpy as np

# The valid-bootstrap-fraction floor is DEFINED in ``responsiveness`` and already imported the same way
# by ``information_gap`` — one constant, one owner, three instruments (no third copy to drift).
from src.inference.responsiveness import MIN_BOOT_VALID_FRACTION

__all__ = ["mediation_analysis"]

#: Guard threshold for ``prop_mediated``: the total effect c must exceed this many bootstrap SEs of 0 for the
#: proportion-mediated ratio to be reported (else NaN). 2.0 ≈ the two-sided 95% z-cut — the same "distinguishable
#: from 0" bar the CI-includes-0 rule enforces; kept as a named constant so the null-regime guard is auditable.
PROP_MEDIATED_C_SE_K: float = 2.0


def _slopes(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS coefficients (intercept first) via least squares; raises ``LinAlgError`` on a singular design."""
    beta, _res, rank, _sv = np.linalg.lstsq(design, y, rcond=None)
    if rank < design.shape[1]:  # rank-deficient (e.g. a constant predictor) -> unstable; signal upstream
        raise np.linalg.LinAlgError("rank-deficient design in mediation OLS")
    return beta


def _standardize(v: np.ndarray) -> np.ndarray:
    sd = float(v.std(ddof=0))
    return (v - v.mean()) / sd if sd > 0 else v - v.mean()


def mediation_analysis(
    x: np.ndarray,
    m: np.ndarray,
    y: np.ndarray,
    *,
    n_boot: int = 5000,
    standardize: bool = True,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Single-mediator linear mediation of ``X → M → Y`` with a bootstrap CI on the indirect effect.

    Parameters
    ----------
    x, m, y : np.ndarray
        Equal-length 1-D arrays, one row per candidate (paired): ``x`` = fed-signal summary (treatment),
        ``m`` = authored-code feature (mediator), ``y`` = realised tail outcome (outcome).
    n_boot : int
        Nonparametric (case-resampling) bootstrap replications for the indirect-effect CI (Preacher–Hayes).
    standardize : bool
        Z-score ``x``, ``m``, ``y`` once up front so effects are in standardised (comparable) units. The
        bootstrap then resamples the standardised data (deterministic; documented).
    rng : np.random.Generator | None
        Seeded generator (default ``default_rng(0)``) → byte-deterministic.

    Returns
    -------
    dict
        ``{"status": "ok", "n", "a", "b", "c_total", "c_direct", "indirect", "prop_mediated",
        "prop_mediated_undefined", "c_total_ci_low", "c_total_ci_high", "c_total_boot_se", "ci_low",
        "ci_high", "mediated", "n_boot_valid", "standardized"}`` where ``mediated`` is True iff the 95%
        percentile CI on the indirect effect ``a·b`` excludes 0. ``prop_mediated`` (= ``indirect / c_total``)
        is STABILITY-GUARDED: it is ``NaN`` and ``prop_mediated_undefined`` is True whenever the total effect
        ``c_total`` is indistinguishable from 0 (its 95% bootstrap CI includes 0, or ``|c_total|`` is within
        ``PROP_MEDIATED_C_SE_K`` bootstrap SEs of 0) — in the predicted-null regime a bare ``small / small``
        ratio would otherwise explode or sign-flip into a misleading proportion. Returns
        ``{"status": "no_data", ...}`` for fewer than 3 paired rows (no residual df for the two-predictor
        regression) or a degenerate (zero-variance) ``x`` or ``m``.
    """
    x = np.asarray(x, dtype=float).ravel()
    m = np.asarray(m, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if not (x.size == m.size == y.size):
        return {"status": "no_data", "reason": "x, m, y must be equal-length paired arrays"}
    n = x.size
    if n < 3:
        return {"status": "no_data", "reason": "need >= 3 paired rows for a two-predictor mediation"}
    if x.std(ddof=0) == 0.0 or m.std(ddof=0) == 0.0:
        return {"status": "no_data", "reason": "degenerate (zero-variance) treatment or mediator"}
    if rng is None:
        rng = np.random.default_rng(0)

    if standardize:
        x, m, y = _standardize(x), _standardize(m), _standardize(y)

    def _effects(xx: np.ndarray, mm: np.ndarray, yy: np.ndarray) -> tuple[float, float, float, float, float]:
        one = np.ones(xx.size)
        a = float(_slopes(np.column_stack([one, xx]), mm)[1])          # M ~ X
        c_total = float(_slopes(np.column_stack([one, xx]), yy)[1])    # Y ~ X
        coef = _slopes(np.column_stack([one, xx, mm]), yy)             # Y ~ X + M
        c_direct, b = float(coef[1]), float(coef[2])
        return a, b, c_total, c_direct, a * b

    a, b, c_total, c_direct, indirect = _effects(x, m, y)

    # Bootstrap BOTH the indirect effect a·b AND the total effect c so that ``prop_mediated = indirect /
    # c_total`` can be STABILITY-guarded (below). In the predicted-null regime the total effect is small and
    # noisy; a bare ``indirect / c_total`` then divides small-by-small and explodes or sign-flips into a
    # meaningless "large/negative proportion mediated". Case-resampling both effects on the SAME resample keeps
    # them paired (the deterministic, seeded bootstrap is unchanged; we simply retain c_total per replicate).
    boots = np.empty(n_boot, dtype=float)
    boots_c = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            a_i, b_i, c_i, _cd_i, ind_i = _effects(x[idx], m[idx], y[idx])
            boots[i] = ind_i
            boots_c[i] = c_i
        except np.linalg.LinAlgError:
            boots[i] = np.nan
            boots_c[i] = np.nan
    valid = boots[np.isfinite(boots)]
    if valid.size:
        ci_low, ci_high = (float(v) for v in np.percentile(valid, [2.5, 97.5]))
    else:
        ci_low = ci_high = float("nan")
    # Gate the percentile CI on the valid-boot FRACTION, mirroring the P7b fix already applied to the
    # sibling instrument (``responsiveness.responsiveness``, which DEFINES ``MIN_BOOT_VALID_FRACTION``,
    # and ``information_gap`` which imports it). Mediation has the IDENTICAL degeneracy mode: ``_effects``
    # raises ``LinAlgError`` on a rank-deficient resample (a constant predictor column), those replicates
    # are dropped to NaN, and ``valid.size`` alone only excluded the ZERO case — so ``mediated=True`` could
    # be declared from a handful of survivors. 2026-07-26 review.
    ci_reliable = bool(valid.size >= MIN_BOOT_VALID_FRACTION * int(n_boot))
    mediated = bool(ci_reliable and valid.size and (ci_low > 0.0 or ci_high < 0.0))

    # STABILITY GUARD on prop_mediated (predicted-null-regime safety). ``prop_mediated`` is only a meaningful
    # ratio when the total effect c is DISTINGUISHABLE from 0. We mark it undefined (NaN, with a flag) when the
    # bootstrap CANNOT separate |c_total| from bootstrap noise — i.e. EITHER the 95% bootstrap CI on c_total
    # includes 0, OR |c_total| < PROP_MEDIATED_C_SE_K · SE_boot(c_total). Otherwise small/small would be
    # reported as a large or sign-flipped ratio (Preacher & Hayes 2008 note the ratio is unstable near c≈0).
    c_valid = boots_c[np.isfinite(boots_c)]
    if c_valid.size:
        c_ci_low, c_ci_high = (float(v) for v in np.percentile(c_valid, [2.5, 97.5]))
        c_se = float(c_valid.std(ddof=1)) if c_valid.size > 1 else float("nan")
    else:
        c_ci_low = c_ci_high = c_se = float("nan")
    # near-zero c: CI straddles 0, OR |c| within k SE of 0, OR the SE itself is undefined/non-finite.
    c_ci_includes_zero = not (c_valid.size and (c_ci_low > 0.0 or c_ci_high < 0.0))
    c_within_noise = bool(np.isfinite(c_se)) and abs(c_total) < PROP_MEDIATED_C_SE_K * c_se
    prop_unstable = bool(c_ci_includes_zero or c_within_noise or not np.isfinite(c_se))
    prop_mediated = float("nan") if prop_unstable else float(indirect / c_total)

    return {
        "status": "ok",
        "n": int(n),
        "a": a,
        "b": b,
        "c_total": c_total,
        "c_direct": c_direct,
        "indirect": indirect,
        "prop_mediated": prop_mediated,
        "prop_mediated_undefined": prop_unstable,
        "c_total_ci_low": c_ci_low,
        "c_total_ci_high": c_ci_high,
        "c_total_boot_se": c_se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "mediated": mediated,
        "n_boot_valid": int(valid.size),
        "ci_reliable": ci_reliable,
        "standardized": bool(standardize),
    }
