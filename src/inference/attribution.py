"""Factor-attribution of frozen-winner TEST returns — the difference-in-alpha Door-C secondary.

Why this module exists (the #1 un-planned reviewer hole)
--------------------------------------------------------
The headline H2 result is "distributional feedback beats scalar feedback" on the held-out test
leg. The sharpest un-planned attack on it (deep-research Part 7, red-team #6;
``00_planning/CAMPAIGN_DEEP_RESEARCH_FINDINGS_2026-06-21.md`` §5.10) is:

    "The agent's edge is just low-vol / betting-against-beta (BAB) beta — a known factor tilt,
     not the reward channel."

This module answers it the only defensible way: re-price each arm's *realized* test returns
against a ladder of published factor models (CAPM -> FF3 -> Carhart-4 -> FF5 -> FF6 -> +BAB ->
+QMJ) and report the **difference in intercept (alpha)** between the distributional and scalar
arms after removing the factor exposure. If the distributional arm's alpha *survives* the factor
controls (and BAB's beta is small while alpha persists), the edge is not "just BAB".

Door-C, not Door-A/B (the governing principle, deep-research Part 0)
-------------------------------------------------------------------
This is a **measurement/inference** sharpening of the dependent variable (alpha instead of raw
Sharpe), computed entirely in the reporting layer. It NEVER touches the env, the reward, the
agent, or the treatment. It is a **separate, declared secondary family** with **disjoint result
keys** (``rung`` / ``factor`` / ``alpha_diff`` / ``seed``) that cannot be confused with the frozen
H2 family's ``arm_a`` / ``arm_b`` / ``metric`` / ``level`` keys — so
``scripts/analyze_campaign.assert_realized_family_matches_frozen`` stays green and the frozen m=6
is never mutated (deep-research Part 2, "amendment-free additions").

The econometrics (be exact; this is the load-bearing part)
----------------------------------------------------------
* **Model.** For one realized daily series ``r_t`` (the frozen winner's test path) and a factor
  matrix ``F_t`` (each column a long/short factor return, already in decimals), OLS

      ``(r_t - rf_t) = alpha + sum_k beta_k * F_{k,t} + eps_t``.

  ``Mkt-RF`` is itself an excess return so it enters raw; the *left-hand side* uses excess returns
  ``r - rf`` so ``alpha`` is the CAPM/multifactor intercept (Jensen's alpha). ``alpha`` is reported
  per period; annualised alpha = ``alpha * 252``.
* **Standard errors.** Daily portfolio residuals are serially correlated (volatility clustering,
  momentum/mean-reversion), so OLS SEs understate sampling error. We use **Newey & West (1987)**
  heteroskedasticity-and-autocorrelation-consistent (HAC) SEs with the **Bartlett kernel**
  ``w_l = 1 - l/(L+1)``. The truncation lag follows the **Newey & West (1994)** automatic-lag rule of
  thumb ``L = floor(4 * (T/100) ** (2/9))`` (cite Newey-West 1994 for the *lag rule*; Newey-West 1987
  for the *estimator* -- NOT Schwert 1989, whose ``12(T/100)^(1/4)`` is a different, ADF unit-root,
  rule). statsmodels' ``OLS.fit(cov_type="HAC")`` defaults ``maxlags`` to exactly this floored value
  and applies the Bartlett kernel; we pass the lag explicitly so it is logged and
  reproducible, and :func:`newey_west_hac_lag` is unit-tested against a hand-rolled Newey-West.
* **Difference-in-alpha (the headline of this module).** We do NOT pool the two arms into one
  series. Mirroring the frozen per-seed rliable inference (Agarwal et al. 2021;
  ``src.inference.bootstrap.paired_seed_difference_test``): for each arm we fit the factor model
  **per seed** (one frozen-winner test path per training seed) to obtain that seed's alpha, then run
  the **paired across-seed bootstrap** on the per-seed alpha differences ``alpha_a(seed) -
  alpha_b(seed)`` over the shared training seeds. This carries the across-seed (training-RNG)
  variance — the dominant uncertainty in a multi-seed RL evaluation — exactly as the headline does,
  rather than a single-path regression that would understate it.

What carries the inference, and what is merely descriptive (read before reporting a t-stat)
-------------------------------------------------------------------------------------------
* **Per-seed factor alpha is a HAC-OLS POINT ESTIMATE; the decision is the across-seed paired
  bootstrap.** The per-``(arm, seed, rung)`` cell's HAC SE / t / p (Newey-West) describe one seed's
  single-path regression and are reported as **descriptive only** — they do NOT carry the inference,
  because they ignore the across-seed (training-RNG) variance. The actual verdict is the sign and
  two-sided p-value of the **paired across-seed bootstrap on the per-seed alpha differences**
  (``difference_in_alpha``); that, not the per-cell HAC t, is what decides whether the
  factor-adjusted edge is real.
* **The factor ladder is a MONOTONE ROBUSTNESS SEQUENCE with a pre-named headline rung, not a
  discovery family.** The rungs (CAPM -> FF3 -> Carhart-4 -> FF5 -> FF6 -> +BAB -> +QMJ) are nested
  controls of increasing strictness; the **pre-named headline rung is Carhart-4 (and +BAB once the
  AQR pull lands)** — the rung the BAB rebuttal lives on. Climbing the ladder is a *sensitivity*
  showing the alpha survives progressively stronger controls, so the **BH correction ACROSS rungs is
  a sensitivity adjustment, NOT a multiple-discovery family** (we are not fishing across rungs for a
  significant one; the headline rung is fixed in advance). This is a separate, declared secondary
  family from the frozen m=6 (disjoint keys) and never enters the headline conjunction.

Factor data — what is ON DISK vs what NEEDS A PULL
-----------------------------------------------------------------------------------------
* **On disk** (``data/raw/``, decimals, daily, 2005-2026): ``Mkt-RF``, ``SMB``, ``HML``, ``RF``
  (``french_F-F_Research_Data_Factors_daily.csv``) and ``Mom`` / UMD
  (``french_F-F_Momentum_Factor_daily.csv``). -> CAPM, FF3, and Carhart-4 are fully constructable
  today.
* **Needs a Ken-French pull**: ``RMW``, ``CMA`` (FF5 profitability/investment) -> FF5, FF6.
* **Needs an AQR pull**: ``BAB`` (Frazzini-Pedersen 2014) and ``QMJ`` (Asness-Frazzini-Pedersen
  2019) -> +BAB, +QMJ rungs. QMJ in particular is NOT constructable from the panel (needs
  fundamentals).

Every rung whose factor columns are not all present is reported with ``status="skipped"`` and the
missing columns named — the harness DEGRADES GRACEFULLY and never fabricates an alpha it cannot
estimate (the no-fabrication rule).

This module is import-light at module load (numpy + pandas only); statsmodels is imported lazily
inside the OLS path so the rest of the analysis stack does not pay for it.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

__all__ = [
    "FACTOR_LADDER",
    "newey_west_hac_lag",
    "factor_alpha",
    "difference_in_alpha",
    "campaign_attribution",
    "load_factor_panel",
    "attribution_markdown",
]

_LOG = logging.getLogger(__name__)

#: The factor ladder (deep-research §5.10): each rung is ``(name, factor_columns)`` read "regress
#: excess return on these factor columns". A rung runs only when EVERY one of its factor columns is
#: present in the supplied factor panel; otherwise it is reported skipped (graceful degradation).
#: ``capm`` uses the market excess return alone; the higher rungs add the published controls. The
#: BAB rung is the headline rival ("the edge is just betting-against-beta"); QMJ is the quality rung.
FACTOR_LADDER: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("capm", ("Mkt-RF",)),
    ("ff3", ("Mkt-RF", "SMB", "HML")),
    ("carhart4", ("Mkt-RF", "SMB", "HML", "Mom")),
    ("ff5", ("Mkt-RF", "SMB", "HML", "RMW", "CMA")),
    ("ff6", ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom")),
    ("ff6_bab", ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "BAB")),
    ("ff6_bab_qmj", ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "BAB", "QMJ")),
)


def newey_west_hac_lag(n: int) -> int:
    """Newey & West (1994) automatic HAC truncation lag ``L = floor(4 * (n/100) ** (2/9))`` (>= 0).

    The deterministic rule-of-thumb maximum lag for a Newey-West (1987) HAC covariance on a
    length-``n`` series. Cite **Newey-West 1994** for this automatic-lag rule and **Newey-West 1987**
    for the Bartlett-kernel estimator it parameterises. (NB the ``4(n/100)^(2/9)`` rule is Newey-West
    1994 -- NOT Schwert 1989, whose ``12(n/100)^(1/4)`` is an ADF unit-root lag rule.) statsmodels'
    ``cov_hac`` uses this EXACT floored value as its ``nlags`` default; we compute it explicitly so the
    lag is logged, reproducible, and testable. Returns ``0`` for a degenerate ``n <= 0`` (an empty
    series has no autocovariance structure to model).
    """
    if n <= 0:
        return 0
    return int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def _newey_west_cov(x: np.ndarray, resid: np.ndarray, lag: int) -> np.ndarray:
    """Hand-rolled Newey-West (1987) HAC covariance of the OLS coefficients (Bartlett kernel).

    Used as the verification reference in the unit tests (and as a statsmodels-free fallback). For
    design matrix ``X`` (with intercept column) and residuals ``e``, the HAC meat is

        ``S = S_0 + sum_{l=1..L} w_l * (Gamma_l + Gamma_l^T)``,  ``w_l = 1 - l/(L+1)`` (Bartlett),

    where ``Gamma_l = sum_t (x_t e_t)(x_{t-l} e_{t-l})^T`` and ``S_0 = sum_t (x_t e_t)(x_t e_t)^T``.
    The sandwich covariance is ``(X^T X)^{-1} S (X^T X)^{-1}`` (no small-sample correction, matching
    statsmodels ``use_correction=False``). This is the textbook estimator; it is positive
    semi-definite by the Bartlett weighting (Newey-West 1987).
    """
    x = np.asarray(x, dtype=float)
    e = np.asarray(resid, dtype=float).ravel()
    n, k = x.shape
    xtx_inv = np.linalg.inv(x.T @ x)
    u = x * e[:, None]  # rows x_t * e_t  (shape n x k)
    s = u.T @ u  # S_0
    for j in range(1, lag + 1):  # j = lag index (named j, not l, to avoid the E741 ambiguous name)
        w = 1.0 - j / (lag + 1.0)  # Bartlett kernel weight w_j = 1 - j/(L+1)
        gamma = u[j:].T @ u[:-j]  # sum_t (x_t e_t)(x_{t-j} e_{t-j})^T
        s = s + w * (gamma + gamma.T)
    return xtx_inv @ s @ xtx_inv


def factor_alpha(
    returns: np.ndarray,
    factors: dict[str, np.ndarray] | np.ndarray,
    *,
    factor_names: tuple[str, ...] | None = None,
    risk_free: np.ndarray | None = None,
    hac_lag: int | None = None,
) -> dict[str, Any]:
    """Estimate the factor-model alpha of one realized return series with Newey-West HAC SEs.

    Fits the OLS ``(r_t - rf_t) = alpha + sum_k beta_k * F_{k,t} + eps_t`` and reports the intercept
    (alpha) and its HAC ``t``-statistic / p-value, the factor betas with their HAC ``t``-statistics,
    and the regression ``R^2``. The market factor ``Mkt-RF`` is itself an excess return; the LHS uses
    excess returns ``r - rf`` so ``alpha`` is Jensen's alpha (the un-priced average return).

    Parameters
    ----------
    returns : np.ndarray
        The realized per-period return series (e.g. a frozen-winner test path), length ``T``.
    factors : dict[str, np.ndarray] | np.ndarray
        Either ``{name: per-period factor series}`` (all aligned to ``returns``) or a ``(T, k)``
        matrix (then ``factor_names`` must be given). Each factor is a per-period decimal return.
    factor_names : tuple of str, optional
        Column names when ``factors`` is a bare matrix.
    risk_free : np.ndarray, optional
        Per-period risk-free decimal series aligned to ``returns``; subtracted from ``returns`` to
        form the excess-return LHS. ``None`` -> rf = 0 (the frozen-headline convention).
    hac_lag : int, optional
        Newey-West truncation lag. ``None`` -> the Newey-West (1994) rule :func:`newey_west_hac_lag` on the
        aligned length. Pass ``0`` for plain (heteroskedasticity-only) White SEs.

    Returns
    -------
    dict
        ``{"status": "ok"|"skipped", "alpha", "alpha_t", "alpha_p", "alpha_se",
        "alpha_ann", "betas": {name: beta}, "betas_t": {name: t}, "r2", "n", "hac_lag",
        "factor_names", ["reason"]}``. ``status="skipped"`` (with a ``reason``, never a raised error)
        when the series is too short, non-finite, or rank-deficient — so a degenerate arm degrades
        gracefully. ``alpha`` is per period; ``alpha_ann = alpha * 252``.
    """
    r = np.asarray(returns, dtype=float).ravel()

    # Normalise the factor block to (names, matrix), aligned to the common leading length.
    if isinstance(factors, dict):
        names = tuple(factors.keys()) if factor_names is None else tuple(factor_names)
        cols = [np.asarray(factors[name], dtype=float).ravel() for name in names]
    else:
        mat = np.asarray(factors, dtype=float)
        if mat.ndim == 1:
            mat = mat[:, None]
        if factor_names is None:
            factor_names = tuple(f"f{i}" for i in range(mat.shape[1]))
        names = tuple(factor_names)
        cols = [mat[:, j] for j in range(mat.shape[1])]

    skip = lambda reason: {  # noqa: E731 - tiny local for the uniform skipped-result shape
        "status": "skipped",
        "reason": reason,
        "alpha": None,
        "alpha_t": None,
        "alpha_p": None,
        "alpha_se": None,
        "alpha_ann": None,
        "betas": {},
        "betas_t": {},
        "r2": None,
        "n": int(min([r.size] + [c.size for c in cols]) if cols else r.size),
        "hac_lag": None,
        "factor_names": list(names),
    }

    if not names:
        return skip("no factor columns supplied")

    lengths = [r.size] + [c.size for c in cols]
    t = int(min(lengths))
    if t == 0:
        return skip("empty return/factor series")
    y_raw = r[:t]
    rf = None if risk_free is None else np.asarray(risk_free, dtype=float).ravel()[:t]
    y = y_raw if rf is None else y_raw - rf
    fmat = np.column_stack([c[:t] for c in cols])

    finite = np.isfinite(y) & np.all(np.isfinite(fmat), axis=1)
    if rf is not None:
        finite &= np.isfinite(rf)
    if finite.sum() < (len(names) + 2):
        # Need strictly more observations than parameters (intercept + k betas) for a non-degenerate
        # fit; the +2 leaves at least one residual degree of freedom.
        return skip(
            f"too few finite aligned observations ({int(finite.sum())}) for {len(names) + 1} parameters"
        )
    y = y[finite]
    fmat = fmat[finite]
    n = int(y.size)

    # Design matrix with a leading intercept column (alpha).
    x = np.column_stack([np.ones(n), fmat])
    # Rank check: a constant/collinear factor (e.g. an all-zero column from a leading alignment gap)
    # makes X'X singular -> skip rather than emit a garbage alpha.
    if np.linalg.matrix_rank(x) < x.shape[1]:
        return skip("rank-deficient design (a factor is constant or collinear over the window)")

    lag = newey_west_hac_lag(n) if hac_lag is None else int(max(0, hac_lag))

    beta_hat: np.ndarray
    cov: np.ndarray
    r2: float
    try:
        import statsmodels.api as sm

        model = sm.OLS(y, x, missing="raise")
        # cov_type="HAC" applies the Bartlett kernel with truncation `maxlags`; use_correction=False
        # matches the textbook Newey-West (and our hand-rolled reference) with no small-sample factor.
        res = model.fit(cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": False})
        beta_hat = np.asarray(res.params, dtype=float)
        cov = np.asarray(res.cov_params(), dtype=float)
        r2 = float(res.rsquared)
    except Exception as exc:  # noqa: BLE001 - fall back to the self-contained OLS+NW reference
        _LOG.warning("statsmodels HAC path failed (%s); using the hand-rolled Newey-West reference", exc)
        xtx_inv = np.linalg.inv(x.T @ x)
        beta_hat = xtx_inv @ (x.T @ y)
        resid = y - x @ beta_hat
        cov = _newey_west_cov(x, resid, lag)
        ss_res = float(resid @ resid)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    # Guard a zero/non-finite SE (a perfectly-fit or degenerate column) before forming t-stats.
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.where(se > 0, beta_hat / se, 0.0)

    # Two-sided p-value from the t distribution with (n - k) degrees of freedom.
    from scipy import stats as _stats

    dof = max(1, n - x.shape[1])
    alpha_p = float(2.0 * _stats.t.sf(abs(tstat[0]), dof))

    betas = {name: float(beta_hat[1 + j]) for j, name in enumerate(names)}
    betas_t = {name: float(tstat[1 + j]) for j, name in enumerate(names)}
    return {
        "status": "ok",
        "alpha": float(beta_hat[0]),
        "alpha_t": float(tstat[0]),
        "alpha_p": alpha_p,
        "alpha_se": float(se[0]),
        "alpha_ann": float(beta_hat[0] * 252.0),
        "betas": betas,
        "betas_t": betas_t,
        "r2": r2,
        "n": n,
        "hac_lag": int(lag),
        "factor_names": list(names),
    }


def difference_in_alpha(
    returns_a_by_seed: dict[int, np.ndarray],
    returns_b_by_seed: dict[int, np.ndarray],
    factors: dict[str, np.ndarray],
    *,
    factor_names: tuple[str, ...],
    risk_free: np.ndarray | None = None,
    hac_lag: int | None = None,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Paired across-seed test of ``alpha(arm_a) - alpha(arm_b)`` for ONE factor rung.

    The headline of this module. Mirroring the frozen per-seed rliable inference: fit the factor
    model **per seed** for each arm (one frozen-winner test path per training seed) to get that seed's
    alpha, then run the **paired across-seed bootstrap**
    (:func:`src.inference.bootstrap.paired_seed_difference_test`) on the per-seed alpha differences
    over the shared training seeds. ``effect > 0`` means the distributional arm (``a``) has the
    higher (factor-adjusted) alpha. The bootstrap is two-sided; the directional decision is layered on
    by the caller via the recorded sign of ``effect``.

    Parameters
    ----------
    returns_a_by_seed, returns_b_by_seed : dict[int, np.ndarray]
        ``{seed: realized test return series}`` for arm ``a`` (predicted better) and arm ``b``.
    factors : dict[str, np.ndarray]
        ``{name: per-period factor series}`` aligned to the test window (the SAME window every
        per-seed series spans, so a single factor block prices all seeds).
    factor_names : tuple of str
        The rung's factor columns (a subset of ``factors`` keys).
    risk_free, hac_lag
        Forwarded to :func:`factor_alpha`.
    n_boot, rng
        Bootstrap replications and generator (forwarded to ``paired_seed_difference_test``).

    Returns
    -------
    dict
        ``{"status": "ok"|"skipped", "effect", "pvalue", "stat", "ci_low", "ci_high",
        "n_seeds", "alpha_a_iqm", "alpha_b_iqm", "per_seed": {seed: (alpha_a, alpha_b)},
        ["reason"]}``. ``effect`` is the IQM-over-seeds of ``alpha_a - alpha_b`` (per period).
        ``status="skipped"`` (never raised) when fewer than two shared seeds yield a usable per-seed
        alpha for BOTH arms.
    """
    from src.inference.bootstrap import iqm, paired_seed_difference_test

    if rng is None:
        rng = np.random.default_rng(0)
    sub = {name: factors[name] for name in factor_names if name in factors}
    missing = [name for name in factor_names if name not in factors]
    if missing:
        return {
            "status": "skipped",
            "reason": f"factor panel missing columns {missing}",
            "effect": None,
            "pvalue": None,
            "stat": None,
            "ci_low": None,
            "ci_high": None,
            "n_seeds": 0,
            "alpha_a_iqm": None,
            "alpha_b_iqm": None,
            "per_seed": {},
        }

    def _alpha(series: np.ndarray) -> float | None:
        res = factor_alpha(
            series, sub, factor_names=tuple(factor_names), risk_free=risk_free, hac_lag=hac_lag
        )
        return res["alpha"] if res["status"] == "ok" else None

    shared = sorted(set(returns_a_by_seed) & set(returns_b_by_seed))
    per_seed: dict[int, tuple[float, float]] = {}
    a_vals: list[float] = []
    b_vals: list[float] = []
    for s in shared:
        aa = _alpha(returns_a_by_seed[s])
        bb = _alpha(returns_b_by_seed[s])
        if aa is None or bb is None:
            continue
        per_seed[s] = (aa, bb)
        a_vals.append(aa)
        b_vals.append(bb)

    if len(a_vals) < 2:
        return {
            "status": "skipped",
            "reason": (
                "no shared seeds with a usable per-seed alpha for both arms"
                if not a_vals
                else "only 1 shared seed with a usable per-seed alpha (need >= 2 for the across-seed bootstrap)"
            ),
            "effect": None,
            "pvalue": None,
            "stat": None,
            "ci_low": None,
            "ci_high": None,
            "n_seeds": len(a_vals),
            "alpha_a_iqm": float(iqm(np.asarray(a_vals))) if a_vals else None,
            "alpha_b_iqm": float(iqm(np.asarray(b_vals))) if b_vals else None,
            "per_seed": per_seed,
        }

    a_arr = np.asarray(a_vals, dtype=float)
    b_arr = np.asarray(b_vals, dtype=float)
    test = paired_seed_difference_test(a_arr, b_arr, statistic=iqm, n_boot=n_boot, rng=rng)
    return {
        "status": "ok",
        "effect": float(test["effect"]),
        "pvalue": float(test["pvalue"]),
        "stat": float(test["stat"]),
        "ci_low": float(test["ci_low"]),
        "ci_high": float(test["ci_high"]),
        "n_seeds": int(a_arr.size),
        "alpha_a_iqm": float(iqm(a_arr)),
        "alpha_b_iqm": float(iqm(b_arr)),
        "per_seed": per_seed,
    }


# --------------------------------------------------------------------------- #
# The campaign driver, shaped to the frozen-winner TEST records                #
# --------------------------------------------------------------------------- #
#: The pre-registered HEADLINE contrasts, mirrored from ``scripts/analyze_campaign.H2_CONTRASTS``
#: so this module does not import the analysis script (which imports torch-adjacent deps). Each
#: entry is ``(arm_a, arm_b)`` read "a predicted to have the HIGHER factor-adjusted alpha than b".
_ATTRIBUTION_CONTRASTS: tuple[tuple[str, str], ...] = (
    ("distributional", "scalar"),
    ("distributional", "placebo"),
    ("distributional", "scalar_cvar5"),
)


def _record_test_returns(record: dict[str, Any]) -> np.ndarray | None:
    """Extract a frozen-winner record's realized per-step TEST return vector, or ``None``.

    Mirrors ``scripts/analyze_campaign._test_returns``: the vector rides inside
    ``metrics['test_returns']`` (with a top-level ``test_returns`` fallback). Returns ``None`` for an
    absent / empty / non-finite vector so a missing arm degrades gracefully.
    """
    metrics = record.get("metrics") or {}
    tr = metrics.get("test_returns")
    if tr is None:
        tr = record.get("test_returns")
    if tr is None:
        return None
    arr = np.asarray(tr, dtype=float).ravel()
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    return arr


def _returns_by_seed(records: list[dict[str, Any]], arm: str) -> dict[int, np.ndarray]:
    """``{seed: realized test series}`` for one arm's frozen-winner TEST records (one per seed)."""
    out: dict[int, np.ndarray] = {}
    for r in records:
        if r.get("arm") != arm or r.get("seed") is None:
            continue
        v = _record_test_returns(r)
        if v is None:
            continue
        out[int(r["seed"])] = v
    return out


def campaign_attribution(
    records: list[dict[str, Any]],
    factors: dict[str, np.ndarray] | None,
    *,
    contrasts: tuple[tuple[str, str], ...] = _ATTRIBUTION_CONTRASTS,
    ladder: tuple[tuple[str, tuple[str, ...]], ...] = FACTOR_LADDER,
    risk_free: np.ndarray | None = None,
    hac_lag: int | None = None,
    n_boot: int = 2000,
    q: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Difference-in-alpha across the factor ladder for every contrast — the Door-C secondary driver.

    Shaped to the per-(arm, seed) frozen-winner TEST records ``scripts/run_campaign.py`` writes (each
    carrying ``metrics['test_returns']``). For each ``(arm_a, arm_b)`` contrast and each factor rung
    whose columns are ALL present, runs :func:`difference_in_alpha` (per-seed factor alphas -> paired
    across-seed bootstrap). The two-sided p-values are Benjamini-Hochberg corrected at ``q`` ACROSS
    the whole secondary family (every ``contrast x rung`` cell that ran), and the rejection set is
    returned alongside a per-cell ``direction_ok`` (distributional alpha higher). The result uses
    DISJOINT keys (``rung`` / ``factor`` / ``alpha_diff`` / ``seed``) so it can never be mistaken for
    the frozen H2 family (``arm_a`` / ``arm_b`` / ``metric`` / ``level``) — the
    ``assert_realized_family_matches_frozen`` guard stays green and the frozen m=6 is untouched.

    DEGRADES GRACEFULLY (status="skipped", never crashes) when:
      * ``factors`` is ``None`` or empty (no factor data on a synthetic-only install);
      * a contrast's arms have no test records (a missing comparator arm yields a credible null);
      * a rung's factor columns are not all present (e.g. FF5/BAB/QMJ before the data is pulled).

    Parameters
    ----------
    records : list of dict
        Per-(arm, seed) frozen-winner TEST records (carry ``metrics['test_returns']``).
    factors : dict[str, np.ndarray] | None
        ``{name: per-period factor series}`` aligned to the test window (e.g. from
        :func:`load_factor_panel`), or ``None`` when no factor data is available.
    contrasts : tuple of (str, str), optional
        Arm pairs ``(a, b)`` read "a predicted higher-alpha than b". Default the H2 contrasts.
    ladder : tuple, optional
        The factor ladder (default :data:`FACTOR_LADDER`).
    risk_free, hac_lag, n_boot, rng
        Forwarded to :func:`difference_in_alpha` / :func:`factor_alpha`.
    q : float, optional
        Benjamini-Hochberg FDR level for THIS secondary family (default 0.05).

    Returns
    -------
    dict
        ``{"status", "family": "factor_attribution_difference_in_alpha", "q", "rungs": [...],
        "cells": [ {contrast, rung, factor, status, alpha_diff, pvalue, direction_ok, reject_bh,
        n_seeds, alpha_a_iqm, alpha_b_iqm}, ... ], "n_family", "skipped": [...], "available": bool}``.
        ``available`` is ``False`` (and ``cells`` empty) when no factor data was supplied — the whole
        secondary is then reported skipped rather than fabricated.
    """
    from src.inference.multiple_testing import benjamini_hochberg

    if rng is None:
        rng = np.random.default_rng(0)

    base: dict[str, Any] = {
        "status": "ok",
        "family": "factor_attribution_difference_in_alpha",
        "q": float(q),
        "rungs": [name for name, _ in ladder],
        "cells": [],
        "n_family": 0,
        "skipped": [],
        "available": True,
    }

    if not factors:
        base.update(
            status="skipped",
            available=False,
            skipped=[{"reason": "no factor data supplied (factors is None/empty)"}],
        )
        return base

    available_cols = {name for name, col in factors.items() if np.asarray(col).size}
    returns_by_arm = {
        arm: _returns_by_seed(records, arm) for arm in {a for pair in contrasts for a in pair}
    }

    cells: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    runnable_pvals: list[float] = []
    runnable_idx: list[int] = []

    for arm_a, arm_b in contrasts:
        a_by_seed = returns_by_arm.get(arm_a, {})
        b_by_seed = returns_by_arm.get(arm_b, {})
        if not a_by_seed or not b_by_seed:
            skipped.append(
                {
                    "contrast": f"{arm_a}>{arm_b}",
                    "reason": "one or both arms have no usable frozen-winner test record",
                }
            )
            continue

        for rung_name, cols in ladder:
            cell: dict[str, Any] = {
                "contrast": f"{arm_a}>{arm_b}",
                "arm_a": arm_a,
                "arm_b": arm_b,
                "rung": rung_name,
                "factor": list(cols),
            }
            missing = [c for c in cols if c not in available_cols]
            if missing:
                cell.update(
                    status="skipped",
                    reason=f"factor columns not available: {missing}",
                    alpha_diff=None,
                    pvalue=None,
                    direction_ok=None,
                    reject_bh=False,
                    n_seeds=0,
                )
                cells.append(cell)
                continue

            res = difference_in_alpha(
                a_by_seed,
                b_by_seed,
                factors,
                factor_names=tuple(cols),
                risk_free=risk_free,
                hac_lag=hac_lag,
                n_boot=n_boot,
                rng=rng,
            )
            if res["status"] != "ok":
                cell.update(
                    status="skipped",
                    reason=res.get("reason", "difference_in_alpha skipped"),
                    alpha_diff=None,
                    pvalue=None,
                    direction_ok=None,
                    reject_bh=False,
                    n_seeds=int(res.get("n_seeds", 0)),
                    alpha_a_iqm=res.get("alpha_a_iqm"),
                    alpha_b_iqm=res.get("alpha_b_iqm"),
                )
                cells.append(cell)
                continue

            cell.update(
                status="ok",
                alpha_diff=float(res["effect"]),
                alpha_diff_ann=float(res["effect"] * 252.0),
                pvalue=float(res["pvalue"]),
                stat=float(res["stat"]),
                ci_low=float(res["ci_low"]),
                ci_high=float(res["ci_high"]),
                direction_ok=bool(res["effect"] > 0.0),
                n_seeds=int(res["n_seeds"]),
                alpha_a_iqm=float(res["alpha_a_iqm"]),
                alpha_b_iqm=float(res["alpha_b_iqm"]),
                reject_bh=False,  # filled after the family-wide BH below
            )
            runnable_idx.append(len(cells))
            runnable_pvals.append(float(res["pvalue"]))
            cells.append(cell)

    # Benjamini-Hochberg across the WHOLE secondary family (the cells that actually ran). Disjoint
    # from the frozen m=6 — corrected internally per deep-research §5.10 ("one declared family each").
    if runnable_pvals:
        reject = benjamini_hochberg(np.asarray(runnable_pvals, dtype=float), q=q)
        for idx, rj in zip(runnable_idx, reject):
            cells[idx]["reject_bh"] = bool(rj)

    base.update(
        cells=cells,
        n_family=len(runnable_pvals),
        skipped=skipped,
    )
    if not runnable_pvals:
        base["status"] = "skipped"
        if not skipped:
            base["skipped"] = [
                {"reason": "no factor rung was runnable (factor columns absent or too few seeds)"}
            ]
    return base


# --------------------------------------------------------------------------- #
# Factor-panel loader (on-disk Ken-French FF3 + Momentum; the pull-needed rest)#
# --------------------------------------------------------------------------- #
_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
_FF_CSV = "french_F-F_Research_Data_Factors_daily.csv"
_MOM_CSV = "french_F-F_Momentum_Factor_daily.csv"
#: Factor columns that COULD be wired but require an external pull (see the module + docs); listed so
#: the loader can report them as ``needs_pull`` rather than silently omitting them.
_NEEDS_PULL: tuple[str, ...] = ("RMW", "CMA", "BAB", "QMJ")


def load_factor_panel(
    dates: np.ndarray,
    *,
    raw_dir: Path | str = _RAW_DIR,
    factor_provider: Callable[[np.ndarray], dict[str, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Load the on-disk daily factor columns aligned to ``dates`` (CAPM/FF3/Carhart-4 constructable).

    Wires the factor columns the project already has on disk (deep-research §5.10 "Factor sourcing"):
    ``Mkt-RF``, ``SMB``, ``HML``, ``RF`` from the Ken-French research-factors daily file (via
    ``src.data.market_reference.load_ff_factors``, which forward-fills with no future leak), and
    ``Mom`` / UMD from the Ken-French momentum daily file. -> the CAPM, FF3, and Carhart-4 ladder
    rungs are estimable today; FF5 (``RMW``/``CMA``) and the AQR ``BAB``/``QMJ`` rungs are reported as
    ``needs_pull`` until that data is added.

    An optional ``factor_provider(dates) -> {name: series}`` is merged on top (its keys win), so a
    caller who has pulled FF5/BAB/QMJ can inject them without changing this signature. DEGRADES
    GRACEFULLY to ``available=False`` with an empty ``factors`` dict when the on-disk files are absent
    (a synthetic-only install) and no provider supplies them.

    Returns
    -------
    dict
        ``{"factors": {name: ndarray aligned to dates}, "available": bool, "present": [...],
        "needs_pull": [...], "rf": ndarray|None}``. ``present`` lists the loaded factor columns;
        ``needs_pull`` lists the ladder columns still missing. ``rf`` is the Ken-French daily RF
        column when available (the LHS excess-return convention may instead use FRED DGS3MO — see the
        module docstring / docs).
    """
    from src.data.market_reference import load_ff_factors

    dates = np.asarray(dates)
    factors: dict[str, np.ndarray] = {}
    rf_series: np.ndarray | None = None

    # FF3 (Mkt-RF/SMB/HML) via the existing reference loader (aligned + forward-filled, no future leak).
    ff = load_ff_factors(dates, raw_dir=raw_dir)
    if ff.available:
        factors.update({k: np.asarray(v, dtype=float) for k, v in ff.factors.items()})

    # RF column (Ken-French daily) and Mom column are read directly here (the reference loader exposes
    # only Mkt-RF/SMB/HML). Reuse the same forward-fill alignment contract via the loader's helper.
    raw = Path(raw_dir)
    try:
        import pandas as pd

        from src.data.market_reference import _aligned_series  # same no-future-leak alignment
        from src.data.market_reference import _raw_path  # refresh-preferring resolver (2026-07-05)

        # Route through the x26-refresh resolver: the direct `raw / _FF_CSV` reads bypassed the
        # frozen refresh files, so RF/Mom ended 2026-04-30 and stale-ffilled the test tail while the
        # sibling FF3 loader (load_ff_factors) already used the fresher data (2026-07-05 fix).
        ff_path = _raw_path(raw, _FF_CSV)
        if ff_path.exists():
            ff_df = pd.read_csv(ff_path)
            date_col = "Date" if "Date" in ff_df.columns else ff_df.columns[0]
            ff_df = ff_df.set_index(date_col)
            if "RF" in ff_df.columns:
                rf_vals, _ = _aligned_series(ff_df, "RF", dates)
                rf_series = np.nan_to_num(rf_vals, nan=0.0)
        mom_path = _raw_path(raw, _MOM_CSV)
        if mom_path.exists():
            mom_df = pd.read_csv(mom_path)
            mdate = "Date" if "Date" in mom_df.columns else mom_df.columns[0]
            mom_df = mom_df.set_index(mdate)
            mom_col = "Mom" if "Mom" in mom_df.columns else ("Mom   " if "Mom   " in mom_df.columns else None)
            if mom_col is not None:
                mom_vals, _ = _aligned_series(mom_df, mom_col, dates)
                factors["Mom"] = np.nan_to_num(mom_vals, nan=0.0)
    except Exception as exc:  # noqa: BLE001 - a factor loader must degrade, never break the analysis
        _LOG.warning("factor-panel on-disk read failed (%s); proceeding with what loaded", exc)

    # Optional injected provider (e.g. a future FF5/BAB/QMJ pull); its columns override.
    if factor_provider is not None:
        try:
            extra = factor_provider(dates)
            for k, v in (extra or {}).items():
                factors[k] = np.asarray(v, dtype=float)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("factor_provider failed (%s); ignoring it", exc)

    present = sorted(factors.keys())
    needs_pull = [c for c in _NEEDS_PULL if c not in factors]
    return {
        "factors": factors,
        "available": bool(factors),
        "present": present,
        "needs_pull": needs_pull,
        "rf": rf_series,
    }


def attribution_markdown(result: dict[str, Any]) -> str:
    """Render the difference-in-alpha secondary as a markdown table (analysis-report fragment).

    Reports, per ``contrast x rung`` cell, the annualised alpha difference (distributional minus the
    comparator), its two-sided across-seed bootstrap p-value, the direction flag, and the family-wide
    BH rejection. Skipped rungs (factor data absent / too few seeds) are shown with their reason so a
    reader sees exactly which rungs the data supports.
    """
    if not result or result.get("status") == "skipped" and not result.get("cells"):
        reason = "no factor data" if not result or not result.get("available") else "no runnable rung"
        return (
            "## Factor attribution — difference-in-alpha (Door-C secondary) — n/a\n\n"
            f"Not computed ({reason}).\n"
        )
    out = [
        "## Factor attribution — difference-in-alpha (Door-C secondary; deep-research §5.10)",
        "",
        "Per-seed factor-model alpha (Newey-West HAC SE, Newey-West-1994 lag) -> paired across-seed "
        f"bootstrap of the alpha difference. Separate declared family (BH q={float(result.get('q', 0.05)):.2f}, "
        f"m={int(result.get('n_family', 0))}); disjoint keys from the frozen m=6. "
        "Positive = the distributional arm keeps the higher alpha AFTER the factor controls.",
        "",
        "| contrast | rung | factors | alpha diff (ann) | p-value | direction_ok | reject_bh | n_seeds | status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in result.get("cells", []):
        if c.get("status") == "ok":
            ad = f"{c.get('alpha_diff_ann', 0.0):+.4f}"
            pv = f"{c.get('pvalue', float('nan')):.4f}"
        else:
            ad = "n/a"
            pv = "n/a"
        status = c.get("status", "?")
        if status == "skipped" and c.get("reason"):
            status = f"skipped ({c['reason']})"
        out.append(
            f"| {c.get('contrast')} | {c.get('rung')} | {'+'.join(c.get('factor', []))} | {ad} | {pv} | "
            f"{c.get('direction_ok')} | {c.get('reject_bh')} | {c.get('n_seeds', 0)} | {status} |"
        )
    skipped = result.get("skipped", [])
    if skipped:
        out += ["", "Skipped contrasts (no usable record / not fabricated):"]
        for s in skipped:
            out.append(f"- {s.get('contrast', '?')}: {s.get('reason', '')}")
    out.append("")
    return "\n".join(out)
