"""Comparative Expected-Shortfall (CVaR) backtesting via the jointly-elicitable (VaR, ES) pair.

Closes the methodology gap that **ES alone is not elicitable** (Gneiting, 2011, *Making and evaluating
point forecasts*, JASA 106(494):746-762 — `gneiting2011making`) — so no strictly consistent loss exists
for ES by itself — whereas the **pair (VaR_alpha, ES_alpha) IS jointly elicitable** (Fissler & Ziegel,
2016, *Higher order elicitability and Osband's principle*, Ann. Statist. 44(4):1680-1707). That joint
elicitability licenses a strictly consistent scoring function, and comparing the *tail-forecast accuracy*
of two models is then a **Diebold-Mariano-style comparative backtest** on the per-period score
differential (Nolde & Ziegel, 2017, *Elicitability and backtesting*, Ann. Appl. Statist.
11(4):1833-1874). The score-difference variance may be estimated by a HAC estimator or the stationary
bootstrap (the latter reused here).

Scope / what this is NOT (read carefully — these answer different questions):
  * THIS module answers: "does risk-FORECAST model 1 predict the tail better than model 2 on the SAME
    realized return series?" — a *comparative backtest* (Nolde-Ziegel). Use it e.g. to check whether the
    measurement instrument's (VaR, ES) forecasts beat a benchmark's.
  * Comparing the *realized* CVaR of two DIFFERENT return series (e.g. the distributional vs scalar arm,
    which have different policies and hence different realizations) is a TWO-SAMPLE comparison of the ES
    functional. No published named test exists for that; use the re-centred stationary block bootstrap in
    ``src.inference.bootstrap.cvar_difference_test`` (size MEASURED — not certified — by
    ``null_calibration``, audit C-7: mildly ANTI-conservative, 0.0573 two-sided / 0.0613 one-sided
    against nominal 0.05 at production settings; not studentized — the bootstrap SE cancels in its
    p-value).

Power caveat (audit B-7 / Bauer 2025, arXiv:2505.23333): comparative tail-risk tests have **low power at
the most extreme quantiles** (alpha = 1%, 2.5%) and short out-of-sample windows; power improves with
alpha and OOS length. Report this for CVaR-1% comparisons.

Convention: lower tail at level ``alpha`` (small, e.g. 0.05); ``var`` = the alpha-quantile of returns,
``es`` = E[r | r <= var] <= var; both negative for a loss tail. The scoring function is the FZ0
(0-homogeneous) strictly consistent loss (Patton, Ziegel & Chen 2019; GAS::FZLoss):

    S(v, e, r) = -(1 / (alpha * e)) * 1{r <= v} * (v - r) + v / e + log(-e) - 1,   e < 0.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from src.inference.bootstrap import stationary_bootstrap_indices

__all__ = [
    "fz0_loss",
    "var_es_estimates",
    "comparative_es_backtest",
    "hln_factor",
    "dm_hln_test",
    "dm_size_power_calibration",
]


def fz0_loss(
    returns: np.ndarray,
    var: float | np.ndarray,
    es: float | np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Per-observation FZ0 strictly consistent score for the (VaR_alpha, ES_alpha) pair.

    Strict consistency: ``E[fz0_loss]`` is uniquely minimized (over ``(v, e)``) at the true
    ``(VaR_alpha, ES_alpha)`` — this is the property exploited by the comparative backtest and asserted
    in the tests.

    Parameters
    ----------
    returns : np.ndarray
        Realized return series (lower values = worse).
    var, es : float or np.ndarray
        The (VaR, ES) forecast(s); scalar (constant forecast) or per-period arrays. ``es`` must be < 0.
    alpha : float
        Tail level in (0, 1).

    Returns
    -------
    np.ndarray
        The per-observation scores (same length as ``returns``).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    r = np.asarray(returns, dtype=float)
    v = np.asarray(var, dtype=float)
    e = np.asarray(es, dtype=float)
    if np.any(e >= 0.0):
        raise ValueError("es must be strictly negative (a loss) for the FZ0 score")
    hit = (r <= v).astype(float)
    return -(1.0 / (alpha * e)) * hit * (v - r) + v / e + np.log(-e) - 1.0


def var_es_estimates(returns: np.ndarray, alpha: float) -> tuple[float, float]:
    """Empirical (VaR_alpha, ES_alpha) of a return series (lower tail).

    Returns ``(var, es)`` with ``var`` the (interpolated) empirical alpha-quantile and ``es`` the
    mean of the returns AT OR BELOW that quantile — the one-convention A-L1 pair guaranteeing
    ES <= VaR (row 30d docstring fix: the old text claimed mean-of-worst-``ceil(alpha*T)``, which
    is the HOUSE headline-CVaR convention but deliberately NOT what this ES-backtest input uses;
    the code below was always the documented A-L1 choice — the words now match it).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]  # strip non-finite so a NaN can't poison VaR (matches bootstrap.cvar, P0-1)
    if r.size == 0:
        return float("nan"), float("nan")
    # Coherent (VaR, ES) on ONE convention: VaR is the empirical alpha-quantile and ES the mean of the returns
    # AT OR BELOW it, so ES <= VaR by construction (A-L1, 2026-07-02). Mixing an interpolated-quantile VaR with a
    # count-based mean-of-worst-k ES could invert on tiny samples; on the large validation windows this touches,
    # the two conventions agree to within sampling noise.
    var = float(np.quantile(r, alpha))
    tail = r[r <= var]  # returns at or below VaR
    es = float(tail.mean()) if tail.size else var
    return var, es


def hln_factor(t: int, h: int = 1) -> float:
    """Harvey-Leybourne-Newbold (1997) small-sample DM rescaling factor.

    Harvey, Leybourne & Newbold (1997, *Testing the equality of prediction mean squared errors*,
    Int. J. Forecasting 13(2):281-291) show that the asymptotic-normal Diebold-Mariano (1995) test
    **over-rejects in small samples**, and propose (i) rescaling the DM statistic by

        sqrt[ (T + 1 - 2h + h(h-1)/T) / T ]

    and (ii) comparing the rescaled statistic against a **Student-t(T-1)** distribution rather than
    the standard normal. Both changes make the test more conservative at small ``T``. Here ``h`` is the
    forecast horizon (``h = 1`` for the one-step (VaR, ES) tail forecast). The factor is < 1 for the
    relevant ``T >> h`` regime, so the corrected statistic is shrunk toward zero.

    Parameters
    ----------
    t : int
        Number of score-differential observations ``T`` (must be >= 1).
    h : int
        Forecast horizon (default 1).

    Returns
    -------
    float
        The multiplicative HLN correction factor.
    """
    if t < 1:
        raise ValueError("t must be >= 1")
    if h < 1:
        raise ValueError("h must be >= 1")
    return float(np.sqrt((t + 1 - 2 * h + h * (h - 1) / t) / t))


def _hac_long_run_variance(d: np.ndarray, h: int) -> float:
    """Newey-West (Bartlett-kernel) long-run variance of the mean of ``d`` for an h-step forecast.

    The classical Diebold-Mariano variance: ``gamma_0 + 2 * sum_{k=1}^{h-1} gamma_k`` with the
    Bartlett (triangular) weights ``1 - k/h`` guaranteeing a non-negative estimate. For ``h = 1`` this
    is just the sample variance ``gamma_0`` (no autocovariance terms), matching the standard one-step
    DM test. Returned value is the variance of the *mean* (i.e. divided by ``T``).
    """
    d = np.asarray(d, dtype=float)
    n = d.size
    dc = d - d.mean()
    gamma0 = float(dc @ dc) / n
    lrv = gamma0
    for k in range(1, h):
        if k >= n:
            break
        gamma_k = float(dc[k:] @ dc[:-k]) / n
        lrv += 2.0 * (1.0 - k / h) * gamma_k
    lrv = max(lrv, 0.0)  # Bartlett guarantees >= 0; guard float residue
    return lrv / n


def dm_hln_test(d: np.ndarray, h: int = 1) -> dict[str, float]:
    """Classical Diebold-Mariano test on a score differential, with the HLN small-sample correction.

    Computes the asymptotic DM statistic ``DM = mean(d) / sqrt(Var_hat[mean(d)])`` using a Newey-West
    (Bartlett) long-run variance (DM 1995; for ``h = 1`` this reduces to the plain sample variance),
    then reports BOTH:

    * the **uncorrected** two-sided p-value against ``N(0, 1)`` (Diebold-Mariano 1995), and
    * the **HLN-corrected** statistic ``DM * hln_factor(T, h)`` with a two-sided p-value against
      ``Student-t(T - 1)`` (Harvey-Leybourne-Newbold 1997) — the conservative, *headline* small-sample
      result.

    This complements the stationary-bootstrap p-value returned by :func:`comparative_es_backtest`: the
    bootstrap is HAC-robust but the parametric DM-HLN gives a closed-form, certifiable small-sample
    reference whose size/power can be Monte-Carlo calibrated (:func:`dm_size_power_calibration`).

    Parameters
    ----------
    d : np.ndarray
        Per-period score differential ``d_t = S(f1) - S(f2)`` (length ``T``).
    h : int
        Forecast horizon (default 1 for the one-step tail forecast).

    Returns
    -------
    dict
        ``{"dm_stat", "pvalue_normal", "dm_stat_hln", "pvalue_hln", "hln_factor", "n"}``. A small
        ``pvalue_hln`` rejects equal predictive accuracy; ``mean(d) < 0`` => model 1 (f1) is better.
    """
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 2:
        return {
            "dm_stat": 0.0,
            "pvalue_normal": 1.0,
            "dm_stat_hln": 0.0,
            "pvalue_hln": 1.0,
            "hln_factor": float("nan"),
            "n": float(n),
        }
    var_mean = _hac_long_run_variance(d, h)
    se = float(np.sqrt(var_mean))
    # Relative near-zero guard: a (near-)constant differential leaves only float residue in the
    # variance, so an exact `se == 0` test silently fails on values like 0.7 (cf. bootstrap.sharpe_ratio
    # P0-1). A degenerate differential carries no evidence against equal accuracy -> p = 1.
    scale = abs(float(d.mean())) + float(np.abs(d).max())
    if not np.isfinite(se) or se <= 1e-12 * (scale + 1e-12) or float(np.ptp(d)) == 0.0:
        return {
            "dm_stat": 0.0,
            "pvalue_normal": 1.0,
            "dm_stat_hln": 0.0,
            "pvalue_hln": 1.0,
            "hln_factor": hln_factor(n, h),
            "n": float(n),
        }
    dm = float(d.mean()) / se
    factor = hln_factor(n, h)
    dm_hln = dm * factor
    pval_normal = float(2.0 * stats.norm.sf(abs(dm)))
    pval_hln = float(2.0 * stats.t.sf(abs(dm_hln), df=n - 1))
    return {
        "dm_stat": dm,
        "pvalue_normal": min(1.0, pval_normal),
        "dm_stat_hln": dm_hln,
        "pvalue_hln": min(1.0, pval_hln),
        "hln_factor": factor,
        "n": float(n),
    }


def dm_size_power_calibration(
    t: int,
    *,
    alpha: float = 0.05,
    h: int = 1,
    effect: float = 0.25,
    ar1: float = 0.0,
    n_reps: int = 2000,
    seed: int = 12345,
) -> dict[str, float]:
    """Monte-Carlo empirical size and power of the DM / DM-HLN test at window length ``T``.

    Simulates ``n_reps`` score-differential series of length ``T`` and reports the empirical rejection
    rate of :func:`dm_hln_test` at the nominal level ``alpha``:

    * **size**  — under ``H0: E[d] = 0`` (zero-mean differential), the fraction of rejections; a
      well-sized test has size near ``alpha``. Reported for BOTH the uncorrected (normal) and the
      HLN-corrected (t) p-values, so the dissertation can state the *actual* size rather than assume
      the nominal one (the HLN size should sit closer to ``alpha`` at small ``T``).
    * **power** — under a fixed alternative ``E[d] = effect`` (in units of the differential's standard
      deviation), the fraction of rejections — the test's ability to detect a real tail-accuracy gap.

    An ``ar1`` parameter injects first-order autocorrelation into the differential (volatility-cluster
    proxy) to exercise the HAC variance. Determinism: seeded from the supplied ``seed`` (the repo seeds
    every stack from the run seed; no wall-clock / global RNG). Keep ``n_reps`` modest in tests.

    Parameters
    ----------
    t : int
        Window length (number of score-differential observations).
    alpha : float
        Nominal two-sided significance level.
    h : int
        Forecast horizon passed through to :func:`dm_hln_test`.
    effect : float
        Alternative mean of the differential, in standard-deviation units (power scenario).
    ar1 : float
        AR(1) coefficient of the simulated differential (0 => i.i.d.).
    n_reps : int
        Monte-Carlo replications.
    seed : int
        Deterministic seed for the simulation RNG.

    Returns
    -------
    dict
        ``{"size_normal", "size_hln", "power_normal", "power_hln", "t", "n_reps", "alpha"}``.
    """
    if t < 2:
        raise ValueError("t must be >= 2 for a DM test")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not -1.0 < ar1 < 1.0:
        raise ValueError("ar1 must be in (-1, 1)")
    rng = np.random.default_rng(seed)

    def _simulate(mean_shift: float) -> tuple[float, float]:
        rej_normal = 0
        rej_hln = 0
        # Unit-variance innovations; for AR(1) scale so the stationary series has ~unit variance.
        innov_sd = np.sqrt(1.0 - ar1 * ar1) if ar1 != 0.0 else 1.0
        for _ in range(n_reps):
            eps = rng.standard_normal(t) * innov_sd
            if ar1 == 0.0:
                d = eps
            else:
                d = np.empty(t, dtype=float)
                d[0] = rng.standard_normal()  # stationary start (unit variance)
                for k in range(1, t):
                    d[k] = ar1 * d[k - 1] + eps[k]
            d = d + mean_shift  # mean_shift already in sd units (series has ~unit sd)
            res = dm_hln_test(d, h=h)
            if res["pvalue_normal"] < alpha:
                rej_normal += 1
            if res["pvalue_hln"] < alpha:
                rej_hln += 1
        return rej_normal / n_reps, rej_hln / n_reps

    size_normal, size_hln = _simulate(0.0)
    power_normal, power_hln = _simulate(effect)
    return {
        "size_normal": size_normal,
        "size_hln": size_hln,
        "power_normal": power_normal,
        "power_hln": power_hln,
        "t": float(t),
        "n_reps": float(n_reps),
        "alpha": alpha,
    }


def comparative_es_backtest(
    realized: np.ndarray,
    forecast1: tuple[float | np.ndarray, float | np.ndarray],
    forecast2: tuple[float | np.ndarray, float | np.ndarray],
    alpha: float = 0.05,
    p: float = 0.1,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
    h: int = 1,
) -> dict[str, object]:
    """Two-sided Diebold-Mariano *equal-accuracy* ES backtest on the FZ0 score differential, on a common series.

    Tests H0: the two (VaR, ES) forecasts have equal predictive accuracy for the tail, via the mean of
    the per-period FZ0 score differential ``d_t = S(f1) - S(f2)`` (Fissler-Ziegel jointly-consistent scoring;
    the Diebold-Mariano score-difference apparatus of Nolde & Ziegel 2017). We run it TWO-SIDED (equal
    predictive accuracy), NOT in Nolde & Ziegel's *one-sided* comparative / three-zone form (their H0:
    "forecast 1 predicts at least as well as forecast 2"): the two-sided null is the conservative choice —
    it never credits a forecast for a directional tail advantage (the sign of ``mean_score_diff`` carries the
    direction; the two-sided p-value is the significance). The standard error of ``mean(d)`` is
    estimated by the stationary bootstrap (Politis-Romano 1994), a valid HAC-robust choice for the
    autocorrelated score differential.

    Parameters
    ----------
    realized : np.ndarray
        The common realized return series both forecasts are scored against.
    forecast1, forecast2 : (var, es)
        Each model's (VaR, ES) forecast — scalar or per-period array (es < 0).
    alpha : float
        Tail level.
    p, n_boot, rng :
        Stationary-bootstrap block-restart probability, replications, and generator.
    h : int
        Forecast horizon for the parametric DM-HLN companion (default 1; the (VaR, ES) tail forecast is
        one-step). Does not affect the bootstrap p-value.

    Returns
    -------
    dict
        ``{"mean_score_diff", "stat", "pvalue", "better", "pvalue_dm_normal", "pvalue_dm_hln",
        "dm_stat", "dm_stat_hln"}``. ``mean_score_diff < 0`` => model 1 scores lower (better tail
        forecast); ``"better"`` is "model1" / "model2" / "tie". ``pvalue`` is the HAC-robust stationary
        bootstrap p-value (unchanged). The ``*_dm_*`` keys add the closed-form Diebold-Mariano companion:
        ``pvalue_dm_normal`` is the uncorrected (N(0,1)) DM p-value and ``pvalue_dm_hln`` is the
        Harvey-Leybourne-Newbold (1997) small-sample-corrected p-value (Student-t(T-1)) — the latter is
        the conservative **headline** small-sample result (``pvalue_dm_hln >= pvalue_dm_normal`` at small
        T). All p-values are report-only and corroborate H2-Tail; none gate the frozen m=6.
    """
    r = np.asarray(realized, dtype=float)
    s1 = fz0_loss(r, forecast1[0], forecast1[1], alpha)
    s2 = fz0_loss(r, forecast2[0], forecast2[1], alpha)
    d = s1 - s2
    obs = float(d.mean())
    if rng is None:
        rng = np.random.default_rng(0)  # P24: deterministic fallback (reported paths pass an explicit rng)

    n = d.size
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boot[i] = d[stationary_bootstrap_indices(n, p, rng)].mean()
    se = boot.std(ddof=1)

    if se > 0.0 and np.isfinite(se):
        stat = obs / se
        centred = (boot - obs) / se
        pvalue = float((np.abs(centred) >= abs(stat)).mean())
    else:
        stat, pvalue = 0.0, 1.0
    pvalue = min(1.0, max(pvalue, 1.0 / (n_boot + 1)))

    # Closed-form Diebold-Mariano companion (Newey-West variance) with the HLN small-sample correction
    # (report-only): gives a certifiable parametric p-value alongside the HAC-robust bootstrap one.
    dm = dm_hln_test(d, h=h)

    # B.5.2 (2026-07-06): the tail-index of the LOSS DIFFERENTIAL — heavy tails of d_t distort the
    # DM companion's size (its Newey-West variance needs finite fourth-ish moments to behave at these
    # sample sizes). Hill estimator on the upper order statistics of |d_t - median(d_t)|, k = the
    # canonical 5% of T floored at 10. hill_alpha <= 4 flags a heavy-tailed differential: read the
    # bootstrap p as the headline and the DM-HLN companion with extra caution.
    abs_c = np.sort(np.abs(d - np.median(d)))
    k = max(10, int(0.05 * n))
    tail_block: dict[str, Any] = {"hill_alpha": float("nan"), "k": int(k),
                                  "heavy_tailed_for_dm_companion": None}
    if abs_c.size > k and abs_c[-(k + 1)] > 0.0:
        logs = np.log(abs_c[-k:] / abs_c[-(k + 1)])
        mean_log = float(np.mean(logs))
        if mean_log > 0.0:
            hill_alpha = 1.0 / mean_log
            tail_block = {"hill_alpha": float(hill_alpha), "k": int(k),
                          "heavy_tailed_for_dm_companion": bool(hill_alpha <= 4.0)}

    better = "model1" if obs < 0 else "model2" if obs > 0 else "tie"
    return {
        "mean_score_diff": obs,
        "stat": float(stat),
        "pvalue": pvalue,
        "better": better,
        "dm_stat": dm["dm_stat"],
        "dm_stat_hln": dm["dm_stat_hln"],
        "pvalue_dm_normal": dm["pvalue_normal"],
        "pvalue_dm_hln": dm["pvalue_hln"],
        "loss_diff_tail": tail_block,
    }
