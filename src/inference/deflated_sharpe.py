"""Deflated Sharpe Ratio and minimum track-record length (FINAL_PLAN F.11).

Purpose
-------
Provide the Probabilistic / Deflated Sharpe Ratio (PSR / DSR) and Minimum Track
Record Length (MinTRL) of Bailey & Lopez de Prado (2012, 2014) as a SECONDARY
overfitting diagnostic. DSR discounts an observed Sharpe ratio for (a) the
number of trials, (b) the length of the track record, and (c) the
non-normality (skewness / kurtosis) of the returns.

Why secondary
-------------
The trial count ``n_trials`` and the cross-sectional Sharpe variance ``var_sr``
are derived under an *independent-trials* assumption (Bailey & Lopez de Prado),
so both are genuinely ill-defined under a guided, sequential LLM search where each
proposal conditions on prior outcomes. Per AFML (2018) the right input is an
*effective* N estimated by clustering correlated trials, used as a conservative
upper bound. PBO/CSCV in ``src.inference.overfitting`` — which is rank-based and
needs no trial count — is therefore the PRIMARY guard; DSR is reported alongside
it with the effective-``n_trials`` caveat stated explicitly.

Formulae
--------
Probabilistic Sharpe Ratio for an observed (per-period) Sharpe ``SR`` over ``n``
observations with skewness ``skew`` and (raw) kurtosis ``kurt``::

    PSR = Phi( (SR - SR_star) * sqrt(n - 1)
               / sqrt(1 - skew * SR + (kurt - 1) / 4 * SR**2) )

Expected maximum Sharpe under ``N`` independent trials of zero true Sharpe::

    E[max SR] = sqrt(var_sr) * ( (1 - g) * Phi^{-1}(1 - 1/N)
                                  + g * Phi^{-1}(1 - 1/(N*e)) )

with ``g`` the Euler-Mascheroni constant. The Deflated Sharpe Ratio is the PSR
evaluated against ``SR_star = E[max SR]``.

Convention
----------
All Sharpe quantities inside the PSR/DSR formula are *non-annualized*
(per-period). ``kurt`` is the *raw* (not excess) kurtosis -- normal => 3.
"""

from __future__ import annotations


import math

import numpy as np
from scipy.stats import norm

__all__ = [
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "min_track_record_length",
    # Backwards-compatible aliases.
    "deflated_sharpe",
]

_EULER_MASCHERONI = 0.5772156649015329


def probabilistic_sharpe_ratio(
    sr: float,
    sr_star: float,
    n: int,
    skew: float,
    kurt: float,
) -> float:
    """Probabilistic Sharpe Ratio (Bailey & Lopez de Prado, 2012).

    Parameters
    ----------
    sr:
        Observed (per-period) Sharpe ratio.
    sr_star:
        Benchmark Sharpe to test against (e.g. ``0`` or the expected maximum).
    n:
        Number of return observations.
    skew:
        Skewness of the returns.
    kurt:
        Raw kurtosis of the returns (normal => ``3``).

    Returns
    -------
    float
        ``Phi((sr - sr_star) * sqrt(n - 1) / sqrt(1 - skew*sr + (kurt-1)/4*sr^2))``
        in ``[0, 1]``.
    """
    denom_var = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    if denom_var <= 0.0 or not np.isfinite(denom_var):
        # Degenerate denominator: fall back to a sign-based decision.
        return 1.0 if sr > sr_star else 0.0
    z = (sr - sr_star) * math.sqrt(max(n - 1, 0)) / math.sqrt(denom_var)
    return float(norm.cdf(z))


def expected_max_sharpe(var_sr: float, n_trials: int) -> float:
    """Expected maximum Sharpe under ``n_trials`` independent zero-Sharpe trials.

    Parameters
    ----------
    var_sr:
        Variance of the Sharpe estimates across trials.
    n_trials:
        Number of trials ``N`` (``>= 1``).

    Returns
    -------
    float
        ``sqrt(var_sr) * ((1-g) * Phi^{-1}(1 - 1/N) + g * Phi^{-1}(1 - 1/(N*e)))``
        where ``g`` is the Euler-Mascheroni constant. Returns ``0.0`` for
        ``N <= 1``.
    """
    n = int(n_trials)
    # N <= 1 has NO multiplicity: the expected maximum of a single zero-skill trial is the expected
    # single Sharpe = 0, so sr_star = 0 and the DSR reduces to the PSR against zero. The guard MUST be
    # ``n <= 1`` (this function's documented contract): with ``n == 1``, ``norm.ppf(1 - 1/n) = ppf(0) =
    # -inf`` made ``sr_star = -inf`` and hence ``deflated_sharpe_ratio(x, n_trials=1) == 1.0`` for EVERY
    # series -- even a strongly NEGATIVE-Sharpe one. That silently broke the H1/T0 benchmark-floor gate
    # (every un-searched benchmark DSR saturated to 1.0, so the winner could never clear the floor) and
    # the ``winner_dsr_undeflated_n1`` transparency value (R65 / DEEP_AUDIT 2026-06-28).
    if n <= 1 or var_sr <= 0.0:
        return 0.0
    g = _EULER_MASCHERONI
    e = math.e
    term = (1.0 - g) * norm.ppf(1.0 - 1.0 / n) + g * norm.ppf(1.0 - 1.0 / (n * e))
    return float(math.sqrt(var_sr) * term)


def _sample_moments(returns: np.ndarray) -> tuple[float, float, float, int]:
    """Return per-period (sr, skew, raw_kurt, n) from a return series.

    P0-1 (audit 2026-06-19): strip non-finite inputs and detect a near-constant
    series with a RELATIVE near-zero guard. An exact ``sd == 0`` test silently
    fails on a near-constant reward (``np.full(750, 1e-3).std()`` is ``~2e-19``,
    not exactly ``0``), which produced a spurious ``sr ~ 1e15`` and a degenerate
    ``DSR == 1.0`` -- so a flat reward would have WON candidate selection.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]  # a pathological reward must not poison the moments
    n = r.size
    if n < 2:
        return 0.0, 0.0, 3.0, n
    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    # Relative near-zero guard (+ exact-constant ptp check): treat a (near-)constant
    # series as zero-information rather than dividing by float residue.
    if not np.isfinite(sd) or sd <= 1e-12 * (abs(mu) + 1e-12) or np.ptp(r) == 0.0:
        return 0.0, 0.0, 3.0, n
    sr = mu / sd
    z = (r - mu) / sd
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4))  # raw kurtosis (normal => 3)
    return float(sr), skew, kurt, n


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
    var_sr: float | None = None,
    periods_per_year: int = 252,
) -> float:
    """Deflated Sharpe Ratio of a return series.

    Parameters
    ----------
    returns:
        One-dimensional array of per-period returns for the selected strategy.
    n_trials:
        Number of trials ``N`` for the expected-maximum benchmark.
    var_sr:
        Optional variance of the trial Sharpe estimates -- the CROSS-TRIAL
        Sharpe DISPERSION that the canonical Bailey-Lopez de Prado DSR requires
        (empirically ``Var`` of the candidate population's per-period Sharpes,
        e.g. ``np.var(per_candidate_val_sharpes, ddof=1)``).

        If omitted (``var_sr=None``) a WITHIN-SERIES proxy is substituted: the
        single-strategy SAMPLING variance of ONE Sharpe estimate,
        ``Var[SR_hat] = (1 - skew*SR + (kurt-1)/4 SR^2)/(n-1)``. This proxy is a
        DIFFERENT quantity from the cross-trial dispersion and coincides with it
        ONLY under the homogeneous zero-skill null (all trials drawn from the
        same zero-Sharpe distribution, so the variance of the Sharpe estimator
        equals the variance of Sharpes across trials). On a HETEROGENEOUS
        candidate population (a real dispersed-skill search) the two diverge, so
        ``var_sr=None`` SILENTLY MIS-STATES the selection statistic; pass the
        empirical cross-trial variance for the canonical DSR (Rank 16). The WIRED
        per-candidate selection path passes ``None`` (the population variance is
        only knowable once an arm finishes); ``scripts/analyze_campaign.py``
        recomputes the headline winner DSR with the empirical ``var_sr``.
    periods_per_year:
        Retained for signature compatibility; the DSR is computed on the
        non-annualized per-period Sharpe and is annualization-invariant.

    Returns
    -------
    float
        DSR (a PSR against the expected-maximum benchmark) in ``[0, 1]``.
    """
    sr, skew, kurt, n = _sample_moments(returns)
    if n < 2:
        return 0.0
    if var_sr is None:
        denom_var = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
        denom_var = max(denom_var, 1e-12)
        var_sr = denom_var / (n - 1)
    sr_star = expected_max_sharpe(var_sr, n_trials)
    return probabilistic_sharpe_ratio(sr, sr_star, n, skew, kurt)


def min_track_record_length(
    sr: float,
    sr_star: float,
    skew: float,
    kurt: float,
    target_prob: float = 0.95,
) -> float:
    """Minimum track-record length for ``sr`` to beat ``sr_star`` at ``target_prob``.

    Parameters
    ----------
    sr:
        Observed (per-period) Sharpe ratio (must exceed ``sr_star``).
    sr_star:
        Benchmark Sharpe to test against.
    skew:
        Skewness of the returns.
    kurt:
        Raw kurtosis of the returns (normal => ``3``).
    target_prob:
        Desired confidence level (e.g. ``0.95``).

    Returns
    -------
    float
        Minimum number of observations ``T`` such that the PSR reaches
        ``target_prob``. ``inf`` if ``sr <= sr_star`` (never significant).
    """
    if sr <= sr_star:
        return float("inf")
    denom_var = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    denom_var = max(denom_var, 1e-12)
    z = norm.ppf(target_prob)
    mintrl = 1.0 + denom_var * (z / (sr - sr_star)) ** 2
    return float(mintrl)


# ---------------------------------------------------------------------------
# Backwards-compatible alias matching the original stub signature.
# ---------------------------------------------------------------------------
def deflated_sharpe(
    returns: np.ndarray,
    n_trials: int,
    *,
    sr_benchmark: float = 0.0,
    trial_sr_variance: float | None = None,
) -> float:
    """Alias for :func:`deflated_sharpe_ratio` (legacy signature).

    ``sr_benchmark`` is NOT supported by the canonical implementation (which always benchmarks
    against the expected-max Sharpe ``sr_star``); a non-zero value is rejected LOUDLY rather than
    silently ignored (final-audit #31 — a silently-dropped parameter is a correctness trap).
    """
    if sr_benchmark != 0.0:
        raise NotImplementedError(
            "deflated_sharpe(sr_benchmark != 0.0) is not supported: the canonical DSR benchmarks "
            "against the expected-max Sharpe sr_star. Call deflated_sharpe_ratio directly if you "
            "need a custom benchmark."
        )
    return deflated_sharpe_ratio(returns, n_trials, var_sr=trial_sr_variance)
