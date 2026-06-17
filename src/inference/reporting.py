"""Robust aggregate reporting statistics (rliable-style) (FINAL_PLAN F.11).

Purpose
-------
Aggregate held-out scores with estimators robust to outlier runs and that
report uncertainty honestly, following the ``rliable`` methodology (Agarwal et
al. 2021).

Estimators
----------
- ``iqm`` -- the interquartile mean: average the middle 50% of scores.
- ``probability_of_improvement`` -- ``P(a > b)`` over all pairs, ties = 0.5.
- ``stratified_bootstrap_ci`` -- a bootstrap confidence interval whose point
  estimate is the IQM, returning ``(point, low, high)``.
- ``performance_profile`` -- the run-score distribution (Agarwal et al. 2021
  §4 "performance profiles"): the fraction of runs scoring above each threshold
  ``tau``, with an optional stratified-bootstrap uncertainty band. Reporting the
  whole score distribution (rather than a single aggregate) exposes the
  run-to-run variability that point estimates hide, and a profile that
  stochastically dominates another is a qualitatively stronger claim than an
  IQM gap.
"""

from __future__ import annotations


import numpy as np

__all__ = [
    "iqm",
    "probability_of_improvement",
    "stratified_bootstrap_ci",
    "performance_profile",
]


def iqm(scores: np.ndarray) -> float:
    """Interquartile mean: mean of the middle 50% of ``scores``.

    Parameters
    ----------
    scores:
        One-dimensional array of scores.

    Returns
    -------
    float
        The mean of the values in the ``[25th, 75th]`` percentile band. Values
        are sorted and the lower/upper 25% are trimmed by count.
    """
    s = np.asarray(scores, dtype=float)
    # Strip non-finite values before sorting so they cannot land in the trimmed
    # band: np.sort pushes NaN/+inf to the end, where they would otherwise be
    # averaged in and silently corrupt the IQM. Mirrors the documented sibling
    # oracle src/inference/bootstrap.iqm (bootstrap.py:93) so the two agree.
    s = s[np.isfinite(s)]
    n = s.size
    if n == 0:
        return float("nan")
    s = np.sort(s)
    lo = int(np.floor(n * 0.25))
    hi = int(np.ceil(n * 0.75))
    # For every n >= 1, floor(n*0.25) < ceil(n*0.75), so s[lo:hi] is never empty
    # (the only empty-band case is n == 0, already returned above).
    middle = s[lo:hi]
    return float(middle.mean())


def probability_of_improvement(a: np.ndarray, b: np.ndarray) -> float:
    """Probability that a draw from ``a`` improves on a draw from ``b``.

    Parameters
    ----------
    a, b:
        Score populations (need not be paired or equal length).

    Returns
    -------
    float
        ``P(a > b)`` in ``[0, 1]`` over all pairs, with ties counted as one half.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    # Pairwise comparison via broadcasting.
    diff = a[:, None] - b[None, :]
    greater = np.count_nonzero(diff > 0)
    ties = np.count_nonzero(diff == 0)
    total = a.size * b.size
    return float((greater + 0.5 * ties) / total)


def stratified_bootstrap_ci(
    scores: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for the IQM of ``scores``.

    Parameters
    ----------
    scores:
        One-dimensional array of run scores.
    n_boot:
        Number of bootstrap replications.
    ci:
        Target two-sided coverage (e.g. ``0.95``).
    rng:
        Optional NumPy generator for reproducibility.

    Returns
    -------
    tuple[float, float, float]
        ``(point, low, high)`` where ``point`` is the IQM of ``scores`` and
        ``low``/``high`` are the percentile bootstrap CI bounds for the IQM.
    """
    s = np.asarray(scores, dtype=float)
    n = s.size
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    if rng is None:
        rng = np.random.default_rng(0)  # P24: deterministic fallback (reported paths pass an explicit rng)

    point = iqm(s)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = s[rng.integers(0, n, size=n)]
        boot[i] = iqm(sample)
    alpha = 1.0 - ci
    low = float(np.quantile(boot, alpha / 2.0))
    high = float(np.quantile(boot, 1.0 - alpha / 2.0))
    return (point, low, high)


def performance_profile(
    scores: np.ndarray,
    taus: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run-score distribution (performance profile) of Agarwal et al. (2021).

    For each threshold ``tau`` returns the fraction of runs whose score exceeds
    ``tau`` -- the empirical survival function ``hat F_X(tau) = mean_i 1{x_i >
    tau}``. Plotted against ``tau`` this is rliable's "performance profile": a
    curve that lies entirely above another indicates (empirical) stochastic
    dominance, a strictly stronger statement than a gap in any single aggregate.

    Parameters
    ----------
    scores:
        One-dimensional array of run scores (non-finite values are dropped).
    taus:
        One-dimensional, increasing array of score thresholds at which to
        evaluate the profile.
    n_boot:
        Bootstrap replications for the pointwise uncertainty band. ``0`` skips
        the bootstrap and returns ``low == high == point``.
    ci:
        Target two-sided pointwise coverage (e.g. ``0.95``).
    rng:
        Optional NumPy generator for reproducibility.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(profile, low, high)`` each shaped like ``taus``. ``profile`` is the
        survival fraction in ``[0, 1]`` and is monotone non-increasing in
        ``tau``; ``low``/``high`` are the pointwise percentile-bootstrap band
        (equal to ``profile`` when ``n_boot == 0``). All-``nan`` if ``scores``
        is empty.
    """
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    t = np.asarray(taus, dtype=float)
    if s.size == 0:
        nan = np.full(t.shape, float("nan"))
        return (nan, nan.copy(), nan.copy())

    # Survival fraction: proportion of runs scoring strictly above each tau.
    profile = (s[:, None] > t[None, :]).mean(axis=0)
    if n_boot <= 0:
        return (profile, profile.copy(), profile.copy())

    if rng is None:
        rng = np.random.default_rng(0)  # P24: deterministic fallback (reported paths pass an explicit rng)
    n = s.size
    boot = np.empty((n_boot, t.size), dtype=float)
    for i in range(n_boot):
        sample = s[rng.integers(0, n, size=n)]
        boot[i] = (sample[:, None] > t[None, :]).mean(axis=0)
    alpha = 1.0 - ci
    low = np.quantile(boot, alpha / 2.0, axis=0)
    high = np.quantile(boot, 1.0 - alpha / 2.0, axis=0)
    return (profile, low, high)
