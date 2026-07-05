"""Regime-stratified tail/return metrics — report-only (T3'; Okhrati "motivate with the data").

The headline equivalence is an average over the whole sealed leg; a marker who works on coherent risk will
immediately ask *where* it holds. This module conditions the per-arm metrics on market regime (the
VIX-threshold calm / normal / stress labels from :mod:`src.regimes.definition`) so the write-up can say
whether the channel is silent everywhere or only on average — and, crucially, it reports the **independent
episode count** per regime, because that (not the per-date count) is what bounds regime-conditional power: a
20-year sample holds only single-digit independent stress episodes, so a regime-stratified contrast is
descriptive, never a powered test.

Report-only and DISJOINT from the frozen ``m=6`` family. Deterministic, numpy-only. The metrics are simple
**per-period** statistics (not annualised) used for *relative* cross-arm / cross-regime comparison; the
inferential headline uses the full inference stack (`deflated_sharpe`, `bootstrap`, `es_backtest`), not these.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.regimes.definition import independent_regime_count

__all__ = ["DEFAULT_METRICS", "cvar", "regime_stratified_metrics", "render_regime_table"]

_REGIME_NAMES = {0: "calm", 1: "normal", 2: "stress"}


def cvar(returns: np.ndarray, q: float = 0.05) -> float:
    """Empirical CVaR at level ``q`` (mean of the worst ``q`` fraction of returns); NaN for an empty input."""
    r = np.asarray(returns, dtype=float).ravel()
    r = np.sort(r[np.isfinite(r)])  # row 30c: strip non-finite (mirrors bootstrap.cvar)
    if r.size == 0:
        return float("nan")
    k = max(1, int(np.ceil(q * r.size)))
    return float(r[:k].mean())


def _sharpe(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float).ravel()
    sd = r.std(ddof=1) if r.size > 1 else 0.0
    return float(r.mean() / sd) if sd > 0 else float("nan")


#: Default per-period descriptive metrics (relative comparison only; not annualised).
DEFAULT_METRICS: dict[str, Callable[[np.ndarray], float]] = {
    "mean": lambda r: float(np.mean(r)) if len(r) else float("nan"),
    "vol": lambda r: float(np.std(r, ddof=1)) if len(r) > 1 else float("nan"),
    "sharpe": _sharpe,
    "cvar_05": lambda r: cvar(r, 0.05),
}


def regime_stratified_metrics(
    returns_by_arm: Mapping[str, np.ndarray],
    labels: np.ndarray,
    *,
    regime_names: Mapping[int, str] | None = None,
    metrics: Mapping[str, Callable[[np.ndarray], float]] | None = None,
    min_obs: int = 10,
) -> dict[str, Any]:
    """Per-arm metrics stratified by regime, with each regime's independent-episode count (the power bound).

    Parameters
    ----------
    returns_by_arm : Mapping[str, np.ndarray]
        ``{arm: per-date realised return array}`` over the sealed test leg; every array length ``T``.
    labels : np.ndarray, shape (T,)
        Per-date integer regime labels (from :func:`src.regimes.definition.label_regimes`).
    regime_names : Mapping[int, str] | None
        Optional label→name map (defaults to calm/normal/stress for 0/1/2; unknown labels use ``regime_<k>``).
    metrics : Mapping[str, callable] | None
        Metric name → function over a returns slice (defaults to :data:`DEFAULT_METRICS`).
    min_obs : int
        A regime slice with fewer than ``min_obs`` dates has its metrics returned as NaN and ``underpowered``
        set True (reported, never silently dropped).

    Returns
    -------
    dict
        ``{"status": "ok", "n_dates", "regimes": [ {label, name, n_dates, n_episodes, underpowered,
        per_arm: {arm: {metric: value}}}, ... ], "power_note"}`` or ``{"status": "no_data", ...}``.
    """
    lab = np.asarray(labels).ravel()
    arms = list(returns_by_arm)
    if not arms or lab.size == 0:
        return {"status": "no_data", "reason": "no arms or empty labels"}
    T = lab.size
    for a in arms:
        if np.asarray(returns_by_arm[a]).ravel().size != T:
            return {"status": "no_data", "reason": f"arm {a!r} returns length != labels length {T}"}

    names = dict(_REGIME_NAMES)
    if regime_names:
        names.update(regime_names)
    mfns = dict(metrics) if metrics is not None else DEFAULT_METRICS

    regimes_out: list[dict[str, Any]] = []
    for k in sorted(set(int(v) for v in lab)):
        mask = lab == k
        n_dates = int(mask.sum())
        # independent episodes = contiguous blocks of THIS label across the FULL series (the power bound).
        n_eps = _episodes_of_label(lab, k)
        underpowered = n_dates < min_obs
        per_arm: dict[str, dict[str, float]] = {}
        for a in arms:
            r = np.asarray(returns_by_arm[a], dtype=float).ravel()[mask]
            per_arm[a] = {
                name: (float("nan") if underpowered else float(fn(r))) for name, fn in mfns.items()
            }
        regimes_out.append({
            "label": int(k),
            "name": names.get(int(k), f"regime_{k}"),
            "n_dates": n_dates,
            "n_episodes": int(n_eps),
            "underpowered": bool(underpowered),
            "per_arm": per_arm,
        })

    return {
        "status": "ok",
        "n_dates": int(T),
        "total_episodes": int(independent_regime_count(lab)),
        "regimes": regimes_out,
        "power_note": (
            "n_episodes (contiguous regime blocks), not n_dates, bounds regime-conditional power; "
            "single-digit independent stress episodes => regime contrasts are DESCRIPTIVE, not powered tests."
        ),
    }


def _episodes_of_label(labels: np.ndarray, k: int) -> int:
    """Number of contiguous runs equal to label ``k`` (the independent-episode count for that regime)."""
    is_k = (np.asarray(labels).ravel() == k).astype(int)
    if is_k.sum() == 0:
        return 0
    starts = int(is_k[0]) + int(np.count_nonzero((is_k[1:] == 1) & (is_k[:-1] == 0)))
    return starts


def render_regime_table(
    result: Mapping[str, Any],
    *,
    arms: Sequence[str] | None = None,
    metric_order: Sequence[str] = ("cvar_05", "sharpe", "mean", "vol"),
) -> str:
    """Render a regime-stratified result as a tidy long-form Markdown table (T3').

    Columns: regime | n (dates) | episodes | arm | <metrics…>. The episode count sits beside every regime so a
    reader sees the power bound inline. Returns a ``"no data"`` line if the result did not run.
    """
    if result.get("status") != "ok":
        return f"_regime table unavailable: {result.get('reason', 'no data')}_"
    all_arms = list(arms) if arms is not None else list(result["regimes"][0]["per_arm"])
    cols = [m for m in metric_order if any(m in r["per_arm"][a] for r in result["regimes"] for a in all_arms)]
    header = "| regime | n | episodes | arm | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (4 + len(cols))
    lines = [header, sep]
    for r in result["regimes"]:
        tag = r["name"] + (" (!)" if r["underpowered"] else "")
        for i, a in enumerate(all_arms):
            cells = []
            for m in cols:
                v = r["per_arm"][a].get(m, float("nan"))
                cells.append("n/a" if v != v else f"{v:+.4f}")  # v!=v catches NaN
            rk = f"{tag} | {r['n_dates']} | {r['n_episodes']}" if i == 0 else "  |  | "
            lines.append(f"| {rk} | {a} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"_{result['power_note']}_")
    return "\n".join(lines)
