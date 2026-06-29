"""World-class performance & risk metric suite for the sealed-test backtest (block B11).

Every metric here is a STANDARD, citation-anchored statistic chosen for a *tail-risk* portfolio study;
nothing is included for show. The suite is deliberately coherent — it spans return, risk-adjusted return,
drawdown, tail, distribution shape, trading cost, benchmark-relative, and overfitting-aware families — so the
dissertation can report a publishable performance picture rather than a single Sharpe.

It REUSES the audited inference primitives (``src.inference.bootstrap.{sharpe_ratio,cvar}`` and
``src.inference.deflated_sharpe.{probabilistic_sharpe_ratio,deflated_sharpe_ratio}``) so there is ONE
definition of each shared quantity (DRY, audit C-1). Across-SEED aggregation (IQM + stratified bootstrap)
stays in the inference layer (``src.inference.reporting`` / ``bootstrap.paired_seed_difference_test``); this
module computes PER-SERIES analytics for one realised return path (an arm/benchmark/seed).

References (one primary anchor per metric family)
-------------------------------------------------
- Sharpe ratio — Sharpe (1966, 1994, *J. Portfolio Management*).
- Sortino ratio / downside deviation — Sortino & Price (1994); Sortino & van der Meer (1991).
- Calmar / MAR ratio — Young (1991, *Futures*).
- Omega ratio — Keating & Shadwick (2002, *J. Performance Measurement*).
- Ulcer index / Martin (Ulcer Performance) ratio — Martin & McCann (1989, *The Investor's Guide to Fidelity Funds*).
- CVaR / expected shortfall — Rockafellar & Uryasev (2000, *J. Risk*); Acerbi & Tasche (2002).
- STARR ratio / tail-adjusted Sharpe (excess return per unit CVaR) — Martin, Rachev & Siboulet (2003,
  *Wilmott*). Report-only; the tail-aware analogue of Sharpe (operationalises "Sharpe is tail-blind").
- Historical VaR (empirical quantile). NB Cornish-Fisher modified VaR was REMOVED (2026-06-20 metrics
  research: non-monotonic / unreliable for the fat-tailed loss distribution; coherent CVaR/ES dominates).
- Deflated Sharpe ratio / PSR — Bailey & López de Prado (2014, *J. Portfolio Management*).
- IQM + stratified bootstrap CI — Agarwal et al. (2021, *NeurIPS*, "rliable").
- 1/N benchmark hardness — DeMiguel, Garlappi & Uppal (2009, *Review of Financial Studies*).

All metrics are computed on the NET (post-cost) per-period return series unless noted. Annualisation uses
``periods_per_year`` (252 trading days by default). Functions are numpy-only + numerically guarded (a constant
or empty series degrades to a finite sentinel, never a crash).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as _sps

from src.inference.bootstrap import cvar as _cvar
from src.inference.bootstrap import sharpe_ratio as _sharpe

#: Trading days per year — the canonical annualisation factor for daily P&L.
PERIODS_PER_YEAR: int = 252

_EPS = 1e-12


def _clean(returns: Any) -> np.ndarray:
    """Coerce to a 1-D finite float array (drop non-finite, as a pathological reward could emit them)."""
    r = np.asarray(returns, dtype=float).ravel()
    return r[np.isfinite(r)]


# --------------------------------------------------------------------------- #
# Drawdown analytics (shared by several ratios)                                #
# --------------------------------------------------------------------------- #
def drawdown_series(returns: Any) -> dict[str, Any]:
    """Wealth curve + drawdown series and their summaries.

    Returns ``{"wealth", "drawdown", "max_drawdown", "avg_drawdown",
    "max_drawdown_duration", "time_under_water"}``. ``drawdown`` is the fractional shortfall from the running
    peak (≤ 0); ``max_drawdown_duration`` is the longest run (in periods) below a peak; ``time_under_water`` is
    the fraction of periods spent below a peak.
    """
    r = _clean(returns)
    if r.size == 0:
        return {
            "wealth": np.ones(0),
            "drawdown": np.zeros(0),
            "max_drawdown": 0.0,
            "avg_drawdown": 0.0,
            "max_drawdown_duration": 0,
            "time_under_water": 0.0,
        }
    wealth = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(wealth)
    # ≤ 0; guard total ruin (peak -> 0 after a -100% period) -> a -100% drawdown, not 0/0 = NaN. Divide by
    # a SAFE peak so the masked-off ruin branch does not emit a RuntimeWarning (would trip warnings-as-errors).
    safe_peak = np.where(peak > _EPS, peak, 1.0)
    dd = np.where(peak > _EPS, wealth / safe_peak - 1.0, -1.0)
    underwater = dd < -_EPS
    # Longest consecutive underwater run.
    longest = cur = 0
    for u in underwater:
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    neg = dd[underwater]
    return {
        "wealth": wealth,
        "drawdown": dd,
        "max_drawdown": max(float(-dd.min()), 0.0) if dd.size else 0.0,  # POSITIVE magnitude (no -0.0)
        "avg_drawdown": float(-neg.mean()) if neg.size else 0.0,
        "max_drawdown_duration": int(longest),
        "time_under_water": float(underwater.mean()),
    }


def _ulcer_index(dd: np.ndarray) -> float:
    """Ulcer index = RMS of the (fractional) drawdown path (Martin & McCann 1989)."""
    return float(np.sqrt(np.mean(dd**2))) if dd.size else 0.0


def _annualised_return(r: np.ndarray, ppy: int) -> float:
    """Geometric (CAGR) annualised return from a per-period simple-return series."""
    if r.size == 0:
        return 0.0
    growth = float(np.prod(1.0 + r))
    if growth <= 0.0:  # a -100% wipeout: report -100% CAGR rather than a complex root
        return -1.0
    return growth ** (ppy / r.size) - 1.0


def _downside_deviation(r: np.ndarray, target: float, ppy: int) -> float:
    """Annualised downside deviation: RMS of shortfalls below ``target`` over ALL periods (Sortino 1994)."""
    if r.size == 0:
        return 0.0
    short = np.minimum(r - target, 0.0)
    return float(np.sqrt(np.mean(short**2)) * np.sqrt(ppy))


# NB: Cornish-Fisher modified VaR was intentionally REMOVED (2026-06-20 metrics research). For a
# fat-tailed loss distribution the CF expansion is non-monotonic / unreliable beyond moderate
# skew-kurtosis; historical VaR + coherent CVaR/ES dominate it here. Keeping only highly-relevant tail
# metrics (mandate: advanced but ONLY extremely relevant) — historical VaR and CVaR are retained.


# --------------------------------------------------------------------------- #
# The full metric suite                                                        #
# --------------------------------------------------------------------------- #
def compute_metrics(
    returns: Any,
    *,
    periods_per_year: int = PERIODS_PER_YEAR,
    risk_free: float | np.ndarray = 0.0,
    benchmark: Any | None = None,
    turnover: Any | None = None,
    cost_bps: float = 0.0,
    var_levels: tuple[float, ...] = (0.01, 0.05, 0.10),
    n_trials: int | None = None,
    var_sr: float | None = None,
) -> dict[str, float]:
    """Compute the full performance/risk suite for one per-period NET return series.

    Parameters
    ----------
    returns : array
        1-D per-period (e.g. daily) simple returns of the strategy/arm/benchmark.
    periods_per_year : int
        Annualisation factor (252 for daily).
    risk_free : float or array
        Per-period risk-free rate — a SCALAR or a per-period series (e.g. the daily FRED DGS3MO T-bill,
        via ``src.data.market_reference.load_risk_free_daily``). A series is reduced to its mean (exact
        for the Sharpe/Sortino numerator; the daily T-bill's variance is negligible in the denominator).
    benchmark : array, optional
        Aligned per-period benchmark returns -> enables information ratio, alpha, beta, tracking error.
    turnover : array, optional
        Aligned per-period one-way turnover (``0.5*L1(w - w_prev_drifted)``) -> annualised turnover + cost drag.
    cost_bps : float
        Per-unit-turnover cost in bps (for the cost-drag estimate; the realised series is already net).
    var_levels : tuple of float
        Lower-tail levels for VaR/CVaR (default 1%, 5%, 10%).
    n_trials, var_sr : optional
        If both given, the **deflated Sharpe** is reported (multiple-testing-aware; Bailey-López de Prado).

    Returns
    -------
    dict[str, float]
        ~30 named metrics across return, risk-adjusted, drawdown, tail, distribution, trading-cost,
        benchmark-relative, and overfitting-aware families. A degenerate input yields finite sentinels.
    """
    r = _clean(returns)
    ppy = int(periods_per_year)
    out: dict[str, float] = {"n_periods": float(r.size)}
    if r.size == 0:
        return out

    # Accept a SCALAR risk-free OR a per-period rf SERIES (e.g. the daily FRED DGS3MO). Reduce a series to
    # its mean: the Sharpe/Sortino numerator is mean(r) - mean(rf) (exact under the mean), and a daily
    # T-bill's own variance is negligible vs portfolio vol so the volatility denominator is unchanged to
    # ~4 dp — and this sidesteps re-aligning an rf series to `r` after the non-finite stripping above.
    _rf = np.asarray(risk_free, dtype=float)
    rf_scalar = float(np.nanmean(_rf)) if _rf.size else 0.0
    mu, sd = float(r.mean()), float(r.std(ddof=1)) if r.size > 1 else 0.0
    excess = r - rf_scalar

    # --- return ---
    out["total_return"] = float(np.prod(1.0 + r) - 1.0)
    out["cagr"] = _annualised_return(r, ppy)
    out["ann_return_arith"] = float(mu * ppy)
    out["ann_volatility"] = float(sd * np.sqrt(ppy))

    # --- risk-adjusted return ---
    out["sharpe"] = _sharpe(excess, periods_per_year=ppy)  # reuse the audited Sharpe (guarded)
    dd_dev = _downside_deviation(r, rf_scalar, ppy)
    out["sortino"] = float(excess.mean() * ppy / dd_dev) if dd_dev > _EPS else 0.0
    dd = drawdown_series(r)
    out.update(
        max_drawdown=dd["max_drawdown"],
        avg_drawdown=dd["avg_drawdown"],
        max_drawdown_duration=float(dd["max_drawdown_duration"]),
        time_under_water=dd["time_under_water"],
    )
    out["calmar"] = float(out["cagr"] / dd["max_drawdown"]) if dd["max_drawdown"] > _EPS else 0.0
    ulcer = _ulcer_index(dd["drawdown"])
    out["ulcer_index"] = ulcer
    out["martin_ratio"] = float((out["cagr"] - rf_scalar * ppy) / ulcer) if ulcer > _EPS else 0.0
    out["pain_index"] = float(np.mean(np.abs(dd["drawdown"]))) if dd["drawdown"].size else 0.0
    out["pain_ratio"] = (
        float((out["cagr"] - rf_scalar * ppy) / out["pain_index"]) if out["pain_index"] > _EPS else 0.0
    )
    # Omega about a FIXED 0 MAR (capital-preservation threshold), NOT the risk-free rate. Omega is a
    # distribution-SHAPE ratio of gains-above / losses-below a minimum acceptable return (Keating-Shadwick
    # 2002); the standard/reported convention is tau=0 (metrics research 2026-06-20), which keeps it
    # rf-INVARIANT — the excess-return metrics (Sharpe/Sortino/Martin) use rf, Omega does NOT, so passing a
    # per-period rf must not silently shift Omega's MAR. With NO downside (degenerate all-gains series)
    # Omega is +inf; report a large FINITE sentinel so it stays sortable and never breaks a table.
    _mar = 0.0
    gains = float(np.maximum(r - _mar, 0.0).sum())
    losses = float(np.maximum(_mar - r, 0.0).sum())
    out["omega"] = float(gains / losses) if losses > _EPS else (1e9 if gains > _EPS else 1.0)

    # --- tail / distribution ---
    for a in var_levels:
        tag = f"{int(round(a * 100)):02d}"
        cv = float(_cvar(r, float(a)))
        out[f"cvar_{tag}"] = cv  # ES (mean of worst a-fraction); negative for losses
        out[f"var_hist_{tag}"] = float(-np.quantile(r, float(a)))  # historical VaR, positive loss magnitude
        # STARR ratio / tail-adjusted Sharpe (Martin, Rachev & Siboulet 2003): per-period excess return per
        # unit of CVaR tail loss -- a tail-AWARE analogue of Sharpe that swaps volatility for the coherent
        # tail measure, quantifying the dissertation's "Sharpe is tail-blind" point. Report-only / descriptive;
        # NOT a confirmatory IUT. Mirrors the module's convention (excess-return numerator, raw-return CVaR).
        tail_loss = abs(cv)
        out[f"starr_{tag}"] = float(excess.mean() / tail_loss) if tail_loss > _EPS else 0.0
    q95, q05 = float(np.quantile(r, 0.95)), float(np.quantile(r, 0.05))
    out["tail_ratio"] = float(abs(q95) / abs(q05)) if abs(q05) > _EPS else 0.0
    out["downside_deviation_ann"] = dd_dev
    # Zero-variance series -> scipy skew/kurtosis are 0/0 = NaN; report 0 (no shape) explicitly.
    out["skew"] = float(_sps.skew(r, bias=False)) if (r.size > 2 and sd > _EPS) else 0.0
    out["excess_kurtosis"] = (
        float(_sps.kurtosis(r, fisher=True, bias=False)) if (r.size > 3 and sd > _EPS) else 0.0
    )
    out["best_period"] = float(r.max())
    out["worst_period"] = float(r.min())
    out["pct_positive"] = float(np.mean(r > 0.0))

    # --- trading / cost ---
    out["hit_rate"] = float(np.mean(r > 0.0))
    wins, loss = r[r > 0.0], r[r < 0.0]
    # No-downside (all-gains) series: use the SAME large finite sentinel as omega (line above) for the
    # gain/loss and profit ratios, so a no-loss series is ranked consistently 'best' across all three —
    # not omega=1e9 (best) alongside profit_factor=0.0 (the same value a genuinely bad series takes)
    # (metrics audit 2026-06-20). 0.0 is reserved for the no-wins case.
    # Guard the divisor by MAGNITUDE (> _EPS), not just presence / != 0, to match the module-wide _EPS
    # convention (sharpe/sortino/calmar/omega/...). A denormal-magnitude loss (|loss| ~ 1e-310) slips past a
    # bare `loss.size` / `!= 0` check and overflows the ratio to +inf, violating the "finite sentinels, never
    # crash" contract above; with the _EPS guard a negligible-loss series correctly takes the 1e9 wins
    # sentinel. The leading `loss.size` short-circuit preserves the no-warning empty-array path.
    out["gain_loss_ratio"] = (
        float(wins.mean() / abs(loss.mean()))
        if (loss.size and wins.size and abs(loss.mean()) > _EPS)
        else (1e9 if wins.size else 0.0)
    )
    out["profit_factor"] = (
        float(wins.sum() / abs(loss.sum()))
        if (loss.size and abs(loss.sum()) > _EPS)
        else (1e9 if wins.size else 0.0)
    )
    if turnover is not None:
        tov = _clean(turnover)
        if tov.size:
            out["turnover_ann"] = float(tov.mean() * ppy)
            out["cost_drag_ann"] = float(tov.mean() * float(cost_bps) * 1e-4 * ppy)

    # --- benchmark-relative ---
    if benchmark is not None:
        # Use ONE shared finite mask over the ALIGNED (returns, benchmark) pair, BEFORE truncation —
        # cleaning each by position independently would desync the pairing (an interior NaN in either
        # shifts the other), corrupting beta/alpha/IR (critical-review 2026-06-20). The strategy returns
        # passed here (env net returns / cleaned arm series) are finite, so this is defensive correctness.
        r_raw = np.asarray(returns, dtype=float).ravel()
        b_raw = np.asarray(benchmark, dtype=float).ravel()
        m = min(r_raw.size, b_raw.size)
        if m > 1:
            pair_ok = np.isfinite(r_raw[:m]) & np.isfinite(b_raw[:m])
            rr, bb = r_raw[:m][pair_ok], b_raw[:m][pair_ok]
            if rr.size > 1:
                active = rr - bb
                te = float(active.std(ddof=1))
                out["information_ratio"] = (
                    float(active.mean() * ppy / (te * np.sqrt(ppy))) if te > _EPS else 0.0
                )
                out["tracking_error_ann"] = float(te * np.sqrt(ppy))
                var_b = float(bb.var(ddof=1))
                beta = float(np.cov(rr, bb, ddof=1)[0, 1] / var_b) if var_b > _EPS else 0.0
                out["beta"] = beta
                out["alpha_ann"] = float((rr.mean() - beta * bb.mean()) * ppy)

    # --- overfitting-aware (Bailey & López de Prado 2014) ---
    # PSR = P(true Sharpe > 0) given the realised skew/kurtosis; uses the PER-PERIOD Sharpe + n (NOT
    # annualised). Computed explicitly (no broad exception swallow — a wrong call must surface, not hide).
    # NOTE (rf convention, P7 2026-07-04): PSR/DSR are deliberately on the RAW (total-return, rf=0) Sharpe
    # even when a ``risk_free`` is passed — the Deflated-Sharpe overfitting statistic is the frozen rf=0
    # selection numeraire (PREREGISTRATION.md §numeraire). Only Sharpe/Sortino/Martin above use excess (r−rf).
    if r.size > 3 and sd > _EPS:
        from src.inference.deflated_sharpe import probabilistic_sharpe_ratio

        sr_per = mu / sd
        skew_r = float(_sps.skew(r, bias=False))
        kurt_raw = float(_sps.kurtosis(r, fisher=False, bias=False))  # RAW kurtosis (normal = 3)
        out["psr"] = float(probabilistic_sharpe_ratio(sr_per, 0.0, int(r.size), skew_r, kurt_raw))
    # Deflated Sharpe needs the multiple-testing context (trial count + cross-trial Sharpe variance).
    if n_trials is not None and var_sr is not None:
        from src.inference.deflated_sharpe import deflated_sharpe_ratio

        out["deflated_sharpe"] = float(
            deflated_sharpe_ratio(r, int(n_trials), var_sr=float(var_sr), periods_per_year=ppy)
        )
    return out


# --------------------------------------------------------------------------- #
# Regime-conditional breakdown                                                 #
# --------------------------------------------------------------------------- #
def regime_conditional_metrics(
    returns: Any,
    regimes: Any,
    *,
    periods_per_year: int = PERIODS_PER_YEAR,
    metrics: tuple[str, ...] = ("cagr", "sharpe", "sortino", "max_drawdown", "cvar_05", "calmar", "hit_rate"),
    **kwargs: Any,
) -> dict[str, dict[str, float]]:
    """Per-regime metric breakdown — does the edge concentrate in calm vs stress periods?

    ``regimes`` is an aligned per-period label array (e.g. {"calm","normal","stress"}). Returns
    ``{regime_label: {metric: value}}`` (the chosen ``metrics`` subset) plus an ``"all"`` group. A regime with
    too few periods for a stable estimate is still reported (caveat emptor) but never crashes.
    """
    r0 = np.asarray(returns, dtype=float).ravel()
    lab = np.asarray(regimes).ravel()
    if r0.size != lab.size:
        raise ValueError(
            f"returns ({r0.size}) and regimes ({lab.size}) must be the same length "
            "(a misalignment must FAIL LOUD, not be silently truncated)."
        )
    # Pop the per-period benchmark/turnover series out of kwargs: they are ALIGNED to `returns` and so
    # MUST be subset by the same finite + regime masks below, never forwarded full-length. Otherwise the
    # benchmark branch of compute_metrics would pair a length-k regime slice positionally against the FIRST
    # k rows of the full benchmark (a date misalignment) and silently corrupt per-regime beta/alpha/IR
    # (metrics audit 2026-06-20). risk_free is left in kwargs — it is reduced to a scalar mean, so it needs
    # no per-regime subsetting.
    bench_full = kwargs.pop("benchmark", None)
    tov_full = kwargs.pop("turnover", None)
    if bench_full is not None:
        bench_full = np.asarray(bench_full, dtype=float).ravel()
        if bench_full.size != r0.size:
            raise ValueError(
                f"benchmark ({bench_full.size}) and returns ({r0.size}) must be the same length for a "
                "per-regime breakdown (the benchmark must be aligned so each regime slice keeps its own "
                "calendar rows; a misalignment must FAIL LOUD, not be silently truncated)."
            )
    if tov_full is not None:
        tov_full = np.asarray(tov_full, dtype=float).ravel()
        if tov_full.size != r0.size:
            raise ValueError(
                f"turnover ({tov_full.size}) and returns ({r0.size}) must be the same length for a "
                "per-regime breakdown (a misalignment must FAIL LOUD, not be silently truncated)."
            )
    finite = np.isfinite(r0)
    r0, lab = r0[finite], lab[finite]  # mask BOTH together so the regime alignment is preserved
    if bench_full is not None:
        bench_full = bench_full[finite]  # keep benchmark row-aligned to the masked returns
    if tov_full is not None:
        tov_full = tov_full[finite]  # keep turnover row-aligned to the masked returns
    out: dict[str, dict[str, float]] = {}
    groups = ["all", *[str(x) for x in sorted(set(lab.tolist()))]] if lab.size else ["all"]
    for g in groups:
        mask = np.ones(r0.shape, dtype=bool) if g == "all" else (lab == g)
        sub = r0[mask]
        kw = dict(kwargs)
        if bench_full is not None:
            kw["benchmark"] = bench_full[mask]  # ALIGNED regime slice, not the first-N rows
        if tov_full is not None:
            kw["turnover"] = tov_full[mask]
        full = compute_metrics(sub, periods_per_year=periods_per_year, **kw)
        out[g] = {"n_periods": full.get("n_periods", 0.0), **{k: full.get(k, float("nan")) for k in metrics}}
    return out


# --------------------------------------------------------------------------- #
# Markdown tearsheet                                                           #
# --------------------------------------------------------------------------- #
_TEARSHEET_ROWS: tuple[tuple[str, str, str], ...] = (
    ("CAGR", "cagr", "pct"),
    ("Ann. volatility", "ann_volatility", "pct"),
    ("Sharpe", "sharpe", "num"),
    ("Sortino", "sortino", "num"),
    ("Calmar", "calmar", "num"),
    ("Omega", "omega", "num"),
    ("Martin (UPI)", "martin_ratio", "num"),
    ("Max drawdown", "max_drawdown", "pct"),
    ("Max DD duration (periods)", "max_drawdown_duration", "int"),
    ("Ulcer index", "ulcer_index", "num"),
    ("CVaR 5%", "cvar_05", "pct"),
    ("CVaR 1%", "cvar_01", "pct"),
    ("Tail ratio", "tail_ratio", "num"),
    ("Skew", "skew", "num"),
    ("Excess kurtosis", "excess_kurtosis", "num"),
    ("Hit rate", "hit_rate", "pct"),
    ("Gain/loss ratio", "gain_loss_ratio", "num"),
    ("Profit factor", "profit_factor", "num"),
    ("Turnover (ann.)", "turnover_ann", "num"),
    ("Information ratio", "information_ratio", "num"),
    ("Alpha (ann.)", "alpha_ann", "pct"),
    ("Beta", "beta", "num"),
    ("PSR", "psr", "num"),
    ("Deflated Sharpe", "deflated_sharpe", "num"),
)


def _fmt(value: float | None, kind: str) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if kind == "pct":
        return f"{value * 100:,.2f}%"
    if kind == "int":
        return f"{int(value)}"
    return f"{value:,.3f}"


def tearsheet_markdown(metrics_by_name: dict[str, dict[str, float]], *, title: str = "Performance tearsheet") -> str:
    """Render one or more named metric dicts (e.g. ``{arm: metrics}``) as a comparison tearsheet table."""
    names = list(metrics_by_name.keys())
    head = "| Metric | " + " | ".join(names) + " |"
    sep = "|" + "---|" * (len(names) + 1)
    lines = [f"## {title}", "", head, sep]
    for label, key, kind in _TEARSHEET_ROWS:
        # Skip a row only if NO column reports it (keeps the table compact).
        if not any(key in metrics_by_name[n] for n in names):
            continue
        cells = [_fmt(metrics_by_name[n].get(key), kind) for n in names]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
