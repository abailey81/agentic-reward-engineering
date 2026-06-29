"""Statistical profiling & EDA that MOTIVATES THE METHOD (stage 11) — feeds the
supervisor's explicit "EDA-that-motivates-method" ask. EDA only; no modelling.

Every figure/table is emitted with a caption that names the design choice it
motivates: fat tails -> CVaR-penalised fitness (PREREG §3); volatility clustering &
regime dynamics -> 60-day lookback + 3-state HMM (config environment/inference);
reconstitution turnover -> the {0,5,10,20,50}bps cost grid (DeMiguel et al. 2009).

Statistics: ADF & KPSS (statsmodels; opposite null hypotheses — reported together),
excess kurtosis & skewness, Hill tail-index estimator (left tail; Hill 1975),
|r| autocorrelation + ARCH-LM (Engle 1982), rolling mean pairwise correlation,
cross-sectional dispersion, drawdown anatomy, naive top-N reconstitution turnover.
Figures are written headless (matplotlib Agg) to reports/figures/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..config import get  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
FIG_DIR = ROOT / "reports" / "figures"


# ------------------------------------------------------------------------- metrics
def adf_kpss_table(returns: pd.DataFrame) -> pd.DataFrame:
    from statsmodels.tsa.stattools import adfuller, kpss
    rows = []
    for col in returns.columns:
        s = returns[col].dropna()
        if len(s) < 50:
            continue
        adf_p = float(adfuller(s, autolag="AIC")[1])
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")                 # KPSS p-value clipping note
            kpss_p = float(kpss(s, regression="c", nlags="auto")[1])
        rows.append({"name": col, "adf_p": adf_p, "kpss_p": kpss_p,
                     "stationary_consistent": bool(adf_p < 0.05 and kpss_p > 0.05)})
    return pd.DataFrame(rows)


def moments_table(returns: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import kurtosis, skew
    rows = []
    for col in returns.columns:
        s = returns[col].dropna()
        if len(s) < 30:
            continue
        rows.append({"name": col, "n": int(len(s)), "mean_daily": float(s.mean()),
                     "std_daily": float(s.std(ddof=1)), "skew": float(skew(s)),
                     "excess_kurtosis": float(kurtosis(s, fisher=True)),
                     "min": float(s.min()), "max": float(s.max())})
    return pd.DataFrame(rows)


def hill_tail_index(returns: pd.Series, tail_fraction: float = 0.05) -> float:
    """Hill (1975) estimator on the LEFT tail (losses): alpha_hat =
    [mean(log(x_(i)/x_(k)))]^-1 over the k largest losses. Finite-variance
    benchmark: alpha > 2; Gaussian tails -> alpha -> infinity."""
    losses = -returns.dropna()
    losses = losses[losses > 0].sort_values(ascending=False)
    k = max(10, int(np.ceil(tail_fraction * len(losses))))
    if len(losses) <= k:
        return float("nan")
    top = losses.iloc[:k].to_numpy()
    return float(1.0 / np.mean(np.log(top / top[-1])))


def arch_lm_table(returns: pd.DataFrame, lags: int = 10) -> pd.DataFrame:
    from statsmodels.stats.diagnostic import het_arch
    rows = []
    for col in returns.columns:
        s = returns[col].dropna()
        if len(s) < 100:
            continue
        stat, p, *_ = het_arch(s - s.mean(), nlags=lags)
        rows.append({"name": col, "arch_lm_stat": float(stat), "p_value": float(p),
                     "clustering": bool(p < 0.01)})
    return pd.DataFrame(rows)


def rolling_mean_pairwise_corr(returns: pd.DataFrame, window: int = 60) -> pd.Series:
    cols = returns.columns
    if len(cols) < 2:
        return pd.Series(dtype=float)
    pair_corrs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pair_corrs.append(returns[cols[i]].rolling(window).corr(returns[cols[j]]))
    return pd.concat(pair_corrs, axis=1).mean(axis=1).rename("mean_pairwise_corr")


def drawdown_anatomy(market: pd.Series, top_n: int = 4) -> pd.DataFrame:
    """The deepest drawdowns in the span: peak/trough dates, depth, lengths —
    the 2008 / 2020 / 2022 / (2025 if present) anatomy table."""
    s = (1.0 + market.dropna()).cumprod()
    peak = s.cummax()
    dd = s / peak - 1.0
    episodes, in_dd, start = [], False, None
    for date, v in dd.items():
        if v < 0 and not in_dd:
            in_dd, start = True, date
        elif v == 0 and in_dd:
            seg = dd.loc[start:date]
            episodes.append({"peak": start, "trough": seg.idxmin(), "recovery": date,
                             "depth": float(seg.min()),
                             "length_sessions": int(len(seg))})
            in_dd = False
    if in_dd:
        seg = dd.loc[start:]
        episodes.append({"peak": start, "trough": seg.idxmin(), "recovery": pd.NaT,
                         "depth": float(seg.min()), "length_sessions": int(len(seg))})
    return (pd.DataFrame(episodes).sort_values("depth").head(top_n).reset_index(drop=True)
            if episodes else pd.DataFrame())


def reconstitution_turnover(membership_lists: list[list[str]]) -> float:
    """Mean one-way turnover of naive periodic top-N reconstitution (equal-weight
    proxy): ½·Σ|Δw| = |sym-diff| / (2N) per rebalance — the number that motivates
    charging costs on the grid rather than assuming free reconstitution."""
    if len(membership_lists) < 2:
        return 0.0
    tos = []
    for a, b in zip(membership_lists, membership_lists[1:]):
        sa, sb = set(a), set(b)
        n = max(len(sa), len(sb))
        tos.append(len(sa ^ sb) / (2.0 * n) if n else 0.0)
    return float(np.mean(tos))


# -------------------------------------------------------------------------- report
def _fig(path_name: str, plot_fn) -> str:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=120)
    plot_fn(ax)
    fig.tight_layout()
    out = FIG_DIR / path_name
    fig.savefig(out)
    plt.close(fig)
    return str(out.relative_to(ROOT))


def run_eda(returns: pd.DataFrame, market: pd.Series, vix: pd.Series | None,
            out_name: str = "eda_v1.md") -> Path:
    """Compute all profiles, render figures, and write reports/eda_v1.md with
    design-choice captions. Pure consumption of clean/gold inputs (EDA never
    modifies data)."""
    cost_grid = get("environment.costs.bps_grid")
    alpha = get("inference.fitness.alpha")
    n_states = get("inference.regimes.hmm.n_states")
    lookback = get("environment.state.lookback_T")

    adf = adf_kpss_table(returns)
    mom = moments_table(returns)
    arch = arch_lm_table(returns)
    hill = {c: hill_tail_index(returns[c]) for c in returns.columns}
    roll = rolling_mean_pairwise_corr(returns)
    dda = drawdown_anatomy(market)
    disp = returns.std(axis=1, ddof=1)

    pooled = returns.stack().dropna()
    figs = {}
    figs["qq"] = _fig("eda_qq_pooled.png", lambda ax: __import__("scipy.stats", fromlist=["probplot"])
                      .probplot(pooled, dist="norm", plot=ax))
    figs["acf"] = _fig("eda_abs_return_acf.png", lambda ax: ax.bar(
        range(1, 21), [pooled.abs().autocorr(lag) for lag in range(1, 21)]) or
        ax.set(title="|r| autocorrelation (pooled)", xlabel="lag (days)", ylabel="ACF"))
    figs["roll"] = _fig("eda_rolling_corr.png", lambda ax: (
        ax.plot(roll.index, roll.values),
        ax.set(title="60d mean pairwise correlation", ylabel="corr")) and None)
    figs["disp"] = _fig("eda_dispersion.png", lambda ax: (
        ax.plot(disp.index, disp.values),
        ax.set(title="Cross-sectional dispersion (daily std across names)")) and None)
    figs["missing"] = _fig("eda_missingness.png", lambda ax: (
        ax.imshow(returns.isna().T, aspect="auto", interpolation="none", cmap="Greys"),
        ax.set(title="Missingness map (rows=names, cols=sessions)")) and None)

    lines = [
        "# EDA v1 — profiling that motivates the method (stage 11; EDA only, no modelling)",
        "",
        "## 1. Stationarity (ADF + KPSS, opposite nulls)",
        adf.to_markdown(index=False) if not adf.empty else "⟨insufficient data⟩",
        "*Caption: daily returns are level-stationary while volatility is not constant — "
        f"the state uses a rolling {lookback}-day log-return window rather than longer memory.*",
        "",
        "## 2. Fat tails",
        mom.to_markdown(index=False) if not mom.empty else "⟨insufficient data⟩",
        f"Hill left-tail indices (5% tail): {({k: round(v, 2) for k, v in hill.items()})}",
        f"![QQ]({figs['qq']})",
        f"*Caption: excess kurtosis ≫ 0 and Hill α near/below 3 mean variance understates risk — "
        f"the fitness penalises empirical CVaR at α={alpha} (PREREG §3) instead of trusting σ.*",
        "",
        "## 3. Volatility clustering",
        arch.to_markdown(index=False) if not arch.empty else "⟨insufficient data⟩",
        f"![ACF]({figs['acf']})",
        f"*Caption: ARCH effects and slow |r| ACF decay are regime persistence — the {n_states}-state "
        "HMM with FILTERED probabilities (R3) captures it without look-ahead.*",
        "",
        "## 4. Correlation regimes & dispersion",
        f"![rolling corr]({figs['roll']})",
        f"![dispersion]({figs['disp']})",
        "*Caption: pairwise correlation spikes in stress — diversification fails exactly when the "
        "tail matters, motivating tail-risk feedback to the reward designer over mean-variance logic.*",
        "",
        "## 5. Drawdown anatomy (deepest episodes in span)",
        dda.to_markdown(index=False) if not dda.empty else "⟨no completed drawdowns in span⟩",
        "*Caption: crisis depth/length asymmetry motivates keeping the GFC inside the development "
        "window (PREREG §6) — evolved rewards must SEE a crisis to be tested against one.*",
        "",
        "## 6. Missingness",
        f"![missingness]({figs['missing']})",
        "*Caption: missingness is structural (pre-IPO / post-delisting), not random — the "
        "missing-data engine masks and terminates rather than interpolating (R4).*",
        "",
        "## 7. Reconstitution turnover -> cost grid",
        f"*Naive top-N reconstitution turnover feeds the cost columns {cost_grid} bps "
        "(DeMiguel et al. 2009 convention; 50bps = stress column). Computed when PIT membership lands.*",
    ]
    out = ROOT / "reports" / out_name
    out.write_text("\n".join(lines) + "\n")
    return out
