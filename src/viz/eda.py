"""The F3 stylised-facts EDA figure (report-only). Returns a ``matplotlib.figure.Figure``; deterministic.

The Data-chapter figure that MOTIVATES the feedback channel from the data (Cont 2001; the examiner's
"motivate the method with the data" lever): one 2×2 panel built from the REAL gold panel's TRAIN window
only — the development top-30, 2005–2016 (Split C train) — so the sealed validation/test years are never read
(snoop-clean by construction; the caption says so). Each panel shows one reason a single SCALAR
performance summary cannot convey the lower tail the reward designer is supposed to care about:

* **(a) heavy tails** — the equal-weight daily-return density against a Normal with the SAME mean/std
  on a log density axis, annotated with the excess kurtosis and the observed-vs-Normal probability
  beyond −3σ / −5σ ("the Normal underestimates the −5σ day by ~N×").
* **(b) the tail is a CURVE, not a number** — empirical CVaR_α vs the Normal-implied CVaR_α as α
  shrinks 0.25 → 0.01, with the four FED levels (``cvar_25/10/05/01`` — the preregistration's
  ``tail_diagnostics.cvar_levels``) marked: the divergence as α → 0 is exactly the multi-level
  information the fed vector carries and any one scalar collapses to a point.
* **(c) tail risk CLUSTERS** — rolling 21-day realized volatility with the top-decile stress episodes
  shaded: a time-averaged scalar cannot say WHEN the risk arrives.
* **(d) diversification fails IN the tail** — the co-crash fraction (share of names simultaneously
  below their OWN 5th percentile) on calm vs stress days against the independence benchmark.

Every panel carries a one-line intuition takeaway in its subtitle. :func:`fig_stylised_facts` takes
ALREADY-COMPUTED arrays (decoupled from the licensed data; trivially testable) and
:func:`stylised_fact_stats` returns every annotated number as one dict (the Data-chapter prose source);
:func:`build_f3` is the thin wrapper that loads the real train window via
:func:`src.data.loaders.load_gold_panel`, renders PNG+PDF, and prints/returns the numbers. The empirical
CVaR is REUSED from :func:`src.inference.bootstrap.cvar` (the repo's canonical signed-return estimator),
so the figure and the inference layer cannot drift apart. Pure numpy/scipy/matplotlib; NO RNG anywhere.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.inference.bootstrap import cvar as empirical_cvar
from src.viz.style import OKABE_ITO, apply_house_style, savefig

__all__ = [
    "FED_CVAR_LEVELS",
    "alive_mask_from_returns",
    "normal_cvar",
    "rolling_realized_vol",
    "stress_episodes",
    "co_crash_fractions",
    "stylised_fact_stats",
    "fig_stylised_facts",
    "build_f3",
]

#: The FED tail levels — ``preregistration.yaml: tail_diagnostics.cvar_levels`` (cvar_25/10/05/01).
#: Panel (b) marks these on the empirical curve: the four points of the curve the distributional arm
#: is actually shown, where the scalar arm gets a single number.
FED_CVAR_LEVELS: tuple[float, ...] = (0.25, 0.10, 0.05, 0.01)

_TRADING_DAYS = 252  # annualisation convention (matches sharpe_ratio's periods_per_year default)


def alive_mask_from_returns(panel_returns: np.ndarray) -> np.ndarray:
    """``(T, N)`` bool mask: True from a name's FIRST to LAST non-zero return (its listed span).

    The gold loader's ``liquidate_to_cash`` policy zero-fills a delisted name's dead tail (and any
    leading gap), so aliveness is recovered as the span between the first and last non-zero daily
    return: every filled cell is exactly ``0.0`` by construction, while a LIVE large-cap essentially
    never prints a sustained exactly-zero boundary run. Panel (d) needs this so per-name crash
    thresholds and day-t denominators count only names actually trading — a zero-filled dead name can
    never "crash", which would silently DILUTE the co-crash fraction. An all-zero column is all-False.
    """
    r = np.asarray(panel_returns, dtype=float)
    if r.ndim != 2:
        raise ValueError(f"panel_returns must be 2-D (T, N); got shape {r.shape}")
    t = r.shape[0]
    nonzero = r != 0.0
    any_nz = nonzero.any(axis=0)
    first = np.where(any_nz, nonzero.argmax(axis=0), t)  # t -> empty span for all-zero columns
    last = np.where(any_nz, t - 1 - nonzero[::-1].argmax(axis=0), -1)
    day = np.arange(t)[:, None]
    return (day >= first[None, :]) & (day <= last[None, :])


def normal_cvar(mu: float, sigma: float, alpha: float) -> float:
    """CVaR_α of ``N(mu, sigma²)`` on SIGNED returns: ``mu − sigma·φ(Φ⁻¹(α))/α``.

    The closed-form Gaussian lower-tail mean, in the SAME convention as the repo's empirical
    estimator (:func:`src.inference.bootstrap.cvar`: the mean of the worst α-tail, negative for
    loss-making tails). Panel (b) plots this against the empirical curve — the benchmark the data
    must beat for the "a scalar cannot carry the tail" argument to hold.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1); got {alpha}")
    from scipy.stats import norm

    z = float(norm.ppf(alpha))
    return float(mu - sigma * float(norm.pdf(z)) / alpha)


def rolling_realized_vol(x: np.ndarray, window: int = 21) -> np.ndarray:
    """Trailing ``window``-day realized std (ddof=1) of a return series, length ``T − window + 1``.

    Element ``i`` covers days ``i..i+window−1`` — i.e. it is the vol KNOWN at day ``window−1+i``
    (window-end aligned, no look-ahead). Deterministic sliding-window numpy; no pandas.
    """
    v = np.asarray(x, dtype=float).ravel()
    if v.size < window:
        raise ValueError(f"need at least window={window} observations; got {v.size}")
    from numpy.lib.stride_tricks import sliding_window_view

    return sliding_window_view(v, window).std(axis=1, ddof=1)


def stress_episodes(
    vol: np.ndarray, stress_quantile: float = 0.90
) -> tuple[float, np.ndarray, list[tuple[int, int]]]:
    """``(threshold, mask, episodes)``: the top-``(1−q)`` vol days and their contiguous runs.

    ``mask`` is ``vol >= quantile(vol, stress_quantile)``; ``episodes`` lists each contiguous True
    run as an inclusive ``(start, end)`` index pair into ``vol``. "Tail risk arrives in clusters" is
    read directly off ``len(episodes)`` vs ``mask.sum()``: hundreds of stress DAYS collapse into a
    handful of EPISODES.
    """
    v = np.asarray(vol, dtype=float).ravel()
    if v.size == 0:
        raise ValueError("vol is empty")
    threshold = float(np.quantile(v, stress_quantile))
    mask = v >= threshold
    episodes: list[tuple[int, int]] = []
    start: int | None = None
    for i, hot in enumerate(mask):
        if hot and start is None:
            start = i
        elif not hot and start is not None:
            episodes.append((start, i - 1))
            start = None
    if start is not None:
        episodes.append((start, int(mask.size) - 1))
    return threshold, mask, episodes


def co_crash_fractions(
    panel_returns: np.ndarray,
    alive_mask: np.ndarray | None = None,
    *,
    crash_quantile: float = 0.05,
) -> np.ndarray:
    """Per-day fraction of ALIVE names below their OWN alive-day ``crash_quantile`` return.

    Each name's crash threshold is the ``crash_quantile`` (default 5th-percentile) quantile of its
    OWN daily returns over its alive days; day ``t``'s fraction is (alive names below their
    threshold) / (alive names). Under cross-sectional independence the expected fraction is
    ≈ ``crash_quantile`` on EVERY day — the benchmark panel (d) draws; tail dependence appears as
    stress-day fractions far above it. Days with zero alive names return NaN.
    """
    r = np.asarray(panel_returns, dtype=float)
    if r.ndim != 2:
        raise ValueError(f"panel_returns must be 2-D (T, N); got shape {r.shape}")
    alive = np.ones(r.shape, dtype=bool) if alive_mask is None else np.asarray(alive_mask, dtype=bool)
    if alive.shape != r.shape:
        raise ValueError(f"alive_mask shape {alive.shape} != panel shape {r.shape}")
    t, n = r.shape
    crashed = np.zeros((t, n), dtype=bool)
    for j in range(n):  # n is small (top-30); an explicit loop beats a masked megabroadcast here
        a = alive[:, j]
        if not a.any():
            continue
        threshold = float(np.quantile(r[a, j], crash_quantile))
        crashed[:, j] = a & (r[:, j] < threshold)
    n_alive = alive.sum(axis=1)
    frac = np.full(t, np.nan)
    live_day = n_alive > 0
    frac[live_day] = crashed.sum(axis=1)[live_day] / n_alive[live_day]
    return frac


def stylised_fact_stats(
    panel_returns: np.ndarray,
    *,
    alive_mask: np.ndarray | None = None,
    vol_window: int = 21,
    stress_quantile: float = 0.90,
    crash_quantile: float = 0.05,
    cvar_levels: Sequence[float] = FED_CVAR_LEVELS,
) -> dict[str, Any]:
    """Every number F3 annotates, as one plain dict — the Data-chapter prose source of truth.

    The running example is the EQUAL-WEIGHT portfolio of the (zero-filled) panel — exactly the
    return stream an equal-weight policy realises in the environment under ``liquidate_to_cash``
    (dead proceeds held flat). Keys: ``n_days``/``n_assets``/``n_dead_by_end``; the EW moments
    (``mean_daily``, ``std_daily``, ``skewness``, ``excess_kurtosis`` — Fisher, bias-corrected,
    matching ``validate_stylized_facts``); ``tail_3sigma``/``tail_5sigma`` (observed count vs the
    Normal's expectation and their ratio — NaN ratio when nothing observed); ``cvar_by_level``
    (empirical vs Normal-implied, signed daily); the stress-clustering block (threshold /
    days / episodes / longest); and the co-crash block (calm vs stress means, ratio, worst day).
    """
    r = np.asarray(panel_returns, dtype=float)
    if r.ndim != 2:
        raise ValueError(f"panel_returns must be 2-D (T, N); got shape {r.shape}")
    from scipy.stats import kurtosis, norm, skew

    portfolio = r.mean(axis=1)
    t = int(portfolio.size)
    mu = float(portfolio.mean())
    sd = float(portfolio.std(ddof=1))
    if not (np.isfinite(sd) and sd > 0.0):
        raise ValueError(f"degenerate portfolio series (std={sd}); check the input panel")

    out: dict[str, Any] = {
        "n_days": t,
        "n_assets": int(r.shape[1]),
        "n_dead_by_end": (
            int((~np.asarray(alive_mask, dtype=bool)[-1]).sum()) if alive_mask is not None else 0
        ),
        "mean_daily": mu,
        "std_daily": sd,
        "skewness": float(skew(portfolio, bias=False)),
        "excess_kurtosis": float(kurtosis(portfolio, fisher=True, bias=False)),
    }

    for k in (3, 5):
        count = int((portfolio < mu - k * sd).sum())
        empirical_prob = count / t
        normal_prob = float(norm.cdf(-k))
        out[f"tail_{k}sigma"] = {
            "count": count,
            "empirical_prob": empirical_prob,
            "normal_prob": normal_prob,
            "normal_expected_days": normal_prob * t,
            "ratio": (empirical_prob / normal_prob) if count > 0 else float("nan"),
        }

    by_level: dict[float, dict[str, float]] = {}
    for a in cvar_levels:
        emp = float(empirical_cvar(portfolio, float(a)))
        gauss = normal_cvar(mu, sd, float(a))
        by_level[float(a)] = {
            "empirical": emp,
            "normal": gauss,
            "ratio": emp / gauss if gauss != 0.0 else float("nan"),
        }
    out["cvar_by_level"] = by_level

    vol = rolling_realized_vol(portfolio, vol_window)
    threshold, stress, episodes = stress_episodes(vol, stress_quantile)
    out["vol_window"] = int(vol_window)
    out["stress_quantile"] = float(stress_quantile)
    out["stress_threshold_ann_pct"] = threshold * math.sqrt(_TRADING_DAYS) * 100.0
    out["n_stress_days"] = int(stress.sum())
    out["n_stress_episodes"] = len(episodes)
    out["longest_episode_days"] = max((e - s + 1 for s, e in episodes), default=0)

    frac = co_crash_fractions(r, alive_mask, crash_quantile=crash_quantile)
    # Align vol to days: vol[i] is known at day (vol_window-1+i); the first vol_window-1 days are
    # UNCLASSIFIED and excluded from BOTH groups (never silently binned as calm).
    day_is_stress = np.zeros(t, dtype=bool)
    day_is_stress[vol_window - 1 :] = stress
    classified = np.zeros(t, dtype=bool)
    classified[vol_window - 1 :] = True
    valid = classified & np.isfinite(frac)
    calm_frac = frac[valid & ~day_is_stress]
    stress_frac = frac[valid & day_is_stress]
    calm_mean = float(calm_frac.mean()) if calm_frac.size else float("nan")
    stress_mean = float(stress_frac.mean()) if stress_frac.size else float("nan")
    out["crash_quantile"] = float(crash_quantile)
    out["co_crash_calm_mean"] = calm_mean
    out["co_crash_stress_mean"] = stress_mean
    out["co_crash_ratio"] = stress_mean / calm_mean if calm_mean > 0.0 else float("nan")
    finite = np.where(np.isfinite(frac), frac, -np.inf)
    worst = int(np.argmax(finite))
    out["worst_day_index"] = worst
    out["worst_day_co_crash"] = float(frac[worst])
    return out


# --------------------------------------------------------------------------------------------------- #
# The figure                                                                                           #
# --------------------------------------------------------------------------------------------------- #
def _fmt_ratio(x: float, mark: str = "×") -> str:
    """Human ratio: '×1,234' / '×8.3' / 'n/a' for a NaN (nothing observed).

    ``mark="x"`` gives the ASCII form for CONSOLE prints: a legacy Windows codepage (cp1251/cp1252
    stdout) cannot encode U+00D7 and would crash ``build_f3``'s report block; figure text keeps ``×``
    (matplotlib renders unicode regardless of the console encoding).
    """
    if not np.isfinite(x):
        return "n/a"
    return f"{mark}{x:,.0f}" if x >= 10 else f"{mark}{x:.1f}"


def fig_stylised_facts(
    panel_returns: np.ndarray,
    *,
    dates: np.ndarray | None = None,
    alive_mask: np.ndarray | None = None,
    vol_window: int = 21,
    stress_quantile: float = 0.90,
    crash_quantile: float = 0.05,
    cvar_levels: Sequence[float] = FED_CVAR_LEVELS,
    n_alpha: int = 41,
    title: str = "Training years, 2005 to 2016",
    footnote: str | None = None,
) -> Any:
    """The 2×2 F3 figure from a ``(T, N)`` daily simple-return panel (see the module docstring).

    ``dates`` (optional ``datetime64``) drives panel (c)'s axis — an integer day index otherwise;
    ``alive_mask`` (optional, from :func:`alive_mask_from_returns`) makes panel (d) count only
    trading names. All annotated numbers come from :func:`stylised_fact_stats` (one source), the
    empirical CVaR from :func:`src.inference.bootstrap.cvar`. Deterministic — no RNG, no jitter.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    r = np.asarray(panel_returns, dtype=float)
    stats = stylised_fact_stats(
        r,
        alive_mask=alive_mask,
        vol_window=vol_window,
        stress_quantile=stress_quantile,
        crash_quantile=crash_quantile,
        cvar_levels=cvar_levels,
    )
    portfolio = r.mean(axis=1)
    t = portfolio.size
    mu, sd = stats["mean_daily"], stats["std_daily"]
    blue, verm = OKABE_ITO["blue"], OKABE_ITO["vermillion"]

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.2))

    # -- (a) heavy tails: observed density vs the matched Normal (log density axis) ------------------ #
    ax = axes[0, 0]
    lo, hi = float(portfolio.min()), float(portfolio.max())
    pad = 0.06 * (hi - lo)
    bins = np.linspace(lo - pad, hi + pad, 120)
    ax.hist(portfolio, bins=bins, density=True, color=blue, alpha=0.55, label="observed (EW daily)")
    xs = np.linspace(lo - pad, hi + pad, 400)
    ax.plot(xs, norm.pdf(xs, mu, sd), color=verm, lw=1.6, ls="--", label="Normal, same μ, σ")
    ax.set_yscale("log")
    single_obs_density = 1.0 / (t * float(bins[1] - bins[0]))
    ax.set_ylim(bottom=single_obs_density / 5.0)  # one-observation bars stay visible; Normal dives off
    from matplotlib.transforms import blended_transform_factory

    sigma_label_transform = blended_transform_factory(ax.transData, ax.transAxes)
    for k in (3, 5):
        ax.axvline(mu - k * sd, color="0.45", lw=0.8, ls=":")
        ax.text(mu - k * sd, 0.98, f"−{k}σ", transform=sigma_label_transform,
                ha="center", va="top", fontsize=7, color="0.35")
    t3, t5 = stats["tail_3sigma"], stats["tail_5sigma"]
    ax.text(
        0.03, 0.04,
        (
            f"excess kurtosis = {stats['excess_kurtosis']:.1f}  (Normal: 0)\n"
            f"< −3σ: {t3['count']} days vs {t3['normal_expected_days']:.1f} expected "
            f"({_fmt_ratio(t3['ratio'])})\n"
            f"< −5σ: {t5['count']} days vs {t5['normal_expected_days']:.4f} expected "
            f"({_fmt_ratio(t5['ratio'])})"
        ),
        transform=ax.transAxes, va="bottom", fontsize=7.5,
        bbox=dict(boxstyle="round", fc="white", ec="0.6"),
    )
    ax.set_xlabel("EW portfolio daily return")
    ax.set_ylabel("density (log)")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title(
        "(a) Heavy tails — the crash days a Normal cannot see\n"
        "a matched Normal misses the deepest crash days",
        loc="left", fontsize=9,
    )

    # -- (b) the lower tail up close: CVaR_α curve, empirical vs Normal-implied ---------------------- #
    ax = axes[0, 1]
    ax.margins(x=0.06, y=0.12)  # keep the deepest fed-level marker + its label off the axis edge
    levels = tuple(float(a) for a in cvar_levels)
    alphas = np.unique(np.concatenate([np.geomspace(min(levels), max(levels), n_alpha), levels]))
    emp_curve = np.array([empirical_cvar(portfolio, float(a)) for a in alphas]) * 100.0
    gauss_curve = np.array([normal_cvar(mu, sd, float(a)) for a in alphas]) * 100.0
    ax.plot(alphas, emp_curve, color=blue, lw=1.8, label="empirical CVaR$_\\alpha$", zorder=3)
    ax.plot(alphas, gauss_curve, color=verm, lw=1.6, ls="--", label="Normal-implied", zorder=2)
    ax.fill_between(alphas, emp_curve, gauss_curve, color=blue, alpha=0.10, zorder=1)
    for a in levels:
        fed = stats["cvar_by_level"][a]
        ax.plot([a], [fed["empirical"] * 100.0], marker="o", ms=6, color=blue,
                markeredgecolor="black", markeredgewidth=0.5, zorder=4)
        ax.annotate(f"cvar_{int(round(a * 100)):02d}", (a, fed["empirical"] * 100.0),
                    textcoords="offset points", xytext=(0, -11), ha="center", fontsize=6.5,
                    color="0.25")
    deepest = min(levels)
    deep = stats["cvar_by_level"][deepest]
    ax.text(
        0.03, 0.04,
        (
            f"at α = {deepest:g}: empirical {deep['empirical'] * 100.0:.2f}% "
            f"vs Normal {deep['normal'] * 100.0:.2f}%  ({_fmt_ratio(deep['ratio'])})"
        ),
        transform=ax.transAxes, va="bottom", fontsize=7.5,
        bbox=dict(boxstyle="round", fc="white", ec="0.6"),
    )
    ax.set_xscale("log")
    ax.invert_xaxis()  # deeper into the tail reads left -> right
    ax.set_xticks(sorted(levels, reverse=True))
    ax.set_xticklabels([f"{a:g}" for a in sorted(levels, reverse=True)])
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_xlabel("tail level α  (log scale, deeper tail to the right)")
    ax.set_ylabel("CVaR$_\\alpha$ (daily %, signed)")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title(
        "(b) The tail is a curve, not a number — fed levels marked\n"
        "the curves diverge as α → 0; a scalar picks one point",
        loc="left", fontsize=9,
    )

    # -- (c) volatility clustering: rolling vol with top-decile stress episodes shaded --------------- #
    ax = axes[1, 0]
    vol = rolling_realized_vol(portfolio, vol_window)
    threshold, stress, episodes = stress_episodes(vol, stress_quantile)
    ann = vol * math.sqrt(_TRADING_DAYS) * 100.0
    if dates is not None:
        x = np.asarray(dates)[vol_window - 1 :]
    else:
        x = np.arange(vol_window - 1, t)
    for s, e in episodes:
        ax.axvspan(x[s], x[min(e + 1, x.size - 1)], color=verm, alpha=0.16, zorder=0, lw=0)
    ax.plot(x, ann, color="0.30", lw=0.9, zorder=2)
    ax.axhline(threshold * math.sqrt(_TRADING_DAYS) * 100.0, color=verm, lw=0.9, ls="--", zorder=1)
    ax.text(
        0.03, 0.96,
        (
            # Render the stress band FROM the parameter (deep review 2026-07-26, #55): this read
            # "top-decile" as a hardcoded word while ``stress_quantile`` is a public argument, so a
            # caller passing 0.95 would have had a top-5% set labelled a decile. Panel (d) already
            # renders its own ``crash_quantile`` dynamically — this now matches it. The paper-facing
            # render was never wrong (``build_f3`` uses the 0.90 default = a true decile).
            f"{stats['n_stress_episodes']} stress episodes hold all "
            f"{stats['n_stress_days']} top-{1.0 - stress_quantile:.0%} vol days;  longest "
            f"{stats['longest_episode_days']} sessions"
        ),
        transform=ax.transAxes, va="top", fontsize=7.5,
        bbox=dict(boxstyle="round", fc="white", ec="0.6"),
    )
    ax.set_ylabel(f"{vol_window}-day realized vol (annualised %)")
    ax.set_xlabel("session" if dates is None else "")
    ax.set_title(
        "(c) Tail risk arrives in clusters, not uniformly\n"
        "a time-averaged scalar hides WHEN the risk strikes",
        loc="left", fontsize=9,
    )

    # -- (d) cross-sectional tail dependence: co-crash fraction, calm vs stress ---------------------- #
    ax = axes[1, 1]
    calm_mean, stress_mean = stats["co_crash_calm_mean"], stats["co_crash_stress_mean"]
    ax.bar([0, 1], [calm_mean * 100.0, stress_mean * 100.0], width=0.58,
           color=["0.65", verm], edgecolor="black", linewidth=0.5, zorder=2)
    ax.axhline(crash_quantile * 100.0, color=blue, lw=1.0, ls="--", zorder=3)
    ax.text(-0.38, crash_quantile * 100.0, f"independence ≈ {crash_quantile:.0%}",
            fontsize=6.8, color=blue, va="bottom", ha="left")
    for i, v in enumerate((calm_mean, stress_mean)):
        if np.isfinite(v):
            ax.text(i, v * 100.0, f" {v * 100.0:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.text(
        0.03, 0.82,
        (
            f"stress / calm {_fmt_ratio(stats['co_crash_ratio'])};  worst day: "
            f"{stats['worst_day_co_crash']:.0%} of names below their own "
            f"{crash_quantile:.0%} tail at once"
        ),
        transform=ax.transAxes, va="top", fontsize=7.5,
        bbox=dict(boxstyle="round", fc="white", ec="0.6"),
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["calm days", "stress days"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel(f"names below their own {crash_quantile:.0%} quantile (mean %)")
    ax.set_title(
        "(d) Co-crashes — diversification fails in the tail\n"
        "in stress the cross-section falls together",
        loc="left", fontsize=9,
    )

    fig.suptitle(title, fontsize=10)
    if footnote:
        # Reserve a bottom strip for the footnote so constrained layout (the house rcParams) does not
        # pack the bottom axes over it; a non-constrained figure keeps its default margin anyway.
        from matplotlib.layout_engine import ConstrainedLayoutEngine

        engine = fig.get_layout_engine()
        if isinstance(engine, ConstrainedLayoutEngine):
            engine.set(rect=(0.0, 0.03, 1.0, 0.97))
        fig.text(0.01, 0.005, footnote, fontsize=6.3, color="0.35", ha="left", va="bottom")
    return fig


# --------------------------------------------------------------------------------------------------- #
# The real-data wrapper                                                                                #
# --------------------------------------------------------------------------------------------------- #
def build_f3(
    out_path: Path | str = "outputs/figures/F3_stylised_facts.png",
    *,
    phase: str = "development",
    end: Any = None,
    gold_dir: Path | str | None = None,
    verify_checksum: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Load the REAL gold TRAIN window, render F3 (PNG @600dpi + vector PDF), return ``(path, stats)``.

    Snoop-cleanliness: ``phase="development"`` with the loader's default ``end`` slices ONLY the
    train split (2005-01-03…2016-12-30, SPLIT C) — the sealed validation/test years are never read, and the
    baked-in figure footnote says so. The panel arrives anonymised (integer ids; no RICs/tickers) and
    zero-filled under ``liquidate_to_cash``; the alive mask recovered from the zero fill keeps panel
    (d) honest about delisted names. Prints the annotated numbers (the Data-chapter prose inputs) and
    returns them alongside the PNG path. Raises :class:`FileNotFoundError` when the licensed gold
    panel is absent (synthetic-only installs render no F3 — never a fabricated one).
    """
    from src.data.loaders import load_gold_panel  # local: keeps src.viz importable without pandas

    kwargs: dict[str, Any] = {} if gold_dir is None else {"gold_dir": gold_dir}
    result = load_gold_panel(phase, end=end, verify_checksum=verify_checksum, **kwargs)
    returns = result.panel.returns
    dates = result.panel.dates
    alive = alive_mask_from_returns(returns)
    d0, d1 = str(dates[0])[:10], str(dates[-1])[:10]
    footnote = (
        f"Descriptive EDA on the TRAIN window only ({phase} top-{returns.shape[1]}, {d0} → {d1}; "
        "anonymised ids; delisted names liquidate-to-cash) — the sealed validation/test years are "
        "never read. Panel (d) counts alive names only."
    )
    apply_house_style()
    fig = fig_stylised_facts(returns, dates=dates, alive_mask=alive, footnote=footnote)
    path = Path(out_path)
    savefig(fig, path)  # PNG + vector-PDF sibling (the engine's standard formats)
    import matplotlib.pyplot as plt

    plt.close(fig)

    stats = stylised_fact_stats(returns, alive_mask=alive)
    stats["window"] = f"{d0}..{d1}"
    t3, t5 = stats["tail_3sigma"], stats["tail_5sigma"]
    worst_date = str(dates[stats["worst_day_index"]])[:10]
    print(
        f"[F3] train window {d0} -> {d1}: {stats['n_days']} sessions x {stats['n_assets']} names "
        f"({stats['n_dead_by_end']} dead by window end)"
    )
    print(
        f"[F3] EW daily: mean {stats['mean_daily']:+.5f}, sd {stats['std_daily']:.5f}, "
        f"skew {stats['skewness']:.2f}, excess kurtosis {stats['excess_kurtosis']:.2f}"
    )
    print(
        f"[F3] < -3sigma: {t3['count']} days vs {t3['normal_expected_days']:.2f} Normal-expected "
        f"({_fmt_ratio(t3['ratio'], 'x')}); < -5sigma: {t5['count']} vs "
        f"{t5['normal_expected_days']:.4f} ({_fmt_ratio(t5['ratio'], 'x')})"
    )
    for a, row in sorted(stats["cvar_by_level"].items(), reverse=True):
        print(
            f"[F3] CVaR_{a:g}: empirical {row['empirical'] * 100.0:+.2f}%/day vs Normal "
            f"{row['normal'] * 100.0:+.2f}% ({_fmt_ratio(row['ratio'], 'x')})"
        )
    print(
        f"[F3] stress: {stats['n_stress_episodes']} episodes / {stats['n_stress_days']} days at "
        f">= {stats['stress_threshold_ann_pct']:.1f}% ann vol; longest "
        f"{stats['longest_episode_days']} sessions"
    )
    print(
        f"[F3] co-crash: calm {stats['co_crash_calm_mean']:.1%} vs stress "
        f"{stats['co_crash_stress_mean']:.1%} ({_fmt_ratio(stats['co_crash_ratio'], 'x')}); "
        f"worst day {worst_date}: {stats['worst_day_co_crash']:.0%}"
    )
    return path, stats
