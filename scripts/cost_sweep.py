"""Transaction-cost robustness sweep — RE-PRICE the frozen winners across `costs.grid_bps`.

Why this exists (the cost-robustness arm)
-----------------------------------------
`config/environment.yaml` declares ``costs.grid_bps = [0, 5, 10, 25, 50]`` but, before Rank
15, nothing consumed it: there was no ``cost_bps`` override on the env and no sweep harness, so
the **cost-robustness arm** — which stands in for the absent market-impact model and is the
viva cost defence — could not be produced. This module is that harness. It answers the key
robustness question (PREREGISTRATION §10 power-caveat companion; FINAL_PLAN cost note):

  **Does a tail-aware reward still win once you charge it more to trade — i.e. is the
  distributional arm's edge a genuine risk-shape effect, or merely an artefact of trading
  less at the headline 10 bps?**

It RE-PRICES the **FROZEN** winning policies WITHOUT retraining (results replay
from the archive; the test leg is touched once). Two re-pricing paths, identical to ~1e-12
(``tests/test_cost_sweep.py``):

  1. **ANALYTIC (preferred).** The campaign persists each frozen-winner test path's per-step
     GROSS and TURNOVER (``metrics['test_gross']`` + ``metrics['test_turnover']``, Rank 15).
     Because the cost is charged AFTER the action is chosen (audit C-5), the realized gross
     and turnover do NOT depend on the cost level, so for any per-unit cost ``c`` (a fraction)
     the net series is exactly::

         net_c = gross - c * turnover,   c = bps * 1e-4

     No env, no policy, no retraining — a vectorised re-price of the sealed test path.
  2. **RE-ROLL (fallback / cross-check).** When a record lacks the decomposition (an older
     archive), and a ``panel`` + ``reward_builder`` + injected ``policy`` are supplied, the
     frozen policy is re-rolled through a ``cost_bps``-OVERRIDDEN ``PortfolioEnv`` (the new
     ``PortfolioEnv(..., cost_bps=c)`` / ``make_env_builder(..., cost_bps=c)`` plumbing). This
     is also what the consistency test exercises against the analytic path.

Output (the headline robustness TABLE)
--------------------------------------
``sweep_markdown`` renders the **winner-identity-vs-cost** table — which arm wins at each cost
level (by test Sharpe) AND every arm's Sharpe + CVaR-5% at each level — exactly the check that
the tail-aware reward does not win merely by trading less. One row per cost level.

This module reads results ONLY through ``src.io.results`` (audit C-1) and is import-light /
torch-free for the analytic path (so the fast suite covers it).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np

__all__ = [
    "ARMS",
    "arm_test_decomposition",
    "reprice_analytic",
    "reprice_reroll",
    "cost_curve",
    "winner_by_cost",
    "sweep_markdown",
    "run_cost_sweep",
]

_LOG = logging.getLogger(__name__)

#: The seven pre-registered arms (PREREGISTRATION §3); the sweep reports each that has a usable
#: frozen-winner test record, and skips the rest (never fabricates a curve for a missing arm).
ARMS: tuple[str, ...] = (
    "distributional",
    "scalar",
    "placebo",
    "scalar_cvar5",
    "placebo_shuffled",
    "random_search",
    "bayes_opt",
)


# --------------------------------------------------------------------------- #
# Extracting the per-step gross/turnover decomposition from test records        #
# --------------------------------------------------------------------------- #
def _finite_vec(value: Any) -> np.ndarray | None:
    """Return ``value`` as a finite 1-D float array, or ``None`` if absent/degenerate."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=float).ravel()
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    return arr


def _record_decomposition(record: dict[str, Any]) -> dict[str, np.ndarray] | None:
    """Pull one frozen-winner record's per-step (gross, turnover), or ``None``.

    The vectors ride inside ``metrics`` (``test_gross`` / ``test_turnover``). ANALYTIC
    re-pricing needs BOTH gross and turnover, so a record carrying only the NET
    ``test_returns`` returns ``None`` here (the caller then falls back to re-rolling). The
    persisted NET series is intentionally NOT carried: nothing downstream consumes it — the
    analytic path recomputes net via :func:`reprice_analytic` and the re-roll path takes
    ``series["net"]`` from the live env — so extracting it here would be dead code.
    """
    metrics = record.get("metrics") or {}
    gross = _finite_vec(metrics.get("test_gross"))
    turnover = _finite_vec(metrics.get("test_turnover"))
    if gross is None or turnover is None:
        return None
    n = min(gross.size, turnover.size)
    return {"gross": gross[:n], "turnover": turnover[:n]}


def arm_test_decomposition(
    records: list[dict[str, Any]], arm: str
) -> dict[str, np.ndarray] | None:
    """Pool one arm's per-seed frozen-winner (gross, turnover) into a single aligned path.

    Mirrors ``analyze_campaign._arm_test_returns``: the campaign writes one record per
    (arm, seed); every seed's frozen-winner test path is the SAME held-out calendar, so the
    per-seed gross/turnover vectors are aligned to the common leading length and AVERAGED
    across seeds (rliable-style per-period mean, PREREGISTRATION §10) to yield ONE
    decomposition per arm. A single-seed campaign reduces to that seed's vectors unchanged.
    Returns ``None`` when no record for the arm carries the (gross, turnover) decomposition.
    """
    decomps = [
        d
        for r in records
        if r.get("arm") == arm and (d := _record_decomposition(r)) is not None
    ]
    if not decomps:
        return None
    t_min = min(d["gross"].size for d in decomps)
    if t_min == 0:
        return None
    gross = np.vstack([d["gross"][:t_min] for d in decomps]).mean(axis=0)
    turnover = np.vstack([d["turnover"][:t_min] for d in decomps]).mean(axis=0)
    return {"gross": gross, "turnover": turnover}


# --------------------------------------------------------------------------- #
# The two re-pricing paths                                                     #
# --------------------------------------------------------------------------- #
def reprice_analytic(
    gross: np.ndarray, turnover: np.ndarray, cost_bps: float
) -> np.ndarray:
    """ANALYTIC re-price: ``net_c = gross - (cost_bps * 1e-4) * turnover`` (vectorised).

    The exact identity the env computes step-for-step (``port_ret = gross - cost``,
    ``cost = c * turnover``), evaluated at the alternative per-unit cost ``c = cost_bps*1e-4``.
    Reproduces the headline net series when ``cost_bps`` equals the headline (10 bps) and
    re-prices any other grid level without touching the env (the gross/turnover are
    cost-independent — audit C-5).
    """
    g = np.asarray(gross, dtype=float).ravel()
    tn = np.asarray(turnover, dtype=float).ravel()
    n = min(g.size, tn.size)
    return g[:n] - (float(cost_bps) * 1e-4) * tn[:n]


def reprice_reroll(
    panel: Any,
    cfg: Any,
    test_window: tuple[int, int],
    reward_fn: Any,
    policy: Any,
    cost_bps: float,
) -> dict[str, np.ndarray]:
    """RE-ROLL the frozen policy through a ``cost_bps``-OVERRIDDEN env; return gross/turnover/net.

    The fallback (and the consistency cross-check): build a fresh ``PortfolioEnv`` over the
    test window with ``cost_bps=cost_bps`` and roll the (frozen) ``policy`` + ``reward_fn``
    through it via ``rollout_port_series``. The realized gross/turnover are identical to the
    headline roll (the policy is the same), and ``net`` equals the analytic re-price to
    machine precision — which ``tests/test_cost_sweep.py`` asserts.
    """
    from src.env.portfolio_env import PortfolioEnv
    from src.env.runner import rollout_port_series

    start, end = int(test_window[0]), int(test_window[1])
    env = PortfolioEnv(panel, cfg, reward_fn, start=start, end=end, cost_bps=float(cost_bps))
    return rollout_port_series(env, policy)


# --------------------------------------------------------------------------- #
# The per-arm cost curve (Sharpe + CVaR at each grid level)                     #
# --------------------------------------------------------------------------- #
def cost_curve(
    records: list[dict[str, Any]],
    grid_bps: "list[float] | tuple[float, ...]",
    *,
    arms: "tuple[str, ...] | list[str]" = ARMS,
    cvar_alpha: float = 0.05,
    panel: Any = None,
    cfg: Any = None,
    test_window: "tuple[int, int] | None" = None,
    reward_for_arm: "Callable[[str], Any] | None" = None,
    policy_for_arm: "Callable[[str], Any] | None" = None,
) -> dict[str, Any]:
    """Re-price every arm at every ``grid_bps`` level; return Sharpe + CVaR curves.

    For each arm, the per-step (gross, turnover) is pooled across seeds
    (:func:`arm_test_decomposition`) and re-priced ANALYTICALLY at each cost. If an arm lacks
    the decomposition, and ``panel`` + ``cfg`` + ``test_window`` + ``reward_for_arm`` +
    ``policy_for_arm`` are supplied, it re-rolls instead (fallback); otherwise that arm is
    reported as skipped (never fabricated).

    Parameters
    ----------
    records : list of dict
        Per-(arm, seed) frozen-winner TEST records (``src.io.results.load_all``).
    grid_bps : sequence of float
        The cost grid in basis points (``config/environment.yaml: costs.grid_bps``).
    arms : sequence of str
        Arms to report (default: the six pre-registered arms).
    cvar_alpha : float
        CVaR tail level for the reported curve (default headline ``0.05``).
    panel, cfg, test_window, reward_for_arm, policy_for_arm
        Optional re-roll-fallback collaborators (only used for an arm lacking the persisted
        decomposition). ``reward_for_arm(arm) -> reward_fn`` and ``policy_for_arm(arm) ->
        policy`` rebuild the frozen winner; the env is ``cost_bps``-overridden per level.

    Returns
    -------
    dict
        ``{"grid_bps": [...], "cvar_alpha": ..., "arms": {arm: {"source": "analytic"|"reroll",
        "sharpe": [...], "cvar": [...], "n_steps": int}}, "skipped": {arm: reason}}``. Each
        ``sharpe``/``cvar`` list is aligned to ``grid_bps``.
    """
    from src.inference.bootstrap import cvar, sharpe_ratio

    grid = [float(b) for b in grid_bps]
    out_arms: dict[str, Any] = {}
    skipped: dict[str, str] = {}

    can_reroll = (
        panel is not None
        and cfg is not None
        and test_window is not None
        and reward_for_arm is not None
        and policy_for_arm is not None
    )

    for arm in arms:
        decomp = arm_test_decomposition(records, arm)
        source: str
        sharpes: list[float] = []
        cvars: list[float] = []

        if decomp is not None:
            source = "analytic"
            gross, turnover = decomp["gross"], decomp["turnover"]
            for b in grid:
                net = reprice_analytic(gross, turnover, b)
                sharpes.append(float(sharpe_ratio(net)))
                cvars.append(float(cvar(net, cvar_alpha)))
            n_steps = int(gross.size)
        elif can_reroll:
            source = "reroll"
            reward_fn = reward_for_arm(arm)  # type: ignore[misc]
            policy = policy_for_arm(arm)  # type: ignore[misc]
            if reward_fn is None or policy is None:
                skipped[arm] = "no decomposition and re-roll reward/policy unavailable"
                continue
            n_steps = 0
            for b in grid:
                series = reprice_reroll(
                    panel, cfg, test_window, reward_fn, policy, b
                )
                net = series["net"]
                sharpes.append(float(sharpe_ratio(net)))
                cvars.append(float(cvar(net, cvar_alpha)))
                n_steps = int(net.size)
        else:
            skipped[arm] = "no test record carrying gross/turnover decomposition"
            continue

        out_arms[arm] = {
            "source": source,
            "sharpe": sharpes,
            "cvar": cvars,
            "n_steps": n_steps,
        }

    return {
        "grid_bps": grid,
        "cvar_alpha": float(cvar_alpha),
        "arms": out_arms,
        "skipped": skipped,
    }


def winner_by_cost(curve: dict[str, Any]) -> list[dict[str, Any]]:
    """Reduce a :func:`cost_curve` to the winner-identity-vs-cost rows (one per cost level).

    The winner at a cost level is the arm with the greatest test Sharpe at that level (ties
    broken by the higher — less negative — CVaR, then arm name, for determinism). Each row also
    records whether the headline (10-bps) winner is dethroned, so a glance at the table answers
    "does the tail-aware arm survive a higher cost?".

    Returns
    -------
    list of dict
        ``[{"bps": float, "winner": str|None, "sharpe": float, "cvar": float,
        "per_arm": {arm: {"sharpe": float, "cvar": float}}}, ...]``, one row per cost level.
    """
    grid = curve["grid_bps"]
    arms = curve["arms"]
    rows: list[dict[str, Any]] = []
    for i, b in enumerate(grid):
        per_arm = {
            arm: {"sharpe": d["sharpe"][i], "cvar": d["cvar"][i]}
            for arm, d in arms.items()
        }
        if per_arm:
            # Highest Sharpe; tie-break on higher CVaR then lexicographic arm name.
            winner = max(
                per_arm,
                key=lambda a: (per_arm[a]["sharpe"], per_arm[a]["cvar"], _neg_name(a)),
            )
            rows.append(
                {
                    "bps": float(b),
                    "winner": winner,
                    "sharpe": float(per_arm[winner]["sharpe"]),
                    "cvar": float(per_arm[winner]["cvar"]),
                    "per_arm": per_arm,
                }
            )
        else:
            rows.append({"bps": float(b), "winner": None, "sharpe": float("nan"),
                         "cvar": float("nan"), "per_arm": {}})
    return rows


def _neg_name(arm: str) -> tuple[int, ...]:
    """A deterministic, descending-name tie-break key (so ``max`` prefers the earlier name)."""
    return tuple(-ord(c) for c in arm)


def sweep_markdown(curve: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    """Render the headline winner-identity-vs-cost robustness TABLE as markdown.

    One row per cost level: the winning arm (by Sharpe) and EVERY arm's Sharpe / CVaR-5% at
    that cost — the key check that a tail-aware reward doesn't win merely by trading less.
    """
    arms = list(curve["arms"].keys())
    a = curve["cvar_alpha"]

    lines = [
        "# Transaction-cost robustness sweep (`costs.grid_bps`; PREREGISTRATION §10 companion)",
        "",
        "Frozen winners RE-PRICED across the cost grid WITHOUT retraining "
        "(`net_c = gross − c·turnover`, `c = bps·1e-4`; the gross/turnover are cost-independent "
        "because cost is charged after the action, audit C-5). The question: does the "
        "tail-aware (`distributional`) arm still win once trading costs more, or did it merely "
        "trade less at the headline 10 bps?",
        "",
    ]

    # Winner-identity row table.
    lines.append("## Winner identity vs cost")
    lines.append("")
    lines.append("| cost (bps) | winning arm | winner Sharpe | winner CVaR%g |".replace("%g", f"{a:g}"))
    lines.append("|---|---|---|---|")
    for row in rows:
        w = row["winner"] if row["winner"] is not None else "—"
        lines.append(
            f"| {row['bps']:g} | {w} | {row['sharpe']:.4f} | {row['cvar']:.5f} |"
        )
    lines.append("")

    # Per-arm Sharpe-at-cost table.
    lines.append("## Sharpe at each cost level")
    lines.append("")
    lines.append("| arm | source | " + " | ".join(f"{b:g} bps" for b in curve["grid_bps"]) + " |")
    lines.append("|---|---|" + "---|" * len(curve["grid_bps"]))
    for arm in arms:
        d = curve["arms"][arm]
        cells = " | ".join(f"{s:.4f}" for s in d["sharpe"])
        lines.append(f"| {arm} | {d['source']} | {cells} |")
    lines.append("")

    # Per-arm CVaR-at-cost table.
    lines.append(f"## CVaR-{a:g} at each cost level (higher = less-negative tail = better)")
    lines.append("")
    lines.append("| arm | source | " + " | ".join(f"{b:g} bps" for b in curve["grid_bps"]) + " |")
    lines.append("|---|---|" + "---|" * len(curve["grid_bps"]))
    for arm in arms:
        d = curve["arms"][arm]
        cells = " | ".join(f"{c:.5f}" for c in d["cvar"])
        lines.append(f"| {arm} | {d['source']} | {cells} |")
    lines.append("")

    if curve.get("skipped"):
        lines.append("## Skipped arms (no usable frozen-winner test record)")
        lines.append("")
        for arm, reason in curve["skipped"].items():
            lines.append(f"- `{arm}`: {reason}")
        lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
def run_cost_sweep(
    test_root: str | Path,
    grid_bps: "list[float] | tuple[float, ...]",
    *,
    arms: "tuple[str, ...] | list[str]" = ARMS,
    cvar_alpha: float = 0.05,
) -> dict[str, Any]:
    """Load a campaign's frozen-winner TEST archive and re-price across ``grid_bps``.

    Reads the per-(arm, seed) test records (one per arm subdirectory) via the canonical loader,
    re-prices each arm analytically across the grid, and returns the curve + the
    winner-identity rows. This is the production (analytic-only) entry point; the re-roll
    fallback is reserved for the consistency test / older archives that lack the decomposition.
    """
    try:
        from scripts.analyze_campaign import load_campaign_records
    except ModuleNotFoundError:  # `python scripts/cost_sweep.py` (documented in the runbook) puts
        import sys as _sys                     # scripts/ on sys.path, NOT the repo root -> `scripts`
        from pathlib import Path as _P          # is not an importable package: this UNGUARDED import
        _sys.path.insert(0, str(_P(__file__).resolve().parent))  # HARD-CRASHED the documented cost-
        from analyze_campaign import load_campaign_records        # sweep command (seal-bug class, 2026-07-24)

    records = load_campaign_records(test_root)
    curve = cost_curve(records, grid_bps, arms=arms, cvar_alpha=cvar_alpha)
    rows = winner_by_cost(curve)
    return {
        "test_root": str(test_root),
        "n_records": len(records),
        "curve": curve,
        "winner_by_cost": rows,
    }


def write_report(result: dict[str, Any], out_dir: str | Path) -> Path:
    """Write ``cost_sweep.md`` + ``cost_sweep.json`` into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = sweep_markdown(result["curve"], result["winner_by_cost"])
    report = out_dir / "cost_sweep.md"
    report.write_text(md, encoding="utf-8")
    (out_dir / "cost_sweep.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Transaction-cost robustness sweep over the frozen winners (costs.grid_bps)."
    )
    p.add_argument(
        "--root",
        default="outputs/campaign/test",
        help="Campaign frozen-winner TEST archive root (default: outputs/campaign/test).",
    )
    p.add_argument(
        "--grid-bps",
        default=None,
        help="Comma list overriding config/environment.yaml costs.grid_bps (e.g. 0,5,10,25,50).",
    )
    p.add_argument(
        "--cvar-alpha", type=float, default=0.05, help="CVaR tail level (default 0.05)."
    )
    p.add_argument(
        "--out-dir", default=None, help="Report directory (default: alongside --root)."
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.grid_bps is not None:
        grid = [float(x) for x in args.grid_bps.split(",") if x.strip() != ""]
    else:
        from src.utils.config import load_config

        grid = [float(b) for b in load_config("environment")["costs"]["grid_bps"]]

    out_dir = args.out_dir or str(Path(args.root).parent / "cost_sweep")
    result = run_cost_sweep(args.root, grid, cvar_alpha=args.cvar_alpha)
    report = write_report(result, out_dir)

    print(
        f"[cost_sweep] re-priced {len(result['curve']['arms'])} arm(s) over "
        f"{len(grid)} cost level(s) from {result['n_records']} records -> {report}"
    )
    for row in result["winner_by_cost"]:
        w = row["winner"] if row["winner"] is not None else "—"
        print(f"  {row['bps']:>5g} bps: winner={w:>14}  Sharpe={row['sharpe']:.4f}")
    if result["curve"].get("skipped"):
        for arm, reason in result["curve"]["skipped"].items():
            print(f"  skipped {arm}: {reason}")


if __name__ == "__main__":
    main()
