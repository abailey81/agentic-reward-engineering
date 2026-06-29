"""PopArt scale-dependence ablation — re-evaluate the FROZEN winners with ``popart=False`` (T2.4).

Why this exists (the scale-dependence robustness check)
-------------------------------------------------------
``src/agents/popart.py`` clamps the running scale ``sigma >= 1.0`` (``min_scale=1.0``), so the PopArt
wrapper is the EXACT IDENTITY for a unit-scale reward but a ``>1`` divisor for a large-magnitude one.
Because SAC's entropy temperature is auto-tuned (``ent_coef="auto"``) against the reward the critic
sees (``raw / sigma``), two H2 arms whose authored rewards differ in *magnitude* are regularised against
different *normalised* scales — a LATENT cross-arm difference inside an ostensibly fixed-agent design
(the agent is meant to differ only by seed). Two defences ship together:

  * **AUDIT (the cheap, always-on one).** Every training path now logs the realised per-candidate scale
    (``sigma_max``/``sigma_last``) into the candidate AND test records
    (``PopArtRewardScaler.realized_scale_stats``); ``sigma_max == 1.0`` across all arms means the wrapper
    was the identity everywhere and there is no scale-driven regularisation confound to worry about.
  * **THIS ablation (the falsifier).** Re-evaluate the FROZEN winners with the wrapper DISABLED
    (``popart=False``) at ONE seed and check the **H2 ordering is unchanged** — i.e. the
    ``distributional`` arm still ranks above ``scalar`` on BOTH the test Sharpe and the test CVaR-5%.
    If the ordering flips when PopArt is removed, the headline could be an artefact of differential
    entropy regularisation; if it holds, the headline is robust to the wrapper.

This RETRAINS (unlike ``scripts/cost_sweep.py``, which re-prices analytically): PopArt is a training-time
transform, so its effect on the learned policy cannot be recovered without re-running the fixed SAC. It is
therefore a GPU job — gated on the user. This module BUILDS the harness and exposes a ``--smoke`` path
(tiny synthetic panel, a handful of steps) for a torch-free-ish sanity run; it deliberately does NOT run
the full 1-seed study (that is the user's GPU call). Pre-registration note: this is a post-hoc ROBUSTNESS
diagnostic on the frozen winners, not a headline inference path — it touches the sealed test leg only
through the SAME ``EnvBundle`` seal the campaign uses (a 3-window bundle built once), and reports an
ORDERING, not a new significance test.

The pure reducers (``ordering_from_returns`` / ``ordering_preserved`` / ``ablation_markdown``) are
torch-free and unit-tested; the heavy driver (``run_popart_ablation``) takes injectable
``trainer`` / ``env_builder`` / ``reward_builder`` / ``winner_loader`` collaborators so the orchestration
is exercised with fakes (no GPU) in ``tests/test_popart_ablation.py``.
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
    "H2_PAIR",
    "ordering_from_returns",
    "ordering_preserved",
    "ablation_markdown",
    "evaluate_arm_no_popart",
    "run_popart_ablation",
    "write_report",
]

_LOG = logging.getLogger(__name__)

#: The pre-registered arms this ablation reports (PREREGISTRATION §3); only those with a usable frozen
#: winner are re-evaluated, the rest are skipped (never fabricated). Mirrors ``cost_sweep.ARMS`` minus
#: the search baselines (random/bayes carry non-executable coeff stubs, not a re-runnable reward here).
ARMS: tuple[str, ...] = ("distributional", "scalar", "placebo", "scalar_cvar5")

#: The H2 ordering whose invariance under ``popart=False`` is the headline claim of this ablation:
#: ``distributional`` is predicted BETTER than ``scalar`` (higher Sharpe AND higher — less negative —
#: CVaR-5%). Kept as the canonical pair so a reader (and the test) sees exactly what "ordering preserved"
#: means. The full H2 conjunction also tests placebo / scalar_cvar5; this diagnostic reports those deltas
#: too, but the load-bearing ordering check is the primary distributional>scalar contrast.
H2_PAIR: tuple[str, str] = ("distributional", "scalar")


# --------------------------------------------------------------------------- #
# Pure reducers (torch-free; unit-tested)                                      #
# --------------------------------------------------------------------------- #
def ordering_from_returns(
    arm_returns: dict[str, np.ndarray | None], *, cvar_alpha: float = 0.05
) -> dict[str, Any]:
    """Reduce per-arm realized TEST return vectors to the per-arm (Sharpe, CVaR) ordering.

    Parameters
    ----------
    arm_returns : dict[str, np.ndarray | None]
        ``{arm: per_step_test_returns}`` (``None`` / empty for a skipped arm). The vectors are the
        realized NET per-step returns of the frozen winner trained WITHOUT PopArt at the ablation seed.
    cvar_alpha : float
        CVaR tail level for the reported ordering (default headline ``0.05``).

    Returns
    -------
    dict
        ``{"cvar_alpha": ..., "metrics": {arm: {"sharpe", "cvar", "n_steps"}},
        "sharpe_rank": [arm, ...], "cvar_rank": [arm, ...]}`` — the arms ranked best→worst by Sharpe
        and (separately) by CVaR (higher = less-negative tail = better). Arms with no returns are
        omitted from both the metrics and the ranks (never fabricated).
    """
    from src.inference.bootstrap import cvar, sharpe_ratio

    metrics: dict[str, dict[str, float]] = {}
    for arm, ret in arm_returns.items():
        if ret is None:
            continue
        arr = np.asarray(ret, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        metrics[arm] = {
            "sharpe": float(sharpe_ratio(arr)),
            "cvar": float(cvar(arr, cvar_alpha)),
            "n_steps": float(arr.size),
        }
    sharpe_rank = sorted(metrics, key=lambda a: metrics[a]["sharpe"], reverse=True)
    cvar_rank = sorted(metrics, key=lambda a: metrics[a]["cvar"], reverse=True)
    return {
        "cvar_alpha": float(cvar_alpha),
        "metrics": metrics,
        "sharpe_rank": sharpe_rank,
        "cvar_rank": cvar_rank,
    }


def ordering_preserved(
    ablation: dict[str, Any],
    *,
    pair: tuple[str, str] = H2_PAIR,
) -> dict[str, Any]:
    """Does the H2 ordering (``pair[0]`` better than ``pair[1]``) survive removing PopArt?

    Checks the primary H2 contrast on BOTH axes: ``pair[0]`` (distributional) must have the higher
    Sharpe AND the higher (less-negative) CVaR than ``pair[1]`` (scalar). Returns the per-axis booleans,
    the signed deltas (``a − b``), and an overall ``preserved`` flag (both axes hold). When either arm is
    missing its metric, the corresponding axis is ``None`` (``preserved`` is then ``False`` — the ablation
    could not confirm the ordering, which is the conservative read).
    """
    a, b = pair
    m = ablation.get("metrics", {})
    out: dict[str, Any] = {"pair": [a, b]}
    if a in m and b in m:
        d_sharpe = float(m[a]["sharpe"] - m[b]["sharpe"])
        d_cvar = float(m[a]["cvar"] - m[b]["cvar"])
        out["sharpe_delta"] = d_sharpe
        out["cvar_delta"] = d_cvar
        out["sharpe_preserved"] = bool(d_sharpe > 0.0)
        out["cvar_preserved"] = bool(d_cvar > 0.0)
        out["preserved"] = bool(d_sharpe > 0.0 and d_cvar > 0.0)
    else:
        out["sharpe_delta"] = None
        out["cvar_delta"] = None
        out["sharpe_preserved"] = None
        out["cvar_preserved"] = None
        out["preserved"] = False
        out["missing"] = [x for x in (a, b) if x not in m]
    return out


def ablation_markdown(result: dict[str, Any]) -> str:
    """Render the PopArt ablation report (per-arm Sharpe/CVaR without PopArt + the ordering verdict)."""
    ablation = result["ablation"]
    verdict = result["ordering"]
    a = ablation["cvar_alpha"]
    seed = result.get("seed")
    metrics = ablation["metrics"]

    lines = [
        "# PopArt scale-dependence ablation (`popart=False`; T2.4)",
        "",
        "Frozen winners RE-TRAINED with the PopArt value-target scaler DISABLED, at a SINGLE seed "
        f"(seed={seed}), then evaluated on the sealed test leg once per arm. The question: is the H2 "
        "ordering (`distributional` > `scalar`) an effect of the reward, or an artefact of the "
        "scale-dependent entropy regularisation PopArt induces (`ent_coef=auto` adapts to `raw/sigma`)? "
        "If the ordering holds with PopArt removed, the headline is robust to the wrapper.",
        "",
        "## H2 ordering verdict (primary contrast: distributional vs scalar)",
        "",
    ]
    pair = verdict.get("pair", list(H2_PAIR))
    if verdict.get("sharpe_delta") is None:
        miss = ", ".join(f"`{x}`" for x in verdict.get("missing", []))
        lines.append(
            f"**INCONCLUSIVE** — missing a frozen-winner test record for {miss}; the "
            f"`{pair[0]}` > `{pair[1]}` ordering could not be evaluated without PopArt."
        )
    else:
        tick = lambda ok: "preserved" if ok else "FLIPPED"  # noqa: E731
        lines.append(
            f"- Sharpe: `{pair[0]} − {pair[1]}` = **{verdict['sharpe_delta']:+.4f}** "
            f"({tick(verdict['sharpe_preserved'])})"
        )
        lines.append(
            f"- CVaR-{a:g}: `{pair[0]} − {pair[1]}` = **{verdict['cvar_delta']:+.5f}** "
            f"({tick(verdict['cvar_preserved'])})"
        )
        lines.append("")
        lines.append(
            f"**Overall: H2 ordering {'PRESERVED' if verdict['preserved'] else 'NOT preserved'} "
            "without PopArt.**"
        )
    lines.append("")

    # Per-arm Sharpe / CVaR table (popart=False).
    lines.append("## Per-arm metrics without PopArt")
    lines.append("")
    lines.append(f"| arm | test Sharpe | test CVaR-{a:g} | n steps |")
    lines.append("|---|---|---|---|")
    for arm in ARMS:
        if arm in metrics:
            d = metrics[arm]
            lines.append(f"| {arm} | {d['sharpe']:.4f} | {d['cvar']:.5f} | {int(d['n_steps'])} |")
        else:
            lines.append(f"| {arm} | — | — | — |")
    lines.append("")
    lines.append(f"Sharpe ranking (best→worst): {' > '.join(ablation['sharpe_rank']) or '—'}")
    lines.append("")
    lines.append(f"CVaR-{a:g} ranking (best→worst): {' > '.join(ablation['cvar_rank']) or '—'}")
    lines.append("")

    if result.get("skipped"):
        lines.append("## Skipped arms (no usable frozen winner)")
        lines.append("")
        for arm, reason in result["skipped"].items():
            lines.append(f"- `{arm}`: {reason}")
        lines.append("")

    # Cross-arm realised-scale audit (sigma_max) — the cheap always-on companion to the ablation.
    if result.get("sigma_audit"):
        lines.append("## Realised PopArt scale at the ablation seed (audit)")
        lines.append("")
        lines.append(
            "The `popart=True` re-train's realised `sigma_max` per arm (1.0 ⇒ the wrapper was the "
            "identity for that reward ⇒ no scale-driven entropy-regularisation difference):"
        )
        lines.append("")
        lines.append("| arm | sigma_max | sigma_last |")
        lines.append("|---|---|---|")
        for arm in ARMS:
            s = result["sigma_audit"].get(arm)
            if s is not None:
                lines.append(f"| {arm} | {s.get('sigma_max', float('nan')):.4g} "
                             f"| {s.get('sigma_last', float('nan')):.4g} |")
        lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The heavy per-arm evaluation (retrain with popart=False; inject for tests)    #
# --------------------------------------------------------------------------- #
def evaluate_arm_no_popart(
    *,
    winner: dict[str, Any],
    arm: str,
    seed: int,
    panel: Any,
    cfg: Any,
    agent_cfg: dict[str, Any],
    train_window: tuple[int, int],
    val_window: tuple[int, int],
    test_window: tuple[int, int],
    embargo: int,
    lookback: int,
    trainer: Callable[[Any, int], Callable[[Any], Any]],
    env_builder: Callable[..., Callable[[Any], Any]],
    reward_builder: Callable[[str | None], Any],
    popart: bool = False,
) -> dict[str, Any]:
    """Retrain ONE frozen winner (PopArt off by default) and roll the sealed test leg once.

    Mirrors the campaign's ``evaluate_winner_on_test`` per-seed body (reseed the full stack, 3-window
    bundle = the ONLY place a test window exists, train the fixed SAC, touch the test leg EXACTLY once)
    but forces ``popart=popart`` in the agent config so the ablation removes the scaler. Returns
    ``{"ok", "arm", "test_returns", "popart_scale"}`` (``ok=False`` + ``error`` on a failure so one arm
    cannot sink the sweep).
    """
    from src.utils.seeding import set_global_seed

    out: dict[str, Any] = {"ok": False, "arm": arm}
    try:
        set_global_seed(int(seed), deterministic_torch=True)
        reward_fn = reward_builder(winner.get("reward_source"))
        builder = env_builder(
            panel,
            cfg,
            train_window,
            val_window,
            test_window=test_window,  # the ONLY 3-window bundle (seal intact)
            embargo=int(embargo),
            lookback=int(lookback),
        )
        bundle = builder(reward_fn)
        # Force the PopArt gate for the ablation; everything else is the matched campaign agent config.
        ablation_cfg = {**agent_cfg, "popart": bool(popart)}
        train_fn = trainer(ablation_cfg, int(seed))
        policy = train_fn(bundle.train_env())
        test_returns = np.asarray(bundle.test_returns(policy), dtype=float)
        out.update(
            ok=True,
            test_returns=test_returns,
            popart_scale=getattr(policy, "popart_scale", None),
        )
    except Exception as exc:  # noqa: BLE001 — a failed arm must not crash the sweep
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


# --------------------------------------------------------------------------- #
# Driver                                                                        #
# --------------------------------------------------------------------------- #
def _default_winner_loader(frozen_root: str | Path) -> dict[str, dict[str, Any]]:
    """Load each arm's FROZEN winner record from ``<frozen_root>`` via the canonical loader (audit C-1).

    The campaign writes one frozen record per arm under ``<frozen_root>/<arm>-winner/record.json``
    (``run_campaign.freeze_winner``). Returns ``{arm: record}`` for the records that carry an executable
    ``reward_source`` (a baseline / coeff-stub winner with no source is skipped — it cannot be re-run
    here). Reads ONLY through ``src.io.results.load_all``.
    """
    from src.io.results import load_all

    records = load_all(frozen_root)
    winners: dict[str, dict[str, Any]] = {}
    for rec in records:
        arm = str(rec.get("arm", ""))
        if arm in ARMS and str(rec.get("reward_source", "")).strip():
            winners[arm] = rec
    return winners


def run_popart_ablation(
    *,
    frozen_root: str | Path,
    seed: int,
    panel: Any,
    cfg: Any,
    agent_cfg: dict[str, Any],
    train_window: tuple[int, int],
    val_window: tuple[int, int],
    test_window: tuple[int, int],
    embargo: int,
    lookback: int,
    trainer: Callable[[Any, int], Callable[[Any], Any]],
    env_builder: Callable[..., Callable[[Any], Any]],
    reward_builder: Callable[[str | None], Any] | None = None,
    winner_loader: Callable[[str | Path], dict[str, dict[str, Any]]] | None = None,
    arms: tuple[str, ...] | list[str] = ARMS,
    cvar_alpha: float = 0.05,
    with_popart_audit: bool = True,
) -> dict[str, Any]:
    """Re-evaluate every frozen winner with ``popart=False`` at ONE seed; report the H2 ordering.

    For each arm with a usable frozen winner: retrain the fixed SAC with PopArt DISABLED
    (:func:`evaluate_arm_no_popart`), roll the sealed test leg once, and reduce to the per-arm
    (Sharpe, CVaR) ordering (:func:`ordering_from_returns`). The primary verdict is whether the
    ``distributional`` > ``scalar`` ordering is preserved (:func:`ordering_preserved`).

    When ``with_popart_audit`` (default), each arm is ALSO retrained ONCE with ``popart=True`` at the
    same seed to record the realised ``sigma_max`` — the cross-arm scale distribution that makes the
    "fixed-agent" claim falsifiable (``sigma_max == 1`` ⇒ the wrapper was the identity for that arm).

    Dependency injection (``trainer`` / ``env_builder`` / ``reward_builder`` / ``winner_loader``) makes
    the orchestration unit-testable with fakes (no torch). Returns the full result dict.
    """
    if reward_builder is None:
        import sys as _sys

        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        from run_campaign import _reinstantiate_frozen_winner  # type: ignore[import-not-found]

        reward_builder = _reinstantiate_frozen_winner
    if winner_loader is None:
        winner_loader = _default_winner_loader

    winners = winner_loader(frozen_root)

    no_popart_returns: dict[str, np.ndarray | None] = {}
    sigma_audit: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}

    for arm in arms:
        winner = winners.get(arm)
        if winner is None:
            skipped[arm] = "no frozen winner with an executable reward_source"
            continue
        res = evaluate_arm_no_popart(
            winner=winner,
            arm=arm,
            seed=seed,
            panel=panel,
            cfg=cfg,
            agent_cfg=agent_cfg,
            train_window=train_window,
            val_window=val_window,
            test_window=test_window,
            embargo=embargo,
            lookback=lookback,
            trainer=trainer,
            env_builder=env_builder,
            reward_builder=reward_builder,
            popart=False,
        )
        if res.get("ok"):
            no_popart_returns[arm] = res["test_returns"]
        else:
            no_popart_returns[arm] = None
            skipped[arm] = f"retrain failed: {res.get('error')}"

        if with_popart_audit:
            ref = evaluate_arm_no_popart(
                winner=winner,
                arm=arm,
                seed=seed,
                panel=panel,
                cfg=cfg,
                agent_cfg=agent_cfg,
                train_window=train_window,
                val_window=val_window,
                test_window=test_window,
                embargo=embargo,
                lookback=lookback,
                trainer=trainer,
                env_builder=env_builder,
                reward_builder=reward_builder,
                popart=True,
            )
            if ref.get("ok") and ref.get("popart_scale") is not None:
                sigma_audit[arm] = dict(ref["popart_scale"])

    ablation = ordering_from_returns(no_popart_returns, cvar_alpha=cvar_alpha)
    verdict = ordering_preserved(ablation)

    return {
        "frozen_root": str(frozen_root),
        "seed": int(seed),
        "n_winners": len(winners),
        "ablation": ablation,
        "ordering": verdict,
        "sigma_audit": sigma_audit,
        "skipped": skipped,
    }


def write_report(result: dict[str, Any], out_dir: str | Path) -> Path:
    """Write ``popart_ablation.md`` + ``popart_ablation.json`` into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = ablation_markdown(result)
    report = out_dir / "popart_ablation.md"
    report.write_text(md, encoding="utf-8")
    (out_dir / "popart_ablation.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    return report


# --------------------------------------------------------------------------- #
# Smoke / full wiring (real panel + windows; the production-shaped run)         #
# --------------------------------------------------------------------------- #
def _resolve_inputs(*, synthetic: bool, train_steps: int) -> dict[str, Any]:
    """Resolve (panel, cfg, agent_cfg, windows, lookback, embargo) exactly like ``run_campaign``.

    Reuses ``run_campaign.resolve_windows`` + the env/inference config + the prototype agent block so the
    ablation trains the SAME fixed agent on the SAME windows as the headline — only ``train_steps`` (for a
    tractable smoke) and the ``popart`` gate differ. ``synthetic=True`` swaps in the 600-day synthetic
    panel (no gold needed; the smoke path).
    """
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_campaign import resolve_windows  # type: ignore[import-not-found]
    from run_prototype import _agent_cfg  # type: ignore[import-not-found]

    from src.utils.config import cfg_get, load_config

    env_cfg = load_config("environment")
    inf_cfg = load_config("inference")
    lookback = int(env_cfg["state"]["lookback_days"])
    embargo = int(cfg_get(load_config("data"), "embargo_days", 21))

    if synthetic:
        from src.data.synthetic import make_synthetic_panel

        # The 600-day synthetic panel ends in 2006, so it cannot host the real 2005-2026 calendar splits
        # resolve_windows derives (they would clamp to a degenerate end-of-panel window). The SMOKE run
        # only needs three short, PURGE-respecting windows that prove the harness wires up — use the same
        # layout tests/test_run_campaign.py uses: train [lb,250) | gap>=lookback | val [310,360) | test [420,T).
        panel = make_synthetic_panel(n_assets=30, n_days=600, seed=0)
        train_window = (lookback, 250)
        val_window = (250 + lookback, 360)
        test_window = (360 + lookback, int(panel.T))
    else:
        from src.data.loaders import load_gold_panel

        ev = cfg_get(cfg_get(inf_cfg, "splits", {}), "evaluation", {})
        span = cfg_get(ev, "span", ["2020-01-01", "2026-06-30"])
        panel = load_gold_panel(phase="development", end=str(span[1])).panel
        splits = cfg_get(inf_cfg, "splits", {})
        train_window, val_window, test_window = resolve_windows(
            panel, lookback, splits, embargo=embargo
        )
    agent_cfg = _agent_cfg(load_config("prototype"), int(train_steps))
    return {
        "panel": panel,
        "cfg": env_cfg,
        "agent_cfg": agent_cfg,
        "train_window": train_window,
        "val_window": val_window,
        "test_window": test_window,
        "lookback": lookback,
        "embargo": embargo,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PopArt scale-dependence ablation: re-evaluate the frozen winners with popart=False (T2.4)."
    )
    p.add_argument(
        "--frozen-root",
        default="outputs/campaign/frozen",
        help="Campaign FROZEN-winner archive root (default: outputs/campaign/frozen).",
    )
    p.add_argument("--seed", type=int, default=0, help="The single ablation seed (default 0).")
    p.add_argument(
        "--train-steps",
        type=int,
        default=None,
        help="Override train_steps_per_candidate (default: the campaign budget; --smoke forces a tiny one).",
    )
    p.add_argument(
        "--out-dir", default=None, help="Report directory (default: alongside --frozen-root)."
    )
    p.add_argument(
        "--no-sigma-audit",
        action="store_true",
        help="Skip the companion popart=True re-train that records the realised sigma_max per arm.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny synthetic-panel + few-step sanity run (no gold, ~no GPU); proves the harness wires up.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.smoke:
        # The harness sanity check: synthetic panel, a handful of steps, the stub-frozen winners below.
        # Proves the load->retrain(popart off/on)->roll->order pipeline wires up end-to-end without gold
        # or a long GPU run. It does NOT certify the science (that is the user's real 1-seed GPU run).
        train_steps = int(args.train_steps or 200)
        synthetic = True
    else:
        train_steps = int(args.train_steps) if args.train_steps is not None else 50000
        synthetic = False

    from src.orchestration.parallel import _FIXTURE  # noqa: F401  (kept for parity / future reward checks)

    from run_campaign import _reinstantiate_frozen_winner, make_agent_trainer_factory  # type: ignore[import-not-found]
    from src.env.runner import make_env_builder

    inputs = _resolve_inputs(synthetic=synthetic, train_steps=train_steps)
    trainer = make_agent_trainer_factory()

    if args.smoke:
        # In smoke mode synthesise two minimal frozen winners (a tiny mean reward) so the pipeline has
        # something executable to re-train WITHOUT needing a real campaign archive on disk.
        import hashlib

        _src = (
            "def reward(weights, returns, prev_weights, port_ret, info):\n"
            "    return float(port_ret), {'r': float(port_ret)}, {}\n"
        )
        _h = hashlib.sha256(_src.encode("utf-8")).hexdigest()
        fake_winners = {
            arm: {
                "arm": arm,
                "reward_source": _src,
                "reward_source_hash": _h,
                "candidate_id": f"{arm}-winner",
                "metrics": {"val_fitness": 0.0},
            }
            for arm in (H2_PAIR)  # distributional + scalar: enough to exercise the ordering verdict
        }
        winner_loader: Callable[[str | Path], dict[str, dict[str, Any]]] = lambda _root: fake_winners  # noqa: E731
    else:
        winner_loader = _default_winner_loader

    result = run_popart_ablation(
        frozen_root=args.frozen_root,
        seed=int(args.seed),
        panel=inputs["panel"],
        cfg=inputs["cfg"],
        agent_cfg=inputs["agent_cfg"],
        train_window=inputs["train_window"],
        val_window=inputs["val_window"],
        test_window=inputs["test_window"],
        embargo=inputs["embargo"],
        lookback=inputs["lookback"],
        trainer=trainer,
        env_builder=make_env_builder,
        reward_builder=_reinstantiate_frozen_winner,
        winner_loader=winner_loader,
        with_popart_audit=not args.no_sigma_audit,
    )

    out_dir = args.out_dir or str(Path(args.frozen_root).parent / "popart_ablation")
    report = write_report(result, out_dir)

    verdict = result["ordering"]
    print(
        f"[popart_ablation] seed={result['seed']} re-evaluated "
        f"{len(result['ablation']['metrics'])} arm(s) without PopArt from "
        f"{result['n_winners']} frozen winner(s) -> {report}"
    )
    if verdict.get("sharpe_delta") is not None:
        print(
            f"  H2 ordering (distributional minus scalar): Sharpe {verdict['sharpe_delta']:+.4f}, "
            f"CVaR {verdict['cvar_delta']:+.5f} -> "
            f"{'PRESERVED' if verdict['preserved'] else 'NOT preserved'} without PopArt"
        )
    else:
        print("  H2 ordering: INCONCLUSIVE (missing a frozen winner for the contrast)")
    for arm, reason in result.get("skipped", {}).items():
        print(f"  skipped {arm}: {reason}")


if __name__ == "__main__":
    main()
