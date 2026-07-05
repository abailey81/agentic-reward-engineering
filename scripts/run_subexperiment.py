"""Report-only mechanism sub-experiment driver (ADR-039; PREREGISTRATION §2a) — DEFAULT-OFF, DISJOINT.

Two registered sub-experiments that the confirmatory campaign does NOT run (they need a SEPARATE
re-authoring pass producing PAIRED reward sources). Both are report-only, never gate the frozen m=6
headline, and are produced here by a thin driver mirroring ``scripts/run_prototype.py``'s
load-panel-once + ``run_loop`` pattern:

  * ``--mode named``   — the NAMED-vs-BLINDED structural A/B. For each seed the SAME anonymised panel +
    base prompts are authored TWICE: condition ``named`` appends a DATA-PROVENANCE paragraph to the
    ``initial`` prompt (the real train-window calendar span, the real RIC tickers, the regime
    composition), condition ``blinded`` leaves the base prompt UNCHANGED. Each record is stamped with a
    top-level ``label`` in ``{"named","blinded"}`` so ``analyze_campaign`` pairs them by
    ``(generation, candidate_id)`` and runs ``named_vs_blinded_structural``.

  * ``--mode legible`` — the numeracy-bottleneck legible-format responsiveness differential. Condition
    ``legible`` renders the distributional tail block in basis-points + decile form
    (``legible_render=True``); condition ``raw`` renders the close-small-float vector
    (``legible_render=False``). Each record is stamped ``condition`` in ``{"legible","raw"}`` so
    ``analyze_campaign`` runs ``legible_format_responsiveness_differential`` over the two conditions.

Both conditions train the SAME fixed SB3-SAC agent (arm = ``"distributional"``) on the SAME anonymised
panel + windows the campaign uses, so the per-candidate arrays are byte-identical to the blinded leg —
the ONLY thing that differs is the feedback RENDERING (legible) or the appended provenance text (named),
exactly as the analysis legs require.

Pairing robustness (FLAG B). The named leg's ``named_vs_blinded_structural`` REQUIRES
``len(named) == len(blinded)`` with index-alignment by ``(generation, candidate_id)``. A sandbox
rejection in only ONE condition would desync the two lists. We resolve this on the DRIVER side, honestly
and at zero extra spend: after authoring both conditions we keep only the ``candidate_id``s ACCEPTED IN
BOTH conditions (the intersection) and DELETE the unpaired records from the archive before analysis reads
them (``_prune_to_intersection``). ``analyze()`` already sorts each leg by ``_by_generation``, so the two
pruned, equal-length lists align element-for-element. This is the no-extra-spend option: it discards a
handful of unpaired candidates rather than re-authoring to backfill them.

Usage
-----
    python scripts/run_subexperiment.py --mode named   --n-seeds 150 --candidates 5 --output-dir runs/named
    python scripts/run_subexperiment.py --mode legible --seeds 0..149 --candidates 5 --output-dir runs/legible
    python scripts/run_subexperiment.py --mode named   --n-seeds 4   --candidates 3 --synthetic   # smoke

The keyless stub designer (Pass A / ``--synthetic`` or no key) lets the WHOLE pipeline run without an API
key; the real Opus author (config/subexperiment.yaml: llm) is used otherwise. ~150-200 seeds give >0.95
power for the named A/B (src/inference/contamination.py:234) but the count is configurable (it costs Opus
calls), read from config/subexperiment.yaml unless overridden on the CLI.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.utils.config import cfg_get, load_config

# Both sub-experiments author under the tail-FED arm so the legible/named contrast is over the same
# six-line distributional block; the analysis legs gate on `_was_fed_tail`, which only this arm passes.
_ARM = "distributional"
_NAMED_CONDITIONS = ("named", "blinded")
_LEGIBLE_CONDITIONS = ("legible", "raw")


# --------------------------------------------------------------------------- #
# Data-provenance paragraph (NAMED condition only)                            #
# --------------------------------------------------------------------------- #
def _provenance_paragraph(panel: Any, ric_by_id: dict[int, str] | None, regime_names: list[str]) -> str:
    """The identity-revealing paragraph appended to the NAMED condition's initial prompt.

    Reveals (a) the real train-window calendar span from ``panel.dates[0]..[-1]``; (b) the real RIC
    tickers ``ric_by_id`` maps; (c) the regime composition/names. The BLINDED condition omits this — the
    A/B asks whether revealing this provenance changes the reward CODE the LLM writes. This text is
    PROVENANCE for the LLM author only; it is never handed to a reward function (the reward still sees
    anonymised numpy arrays only — contamination defence)."""
    import numpy as np

    dates = np.asarray(panel.dates)
    span = f"{str(dates[0])[:10]} to {str(dates[-1])[:10]}" if dates.size else "unknown"
    tickers = ", ".join(str(r) for r in (ric_by_id or {}).values()) or "unknown"
    regimes = ", ".join(regime_names) or "unknown"
    return (
        "\n\nDATA PROVENANCE (for context; the reward you write still receives ANONYMISED numpy arrays "
        "only, no tickers or dates):\n"
        f"- The training panel spans {span} ({int(getattr(panel, 'T', 0))} trading sessions).\n"
        f"- The assets are the real tickers: {tickers}.\n"
        f"- The market regimes present (by VIX): {regimes}."
    )


def _regime_names(panel: Any) -> list[str]:
    """Ordered distinct regime names present in ``panel`` (calm/normal/stress), for the provenance text."""
    try:
        import numpy as np

        from src.regimes.definition import label_regimes

        labels = np.asarray(label_regimes(panel, load_config("regimes")))
        names = {0: "calm", 1: "normal", 2: "stress"}
        present = sorted(set(int(x) for x in labels.tolist()))
        return [names.get(i, f"regime_{i}") for i in present]
    except Exception:  # noqa: BLE001 - provenance text is best-effort; an absent regimes config => no names
        return []


# --------------------------------------------------------------------------- #
# Pairing robustness (FLAG B): keep only candidate_ids accepted in BOTH conditions
# --------------------------------------------------------------------------- #
def _prune_to_intersection(output_dir: Path, conditions: tuple[str, str]) -> int:
    """Delete unpaired candidate records so the two conditions are equal-length + (gen, cid)-alignable.

    Walks the archive, groups accepted records by their top-level discriminator (``label``/``condition``),
    and removes any ``candidate_id`` not accepted under BOTH conditions. Returns the number of paired
    candidate_ids retained. Idempotent (a second call is a no-op). This is the honest, no-extra-spend
    resolution of the sandbox-rejection desync (see module docstring, FLAG B).
    """
    import shutil

    # Map candidate_id -> {condition: run_dir} over every record.json under output_dir.
    by_cid: dict[str, dict[str, Path]] = {}
    discrim = "label" if conditions == _NAMED_CONDITIONS else "condition"
    for rec_path in output_dir.rglob("record.json"):
        try:
            rec = json.loads(rec_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        cond = rec.get(discrim)
        cid = rec.get("candidate_id")
        if cond in conditions and cid is not None:
            by_cid.setdefault(str(cid), {})[str(cond)] = rec_path.parent

    paired = {cid: dirs for cid, dirs in by_cid.items() if set(dirs) == set(conditions)}
    # Delete the run dirs of any candidate_id NOT accepted under both conditions.
    for cid, dirs in by_cid.items():
        if cid not in paired:
            for run_dir in dirs.values():
                shutil.rmtree(run_dir, ignore_errors=True)
    return len(paired)


# --------------------------------------------------------------------------- #
# Collaborators (real by default; injected fakes in tests)                    #
# --------------------------------------------------------------------------- #
def _load_real_collaborators(synthetic: bool, sub_cfg: Any, seed: int):
    """Build the real (panel, env_builder, agent_trainer, measurement, prompts, ric_by_id, regime_names).

    Mirrors ``run_prototype.run_arm``'s load-panel-once + builders pattern: the campaign's SAME
    anonymised panel + windows, the fixed SB3-SAC trainer, and the empirical+EVT ``ReturnDistribution``.
    Imported lazily so the test path (which injects fakes) needs no torch.
    """
    import numpy as np  # noqa: F401 - imported for parity with run_prototype's seeded path

    from src.agents.trainer import make_agent_trainer
    from src.env.runner import make_env_builder
    from src.feedback.measurement import ReturnDistribution
    from src.llm.prompts import build_prompt_set
    from src.utils.preload import preload
    from src.utils.seeding import set_global_seed

    # Reuse run_prototype's panel loader (the campaign's anonymised panel + purged windows).
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_prototype import _load_panel_and_windows  # noqa: E402

    preload(strict=True)  # pyarrow BEFORE torch (ABI conflict); H2 fail-loud in production
    set_global_seed(seed, deterministic_torch=True)
    env_cfg = load_config("environment")
    lookback = int(env_cfg["state"]["lookback_days"])
    data_cfg = cfg_get(sub_cfg, "data", {})

    ric_by_id: dict[int, str] | None = None
    if synthetic:
        from src.data.synthetic import make_synthetic_panel

        panel = make_synthetic_panel(n_assets=30, n_days=600, seed=0)
        train_window, val_window = (lookback, 400), (400, panel.T)
    else:
        from src.data.loaders import load_gold_panel

        res = load_gold_panel(
            phase=str(cfg_get(data_cfg, "phase", "development")),
            # Split-C val end (R73). The old fallback was the stale pre-Split-C 2017-12-31: a dropped
            # config key would have silently reverted this leg to the superseded window (2026-07-05).
            end=str(cfg_get(data_cfg, "val_end", "2019-12-31")),
            on_missing=str(cfg_get(data_cfg, "on_missing", "liquidate_to_cash")),
        )
        ric_by_id = res.ric_by_id
        panel, train_window, val_window = _load_panel_and_windows(False, data_cfg, lookback)

    # Thread the R18 purge guard exactly like run_prototype.run_arm / parallel.train_candidate do —
    # the sub-experiment path previously built its env with purge=0 (guard unarmed; 2026-07-05).
    env_builder = make_env_builder(
        panel, env_cfg, train_window, val_window,
        embargo=int(cfg_get(data_cfg, "embargo_days", 21)), lookback=int(lookback),
    )
    a = cfg_get(sub_cfg, "agent", {})
    train_steps = int(cfg_get(a, "train_steps_per_candidate", 25000))
    agent_cfg = {
        "train_steps_per_candidate": train_steps,
        "buffer_size": int(cfg_get(a, "buffer_size", train_steps)),
        "batch_size": int(cfg_get(a, "batch_size", 256)),
        "learning_rate": float(cfg_get(a, "learning_rate", 3e-4)),
        "gamma": float(cfg_get(a, "gamma", 0.99)),
    }
    agent_trainer = make_agent_trainer(agent_cfg, seed)
    prompts = build_prompt_set(env_cfg, panel.N)
    return {
        "panel": panel,
        "env_builder": env_builder,
        "agent_trainer": agent_trainer,
        "measurement": ReturnDistribution,
        "prompts": {"system": prompts.system, "initial": prompts.initial},
        "ric_by_id": ric_by_id,
        "regime_names": _regime_names(panel),
    }


def _make_llm(transport: Any, model_id: str):
    """Wrap a transport in an ``LLMClient`` (the loop calls ``llm.complete``)."""
    from src.llm.client import LLMClient

    return LLMClient({"model": model_id}, transport=transport)


# --------------------------------------------------------------------------- #
# Core driver                                                                 #
# --------------------------------------------------------------------------- #
def run_subexperiment(
    mode: str,
    *,
    seeds: list[int],
    candidates: int,
    output_dir: str | Path,
    synthetic: bool = False,
    model: str | None = None,
    sub_cfg: Any = None,
    collaborators_factory: Any = None,
    transport_factory: Any = None,
    allow_inert_legible: bool = False,
) -> dict[str, Any]:
    """Run the named/legible sub-experiment over ``seeds`` and return a summary.

    For each seed, the fixed SB3-SAC agent is trained ONCE PER CONDITION (generations=1,
    candidates_per_gen=``candidates``) via ``run_loop``, under arm ``"distributional"``. The two
    conditions differ only by the appended provenance text (named) or the feedback rendering (legible);
    each record is stamped with the leg's top-level discriminator (``label``/``condition``).

    Dependency injection (for tests, no torch/real LLM):
      * ``collaborators_factory(seed) -> dict`` supplies panel/env_builder/agent_trainer/measurement/
        prompts/ric_by_id/regime_names (default: the real loaders).
      * ``transport_factory(seed) -> (transport, model_id)`` supplies the reward author (default: the
        keyless stub designer, or the configured Opus author when a key is present).

    Returns ``{"mode", "conditions", "n_seeds", "candidates", "n_paired", "output_dir"}``.
    """
    from src.llm.loop import run_loop
    from src.selection.fitness import held_out_fitness

    if mode not in ("named", "legible"):
        raise ValueError(f"mode must be 'named' or 'legible', got {mode!r}")
    conditions = _NAMED_CONDITIONS if mode == "named" else _LEGIBLE_CONDITIONS
    output_dir = Path(output_dir)
    sub_cfg = sub_cfg if sub_cfg is not None else _safe_load_sub_cfg()

    if collaborators_factory is None:
        def collaborators_factory(seed: int):  # type: ignore[misc]
            return _load_real_collaborators(synthetic, sub_cfg, seed)
    if transport_factory is None:
        transport_factory = _default_transport_factory(synthetic, sub_cfg, model)

    for seed in seeds:
        collab = collaborators_factory(seed)
        for condition in conditions:
            # CRASH-RESUME (2026-07-05): each (condition, seed) cell is write-once — if its full
            # candidate slate is already archived, skip re-authoring (a paid multi-day 150-seed run
            # previously had ZERO crash recovery and re-billed every prior seed on restart).
            if _condition_complete(output_dir / _ARM, condition, seed, candidates):
                continue
            transport, model_id = transport_factory(seed)
            llm = _make_llm(transport, model_id)
            prompts = dict(collab["prompts"])  # copy: the NAMED leg appends to `initial`
            extra_record_fields = {("label" if mode == "named" else "condition"): condition}
            legible_render = mode == "legible" and condition == "legible"
            if mode == "named" and condition == "named":
                prompts["initial"] = prompts["initial"] + _provenance_paragraph(
                    collab["panel"], collab.get("ric_by_id"), collab.get("regime_names", [])
                )
            if mode == "legible":
                # SQ3b REDESIGN (2026-07-05, M16): at generations=1 the reflection turn never fires,
                # so the old `legible_render` flag only re-rendered the ARCHIVED block — the designer
                # never SAW any tail rendering and the raw-vs-legible differential was ~0 BY
                # CONSTRUCTION (a paid ~seeds x conditions x candidates Opus burn for nothing). Both
                # conditions now receive the SAME real, per-seed reference tail CONTENT injected into
                # the initial prompt — rendered raw for `raw`, legibly for `legible` — so the ablation
                # manipulates exactly the ENCODING of identical information, as registered.
                ref_block = _reference_tail_block(collab, seed, legible=(condition == "legible"))
                prompts["initial"] = (
                    prompts["initial"]
                    + "\n\nFor context, a REFERENCE portfolio's summary metrics on this market "
                    "(training period) — use them to inform how the reward should shape risk:\n"
                    + ref_block
                )
                if not allow_inert_legible and "CVaR 5%" not in prompts["initial"]:
                    raise RuntimeError(
                        "legible sub-experiment PROOF-OF-FEED failed: the reference tail block did "
                        "not reach the initial prompt — the run would be inert (M16). Refusing to "
                        "burn paid authoring; pass allow_inert_legible=True only for mocked smokes."
                    )
            loop_cfg = {
                "generations": 1,
                "candidates_per_gen": int(candidates),
                "budget": int(candidates),
                "seed": int(seed),
                "n_trials": int(candidates),
                "model": model_id,
                "run_prefix": f"{condition}-s{seed}",
                "fold": 0,
                "prompts": prompts,
                "env_fingerprint": f"subexp:{mode}:{condition}",
                "extra_record_fields": extra_record_fields,
                "legible_render": legible_render,
                # Opus rejects sampler knobs -> per-candidate diversity comes from prompt variation;
                # the config key existed but was never threaded (2026-07-05 fix), so all K candidates
                # per condition were authored from the IDENTICAL prompt.
                "diversity_prompt_variation": bool(
                    cfg_get(sub_cfg, "diversity_prompt_variation", True)
                ),
            }
            # Each condition writes under <output_dir>/<arm>/<run_id>/ (load_campaign_records walks any depth).
            run_loop(
                _ARM, collab["env_builder"], llm, collab["agent_trainer"],
                collab["measurement"], held_out_fitness, loop_cfg, output_dir / _ARM,
            )

    n_paired = _prune_to_intersection(output_dir, conditions)
    return {
        "mode": mode,
        "conditions": list(conditions),
        "n_seeds": len(seeds),
        "candidates": int(candidates),
        "n_paired": n_paired,
        "output_dir": str(output_dir),
    }


def _safe_load_sub_cfg() -> Any:
    """Load config/subexperiment.yaml, falling back to an empty dict if absent (so tests need no config)."""
    try:
        return load_config("subexperiment")
    except Exception:  # noqa: BLE001 - the driver's defaults stand in for a missing config (smoke/tests)
        return {}


def _condition_complete(arm_root: Path, condition: str, seed: int, candidates: int) -> bool:
    """Crash-resume read path: is the full candidate slate for this (condition, seed) archived?

    ``run_loop`` writes one ``<arm_root>/<run_prefix>-...>/record.json`` per candidate with
    ``run_prefix = f"{condition}-s{seed}"``; the cell is complete when at least ``candidates``
    such records exist. Write-once semantics: a complete cell is never re-authored (2026-07-05 —
    the paid multi-day sub-experiment previously restarted from seed 0 on any crash)."""
    root = Path(arm_root)
    if not root.is_dir():
        return False
    prefix = f"{condition}-s{seed}-"
    n = sum(1 for d in root.iterdir() if d.name.startswith(prefix) and (d / "record.json").is_file())
    return n >= int(candidates)


def _reference_tail_block(collab: dict, seed: int, *, legible: bool) -> str:
    """A REAL per-seed reference feedback block for the SQ3b encoding ablation (2026-07-05 redesign).

    Content: the frozen six-field tail measurement fitted on a seeded moving-block resample of the
    collaborator panel's equal-weight TRAINING-window returns — real market content, deterministic
    per seed, varying across seeds. The SAME dict is rendered raw (``legible=False``) for the ``raw``
    condition and legibly (bps/percent/decile) for the ``legible`` condition, so the two prompts
    differ in ENCODING only. Falls back (with a loud print) to a fixed representative vector when the
    collaborators carry no usable panel/measurement (keyless or mocked smokes)."""
    import numpy as np

    from src.feedback.schema import build_block

    stats: dict | None = None
    panel = (collab or {}).get("panel")
    rd_cls = (collab or {}).get("measurement")
    try:
        if panel is not None and rd_cls is not None:
            rets = np.nanmean(np.asarray(panel.returns, dtype=float), axis=1)
            rets = rets[np.isfinite(rets)]
            n = int(rets.size)
            if n >= 250:
                rng = np.random.default_rng(int(seed))
                blk = 20
                starts = rng.integers(0, n, size=n // blk + 1)
                idx = np.concatenate([(np.arange(blk) + s) % n for s in starts])[:n]
                stats = rd_cls().fit(rets[idx]).tail_stats()
    except Exception as exc:  # noqa: BLE001 — the reference block must never sink the leg
        print(f"[run_subexperiment] reference tail measurement failed ({exc}); using the fallback vector")
        stats = None
    if stats is None:
        # Representative of the real univ5 train window (disclosed fallback for stub collaborators),
        # deterministically JITTERED per seed: cross-seed variation in the fed content is what the
        # SQ3b responsiveness statistic correlates against, so a seed-constant fallback would make
        # the instrument degenerate by construction even on a degraded run.
        # Jitter scale 0.25 (clamped): a narrow jitter can QUANTIZE two seeds' cvar values to the
        # same rendered string (the numeracy thesis biting its own fixture) and degenerate the raw
        # condition's x. NOTE: This said "the RAW rendering prints 3 decimals" until 2026-07-27 (deep
        # review #109) — #98 moved ``schema._fmt`` to ``.4f``, so the raw resolution is now 1e-4, not
        # 1e-3. The 0.25 stands and is now MORE conservative: over the clamp the cvar_05 fallback
        # spans ~0.041 (≈410 quantization steps at 1e-4, vs ≈41 at the old 1e-3), so the degeneracy
        # this scale defends against is an order of magnitude further away than when it was chosen.
        # The renderer change also gave the two conditions EXACT RESOLUTION PARITY — raw 1e-4 ≡
        # legible 1 bps ≡ 0.01% on every one of the six fields (rendered and checked 2026-07-27) —
        # which is what keeps this an ENCODING ablation: before #98 the legible condition resolved
        # 10x finer than raw, so a positive differential was confounded by extra INFORMATION.
        jit = np.clip(np.random.default_rng(int(seed)).normal(1.0, 0.25, size=6), 0.5, 1.5)
        base = {"cvar_05": -0.0410, "cvar_10": -0.0296, "cvar_25": -0.0173,
                "cvar_01": -0.0575, "left_tail_mass": 0.021, "robust_skew": -0.08}
        stats = {k: float(v * jit[i]) for i, (k, v) in enumerate(base.items())}
    return build_block("distributional", 0.0, stats, legible=legible)


def _default_transport_factory(synthetic: bool, sub_cfg: Any, model: str | None):
    """Real reward-author factory: keyless stub for --synthetic or when no API key is set; else the
    configured Opus author. Returns a ``(seed) -> (transport, model_id)`` callable."""
    import os

    llm_cfg = cfg_get(sub_cfg, "llm", {})
    key_env = str(cfg_get(llm_cfg, "api_key_env", "ANTHROPIC_API_KEY"))
    use_stub = synthetic or str(cfg_get(llm_cfg, "pass", "B")).upper() == "A" or not os.environ.get(key_env)

    if use_stub:
        from src.llm.stub_designer import StubDesignerTransport

        def _factory(seed: int):
            return StubDesignerTransport(seed=seed), f"stub-designer/seed{seed}"
        return _factory

    from src.llm.client import build_transport, default_key_env

    provider = str(cfg_get(llm_cfg, "provider", "anthropic"))
    model_id = str(model or cfg_get(llm_cfg, "model_snapshot", "claude-opus-5"))
    temp_raw = cfg_get(llm_cfg, "temperature", None)
    temperature = float(temp_raw) if temp_raw is not None else None
    key_env = str(cfg_get(llm_cfg, "api_key_env", default_key_env(provider)))

    def _factory(seed: int):
        return build_transport(provider, model_id, key_env, temperature=temperature), model_id
    return _factory


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def _parse_seeds(args: argparse.Namespace, sub_cfg: Any) -> list[int]:
    """Resolve the seed list from --seeds N0..N1, --n-seeds N, or config/subexperiment.yaml."""
    if args.seeds:
        if ".." in args.seeds:
            lo, hi = args.seeds.split("..", 1)
            return list(range(int(lo), int(hi) + 1))
        return [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    n = args.n_seeds if args.n_seeds is not None else int(cfg_get(sub_cfg, "n_seeds", cfg_get(sub_cfg, "seeds", 150)))
    return list(range(int(n)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report-only mechanism sub-experiment driver (ADR-039).")
    parser.add_argument("--mode", choices=("named", "legible"), required=True)
    parser.add_argument("--seeds", default=None, help="inclusive range 'N0..N1' or a comma list 'a,b,c'")
    parser.add_argument("--n-seeds", type=int, default=None, help="run seeds 0..n-1 (overridden by --seeds)")
    parser.add_argument("--candidates", type=int, default=None, help="candidates_per_gen K (the budget per seed)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--synthetic", action="store_true", help="synthetic panel + keyless stub author")
    parser.add_argument("--model", default=None, help="override the reward-author model id")
    args = parser.parse_args(argv)

    sub_cfg = _safe_load_sub_cfg()
    seeds = _parse_seeds(args, sub_cfg)
    candidates = int(args.candidates if args.candidates is not None else cfg_get(sub_cfg, "candidates", 5))
    summary = run_subexperiment(
        args.mode, seeds=seeds, candidates=candidates, output_dir=args.output_dir,
        synthetic=args.synthetic, model=args.model, sub_cfg=sub_cfg,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
