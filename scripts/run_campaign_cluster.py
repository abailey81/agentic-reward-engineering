"""Cluster campaign entry point — run the headline campaign on UCL Myriad (PLAN §7 / §12 B-A1).

This is the thin, professional GLUE that turns the certified orchestrator (``src.cluster.campaign``)
into a runnable campaign: it ASSEMBLES the campaign config by reusing the SAME helpers the laptop
``run_campaign.run_headline_campaign`` uses (``load_config`` / panel loader / ``resolve_windows`` /
``run_prototype._agent_cfg`` / ``build_parallel_opts``), so the cluster runs byte-identical science —
there is no second, drift-prone config path. It then wires :func:`build_cluster_run` and calls
:func:`run_campaign_on_cluster`.

The GO-time runsheet (PLAN §12 B-B) precedes this: VPN up, ``~/.ssh/config`` username filled + key
installed, repo pushed (``git archive HEAD | ssh myriad tar -x -C ~/llmrp``), gold staged to ACFS,
``scripts/myriad/build_env.sh`` run + G1 cert passed. THEN, on the laptop driver::

    # validate the wiring WITHOUT touching the cluster (renders one jobscript + one gen's specs):
    python scripts/run_campaign_cluster.py --dry-run --synthetic
    # the real run (over the VPN):
    python scripts/run_campaign_cluster.py \
        --remote-root ~/Scratch/llmrp --gold-dir /acfs/users/<u>/llmrp-inputs \
        --arms distributional scalar scalar_cvar5 ... --seeds 0-402 --resume

Nothing here runs a confirmatory unit on the laptop — training is 100% on Myriad; authoring/reflection
/selection are laptop-side (keys are laptop-only, R10). The laptop track remains the certified default
until the launch decision is given; this script is inert unless invoked.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

_LOG = logging.getLogger("run_campaign_cluster")

#: The pre-2026-07-27 ``--arms`` argparse default, kept ONLY for the non-tiered rehearsal/probe/D1
#: paths that relied on it. It is deliberately NOT the frozen roster: the headline is ``--tiered``
#: and resolves the roster from config, so anything reaching this constant is by definition not the
#: confirmatory campaign. See :func:`resolve_cluster_arms`.
_LEGACY_DEFAULT_ARMS = ("distributional", "scalar")


def _parse_seeds(spec: str) -> list[int]:
    """Parse ``--seeds`` as a comma list and/or ``a-b`` inclusive ranges (e.g. ``0-402`` or ``0,1,5``)."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = (t.strip() for t in part.split("-", 1))
            # str.isdigit() rejects underscores ('0_567'.isdigit() is False -> CPython would parse it to
            # 567), signs, and spaces — silent malformations that would run the WRONG seed set (audit
            # 2026-07-19). Also reject a reversed range, which range() turns into a silent empty set.
            if not (a.isdigit() and b.isdigit()):
                raise SystemExit(f"--seeds range {part!r} must be 'A-B' with non-negative integers")
            ai, bi = int(a), int(b)
            if bi < ai:
                raise SystemExit(f"--seeds range {part!r} is reversed (A > B: {ai} > {bi})")
            out.extend(range(ai, bi + 1))
        else:
            if not part.isdigit():
                raise SystemExit(f"--seeds token {part!r} must be a non-negative integer")
            out.append(int(part))
    if not out:
        raise SystemExit(f"--seeds {spec!r} parsed to an empty seed set")
    return sorted(set(out))


def assemble_cluster_inputs(
    *,
    arms: list[str],
    seeds: list[int],
    output_dir: str | Path,
    synthetic: bool,
    train_steps: int | None,
    n_trials: int,
    candidates: int,
    generations: int,
    search_seed: int,
    embargo: int,
    pass_mode: str,
    provider: str,
    llm_cfg: dict[str, Any] | None,
    resume: bool,
) -> dict[str, Any]:
    """Assemble ``(opts_for, seeds, test_leg_kwargs, frozen_root, agent_cfg, panel_descriptor)``.

    A pure, cluster-free function (loads the panel on the DRIVER only to resolve the integer windows
    the nodes reuse) — MIRRORS ``run_headline_campaign``'s synthetic/gold + window + agent-config +
    opts assembly so the two paths cannot drift. Unit-testable without a cluster.
    """
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts on path (codebase convention)
    from run_campaign import (  # noqa: F401  (parity import)
        _assert_expected_windows,
        make_agent_trainer_factory,
        resolve_windows,
    )
    from run_prototype import _agent_cfg, build_parallel_opts

    from src.utils.config import cfg_get, load_config

    env_cfg = load_config("environment")
    inf_cfg = load_config("inference")
    lookback = int(env_cfg["state"]["lookback_days"])

    # Panel (driver-side, ONLY to derive the integer windows the nodes reuse) + the picklable
    # descriptor the on-node worker reloads from (mirror of run_headline_campaign 1424-1490).
    if synthetic:
        from src.data.synthetic import make_synthetic_panel

        panel = make_synthetic_panel(n_assets=30, n_days=7800, seed=0)
        panel_descriptor: dict[str, Any] = {"synthetic": True, "n_assets": 30, "n_days": 7800}
    else:
        from src.data.loaders import load_gold_panel

        span = cfg_get(cfg_get(cfg_get(inf_cfg, "splits", {}), "evaluation", {}), "span",
                       ["2020-01-01", "2026-06-30"])
        panel = load_gold_panel(phase="development", end=str(span[1]),
                                verify_checksum=True, validate=True).panel
        panel_descriptor = {"synthetic": False, "phase": "development", "end": str(span[1]),
                            "on_missing": "liquidate_to_cash"}

    splits = cfg_get(inf_cfg, "splits", {})
    train_window, val_window, test_window = resolve_windows(panel, lookback, splits, embargo=embargo)

    # 2026-07-19 (35-agent audit, CONFIRMED major): the laptop path guards resolved windows against
    # config/inference.yaml splits.expected_windows (M1 drift guard, run_campaign._assert_expected_
    # windows) but the Myriad entry point did NOT — a rebuilt panel whose session axis shifts could
    # slide the integer windows THROUGH the searchsorted clamps silently and train/score the wrong
    # span. Mirror the laptop guard exactly on the real-gold path (synthetic has no frozen calendar).
    if not synthetic:
        from src.data.loaders import gold_suffix

        _assert_expected_windows(gold_suffix(), train_window, val_window, test_window, splits)

    # Agent config (matched-compute B*): prototype base + campaign overlay (mirror 1522-1540).
    # 2026-07-18 LAUNCH-CRITICAL FIX (caught by the pre-flight coherence assert, the day before
    # launch): --train-steps None claimed "default: campaign.yaml" but _agent_cfg fell back to
    # PROTOTYPE.yaml's 25,000 — the entire campaign would have trained at 1/16th the registered
    # B*. Resolve None from campaign.yaml's top-level train_steps_per_candidate (the R77 value),
    # and HARD-ASSERT the assembled budget equals the PRE-REGISTERED B* (config/preregistration
    # .yaml) — the freeze guard checks config mirrors, not the runtime assembly; this closes the
    # runtime side so the class can never recur.
    if train_steps is None:
        train_steps = int(cfg_get(load_config("campaign"), "train_steps_per_candidate", 0) or 0)
        if not train_steps:
            raise SystemExit("campaign.yaml train_steps_per_candidate missing — B* unresolved")
        _prereg_bstar = int(cfg_get(load_config("preregistration"),
                                    "train_steps_per_candidate", 0) or 0)
        if _prereg_bstar and train_steps != _prereg_bstar:
            raise SystemExit(
                f"campaign.yaml B* ({train_steps}) != pre-registered B* ({_prereg_bstar}) — "
                f"mirror drift; fix the configs (a dated amendment) before assembling")
    agent_cfg = _agent_cfg(load_config("prototype"), train_steps)
    for _k, _v in dict(cfg_get(load_config("campaign"), "agent", {}) or {}).items():
        agent_cfg[_k] = _v
    # 2026-07-13 audit: the LAPTOP-calibrated thermal governor (88C hi on the RTX-4050) must not
    # ship onto V100 nodes — a co-tenanted card crossing 88C would pause trainings up to 1800s
    # each inside an already-tight h_rt. Nodes have datacenter thermal management; the governor is
    # a laptop concern. Result-neutral (wall-clock-only knob).
    agent_cfg["thermal_guardian"] = None

    # Search opts — the SAME build_parallel_opts the laptop search uses (mirror _search_parallel_arm).
    proto = load_config("prototype")
    structural = {"agent": agent_cfg, "reward_family": cfg_get(proto, "reward_family", {}),
                  "data": cfg_get(proto, "data", {})}
    opts = build_parallel_opts(
        structural, env_cfg, llm_block=(llm_cfg if llm_cfg is not None else cfg_get(proto, "llm", {})),
        train_steps=int(agent_cfg["train_steps_per_candidate"]), n_trials=n_trials, synthetic=synthetic,
        seed=search_seed, candidates=candidates, generations=generations, pass_mode=pass_mode,
        provider=provider, resume=resume,
    )

    frozen_root = Path(output_dir) / "frozen"
    test_leg_kwargs = dict(
        panel_descriptor=panel_descriptor, env_cfg=env_cfg, agent_cfg=agent_cfg,
        train_window=train_window, val_window=val_window, test_window=test_window,
        embargo=embargo, lookback=lookback,
    )
    return {
        "opts_for": (lambda _arm: opts),  # opts are arm-independent; the arm is a run_search_arm param
        "opts": opts, "seeds": seeds, "test_leg_kwargs": test_leg_kwargs,
        "frozen_root": frozen_root, "agent_cfg": agent_cfg, "panel_descriptor": panel_descriptor,
        "windows": (train_window, val_window, test_window),
    }


def _dry_run(inputs: dict[str, Any], arms: list[str], *, remote_root: str, gold_dir: str,
             pool: str, pack: int) -> int:
    """Validate wiring WITHOUT the cluster: build one generation's specs (fails loud on a
    non-JSON-serializable config, BUG-3) + render one jobscript. No ssh, no submit."""
    import tempfile

    from src.cluster.campaign import _search_spec
    from src.cluster.jobscript import render_jobscript
    from src.cluster.spec_io import write_specs

    opts = inputs["opts"]
    arm = arms[0]
    cpg = max(1, int(opts["candidates"]) // max(1, int(opts["generations"])))
    specs = [_search_spec(arm, f"# stub reward {k}\n", f"{arm}-g0-c{k}", opts,
                          "PROMPT", 0, f"{remote_root}/outputs/search") for k in range(cpg)]
    with tempfile.TemporaryDirectory() as td:
        n = write_specs(specs, Path(td) / f"{arm}_g0")  # strict-JSON guard runs here
    js = render_jobscript(f"{arm}_g0", cpg, remote_root, gold_dir, pool=pool, pack=pack)
    _LOG.info("DRY-RUN OK: %d arms, %d seeds; one gen = %d JSON-clean specs; jobscript renders (%d lines)",
              len(arms), len(inputs["seeds"]), n, len(js.splitlines()))
    print(f"[dry-run] wiring valid — {len(arms)} arms, {len(inputs['seeds'])} seeds, "
          f"{cpg} candidates/gen, windows={inputs['windows']}, pool={pool}, pack={pack}")
    return 0


def _write_campaign_summary(output_dir: str | Path, inputs: dict[str, Any], *,
                            freeze_stamp: dict[str, Any] | None,
                            extra: dict[str, Any],
                            filename: str = "campaign_summary.json") -> None:
    """P7 (2026-07-13 pre-spend audit): the cluster mirror had NO ``campaign_summary.json``, so
    ``analyze_campaign.py`` silently skipped the DeMiguel benchmark floor (it reads ``test_window``
    from the summary) and the sentinel/monitor had no terminal-state sentinel to key off. Mirrors
    ``run_campaign``'s summary keys (windows + freeze stamp + gold-panel provenance) and REUSES its
    atomic writer. Written ONLY on terminal completion — never at the C3 review stop, where the
    watcher would misread the file as "campaign done"."""
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_campaign import _write_summary_atomic

    train_window, val_window, test_window = inputs["windows"]
    if inputs["panel_descriptor"].get("synthetic"):
        provenance: dict[str, Any] = {"synthetic": True}
    else:
        try:
            from capture_env import _gold_panel_provenance

            provenance = _gold_panel_provenance()
        except Exception as exc:  # noqa: BLE001 — provenance is best-effort; the descriptor still names the panel
            print(f"[cluster] gold-panel provenance unavailable ({type(exc).__name__}: {exc}); "
                  "recording the panel descriptor instead", flush=True)
            provenance = dict(inputs["panel_descriptor"])
    summary = {
        "source": "run_campaign_cluster",
        "train_window": list(train_window),
        "val_window": list(val_window),
        "test_window": list(test_window),
        "freeze": freeze_stamp if freeze_stamp is not None else {"enforced": False, "frozen": None},
        "gold_panel": provenance,
        **extra,
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    _write_summary_atomic(Path(output_dir), summary, filename=filename)
    # Parity with run_campaign.py's run-end seal (audit 2026-07-24): the cluster mirror is
    # AUTHORITATIVELY sealed by the bank_gate ritual (runbook §step-1), but a bare `analyze_campaign`
    # BEFORE the gate reports "not_sealed" and this summary otherwise lacks the integrity root (F4).
    # Seal the as-pulled archive here too (this fires only on TERMINAL completion per the docstring;
    # bank_gate re-seals idempotently) and stamp the root for summary parity. Best-effort — a seal
    # failure must never sink a completed campaign. Sibling import: scripts/ is already on sys.path
    # (line ~220); `from scripts.archive_integrity` would itself hit the import-bug class on direct launch.
    try:
        import json as _json

        from archive_integrity import write_manifest

        _manifest = write_manifest(output_dir)
        summary["archive_integrity_root"] = _json.loads(
            Path(_manifest).read_text(encoding="utf-8")).get("root")
        _write_summary_atomic(Path(output_dir), summary, filename=filename)
    except Exception as exc:  # noqa: BLE001 — sealing is provenance, never a run-blocker
        print(f"[cluster] archive-integrity seal skipped: {exc}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the headline campaign on UCL Myriad.")
    p.add_argument("--arms", nargs="+", default=None,
                   help="Arms to run. NORMALLY OMIT IT: under --tiered the frozen "
                        "config/campaign.yaml roster is resolved automatically, and under --leg "
                        "the five LLM feedback arms are; a hand-typed list is a copy of a frozen "
                        "value and DRIFTS (it still said 7 after R108 took the roster to 9, which "
                        "would have made confirmatory node N4 unsatisfiable). If passed under "
                        "--tiered it must be EXACTLY the frozen roster. Omitted on any other path "
                        "keeps the historical two-arm default, with a warning. See "
                        "resolve_cluster_arms().")
    p.add_argument("--baselines", nargs="*", default=None,
                   help="H1 hand-designed baseline REWARD_CANON names (fixed rewards, no search; "
                        "flood the pool from minute 0). NORMALLY OMIT IT: under --tiered the frozen "
                        "config/campaign.yaml h1_baselines family is used automatically (never "
                        "hand-mirror a frozen list — it drifts); omitted on any other path skips "
                        "the H1 leg. If passed, it must be EXACTLY the frozen family. See "
                        "resolve_cluster_baselines().")
    p.add_argument("--seeds", default="0-567", help="Test seeds for the NON-tiered path: comma list "
                   "and/or a-b ranges. Default = the full E1 ladder [0..567]. IGNORED under --tiered "
                   "(the config seed schema drives the tiers).")
    p.add_argument("--search-seed", type=int, default=0)
    # 2026-07-18 DEFAULTS-CLASS SWEEP (the B*-resolution bug generalized): every design value
    # below used to carry a HARDCODED MIRROR of a frozen config value as its argparse default
    # (30/6/30/21). Mirrors drift silently — the B* instance would have trained the whole
    # campaign at prototype.yaml's 25k. All five now default to None and resolve in main()
    # from the SAME config keys the laptop main reads (run_campaign.py:2134-2151).
    p.add_argument("--candidates", type=int, default=None,
                   help="Candidate budget per arm (default: campaign.yaml candidates_per_arm).")
    p.add_argument("--generations", type=int, default=None,
                   help="Reflection generations (default: campaign.yaml llm.generations).")
    p.add_argument("--train-steps", type=int, default=None, help="B* (default: campaign.yaml).")
    p.add_argument("--n-trials", type=int, default=None,
                   help="DSR expected-max trial count (default: = candidates, laptop parity).")
    p.add_argument("--embargo", type=int, default=None,
                   help="Embargo trading days (default: inference.yaml splits.embargo_trading_days).")
    p.add_argument("--pass-mode", default="A")
    p.add_argument("--provider", default="stub")
    p.add_argument("--synthetic", action="store_true", help="Synthetic panel (dry-run / cert only).")
    p.add_argument("--output-dir", default="outputs/campaign_cluster")
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--tiered", action="store_true",
        help="Run the Stage-1 C-LADDER (config/campaign.yaml 'seeds' schema): C0 canary -> C1-C3 "
        "priority-laddered core (n=30 floor, H2 pair-adjacent) -> EFFECT-BLIND review gate -> C4 "
        "uniform-n round-robin sweep in assurance blocks (90/95/99).",
    )
    p.add_argument(
        "--approve-tier1", action="store_true",
        help="Create the TIER1_APPROVED file (after reviewing the effect-blind integrity report) so "
        "a tiered --resume run proceeds past the C3 review gate into the C4 sweep.",
    )
    p.add_argument(
        "--no-review-gate", action="store_true",
        help="Tiered mode: skip the C3 review stop (NOT recommended for the real campaign).",
    )
    p.add_argument(
        "--hold-at-gate", action="store_true",
        help="Tiered mode: STOP at the C3 gate for a manual eyeball of the effect-blind integrity "
        "report even when execution health is green (default: auto-proceed on green, stop only on a "
        "real execution defect). Release with --approve-tier1 --resume.",
    )
    p.add_argument(
        "--canary", nargs="*", default=None,
        help="C0 canary baseline names (default: the first 3 of --baselines). Pass an empty list "
        "to skip the canary (NOT recommended).",
    )
    p.add_argument("--dry-run", action="store_true", help="Validate wiring; no ssh, no submit.")
    # Cluster wiring
    p.add_argument("--host", default="myriad")
    p.add_argument("--remote-root", default="~/Scratch/llmrp", help="Scratch working root on Myriad.")
    p.add_argument("--gold-dir", default="~/Scratch/llmrp/inputs", help="Staged gold dir on Myriad.")
    p.add_argument("--pool", default="EF", help="Confirmatory GPU pool (device homogeneity).")
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"),
                   help="Training SUBSTRATE (2026-07-26 CPU lane). 'cuda' = the GPU pools "
                        "(-l gpu=1 -ac allow=<pool>): ~7.8x faster PER TRAINING, so it is the lane "
                        "for the SEQUENTIAL critical paths (the bayes_opt GP chain, the reflection "
                        "chains) — but only 74 GPUs exist cluster-wide and they are heavily "
                        "contended. 'cpu' = no GPU request, no pool pin, 1 core/training: ~14 "
                        "steps/s/core but the d pool alone is 10,584 cores and 636 concurrent were "
                        "measured, so it is the lane for the embarrassingly-parallel TEST flood "
                        "(93-97%% of all trainings). Device homogeneity is enforced per CONTRAST, "
                        "so a GPU search + CPU test split is legitimate — see the dossier §0-PRE.")
    p.add_argument("--pack", type=int, default=1, help="§15 GPU packing (concurrent trainings/job).")
    p.add_argument("--specs-per-task", type=int, default=None, metavar="N",
                   help="THE DURATION LEVER (2026-08-06). Trainings per TASK, independent of "
                        "--pack (trainings at ONCE). N>pack runs multiple WAVES inside one job, so "
                        "the job holds its cores ~N/pack times longer at the SAME dispatch rate — "
                        "the only free variable once fair share caps acquisition. It also HALVES "
                        "the job count at N=2*pack, which raises per-job functional tickets. "
                        "NOTE: YOU MUST size --h-rt for the extra waves: at pack 8, N=16 is 2 waves, "
                        "~18.2h of work, so --h-rt 30:0:0 keeps today's 61%% utilisation margin.")
    p.add_argument("--apptainer-sif", default="~/python311.sif",
                   help="Container image the node trains through (the cluster venv is built INSIDE it; "
                        "RHEL7 glibc is too old for the cu124 wheels natively). Empty string = native venv.")
    p.add_argument("--leg", default=None, metavar="LABEL",
                   help="v2 replication leg (R80/R82): author THIS invocation with the named "
                        "config/legs.yaml leg (e.g. deepseek-v4-pro) — the leg's pinned transport "
                        "(provider/quantization/reasoning/max-tokens + the usage-cost request) "
                        "replaces the campaign llm block, and the archive namespace is FORCED to "
                        "leg_<label> (disjoint roots; the leg_aggregate input contract). "
                        "Report-only: pass --priority -200 per runbook §14.3. Not combinable with "
                        "--llm-from prototype or --h3-singleshot.")
    p.add_argument("--llm-from", choices=["campaign", "prototype"], default="campaign",
                   help="Which config's `llm` block authors the rewards (2026-07-13 pre-spend audit "
                        "fix: llm_cfg was hardcoded None, silently inheriting prototype.yaml's block "
                        "— the WRONG MODEL for the confirmatory run). Default `campaign` = "
                        "config/campaign.yaml's Opus 5 block (the campaign OWNS its author,"
                        "ADR-035). Rehearsals/prototypes pass `prototype` EXPLICITLY.")
    p.add_argument("--batch-tag", default=None,
                   help="Per-run batch-name namespace (2026-07-11d bug fix): the driver's "
                        "double-submit guard matches queued jobs by NAME across the whole user "
                        "queue, so concurrent runs sharing arm names collide (the later run adopts "
                        "the earlier run's array and polls forever). Tag every concurrent run, "
                        "e.g. --batch-tag pm.")
    p.add_argument("--seed-pool-blocks", default=None, metavar="POOL:LO-HI,...",
                   help='Device-stratified seed blocks (RATIFIED 2026-07-11c), e.g. '
                        '"EF:0-283,L:284-567": whole seed blocks run on different GPU pools while '
                        'every CRN pair (same seed, all arms) stays device-homogeneous — a '
                        'randomized-block design that adds the A100 pools to confirmatory C. '
                        'Blocks must be disjoint; unassigned seeds fall back to --pool.')
    p.add_argument("--h-rt", default=None, metavar="H:M:S",
                   help="Per-array walltime request. Default: AUTO-SIZED from the measured aggregate "
                        "throughput curve at worst-node contention (steps x pack / (0.5 x F-curve) + "
                        "overhead, x1.3) — NEVER a per-training number: at pack=5 the per-training "
                        "wall stretches ~2x under GPU time-slicing and a per-training h_rt would "
                        "walltime-kill every packed task (the 2026-07-12 p6ext incident class).")
    p.add_argument("--cores-per-training", type=int, default=None,
                   help="CPU cores requested per packed training (total job cores = this × --pack). "
                        "Default (None) keeps the jobscript's 4×pack. Myriad GPU-node CORES are the "
                        "binding scheduling constraint, so lowering this (e.g. 2) makes packed jobs place.")
    p.add_argument("--search-pack", type=int, default=None, metavar="P",
                   help="MODE-D phase-adaptive packing (2026-07-21): run SEARCH waves (the 6-deep "
                        "reflection critical path) at this pack with a matching auto-sized tight "
                        "walltime, while winner/rung bursts keep --pack's throughput. The measured "
                        "curve makes pack a latency/throughput dial (pack-2 halves chain latency "
                        "at ~half throughput on ~20%% of the work). Ops-only — identical "
                        "seeds/steps/maths. Recommended: 2. Default None = uniform pack (legacy).")
    p.add_argument("--search-poll-secs", type=float, default=None, metavar="S",
                   help="MODE-D chain-lane polling (2026-07-21b): poll SEARCH/BO chain batches at "
                        "this cadence (recommended 45) while burst arrays keep --poll-secs. Every "
                        "chain handoff (6 generations x 10 lines; the 30-step BO chain) pays up "
                        "to --poll-secs of notice latency — ~1h+ on the BO chain alone at 180s. "
                        "Fast polling runs only while small chain batches are outstanding.")
    p.add_argument("--pipeline-rungs", action="store_true",
                   help="MODE-D: submit ALL C4 assurance blocks at once under a descending "
                        "priority ladder (tier-100 at -100 above the legs; tier-189+ from -300 "
                        "below them = the registered unified queue enforced natively) instead of "
                        "draining each block before submitting the next. Rungs still COMPLETE in "
                        "order; a rung only BANKS when it and all below are clean (cumulative "
                        "tiers). Removes every inter-block drain bubble.")
    p.add_argument("--chunk-tasks", type=int, default=None, metavar="K",
                   help="Split every submission round into MANY SMALL ARRAYS of K tasks each "
                        "(2026-07-13 max-throughput lever): the scheduler's serialization policy "
                        "(snx=1, observed ACTIVE) holds a big array's tail in hqw at ~1 task/2h — "
                        "chunked arrays have no tail to hold, every part immediately eligible. "
                        "Campaign launch uses --chunk-tasks 1 (task = a pack-5 bundle). Default "
                        "None = one array per round (legacy).")
    p.add_argument("--exclude-hosts", default=None, metavar="H1,H2",
                   help="Comma-separated nodes to FENCE OFF. A node that fails a job in SECONDS "
                        "is always free, so the scheduler keeps feeding it work and it keeps "
                        "destroying it -- a job vacuum. Live 2026-07-27: node-d00a-230 has no "
                        "apptainer and ate 5 of our jobs in ~40 min. Self-healing (resume re-runs "
                        "the spec) but pure waste across 42,128 trainings.")
    p.add_argument("--search-threads", type=int, default=None, metavar="N",
                   help="Intra-op threads per SEARCH/chain training (R107). Default None = the "
                        "REGISTERED value (lanes.CPU_CHAIN_THREADS) on the CPU lane, so code and "
                        "register agree. NOTE: MEASURED 2026-07-27: at the real workload 8 threads give "
                        "15.4 steps/s = 1.93/core vs 13.0/core at 1 thread, so R107's 2.72x is "
                        "really ~1.18x -- and running the 1,800-training search leg at 8 threads "
                        "costs ~88,500 EXTRA core-hours, +24.6%% on the whole campaign, to buy a "
                        "1.18x latency gain on chains that only bind above 1,685 cores. Pass 1 to "
                        "decline that trade (and amend R107 to match).")
    p.add_argument("--poll-secs", type=float, default=600.0)
    p.add_argument("--max-author-calls", type=int, default=None, help="Hard authoring spend cap.")
    p.add_argument("--allow-unfrozen", action="store_true",
                   help="Dev escape hatch (P19, 2026-07-13 audit): downgrade the freeze "
                        "verify-or-refuse gate to a WARNING. ONLY for rehearsals/prototypes "
                        "(e.g. the pm2 prototype) — the real confirmatory campaign runs frozen.")
    p.add_argument("--root-suffix", default=None, metavar="NAME",
                   help="C6-class namespace guard (2026-07-13): ANY report-only re-search "
                        "invocation (e.g. the D1 search-saturation curve levels) MUST pass a "
                        "suffix — its archives go to search_<NAME>/, test_<NAME>/, frozen_<NAME>/ "
                        "and its batch names are prefixed <NAME>_. Sharing the confirmatory roots "
                        "would let the compacted resume ADOPT headline run_ids and fabricate the "
                        "exhibit (the P4 hazard class). Lowercase [a-z0-9_]+ only.")
    p.add_argument("--priority", type=int, default=None,
                   help="SGE -p override for THIS invocation's arrays (default: the mode's "
                        "ladder value — confirmatory 0/-100; report-only invocations should "
                        "pass -200 per §14.3). A NEGATIVE value is REFUSED unless "
                        "--allow-deprioritise is also given (#96).")
    p.add_argument("--allow-deprioritise", action="store_true",
                   help="Explicitly permit a NEGATIVE --priority. Required by #96 because the standing "
                        "standing rule is that our jobs never sit below full fair-share standing; "
                        "the only sanctioned use is a report-only §14.3 invocation. NEVER pass this "
                        "for the confirmatory campaign.")
    p.add_argument("--h3-singleshot", action="store_true",
                   help="C5 (P4 closed 2026-07-13): run the pre-registered H3 SINGLE-SHOT control "
                        "on the cluster — the distributional arm at generations=1 (no reflection), "
                        "same candidate budget + agent config + seeds, archived to the SEPARATE "
                        "*_h3_singleshot/ roots at -p -100. A separate invocation from the "
                        "headline campaign (shares --output-dir; roots are disjoint by "
                        "construction; batch names force-prefixed h3ss_). Forces "
                        "--arms distributional and --generations from campaign.yaml's "
                        "h3_singleshot_generations.")
    return p


def resolve_leg_override(label: str, explicit_root_suffix: str | None) -> tuple[dict[str, Any], str, str]:
    """Resolve a v2 replication leg into ``(llm_cfg override, provider, forced root suffix)``.

    The llm block mirrors the campaign-block schema ``build_parallel_opts`` consumes
    (``model_snapshot``/``api_key_env``/``max_tokens``/``extra_body``), built from the SAME
    ``transport_kwargs`` translation the pre-launch gates use — one translation point, no drift.
    ``temperature`` stays None (provider default, recorded — the registered diversity protocol is
    prompt-variation, unified across legs). The archive suffix is forced to ``leg_<label>``
    (sanitized to the [a-z0-9_]+ suffix grammar); an explicitly-passed different suffix is refused
    rather than silently overridden (ambiguity kills resumability).
    """
    import re as _re

    from src.llm.legs import leg_by_label, transport_kwargs

    leg = leg_by_label(label)          # loud KeyError listing known labels on a typo
    tk = transport_kwargs(leg)
    llm_cfg: dict[str, Any] = {
        "pass": "B",
        "provider": tk["provider"],
        "model_snapshot": tk["model"],
        "api_key_env": tk["api_key_env"],
        "max_tokens": tk["max_tokens"],
        # R85: the leg's registered decoding pin (1.0 on OpenRouter legs; absent on Anthropic).
        "temperature": tk.get("temperature"),
        "diversity_prompt_variation": True,
    }
    if tk.get("extra_body"):
        llm_cfg["extra_body"] = tk["extra_body"]
    # R106: the Anthropic legs' reasoning pin travels as ``thinking`` (that transport rejects
    # extra_body). Dropping it here would strip the registered reasoning-off from haiku/sonnet at
    # authoring time — the leg would run on the vendor default while the registration said otherwise.
    if tk.get("thinking"):
        llm_cfg["thinking"] = tk["thinking"]
    suffix = "leg_" + _re.sub(r"[^a-z0-9_]", "_", str(label).lower())
    if explicit_root_suffix and explicit_root_suffix != suffix:
        raise SystemExit(
            f"--leg {label!r} forces --root-suffix {suffix!r}; the explicit --root-suffix "
            f"{explicit_root_suffix!r} conflicts — drop it (the leg owns its namespace)."
        )
    return llm_cfg, str(tk["provider"]), suffix


def autosize_h_rt(pack: int, train_steps: int, *, device: str = "cuda") -> str:
    """Walltime default for a pack-``pack`` task at ``train_steps``, **per SUBSTRATE**.

    ``cuda`` — the measured clean aggregate curve at its WORST (x0.5 contention) + 1200 s overhead
    + 30 % margin, rounded up to whole hours (unit-testable — the 2026-07-18 defaults-class sweep
    found the inline version sized on a stale hardcoded 200k).

    ``cpu`` — **LANE-AWARE (2026-07-27, launch-gate catch).** The GPU branch was the ONLY branch,
    and it is wrong on CPU in both of its terms, in the same direction:

    1. **The rate.** ``_agg_clean`` is a GPU aggregate-throughput curve. A CPU training runs at the
       registered ``CPU_STEPS_PER_S_PER_CORE`` (13.0), so it must be priced off the CPU planning
       floor, not off a card.
    2. **The pack term.** ``x int(pack)`` models GPU TIME-SLICING, where packed trainings share one
       device and the task's wall grows with the pack. On the CPU lane ``pack N`` +
       ``cores_per_training 1`` is N INDEPENDENT trainings on N OWN cores (the 2026-07-27
       packing-is-not-threading correction), so the task's wall is ONE training's wall, flat in
       pack. Multiplying by pack does not make the CPU estimate conservative — it makes it wrong in
       a way that happens to cancel part of the rate error, and only part of it.

    MEASURED consequence of the old formula, which is why this is a launch-blocker and not a tidy-up:
    ``autosize_h_rt(4, 400_000)`` returned ``"6:0:0"`` while a 400k CPU training needs **8.55 h** at
    the registered 13.0 steps/s and **6.11 h** even at the fastest rate ever observed (18.2). EVERY
    task of the confirmatory campaign would have been SIGKILLed at its walltime having archived
    nothing — the ``p6ext800/1600`` incident class, at campaign scale. (The launch-readiness
    review §8 asserted "``_auto_h_rt`` is lane-aware now"; that fix
    landed in ``scripts/p6_authored_ladder.py`` only, and this entry point never got it.)

    The rate used is :data:`src.cluster.lanes.CPU_PLANNING_STEPS_PER_SEC` — deliberately the SAME
    object the ladder sizes from, so the two CPU walltime estimators cannot drift apart again.
    ``h_rt`` is a LIMIT, not a reservation: over-asking costs only backfill position (and walltime
    was measured IRRELEVANT to placement — an 11 h request placed as fast as a 50 min one, 15/15),
    whereas under-asking costs the entire job.
    """
    if str(device) == "cpu":
        from src.cluster.lanes import CPU_PLANNING_STEPS_PER_SEC

        secs = (int(train_steps) / float(CPU_PLANNING_STEPS_PER_SEC) + 1200.0) * 1.3
        return f"{int(secs // 3600) + 1}:0:0"
    _agg_clean = {1: 102.0, 2: 133.0, 3: 220.0, 4: 240.0, 5: 253.0, 8: 257.0}
    agg = 0.5 * _agg_clean.get(int(pack), 253.0)
    secs = (int(train_steps) * int(pack) / agg + 1200.0) * 1.3
    return f"{int(secs // 3600) + 1}:0:0"


def assert_remote_gold(runner: Any, gold_dir: str, *, real_spend: bool) -> dict[str, str]:
    """Prove the LICENSED GOLD PANEL is on the cluster, at the right bytes, BEFORE anything is
    submitted. Returns ``{basename: sha256}`` for what was verified.

    THE DEFECT THIS CLOSES (2026-07-27 launch gate). ``--gold-dir`` defaults to
    ``~/Scratch/llmrp/inputs``. That directory exists on Myriad and is **EMPTY** — the licensed gold
    actually lives on ACFS at ``/acfs/users/ucestes/gold`` (which is what ``p6_authored_ladder``
    passes, and why the ladder works). Nothing checked. Worse, the jobscript deliberately
    ``mkdir -p``s the bind source so Apptainer cannot FATAL on a missing mount (a fix for a
    different bug), so an empty gold dir produces a *successful container start* and then a loader
    failure on EVERY task — thousands of core-hours of uniform, late, per-task failure with no
    single loud cause. Gold absence must fail ONCE, at t0, on the laptop.

    It also closes the matching REPRODUCIBILITY hole: the LAPTOP-side panel is checksum-verified
    (``load_gold_panel(verify_checksum=True)``) and the REMOTE one never was, even though the remote
    copy is the one every training actually reads. A wrong-but-present panel is worse than an absent
    one — it produces plausible numbers. We therefore compare the remote SHA-256 against the frozen
    manifest (``data/manifest/checksums.txt`` via :func:`src.data.loaders._expected_sha256`), which
    is the same authority the laptop loader uses, so both ends assert the same bytes.

    Non-fatal (WARN) when ``real_spend`` is False: rehearsals and ``--synthetic`` runs legitimately
    need no gold.
    """
    from pathlib import Path as _P

    from src.data.loaders import _expected_sha256, gold_suffix

    suffix = gold_suffix()
    required = [f"{stem}_{suffix}.parquet" for stem in
                ("returns_panel", "cash_features", "splits", "top30_selection")]
    # ``ssh_runner`` takes an ARGV LIST and shlex-quotes each element (submit.ssh_runner); handing
    # it a plain string makes Python iterate the CHARACTERS and quote each one, so the remote sees
    # `m k d i r ' ' - p ...` and returns 127. Caught by the live rehearsal 2026-07-28 — the unit
    # tests could not see it because their fake runners accepted any object. `bash -c <script>` is
    # the form that docstring documents.
    quoted = " ".join(f"'{gold_dir}/{n}'" for n in required)
    out = runner(["bash", "-c", f"sha256sum {quoted} 2>&1 || true"])
    seen: dict[str, str] = {}
    for line in str(out).splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 64:
            seen[_P(parts[1]).name] = parts[0].lower()
    missing = [n for n in required if n not in seen]
    if missing:
        msg = (f"GOLD PANEL NOT ON THE CLUSTER: {missing} absent from --gold-dir {gold_dir!r}. "
               f"Every training would start its container and then fail in the loader. The "
               f"licensed gold is staged on ACFS (e.g. /acfs/users/<user>/gold) — pass that path "
               f"as --gold-dir, or stage the panel into this one.")
        if real_spend:
            raise SystemExit(msg)
        _LOG.warning("%s (non-fatal: not a real-spend run)", msg)
        return seen
    drifted = []
    for name, got in sorted(seen.items()):
        want = _expected_sha256(_P("data") / "gold" / name)
        if want and want.lower() != got:
            drifted.append(f"{name}: remote {got[:12]} != frozen {want[:12]}")
    if drifted:
        msg = ("REMOTE GOLD DOES NOT MATCH THE FROZEN MANIFEST — the cluster would train on a "
               f"DIFFERENT panel than the design registers: {drifted}")
        if real_spend:
            raise SystemExit(msg)
        _LOG.warning("%s (non-fatal: not a real-spend run)", msg)
    else:
        _LOG.info("remote gold VERIFIED at %s — %d files, sha256 == the frozen manifest (%s)",
                  gold_dir, len(seen), ", ".join(f"{n}:{s[:8]}" for n, s in sorted(seen.items())))
    return seen


def assert_no_foreign_remote_records(runner: Any, remote_outputs_root: str,
                                     local_archive_root: str, roots: list[str], *,
                                     real_spend: bool, batch_tag: str | None = None) -> int:
    """Refuse to start a FRESH campaign on top of records this campaign did not create.

    THE DEFECT THIS CLOSES (2026-07-27, found by inspecting the cluster rather than the code).
    ``~/Scratch/llmrp/outputs/search/`` — the CORE LINE's confirmatory search root — held **8
    records from probe runs on 2026-07-24/25**, with run_ids in exactly the campaign's namespace:
    ``distributional-g0-c0..c4`` (the COMPLETE generation-0 candidate set for that arm, since
    ``candidates=30 / generations=6`` gives 5 per generation) plus ``scalar-g0-c2..c4``.

    What would have happened, quietly: the driver's first act is a pull; ``pull_archive`` mirrors
    those records into the local archive; ``pending_specs`` then sees those run_ids as ALREADY
    ARCHIVED; and ``run_search_arm`` under ``--resume`` REPLAYS an archived candidate instead of
    authoring a new one. The confirmatory search leg would have adopted three-day-old probe
    candidates — authored under a different config, possibly by the stub — as its own generation 0,
    and reflected on them. Nothing would have failed. The records look valid, because they ARE
    valid records; they are just not this experiment's.

    Every existing guard misses it. The F2 guard checks the LOCAL directory and only when
    ``--resume`` is ABSENT — and the confirmatory launch correctly passes ``--resume`` on every
    line, because that is what makes a driver death survivable. So on the one path the campaign
    actually uses, neither side was checked.

    THE DISCRIMINATOR, and why it has no false positives: a genuine resume has already mirrored its
    own remote records locally, so ``local == 0 and remote > 0`` can only mean the remote records
    were produced by something else. A fresh run starts with an empty local archive by definition;
    a resumed run does not.

    NOTE: THE SECOND HALF OF THE DISCRIMINATOR, and the reason it is not just the local-record test.
    There is a window in which the local-record test alone would produce a FALSE POSITIVE that
    wedges a line permanently: this line submits, its records land REMOTELY, and the driver dies
    before its next pull mirrors them. The supervisor relaunches (that is its whole job), the guard
    now sees ``local == 0 and remote > 0``, refuses, and the supervisor relaunches again — forever,
    at 600 s intervals, on records the line produced ITSELF. So a line that has ever SUBMITTED is
    never treated as fresh: ``local_batch_root/<batch_tag>_*`` is written by ``write_specs`` before
    any qsub, so its existence is proof that the remote records under these roots can be this
    line's own. Both halves are needed — the local-record test alone false-positives, and the
    batch-dir test alone would go inert for a line that shares an output dir with eleven others.
    """
    import re as _re
    from pathlib import Path as _P

    # (a) HAS THIS LINE EVER SUBMITTED? If so it is not a fresh launch, whatever the mirror says.
    _tag = _re.sub(r"[^A-Za-z0-9_]", "_", str(batch_tag or "core"))
    if next(_P(local_archive_root).joinpath("batches").glob(f"{_tag}_*"), None) is not None:
        return 0

    # (b) Scoped to THIS line's roots, deliberately. All twelve MODE-D lines share one
    # --output-dir, so a whole-directory check would go inert for every line that starts after the
    # first one wrote a record — and the legs start an hour behind the core by design (the canary
    # shield). Per-root scoping keeps the discriminator meaningful for each line independently.
    for _r in roots:
        if next(_P(local_archive_root).joinpath(_r).rglob("record.json"), None) is not None:
            return 0                               # a genuine resume: this line's mirror is present
    quoted = " ".join(f"'{remote_outputs_root}/{r}'" for r in roots)
    # argv LIST, not a string — see the note in assert_remote_gold.
    out = runner(["bash", "-c", f"find {quoted} -name record.json 2>/dev/null | wc -l || true"])
    try:
        n = int(str(out).strip().splitlines()[-1])
    except (ValueError, IndexError):
        # FAIL CLOSED on a real-spend run. This repository's own 2026-07-26 review named
        # "fail-open-on-ABSENT-evidence" as one of three recurring bug CLASSES (#28/#29): a check
        # that cannot see is not a check that passed. Silence here means we do not know whether the
        # confirmatory roots are clean, and the failure this guards against is the SILENT adoption
        # of foreign rewards into the confirmatory search leg — the one class of error that yields a
        # plausible result rather than an obvious one. A transient ssh hiccup costing one relaunch
        # is the cheaper mistake by a wide margin.
        msg = (f"could not determine whether {remote_outputs_root} holds foreign records "
               f"(unparseable probe output: {str(out)[:200]!r}). REFUSING rather than assuming "
               f"clean — re-run once the cluster answers, or sweep the roots by hand.")
        if real_spend:
            raise SystemExit(msg)
        _LOG.warning("%s (non-fatal: not a real-spend run)", msg)
        return 0
    if not n:
        return 0
    msg = (
        f"{n} record(s) ALREADY EXIST under the confirmatory archive roots on the cluster "
        f"({remote_outputs_root}) while the local archive {local_archive_root!r} is EMPTY. A fresh "
        f"campaign cannot have produced them, so they are FOREIGN — left over from a probe, a "
        f"rehearsal or an earlier run. The driver resumes from the ARCHIVE, so it would mirror "
        f"them and then ADOPT any whose run_id matches one of its own candidates, silently "
        f"substituting foreign rewards into the confirmatory search leg. Move them aside on the "
        f"cluster first, e.g.\n"
        f"    ssh <host> \"cd {remote_outputs_root} && mv search _quarantined_$(date -u "
        f"+%Y%m%dT%H%M%SZ)\"\n"
        f"(MOVE, never delete — they are evidence.) Then relaunch."
    )
    if real_spend:
        raise SystemExit(msg)
    _LOG.warning("%s (non-fatal: not a real-spend run)", msg)
    return n


def frozen_arm_roster() -> list[str]:
    """The registered arm roster, read from the hash-bound configs — never hand-typed.

    ``config/arms.yaml`` calls itself "THE AUTHORITATIVE ROSTER"; ``config/campaign.yaml`` mirrors
    it and ``freeze.py::assert_executed_arms_match`` pins BOTH against the pre-registration. So the
    frozen truth is a config read, and any list typed into a launcher is a copy that can rot.
    """
    from src.utils.config import cfg_get, load_config

    roster = [str(a) for a in (cfg_get(load_config("campaign"), "arms", []) or [])]
    if not roster:
        raise SystemExit("config/campaign.yaml has no `arms` roster — arm set unresolved")
    return roster


def llm_arm_roster() -> list[str]:
    """The five LLM feedback arms, DERIVED from ``config/arms.yaml`` rather than hand-listed.

    A replication leg runs "the identical five LLM arms" (``model_suite``); the four DFO arms carry
    ``llm: false`` and author nothing, so they belong to the core line only. Deriving the split from
    the authoritative table means seating a sixth feedback arm cannot silently miss the legs.
    """
    from src.utils.config import cfg_get, load_config

    table = cfg_get(load_config("arms"), "arms", {}) or {}
    frozen = frozen_arm_roster()
    llm = [a for a in frozen if not (isinstance(table.get(a), dict)
                                     and table[a].get("llm") is False)]
    if not llm:
        raise SystemExit("config/arms.yaml resolved ZERO LLM arms — roster unreadable")
    return llm


def resolve_cluster_arms(arms: list[str] | None, *, tiered: bool,
                         leg: str | None = None) -> list[str]:
    """Resolve the arm roster for a cluster launch — the DRIFT-PROOF path (2026-07-27).

    THE DEFECT THIS CLOSES, and it is the SAME one ``resolve_cluster_baselines`` closed for H1 the
    day before. ``--arms`` was taken VERBATIM with an argparse default of ``["distributional",
    "scalar"]``, and **nothing anywhere validated it against the frozen roster** —
    ``freeze.py::assert_executed_arms_match`` compares *config to pre-registration*, never *CLI to
    config*. Measured consequence at the 2026-07-27 launch gate: the ratified MODE-D core line
    (``scripts/mode_d_supervisor.ps1``), the runbook §2 line, ``campaign_supervisor.ps1`` and
    ``install_onstart_task.ps1`` ALL hand-typed a roster that still said **7** after R108 took it to
    **9**, so ``cma_es`` and ``tpe`` — two of the four comparators of the CONFIRMATORY node N4 —
    would never have been trained, and N4's beat-the-max IUT (``p = max`` over the portfolio) would
    have been permanently unsatisfiable. The launch-ready doc's own command was worse: omitting the
    flag ran **two** arms. ``install_onstart_task.ps1`` even carries a comment saying "never
    hand-type a roster" directly above a hand-typed roster of seven.

    Semantics, chosen so every existing non-headline caller keeps working unchanged:

    - ``--leg`` -> the five LLM feedback arms (``llm_arm_roster``), whatever is passed. A leg runs
      no DFO arm and no H1 canon; passing anything else is refused rather than silently narrowed.
    - omitted on the headline ``--tiered`` path -> the FROZEN roster from config.
    - omitted on any other path -> the historical two-arm default, with a LOUD warning (rehearsals,
      probes and the D1 curve levels rely on it; a real-spend run cannot reach here because the
      headline is ``--tiered``).
    - provided -> every name must be a known arm, and on ``--tiered`` the list must be EXACTLY the
      frozen roster (set equality). A partial roster is refused up front, before ssh or any spend.
    """
    frozen = frozen_arm_roster()
    if leg:
        want = llm_arm_roster()
        if arms is not None and set(arms) != set(want):
            raise SystemExit(
                f"--leg {leg!r} runs the five LLM feedback arms ({want}); got {sorted(arms)}. "
                f"A leg authors with its own pinned model and runs no DFO arm and no H1 canon — "
                f"omit --arms and let it resolve.")
        return list(want)
    if arms is None:
        if tiered:
            return list(frozen)
        _LOG.warning(
            "--arms omitted on a NON-tiered run: falling back to the historical two-arm default "
            "%s. The confirmatory campaign is --tiered and resolves the frozen %d-arm roster; if "
            "you meant the headline, pass --tiered.", _LEGACY_DEFAULT_ARMS, len(frozen))
        return list(_LEGACY_DEFAULT_ARMS)
    from src.arms.factory import all_arms

    # ``all_arms()`` yields Arm OBJECTS, not names — comparing a str against them silently makes
    # every name "unknown" (and then blows up sorting them). Key on the name attribute, and fall
    # back to the object itself so a future plain-string factory keeps working.
    known = {str(getattr(a, "name", a)) for a in all_arms()}
    unknown = [a for a in arms if a not in known]
    if unknown:
        raise SystemExit(f"--arms: unknown arm(s) {unknown}; valid: {sorted(known)}")
    if tiered:
        missing = sorted(set(frozen) - set(arms))
        extra = sorted(set(arms) - set(frozen))
        if missing or extra:
            raise SystemExit(
                f"--arms must be the FROZEN roster ({len(frozen)} arms; "
                f"freeze.py::assert_executed_arms_match pins config/arms.yaml + "
                f"config/campaign.yaml against the pre-registration). missing={missing} "
                f"unexpected={extra}. Omit the flag to use the frozen roster.")
    return [str(a) for a in arms]


def resolve_cluster_baselines(baselines: list[str] | None, *, tiered: bool,
                              leg: str | None = None) -> list[str] | None:
    """Resolve the H1 hand-reward family for a cluster launch — the DRIFT-PROOF path.

    The laptop driver has always taken H1 from the FROZEN config and REFUSES a hand-typed list on
    the headline path (``run_campaign.py::resolve_baseline_names``, R97). The cluster driver did
    not: ``--baselines`` was taken verbatim, so the runbook's headline line carried a hand-mirrored
    copy of a frozen config value — precisely the failure mode the 2026-07-18 DEFAULTS-CLASS SWEEP
    closed for B*/candidates/generations/n_trials/embargo. It then drifted: the runbook still named
    the 4 pre-2026-07-26 baselines after the H1 canon expanded to 11, so the headline launch would
    have run a SUBSET of the registered family and silently made the N6 intersection-union node
    (its p = max over the 11 one-sided leg p-values) unsatisfiable — and, because ``--canary``
    defaults to the first 3 of this list, would also have mis-sized the C0 canary hard-gate.

    Semantics (chosen so the h3 / C6 re-search lines keep working unchanged):

    - omitted on the headline ``--tiered`` path -> the frozen ``config/campaign.yaml h1_baselines``
      (the list ``freeze.py::assert_h1_baselines_match`` pins against the pre-registration);
    - omitted on any other path -> ``None`` (skip the H1 leg — what ``--h3-singleshot`` and the
      ``--root-suffix`` re-search invocations rely on);
    - provided -> every name must be a ``REWARD_CANON`` key AND the list must be exactly the frozen
      family (set equality). A partial family is refused up front, before ssh or any spend.
    - **``--leg`` -> ALWAYS None, even under ``--tiered`` (2026-07-27).** The H1 canon is eleven
      HAND-DESIGNED rewards; they contain no LLM-authored code, so they are model-INDEPENDENT and
      belong to the core line exactly once. Attaching them to a leg would train 11 x (the achieved
      rung) identical baseline units per leg — ~10x the entire H1 leg in wasted compute — and would
      let a leg's archive masquerade as an H1 replication that the design never registered. This
      branch is what makes ``--leg --tiered`` (R101 lockstep: every model climbs the SAME ladder)
      safe to run at all.
    """
    from src.utils.config import cfg_get, load_config

    frozen = [str(b) for b in (cfg_get(load_config("campaign"), "h1_baselines", []) or [])]
    if leg:
        if baselines:
            raise SystemExit(
                f"--leg {leg!r} does not run the H1 hand-reward canon: those eleven rewards are "
                f"hand-designed and model-INDEPENDENT, so they belong to the core line once. Drop "
                f"--baselines.")
        return None
    if baselines is None:
        return list(frozen) if (tiered and frozen) else None

    from src.baselines.rewards import REWARD_CANON

    unknown = [b for b in baselines if b not in REWARD_CANON]
    if unknown:
        raise SystemExit(
            f"--baselines: unknown REWARD_CANON key(s) {unknown}; valid: {sorted(REWARD_CANON)}")
    missing = sorted(set(frozen) - set(baselines))
    extra = sorted(set(baselines) - set(frozen))
    if frozen and (missing or extra):
        raise SystemExit(
            f"--baselines must be the FROZEN config h1_baselines family ({len(frozen)} names; "
            f"freeze.py::assert_h1_baselines_match pins it against the pre-registration). "
            f"missing={missing} unexpected={extra}. Omit the flag to use the frozen family.")
    return [str(b) for b in baselines]


def main(argv: list[str] | None = None) -> int:
    from src.utils.console import make_console_safe
    make_console_safe()   # a console codepage must never kill the launcher (src/utils/console.py)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Load .env -> os.environ so the authoring key (ANTHROPIC_API_KEY / OPENROUTER_API_KEY) is available
    # to the LAPTOP-side driver that authors before shipping specs (parity with run_campaign.py:2111).
    # Without this, real-authoring (--pass-mode B) crashes with a "key unset" RuntimeError.
    from src.cluster.submit import expand_remote, remote_home, ssh_runner
    from src.utils.config import load_config as _load_config
    from src.utils.env import load_env

    load_env()
    args = build_parser().parse_args(argv)

    # MYRIAD PRIORITY — ABSOLUTE (standing rule, 2026-07-24): "NEVER lower the SGE/queue
    # priority of any of our jobs, EVER. No `qalter -p <negative>`, no deprioritize, ever — our work
    # always sits at full fair-share standing." The DEFAULT here is already 0 (line ~974), so the
    # confirmatory launch is safe. But `--priority` accepts any int with NO guard, and this file's own
    # help text instructs operators to pass `-200` (the --leg help) and `-300` (the ladder re-run) per
    # runbook §14.3 — so the tooling actively invites the one thing the standing rule forbids, and a
    # deprioritised array in a 23-day queue also works directly against the standing CAMPAIGN-SPEED
    # priority. Refusing outright would break a documented workflow with no arbitration
    # available, so this makes the conflict IMPOSSIBLE TO HIT SILENTLY and leaves the call to him
    # (deep review, finding #96 — raised, not resolved).
    # RESOLVED 2026-07-27 (#96, under the close-all-gaps instruction). The warning below used to PROCEED,
    # because refusing would have broken runbook §14.3's documented report-only ladder while he was
    # unavailable to arbitrate. He is available and the standing rule is ABSOLUTE, so the default is
    # now REFUSAL and the documented path survives behind an explicit opt-in — the same shape as
    # `--allow-unfrozen`. A typo or a copied command line can no longer silently park the CONFIRMATORY
    # campaign below full fair-share standing in a 23-day queue (which also works against the standing
    # CAMPAIGN-SPEED priority); deprioritising now requires stating that you mean it.
    if args.priority is not None and args.priority < 0 and not args.allow_deprioritise:
        raise SystemExit(
            f"REFUSING --priority {args.priority}: the standing rule is ABSOLUTE — never lower our "
            "queue priority, EVER ('MYRIAD PRIORITY — ABSOLUTE'). Full fair-share standing is "
            "-p 0. Runbook §14.3's report-only ladder is the ONLY sanctioned exception; if that is what "
            "you are launching, pass --allow-deprioritise to say so explicitly. NEVER pass it for the "
            "confirmatory campaign."
        )
    if args.priority is not None and args.priority < 0:
        _LOG.warning(
            "NEGATIVE SGE PRIORITY (--priority %d) accepted ONLY because --allow-deprioritise was "
            "given. This run will sit BELOW full fair-share standing. This must be a report-only "
            "§14.3 invocation, never the confirmatory campaign.", args.priority,
        )

    # Capture which design values the CALLER set explicitly, BEFORE any programmatic forcing
    # (the H3 block below sets args.generations itself — that must not read as a caller override).
    _explicit_design = {name: getattr(args, name) is not None
                        for name in ("train_steps", "candidates", "generations",
                                     "n_trials", "embargo")}

    # P19 (2026-07-13 pre-spend audit): the cluster entry point had NO freeze gate at all — the
    # laptop campaign REFUSES to launch on an unfrozen/drifted pre-registration (verify-or-refuse,
    # run_campaign.py enforce_freeze) but the Myriad path would happily author + train the whole
    # confirmatory campaign against an unfrozen design. Mirror the laptop semantics exactly: the
    # keyless --dry-run is exempt; --allow-unfrozen downgrades to a warning (rehearsals only).
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_campaign import CampaignNotFrozenError, enforce_freeze

    freeze_stamp: dict[str, Any] | None = None
    if not args.dry_run:
        try:
            freeze_stamp = enforce_freeze(allow_unfrozen=bool(args.allow_unfrozen))
        except CampaignNotFrozenError as exc:
            raise SystemExit(f"[run_campaign_cluster] {exc}") from exc

    # v2 REPLICATION LEG (R80/R82): resolve BEFORE the root-suffix validation so the forced
    # leg_<label> namespace flows through the same guards as an explicit suffix. The leg supplies
    # its own pinned author block (one translation point with the pre-launch gates: transport_kwargs).
    leg_llm_cfg: dict[str, Any] | None = None
    if args.leg:
        if args.h3_singleshot:
            raise SystemExit("--leg and --h3-singleshot are SEPARATE invocations — a leg runs the "
                             "standard five LLM arms, never the H3 control.")
        if args.llm_from != "campaign":
            raise SystemExit("--leg supplies its own author block from config/legs.yaml — do not "
                             "combine it with --llm-from prototype.")
        leg_llm_cfg, _leg_provider, _forced_suffix = resolve_leg_override(args.leg, args.root_suffix)
        args.root_suffix = _forced_suffix
        logging.getLogger(__name__).info(
            "leg %s: author=%s/%s, root-suffix FORCED to %s",
            args.leg, _leg_provider, leg_llm_cfg["model_snapshot"], _forced_suffix)

    # --root-suffix VALIDATION (C6-class): format + the h3 conflict, hoisted here so a bad flag
    # is rejected BEFORE any cluster contact (2026-07-19 audit: the checks lived after remote_home,
    # so an invalid suffix hit the ssh path first — the validation could not run offline and its
    # own "before any cluster contact" contract was violated). The root-suffix APPLICATION stays
    # after assembly (it rewrites the ClusterRun); only the guards move up.
    if args.root_suffix:
        import re as _re

        if not _re.fullmatch(r"[a-z0-9_]+", args.root_suffix):
            raise SystemExit(f"--root-suffix {args.root_suffix!r} must be lowercase [a-z0-9_]+")
        if args.h3_singleshot:
            raise SystemExit("--h3-singleshot manages its own roots — do not combine it with "
                             "--root-suffix.")

    # C5 (P4): the H3 single-shot control is its OWN invocation — force its fixed shape BEFORE
    # the guards/assembly so the spend cap, divisibility check, and dry-run all see the truth.
    if args.h3_singleshot:
        if args.tiered:
            raise SystemExit("--h3-singleshot and --tiered are SEPARATE invocations (C5 vs "
                             "C1-C4) — run the H3 control after (or alongside) the ladder.")
        from src.utils.config import cfg_get as _h3cg
        args.arms = ["distributional"]
        # h3_singleshot_generations is nested under the campaign.yaml `llm:` block — read it the SAME way
        # the laptop entry (run_campaign.py:2155) and preflight (preflight.py:589) do. A flat top-level
        # cfg_get missed the nested key and silently fell to the default, a laptop/cluster divergence
        # that would bite the moment the config value differs from 1 (audit 2026-07-19).
        _camp_llm = (_load_config("campaign") or {}).get("llm") or {}
        args.generations = int(_h3cg(_camp_llm, "h3_singleshot_generations", 1) or 1)
        _LOG.info("C5 H3 single-shot: arms forced to ['distributional'], generations=%d",
                  args.generations)
    else:
        # ARM ROSTER (2026-07-27): resolved from the FROZEN config, never from an argparse mirror.
        # Placed after the H3 block so C5's deliberately-forced single arm is not overwritten, and
        # before assembly so the spend cap, the divisibility check and the dry-run all see the
        # roster that will actually run. See resolve_cluster_arms() for the drift this closes.
        args.arms = resolve_cluster_arms(args.arms, tiered=bool(args.tiered), leg=args.leg)
        _LOG.info("arms RESOLVED (%d): %s", len(args.arms), " ".join(args.arms))

    # 2026-07-19 (35-agent audit, CONFIRMED major): the LLM_RP_GOLD_SUFFIX env var silently
    # OUTRANKS the freeze-bound config/data.yaml gold.suffix (loaders.gold_suffix precedence), so a
    # stray override would run the whole confirmatory campaign on the WRONG panel with every gate
    # green and no record of it. That variable is an explicit sensitivity-band tool only; forbid it
    # on a real-spend headline run (rehearsals pass --allow-unfrozen).
    if args.pass_mode.upper() == "B" and not args.synthetic and not args.allow_unfrozen:
        import os as _os

        _ovr = _os.environ.get("LLM_RP_GOLD_SUFFIX", "").strip()
        if _ovr:
            raise SystemExit(
                f"LLM_RP_GOLD_SUFFIX={_ovr!r} is set on a real-spend headline run — it OUTRANKS the "
                f"freeze-bound config/data.yaml gold.suffix and would run the campaign on the wrong "
                f"panel silently. Unset it (the frozen panel governs), or pass --allow-unfrozen for "
                f"an explicit sensitivity-band rehearsal.")

    # 2026-07-18 (the B*-resolution catch, GENERALIZED to the class): on a REAL-SPEND run an
    # EXPLICIT design-value flag overrides the pre-registered/frozen configuration — only
    # rehearsals (--allow-unfrozen) may do that. Uses the pre-forcing capture so the H3 block's
    # programmatic generations never trips it.
    if args.pass_mode.upper() == "B" and not args.synthetic and not args.allow_unfrozen:
        _overridden = [n for n, was_set in _explicit_design.items() if was_set]
        if _overridden:
            raise SystemExit(
                f"explicit design override(s) {_overridden} on a real-spend run: the confirmatory "
                f"campaign must NOT pass --train-steps/--candidates/--generations/--n-trials/"
                f"--embargo (the frozen campaign.yaml/inference.yaml values are resolved and "
                f"asserted); rehearsals pass --allow-unfrozen.")

    # DEFAULTS-CLASS RESOLUTION (2026-07-18): fill every None design value from the SAME config
    # keys the laptop main reads (run_campaign.py:2134-2151) — the freeze hash binds these files,
    # so config-read IS the frozen truth; a hardcoded argparse mirror is drift waiting to fire.
    from src.utils.config import cfg_get as _rescfg
    _camp_cfg = _load_config("campaign")
    if args.candidates is None:
        args.candidates = int(_rescfg(_camp_cfg, "candidates_per_arm", 0) or 0)
        if not args.candidates:
            raise SystemExit("campaign.yaml candidates_per_arm missing — candidate budget unresolved")
        _prereg_budget = int(_rescfg(_load_config("preregistration"), "matched_budget", 0) or 0)
        if _prereg_budget and args.candidates != _prereg_budget:
            raise SystemExit(
                f"campaign.yaml candidates_per_arm ({args.candidates}) != pre-registered "
                f"matched_budget ({_prereg_budget}) — mirror drift; fix the configs (a dated "
                f"amendment) before launching")
    if args.generations is None:
        args.generations = int(_rescfg(_rescfg(_camp_cfg, "llm", {}) or {}, "generations", 1) or 1)
    if args.n_trials is None:
        args.n_trials = int(args.candidates)  # laptop parity: DSR expected-max = per-arm candidates
    if args.embargo is None:
        args.embargo = int(_rescfg(_rescfg(_load_config("inference"), "splits", {}) or {},
                                   "embargo_trading_days", 21))
    _LOG.info("design values RESOLVED: candidates=%d generations=%d n_trials=%d embargo=%d "
              "train_steps=%s", args.candidates, args.generations, args.n_trials, args.embargo,
              args.train_steps if args.train_steps is not None else "campaign.yaml (assembly)")

    # P17/A3-F12 (2026-07-13 audit): --hold-at-gate is meaningless without the review gate —
    # the hold would be SILENTLY ignored and C4 would launch unreviewed.
    if args.hold_at_gate and args.no_review_gate:
        raise SystemExit("--hold-at-gate requires the review gate, but --no-review-gate disables "
                         "it — drop one of the two flags.")
    # P17/A3-F9 (2026-07-13 audit): the search authors candidates//generations per generation
    # (floor division, laptop-parity). A non-divisible pair silently DROPS the remainder from
    # the budget — e.g. a D1 curve level would train fewer candidates than its x-axis claims.
    if args.candidates % max(1, args.generations) != 0:
        _dropped = args.candidates - (args.candidates // max(1, args.generations)) * max(1, args.generations)
        raise SystemExit(
            f"--candidates {args.candidates} is not divisible by --generations "
            f"{args.generations}: {_dropped} candidate(s) would be SILENTLY dropped from the "
            f"budget (candidates-per-generation is floor division, laptop-parity). Pick a "
            f"divisible pair (e.g. --generations 1 for small D1 curve levels)."
        )

    # ---- 2026-07-13 PRE-SPEND AUDIT GUARDS (all fail LOUD before any submission/authoring) ---- #
    from src.utils.config import cfg_get as _cfg_get
    llm_cfg = None  # None -> assemble falls back to prototype.yaml (the legacy rehearsal path)
    if leg_llm_cfg is not None:
        llm_cfg = leg_llm_cfg  # v2 leg: the pinned author block from config/legs.yaml wins
    elif args.llm_from == "campaign":
        llm_cfg = dict(_cfg_get(_load_config("campaign"), "llm", {}) or {})
    # AUDIT 2026-07-22 (MAJOR, R83): the cluster path never set spend_ledger -> LLMClient's guard
    # silently recorded NOTHING for the whole campaign. Per-LINE namespaced ledgers (batch_tag)
    # also avoid the cross-process torn-append undercount on a shared file; report-time spend =
    # the sum over spend_ledger_*.jsonl.
    if llm_cfg is not None:
        # Audit 2026-07-24 (minor): emptiness must be tested BEFORE the ledger key is injected —
        # setdefault made the dict non-empty, so this guard was unreachable dead code.
        if not llm_cfg:
            raise SystemExit("--llm-from campaign but config/campaign.yaml has no `llm` block")
        import re as _re_tag
        _tag = _re_tag.sub(r"[^A-Za-z0-9_]", "_", str(args.batch_tag or "core"))
        llm_cfg.setdefault("spend_ledger",
                           str(Path(args.output_dir) / f"spend_ledger_{_tag}.jsonl"))
    # F1 (agent audit, CRITICAL): pass-mode B with the DEFAULT provider=stub silently authors the
    # whole "Opus" campaign with the keyless stub and completes "successfully". Derive the provider
    # from the resolved llm block when possible (campaign.yaml carries provider: anthropic), else
    # fail loud — a real-spend run must never fall through to the stub.
    if args.pass_mode.upper() == "B" and args.provider == "stub":
        _blk = llm_cfg if llm_cfg is not None else _cfg_get(_load_config("prototype"), "llm", {})
        _blk_provider = str(_cfg_get(_blk, "provider", "") or "")
        if _blk_provider and _blk_provider != "stub":
            args.provider = _blk_provider
            _src = f"leg:{args.leg}" if leg_llm_cfg is not None else args.llm_from
            _LOG.info("provider DERIVED from the %s llm block: %s", _src, args.provider)
        else:
            raise SystemExit(
                "pass-mode B with provider=stub: a real-spend run would silently author with the "
                "keyless stub designer. Pass --provider explicitly (or --llm-from campaign, whose "
                "block carries provider: anthropic)."
            )
    # F2 (agent audit, CRITICAL): restarting WITHOUT --resume re-authors every candidate (paid)
    # while archive-truth training discards the new sources — the full authoring budget wasted.
    if not args.resume and not args.dry_run:
        _dirty = list(Path(args.output_dir).glob("search*/*/*/record.json"))  # audit 2026-07-22: search_leg_*/search_h3_* too
        if _dirty:
            raise SystemExit(
                f"{len(_dirty)} search records already exist under {args.output_dir} but --resume "
                "was not passed. A fresh start here RE-AUTHORS every candidate (paid API calls) and "
                "then DISCARDS the new sources (training is archive-truth). Pass --resume, or move "
                "the output dir aside if you truly want a fresh run."
            )
    # Provider<->model consistency: authoring with a mismatched pair either 400s (burning retries)
    # or silently authors with the wrong family — both unacceptable at real spend.
    if args.pass_mode.upper() == "B" and args.provider != "stub":
        _block = llm_cfg if llm_cfg is not None else _cfg_get(_load_config("prototype"), "llm", {})
        _model = str(_cfg_get(_block, "model_snapshot", ""))
        if args.provider == "anthropic" and not _model.startswith("claude"):
            raise SystemExit(f"provider=anthropic but the {args.llm_from} llm block's model is "
                             f"{_model!r} — wrong-model spend guard (pass --llm-from correctly)")
        if args.provider == "openrouter" and _model.startswith("claude"):
            raise SystemExit(f"provider=openrouter with model {_model!r} — the campaign authors "
                             "Opus via the NATIVE anthropic provider (ADR-035)")

    inputs = assemble_cluster_inputs(
        arms=list(args.arms), seeds=_parse_seeds(args.seeds), output_dir=args.output_dir,
        synthetic=bool(args.synthetic), train_steps=args.train_steps, n_trials=args.n_trials,
        candidates=args.candidates, generations=args.generations, search_seed=args.search_seed,
        embargo=args.embargo, pass_mode=args.pass_mode, provider=args.provider, llm_cfg=llm_cfg,
        resume=bool(args.resume),
    )
    # R97 fail-before-ssh guard (mirrors run_campaign.py --baselines), placed ABOVE the dry-run
    # exit so the keyless pre-flight validates baseline names too (audit 2026-07-22: the guard
    # originally sat after the dry-run return — live for real launches, dead for dry-runs).
    # 2026-07-26: widened from a name check to full frozen-family resolution — see
    # resolve_cluster_baselines() for why a hand-typed partial list was a launch-breaking defect.
    _resolved_baselines = resolve_cluster_baselines(args.baselines, tiered=bool(args.tiered),
                                                    leg=args.leg)
    # DEVICE COHERENCE (2026-07-26, added WITH the CPU lane so the lane cannot create an
    # incoherent run). --seed-pool-blocks stripes seeds across GPU POOLS (EF/L/U/V) to keep every
    # CRN pair device-homogeneous; under --device cpu those pool names denote nothing the job
    # requests, so the stripe would silently claim a device stratification the run does not have.
    # CPU and CUDA are NOT bit-identical, so a false homogeneity claim is a reproducibility defect,
    # not a cosmetic one. Fail loud at the CLI boundary rather than mislabel the archive.
    if args.device == "cpu" and args.seed_pool_blocks:
        raise SystemExit(
            "--device cpu is incompatible with --seed-pool-blocks: the blocks stratify seeds "
            "across GPU pools (EF/L/U/V) for CRN device-homogeneity, but a CPU job requests no "
            "GPU and pins no pool, so the stripe would assert a stratification that does not "
            "exist. Drop --seed-pool-blocks for a CPU run (a CPU run is device-homogeneous by "
            "construction).")
    if getattr(args, "canary", None):
        # Same guard for --canary names (audit 2026-07-22: they bypassed validation → fail-after-submit).
        from src.baselines.rewards import REWARD_CANON as _RC2
        _unknown_c = [b for b in args.canary if b not in _RC2]
        if _unknown_c:
            raise SystemExit(
                f"--canary: unknown REWARD_CANON key(s) {_unknown_c}; valid: {sorted(_RC2)}")

    if args.dry_run:
        # No ssh in a dry-run: expand a leading '~' against a documented STUB home so the render
        # is representative and passes the tilde-free jobscript contract (2026-07-11 incident).
        # 2026-07-13 (launch readiness): the dry-run now ALSO validates the two launch-critical
        # configs that previously only failed at real launch — the seed-pool block spec and the
        # tiered seeds schema — so the keyless pre-flight step 3 exercises the WHOLE launch shape.
        if args.seed_pool_blocks:
            from src.cluster.campaign import parse_seed_pool_blocks as _pspb
            _blocks = _pspb(args.seed_pool_blocks)  # fail-loud on overlap/shape
            _LOG.info("dry-run: seed-pool blocks OK — %s",
                      {p: len(sd) for p, sd in _blocks})
        if args.tiered:
            from src.utils.seeds import seed_tiers as _st
            _tiers = _st(_load_config("campaign").get("seeds"))  # fail-loud on a bad schema
            _LOG.info("dry-run: tiered seeds schema OK — %d tiers, sizes %s",
                      len(_tiers), [len(t) for t in _tiers])
        if args.search_pack is not None:
            # MODE-D lane validation belongs in the keyless pre-flight too: size the search
            # walltime here so a bad --search-pack/B* combination fails OFFLINE, not at launch.
            _dr_steps = int(args.train_steps
                            or _cfg_get(_load_config("campaign"), "train_steps_per_candidate", 0)
                            or 0)
            if not _dr_steps:
                raise SystemExit("--search-pack: B* unresolved — refusing to size the search lane")
            # device= is NOT optional here (2026-07-27): the dry-run is the operator's only offline
            # look at the walltime, and without it this line printed the GPU-curve number for a CPU
            # launch — an instrument quietly reporting a value the real run would not use. It is
            # the same omission, in the same file, that made the real sizing kill every job.
            _LOG.info("dry-run: MODE-D search lane OK — pack=%d h_rt=%s (bursts pack=%d h_rt=%s)%s",
                      args.search_pack,
                      autosize_h_rt(int(args.search_pack), _dr_steps, device=args.device),
                      args.pack, autosize_h_rt(int(args.pack), _dr_steps, device=args.device),
                      "; pipelined rungs ON" if args.pipeline_rungs else "")
        stub = "/home/USER"
        return _dry_run(inputs, list(args.arms),
                        remote_root=expand_remote(args.remote_root, stub),
                        gold_dir=expand_remote(args.gold_dir, stub),
                        pool=args.pool, pack=args.pack)

    from src.cluster.campaign import build_cluster_run, parse_seed_pool_blocks, run_campaign_on_cluster

    # 2026-07-11 incident fix: '~' survives LITERALLY through the quoted ssh runner, SGE '#$'
    # directives, and the jobscript's quoted strings (the rehearsal arrays Eqw-died at dispatch
    # and were admin-purged without a qacct trace). Resolve the REAL remote home once and expand
    # every user-supplied remote path before anything is rendered, pushed, or submitted.
    if any(str(p).startswith("~") for p in (args.remote_root, args.gold_dir, args.apptainer_sif or "")):
        home = remote_home(ssh_runner(args.host))
        args.remote_root = expand_remote(args.remote_root, home)
        args.gold_dir = expand_remote(args.gold_dir, home)
        if args.apptainer_sif:
            args.apptainer_sif = expand_remote(args.apptainer_sif, home)
        _LOG.info("remote '~' paths expanded against home=%s", home)

    # ══════════════════════════════════════════════════════════════════════════════════════════
    # PRE-SUBMISSION PRECONDITIONS (2026-07-27 launch gate). All three run exactly ONCE, here, on
    # the laptop, after '~' expansion and before build_cluster_run — i.e. before a single qsub.
    # Each closes a failure that was silent, uniform and LATE: discovered per-task, thousands of
    # times, hours in, with no single loud cause.
    # ══════════════════════════════════════════════════════════════════════════════════════════
    _real_spend = args.pass_mode.upper() == "B" and not args.synthetic

    # ---- PRECONDITION 1: the remote working roots must EXIST before the driver's first poll ---- #
    # COLD-START DEADLOCK (2026-07-27): the driver's first action every cycle is a pull, and
    # `poll.remote_completed_dirs` runs `find <remote_outputs_root> -name record.json`, which GNU
    # find exits 1 on for a MISSING directory -> CalledProcessError -> RuntimeError, which is in
    # `driver._TRANSPORT_ERRORS`, so the driver sleeps and retries. The only code that CREATES that
    # directory is `prepare_remote`, which lives inside `submit_batch` — i.e. AFTER the pull it can
    # never reach. On a virgin remote root the driver would poll for the full 12 h transport-outage
    # budget and then die having submitted NOTHING. `remote_completed_dirs`' own docstring asserts
    # "prepare_remote pre-creates outputs/", and that ordering does not hold on a cold start.
    # One idempotent mkdir removes the whole class. (The campaign's own root already exists, so this
    # is insurance for a fresh --remote-root, e.g. every rehearsal.)
    _rr = args.remote_root.rstrip("/")
    # argv LIST, not a string — see the note in assert_remote_gold.
    ssh_runner(args.host)(
        ["bash", "-c", f"mkdir -p '{_rr}/outputs' '{_rr}/ledger' '{_rr}/specs' '{_rr}/logs'"])
    _LOG.info("remote working roots ensured under %s", _rr)

    # ---- PRECONDITION 2: the licensed gold must BE there, at the FROZEN bytes ---- #
    assert_remote_gold(ssh_runner(args.host), args.gold_dir, real_spend=_real_spend)

    # ---- PRECONDITION 3: no FOREIGN records under THIS run's archive roots ---- #
    # The three namespaces mirror how the roots are actually built below (--leg forces
    # root_suffix=leg_<label>, so every leg line is covered by the middle branch).
    if args.h3_singleshot:
        _roots = [f"{b}_h3_singleshot" for b in ("search", "test", "frozen")]
    elif args.root_suffix:
        _roots = [f"{b}_{args.root_suffix}" for b in ("search", "test", "frozen")]
    else:
        _roots = ["search", "test", "frozen"]
    assert_no_foreign_remote_records(
        ssh_runner(args.host), f"{args.remote_root.rstrip('/')}/outputs", args.output_dir, _roots,
        real_spend=_real_spend, batch_tag=args.batch_tag,
    )

    # ---- 2026-07-13 pre-spend audit: safe DEFAULTS for the two silent money sinks ---- #
    # (a) Spend cap: --max-author-calls defaulted to None = the spend_guard was a NO-OP on an
    #     unattended run. Default it to 2x the structural authoring bound + slack, logged.
    #     The LLM-arm set is DERIVED from config/arms.yaml (2026-07-27), not hand-listed: a
    #     hand-listed copy of a frozen roster is exactly the drift resolve_cluster_arms() closes,
    #     and here it would silently mis-size the spend cap if a feedback arm were ever seated.
    _LLM_ARMS = set(llm_arm_roster())
    if args.max_author_calls is None and args.pass_mode.upper() == "B" and args.provider != "stub":
        n_llm = len(set(args.arms) & _LLM_ARMS)
        args.max_author_calls = max(30, 2 * n_llm * int(args.candidates) + 60)
        _LOG.info("spend cap DEFAULTED: max_author_calls=%d (2 x %d LLM arms x %d candidates + 60 "
                  "slack); override with --max-author-calls", args.max_author_calls, n_llm,
                  args.candidates)
    # (b) Walltime: the renderer's pack>1 default (1:30) was sized for 50k probes — a pack-5 B*
    #     task needs ~2.4h on a contended node and would be h_rt-KILLED after burning the GPU
    #     (the p6ext800/1600 incident class). Size on the measured aggregate curve at its worst
    #     (x0.5 contention) + overhead + 30% margin.
    if args.h_rt is None:
        # 2026-07-18 DEFAULTS-CLASS catch #2 (launch-critical): this read
        # campaign.agent.train_steps_per_candidate — a key that DOES NOT EXIST (B* is top-level)
        # — then fell to a stale hardcoded 200000. At B*=400k the auto-h_rt would have been sized
        # for 200k (~4h vs the ~6:09 a pack-5 400k task needs) and EVERY chunked array task would
        # have been walltime-killed after burning ~4 GPU-h. Read the SAME top-level key the
        # assembly resolves, and fail loud rather than guess.
        steps = int(args.train_steps
                    or _cfg_get(_load_config("campaign"), "train_steps_per_candidate", 0) or 0)
        if not steps:
            raise SystemExit("h_rt autosize: B* unresolved (campaign.yaml "
                             "train_steps_per_candidate missing) — refusing to guess a walltime")
        args.h_rt = autosize_h_rt(int(args.pack), steps, device=args.device)
        _LOG.info("walltime DEFAULTED: h_rt=%s (steps=%d, pack=%d, device=%s); override with --h-rt",
                  args.h_rt, steps, args.pack, args.device)

    # MODE-D phase-adaptive packing (2026-07-21): search waves (the 6-deep reflection critical
    # path) run at --search-pack with a MATCHING auto-sized tight walltime — short low-pack
    # requests are prime backfill and roughly halve every chain's latency; burst work keeps
    # --pack's throughput. Ops-only: identical seeds/steps/maths.
    search_h_rt = None
    if args.search_pack is not None:
        _sp_steps = int(args.train_steps
                        or _cfg_get(_load_config("campaign"), "train_steps_per_candidate", 0) or 0)
        if not _sp_steps:
            raise SystemExit("--search-pack: B* unresolved — refusing to size the search walltime")
        search_h_rt = autosize_h_rt(int(args.search_pack), _sp_steps, device=args.device)
        _LOG.info("MODE-D search lane: pack=%d h_rt=%s (bursts stay pack=%d h_rt=%s)",
                  args.search_pack, search_h_rt, args.pack, args.h_rt)

    remote_root = args.remote_root.rstrip("/")
    # local_archive_root == output_dir so the pulled mirror is output_dir/{search,test,frozen}/... —
    # EXACTLY the laptop campaign's layout, so analyze_campaign reads the cluster archive identically
    # (parity). Batches/logs live under output_dir/batches, disjoint from the archive dirs.
    run = build_cluster_run(
        remote_root=remote_root, remote_outputs_root=f"{remote_root}/outputs",
        local_batch_root=f"{args.output_dir}/batches", local_archive_root=args.output_dir,
        gold_dir=args.gold_dir, host=args.host, pool_confirmatory=args.pool, pack=args.pack,
        chunk_tasks=args.chunk_tasks, specs_per_task=args.specs_per_task,
        poll_secs=args.poll_secs, max_author_calls=args.max_author_calls, concurrent=True,
        apptainer_sif=(args.apptainer_sif or None), cores_per_training=args.cores_per_training,
        h_rt=(args.h_rt or None),
        seed_pool_blocks=(parse_seed_pool_blocks(args.seed_pool_blocks)
                          if args.seed_pool_blocks else None),
        batch_tag=(args.batch_tag or None),
        search_threads=args.search_threads,
        exclude_hosts=([h.strip() for h in args.exclude_hosts.split(',') if h.strip()]
                       if args.exclude_hosts else None),
        search_pack=args.search_pack, search_h_rt=search_h_rt,
        search_poll_secs=args.search_poll_secs, device=args.device,
    )
    baselines = _resolved_baselines  # frozen-family resolved + validated pre-dry-run (R97)

    if args.root_suffix:
        # C6-class APPLICATION: namespaced roots + batch names for report-only re-search
        # invocations (the format + h3-conflict VALIDATION already ran early, pre-ssh).
        from dataclasses import replace as _dc_replace

        _sfx = args.root_suffix
        _base_rb = run.run_batch

        def _sfx_run_batch(specs, name, **kw):
            return _base_rb(specs, f"{_sfx}_{name}", **kw)

        run = _dc_replace(run, run_batch=_sfx_run_batch,
                          search_subdir=f"search_{_sfx}", test_subdir=f"test_{_sfx}")
        inputs["frozen_root"] = Path(args.output_dir) / f"frozen_{_sfx}"
        _LOG.info("root-suffix %r: archives -> search_%s/ test_%s/ frozen_%s/; batches %s_*",
                  _sfx, _sfx, _sfx, _sfx, _sfx)

    if args.h3_singleshot:
        # C5: the H3 single-shot control — disjoint *_h3_singleshot roots, h3ss_ batch names,
        # -p -100. Its sentinel is its OWN file (h3_singleshot_summary.json): clobbering the
        # HEADLINE campaign_summary.json in the shared output dir would fool the watcher/analyze.
        from src.cluster.campaign import run_h3_singleshot_on_cluster

        out = run_h3_singleshot_on_cluster(
            inputs["opts"], inputs["seeds"], run,
            test_leg_kwargs=inputs["test_leg_kwargs"],
            frozen_root=Path(args.output_dir) / "frozen_h3_singleshot",
            search_seed=args.search_seed, resume=bool(args.resume),
            priority=args.priority,  # audit 2026-07-22: the ladder re-run passes --priority -300
        )
        ok = bool(out.get("ok"))
        _write_campaign_summary(args.output_dir, inputs, freeze_stamp=freeze_stamp, extra={
            "h3_singleshot": True, "all_arms_tested": ok, "exit_code": 0 if ok else 1,
            "winner_id": out.get("winner_id"),
        }, filename="h3_singleshot_summary.json")
        print(f"[campaign] H3 SINGLE-SHOT {'OK' if ok else 'INCOMPLETE'} — "
              f"winner={out.get('winner_id')} ({out.get('reason', 'tested at the campaign seeds')})")
        return 0 if ok else 1

    if args.tiered:
        from src.cluster.campaign import run_campaign_tiered
        from src.utils.config import load_config

        if args.approve_tier1:
            approval = Path(args.output_dir) / "TIER1_APPROVED"
            approval.parent.mkdir(parents=True, exist_ok=True)
            approval.write_text("approved via --approve-tier1\n", encoding="utf-8")
            _LOG.info("review gate approved: %s", approval)
        canary = args.canary if args.canary is not None else (baselines or [])[:3]
        seeds_cfg = load_config("campaign").get("seeds")  # the schema (list -> one tier; tiered -> ladder)
        out = run_campaign_tiered(
            list(args.arms), inputs["opts_for"], seeds_cfg, run,
            test_leg_kwargs=inputs["test_leg_kwargs"], frozen_root=inputs["frozen_root"],
            baseline_names=baselines, canary_baselines=(canary or None),
            review_gate=not args.no_review_gate, hold_at_gate=bool(args.hold_at_gate),
            resume=bool(args.resume), pipeline_rungs=bool(args.pipeline_rungs),
        )
        if out.get("awaiting_review"):
            reason = out.get("gate", "review")
            why = ("a REAL execution defect (a short/incomplete unit or device inhomogeneity) — "
                   "inspect and fix it before releasing" if reason == "RED-execution-health"
                   else "you asked to hold for a manual eyeball (--hold-at-gate)")
            print(f"[campaign] C3 FLOOR COMPLETE, gate STOPPED: {why}. Review the EFFECT-BLIND report "
                  f"({out['integrity_report']}), then re-run with --approve-tier1 --resume. "
                  f"(On green health without --hold-at-gate the gate auto-proceeds — no manual wait.)")
            # ── D12 (applied 2026-08-01, record §97) ────────────────────────────────────────────
            # A GATE STOP IS NOT A SUCCESS. This returned 0, so the supervisor's `if ($rc -eq 0)`
            # logged "LINE COMPLETE" and EXITED the line — six legs reported complete on 2026-07-29
            # having produced nothing, and only the watchdog's 300 s revive loop kept them alive.
            #
            # NOTE: APPLIED NOW BECAUSE D16 MADE IT URGENT, not merely overdue. D16 (same commit) folds
            # the substrate census into `health_ok`, which makes gate stops MORE likely — and a stop
            # that reports success would have turned the confirmatory line into a silent 300-second
            # relaunch loop logging "LINE COMPLETE" on every pass. The two fixes are hard-coupled and
            # neither is safe to ship without the other.
            #
            # VERIFIED before changing it: the watchdog decides "dead line" by process ABSENCE, not
            # by exit code (scripts/mode_d_watchdog.ps1 polls the
            # process table), so it is unaffected by a new code — the deferred-fix note asked for
            # that to be checked rather than assumed.
            return 3   # EXIT_AWAITING_REVIEW
        ok = bool(out.get("ok"))
        # row 30n/C7 (audit 2026-07-22): a --tiered --root-suffix combo previously clobbered the
        # HEADLINE campaign_summary.json (only the non-tiered path namespaced). Mirror it.
        _tiered_fname = (f"campaign_summary_{args.root_suffix}.json" if args.root_suffix
                         else "campaign_summary.json")
        _write_campaign_summary(args.output_dir, inputs, freeze_stamp=freeze_stamp, extra={
            "tiered": True, "all_arms_tested": ok, "exit_code": 0 if ok else 1,
            "n_tiers": out.get("n_tiers"), "tier_sizes": out.get("tier_sizes"),
        }, filename=_tiered_fname)
        print(f"[campaign] TIERED {'OK' if ok else 'INCOMPLETE'} — "
              f"{out['n_tiers']} tiers, sizes {out['tier_sizes']}")
        return 0 if ok else 1

    results = run_campaign_on_cluster(
        list(args.arms), inputs["opts_for"], inputs["seeds"], run,
        test_leg_kwargs=inputs["test_leg_kwargs"], frozen_root=inputs["frozen_root"],
        baseline_names=baselines, resume=bool(args.resume),
        priority=(args.priority if args.priority is not None else 0),
    )
    # ``all({})`` is True: an empty results dict would score a no-op run as ALL OK / exit 0. The
    # library refuses a zero-unit call outright; this is the same guard at the point of USE, so the
    # exit code can never claim success for a campaign that tested nothing (2026-07-26 review).
    ok = bool(results) and all(r.get("ok") for r in results.values())
    for arm, r in results.items():
        _LOG.info("[%s] ok=%s %s", arm, r.get("ok"),
                  {k: v for k, v in r.items() if k not in ("search", "test")})
    # MODE-D audit catch (2026-07-21): a --root-suffix invocation (every LEG line; the C6 dose
    # class) sharing --output-dir would CLOBBER the headline campaign_summary.json — the exact
    # hazard the H3 path already guards against for itself (its own comment warns it). Namespace
    # the summary per suffix, exactly like the archive roots.
    _summary_name = (f"campaign_summary_{args.root_suffix}.json" if args.root_suffix
                     else "campaign_summary.json")
    _write_campaign_summary(args.output_dir, inputs, freeze_stamp=freeze_stamp, extra={
        "tiered": False, "all_arms_tested": ok, "exit_code": 0 if ok else 1,
        "arms": {arm: bool(r.get("ok")) for arm, r in results.items()},
        **({"root_suffix": args.root_suffix} if args.root_suffix else {}),
    }, filename=_summary_name)
    print(f"[campaign] {'ALL OK' if ok else 'INCOMPLETE'} — "
          f"{sum(1 for r in results.values() if r.get('ok'))}/{len(results)} arms")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
