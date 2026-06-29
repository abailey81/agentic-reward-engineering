"""Advanced 40-candidate prototype orchestrator (MASTER_EXECUTION_PLAN P5).

Runs the six arms (``distributional, scalar, placebo, scalar_cvar5, random_search, bayes_opt``)
on the FIXED SB3-SAC agent at a matched per-arm budget, with **arm-level parallelism** across
NON-daemon worker processes (so the sandbox's killable validate-once child can spawn; the C2
inline fallback is the backstop). The LLM arms run through ``run_loop`` with the keyless
``StubDesignerTransport`` (Pass A) or a real provider transport (Pass B); the search arms run
through the C1 evaluator (``src.agents.evaluator``) + the live H4 reward family
(``src.baselines.reward_family``), so all six arms consume the identical train->rollout->select
pipeline and "matched compute" actually holds (PREREGISTRATION §3; review C1/M2).

DIRECTIONAL / plumbing-only: per review M3/M4 (liquidate_to_cash fill + held 2005 cohort bias the
measured tails), NO prototype number enters the dissertation. Built + verified by a tiny dry run;
the full run is gated on the user.

Usage
-----
    python scripts/run_prototype.py --dry-run            # tiny synthetic verification (ALWAYS keyless stub)
    python scripts/run_prototype.py                      # full prototype; Pass/provider per config/prototype.yaml
    python scripts/run_prototype.py --pass A             # FORCE the keyless stub designer (no API key, no cost)
    python scripts/run_prototype.py --pass B             # FORCE the real reward-author (config default: Claude Sonnet 4.6)
    python scripts/run_prototype.py --arms distributional,scalar

Note: the in-repo default Pass/provider come from config/prototype.yaml (currently Pass B / Sonnet 4.6), so a
bare run is billed; --pass A forces the keyless stub, and --dry-run is always keyless regardless of config.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src.utils.config import cfg_get, load_config

# Schema arms that train an LLM-designed reward via the discovery loop.
_LLM_ARMS = ("distributional", "scalar", "placebo", "scalar_cvar5", "placebo_shuffled")
_SEARCH_ARMS = ("random_search", "bayes_opt")


def _env_fp_label(synthetic: bool, steps: Any, *, n_assets: int | None = None) -> str:
    """Provenance label for a run's ``env_fp`` naming the REAL active panel (C5).

    For a real-gold run the panel token is ``gold_suffix()`` (reads the freeze-bound
    ``config/data.yaml: gold.suffix``), NOT a hardcoded ``'univ3'`` — so a new-suffix rebuild is
    reflected in archived records. ``n_assets`` adds the ``:N<k>`` universe token when known.
    """
    if synthetic:
        panel = "synthetic"
    else:
        from src.data.loaders import gold_suffix

        panel = gold_suffix()
    n_token = f":N{n_assets}" if n_assets is not None else ""
    return f"{panel}{n_token}:steps{steps}"


# --------------------------------------------------------------------------- #
# Panel + windows                                                             #
# --------------------------------------------------------------------------- #
def _load_panel_and_windows(synthetic: bool, data_cfg: Any, lookback: int):
    """Return ``(panel, train_window, val_window)`` for the prototype (real or synthetic)."""
    import numpy as np

    if synthetic:
        from src.data.synthetic import make_synthetic_panel

        panel = make_synthetic_panel(n_assets=30, n_days=600, seed=0)
        # Index windows: train [lookback, 400), val [400, 600).
        return panel, (lookback, 400), (400, panel.T)

    from src.data.loaders import embargoed_val_start, load_gold_panel

    phase = str(cfg_get(data_cfg, "phase", "development"))
    val_end = str(cfg_get(data_cfg, "val_end", "2019-12-31"))
    train_end = str(cfg_get(data_cfg, "train_end", "2016-12-31"))
    # Embargo fallback = the canonical config/data.yaml floor, NOT a bare literal 21 (no-hardcoding audit).
    _emb_floor = int(load_config("data").get("embargo_days", 21))
    embargo_days = int(cfg_get(data_cfg, "embargo_days", _emb_floor))
    on_missing = str(cfg_get(data_cfg, "on_missing", "liquidate_to_cash"))
    res = load_gold_panel(phase=phase, end=val_end, on_missing=on_missing)
    panel = res.panel
    dates = np.asarray(panel.dates)
    # Train ends at ``train_end``; validation begins at the PURGED boundary (PREREGISTRATION §7, R18).
    # ``embargoed_val_start(lookback=lookback)`` returns max(materialized boundary — stale pre-Split-C
    # 2015-02-03, inert under the Split-C train_end — first_post_train + max(embargo, lookback)); with
    # lookback=60 the lookback purge dominates so val starts 2017-03-30 (train_end + 60 sessions). The
    # purged (train_end, val_start) gap is dropped.
    # side='right': half-open train end — identical to the old searchsorted+1 when train_end IS a
    # session; correct when it is not (SPLIT C's 2016-12-31 is a Saturday — +1 leaked 2017-01-03).
    train_split = int(np.searchsorted(dates, np.datetime64(train_end), side="right"))
    train_split = max(lookback + 1, min(train_split, panel.T - 1))
    val_split = embargoed_val_start(
        dates, train_end, phase=phase, embargo_days=embargo_days, lookback=lookback
    )
    val_split = max(train_split, min(val_split, panel.T - 1))
    return panel, (lookback, train_split), (val_split, panel.T)


# --------------------------------------------------------------------------- #
# Builders                                                                    #
# --------------------------------------------------------------------------- #
def _agent_cfg(proto_cfg: Any, train_steps: int | None) -> dict[str, Any]:
    from src.agents.factory import campaign_replay_cap  # local: factory lazy-imports torch (keep top-level torch-free)

    a = cfg_get(proto_cfg, "agent", {})
    steps = int(train_steps if train_steps is not None else cfg_get(a, "train_steps_per_candidate", 25000))
    return {
        "train_steps_per_candidate": steps,
        # buffer = min(train_steps, HARD cap) — the SAME rule as the TEST leg / resolve_agent_kwargs, so the
        # winner is SELECTED under the replay dynamics it is EVALUATED under. At the prototype's 25k steps this
        # is min(25000, 50000)=25000 (== train_steps, prototype.yaml's intent preserved); at the campaign's 50k/
        # 200k budget it yields 50000, eliminating the serial-SEARCH-25k vs TEST-50k skew (prototype.yaml pins
        # buffer_size=25000, which we intentionally ignore in favour of the step-coupled+capped value).
        "buffer_size": min(int(steps), campaign_replay_cap()),
        "batch_size": int(cfg_get(a, "batch_size", 256)),
        "learning_rate": float(cfg_get(a, "learning_rate", 3e-4)),
        "gamma": float(cfg_get(a, "gamma", 0.99)),
        "ent_coef": cfg_get(a, "ent_coef", "auto"),
        # learning_starts: warm-up steps before the critic regresses. SB3's UNSET default is 100, but the
        # Phase-0 GATE validated 1000 — thread the config value (default 1000) so the serial/SEARCH/campaign
        # paths all train under the gated warmup, not a silent SB3-100 (resolve_agent_kwargs floors at 1000).
        "learning_starts": int(cfg_get(a, "learning_starts", 1000)),
        # PopArt value-target scale normalization (src/agents/popart.py): config-driven (default on); the
        # critic is made invariant to the reward SCALE so a variance-floored Sharpe can't explode the loss.
        "popart": bool(cfg_get(a, "popart", True)),
        "popart_beta": float(cfg_get(a, "popart_beta", 1e-3)),
        "popart_min_scale": float(cfg_get(a, "popart_min_scale", 1.0)),
        "popart_warmup": int(cfg_get(a, "popart_warmup", 0)),
        "normalize_obs": bool(cfg_get(a, "normalize_obs", True)),
        "tf32": bool(cfg_get(a, "tf32", True)),  # config-driven matmul precision (train_agent applies it; default on)
        # final-audit #26: honor the configured device on the sequential/campaign-search path (was
        # dropped, so the trainer silently fell back to SB3 'auto' instead of the configured value).
        "device": cfg_get(a, "device", "auto"),
    }


def build_parallel_opts(
    structural_cfg: Any,
    env_cfg: Any,
    *,
    llm_block: Any,
    train_steps: int,
    n_trials: int,
    synthetic: bool,
    seed: int,
    candidates: int,
    generations: int,
    pass_mode: str,
    provider: str,
    resume: bool = False,
    max_tasks_per_child: Any = None,
) -> dict[str, Any]:
    """Assemble the ``opts`` dict consumed by ``run_parallel`` / ``train_candidate`` / ``_drive_llm_arm``.

    SHARED by the prototype ``--parallel`` path and the campaign ``--search-gpu`` path so the two
    cannot drift. ``structural_cfg`` supplies the ``agent`` (batch_size, normalize_obs), ``reward_family``
    (cvar_alpha, window) and ``data`` blocks; ``llm_block`` is the reward-author block
    (``model_snapshot`` / ``api_key_env`` / ``temperature`` / ``diversity_prompt_variation``) — the
    prototype's own ``llm`` or the campaign's Opus block. ``train_steps`` is threaded straight into the
    worker, which couples ``buffer_size == train_steps`` (``parallel.train_candidate``) — so passing the
    campaign's 50k both matches the TEST leg AND resolves the serial-search 25k-buffer skew.
    """
    from src.llm.client import default_key_env

    agent = cfg_get(structural_cfg, "agent", {})
    rf = cfg_get(structural_cfg, "reward_family", {})
    steps = int(train_steps)
    return {
        "train_steps": steps,
        "batch_size": int(cfg_get(agent, "batch_size", 256)),  # ONE canonical default (SB3); 512 -> 256/512 drift
        "normalize_obs": bool(cfg_get(agent, "normalize_obs", True)),
        # Thread the FULL agent block so the parallel SEARCH worker (parallel.train_candidate) honors the
        # campaign config for these too — parity with the serial path + the parallel TEST worker, not just
        # train_agent's defaults (they coincide today, so this is behaviour-preserving; closes a latent skew).
        "learning_rate": float(cfg_get(agent, "learning_rate", 3e-4)),
        "gamma": float(cfg_get(agent, "gamma", 0.99)),
        "ent_coef": cfg_get(agent, "ent_coef", "auto"),
        # learning_starts (gated 1000) + PopArt scale-normalization (default on): threaded so the parallel
        # SEARCH worker trains the SAME fixed agent as the serial path + the parallel TEST worker.
        "learning_starts": int(cfg_get(agent, "learning_starts", 1000)),
        # P10 (2026-07-13 audit): stamp the replay cap into opts -> specs so a cluster task's JSON
        # is SELF-CONTAINED. Without it the node's cap came from the REMOTE checkout's campaign
        # yaml — a divergent sync would silently uncap the buffer (5 workers x ~2.8 GiB = cgroup
        # OOM cascade at pack-5/B*). None (absent in the agent block) keeps prior behavior.
        "buffer_size": cfg_get(agent, "buffer_size", None),
        "popart": bool(cfg_get(agent, "popart", True)),
        "popart_beta": float(cfg_get(agent, "popart_beta", 1e-3)),
        "popart_min_scale": float(cfg_get(agent, "popart_min_scale", 1.0)),
        "popart_warmup": int(cfg_get(agent, "popart_warmup", 0)),
        "tf32": bool(cfg_get(agent, "tf32", True)),  # threaded so the parallel SEARCH worker shares the precision setting
        # M6 (ops audit 2026-07-02): thread the agent block's thermal_guardian ({hi, lo, poll_secs} —
        # src/utils/guardian.ThermalGovernor's schema) into the parallel SEARCH worker, so SEARCH
        # trainings cooperatively pause-and-cool like the serial/TEST paths. Result-neutral: the
        # governor spends only wall-clock, never a weight. None/absent -> off (unchanged behaviour).
        "thermal_guardian": cfg_get(agent, "thermal_guardian", None),
        "n_trials": n_trials,
        "synthetic": synthetic,
        "data": dict(cfg_get(structural_cfg, "data", {})),
        "cvar_alpha": float(cfg_get(rf, "cvar_alpha", 0.05)),
        "window": int(cfg_get(rf, "window", 20)),
        "seed": seed,
        "candidates": candidates,
        "generations": generations,
        "pass_mode": pass_mode,
        "provider": provider,
        # Search-replay (resume): when True, ``_drive_llm_arm`` REPLAYS already-archived candidates
        # (success or sandbox-failure) from disk instead of re-calling the (paid, non-deterministic)
        # LLM + retraining — mirroring the serial loop's resume cache. Default False -> a fresh run is
        # byte-for-byte unchanged.
        "resume": bool(resume),
        "model": str(cfg_get(llm_block, "model_snapshot", "<unset>")),
        "api_key_env": str(cfg_get(llm_block, "api_key_env", default_key_env(provider))),
        "temperature": cfg_get(llm_block, "temperature", None),
        # F17 (review 2026-07-02): thread the author block's max_tokens/max_retries into the
        # parallel driver's transport (defaults = the historical hardcodes 4096/6) — raising the
        # config value must not silently no-op into a truncated reward misread as a 'bad candidate'.
        "max_tokens": int(cfg_get(llm_block, "max_tokens", 4096)),
        "max_retries": int(cfg_get(llm_block, "max_retries", 6)),
        # v2 legs (R80/R82): OpenRouter routing/reasoning pins + the usage-cost request ride the
        # llm block's ``extra_body`` into the transport — without this key a leg's provider pin /
        # quantization / reasoning pins would be SILENTLY DROPPED at authoring (a registered-design
        # violation). None (absent) keeps every v1 path byte-identical.
        "extra_body": cfg_get(llm_block, "extra_body", None),
        # R83 advisory spend ledger: the campaign author paths record per-call realized/estimated
        # cost here (LLMClient skips recording when the transport surfaces neither cost nor usage —
        # fakes/stubs stay ledger-silent). Explicit null in the block disables.
        "spend_ledger": cfg_get(llm_block, "spend_ledger", "outputs/spend_ledger.jsonl"),
        "diversity_prompt_variation": bool(cfg_get(llm_block, "diversity_prompt_variation", False)),
        "env_cfg": env_cfg,
        # Universe size from config (environment.yaml: universe.n_assets) — NOT a hardcoded 30 on the
        # parallel LLM-prompt path (no-hardcoding audit), mirroring the sequential path's panel.N.
        "n_assets": int(cfg_get(cfg_get(env_cfg, "universe", {}), "n_assets", 30)),
        # env_fp provenance label: name the REAL active panel (C5) — gold_suffix() reads the
        # freeze-bound config/data.yaml, not the hardcoded 'univ3', so a new-suffix rebuild is
        # reflected in archived records.
        "env_fp": _env_fp_label(synthetic, steps),
        # Recycle pool workers every N candidates to reclaim fragmented heap (RAM-creep fix, 2026-06-20).
        "max_tasks_per_child": max_tasks_per_child,
        "proto_cfg": structural_cfg,
    }


def _run_env_fp(arm_root: str, run_id: str, label: str, seed: int) -> Any:
    """Write ``<arm_root>/<run_id>/env.json`` and return the record ``env_fingerprint`` (Rank 14).

    Mirrors ``src.orchestration.parallel._run_env_fp`` so BOTH executed paths (sequential here +
    the parallel scheduler) archive a full, content-hashed env.json and persist {label, sha256}
    instead of a bare label string (audit C-2/C-6). Best-effort: falls back to the bare label on any
    capture failure so archiving never breaks.

    F12 (review 2026-07-02): when the snapshot ALREADY exists (a --resume of a partially-run
    arm) it is REUSED — read back and re-hashed — never recaptured. A fresh capture hashes
    differently (timestamps, nvidia-smi, pip state can all drift), so rewriting the file would
    ORPHAN the ``env_json_sha256`` already embedded in the arm's previously-archived records: resume
    must not orphan recorded shas. ``env_json_sha256(env=<loaded>)`` equals the recorded digest by
    construction (sha256_obj canonical JSON — "the same string whether computed here or recomputed
    from a written env.json", capture_env docstring).
    """
    try:
        try:
            from scripts.capture_env import capture_env, env_json_sha256
        except ModuleNotFoundError:  # direct `python scripts/run_prototype.py` puts scripts/ on
            import sys as _sys        # sys.path, not the repo root -> `scripts` not a package; the
            _sys.path.insert(0, str(Path(__file__).resolve().parent))  # broad except below would else
            from capture_env import capture_env, env_json_sha256       # SILENTLY drop env.json (seal-bug class)

        run_dir = Path(arm_root) / str(run_id)
        env_path = run_dir / "env.json"
        if env_path.is_file():
            with env_path.open(encoding="utf-8") as fh:
                env = json.load(fh)
            return {"label": label, "env_json_sha256": env_json_sha256(env=env)}
        env = capture_env(seed=int(seed))
        run_dir.mkdir(parents=True, exist_ok=True)
        with env_path.open("w", encoding="utf-8") as fh:
            json.dump(env, fh, indent=2, sort_keys=True, default=str)
        return {"label": label, "env_json_sha256": env_json_sha256(env=env)}
    except Exception:  # noqa: BLE001 - provenance capture must never crash a candidate's archive
        return label


def _archive_record(
    *, run_id, arm, seed, fold, candidate_id, generation, source, score, env_fp,
    wall=0.0, feedback="", val_returns=None,
) -> dict:
    import hashlib

    metrics: dict[str, Any] = {"val_fitness": float(score)}
    # Additive (Rank 3): persist the per-period VALIDATION return vector so the search
    # arms carry metrics['val_returns'] exactly like the LLM-loop candidate records do
    # (loop.py). PBO/CSCV (scripts/analyze_campaign.py) stacks these per arm. Absent for
    # arms/paths that did not surface a vector -> the key is simply omitted (back-compat).
    if val_returns is not None:
        metrics["val_returns"] = [float(x) for x in val_returns]
    return {
        "run_id": run_id,
        "arm": arm,
        "seed": int(seed),
        "fold": int(fold),
        "candidate_id": candidate_id,
        "generation": int(generation),
        "reward_source": source,
        "reward_source_hash": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "feedback_block": feedback,
        "metrics": metrics,
        "wall_clock": float(wall),
        "env_fingerprint": env_fp,
    }


def _load_search_cache(arm_root: str | Path, arm: str, seed: int) -> dict[int, dict[str, Any]]:
    """Archived search-arm candidate records keyed by candidate index (crash-resume read path).

    The search arms draw their candidate sequence deterministically from the run seed, so on a
    ``--resume`` relaunch the SAME candidates are re-drawn in the same order; a record archived for
    index ``i`` lets the driver skip that candidate's training (the caller hash-verifies the
    re-drawn source against ``reward_source_hash`` and fails LOUD on any drift, so a stale archive
    from a different draw sequence can never be silently reused). Records missing a
    ``metrics.val_fitness`` are ignored (never written by the checkpoint path)."""
    import re as _re

    from src.io.results import load_all

    out: dict[int, dict[str, Any]] = {}
    root = Path(arm_root)
    if not root.is_dir():
        return out
    pat = _re.compile(rf"^{_re.escape(arm)}-s{int(seed)}-c(\d+)$")
    for rec in load_all(root):
        m = pat.match(str(rec.get("run_id", "")))
        if m and "val_fitness" in (rec.get("metrics") or {}):
            out[int(m.group(1))] = rec
    return out


# --------------------------------------------------------------------------- #
# Per-arm execution (the picklable worker)                                    #
# --------------------------------------------------------------------------- #
def run_arm(
    arm: str,
    *,
    synthetic: bool,
    candidates: int,
    generations: int,
    train_steps: int | None,
    n_trials: int,
    seed: int,
    pass_mode: str,
    provider: str,
    archive_root: str,
    llm_cfg: dict[str, Any] | None = None,
    monitor: Any = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run ONE arm end-to-end and return a summary. Loads config/panel/builders INSIDE so it is
    picklable for the process pool (no closures crossing the spawn boundary).

    ``llm_cfg`` is the reward-author block to use for Pass B: the campaign threads its OWN
    (Opus 5) so it doesn't inherit the prototype's; ``None`` falls back to
    ``config/prototype.yaml: llm`` (Sonnet 4.6), which is the prototype's own author."""
    import numpy as np  # used by the seeded search-arm RNGs below (final-audit #2)

    from src.agents.trainer import make_agent_trainer
    from src.env.runner import make_env_builder
    from src.io.results import write_run
    from src.selection.fitness import held_out_fitness
    from src.utils.config import load_config
    from src.utils.preload import preload
    from src.utils.seeding import set_global_seed

    preload()  # pyarrow BEFORE torch: loading the real gold parquet after torch SIGSEGVs (ABI conflict)
    # Seed EVERY RNG stack deterministically for parity with parallel.py::train_candidate (~line 111):
    # Python random, the legacy np.random global (read by VecNormalize/SB3), PYTHONHASHSEED, torch +
    # cuDNN, and the deterministic-algorithm flags (Rank 18).
    set_global_seed(seed, deterministic_torch=True)
    t0 = time.perf_counter()
    env_cfg = load_config("environment")
    proto_cfg = load_config("prototype")
    lookback = int(env_cfg["state"]["lookback_days"])
    data_cfg = cfg_get(proto_cfg, "data", {})
    panel, train_window, val_window = _load_panel_and_windows(synthetic, data_cfg, lookback)
    # R18 purge-guard args (fix 2026-07-03, mirroring run_campaign's builder call): pass the REAL
    # embargo + lookback so make_env_builder's max(embargo, lookback) leakage guard is ARMED on the
    # gold SEARCH path — the bare 4-arg call left purge=0, i.e. the guard could never catch a future
    # windows regression. Gold windows already satisfy the purge (embargoed_val_start builds them), so
    # this only fails loud on a violation. The SYNTHETIC dev/dry-run windows deliberately ABUT
    # ((lookback, 400)/(400, T), no purge gap), so the guard stays legacy-inert (0/0) there.
    if synthetic:
        _embargo = _lookback_guard = 0
    else:
        # Same resolution as _load_panel_and_windows: the data block, else the config/data.yaml floor.
        _emb_floor = int(load_config("data").get("embargo_days", 21))
        _embargo = int(cfg_get(data_cfg, "embargo_days", _emb_floor))
        _lookback_guard = lookback
    env_builder = make_env_builder(
        panel, env_cfg, train_window, val_window, embargo=_embargo, lookback=_lookback_guard
    )
    agent_cfg = _agent_cfg(proto_cfg, train_steps)
    agent_trainer = make_agent_trainer(agent_cfg, seed, monitor=monitor)
    arm_root = str(Path(archive_root) / arm)
    # T2.8b arm-scoped FED-estimator audit (fix 2026-07-03): clear the process-level EVT<->empirical
    # switch registry at ARM START — arms run sequentially/reused within one process (the campaign's
    # serial SEARCH loop; p_arms worker reuse), so without this reset the "estimator switched across
    # candidates" warning mis-scopes ACROSS arms (a legitimate cross-arm difference would fire the
    # within-arm consistency alarm).
    from src.feedback.measurement import reset_fed_estimator_log

    reset_fed_estimator_log()
    env_fp = _env_fp_label(
        synthetic, agent_cfg["train_steps_per_candidate"], n_assets=panel.N
    )  # C5: name the REAL active panel via gold_suffix(), not a hardcoded 'univ3'
    # Rank 14: the REAL provenance fingerprint (full CI-grade env.json captured once per arm under
    # <arm_root>/env.json) + its sha256, threaded into every candidate record so env_fingerprint is a
    # replayable, content-hashed snapshot rather than the bare label (audit C-2/C-6). Falls back to
    # the bare label if torch/capture is unavailable.
    env_fp_record: Any = _run_env_fp(arm_root, "_env", env_fp, seed)
    n_expected = candidates

    if arm in _LLM_ARMS:
        from src.feedback.measurement import ReturnDistribution
        from src.llm.client import JsonlArchiveSink, LLMClient
        from src.llm.loop import run_loop
        from src.llm.prompts import build_prompt_set

        # Reward-author config: the campaign threads its OWN block (Opus 5) via `llm_cfg`; a
        # direct prototype run falls back to config/prototype.yaml: llm (Claude Sonnet 4.6).
        _llm = llm_cfg if llm_cfg is not None else cfg_get(proto_cfg, "llm", {})

        # Transport: Pass A = keyless stub; Pass B = real provider (via the central factory).
        if pass_mode.upper() == "A" or provider == "stub":
            from src.llm.stub_designer import StubDesignerTransport

            transport: Any = StubDesignerTransport(seed=seed)
            model_id = f"stub-designer/seed{seed}"
        else:
            from src.llm.client import build_transport, default_key_env

            model_id = str(cfg_get(_llm, "model_snapshot", "<unset>"))
            key_env = str(cfg_get(_llm, "api_key_env", default_key_env(provider)))
            temp_raw = cfg_get(_llm, "temperature", None)
            temperature = float(temp_raw) if temp_raw is not None else None
            # F17: honor the author block's max_tokens/max_retries (defaults = the historical
            # hardcodes 4096/6) — a raised config value must not silently no-op into truncation.
            transport = build_transport(
                provider, model_id, key_env, temperature=temperature,
                max_tokens=int(cfg_get(_llm, "max_tokens", 4096)),
                max_retries=int(cfg_get(_llm, "max_retries", 6)),
            )

        prompts = build_prompt_set(env_cfg, panel.N)
        # F11: llm_calls.jsonl is appended PER CALL at call time via the archive sink — the old
        # end-of-arm "w" dump lost the whole arm's call provenance (incl. the R71 served_model
        # reproducibility anchor, recorded at the FIRST live call per config/llm.yaml) on a mid-arm
        # crash, and on --resume it TRUNCATED the first run's calls down to just the new ones.
        llm = LLMClient({"model": model_id}, transport=transport,
                        archive=JsonlArchiveSink(Path(arm_root) / "llm_calls.jsonl"))
        loop_cfg = {
            "generations": generations,
            "candidates_per_gen": max(1, candidates // max(1, generations)),
            "budget": candidates,
            "seed": seed,
            "n_trials": n_trials,
            "model": model_id,
            "run_prefix": f"{arm}-s{seed}",
            "fold": 0,
            "prompts": {"system": prompts.system, "initial": prompts.initial},
            # Rank 14: the loop persists this on every candidate record (replaces its "injected"
            # default) so the LLM-arm records carry the real, content-hashed env fingerprint.
            "env_fingerprint": env_fp_record,
            # Within-generation diversity by per-candidate prompt variation (uniform across arms) —
            # required for temperature-rejecting reward-authors (Opus 5). Off by default.
            "diversity_prompt_variation": bool(cfg_get(_llm, "diversity_prompt_variation", False)),
            "monitor": monitor,  # RunMonitor for live progress/logs/anomalies (sequential path; no-op if None)
            # Search-replay cache: on --resume the loop REPLAYS the archived candidates/failures of an
            # interrupted arm instead of re-billing Opus + retraining (byte-faithful; src/llm/loop.py).
            "resume": resume,
        }
        archive = run_loop(
            arm, env_builder, llm, agent_trainer, ReturnDistribution, held_out_fitness, loop_cfg, arm_root
        )
        winner = archive.winner()
        n_done = len(archive.candidates) + len(archive.failures)
        summary = {
            "arm": arm,
            "n_candidates": len(archive.candidates),
            "n_failed": len(archive.failures),
            # C3c (ops audit 2026-07-02): LLM-error skips (API exhaustion after retries / unparseable
            # response) leave slots that are neither archived nor ledgered — surface the count so the
            # campaign's winner-selection floor (run_campaign C3b) can see a resumably-short pool.
            "n_llm_error_skips": int(archive.meta.get("llm_error_skips", 0)),
            "winner_fitness": None if winner is None else float(winner.val_fitness),
            "winner_id": None if winner is None else winner.candidate_id,
        }
        # Persist failed candidate sources (final-audit #21) so a formatting/gate failure is
        # diagnosable from the archive instead of living only in a transient WARNING log line.
        if archive.failures:
            Path(arm_root).mkdir(parents=True, exist_ok=True)
            with (Path(arm_root) / "failures.jsonl").open("w", encoding="utf-8") as _fh:
                for _fail in archive.failures:
                    _fh.write(json.dumps(_fail) + "\n")
        # NB (F11, review 2026-07-02): NO end-of-arm llm_calls.jsonl dump here any more — the
        # raw LLM provenance (system+user+response + per-call token usage, ADR-016 / final-audit #16)
        # is appended PER CALL by the JsonlArchiveSink the LLMClient above was constructed with, so a
        # mid-arm crash loses at most the in-flight call (and a resume APPENDS instead of truncating).
    else:
        # Search arms — wired to matched compute via the C1 evaluator.
        from src.agents.evaluator import evaluate_reward_with_returns
        from src.baselines.reward_family import family_bounds, params_to_reward, params_to_source
        from src.search.bayes_opt import bayes_opt_over_template
        from src.search.random_search import random_search_over_code

        # Capture the per-candidate VALIDATION return VECTOR in evaluation order (Rank 3).
        # random_search/bayes_opt call the injected evaluator EXACTLY ONCE per archived
        # candidate (random_search.archive / bayes_opt.history are in eval order), so this
        # ordered sink aligns 1:1 by index with `evaluated` below. ADDITIVE: the evaluator's
        # scalar (reward|coeffs)->float contract that the search functions depend on is
        # preserved; the vector is recorded as a side effect into this closure list.
        _sc = [0]  # search-arm candidate counter (monitor progress; search loops have no run_loop)

        # Carry the frozen reward_family ranges so random_search draws the six family
        # weights from the SAME box the BO arm searches (both H4 arms => identical space).
        search_cfg = {
            "matched_budget": candidates,
            "reward_family": cfg_get(proto_cfg, "reward_family", {}),
        }

        # ------------------------------------------------------------------ #
        # Crash-resume + per-candidate checkpointing (2026-07-05 hardening).
        # Before this, search-arm records were written only AFTER the whole arm
        # completed, so a crash mid-arm lost every finished training (up to
        # ~30 x 85 min at the campaign budget) and a deterministic mid-arm fault
        # became an infinite supervisor crash-loop. Now: (a) every FRESH
        # candidate is archived the moment its evaluation completes (the
        # checkpoint — always on); (b) on --resume, archived candidates are
        # replayed from disk with the training SKIPPED — the seeded draw
        # sequence is regenerated identically (the drivers consume rng
        # unconditionally), and the re-drawn source is hash-verified against
        # the archived ``reward_source_hash`` so a cache from a different draw
        # sequence fails LOUD instead of silently polluting the arm.
        # ------------------------------------------------------------------ #
        import hashlib as _hl

        _search_cache: dict[int, dict[str, Any]] = (
            _load_search_cache(arm_root, arm, seed) if resume else {}
        )
        _last_vr: list[Any] = [None]  # val_returns holder: fitness -> checkpoint (synchronous)
        _vr_by_idx: dict[int, Any] = {}
        # Resume the live-monitor candidate counter PAST the cached prefix (2026-07-06 audit MINOR):
        # cached candidates report their TRUE idx via _cached_score, so the fresh-candidate counter must
        # start after them or the monitor sees duplicated indices on --resume (archive is unaffected —
        # it keys off the search driver's own idx — this is monitor-precision only).
        _sc[0] = len(_search_cache)

        def _cached_score(idx: int, source: str) -> float | None:
            rec = _search_cache.get(idx)
            if rec is None:
                return None
            want = _hl.sha256(source.encode("utf-8")).hexdigest()
            got = rec.get("reward_source_hash")
            if got != want:
                raise RuntimeError(
                    f"search resume cache MISMATCH for {arm}-s{seed}-c{idx}: the archived candidate "
                    f"hash {str(got)[:12]}... != the re-drawn candidate hash {want[:12]}... — the "
                    "archive was produced by a DIFFERENT draw sequence (seed/config/grammar drift). "
                    "Refusing to reuse it; quarantine the arm directory and re-run without --resume."
                )
            if monitor is not None:
                monitor.candidate_done(arm, idx, fitness=float(rec["metrics"]["val_fitness"]),
                                       status="cached", secs=0.0)
            return float(rec["metrics"]["val_fitness"])

        def _checkpoint(idx: int, source: str, score: float) -> None:
            _vr_by_idx[idx] = _last_vr[0]
            write_run(
                _archive_record(
                    run_id=f"{arm}-s{seed}-c{idx}", arm=arm, seed=seed, fold=0,
                    candidate_id=f"{arm}-s{seed}-c{idx}", generation=0,
                    source=source or f"# {arm} candidate {idx}\n", score=score,
                    env_fp=env_fp_record, val_returns=_last_vr[0],
                ),
                arm_root,
            )
            _last_vr[0] = None
        if arm == "random_search":
            def fitness(reward_fn: Any) -> float:
                if monitor is not None:
                    monitor.candidate_start(arm, _sc[0])
                _c0 = time.perf_counter()
                score, val_returns = evaluate_reward_with_returns(
                    reward_fn, env_builder, agent_trainer, held_out_fitness, n_trials
                )
                _last_vr[0] = val_returns
                if monitor is not None:
                    monitor.candidate_done(arm, _sc[0], fitness=score, status="ok",
                                           secs=time.perf_counter() - _c0)
                _sc[0] += 1
                return score

            # Seed the sampler from the run seed (final-audit #2: a bare default_rng() draws OS
            # entropy and does NOT consult set_global_seed's legacy np.random, so the SELECTED
            # winner was non-reproducible; the parallel scheduler's search arms are seeded the same way).
            result = random_search_over_code(
                env_builder, fitness, search_cfg, rng=np.random.default_rng(seed),
                cache_lookup=_cached_score, on_evaluated=_checkpoint,
            )
            evaluated = [(c.get("source", ""), float(c["score"])) for c in result["archive"]]
        else:  # bayes_opt
            rf = cfg_get(proto_cfg, "reward_family", {})
            alpha = float(cfg_get(rf, "cvar_alpha", 0.05))
            window = int(cfg_get(rf, "window", 20))
            p2r = lambda coeffs: params_to_reward(coeffs, cvar_alpha=alpha, window=window)  # noqa: E731

            def template_eval(coeffs: Any) -> float:
                reward_fn = p2r(coeffs)
                if monitor is not None:
                    monitor.candidate_start(arm, _sc[0])
                _c0 = time.perf_counter()
                score, val_returns = evaluate_reward_with_returns(
                    reward_fn, env_builder, agent_trainer, held_out_fitness, n_trials
                )
                _last_vr[0] = val_returns
                if monitor is not None:
                    monitor.candidate_done(arm, _sc[0], fitness=score, status="ok",
                                           secs=time.perf_counter() - _c0)
                _sc[0] += 1
                return score

            # The resume/checkpoint hooks are keyed by the MATERIALIZED executable source for these
            # coefficients — the exact text the archive stores — so the hash verification protects
            # against template/alpha/window drift as well as seed drift.
            _p2src = lambda x: params_to_source(x, cvar_alpha=alpha, window=window)  # noqa: E731
            result = bayes_opt_over_template(
                template_eval, family_bounds(proto_cfg), search_cfg, rng=np.random.default_rng(seed),
                cache_lookup=lambda i, x: _cached_score(i, _p2src(x)),
                on_evaluated=lambda i, x, s: _checkpoint(i, _p2src(x), s),
            )
            # Rank 2c: archive the MATERIALIZED EXECUTABLE reward source (not a comment stub) so the
            # frozen BO winner rehydrates through validate_once for the sealed TEST leg (H4).
            evaluated = [
                (params_to_source(h["coeffs"], cvar_alpha=alpha, window=window), float(h["score"]))
                for h in result["history"]
            ]

        # Defensive fill-only pass: since the 2026-07-05 hardening, every candidate is archived
        # INCREMENTALLY at evaluation time (fresh -> _checkpoint; resumed -> the record already
        # exists from the prior run), so this loop normally writes NOTHING. It survives as a
        # belt-and-braces net for any candidate whose incremental write did not land — existing
        # records are never overwritten (a resumed arm's cached records keep their original bytes).
        for i, (source, score) in enumerate(evaluated):
            cid = f"{arm}-s{seed}-c{i}"
            if (Path(arm_root) / cid / "record.json").is_file():
                continue
            write_run(
                _archive_record(
                    run_id=cid, arm=arm, seed=seed, fold=0, candidate_id=cid, generation=0,
                    source=source or f"# {arm} candidate {i}\n", score=score, env_fp=env_fp_record,
                    val_returns=_vr_by_idx.get(i),
                ),
                arm_root,
            )
        n_done = len(evaluated)
        best_score = float(result["best_score"])
        summary = {"arm": arm, "n_candidates": n_done, "n_failed": 0, "winner_fitness": best_score}

    # Matched-compute check (review M2): every arm draws the same budget AND lands at least one
    # ACCEPTED candidate. An arm whose candidates all fail the gate (e.g. fenced LLM output) draws
    # the full budget as FAILURES — n_done would still equal n_expected and falsely report healthy;
    # requiring n_candidates>0 surfaces a total-failure arm before auto-shutdown (final-audit #21).
    summary["matched_budget_ok"] = bool(n_done == n_expected and summary["n_candidates"] > 0)
    summary["wall_clock_s"] = round(time.perf_counter() - t0, 1)
    Path(arm_root).mkdir(parents=True, exist_ok=True)
    (Path(arm_root) / "COMPLETE").write_text(json.dumps(summary), encoding="utf-8")
    return summary


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Advanced 40-candidate prototype orchestrator (P5).")
    p.add_argument("--dry-run", action="store_true", help="Tiny synthetic verification (2 arms, 2 cand, 200 steps).")
    p.add_argument("--synthetic", action="store_true", help="Use the synthetic panel (no gold load).")
    p.add_argument("--arms", default=None, help="Comma list overriding the configured arms.")
    p.add_argument("--pass", dest="pass_mode", default=None, help="'A' (keyless stub) or 'B' (real LLM).")
    p.add_argument("--p-arms", type=int, default=None, help="Concurrent arm workers (override config).")
    p.add_argument("--parallel", action="store_true",
                   help="Use the heterogeneous GPU+CPU candidate scheduler (MAX throughput).")
    p.add_argument("--gpu", type=int, default=None, help="GPU worker slots (default: config/auto).")
    p.add_argument("--cpu", type=int, default=None, help="CPU worker slots (default: config).")
    p.add_argument("--out", default=None,
                   help="Override the output dir (config output_dir / the --dry-run default). Used by "
                        "scripts/crash_rehearsal.py to run reference + crash trees side by side.")
    p.add_argument("--resume", action="store_true",
                   help="FORCE resume on (skip COMPLETE arms; replay archived candidates), even under "
                        "--dry-run (whose default is a fresh run). The crash-resume rehearsal's hook.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    from src.utils.config import load_config
    from src.utils.env import load_env
    from src.utils.preload import preload

    preload(strict=True)  # pyarrow before torch (gold-parquet ABI segfault guard) -- BEFORE any torch import; H2 fail-loud
    load_env()  # .env -> os.environ so the LLM key is available (ADR-038); workers inherit it
    proto = load_config("prototype")
    arms = (args.arms.split(",") if args.arms else list(proto["arms"]))
    candidates = int(proto["candidates_per_arm"])
    generations = int(proto["generations"])
    n_trials = int(proto["n_trials"])
    seed = int(proto["seeds"][0])
    pass_mode = str(args.pass_mode or cfg_get(proto.get("llm", {}), "pass", "A"))
    provider = str(cfg_get(proto.get("llm", {}), "provider", "stub"))
    p_arms = int(args.p_arms or cfg_get(proto.get("parallel", {}), "p_arms", 2))
    output_dir = str(proto.get("output_dir", "outputs/prototype"))
    train_steps: int | None = None
    synthetic = bool(args.synthetic)

    if args.dry_run:
        arms = ["distributional", "random_search", "bayes_opt"]  # one LLM + both search paths
        candidates, generations, train_steps, synthetic = 2, 1, 200, True
        # honor an explicit --p-arms under --dry-run (the crash rehearsal forces 1 = the monitored
        # sequential path); the dry-run default stays 2.
        p_arms = int(args.p_arms) if args.p_arms else 2
        # A dry run is a FREE, offline machinery smoke: force the keyless stub regardless of config
        # (final-audit #6 — the prototype now defaults to Pass B / Gemini, so without this override a
        # --dry-run would issue real billed API calls / require GEMINI_API_KEY). Mirrors run_campaign.
        pass_mode, provider = "A", "stub"
        output_dir = "outputs/prototype_dryrun"
        print("[run_prototype] DRY RUN — 3 arms x 2 candidates x 200 steps on a synthetic panel (keyless stub).")

    if args.out:
        output_dir = str(args.out)  # explicit CLI override wins (crash-rehearsal trees)

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    if args.parallel:
        from src.orchestration.parallel import run_parallel

        env_cfg = load_config("environment")
        try:
            import torch

            has_gpu = bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001
            has_gpu = False
        par = proto.get("parallel", {})
        _cfg_gpu = cfg_get(par, "n_gpu", "auto")
        if args.gpu is not None:
            n_gpu = int(args.gpu)
        elif not has_gpu:
            n_gpu = 0
        elif str(_cfg_gpu).lower() == "auto":
            from src.orchestration.parallel import auto_n_gpu

            _ts = int(train_steps or cfg_get(proto.get("agent", {}), "train_steps_per_candidate", 25000))
            n_gpu = auto_n_gpu(_ts)
            print(f"[run_prototype] auto n_gpu={n_gpu} (max concurrency bounded by RAM/VRAM/physical-cores)")
        else:
            n_gpu = int(_cfg_gpu)
        n_cpu = args.cpu if args.cpu is not None else int(cfg_get(par, "n_cpu", 0))
        agent = proto.get("agent", {})
        steps = train_steps or int(cfg_get(agent, "train_steps_per_candidate", 25000))
        # Shared opts builder (no drift with the campaign --search-gpu path; see build_parallel_opts).
        opts = build_parallel_opts(
            proto,
            env_cfg,
            llm_block=proto.get("llm", {}),
            train_steps=steps,
            n_trials=n_trials,
            synthetic=synthetic,
            seed=seed,
            candidates=candidates,
            generations=generations,
            pass_mode=pass_mode,
            provider=provider,
            max_tasks_per_child=cfg_get(par, "max_tasks_per_child", None),
        )
        # Honor AND enable resume on the --parallel path too (audit 2026-06-20): the sequential path skips
        # arms with a COMPLETE marker (run_arm writes one at :366), but the parallel scheduler did NEITHER,
        # so it silently RE-RAN already-complete arms. Filter before scheduling; write the per-arm marker
        # after — mirroring the sequential contract so a later --parallel re-run resumes correctly.
        resume_par = bool(args.resume) or (bool(proto.get("resume", True)) and not args.dry_run)
        par_todo = [a for a in arms if not (resume_par and (root / a / "COMPLETE").exists())]
        for a in arms:
            if a not in par_todo:
                print(f"[run_prototype] skip {a} (COMPLETE marker present).")
        if not par_todo:
            print("[run_prototype] all arms COMPLETE; nothing to do (resume).")
            return
        print(f"[run_prototype] PARALLEL scheduler: gpu={n_gpu} cpu={n_cpu} arms={par_todo} "
              f"candidates={candidates} steps={steps}")
        t0 = time.perf_counter()
        summaries = run_parallel(par_todo, opts, n_gpu, n_cpu, output_dir)
        wall = round(time.perf_counter() - t0, 1)
        summary = {
            "arms": summaries,
            "wall_clock_s": wall,
            "matched_budget_ok": all(s.get("matched_budget_ok") for s in summaries),
            "n_gpu": n_gpu,
            "n_cpu": n_cpu,
        }
        (root / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        for s in summaries:
            # Per-arm COMPLETE marker so a later --parallel re-run resumes past this arm (mirrors run_arm:366).
            marker = root / s["arm"] / "COMPLETE"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps(s), encoding="utf-8")
            print(f"  {s['arm']:>14}: {s['n_candidates']} cand, winner_fitness={s.get('winner_fitness')}, "
                  f"matched={s.get('matched_budget_ok')}")
        print(f"[run_prototype] parallel done in {wall}s -> {root / 'run_summary.json'}")
        return

    resume = bool(args.resume) or (bool(proto.get("resume", True)) and not args.dry_run)

    todo = []
    for arm in arms:
        if resume and (root / arm / "COMPLETE").exists():
            print(f"[run_prototype] skip {arm} (COMPLETE marker present).")
            continue
        todo.append(arm)

    print(f"[run_prototype] arms={todo} candidates={candidates} gens={generations} "
          f"steps={train_steps or proto['agent']['train_steps_per_candidate']} p_arms={p_arms} "
          f"pass={pass_mode} provider={provider} -> {output_dir}")

    opts = dict(
        synthetic=synthetic, candidates=candidates, generations=generations, train_steps=train_steps,
        n_trials=n_trials, seed=seed, pass_mode=pass_mode, provider=provider, archive_root=output_dir,
    )
    summaries: list[dict] = []
    t0 = time.perf_counter()
    if p_arms <= 1 or len(todo) <= 1:
        # SEQUENTIAL path -> attach the live RunMonitor: precise multi-level progress (arms > candidates >
        # training steps), deep JSONL event logs, GPU/CPU/RAM telemetry, and real-time anomaly detection.
        # (The --parallel scheduler has its OWN queue-based ParallelMonitor — spawn workers stream per-step
        # SAC metrics via QueueSink to a Manager queue, drained by a pump thread; this in-process RunMonitor
        # is the sequential path's equivalent. Both write progress.json + events.jsonl + anomalies.jsonl.)
        from src.utils.monitoring import RunMonitor

        _steps = int(train_steps or cfg_get(proto.get("agent", {}), "train_steps_per_candidate", 25000))
        _model = str(cfg_get(proto.get("llm", {}), "model_snapshot", provider))
        monitor = RunMonitor(root, title="Prototype", total_arms=len(todo),
                             candidates_per_arm=candidates, train_steps=_steps, model=_model)
        try:
            for i, arm in enumerate(todo):
                monitor.arm_start(arm, i)
                a_t0 = time.perf_counter()
                s = run_arm(arm, monitor=monitor, resume=resume, **opts)
                monitor.arm_done(arm, winner_fitness=s.get("winner_fitness"),
                                 secs=time.perf_counter() - a_t0)
                summaries.append(s)
            monitor.close(status="done")
        except BaseException:
            monitor.close(status="error")
            raise
    else:
        with ProcessPoolExecutor(max_workers=p_arms) as ex:
            # resume threads through to the per-arm search cache (2026-07-06 fix: this arm-parallel
            # path silently dropped it — arm-level COMPLETE skip worked, but a PARTIAL arm restarted
            # from scratch and re-billed the LLM instead of replaying its archived candidates).
            futures = {ex.submit(run_arm, arm, resume=resume, **opts): arm for arm in todo}
            for fut in as_completed(futures):
                summaries.append(fut.result())

    summary = {
        "arms": summaries,
        "wall_clock_s": round(time.perf_counter() - t0, 1),
        "matched_budget_ok": all(s.get("matched_budget_ok") for s in summaries),
    }
    (root / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[run_prototype] summary:")
    for s in summaries:
        print(f"  {s['arm']:>14}: {s['n_candidates']} cand, winner_fitness={s.get('winner_fitness')}, "
              f"matched={s.get('matched_budget_ok')}, {s.get('wall_clock_s')}s")
    print(f"[run_prototype] done in {summary['wall_clock_s']}s -> {root / 'run_summary.json'}")


if __name__ == "__main__":
    main()
    # The GPU/process-pool atexit teardown can set a nonzero exit code on Windows AFTER a fully
    # successful run (cosmetic, but confuses long-run monitoring). Results are already written, so
    # flush and exit cleanly. Only here (script entrypoint) -- tests that call main() are unaffected.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
