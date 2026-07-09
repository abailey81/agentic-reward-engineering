"""Cluster campaign orchestrator — the generation-level composition (PLAN §7 Phase A / §12 B-A1).

Turns the batch driver (:func:`src.cluster.driver.run_batch`) into the full multi-arm campaign,
split laptop/cluster: **authoring + reflection + selection run on the LAPTOP** (API keys are
laptop-only, R10); **training runs on Myriad**. The load-bearing invariant is LAPTOP↔CLUSTER
PARITY — every science primitive is the SAME object the certified local paths use:

* authoring/reflection: ``build_prompt_set`` + ``LLMClient`` + ``schema.build_block`` +
  ``_diversity_directive`` + ``_REFLECTION_PREAMBLE`` (identical to ``parallel._drive_llm_arm``);
* the reflect-on-generation-BEST protocol (``loop.py`` M5) — candidates within a generation are
  INDEPENDENT (they reflect on the PREVIOUS generation's best, never on each other), so training
  them as ONE SGE array is scientifically identical to pool-training them ("adaptive execution,
  invariant design", §13);
* the search worker (``parallel.train_candidate``) + the sealed-leg worker
  (``test_leg._test_seed_worker``) + the record schema (``parallel._archive`` /
  ``test_leg.build_test_record``) — a cluster record is byte-compatible with the local one;
* the F5 failures ledger + search-replay resume (so a mid-search resume never re-authors a
  ledgered candidate against the non-deterministic LLM, which could silently change the winner).

Only the SCHEDULING differs. Throughput (PLAN §5/§14) is maximised WITHOUT breaking any of this:

* the **test leg is one embarrassingly-parallel array** (``-tc`` = full pool) — 93% of the GPU
  hours at maximum concurrency;
* **per-arm search→test pipelines run concurrently** with ZERO barriers — the moment an arm's
  search completes, its winner freezes and its test array floods WHILE other arms still search;
* authoring is serialised across arms (a shared lock) so the API stays rate-limit-safe, while the
  training arrays interleave freely;
* one GPU pool per contrast (device homogeneity holds within every analysis unit).

Everything is injectable (``run_batch``/``select_winner``/``freeze_winner``/author lock/guard) so
the WHOLE campaign is unit-tested against a fake cluster + the keyless stub author, no network/GPU.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

__all__ = [
    "ClusterRun",
    "build_cluster_run",
    "run_search_arm",
    "run_family_search_arm",
    "run_test_leg",
    "run_baselines_on_cluster",
    "run_arm_pipeline",
    "run_campaign_on_cluster",
    "run_campaign_tiered",
    "spend_guard",
]

_LOG = logging.getLogger(__name__)

# Arms whose reflection block carries the fed tail vector (the manipulated variable). Mirrors
# parallel._drive_llm_arm:1030-1034 EXACTLY — the scalar/H1 arms get a scalar-only block.
_TAIL_FED_ARMS = ("distributional", "scalar_cvar5", "placebo_shuffled")


@dataclass
class ClusterRun:
    """The cluster wiring every orchestrator function threads through.

    ``run_batch(specs, name, *, pool, pack) -> summary`` is the ONLY cluster seam — production
    binds a partial of :func:`src.cluster.driver.run_batch` (with the ssh/transport infra); tests
    inject a fake that simulates "jobs ran, records appeared in ``read_root``". ``spec_archive_root``
    is the REMOTE outputs path stamped into each spec (where on-node ``run_one`` archives);
    ``read_root`` is the LOCAL mirror this orchestrator READS (``load_run``) — production keeps them
    distinct (the driver's pull bridges them), tests point both at one tmp dir.
    """

    run_batch: Callable[..., dict]
    spec_archive_root: str
    read_root: Path
    pool_confirmatory: str = "EF"
    pool_report_only: str = "L"
    pack: int = 1
    # Serialises authoring across concurrent arm threads (rate-limit safety); a plain no-op single-arm.
    author_lock: Any = field(default_factory=nullcontext)
    # Called under the author lock before each LLM call — the hard spend cap (see :func:`spend_guard`).
    author_guard: Callable[[], None] = field(default_factory=lambda: (lambda: None))
    # SELECT/FREEZE are injected (default = the real run_campaign ones, resolved lazily) so the
    # pipeline is testable with fakes and reuses the certified selection rule in production.
    # NOTE: R115 (2026-07-28): the REGISTERED winner-eligibility execution floor lives in
    # `run_campaign.select_winner`. Injecting a custom selector therefore BYPASSES a
    # pre-registered design rule — silently. The one live reason to inject in production is the
    # k>1 multiseed selector this file already warns about (`search_seeds_per_candidate > 1`;
    # currently 1, so the default is used and the floor applies). ANY replacement must re-apply
    # `run_campaign._winner_eligible`, or the amendment becomes registered-but-inert.
    select_winner: Callable[[Any], Any] | None = None
    freeze_winner: Callable[..., Any] | None = None
    # P11 (2026-07-13 audit): the shared throttled pull, exposed so resume paths can refresh the
    # mirror BEFORE making replay/authoring decisions (a stale mirror re-authors PAID candidates).
    pull: Callable[[], int] | None = None
    # Device-stratified seed blocks (RATIFIED 2026-07-11, CHANGELOG [2026-07-11c]): whole SEED
    # BLOCKS may run on different GPU pools — the inference is CRN-PAIRED per seed, so every
    # contrast D_s compares arms trained on the SAME device (device cancels in the pair; a
    # randomized-block design; the device×arm interaction is reported via the per-record env_fp).
    # ``[("EF", {0..283}), ("L", {284..567})]``; None (default) = single-pool, byte-identical to
    # the certified path. Seeds in no block fall back to the call's pool (fail-safe, never dropped).
    seed_pool_blocks: list[tuple[str, set[int]]] | None = None
    # MODE-D phase-adaptive packing (2026-07-21, ops-only — no registered quantity changes): the
    # measured aggregate curve {1:102, 2:133, 5:253 steps/s} makes pack a LATENCY/THROUGHPUT dial —
    # pack-5 maximizes trainings/GPU-hour but stretches each training's wall time ~2.5x. The
    # 6-generation reflection chains are LATENCY-critical (each generation's authoring waits on the
    # previous wave), so search waves run at ``search_pack`` (typically 2) with a matching tight
    # ``search_h_rt`` (short requests backfill into idle windows pack-5's cannot use), while the
    # embarrassingly-parallel winner/rung bursts keep the run-level throughput pack. None (default)
    # = uniform pack, byte-identical to the certified path.
    search_pack: int | None = None
    search_h_rt: str | None = None
    # MODE-D chain-lane polling (2026-07-21b): every chain handoff (6 search generations x 10
    # lines; the 30-step BO chain) pays up to poll_secs of driver-notice latency before the next
    # authoring/proposal can start — ~45-75 min on the BO chain alone at 180s. Chain batches poll
    # at this faster cadence; burst arrays keep the run-level poll (login-node kindness: fast
    # polling only while small chain batches are outstanding). None = legacy.
    search_poll_secs: float | None = None
    # SEARCH-LEG THREAD COUNT — amendment R107, wired 2026-07-27. This is the EXECUTION half of a
    # value that was ratified, fed to the makespan planner via ``lanes.CPU_CHAIN_THREADS``, and
    # never actually applied: ``_worker_init`` pinned 1 thread unconditionally, so the planner
    # believed the bayes_opt GP-EI chain took 3.3 d while the code made it 8.9 d, and the core
    # crossover sat at 1,685 instead of 4,584. Resolved from the DEVICE at build time (see
    # ``build_cluster_run``) rather than left to a CLI flag, so the registered value cannot drift
    # from the executed one again by omission. The TEST leg never passes it: R107's scope is the
    # SEARCH/chain leg only, so every scored contrast keeps one float-reduction order.
    search_threads: int = 1

    # C5 (P4 closed 2026-07-13): the H3 single-shot control runs through a ``dataclasses.replace``d
    # ClusterRun whose subdirs are the ``*_h3_singleshot`` names — the headline and H3 archives are
    # DISJOINT BY CONSTRUCTION (same-root reuse would let the compacted resume ADOPT headline
    # run_ids and fabricate the H3 null via run-id collision).
    search_subdir: str = "search"
    test_subdir: str = "test"

    # SEARCH and TEST records live in SEPARATE sub-roots (parity with the laptop campaign's
    # search_root/test_root split). Critical: a test record carries the winner's ``val_fitness``
    # (test_leg.build_test_record), so if it shared the arm dir with the search candidates,
    # ``select_winner`` (max val_fitness) could pick a test record on RESUME. The split makes that
    # impossible. The driver is subdir-agnostic (it keys the compacted diff on run_id), so this is
    # invisible to it.
    def search_read(self) -> Path:
        return Path(self.read_root) / self.search_subdir

    def test_read(self) -> Path:
        return Path(self.read_root) / self.test_subdir

    def search_spec_root(self) -> str:
        return f"{self.spec_archive_root.rstrip('/')}/{self.search_subdir}"

    def test_spec_root(self) -> str:
        return f"{self.spec_archive_root.rstrip('/')}/{self.test_subdir}"

    def line_tag(self) -> str:
        """This LINE's identity, for per-line files that live in the SHARED output root.

        2026-07-28, found while sweeping for further instances of the cross-line collision that
        invalidated RUN 1. `read_root` is shared by all twelve supervised lines, so ANY unqualified
        filename under it is a collision waiting to happen. The archive sub-roots
        (``search`` / ``search_leg_<model>`` / ``search_h3_singleshot``) are already the mechanism
        that keeps lines disjoint, so they are the natural per-line key here too.
        """
        sub = str(self.search_subdir)
        return sub[len("search_"):] if sub.startswith("search_") else sub


def build_cluster_run(
    *,
    remote_root: str,
    remote_outputs_root: str,
    local_batch_root: str | Path,
    local_archive_root: str | Path,
    gold_dir: str,
    host: str = "myriad",
    pool_confirmatory: str = "EF",
    pool_report_only: str = "L",
    pack: int = 1,
    chunk_tasks: int | None = None,
    # 2026-08-06 DURATION LEVER: trainings per TASK, independent of `pack` (trainings at ONCE).
    # >pack means multiple waves in one job, which holds its cores longer at the same dispatch
    # rate — the only variable left once fair share caps acquisition. Caller must size `h_rt`.
    specs_per_task: int | None = None,
    max_consecutive_errors: int = 240,
    poll_secs: float = 600.0,
    min_pull_interval: float = 60.0,
    max_author_calls: int | None = None,
    concurrent: bool = True,
    apptainer_sif: str | None = None,
    cores_per_training: int | None = None,
    h_rt: str | None = None,
    seed_pool_blocks: list[tuple[str, set[int]]] | None = None,
    batch_tag: str | None = None,
    search_pack: int | None = None,
    search_h_rt: str | None = None,
    search_poll_secs: float | None = None,
    search_threads: int | None = None,
    exclude_hosts: list[str] | None = None,
    device: str = "cuda",
) -> ClusterRun:
    """Wire a production :class:`ClusterRun` over :func:`src.cluster.driver.run_batch`.

    Binds the ssh/transport infra once and hands the orchestrator the clean
    ``run_batch(specs, name, *, pool, pack)`` seam. Two throughput/safety details are handled here
    so the concurrent arm threads behave:

    * a **shared throttled puller** — ``run_batch``'s per-cycle pull is de-duplicated across the arm
      threads (any thread's pull refreshes the WHOLE local mirror, so a second pull within
      ``min_pull_interval`` reuses it) — no redundant ssh ``find``/transfer storms;
    * a shared **authoring lock** (arm-serial API) and, when ``max_author_calls`` is set, the hard
      **spend cap** (:func:`spend_guard`), both threaded onto the ``ClusterRun``.

    ``spec_archive_root`` (into each spec) is the REMOTE outputs root; ``read_root`` (the mirror the
    orchestrator reads) is ``local_archive_root`` — the driver's pull bridges them.
    """
    from src.cluster import driver
    from src.cluster.poll import pull_archive, sync_epilogue_ledgers
    from src.cluster.submit import ssh_runner

    runner = ssh_runner(host)
    pull_lock = threading.Lock()
    pull_state: dict[str, Any] = {"t": None, "n": 0, "busy": False, "last_error": None}

    def shared_pull() -> int:
        with pull_lock:
            now = time.monotonic()
            last = pull_state["t"]
            # Skip if a pull is IN PROGRESS (concurrent pull_archive would race on the shared
            # ``.pull_tmp.<pid>`` staging dirs (per-pid since the 2026-07-24 audit fix; the lock now mainly dedupes in-process pulls) — a real bug when a pull outlasts min_pull_interval) OR if
            # one SUCCEEDED within the interval. Either way reuse the last count; run_batch
            # re-derives completion from the shared local archive the live pull is updating.
            # 2026-07-13 audit fix (HIGH): the window timestamp was stamped BEFORE the attempt and
            # never rolled back on failure — for min_pull_interval after every FAILED pull, every
            # other arm thread got a cached "success", resetting each thread's outage streak so the
            # 12 h fatal bound could NEVER trip during a permanent breakage (the campaign would
            # idle forever with green heartbeats). The window now opens only on SUCCESS; a failure
            # RAISES to every caller in the failed window so outage streaks accumulate honestly.
            if pull_state["busy"] or (last is not None and (now - last) < min_pull_interval):
                if pull_state.get("last_error") is not None and not pull_state["busy"]:
                    raise RuntimeError(f"shared pull failed recently: {pull_state['last_error']}")
                return int(pull_state["n"])
            pull_state["busy"] = True
        try:
            n = pull_archive(remote_outputs_root, local_archive_root, host=host, runner=runner)
            # The epilogue ledgers ride the same rate-limited window (2026-07-26). They live in
            # ``{remote_root}/ledger``, a SIBLING of the outputs root pull_archive walks, so without
            # this the driver never sees the tasks that died WITHOUT archiving anything — a walltime
            # kill, a node failure, or a host missing apptainer. Those are precisely the events the
            # kill-classifier and the bad-node check exist for. Best-effort and swallowed: the
            # archive is the completion truth, so failing forensics must never fail a pull.
            try:
                sync_epilogue_ledgers(f"{remote_root.rstrip('/')}/ledger",
                                      Path(local_archive_root) / "ledger", runner)
                _enforce_kill_switch(local_archive_root, _h_rt_seconds(h_rt, search_h_rt))
            except Exception:  # noqa: BLE001 — observability never breaks the run
                pass
        except BaseException as exc:
            with pull_lock:
                pull_state["busy"] = False
                pull_state["t"] = time.monotonic()  # rate-limit RETRIES of a failing remote too
                pull_state["last_error"] = f"{type(exc).__name__}: {exc}"
            raise
        with pull_lock:
            pull_state["n"] = n
            pull_state["t"] = time.monotonic()
            pull_state["busy"] = False
            pull_state["last_error"] = None
        return n

    # Per-batch driver heartbeat → <mirror>/driver_status/<name>.json (atomic). The concurrent arm
    # threads each write their OWN file (no cross-thread races); the sentinel reads them READ-ONLY —
    # the freshest mtime is the driver LEASE (host-death / hang deadman), and the union of queue_names
    # + summed pending is the Myriad QUEUE panel. Effect-blind (counts + SGE states only) and
    # best-effort (a write failure never touches the run).
    status_dir = Path(local_archive_root) / "driver_status"

    def emit_heartbeat(payload: dict[str, Any]) -> None:
        status_dir.mkdir(parents=True, exist_ok=True)
        base = str(payload.get("base_name", "batch")).replace("/", "_").replace("\\", "_")
        dst = status_dir / f"{base}.json"
        tmp = dst.with_name(f"{base}.json.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps({**payload, "wall_ts": time.time()}), encoding="utf-8")
        tmp.replace(dst)  # atomic same-dir rename

    def run_batch(specs: list[dict], name: str, *, pool: str = "EF", pack: int = pack,
                  priority: int = 0, h_rt_call: str | None = None,
                  poll_call: float | None = None, threads: int = 1) -> dict:
        # batch_tag (2026-07-11d, BUG FIX): the driver's double-submit guard matches queued jobs by
        # NAME across the user's WHOLE queue, so two concurrent runs sharing arm names (e.g. a
        # rehearsal's `distributional_g0` and the prototype's) COLLIDE — the later run silently
        # "adopts" the earlier run's queued array and polls an archive that job will never write
        # to. A per-run tag namespaces every batch at this single choke point (markers and local
        # batch dirs inherit it); same-run resume adoption still works (same tag).
        if batch_tag:
            name = f"{batch_tag}_{name}"
        # KILL-INCIDENT GATE (2026-07-26). Every batch on every path funnels through here, so this
        # is the one place that can guarantee we do not resubmit into an unresolved administrative
        # kill — the behaviour that turns "our jobs were killed" into "our account was suspended".
        # Human-in-the-loop by design (like the tier-1 review gate): a machine may stop the
        # campaign, only a person may restart it (killswitch.clear_incident).
        from src.cluster.killswitch import incident_blocks_submission

        _blocked, _why = incident_blocks_submission(local_archive_root)
        if _blocked:
            raise RuntimeError(f"submission BLOCKED — {_why}")
        # ``priority`` = the §14.3 intra-user -p ladder value (0 / -100 / -200 / -500): SGE itself
        # executes the C-ladder value order even when everything is queued at once.
        # Jobscript kwargs threaded to render_jobscript via driver.run_batch's **jobscript_kwargs:
        #  * apptainer_sif (2026-07-10): the cluster venv is built INSIDE python311.sif (RHEL7 glibc is
        #    too old for the cu124 wheels natively), so every training must launch through apptainer.
        #  * cores (2026-07-11): the LIVE rehearsal found GPU-node CORES (not GPUs) are the binding
        #    scheduling constraint (free-GPU nodes sit at load=36). The default 4-cores/training is
        #    over-provisioned (a training uses <1 core), so cores_per_training lets the campaign shrink
        #    the footprint (cores = cores_per_training × the call's pack) so packed jobs actually place.
        #  * h_rt (2026-07-11, max-throughput): the renderer's default walltime (3h at pack=1) is a
        #    ~5.5x over-request at the measured 33 min/training — a tight, MEASURED h_rt (wall x1.5)
        #    makes every task a prime BACKFILL candidate (schedulers slot short jobs into reservation
        #    gaps), which is a pure placement win on a saturated queue. Hardware/scheduling only.
        #  * device (2026-07-26, the CPU lane): selects the SUBSTRATE. It must reach BOTH the
        #    jobscript (so the array requests no GPU and pins no GPU pool) AND every spec (so
        #    run_one._run_single trains on cpu instead of its "cuda" default) — setting only one
        #    of the two would either strand a CPU array on a GPU queue or run torch on a device
        #    the job never requested. Injected at this single choke point.
        _jk: dict[str, Any] = {}
        if device != "cuda":
            _jk["device"] = device
            specs = [{**s, "device": device} for s in specs]
        if apptainer_sif:
            _jk["apptainer_sif"] = apptainer_sif
        if exclude_hosts:
            _jk["exclude_hosts"] = list(exclude_hosts)
        #  * threads (R107, wired 2026-07-27): the intra-op BLAS/torch thread count for THIS batch.
        #    Like `device` it must reach BOTH the specs (so `run_one._task_threads` pins the worker)
        #    AND the jobscript (so SGE actually grants the cores) — threads without matching cores
        #    is oversubscription and measurably SLOWER than 1 thread, so the two move together at
        #    this single choke point and can never be set independently.
        #    CPU-ONLY BY CONSTRUCTION: on the GPU lane the card does the arithmetic, so extra CPU
        #    threads buy nothing and would inflate the core request on exactly the pools where
        #    CORES (not GPUs) are the binding scheduling constraint — i.e. it would HURT placement.
        #    Callers pass `threads=CPU_CHAIN_THREADS` on the SEARCH/chain leg only; the TEST leg
        #    keeps the default 1, so every scored contrast stays on one arithmetic (R107's scope).
        #    The SPEC carries the count (never `_jk`): `_worker_init` sets the BLAS env vars inside
        #    the worker process BEFORE torch is imported, which is the only point where they bind.
        #    The jobscript's job is solely to REQUEST the matching cores.
        eff_threads = max(1, int(threads)) if device == "cpu" else 1
        if eff_threads > 1:
            specs = [{**s, "threads": eff_threads} for s in specs]
        if cores_per_training or eff_threads > 1:
            per_training = max(int(cores_per_training or 1), eff_threads)
            _jk["cores"] = per_training * int(pack)
        # MODE-D: a per-call h_rt (search waves at a lower pack need a TIGHTER walltime than the
        # run-level pack-5 sizing — short requests are prime backfill) overrides the run-level one.
        if h_rt_call:
            _jk["h_rt"] = str(h_rt_call)
        elif h_rt:
            _jk["h_rt"] = str(h_rt)
        return driver.run_batch(
            specs, name,
            local_batch_root=local_batch_root, local_archive_root=local_archive_root,
            remote_root=remote_root, remote_outputs_root=remote_outputs_root, gold_dir=gold_dir,
            host=host, runner=runner, pull=shared_pull, pack=pack, chunk_tasks=chunk_tasks,
            specs_per_task=specs_per_task,
            max_consecutive_errors=max_consecutive_errors,
            poll_secs=(poll_call if poll_call is not None else poll_secs), pool=pool,
            priority=priority, heartbeat=emit_heartbeat, **_jk,
        )

    author_lock: Any = threading.Lock() if concurrent else nullcontext()
    # F7 (agent audit): `if max_author_calls` treated an EXPLICIT 0 as "uncapped" — the exact
    # inverse of the operator's intent ("forbid authoring"). None = uncapped; 0 = forbid.
    author_guard = spend_guard(max_author_calls) if max_author_calls is not None else (lambda: None)
    from src.cluster.lanes import CPU_CHAIN_THREADS

    return ClusterRun(
        run_batch=run_batch, spec_archive_root=remote_outputs_root,
        read_root=Path(local_archive_root), pool_confirmatory=pool_confirmatory,
        pool_report_only=pool_report_only, pack=pack, author_lock=author_lock,
        author_guard=author_guard, seed_pool_blocks=seed_pool_blocks, pull=shared_pull,
        search_pack=search_pack, search_h_rt=search_h_rt,
        search_poll_secs=search_poll_secs,
        # R107 is CPU-ONLY by construction (CPU_CHAIN_THREADS imported just above): on the GPU lane the card does the arithmetic, so extra
        # CPU threads buy nothing and would inflate the core request on precisely the pools where
        # CORES are the binding scheduling constraint. `run_batch` enforces the same guard.
        # None = the REGISTERED value, so code and register agree by default (the R85 rule).
        # An explicit override is how an operator DECLINES the trade after measuring it -- and it
        # must be paired with an amendment, or the register is fiction again.
        search_threads=(int(search_threads) if search_threads is not None
                        else (CPU_CHAIN_THREADS if device == "cpu" else 1)),
    )


def parse_seed_pool_blocks(text: str) -> list[tuple[str, set[int]]]:
    """Parse ``"EF:0-283,L:284-567"`` into device-stratified seed blocks (fail-loud).

    Blocks must be pairwise DISJOINT — one seed on two devices would break the CRN pairing the
    stratification's validity rests on (2026-07-11c).
    """
    blocks: list[tuple[str, set[int]]] = []
    seen: set[int] = set()
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        pool, _, rng = part.partition(":")
        if not pool or "-" not in rng:
            raise ValueError(f"bad seed-pool block {part!r} (expected POOL:LO-HI)")
        lo, hi = (int(x) for x in rng.split("-", 1))
        if hi < lo:
            raise ValueError(f"bad seed range in {part!r} (hi < lo)")
        seeds = set(range(lo, hi + 1))
        overlap = seen & seeds
        if overlap:
            raise ValueError(f"seed-pool blocks overlap (e.g. seed {min(overlap)}): CRN pairing "
                             "requires each seed on exactly one device class")
        seen |= seeds
        blocks.append((pool.strip(), seeds))
    if not blocks:
        raise ValueError(f"no seed-pool blocks parsed from {text!r}")
    # STRIPED splits (2026-07-13, launch ratification): a pool may appear in MULTIPLE ranges
    # (e.g. "EF:0-14,L:15-29,EF:30-64,..." engages both pools at EVERY ladder rung instead of
    # idling the A100s until seed 284). Merge same-pool entries into ONE block — without this,
    # run_test_leg would submit two arrays under the SAME batch name (t_EF twice: a P12 lock
    # collision + queue-name adoption). Order-preserving on first occurrence.
    merged: dict[str, set[int]] = {}
    order: list[str] = []
    for pool, seeds in blocks:
        if pool not in merged:
            merged[pool] = set()
            order.append(pool)
        merged[pool] |= seeds
    return [(pool, merged[pool]) for pool in order]


# --------------------------------------------------------------------------- #
# Authoring + result reconstruction (reused primitives, parity with parallel)  #
# --------------------------------------------------------------------------- #
def _advance(llm: Any) -> None:
    """Consume one author-stream position (positional stub transports stay index-aligned on resume;
    a real LLM transport no-ops). Duck-typed — injected fakes need not implement it."""
    adv = getattr(llm, "advance_author_stream", None)
    if adv is not None:
        adv()


def _build_cluster_author(arm: str, opts: dict, arm_root: Path) -> tuple[Any, Any, bool]:
    """Construct the author EXACTLY as ``parallel._drive_llm_arm`` does (same transport selection,
    same prompt set, same per-call JSONL provenance sink) → returns ``(llm, prompts, diversity)``."""
    from src.llm.client import JsonlArchiveSink, LLMClient
    from src.llm.prompts import build_prompt_set

    if str(opts["pass_mode"]).upper() == "A" or opts["provider"] == "stub":
        from src.llm.stub_designer import StubDesignerTransport

        transport: Any = StubDesignerTransport(seed=opts["seed"])
        model = f"stub-designer/seed{opts['seed']}"
    else:
        from src.llm.client import build_transport, default_key_env

        model = opts["model"]
        temp_raw = opts.get("temperature")
        temperature = float(temp_raw) if temp_raw is not None else None
        key_env = opts.get("api_key_env") or default_key_env(opts["provider"])
        transport = build_transport(
            opts["provider"], model, key_env, temperature=temperature,
            max_tokens=int(opts.get("max_tokens") or 4096),
            max_retries=int(opts.get("max_retries") or 6),
            # v2 legs: the registered OpenRouter pins (provider/quantization/reasoning + usage-cost)
            # must survive into the cluster author — parity with parallel._drive_llm_arm.
            extra_body=opts.get("extra_body") or None,
            # R106: the ANTHROPIC equivalent of those pins. extra_body is rejected on that transport,
            # so without this an Anthropic leg's registered reasoning-off would silently not be sent.
            thinking=opts.get("thinking") or None,
        )
    diversity = bool(opts.get("diversity_prompt_variation", False))
    prompts = build_prompt_set(opts["env_cfg"], opts["n_assets"])
    llm = LLMClient(
        # 2026-07-28: `provider` MUST be threaded. The transport above is built from the real
        # opts["provider"], so routing was always correct — but this cfg used to carry only
        # model+spend_ledger, and client.py resolves `cfg_get(cfg, "provider", "anthropic")`, so
        # every spend row for the eight OpenRouter legs was stamped "anthropic". Measured across
        # RUNs 1-3: all 1,361 rows said anthropic, including models like `deepseek/deepseek-v4-pro`.
        # Cost attribution by provider is a reported artifact, so a wrong label is a real defect.
        {"model": model, "provider": opts["provider"],
         "spend_ledger": opts.get("spend_ledger")}, transport=transport,
        archive=JsonlArchiveSink(Path(arm_root) / "llm_calls.jsonl"),
    )
    return llm, prompts, diversity


#: ``str(arm_root) -> train_steps`` this run configured, populated by ``run_search_arm`` before any
#: replay decision. Keyed by root rather than passed down because ``_result_from_record`` is reached
#: through four call sites and threading a parameter through all of them would be a wider change than
#: the guard it serves. Absent key = no expectation recorded = the guard stays inert (old behaviour).
_EXPECTED_TRAIN_STEPS: dict[str, int] = {}


def _result_from_record(cid: str, arm: str, arm_root: Path) -> dict[str, Any] | None:
    """Reconstruct a search result from an archived record (the ``_drive_llm_arm`` replay shape).

    ``None`` when the record is absent (candidate not yet trained, or FAILED — sandbox reject /
    exhausted retries). A CORRUPT record (KeyError/ValueError from ``load_run``) PROPAGATES — failing
    loud beats silently regenerating (re-billing the LLM + desyncing the search)."""
    from src.io.results import load_run

    try:
        hit = load_run(cid, str(arm_root))
    except FileNotFoundError:
        return None
    m = hit.get("metrics", {}) or {}
    # NOTE: A RECORD THAT EXISTS IS NOT NECESSARILY A RECORD OF THE RUN WE ARE RESUMING.
    #
    # Resume matched candidates on IDENTITY alone, so any archived record under this arm root was
    # replayed whatever produced it. Measured 2026-08-21 on the authoring-variance sub-experiment: a
    # 5-minute prototype had written ten 2,000-step records into `haiku_4_5/r1`, and the full run
    # replayed all ten as if they were its own 400,000-step work. The consequence is not a slow run
    # but a WRONG one — the generation-0 winner was selected from undertrained scores, and
    # generation 1 was then authored from that winner's feedback block, so the whole chain-arm
    # descended from numbers that never meant anything. It is invisible in the logs, because a
    # replayed record looks exactly like a completed one.
    #
    # `train_safe_call_count` is the environment's own step counter, written by the node, so it is
    # the training that actually happened rather than the training that was requested. Refuse the
    # record when it disagrees with the configured budget, and say so — a re-run costs one training,
    # while a silent replay costs the arm's scientific validity.
    want_steps = _EXPECTED_TRAIN_STEPS.get(str(arm_root))
    got_steps = m.get("train_safe_call_count")
    if want_steps and isinstance(got_steps, (int, float)) and int(got_steps) != int(want_steps):
        _LOG.warning(
            "[%s] REFUSING archived record %s: trained %s steps, this run wants %s. It will be "
            "re-trained. A record from a different training budget must never seed a selection.",
            arm, cid, int(got_steps), int(want_steps),
        )
        return None
    r: dict[str, Any] = {
        "ok": True, "candidate_id": cid, "arm": arm,
        "fitness": float(m["val_fitness"]),
        "val_returns": m.get("val_returns"), "tail_stats": m.get("tail_stats"),
        "reward_source": hit.get("reward_source", ""),
        "reward_hash": hit.get("reward_source_hash", ""),
        "prompt": hit.get("prompt", ""),
    }
    if m.get("train_returns") is not None:  # k-seed fed-tail input (B-A2; absent on k=1 records)
        r["train_returns"] = m["train_returns"]
    if m.get("popart_scale") is not None:
        r["popart_scale"] = m["popart_scale"]
    return r


#: TEXTUAL transient markers — safe as substrings because they are WORDS, not digits.
_TRANSIENT_AUTHOR_MARKERS = (
    "overloaded", "timeout", "timed out", "connection",
    "rate limit", "rate_limit", "temporarily", "unavailable", "getaddrinfo", "reset by peer",
)
#: Transient HTTP statuses, matched as NUMBERS not substrings (2026-07-26 review). The bare strings
#: "500"/"502"/"503"/"504"/"529" used to live in the tuple above and matched any INCIDENTAL digits in
#: an error message, so PERMANENT 4xx errors were classified transient and rode out the full 2 h
#: budget — verified on realistic messages: ``max_tokens: 8500 > 8192``, ``claude-opus-5000-preview``,
#: ``req_011CQ500ABCD``, ``prompt is 1500 tokens over the limit`` ALL contain "500". That directly
#: defeats this function's stated guarantee ("no spend-burning retries on a permanent problem") on
#: exactly the failure this project has already hit twice (the R97a / R102 max_tokens 400s).
#: 429 is included deliberately: rate-limiting IS transient, and the old tuple only caught it when the
#: message happened to spell "rate limit" — matching client.py, where 429 retries on the SAME key.
_TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504, 529})
_STATUS_CODE_RE = re.compile(r"\b(?:error code|status(?:[ _]code)?|http)\D{0,3}(\d{3})\b")


def _is_transient_author_error(exc: BaseException) -> bool:
    """True iff an authoring failure should be RIDDEN OUT rather than re-raised.

    FAIL-CLOSED: anything not positively recognised as transient re-raises immediately, so an
    unknown error never burns the retry budget. A definitive HTTP status (from the SDK attribute, else
    parsed from the message) DECIDES on its own — a 4xx is permanent even if the prose happens to
    contain a transient-looking word. Only when no status is available do the textual markers apply.
    """
    status = getattr(exc, "status_code", None)
    msg = f"{type(exc).__name__}: {exc}".lower()
    if not isinstance(status, int):
        m = _STATUS_CODE_RE.search(msg)
        status = int(m.group(1)) if m else None
    if isinstance(status, int):
        return status in _TRANSIENT_STATUS_CODES
    return any(marker in msg for marker in _TRANSIENT_AUTHOR_MARKERS)


def _complete_with_outage_tolerance(
    llm: Any, system: str, user: str, *, label: str = "",
    budget_secs: float = 7200.0, retry_wait: float = 60.0,
) -> str:
    """P2 (2026-07-13 audit): authoring rode out only ~31 s of API trouble (the transport's
    tenacity budget) while training transport rides out 12 h — a few-minute 529/VPN window killed
    the whole arm days into an unattended run. Ride out TRANSIENT authoring failures for up to
    ``budget_secs`` (2 h), retrying each ``retry_wait``; non-transient errors (401/404/400 — wrong
    key/model/request) re-raise IMMEDIATELY (no spend-burning retries on a permanent problem).
    Held under the author lock BY DESIGN: during an API outage no arm can author anyway, and the
    serialized wait preserves the arm-serial authoring order."""
    deadline = time.monotonic() + float(budget_secs)
    while True:
        try:
            return str(llm.complete(system, user))
        except Exception as exc:  # noqa: BLE001 — classify, then re-raise or ride out
            msg = f"{type(exc).__name__}: {exc}".lower()
            if not _is_transient_author_error(exc) or time.monotonic() >= deadline:
                raise
            _LOG.warning("[author%s] transient API failure (%.0f min budget left): %s",
                         f":{label}" if label else "", (deadline - time.monotonic()) / 60.0,
                         msg[:200])
            time.sleep(retry_wait)


def _search_spec(
    arm: str, src: str, cid: str, opts: dict, prompt: str, gen: int, spec_archive_root: str
) -> dict[str, Any]:
    """The on-node search spec: the certified ``parallel._spec`` PLUS the three fields the cluster
    archival needs — ``archive_root`` (where ``run_one`` writes), ``generation``, and ``prompt``
    (threaded so the record carries the exact authored prompt, provenance parity, directive 6)."""
    from src.orchestration.parallel import _spec

    s = _spec(arm, "source", src, cid, opts)
    s["archive_root"] = str(spec_archive_root)
    s["generation"] = int(gen)
    s["prompt"] = prompt
    s["monitor_queue"] = None  # a Manager queue is unpicklable to JSON; the cluster is unmonitored in-job
    return s


def _load_failures(fail_ledger: Path) -> dict[str, dict]:
    """Read the arm's F5 failures ledger (candidate_id -> row); torn last line is tolerated."""
    cached: dict[str, dict] = {}
    if not fail_ledger.is_file():
        return cached
    for line in fail_ledger.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            f = json.loads(line)
            cached[str(f["candidate_id"])] = f
        except Exception:  # noqa: BLE001 — a torn line just re-authors that one candidate
            pass
    return cached


def _h_rt_seconds(*values: str | None) -> float | None:
    """Largest of the given ``H:M:S`` walltime requests, in seconds (``None`` if none parse).

    LARGEST, deliberately. It feeds :func:`killswitch.classify_task_deaths`'s walltime
    discriminator, which calls a death "walltime-proximate" when it ran >= 90 % of ``h_rt_secs``.
    A LARGER threshold classifies FEWER deaths as walltime kills, i.e. leaves more of them on the
    admin-kill path — the conservative direction under the killswitch's own stated asymmetry (a
    false positive costs hours a human can undo; a false negative costs the account). It is also
    the walltime that governs ~97 % of tasks, the test flood.
    """
    best: float | None = None
    for v in values:
        if not v:
            continue
        try:
            h, m, s = (int(x) for x in str(v).split(":"))
        except ValueError:
            continue
        secs = float(h * 3600 + m * 60 + s)
        if secs > 0 and (best is None or secs > best):
            best = secs
    return best


def _enforce_kill_switch(local_archive_root: str | Path,
                         h_rt_secs: float | None = None) -> None:
    """Classify the mirrored epilogue rows and, on a RETREAT verdict, write the incident file.

    RATIFIED 2026-07-26 (deep review #57). The
    detector, the incident writer and the submission gate all existed and were individually tested,
    but nothing ever called the first two — ``incident_blocks_submission`` was wired at the top of
    this module against a file no code path could create, so the automated access guard could not
    fire. Detection alone is not enough here: the module's premise is that blind resubmission right
    after an administrative ``qdel`` is what escalates "your jobs were killed" into "your account was
    suspended", and an alert nobody reads at 03:00 does not stop the driver resubmitting.

    THE ASYMMETRY (killswitch's own docstring): a false positive costs a few hours on a run with ~7
    days of slack and is undone by a human clearing one file; a false negative costs the account. So
    this enforces — but only on evidence it can trust:

    NOTE: **``h_rt_secs`` IS LOAD-BEARING, and it was missing until 2026-07-27.** ``classify_task_deaths``
    only applies its walltime discriminator ``if h_rt_secs:``; this function called it with nothing,
    so the discriminator was DEAD and **every walltime kill counted as admin-kill evidence**. That
    is not a theoretical gap: the same launch-gate review found ``autosize_h_rt`` sizing CPU tasks at
    6 h against a real 8.55 h need, i.e. a campaign in which ~142 concurrently-dispatched tasks would
    all die at their walltime, on distinct hosts, within minutes of each other — the exact shape
    (>= 8 deaths, >= 4 hosts, <= 300 s) that writes ``MYRIAD_KILL_INCIDENT.json`` and hard-blocks
    EVERY subsequent submission until a human clears it by hand. A sizing bug would have become a
    total, silent campaign halt, and the retreat would have looked like a correctly-working safety
    system. The two defects were fixed together, and this one stays fixed even if a walltime is ever
    mis-sized again.

    * ``n_undated > 0`` ⇒ **do NOT enforce.** Undated rows are exactly the #56 false-positive shape
      (a whole campaign's scattered failures read as one burst). A ledger written by a jobscript
      predating the ``ts`` fix, or a torn row, must never trigger a retreat.
    * only ``action == "retreat"`` writes; every other verdict is inert.

    Best-effort by contract: the caller wraps this in the same swallow as the ledger sync, so a
    forensics failure can never fail a pull. Writing is idempotent-safe — ``write_incident``
    replaces atomically, and the gate blocks on ANY uncleared incident, so a repeated verdict simply
    rewrites the same block rather than compounding.
    """
    from src.cluster.killswitch import classify_task_deaths, write_incident
    from src.cluster.ledger import read_epilogue

    ledger_dir = Path(local_archive_root) / "ledger"
    if not ledger_dir.is_dir():
        return
    rows: list[dict[str, Any]] = []
    for p in sorted(ledger_dir.glob("*.epilogue.jsonl")):
        rows.extend(read_epilogue(p))
    if not rows:
        return
    verdict = classify_task_deaths(rows, h_rt_secs=h_rt_secs)
    if verdict.n_undated:
        _LOG.warning(
            "killswitch NOT enforced: %d undated epilogue rows (a pre-`ts` jobscript or torn "
            "lines). Detection is degraded and an undated ledger is the known false-positive "
            "shape, so no incident was written. Re-render the jobscript and resubmit.",
            verdict.n_undated,
        )
        return
    if verdict.action != "retreat":
        return
    path = write_incident(local_archive_root, verdict)
    _LOG.critical(
        "MYRIAD ADMIN-KILL RETREAT: %s — incident written to %s. Submission is now BLOCKED until a "
        "human clears it (killswitch.clear_incident). Do NOT resubmit; check for mail from "
        "rc-support@ucl.ac.uk first.", verdict.reason, path,
    )


def _ledger_failure(fail_ledger: Path, row: dict) -> None:
    """Append+flush one F5 failure row (best-effort — a ledger write must never crash the arm)."""
    try:
        fail_ledger.parent.mkdir(parents=True, exist_ok=True)
        with fail_ledger.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()
    except Exception as exc:  # noqa: BLE001 — best-effort by design, but never SILENT (2026-07-26)
        # Swallowing stays correct (a ledger write must never crash the arm), but the loss was
        # invisible, and this ledger is READ twice: `_load_failures` seeds `cached_failures` on
        # --resume (a dropped row re-authors that candidate = a wasted PAID LLM call), and
        # `integrity.py:42` counts failures.jsonl for the campaign integrity census (a dropped row
        # under-reports it). Both corruptions were undetectable. Warn, then continue as before.
        _LOG.warning("failure-ledger write FAILED (%s: %s) — row lost from %s; --resume may "
                     "re-author this candidate and the integrity census will under-count",
                     type(exc).__name__, exc, fail_ledger)


# --------------------------------------------------------------------------- #
# k-seed search selection (B-A2) — the σ_seed selection-noise denoiser          #
# --------------------------------------------------------------------------- #
def _candidate_seed_runs(cid: str, k_seeds: int, base_seed: int) -> list[tuple[str, int]]:
    from src.search.multiseed import candidate_seed_ids

    return candidate_seed_ids(cid, k_seeds, base_seed)


def _read_candidate(
    cid: str, arm: str, arm_root: Path, k_seeds: int, base_seed: int
) -> dict[str, Any] | None:
    """A candidate's result from the archive: the single record (k=1) or the k-seed AGGREGATE (k>1);
    ``None`` if any required seed record is missing (candidate incomplete / failed)."""
    if k_seeds <= 1:
        return _result_from_record(cid, arm, arm_root)
    from src.search.multiseed import aggregate_k_seeds

    per_seed: list[dict[str, Any]] = []
    for run_id, _ in _candidate_seed_runs(cid, k_seeds, base_seed):
        r = _result_from_record(run_id, arm, arm_root)
        if r is None:
            return None
        per_seed.append(r)
    agg = aggregate_k_seeds(cid, arm, per_seed)
    return agg if agg.get("ok") else None


def failed_marker_path(arm_root: Path | str, cid: str) -> Path:
    """Sidecar recording that candidate ``cid`` was ATTEMPTED and failed (deep review #62).

    A failed candidate archives no ``record.json``, so ``_read_candidate`` returns None for it —
    which on a resume is indistinguishable from "never attempted", and the candidate is RE-TRAINED.
    If the retry succeeds, the search sees a different fitness at that index than the original run
    did, and every subsequent GP proposal diverges. MEASURED: an interrupted chain and an
    uninterrupted one selected entirely different winners.

    This is a ``.failed.json`` SIDECAR, deliberately not a ``record.json``: the results archive keeps
    its "a record means a completed training" meaning, and nothing downstream (``src/io/results.py``,
    the analysis) can mistake a failure for a result. Only the search-resume path consults it.
    """
    return Path(arm_root) / f"{cid}.failed.json"


def record_failed_candidate(arm_root: Path | str, cid: str, *, reason: str = "") -> None:
    """Mark ``cid`` as attempted-and-failed so a resume reproduces the sentinel deterministically."""
    p = failed_marker_path(arm_root, cid)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"candidate_id": cid, "failed": True, "ts": time.time(),
                                   "reason": str(reason)[:400]}, indent=2), encoding="utf-8")
        tmp.replace(p)
    except OSError as exc:  # noqa: BLE001 — a marker write must never crash the arm
        _LOG.warning("could not record failed-candidate marker for %s: %s", cid, exc)


def candidate_failed_before(arm_root: Path | str, cid: str) -> bool:
    """True iff ``cid`` was already attempted and failed (so replay must NOT re-train it)."""
    return failed_marker_path(arm_root, cid).is_file()


def _archived_source(
    cid: str, arm: str, arm_root: Path, k_seeds: int, base_seed: int
) -> tuple[str, str] | None:
    """``(reward_source, prompt)`` from ANY already-archived seed of this candidate — so a
    PARTIAL-candidate resume (some of the k seeds done) re-submits the missing seeds with the
    SAME source instead of re-authoring (a fresh non-deterministic author would orphan the
    completed seeds), and the remaining seeds' records carry the REAL authored prompt
    (P17/A1-F10: they previously archived a ``<resumed>`` placeholder — a provenance gap
    against directive 6). ``None`` when no seed is archived yet (a genuinely fresh candidate)."""
    for run_id, _ in _candidate_seed_runs(cid, k_seeds, base_seed):
        r = _result_from_record(run_id, arm, arm_root)
        if r is not None:
            return str(r.get("reward_source", "")), str(r.get("prompt", ""))
    return None


# --------------------------------------------------------------------------- #
# SEARCH — the reflection generation loop, per-generation cluster arrays        #
# --------------------------------------------------------------------------- #
def run_search_arm(arm: str, opts: dict, run: ClusterRun, *, resume: bool = False,
                   priority: int = 0) -> dict[str, Any]:
    """Reflect-on-BEST SEARCH for ONE arm, batching each generation as a single SGE array.

    Mirrors ``parallel._drive_llm_arm`` step-for-step (author cpg candidates per generation →
    select the generation BEST by val DSR → build the next reflection block via
    ``schema.build_block``), swapping ONLY the per-candidate ``pool.submit`` for a per-generation
    ``run.run_batch``. Resume replays already-archived candidates (no re-author, no re-train) and
    honours the F5 failures ledger, so an interrupted search resumes byte-faithfully.
    """
    from src.feedback import schema
    from src.llm.loop import _REFLECTION_PREAMBLE, _diversity_directive
    from src.orchestration.parallel import _summary
    from src.sandbox.executor import extract_reward_source

    arm_root = run.search_read() / arm  # SEARCH sub-root (disjoint from the test records)
    fail_ledger = arm_root / "failures.jsonl"
    # Register what THIS run trains for, so `_result_from_record` can refuse a record produced under
    # a different budget (see the guard there; a 2,000-step prototype record once seeded a real
    # 400,000-step run's selection).
    _steps = opts.get("train_steps")
    if _steps:
        _EXPECTED_TRAIN_STEPS[str(arm_root)] = int(_steps)
    if resume and run.pull is not None:
        # P11 (2026-07-13 audit): replay/authoring decisions below read the LOCAL mirror — a crash
        # between the last pull and the arrays finishing makes completed candidates look absent,
        # and each would be RE-AUTHORED (paid) then discarded by archive-truth training. Refresh
        # first; if the mirror cannot be refreshed, fail loud rather than pay twice.
        for _attempt in range(3):
            try:
                run.pull()
                break
            except Exception as exc:  # noqa: BLE001 — bounded retry, then loud
                if _attempt == 2:
                    raise RuntimeError(
                        f"[{arm}] resume requires a FRESH mirror (3 pull attempts failed: {exc}) "
                        "— resuming on stale state would re-author paid candidates"
                    ) from exc
                time.sleep(30.0)
    cached_failures = _load_failures(fail_ledger) if resume else {}
    llm, prompts, diversity = _build_cluster_author(arm, opts, arm_root)

    gens = max(1, int(opts["generations"]))
    cpg = max(1, int(opts["candidates"]) // gens)
    # k-seed search selection (B-A2): each candidate trains at k seeds {base..base+k-1} and is
    # SELECTED on the IQM of its per-seed scores. Default 1 = byte-identical to the single-seed path
    # (run_id == cid, no aggregation). k>1 fans each candidate out to {cid}-s{j} at consecutive seeds.
    k_seeds = max(1, int(opts.get("search_seeds_per_candidate", 1)))
    base_seed = int(opts["seed"])
    accepted: list[dict[str, Any]] = []
    failed = 0
    prev_block: str | None = None

    for gen in range(gens):
        user = prompts.initial if prev_block is None else f"{_REFLECTION_PREAMBLE}\n{prev_block}"
        cids: list[str] = []
        replayed: dict[str, dict] = {}
        fresh: list[tuple[str, str, str]] = []  # (cid, source, prompt) to train THIS generation
        for ci in range(cpg):
            cid = f"{arm}-g{gen}-c{ci}"
            cids.append(cid)
            if resume:
                done = _read_candidate(cid, arm, arm_root, k_seeds, base_seed)
                if done is not None:  # all k seeds archived -> replay, no re-author, no re-train
                    replayed[cid] = done
                    _advance(llm)
                    continue
                if cid in cached_failures:
                    # P3 (2026-07-13 audit): an INFRA-killed candidate (h_rt kill, node death,
                    # requeue exhaustion) was ledgered permanently and its ~paid authoring
                    # stranded — resume now RESUBMITS from the ledger row's stored source
                    # (no re-author; candidate identity preserved). A deterministic sandbox
                    # reject simply fails again and re-ledgers (bounded by the node retry cap).
                    # P8: rows marked permanent (author-side AST rejects) are NEVER re-shipped.
                    _row = cached_failures[cid]
                    _led_src = str(_row.get("reward_source") or "")
                    _advance(llm)
                    if _led_src.strip() and not _row.get("permanent"):
                        # P17/A1-F10: carry the REAL authored prompt from the ledger row (it has
                        # always stored it) — the old placeholder string became the archived
                        # prompt of the recovered record, a provenance gap (directive 6).
                        fresh.append((cid, _led_src,
                                      str(_row.get("prompt") or "<resubmitted-from-failure-ledger>")))
                    else:
                        failed += 1  # permanent reject or no stored source -> stays abandoned
                    continue
                prior = _archived_source(cid, arm, arm_root, k_seeds, base_seed)
                if prior is not None:  # PARTIAL candidate -> reuse the archived source (no re-author)
                    _advance(llm)
                    prior_src, prior_prompt = prior
                    fresh.append((cid, prior_src, prior_prompt or "<resumed>"))
                    continue
            cand_user = user
            if diversity and cpg > 1:
                cand_user = f"{user}\n\n{_diversity_directive(ci, cpg)}"
            with run.author_lock:
                run.author_guard()  # hard spend cap, under the lock (thread-safe shared counter)
                src = extract_reward_source(
                    _complete_with_outage_tolerance(llm, prompts.system, cand_user, label=cid)
                )
            # P8 (2026-07-13 audit): a truncated/refused completion previously SHIPPED to a node,
            # burned a queue slot + a poll cycle, fast-failed 3x, and its F5 row blamed the sandbox.
            # A spawn-free AST gate here catches it at zero cost; the row is marked PERMANENT so
            # the P3 resume-resubmit never re-ships a deterministic author-reject.
            from src.sandbox.executor import ast_gate, defines_reward

            # Deep review 2026-07-26: `ast_gate` alone did NOT deliver what P8 above promises. It
            # answers "is this SAFE", and `ast_gate("")` is True — nothing dangerous in nothing — so
            # an EMPTY completion (the canonical refusal/content_filter output per client.py
            # `_INCOMPLETE_STOP_REASONS`), a whitespace/comment-only block, or valid Python with no
            # reward at all still SHIPPED to a node: precisely the truncated/refused case P8 exists
            # to stop. Only the syntactically-invalid subset (e.g. prose) was caught. Require the
            # function the contract needs as well — still spawn-free, still zero cost.
            if not ast_gate(src) or not defines_reward(src):
                failed += 1
                _ledger_failure(fail_ledger, {
                    "candidate_id": cid, "generation": gen, "permanent": True,
                    "error": (
                        "author_reject: ast_gate (unsafe construct)" if not ast_gate(src)
                        else "author_reject: no top-level 'reward' binding "
                             "(empty/refused/truncated completion)"
                    ),
                    "reward_source": src, "prompt": cand_user,
                })
                continue
            fresh.append((cid, src, cand_user))

        # Train: k_seeds specs per fresh candidate ({cid}-s{j} at base+j), ONE array for the generation.
        if fresh:
            specs = []
            for cid, src, prompt in fresh:
                for run_id, seed in _candidate_seed_runs(cid, k_seeds, base_seed):
                    s = _search_spec(arm, src, run_id, opts, prompt, gen, run.search_spec_root())
                    s["seed"] = seed  # per-seed seed (base+j); k=1 keeps base
                    if k_seeds > 1:
                        s["emit_train_returns"] = True  # the k-seed FED tail is TRAIN-window (B-A2)
                    specs.append(s)
            # MODE-D phase-adaptive packing: the generation wave is on the 6-deep reflection
            # CRITICAL PATH — run it at the latency pack (with its matching tight h_rt) when
            # configured; None keeps the certified uniform-pack path byte-identical.
            _skw: dict[str, Any] = {}
            if run.search_h_rt:
                _skw["h_rt_call"] = run.search_h_rt
            if run.search_poll_secs:
                _skw["poll_call"] = run.search_poll_secs
            if run.search_threads > 1:                   # R107 (search/chain leg only); the
                _skw["threads"] = run.search_threads     # 1-thread default stays byte-identical
            run.run_batch(specs, f"{arm}_g{gen}", pool=run.pool_confirmatory,
                          pack=(run.search_pack or run.pack), priority=priority, **_skw)

        # Collect in candidate order (parity); aggregate each candidate's k seeds; ledger failures (F5).
        fresh_by_cid = {cid: (src, prompt) for (cid, src, prompt) in fresh}
        best: dict[str, Any] | None = None
        for cid in cids:
            r = replayed.get(cid)
            if r is None:
                r = _read_candidate(cid, arm, arm_root, k_seeds, base_seed)
            if r is None:  # a seed missing/failed after run_batch -> the candidate FAILED
                if cid in fresh_by_cid:
                    failed += 1
                    src, prompt = fresh_by_cid[cid]
                    # P9 (2026-07-13 audit): consult the node-side reject markers (mirrored under
                    # <search>/_rejects/) — a PERMANENT one (deterministic sandbox/contract reject)
                    # marks the F5 row permanent, so the P3 resume-resubmit never re-ships the same
                    # invalid source, and the row carries the node's ACTUAL error (diagnosability).
                    _perm, _err = False,                         "cluster training failed (a seed sandbox-rejected or exhausted retries)"
                    for _rid, _ in _candidate_seed_runs(cid, k_seeds, base_seed):
                        _mp = arm_root.parent / "_rejects" / f"{_rid}.json"
                        if _mp.is_file():
                            try:
                                _m = json.loads(_mp.read_text(encoding="utf-8"))
                            except Exception:  # noqa: BLE001 — a torn marker keeps the default row
                                continue
                            _err = f"node reject: {str(_m.get('error'))[:400]}"
                            if _m.get("permanent"):
                                _perm = True
                                break
                    _ledger_failure(fail_ledger, {
                        "candidate_id": cid, "generation": gen, "prompt": prompt,
                        "reward_source": src, "error": _err,
                        **({"permanent": True} if _perm else {}),
                    })
                continue
            accepted.append(r)
            if best is None or r["fitness"] > best["fitness"]:
                best = r

        if best is not None:
            tail_for = best.get("tail_stats") if arm in _TAIL_FED_ARMS else None
            prev_block = schema.build_block(
                arm, best["fitness"], tail_for,
                shuffle_seed=(
                    schema.shuffle_seed_from_id(str(best.get("candidate_id", "")))
                    if arm == "placebo_shuffled" else None
                ),
            )
    return _summary(arm, accepted, failed, gens * cpg)


#: The H4 family-search arms — NOT LLM-authored; they sample/optimize the reward FAMILY. Mirrors
#: parallel._LLM_ARMS's complement (run_parallel dispatches _drive_search_arm for these).
_FAMILY_ARMS = ("random_search", "bayes_opt", "cma_es", "tpe")   # H4 optimiser portfolio (N4 confirmatory, 2026-07-26)


def _family_spec(arm: str, kind: str, reward: Any, cid: str, opts: dict, spec_archive_root: str
                 ) -> dict[str, Any]:
    """The on-node family-search spec: ``parallel._spec`` (kind ``source`` for random_search /
    ``coeffs`` for bayes_opt) + the cluster archival fields. No authored prompt (no LLM)."""
    from src.orchestration.parallel import _spec

    s = _spec(arm, kind, reward, cid, opts)
    s["archive_root"] = str(spec_archive_root)
    s["generation"] = 0
    s["monitor_queue"] = None
    return s


def run_family_search_arm(arm: str, opts: dict, run: ClusterRun, *, resume: bool = False,
                          priority: int = 0) -> dict[str, Any]:
    """H4 family SEARCH for ``random_search`` / ``bayes_opt`` (mirrors ``parallel._drive_search_arm``).

    ``random_search`` samples K reward sources UP FRONT (deterministic rng → byte-identical stream +
    winner to the laptop) and trains them as ONE array (no reflection). ``bayes_opt`` runs the GP on
    the DRIVER (``bayes_opt_over_template``) and trains each proposed coefficient vector as an
    array-of-1 — inherently sequential (the GP proposes from prior fitnesses), hidden under the other
    arms' floods on the cluster. Both reuse the certified search primitives + the search-replay resume.
    """
    import numpy as np

    from src.orchestration.parallel import _summary

    arm_root = run.search_read() / arm
    n = max(1, int(opts["candidates"]))
    # k-seed selection (B-A2) applies to the H4 family arms too, so the LLM-vs-family comparison is
    # NOT confounded by k. Default 1 = byte-identical single-seed.
    k_seeds = max(1, int(opts.get("search_seeds_per_candidate", 1)))
    base_seed = int(opts["seed"])

    def _family_specs(cid: str, kind: str, reward: Any) -> list[dict[str, Any]]:
        out = []
        for run_id, seed in _candidate_seed_runs(cid, k_seeds, base_seed):
            s = _family_spec(arm, kind, reward, run_id, opts, run.search_spec_root())
            s["seed"] = seed  # per-seed seed (base+j); k=1 keeps base
            if k_seeds > 1:
                s["emit_train_returns"] = True  # the k-seed FED tail is TRAIN-window (B-A2)
            out.append(s)
        return out

    if arm == "random_search":
        from src.search.random_search import sample_reward_source

        rng = np.random.default_rng(opts["seed"])
        sources = [sample_reward_source(rng, opts.get("proto_cfg")) for _ in range(n)]  # ALL up front
        cids = [f"{arm}-c{i}" for i in range(n)]
        fresh = [
            (cid, src) for cid, src in zip(cids, sources)
            if not (resume and _read_candidate(cid, arm, arm_root, k_seeds, base_seed) is not None)
        ]
        if fresh:
            specs = [s for cid, src in fresh for s in _family_specs(cid, "source", src)]
            # R107: pass threads ONLY when raised, so the 1-thread path is byte-identical (and
            # the ``run_batch`` seam keeps its pre-R107 signature for every existing caller).
            _sekw = {"threads": run.search_threads} if run.search_threads > 1 else {}
            # SEARCH-LANE PACK (2026-07-27). This is a SEARCH batch, so it takes the search lane's
            # pack exactly like the LLM chain at run_llm_search does — it did not, and that was the
            # whole reason R107's 8 threads looked unusable. Job cores are
            # ``max(cores_per_training, threads) x pack``, so 8 threads against the TEST flood's
            # pack (4) asks for 32 cores: under the 36-core exclusivity refusal, so not caught, but
            # far past the measured placement cliff (8-core ~19 min, 16-core never placed in 28).
            # With the search lane at pack 1 the same request is 8 cores — placeable — which is
            # what ``--search-pack`` exists for. It also matters that this batch is NOT throughput
            # work despite looking like a burst: the generation is a BARRIER (nothing reflects
            # until every candidate lands), so the wave's cost is ONE training's latency, which is
            # precisely where the register says to spend threads.
            run.run_batch(specs, f"{arm}_search", pool=run.pool_confirmatory,
                          pack=(run.search_pack or run.pack), priority=priority, **_sekw)
        accepted, failed = [], 0
        for cid in cids:
            r = _read_candidate(cid, arm, arm_root, k_seeds, base_seed)
            if r is None:
                failed += 1
            else:
                accepted.append(r)
        return _summary(arm, accepted, failed, n)

    # bayes_opt — driver-side GP, per-iteration cluster training (each iteration = one k-seed array).
    from src.baselines.reward_family import family_bounds
    from src.search.dfo_toolkit import over_template_optimizer  # GP-EI / CMA-ES / TPE, resolved by arm

    state: dict[str, Any] = {"i": 0, "accepted": [], "failed": 0}

    def template_eval(coeffs: Any) -> float:
        i = state["i"]
        state["i"] = i + 1
        cid = f"{arm}-c{i}"
        if resume:
            r = _read_candidate(cid, arm, arm_root, k_seeds, base_seed)
            if r is not None:
                state["accepted"].append(r)
                return float(r["fitness"])
            # Same determinism guard as the on-node chain (#62): a candidate that was attempted and
            # FAILED archives no record, so replay would re-train it, and a retry that succeeds
            # changes the fitness at this index and every proposal after it. Replay the sentinel.
            if candidate_failed_before(arm_root, cid):
                state["failed"] += 1
                return -1e9
        # MODE-D: the BO chain (30 sequential GP proposals) is the campaign's LONGEST serial path —
        # already pack-1 per proposal; give it the tight search h_rt when configured (a 1.1h
        # training under a 6-7h pack-5 walltime request is a poor backfill candidate).
        _bkw: dict[str, Any] = {}
        if run.search_h_rt:
            _bkw["h_rt_call"] = run.search_h_rt
        if run.search_poll_secs:
            _bkw["poll_call"] = run.search_poll_secs   # 30 chain steps x up to 180s notice = ~1h+
        if run.search_threads > 1:                     # R107 (search/chain leg only); the
            _bkw["threads"] = run.search_threads       # 1-thread default stays byte-identical
        run.run_batch(_family_specs(cid, "coeffs", list(coeffs)), f"{arm}_c{i}",
                      pool=run.pool_confirmatory, pack=(run.pack if k_seeds > 1 else 1),
                      priority=priority, **_bkw)
        r = _read_candidate(cid, arm, arm_root, k_seeds, base_seed)
        if r is None:
            state["failed"] += 1
            record_failed_candidate(arm_root, cid, reason="no record after run_batch")
            return -1e9
        state["accepted"].append(r)
        return float(r["fitness"])

    def template_eval_batch(coeffs_list: Any) -> list[float]:
        """Evaluate a WHOLE batch concurrently as ONE cluster array (TPE's startup phase).

        TPE's ``n_startup`` trials come from Optuna's RANDOM sampler and depend on no observed
        value, so dispatching them together is a pure DISPATCH change with identical results — it
        cuts TPE's SERIAL chain from the full budget (30) to ``budget - n_startup`` (~20), which
        matters because a serial chain is the campaign's critical path (``src/cluster/lanes.py``).
        Candidate ids stay contiguous with the sequential phase via the shared ``state["i"]``, and
        archived candidates replay for free so resume is unaffected.
        """
        coeffs_list = [list(c) for c in coeffs_list]
        idx0 = state["i"]
        state["i"] = idx0 + len(coeffs_list)
        cids = [f"{arm}-c{idx0 + j}" for j in range(len(coeffs_list))]

        fresh = []
        for cid, coeffs in zip(cids, coeffs_list):
            if resume and _read_candidate(cid, arm, arm_root, k_seeds, base_seed) is not None:
                continue                      # already archived -> replayed below, no retraining
            fresh.append((cid, coeffs))
        if fresh:
            _bkw_b: dict[str, Any] = {}
            if run.search_h_rt:
                _bkw_b["h_rt_call"] = run.search_h_rt
            if run.search_poll_secs:
                _bkw_b["poll_call"] = run.search_poll_secs
            if run.search_threads > 1:                    # R107 (search/chain leg only); the
                _bkw_b["threads"] = run.search_threads    # 1-thread default is byte-identical
            specs = [s for cid, coeffs in fresh for s in _family_specs(cid, "coeffs", coeffs)]
            # SEARCH-LANE PACK (2026-07-27), same reasoning as the ``{arm}_search`` batch above:
            # these are the DFO startup design points, a BARRIER before the strictly-sequential
            # ask/tell loop, so the wave costs one training's latency and must not be sized against
            # the test flood's pack (8 threads x pack 4 = an unplaceable 32-core request).
            # D27 (2026-08-02): CMA-ES calls this ONCE PER GENERATION, so a FIXED batch name would put
            # three generations into one batch directory, one `.driver.lock` and one
            # `.permanent.jsonl` — and a stale lock of exactly that kind cost this campaign 4.5 h on
            # `cma_es-c5` on 2026-08-01. Keying the name on `idx0` makes each call unique, and
            # `idx0 == 0` keeps it BYTE-IDENTICAL for tpe/bayes_opt, which batch exactly once.
            _bname = f"{arm}_startup" if idx0 == 0 else f"{arm}_gen{idx0}"
            run.run_batch(specs, _bname, pool=run.pool_confirmatory,
                          pack=(run.search_pack or run.pack), priority=priority, **_bkw_b)

        out: list[float] = []
        for cid in cids:
            r = _read_candidate(cid, arm, arm_root, k_seeds, base_seed)
            if r is None:
                state["failed"] += 1
                out.append(-1e9)              # the same sentinel the sequential path feeds
            else:
                state["accepted"].append(r)
                out.append(float(r["fitness"]))
        return out

    # BATCH-CAPABLE ARMS. Both GP-EI and TPE open with a phase of points fixed by the rng BEFORE
    # any training runs (bayes_opt's n_init=5 i.i.d. uniform draws; TPE's n_startup random trials),
    # so dispatching each phase as ONE array is a pure DISPATCH change with identical results.
    # It matters because the driver turns every `template_eval` into its own array-of-1 job: left
    # sequential, bayes_opt's "5 parallel" init is really 5 more SERIAL queue-and-train steps,
    # making the GP chain 30 rather than 25 on the campaign's critical path.
    # random_search is already ONE array and does not accept the kwarg.
    #
    # IMPORTANT: CORRECTED 2026-08-02 (D27). This block used to read "CMA-ES already dispatches a whole
    # population per generation; neither accepts the kwarg" — and that was FALSE. CMA-ES *proposes* a
    # whole population per generation; `dfo_toolkit.cma_es_over_template` then evaluated it with a
    # SERIAL list comprehension, and on the cluster path every one of those calls is a blocking
    # `run_batch`. The false comment is why `batch_eval_fn` was wired here for tpe and bayes_opt and
    # never for cma_es, and why `lanes._CMA_SERIAL_GENERATIONS` was set to 4 for a chain that is
    # really 30. Measured cost on RUN 4: `cma_es` at 9/30 with a 21-step remaining chain of ~8.6 h per
    # step, gating the CORE line's C1 barrier and therefore the campaign's COMMON rung — while the
    # sentinel, reading the same wrong constant, reported the arm COMPLETE at "9/4".
    #
    # The classic failure: when a comment and the code disagree, the COMMENT is the more
    # dangerous artefact. The kwarg is the same one, the safety argument is the same one, and
    # `tests/test_dfo_cma_batch.py` measures the identity rather than asserting it.
    _opt_kwargs: dict[str, Any] = {"rng": np.random.default_rng(opts["seed"])}
    if arm in ("tpe", "bayes_opt", "cma_es"):
        _opt_kwargs["batch_eval_fn"] = template_eval_batch

    over_template_optimizer(arm)(
        template_eval, family_bounds(opts.get("proto_cfg")), {"matched_budget": n},
        **_opt_kwargs,
    )
    return _summary(arm, state["accepted"], state["failed"], n)


# --------------------------------------------------------------------------- #
# TEST — the sealed leg as ONE embarrassingly-parallel array                    #
# --------------------------------------------------------------------------- #
def run_test_leg(
    winners: list[tuple[str, dict[str, Any]]],
    seeds: list[int],
    run: ClusterRun,
    *,
    panel_descriptor: dict[str, Any],
    env_cfg: Any,
    agent_cfg: dict[str, Any],
    train_window: tuple[int, int],
    val_window: tuple[int, int],
    test_window: tuple[int, int],
    embargo: int,
    lookback: int,
    name: str = "test",
    pool: str | None = None,
    resume: bool = True,
    priority: int = 0,
    interleave: bool = False,
) -> dict[str, Any]:
    """Run the sealed test leg for ``winners`` × ``seeds`` as ONE array (``-tc`` = full pool).

    Reuses ``test_leg.build_test_specs`` (the single-source spec schema + the frozen/test desync
    guard) so the cluster trains EXACTLY what the local pool would; stamps the remote
    ``archive_root`` onto each spec so on-node ``run_one`` routes to ``_test_seed_worker`` and
    archives the record. Resume skips already-archived ``{arm}-s{seed}`` records.

    ``interleave=True`` reorders the taskfile SEED-MAJOR (pair-adjacent / round-robin across the
    winners: unit1-s0, unit2-s0, …, unit1-s1, …) — the §14.3/B-A3 schedule, so at ANY truncation
    point every unit holds a (near-)equal, CRN-paired seed count instead of one unit hogging the
    early tasks. ``priority`` = the §14.3 -p ladder value for this array.
    """
    from src.cluster.poll import completed_run_ids
    from src.orchestration.test_leg import build_test_specs

    test_read = run.test_read()  # TEST sub-root (disjoint from the search records)
    done = completed_run_ids(test_read) if resume else set()
    specs = build_test_specs(
        winners=winners, seeds=seeds, panel_descriptor=panel_descriptor, env_cfg=env_cfg,
        agent_cfg=agent_cfg, train_window=train_window, val_window=val_window,
        test_window=test_window, embargo=embargo, lookback=lookback, test_root=test_read,
        done_ids=done,
    )
    for s in specs:
        s["archive_root"] = run.test_spec_root()  # where on-node run_one writes the test record
    if interleave and len(winners) > 1:
        unit_order = {arm: i for i, (arm, _) in enumerate(winners)}
        specs.sort(key=lambda s: (int(s["seed"]), unit_order.get(str(s["arm"]), len(unit_order))))
    if not specs:
        return {"name": name, "ok": True, "submitted": 0, "note": "all seeds already archived"}
    base_pool = pool or run.pool_confirmatory
    if run.seed_pool_blocks:
        # Device-stratified seed blocks (2026-07-11c): partition the taskfile BY SEED so each
        # block's array runs on its pool. Every CRN pair (same seed, all arms) stays on ONE device
        # class; the interleave order is preserved WITHIN each block. Unassigned seeds fall back to
        # ``base_pool`` under the original batch name — no spec is ever dropped.
        assigned: set[int] = set()
        jobs: list[tuple[str, str, list[dict[str, Any]]]] = []  # (pool, batch name, taskfile part)
        for block_pool, block_seeds in run.seed_pool_blocks:
            part = [s for s in specs if int(s["seed"]) in block_seeds]
            assigned |= block_seeds
            if part:
                jobs.append((block_pool, f"{name}_{block_pool}", part))
        rest = [s for s in specs if int(s["seed"]) not in assigned]
        if rest:
            jobs.append((base_pool, name, rest))
        # P16 (2026-07-13 audit): blocks drive CONCURRENTLY (one thread per pool) — the old
        # serial loop idled the second pool for the whole first block, forfeiting exactly the
        # extra confirmatory C the device-stratified blocks were ratified to add. Concurrent
        # run_batch is the established pattern (the per-arm pipelines already do it): distinct
        # batch names, per-batch P12 locks, and the shared pull is internally serialized.
        out: dict[str, Any] = {"name": name, "ok": True, "blocks": {}}
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as ex:
            futs = {ex.submit(run.run_batch, part, bname, pool=bpool, pack=run.pack,
                              priority=priority): bpool
                    for bpool, bname, part in jobs}
            for f, bpool in futs.items():
                r = f.result()
                out["blocks"][bpool] = r
                out["ok"] = out["ok"] and bool(r.get("ok", True))
        return out
    return run.run_batch(specs, name, pool=base_pool, pack=run.pack,
                         priority=priority)


def run_baselines_on_cluster(
    baseline_names: list[str],
    seeds: list[int],
    run: ClusterRun,
    *,
    test_leg_kwargs: dict[str, Any],
    name: str = "baselines",
    resume: bool = True,
    priority: int = 0,
    interleave: bool = False,
) -> dict[str, Any]:
    """Run the H1 hand-designed baselines (PREREG §1) as ONE embarrassingly-parallel test array.

    A baseline is a FIXED canonical reward — no search/select/freeze — so it depends on NOTHING the
    search produces and floods the pool from minute 0 (PLAN §5: "H1 flood, 1,612 parallel"). Reuses
    the single-source ``run_campaign._baseline_winner_record`` (arm=``baseline_<name>``,
    reward_kind=``baseline``, reward_name=``<name>``) so the on-node ``_test_seed_worker`` resolves the
    canonical callable BY NAME from ``src.baselines.rewards`` — identical to the laptop H1 leg.
    """
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from run_campaign import _baseline_winner_record  # type: ignore[import-not-found]

    winners = [(f"baseline_{nm}", _baseline_winner_record(nm)) for nm in baseline_names]
    return run_test_leg(winners, seeds, run, name=name, resume=resume, priority=priority,
                        interleave=interleave, **test_leg_kwargs)


# --------------------------------------------------------------------------- #
# PIPELINE — per-arm search → select → freeze → test (zero barrier)             #
# --------------------------------------------------------------------------- #
def _guard_k_seed_selector(opts: dict, run: ClusterRun) -> None:
    """P15 (2026-07-13 pre-spend audit; DORMANT at the ratified k=1): under k-seed search (B-A2)
    the SELECTION metric is the IQM aggregate (``multiseed.aggregate_k_seeds`` — the value the
    LLM reflection also sees), but the default ``select_winner`` reads the PER-SEED records and
    takes the max — a max-single-lucky-seed freeze would contradict the aggregate semantics and
    re-open the exact selection-noise channel k-seed search was built to close. Fail loud until
    an aggregate-aware selector is injected."""
    if int(opts.get("search_seeds_per_candidate", 1) or 1) > 1 and run.select_winner is None:
        raise NotImplementedError(
            "k-seed search (search_seeds_per_candidate>1) requires an aggregate-aware winner "
            "selector: the default select_winner picks the max SINGLE-SEED val_fitness from "
            "per-seed records, contradicting the IQM selection semantics "
            "(src.search.multiseed.aggregate_k_seeds). Inject run.select_winner before "
            "enabling k>1."
        )


def _refuse_winner_swap(arm: str, winner: dict[str, Any], frozen_root: str | Path) -> None:
    """P5 winner-swap refusal (2026-07-19 audit: hoisted out of the tiered path so EVERY freeze
    site enforces it). If a frozen winner already exists for ``arm`` with a DIFFERENT reward hash,
    resume must refuse rather than silently overwrite — a mixed-winner unit (some seeds trained on
    winner A, some on B) is scientifically meaningless. A P3 ledger-resubmit recovery makes the
    swap non-exotic, and the non-tiered pipeline + the H3 C5 control previously lacked this guard."""
    existing = Path(frozen_root) / f"{arm}-winner" / "record.json"
    if not existing.is_file():
        return
    try:
        old = json.loads(existing.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise RuntimeError(f"corrupt frozen winner record at {existing}: {exc}") from exc
    old_h = str(old.get("reward_source_hash", ""))
    new_h = str(winner.get("reward_source_hash", ""))
    if old_h and new_h and old_h != new_h:
        raise RuntimeError(
            f"[{arm}] WINNER-SWAP REFUSED: a frozen winner already exists with reward hash "
            f"{old_h[:12]}… but this resume selected {new_h[:12]}… — the search archive changed "
            "between runs (config drift?). A mixed-winner unit is scientifically meaningless. "
            "Investigate; delete the frozen record ONLY with an explicit, dated decision.")


def _select_eligible_winner(
    select: Callable[[Any], Any], arm_search_root: Any, label: str
) -> tuple[Any, str | None]:
    """Select an arm's winner, turning the R115 all-contaminated case into a REASON, not a crash.

    Returns ``(winner_or_None, reason_or_None)``; ``reason`` is set only when R115 rejected every
    candidate.

    2026-07-28: R115 made ``select_winner`` raise when NO candidate clears the execution floor. That
    must not propagate here. :func:`run_arm_pipeline` documents that it "never raises for a 'no
    winner' arm", and nothing wraps the three selection sites — so an uncaught raise would kill the
    arm, the supervisor would relaunch the line into the same error, and an unattended multi-day run
    would sit in a 600 s hot loop.

    Degrading is not the same as hiding. The reason is DISTINCT from ``no_winner`` (candidates
    existed; none was eligible — a different diagnosis needing a different response), it is logged at
    ERROR, and the incompleteness is caught downstream by construction: a missing arm makes
    ``present != expected`` in the integrity census, so ``health_ok`` goes false and the C3 review
    gate STOPS the line before the expensive C4 sweep.
    """
    from src.selection.fitness import NoEligibleWinnerError

    try:
        return select(arm_search_root), None
    except NoEligibleWinnerError as exc:
        _LOG.error("[%s] R115 winner-eligibility: %s", label, exc)
        return None, "no_eligible_winner"


def _resolve_select_freeze(run: ClusterRun) -> tuple[Callable[[Any], Any], Callable[..., Any]]:
    """The SELECT/FREEZE callables — injected on ``run`` for tests, else the certified
    ``run_campaign`` implementations (imported via the codebase's scripts-on-path pattern)."""
    select = run.select_winner
    freeze = run.freeze_winner
    if select is not None and freeze is not None:
        return select, freeze
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from run_campaign import freeze_winner as _freeze  # type: ignore[import-not-found]
    from run_campaign import select_winner as _select  # type: ignore[import-not-found]

    return (select or _select), (freeze or _freeze)


def run_arm_pipeline(
    arm: str,
    opts: dict,
    seeds: list[int],
    run: ClusterRun,
    *,
    test_leg_kwargs: dict[str, Any],
    frozen_root: str | Path,
    search_seed: int = 0,
    resume: bool = False,
    priority: int = 0,
) -> dict[str, Any]:
    """The full per-arm pipeline: SEARCH → SELECT winner → FREEZE → sealed TEST leg.

    Run concurrently across arms by :func:`run_campaign_on_cluster`, so an arm's test array floods
    the pool the instant its search finishes — while the other arms are still searching (§5 zero
    barriers). Returns a per-arm result dict; never raises for a "no winner" arm (that arm simply
    contributes no test records — surfaced as ``ok=False, reason="no_winner"``)."""
    select_winner, freeze_winner = _resolve_select_freeze(run)
    # Dispatch on arm type (parity with run_parallel): the 5 LLM arms author+reflect; the H4 family
    # arms (random_search/bayes_opt) sample/optimize the reward family — NOT via an LLM.
    if arm in _FAMILY_ARMS:
        search = run_family_search_arm(arm, opts, run, resume=resume, priority=priority)
    else:
        search = run_search_arm(arm, opts, run, resume=resume, priority=priority)

    arm_search_root = run.search_read() / arm  # SELECT reads ONLY the search sub-root
    _guard_k_seed_selector(opts, run)  # P15: max-single-seed select is WRONG under k>1
    winner, _r115 = _select_eligible_winner(select_winner, arm_search_root, arm)
    if winner is None:
        _LOG.warning("[%s] search produced no %swinner — no test leg",
                     arm, "ELIGIBLE " if _r115 else "")
        return {"arm": arm, "ok": False, "reason": _r115 or "no_winner", "search": search}

    env_fp = winner.get("env_fingerprint") or f"cluster:{arm}:frozen"
    _refuse_winner_swap(arm, winner, frozen_root)  # P5 (hoisted 2026-07-19): non-tiered path too
    freeze_winner(arm, winner, search_seed=search_seed, frozen_root=frozen_root, env_fingerprint=env_fp)

    test = run_test_leg(
        [(arm, winner)], seeds, run, name=f"{arm}_test", resume=resume, priority=priority,
        **test_leg_kwargs
    )
    return {
        "arm": arm, "ok": True, "search": search, "test": test,
        "winner_id": winner.get("candidate_id"),
    }


#: The C5 H3-single-shot invocation. **Retired to 0 with the rest of the §14.3 ladder (2026-07-31).**
#: It was -100 "below the confirmatory core, same report-only class as stage-1"; that ordering is an
#: intra-user intent expressed through a GLOBAL field, which is the error documented on
#: PRIORITY_CORE below. h3ss is a registered line of the campaign and rides at full fair-share
#: standing like every other.
PRIORITY_H3_SINGLESHOT = 0


def run_h3_singleshot_on_cluster(
    opts: dict,
    seeds: list[int],
    run: ClusterRun,
    *,
    test_leg_kwargs: dict[str, Any],
    frozen_root: str | Path,
    search_seed: int = 0,
    resume: bool = False,
    priority: int | None = None,
) -> dict[str, Any]:
    """C5 — the pre-registered H3 SINGLE-SHOT control on the cluster (P4 closed 2026-07-13,
    the standing directive being that the whole campaign runs on Myriad).

    Mirrors ``scripts/run_campaign.run_h3_singleshot`` stage-for-stage on the cluster primitives:
    the **distributional** arm re-run at ``generations = h3_singleshot_generations`` (1 — every
    candidate authors from the INITIAL prompt; ``run_search_arm``'s reflection block never fires
    because ``prev_block`` only becomes non-None after a completed generation), the SAME candidate
    budget and agent config (matched compute, DEEP_H3 §4), the IDENTICAL winner selector, and the
    sealed test leg at the SAME campaign seeds.

    Archives to the SEPARATE ``search_h3_singleshot/`` / ``frozen_h3_singleshot/`` /
    ``test_h3_singleshot/`` roots — laptop-layout parity, so ``analyze_campaign``'s H3 loader reads
    both substrates identically. Root disjointness is STRUCTURAL (a ``dataclasses.replace``d
    ClusterRun): the P4 audit hazard was that same-root reuse would let the compacted resume ADOPT
    headline run_ids and fabricate the H3 null. Batch names are force-prefixed ``h3ss_`` so this
    invocation can never adopt (or be adopted by) the headline campaign's queued arrays even when
    both share a ``--batch-tag`` namespace.
    """
    from dataclasses import replace

    arm = "distributional"  # the H3 contrast lives wholly on the distributional arm (DEEP_H3 §1)
    _headline_rb = run.run_batch

    def _h3_run_batch(specs: list[dict[str, Any]], name: str, **kw: Any) -> dict[str, Any]:
        return _headline_rb(specs, f"h3ss_{name}", **kw)

    h3_run = replace(run, run_batch=_h3_run_batch,
                     search_subdir="search_h3_singleshot", test_subdir="test_h3_singleshot")
    gens = int(opts.get("h3_singleshot_generations", 1) or 1)
    h3_opts = {**opts, "generations": gens}

    h3_prio = PRIORITY_H3_SINGLESHOT if priority is None else int(priority)
    # AUDIT 2026-07-22 (MAJOR): the full-ladder H3 re-run (--seeds 0-567 --resume) must run
    # at RUNG priority per the registered queue; the hardcoded -100 would jump all 10 legs.
    search = run_search_arm(arm, h3_opts, h3_run, resume=resume, priority=h3_prio)
    select_winner, freeze_winner = _resolve_select_freeze(run)
    _guard_k_seed_selector(h3_opts, h3_run)  # P15: max-single-seed select is WRONG under k>1
    winner, _r115 = _select_eligible_winner(select_winner, h3_run.search_read() / arm, "h3ss")
    if winner is None:
        _LOG.warning("[h3ss] single-shot search produced no %swinner — no test leg",
                     "ELIGIBLE " if _r115 else "")
        return {"arm": "distributional_singleshot", "ok": False,
                "reason": _r115 or "no_winner", "search": search}
    winner.setdefault("reward_source", "")
    # Laptop-parity provenance string (run_h3_singleshot freezes with exactly this fingerprint).
    env_fp = winner.get("env_fingerprint") or "campaign:distributional_singleshot"
    _refuse_winner_swap(arm, winner, str(frozen_root))  # P5 (hoisted 2026-07-19): H3 C5 path too
    freeze_winner(arm, winner, search_seed=search_seed, frozen_root=str(frozen_root),
                  env_fingerprint=env_fp)
    test = run_test_leg([(arm, winner)], seeds, h3_run, name=f"{arm}_test",
                        resume=resume, priority=h3_prio, **test_leg_kwargs)
    return {"arm": "distributional_singleshot", "ok": True, "search": search, "test": test,
            "winner_id": winner.get("candidate_id")}


def run_campaign_on_cluster(
    arms: list[str],
    opts_for: Callable[[str], dict],
    seeds: list[int],
    run: ClusterRun,
    *,
    test_leg_kwargs: dict[str, Any],
    frozen_root: str | Path,
    baseline_names: list[str] | None = None,
    resume: bool = False,
    max_concurrent_arms: int | None = None,
    priority: int = 0,
) -> dict[str, dict[str, Any]]:
    """Drive ALL arms' search→test pipelines CONCURRENTLY (§5 all-arms-parallel, zero barriers).

    Each arm runs :func:`run_arm_pipeline` in its own thread; the training arrays interleave freely
    on the cluster while ``run.author_lock`` keeps authoring rate-limit-safe (arm-serial API). The
    H1 ``baseline_names`` (fixed rewards, no search) run as ONE concurrent test flood from minute 0
    (:func:`run_baselines_on_cluster`), keyed ``__baselines__`` in the result. A per-unit crash is
    captured into that unit's result, never taking down the others. The GPU pool is pinned per
    contrast by ``run.pool_confirmatory`` (all confirmatory units share it -> device homogeneity
    within every analysis unit)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    has_baselines = bool(baseline_names)
    if not arms and not has_baselines:
        # VACUOUS-TRUTH GUARD (2026-07-26 review). With zero units this returns {}, and every caller
        # scores the campaign with ``all(r.get("ok") for r in results.values())`` — ``all({})`` is
        # True, so a no-op run reports "ALL OK", writes all_arms_tested=true, and exits 0. A campaign
        # that tested NOTHING must never be indistinguishable from one that tested everything.
        raise ValueError(
            "run_campaign_on_cluster called with no units: arms=[] and baseline_names empty "
            "(nothing would run, yet the caller's all(...) would report the campaign OK)"
        )
    workers = max_concurrent_arms or max(1, len(arms) + (1 if has_baselines else 0))
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="unit") as ex:
        futs = {
            ex.submit(
                run_arm_pipeline, arm, opts_for(arm), seeds, run,
                test_leg_kwargs=test_leg_kwargs, frozen_root=frozen_root, resume=resume,
                # F9 (agent audit): the frozen-winner provenance stamped search_seed=0 regardless
                # of the actual search seed; thread the arm's own opts value (science unaffected —
                # authoring used opts["seed"] correctly — provenance now truthful).
                search_seed=int(opts_for(arm).get("seed", 0)),
                priority=priority,
            ): arm
            for arm in arms
        }
        if has_baselines:
            futs[ex.submit(
                run_baselines_on_cluster, list(baseline_names or []), seeds, run,
                test_leg_kwargs=test_leg_kwargs, resume=resume, priority=priority,
            )] = "__baselines__"
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                res = fut.result()
                # FAIL CLOSED on a missing `ok` (2026-07-28). This defaulted to True, so a result
                # dict without `ok` counted as SUCCESS — and the launcher judges the same value with
                # `bool(out.get("ok"))`, i.e. FAILURE. The two layers disagreed, and the campaign
                # side took the fail-OPEN direction: an unreported failure here proceeds to freeze a
                # winner and run the sealed leg. Every producer sets `ok` explicitly today (verified
                # across all return sites), so this changes NOTHING now; it removes a fail-open
                # default in the class this project has already been bitten by three times
                # (#26 vacuous-truth-on-empty, #28/#29 fail-open-on-absent-evidence).
                results[key] = res if key != "__baselines__" else {"ok": res.get("ok", False),
                                                                    "test": res}
            except Exception as exc:  # noqa: BLE001 — one unit's failure must not sink the campaign
                _LOG.exception("[%s] pipeline crashed", key)
                results[key] = {"arm": key, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return results


#: THE INTRA-USER ``-p`` LADDER IS RETIRED. EVERY VALUE IS 0. (2026-07-31, record §54.)
#:
#: **The conceptual error, stated plainly: ``-p`` is not an intra-user knob.** R88 registered the
#: queue order as an SGE priority ladder on the reasoning that *"under scarce capacity the scheduler
#: starves back-of-queue work first, reproducing the serial semantics"*. That is only true while WE
#: are effectively the sole consumer of the pool. ``-p`` is a GLOBAL POSIX priority, and Myriad
#: weights it at ``weight_priority = 4.0`` — the LARGEST weight in ``qconf -ssconf`` (ticket 1.5,
#: waiting_time 1.0, urgency 0.0). Against a contended queue it does not order our work among
#: itself; it sinks us beneath EVERY other user on the cluster.
#:
#: **R101 (Okhrati seed-parity — all lines in LOCKSTEP at equal seeds) SUPERSEDED R88's ladder.**
#: ``mode_d_launch.ps1`` was updated at the time ("No line passes --priority any more; the default 0
#: is full fair-share standing"). The ARM-level and RUNG-level ladders in *this* module were not —
#: a half-applied amendment, the same failure mode as R106.
#:
#: **MEASURED LIVE 2026-07-31, at 384 cores and falling from 728:**
#:   * our pending jobs  ``prior`` 1.811–1.828, every one at ``-p -100``
#:   * other users       ``prior`` 2.000–2.082, every one at ``-p 0`` (four users sampled)
#:   * 2,144 of 2,646 pending jobs cluster-wide outranked us
#:   * 120 of our 124 queued jobs were the three CONTROL arms; every RUNNING treatment-arm job
#:     was at ``-p 0``
#:   * the gap IS the flag: predicted ``4.0 x [(0+1023)/2047 - (-100+1023)/2047] = 0.195``,
#:     observed 0.189
#:   * pool d had 2,480 free slots and 246 placeable 8-slot jobs — the capacity was there
#:
#: So the ladder was starving the three CONTROL arms (``scalar_cvar5``, ``placebo``,
#: ``placebo_shuffled``) while the two H2 treatment arms ran at full standing. A line finishes when
#: its SLOWEST arm finishes, so it gated the whole campaign — and it threatened IDENTIFICATION, by
#: searching treatment arms to greater depth than their own controls. At C4 it was worse still:
#: blocks 2+ descended to -300…-340, i.e. the decisive top rungs the power analysis rests on would
#: have been submitted at the LOWEST standing we can hold.
#:
#: **Block ORDER is preserved without the ladder.** Blocks are submitted in rung order and
#: ``weight_waiting_time = 1.0``, so an earlier-submitted block accrues priority over a later one;
#: the cumulative-tier BANKING rule is enforced in analysis, never by the scheduler.
#:
#: 0 = full fair-share standing. This is R101's requirement and the absolute standing rule:
#: *never lower the SGE priority of any of our jobs, ever.*
PRIORITY_CORE = 0
PRIORITY_STAGE1 = 0
PRIORITY_RUNG_BASE = 0


def _core_priority(arm: str, h2_arms: tuple[str, ...] | list[str]) -> int:
    """The C1-C3 per-arm ``-p`` value. **Now PRIORITY_CORE (0) for EVERY arm.**

    It used to return ``PRIORITY_CORE`` for the two H2 arms and ``bayes_opt``, and
    ``PRIORITY_STAGE1`` (-100) for everything else. That split is retired for the reasons recorded
    on the constants above, and one of them is scientific rather than operational:

    **the arms it deprioritised were the three CONTROLS** — ``scalar_cvar5``, ``placebo``,
    ``placebo_shuffled``. Those are precisely the arms that isolate the effect of the feedback
    CONTENT, and searching them to a shallower depth than the treatment arms they are compared
    against is an identification threat, not merely a slowdown. Measured on 2026-07-31 it had
    already opened a four-generation gap (treatments at g5, controls at g1–g3).

    The signature is kept (rather than deleting the call sites) so that the retirement is visible
    at the point of use and any future reintroduction has to argue with this docstring.
    """
    del h2_arms          # every arm rides at full fair-share standing under R101
    return PRIORITY_CORE


def run_campaign_tiered(
    arms: list[str],
    opts_for: Callable[[str], dict],
    seeds_cfg: Any,
    run: ClusterRun,
    *,
    test_leg_kwargs: dict[str, Any],
    frozen_root: str | Path,
    h2_arms: tuple[str, ...] = ("distributional", "scalar"),
    baseline_names: list[str] | None = None,
    canary_baselines: list[str] | None = None,
    review_gate: bool = True,
    hold_at_gate: bool = False,
    resume: bool = False,
    pipeline_rungs: bool = False,
) -> dict[str, Any]:
    """The Stage-1 **C-ladder** (PLAN §13.1) — the value-per-hour greedy checkpoint order, with the
    §14.3 priority ladder enforcing it IN the scheduler and the effect-blind review gate at the
    design-floor boundary.

    * **C0 canary** — ``canary_baselines`` (the first 3 H1 units) through the FULL production path
      at the core seeds, BEFORE any Opus spend. A canary failure raises loud (abort, fix, re-run).
    * **C1–C3** — all arms' search→select→freeze pipelines CONCURRENT (**every arm at ``-p 0``
      since 2026-07-31; the H2-at-0 / rest-at--100 split is retired — see PRIORITY_CORE**). Non-H2
      arms' core tests flood per-arm the instant their search ends (zero barrier); the H2 pair's
      core test is ONE **pair-adjacent interleaved** array (B-A3: the taskfile alternates
      dist-s_k, scalar-s_k, so any truncation leaves the pair CRN-balanced).
    * **REVIEW GATE** (after the C3 all-arms floor; ``review_gate=True``) — writes the EFFECT-BLIND
      integrity report (:mod:`src.cluster.integrity` — counts/health/homogeneity ONLY; the docs'
      "counts-only monitoring; single-look discipline") and STOPS. The operator reviews and creates the
      approval file (``<read_root>/TIER1_APPROVED``); re-running with ``resume=True`` then skips
      straight through C0–C3 (archive-complete) into C4. Looking at RESULTS here would be optional
      stopping — the gate is execution-health only, by construction.
    * **C4 sweep** — the uniform-n completion (n_core→n_max) for ALL units (the 9 arms of
      ``config/arms.yaml`` after R108's 7 -> 9, plus the 11-member H1 canon; both RESOLVED at
      runtime — ``sweep_units`` is built from the passed ``arms`` and ``baseline_names``, never from
      a hand-typed count), ROUND-ROBIN interleaved per seed, in the assurance-checkpoint blocks the seed
      schema declares (e.g. 30→340 @90% → 403 @95% → 568 @99%): every block boundary is a clean,
      complete design at that assurance level.

    H3 single-shot and the D1 curve levels run as separate entry invocations, **at ``-p 0`` since
    2026-07-31** (they were C5/C6 at -100/-200). Fully resume-safe at every step: rerun and every
    archived unit is skipped.
    """
    from src.utils.seeds import seed_tiers

    select_winner, _ = _resolve_select_freeze(run)
    tiers = seed_tiers(seeds_cfg)
    core = tiers[0]
    out: dict[str, Any] = {
        "n_tiers": len(tiers), "tier_sizes": [len(t) for t in tiers], "results": {},
    }

    # ---- C0 + C1–C3 (MODE-D 2026-07-21c): the canary runs CONCURRENTLY with the no-spend work - #
    # The canary's purpose is SPEND protection: prove the full path before any *Opus authoring*
    # is billed. The family arms (random_search/bayes_opt) author NOTHING — they are the same
    # fixed-machinery risk class as the canary's own baseline trainings — yet the old sequential
    # structure made the floor's LONGEST chain (the 30-step BO sequence) wait ~4-6h for a gate
    # that protects nothing it does. Now: the canary + family arms + H1 baselines all start at
    # L+0; ONLY the LLM arms' authoring waits on the canary verdict (``canary_ok``). A canary
    # failure still aborts loudly with ZERO Opus spent: LLM arms return un-authored, the loop
    # drains (in-flight family arrays are legitimate trainings; a truly broken path fails them
    # fast via the driver's error caps), and the canary error re-raises after the drain.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    winners: dict[str, dict[str, Any]] = {}
    core_results: dict[str, dict[str, Any]] = {}
    canary_ok = threading.Event()
    canary_abort = threading.Event()
    canary_failure: list[BaseException] = []
    if not canary_baselines:
        canary_ok.set()  # no canary configured -> nothing gates authoring

    def _run_canary() -> dict[str, Any]:
        _LOG.info("[C0] canary: %s at %d core seeds (concurrent with the no-spend arms)",
                  canary_baselines, len(core))
        try:
            # AUDIT 2026-07-22 (CRITICAL): this call sat OUTSIDE the try — a canary-batch exception
            # (driver outage, spec desync, qsub parse) set NEITHER event, so the LLM-arm waiters
            # spun forever and the line hung silently. Everything canary-fatal now aborts loudly.
            canary = run_baselines_on_cluster(
                canary_baselines, core, run, test_leg_kwargs=test_leg_kwargs, name="canary",
                priority=PRIORITY_CORE, resume=resume,
            )
            if not canary.get("ok", False):
                raise RuntimeError(
                    f"C0 CANARY FAILED ({canary}) — the production path does not work end-to-end; "
                    f"fix and re-run BEFORE any Opus authoring is spent (the point of the canary)"
                )
            # ANALYSIS-SMOKE (2026-07-12 upgrade): the canary's definition of success extends from
            # "trainings ran" to "the READER side parses them" — load_all validates every canary
            # record against the full schema (fail-loud) so an analysis-side break surfaces day 0.
            from src.io.results import load_all as _load_all

            for nm in canary_baselines:
                unit = f"baseline_{nm}"
                recs = _load_all(str(run.test_read() / unit))
                got = {int(r["seed"]) for r in recs}
                if not set(core) <= got:
                    raise RuntimeError(
                        f"C0 canary ANALYSIS-SMOKE failed: unit {unit} parsed {sorted(got)} but "
                        f"the core seeds are {core} — records exist per the census yet the reader "
                        "cannot recover them all; fix the archive/reader before any Opus spend"
                    )
            _LOG.info("[C0] analysis-smoke: all canary records parse + full seed coverage")
        except BaseException as exc:
            canary_failure.append(exc)
            canary_abort.set()
            raise
        canary_ok.set()
        return canary

    def _arm_core(arm: str) -> dict[str, Any]:
        prio = _core_priority(arm, h2_arms)
        opts = opts_for(arm)
        if arm in _FAMILY_ARMS:
            search = run_family_search_arm(arm, opts, run, resume=resume, priority=prio)
        else:
            # LLM arms WAIT for the canary verdict — authoring is the spend the canary protects.
            while not canary_ok.wait(timeout=5.0):
                if canary_abort.is_set():
                    return {"arm": arm, "ok": False, "reason": "canary_failed_no_authoring"}
            search = run_search_arm(arm, opts, run, resume=resume, priority=prio)
        _guard_k_seed_selector(opts, run)  # P15: max-single-seed select is WRONG under k>1
        winner, _r115 = _select_eligible_winner(select_winner, run.search_read() / arm, arm)
        if winner is None:
            return {"arm": arm, "ok": False, "reason": _r115 or "no_winner", "search": search}
        winner.setdefault("reward_source", "")
        _, freeze_winner = _resolve_select_freeze(run)
        env_fp = winner.get("env_fingerprint") or f"cluster:{arm}:frozen"
        # P5 (2026-07-13 audit): a resume after search-archive drift (different --candidates/
        # --generations, an F5 bypass) could re-derive a DIFFERENT winner and silently re-freeze
        # it — already-tested seeds keep the old reward, the rest train the new one (a mixed-
        # winner unit). Refuse to overwrite an existing frozen winner with a different hash.
        _refuse_winner_swap(arm, winner, frozen_root)  # P5 (2026-07-19: shared helper, all paths)
        freeze_winner(arm, winner, search_seed=int(opts.get("seed", 0)), frozen_root=frozen_root,
                      env_fingerprint=env_fp)
        winners[arm] = winner
        if arm in h2_arms:
            return {"arm": arm, "ok": True, "search": search, "test": "deferred_to_pair_array"}
        test = run_test_leg([(arm, winner)], core, run, name=f"{arm}_test",
                            priority=PRIORITY_STAGE1, resume=resume, **test_leg_kwargs)
        return {"arm": arm, "ok": True, "search": search, "test": test}

    with ThreadPoolExecutor(max_workers=len(arms) + 2, thread_name_prefix="unit") as ex:
        futs = {ex.submit(_arm_core, arm): arm for arm in arms}
        if canary_baselines:
            futs[ex.submit(_run_canary)] = "__canary__"
        if baseline_names:
            # The canary IS the first canary_baselines' core run — now that the two tasks are
            # CONCURRENT, the baselines batch must EXCLUDE them or the same run_ids double-submit
            # (two live jobs writing one record — the P4 write-race class; sequentially the old
            # code merely re-submitted redundantly). The exclusion is also strictly faster.
            _rest = [n for n in baseline_names if n not in set(canary_baselines or [])]
            if _rest:
                futs[ex.submit(
                    run_baselines_on_cluster, _rest, core, run,
                    test_leg_kwargs=test_leg_kwargs, name="baselines",
                    priority=PRIORITY_STAGE1, resume=resume,
                )] = "__baselines__"
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                res = fut.result()
                if key == "__canary__":
                    out["results"]["canary"] = res
                elif key == "__baselines__":
                    # FAIL CLOSED, same reasoning as the non-tiered path above (2026-07-28).
                    core_results[key] = {"ok": res.get("ok", False), "test": res}
                else:
                    core_results[key] = res
            except Exception as exc:  # noqa: BLE001 — one unit must not sink the ladder
                if key == "__canary__":
                    _LOG.exception("[C0] canary crashed (abort raised; re-raised after the drain)")
                    continue  # recorded in canary_failure; re-raised loudly after the drain
                _LOG.exception("[%s] core pipeline crashed", key)
                core_results[key] = {"arm": key, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if canary_failure:
        # The loud abort semantics are PRESERVED: zero Opus was authored (LLM arms returned
        # un-authored the moment the abort flag rose); the drain above only let already-submitted
        # no-spend arrays finish. Fix the path, then re-run — resume skips completed work.
        raise RuntimeError(str(canary_failure[0])) from canary_failure[0]

    # ---- D14: A PARTIAL ARM FAILURE MUST NOT BE SILENT, AND MUST NOT BE BUILT ON ------------- #
    # (Applied 2026-08-01, RUN 10.)
    #
    # `_arm_core`'s `except` above is deliberate — one unit must not sink the ladder — and it is
    # right for throughput. What was missing is any reconciliation AFTERWARDS. The asymmetry that
    # made this dangerous: when EVERY arm crashes the line exits and the watchdog revives it, but
    # when only SOME crash the survivors carry the process onward, so nothing exits, nothing
    # revives, and the dead arms are stranded for the life of the run. `nemotron-3-super` ran
    # 8 h 29 m with 3 of its 5 arms exactly this way.
    #
    # AND THE REAL DAMAGE IS SCIENTIFIC, NOT MERELY WASTED COMPUTE. The very next statement
    # builds the H2 pair test from `winners`, as ONE `interleave=True` CRN-paired array. A crashed
    # arm has no winner, so it is silently ABSENT from that array — the pair test would run with
    # four of five arms, and every seed in it would be paired against a comparator set that is not
    # the registered one. Stopping here is what prevents that.
    #
    # DELIBERATELY NARROW: only an arm that raised (an `error` key) stops the line. A designed
    # outcome — `no_winner`, an R115 ineligibility, a canary-gated arm — carries a `reason` and is
    # a legitimate result of the experiment, not a fault, and must NOT stop anything.
    _crashed = {k: v.get("error") for k, v in core_results.items() if v.get("error")}
    if _crashed:
        _LOG.critical(
            "[C1] %d core arm(s) CRASHED: %s — STOPPING this pass BEFORE the H2 pair test rather "
            "than building a CRN-paired array with a missing arm. The line exits non-zero, the "
            "supervisor relaunches it, and --resume re-runs only the crashed arm(s).",
            len(_crashed), ", ".join(f"{k} ({v})" for k, v in sorted(_crashed.items())))
        try:
            # A marker the monitoring cycle can see within one sweep. A CRITICAL log line in a
            # multi-hundred-megabyte driver log is not an alarm; a file is.
            # NOTE: `read_root` is SHARED BY ALL TWELVE SUPERVISED LINES (see the warning at the top of
            # this module and the `TIER1_APPROVED_{line_tag}` precedent below) — an unqualified
            # filename here would have twelve writers racing one path and would report the wrong
            # line. Qualify it exactly the way the approval file is qualified.
            _marker = Path(run.read_root) / f"ARM_CRASH_{run.line_tag()}.json"
            _marker.write_text(json.dumps(
                {"ts": time.time(), "line": run.line_tag(), "arms": _crashed,
                 "note": "core arm(s) raised; the pass stopped before C2 to protect the CRN "
                         "pairing of the H2 pair test. Cleared automatically on a clean pass."},
                indent=2), encoding="utf-8")
        except Exception:  # noqa: BLE001 — observability must never be the thing that fails
            _LOG.exception("[C1] could not write the ARM_CRASH marker")
        out["results"]["core"] = core_results
        out["arm_crash"] = _crashed
        out["ok"] = False
        return out
    try:
        _m = Path(run.read_root) / f"ARM_CRASH_{run.line_tag()}.json"
        if _m.exists():
            _m.unlink()          # a clean pass clears the marker — the alarm must not be sticky
    except Exception:  # noqa: BLE001
        pass

    # C2: the H2 pair-test at the core n — ONE pair-adjacent interleaved array at -p 0.
    h2_present = [a for a in h2_arms if a in winners]
    if h2_present:
        pair = run_test_leg(
            [(a, winners[a]) for a in h2_present], core, run, name="h2_pair_test",
            priority=PRIORITY_CORE, interleave=True, resume=resume, **test_leg_kwargs,
        )
        for a in h2_present:
            core_results[a]["test"] = pair
    out["results"]["core"] = core_results
    core_ok = all(r.get("ok", True) for r in core_results.values())

    # ---- THE REVIEW GATE (effect-blind; auto-proceeds on green health) ------------------------- #
    # The gate reads ONLY execution health (counts + homogeneity censuses — never a performance
    # statistic), so releasing C4 on green does NOT condition continuation on observed effects and
    # cannot become optional stopping. It therefore AUTO-PROCEEDS when execution is clean (no manual
    # latency on the happy path — the time-security requirement), and STOPS only when (a) a real
    # execution defect is found (a short/incomplete unit or device inhomogeneity), or (b) the operator set an
    # explicit ``hold_at_gate`` to eyeball the effect-blind report first. Either stop is released by
    # creating ``<read_root>/TIER1_APPROVED`` and re-running with ``resume=True``.
    # 2026-07-28 CROSS-LINE COLLISION FIX (same class as the reject-marker defect that invalidated
    # RUN 1). Both of these files live in `read_root`, which ALL TWELVE supervised lines share, and
    # both were unqualified. Two distinct failures followed:
    #   * `TIER1_APPROVED` was ONE file for twelve gates, and the passage CONSUMES it (unlink). An
    #     approval granted after reviewing line A's report could therefore be eaten by whichever
    #     line reached its gate first, letting THAT line proceed to the expensive C4 sweep on a
    #     review it never had, while line A still stopped.
    #   * `tier1_integrity.json` was likewise one file, rewritten by every line — so the report a
    #     reviewer read was whichever line wrote last, and the staleness comparison below
    #     (approval mtime vs report mtime) raced against OTHER lines' report writes, silently
    #     invalidating legitimate approvals.
    # Verified on the RUN 1 archive: exactly one tier1_integrity.json/.md existed for all twelve
    # lines. Both are now scoped by `run.line_tag()`. Note the rename fails CLOSED: a pre-existing
    # unqualified TIER1_APPROVED is now ignored, so the gate stops rather than passing unreviewed.
    approval = Path(run.read_root) / f"TIER1_APPROVED_{run.line_tag()}"
    if review_gate:
        from src.cluster.integrity import write_integrity_report

        # P6 (2026-07-13 audit): a STALE approval file (left over from a rehearsal in a reused
        # output dir, or created BEFORE any report existed to review) bypassed even a RED gate.
        # An approval counts only if it postdates the report THE REVIEWER SAW — i.e. the report
        # file as it existed BEFORE this run regenerates it (regeneration always postdates any
        # legitimate approval, so comparing against the fresh report would reject everything).
        _prior_report = Path(run.read_root) / f"tier1_integrity_{run.line_tag()}.json"
        approved = False
        if approval.exists():
            if _prior_report.exists() and approval.stat().st_mtime >= _prior_report.stat().st_mtime:
                approved = True
            else:
                _LOG.warning("[gate] STALE approval (predates the reviewed report, or no report "
                             "was ever written) — IGNORED; re-create %s AFTER reviewing the "
                             "integrity report", approval)

        report, _report_json, report_md = write_integrity_report(
            run, arms=arms, h2_arms=list(h2_present), baseline_names=list(baseline_names or []),
            core_seeds=core, opts_for=opts_for, winners=winners,
        )
        out["integrity_report"] = str(report_md)
        health_ok = bool(report.get("verdict", {}).get("health_ok"))
        out["gate_health_ok"] = health_ok
        if not approved and (not health_ok or hold_at_gate):
            reason = "RED-execution-health" if not health_ok else "manual-hold"
            out.update(ok=core_ok, gate=reason, awaiting_review=True,
                       health_ok=health_ok, approve_by_creating=str(approval))
            _LOG.warning("[gate] STOPPING before C4 (%s) — review %s then create %s and resume",
                         reason, report_md, approval)
            return out
        if approved:
            try:
                approval.unlink()  # consume: the NEXT gate passage needs its own explicit approval
            except OSError:
                _LOG.warning("[gate] could not consume %s (continuing)", approval)
        _LOG.info("[gate] %s — PROCEEDING to C4 sweep",
                  "explicit approval consumed" if approved else "green execution health (auto)")

    # ---- C4: the uniform-n round-robin sweep in assurance-checkpoint blocks ------------------- #
    sweep_units: list[tuple[str, dict[str, Any]]] = [(a, winners[a]) for a in arms if a in winners]
    if baseline_names:
        import sys as _sys

        _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from run_campaign import _baseline_winner_record  # type: ignore[import-not-found]

        sweep_units += [(f"baseline_{nm}", _baseline_winner_record(nm)) for nm in baseline_names]
    if pipeline_rungs:
        # MODE-D (2026-07-21): submit ALL assurance blocks AT ONCE under a descending priority
        # ladder instead of draining each block before submitting the next. The scheduler then
        # (a) keeps the eligible backlog deep at every instant (idle-window harvesting — the
        # sequential path forfeits capacity during every block's drain), and (b) still COMPLETES
        # blocks in rung order (block 1 = tier-100 at -100 above the legs; blocks 2+ descend from
        # PRIORITY_RUNG_BASE below the legs — the registered unified queue, enforced natively).
        # P17 semantics are PRESERVED where they bind: a rung only BANKS when it and every rung
        # below it are complete (the cumulative-tier definition), so a broken lower level can
        # never carry results; the exposure is bounded GPU-hours of above-a-broken-level work,
        # traded deliberately for zero drain bubbles (documented in the runbook §10).
        from concurrent.futures import ThreadPoolExecutor as _TPE

        def _block(i: int, tier: list[int]) -> tuple[int, dict[str, Any]]:
            # 2026-07-31: the descending ladder (-100, then -300, -310, -320 ...) is RETIRED. It put
            # the HIGHEST rungs — the ones the power analysis is built on and the ones the Aug-27
            # calendar threatens — at the LOWEST standing we can hold. At -340 the penalty is
            # 4.0 x 0.166 = 0.66 of normalised priority, against competing users sitting at 2.00+.
            # Ordering survives without it: blocks are submitted in rung order and
            # weight_waiting_time = 1.0, so the earlier block outranks the later one on age alone.
            prio = PRIORITY_RUNG_BASE
            _LOG.info("[C4|pipelined] block %d: %d units x %d seeds at -p %d",
                      i, len(sweep_units), len(tier), prio)
            return i, run_test_leg(
                sweep_units, tier, run, name=f"sweep_t{i}", priority=prio, interleave=True,
                resume=resume, **test_leg_kwargs,
            )

        with _TPE(max_workers=max(1, len(tiers) - 1), thread_name_prefix="rung") as ex:
            for i, r in ex.map(lambda it: _block(*it), list(enumerate(tiers[1:], start=1))):
                out["results"][f"sweep_t{i}"] = r
                if not r.get("ok", True):
                    _LOG.error("[C4|pipelined] block %d INCOMPLETE — the ladder BANKS only up to "
                               "the last clean level (fix + resume completes it); higher-block "
                               "work already run above this level is reported, never banked", i)
    else:
        for i, tier in enumerate(tiers[1:], start=1):
            # row 30n/C6 (audit 2026-07-22) had made the sequential path MIRROR the pipelined
            # ladder. Both are retired together (2026-07-31) — mirroring is still the right
            # property, it is now mirroring 0.
            prio = PRIORITY_RUNG_BASE
            _LOG.info("[C4] sweep block %d: %d units x %d seeds (round-robin, -p %d)", i,
                      len(sweep_units), len(tier), prio)
            r = run_test_leg(
                sweep_units, tier, run, name=f"sweep_t{i}", priority=prio,
                interleave=True, resume=resume, **test_leg_kwargs,
            )
            out["results"][f"sweep_t{i}"] = r
            if not r.get("ok", True):
                # P17/A3-F10 (2026-07-13 audit): a failed assurance block STOPS the ladder — the
                # old loop marched on to the next block, spending GPU-days ON TOP of a broken
                # level (every block boundary must be a clean, complete design). Resume continues.
                _LOG.error("[C4] sweep block %d INCOMPLETE — stopping the ladder (fix + resume; "
                           "later blocks are not run on top of a broken assurance level)", i)
                break
    out["ok"] = core_ok and all(
        out["results"].get(f"sweep_t{i}", {}).get("ok", True) for i in range(1, len(tiers))
    )
    return out


# --------------------------------------------------------------------------- #
# The hard authoring spend cap (defense-in-depth for unattended multi-day runs) #
# --------------------------------------------------------------------------- #
def spend_guard(max_calls: int) -> Callable[[], None]:
    """A thread-safe author guard that raises once ``max_calls`` LLM authorings are reached.

    The generation loop is already structurally bounded (Σ arms × gens × cpg), so this is
    defense-in-depth: a bug that loops authoring (or a mis-sized config) can never silently burn
    the API budget on an unattended run — it fails LOUD at the ceiling. Wire onto
    ``ClusterRun.author_guard``; it is invoked under the author lock, so the counter is safe across
    the concurrent arm threads."""
    lock = threading.Lock()
    state = {"n": 0}

    def _guard() -> None:
        with lock:
            state["n"] += 1
            if state["n"] > max_calls:
                raise RuntimeError(
                    f"authoring spend cap reached ({max_calls} LLM calls) — refusing to author more; "
                    f"raise max_author_calls if this is expected, else a loop/config bug is burning budget"
                )

    return _guard
