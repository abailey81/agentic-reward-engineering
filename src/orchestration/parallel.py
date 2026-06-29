"""Heterogeneous GPU+CPU candidate-training scheduler — MAX laptop throughput (P5+).

Decouples the (fast, sequential) arm logic — LLM reflection / search — from the (heavy, parallel)
per-candidate SAC training. Arm DRIVERS run as threads in the main process and submit candidate
TASKS to a shared process pool of device-tagged workers; a device-token queue load-balances
``n_gpu`` 'cuda' + ``n_cpu`` 'cpu' concurrent trainings so the (faster) GPU AND all CPU threads stay
saturated, and the slow per-arm-serial bottleneck of arm-level parallelism is removed.

Per the compute deep-research (sources recorded in CHANGELOG): each worker pins threads=1 (OMP/MKL
+ torch) BEFORE imports; per job ``batch_size=512`` + TF32 on (AMP off); ``DummyVecEnv``;
``ProcessPoolExecutor`` with NON-daemon workers (so the sandbox validate-once child can spawn);
``empty_cache()`` between GPU tasks; the panel is loaded ONCE per worker and cached.

This is the FAST path for the real run; ``scripts/run_prototype.py`` keeps the simple arm-level path
(verified by the dry-run + tests). DIRECTIONAL / plumbing only — no number enters the dissertation.
"""
from __future__ import annotations

import hashlib
import multiprocessing as mp
import queue
import time
from collections.abc import Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np

__all__ = [
    "DevicePool",
    "train_candidate",
    "run_parallel",
    "run_recycling",
    "auto_n_gpu",
    "max_power_config",
]


def auto_n_gpu(train_steps: int = 25000, *, obs_dim: int = 1900, reserve_gb: float = 4.0,
               vram_ctx_mb: int = 1400, cap: int = 12) -> int:
    """Max sustainable CONCURRENT training workers on THIS machine — bounded by RAM, VRAM, and cores.

    Each worker holds a replay buffer of ``train_steps x obs_dim x 4 B x 2`` (obs+next_obs, float32) plus
    a torch/SB3/panel base (~0.8 GB) in RAM, and a CUDA context + model/batch (~``vram_ctx_mb``) in VRAM.
    We pick the largest worker count that fits available RAM (minus ``reserve_gb`` for the OS/main process)
    AND free VRAM AND the physical-core count — pushing the laptop to its real ceiling WITHOUT swapping
    (which would be slower). Measured + tuned by the Phase-0 scaling probe; override with ``--gpu N``.
    """
    import psutil

    phys = psutil.cpu_count(logical=False) or 4
    buf_gb = train_steps * obs_dim * 4 * 2 / 2**30
    # Per-worker RAM = replay buffer + torch/SB3/panel/CUDA-context overhead. The ~1.4 GB base is calibrated
    # to the MEASURED ceiling (2026-06-20): n_gpu=5 hit CPU MemoryError + CUDA-OOM, n_gpu=4 was the max.
    per_worker_gb = buf_gb + 1.4
    vm = psutil.virtual_memory()
    # Budget off TOTAL RAM minus a fixed OS/main reserve (steady state) — NOT the momentary ``available``,
    # which is depressed by transient caches / other processes; the run's lasting working set is the worker
    # replay buffers. Take the larger of the two so a transient dip doesn't under-provision the max run.
    budget_gb = max(vm.available / 2**30, vm.total / 2**30 - reserve_gb)
    ram_bound = max(1, int(budget_gb / max(per_worker_gb, 0.1)))
    vram_bound = cap
    try:
        import pynvml

        pynvml.nvmlInit()
        free_mb = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).free // 2**20
        vram_bound = max(1, int(free_mb / vram_ctx_mb))
    except Exception:  # pragma: no cover - no NVML / no GPU
        pass
    return max(1, min(phys, ram_bound, vram_bound, cap))


def max_power_config(
    train_steps: int = 50000,
    *,
    gpu_cap: int = 2,
    obs_dim: int = 1900,
    reserve_gb: float = 4.0,
    cpu_only: bool = False,
) -> tuple[int, int]:
    """Recommended ``(n_gpu, n_cpu)`` to drive THIS machine at its full *heterogeneous* throughput.

    Rationale (compute research + the RAM/thermal spec): SAC on the small portfolio
    nets is OVERHEAD-bound, so a CPU worker trains within ~1.5x of a GPU worker AND bypasses the 6 GiB VRAM
    ceiling entirely. GPU workers are therefore capped at ``gpu_cap`` (the transition-wave-safe ceiling at
    50k — ``n_gpu=4`` OOMs the 6 GiB RTX-4050), and CPU workers fill the remaining PHYSICAL cores within a
    RAM budget (each worker holds a ~``train_steps`` replay buffer + ~1.4 GB torch/SB3/panel/CUDA base; one
    thread per worker). ``cpu_only=True`` returns ``(0, all-safe-cores)`` for a GPU-less box.

    .. warning::
       This is a *hardware* recommendation; it is NOT automatically safe for the frozen confirmatory run.
       Two pre-registration constraints cap the usable parallelism BELOW this hardware ceiling:

       * **TEST leg (the winner seeds = inference data):** the parallel path is byte-identical to serial
         only on a SINGLE device. Mixing CPU + GPU workers makes results device-dependent (CPU != CUDA
         bit-for-bit), so for the frozen run use **GPU-only** here (``n_cpu=0``), not this heterogeneous mix.
       * **SEARCH leg:** ``run_parallel`` reflects on the generation's BEST candidate, whereas the FROZEN
         protocol is ``serial_reflect_on_best`` (§6 CORRECTION 2026-07-02; the serial loop reflects on the generation BEST, M5) — so enabling search parallelism is an *amendment-gated
         frozen-decision change*, not a free speed-up.

       The heterogeneous ``(n_gpu, n_cpu)`` returned here is therefore appropriate for the ``--synthetic`` /
       dev / ``bench_compute`` paths (no pre-registration in force), or for a leg the user has explicitly
       amended. Pure recommendation — it trains nothing.
    """
    import psutil

    phys = int(psutil.cpu_count(logical=False) or 4)
    buf_gb = train_steps * obs_dim * 4 * 2 / 2**30
    per_worker_gb = buf_gb + 1.4
    vm = psutil.virtual_memory()
    budget_gb = max(vm.available / 2**30, vm.total / 2**30 - reserve_gb)
    ram_workers = max(1, int(budget_gb / max(per_worker_gb, 0.1)))
    n_gpu = 0 if cpu_only else max(1, min(int(gpu_cap), ram_workers))
    n_cpu = max(0, min(phys - n_gpu, ram_workers - n_gpu))
    return int(n_gpu), int(n_cpu)


#: Per-worker caches (loaded once; workers are reused, max_tasks_per_child=None).
_PANEL_CACHE: dict[str, Any] = {}

#: Anonymised validation fixture for the sandbox (no tickers/dates). Real-ish length (~30 risky +
#: cash) so an allocation that scales with the input surfaces at validate_once (final-audit #12).
#: SHAPE PARITY (review batch 3 #1, 2026-07-03): production calls reward(weights(N+1),
#: returns(N), prev(N+1), ...) — portfolio_env.py:347-348 — and the FROZEN prompt promises exactly
#: that (weights (31,), returns (30,)). A 31/31/31 fixture INVERTED the gate for shape-aware rewards:
#: the spec-faithful `weights[:-1] @ returns` was falsely REJECTED (fixture shape mismatch) while the
#: sloppy `weights @ returns` was falsely ACCEPTED and then zero-trained via SAFE_DEFAULT on every
#: real step. The fixture must mirror the production shape contract: returns has ONE FEWER element.
#: DEGENERACY PARITY (audit 2026-07-19): a degenerate all-positive, zero-variance returns vector with an
#: EMPTY info dict falsely REJECTED valid rewards at validate_once. A left-tail/downside reward hits an
#: empty ``returns[returns < 0]`` subset (nan / ValueError on min) or a zero-variance ``std()`` (÷0), and a
#: stateful reward reading the prompt-documented ``info["reward_state"]`` raised KeyError — none of which
#: occur in production, where returns are mixed-sign with variance and info always carries the three keys
#: portfolio_env.py:323-327 supplies (weights, prev_weights, reward_state=None at reset). The fixture now
#: mirrors that so the treatment arm's characteristic tail-aware rewards are gated faithfully, not starved.
_FIX_WEIGHTS = np.full(31, 1.0 / 31)
_FIX_PREV = np.full(31, 1.0 / 31)
_FIXTURE: tuple[Any, ...] = (
    _FIX_WEIGHTS,
    ((np.arange(30) % 5) - 2) * 0.01,     # returns (N=30): mixed-sign, non-zero variance
    _FIX_PREV,
    0.002,                                # port_ret (non-zero)
    {"weights": _FIX_WEIGHTS, "prev_weights": _FIX_PREV, "reward_state": None},
)
_LLM_ARMS = ("distributional", "scalar", "placebo", "scalar_cvar5", "placebo_shuffled")


def _parse_affinity(spec: str) -> list[int]:
    """``"0-11"`` / ``"0,1,2"`` / mixed ``"0-5,8"`` -> a sorted logical-CPU id list (total on junk)."""
    out: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, _, hi = part.partition("-")
            try:
                out.update(range(int(lo), int(hi) + 1))
            except ValueError:
                continue
        else:
            try:
                out.add(int(part))
            except ValueError:
                continue
    return sorted(out)


def _apply_cpu_placement() -> None:
    """P-core affinity + anti-throttling for THIS process (2026-07-06 throughput; result-neutral).

    The campaign host's i7-13620H is HYBRID (6 P-cores = logical 0-11 with HT, 4 E-cores = 12-15 at
    ~3.6 vs ~4.9 GHz and weaker IPC). Windows freely migrates the per-step SB3 hot loop onto
    E-cores — and applies EcoQoS power-throttling to background-ish processes — silently slowing
    the CPU-bound fraction of every training step. When ``LLMRP_WORKER_CPU_AFFINITY`` is set (the
    driver exports it from ``config/campaign.yaml compute.worker_cpu_affinity``), pin this process
    to those logical CPUs, raise it to ABOVE_NORMAL, and disable Windows power throttling.
    PLACEMENT ONLY — identical instructions on identical data, byte-identical results. Best-effort:
    a failure never blocks training."""
    import os

    spec = os.environ.get("LLMRP_WORKER_CPU_AFFINITY", "")
    if not spec:
        return
    cpus = _parse_affinity(spec)
    if not cpus:
        return
    try:
        import psutil

        p = psutil.Process()
        p.cpu_affinity(cpus)
        try:
            p.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 — POSIX/nice differences
            pass
    except Exception:  # noqa: BLE001 — affinity is an optimization, never a blocker
        return
    try:  # EcoQoS OFF (Windows 11 power throttling — the hybrid-CPU background-process trap)
        import ctypes

        class _PT(ctypes.Structure):
            _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong),
                        ("StateMask", ctypes.c_ulong)]

        # PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1; ControlMask set + StateMask 0 = NEVER throttle.
        st = _PT(1, 0x1, 0x0)
        ctypes.windll.kernel32.SetProcessInformation(
            ctypes.windll.kernel32.GetCurrentProcess(), 4,  # ProcessPowerThrottling
            ctypes.byref(st), ctypes.sizeof(st),
        )
    except Exception:  # noqa: BLE001
        pass


def _worker_init(threads: int = 1) -> None:
    """ProcessPoolExecutor initializer: pin threads BEFORE any heavy import (research §2/§3).

    ``threads`` is the INTRA-OP BLAS/torch thread count for this worker, and it is the execution
    half of amendment **R107** (2026-07-27 wiring). R107 ratified 1 -> 8 threads on the SEARCH/chain
    leg on a measured 2.72x single-training speed-up (jobs 17784/17836), and `lanes.CPU_CHAIN_THREADS
    = 8` has been feeding the makespan planner ever since — but this function pinned "1"
    unconditionally, so the ratified value was MODELLED and never EXECUTED. The planner believed the
    `bayes_opt` GP-EI chain took 3.3 d; the code made it 8.9 d, and the crossover where extra cores
    stop helping sat at 1,685 rather than 4,584. That gap is invisible to a registry-vs-constant
    test, which is why one existed and still missed it; see the constant-vs-EXECUTION test.

    NOTE: THREADS ARE PART OF THE DETERMINISM ENVELOPE. Multi-threaded BLAS changes the ORDER of float
    reductions, so an 8-thread training is NOT bit-identical to a 1-thread one. R107 confines the
    change to the SEARCH/chain leg: every SCORED comparison (H1/H2/H3/H4, every paired contrast,
    all CRN pairing) lives on the uniformly 1-thread TEST leg, so threads change only WHICH
    candidate is selected, never a measured quantity. **The default is therefore 1** — every
    existing caller keeps today's exact behaviour, and 8 is opt-in per batch.

    NOTE: THREADS MUST BE MATCHED BY CORES. 8 threads on a 1-core SGE allocation is oversubscription
    and is SLOWER than 1 thread, so any caller raising this MUST raise the job's core request in
    the same change (``cores = threads x pack``). The two are one decision, never two.

    Interop threads stay at 1: R107 benched the intra-op axis only, and an unbenched second axis
    is exactly the kind of unregistered envelope change this docstring exists to prevent.
    """
    import os

    n = max(1, int(threads))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)
    _apply_cpu_placement()  # 2026-07-06: P-core pinning + EcoQoS-off (config-gated; result-neutral)
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.8"
    )
    # Preload pyarrow BEFORE torch: the worker loads the REAL gold parquet (pyarrow) + trains (torch), and
    # importing pyarrow AFTER torch SIGSEGVs (ABI conflict, verified 2026-06-20). pyarrow must win the native
    # load. See src.utils.preload. Inline (not the helper) so it runs before ANY src import in the spawn child.
    import pyarrow.parquet  # noqa: F401

    import torch

    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(1)  # deliberately NOT scaled — see the docstring
    except Exception:  # noqa: BLE001 - only settable once
        pass


def _panel_and_windows(synthetic: bool, data: dict, lookback: int):
    # Cache key = EVERY input that shapes the result (fix 2026-07-03; mirrors test_leg._load_test_panel's
    # keying): the old key omitted on_missing (changes the PANEL content — the delisting fill) and
    # embargo_days (changes the VAL window via embargoed_val_start), so two specs differing only there
    # silently shared one cached (panel, windows). lookback (shapes both windows) is keyed too; it is
    # process-constant today (read from config/environment.yaml) so that part is purely defensive.
    key = (
        f"syn:{lookback}"
        if synthetic
        else "gold:{}:{}:{}:{}:{}:{}".format(
            data.get("phase"),
            data.get("train_end"),
            data.get("val_end"),
            data.get("on_missing"),
            data.get("embargo_days"),
            lookback,
        )
    )
    if key in _PANEL_CACHE:
        return _PANEL_CACHE[key]
    if synthetic:
        from src.data.synthetic import make_synthetic_panel

        panel = make_synthetic_panel(n_assets=30, n_days=600, seed=0)
        res = (panel, (lookback, 400), (400, panel.T))
    else:
        from src.data.loaders import OnMissing, embargoed_val_start, load_gold_panel

        phase = str(data.get("phase", "development"))
        train_end = str(data.get("train_end", "2016-12-31"))
        from src.utils.config import load_config

        # Embargo fallback = the canonical config/data.yaml floor, NOT a bare literal 21 (no-hardcoding audit).
        _emb_floor = int(load_config("data").get("embargo_days", 21))
        embargo_days = int(data.get("embargo_days", _emb_floor))
        r = load_gold_panel(
            phase=phase,
            end=str(data.get("val_end", "2019-12-31")),
            on_missing=cast(OnMissing, data.get("on_missing", "liquidate_to_cash")),
            # 2026-07-13 audit: on-node reads were NOT checksum-verified (the jobscript comment
            # claimed they were) — a stale/diverged staged panel would silently mis-slice every
            # integer window resolved on the laptop panel. Fail loud against the frozen manifest.
            verify_checksum=True,
        )
        panel = r.panel
        dates = np.asarray(panel.dates)
        # Train ends at ``train_end``; validation begins at the PURGED boundary (PREREGISTRATION §7, R18):
        # embargoed_val_start(lookback=lookback) = max(materialized boundary — stale pre-Split-C
        # 2015-02-03, inert under the Split-C train_end — first_post_train + max(embargo, lookback)).
        # With lookback=60 the lookback purge dominates (val 2017-03-30 = train_end + 60 sessions).
        # Matches run_prototype._load_panel_and_windows so BOTH executed paths drop the same gap (R18).
        # side='right': half-open train end — identical to the old searchsorted+1 when train_end IS a
        # session; correct when it is not (SPLIT C's 2016-12-31 is a Saturday — +1 leaked 2017-01-03).
        train_split = int(np.searchsorted(dates, np.datetime64(train_end), side="right"))
        train_split = max(lookback + 1, min(train_split, panel.T - 1))
        val_split = embargoed_val_start(
            dates, train_end, phase=phase, embargo_days=embargo_days, lookback=lookback
        )
        val_split = max(train_split, min(val_split, panel.T - 1))
        res = (panel, (lookback, train_split), (val_split, panel.T))
    _PANEL_CACHE[key] = res
    return res


def train_candidate(spec: dict) -> dict:
    """Worker: build the reward, train the fixed SAC on ``spec['device']``, return fitness + tails.

    ``spec['reward_kind']`` is ``"source"`` (untrusted LLM/search code → sandbox-validated),
    ``"coeffs"`` (H4 reward family), or ``"baseline"`` (a hand-designed reward by name). Returns a
    picklable dict ``{ok, candidate_id, arm, fitness, val_returns, tail_stats, reward_source, ...}``.
    """
    from src.utils.env import load_env

    load_env()  # ensure the LLM key is in this worker's env (ADR-038) in case spawn didn't inherit it
    import torch  # kept for cuda.empty_cache below; TF32 is now set by train_agent
    # (config-driven, ``tf32`` default on) so the SEARCH worker, the serial trainer, and the parallel TEST
    # worker share IDENTICAL float32 numerics (no scheduler-dependent precision drift — manager review P2).

    out: dict[str, Any] = {"ok": False, "candidate_id": spec.get("candidate_id"), "arm": spec.get("arm")}
    # Candidate-parallel monitoring: stream per-step training metrics + lifecycle to the main
    # ParallelMonitor over ``spec['monitor_queue']`` (a Manager queue; absent in tests/unmonitored runs).
    sink = None
    _mq = spec.get("monitor_queue")
    if _mq is not None:
        from src.utils.monitoring import QueueSink

        sink = QueueSink(_mq, str(spec.get("candidate_id")), str(spec.get("arm")))
    _cand_t0 = time.perf_counter()
    try:
        from src.utils.seeding import set_global_seed

        # P0-3 (audit 2026-06-19): seed EVERY RNG stack in the worker process -- not just the
        # SAC kwarg. Python random, the legacy np.random global (read by VecNormalize/SB3),
        # PYTHONHASHSEED, torch + cuDNN, and the deterministic-algorithm flags.
        set_global_seed(int(spec["seed"]), deterministic_torch=True)

        from src.agents.trainer import make_agent_trainer
        from src.env.runner import make_env_builder
        from src.feedback.measurement import ReturnDistribution
        from src.selection.fitness import held_out_fitness
        from src.utils.config import load_config

        env_cfg = load_config("environment")
        lookback = int(env_cfg["state"]["lookback_days"])
        panel, tw, vw = _panel_and_windows(bool(spec["synthetic"]), spec.get("data", {}), lookback)
        device = spec.get("device", "cpu")

        kind = spec["reward_kind"]
        if kind == "source":
            from src.sandbox.executor import SandboxEnvironmentError, SandboxError, validate_once

            src = spec["reward"]
            out["reward_source"] = src
            out["reward_hash"] = hashlib.sha256(src.encode("utf-8")).hexdigest()
            try:
                reward_fn = validate_once(src, _FIXTURE)
            except SandboxEnvironmentError:
                # CONTRACT (src/sandbox/executor.py::SandboxEnvironmentError): a STARVED spawn
                # environment is NOT a candidate defect, and any site that permanently ledgers a
                # rejection must catch it FIRST. Marking `failed_validation` here would poison a
                # good, PAID candidate into the frozen reject set that ``--resume`` replays — the
                # exact failure src/llm/loop.py:531-539 was written to prevent. Fail LOUD instead so
                # the supervisor relaunches and ``--resume`` re-attempts this one on a freed box.
                # (deep-review 2026-07-26, loop 2 — the contract was documented but unhonoured here.)
                raise
            except SandboxError as exc:
                out["error"] = f"sandbox: {exc}"
                out["failed_validation"] = True
                return out
        elif kind == "coeffs":
            from src.baselines.reward_family import params_to_reward, params_to_source

            alpha, window = spec.get("cvar_alpha", 0.05), spec.get("window", 20)
            reward_fn = params_to_reward(spec["reward"], alpha, window)
            # Rank 2c: archive the MATERIALIZED EXECUTABLE source so the frozen BO winner rehydrates
            # through validate_once for the sealed TEST leg (H4) — not a non-executable comment stub.
            out["reward_source"] = params_to_source(spec["reward"], cvar_alpha=alpha, window=window)
            out["reward_hash"] = hashlib.sha256(out["reward_source"].encode("utf-8")).hexdigest()
        else:  # baseline
            from src.baselines import rewards as R

            reward_fn = getattr(R, spec["reward"])
            out["reward_source"] = f"# baseline:{spec['reward']}\n"
            out["reward_hash"] = hashlib.sha256(out["reward_source"].encode("utf-8")).hexdigest()

        # R18 purge-guard args (fix 2026-07-03, mirroring run_campaign's builder call): pass the REAL
        # embargo + lookback so make_env_builder's max(embargo, lookback) leakage guard is ARMED on the
        # gold SEARCH path — the bare 4-arg call left purge=0, i.e. the guard could never catch a future
        # windows regression. Gold windows already satisfy the purge (embargoed_val_start builds them),
        # so this only fails loud on a violation. The SYNTHETIC dev/dry-run windows deliberately ABUT
        # ((lookback, 400)/(400, T), no purge gap), so the guard stays legacy-inert (0/0) there — arming
        # it would reject every --synthetic run, which run_campaign avoids via purged resolve_windows.
        if bool(spec["synthetic"]):
            _embargo = _lookback_guard = 0
        else:
            # Same resolution as _panel_and_windows above: spec data block, else the config/data.yaml floor.
            _emb_floor = int(load_config("data").get("embargo_days", 21))
            _embargo = int(spec.get("data", {}).get("embargo_days", _emb_floor))
            _lookback_guard = lookback
        env_builder = make_env_builder(panel, env_cfg, tw, vw, embargo=_embargo, lookback=_lookback_guard)
        if sink is not None:
            sink.candidate_start()
        trainer = make_agent_trainer(
            {
                "train_steps_per_candidate": int(spec["train_steps"]),
                # Buffer-cap (ADR-025): honor a spec-supplied cap, else fall back to train_steps, never
                # exceeding it. When B* rises (250-350k) the campaign pins buffer_size=50000 so the replay
                # buffer does NOT scale with the step budget (the silent OOM trap on 6 GB / 16 GB). Result-
                # neutral when the spec omits buffer_size (== prior behavior of buffer_size == train_steps).
                "buffer_size": min(int(spec.get("buffer_size", spec["train_steps"])), int(spec["train_steps"])),
                "batch_size": int(spec.get("batch_size", 256)),  # ONE canonical default (SB3); was 512 -> 256/512 drift
                "normalize_obs": bool(spec.get("normalize_obs", True)),
                "learning_rate": float(spec.get("learning_rate", 3e-4)),  # honor the full agent block (parity)
                "gamma": float(spec.get("gamma", 0.99)),
                "ent_coef": spec.get("ent_coef", "auto"),
                # learning_starts (gated 1000, not SB3's silent 100) + PopArt scale-normalization (default on):
                # thread them through the parallel worker so the SEARCH/TEST path trains the SAME agent as the
                # serial path (the trainer floors learning_starts at 1000 and gates popart on `popart`).
                "learning_starts": int(spec.get("learning_starts", 1000)),
                "popart": bool(spec.get("popart", True)),
                "popart_beta": float(spec.get("popart_beta", 1e-3)),
                "popart_min_scale": float(spec.get("popart_min_scale", 1.0)),
                "popart_warmup": int(spec.get("popart_warmup", 0)),
                "tf32": bool(spec.get("tf32", True)),
                # M6 (ops audit 2026-07-02): thermal governor for the SEARCH worker too. The trainer's
                # _make_governor reads cfg['thermal_guardian'] ({hi, lo, poll_secs}); absent/None -> off.
                # Result-neutral (pause-and-cool spends wall-clock only), so a 24/7 laptop SEARCH leg is
                # protected from thermal shutdown exactly like the serial/TEST paths.
                "thermal_guardian": spec.get("thermal_guardian"),
                "device": device,
            },
            int(spec["seed"]),
            monitor=sink,  # streams per-step SAC metrics to the main ParallelMonitor (anomaly detection)
        )
        bundle = env_builder(reward_fn)
        policy = trainer(bundle.train_env())
        popart_scale = getattr(policy, "popart_scale", None)  # T2.4 realised scale (read before the del below)
        # R66 (2026-07-03): training-window SAFE_DEFAULT substitution counts attached by train_agent —
        # read from the policy ATTRS (frozen at train end), not the live executor counters, which the
        # val/train rollouts below re-zero. None when a trainer doesn't surface them (back-compat).
        train_sd_count = getattr(policy, "train_safe_default_count", None)
        train_call_count = getattr(policy, "train_safe_call_count", None)
        val = np.asarray(bundle.val_returns(policy), dtype=float)
        train = np.asarray(bundle.train_returns(policy), dtype=float)
        # n_trials is MANDATORY in the spec (built as opts["n_trials"], ~L494) — fail loud rather than fall
        # back: the old `.get("n_trials", 40)` default was the PROTOTYPE candidate count and would silently
        # over-deflate a 30-candidate campaign arm's DSR if a hand-built spec ever omitted the key.
        fitness = float(held_out_fitness(val, int(spec["n_trials"])))
        tail = ReturnDistribution().fit(train).tail_stats()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        out.update(ok=True, fitness=fitness, val_returns=[float(x) for x in val], tail_stats=tail,
                   popart_scale=popart_scale,
                   # 2026-07-13 pre-spend audit: real elapsed seconds -> the record's wall_clock.
                   # _archive had HARDCODED 0.0 since inception, so NO search record (laptop or
                   # cluster) ever carried wall-clock — the compute-reporting hole Okhrati docks.
                   elapsed_secs=float(time.perf_counter() - _cand_t0),
                   # R66: training-window SAFE_DEFAULT counts -> archived by _archive (additive/optional).
                   train_safe_default_count=train_sd_count, train_safe_call_count=train_call_count)
        if spec.get("emit_train_returns"):
            # k-seed search (B-A2): the aggregator re-fits the FED tail on the CONCATENATION of the
            # k seeds' TRAIN-window returns — the same in-sample window the single-seed fed tail uses
            # (fed in-sample / scored out-of-sample, the ratified selection-overfitting defense) with
            # 3x the tail data for the EVT fit. Flag-gated: absent (every k=1 path) -> byte-identical.
            out["train_returns"] = [float(x) for x in train]
        if sink is not None:
            sink.candidate_done(fitness=fitness, status="ok", secs=time.perf_counter() - _cand_t0)
        # Reclaim this candidate's heavy objects in the PERSISTENT pool worker BEFORE the next candidate.
        # SB3's SAC holds cyclic refs (policy <-> optimizer <-> replay buffer, ~0.36 GiB obs arrays), so
        # without an explicit del + gc.collect() the freed memory is NOT reclaimed and the worker RSS CREEPS
        # across candidates -> OOM over a long run (measured without this: n_gpu=2 71 -> 85% in 40 min; n_gpu=3
        # 89.6 -> 96.3%). The clean fix (max_tasks_per_child worker recycling) DEADLOCKS on Python 3.11.9 +
        # Windows spawn, so reclaim in-process instead. Results are already captured in `out` (no science
        # change); numpy replay buffers are mmap'd -> returned to the OS on collection.
        del trainer, bundle, policy, val, train
        import gc

        gc.collect()
    except Exception as exc:  # noqa: BLE001 - a failed candidate must not crash the pool
        import traceback

        out["error"] = f"{type(exc).__name__}: {exc}"
        out["trace"] = traceback.format_exc().splitlines()[-3:]
        if sink is not None:
            # Surface the ERROR REASON to the live monitor (not just status) so a failure wave -- e.g. the
            # CUDA-OOM cascade at too-high n_gpu -- is caught immediately, not buried in failures.jsonl.
            sink.candidate_done(fitness=None, status="error", secs=time.perf_counter() - _cand_t0,
                                error=str(out["error"])[:200])
        if str(spec.get("device", "")).startswith("cuda"):
            try:
                import torch

                torch.cuda.empty_cache()  # release this candidate's VRAM after an OOM/error before the next
            except Exception:  # noqa: BLE001
                pass
    return out


#: Sentinel so ``DevicePool(initializer=None)`` means "no worker initializer" (used by fast unit
#: tests that must not import torch/pyarrow), while the default stays the production ``_worker_init``.
_DEFAULT_INIT = object()


class DevicePool:
    """A device-load-balanced process pool: ``n_gpu`` cuda + ``n_cpu`` cpu concurrent training slots.

    ``submit(spec)`` blocks until a device token frees, tags the spec with that device, submits
    ``train_candidate`` to the pool, and returns the token when the future completes — so at most
    ``n_gpu`` trainings run on the GPU and ``n_cpu`` on the CPU at any instant (no VRAM/RAM blow-up,
    GPU time-slicing bounded).
    """

    def __init__(
        self,
        n_gpu: int,
        n_cpu: int,
        max_tasks_per_child: int | None = None,
        initializer: Any = _DEFAULT_INIT,
        init_threads: int = 1,
    ) -> None:
        self.n_gpu = max(0, int(n_gpu))
        self.n_cpu = max(0, int(n_cpu))
        if self.n_gpu + self.n_cpu == 0:
            self.n_cpu = 1  # CPU-only fallback when no GPU is present
        self._tokens: queue.Queue[str] = queue.Queue()
        for _ in range(self.n_gpu):
            self._tokens.put("cuda")
        for _ in range(self.n_cpu):
            self._tokens.put("cpu")
        ctx = mp.get_context("spawn")
        # max_tasks_per_child: RECYCLE each worker after N candidates so the OS reclaims its heap (each
        # candidate alloc/frees a ~0.36 GiB replay buffer; without recycling the persistent workers' RSS
        # creeps from fragmentation). None = reuse workers forever (the PROVEN behavior). WARNING: a non-None
        # value HANGS on Windows + spawn across CPython 3.11-3.14 (deadlock measured 2026-06-24, all four) ->
        # the campaign reclaims RAM via ``run_recycling`` (fresh pools per batch) instead, and config keeps
        # this null. Passed explicitly (not via ``**kwargs``) so the call type-checks cleanly.
        # The initializer pins worker threads + preloads pyarrow-before-torch (production). Tests pass
        # ``initializer=None`` to spawn bare workers (no heavy imports), keeping the recycling unit test fast.
        _init: Callable[[], object] | None = (
            _worker_init if initializer is _DEFAULT_INIT else initializer
        )
        # ``init_threads`` reaches the worker as initargs (R107 wiring, 2026-07-27) so the packed
        # path honours the same thread regime as the inline one. Only passed when the initializer
        # actually takes it -- a test-supplied bare initializer takes no arguments, and handing it
        # initargs would raise inside the worker, where the traceback is far from the cause.
        _initargs = (int(init_threads),) if _init is _worker_init else ()
        self._ex = ProcessPoolExecutor(
            max_workers=self.n_gpu + self.n_cpu,
            mp_context=ctx,
            initializer=_init,
            initargs=_initargs,
            max_tasks_per_child=None if max_tasks_per_child is None else int(max_tasks_per_child),
        )

    def submit(self, spec: dict) -> Future:
        token = self._tokens.get()  # blocks until a device is free
        try:
            fut = self._ex.submit(train_candidate, {**spec, "device": token})
        except BaseException:
            # _ex.submit raised (pool already shut down, un-picklable spec, ...) BEFORE the done-callback
            # that returns the token could attach — return it here or the pool leaks a device forever and
            # eventually deadlocks on the next .get() (audit 2026-07-19).
            self._tokens.put(token)
            raise
        # token is a fresh per-call local (NOT a mutating loop var), so a plain closure captures it
        # correctly; no `t=token` default needed (which also broke the callback's type inference).
        fut.add_done_callback(lambda _f: self._tokens.put(token))
        return fut

    def submit_with(self, fn: Any, spec: dict) -> Future:
        """Like :meth:`submit` but runs an ARBITRARY picklable ``fn(spec)`` on the next free device
        token, instead of the default :func:`train_candidate`. Used for the TEST-leg worker so the
        winner re-runs share the SAME device-load-balanced pool as the search trainings.
        """
        token = self._tokens.get()  # blocks until a device is free
        try:
            fut = self._ex.submit(fn, {**spec, "device": token})
        except BaseException:
            # See submit(): return the token if _ex.submit raises before the callback attaches, else the
            # pool leaks a device and deadlocks on the next .get() (audit 2026-07-19).
            self._tokens.put(token)
            raise
        # token is a fresh per-call local (NOT a mutating loop var), so a plain closure captures it
        # correctly; no `t=token` default needed (which also broke the callback's type inference).
        fut.add_done_callback(lambda _f: self._tokens.put(token))
        return fut

    def __enter__(self) -> "DevicePool":
        return self

    def __exit__(self, *exc: Any) -> None:
        self._ex.shutdown(wait=True)


def run_recycling(
    specs: list[dict],
    *,
    worker: Any,
    n_gpu: int,
    n_cpu: int,
    recycle_every: int,
    initializer: Any = _DEFAULT_INIT,
    on_result: Callable[[dict], None] | None = None,
    stall_after_s: float | None = None,
    on_stall: Callable[[dict], None] | None = None,
) -> list:
    """Run ``specs`` through a SEQUENCE of fresh :class:`DevicePool`s of ``recycle_every`` tasks each.

    Tearing each pool down between batches (the ``with`` exit calls ``shutdown(wait=True)`` ->
    worker processes terminate) makes the OS reclaim each worker's ENTIRE address space — the
    fragmented SAC replay-buffer heap included — so per-worker RSS cannot creep across a long run.
    This is the deadlock-free substitute for ``max_tasks_per_child`` worker recycling, which HANGS on
    Windows + spawn across CPython 3.11-3.14 (measured 2026-06-24). The pool re-spawn (~15 s) is
    amortized over ``recycle_every`` trainings. Results are returned in submission order.

    Streaming archival (F1, review 2026-07-02; AS-COMPLETED 2026-07-06): futures are collected
    with :func:`concurrent.futures.wait` as they COMPLETE — not in submission order — so ``on_result``
    (when given) receives each result the moment its training finishes, success and captured-exception
    rows alike, and the caller can persist finished work IMMEDIATELY. Under the old submission-order
    collection a completed training's archival waited behind every earlier-submitted future (a driver
    crash mid-batch could orphan up to ``recycle_every - 1`` COMPLETED trainings ≈ 17 h at campaign
    scale, and ONE wedged training blocked the whole batch's archival). The RETURNED list stays in
    submission order (``scripts/run_sigma_pilot_train.py`` re-attributes results by position).

    Parameters
    ----------
    specs:
        The per-task spec dicts (e.g. one per ``(winner, seed)`` TEST run).
    worker:
        The picklable worker callable ``worker(spec) -> result`` run in each pool process. It SHOULD
        catch its own failures and return a result (e.g. ``{"ok": False, "error": ...}`` like
        ``train_candidate``); if it instead RAISES, the exception is captured as a failed result so a
        single worker crash (e.g. a CUDA OOM) never aborts the batch or the remaining batches.
    n_gpu, n_cpu:
        Concurrent cuda / cpu slots per pool (see :class:`DevicePool`).
    recycle_every:
        Tasks per pool before it is torn down and a fresh one is spawned (the RAM-reclaim cadence).
    initializer:
        Worker initializer; defaults to the production ``_worker_init`` (thread pinning +
        pyarrow-before-torch). Tests pass ``None`` for bare workers.
    on_result:
        Optional per-result callback ``on_result(row)`` invoked as each future completes (success and
        captured-exception rows both). An exception it raises is swallowed after stamping
        ``row['archive_error'] = "<Type>: <msg>"`` — a transient disk error must not abort the batch;
        the caller's post-loop writer retries the stamped rows.
    stall_after_s / on_stall:
        Wedge DETECTION (2026-07-06): when no future completes for ``stall_after_s`` seconds,
        ``on_stall({"pending": [...spec identity dicts...], "since_last_completion_s": ...})`` fires
        (then again every further ``stall_after_s`` of silence) so the caller can journal/alert a
        hung training — the failure mode a process-liveness check cannot see (driver alive, one CUDA
        training wedged). Detection only: a running future cannot be cancelled from here; the
        supervisor/sentinel own the response. ``None`` (default) disables the check (no behavior
        change). An ``on_stall`` exception is swallowed — monitoring must never kill the batch.
    """
    out: list = []
    specs = list(specs)
    step = max(1, int(recycle_every))
    for i in range(0, len(specs), step):
        batch = specs[i : i + step]
        with DevicePool(n_gpu, n_cpu, initializer=initializer) as pool:
            # Submit through a BOUNDED sliding window (2026-07-06 S15): ``submit_with`` BLOCKS on a
            # device token, so submitting the whole batch up-front (the old dict comprehension)
            # stalled COLLECTION until all but n_workers trainings had already completed — quietly
            # resurrecting the crash-loss window the as-completed rewrite exists to close (~9
            # finished trainings unarchived for hours at recycle_every=12/3 slots) and keeping
            # ``on_stall`` blind through the submission phase. With ``len(pending) < capacity`` a
            # token is guaranteed free (completed futures return theirs at completion), so
            # submission never blocks and archival/journaling/stall-detection stay live from the
            # FIRST completion. Same pattern as _drive_search_arm.
            capacity = max(1, pool.n_gpu + pool.n_cpu)
            results: list = [None] * len(batch)
            pending: dict[Future, int] = {}
            next_j = 0
            last_done = time.monotonic()
            while next_j < len(batch) or pending:
                while next_j < len(batch) and len(pending) < capacity:
                    pending[pool.submit_with(worker, batch[next_j])] = next_j
                    next_j += 1
                done, _ = wait(pending, timeout=stall_after_s, return_when=FIRST_COMPLETED)
                if not done:
                    # NOTHING completed within the stall window. Surface the pending identities so
                    # the caller can journal/alert, then keep waiting — the completed work of this
                    # batch is ALREADY archived (as-completed above), so a wedge now risks only the
                    # in-flight trainings.
                    if on_stall is not None:
                        try:
                            on_stall({
                                "pending": [
                                    {
                                        k: batch[j][k]
                                        for k in ("run_id", "arm", "cid", "seed", "candidate_id")
                                        if batch[j].get(k) is not None
                                    }
                                    if isinstance(batch[j], dict)
                                    else {"index": j}
                                    for j in sorted(pending.values())
                                ],
                                "since_last_completion_s": time.monotonic() - last_done,
                            })
                        except Exception:  # noqa: BLE001 — monitoring must never kill the batch
                            pass
                    continue
                for f in done:
                    j = pending.pop(f)
                    s = batch[j]
                    try:
                        row: Any = f.result()
                    except Exception as exc:  # noqa: BLE001 — a worker that RAISES (rather than
                        # returning an error result like ``train_candidate``) must NOT abort the batch
                        # or the remaining batches; capture it so every spec is still attempted (the
                        # caller checks ``ok`` and the matched-budget guard surfaces a failure wave).
                        row = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                        # F19: stamp the spec's identity keys onto the captured-exception row — a
                        # pool-level crash (worker killed by the OS, unpicklable result) otherwise
                        # yields an ANONYMOUS failure the caller cannot attribute to a
                        # (run_id, arm, seed) cell for resume/triage.
                        if isinstance(s, dict):
                            row.update(
                                {k: s[k] for k in ("run_id", "arm", "cid", "seed") if s.get(k) is not None}
                            )
                    results[j] = row
                    last_done = time.monotonic()
                    if on_result is not None:
                        # Stream the row to the caller's archiver NOW (success + failure alike). Never
                        # let an archive error abort the batch: stamp it and continue — the post-loop
                        # writer is the retry (it skips only rows it knows were durably written).
                        try:
                            on_result(row)
                        except Exception as exc:  # noqa: BLE001 — archiving must not kill the batch
                            if isinstance(row, dict):
                                row["archive_error"] = f"{type(exc).__name__}: {exc}"
            out.extend(results)
    return out


def _spec(arm: str, kind: str, reward: Any, cid: str, opts: dict) -> dict:
    return {
        "arm": arm,
        "reward_kind": kind,
        "reward": reward,
        "candidate_id": cid,
        "train_steps": opts["train_steps"],
        "batch_size": opts["batch_size"],
        "normalize_obs": opts["normalize_obs"],
        "learning_rate": opts.get("learning_rate", 3e-4),  # honor the full agent block (parity w/ serial + TEST)
        "gamma": opts.get("gamma", 0.99),
        "ent_coef": opts.get("ent_coef", "auto"),
        # learning_starts (gated 1000) + PopArt scale-normalization: threaded so the SEARCH worker trains the
        # SAME agent as serial/TEST (parity; the trainer floors learning_starts at 1000 and gates popart).
        "learning_starts": opts.get("learning_starts", 1000),
        # P10 (2026-07-13): the replay cap travels IN the spec (self-contained task JSON); absent/
        # None -> train_candidate's prior behavior (buffer==train_steps, node-side clamp).
        **({"buffer_size": int(opts["buffer_size"])} if opts.get("buffer_size") else {}),
        "popart": opts.get("popart", True),
        "popart_beta": opts.get("popart_beta", 1e-3),
        "popart_min_scale": opts.get("popart_min_scale", 1.0),
        "popart_warmup": opts.get("popart_warmup", 0),
        "tf32": opts.get("tf32", True),
        # M6: thermal_guardian threaded from the campaign agent block (build_parallel_opts) into every
        # worker spec so SEARCH trainings are governed too (None -> off; result-neutral wall-clock pause).
        "thermal_guardian": opts.get("thermal_guardian"),
        "n_trials": opts["n_trials"],
        "synthetic": opts["synthetic"],
        "data": opts["data"],
        "cvar_alpha": opts["cvar_alpha"],
        "window": opts["window"],
        "seed": opts["seed"],
        "monitor_queue": opts.get("monitor_queue"),  # main ParallelMonitor's queue (None when unmonitored)
    }


#: Process-local cache of the (expensive) env capture + its sha256, keyed by seed. The environment
#: doesn't change within a run, so capture pip-freeze/nvidia-smi/torch-cuda ONCE per (process, seed).
_ENV_CACHE: dict[Any, tuple[dict, str]] = {}


def _capture_env_cached(seed: Any) -> tuple[dict, str]:
    key = seed
    if key not in _ENV_CACHE:
        try:
            from scripts.capture_env import capture_env, env_json_sha256
        except ModuleNotFoundError:  # a directly-launched `python scripts/run_campaign.py` puts
            import sys as _sys        # scripts/ on sys.path, not the repo root, so `scripts` is not an
            from pathlib import Path as _Path  # importable package (same class as the seal bug,
            _sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "scripts"))  # 2026-07-24);
            from capture_env import capture_env, env_json_sha256   # Myriad's `-m` run keeps this dead.

        env = capture_env(seed=int(seed) if isinstance(seed, (int, float)) else None)
        _ENV_CACHE[key] = (env, env_json_sha256(env=env))
    return _ENV_CACHE[key]


def _run_env_fp(arm_root: str, run_id: str, opts: dict) -> Any:
    """Write ``<arm_root>/<run_id>/env.json`` and return the record's ``env_fingerprint`` (Rank 14).

    Best-effort: on any failure (e.g. torch absent in a degraded env) it falls back to the bare
    ``opts['env_fp']`` label so archiving never breaks. On success the returned dict pairs the
    human-readable label with the content-hash of the on-disk ``env.json`` so the record resolves
    back to a full, replayable environment snapshot (audit C-2/C-6).
    """
    import json as _json

    label = opts.get("env_fp", "")
    try:
        env, digest = _capture_env_cached(opts.get("seed"))
        run_dir = Path(arm_root) / str(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "env.json").open("w", encoding="utf-8") as fh:
            _json.dump(env, fh, indent=2, sort_keys=True, default=str)
        return {"label": label, "env_json_sha256": digest}
    except Exception:  # noqa: BLE001 - provenance capture must never crash a candidate's archive
        # P17/A4-F11 (2026-07-13 audit): the bare fallback label is the DRIVER-built fingerprint
        # (on the cluster: the LAPTOP env captured at spec-build time) — returning it unmarked
        # made the S6 homogeneity audit count a phantom laptop environment as if a node had
        # captured it. Mark the fallback so the census sees a capture FAILURE (loud,
        # distinguishable) instead of a silently wrong environment.
        return f"capture-failed:{label}" if label else "capture-failed"


def _archive(result: dict, arm: str, opts: dict, archive_root: str, generation: int = 0) -> None:
    from src.io.results import write_run

    cid = result["candidate_id"]
    arm_root = str(Path(archive_root) / arm)
    write_run(
        {
            "run_id": cid,
            "arm": arm,
            "seed": opts["seed"],
            "fold": 0,
            "candidate_id": cid,
            "generation": generation,
            # Rank 14: persist the rendered prompt so the replay archive doesn't DROP it. The LLM driver injects ``prompt``; search arms have no prompt -> "".
            "prompt": result.get("prompt", ""),
            "reward_source": result.get("reward_source", ""),
            "reward_source_hash": result.get("reward_hash", ""),
            "feedback_block": "",
            "metrics": {
                "val_fitness": result.get("fitness", float("-inf")),
                "val_returns": result.get("val_returns"),
                "tail_stats": result.get("tail_stats"),
                # B-A2 (k-seed search): TRAIN-window returns, archived ONLY when the worker was asked
                # to emit them (k>1 specs) so the k-seed fed-tail aggregate replays from the archive.
                **(
                    {"train_returns": result["train_returns"]}
                    if result.get("train_returns") is not None
                    else {}
                ),
                # T2.4: realised PopArt scale (sigma_max/last) the critic saw, for the cross-arm sigma audit.
                # Optional/back-compatible — omitted when the worker didn't surface it (older records / fakes).
                **(
                    {"popart_scale": result["popart_scale"]}
                    if result.get("popart_scale") is not None
                    else {}
                ),
                # R66 (2026-07-03): training-window SAFE_DEFAULT substitution counts (train_agent attach)
                # — additive/optional, mirroring the popart_scale pattern above.
                **(
                    {
                        "train_safe_default_count": int(result["train_safe_default_count"]),
                        "train_safe_call_count": int(result["train_safe_call_count"]),
                    }
                    if result.get("train_safe_default_count") is not None
                    and result.get("train_safe_call_count") is not None
                    else {}
                ),
            },
            # 2026-07-13 pre-spend audit fix: was HARDCODED 0.0 — the worker's measured elapsed
            # seconds now flow through (0.0 only for legacy results lacking the field).
            "wall_clock": float(result.get("elapsed_secs", 0.0)),
            # Rank 14: the env_fingerprint is the REAL provenance now, not a bare label. ``_run_env_fp``
            # writes <run_dir>/env.json (full CI-grade capture) and returns {label, env_json_sha256} so
            # the record points at a replayable, content-hashed environment snapshot (audit C-2/C-6).
            "env_fingerprint": _run_env_fp(arm_root, cid, opts),
        },
        arm_root,
    )


def _summary(arm: str, accepted: list, failed: int, expected: int) -> dict:
    winner = max(accepted, key=lambda r: r["fitness"]) if accepted else None
    return {
        "arm": arm,
        "n_candidates": len(accepted),
        "n_failed": failed,
        "winner_fitness": None if winner is None else float(winner["fitness"]),
        # Require the full budget AND at least one ACCEPTED candidate (re-audit: this parallel twin
        # of run_arm's matched-budget check lacked the accepted>0 guard from final-audit #21, so an
        # all-failures arm reported healthy; this flag drives the campaign-level health verdict).
        "matched_budget_ok": (len(accepted) + failed) == expected and len(accepted) > 0,
    }


def _drive_llm_arm(arm: str, pool: DevicePool, opts: dict, archive_root: str) -> dict:
    import json as _json

    from src.feedback import schema
    from src.io.results import load_run
    from src.llm.client import JsonlArchiveSink, LLMClient
    from src.llm.loop import _REFLECTION_PREAMBLE, _diversity_directive
    from src.llm.prompts import build_prompt_set
    from src.sandbox.executor import extract_reward_source

    # ── Search-replay cache (resume) ──────────────────────────────────────────────────
    # On resume a previously-archived candidate is REPLAYED from disk instead of re-calling the
    # (paid, non-deterministic) LLM and re-training: a mid-search resume becomes byte-faithful to the
    # original run (same candidates -> same generation-best -> same reflection seed -> same winner) AND
    # saves the Opus spend + GPU time. Mirrors src/llm/loop.py's serial cache, adapted to the parallel
    # archive scheme: successes reload via ``load_run(cid, arm_root)`` (parallel ids are bare ``cid``,
    # NOT ``<prefix>-<cid>``); the EMPTY stored feedback_block is irrelevant because this path rebuilds
    # the reflection block LIVE from the generation's BEST candidate; sandbox FAILURES replay from the
    # arm's failures.jsonl ledger so a previously-rejected candidate is NOT re-generated (which, against
    # a non-deterministic author, could newly SUCCEED and silently change the candidate set / winner).
    # Default OFF -> a fresh run is byte-for-byte unchanged.
    resume = bool(opts.get("resume", False))
    arm_root = str(Path(archive_root) / arm)
    # T2.8b arm-scoped FED-estimator audit (fix 2026-07-03): clear the process-level EVT<->empirical
    # switch registry at ARM START so the "estimator switched across candidates" warning scopes to THIS
    # arm's in-process measurements, not across previously-driven arms (the registry was never reset in
    # production). NB the parallel workers measure tail stats in their OWN processes, whose registries
    # this cannot reach — the reset here covers the driver process (parity with run_prototype.run_arm).
    from src.feedback.measurement import reset_fed_estimator_log

    reset_fed_estimator_log()
    # F5 (review 2026-07-02): the failures ledger is APPENDED per rejection AT THE MOMENT it
    # happens, mirroring the serial loop's crash-robust append ledger (src/llm/loop.py) — the old
    # end-of-arm "w" write meant a mid-arm crash lost EVERY ledgered failure, so --resume re-generated
    # previously-rejected candidates against the non-deterministic author (which could newly succeed
    # and silently change the candidate set / winner).
    fail_ledger = Path(arm_root) / "failures.jsonl"
    # (batch-5 m5) The SERIAL loop writes a DIFFERENTLY-named ledger — <root>/<prefix>-<arm>.failures.jsonl
    # (src/llm/loop.py ~317) — but a mid-campaign SERIAL<->PARALLEL search-mode switch is refused up front
    # by run_campaign._assert_search_mode_unchanged, so any one arm is only ever driven by a SINGLE mode and
    # the two ledger layouts never need cross-replay (each mode's resume reads only its own ledger).

    def _ledger_failure(row: dict) -> None:
        # Best-effort append+flush (a ledger write must never crash the arm) — the serial loop's exact
        # contract; the resume reader below and _search_pool_counts both key on ``candidate_id``.
        try:
            fail_ledger.parent.mkdir(parents=True, exist_ok=True)
            with fail_ledger.open("a", encoding="utf-8") as _fh:
                _fh.write(_json.dumps(row, default=str) + "\n")
                _fh.flush()
        except Exception:  # noqa: BLE001
            pass

    cached_failures: dict[str, dict] = {}
    if resume and fail_ledger.is_file():
        for _ln in fail_ledger.read_text(encoding="utf-8").splitlines():
            _ln = _ln.strip()
            if not _ln:
                continue
            try:  # a torn last line just means that one candidate regenerates — never crash resume
                _f = _json.loads(_ln)
                cached_failures[str(_f["candidate_id"])] = _f
            except Exception:  # noqa: BLE001
                pass

    if opts["pass_mode"].upper() == "A" or opts["provider"] == "stub":
        from src.llm.stub_designer import StubDesignerTransport

        transport: Any = StubDesignerTransport(seed=opts["seed"])
        model = f"stub-designer/seed{opts['seed']}"
    else:
        from src.llm.client import build_transport, default_key_env

        model = opts["model"]
        temp_raw = opts.get("temperature")
        temperature = float(temp_raw) if temp_raw is not None else None
        key_env = opts.get("api_key_env") or default_key_env(opts["provider"])
        # F17: thread the author block's max_tokens/max_retries through (defaults = the historical
        # hardcodes 4096/6) — raising the config value must not silently no-op into a truncated reward.
        # v2 legs: extra_body carries the OpenRouter provider/quantization/reasoning pins + the
        # usage-cost request — dropping it here would strip a leg's registered pins at authoring.
        transport = build_transport(
            opts["provider"], model, key_env, temperature=temperature,
            max_tokens=int(opts.get("max_tokens") or 4096),
            max_retries=int(opts.get("max_retries") or 6),
            extra_body=opts.get("extra_body") or None,
            # R106: the ANTHROPIC equivalent of the extra_body pins (that transport rejects
            # extra_body), so a registered reasoning-off actually reaches the API.
            thinking=opts.get("thinking") or None,
        )

    diversity = bool(opts.get("diversity_prompt_variation", False))

    prompts = build_prompt_set(opts["env_cfg"], opts["n_assets"])
    # F11: llm_calls.jsonl is appended PER CALL at call time via the archive sink — the old end-of-arm
    # "w" dump lost the whole arm's call provenance (incl. the R71 served_model reproducibility anchor,
    # which config/llm.yaml requires recorded at the FIRST live call) on any mid-arm crash.
    llm = LLMClient(
        # 2026-07-28: `provider` threaded here for the same reason as in the cluster author
        # (src/cluster/campaign.py) — omitting it let client.py fall back to its "anthropic"
        # default, mis-attributing every OpenRouter leg's cost in the spend ledger. Routing was
        # never affected: the transport above is built from the real opts["provider"].
        {"model": model, "provider": opts["provider"],
         "spend_ledger": opts.get("spend_ledger")},
        transport=transport,
        archive=JsonlArchiveSink(Path(arm_root) / "llm_calls.jsonl"),
    )
    gens = max(1, int(opts["generations"]))
    cpg = max(1, int(opts["candidates"]) // gens)
    accepted: list = []
    failed = 0
    prev_block: str | None = None
    for gen in range(gens):
        # Reflection PREAMBLE identical to src/llm/loop.py (final-audit #34: both paths share the same
        # preamble string, not two different hand-written ones). NB the appended feedback `prev_block`
        # itself derives from each path's OWN selection — the serial loop seeds from its generation-BEST candidate (M5),
        # this scheduler from the generation's BEST — so the full reflection prompt is not byte-identical
        # across paths for multi-candidate generations; the headline campaign uses the SERIAL path.
        user = prompts.initial if prev_block is None else f"{_REFLECTION_PREAMBLE}\n{prev_block}"
        # Each item is ("replay", result_dict) — an already-archived candidate replayed from disk —,
        # ("fail", cid) — a previously-archived sandbox failure replayed from the ledger —, or
        # ("live", cand_user, future) — a freshly generated candidate trained in the pool. Built in `k`
        # order so the best-selection + archive ordering below is IDENTICAL whether a candidate is
        # replayed or freshly generated (parity with a non-resume run).
        items: list[tuple] = []
        for k in range(cpg):
            cid = f"{arm}-g{gen}-c{k}"
            if resume:
                hit = None
                try:
                    hit = load_run(cid, arm_root)
                except FileNotFoundError:
                    hit = None  # not yet done -> fall through and generate
                # A corrupt/integrity-failed record (KeyError/ValueError from load_run) PROPAGATES —
                # failing loud beats silently regenerating (which would re-bill Opus + could desync the
                # search), matching the serial path's rationale.
                if hit is not None:
                    _m = hit.get("metrics", {}) or {}
                    r: dict[str, Any] = {
                        "ok": True,
                        "candidate_id": cid,
                        "arm": arm,
                        "fitness": float(_m["val_fitness"]),
                        "val_returns": _m.get("val_returns"),
                        "tail_stats": _m.get("tail_stats"),
                        "reward_source": hit.get("reward_source", ""),
                        "reward_hash": hit.get("reward_source_hash", ""),
                        "prompt": hit.get("prompt", ""),
                    }
                    if _m.get("popart_scale") is not None:
                        r["popart_scale"] = _m["popart_scale"]
                    items.append(("replay", r))
                    # Stream-faithful replay (2026-07-06): the slot was AUTHORED originally —
                    # consume its author-stream position so a POSITIONAL (stub) transport keeps
                    # later fresh candidates index-aligned; real LLM transports no-op.
                    llm.advance_author_stream()
                    continue
                if cid in cached_failures:
                    items.append(("fail", cid))
                    llm.advance_author_stream()  # a ledgered failure was authored too (one position)
                    continue
            # Per-candidate prompt variation -> within-generation diversity without temperature
            # (mirrors src/llm/loop.py; uniform across arms, not a feedback confound).
            cand_user = user
            if diversity and cpg > 1:
                cand_user = f"{user}\n\n{_diversity_directive(k, cpg)}"
            # Salvage fenced / prose-wrapped LLM output before it reaches the gate (final-audit P0).
            src = extract_reward_source(llm.complete(prompts.system, cand_user))
            # Rank 14: carry the rendered user prompt alongside the future so the archive can persist
            # the EXACT prompt sent (the worker result dict has no prompt; directive 6). candidate_id
            # uses the per-generation index `k` (final-audit #22/#33: matches src/llm/loop.py and keeps
            # the archived directive index consistent with the id, instead of a global counter).
            items.append(("live", cand_user, pool.submit(_spec(arm, "source", src, cid, opts))))
        best = None
        for item in items:
            if item[0] == "fail":
                # A previously-LEDGERED sandbox failure: count it and do NOT regenerate (a
                # non-deterministic re-author could newly succeed and change the set). Its row is
                # ALREADY on disk (F5 appends at rejection time), so it is NOT re-appended here —
                # re-appending would duplicate ledger lines on every resume (readers dedupe by
                # candidate_id, so duplicates are harmless, but we do not create them).
                failed += 1
                continue
            if item[0] == "replay":
                # An already-archived accepted candidate, treated EXACTLY like a live-accepted result
                # (feeds `accepted` + `best`) — but NOT re-archived (it is already on disk) and NOT
                # submitted for training.
                r = item[1]
            else:  # "live"
                user_prompt, f = item[1], item[2]
                r = f.result()
                if not r.get("ok"):
                    failed += 1
                    # Capture the failed source + error (re-audit / final-audit #21 parity: the parallel
                    # path discarded failures, so a total-failure arm was undiagnosable from the archive).
                    # F5: appended to the on-disk ledger IMMEDIATELY (_ledger_failure) in the serial
                    # loop's shape (candidate_id/generation/reward_source/error[:500]) PLUS the prompt
                    # the parallel ledger has always carried (the #21 diagnosability payload; every
                    # reader keys on candidate_id and ignores extra fields).
                    _ledger_failure({
                        "candidate_id": r.get("candidate_id"),
                        "generation": gen,
                        "prompt": user_prompt,
                        "reward_source": r.get("reward_source"),
                        "error": str(r.get("error"))[:500],
                    })
                    continue
                r["prompt"] = user_prompt
                _archive(r, arm, opts, archive_root, generation=gen)
            accepted.append(r)
            if best is None or r["fitness"] > best["fitness"]:
                best = r
        if best is not None:
            tail_for = (
                best.get("tail_stats")
                if arm in ("distributional", "scalar_cvar5", "placebo_shuffled")
                else None
            )
            prev_block = schema.build_block(
                arm,
                best["fitness"],
                tail_for,
                shuffle_seed=(
                    schema.shuffle_seed_from_id(str(best.get("candidate_id", "")))
                    if arm == "placebo_shuffled"
                    else None
                ),
            )
    # NB (F5 + F11, review 2026-07-02): NO end-of-arm ledger writes here any more. The raw LLM
    # provenance (llm_calls.jsonl, final-audit #36) is appended PER CALL by the JsonlArchiveSink the
    # LLMClient above was constructed with, and each failure row (final-audit #21) is appended at
    # rejection time by _ledger_failure — so a mid-arm crash loses at most the in-flight call/row,
    # never the arm's whole provenance (the old "w"-mode dumps did exactly that).
    return _summary(arm, accepted, failed, gens * cpg)


def _drive_search_arm(arm: str, pool: DevicePool, opts: dict, archive_root: str) -> dict:
    n = int(opts["candidates"])
    # Search-replay cache (resume, 2026-07-06 S21): the serial path got per-candidate checkpoint +
    # hash-verified resume on 2026-07-05, but this parallel driver re-trained ALL candidates on
    # --resume (~42 GPU-h re-paid per search arm at campaign budget) and silently overwrote the
    # archived records. Mirror the serial discipline: draws stay UNCONDITIONAL (the rng stream is
    # position-faithful), an archived candidate is REPLAYED instead of re-trained, and a drawn
    # source whose hash mismatches its archived record fails LOUD (draw-sequence drift).
    resume = bool(opts.get("resume", False))
    arm_root = str(Path(archive_root) / arm)

    def _replay(cid: str, expected_source: str) -> dict | None:
        """The archived result for ``cid`` (None when absent); hash-verifies against the fresh draw."""
        if not resume:
            return None
        from src.io.results import load_run

        try:
            hit = load_run(cid, arm_root)
        except FileNotFoundError:
            return None  # not yet done -> train it
        # Corrupt records (KeyError/ValueError) PROPAGATE — same fail-loud rationale as every
        # other resume path (silently regenerating could desync the search).
        recorded = str(hit.get("reward_source_hash", ""))
        actual = hashlib.sha256(expected_source.encode("utf-8")).hexdigest()
        if recorded and recorded != actual:
            raise RuntimeError(
                f"parallel search resume cache MISMATCH for {cid}: the archived candidate was built "
                f"from a different source than this run drew at its index (recorded {recorded[:12]}.. "
                f"!= drawn {actual[:12]}..) — config/seed drift; refusing to mix archives"
            )
        _m = hit.get("metrics", {}) or {}
        r: dict[str, Any] = {
            "ok": True, "candidate_id": cid, "arm": arm,
            "fitness": float(_m["val_fitness"]),
            "val_returns": _m.get("val_returns"), "tail_stats": _m.get("tail_stats"),
            "reward_source": hit.get("reward_source", ""),
            "reward_hash": recorded,
        }
        if _m.get("popart_scale") is not None:
            r["popart_scale"] = _m["popart_scale"]
        return r

    if arm == "random_search":
        from src.search.random_search import sample_reward_source

        rng = np.random.default_rng(opts["seed"])
        # Draw the six family weights from the SAME frozen ranges the BO arm uses
        # (proto_cfg.reward_family.weights), so both H4 arms search the identical space.
        # ALL sources are drawn UP FRONT, in candidate order, so the rng stream is byte-identical
        # to the old draw-at-submit code regardless of the as-completed scheduling below.
        _rf_cfg = opts.get("proto_cfg")
        sources = [sample_reward_source(rng, _rf_cfg) for _ in range(n)]
        # As-completed archival (2026-07-06): the old code archived ONLY after every future resolved,
        # so a driver crash mid-arm lost the WHOLE arm's completed trainings. Submit through a sliding
        # window (``pool.submit`` blocks on a device token, so unbounded up-front submission would
        # stall collection too) and archive each ok result the moment it completes. The results list
        # stays in submission order, so ``_summary``'s winner/tie-break semantics are unchanged.
        capacity = max(1, pool.n_gpu + pool.n_cpu)
        results: list[Any] = [None] * n
        pending: dict[Future, int] = {}
        next_i = 0
        while next_i < n or pending:
            while next_i < n and len(pending) < capacity:
                cid = f"{arm}-c{next_i}"
                cached = _replay(cid, sources[next_i])
                if cached is not None:
                    results[next_i] = cached  # replayed — neither re-trained nor re-archived
                    next_i += 1
                    continue
                pending[pool.submit(_spec(arm, "source", sources[next_i], cid, opts))] = next_i
                next_i += 1
            if not pending:
                continue  # everything in this window replayed from the archive
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for f in done:
                i = pending.pop(f)
                r = f.result()  # train_candidate catches its own failures -> always a result dict
                results[i] = r
                if r.get("ok"):
                    _archive(r, arm, opts, archive_root)
        accepted = [r for r in results if r.get("ok")]
        return _summary(arm, accepted, len(results) - len(accepted), n)

    # bayes_opt — sequential BO (each evaluation trained via the shared pool, so it overlaps the
    # other arms; BO itself is inherently sequential).
    from src.baselines.reward_family import family_bounds, params_to_source
    from src.search.dfo_toolkit import over_template_optimizer  # GP-EI / CMA-ES / TPE, resolved by arm

    state: dict[str, Any] = {"i": 0, "results": []}

    def template_eval(coeffs) -> float:
        cid = f"{arm}-c{state['i']}"
        state["i"] += 1
        # Resume replay keyed by the MATERIALIZED executable source for these coefficients — the
        # exact text the archive stores — so hash verification catches template/alpha/window drift
        # as well as seed drift (parity with the serial BO cache).
        src = params_to_source(
            list(coeffs), cvar_alpha=float(opts.get("cvar_alpha", 0.05)),
            window=int(opts.get("window", 20)),
        )
        cached = _replay(cid, src)
        if cached is not None:
            state["results"].append(cached)
            return float(cached["fitness"])
        r = pool.submit(_spec(arm, "coeffs", list(coeffs), cid, opts)).result()
        state["results"].append(r)
        # As-completed archival (2026-07-06): archive each ok evaluation IMMEDIATELY — the old
        # end-of-arm loop lost every completed BO training on a mid-arm driver crash.
        if r.get("ok"):
            _archive(r, arm, opts, archive_root)
        return float(r["fitness"]) if r.get("ok") else -1e9

    # Seed the BO sampler from the run seed (re-audit: this parallel BO arm still drew OS entropy via
    # the rng=None default, so its winner was non-reproducible — the exact defect final-audit #2 closed
    # on the sequential path; random_search above already seeds the same way).
    over_template_optimizer(arm)(
        template_eval,
        cast(Sequence[Sequence[float]], family_bounds(opts.get("proto_cfg"))),
        {"matched_budget": n},
        rng=np.random.default_rng(opts["seed"]),
    )
    accepted = [r for r in state["results"] if r.get("ok")]
    return _summary(arm, accepted, len(state["results"]) - len(accepted), n)


def run_parallel(arms: list[str], opts: dict, n_gpu: int, n_cpu: int, archive_root: str) -> list[dict]:
    """Run all ``arms`` concurrently across a heterogeneous GPU+CPU candidate pool.

    Arm drivers run as threads (they only generate reward code / run light BO + archive); the heavy
    SAC trainings run in the ``DevicePool`` process workers, device-load-balanced.
    """
    # Live monitoring (candidate-parallel): the main ParallelMonitor drains a Manager queue that every
    # training worker streams per-step SAC metrics + lifecycle into — precise progress (run_done/run_total,
    # N concurrent trainings, ETA), deep JSONL event logs (incl. the driver threads' LLM/sandbox logs),
    # GPU/CPU/RAM telemetry, and CENTRAL anomaly detection. scripts/monitor.py renders it from progress.json.
    import threading

    from src.utils.monitoring import ParallelMonitor

    mgr = mp.Manager()
    opts = {**opts, "monitor_queue": mgr.Queue()}
    monitor = ParallelMonitor(
        archive_root,
        title="Prototype (parallel)",
        total_arms=len(arms),
        candidates_per_arm=int(opts.get("candidates", 0) or 0),
        train_steps=int(opts.get("train_steps") or 25000),
        model=str(opts.get("model", "?")),
    )
    stop = threading.Event()
    pump = threading.Thread(target=monitor.pump, args=(opts["monitor_queue"], stop), daemon=True)
    pump.start()
    summaries: list[dict] = []
    try:
        with DevicePool(n_gpu, n_cpu, opts.get("max_tasks_per_child")) as pool, \
                ThreadPoolExecutor(max_workers=max(1, len(arms))) as drivers:
            futs = {}
            for arm in arms:
                drive = _drive_llm_arm if arm in _LLM_ARMS else _drive_search_arm
                futs[drivers.submit(drive, arm, pool, opts, archive_root)] = arm
            for f in as_completed(futs):
                summaries.append(f.result())
        monitor.close(status="done")
    except BaseException:
        monitor.close(status="error")
        raise
    finally:
        stop.set()
        pump.join(timeout=3)
        # m12 (ops audit 2026-07-02): shut the per-call mp.Manager down explicitly. run_parallel is
        # invoked ONCE PER ARM by the campaign's parallel search, and each Manager is a live helper
        # PROCESS + socket; without shutdown they only die at interpreter exit, so a 7-arm campaign
        # accumulates orphaned manager processes for the whole multi-week run. Best-effort: a manager
        # that already died must not mask a real exception from the try block above.
        try:
            mgr.shutdown()
        except Exception:  # noqa: BLE001 - cleanup must never raise over the run's real outcome
            pass
    return summaries
