"""LANE PLANNING — the makespan model that decides what runs on CPU, what runs on GPU, and why.

WHY THIS EXISTS. "Maximise campaign speed" is not "get more cores": the campaign's wall-clock is a
MAKESPAN, and a makespan is the max of a throughput term and a critical-path term::

    makespan = max( total_work / capacity ,  longest_sequential_chain )

The work decomposes into three shapes with wildly different scheduling character (all counts from
``config/preregistration.yaml``; totals validated in the dossier §0-PRE M7):

============================  ===============  ==========================  ==================
component                     trainings @n=568 structure                   CPU critical path
============================  ===============  ==========================  ==================
test flood + H1 baselines     40,328           embarrassingly parallel     --
LLM reflection chains         1,650            55 chains x 6 SEQUENTIAL    6 x 8.54 h = 2.1 d
``bayes_opt`` GP-EI chain     30               5 parallel + **25 SERIAL**  25 x 8.54 h = 8.9 d
TPE (batched startup)         30               10 batched + **20 SERIAL**  20 x 8.54 h = 7.1 d
CMA-ES                        30               populations => ~4 SERIAL    ~1.4 d
============================  ===============  ==========================  ==================

THE RESULT THAT DRIVES EVERYTHING: with ~360,000 core-hours of total work (9 arms since
2026-07-26), the throughput term falls below the 8.9-day ``bayes_opt`` chain at about **1,685 CPU
cores**. Past that the campaign is
LATENCY-bound, and additional CPU capacity buys **literally nothing** -- while the binding chain is
just **30 trainings**, which one GPU at pack-1 finishes in ~27 h instead of 8.9 days.

AND THEN THE THREAD SWEEP CHANGED THE ANSWER AGAIN (jobs 17784/17836, 2026-07-26). Every training
is pinned to ONE thread -- correct for the test flood, but the chain's cost is pure LATENCY, where
aggregate throughput is worthless. Measured: **8 threads = 2.72x on a single training** (and 16
threads REGRESSES to 2.11x). Applied to the contended campaign rate that turns the ``bayes_opt``
chain from **8.9 days into ~3.3 days on plain CPU** -- below the throughput term at any realistic
core count. **So the GPU is no longer NEEDED at all**; it remains a nice-to-have that would take the
chain to 1.13 d, but the campaign is throughput-bound on pure CPU either way.

NOTE: **CORRECTED 2026-07-30 (record §39), APPLIED 2026-08-01 (RUN 10).** The 2.72x above came from an
ISOLATED BENCH — one 8-core box with nothing else running. In PRODUCTION, across **740 timed
trainings on shared 36-core nodes**, 8 threads is **1.92x by median / 1.75x by mean**: co-tenants
take memory bandwidth an idle bench never loses. The paragraph above is kept as DATED HISTORY (it
is the reasoning that retired the GPU requirement, and that conclusion still holds — it holds MORE
strongly at the pessimistic constant, since the chain still falls below the throughput term). But
the NUMBERS in it are superseded: the chain floor is **4.64 d, not ~3.3 d**, and the crossover is
**~3,235 cores at rung 568, not ~4,584**.

So the plan is a THREAD/CORE split, not a CPU/GPU one:

* **LANE 1 (CPU, LATENCY -- the sequential chains):** ``bayes_opt``'s 25 serial GP-EI steps, the 55
  LLM reflection chains, and the TPE/CMA-ES DFO comparators, each run at
  :data:`CPU_CHAIN_THREADS` = 8 threads. These run CONCURRENTLY with one another, so the critical
  path is the MAX over them (~25-30 serial trainings), never the sum.
* **LANE 2 (CPU, THROUGHPUT -- 97%+ of all trainings):** the test flood + H1 baselines, at **1
  thread per core**, scaled as wide as the scheduler grants (``killswitch.plan_footprint``).
  Measured justification: 8 cores as 8x 1-thread trainings give ~104 steps/s aggregate versus ~35
  for one 8-thread training -- 1 thread is ~3x better here, exactly inverting the chain's rule.
* **LANE 3 (GPU, purely opportunistic):** if a GPU is granted, put the longest chain on it. Never a
  dependency, and after the thread lever, never a necessity either.

NOTE: THE THREAD COUNT IS PART OF THE DETERMINISM ENVELOPE. Multi-threaded BLAS changes the ORDER of
float reductions, so an 8-thread training is NOT bit-identical to a 1-thread one.
The throughput run plan prescribes exactly the procedure followed here -- BENCH, then
RATIFY pre-freeze -- and the bench above is the material win it required. Two conditions make it
sound: the split is applied to the SEARCH/chain leg only (every SCORED comparison lives on the
uniformly 1-thread test leg, so no paired contrast is affected), and ``scripts/capture_env.py`` now
records ``OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS`` plus ``torch.get_num_threads()`` so a heterogeneous
thread regime is DETECTABLE by audit rather than merely improbable.

NOTE: TWO SCIENCE CONSTRAINTS ARE ENCODED HERE, NOT LEFT TO OPERATOR MEMORY:

1. **CPU-microarchitecture homogeneity.** The ``t`` pool is AMD EPYC 9554P (Zen4); every other pool
   is Intel Xeon. PyTorch-CPU dispatches different oneDNN kernels per microarchitecture, which
   changes float reduction ORDER -- exactly the bit-exactness the CRN pairing of every paired
   contrast depends on. ``t`` is therefore EXCLUDED from the confirmatory lane (it also measured no
   faster: 14.75 vs 14.2 steps/s/core), so the exclusion costs nothing.
2. **Do not harvest idle CPU cores on GPU nodes.** e/f/l/u/v had ~850 free cores, but GPU jobs
   request ``-pe smp 4`` alongside ``gpu=1``; taking those cores would block GPU users. This is the
   one case where our campaign would genuinely impair others, so it is refused by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "CPU_STEPS_PER_S_PER_CORE",
    "CPU_PLANNING_STEPS_PER_SEC",
    "GPU_PACK1_STEPS_PER_S",
    "SERIAL_CHAIN_STEPS",
    "SERIAL_CHAIN_BUDGET",
    "CONFIRMATORY_CPU_POOLS",
    "EXCLUDED_CPU_POOLS",
    "training_core_hours",
    "total_trainings",
    "LanePlan",
    "plan_lanes",
    "cpu_saturation_cores",
]

#: Sustained CPU rate, warmup-corrected, measured 2026-07-26 over 148 completed jobs under our own
#: ~600-core load (mean 14.19 steps/s raw; ``learning_starts: 1000`` makes 1/12 of a 12k bench
#: gradient-free). Flat across footprints 1-28 workers.
CPU_STEPS_PER_S_PER_CORE = 13.0

#: CPU WALLTIME-PLANNING FLOOR -- the rate every ``h_rt`` request on the CPU lane is sized from.
#: DELIBERATELY BELOW the measured 13.0 above: that is a MEAN over completed jobs, and a walltime
#: request must survive the TAIL. Evidence for the tail being heavy, in order of discovery
#: (2026-07-27): a 102 steps/s GPU-era anchor, then a co-tenanted node at 51, then a ladder cell
#: h_rt-KILLED at its full 6 h (``qacct failed=37``, ``ru_wallclock 21,612``) implying that node
#: sustained UNDER 37 -- two successive downward surprises. 10.0 prices the measured 13.0 with
#: ~30 % co-tenancy margin and is conservative for the 8-thread chain regime too (15.4 measured).
#:
#: ONE OWNER, deliberately: exactly one module defines this constant, and it is imported by
#: BOTH CPU walltime estimators -- ``scripts/p6_authored_ladder._auto_h_rt`` and
#: ``scripts/run_campaign_cluster.autosize_h_rt``. It was defined in the ladder alone until
#: 2026-07-27, and the campaign launcher consequently sized 400k CPU tasks at 6 h against a real
#: 8.55 h -- i.e. every task SIGKILLed. A second copy is how that recurs; there is now no copy.
#:
#: ``h_rt`` is a LIMIT, not a reservation. Over-asking costs only backfill position, and walltime
#: was MEASURED irrelevant to placement (an 11 h request placed as fast as a 50 min one, 15/15);
#: under-asking costs the whole job. Asymmetric loss -> price the floor, not the mean.
CPU_PLANNING_STEPS_PER_SEC = 10.0

#: One GPU, ONE training (pack=1) -- the latency figure, not a throughput figure. From the
#: launcher's own measured curve (``run_campaign_cluster.autosize_h_rt::_agg_clean``) and confirmed
#: by the G1 anchor (102.2 steps/s, real SAC at obs 1893, job 764154 on a V100-PCIE-32GB).
GPU_PACK1_STEPS_PER_S = 102.0

#: MEASURED CPU THREAD SCALING for ONE training (jobs 17784/17836, 2026-07-26, b-pool, campaign
#: profile). Two independent runs agree closely, and the curve has a REGRESSION past 8 threads --
#: classic small-matmul oversubscription (the nets are 256x256 at batch 256, so each parallel
#: region is tiny relative to its launch overhead).
#:
#:     threads:  1      2      4      8      16
#:     16-core:  20.9   32.9   46.4   55.1   44.0   steps/s   (speedup 1.00 1.57 2.22 2.64 2.11)
#:      8-core:  21.5   33.7   48.1   60.0    --              (speedup 1.00 1.57 2.24 2.80)
#:
#: THE RULE THIS ESTABLISHES -- threads where LATENCY binds, cores where THROUGHPUT binds:
#: 8 cores run as 8 separate 1-thread trainings deliver ~8x13.0 = 104 steps/s AGGREGATE, versus
#: ~35 steps/s for one 8-thread training. So 1 thread is ~3x better for the TEST FLOOD, while
#: 8 threads is ~2.7x better for a SEQUENTIAL CHAIN, where aggregate throughput is worthless.
#: NOTE: 2026-07-30 (record §39) / applied 2026-08-01 (RUN 10), deferred item 9: THE ROW ABOVE IS AN
#: ISOLATED BENCH (one 8-core box, 21.5 -> 60.0 steps/s, nothing else running). In PRODUCTION,
#: across 740 timed trainings on shared 36-core nodes -- 60 packed 1-thread baseline tasks at a
#: p50 of 8.33 h against 680 8-thread search records at a p50 of 4.34 h, on the SAME 400,000-step
#: training -- 8 threads is **1.92x by median / 1.75x by mean**, not 2.72x. Co-tenants take memory
#: bandwidth an idle bench never loses. The packed lane's own four-way sharing biases the 1-thread
#: figure UP, so 1.92 is an UPPER bound.
#:
#: SCOPE: this is a MODEL INPUT, not a behaviour change -- nothing the campaign computes moves,
#: only what we PREDICT about it. At 2.72x the critical-chain floor reports 3.27 d; at the measured
#: 1.92x it is 4.64 d, so the optimistic value understated the front of the ladder by ~1.4 days.
#: Any prose quoting "2.72x" or a "3.27 d critical-chain floor" is wrong and must be corrected
#: before it reaches the PDF. Bench values kept for provenance: {1: 1.00, 2: 1.57, 4: 2.23,
#: 8: 2.72, 16: 2.11}.
CPU_THREAD_SPEEDUP = {1: 1.00, 2: 1.45, 4: 1.75, 8: 1.92, 16: 1.55}   # field-measured at 8

#: The optimum, and the only value worth using for a chain. NEVER exceed it -- 16 threads is
#: measurably SLOWER than 8.
CPU_CHAIN_THREADS = 8

#: Intel-Xeon CPU pools eligible for the confirmatory lane (microarchitecture-homogeneous).
CONFIRMATORY_CPU_POOLS = ("d", "b")

#: Excluded, with the reason. Encoded so nobody re-adds them from a free-capacity table.
_GPU_NODE_REASON = ("GPU node: harvesting its idle CPU cores blocks GPU jobs, which request "
                    "-pe smp 4 alongside gpu=1. The one case where our campaign would genuinely "
                    "impair other users, so it is refused by construction.")
EXCLUDED_CPU_POOLS = {
    "t": "AMD EPYC (Zen4) vs Intel Xeon elsewhere -> different oneDNN kernels -> different float "
         "reduction order -> breaks CRN bit-exactness. Measured no faster anyway (14.75 vs 14.2).",
    "e": _GPU_NODE_REASON,
    "f": _GPU_NODE_REASON,
    "l": _GPU_NODE_REASON,
    "u": _GPU_NODE_REASON,
    "v": _GPU_NODE_REASON,
}

#: Structure of the two sequential chains (``config/preregistration.yaml`` + ``src/search/bayes_opt.py``).
# NOTE: 25 is TRUE ONLY BECAUSE THE INIT IS NOW BATCHED (fixed 2026-07-26). `bayes_opt_over_template`
# ran `for x in init_points: _evaluate(x, "init")` — a SEQUENTIAL loop — and the cluster driver
# turns every eval into its own array-of-1 job, so the "5 parallel" init this project documented
# everywhere was really 5 more SERIAL queue-and-train steps: the chain was 30, not 25 (8.9 d ->
# 10.7 d at 1 thread). `campaign.run_family_search_arm` now passes `batch_eval_fn` for bayes_opt
# (as well as tpe), dispatching the i.i.d.-uniform init as ONE array. Identity proven in
# tests/test_dfo_tpe_batch.py; the GUIDED phase stays strictly serial by algorithmic necessity.
_BAYES_SERIAL_STEPS = 25          # 30 budget - 5 BATCHED init; the GP-EI loop is strictly serial
_LLM_CHAIN_GENERATIONS = 6        # K=5 candidates/generation x 6 = the 30-candidate budget

#: The H4 DFO comparators (``src/search/dfo_toolkit.py``, added 2026-07-26 by the FEATURE/BUILD
#: lane). Counted ONLY when ``include_dfo=True``, because as of this writing they are REPORT-ONLY
#: (prunable ⇒ not on the confirmatory critical path). **If TPE is ever seated confirmatorily it
#: becomes the LONGEST chain in the campaign** — longer than GP-EI — so the option is modelled
#: explicitly rather than left as an assumption:
#:
#: * **TPE = 30 serial by default, ~21 when batched.** ``tpe_over_template`` drives Optuna with
#:   ``study.optimize(...)``, which evaluates ONE trial at a time, so the full budget was
#:   sequential. **FIXED 2026-07-26 (T5-a):** the function now accepts ``batch_eval_fn`` and, when
#:   given, dispatches the ``n_startup_trials`` (default ``min(10, budget)``) as ONE batch — those
#:   points come from Optuna's RANDOM sampler and depend on no observed value, so the results are
#:   IDENTICAL (proven by ``tests/test_dfo_tpe_batch.py``, which asserts the same points, order,
#:   scores and winner as the sequential path). Pure dispatch; matched budget untouched.
#:   **WIRED 2026-07-26** — ``campaign.run_family_search_arm`` now passes ``batch_eval_fn`` for
#:   the ``tpe`` arm, dispatching the startup trials as ONE cluster array. So the SERIAL chain is
#:   ``budget - n_startup`` = **30 - 10 = 20**, and the value below is 20 (the measured reality,
#:   not an aspiration). This also restores ``bayes_opt`` (25) as the longest DFO chain.
#:   NB the LAPTOP path (``orchestration/parallel.py``) stays sequential — harmless, because
#:   batching does not change results, so LAPTOP<->CLUSTER parity is preserved either way.
#: * **CMA-ES ≈ 4 serial generations.** ``es.ask()`` proposes a whole population per generation
#:   (parallel within a generation), so at budget 30 with the default popsize ~9 it is ~4 steps —
#:   never the binding chain.
_TPE_SERIAL_STEPS = 20   # budget 30 - n_startup 10, now BATCHED on the cluster path
_CMA_SERIAL_GENERATIONS = 4

#: The CRITICAL PATH, per arm: how many trainings that arm must run STRICTLY IN SEQUENCE. The
#: makespan is ``max(total_work / capacity, longest chain)``, so these floor the campaign no matter
#: how many cores Myriad grants — which is exactly why the live monitor watches them (a stalled
#: chain slips the finish date while the parallel test flood keeps every other indicator green).
#: Derived from the constants above so there is ONE source of truth for the chain lengths.
SERIAL_CHAIN_STEPS: dict[str, int] = {
    "bayes_opt": _BAYES_SERIAL_STEPS,
    "tpe": _TPE_SERIAL_STEPS,
    "cma_es": _CMA_SERIAL_GENERATIONS,
}

#: What the SENTINEL must compare its record count against — and it is NOT the constant above.
#:
#: NOTE: FOUND 2026-08-02 (D27, record §101.2). ``SERIAL_CHAIN_STEPS`` counts DISPATCH steps, which is the
#: right quantity for the makespan model in ``_critical_chain_days``. ``scripts/sentinel.py`` counts
#: RECORDS and compared them against it, so ``campaign_health.check_chain_progress`` — whose whole
#: purpose is that "a stalled chain is the campaign's worst silent failure" — read **"cma_es 9/4"** and
#: classified the campaign's longest chain as COMPLETE. The same unit mismatch under-counts the other
#: two (tpe would be "done" at 20 of 30 records, bayes_opt at 25 of 30), it just never bit as hard.
#:
#: Both numbers are correct for their own purpose; using one for the other is the defect. Every DFO arm
#: spends the SAME matched candidate budget — that is the point of the H4 comparison — so the record
#: count to expect is ``config/campaign.yaml: candidates_per_arm``, which the freeze binds at 30.
_MATCHED_CANDIDATE_BUDGET = 30
SERIAL_CHAIN_BUDGET: dict[str, int] = {a: _MATCHED_CANDIDATE_BUDGET for a in SERIAL_CHAIN_STEPS}
# NOTE: UPDATED 2026-07-26 (late): the registered roster grew 7 -> 9 arms the same day
# (`config/preregistration.yaml: arms` gained `cma_es` + `tpe` as the H4 optimiser portfolio,
# N4 CONFIRMATORY — not report-only). The DFO arms are CORE-only: the 10 replication legs still
# run 5 LLM arms each, so the leg total is unchanged at 50.
_SEARCH_TRAININGS = 1_800         # 9x30 core + 30 H3 + 10 legs x 5 arms x 30
_TEST_UNITS_PER_RUNG = 71         # 9 core arms + 50 leg arms + 11 H1 canon + 1 H3


def training_core_hours(train_steps: int = 400_000,
                        steps_per_s: float = CPU_STEPS_PER_S_PER_CORE) -> float:
    """Core-hours for ONE training at ``steps_per_s``. 400k @ 13.0 -> 8.547 core-hours."""
    if steps_per_s <= 0:
        raise ValueError(f"steps_per_s must be > 0, got {steps_per_s}")
    return train_steps / steps_per_s / 3600.0


def total_trainings(rung: int) -> int:
    """Total trainings to complete the campaign at seed rung ``n``: ``1,800 + 71n``.

    NOTE: The formula in this docstring read ``1,740 + 69n`` until 2026-07-26 — STALE since the DFO arms
    landed, and wrong by 1,196 trainings at n=568 (40,932 vs the true 42,128, which is the value the
    registered ladder actually carries). Not cosmetic: this is the function the whole campaign is
    planned from, and the ``69`` hid the fact that the **11-member H1 canon sits in the per-rung
    denominator** — precisely the question amendment R111 had to answer. The CONSTANTS were correct
    throughout; only the prose drifted.
    """
    if rung < 0:
        raise ValueError(f"rung must be >= 0, got {rung}")
    return _SEARCH_TRAININGS + _TEST_UNITS_PER_RUNG * int(rung)


def cpu_saturation_cores(rung: int = 568, *, bayes_on_gpu: bool = False,
                         train_steps: int = 400_000, chain_threads: int = 1,
                         include_dfo: bool = True) -> float:
    """CPU cores beyond which MORE CORES BUY NOTHING (the throughput/critical-path crossover).

    NOTE: **READ THE THREAD REGIME BEFORE QUOTING A NUMBER — the default is NOT the campaign's.**
    ``chain_threads`` DEFAULTS TO 1, which returns the pre-R107 crossover **~1,685**. The RATIFIED
    campaign runs its chains at :data:`CPU_CHAIN_THREADS` = 8 (R107), where the crossover is
    **~3,235 at rung 568** (and ~2,336 at the registered primary target of 403) — the figure this
    docstring quotes below and the one the GO-day advisor reports
    (``allocation.advise_cpu_lane`` passes the value explicitly, as does :func:`plan_lanes`, so the
    operational path is correct). The default is retained at 1 only so the two regimes stay
    comparable; **a bare ``cpu_saturation_cores()`` models the SUPERSEDED design.** Pass
    ``chain_threads=CPU_CHAIN_THREADS`` for the campaign as ratified. (Flagged 2026-07-27 after a
    bare call returned 1,685 and appeared to contradict this very docstring.)

    NOTE: THIS IS A PROPERTY OF THE CURVE, **NOT A CLAIM THAT SUCH CAPACITY IS AVAILABLE.** It answers
    "above what point does adding cores stop helping?", nothing else. Do not quote it beside
    measured or projected capacity without saying which is which. The ladder of what is actually
    known (2026-07-26): **636 cores MEASURED** (peak, reproducible over three ramps);
    ~2,000-3,000 PROJECTED from the flow model and still UNVERIFIED over multiple hours; and a
    ~3,576 available under the policy at the best free capacity we happened to SAMPLE (4,576 free
    minus ``killswitch.FREE_CORE_RESERVE``). **That sample is not a ceiling.** SGE's
    ``maxujobs = 1000`` at 8 cores/job structurally permits ~**8,000** cores, and d+b hold 11,160,
    so saturation is well inside what the scheduler allows — it is UNMEASURED, not unattainable.
    PUSH FOR IT: every core up to ~3,235 shortens the campaign, and ``plan_footprint`` scales with
    whatever is free, so a generous cluster is exploited automatically. Past ~3,235 the makespan
    genuinely stops improving (the 4.64 d chain floor) — that part is real.

    **THE "WE NEED 4,000+ CORES" TARGET WAS ITSELF AN ARTEFACT (RUN 10, 2026-08-01).** 4,584 and
    the 3.27 d floor were both computed from the ISOLATED-BENCH 2.72x thread speed-up. On the
    field-measured 1.92x (740 timed trainings, record §39) the crossover is **3,235 at rung 568 and
    2,336 at rung 403**. So the standing ambition to reach ~4,000 cores was chasing a number that
    does not exist: past ~3,235 the CRITICAL CHAIN binds and extra cores buy NOTHING. This makes
    the honest capacity story better, not worse — the target is ~30 % closer than anyone thought.

    This is the number that makes the whole plan legible: below it, buy cores; above it, buy
    LATENCY on the binding chain instead. With ``bayes_opt`` on CPU the crossover is ~1,640 cores;
    moving those 30 trainings to one GPU pushes it out to ~6,850 (where the LLM chains bind).
    """
    work = total_trainings(rung) * training_core_hours(train_steps)
    chain_days = _critical_chain_days(bayes_on_gpu=bayes_on_gpu, train_steps=train_steps,
                                      chain_threads=chain_threads, include_dfo=include_dfo)
    return work / (chain_days * 24.0)


def _critical_chain_days(*, bayes_on_gpu: bool, train_steps: int,
                         chain_threads: int = 1, include_dfo: bool = True) -> float:
    """Longest sequential chain, in days.

    ``chain_threads`` applies the MEASURED thread speed-up to the CHAIN ONLY — legitimate because
    a chain's cost is pure latency, so the aggregate-throughput argument that pins the test flood
    to 1 thread simply does not apply to it. The chains run CONCURRENTLY with one another (and with
    the TPE/CMA-ES DFO comparators), so the critical path is the MAX over them, never the sum.
    """
    speedup = CPU_THREAD_SPEEDUP.get(int(chain_threads), 1.0)
    cpu_h = training_core_hours(train_steps) / speedup
    gpu_h = training_core_hours(train_steps, GPU_PACK1_STEPS_PER_S)
    chains = [
        _BAYES_SERIAL_STEPS * (gpu_h if bayes_on_gpu else cpu_h),
        _LLM_CHAIN_GENERATIONS * cpu_h,
    ]
    if include_dfo:
        chains += [_TPE_SERIAL_STEPS * cpu_h, _CMA_SERIAL_GENERATIONS * cpu_h]
    return max(chains) / 24.0


@dataclass(frozen=True)
class LanePlan:
    """The plan, its predicted makespan, and — crucially — WHAT IS BINDING."""

    rung: int
    cpu_cores: int
    bayes_on_gpu: bool
    throughput_days: float
    critical_chain_days: float
    makespan_days: float
    binding: str                       # "throughput" | "critical_chain"
    saturation_cores: float
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"LANE PLAN  rung={self.rung}  cpu_cores={self.cpu_cores}  "
            f"bayes_on_gpu={self.bayes_on_gpu}",
            f"  throughput      : {self.throughput_days:6.2f} d",
            f"  critical chain  : {self.critical_chain_days:6.2f} d",
            f"  => MAKESPAN     : {self.makespan_days:6.2f} d   (BINDING: {self.binding})",
            f"  cores beyond which more buy nothing: {self.saturation_cores:.0f}",
        ]
        lines.extend(f"  * {n}" for n in self.notes)
        return "\n".join(lines)


def plan_lanes(*, rung: int = 568, cpu_cores: int, bayes_on_gpu: bool = False,
               train_steps: int = 400_000, chain_threads: int = 1,
               include_dfo: bool = True) -> LanePlan:
    """Compute the makespan and say plainly which lever is worth pulling next.

    The ``notes`` are the actionable part: they name the ONE change that would most reduce the
    makespan given where the plan currently sits, so the operator is never guessing between "more
    cores" and "get a GPU".
    """
    if cpu_cores <= 0:
        raise ValueError(f"cpu_cores must be > 0, got {cpu_cores}")
    work = total_trainings(rung) * training_core_hours(train_steps)
    throughput = work / (cpu_cores * 24.0)
    chain = _critical_chain_days(bayes_on_gpu=bayes_on_gpu, train_steps=train_steps,
                                 chain_threads=chain_threads, include_dfo=include_dfo)
    binding = "throughput" if throughput >= chain else "critical_chain"
    sat = cpu_saturation_cores(rung, bayes_on_gpu=bayes_on_gpu, train_steps=train_steps,
                               chain_threads=chain_threads, include_dfo=include_dfo)

    notes: list[str] = []
    if binding == "critical_chain" and not bayes_on_gpu:
        notes.append(
            f"MORE CPU CORES BUY NOTHING (saturated at ~{sat:.0f}). The binding constraint is the "
            f"bayes_opt GP chain: {_BAYES_SERIAL_STEPS} STRICTLY SERIAL trainings. Move those 30 "
            "trainings to ONE GPU at pack=1 -> the chain drops from "
            f"{_BAYES_SERIAL_STEPS * training_core_hours(train_steps) / 24:.2f} d to "
            f"{_BAYES_SERIAL_STEPS * training_core_hours(train_steps, GPU_PACK1_STEPS_PER_S) / 24:.2f} d. "
            "Highest-leverage action in the whole campaign.")
    elif binding == "critical_chain" and bayes_on_gpu:
        notes.append(
            "Saturated with bayes already on GPU; the LLM reflection chains "
            f"({_LLM_CHAIN_GENERATIONS} sequential generations) now bind. Only worth attacking if "
            "many GPUs become available - otherwise accept it.")
    else:
        notes.append(
            f"Throughput-bound: more CPU cores DO still help, up to ~{sat:.0f} cores.")
    if not bayes_on_gpu:
        notes.append("bayes_opt on GPU is OPPORTUNISTIC, never a blocker: with no GPU the campaign "
                     "is slower, not stopped.")
    notes.append("CPU lane = pools " + "/".join(CONFIRMATORY_CPU_POOLS) +
                 " only; t excluded (AMD vs Intel -> CRN), GPU-node cores excluded (would block "
                 "GPU users).")
    return LanePlan(rung=rung, cpu_cores=int(cpu_cores), bayes_on_gpu=bayes_on_gpu,
                    throughput_days=throughput, critical_chain_days=chain,
                    makespan_days=max(throughput, chain), binding=binding,
                    saturation_cores=sat, notes=notes)
