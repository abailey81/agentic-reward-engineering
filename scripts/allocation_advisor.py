"""The adaptive allocation advisor CLI — one command = the current optimal launch settings.

Usage (GO morning, or any time during the campaign):
    python scripts/allocation_advisor.py                       # live snapshot -> the plan
    python scripts/allocation_advisor.py --probe-age-hours 30  # age the U/V probe verdict
    python scripts/allocation_advisor.py --vram-per-training 2.8 --rate 700 \
        --remaining "tier403=6800,legs=3000"                   # canary-measured facts -> ETAs

Prints the PLAN (regime, chunk-tasks, search-lane pool, the exact --seed-pool-blocks string,
pack depths, ETAs) and appends the raw snapshot to outputs/myriad_telemetry.jsonl. Advisory
only: nothing is submitted, nothing is altered, priorities are never touched.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:  # Windows consoles default to legacy codepages — never let a glyph kill a live cycle
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[union-attr]
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cluster.allocation import advise, plan_delta  # noqa: E402
from src.cluster.telemetry import (append_log, collect, contention_trend,  # noqa: E402
                                   load_state, measure_rate, observed_gpus, save_state)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="myriad")
    ap.add_argument("--probe-age-hours", type=float, default=0.0,
                    help="Hours since the U/V probes were submitted (drives the 48h verdict).")
    ap.add_argument("--vram-per-training", type=float, default=None,
                    help="Canary-measured GB per training; unlocks pack>5 recommendations.")
    ap.add_argument("--rate", type=float, default=None,
                    help="Measured trainings/day from the live archive; unlocks ETAs.")
    ap.add_argument("--remaining", default=None,
                    help='Milestones as "name=count,...", e.g. "tier403=6800,legs=3000".')
    ap.add_argument("--archive-root", default=None,
                    help="Live archive root: AUTO-measures the rate + the granted GPU models "
                         "(supersedes --rate when records exist).")
    ap.add_argument("--prev-regime", choices=["CONTENDED", "QUIET"], default=None,
                    help="Override the hysteresis anchor (default: the persisted state file).")
    ap.add_argument("--watch", type=int, default=None, metavar="SECS",
                    help="LIVE MODE: re-snapshot + re-advise every SECS seconds forever; prints "
                         "a heartbeat each cycle and LOUD alerts only on actionable changes "
                         "(regime flip, U/V unlock, pool changes). Ctrl+C to stop.")
    args = ap.parse_args(argv)

    remaining = None
    if args.remaining:
        remaining = {}
        for part in args.remaining.split(","):
            k, v = part.split("=", 1)
            remaining[k.strip()] = int(v)

    import time as _time
    watch_started = _time.time()

    def one_cycle(prev_regime: str | None, old_fields: dict | None, *,
                  always_render: bool) -> dict:
        """Snapshot -> advise -> persist -> print (heartbeat always; plan per render policy)."""
        # save_state REPLACES the file, so anything else the state carries (the lane facts the
        # sentinel reads) must be carried forward explicitly or a single advisor cycle would erase
        # it — silently switching the capacity check back off mid-campaign.
        lane_state = {k: v for k, v in (load_state() or {}).items()
                      if k not in ("prev_regime", "last_plan")}
        # Audit m10b: in watch mode the probe age ADVANCES with wall time, so the 48h
        # RESTRICTED transition can actually fire mid-watch.
        probe_age = args.probe_age_hours + (_time.time() - watch_started) / 3600.0
        snap = collect(args.host, probe_age_hours=probe_age)
        # NEVER log a frame whose PROBE FAILED (2026-07-28). `collect` captures transport errors and
        # never raises, so a timed-out ssh returns a Snapshot with EMPTY sections — which appends as
        # "0 jobs, 0 slots" and reads on the capacity curve as a total capacity COLLAPSE. Observed
        # live at 2026-07-28T10:25Z during a transport wobble, sitting between frames of ~1,400
        # slots. `accumulation_report` already filters to slot-bearing frames, so the forecast was
        # safe — but a human reading the curve sees the campaign lose the cluster. Absence is honest;
        # a false zero is not. The failure is still surfaced on stdout, so it is never silent.
        if snap.errors and not snap.our_jobs:
            print(f"[telemetry] probe FAILED ({snap.errors[0][:120]}) — frame NOT logged; a false "
                  "zero would read as a capacity collapse")
        else:
            append_log(snap)
        rate = args.rate
        pool_vram = None
        if args.archive_root:
            measured, n = measure_rate(args.archive_root)
            gpus = observed_gpus(args.archive_root)
            if n:
                rate = measured if measured > 0 else None
                stale_note = "" if measured > 0 else " (STALE >3h since the last record — rate withheld)"
                print(f"[self-measured] {n} records{stale_note}; granted GPUs so far: "
                      f"{gpus or 'none archived'}")
            # Audit m8: feed the empirically observed EF card back into the pack ceiling.
            if any(k.startswith("V100-32") for k in gpus):
                pool_vram = {"EF": 32, "L": 40, "U": 80, "V": 80}
        plan = advise(snap,
                      measured_vram_per_training_gb=args.vram_per_training,
                      remaining_trainings=remaining,
                      measured_trainings_per_day=rate,
                      prev_regime=prev_regime,
                      pool_vram=pool_vram)
        fields = {"regime": plan.regime, "chunk_tasks": plan.chunk_tasks,
                  "search_pool": plan.search_pool, "stripe_pools": plan.stripe_pools,
                  "seed_pool_blocks": plan.seed_pool_blocks, "pack": plan.pack, "ts": snap.ts}
        alerts = plan_delta(old_fields, fields)
        print(f"[telemetry {snap.ts}] pools free: {snap.pool_free} | cluster qw: "
              f"{snap.cluster_qw} ({snap.cluster_users} users) | our jobs: "
              f"{len(snap.our_jobs)} | trend: {contention_trend()}")
        for a in alerts:
            print(f"[ALERT] {a}")
        # Audit M2: one-shot ALWAYS renders the full plan (the runbook's daily-ETA run was
        # printing only a heartbeat); watch mode stays change-gated to keep alerts meaningful.
        if always_render or alerts or old_fields is None:
            print(plan.render())
            # CPU LANE (2026-07-26). The campaign's throughput lane is CPU, not GPU: 636 cores were
            # measured concurrently vs ZERO GPUs granted in three days. Printed alongside the GPU
            # plan so GO day READS the target instead of a human re-deriving it from the dossier.
            from src.cluster.allocation import advise_cpu_lane

            cpu = advise_cpu_lane(snap)
            # Hand the CPU target to the sentinel's capacity check (2026-07-26). It is the ONE lane
            # input that cannot be derived from the archive or the pre-registration — it is the
            # forecast we are choosing to hold ourselves to — and recording it HERE means it is
            # captured by the tool that already computes it, with no GO-day step to forget. Without
            # it the capacity check can only report the measurement, never judge it.
            if cpu.get("target_cores") is not None:
                lane_state["lane_expected_cores"] = int(cpu["target_cores"])
                lane_state["lane_forecast_utc"] = _time.time()
            if cpu.get("target_cores") is None:
                print(f"CPU LANE: {cpu['why']} — fall back to runbook §11.3 (manual sizing)")
            else:
                print(
                    f"CPU LANE: hold ~{cpu['target_cores']} cores of {cpu['free_cores']} free "
                    f"{cpu['free_by_pool']} | {cpu['job_shape']}\n"
                    f"  threads: flood={cpu['flood_threads']} chain={cpu['chain_threads']} "
                    f"(threads where LATENCY binds, cores where THROUGHPUT binds)\n"
                    f"  makespan ~{cpu['makespan_days']} d at rung 568 (BINDING: {cpu['binding']}; "
                    f"more cores stop helping past ~{cpu['saturation_cores']})\n"
                    f"  why: {cpu['why']}")
        save_state({**lane_state, "prev_regime": plan.regime, "last_plan": fields})
        return fields

    state = load_state()
    prev = args.prev_regime or state.get("prev_regime")
    fields = state.get("last_plan")

    if args.watch is None:
        one_cycle(prev, fields if fields else None, always_render=True)
        return 0

    # LIVE MODE: forever, degrade-never-crash (a bad cycle logs and waits for the next).
    interval = max(60, args.watch)  # audit m10c: the 60s floor is now STATED, not silent
    print(f"[watch] live every {interval}s (min 60) — Ctrl+C to stop; "
          "alerts on actionable change only")
    while True:
        try:
            fields = one_cycle(prev, fields, always_render=False)
            prev = fields["regime"]
            _time.sleep(interval)
        except KeyboardInterrupt:  # audit m10c: clean exit even mid-sleep
            print("[watch] stopped")
            return 0
        except Exception as exc:  # noqa: BLE001 — the watcher outlives any single bad cycle
            print(f"[watch] cycle failed ({type(exc).__name__}: {exc}) — retrying next tick")
            _time.sleep(interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
