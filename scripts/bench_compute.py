"""Calibrate MAX training throughput across GPU+CPU (heterogeneous concurrency).

Runs several short SB3-SAC trainings concurrently across a mix of GPU and CPU workers and
reports the AGGREGATE steps/s (sum of per-worker rates — each worker measures its own rate, so
the sum correctly accounts for GPU time-slicing / CPU oversubscription). The configuration with
the highest aggregate throughput is the one to use for the prototype run; its aggregate rate
directly gives the wall-clock for the full campaign (total_steps / aggregate_steps_s).

Usage:
    python scripts/bench_compute.py                 # default sweep, 2500 steps/worker
    python scripts/bench_compute.py --steps 4000    # longer (more steady-state, less spawn noise)
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time


def _worker(device: str, steps: int, idx: int, threads: int, n_assets: int, batch: int, q) -> None:
    """Train one SAC for ``steps`` on ``device``; report its own steps/s on ``q``.

    ``n_assets`` sets the synthetic panel width so the observation dim (~2*n_assets) MATCHES the campaign's
    real 953-asset / 1893-dim obs — per-worker VRAM/compute (hence the concurrency ceiling) tracks the obs
    SHAPE, not the data values. ``batch`` mirrors the campaign batch_size and the replay buffer uses the
    campaign 50k cap (ADR-042), so this benchmark reflects the ACTUAL campaign memory/compute, not a toy.
    """
    try:
        # Limit CPU threads per worker so concurrent CPU jobs don't oversubscribe the 16 threads.
        os.environ["OMP_NUM_THREADS"] = str(threads)
        import torch

        torch.set_num_threads(max(1, threads))
        from src.agents.factory import campaign_replay_cap, make_headline_agent
        from src.data.synthetic import make_synthetic_panel
        from src.env.portfolio_env import PortfolioEnv
        from src.utils.config import load_config

        panel = make_synthetic_panel(n_assets=n_assets, n_days=600, seed=idx)
        cfg = load_config("environment")

        def reward(w, r, pw, pr, info):  # noqa: ANN001
            return float(pr), {"pr": float(pr)}, None

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:  # noqa: BLE001
            pass
        env = PortfolioEnv(panel, cfg, reward)
        agent = make_headline_agent(
            env,
            {
                # FULL 50k campaign buffer (ADR-042): SB3 sizes the replay array at construction, so allocate
                # the REAL per-worker RAM up front even though this bench trains only a few steps — the
                # concurrency ceiling then reflects the campaign's actual memory footprint, not the tiny run.
                "buffer_size": campaign_replay_cap(),
                "batch_size": batch,
                "device": device,
                "seed": idx,
                "verbose": 0,
                # MEASURE STEADY-STATE THROUGHPUT (deep review 2026-07-26). Without this the factory
                # default `learning_starts=1000` applied, so 1000 of the default 2500 steps were SAC
                # WARMUP — random-action rollout with NO gradient update. Measured on this machine:
                # warmup runs at ~5407 steps/s vs ~30 steps/s once training starts (~181x cheaper), so
                # the reported rate was ~49.7 steps/s against a true steady state of ~29.9 — a 66%
                # OVERSTATEMENT, which understated the `proj days` wall-clock projection below by ~40%.
                # The projection spans 400k-step trainings where the 1000-step warmup is ~0.25% of the
                # work, so steady state IS the right basis; training from step 1 makes the measured
                # rate equal it. (Sampling a full batch from a near-empty buffer is fine — SB3 samples
                # with replacement, and gradient-step cost is independent of buffer contents.)
                "learning_starts": 0,
            },
        )
        t0 = time.perf_counter()
        agent.learn(total_timesteps=steps)
        dt = time.perf_counter() - t0
        q.put((device, steps / dt if dt > 0 else 0.0))
    except Exception as exc:  # noqa: BLE001
        q.put((device, -1.0))
        import sys

        print(f"[worker {idx} {device}] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)


def bench(n_gpu: int, n_cpu: int, steps: int, n_assets: int = 30, batch: int = 256) -> dict:
    """Run ``n_gpu`` cuda + ``n_cpu`` cpu workers concurrently; return aggregate throughput.

    ``n_assets``/``batch`` default to the CAMPAIGN profile so the measured concurrency ceiling reflects
    the real run, not a toy panel. **The campaign trades a PIT top-30 universe** (``load_gold_panel
    phase="development"`` -> N=30), so obs = 63*N+3 = **1893-dim** and the 50k buffer is ~0.71 GB.
    DEFECT FIXED 2026-07-25: the previous default was ``n_assets=953`` with a comment claiming
    "953-asset -> 1893-dim" — but 953 assets give a **60,042-dim** obs (~24 GB buffer), a 30x-HEAVIER
    toy that is NOT the campaign. Running this tool with the old default silently measured that toy.
    Use ``--n-assets 30`` for the campaign scale (now the default); pass 953 only to probe the heavy case.
    """
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    # Per the compute research: 1 thread per worker (avoid oversubscribing the 16 hardware threads;
    # SB3 SAC's small matmuls don't parallelize well, so many 1-thread workers beat few fat ones).
    specs = [("cuda", i, 1) for i in range(n_gpu)] + [("cpu", n_gpu + i, 1) for i in range(n_cpu)]
    procs = [
        ctx.Process(target=_worker, args=(dev, steps, idx, th, n_assets, batch, q))
        for dev, idx, th in specs
    ]
    t0 = time.perf_counter()
    for p in procs:
        p.start()
    rates = [q.get() for _ in procs]
    for p in procs:
        p.join()
    wall = time.perf_counter() - t0
    gpu = [r for d, r in rates if d == "cuda" and r > 0]
    cpu = [r for d, r in rates if d == "cpu" and r > 0]
    return {
        "n_gpu": n_gpu,
        "n_cpu": n_cpu,
        "agg": sum(r for _, r in rates if r > 0),
        "wall": wall,
        "gpu_each": gpu,
        "cpu_each": cpu,
        "n_failed": sum(1 for _, r in rates if r <= 0),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Calibrate campaign training throughput (GPU concurrency).")
    p.add_argument("--steps", type=int, default=2500, help="Steps per worker (default 2500).")
    p.add_argument(
        "--configs",
        default="2x0,3x0,4x0,5x0,6x0",
        help="Comma list of '<gpu>x<cpu>' worker configs. Default = GPU-only 2..6: the campaign uses ONE "
             "GPU, and CPU workers would train on a DIFFERENT device => a hardware confound across arms.",
    )
    p.add_argument("--n-assets", type=int, default=30,
                   help="Synthetic panel width (default 30 => 1893-dim obs = the campaign PIT top-30 "
                        "profile). NOTE 953 => 60,042-dim (~24 GB buffer) = a 30x-heavier toy, NOT the "
                        "campaign — the old default; fixed 2026-07-25.")
    p.add_argument("--batch", type=int, default=256, help="batch_size (default 256, the campaign value).")
    p.add_argument("--total-steps", type=int, default=None,
                   help="Total campaign steps for the wall-clock projection (e.g. 600 * B*).")
    args = p.parse_args()
    configs = [tuple(int(x) for x in c.split("x")) for c in args.configs.split(",")]
    total = int(args.total_steps) if args.total_steps else 600 * 200_000  # 600 trainings x 200k default

    print(f"Calibrating ({args.steps} steps/worker; n_assets={args.n_assets}, batch={args.batch}). "
          f"Projection total = {total:,} steps.\n")
    print(f"{'gpu':>3} {'cpu':>3} | {'agg steps/s':>11} | {'proj days':>9} | gpu_each / cpu_each")
    print("-" * 78)
    best = None
    for g, c in configs:
        r = bench(g, c, args.steps, args.n_assets, args.batch)
        agg = r["agg"]
        proj_d = (total / agg / 86400) if agg > 0 else float("inf")
        ge = ",".join(f"{x:.0f}" for x in r["gpu_each"]) or "-"
        ce = ",".join(f"{x:.0f}" for x in r["cpu_each"]) or "-"
        fail = f"  FAILED={r['n_failed']}" if r["n_failed"] else ""
        print(f"{g:>3} {c:>3} | {agg:>11.0f} | {proj_d:>9.1f} | [{ge}] / [{ce}]{fail}")
        if r["n_failed"] == 0 and (best is None or agg > best[2]):
            best = (g, c, agg)
    if best:
        g, c, agg = best
        print("-" * 78)
        print(f"BEST (no failures): gpu={g} cpu={c} -> {agg:.0f} aggregate steps/s "
              f"=> projected campaign ~ {total / agg / 86400:.1f} days for {total:,} steps. "
              f"Pick for SUSTAINED stability (RAM/VRAM/thermal), NOT just peak agg.")


if __name__ == "__main__":
    main()
