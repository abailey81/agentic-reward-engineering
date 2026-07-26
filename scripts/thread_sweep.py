"""Measure SINGLE-TRAINING latency vs CPU thread count (the bayes_opt chain question).

Every campaign training is pinned to 1 thread — correct for the TEST FLOOD, where many 1-thread
trainings maximise AGGREGATE throughput. But the bayes_opt GP chain is 25 STRICTLY SEQUENTIAL
trainings, so its only cost is PER-TRAINING LATENCY. This sweeps torch threads for one training at
the exact campaign profile (n_assets=30 -> 1893-dim obs, batch 256, 50k replay cap) and reports
steps/s, so the CPU-vs-GPU choice for the chain rests on a measurement, not an assumption.

Each config runs in a FRESH subprocess: torch's thread count and its intra-op pools cannot be
reliably re-tuned in-process after the first parallel region.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time


def _worker(threads: int, steps: int, n_assets: int, batch: int, q) -> None:
    try:
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)
        import torch

        torch.set_num_threads(max(1, threads))
        from src.agents.factory import campaign_replay_cap, make_headline_agent
        from src.data.synthetic import make_synthetic_panel
        from src.env.portfolio_env import PortfolioEnv
        from src.utils.config import load_config

        panel = make_synthetic_panel(n_assets=n_assets, n_days=600, seed=0)
        cfg = load_config("environment")

        def reward(w, r, pw, pr, info):  # noqa: ANN001
            return float(pr), {"pr": float(pr)}, None

        env = PortfolioEnv(panel, cfg, reward)
        agent = make_headline_agent(env, {
            "buffer_size": campaign_replay_cap(), "batch_size": batch,
            "device": "cpu", "seed": 0, "verbose": 0,
            # Steady-state throughput, same defect as bench_compute (deep review 2026-07-26): the
            # factory default `learning_starts=1000` made 1000 of the default 6000 steps SAC WARMUP
            # (random rollout, NO gradient update) — measured ~5407 steps/s vs ~30 steps/s training,
            # so the reported per-thread rate was inflated ~20%. A thread sweep compares BLAS thread
            # counts, and warmup barely touches BLAS, so the warmup fraction also diluted the very
            # signal being swept. Train from step 1 so every measured step is the work being timed.
            "learning_starts": 0,
        })
        t0 = time.perf_counter()
        agent.learn(total_timesteps=steps)
        dt = time.perf_counter() - t0
        q.put((threads, steps / dt if dt > 0 else 0.0, torch.get_num_threads()))
    except Exception as exc:  # noqa: BLE001
        q.put((threads, -1.0, str(exc)[:120]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--threads", default="1,2,4,8,16")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--n-assets", type=int, default=30)
    p.add_argument("--batch", type=int, default=256)
    a = p.parse_args()

    ctx = mp.get_context("spawn")
    print(f"single-training thread sweep (steps={a.steps}, n_assets={a.n_assets}, batch={a.batch})")
    print(f"{'threads':>8}{'steps/s':>10}{'speedup':>9}{'h per 400k':>12}{'chain 25x (d)':>15}")
    base = None
    for t in [int(x) for x in a.threads.split(",")]:
        q = ctx.Queue()
        proc = ctx.Process(target=_worker, args=(t, a.steps, a.n_assets, a.batch, q))
        proc.start()
        thr, rate, info = q.get()
        proc.join()
        if rate <= 0:
            print(f"{t:>8}   FAILED: {info}")
            continue
        base = base or rate
        hours = 400_000 / rate / 3600
        print(f"{t:>8}{rate:>10.1f}{rate / base:>9.2f}{hours:>12.2f}{25 * hours / 24:>15.2f}")


if __name__ == "__main__":
    main()
