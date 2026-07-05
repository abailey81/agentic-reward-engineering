"""SCIENCE-NEUTRALITY proof for the parallel TEST leg (slow: real SAC training).

The load-bearing claim of the TEST-leg parallelization is that it changes NOTHING observable: the
parallel ``evaluate_winners_on_test_parallel`` must produce the SAME per-seed ``test_returns`` as the
serial ``evaluate_winner_on_test`` for the same frozen winner / seed / windows / agent config. This test
trains the fixed SAC for real on a synthetic panel, on CPU + single-threaded so the run is deterministic
(GPU/CUDA + multi-thread reductions are accepted as statistical-not-bitwise elsewhere), and asserts the
two paths' realized test paths are numerically equivalent to abs 1e-6 (the assert tolerance — float
reduction ORDER can differ by ~1 ULP between the serial and parallel drivers even on CPU, so the claim
is "numerically equivalent", not literally byte-identical; 2026-07-05 wording reconcile).

Marked ``slow`` — it trains two tiny agents and spawns a worker process; run it explicitly before the
campaign (``pytest -m slow -k equivalence``), not in the fast suite.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


_WINNER = {
    "candidate_id": "scalar-g0-c0",
    "generation": 0,
    "reward_source": (
        "def reward(weights, returns, prev_weights, port_ret, info):\n"
        "    pr = float(port_ret)\n"
        "    return pr, {'r': pr}, None\n"
    ),
    "reward_source_hash": "",
    "feedback_block": "",
    "metrics": {"val_fitness": 0.0},
}


@pytest.mark.slow
def test_parallel_test_leg_equals_serial_on_cpu(tmp_path) -> None:
    import torch

    import run_campaign as rc  # type: ignore[import-not-found]
    from src.data.synthetic import make_synthetic_panel
    from src.env.runner import make_env_builder
    from src.orchestration.test_leg import evaluate_winners_on_test_parallel
    from src.utils.config import load_config

    env_cfg = load_config("environment")
    lookback = int(env_cfg["state"]["lookback_days"])
    # The parallel worker reloads make_synthetic_panel(n_assets=30, n_days=600, seed=0); use the SAME
    # here so the serial path trains on an identical panel. Windows are explicit + FITTING (resolve_windows
    # would map the 2020-2026 calendar onto a 600-day panel and degenerate — that is a dry-run artefact).
    panel = make_synthetic_panel(n_assets=30, n_days=600, seed=0)
    train_w = (lookback, 250)
    val_w = (250 + lookback, 360)
    test_w = (360 + lookback, panel.T)
    agent_cfg = {
        "train_steps_per_candidate": 300, "buffer_size": 300, "batch_size": 64,
        "learning_rate": 3e-4, "device": "cpu", "normalize_obs": True, "tf32": True,
    }

    torch.set_num_threads(1)  # match the parallel worker's _worker_init pinning -> deterministic CPU run
    serial = rc.evaluate_winner_on_test(
        _WINNER, panel, env_cfg, [0], test_w,
        train_window=train_w, val_window=val_w, embargo=21,
        trainer=rc.make_agent_trainer_factory(), env_builder=make_env_builder,
        archive_root=str(tmp_path / "serial"), arm="scalar", agent_cfg=agent_cfg,
    )
    parallel = evaluate_winners_on_test_parallel(
        winners=[("scalar", _WINNER)], seeds=[0], panel_descriptor={"synthetic": True},
        env_cfg=env_cfg, agent_cfg=agent_cfg, train_window=train_w, val_window=val_w,
        test_window=test_w, embargo=21, lookback=lookback,
        n_gpu=0, n_cpu=1, recycle_every=4, test_root=tmp_path / "par",
    )

    assert parallel["n_written"] == 1 and parallel["n_failed"] == 0, parallel["failures"]
    s = serial[0]["metrics"]["test_returns"]
    p = parallel["written"][0]["metrics"]["test_returns"]
    assert len(p) == len(s) > 0
    # identical winner + seed + windows + panel + numerics (CPU, 1-thread) -> identical realized test path
    assert p == pytest.approx(s, abs=1e-6)
    # and the full record schema matches (run_id, frozen flag, the metric keys)
    assert parallel["written"][0]["run_id"] == serial[0]["run_id"] == "scalar-s0"
    assert parallel["written"][0]["frozen"] is True and serial[0]["frozen"] is True
