"""Coverage tests for src/utils/monitoring.py RunMonitor — the headless (no-TTY) event/anomaly/JSONL logic.
The rich-Live UI (`_render`/`_start_live`) and NVML/GPU telemetry are `# pragma: no cover` (TTY/hardware-only);
everything here runs in-process with `_progress is None`, writing progress.json + the structured log.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import monitoring as mon  # noqa: E402


def test_run_monitor_full_lifecycle_and_anomalies(tmp_path: Path) -> None:
    m = mon.RunMonitor(
        tmp_path, title="cov", total_arms=2, candidates_per_arm=3, train_steps=100, model="stub"
    )
    m.arm_start("distributional", 0)
    m.candidate_start("distributional", 0, gen=0)
    m.llm_call("distributional", 0, secs=1.2, in_tok=10, out_tok=20)
    m.sandbox_result("distributional", 0, ok=True)
    m.sandbox_result("distributional", 0, ok=False, reason="gate-reject")
    # A normal training snapshot, then one that trips every anomaly guard.
    m.train_metrics(50, {"time/fps": 120.0, "train/critic_loss": 1.0, "train/ent_coef": 0.5,
                         "train/actor_loss": 0.1, "rollout/ep_rew_mean": 0.0})
    m.train_metrics(60, {"time/fps": 0.001, "train/critic_loss": 1e12, "train/ent_coef": 1e-9,
                         "train/actor_loss": float("nan"), "rollout/ep_rew_mean": 0.0})
    m.candidate_done("distributional", 0, fitness=0.5, status="ok", secs=2.0)
    m.candidate_done("distributional", 1, fitness=float("nan"), status="ok", secs=2.0)  # fitness_nan
    m.arm_done("distributional", winner_fitness=0.5, secs=10.0)
    m.anomaly("custom_kind", "a custom anomaly", level=40, extra=1)
    m.close(status="done")

    state = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert state["arms"]["done"] == 1
    assert state["candidates"]["run_done"] == 2
    assert state["best_fitness"]["distributional"] == 0.5
    # critic_explosion + entropy_collapse + fps_collapse + nonfinite_metric + fitness_nan + custom_kind.
    assert state["anomalies"]["count"] >= 4


def test_fmt_eta_formats_and_handles_missing() -> None:
    m = mon.RunMonitor(_tmp(), title="x", total_arms=1, candidates_per_arm=1, train_steps=10)
    m.state["eta_s"] = None
    assert m._fmt_eta() == "?"
    m.state["eta_s"] = 3661
    assert m._fmt_eta() == "1h01m"


def test_resources_sample_returns_dict() -> None:
    out = mon._Resources().sample()
    assert isinstance(out, dict)  # psutil keys if present; empty/partial otherwise — never raises


def test_queue_sink_forwards_events() -> None:
    puts: list[tuple] = []

    class _Q:
        def put(self, item: object) -> None:
            puts.append(item)

    sink = mon.QueueSink(_Q(), cand_id="c0", arm="distributional")
    sink.candidate_start("distributional", 0)
    sink.train_metrics(10, {"train/critic_loss": 1.0})
    sink.candidate_done("distributional", 0, fitness=0.3, status="ok", secs=1.0)
    sink.anything_else_is_a_noop()  # __getattr__ path
    assert len(puts) >= 3


def _tmp() -> Path:
    import tempfile

    return Path(tempfile.mkdtemp())
