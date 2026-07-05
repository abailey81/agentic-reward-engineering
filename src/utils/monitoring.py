"""Precise, real-time run monitoring: multi-level progress, deep event logs, resource telemetry, anomalies.

One :class:`RunMonitor` owns a run's observability:

* **Multi-level rich progress** — Arms > Candidates > Training-steps — shown live when attached to a TTY,
  with steps/sec, ETA, and the live loss; a no-op display otherwise (so a backgrounded run is unaffected).
* **A disk state file** (``progress.json``, atomically rewritten every update) carrying the FULL snapshot —
  phase, arm/candidate/step counters, the latest training metrics, per-arm best fitness, resource usage,
  ETA, and the anomaly tally — so a background run is precisely inspectable at any instant (by a human, by
  ``scripts/monitor.py``, or by the agent) without attaching to the process.
* **Structured events** — every LLM call, sandbox verdict, candidate, arm, and anomaly is emitted via
  ``src.utils.logging.log_event`` to ``events.jsonl`` (machine-parseable) + ``run.log`` (human).
* **Resource telemetry** — GPU util/mem/temp (NVML), CPU%, RAM%, process RSS (psutil), sampled cheaply.
* **Anomaly detection** — NaN/inf losses or reward, critic-loss explosion, entropy-coef collapse/explosion,
  and FPS collapse are flagged the instant they appear, so anything even slightly off is caught.

:func:`make_training_callback` builds the SB3 hook that streams per-step training metrics into the monitor.
"""

from __future__ import annotations

import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

from src.utils.logging import CONSOLE, attach_run_logging, get_logger, log_event

_LOG = get_logger("monitor")

# Anomaly thresholds (RL-training early-warning signals; web-research-grounded 2026-06-20).
_CRITIC_EXPLOSION_ABS = 1e7  # absolute critic-loss ceiling (Q-divergence precursor)
_CRITIC_EXPLOSION_REL = 100.0  # critic loss > REL x its running median => explosion
_ENT_COEF_COLLAPSE = 1e-4  # auto entropy coef collapsing to ~0 (premature determinism)
_ENT_COEF_EXPLODE = 1e3
_FPS_COLLAPSE = 3.0  # steps/sec below this => training effectively stalled

# Anomaly kinds that describe a persistent STATE rather than a discrete EVENT (m9). Once SAC's
# auto-tuned entropy coef legitimately decays below the collapse floor (or a candidate genuinely
# stalls) the condition holds for the rest of the candidate, so the per-(kind, step) dedup — which
# only blocks IDENTICAL steps — lets it re-alarm every ``log_every`` steps. These fire AT MOST ONCE
# per (arm, candidate) instead (tracked in ``RunMonitor._fired_kinds``). Genuine EVENT kinds
# (nonfinite_metric, critic_explosion) stay re-fireable — each occurrence is fresh news.
_ONCE_PER_CANDIDATE_KINDS = frozenset({"entropy_collapse", "entropy_explosion", "fps_collapse"})

# Long-run laptop resource guards (campaign-robustness §E / ram_thermal §4). The resource sampler
# already records ``gpu_temp`` + ``ram_pct`` every flush but nothing thresholds them, so a 27 h run on
# the RTX-4050 laptop can silently thermal-throttle or creep into the OOM band. These fire WARN/ABORT
# anomalies through the SAME advisory ``anomaly()`` path the training checks use (non-halting by design;
# they surface the condition loudly in anomalies.jsonl + run.log so the operator can act / a sidecar can
# trip). Measured on the target GPU 2026-06-24: HW thermal slowdown begins ~91 C. RAM 92% is the band the
# prototype's n_gpu=4 transition-wave OOM actually fired in.
_GPU_TEMP_WARN_C = 87  # approaching the ~91 C HW slowdown -> visibly throttling before throughput collapses
_GPU_TEMP_CRIT_C = 91  # HW thermal slowdown / imminent throttle on this card
_RAM_PCT_WARN = 85.0   # RAM pressure building (per-candidate gc reclaim may be losing ground)
_RAM_PCT_CRIT = 92.0   # the measured OOM band (prototype lost 52 candidates at n_gpu=4 here)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


class _Resources:
    """Cheap GPU (NVML) + CPU/RAM (psutil) sampler; every probe is best-effort and never raises."""

    def __init__(self) -> None:
        self._nvml = None
        self._proc = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._pynvml = pynvml
        except Exception:  # pragma: no cover - no GPU / NVML
            self._nvml = None
        try:
            import psutil

            self._psutil = psutil
            self._proc = psutil.Process(os.getpid())
        except Exception:  # pragma: no cover
            self._psutil = None

    def sample(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self._nvml is not None:  # pragma: no cover - NVML/GPU telemetry; no GPU on the unit-test host (Linux/4090 only)
            try:
                p = self._pynvml
                u = p.nvmlDeviceGetUtilizationRates(self._nvml)
                m = p.nvmlDeviceGetMemoryInfo(self._nvml)
                out["gpu_util"] = int(u.gpu)
                out["gpu_mem_mib"] = [int(m.used // 2**20), int(m.total // 2**20)]
                out["gpu_temp"] = int(p.nvmlDeviceGetTemperature(self._nvml, 0))
            except Exception:  # pragma: no cover
                pass
        if self._psutil is not None:
            try:
                out["cpu_pct"] = float(self._psutil.cpu_percent())
                out["ram_pct"] = float(self._psutil.virtual_memory().percent)
                if self._proc is not None:
                    out["proc_rss_mib"] = int(self._proc.memory_info().rss // 2**20)
            except Exception:  # pragma: no cover
                pass
        return out


class RunMonitor:
    """Live + on-disk monitor for one prototype/campaign run. Safe to use whether or not rich/a TTY exist."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        title: str,
        total_arms: int,
        candidates_per_arm: int,
        train_steps: int,
        model: str = "?",
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.paths = attach_run_logging(self.run_dir)
        self.state_path = self.run_dir / "progress.json"
        self.title = title
        self.model = model
        self.train_steps = int(train_steps)
        self.t0 = time.time()
        self._res = _Resources()
        self._cand_secs: deque[float] = deque(maxlen=20)  # rolling per-candidate wall-clock for ETA
        self._critic_hist: deque[float] = deque(maxlen=50)
        self._best: dict[str, float] = {}
        self._anomalies: list[dict[str, Any]] = []
        # Identity context (M6/FIX B): the (arm, candidate) currently in flight, stamped onto every
        # anomaly row that does not carry its own ids so a diverged-run alarm is traceable to the exact
        # candidate that emitted it. Set in arm_start/candidate_start, cleared in candidate_done/arm_done.
        self._cur_arm: str | None = None
        self._cur_cand: Any = None
        # STATE-like anomaly kinds already fired this candidate (m9): keyed by (kind, arm, cand) so the
        # once-per-candidate suppression is correct on BOTH the serial path (cleared each candidate_start)
        # AND the parallel path (concurrent candidates never share a key; see ParallelMonitor).
        self._fired_kinds: set[Any] = set()
        self.state: dict[str, Any] = {
            "title": title,
            "model": model,
            "pid": os.getpid(),
            "started": _now_iso(),
            "updated": _now_iso(),
            "elapsed_s": 0.0,
            "phase": "starting",
            "arms": {"total": int(total_arms), "done": 0, "current": None, "current_idx": 0},
            "candidates": {
                "in_arm_total": int(candidates_per_arm),
                "in_arm_done": 0,
                "run_total": int(total_arms) * int(candidates_per_arm),
                "run_done": 0,
                "current": 0,
                "gen": 0,
            },
            "training": {},
            "best_fitness": {},
            "anomalies": {"count": 0, "recent": []},
            "resources": {},
            "eta_s": None,
            "eta": None,
        }
        self._live: Any = None
        self._progress: Any = None
        self._tasks: dict[str, Any] = {}
        self._status = ""
        self._start_live()
        log_event(
            _LOG, "run_start", title=title, model=model, arms=total_arms,
            candidates_per_arm=candidates_per_arm, train_steps=train_steps, run_dir=str(self.run_dir),
        )
        self._flush()

    # ----- rich live display (TTY only) ------------------------------------------------------------ #
    def _start_live(self) -> None:
        try:
            import sys

            if CONSOLE is None or not sys.stderr.isatty():
                return
            from rich.progress import (
                BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn,
            )

            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold]{task.fields[label]}"),
                BarColumn(bar_width=28),
                TaskProgressColumn(),
                TextColumn("{task.fields[note]}"),
                TimeElapsedColumn(),
                console=CONSOLE,
            )
            self._tasks["arms"] = self._progress.add_task("", total=self.state["arms"]["total"], label="Arms ", note="")
            self._tasks["cand"] = self._progress.add_task("", total=self.state["candidates"]["in_arm_total"], label="Cand ", note="")
            self._tasks["train"] = self._progress.add_task("", total=self.train_steps, label="Train", note="")
            from rich.live import Live

            self._live = Live(self._render(), console=CONSOLE, refresh_per_second=4, transient=False)
            self._live.start()
        except Exception:  # pragma: no cover - any rich/term issue -> headless (files + state only)
            self._live = self._progress = None

    def _render(self) -> Any:  # pragma: no cover - rich Live panel; rendered only on a TTY (headless runs skip it)
        from rich.console import Group
        from rich.panel import Panel

        r = self.state["resources"]
        gpu = (
            f"GPU {r.get('gpu_util', '?')}% {r.get('gpu_mem_mib', ['?', '?'])[0]}/"
            f"{r.get('gpu_mem_mib', ['?', '?'])[1]}MiB {r.get('gpu_temp', '?')}C"
        )
        line = (
            f"{gpu}  CPU {r.get('cpu_pct', '?')}%  RAM {r.get('ram_pct', '?')}%  "
            f"| anomalies: {self.state['anomalies']['count']}  | ETA {self._fmt_eta()}  {self._status}"
        )
        return Panel(Group(self._progress, line), title=f"{self.title} [{self.model}]", border_style="cyan")

    def _refresh(self) -> None:
        if self._live is not None:
            try:
                self._live.update(self._render())
            except Exception:  # pragma: no cover
                pass

    def _fmt_eta(self) -> str:
        s = self.state.get("eta_s")
        if not s or s <= 0:
            return "?"
        h, rem = divmod(int(s), 3600)
        m, _ = divmod(rem, 60)
        return f"{h}h{m:02d}m"

    # ----- state file -------------------------------------------------------------------------------- #
    def _flush(self) -> None:
        import json

        self.state["updated"] = _now_iso()
        self.state["elapsed_s"] = round(time.time() - self.t0, 1)
        self.state["resources"] = self._res.sample()
        self._check_resource_anomalies(self.state["resources"])  # GPU-temp + RAM guards (long laptop run)
        self.state["best_fitness"] = dict(self._best)
        self.state["anomalies"] = {"count": len(self._anomalies), "recent": self._anomalies[-5:]}
        try:
            tmp = self.state_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.state, default=str, indent=1), encoding="utf-8")
            tmp.replace(self.state_path)  # atomic
        except Exception:  # pragma: no cover
            pass
        self._refresh()

    # ----- lifecycle events -------------------------------------------------------------------------- #
    def arm_start(self, arm: str, idx: int) -> None:
        self.state["phase"] = "arm"
        self._cur_arm, self._cur_cand = arm, None  # identity context for anomaly rows (FIX B)
        self.state["arms"].update(current=arm, current_idx=idx)
        self.state["candidates"].update(in_arm_done=0, current=0, gen=0)
        if self._progress is not None:
            self._progress.update(self._tasks["arms"], completed=idx, note=arm)
            self._progress.reset(self._tasks["cand"], total=self.state["candidates"]["in_arm_total"], note="")
        log_event(_LOG, "arm_start", arm=arm, idx=idx, total=self.state["arms"]["total"])
        self._flush()

    def arm_done(self, arm: str, *, winner_fitness: float | None, secs: float) -> None:
        self.state["arms"]["done"] += 1
        if winner_fitness is not None:
            self._best[arm] = float(winner_fitness)
        if self._progress is not None:
            self._progress.update(self._tasks["arms"], completed=self.state["arms"]["done"])
        log_event(_LOG, "arm_done", arm=arm, winner_fitness=winner_fitness, secs=round(secs, 1))
        self._cur_arm, self._cur_cand = None, None  # arm finished: clear identity context (FIX B)
        self._flush()

    def candidate_start(self, arm: str, cand_idx: int, gen: int = 0) -> None:
        self.state["phase"] = "candidate"
        self._cur_arm, self._cur_cand = arm, cand_idx  # identity context for anomaly rows (FIX B)
        self._fired_kinds.clear()  # reset once-per-candidate STATE-anomaly suppression (m9) for the serial path
        self.state["candidates"].update(current=cand_idx, gen=gen)
        self.state["training"] = {}
        self._cand_t0 = time.time()
        if self._progress is not None:
            self._progress.reset(self._tasks["train"], total=self.train_steps, note="")
        self._flush()

    def candidate_done(
        self, arm: str, cand_idx: int, *, fitness: float | None, status: str = "ok", secs: float | None = None
    ) -> None:
        secs = secs if secs is not None else (time.time() - getattr(self, "_cand_t0", time.time()))
        self._cand_secs.append(secs)
        self.state["candidates"]["in_arm_done"] += 1
        self.state["candidates"]["run_done"] += 1
        self.state["last_fitness"] = fitness
        if fitness is not None and math.isnan(float(fitness)):
            self.anomaly("fitness_nan", f"{arm} cand {cand_idx} fitness=NaN", arm=arm, cand=cand_idx)
        if fitness is not None and not (fitness != fitness) and fitness > self._best.get(arm, -math.inf):
            self._best[arm] = float(fitness)
        # ETA from rolling mean candidate time x remaining candidates.
        if self._cand_secs:
            avg = sum(self._cand_secs) / len(self._cand_secs)
            remaining = self.state["candidates"]["run_total"] - self.state["candidates"]["run_done"]
            self.state["eta_s"] = round(avg * remaining, 0)
            self.state["eta"] = time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(time.time() + avg * remaining)
            )
        if self._progress is not None:
            self._progress.update(self._tasks["cand"], completed=self.state["candidates"]["in_arm_done"])
        log_event(
            _LOG, "candidate_done", arm=arm, cand=cand_idx, fitness=fitness, status=status,
            secs=round(secs, 1), run_done=self.state["candidates"]["run_done"],
            run_total=self.state["candidates"]["run_total"],
        )
        self._cur_cand = None  # candidate finished: a between-candidates anomaly carries no cand id (FIX B)
        self._flush()

    def llm_call(self, arm: str, cand: int, *, secs: float, in_tok: int | None = None,
                 out_tok: int | None = None, model: str | None = None) -> None:
        self.state["phase"] = "llm"
        self._status = f"LLM {arm} c{cand}"
        log_event(_LOG, "llm_call", arm=arm, cand=cand, secs=round(secs, 2), in_tok=in_tok,
                  out_tok=out_tok, model=model or self.model)
        self._refresh()

    def sandbox_result(self, arm: str, cand: int, *, ok: bool, reason: str = "") -> None:
        self.state["phase"] = "sandbox"
        lvl = 20 if ok else 30  # INFO / WARNING
        log_event(_LOG, "sandbox_ok" if ok else "sandbox_reject", level=lvl, arm=arm, cand=cand, reason=reason)
        if not ok:
            self._refresh()

    def train_metrics(self, step: int, metrics: dict[str, float]) -> None:
        """Stream one training-metrics snapshot: update the bar, persist, and run anomaly checks."""
        self.state["phase"] = "training"
        tr = {
            "step": int(step),
            "total": self.train_steps,
            "fps": metrics.get("time/fps"),
            "actor_loss": metrics.get("train/actor_loss"),
            "critic_loss": metrics.get("train/critic_loss"),
            "ent_coef": metrics.get("train/ent_coef"),
            "ep_rew_mean": metrics.get("rollout/ep_rew_mean"),
        }
        self.state["training"] = tr
        if self._progress is not None:
            note = f"{(tr['fps'] or 0):.0f}fps critic={_g(tr['critic_loss'])}"
            self._progress.update(self._tasks["train"], completed=int(step), note=note)
        self._check_training_anomalies(step, tr)
        self._flush()

    def _check_training_anomalies(
        self, step: int, tr: dict[str, Any], *, arm: str | None = None, cand: Any = None
    ) -> None:
        # FIX B: the PARALLEL pump passes the (arm, candidate) EXPLICITLY (it has no per-candidate identity
        # context, and several candidates train at once), while the serial path omits them and lets
        # ``anomaly`` stamp the in-flight ``_cur_arm``/``_cur_cand``. ``ident`` rides on EVERY row below.
        ident: dict[str, Any] = {}
        if arm is not None:
            ident["arm"] = arm
        if cand is not None:
            ident["cand"] = cand
        for k in ("actor_loss", "critic_loss", "ent_coef", "ep_rew_mean"):
            v = tr.get(k)
            if v is not None and (math.isnan(v) or math.isinf(v)):
                self.anomaly("nonfinite_metric", f"{k}={v} at step {step}", metric=k, step=step, **ident)
        cl = tr.get("critic_loss")
        if cl is not None and math.isfinite(cl):
            if cl > _CRITIC_EXPLOSION_ABS:
                self.anomaly("critic_explosion", f"critic_loss={cl:.3g} > {_CRITIC_EXPLOSION_ABS:.0g}", step=step, **ident)
            elif len(self._critic_hist) >= 10:
                med = sorted(self._critic_hist)[len(self._critic_hist) // 2]
                if med > 0 and cl > _CRITIC_EXPLOSION_REL * med:
                    self.anomaly("critic_explosion", f"critic_loss={cl:.3g} >> median {med:.3g}", step=step, **ident)
            self._critic_hist.append(cl)
        ec = tr.get("ent_coef")
        if ec is not None and math.isfinite(ec):
            if ec < _ENT_COEF_COLLAPSE:
                self.anomaly("entropy_collapse", f"ent_coef={ec:.3g} (premature determinism)", step=step, level=30, **ident)
            elif ec > _ENT_COEF_EXPLODE:
                self.anomaly("entropy_explosion", f"ent_coef={ec:.3g}", step=step, level=30, **ident)
        fps = tr.get("fps")
        if fps is not None and 0 < fps < _FPS_COLLAPSE:
            self.anomaly("fps_collapse", f"fps={fps:.2f} (training stalled?)", step=step, level=30, **ident)

    def _check_resource_anomalies(self, res: dict[str, Any]) -> None:
        """Flag GPU thermal-throttle + RAM-pressure on a long laptop run (deduped via a ~60 s cooldown).

        Runs on every ``_flush()`` (training-metric updates AND idle pump ticks), so it catches a hot
        GPU / RAM creep independent of training callbacks. Inherited by :class:`ParallelMonitor`, so ONE
        method covers both the serial-search in-process monitor and the parallel pump. The ~60 s cooldown
        (separate per family) stops a sustained-hot run spamming ``anomalies.jsonl`` every ~0.3 s.
        """
        now = time.time()
        t = res.get("gpu_temp")
        if isinstance(t, (int, float)) and now - getattr(self, "_last_thermal_ts", 0.0) >= 60.0:
            if t >= _GPU_TEMP_CRIT_C:
                self._last_thermal_ts = now
                self.anomaly("gpu_thermal_critical",
                             f"gpu_temp={t}C >= {_GPU_TEMP_CRIT_C}C (HW slowdown/throttle imminent)", level=40)
            elif t >= _GPU_TEMP_WARN_C:
                self._last_thermal_ts = now
                self.anomaly("gpu_thermal_warn", f"gpu_temp={t}C >= {_GPU_TEMP_WARN_C}C", level=30)
        r = res.get("ram_pct")
        if isinstance(r, (int, float)) and now - getattr(self, "_last_ram_ts", 0.0) >= 60.0:
            if r >= _RAM_PCT_CRIT:
                self._last_ram_ts = now
                self.anomaly("ram_pressure_critical",
                             f"ram_pct={r:.1f}% >= {_RAM_PCT_CRIT}% (OOM band)", level=40)
            elif r >= _RAM_PCT_WARN:
                self._last_ram_ts = now
                self.anomaly("ram_pressure_warn", f"ram_pct={r:.1f}% >= {_RAM_PCT_WARN}%", level=30)

    def anomaly(self, kind: str, detail: str, *, level: int = 40, **fields: Any) -> None:
        """Record + loudly log an anomaly (default ERROR).

        Every row is stamped with the in-flight (arm, candidate) identity unless the caller passed its
        own ids (FIX B), so a diverged-run alarm is traceable to the exact candidate that emitted it.
        Consecutive-duplicate rows are deduped per (kind, step, cand) — the candidate id is IN the key so
        a second candidate's same-kind/same-step row is never swallowed by the first's. STATE-like kinds
        (``_ONCE_PER_CANDIDATE_KINDS``) additionally fire at most once per (arm, candidate), so a
        legitimately-decayed entropy coef cannot re-alarm every log step for the rest of the run (m9).
        """
        # Stamp the identity context onto rows without their own ids (the serial training + resource
        # checks call anomaly() bare; the parallel pump passes arm/cand explicitly, which win here).
        if "arm" not in fields and self._cur_arm is not None:
            fields["arm"] = self._cur_arm
        if "cand" not in fields and self._cur_cand is not None:
            fields["cand"] = self._cur_cand
        cand_key = fields.get("cand")
        # Once-per-candidate STATE-anomaly suppression (m9): keyed by (kind, arm, cand) so it holds on the
        # serial path (cand id resets per arm -> arm disambiguates) AND the parallel path (candidate_start
        # never clears the set there, but each concurrent candidate carries a DISTINCT key).
        if kind in _ONCE_PER_CANDIDATE_KINDS:
            fired_key = (kind, fields.get("arm"), cand_key)
            if fired_key in self._fired_kinds:
                return
            self._fired_kinds.add(fired_key)
        rec = {"ts": _now_iso(), "kind": kind, "detail": detail, **fields}
        if (
            self._anomalies
            and self._anomalies[-1].get("kind") == kind
            and self._anomalies[-1].get("step") == fields.get("step")
            and self._anomalies[-1].get("cand") == cand_key
        ):
            return
        self._anomalies.append(rec)
        log_event(_LOG, "ANOMALY", level=level, kind=kind, detail=detail, **fields)
        try:
            import json

            with (self.run_dir / "anomalies.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
        except Exception:  # pragma: no cover
            pass

    def close(self, *, status: str = "done") -> None:
        self.state["phase"] = status
        log_event(
            _LOG, "run_end", status=status, elapsed_s=round(time.time() - self.t0, 1),
            anomalies=len(self._anomalies), best_fitness=dict(self._best),
        )
        self._flush()
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:  # pragma: no cover
                pass


def _g(v: Any) -> str:
    return f"{v:.3g}" if isinstance(v, (int, float)) and v == v else "?"


def make_training_callback(monitor: RunMonitor | None, *, log_every: int = 1000, governor: Any = None) -> Any:
    """Build a real SB3 ``BaseCallback`` bound to ``monitor`` (or a no-op stand-in if SB3 absent/monitor None).

    When a ``governor`` (``src.utils.guardian.ThermalGovernor``) is supplied it is *ticked* every ``log_every``
    steps: on overheat it cooperatively PAUSES training until the GPU cools (result-neutral — only wall-clock is
    spent, no weight changes), protecting a 24/7 laptop run from thermal throttling/shutdown. ``None`` = off.
    """
    try:
        from stable_baselines3.common.callbacks import BaseCallback
    except Exception:  # pragma: no cover
        return None

    class _Cb(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self._next = log_every
            self._gov_next = log_every

        def _on_step(self) -> bool:
            if governor is not None and self.num_timesteps >= self._gov_next:
                self._gov_next += log_every
                governor.govern()  # thermal pause on overheat (result-neutral); no-op when cool
            if monitor is not None and self.num_timesteps >= self._next:
                self._next += log_every
                vals = dict(self.model.logger.name_to_value)
                monitor.train_metrics(self.num_timesteps, vals)
            return True

        def _on_training_end(self) -> None:
            if monitor is not None:
                monitor.train_metrics(self.num_timesteps, dict(self.model.logger.name_to_value))

    return _Cb()


# --------------------------------------------------------------------------- #
# Candidate-parallel monitoring: workers stream metrics over a Queue; the main #
# process aggregates + detects anomalies centrally (ParallelMonitor).          #
# --------------------------------------------------------------------------- #
#: Training metrics streamed worker -> main (kept tiny).
_STREAM_KEYS = ("time/fps", "train/actor_loss", "train/critic_loss", "train/ent_coef", "rollout/ep_rew_mean")


class QueueSink:
    """Worker-side monitor stand-in (candidate-parallel path): forwards candidate + per-step training
    events to the main process over a multiprocessing Queue, where :class:`ParallelMonitor` renders them
    and runs the anomaly checks centrally. Any other ``RunMonitor`` method is a safe no-op, so the SB3
    callback and the worker can call it exactly like a real monitor."""

    def __init__(self, q: Any, cand_id: str, arm: str) -> None:
        self._q, self._cid, self._arm = q, cand_id, arm

    def _put(self, m: str, **kw: Any) -> None:
        try:
            self._q.put({"m": m, "cand": self._cid, "arm": self._arm, **kw})
        except Exception:  # pragma: no cover - queue closed / full
            pass

    def candidate_start(self, *a: Any, **k: Any) -> None:
        self._put("candidate_start")

    def train_metrics(self, step: int, metrics: dict[str, Any]) -> None:
        m = {k: (float(v) if isinstance(v, (int, float)) else None) for k, v in metrics.items() if k in _STREAM_KEYS}
        self._put("train_metrics", step=int(step), metrics=m)

    def candidate_done(self, *a: Any, fitness: Any = None, status: str = "ok", secs: Any = None,
                       error: Any = None) -> None:
        self._put("candidate_done", fitness=fitness, status=status, secs=secs, error=error)

    def __getattr__(self, _name: str) -> Any:
        return lambda *a, **k: None


class ParallelMonitor(RunMonitor):
    """Main-side monitor for the candidate-parallel scheduler (``n_gpu`` concurrent trainings).

    Drains a worker Queue (per-step training metrics + candidate lifecycle) into the shared
    ``progress.json`` + ``events.jsonl`` + ``anomalies.jsonl``, tracking the N candidates training at once
    and detecting anomalies CENTRALLY. No in-process rich Live (a parallel run is typically backgrounded) —
    ``scripts/monitor.py`` renders the dashboard from ``progress.json``; all driver-thread ``_LOG`` output
    is captured to ``events.jsonl`` too (so LLM/sandbox events are logged even though they don't flow over
    the queue).
    """

    def _start_live(self) -> None:  # no rich Live for the (backgrounded) parallel path
        self._live = self._progress = None

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self.active: dict[str, dict[str, Any]] = {}
        self._errors = 0
        self.state["active"] = []
        self.state["parallel"] = True
        self.state["candidates"]["errors"] = 0

    def dispatch(self, ev: dict[str, Any]) -> None:
        # cid/arm are the (always-present) candidate + arm ids; ``str(...)`` aligns the
        # runtime key with the ``dict[str, ...]`` declarations (no-op for the string ids
        # every event carries) and clears the Optional-index mypy errors.
        m = ev.get("m")
        cid, arm = str(ev.get("cand")), str(ev.get("arm"))
        if m == "candidate_start":
            self.active[cid] = {"cand": cid, "arm": arm, "step": 0, "total": self.train_steps}
        elif m == "train_metrics":
            mt = ev.get("metrics", {})
            tr = {
                "step": ev.get("step"), "total": self.train_steps, "fps": mt.get("time/fps"),
                "actor_loss": mt.get("train/actor_loss"), "critic_loss": mt.get("train/critic_loss"),
                "ent_coef": mt.get("train/ent_coef"), "ep_rew_mean": mt.get("rollout/ep_rew_mean"),
            }
            self.active.setdefault(cid, {"cand": cid, "arm": arm, "total": self.train_steps}).update(
                step=ev.get("step"), fps=tr["fps"], critic_loss=tr["critic_loss"]
            )
            self.state["training"] = tr
            # FIX B: pass the (arm, candidate) explicitly — the central pump has no per-candidate identity
            # context, and several candidates train at once, so every anomaly row must carry its own ids.
            self._check_training_anomalies(int(ev.get("step") or 0), tr, arm=arm, cand=cid)
        elif m == "candidate_done":
            self.active.pop(cid, None)
            c = self.state["candidates"]
            c["run_done"] += 1
            if ev.get("status") == "error":
                self._errors += 1
                c["errors"] = self._errors
                log_event(_LOG, "candidate_error", level=30, arm=arm, cand=cid, error=ev.get("error"),
                          total_errors=self._errors)
                # Flag a FAILURE WAVE loudly (e.g. a CUDA-OOM cascade): >=3 errors AND >50% of completed.
                if self._errors >= 3 and self._errors > 0.5 * c["run_done"]:
                    self.anomaly("failure_wave",
                                 f"{self._errors}/{c['run_done']} candidates FAILED; last: {ev.get('error')}",
                                 arm=arm, cand=cid)  # FIX B: identify the candidate whose failure tipped the wave
            fit = ev.get("fitness")
            # FIX D: parity with the serial candidate_done — a NaN winner-fitness must be FLAGGED, not
            # silently dropped by the ``fit == fit`` best-update guard below (NaN fails that guard).
            if fit is not None and math.isnan(float(fit)):
                self.anomaly("fitness_nan", f"{arm} cand {cid} fitness=NaN", arm=arm, cand=cid)
            if fit is not None and fit == fit and fit > self._best.get(arm, -math.inf):
                self._best[arm] = float(fit)
            secs = ev.get("secs")
            if secs:
                self._cand_secs.append(float(secs))
                avg = sum(self._cand_secs) / len(self._cand_secs)
                self.state["eta_s"] = round(avg * max(0, c["run_total"] - c["run_done"]), 0)
            log_event(_LOG, "candidate_done", arm=arm, cand=cid, fitness=fit, status=ev.get("status"),
                      secs=secs, run_done=c["run_done"], run_total=c["run_total"])
        self.state["active"] = list(self.active.values())
        self.state["phase"] = "training" if self.active else "scheduling"
        self._flush()

    def pump(self, q: Any, stop: Any) -> None:
        """Drain the worker queue until ``stop`` is set AND the queue is empty (run in a daemon thread)."""
        import queue as _q

        while not (stop.is_set() and self._empty(q)):
            try:
                ev = q.get(timeout=0.3)
            except _q.Empty:
                self._flush()  # refresh resources/ETA while idle between candidates
                continue
            except Exception:  # pragma: no cover - queue broken
                break
            try:
                self.dispatch(ev)
            except Exception:  # pragma: no cover - monitoring must never crash the run
                pass

    @staticmethod
    def _empty(q: Any) -> bool:
        try:
            return bool(q.empty())
        except Exception:  # pragma: no cover
            return True
