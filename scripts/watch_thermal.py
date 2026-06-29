"""Standalone GPU-thermal + RAM watchdog sidecar (ram_thermal §4) — monitoring ONLY.

Run this in a SECOND terminal while the headline campaign runs. It is **zero-touch**: it does NOT
import the run, the campaign package, torch, or any heavy dependency — it reads NVML directly (GPU
temperature + the hardware throttle-reason bitmask) and, optionally, the campaign's already-written
``<run_dir>/progress.json`` (for ``ram_pct``, written by ``src.utils.monitoring.RunMonitor``). It only
prints WARN / ABORT lines; it never alters or signals the run. Its job is to (a) catch the RAM red line
*before* an OOM corrupts an arm, and (b) tell you if heat is silently inflating wall-clock — both are
operational signals, NOT correctness signals (the science is seed-deterministic; a throttled run
produces identical numbers, just later).

Thresholds (measured on the RTX-4050 laptop, 2026-06-24): HW thermal slowdown begins ~91 C; the
prototype's n_gpu=4 transition-wave OOM fired in the ~92% RAM band. They mirror the in-run guards added
to ``src/utils/monitoring.py`` so the sidecar and the run agree.

Usage::

    .venv/Scripts/python.exe scripts/watch_thermal.py                       # NVML only
    .venv/Scripts/python.exe scripts/watch_thermal.py outputs/campaign/search  # + ram_pct from progress.json
    .venv/Scripts/python.exe scripts/watch_thermal.py outputs/campaign --interval 15

Action on a sustained ABORT: improve cooling (elevate/clean the laptop, cap room temp) or drop a GPU
worker slot (``--search-gpu 2``). Do NOT change any training/eval parameter in response — that would
break matched compute.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Resource thresholds (kept in lockstep with src/utils/monitoring.py).
GPU_TEMP_WARN_C = 87
GPU_TEMP_ABORT_C = 91  # HW thermal slowdown on this card (also read live from NVML when available)
RAM_PCT_WARN = 85.0
RAM_PCT_ABORT = 92.0

# NVML clock-throttle-reason bits indicating an ACTIVE thermal throttle right now.
_THERMAL_BITS = 0x8 | 0x20 | 0x40  # HW_SLOWDOWN | SW_THERMAL_SLOWDOWN | HW_THERMAL_SLOWDOWN


def _open_nvml() -> tuple[Any, Any, int]:
    """Return ``(pynvml, handle, slowdown_threshold_C)`` or ``(None, None, GPU_TEMP_ABORT_C)``."""
    try:
        import pynvml as p

        p.nvmlInit()
        h = p.nvmlDeviceGetHandleByIndex(0)
        try:
            slow = int(p.nvmlDeviceGetTemperatureThreshold(h, p.NVML_TEMPERATURE_THRESHOLD_SLOWDOWN))
        except Exception:  # pragma: no cover - threshold API absent
            slow = GPU_TEMP_ABORT_C
        return p, h, slow
    except Exception as exc:  # pragma: no cover - no NVML / no GPU
        print(f"[watch_thermal] NVML unavailable ({type(exc).__name__}: {exc}); "
              f"GPU temperature will not be polled.", file=sys.stderr, flush=True)
        return None, None, GPU_TEMP_ABORT_C


def _read_ram_pct(progress_path: Path | None) -> float | None:
    """Read ``resources.ram_pct`` from a run's progress.json (best-effort; None if unavailable)."""
    if progress_path is None or not progress_path.is_file():
        return None
    try:
        st = json.loads(progress_path.read_text(encoding="utf-8"))
        val = st.get("resources", {}).get("ram_pct")
        return float(val) if isinstance(val, (int, float)) else None
    except Exception:  # pragma: no cover - partial write / parse race -> skip this tick
        return None


def _gpu_status(p: Any, h: Any, slow_c: int) -> tuple[str | None, str]:
    """Return ``(level, message)`` for the GPU; ``level`` is 'ABORT' / 'WARN' / None."""
    if p is None or h is None:
        return None, ""
    try:
        t = int(p.nvmlDeviceGetTemperature(h, 0))
    except Exception:  # pragma: no cover
        return None, ""
    try:
        thr = int(p.nvmlDeviceGetCurrentClocksThrottleReasons(h))
    except Exception:  # pragma: no cover - throttle-reason API absent
        thr = 0
    throttling = bool(thr & _THERMAL_BITS)
    if t >= slow_c or t >= GPU_TEMP_ABORT_C or throttling:
        return "ABORT", (f"gpu_temp={t}C (slowdown>={slow_c}C) "
                         f"throttle_reasons={hex(thr)} thermal_bit={throttling}")
    if t >= GPU_TEMP_WARN_C:
        return "WARN", f"gpu_temp={t}C approaching slowdown {slow_c}C"
    return None, f"gpu_temp={t}C ok"


def _ram_status(ram_pct: float | None) -> tuple[str | None, str]:
    if ram_pct is None:
        return None, ""
    if ram_pct >= RAM_PCT_ABORT:
        return "ABORT", f"ram_pct={ram_pct:.1f}% >= {RAM_PCT_ABORT}% (OOM band)"
    if ram_pct >= RAM_PCT_WARN:
        return "WARN", f"ram_pct={ram_pct:.1f}% >= {RAM_PCT_WARN}%"
    return None, f"ram_pct={ram_pct:.1f}% ok"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Standalone GPU-thermal + RAM watchdog (monitoring only).")
    ap.add_argument(
        "run_dir", nargs="?", default=None,
        help="Optional run dir containing progress.json (adds ram_pct from the live run). "
             "Without it, only NVML GPU temperature/throttle is polled.",
    )
    ap.add_argument("--interval", type=float, default=30.0, help="poll seconds (default 30)")
    ap.add_argument("--once", action="store_true", help="print one snapshot line and exit")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    progress_path = (Path(args.run_dir) / "progress.json") if args.run_dir else None
    p, h, slow_c = _open_nvml()
    if p is None and progress_path is None:
        raise SystemExit("[watch_thermal] nothing to watch: NVML unavailable and no run_dir given.")
    print(f"[watch_thermal] watching (gpu_warn={GPU_TEMP_WARN_C}C abort>={slow_c}C, "
          f"ram_warn={RAM_PCT_WARN}% abort>={RAM_PCT_ABORT}%, interval={args.interval}s); Ctrl-C to stop.",
          flush=True)
    try:
        while True:
            ram = _read_ram_pct(progress_path)
            for level, msg in (_gpu_status(p, h, slow_c), _ram_status(ram)):
                if level is not None:
                    print(f"[watch_thermal] {level}: {msg}", flush=True)
            if args.once:
                break
            time.sleep(max(1.0, float(args.interval)))
    except KeyboardInterrupt:
        print("[watch_thermal] stopped.", flush=True)
    finally:
        if p is not None:
            try:
                p.nvmlShutdown()
            except Exception:  # pragma: no cover
                pass


if __name__ == "__main__":
    main()
