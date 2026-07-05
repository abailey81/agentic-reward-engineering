"""Auto-guardian — turns the (read-only) anomaly *detection* of ``monitoring.py`` into *action*.

The monitor records thermal/RAM/divergence anomalies but never halts. The guardian adds the missing
controller, with two **result-neutral** governors:

* **Thermal governor** — when the GPU crosses a hot threshold, *pause* training (a cooperative sleep) until it
  cools past a lower threshold (hysteresis, so it doesn't flap). Pausing preserves the training state exactly:
  it changes only wall-clock, never a learned weight, so byte-determinism and every reported number are
  untouched. This protects a 24/7 laptop run from sustained throttling / thermal shutdown.
* **RAM governor** — when system RAM crosses a danger threshold it requests a back-off (the orchestrator can
  defer launching the next concurrent candidate / trigger an earlier pool recycle). Also result-neutral: it
  reorders *when* work runs, not *what* it computes.

The decision logic is pure (unit-tested in ``tests/test_guardian.py``); the only side effect is ``time.sleep``
inside :meth:`ThermalGovernor.govern`, injected so tests stay instant.
"""
from __future__ import annotations

from collections.abc import Callable


def should_pause(temp_c: float | None, *, hi: float, lo: float, paused: bool) -> bool:
    """Hysteresis thermal gate. Start pausing at/above ``hi``; once paused, stay paused until the temperature
    falls BELOW ``lo`` (so it doesn't oscillate around a single threshold). ``None`` (no telemetry) never pauses.
    """
    if temp_c is None:
        return False
    if paused:
        return temp_c >= lo
    return temp_c >= hi


def should_throttle_ram(ram_pct: float | None, *, danger: float) -> bool:
    """Request a concurrency back-off when system RAM is at/above the danger band (the measured OOM zone)."""
    if ram_pct is None:
        return False
    return ram_pct >= danger


class ThermalGovernor:
    """Cooperative thermal pause with hysteresis. Call :meth:`govern` from a training callback / loop tick."""

    def __init__(
        self,
        *,
        hi: float = 88.0,
        lo: float = 80.0,
        poll_secs: float = 5.0,
        read_temp: Callable[[], float | None] | None = None,
        sleep: Callable[[float], None] | None = None,
        log: Callable[[str], None] | None = None,
        max_pause_secs: float = 1800.0,
    ) -> None:
        if not (lo < hi):
            raise ValueError(f"thermal lo ({lo}) must be < hi ({hi})")
        self.hi, self.lo, self.poll_secs, self.max_pause_secs = hi, lo, poll_secs, max_pause_secs
        self._read_temp = read_temp or _default_read_temp
        self._sleep = sleep or _default_sleep
        self._log = log or (lambda _m: None)
        self.total_paused_secs = 0.0
        self.pause_events = 0

    def govern(self) -> float:
        """If the GPU is too hot, sleep (polling) until it cools past ``lo`` or ``max_pause_secs`` elapses.
        Returns the seconds paused (0.0 if no pause was needed). Result-neutral: only wall-clock is spent."""
        temp = self._read_temp()
        if not should_pause(temp, hi=self.hi, lo=self.lo, paused=False):
            return 0.0
        self.pause_events += 1
        self._log(f"[guardian] GPU {temp:.0f}C >= {self.hi:.0f}C — pausing training to cool to < {self.lo:.0f}C")
        waited = 0.0
        paused = True
        while paused and waited < self.max_pause_secs:
            self._sleep(self.poll_secs)
            waited += self.poll_secs
            temp = self._read_temp()
            paused = should_pause(temp, hi=self.hi, lo=self.lo, paused=True)
        self.total_paused_secs += waited
        self._log(f"[guardian] resumed after {waited:.0f}s (GPU {('%.0f' % temp) if temp is not None else '?'}C)")
        return waited


def _default_read_temp() -> float | None:
    """Best-effort GPU temperature via NVML; ``None`` when unavailable (CPU box / no telemetry)."""
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
    except Exception:
        return None


def _default_sleep(secs: float) -> None:
    import time

    time.sleep(secs)
