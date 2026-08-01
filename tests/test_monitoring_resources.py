"""Focused tests for the long-run GPU-thermal + RAM resource guards (campaign-robustness §E / §C/§D).

The resource sampler already records ``gpu_temp`` + ``ram_pct``; these tests pin the WARN/ABORT
thresholds added to :meth:`RunMonitor._check_resource_anomalies` (inherited by ``ParallelMonitor``,
so one method covers serial + parallel monitors) and the ~60 s cooldown that prevents anomaly spam.
No torch / no SAC; the monitor runs headless (no TTY).
"""
from __future__ import annotations

import src.utils.monitoring as M
from src.utils.monitoring import ParallelMonitor, RunMonitor


def _monitor(tmp_path) -> RunMonitor:
    return RunMonitor(
        tmp_path / "run", title="t", total_arms=1, candidates_per_arm=1, train_steps=10, model="stub"
    )


def _kinds(mon: RunMonitor) -> list[str]:
    return [a["kind"] for a in mon._anomalies]


def test_no_anomaly_below_thresholds(tmp_path) -> None:
    mon = _monitor(tmp_path)
    before = len(mon._anomalies)
    mon._check_resource_anomalies({"gpu_temp": 70, "ram_pct": 60.0})
    assert len(mon._anomalies) == before  # nothing flagged in the safe band


def test_gpu_temp_warn_and_critical_fire_at_thresholds(tmp_path) -> None:
    mon = _monitor(tmp_path)
    mon._check_resource_anomalies({"gpu_temp": M._GPU_TEMP_WARN_C, "ram_pct": 50.0})
    assert "gpu_thermal_warn" in _kinds(mon)

    mon2 = _monitor(tmp_path / "b")
    mon2._check_resource_anomalies({"gpu_temp": M._GPU_TEMP_CRIT_C, "ram_pct": 50.0})
    assert "gpu_thermal_critical" in _kinds(mon2)


def test_ram_warn_and_critical_fire_at_thresholds(tmp_path) -> None:
    """The RAM cooldown is shared between the WARN and CRITICAL families, so this test must not
    inherit one from the REAL host.

    The monitor samples actual machine RAM on every flush. When the host crosses ``_RAM_PCT_CRIT``
    (92 %) the sampler fires ``ram_pressure_critical`` FIRST and stamps ``_last_ram_ts`` — and the
    explicit WARN call below is then swallowed by the 60 s cooldown, so the assertion fails for a
    reason that has nothing to do with the thresholds. That is exactly what happened during the
    2026-07-28 campaign run (12 drivers + pytest + editor pushed the laptop into the OOM band):
    the test passed alone and in-module, and failed only inside the full suite. Reproduced
    deterministically as CRIT-then-WARN -> ``ram_pressure_warn`` absent.

    Zeroing the stamp makes the test assert what it claims — the THRESHOLDS — instead of the
    machine's incidental memory pressure.
    """
    mon = _monitor(tmp_path)
    mon._last_ram_ts = 0.0
    mon._check_resource_anomalies({"gpu_temp": 60, "ram_pct": M._RAM_PCT_WARN})
    assert "ram_pressure_warn" in _kinds(mon)

    mon2 = _monitor(tmp_path / "b")
    mon2._last_ram_ts = 0.0
    mon2._check_resource_anomalies({"gpu_temp": 60, "ram_pct": M._RAM_PCT_CRIT})
    assert "ram_pressure_critical" in _kinds(mon2)


def test_a_HOST_fired_critical_must_not_swallow_the_warn_assertion() -> None:
    """Pin the mechanism itself, so the flake above can never silently return.

    WARN and CRITICAL share one `_last_ram_ts`, so a CRITICAL inside the 60 s window suppresses a
    subsequent WARN. That is CORRECT behaviour for the monitor (it exists to stop anomaly spam) and
    is only a problem when a test lets the real machine fire the first one.

    NOTE: FIXED 2026-07-31 (RUN 9, record §92) — THIS TEST WAS VULNERABLE TO THE VERY FLAKE IT PINS,
    in the mirror direction. Zeroing `_last_ram_ts` protects the test above, whose assertion is
    "the WARN IS present"; it does NOT protect this one, whose assertion is "the WARN is ABSENT".
    The monitor samples real machine RAM on construction, and the WARN band (85 %) is a LOWER bar
    than the CRITICAL band (92 %) — so any host above 85 % puts a REAL `ram_pressure_warn` into
    `_anomalies` before this test makes a single call, and `not in kinds` can never hold however
    perfectly the cooldown works. Observed live during this session's full-suite run (the laptop was
    at 85.5 % with 12 supervisors, 24 drivers and pytest resident):

        WARNING  ANOMALY kind=ram_pressure_warn      detail=ram_pct=85.5% >= 85.0%   <- the HOST
        ERROR    ANOMALY kind=ram_pressure_critical  detail=ram_pct=92.0% >= 92.0%   <- the test
        AssertionError: assert 'ram_pressure_warn' not in ['ram_pressure_warn', 'ram_pressure_critical']

    The fix asserts over the anomalies THIS TEST PRODUCED rather than over the monitor's whole
    history, so no amount of real-machine pressure can reach the assertion. The mechanism being
    pinned is unchanged and the test still fails if the shared cooldown stops suppressing.
    """
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        mon = _monitor(Path(td))
        mon._last_ram_ts = 0.0
        mon._anomalies.clear()          # discard anything the REAL host fired at construction
        mon._check_resource_anomalies({"gpu_temp": 60, "ram_pct": M._RAM_PCT_CRIT})
        mon._check_resource_anomalies({"gpu_temp": 60, "ram_pct": M._RAM_PCT_WARN})
        kinds = _kinds(mon)
        assert "ram_pressure_critical" in kinds
        assert "ram_pressure_warn" not in kinds, (
            "the shared cooldown must suppress the WARN — if this ever changes, the isolation "
            "workaround above can be dropped")


def test_cooldown_dedups_repeated_thermal_anomalies(tmp_path) -> None:
    """A sustained-hot GPU must not spam anomalies on every ~0.3 s flush (60 s cooldown)."""
    mon = _monitor(tmp_path)
    for _ in range(5):
        mon._check_resource_anomalies({"gpu_temp": M._GPU_TEMP_CRIT_C, "ram_pct": 50.0})
    assert _kinds(mon).count("gpu_thermal_critical") == 1  # deduped by the cooldown


def test_missing_or_nonnumeric_fields_are_ignored(tmp_path) -> None:
    """A probe without gpu_temp/ram_pct (NVML/psutil absent) is a safe no-op."""
    mon = _monitor(tmp_path)
    before = len(mon._anomalies)
    mon._check_resource_anomalies({})  # empty sample (no GPU, no psutil)
    mon._check_resource_anomalies({"gpu_temp": None, "ram_pct": "n/a"})
    assert len(mon._anomalies) == before


def test_parallel_monitor_inherits_the_resource_guard(tmp_path) -> None:
    """ParallelMonitor gets the SAME guard (one edit covers serial + parallel)."""
    pm = ParallelMonitor(
        tmp_path / "par", title="t", total_arms=1, candidates_per_arm=1, train_steps=10, model="stub"
    )
    assert hasattr(pm, "_check_resource_anomalies")
    pm._check_resource_anomalies({"gpu_temp": M._GPU_TEMP_CRIT_C, "ram_pct": 50.0})
    assert "gpu_thermal_critical" in [a["kind"] for a in pm._anomalies]
