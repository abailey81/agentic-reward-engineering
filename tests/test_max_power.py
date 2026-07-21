"""Tests for ``src.orchestration.parallel.max_power_config`` — the heterogeneous full-throughput sizer.

The helper is pure (no training, no side effects): it reads cpu/RAM via psutil and returns a recommended
``(n_gpu, n_cpu)``. We pin the machine via monkeypatch so the arithmetic is checked EXACTLY, then assert
the real-machine call satisfies the safety invariants.
"""
from __future__ import annotations

import types

import psutil

from src.orchestration.parallel import max_power_config


def _pin(monkeypatch, *, cores: int, ram_gb: float) -> None:
    monkeypatch.setattr(psutil, "cpu_count", lambda logical=False: cores)
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: types.SimpleNamespace(available=int(ram_gb * 2**30), total=int(ram_gb * 2**30)),
    )


def test_exact_fill_on_a_16gb_8core_box(monkeypatch) -> None:
    """50k buffer ~0.71 GB -> per-worker ~2.11 GB; 16 GB budget -> 7 workers; 2 GPU + the rest CPU."""
    _pin(monkeypatch, cores=8, ram_gb=16.0)
    assert max_power_config(50000, gpu_cap=2) == (2, 5)


def test_cpu_only_uses_no_gpu(monkeypatch) -> None:
    _pin(monkeypatch, cores=8, ram_gb=16.0)
    n_gpu, n_cpu = max_power_config(50000, cpu_only=True)
    assert n_gpu == 0 and n_cpu == 7


def test_gpu_cap_is_respected(monkeypatch) -> None:
    """Even with abundant RAM/cores, GPU workers never exceed the VRAM-safe cap."""
    _pin(monkeypatch, cores=32, ram_gb=256.0)
    n_gpu, _ = max_power_config(50000, gpu_cap=2)
    assert n_gpu == 2


def test_low_ram_clamps_to_a_single_worker(monkeypatch) -> None:
    """A box that can only fit one worker yields exactly one GPU worker and no CPU workers."""
    _pin(monkeypatch, cores=8, ram_gb=2.0)
    assert max_power_config(50000, gpu_cap=2) == (1, 0)


def test_cpu_workers_nonincreasing_in_train_steps(monkeypatch) -> None:
    """Bigger replay buffers (more steps) cost more RAM per worker -> fewer concurrent CPU workers."""
    _pin(monkeypatch, cores=8, ram_gb=16.0)
    counts = [max_power_config(s, gpu_cap=2)[1] for s in (25_000, 50_000, 100_000, 200_000)]
    assert counts == sorted(counts, reverse=True)
    assert all(c >= 0 for c in counts)


def test_total_workers_never_exceed_physical_cores(monkeypatch) -> None:
    _pin(monkeypatch, cores=4, ram_gb=64.0)
    n_gpu, n_cpu = max_power_config(50000, gpu_cap=2)
    assert n_gpu + n_cpu <= 4


def test_real_machine_invariants() -> None:
    """On THIS machine (no mock): at least one worker, GPU within cap, total within physical cores."""
    n_gpu, n_cpu = max_power_config(50000, gpu_cap=2)
    phys = psutil.cpu_count(logical=False) or 4
    assert 1 <= n_gpu <= 2
    assert n_cpu >= 0
    assert 1 <= n_gpu + n_cpu <= phys
