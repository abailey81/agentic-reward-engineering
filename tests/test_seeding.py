"""Tests for global seeding (FINAL_PLAN F.14, audit C-6).

PD-6 (results must replay byte-identically from the run seed) is load-bearing for the
pre-registered dissertation. These tests PROVE the determinism / reproducibility claim
made by ``src.utils.seeding`` against its REAL public API (``set_global_seed`` / ``rng``).
"""
from __future__ import annotations

import os
import random

import numpy as np
import pytest

from src.utils.seeding import rng, set_global_seed


def test_set_global_seed_returns_seed_and_seeds_numpy() -> None:
    """``set_global_seed`` returns the seed and makes the legacy global RNG reproducible."""
    assert set_global_seed(123) == 123
    a = np.random.random(5)
    set_global_seed(123)
    b = np.random.random(5)
    np.testing.assert_array_equal(a, b)


def test_set_global_seed_rejects_non_int() -> None:
    """Non-integer seeds fail loudly."""
    import pytest

    with pytest.raises(TypeError):
        set_global_seed(1.5)  # type: ignore[arg-type]


def test_rng_is_isolated_and_reproducible() -> None:
    """Two generators with the same seed produce identical streams; different seeds differ."""
    s1 = rng(7).standard_normal(1000)
    s2 = rng(7).standard_normal(1000)
    s3 = rng(8).standard_normal(1000)
    np.testing.assert_array_equal(s1, s2)
    assert not np.array_equal(s1, s3)


def test_global_seed_does_not_leak_into_isolated_generator() -> None:
    """An isolated ``rng`` stream is unaffected by interleaved global RNG use (no shared state)."""
    set_global_seed(0)
    expected = rng(42).standard_normal(100)
    set_global_seed(0)
    _ = np.random.random(50)  # perturb the global RNG
    got = rng(42).standard_normal(100)
    np.testing.assert_array_equal(expected, got)


# --------------------------------------------------------------------------- #
# PD-6: same seed -> identical streams; different seed -> different streams     #
# --------------------------------------------------------------------------- #
def test_set_global_seed_same_seed_identical_numpy_streams() -> None:
    """PD-6: re-seeding with the SAME seed re-establishes a byte-identical NumPy stream.

    Covers BOTH the legacy global RNG (np.random) and the stdlib ``random`` module,
    which ``set_global_seed`` also seeds (SB3 / older libs read both).
    """
    set_global_seed(2024)
    np_a = np.random.random(64)
    py_a = [random.random() for _ in range(64)]

    set_global_seed(2024)
    np_b = np.random.random(64)
    py_b = [random.random() for _ in range(64)]

    np.testing.assert_array_equal(np_a, np_b)
    assert py_a == py_b


def test_set_global_seed_different_seeds_diverge() -> None:
    """PD-6 (sensitivity): a DIFFERENT seed yields a DIFFERENT stream (seed actually matters).

    A seeder that ignored its argument would also pass the same-seed test; this guards
    against that degenerate failure mode.
    """
    set_global_seed(2024)
    np_a = np.random.random(64)
    set_global_seed(2025)
    np_b = np.random.random(64)
    assert not np.array_equal(np_a, np_b)


# --------------------------------------------------------------------------- #
# PD-6: documented environment / torch flags are set to their documented values #
# --------------------------------------------------------------------------- #
def test_set_global_seed_sets_pythonhashseed() -> None:
    """PD-6: ``PYTHONHASHSEED`` is set to the run seed (str) so hash-randomisation is pinned."""
    set_global_seed(777)
    assert os.environ["PYTHONHASHSEED"] == "777"


def test_set_global_seed_sets_cublas_workspace_config() -> None:
    """PD-6: ``CUBLAS_WORKSPACE_CONFIG`` is present for deterministic cuBLAS matmul.

    ``set_global_seed`` guarantees the env var holds one of the two documented deterministic values
    (``:4096:8`` / ``:16:8``) after the call. (The module does an EXPLICIT set, NOT ``setdefault`` — the
    exact R66 overwrite-vs-preserve semantics are pinned by
    ``test_set_global_seed_cublas_config_r66_overwrites_nondeterministic_preserves_valid`` below.)
    """
    set_global_seed(777)
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") in {":4096:8", ":16:8"}


def test_set_global_seed_cublas_config_r66_overwrites_nondeterministic_preserves_valid(monkeypatch) -> None:
    """R66 (audit 2026-06-28): the ``CUBLAS_WORKSPACE_CONFIG`` assignment is an EXPLICIT set, NOT
    ``setdefault``. This governs campaign GPU reproducibility (PD-6), so pin all three cases:

    * a pre-existing NON-deterministic value must be OVERWRITTEN to ``:4096:8`` — otherwise it would
      SILENTLY defeat reproducible cuBLAS matmul on the campaign GPU (the exact bug R66 fixed; a
      ``setdefault`` would have kept the bad value and this assertion would fail);
    * an already-deterministic value (``:16:8`` or ``:4096:8``) must be PRESERVED, not clobbered;
    * an absent value is set to the canonical ``:4096:8``.

    ``monkeypatch`` restores the env afterwards so mutating this process-global cannot leak into the suite.
    """
    # (1) pre-existing non-deterministic value -> OVERWRITTEN (setdefault would have kept ":2:2").
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":2:2")
    set_global_seed(1)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    # (2) already-deterministic ":16:8" -> PRESERVED (not clobbered to ":4096:8").
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    set_global_seed(1)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    # (3) already-deterministic ":4096:8" -> PRESERVED.
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    set_global_seed(1)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    # (4) absent -> set to the canonical ":4096:8".
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    set_global_seed(1)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_set_global_seed_sets_torch_cudnn_flags() -> None:
    """PD-6: cuDNN is pinned deterministic (deterministic=True, benchmark=False) after seeding.

    These flags are set unconditionally by ``set_global_seed`` (independent of CUDA presence),
    so they are assertable on a CPU-only host. torch is optional -> importorskip.
    """
    torch = pytest.importorskip("torch")
    set_global_seed(777)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_set_global_seed_deterministic_torch_enables_global_algorithms() -> None:
    """PD-6: ``deterministic_torch=True`` (the training/campaign path) turns on global determinism.

    Asserts ``torch.are_deterministic_algorithms_enabled()`` flips True. Restores the prior
    state afterwards so this test does not leak global determinism config into the rest of the
    suite. Guarded for older torch lacking the query helper.
    """
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "are_deterministic_algorithms_enabled"):
        pytest.skip("torch too old to query global deterministic-algorithms state")
    prior = torch.are_deterministic_algorithms_enabled()
    try:
        set_global_seed(777, deterministic_torch=True)
        assert torch.are_deterministic_algorithms_enabled() is True
    finally:
        torch.use_deterministic_algorithms(prior, warn_only=True)


# --------------------------------------------------------------------------- #
# PD-6: idempotence / independence from prior global RNG state                  #
# --------------------------------------------------------------------------- #
def test_set_global_seed_independent_of_prior_global_state() -> None:
    """PD-6: the post-seed stream does NOT depend on RNG state consumed before seeding.

    Interleaves heavy global draws (np.random + stdlib random) before the seed call; the
    stream produced AFTER seeding must be identical regardless of that prior consumption.
    A seeder that merely re-seeded from system entropy, or failed to reset both RNGs, would
    fail this.
    """
    # Path A: seed from a "clean" state.
    set_global_seed(99)
    set_global_seed(13)
    np_clean = np.random.random(50)
    py_clean = [random.random() for _ in range(50)]

    # Path B: consume a lot of entropy from BOTH global RNGs, THEN seed identically.
    set_global_seed(99)
    _ = np.random.random(10_000)
    _ = [random.random() for _ in range(10_000)]
    set_global_seed(13)
    np_dirty = np.random.random(50)
    py_dirty = [random.random() for _ in range(50)]

    np.testing.assert_array_equal(np_clean, np_dirty)
    assert py_clean == py_dirty


# --------------------------------------------------------------------------- #
# PD-6: distinct seeds (the parallel-worker model) give non-overlapping streams #
# --------------------------------------------------------------------------- #
# NOTE ON COVERAGE: ``src.utils.seeding`` exposes NO per-worker seed-derivation
# function. The parallel/test-leg workers (src.orchestration.parallel.train_candidate,
# src.orchestration.test_leg._test_seed_worker) each receive a DISTINCT seed assigned
# upstream and call ``set_global_seed(seed, ...)`` in their own spawn process. The
# testable determinism property the worker model relies on is therefore: distinct seeds
# fed to the seeder yield distinct, deterministic, NON-overlapping global streams. We
# assert exactly that against the real API (not an invented derivation helper).
def test_distinct_seeds_give_nonoverlapping_deterministic_streams() -> None:
    """PD-6 (parallel-worker model): distinct seeds -> distinct, deterministic, non-prefix streams.

    For a set of seeds (as the campaign assigns to parallel workers): each seed's first-K
    legacy-global draws are (a) reproducible on re-seed and (b) not a prefix of any other
    seed's draws. Non-overlap is the property that makes per-worker seeds independent rather
    than aliased.
    """
    seeds = [0, 1, 2, 7, 12345, 2024]
    k = 256

    def first_k(s: int) -> np.ndarray:
        set_global_seed(s)
        return np.random.random(k)

    streams = {s: first_k(s) for s in seeds}

    # (a) deterministic: re-seeding reproduces each stream byte-identically.
    for s in seeds:
        np.testing.assert_array_equal(streams[s], first_k(s))

    # (b) pairwise distinct AND no stream is a prefix of another (non-overlap).
    for i, si in enumerate(seeds):
        for sj in seeds[i + 1 :]:
            assert not np.array_equal(streams[si], streams[sj])
            # prefix check on the shared head (here equal length, so == full-array check)
            head = min(len(streams[si]), len(streams[sj]))
            assert not np.array_equal(streams[si][:head], streams[sj][:head])


# --------------------------------------------------------------------------- #
# PD-6: end-to-end replay — seed -> draw (numpy + torch) -> reseed -> redraw     #
# --------------------------------------------------------------------------- #
def test_end_to_end_replay_numpy_and_torch_byte_identical() -> None:
    """PD-6 (replay): a seed -> draw -> reseed -> redraw cycle is byte-identical across stacks.

    Exercises the full stack ``set_global_seed`` touches in the import-light env: stdlib
    ``random``, the legacy ``np.random`` global, AND (importorskip) ``torch.rand`` on CPU.
    This is the smallest faithful proxy of a run replaying from its recorded seed (PD-6).
    """
    torch = pytest.importorskip("torch")

    def draw() -> tuple[list[float], np.ndarray, "torch.Tensor"]:
        set_global_seed(31337)
        py = [random.random() for _ in range(32)]
        npy = np.random.random(32)
        tch = torch.rand(32)  # CPU; CUDA replay is documented as non-bitwise in the module
        return py, npy, tch

    py_a, np_a, t_a = draw()
    py_b, np_b, t_b = draw()

    assert py_a == py_b
    np.testing.assert_array_equal(np_a, np_b)
    assert torch.equal(t_a, t_b)
