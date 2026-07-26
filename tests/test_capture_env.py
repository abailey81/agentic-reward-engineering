"""Tests for the run-provenance env capture (scripts/capture_env.py).

Focus: the C1 gold-panel provenance block — every run RECORD must name EXACTLY which gold panel
(the active suffix + per-artifact manifest SHA-256s) produced it, so the headline panel's identity
is recoverable from ``env.json`` (and, via run_campaign, from ``campaign_summary.json``).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import capture_env  # noqa: E402


def test_gold_panel_provenance_records_active_suffix(monkeypatch) -> None:
    """``_gold_panel_provenance`` records the ACTIVE gold suffix (C1)."""
    from src.data import loaders

    monkeypatch.setattr(loaders, "gold_suffix", lambda: "univ7")
    gp = capture_env._gold_panel_provenance()
    assert gp["available"] is True
    assert gp["suffix"] == "univ7"
    # Every production gold artifact is keyed (value may be None if that suffix isn't manifested).
    assert set(gp["manifest_sha256"]) == {"returns_panel", "cash_features", "top30_selection", "splits"}


def test_capture_env_includes_gold_panel_block() -> None:
    """capture_env() carries the C1 gold_panel provenance + the bumped schema tag."""
    env = capture_env.capture_env(seed=0)
    assert env["schema"] == "capture_env/4"   # /3 applied tf32; /4 the CPU identity (2026-07-27)
    # /4: the confirmatory lane moved to CPU and the archive recorded NOTHING about the processor.
    # env_fingerprint's only platform field is platform.platform() (kernel + glibc), and
    # integrity._record_device reads only nvidia_smi.gpus[0] -- which on a CPU node returns
    # "<absent>", a value the integrity report treats as a WILDCARD. So crn_pair_device_consistent
    # was green whatever silicon the pair ran on, while lanes.EXCLUDED_CPU_POOLS excludes the AMD
    # pool precisely because a different microarchitecture changes float reduction order and breaks
    # the CRN bit-exactness every paired contrast rests on. Now auditable after the fact.
    assert "cpu" in env, "the processor identity must reach the archived env"
    assert env["cpu"].get("machine"), "no machine architecture recorded"
    # TF32 is a REGISTERED design parameter and the largest precision lever in the stack (~10-bit
    # vs 23-bit matmul mantissa, on the ~97%-of-runtime gradient update). freeze.py verified the
    # CONFIG; nothing recorded what was IN FORCE at training time, so the pin was unverifiable at
    # the execution boundary -- the R85 "a pin nobody can verify is FICTIONAL" failure. It is also
    # CUDA-only, so it differs silently between the GPU pilots and the CPU campaign lane.
    for _k in ("matmul_allow_tf32", "cudnn_allow_tf32", "float32_matmul_precision"):
        assert _k in env["torch_cuda"], f"applied TF32 field {_k!r} is not archived"
    assert "gold_panel" in env
    assert "suffix" in env["gold_panel"]


def test_gold_panel_provenance_matches_default_suffix() -> None:
    """With no override, the recorded suffix is the config-primary default (univ5, Split C ADR-044/051)."""
    from src.data.loaders import gold_suffix

    gp = capture_env._gold_panel_provenance()
    assert gp["suffix"] == gold_suffix() == "univ5"


# --- BLAS thread count: part of the determinism envelope (2026-07-26) -----------------------
# Multi-threaded BLAS changes the ORDER of float reductions, so a training at 4 threads is NOT
# bit-identical to the same training at 1. The throughput run plan calls the thread count
# "part of the determinism envelope" and gates any change behind a pre-freeze ratification -- but
# the archive did not RECORD it, leaving the S6 homogeneity audit blind to a heterogeneous thread
# regime exactly as it was blind to a CPU/GPU device mix.


def test_determinism_env_records_the_BLAS_thread_counts(monkeypatch) -> None:
    """The REQUESTED thread regime must land in the archive, so a mix is detectable by audit."""
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setenv("MKL_NUM_THREADS", "4")
    env = capture_env.capture_env(seed=0)
    det = env["determinism_env"]
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        assert key in det, f"{key} is not recorded — a thread-regime change would be invisible"
    assert det["OMP_NUM_THREADS"] == "4"
    # the pre-existing keys must survive
    for key in ("CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED", "CUDA_VISIBLE_DEVICES"):
        assert key in det


def test_torch_block_records_the_EFFECTIVE_thread_count() -> None:
    """The env var records what was REQUESTED; torch.set_num_threads() can override it in-process,
    and it is the EFFECTIVE value that fixes the reduction order — so record that too."""
    tc = capture_env._torch_cuda()
    if not tc.get("available"):
        import pytest

        pytest.skip("torch not installed in this environment")
    assert "num_threads" in tc and "num_interop_threads" in tc
    assert tc["num_threads"] is None or tc["num_threads"] >= 1
