"""Global deterministic seeding (FINAL_PLAN F.14, audit C-6).

Seeds every stochastic stack from a single run seed so a run reproduces (modulo documented
nondeterminism: some CUDA kernels; and LLM calls, which are replayed from the archive, not
regenerated — audit C-2). The seed is recorded in every run artifact (``src.io.results``).
"""
from __future__ import annotations

import os
import random

import numpy as np

__all__ = ["set_global_seed", "rng"]


def set_global_seed(seed: int, *, deterministic_torch: bool = False) -> int:
    """Seed Python ``random``, NumPy, ``PYTHONHASHSEED`` and (if importable) torch + cuDNN.

    With ``deterministic_torch=True`` (the training/campaign path) also enables
    ``torch.use_deterministic_algorithms(True, warn_only=True)`` and sets
    ``CUBLAS_WORKSPACE_CONFIG`` so deterministic cuBLAS matmul is used (M4, audit
    2026-06-19). Residual GPU non-determinism is documented in METHODS: results are
    NOT bit-reproducible across a CPU<->GPU or a different-GPU boundary, nor across
    torch releases (PyTorch randomness notes) -- which is why we report mean +/- error
    bars over N seeds (statistical, not bitwise, reproducibility) and replay LLM calls
    from the archive rather than regenerating them.

    Returns the seed so callers can record it in the run artifact.
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed)!r}")
    # PYTHONHASHSEED governs hash randomisation in CHILD processes spawned AFTER this call (the
    # ``spawn`` sandbox / test-leg workers read it at their own interpreter start); the PARENT's hash
    # seed was fixed when this interpreter started, so this assignment does not re-randomise the current
    # process. Run records are written with ``sort_keys=True``, so parent dict-order is not a determinism
    # risk — the env var is what makes the spawned workers' hashing reproducible.
    os.environ["PYTHONHASHSEED"] = str(seed)
    # CUBLAS_WORKSPACE_CONFIG must be set BEFORE the first CUDA op for deterministic cuBLAS matmul.
    # Use an explicit set (NOT ``setdefault``): a pre-existing NON-deterministic value would otherwise
    # SILENTLY defeat reproducible cuBLAS on the campaign GPU (audit 2026-06-28, R66). A pre-existing
    # value that is ALREADY deterministic (":16:8" is the only other valid setting) is preserved; any
    # other value (or none) is set to the canonical ":4096:8".
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)  # legacy global RNG (SB3 / older libs read this)

    try:  # torch is optional in the analysis/test environment
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if deterministic_torch:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:  # noqa: BLE001 - older torch / op without a deterministic impl
                pass
    except ImportError:
        pass

    return seed


def rng(seed: int) -> np.random.Generator:
    """Return a fresh, isolated NumPy ``Generator`` (preferred over the legacy global RNG)."""
    return np.random.default_rng(seed)
