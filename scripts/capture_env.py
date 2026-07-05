"""CI-grade environment capture for run reproducibility (Rank 14; audit C-2/C-6).

Every run directory should record *exactly which machine + software stack* produced it so
results replay rather than regenerate (the replay-from-archive rule). This module EXTENDS the
canonical :func:`src.utils.provenance.env_fingerprint` (it imports it; it never re-implements the
Python / platform / git / tracked-package fields) and enriches it with the fields a reproducibility
audit on a GPU box actually needs but the lightweight fingerprint deliberately omits:

  - the FULL installed package set (``pip freeze``-equivalent via ``importlib.metadata``), not just
    the curated ``_TRACKED_PACKAGES``;
  - the GPU/CUDA stack: the ``nvidia-smi`` driver line (best-effort — skipped cleanly when absent),
    ``torch.version.cuda``, and the cuDNN version;
  - the determinism knobs that decide whether a CUDA run is bitwise-stable: the relevant
    ``os.environ`` snapshot (``CUBLAS_WORKSPACE_CONFIG`` / ``PYTHONHASHSEED`` / ``CUDA_VISIBLE_DEVICES``)
    and ``torch.are_deterministic_algorithms_enabled()``;
  - the run ``seed``.

:func:`capture_env` returns the dict; :func:`write_env_json` writes ``<run_dir>/env.json`` and
returns its path. :func:`env_json_sha256` hashes the serialized dict so a run RECORD can persist a
single fingerprint string that resolves back to the on-disk ``env.json`` (replacing the old bare
``env_fingerprint`` literal such as ``"synthetic:steps200"``).

Used as a script it writes ``outputs/<run>/env.json``::

    python scripts/capture_env.py --run-dir outputs/myrun --seed 7
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

from src.utils.provenance import env_fingerprint, sha256_obj

__all__ = ["capture_env", "write_env_json", "env_json_sha256"]

ENV_JSON_NAME = "env.json"

#: ``os.environ`` keys that govern deterministic CUDA / hashing (see src/utils/seeding.py).
_DETERMINISM_ENV_KEYS = (
    "CUBLAS_WORKSPACE_CONFIG",
    "PYTHONHASHSEED",
    "CUDA_VISIBLE_DEVICES",
    # 2026-07-26: BLAS THREAD COUNTS. Multi-threaded BLAS changes the ORDER of float reductions,
    # so a training at 4 threads is NOT bit-identical to the same training at 1 — which is why
    # The throughput run plan calls the thread count "part of the determinism envelope"
    # and gates any change behind a pre-freeze ratification. Yet the archive did not record it,
    # so a node running under different OMP settings would have been INVISIBLE to the S6
    # homogeneity audit (the same blindness as the device gap fixed the same day). Record it, so
    # a heterogeneous thread regime is DETECTABLE rather than merely improbable.
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _pip_freeze() -> dict[str, str]:
    """The FULL installed distribution set as ``{name: version}`` (a structured ``pip freeze``).

    Uses :mod:`importlib.metadata` rather than shelling out to ``pip`` so it works in a frozen /
    network-less environment and needs no subprocess. Names are normalised to lower-case so the
    map is stable across distributions that vary the casing of their metadata.
    """
    out: dict[str, str] = {}
    for dist in metadata.distributions():
        try:
            name = dist.metadata["Name"]
        except Exception:  # noqa: BLE001 - a malformed dist must not abort the snapshot
            continue
        if not name:
            continue
        out[str(name).lower()] = dist.version
    return dict(sorted(out.items()))


def _cpu_identity() -> dict[str, Any]:
    """The PROCESSOR the training actually ran on — vendor, model string, and core count.

    NOTE: WHY THIS EXISTS (2026-07-27, launch-gate catch). The confirmatory campaign moved to the CPU
    lane, and on that lane the archive recorded **nothing about the CPU**. ``env_fingerprint``'s only
    platform field is ``platform.platform()`` — kernel and glibc, no processor — and the whole
    device-homogeneity apparatus keys on the GPU: ``integrity._record_device`` reads only
    ``env.json -> nvidia_smi.gpus[0]``, so on a CPU node it returns ``"<absent>"``, which
    ``write_integrity_report`` treats as a WILDCARD. The result was a gate that reports
    ``crn_pair_device_consistent`` GREEN no matter what silicon the pair ran on.

    That is not a hypothetical risk on this cluster. ``src.cluster.lanes.EXCLUDED_CPU_POOLS``
    excludes the ``t`` pool precisely because AMD EPYC (Zen4) selects different oneDNN kernels,
    giving a different float reduction order and **breaking the CRN bit-exactness every paired
    contrast in this design rests on** — and the only thing keeping us off it is an ``-ac allow=``
    token whose effect on the CPU lane had never been submitted, let alone verified. So the one
    failure that would invalidate the headline comparison silently was also the one failure no
    recorded field could reveal.

    The determinism envelope is explicit: *"RECORDED, not merely chosen — a pin
    nobody can verify is FICTIONAL ... If you add a knob that can vary across records, you MUST also
    make it visible in the archive in the same change."* A substrate whose microarchitecture can
    vary across records is exactly such a knob. With this field an Intel/AMD mix becomes a
    detectable, post-hoc-auditable fact instead of an invisible one.

    Best-effort and never raising: ``/proc/cpuinfo`` on Linux (the cluster), ``platform`` elsewhere.
    """
    out: dict[str, Any] = {
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cores": os.cpu_count(),
    }
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                key, _, val = line.partition(":")
                key = key.strip().lower()
                if key == "model name" and "model_name" not in out:
                    out["model_name"] = val.strip()
                elif key == "vendor_id" and "vendor_id" not in out:
                    out["vendor_id"] = val.strip()
                if "model_name" in out and "vendor_id" in out:
                    break
    except OSError:
        pass  # not Linux (the laptop/CI) — platform fields above are the record
    return out


def _nvidia_smi() -> dict[str, Any]:
    """Best-effort GPU driver/name via ``nvidia-smi`` — skipped gracefully if absent (CPU box/CI).

    Returns ``{"available": False, "reason": ...}`` when the binary is missing or errors, so the
    capture never fails on a machine without an NVIDIA GPU.
    """
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": f"{type(exc).__name__}"}
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    drivers = sorted({ln.split(",")[0].strip() for ln in lines})
    return {
        "available": True,
        "driver_version": drivers[0] if len(drivers) == 1 else drivers,
        "gpus": lines,
    }


def _torch_cuda() -> dict[str, Any]:
    """The torch-side CUDA/cuDNN stack + the deterministic-algorithm flag (best-effort).

    torch is OPTIONAL in the analysis/test environment, so an ImportError yields
    ``{"available": False}`` rather than raising.
    """
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - torch absent in the light analysis env
        return {"available": False, "reason": f"{type(exc).__name__}"}
    try:
        cudnn_version = torch.backends.cudnn.version()
    except Exception:  # noqa: BLE001
        cudnn_version = None
    try:
        deterministic = bool(torch.are_deterministic_algorithms_enabled())
    except Exception:  # noqa: BLE001 - older torch lacks the getter
        deterministic = None
    # The ENV vars in _DETERMINISM_ENV_KEYS record what was REQUESTED; these record what torch
    # actually resolved to (a process that calls torch.set_num_threads() overrides the env, and it
    # is the effective value that determines the float reduction order).
    try:
        num_threads = int(torch.get_num_threads())
        num_interop_threads = int(torch.get_num_interop_threads())
    except Exception:  # noqa: BLE001
        num_threads = num_interop_threads = None
    # TF32 — the APPLIED value (deep review, 2026-07-27). By the same principle as the thread counts
    # above: ``agent_numerics.tf32`` is a REGISTERED design parameter and ``freeze.py``'s
    # ``assert_executed_tf32_matches_frozen`` verifies the CONFIG mirrors the register — but nothing
    # recorded what was actually IN FORCE when a training ran, so the pin was verified at the config
    # boundary and unverifiable at the execution boundary. That is the R85 failure mode ("a pin
    # nobody can verify is FICTIONAL") on the single largest precision lever in the stack: TF32
    # rounds matmul inputs to ~10 explicit mantissa bits versus float32's 23, and it dominates
    # because ~97 % of per-training time is the SAC gradient update. It is also process-global and
    # CUDA-ONLY, so it silently differs between the GPU pilots and the CPU campaign lane — recording
    # it is what makes that difference auditable instead of inferred. ``float32_matmul_precision``
    # is the modern API ('highest' | 'high' | 'medium'); the two ``allow_tf32`` booleans are the
    # backend flags ``_apply_tf32`` actually sets.
    try:
        tf32 = {
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        }
    except Exception:  # noqa: BLE001 - older torch lacks one of these getters
        tf32 = {"matmul_allow_tf32": None, "cudnn_allow_tf32": None,
                "float32_matmul_precision": None}
    return {
        "available": True,
        "torch_version": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": cudnn_version,
        "cuda_available": bool(torch.cuda.is_available()),
        "deterministic_algorithms_enabled": deterministic,
        "num_threads": num_threads,
        "num_interop_threads": num_interop_threads,
        **tf32,
    }


def _gold_panel_provenance() -> dict[str, Any]:
    """The ACTIVE gold panel's identity + integrity (C1): its suffix + per-artifact SHA-256s.

    Records ``gold_suffix()`` (the freeze-bound panel selector) and the frozen manifest SHA-256 of
    each production gold parquet (``returns_panel`` / ``cash_features`` / ``top30_selection`` /
    ``splits`` under the active suffix) so every run RECORD names EXACTLY which panel produced it.
    Best-effort: any import/lookup failure yields ``{"available": False, ...}`` so env capture never
    fails on a synthetic-only / minimal install without the gold or manifest.
    """
    try:
        from src.data import loaders
    except Exception as exc:  # noqa: BLE001 - loaders/pandas absent in a minimal env
        return {"available": False, "reason": f"{type(exc).__name__}"}
    suffix = loaders.gold_suffix()
    gold_dir = loaders._GOLD_DIR
    shas: dict[str, str | None] = {}
    for stem in ("returns_panel", "cash_features", "top30_selection", "splits"):
        path = gold_dir / f"{stem}_{suffix}.parquet"
        try:
            shas[stem] = loaders._expected_sha256(path, loaders._MANIFEST)
        except Exception:  # noqa: BLE001 - a manifest read hiccup must not abort capture
            shas[stem] = None
    return {"available": True, "suffix": suffix, "manifest_sha256": shas}


def capture_env(seed: int | None = None) -> dict[str, Any]:
    """Build the full CI-grade environment fingerprint dict (Rank 14).

    Parameters
    ----------
    seed : int | None
        The run seed to record, if known. ``None`` leaves the field ``null``.

    Returns
    -------
    dict
        ``env_fingerprint()`` (the canonical Python/platform/git/tracked-package core) plus the
        enrichment keys ``pip_freeze``, ``nvidia_smi``, ``torch_cuda``, ``determinism_env``,
        ``gold_panel`` (the active suffix + panel SHA-256s, C1), ``seed`` and ``schema``.
    """
    fp = dict(env_fingerprint())  # canonical core — NOT re-implemented here
    # /2 +gold_panel (C1); /3 +APPLIED tf32; /4 +cpu (2026-07-27 — the CPU lane recorded NO
    # processor identity, so an Intel/AMD microarchitecture mix, which breaks CRN bit-exactness,
    # was undetectable by audit; see _cpu_identity).
    fp["schema"] = "capture_env/4"
    fp["seed"] = seed
    fp["pip_freeze"] = _pip_freeze()
    fp["cpu"] = _cpu_identity()
    fp["nvidia_smi"] = _nvidia_smi()
    fp["torch_cuda"] = _torch_cuda()
    fp["determinism_env"] = {k: os.environ.get(k) for k in _DETERMINISM_ENV_KEYS}
    fp["gold_panel"] = _gold_panel_provenance()
    return fp


def env_json_sha256(seed: int | None = None, *, env: dict[str, Any] | None = None) -> str:
    """SHA-256 of the (order-stable) environment fingerprint — a single replayable provenance id.

    Reuses :func:`src.utils.provenance.sha256_obj` (canonical sorted-key JSON) so the digest is the
    same string whether computed here or recomputed from a written ``env.json``.
    """
    return sha256_obj(env if env is not None else capture_env(seed))


def write_env_json(run_dir: str | Path, seed: int | None = None) -> Path:
    """Write ``<run_dir>/env.json`` and return its path (creating ``run_dir`` if needed).

    Wired into the orchestration archive step (``src/orchestration/parallel.py``) so every run dir
    gets an ``env.json`` alongside its ``record.json`` / ``reward.py`` / ``prompt.txt``.
    """
    target = Path(run_dir)
    target.mkdir(parents=True, exist_ok=True)
    env = capture_env(seed)
    path = target / ENV_JSON_NAME
    with path.open("w", encoding="utf-8") as fh:
        json.dump(env, fh, indent=2, sort_keys=True, default=str)
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Capture a CI-grade env.json for run reproducibility.")
    p.add_argument("--run-dir", required=True, help="Run directory to write env.json into.")
    p.add_argument("--seed", type=int, default=None, help="Run seed to record (optional).")
    return p


def main() -> None:
    args = build_parser().parse_args()
    path = write_env_json(args.run_dir, args.seed)
    print(f"[capture_env] wrote {path} (sha256={env_json_sha256(args.seed)[:12]}...)")


if __name__ == "__main__":
    main()
