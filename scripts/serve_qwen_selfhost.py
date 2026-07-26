#!/usr/bin/env python
"""Serve the self-hosted, HF-commit-PINNED, bf16 Qwen leg on a GPU node (the A5 reproducibility-
permanence anchor: the lineage's first fully-pinned open-weight author). vLLM exposes the
OpenAI-compatible /v1/chat/completions surface that ``src/llm/client.py``'s ``vllm_selfhost`` provider
consumes, so NO new transport is needed.

WHY this closes the gap closed / aggregator-served models cannot (Stefan #1/#3; R85):
  * ``--revision <commit>`` makes vLLM download and serve EXACTLY the pinned HF commit -- the pin is
    ENFORCED at serve time, not merely reported by an aggregator whose serving stack / quantization can
    drift silently. This is the first leg in the lineage whose weight pin is enforceable, not advisory.
  * ``--dtype bfloat16`` pins the served numeric variant (recorded, never inferred).
  * ``--reasoning-parser qwen3 --default-chat-template-kwargs {"enable_thinking": false}`` pins thinking
    OFF (the R103 fix: Qwen3 defaults to thinking-ON, which spends the whole output budget on hidden
    reasoning -> EMPTY authored code) AND separates reasoning so the archived response PROVES it
    (``reasoning_tokens == 0``), closing the R85 round-trip. The client sends the request-level override
    too (belt-and-suspenders; request-level wins over the server default).
  * a ``served-manifest.json`` records the enforced commit + dtype + vLLM/torch/GPU versions -- the
    serving-NODE provenance the driver-side ``capture_env`` cannot see.

Usage (on the GPU node):
    python -m scripts.serve_qwen_selfhost --leg qwen3.5-9b --port 8000   # serves Qwen/Qwen3.5-9B@<pin> in bf16
then, for the campaign driver:  export VLLM_BASE_URL=http://<node>:8000/v1  VLLM_API_KEY=<key>
``--dry-run`` prints the exact command + writes the manifest WITHOUT launching (CI / off-GPU safe).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

#: vLLM's ``--dtype`` spellings (the leg records the short form in ``quantizations``).
_DTYPE_ALIASES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


def hf_snapshot_dir(hf_home: str | Path, repo: str, commit: str) -> Path:
    """Where the HuggingFace cache stores EXACTLY this repo@commit.

    Pure path arithmetic over the documented cache layout
    (``<HF_HOME>/hub/models--<org>--<name>/snapshots/<commit>``) so the preflight below needs no
    ``huggingface_hub`` import and is unit-testable with no network and no GPU.
    """
    return (Path(hf_home) / "hub" / ("models--" + str(repo).replace("/", "--"))
            / "snapshots" / str(commit))


def preflight_weights(hf_home: str | Path | None, repo: str, commit: str) -> tuple[bool, str]:
    """Are the PINNED weights already on disk? ``(ok, actionable message)``.

    THE FAILURE THIS EXISTS TO PREVENT (2026-07-26). Myriad compute nodes have **no internet**, and
    `vllm serve <repo> --revision <commit>` tries to DOWNLOAD when the revision is not cached. On a
    GPU node that means: the scarce allocation is granted, the job starts, vLLM stalls then dies on a
    network error, and the log blames vLLM rather than the real cause. The weights (~18 GB at bf16)
    must therefore be staged on a LOGIN node first (``--prestage``) into a SHARED filesystem, and the
    serve must run with ``HF_HUB_OFFLINE=1``.

    Checking cheaply and failing LOUDLY here converts a wasted GPU allocation into an instant,
    self-explaining error — the same discipline as the jobscript's apptainer-presence guard, which
    exists because a missing ``.sif`` burned a granted slot with a bare ``rc=127``.
    """
    if not hf_home:
        return False, ("HF_HOME is unset. Myriad compute nodes have no internet, so vLLM cannot "
                       "download the pinned revision at serve time. Set HF_HOME to a SHARED path "
                       "and pre-stage on a login node:\n"
                       "    export HF_HOME=$HOME/Scratch/hf\n"
                       "    python -m scripts.serve_qwen_selfhost --prestage --leg <leg>")
    snap = hf_snapshot_dir(hf_home, repo, commit)
    if snap.is_dir() and any(snap.iterdir()):
        return True, f"pinned weights present: {snap}"
    return False, (f"pinned weights NOT staged for {repo}@{commit[:12]} (expected {snap}).\n"
                   "Run this ON A LOGIN NODE (which has internet), then resubmit:\n"
                   f"    export HF_HOME={hf_home}\n"
                   f"    python -m scripts.serve_qwen_selfhost --prestage --leg <leg>\n"
                   "Serving without it would burn the GPU allocation on a download that cannot "
                   "succeed on a compute node.")


#: bf16 bytes per parameter, and the headroom vLLM needs beyond raw weights (KV cache + activations
#: + its ~0.9 default `gpu_memory_utilization`). Deliberately conservative: refusing a serve that
#: WOULD have fitted costs one resubmission, while attempting one that cannot fit costs the whole
#: granted allocation and produces an OOM traceback that blames vLLM rather than the GPU class.
_BYTES_PER_PARAM_BF16 = 2
_VRAM_HEADROOM = 1.25


def required_vram_gb(param_count_b: float = 9.0) -> float:
    """GB of VRAM the pinned bf16 model needs, with serving headroom."""
    return param_count_b * _BYTES_PER_PARAM_BF16 * _VRAM_HEADROOM


def preflight_vram(total_vram_gb: float | None, param_count_b: float = 9.0) -> tuple[bool, str]:
    """Will the pinned bf16 weights actually FIT on the GPU this job was placed on?

    THE FAILURE THIS PREVENTS. Myriad's default EF pool is `2x V100` at **16G or 32G** and the class
    is not verified until the job lands (dossier §pools). A 9B model in bf16 is ~18 GB, so a 16 GB
    V100 cannot hold it — vLLM would die with a CUDA OOM *after* the scarce GPU allocation was
    granted, and the log would blame vLLM rather than the GPU class. The A100-80G U/V pools were
    measured LESS contended for us than EF (probe_u/probe_v both placed while the EF control was
    still queued), so they are both the safe and the fast choice for this serve.
    """
    need = required_vram_gb(param_count_b)
    if total_vram_gb is None:
        return True, (f"GPU VRAM not detectable — proceeding, but this serve needs ~{need:.0f} GB "
                      "and a 16 GB V100 cannot hold it")
    if total_vram_gb + 1e-9 < need:
        return False, (f"GPU has {total_vram_gb:.0f} GB VRAM but the pinned bf16 weights need "
                       f"~{need:.0f} GB. Resubmit onto an A100 pool, e.g. "
                       "`qsub -ac allow=U ...` (A100-80G; U/V measured less contended than EF), "
                       "or `-ac allow=L` (A100-40G). EF may place a 16 GB V100, which cannot "
                       "load this model.")
    return True, f"GPU VRAM {total_vram_gb:.0f} GB >= the ~{need:.0f} GB the bf16 weights need"


def _detect_vram_gb() -> Optional[float]:  # pragma: no cover (GPU node)
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:  # noqa: BLE001
        pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30)
        if out.returncode == 0 and out.stdout.strip():
            return float(out.stdout.strip().splitlines()[0]) / 1024.0
    except Exception:  # noqa: BLE001
        pass
    return None


def prestage(repo: str, commit: str) -> int:  # pragma: no cover (network; run on a login node)
    """Download EXACTLY the pinned revision into HF_HOME. Login-node step; needs internet."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("--prestage needs huggingface_hub (available inside the vLLM image / on "
                         "the login node): pip install huggingface_hub") from None
    path = snapshot_download(repo_id=repo, revision=commit)
    print(f"[serve_qwen_selfhost] staged {repo}@{commit[:12]} -> {path}", file=sys.stderr)
    return 0


def build_serve_command(
    repo: str, commit: str, dtype: str, served_name: str, port: int,
    *, enable_thinking: bool, api_key: Optional[str] = None,
) -> list[str]:
    """The exact ``vllm serve`` argv that ENFORCES the pin (``--revision``) + dtype + thinking state.

    Pure + deterministic so it is unit-tested without a GPU; the enforcement (not merely the recording)
    of the commit is what makes this leg's reproducibility real rather than advisory (R85).
    """
    dtype_arg = _DTYPE_ALIASES.get(str(dtype).lower(), str(dtype))
    cmd = [
        "vllm", "serve", str(repo),
        "--revision", str(commit),          # ENFORCE the pinned weights -- the first enforceable pin (R85)
        "--dtype", dtype_arg,
        "--served-model-name", str(served_name),
        "--reasoning-parser", "qwen3",      # separate reasoning so reasoning_tokens PROVES thinking-off
        "--default-chat-template-kwargs", json.dumps({"enable_thinking": bool(enable_thinking)}),
        "--port", str(int(port)),
    ]
    if api_key:
        cmd += ["--api-key", str(api_key)]
    return cmd


def build_manifest(
    repo: str, commit: str, dtype: str, served_name: str, port: int,
    *, enable_thinking: bool, versions: dict[str, Any],
) -> dict[str, Any]:
    """The served-manifest: serving-NODE provenance the driver-side ``capture_env`` cannot capture."""
    return {
        "leg": str(served_name),
        "served_model_name": str(served_name),
        "hf_repo": str(repo),
        "hf_commit": str(commit),           # the ENFORCED pin (--revision)
        "served_dtype": str(dtype),
        "reasoning_enabled": bool(enable_thinking),
        "port": int(port),
        "vllm_version": versions.get("vllm"),
        "torch_version": versions.get("torch"),
        "gpu": versions.get("gpu"),
    }


def _versions() -> dict[str, Any]:
    """Best-effort serving-stack versions (absence recorded as null, never fabricated)."""
    v: dict[str, Any] = {"vllm": None, "torch": None, "gpu": None}
    try:
        import vllm  # type: ignore
        v["vllm"] = getattr(vllm, "__version__", None)
    except Exception:  # noqa: BLE001 -- provenance is best-effort; a missing serving dep is recorded null
        pass
    try:
        import torch  # type: ignore
        v["torch"] = torch.__version__
        if torch.cuda.is_available():
            v["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    return v


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Serve the pinned bf16 Qwen self-host leg (A5).")
    p.add_argument("--leg", default="qwen3.5-9b",
                   help="leg in config/legs.yaml whose hf_pin (repo@commit) weights to self-host (bf16)")
    p.add_argument("--dtype", default="bfloat16",
                   help="served numeric variant: bf16 self-hosts the ACTUAL weights, distinct from an "
                        "aggregator leg's fp8 -- this is what lets us claim the bf16 weights authored")
    p.add_argument("--served-model-name", dest="served_model_name", default=None,
                   help="the vLLM served id the client requests (default: the leg label)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--manifest", default=None, help="where to write served-manifest.json")
    p.add_argument("--dry-run", action="store_true",
                   help="print the command + write the manifest; do NOT launch (CI / off-GPU safe)")
    p.add_argument("--prestage", action="store_true",
                   help="LOGIN-NODE step: download exactly the pinned revision into HF_HOME, so the "
                        "GPU node (which has no internet) can serve it offline")
    p.add_argument("--skip-preflight", action="store_true",
                   help="serve even if the pinned weights look unstaged (escape hatch; the default "
                        "refuses, because a download cannot succeed on a compute node)")
    args = p.parse_args(argv)

    from src.llm.legs import leg_by_label
    # A reproducibility DEMONSTRATION (not a full-loop campaign leg): self-host the SAME hash-pinned
    # weights an existing leg records, at bf16 -- so it never enters the 10-leg model_suite roster and
    # cannot dilute the R101 lockstep seed rung (which would cost confirmatory power). freeze stays green.
    leg = leg_by_label(args.leg)
    hf = leg.get("hf_pin") or {}
    repo, commit = hf.get("repo"), hf.get("commit")
    if not repo or not commit:
        raise SystemExit(f"leg {args.leg!r} needs hf_pin.repo + hf_pin.commit (R85: the enforceable pin)")
    dtype = args.dtype
    enable_thinking = bool((leg.get("reasoning") or {}).get("enabled", False))

    import os
    if args.prestage:
        return prestage(repo, commit)
    api_key = os.environ.get("VLLM_API_KEY")
    served_name = args.served_model_name or args.leg

    cmd = build_serve_command(repo, commit, dtype, served_name, args.port,
                              enable_thinking=enable_thinking, api_key=api_key)
    manifest = build_manifest(repo, commit, dtype, served_name, args.port,
                              enable_thinking=enable_thinking, versions=_versions())
    manifest_path = Path(args.manifest) if args.manifest else Path(f"served-manifest-{args.leg}.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[serve_qwen_selfhost] manifest -> {manifest_path}", file=sys.stderr)
    print(f"[serve_qwen_selfhost] command : {' '.join(cmd)}", file=sys.stderr)
    if args.dry_run:
        return 0
    # PREFLIGHT before we consume the GPU allocation: refuse to start a serve whose weights are not
    # already on disk, because on a compute node the download it would attempt cannot succeed.
    ok, why = preflight_weights(os.environ.get("HF_HOME"), repo, commit)
    print(f"[serve_qwen_selfhost] preflight weights: {why}", file=sys.stderr)
    vram_ok, vram_why = preflight_vram(_detect_vram_gb())
    print(f"[serve_qwen_selfhost] preflight vram   : {vram_why}", file=sys.stderr)
    if not (ok and vram_ok) and not args.skip_preflight:
        raise SystemExit(2)
    return subprocess.call(cmd)  # pragma: no cover (GPU-node launch)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
