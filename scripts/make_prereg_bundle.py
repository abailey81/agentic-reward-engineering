#!/usr/bin/env python3
"""Emit the external-deposit bundle of the frozen pre-registration (the OSF timestamp artifact).

Why this exists
---------------
The pre-registration's credibility currently rests on a LOCAL SHA-256 + git history, which an
examiner cannot easily verify. Depositing the frozen design on OSF (or minimally a public, dated
GitHub release) gives a THIRD-PARTY timestamp the PDF can cite — converting "the null was
predicted in advance" from trust-my-repo into independently verifiable fact (audit 2026-07-02,
ADR-050 ultraplan). This script packages EXACTLY the hash-bound files (the same eight
``scripts/freeze.py`` hashes) + a manifest + deposit instructions, so the deposit is one upload.

Usage
-----
    python scripts/make_prereg_bundle.py            # -> outputs/prereg_bundle_<hash8>.zip
    python scripts/make_prereg_bundle.py --out X.zip

Run it AT FREEZE TIME (after ``scripts/freeze.py`` writes ``frozen: true``): the bundle embeds
the canonical hash + the recorded freeze state, and the README tells the operator how to deposit.
Running pre-freeze is allowed for a dry-run (the README then carries a loud PRE-FREEZE banner).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.freeze import (  # noqa: E402
    PREREG_MD,
    PREREG_YAML,
    _BOUND_CONFIGS,
    _BOUND_TREATMENT,
    canonical_hash,
    load_yaml,
)


def bundle_members() -> tuple[str, ...]:
    """The hash-bound file set, in the freeze's own fixed order (prose, yaml, configs, treatment)."""
    return (PREREG_MD, PREREG_YAML, *_BOUND_CONFIGS, *_BOUND_TREATMENT)


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def make_bundle(root: Path | None = None, out: Path | None = None) -> Path:
    """Build the deposit zip; returns its path. Fail-loud on any missing bound file."""
    root = root or REPO
    yml = load_yaml(root)
    frozen = bool(yml.get("frozen", False))
    recorded = yml.get("freeze_hash") or None
    digest = canonical_hash(root)
    # 2026-07-13 audit fix: the bundle recorded BOTH hashes but never compared them — a drifted
    # bound file would be silently bundled as "frozen: true" with a mismatching digest. Fail loud.
    if frozen and recorded and digest != recorded:
        raise RuntimeError(
            f"FROZEN-DESIGN DRIFT: canonical hash {digest[:12]}… != recorded freeze_hash "
            f"{str(recorded)[:12]}… — a hash-bound file changed after the freeze. Do NOT bundle; "
            "run scripts/freeze.py --check and investigate."
        )

    members = bundle_members()
    missing = [m for m in members if not (root / m).is_file()]
    if missing:
        raise FileNotFoundError(f"hash-bound file(s) missing, cannot bundle: {missing}")

    manifest = {
        "what": "Frozen pre-registration deposit bundle (agentic-reward-engineering)",
        "canonical_sha256": digest,
        "frozen": frozen,
        "recorded_freeze_hash": recorded,
        "files": {m: _sha256(root / m) for m in members},
        "hash_definition": (
            "canonical_sha256 = scripts/freeze.py::canonical_hash — SHA-256 over the member files "
            "in fixed order, with the prereg yaml's mutable frozen/freeze_hash lines blanked; "
            "recompute with: python scripts/freeze.py --check"
        ),
    }

    banner = "" if frozen else (
        "> **PRE-FREEZE DRY-RUN** — `frozen: false` at bundle time. Re-run this script AFTER\n"
        "> `scripts/freeze.py` records the freeze; only a post-freeze bundle should be deposited.\n\n"
    )
    readme = (
        f"{banner}# Pre-registration deposit bundle\n\n"
        f"Frozen design of the dissertation *LLM-designed reward code under distributional (tail)\n"
        f"feedback for risk-sensitive portfolio RL* — the exact byte content bound by the freeze hash.\n\n"
        f"* canonical SHA-256: `{digest}`\n"
        f"* recorded freeze_hash: `{recorded}`\n"
        f"* frozen: `{frozen}`\n\n"
        "## How to deposit (one-time, at freeze)\n\n"
        "1. Create an OSF project (osf.io) titled with the dissertation title; upload this zip.\n"
        "2. Create an OSF **Registration** of the project (the frozen, timestamped snapshot).\n"
        "3. Record the OSF DOI/URL + date in `DECISIONS.md` and cite it in the PDF (CH4 SS4.8):\n"
        '   "pre-registered and frozen on <date>; OSF <link>; SHA-256 <hash>".\n'
        "4. Alternative/additional anchor: push the `prereg-v1.0` git tag to the public remote and\n"
        "   create a dated GitHub release attaching this zip.\n"
    )

    out = out or (root / "outputs" / f"prereg_bundle_{digest[:8]}.zip")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for m in members:
            zf.write(root / m, arcname=m)
        zf.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
        zf.writestr("DEPOSIT_README.md", readme)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Package the frozen pre-registration for external deposit.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    path = make_bundle(out=args.out)
    print(f"[prereg_bundle] wrote {path} ({path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
