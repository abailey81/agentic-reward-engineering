#!/usr/bin/env python3
"""Content-addressed integrity seal over the campaign RESULT ARCHIVE (2026-07-05).

The archive (per-candidate/per-seed ``record.json`` + reward/prompt sidecars) is the ONE irreplaceable
artifact in this project: results REPLAY from it, they cannot be regenerated (LLM calls are
non-deterministic; the replay-from-archive rule). The freeze hash seals the *design*; the data checksums
seal the *inputs*; nothing sealed the *results*. This module adds that missing seal — a content-addressed
manifest (every record's SHA-256, sorted by ``run_id``) with a single verifiable **root** = the SHA-256 of
the sorted ``run_id\tdigest`` lines. The root is the "results fingerprint": recorded at run end and
re-checked before analysis trusts the archive, so ANY post-hoc corruption / silent edit / dropped or added
record is caught — a tamper-evident reproducibility guarantee, not a vibe.

This is a flat Merkle (a sorted leaf list under one root hash): deterministic, order-independent, and
sufficient to detect any add/remove/modify. A binary tree buys per-leaf inclusion proofs we do not need,
so the flat form is the honest YAGNI choice.

Usage:
    python scripts/archive_integrity.py write  outputs/campaign            # seal at run end
    python scripts/archive_integrity.py verify outputs/campaign            # re-check before analysis
Both are READ-ONLY except ``write``, which emits a single manifest file (default
``<root>/archive_integrity.json``); it never touches a record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_RECORD_NAME = "record.json"
_MANIFEST_NAME = "archive_integrity.json"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def record_digests(root: Path | str) -> dict[str, str]:
    """``{run_id: sha256(record.json bytes)}`` over EVERY record under ``root`` (any depth).

    The digest is over the raw file BYTES (line-ending-normalised to ``\\n`` so a Windows-authored /
    Linux-analysed archive verifies identically — the same invariance the freeze hash guarantees), keyed
    by the record's own ``run_id`` (falling back to the directory path if a record lacks one). Total: an
    unreadable/undecodable record is recorded under a ``__UNREADABLE__:<path>`` key so a corrupted file
    still perturbs the root rather than silently vanishing."""
    root = Path(root)
    out: dict[str, str] = {}
    if not root.is_dir():
        return out
    for rec_path in sorted(root.rglob(_RECORD_NAME)):
        try:
            raw = rec_path.read_bytes()
        except OSError:
            out[f"__UNREADABLE__:{rec_path.relative_to(root).as_posix()}"] = "unreadable"
            continue
        # Line-ending-invariant digest (CRLF <-> LF), matching the freeze-hash convention.
        norm = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest = _sha256_bytes(norm)
        try:
            run_id = str(json.loads(norm.decode("utf-8")).get("run_id") or "")
        except (ValueError, UnicodeDecodeError):
            run_id = ""
        key = run_id or f"__PATH__:{rec_path.parent.relative_to(root).as_posix()}"
        # A duplicate run_id (two dirs, same id) must not silently overwrite: disambiguate by path.
        if key in out:
            key = f"{key}@{rec_path.parent.relative_to(root).as_posix()}"
        out[key] = digest
    return out


def merkle_root(digests: dict[str, str]) -> str:
    """The single verifiable root over the archive: SHA-256 of the sorted ``run_id\\tdigest`` lines.

    Deterministic and order-independent (sorted keys); any add / remove / modify changes the root."""
    lines = "\n".join(f"{k}\t{digests[k]}" for k in sorted(digests))
    return _sha256_bytes(lines.encode("utf-8"))


def build_manifest(root: Path | str) -> dict[str, Any]:
    """The manifest dict: the per-record digests, the count, and the root fingerprint."""
    digests = record_digests(root)
    return {"schema": "archive_integrity/1", "n_records": len(digests),
            "root": merkle_root(digests), "digests": digests}


def write_manifest(root: Path | str, manifest_path: Path | str | None = None) -> Path:
    """Seal the archive: write the manifest (atomically) and return its path. The ONLY write path."""
    root = Path(root)
    path = Path(manifest_path) if manifest_path is not None else root / _MANIFEST_NAME
    manifest = build_manifest(root)
    # The manifest's OWN file is never a record, so it cannot be part of what it seals (rglob matches
    # record.json only) — no self-reference paradox.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


@dataclass
class VerifyResult:
    """The verdict of re-checking a live archive against its sealed manifest."""

    ok: bool
    expected_root: str
    actual_root: str
    added: list[str]     # run_ids present now but not at seal time
    removed: list[str]   # run_ids sealed but now missing
    changed: list[str]   # run_ids whose record bytes differ from the seal
    n_verified: int = 0  # records that matched the seal byte-for-byte

    def summary(self) -> str:
        if self.ok:
            return (f"OK: archive matches the sealed root {self.expected_root[:12]}... "
                    f"({self.n_verified} records verified)")
        parts = [f"MISMATCH: root {self.actual_root[:12]}... != sealed {self.expected_root[:12]}..."]
        if self.changed:
            parts.append(f"{len(self.changed)} CHANGED record(s): {self.changed[:5]}")
        if self.removed:
            parts.append(f"{len(self.removed)} REMOVED: {self.removed[:5]}")
        if self.added:
            parts.append(f"{len(self.added)} ADDED: {self.added[:5]}")
        return " | ".join(parts)


def verify_manifest(root: Path | str, manifest_path: Path | str | None = None) -> VerifyResult:
    """Re-hash the live archive and diff it against the sealed manifest — the verify-before-trust gate."""
    root = Path(root)
    path = Path(manifest_path) if manifest_path is not None else root / _MANIFEST_NAME
    sealed = json.loads(Path(path).read_text(encoding="utf-8"))
    sealed_digests: dict[str, str] = sealed.get("digests", {})
    expected_root = str(sealed.get("root", ""))
    live = record_digests(root)
    actual_root = merkle_root(live)
    added = sorted(k for k in live if k not in sealed_digests)
    removed = sorted(k for k in sealed_digests if k not in live)
    changed = sorted(k for k in live if k in sealed_digests and live[k] != sealed_digests[k])
    ok = (actual_root == expected_root) and not (added or removed or changed)
    n_verified = sum(1 for k in live if k in sealed_digests and live[k] == sealed_digests[k])
    return VerifyResult(ok, expected_root, actual_root, added, removed, changed, n_verified=n_verified)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Content-addressed integrity seal for the result archive.")
    ap.add_argument("mode", choices=["write", "verify", "verify-mirror"],
                    help="write the seal; verify against it (strict); or verify-mirror (a BACKUP "
                         "copy: sealed records must be intact — removed/changed fail — but records "
                         "ADDED after the seal are tolerated, since a mid-campaign mirror lawfully "
                         "carries newer work than the last sealed manifest).")
    ap.add_argument("root", help="Archive root (e.g. outputs/campaign, or the mirror copy).")
    ap.add_argument("--manifest", default=None, help="Manifest path (default <root>/archive_integrity.json).")
    args = ap.parse_args(argv)
    if args.mode == "write":
        path = write_manifest(args.root, args.manifest)
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        print(f"[archive_integrity] sealed {manifest['n_records']} record(s) -> {path}")
        print(f"[archive_integrity] root = {manifest['root']}")
        return 0
    result = verify_manifest(args.root, args.manifest)
    if args.mode == "verify-mirror":
        ok = not (result.removed or result.changed)
        note = f" ({len(result.added)} newer-than-seal record(s) tolerated)" if result.added else ""
        print(f"[archive_integrity] mirror {'OK' if ok else 'CORRUPT'}: "
              f"{result.n_verified} sealed record(s) intact{note}"
              + ("" if ok else f" | removed={result.removed[:5]} changed={result.changed[:5]}"))
        return 0 if ok else 1
    print(f"[archive_integrity] {result.summary()}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
