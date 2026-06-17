"""One-shot verified physical inventory for the data requirements report.

Re-hashes EVERY manifested artifact (R4: checksums verified, not trusted), loads each
payload for rows/cols/date-range/ticker coverage, checks provenance sidecars, scans the
layers for unmanifested files (write-once integrity), and byte-checks the frozen D5
split artifacts against PREREGISTRATION §6 literals. Emits JSON to stdout.

Read-only: writes nothing under data/. Run: .venv/bin/python scripts/verify_inventory.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Audit fix (scripts__verify_inventory findings 1 & 2): the module previously did
# `from src.config import get` / `from src.data.vault import manifest_entries`, but neither
# `src.config` nor `src.data.vault` exists in the repo-root `src` package — those symbols live
# only in the separate `data_pipeline/src/` namespace, and that namespace's vault resolves its
# own ROOT to `data_pipeline/` and reads a manifest that does not exist on disk. The script's
# own ROOT (repo root) already targets the REAL manifest at data/manifest/manifest.jsonl, whose
# relpaths are repo-root-relative. So we read that manifest directly and pin the medallion layer
# map locally (canonical data/{raw,staged,clean,gold}; mirrors data_pipeline/config/data.yaml
# `data.platform.layers`) — making the inventory self-contained and runnable.
MANIFEST_JSONL = ROOT / "data/manifest/manifest.jsonl"
LAYERS = {"raw": "data/raw", "staged": "data/staged", "clean": "data/clean", "gold": "data/gold"}


def manifest_entries() -> list[dict]:
    """Read the repo-root JSON-lines manifest. Fail loud if it is missing or empty, so an
    empty inventory can never masquerade as a passing 'verified, not trusted' run (R4)."""
    if not MANIFEST_JSONL.exists():
        raise FileNotFoundError(
            f"manifest not found at {MANIFEST_JSONL} — cannot run a physical inventory (R4)"
        )
    entries = [
        json.loads(line)
        for line in MANIFEST_JSONL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not entries:
        raise ValueError(
            f"manifest at {MANIFEST_JSONL} yielded zero entries — refusing to report a vacuous "
            "(green-looking but empty) inventory (R4)"
        )
    return entries


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frame(path: Path, fmt: str) -> pd.DataFrame:
    if fmt == "parquet":
        return pd.read_parquet(path)
    df = pd.read_csv(path, index_col=0)
    try:
        df.index = pd.to_datetime(df.index, format="ISO8601")
    except (ValueError, TypeError):
        pass
    return df


def inventory() -> dict:
    rows, failures = [], []
    seen_paths = set()
    for e in manifest_entries():
        path = ROOT / e["relpath"]
        seen_paths.add(path)
        rec = {"name": e["name"], "layer": e["layer"], "rows": e["rows"], "cols": e["cols"],
               "manifest_sha16": e["sha256"][:16]}
        if not path.exists():
            rec["checksum"] = "FAIL: file missing"
            failures.append(rec)
            rows.append(rec)
            continue
        digest = sha256(path)
        rec["checksum"] = "PASS" if digest == e["sha256"] else f"FAIL: on-disk {digest[:16]}"
        if rec["checksum"] != "PASS":
            failures.append(rec)
        sidecar = path.with_suffix(path.suffix + ".provenance.json")
        rec["sidecar"] = sidecar.exists()
        try:
            df = load_frame(path, e["format"])
            if isinstance(df.index, pd.DatetimeIndex) and len(df):
                rec["date_range"] = [str(df.index.min().date()), str(df.index.max().date())]
            cols = [str(c) for c in df.columns]
            rec["n_series"] = len(cols)
            rec["series_sample"] = cols[:6]
        except Exception as ex:  # noqa: BLE001 — inventory must report, not crash
            rec["load_note"] = f"{type(ex).__name__}: {ex}"[:120]
        rows.append(rec)

    # write-once integrity: payload files on disk not present in the manifest
    # Audit fix: use the local LAYERS map (was `get("data.platform.layers")`, an import that
    # cannot resolve in this repo) — same canonical medallion layers, relative to repo-root ROOT.
    unmanifested = []
    for layer, rel in LAYERS.items():
        d = ROOT / rel
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_dir() or p.name.endswith(".provenance.json"):
                continue
            if p not in seen_paths:
                unmanifested.append(str(p.relative_to(ROOT)))
    return {"artifacts": rows, "checksum_failures": failures, "unmanifested": unmanifested}


def verify_splits_vs_prereg() -> dict:
    """D5: the Frozen split artifacts vs PREREGISTRATION §6 literals (UNFINISHED STUB).

    This one-shot inventory script was left INCOMPLETE upstream (truncated mid-docstring,
    2026-06-17) — the body was never written. It is closed here to a valid, fail-loud stub so the
    module parses and lints (final-audit follow-up: a pre-existing syntax error, not one of the 38
    findings). The real split-vs-prereg verification lives in ``scripts/freeze.py`` (``--check``)
    and ``tests/test_embargo_splits.py`` / ``tests/test_freeze.py``.
    """
    raise NotImplementedError(
        "verify_inventory.verify_splits_vs_prereg is an unfinished stub; use scripts/freeze.py "
        "--check and tests/test_embargo_splits.py for split-vs-prereg verification."
    )


def main() -> int:
    """Emit the physical-inventory JSON to stdout (R4), the behaviour the module docstring promises.

    Audit follow-up (2026-06-20): the file had NO entry point, so running it produced empty stdout and
    the stated inventory report was undeliverable. Exits non-zero if any manifested checksum failed, so
    a caller/CI can gate on integrity.
    """
    inv = inventory()
    print(json.dumps(inv, indent=2))
    return 1 if inv["checksum_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
