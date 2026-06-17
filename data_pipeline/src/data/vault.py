"""Layered artifact vault: write-once storage, SHA-256 manifests, provenance
sidecars, verified reads, and the lineage graph (stages 2 & 12; R4).

Guarantees
    * WRITE-ONCE: freezing onto an existing path raises — re-pulls go to NEW
      versioned names plus an ADR (R4). Nothing in this module can overwrite.
    * VERIFIED READS: `read_verified` recomputes SHA-256 and refuses to parse a
      payload whose digest does not match its manifest entry (R4: "code must verify
      checksums before reading").
    * PROVENANCE: every artifact carries a `<name>.provenance.json` sidecar (vendor,
      query, params, timestamps, row hash, library versions) — the byte-exact trace
      from gold back to the API call.
    * LINEAGE: `record_lineage` appends stage edges (outputs <- inputs by sha256);
      `lineage_chain` walks any artifact back to raw.

Layout (config data.platform): data/{raw,staged,clean,gold} + data/manifest/
{checksums.txt (legacy line format), manifest.jsonl (rich), lineage.jsonl}.
"""
from __future__ import annotations

import hashlib
import io
import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd

from ..config import get

ROOT = Path(__file__).resolve().parent.parent.parent
Layer = Literal["raw", "staged", "clean", "gold"]
_CSV_KW = {"index": True}
_FREEZE_LOCK = threading.Lock()    # parallel pull workers freeze concurrently; the
                                   # write-once existence check + manifest appends
                                   # must be atomic per artifact


def layer_dir(layer: Layer) -> Path:
    layers = get("data.platform.layers")
    if layer not in layers:
        raise KeyError(f"unknown layer '{layer}' — config data.platform.layers")
    return ROOT / layers[layer]


def _manifest_jsonl() -> Path:
    return ROOT / get("data.platform.manifest_jsonl")


def _lineage_jsonl() -> Path:
    return ROOT / get("data.platform.lineage_jsonl")


def _legacy_checksums() -> Path:
    return ROOT / get("data.manifest")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def row_hash(df: pd.DataFrame) -> str:
    """Order-sensitive content digest of a frame (canonical CSV bytes)."""
    buf = io.StringIO()
    df.to_csv(buf)
    return sha256_bytes(buf.getvalue().encode())


@dataclass(frozen=True)
class FrozenArtifact:
    path: Path
    relpath: str
    sha256: str
    rows: int
    cols: int


def _serialize(df: pd.DataFrame, fmt: str) -> bytes:
    if fmt == "csv":
        buf = io.StringIO()
        df.to_csv(buf, **_CSV_KW)
        return buf.getvalue().encode()
    if fmt == "parquet":
        import pyarrow  # noqa: F401  (explicit: parquet path requires pyarrow, ADR-013)
        bio = io.BytesIO()
        df.to_parquet(bio)
        return bio.getvalue()
    raise ValueError(f"unsupported format '{fmt}' (csv|parquet)")


def freeze(
    df: pd.DataFrame,
    layer: Layer,
    name: str,
    provenance: dict,
    fmt: str = "csv",
    run_id: str = "adhoc",
) -> FrozenArtifact:
    """Freeze one artifact into a layer. Write-once; manifested; provenance-sidecar'd.

    `provenance` must say where the bytes came from (vendor, query, params; for
    derived layers: the stage and its inputs). It is stored verbatim in the sidecar
    and summarized into manifest.jsonl.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("freeze expects a DataFrame")
    payload = _serialize(df, fmt)               # serialize OUTSIDE the lock (CPU-bound)
    digest = sha256_bytes(payload)
    with _FREEZE_LOCK:
        target_dir = layer_dir(layer)
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / name
        sidecar = path.with_suffix(path.suffix + ".provenance.json")
        if path.exists() or sidecar.exists():
            raise FileExistsError(
                f"{path} exists — layers are write-once (R4); version the filename + ADR"
            )
        path.write_bytes(payload)

        record = {
            # POSIX separators ALWAYS: str() on Windows yields backslashes, which broke
            # exact-relpath matching for univ5/univ4-era entries (consumers survived only
            # via the name fallback in find_entry). Historical lines stay as written
            # (write-once ledger); this fixes forward-going freezes only.
            "relpath": path.relative_to(ROOT).as_posix(),
            "layer": layer,
            "name": name,
            "format": fmt,
            "sha256": digest,
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "row_hash": row_hash(df),
            "frozen_utc": _utcnow(),
            "run_id": run_id,
            "vendor": provenance.get("vendor"),
        }
        # Explicit UTF-8 on every provenance write (deep-review 2026-07-26, loop 1): these records
        # are the reproducibility chain, so their bytes must not depend on the writer's locale.
        sidecar.write_text(json.dumps({**record, "provenance": provenance}, indent=2,
                                      sort_keys=True, default=str), encoding="utf-8")
        _manifest_jsonl().parent.mkdir(parents=True, exist_ok=True)
        with open(_manifest_jsonl(), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        with open(_legacy_checksums(), "a", encoding="utf-8") as f:  # legacy line format
            f.write(f"{digest}  {record['relpath']}  frozen={_utcnow()[:10]}\n")
    return FrozenArtifact(path, record["relpath"], digest, record["rows"], record["cols"])


def manifest_entries() -> list[dict]:
    p = _manifest_jsonl()
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def find_entry(relpath_or_name: str) -> dict | None:
    entries = manifest_entries()
    for e in reversed(entries):                                     # latest wins
        if e["relpath"] == relpath_or_name or e["name"] == relpath_or_name:
            return e
    return None


def read_verified(relpath_or_name: str) -> pd.DataFrame:
    """Checksum-verified read (R4). Raises on missing manifest entry or digest drift."""
    entry = find_entry(relpath_or_name)
    if entry is None:
        raise FileNotFoundError(f"no manifest entry for '{relpath_or_name}' — unmanifested reads are forbidden (R4)")
    path = ROOT / entry["relpath"]
    payload = path.read_bytes()
    digest = sha256_bytes(payload)
    if digest != entry["sha256"]:
        raise ValueError(
            f"CHECKSUM MISMATCH for {entry['relpath']}: manifest {entry['sha256'][:12]}… "
            f"vs on-disk {digest[:12]}… — data has been altered (R4 violation); investigate before use"
        )
    if entry["format"] == "parquet":
        return pd.read_parquet(io.BytesIO(payload))
    df = pd.read_csv(io.BytesIO(payload), index_col=0)
    try:                                     # restore datetime indices lost by CSV
        df.index = pd.to_datetime(df.index, format="ISO8601")
    except (ValueError, TypeError):
        pass                                 # non-date index: keep as parsed
    return df


def record_lineage(output: FrozenArtifact, inputs: list[FrozenArtifact | dict],
                   stage: str, params: dict | None = None) -> None:
    """Append one lineage edge-set: `output` was produced by `stage` from `inputs`."""
    def _ref(a):
        if isinstance(a, FrozenArtifact):
            return {"relpath": a.relpath, "sha256": a.sha256}
        return {"relpath": a["relpath"], "sha256": a["sha256"]}
    _lineage_jsonl().parent.mkdir(parents=True, exist_ok=True)
    with open(_lineage_jsonl(), "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "output": _ref(output), "inputs": [_ref(i) for i in inputs],
            "stage": stage, "params": params or {}, "recorded_utc": _utcnow(),
        }, sort_keys=True, default=str) + "\n")


def lineage_chain(relpath: str) -> list[dict]:
    """Walk the lineage graph from an artifact back to raw (breadth-first edges)."""
    p = _lineage_jsonl()
    if not p.exists():
        return []
    edges = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    chain, frontier, seen = [], {relpath}, set()
    while frontier:
        nxt = set()
        for e in edges:
            if e["output"]["relpath"] in frontier and e["output"]["relpath"] not in seen:
                chain.append(e)
                seen.add(e["output"]["relpath"])
                nxt.update(i["relpath"] for i in e["inputs"])
        frontier = nxt
    return chain
