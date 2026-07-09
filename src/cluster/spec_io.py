"""Task-spec batches for SGE arrays (§12.4 B-A1; content-addressed per §14.2).

A batch directory holds ``task_1.json .. task_N.json`` (1-indexed to match ``$SGE_TASK_ID``)
plus ``index.json`` mapping task number -> the sha256 of its canonical payload. The sha gives
(a) a storage-level identity check on top of the pipeline's own reward-hash replay guard and
(b) no-op resubmission semantics: an identical payload always produces an identical sha, so a
compacted resume batch (§14.1) can be diffed against what already ran.

A task file is either ONE spec dict (pack=1) or a LIST of spec dicts (a §15 GPU pack) — the
on-node ``run_one`` handles both.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

__all__ = ["write_specs", "read_spec", "payload_sha"]

_INDEX = "index.json"


def payload_sha(payload: Any) -> str:
    """sha256 of the canonical (sorted-keys) JSON encoding — the task's content identity."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def write_specs(specs: list[Any], batch_dir: str | Path) -> int:
    """Write ``task_1..N.json`` + ``index.json`` under ``batch_dir``; return N.

    ``specs[i]`` may be a dict (single training) or a list of dicts (a GPU pack). Files are
    written with sorted keys so identical payloads are byte-identical (content addressing).
    """
    out = Path(batch_dir)
    out.mkdir(parents=True, exist_ok=True)
    # V6 audit fix (identity-at-write): every spec must carry the run_id/candidate_id the
    # compacted resume diffs on — catching it here beats a KeyError mid-campaign.
    for i, payload in enumerate(specs, start=1):
        members = payload if isinstance(payload, list) else [payload]
        for m in members:
            if not (isinstance(m, dict) and (m.get("run_id") or m.get("candidate_id"))):
                raise ValueError(
                    f"spec {i} carries no run_id/candidate_id — resume identity is mandatory"
                )
    index: dict[str, str] = {}
    for i, payload in enumerate(specs, start=1):
        sha = payload_sha(payload)
        # STRICT json for the on-disk task file (NO default=str). A non-JSON-serializable field
        # (e.g. a numpy scalar smuggled into a config dict) must FAIL LOUD here on the driver, not
        # be silently coerced to a string that the node then mis-reads. The sha check cannot catch
        # this on its own: payload_sha ALSO uses default=str, so the coerced write and the coerced
        # read+verify agree and the corruption passes verification. Strict-writing is the guard.
        try:
            text = json.dumps(payload, indent=1, sort_keys=True)
        except TypeError as exc:
            raise TypeError(
                f"cluster spec {i} is not cleanly JSON-serializable ({exc}) — every spec must "
                f"round-trip through JSON to the node losslessly; a config value coerced to str "
                f"would silently mis-train. Convert the offending field to a native JSON type."
            ) from exc
        (out / f"task_{i}.json").write_text(text, encoding="utf-8")
        index[str(i)] = sha
    (out / _INDEX).write_text(json.dumps(index, indent=1, sort_keys=True), encoding="utf-8")
    return len(specs)


def read_spec(path: str | Path) -> Any:
    """Load one task file and VERIFY it against the batch index (fail loud on drift).

    Raises ``ValueError`` when the on-disk payload's sha disagrees with ``index.json`` — a
    truncated/tampered spec must never silently train the wrong thing (mirrors the archive's
    sidecar verification philosophy).
    """
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    idx_path = p.parent / _INDEX
    if not idx_path.is_file():
        # V6 audit fix (fail-CLOSED): write_specs always writes the index, so its absence means
        # a partial transfer or a tampered batch — running UNVERIFIED specs is exactly the
        # silent failure mode this layer exists to prevent.
        raise FileNotFoundError(
            f"batch index missing next to {p.name} ({idx_path}) — partial transfer? refusing "
            f"to run an unverifiable spec"
        )
    index = json.loads(idx_path.read_text(encoding="utf-8"))
    task_no = p.stem.split("_")[-1]
    expected = index.get(task_no)
    actual = payload_sha(payload)
    if expected is None or expected != actual:
        raise ValueError(
            f"cluster spec sha MISMATCH for {p.name}: index {str(expected)[:12]}.. != "
            f"on-disk {actual[:12]}.. — refusing to run a drifted spec"
        )
    return payload
