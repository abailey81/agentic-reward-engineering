"""Per-run results schema and the canonical results loader (FINAL_PLAN F.13, audit C-1).

This module defines the per-run record schema and provides the ONLY interface that
analysis code may use to read results (audit C-1: never parse run files ad hoc).
Every campaign run writes exactly one record under ``<root>/<run_id>/record.json``;
every table, figure and inference step reads it back through :func:`load_run` /
:func:`load_all`. Centralising IO here guarantees the provenance fields (hashes,
seeds, environment fingerprint) are always present and that the loader fails loudly
when a required field is missing.

If a record carries a ``reward_source`` string, it is archived alongside the record
as ``reward.py`` so results *replay from the archive* and are never regenerated
(audit C-2): the LLM that produced the reward is non-deterministic. Likewise a
``prompt`` is archived as ``prompt.txt`` (Rank 14) so the prompt that produced the
reward is never lost (the "archive every prompt" rule).

The self-contained ``record.json`` is the authoritative copy (it carries the embedded
``reward_source``/``prompt`` so analysis never depends on a separate file). The
``reward.py``/``prompt.txt`` sidecars are the on-disk archive: :func:`load_run` VERIFIES
each sidecar byte-for-byte against the embedded copy and fails loudly on any mismatch, so
the "replay from the archive" guarantee is enforced — a tampered sidecar cannot pass
silently — rather than merely asserted (symmetric with the env.json integrity check).

Audit refs: C-1 (analysis reads only through this module), C-2 (replay from archive).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

__all__ = ["REQUIRED_FIELDS", "OPTIONAL_FIELDS", "write_run", "load_run", "load_all"]

#: Fields every persisted run record must contain. :func:`write_run` validates the
#: record against this set and raises ``KeyError`` naming the first missing field.
REQUIRED_FIELDS: tuple[str, ...] = (
    "run_id",
    "arm",
    "seed",
    "fold",
    "candidate_id",
    "generation",
    "reward_source_hash",
    "feedback_block",
    "metrics",
    "wall_clock",
    "env_fingerprint",
)

#: Additive, OPTIONAL provenance fields a record MAY carry (Rank 2/3). They are
#: documented here so writers/readers share one vocabulary, but they are deliberately
#: NOT in :data:`REQUIRED_FIELDS`: adding them as required would break every existing
#: writer (the loop, the search arms, ``run_prototype``) and the loader round-trip
#: tests. The headline campaign (``scripts/run_campaign.py``) writes the frozen-winner
#: test record with these set; the realized per-step test-return vector and per-period
#: P&L additionally ride inside ``metrics`` (where ``analyze_results`` already reads
#: ``metrics['val_returns']``) so Rank 3's PBO can consume them back-compatibly.
#:
#:  - ``frozen``            : ``True`` on a sealed frozen-winner record.
#:  - ``test_returns``      : realized per-step held-out (2020-2026) NET return vector.
#:  - ``per_period_pnl``    : per-period P&L vector aligned to ``test_returns``.
#:  - ``reward_source``     : the winner's reward source (already archived as reward.py).
#:  - ``prompt``            : (Rank 14) the rendered LLM user prompt that produced the candidate.
#:                           ``write_run`` dumps it to a ``prompt.txt`` sidecar (next to reward.py)
#:                           and ``load_run`` verifies it byte-for-byte against the embedded copy, so
#:                           results *replay from the archive* ("archive every prompt";
#:                           audit C-2 — a tampered sidecar raises ``ValueError``). The LLM arms
#:                           (``src/llm/loop.py`` + the parallel/run_prototype writers) set it;
#:                           the search arms and sealed-test records omit it (no sidecar written).
#:
#: Rank 15 (cost-sweep) additively persists the per-step GROSS and TURNOVER decomposition of
#: the frozen-winner test path inside ``metrics`` (``metrics['test_gross']`` +
#: ``metrics['test_turnover']``, mirroring how ``metrics['test_returns']`` rides). These let
#: ``scripts/cost_sweep.py`` ANALYTICALLY re-price the FROZEN winner at every ``costs.grid_bps``
#: level without retraining (``net_c = gross - c*1e-4 * turnover``), since the realized gross
#: and turnover do not depend on the cost level (audit C-5: the cost is charged AFTER the
#: action). They are OPTIONAL (a record lacking them just triggers the re-roll fallback), so
#: the schema stays back-compatible with every existing writer.
#: Rank 3 (PBO/CSCV) additionally relies on a per-CANDIDATE per-period VALIDATION return
#: vector riding inside ``metrics['val_returns']``. ALL SIX arms now write it during
#: search: the LLM arms via ``src/llm/loop.py``, the search arms via
#: ``scripts/run_prototype.py`` (sequential) and ``src/orchestration/parallel.py``
#: (parallel). ``scripts/analyze_campaign.py`` stacks those vectors per arm into the
#: CSCV performance matrix. It is OPTIONAL (a record that lacks it is skipped with a
#: logged warning) so the schema stays back-compatible with every existing writer.
#:
#: 2026-07-26 (M1/M1b/M2/M3) — report-only, determinism-safe RUN DIAGNOSTICS additionally ride inside
#: ``metrics`` on the frozen-winner TEST record (same pattern as ``test_gross``; no schema change), so the
#: replay-only campaign records the data the equity/drawdown/exposure/allocation/learning-curve figures
#: need (a frozen run can only plot what it logged):
#:   - ``metrics['test_exposure']``   : per-step Herfindahl / effective-N / max / top-5 concentration (M1)
#:   - ``metrics['test_alloc']``      : top-K-by-mean monthly allocation snapshots for the heatmap (M1b)
#:   - ``metrics['test_components']`` : per-component mean of the reward decomposition over the test path (M3)
#:   - ``metrics['train_curve']``     : this seed's downsampled training curve (critic/actor loss + return, M2)
#: All are OPTIONAL + best-effort (a capture miss never crashes an irreplaceable sealed seed).
OPTIONAL_FIELDS: tuple[str, ...] = (
    "frozen",
    "test_returns",
    "per_period_pnl",
    "reward_source",
    "prompt",
    # NB (final-audit #38): the unused ``prompt_hash`` was removed — no writer ever set it and no
    # reader consumed it, so it falsely implied a hash-only prompt mode. The full prompt.txt sidecar
    # is the prompt provenance; integrity is via that text, not a separate hash.
)

_RECORD_NAME = "record.json"
_REWARD_NAME = "reward.py"
_PROMPT_NAME = "prompt.txt"


def _validate(record: dict[str, Any]) -> None:
    """Raise ``KeyError`` naming the first missing required field."""
    for field in REQUIRED_FIELDS:
        if field not in record:
            raise KeyError(f"run record is missing required field {field!r}")


def write_run(record: dict[str, Any], root: str | Path) -> Path:
    """Validate and persist a single per-run record.

    Parameters
    ----------
    record : dict
        Mapping containing at least every field in :data:`REQUIRED_FIELDS`.
        If it carries a ``reward_source`` string, that source is written next to
        the record as ``reward.py``.
    root : str | Path
        Directory under which a ``<run_id>/`` subdirectory is created.

    Returns
    -------
    Path
        Path to the written ``record.json``.

    Raises
    ------
    KeyError
        If any field in :data:`REQUIRED_FIELDS` is absent (named in the message).
    """
    _validate(record)
    run_dir = Path(root) / str(record["run_id"])
    run_dir.mkdir(parents=True, exist_ok=True)

    # SIDECARS FIRST, each durably flushed+fsynced (F2, review 2026-07-02): the record.json
    # atomic replace below is the COMMIT POINT of the run dir — once a record is present, ``load_run``
    # VERIFIES the reward.py / prompt.txt sidecars byte-for-byte against it, and the sidecar-mismatch
    # ``ValueError`` PROPAGATES by design (provenance integrity). Writing the sidecars AFTER the commit
    # left a window where a crash produced a committed record with a MISSING/TRUNCATED sidecar — which
    # bricked EVERY subsequent ``--resume`` (``load_all`` -> ``load_run`` raises on that one dir).
    # Sidecars-first means a crash leaves either NO new record (the old state stands; a fresh dir with
    # orphan sidecars has no record.json and is skipped by ``load_all``) or a COMPLETE, verifiable set.
    # ``open("w")+flush+os.fsync`` (not ``write_text``) so the sidecar bytes are durable BEFORE the
    # record commits — an fsynced record must never point at un-fsynced sidecars.
    reward_source = record.get("reward_source")
    if reward_source is not None:
        with (run_dir / _REWARD_NAME).open("w", encoding="utf-8") as fh:
            fh.write(str(reward_source))
            fh.flush()
            os.fsync(fh.fileno())

    # Rank 14: if the record carries the rendered prompt, archive it next to reward.py as a
    # prompt.txt sidecar so results *replay from the archive* — the LLM that produced the reward is
    # non-deterministic and "archive every prompt" is a standing rule (audit C-2).
    # OPTIONAL: a record without ``prompt`` (the search arms, sealed test records) writes no sidecar.
    prompt = record.get("prompt")
    if prompt is not None and str(prompt) != "":
        with (run_dir / _PROMPT_NAME).open("w", encoding="utf-8") as fh:
            fh.write(str(prompt))
            fh.flush()
            os.fsync(fh.fileno())

    # ATOMIC write (campaign-robustness §F): a long campaign WILL be interrupted (Ctrl-C, OS sleep,
    # thermal trip, power loss). A plain ``open("w") + json.dump`` killed mid-write leaves a TRUNCATED
    # ``record.json`` that ``load_run`` (``json.load``) then fails to parse — and because ``load_all``
    # calls ``load_run`` on every dir, a single truncated record CRASHES the whole ``--resume``. We
    # therefore write to a temp sibling, flush+fsync so the bytes are durable, then ``os.replace`` —
    # which is atomic within a directory on BOTH Windows and POSIX. A crash now leaves either the OLD
    # record or the COMPLETE new one, never a half. ``load_all`` looks for exactly ``record.json`` so a
    # stray ``.json.tmp`` left by a crash mid-write is simply ignored (no cleanup needed).
    record_path = run_dir / _RECORD_NAME
    tmp_path = record_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, sort_keys=True, default=str)
        fh.flush()
        os.fsync(fh.fileno())  # durability: the bytes hit disk before the rename
    os.replace(tmp_path, record_path)  # atomic on Windows + POSIX (same directory) — the COMMIT POINT

    return record_path


def load_run(run_id: str, root: str | Path) -> dict[str, Any]:
    """Load a single per-run record by id.

    Parameters
    ----------
    run_id : str
        Identifier of the run to load.
    root : str | Path
        Directory containing the ``<run_id>/`` subdirectory.

    Returns
    -------
    dict
        The validated record. If a ``reward.py`` / ``prompt.txt`` sidecar is present
        it is verified byte-for-byte against the embedded ``reward_source`` / ``prompt``
        (or reattached when the record carries no embedded copy).

    Raises
    ------
    FileNotFoundError
        If the run record does not exist.
    KeyError
        If the loaded record fails schema validation.
    ValueError
        If a ``reward.py`` / ``prompt.txt`` / ``env.json`` sidecar exists but does not
        match the record (provenance integrity; audit C-2).
    """
    run_dir = Path(root) / str(run_id)
    record_path = run_dir / _RECORD_NAME
    if not record_path.is_file():
        raise FileNotFoundError(f"no run record at {record_path}")
    with record_path.open(encoding="utf-8") as fh:
        record = json.load(fh)
    _validate(record)

    # audit C-2 (results "replay from the archive"): make the reward.py sidecar LOAD-BEARING, not
    # dead duplication. ``write_run`` embeds ``reward_source`` in record.json AND writes the sidecar,
    # so on every normal record both exist. When both are present we VERIFY the on-disk sidecar is
    # byte-identical to the embedded copy and raise ``ValueError`` on mismatch — mirroring the env.json
    # hash check below — so a tampered reward.py is caught on the canonical read path instead of being
    # silently ignored. When only the sidecar exists (a hand-placed record with no embedded copy) we
    # reattach it, preserving the original round-trip contract.
    reward_path = run_dir / _REWARD_NAME
    if reward_path.is_file():
        sidecar_source = reward_path.read_text(encoding="utf-8")
        embedded_source = record.get("reward_source")
        if embedded_source is None:
            record["reward_source"] = sidecar_source
        elif str(embedded_source) != sidecar_source:
            raise ValueError(
                f"reward.py sidecar mismatch for run {run_id}: archived reward.py differs from the "
                f"reward_source embedded in record.json (provenance integrity, audit C-2)"
            )

    # Rank 14: the prompt.txt sidecar is verified symmetrically with reward.py so the archived prompt
    # that replays the candidate cannot silently diverge from the embedded copy.
    prompt_path = run_dir / _PROMPT_NAME
    if prompt_path.is_file():
        sidecar_prompt = prompt_path.read_text(encoding="utf-8")
        embedded_prompt = record.get("prompt")
        if embedded_prompt is None:
            record["prompt"] = sidecar_prompt
        elif str(embedded_prompt) != sidecar_prompt:
            raise ValueError(
                f"prompt.txt sidecar mismatch for run {run_id}: archived prompt.txt differs from the "
                f"prompt embedded in record.json (provenance integrity, audit C-2)"
            )

    # final-audit #37: if an env.json snapshot sits in this run dir and the record's env_fingerprint
    # carries its content hash, VERIFY it and reattach (symmetric with reward.py/prompt.txt, which are
    # verified by reattachment) so a tampered/missing env snapshot is caught on the canonical read
    # path. Uses the SAME sha256_obj canonicalisation capture_env.env_json_sha256 used, so the digest
    # matches exactly. A normal candidate record (no env.json in its dir) is a no-op.
    env_json_path = run_dir / "env.json"
    fp = record.get("env_fingerprint")
    if not env_json_path.is_file() and isinstance(fp, dict) and fp.get("env_json_sha256"):
        # 2026-07-06: the SERIAL search path (and the test leg's per-arm capture) write ONE shared
        # snapshot at <root>/_env/env.json rather than per run dir — without this fallback the
        # env-hash verification was a permanent no-op for every record pointing at a shared
        # snapshot (the digest was checkable only by hand).
        shared = Path(root) / "_env" / "env.json"
        if shared.is_file():
            env_json_path = shared
    if env_json_path.is_file() and isinstance(fp, dict) and fp.get("env_json_sha256"):
        from src.utils.provenance import sha256_obj

        with env_json_path.open(encoding="utf-8") as ef:
            env_obj = json.load(ef)
        actual = sha256_obj(env_obj)
        if actual != fp["env_json_sha256"]:
            raise ValueError(
                f"env.json sha256 mismatch for run {run_id}: record "
                f"{str(fp['env_json_sha256'])[:12]}.. != on-disk {actual[:12]}.. (provenance integrity)"
            )
        record.setdefault("env", env_obj)
    return record


def load_all(
    root: str | Path, filter: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Load all per-run records under ``root``, optionally filtered.

    Parameters
    ----------
    root : str | Path
        Directory containing per-run subdirectories.
    filter : dict | None
        Optional mapping of field -> required value; only records matching every
        constraint are returned (e.g. ``{"arm": "scalar"}``).

    Returns
    -------
    list[dict]
        Validated records matching the filter (all records when ``filter`` is
        ``None``), ordered by ``run_id``.
    """
    root_path = Path(root)
    records: list[dict[str, Any]] = []
    if not root_path.is_dir():
        return records
    for run_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
        if not (run_dir / _RECORD_NAME).is_file():
            continue
        try:
            record = load_run(run_dir.name, root_path)
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            # FAIL LOUD with an ACTIONABLE message (2026-07-06 A4). Deliberately NOT auto-quarantined:
            # an LLM-arm record is irreplaceable (a non-deterministic author would regenerate a
            # DIFFERENT candidate — silently changing the study), and the mirror usually still holds
            # the intact copy. A human decides: restore the run dir from the archive mirror, or —
            # for a DETERMINISTIC unit only (test seed / random_search / bayes_opt candidate) —
            # delete the dir so --resume re-runs it to the identical result.
            raise ValueError(
                f"CORRUPT archive record at {run_dir / _RECORD_NAME}: {exc}. "
                "Recover it from the archive mirror (preferred; verify with "
                "scripts/archive_integrity.py), or delete this run dir to let --resume re-run the "
                "unit — ONLY safe for deterministic units (TEST seeds / search-arm candidates); "
                "deleting an LLM-arm record changes the candidate set."
            ) from exc
        if filter is not None and not all(
            record.get(k) == v for k, v in filter.items()
        ):
            continue
        records.append(record)
    return records
