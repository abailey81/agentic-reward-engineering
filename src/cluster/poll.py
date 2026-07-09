"""Polling layer: pull the remote archive, read completion truth, plan compacted resumes (§14.1).

The archive IS the queue: a task is DONE iff its record.json exists (the same atomic-commit
truth the whole pipeline rests on). Completion therefore survives any scheduler/driver/network
failure. ``pending_specs`` implements the COMPACTED-RESUME diff: given the full task list and
the pulled archive, return only the specs whose run_ids are missing — the next array contains
exactly the unfinished work and nothing else (zero wasted fair-share).

Transfer (V9 audit fix): the Windows driver host has NO rsync (verified 2026-07-06 — only
scp/ssh in PATH), so the pull is tar-over-ssh with EXACT incrementality derived from our own
architecture: records are immutable after their atomic commit, so the remote-vs-local run-dir
set DIFF is the complete transfer list — nothing approximate about it. Dirs land in a staging
area and move into the mirror only once whole (record.json verified), preserving the archive's
write-then-rename discipline end-to-end; the nightly mirror chain re-verifies content hashes
downstream.
"""
from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

__all__ = [
    "pull_archive",
    "remote_completed_dirs",
    "remote_reject_files",
    "completed_run_ids",
    "permanent_reject_ids",
    "permanently_rejected_specs",
    "pending_specs",
    "spec_local_root",
    "spec_run_id",
    "sync_epilogue_ledgers",
]

#: fetch(relpaths, staging_dir) — stream those run dirs from the cluster into staging_dir.
Fetch = Callable[[list[str], Path], None]

_STAGING = ".pull_tmp"


def _is_staging_part(part: str) -> bool:
    """True for ANY staging dir — ours or a foreign process's (audit 2026-07-22)."""
    return part.startswith(_STAGING)


def remote_completed_dirs(
    remote_outputs_root: str, runner: Callable[[list[str]], str]
) -> set[str]:
    """POSIX relpaths of every COMMITTED record dir on the cluster (one ssh round-trip).

    record.json is written LAST by the atomic archive commit, so find-by-record.json lists only
    COMPLETE run dirs — the same completion truth :func:`completed_run_ids` reads locally.
    Fails LOUD (with guidance) on a missing root: silently returning 0 forever would mask a
    wrong-path config bug; ``prepare_remote`` pre-creates ``outputs/`` so empty-but-existing
    cleanly means "0 completed".
    """
    root = remote_outputs_root.rstrip("/")
    try:
        out = runner(["find", root, "-name", "record.json", "-type", "f"])
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"listing {root} on the cluster failed — wrong outputs root, or prepare_remote was "
            f"never run? ({exc})"
        ) from exc
    dirs: set[str] = set()
    for line in out.splitlines():
        line = line.strip()
        if not line.endswith("/record.json"):
            continue
        rel = line[len(root):].lstrip("/")
        if "/" not in rel:  # a stray root-level record.json is not a run dir
            continue
        dirs.add(rel.rsplit("/", 1)[0])
    return dirs


def remote_reject_files(
    remote_outputs_root: str, runner: Callable[[list[str]], str]
) -> set[str]:
    """POSIX relpaths of every node-side reject MARKER on the cluster (P9): flat JSON files under
    any ``_rejects/`` dir (written atomically by ``run_one._write_reject_marker``). Same loud
    missing-root behavior as :func:`remote_completed_dirs` (the pull calls that first anyway)."""
    root = remote_outputs_root.rstrip("/")
    try:
        out = runner(["find", root, "-path", "*/_rejects/*.json", "-type", "f"])
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"listing reject markers under {root} on the cluster failed ({exc})"
        ) from exc
    rels: set[str] = set()
    for line in out.splitlines():
        line = line.strip()
        if not line.endswith(".json"):
            continue
        rel = line[len(root):].lstrip("/")
        if "_rejects/" not in rel:
            continue
        rels.add(rel)
    return rels


def sync_epilogue_ledgers(
    remote_ledger_dir: str, local_dir: str | Path, runner: Callable[[list[str]], str],
) -> dict[str, int]:
    """Mirror the per-array epilogue ledgers locally; return ``{filename: bytes}`` for what landed.

    The ledgers are the ONLY record of tasks that died without archiving anything (walltime kill,
    node failure, a host missing ``apptainer``), and they are written on the cluster while every
    consumer of them — :func:`src.cluster.killswitch.classify_task_deaths` and the bad-node check —
    runs on the driver. ``pull_archive`` cannot carry them: it transfers committed run dirs under
    ``outputs/``, and ``ledger/`` is a SIBLING of that root, so without this the driver was blind to
    exactly the failures the ledger exists to record.

    Transfer discipline: the files are append-only, so a remote-vs-local SIZE difference is an exact
    "has anything been added" test — a finished array's ledger stops changing and is never re-sent,
    and only the active array's file moves. Wholesale re-fetch of a changed file (rather than a
    byte-offset tail) is deliberate: array tasks append CONCURRENTLY, so an offset-based tail can
    split a line across two fetches and leave a permanently torn local row, whereas a whole-file copy
    is always internally consistent. Best-effort by contract — the caller degrades to "no ledger
    data" rather than failing, because this is monitoring, never the completion truth (the archive is
    the queue).
    """
    root = remote_ledger_dir.rstrip("/")
    local = Path(local_dir)
    local.mkdir(parents=True, exist_ok=True)
    try:
        listing = runner(["find", root, "-name", "*.epilogue.jsonl", "-type", "f",
                          "-printf", "%s %p\n"])
    except (subprocess.SubprocessError, OSError):
        return {}
    landed: dict[str, int] = {}
    for line in listing.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        try:
            remote_size = int(parts[0])
        except ValueError:
            continue
        remote_path = parts[1]
        name = remote_path.rsplit("/", 1)[-1]
        if not name.endswith(".epilogue.jsonl"):
            continue
        target = local / name
        if target.is_file() and target.stat().st_size == remote_size:
            continue                                  # unchanged — the array is done appending
        try:
            body = runner(["cat", remote_path])
        except (subprocess.SubprocessError, OSError):
            continue
        tmp = target.with_suffix(".tmp")
        # newline="" + explicit "\n": Windows CRLF translation would make the local size differ from
        # the remote size on every byte, so the skip-if-unchanged test would never fire again.
        with tmp.open("w", encoding="utf-8", newline="") as fh:
            fh.write(body)
        tmp.replace(target)
        landed[name] = target.stat().st_size
    return landed


def _default_fetch(host: str, remote_outputs_root: str) -> Fetch:
    """Production transport: remote ``tar -cf -`` | ssh | local ``tar -xf -`` into staging.

    check=True on BOTH ends — ssh exits nonzero on a dropped connection and tar -x errors on a
    truncated stream ("Unexpected EOF in archive") — so a torn pull fails loud here rather than
    leaving a silently short mirror. The remote side is POSIX-quoted; the local tar runs with
    cwd=staging (no -C: Windows bsdtar/GNU-tar path-flavor quirks).
    """
    from src.cluster.submit import reap, ssh_base

    root = remote_outputs_root.rstrip("/")

    def _fetch(relpaths: list[str], staging: Path) -> None:
        remote_cmd = (
            "tar -C " + shlex.quote(root) + " -cf - "
            + " ".join(shlex.quote(p) for p in relpaths)
        )
        # stdin=DEVNULL (2026-07-28): ssh forwards stdin to the remote command unless told not to,
        # and this pull is the LONGEST-LIVED ssh the driver spawns (3600 s budget), so it is the one
        # most likely to sit holding an inherited handle. Correct practice for unattended ssh.
        ssh = subprocess.Popen([*ssh_base(host), remote_cmd], stdout=subprocess.PIPE,
                               stdin=subprocess.DEVNULL)
        drained = False
        try:
            subprocess.run(
                ["tar", "-xf", "-"], stdin=ssh.stdout, check=True, timeout=3600,
                cwd=str(staging),
            )
            drained = True
        finally:
            # Close our read end FIRST (that is what breaks the remote tar's pipe and lets ssh
            # exit on its own), then guarantee the reap — see :func:`reap` for why the unreaped
            # child was doing real damage rather than being untidy.
            if ssh.stdout is not None:
                ssh.stdout.close()
            reap(ssh, grace=300.0 if drained else 10.0)
        if ssh.returncode != 0:
            # Reached only when tar consumed the whole stream, so a nonzero code here (including
            # the -9 of a reap that had to kill) means a torn transfer and must fail LOUD.
            raise subprocess.CalledProcessError(ssh.returncode, "ssh tar -cf - (remote)")

    return _fetch


def pull_archive(
    remote_outputs_root: str,
    local_root: str | Path,
    *,
    host: str = "myriad",
    runner: Callable[[list[str]], str] | None = None,
    fetch: Fetch | None = None,
    chunk: int = 150,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    """Pull every remote-complete run dir the local mirror lacks; return how many were pulled.

    EXACT incremental: the run-dir diff (immutable records) is the transfer list. Each chunk is
    fetched into ``<local_root>/.pull_tmp`` and every dir is moved into the mirror ONLY after
    its record.json is verified present — the local archive never contains a torn dir, even if
    the process is killed mid-extract (stale staging is swept on the next call and never counts
    as completed). ``chunk`` bounds the remote argv length (~150 relpaths ≈ 12 KB, far under
    ARG_MAX) and keeps each pipe short and restartable.
    """
    from src.cluster.submit import ssh_runner

    local = Path(local_root)
    local.mkdir(parents=True, exist_ok=True)
    # Per-PROCESS staging (audit 2026-07-22 CRITICAL): 12 concurrent mode-D driver processes
    # share this output dir; a fixed shared staging raced (one process rmtree'd another's
    # mid-extract tree; a torn record.json could be committed via the copytree fallback).
    # Each process stages under its own pid-suffixed dir and NEVER touches foreign staging
    # (a dead process's leftovers are inert: every reader excludes staging-prefixed dirs).
    import os as _os
    staging = local / f"{_STAGING}.{_os.getpid()}"
    if staging.exists():  # OUR previous pull died mid-extract — its content is untrusted
        shutil.rmtree(staging)
    runner = runner or ssh_runner(host)
    fetch = fetch or _default_fetch(host, remote_outputs_root)

    remote = remote_completed_dirs(remote_outputs_root, runner)
    have = {
        p.parent.relative_to(local).as_posix()
        for p in local.rglob("record.json")
        if not any(_is_staging_part(x) for x in p.parts)
    }
    missing = sorted(remote - have)
    # P9: reject MARKERS ride the same pull (flat immutable JSON files under _rejects/) so the
    # driver can abandon deterministic node-side rejects without waiting out requeue rounds.
    rej_have = {
        p.relative_to(local).as_posix()
        for p in local.rglob("_rejects/*.json")
        if not any(_is_staging_part(x) for x in p.parts)
    }
    rej_missing = sorted(remote_reject_files(remote_outputs_root, runner) - rej_have)
    if not missing and not rej_missing:
        return 0

    # P14: a big pull is MANY chunked pipes (each up to an hour) — report per-chunk progress so
    # the driver can heartbeat mid-pull instead of looking hung to the sentinel. Best-effort.
    n_chunks = -(-len(missing) // chunk) + -(-len(rej_missing) // chunk)
    chunks_done = 0

    def _tick() -> None:
        nonlocal chunks_done
        chunks_done += 1
        if progress is not None:
            try:
                progress(chunks_done, n_chunks)
            except Exception:  # noqa: BLE001 — observability must never break the pull
                pass

    for i in range(0, len(missing), chunk):
        batch = missing[i : i + chunk]
        staging.mkdir(parents=True, exist_ok=True)
        fetch(batch, staging)
        for rel in batch:
            src = staging / rel
            if not (src / "record.json").is_file():
                raise RuntimeError(
                    f"pulled dir {rel!r} arrived without record.json — torn transfer; the "
                    f"local mirror was left untouched for it"
                )
            dest = local / rel
            if dest.exists():
                continue  # landed by an earlier overlapping pull — keep the committed copy
            dest.parent.mkdir(parents=True, exist_ok=True)
            # ── D18 ROOT CAUSE, found 2026-08-01 (RUN 10, record §100) ──────────────────────────
            # `shutil.move(src, dst)` where `dst` is an EXISTING DIRECTORY does not fail — it moves
            # `src` INSIDE it, producing `<candidate>/<candidate>/record.json`. The `dest.exists()`
            # guard above cannot close that, because `local` (`read_root`) is SHARED BY ALL TWELVE
            # SUPERVISED LINES: another line's driver can commit the same record in the window
            # between our check and our move. That is precisely the observed signature — both
            # duplicates sit on LEG lines (the many-concurrent-drivers case), the remote side is
            # FLAT (verified on the node: no nesting exists there), and the count grew 1 -> 2 as
            # concurrency rose. §44.6/§65.4 had the defect right and the mechanism unstated.
            #
            # `os.rename` is the fix because it CANNOT nest: renaming onto an existing directory
            # raises rather than descending into it. Same filesystem by construction (staging is a
            # child of `local`), so it is also atomic.
            try:
                _os.rename(str(src), str(dest))
            except OSError:
                # Lost the race: another driver committed this record between the check and here.
                # Both copies are byte-identical (verified for both live instances), so KEEP THE
                # COMMITTED ONE and drop ours. Never overwrite — the committed copy may already
                # have been read.
                shutil.rmtree(src, ignore_errors=True)
        shutil.rmtree(staging, ignore_errors=True)
        _tick()

    for i in range(0, len(rej_missing), chunk):
        batch = rej_missing[i : i + chunk]
        staging.mkdir(parents=True, exist_ok=True)
        fetch(batch, staging)
        for rel in batch:
            src = staging / rel
            if not src.is_file():
                raise RuntimeError(
                    f"pulled reject marker {rel!r} missing after extract — torn transfer"
                )
            dest = local / rel
            if dest.exists():
                continue  # markers are effectively immutable — keep the committed copy
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Same D18 race, same fix. A marker is a FILE, so losing the race would either raise
            # (Windows) or SILENTLY OVERWRITE a committed marker (POSIX) — and if a directory ever
            # occupied that name it would nest exactly as above. `os.rename` makes all three cases
            # one deterministic outcome: keep what is already committed.
            try:
                _os.rename(str(src), str(dest))
            except OSError:
                src.unlink(missing_ok=True)
        shutil.rmtree(staging, ignore_errors=True)
        _tick()
    return len(missing) + len(rej_missing)


def completed_run_ids(local_root: str | Path) -> set[str]:
    """Every run_id with a committed record under ``local_root`` (recursive; layout-agnostic).

    Staging content (``.pull_tmp`` — a torn pull's leftovers) is EXCLUDED: a record is complete
    only once it has been moved whole into the mirror proper.
    """
    root = Path(local_root)
    if not root.is_dir():
        return set()
    return {
        p.parent.name for p in root.rglob("record.json") if not any(_is_staging_part(x) for x in p.parts)
    }


def permanent_reject_ids(local_root: str | Path) -> set[str]:
    """Run_ids with a PERMANENT node-side reject marker in the mirror (P9): the deterministic
    validation-reject class the driver abandons immediately (no requeue rounds — same-source
    retry is provably futile). Transient markers (``permanent: false``) are diagnostics only;
    torn/unreadable markers are skipped (that spec just keeps the normal bounded retry)."""
    root = Path(local_root)
    if not root.is_dir():
        return set()
    ids: set[str] = set()
    for p in root.rglob("_rejects/*.json"):
        if any(_is_staging_part(x) for x in p.parts):
            continue
        try:
            row = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — a torn marker must not sink the poll cycle
            continue
        if row.get("permanent"):
            ids.add(str(row.get("run_id") or p.stem))
    return ids


def spec_local_root(spec: dict[str, Any], local_root: str | Path) -> Path:
    """The LOCAL archive sub-root a spec's own run_ids live under.

    Extracted 2026-07-28 so completion truth (:func:`pending_specs`) and reject truth
    (:func:`permanently_rejected_specs`) resolve a spec's archive by ONE rule. They disagreed
    before — completions were sub-root scoped by the 2026-07-19 audit and rejects were not — and
    that asymmetry is exactly what let one line's reject marker condemn another line's candidate.

    ``archive_root`` means two things by substrate. In a CLUSTER run it is the REMOTE sub-root
    path, whose basename (``test`` / ``test_h3_singleshot`` / ``search`` / ``search_<suffix>``) is
    what the pull mirrors under ``local_root``. In a LOCAL/pack run ``run_one`` archives straight
    to ``archive_root``, so it IS the local record directory. Distinguish by existence. A spec
    with no ``archive_root`` falls back to the mirror-wide root (unchanged legacy behaviour).
    """
    local_root = Path(local_root)
    ar = str(spec.get("archive_root", ""))
    if ar and Path(ar).is_dir():
        return Path(ar)                    # local/pack run: records live at archive_root itself
    if ar:
        return local_root / Path(ar).name  # cluster run: remote sub-root mirrored under local_root
    return local_root                      # no archive_root: mirror-wide (legacy behaviour)


def spec_run_id(spec: dict[str, Any]) -> str:
    """The run_id a spec will archive under (search: candidate_id; test: run_id)."""
    rid = spec.get("run_id") or spec.get("candidate_id")
    if not rid:
        raise KeyError(f"spec carries neither run_id nor candidate_id: {sorted(spec)}")
    return str(rid)


def pending_specs(
    all_specs: list[dict[str, Any]], local_root: str | Path
) -> list[dict[str, Any]]:
    """The COMPACTED-RESUME diff: specs whose run_ids are not yet in the archive.

    2026-07-19 (35-agent audit, CONFIRMED critical): completion truth is scoped to each spec's
    OWN archive sub-root, NOT the whole mirror. Test run_ids are bare ``{arm}-s{seed}``
    (``test_leg.py``), so a disjoint-root invocation — the H3 C5 single-shot (``test_h3_singleshot/``)
    or a ``--root-suffix`` re-search — reuses run_ids like ``distributional-s0`` that ALSO exist
    under the headline ``test/`` root. A mirror-wide diff would see the headline's record, mark the
    H3/re-search unit "done", never submit its training, and leave the disjoint archive EMPTY — a
    silently fabricated H3 null. Grouping by sub-root (the ``archive_root`` basename: ``test`` vs
    ``test_h3_singleshot`` vs ``search`` vs ``search_<suffix>``) restores disjointness; a spec with
    no ``archive_root`` falls back to the mirror-wide check (unchanged behaviour).

    2026-07-19 (verification follow-up): ``archive_root`` means two things by substrate, so resolving
    the LOCAL completion root must handle both. In a CLUSTER run it is the REMOTE sub-root path (its
    basename — ``test`` / ``test_h3_singleshot`` / ``search`` / ``search_<suffix>`` — is what the pull
    mirrors under ``local_root``), so the local root is ``local_root / basename``. In a LOCAL / laptop
    pack run (``run_one`` archives straight to ``archive_root``) it IS the local record directory, so
    the local root is ``archive_root`` itself. Distinguish by existence: if ``archive_root`` is already
    a local directory, read it directly; else map its basename under the mirror. Both keep the sub-roots
    disjoint (the H3-null-fabrication fix); the earlier basename-only form pointed at a non-existent
    ``local_root/<leaf>`` whenever ``archive_root == local_root`` and marked completed specs pending —
    a resume would then re-run finished work (caught by test_pack_path_end_to_end_two_real_specs)."""
    local_root = Path(local_root)
    done_by_root: dict[str, set[str]] = {}
    out: list[dict[str, Any]] = []
    for s in all_specs:
        root = spec_local_root(s, local_root)
        key = str(root)
        if key not in done_by_root:
            done_by_root[key] = completed_run_ids(root)
        if spec_run_id(s) not in done_by_root[key]:
            out.append(s)
    return out


def permanently_rejected_specs(
    all_specs: list[dict[str, Any]], local_root: str | Path
) -> list[dict[str, Any]]:
    """The specs a PERMANENT node-side reject marker condemns — scoped to each spec's OWN sub-root.

    2026-07-28, found live at T+11.5 h of the confirmatory run. This is the SAME defect the
    2026-07-19 35-agent audit called "CONFIRMED critical" and fixed for :func:`pending_specs`,
    left open in its sibling: the driver resolved rejects with a MIRROR-WIDE
    ``permanent_reject_ids(local_archive_root)``, and ``local_archive_root`` is the ONE tree all
    twelve supervised lines share. Reject markers are keyed on the BARE candidate id
    (``scalar-g1-c0``) and EVERY line reuses that scheme, so one line's marker abandoned every
    other line's identically-named candidate — silently, WITHOUT SUBMITTING IT, and permanently
    (the row is ledgered so no resume ever retries it).

    MEASURED on the live archive before the fix: of 498 abandonments across the twelve lines,
    **439 (88 %) were spurious** — condemned by a foreign marker, never submitted, never judged.
    402 of those trace to a single line, ``search_leg_qwen3_5_9b``, the deliberately weakest model
    in the suite, whose 47 legitimate rejects sterilised the search of all eleven other lines.
    The confirmatory ``claude-opus-5`` core line was hit 36 times out of 36, i.e. every core
    candidate abandoned so far was killed by another model's failure.

    Scoping is by the same rule as completion truth, so the two can never disagree about which
    archive a spec belongs to.
    """
    local_root = Path(local_root)
    perm_by_root: dict[str, set[str]] = {}
    out: list[dict[str, Any]] = []
    for s in all_specs:
        root = spec_local_root(s, local_root)
        key = str(root)
        if key not in perm_by_root:
            perm_by_root[key] = permanent_reject_ids(root)
        if spec_run_id(s) in perm_by_root[key]:
            out.append(s)
    return out
