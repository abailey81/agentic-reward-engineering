#!/usr/bin/env python3
"""Automated CRASH-RESUME rehearsal (A6, 2026-07-06): kill a live run, resume it, PROVE equality.

The resume machinery is unit-tested piecewise (search-cache byte-identity, TEST-leg skip-done,
LLM-arm replay), but the guarantee this campaign actually rests on is END-TO-END: *a run killed
at an arbitrary moment and resumed produces the identical archive a never-killed run produces.*
This script certifies exactly that, automatically, on the tiny keyless ``--dry-run`` config
(3 arms x 2 candidates x 200 steps, synthetic panel, stub author — free, minutes, no API key):

  1. REFERENCE   — an uninterrupted ``run_prototype --dry-run --out <ref>``.
  2. CONTROL     — a second uninterrupted run into ``<ref2>`` (the determinism control: if ref != ref2
                   the ENVIRONMENT is nondeterministic and a resume mismatch must not be blamed on the
                   resume machinery). Skippable with ``--skip-control``.
  3. CRASH       — the same run into ``<crash>``, HARD-KILLED (the whole process TREE — pool workers
                   included, like a real power event) once ``--kill-after-records`` records exist.
  4. RESUME      — ``--resume`` on the crash tree, to completion.
  5. VERDICT     — canonical-compare (volatile fields dropped: wall_clock, env_fingerprint) the
                   resumed tree against the reference: same record set, same science bytes -> PASS.

Run it in the pre-freeze gauntlet (CAMPAIGN_RUNBOOK §3e-bis) so crash-resume is CERTIFIED on this
box, not assumed:

    .venv/Scripts/python.exe scripts/crash_rehearsal.py

READ-ONLY with respect to the repo: all trees live under ``--work-root`` (default
``outputs/crash_rehearsal``), which is wiped per pass. Exit 0 = certified; 1 = mismatch/failure.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]

#: Volatile per-run fields excluded from the equality verdict (timing + env-capture provenance —
#: legitimately different across runs; every SCIENCE field stays compared byte-for-byte).
_VOLATILE_KEYS = ("wall_clock", "env_fingerprint")


def canonical_records(root: Path) -> dict[str, dict[str, Any]]:
    """``{relative_run_dir: canonical record}`` over every record.json under ``root``.

    Canonical = the parsed record minus the volatile keys, so two runs of a seeded-deterministic
    pipeline compare EQUAL iff every science field (reward source + hash, metrics incl. val_returns,
    seeds, generations, prompts) is identical."""
    out: dict[str, dict[str, Any]] = {}
    for rec_path in sorted(root.rglob("record.json")):
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        for k in _VOLATILE_KEYS:
            rec.pop(k, None)
        out[rec_path.parent.relative_to(root).as_posix()] = rec
    return out


def compare_trees(ref: Path, other: Path) -> list[str]:
    """Human-readable differences between two archive trees ([] == canonically identical)."""
    a, b = canonical_records(ref), canonical_records(other)
    diffs: list[str] = []
    for key in sorted(set(a) | set(b)):
        if key not in b:
            diffs.append(f"MISSING in resumed tree: {key}")
        elif key not in a:
            diffs.append(f"EXTRA in resumed tree: {key}")
        elif a[key] != b[key]:
            fields = [f for f in set(a[key]) | set(b[key]) if a[key].get(f) != b[key].get(f)]
            diffs.append(f"DIFFERS: {key} (fields: {sorted(fields)[:6]})")
    return diffs


def _kill_tree(proc: subprocess.Popen) -> None:
    """Hard-kill the WHOLE process tree (a real crash takes the pool workers down too).

    On Windows ``Popen.kill`` is TerminateProcess on the PARENT ONLY — orphaned arm-pool workers
    would keep training and WRITING RECORDS into the "crashed" tree, corrupting the rehearsal's
    semantics. ``taskkill /T /F`` kills the tree."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       capture_output=True, check=False)
    else:  # pragma: no cover - POSIX fallback
        proc.kill()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:  # pragma: no cover
        pass


def _launch(out_dir: Path, *, resume: bool = False) -> subprocess.Popen:
    cmd = [sys.executable, str(_REPO / "scripts" / "run_prototype.py"),
           "--dry-run", "--p-arms", "1", "--out", str(out_dir)]
    if resume:
        cmd.append("--resume")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    # The child's output goes to a log BESIDE its tree (never DEVNULL): when a pass fails, the
    # forensics are one file away instead of gone.
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    log = (out_dir.parent / f"{out_dir.name}.log").open("ab")
    try:
        return subprocess.Popen(cmd, cwd=str(_REPO), env=env, stdout=log, stderr=subprocess.STDOUT)
    finally:
        log.close()  # the child holds its own inherited handle


def _run_to_completion(out_dir: Path, *, resume: bool = False, timeout_s: float) -> None:
    proc = _launch(out_dir, resume=resume)
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        raise RuntimeError(f"run into {out_dir} exceeded {timeout_s:.0f}s") from None
    summary = out_dir / "run_summary.json"
    if not summary.is_file():
        raise RuntimeError(f"run into {out_dir} finished without run_summary.json (rc={proc.returncode})")
    ok = json.loads(summary.read_text(encoding="utf-8")).get("matched_budget_ok")
    if not ok:
        raise RuntimeError(f"run into {out_dir} reports matched_budget_ok={ok}")


def _run_and_kill(out_dir: Path, *, kill_after_records: int, timeout_s: float) -> int:
    """Launch, wait for ``kill_after_records`` archived records, hard-kill the tree.

    Returns the record count at the kill moment. Fails loud if the run COMPLETES before the
    threshold is reached (the kill came too late to exercise resume)."""
    proc = _launch(out_dir)
    t0 = time.monotonic()
    while True:
        n = len(list(out_dir.rglob("record.json"))) if out_dir.is_dir() else 0
        if n >= kill_after_records:
            _kill_tree(proc)
            print(f"[crash_rehearsal] KILLED the run tree at {n} archived record(s).")
            return n
        if proc.poll() is not None:
            raise RuntimeError(
                f"crash run finished (rc={proc.returncode}) before reaching "
                f"{kill_after_records} records — lower --kill-after-records"
            )
        if time.monotonic() - t0 > timeout_s:
            _kill_tree(proc)
            raise RuntimeError(f"crash run produced {n} records in {timeout_s:.0f}s; aborting")
        time.sleep(0.2)  # the dry run archives a record every ~0.5-1.5 s — poll fast enough to kill MID-run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Automated crash-resume rehearsal (kill -> resume -> prove equality).")
    ap.add_argument("--work-root", default=str(_REPO / "outputs" / "crash_rehearsal"),
                    help="Scratch root for the reference/control/crash trees (wiped per pass).")
    ap.add_argument("--kill-after-records", type=int, default=1,
                    help="Hard-kill the crash run once this many records are archived (default 1 of 6 — "
                         "the dry run is FAST, so the earliest kill maximises the mid-run recovery left "
                         "for --resume to prove).")
    ap.add_argument("--timeout", type=float, default=1800.0, help="Per-pass timeout seconds.")
    ap.add_argument("--skip-control", action="store_true",
                    help="Skip the second uninterrupted run (the cross-run determinism control).")
    args = ap.parse_args(argv)

    work = Path(args.work_root)
    ref, ref2, crash = work / "reference", work / "control", work / "crashed"
    for d in (ref, ref2, crash):
        if d.exists():
            shutil.rmtree(d)
    work.mkdir(parents=True, exist_ok=True)

    print(f"[crash_rehearsal] 1/4 REFERENCE run -> {ref}")
    _run_to_completion(ref, timeout_s=args.timeout)
    n_expected = len(canonical_records(ref))
    print(f"[crash_rehearsal]     reference complete: {n_expected} records.")

    if not args.skip_control:
        print(f"[crash_rehearsal] 2/4 CONTROL run (determinism check) -> {ref2}")
        _run_to_completion(ref2, timeout_s=args.timeout)
        control_diffs = compare_trees(ref, ref2)
        if control_diffs:
            print("[crash_rehearsal] DETERMINISM CONTROL FAILED — two uninterrupted runs differ; a "
                  "resume mismatch below would NOT be the resume machinery's fault:")
            for d in control_diffs[:10]:
                print(f"    {d}")
            return 1
        print("[crash_rehearsal]     control identical to reference (environment is deterministic).")

    print(f"[crash_rehearsal] 3/4 CRASH run (kill at {args.kill_after_records} records) -> {crash}")
    _run_and_kill(crash, kill_after_records=args.kill_after_records, timeout_s=args.timeout)
    n_at_kill = len(canonical_records(crash))
    if n_at_kill >= n_expected:
        print(f"[crash_rehearsal] WARNING: all {n_at_kill} records landed before the kill — the "
              "resume below is a no-op pass, not a mid-run recovery; lower --kill-after-records.")

    print("[crash_rehearsal] 4/4 RESUME the crashed tree -> completion")
    _run_to_completion(crash, resume=True, timeout_s=args.timeout)

    diffs = compare_trees(ref, crash)
    if diffs:
        print("[crash_rehearsal] FAIL — resumed tree differs from the uninterrupted reference:")
        for d in diffs[:20]:
            print(f"    {d}")
        return 1
    print(f"[crash_rehearsal] PASS — killed-at-{n_at_kill}-records + resume == uninterrupted run "
          f"({n_expected} records, science fields byte-identical). CERTIFIED path: the serial "
          f"run_prototype driver (LLM-stub replay + author-stream advance + both search-arm caches). "
          f"The campaign's run_recycling/test-leg resume is unit-certified separately "
          f"(tests/test_test_leg.py, test_parallel_*); the multi-point supervisor kill-storm "
          f"(runbook §3e-bis) remains the optional belt-and-braces end-to-end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
