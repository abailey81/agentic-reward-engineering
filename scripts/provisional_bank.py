#!/usr/bin/env python3
"""PROVISIONAL RUNG BANKING — see results as they accumulate, without spending your inference.

THE QUESTION THIS ANSWERS
-------------------------
"Can we bank results every 10 seeds so we see early whether anything is wrong?" Yes — and the
registration already anticipates it (``model_suite.interim_report``: *floor-tier results labeled
provisional … effect-on-data-collection: none*). But it is legitimate ONLY under a condition that is
easy to state and easy to violate, so this tool enforces and RECORDS it rather than trusting memory.

WHY REPEATED LOOKS ARE NORMALLY FATAL, AND WHY THEY ARE NOT HERE
----------------------------------------------------------------
Looking many times inflates the false-positive rate only because you might ACT on what you see —
stop early, add seeds, change a knob. The inflation comes from the sample size being a function of
the data. In this design it is not: the stopping rule is **exogenous** — the achieved rung is set by
measured throughput against a calendar date fixed in advance (``exogenous_stop: 2026-08-27``), never
by the numbers. `config/preregistration.yaml` states it directly: *"the tier is EXOGENOUS (measured
throughput vs deadline, never results) -> the single look is preserved."*

So the rule is simple and absolute:

    **Nothing about data collection may change because of what a provisional bank shows.**
    Not the rung, not the seeds, not the arms, not the stop date, not a knob.

If that holds, interim banking costs nothing statistically. If it is broken even once, the
confirmatory inference is gone and no later analysis can repair it.

WHAT MAKES THIS DEFENSIBLE RATHER THAN A CONFESSION
---------------------------------------------------
Every look is APPENDED TO A LOOK LOG — when, at which rung, over how many records, at which commit,
with the exogeneity attestation. An examiner asking "did you peek?" gets a complete, dated, honest
answer showing the stopping rule never moved. Undisclosed peeking is a research-integrity problem;
disclosed, exogenous, logged monitoring is ordinary good practice. The log is the difference.

Provisional numbers are stamped PROVISIONAL in the payload and the filename, and are never the
confirmatory result: that remains ONE analysis at the achieved rung via ``bank_gate.py``.

Usage::

    python scripts/provisional_bank.py outputs/campaign_cluster              # bank if a rung is due
    python scripts/provisional_bank.py outputs/campaign_cluster --every 10
    python scripts/provisional_bank.py outputs/campaign_cluster --status     # what would happen
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BANK_DIR = "banked_provisional"
LOOK_LOG = "look_log.jsonl"

#: Recorded verbatim in every banked payload and every look-log row, so the condition travels WITH
#: the numbers and cannot be separated from them later.
EXOGENEITY_ATTESTATION = (
    "PROVISIONAL. The stopping rule for this campaign is EXOGENOUS: the achieved seed rung is "
    "determined by measured throughput against the pre-registered calendar stop, never by any "
    "observed effect. Nothing about data collection (rung, seeds, arms, stop date, or any "
    "parameter) was changed on the basis of this or any other provisional bank. The confirmatory "
    "analysis is performed ONCE, at the achieved rung, via scripts/bank_gate.py."
)


def achieved_rung(root: str | Path) -> tuple[int, dict[str, int]]:
    """The COMMON seed depth across scored arms — the rung actually banked so far.

    Deliberately the MINIMUM across arms, not the maximum or the mean: a paired contrast can only
    use seeds present in every arm it compares, so the honest rung is the one every arm has reached.
    """
    base = Path(root)
    per_arm: dict[str, set] = {}
    for sroot in [base / "test", *sorted(base.glob("test_leg_*"))]:
        if not sroot.is_dir():
            continue
        for rec in sroot.rglob("record.json"):
            if any(x.startswith(".pull_tmp") for x in rec.parts):
                continue
            try:
                d = json.loads(rec.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            arm, seed = d.get("arm"), d.get("seed")
            if arm is not None and seed is not None:
                per_arm.setdefault(f"{sroot.name}/{arm}", set()).add(int(seed))
    if not per_arm:
        return 0, {}
    counts = {a: len(s) for a, s in per_arm.items()}
    return min(counts.values()), counts


def already_banked(root: str | Path) -> set:
    """Rungs already recorded in the look log — banking twice would just clutter the record."""
    log = Path(root) / BANK_DIR / LOOK_LOG
    if not log.is_file():
        return set()
    out = set()
    for line in log.read_text(encoding="utf-8").splitlines():
        try:
            out.add(int(json.loads(line).get("rung", -1)))
        except Exception:  # noqa: BLE001
            continue
    return out


def due_rung(rung: int, every: int, banked: set) -> int | None:
    """The largest un-banked multiple of ``every`` at or below the achieved rung."""
    if rung < every:
        return None
    candidate = (rung // every) * every
    return candidate if candidate not in banked else None


def _git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                           text=True, timeout=30)
        return (r.stdout or "").strip()[:12] if r.returncode == 0 else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def bank(root: str | Path, rung: int, *, counts: dict[str, int],
         run_analysis: bool = True) -> dict[str, Any]:
    """Write a PROVISIONAL bank for ``rung`` and append the look to the log. Read-only w.r.t. the run."""
    base = Path(root)
    out_dir = base / BANK_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    payload: dict[str, Any] = {
        "PROVISIONAL": True,
        "attestation": EXOGENEITY_ATTESTATION,
        "rung": rung,
        "utc": stamp,
        "git_commit": _git_commit(),
        "records_per_arm": counts,
        "confirmatory": False,
    }
    if run_analysis:
        r = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "analyze_campaign.py"), "--root", str(base)],
            cwd=REPO, capture_output=True, text=True, timeout=3600)
        payload["analysis_returncode"] = r.returncode
        payload["analysis_tail"] = (r.stdout or "")[-1500:]
        if r.returncode != 0:
            payload["analysis_stderr_tail"] = (r.stderr or "")[-800:]
    path = out_dir / f"rung_{rung:04d}_{stamp}_PROVISIONAL.json"
    # ATOMIC commit, mirroring src/io/results.py::write_run (deep review #119, 2026-07-27). This was a
    # plain ``path.write_text(...)`` — the exact pattern that writer abandoned, in its words: "a plain
    # open('w') + json.dump killed mid-write leaves a TRUNCATED record.json". Here the stakes are lower
    # (no code reads these payloads — the only programmatic reader is ``already_banked``, over
    # ``look_log.jsonl``, and it tolerates a torn line), so nothing could be BRICKED by a truncation.
    # But this file is an INTEGRITY artifact: it carries EXOGENEITY_ATTESTATION and records that a LOOK
    # was taken at the data in a sequential design, and it is what goes to the supervisors in the R81
    # interim report. A half-written attestation is exactly the artifact that must not exist, and the
    # repo already has the one-line pattern that makes it impossible. Temp sibling + fsync + os.replace
    # (atomic on Windows and POSIX within a directory); a crash now leaves either no file or a complete
    # one, and a stray ``.tmp`` is ignored by the ``rung_*_PROVISIONAL.json`` glob.
    tmp_path = path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, path)
    with (out_dir / LOOK_LOG).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"utc": stamp, "rung": rung, "git_commit": payload["git_commit"],
                             "records_per_arm": counts, "file": path.name,
                             "attestation": EXOGENEITY_ATTESTATION}, default=str) + "\n")
    payload["path"] = str(path)
    return payload


def main(argv: list[str] | None = None) -> int:
    from src.utils.console import make_console_safe
    make_console_safe()   # src/utils/console.py — it prints a captured analyze_campaign tail
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--every", type=int, default=10, help="bank every N common seeds (default 10)")
    ap.add_argument("--status", action="store_true", help="report only; bank nothing")
    ap.add_argument("--no-analysis", action="store_true",
                    help="record the look without running the analysis (fast check)")
    args = ap.parse_args(argv)

    rung, counts = achieved_rung(args.root)
    banked = already_banked(args.root)
    due = due_rung(rung, args.every, banked)
    print(f"achieved common rung: {rung}   (arms: {len(counts)})")
    print(f"already banked      : {sorted(banked) or 'none'}")
    if due is None:
        print(f"nothing due at every={args.every} — next bank at "
              f"{((rung // args.every) + 1) * args.every}")
        return 0
    if args.status:
        print(f"DUE: rung {due} would be banked now (PROVISIONAL)")
        return 0
    payload = bank(args.root, due, counts=counts, run_analysis=not args.no_analysis)
    print(f"banked PROVISIONAL rung {due} -> {payload['path']}")
    print("  " + EXOGENEITY_ATTESTATION[:150] + " …")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
