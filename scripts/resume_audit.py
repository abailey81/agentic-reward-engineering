#!/usr/bin/env python3
"""Resume AUDIT — the pre-relaunch confidence check for the unattended Myriad campaign (2026-07-08).

The resume system itself is archive-as-truth: on ``--resume`` the driver re-derives what is DONE from
the archive and re-submits only the rest, idempotently, so it recovers from ANY stoppage (crash, hang,
reboot, power-loss, VPN outage, Scratch purge). This tool makes that recovery *legible and verified*
BEFORE you relaunch: it reads the local archive (the pulled mirror) READ-ONLY and reports, per stage
and per unit, exactly what the next ``--resume`` will re-do — the precise missing sealed-leg seeds, the
search completeness, whether each winner is frozen — and it VERIFIES integrity (every record parses)
and, when a second-disk mirror is present, cross-checks the two archives agree.

It is effect-blind by construction (it reads run_ids + parse-ability + COUNTS, never a ``val_fitness``/
``test_sharpe`` value), so running it never risks the single-look inference. Exit code: ``0`` when the
archive is consistent and resumable; ``2`` when a real integrity problem is found (a corrupt record, a
mirror divergence on SEALED records) that a human should look at before relaunching.

Usage (over the pulled mirror, no cluster needed)::

    python scripts/resume_audit.py outputs/campaign_cluster \
        --arms distributional scalar scalar_cvar5 placebo placebo_shuffled random_search bayes_opt \
        --baselines differential_sharpe raw_return return_minus_cvar return_minus_variance \
        --seeds 0-402 --candidates 30 --k-search 3 --mirror D:/llm_rp_archive_mirror/campaign_cluster
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _parse_seeds(spec: str) -> list[int]:
    """A comma list and/or ``a-b`` inclusive ranges (e.g. ``0-402`` or ``0,1,5``)."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _archived_run_ids(arm_dir: Path) -> tuple[set[str], int]:
    """The set of archived run_ids under one unit dir + the count of UNPARSEABLE records (integrity).

    Walks ``record.json`` directly (NOT ``load_all``, which fails loud on the first corrupt record —
    correct for analysis, wrong for an audit that must SURVEY corruption without crashing): a parseable
    record contributes its run_id (the driver's completion key = the dir name); an unparseable one is
    counted corrupt and NOT counted done, so its slot stays on the resume plan and is re-run."""
    ids: set[str] = set()
    corrupt = 0
    if arm_dir.is_dir():
        for rec in arm_dir.rglob("record.json"):
            try:
                data = json.loads(rec.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                corrupt += 1
                continue
            ids.add(str(data.get("run_id") or rec.parent.name))
    return ids, corrupt


def _ledgered_failures(search_arm_dir: Path) -> int:
    """Count distinct permanently-ledgered failed candidates for one search arm (F5)."""
    seen: set[str] = set()
    for lp in list(search_arm_dir.glob("failures.jsonl")) + list(search_arm_dir.glob("*.failures.jsonl")):
        try:
            for ln in lp.read_text(encoding="utf-8", errors="replace").splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    row = json.loads(ln)
                except (json.JSONDecodeError, ValueError):
                    continue
                seen.add(str(row.get("candidate_id") or f"{lp.name}#{len(seen)}"))
        except OSError:
            continue
    return len(seen)


def audit_resume(
    archive_root: str | Path,
    *,
    arms: list[str],
    test_seeds: list[int],
    candidates: int,
    k_search: int = 1,
    baselines: list[str] | None = None,
    frozen_root: str | Path | None = None,
    mirror_root: str | Path | None = None,
) -> dict[str, Any]:
    """Compute the exact ``--resume`` plan + integrity verdict over a LOCAL archive. Pure + cluster-free.

    Returns a dict with ``search`` (per-arm done/expected/ledgered/complete), ``test`` (per-unit
    present/expected + the EXACT missing seed ids the next resume will re-train), ``winners_frozen``,
    ``corrupt_records`` (integrity), ``mirror`` (per-unit parity when a mirror is given), and a
    top-level ``verdict`` {resumable, integrity_ok, remaining_test_units}."""
    root = Path(archive_root)
    search_root, test_root = root / "search", root / "test"
    frozen = Path(frozen_root) if frozen_root is not None else (root / "frozen")
    baselines = baselines or []
    k = max(1, int(k_search))

    report: dict[str, Any] = {"search": {}, "test": {}, "winners_frozen": {}, "mirror": {}}
    corrupt_total = 0

    # SEARCH: per-arm completeness (archived candidate records + ledgered failures vs candidates x k).
    for arm in arms:
        ids, corrupt = _archived_run_ids(search_root / arm)
        corrupt_total += corrupt
        ledgered = _ledgered_failures(search_root / arm)
        expected = int(candidates) * k
        resolved = len(ids) + ledgered * k
        report["search"][arm] = {
            "archived": len(ids), "ledgered_failures": ledgered, "expected": expected,
            "complete": resolved >= expected, "corrupt": corrupt,
        }
        report["winners_frozen"][arm] = (frozen / arm / "record.json").is_file() or bool(
            list(frozen.glob(f"{arm}*/record.json")) if frozen.is_dir() else []
        )

    # TEST: per-unit exact missing seed ids (the precise resume plan for the sealed leg — 93% of GPU-h).
    remaining_test = 0
    units = [(arm, arm) for arm in arms] + [(f"baseline_{b}", f"baseline_{b}") for b in baselines]
    for unit, arm_label in units:
        ids, corrupt = _archived_run_ids(test_root / unit)
        corrupt_total += corrupt
        expected_ids = [f"{arm_label}-s{s}" for s in test_seeds]
        missing = [rid for rid in expected_ids if rid not in ids]
        remaining_test += len(missing)
        report["test"][unit] = {
            "present": len(expected_ids) - len(missing), "expected": len(expected_ids),
            "missing": missing[:50], "n_missing": len(missing), "corrupt": corrupt,
        }

    # MIRROR parity (the "across all sites" check) — TWO directions with DIFFERENT severities:
    #  * mirror_behind (local has records the mirror doesn't) = the BENIGN normal case: the 6-hourly
    #    mirror is a few fresh pulls behind. A soft warning ("sync the mirror"), NEVER an integrity fail —
    #    else the audit cries wolf (exit 2) on every relaunch and the operator learns to ignore it.
    #  * local_lost (the mirror has SEALED records absent locally) = the REAL alarm: the local archive
    #    lost data the backup still holds → restore from the mirror. THIS fails integrity_ok / exits 2.
    mirror_behind = 0
    mirror_lost = 0
    if mirror_root is not None:
        mroot = Path(mirror_root)
        for unit, _arm_label in units:
            local_ids, _ = _archived_run_ids(test_root / unit)
            mirror_ids, _ = _archived_run_ids(mroot / "test" / unit)
            only_local = sorted(local_ids - mirror_ids)   # mirror behind (benign lag)
            only_mirror = sorted(mirror_ids - local_ids)  # LOCAL lost these — recover from the mirror
            report["mirror"][unit] = {
                "local": len(local_ids), "mirror": len(mirror_ids),
                "mirror_behind": len(only_local), "local_lost": len(only_mirror),
                "recover_from_mirror": only_mirror[:25],
            }
            mirror_behind += len(only_local)
            mirror_lost += len(only_mirror)

    report["corrupt_records"] = corrupt_total
    report["verdict"] = {
        # Integrity fails ONLY on a real archive problem: a corrupt local record, OR a sealed record the
        # LOCAL archive lost that only survives on the mirror. A mirror merely BEHIND is current-enough.
        "integrity_ok": corrupt_total == 0 and mirror_lost == 0,
        "mirror_behind": mirror_behind,
        "mirror_current": mirror_behind == 0,
        "local_lost_recoverable_from_mirror": mirror_lost,
        "remaining_test_units": remaining_test,
        "search_all_complete": all(s["complete"] for s in report["search"].values()),
        "resumable": True,  # archive-as-truth: the driver can always resume; this flags what it will do
    }
    return report


def render(report: dict[str, Any]) -> str:
    v = report["verdict"]
    lines = ["RESUME AUDIT — archive-as-truth pre-relaunch check (effect-blind: counts only)", ""]
    lines.append(f"  integrity_ok={v['integrity_ok']}  corrupt_records={report['corrupt_records']}  "
                 f"remaining_test_units={v['remaining_test_units']}  "
                 f"search_all_complete={v['search_all_complete']}")
    lines.append("")
    lines.append("  SEARCH (candidate completeness):")
    for arm, s in report["search"].items():
        frozen = "frozen" if report["winners_frozen"].get(arm) else "NOT-frozen"
        flag = "OK" if s["complete"] else "INCOMPLETE"
        lines.append(f"    {arm:<20} {s['archived']}+{s['ledgered_failures']}f/{s['expected']} "
                     f"[{flag}] winner={frozen}" + (f"  CORRUPT×{s['corrupt']}" if s["corrupt"] else ""))
    lines.append("  TEST (sealed leg — exact seeds the next --resume will re-train):")
    for unit, t in report["test"].items():
        tail = "" if t["n_missing"] == 0 else (
            "  next: " + ", ".join(t["missing"][:6]) + (" …" if t["n_missing"] > 6 else ""))
        lines.append(f"    {unit:<26} {t['present']}/{t['expected']} present, "
                     f"{t['n_missing']} to run{tail}" + (f"  CORRUPT×{t['corrupt']}" if t["corrupt"] else ""))
    if report["mirror"]:
        lines.append("  MIRROR parity (second-disk copy):")
        for unit, m in report["mirror"].items():
            if m["local_lost"]:
                flag = f"  {m['local_lost']} LOCALLY LOST — recover from mirror!"
            elif m["mirror_behind"]:
                flag = f"  mirror {m['mirror_behind']} behind (benign; sync it)"
            else:
                flag = "  in sync"
            lines.append(f"    {unit:<26} local={m['local']} mirror={m['mirror']}{flag}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pre-relaunch resume audit over the local archive mirror.")
    ap.add_argument("archive_root", help="Campaign output dir (the pulled mirror, e.g. outputs/campaign_cluster).")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--baselines", nargs="*", default=None)
    ap.add_argument("--seeds", default="0-567", help="Test seeds: comma list and/or a-b ranges. "
                    "Default = the full E1 ladder [0..567]; pass a narrower range only to audit one tier.")
    ap.add_argument("--candidates", type=int, default=30)
    ap.add_argument("--k-search", type=int, default=1, help="search_seeds_per_candidate (k).")
    ap.add_argument("--frozen-root", default=None)
    ap.add_argument("--mirror", default=None, help="A second-disk archive root to cross-check (parity).")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    report = audit_resume(
        args.archive_root, arms=list(args.arms), test_seeds=_parse_seeds(args.seeds),
        candidates=args.candidates, k_search=args.k_search,
        baselines=list(args.baselines) if args.baselines else None,
        frozen_root=args.frozen_root, mirror_root=args.mirror,
    )
    print(json.dumps(report, indent=1) if args.json else render(report), flush=True)
    return 0 if report["verdict"]["integrity_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
