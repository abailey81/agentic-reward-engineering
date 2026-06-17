#!/usr/bin/env python3
"""Validate a REBUILT gold returns panel against the frozen reference (C4 lean validator).

Purpose
-------
The Phase-1 data rebuild produces a new-suffix returns panel. Before it can replace the frozen
``univ3`` headline panel we need a REAL, executable check that it does not silently perturb the shared
2005-2025 history — a rebuild that changed historical returns would move the headline tail. This tool
runs the lean cell-level byte-diff (``src.data.validation.panel_overlap_diff``) over the (date × RIC)
OVERLAP of the candidate vs the frozen reference panel and reports every changed cell + a summary.

It compares the RAW ``returns_panel_<suffix>.parquet`` frames (date-indexed, RIC-columned) — NOT the
anonymised, window-sliced :class:`~src.data.panel.Panel` — so the comparison is on the source history.

Exit code is 0 when the overlap is byte-identical (schema/calendar drift — added/retired names or
sessions — is reported but does NOT fail, since a rebuild legitimately changes the universe over time),
non-zero when any SHARED historical cell changed (unless ``--allow-changes``).

Flags
-----
  --candidate    Candidate returns-panel parquet (default: the ACTIVE gold_suffix() panel).
  --reference    Frozen reference returns-panel parquet (default: data/gold/returns_panel_univ3.parquet).
  --gold-dir     Gold directory (default: data/gold under the repo root).
  --max-examples Max changed cells to print (default 20).
  --allow-changes  Exit 0 even if shared cells changed (report-only mode).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a rebuilt gold returns panel vs the frozen reference (C4 byte-diff).",
    )
    parser.add_argument("--candidate", default=None, help="Candidate returns-panel parquet (default: active suffix).")
    parser.add_argument("--reference", default=None, help="Frozen reference parquet (default: returns_panel_univ3.parquet).")
    parser.add_argument("--gold-dir", default=None, help="Gold directory (default: <repo>/data/gold).")
    parser.add_argument("--max-examples", type=int, default=20, help="Max changed cells to print.")
    parser.add_argument("--allow-changes", action="store_true", help="Exit 0 even if shared cells changed (report-only).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = _repo_root()
    sys.path.insert(0, str(root))

    from src.data.loaders import gold_suffix
    from src.data.validation import panel_overlap_diff

    gold_dir = Path(args.gold_dir) if args.gold_dir else root / "data" / "gold"
    reference = Path(args.reference) if args.reference else gold_dir / "returns_panel_univ3.parquet"
    candidate = (
        Path(args.candidate)
        if args.candidate
        else gold_dir / f"returns_panel_{gold_suffix()}.parquet"
    )

    print("=" * 72)
    print("VERIFY GOLD — candidate vs frozen reference (byte-diff over the shared overlap)")
    print("=" * 72)
    print(f"  candidate: {candidate}")
    print(f"  reference: {reference}")
    for p in (candidate, reference):
        if not p.is_file():
            print(f"[verify_gold] FAIL: not found: {p}", file=sys.stderr)
            return 2

    diff = panel_overlap_diff(candidate, reference, max_examples=int(args.max_examples))
    print("-" * 72)
    print("  " + diff.summary())
    print("-" * 72)

    if diff.identical_over_overlap:
        print("[verify_gold] PASS: the shared 2005-2025 overlap is byte-identical to the reference.")
        return 0
    msg = f"[verify_gold] {diff.n_changed_cells} shared cell(s) CHANGED vs the frozen reference."
    if args.allow_changes:
        print(msg + " (--allow-changes: exit 0, report-only.)")
        return 0
    print(msg + " Re-run the rebuild or bless the new panel explicitly.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
