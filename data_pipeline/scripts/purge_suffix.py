#!/usr/bin/env python3
"""Surgically remove a NEVER-CONSUMED build-suffix from the vault (files + manifest lines).

WHY THIS EXISTS (ADR-051 execution log): the first univ5 build ran against the un-bumped
``data_pipeline/config/data.yaml: window.end`` and clipped the calendar at 2025-12-31 — the
_univ5/_univ5s artifacts were structurally wrong (5,283 sessions, no 2026 rows) minutes after
creation and referenced by NOTHING. The vault is write-once by design, so a corrected rebuild
under the SAME suffix requires explicitly removing the poisoned set first. This script does
exactly that, loudly: it lists every file and manifest/lineage/checksum line it will touch,
requires ``--yes`` to act, and refuses suffixes that are consumed by the engine
(``config/data.yaml gold.suffix``) or protected (univ3/univ4).

Usage:
    python data_pipeline/scripts/purge_suffix.py --suffix _univ5          # DRY-RUN list
    python data_pipeline/scripts/purge_suffix.py --suffix _univ5 --yes   # remove
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "data_pipeline"))

from src.config import get  # noqa: E402

ROOT = _REPO / "data_pipeline"
PROTECTED = ("_univ3", "_univ4", "univ3", "univ4")


def main() -> int:
    ap = argparse.ArgumentParser(description="Purge a never-consumed build suffix from the vault.")
    ap.add_argument("--suffix", required=True, help="e.g. _univ5 (also matches _univ5s files)")
    ap.add_argument("--yes", action="store_true", help="actually delete (default: dry-run)")
    args = ap.parse_args()
    sfx = args.suffix
    # Guard-vs-match symmetry (2026-07-05 M18): the protected/active guards compare by EQUALITY
    # but victim selection used to SUBSTRING-match, so a partial suffix like "_univ" passed the
    # guards yet matched EVERY universe artifact (including the frozen headline panel). Require a
    # digit-bearing suffix and match on an exact token boundary (optional trailing "s" variant,
    # per the --suffix help), never a bare substring.
    if not re.search(r"\d", sfx):
        raise SystemExit(f"REFUSED: {sfx!r} has no digit — a partial suffix would match every universe")
    sfx_re = re.compile(re.escape(sfx) + r"s?(?=[._]|$)")
    if any(p == sfx or p == sfx.lstrip("_") for p in PROTECTED):
        raise SystemExit(f"REFUSED: {sfx} is a protected frozen suffix")
    # Engine-side active suffix must not be purged.
    import yaml
    engine_cfg = yaml.safe_load((_REPO / "config" / "data.yaml").read_text(encoding="utf-8"))
    active = str(((engine_cfg or {}).get("gold") or {}).get("suffix", ""))
    if active and (sfx == f"_{active}" or sfx == active):
        raise SystemExit(f"REFUSED: {sfx} is the ACTIVE engine gold suffix ({active})")

    layers = get("data.platform.layers")
    victims: list[Path] = []
    for layer_rel in layers.values():
        d = ROOT / layer_rel
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if sfx_re.search(p.name):
                victims.append(p)
    print(f"[purge] files matching '{sfx}': {len(victims)}")
    for p in victims:
        print(f"    {p.relative_to(ROOT)}")

    line_files = [ROOT / get("data.platform.manifest_jsonl"),
                  ROOT / get("data.platform.lineage_jsonl")]
    # legacy checksum file lives next to the manifest (vault._legacy_checksums)
    line_files.append((ROOT / get("data.platform.manifest_jsonl")).parent / "checksums.txt")
    plans: list[tuple[Path, list[str], list[str]]] = []
    for lf in line_files:
        if not lf.exists():
            continue
        keep, drop = [], []
        for line in lf.read_text(encoding="utf-8").splitlines():
            (drop if sfx_re.search(line) else keep).append(line)
        plans.append((lf, keep, drop))
        print(f"[purge] {lf.relative_to(ROOT)}: dropping {len(drop)} line(s)")
        for ln in drop[:12]:
            print(f"    - {ln[:150]}")

    if not args.yes:
        print("[purge] DRY-RUN (no changes). Re-run with --yes to execute.")
        return 0

    for p in victims:
        p.unlink()
    for lf, keep, drop in plans:
        if drop:
            lf.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
    print(f"[purge] DONE: removed {len(victims)} file(s) + "
          f"{sum(len(d) for _, _, d in plans)} manifest/lineage/checksum line(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
