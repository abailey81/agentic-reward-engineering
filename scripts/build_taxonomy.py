"""Standalone reward-program taxonomy builder — walks an archive of authored ``reward.py`` files.

Why a standalone CLI: the prototype archive lays out one authored program per candidate directory
(``<root>/<arm>/<candidate>/reward.py`` — written by ``src.io.results.write_run`` next to each
``record.json``), and the taxonomy question ("do different feedback arms author different KINDS of
programs?") needs nothing else — no configs, no panel, no headline inference. This tool builds JUST the
taxonomy from the on-disk sources: one pooled induction (``src.inference.reward_taxonomy``), the per-arm
kind composition, and the threshold-sensitivity sweep, then writes
``reward_taxonomy_<rootname>.json`` + ``.md`` (the same renderer ``analyze_campaign.write_report`` uses,
so the standalone table and the in-report section can never drift).

The campaign analysis path does NOT need this tool: ``scripts/analyze_campaign.py::analyze()`` emits the
identical ``reward_taxonomy`` block from the loaded records (which carry ``reward_source``) for any
archive depth, including the campaign's ``<leg>/<arm>/<candidate>`` layout. This walker is deliberately
prototype-shaped (``<root>/<arm>/<candidate>/reward.py``, one level of arms).

Report-only and DISJOINT from the frozen m=6 confirmatory family: nothing here reads or writes a
hypothesis decision, and the output carries no ``arm_a/arm_b/metric/level`` family-tuple keys.
Deterministic: no randomness; sorted walks; ``sort_keys`` JSON — byte-identical output on re-run.

Flags
-----
  --root           Archive root with <arm>/<candidate>/reward.py leaves (default outputs/prototype).
  --out-dir        Where to write reward_taxonomy_<rootname>.{json,md} (default outputs/tables).
  --depth          Canonical AST subtree-shape depth (default 4, the reward-code-distance signature).
  --sim-threshold  Jaccard threshold for the kind graph (default 0.6; the sweep reports 0.5/0.6/0.7).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.inference.reward_taxonomy import (
    pool_sources,
    taxonomy_by_arm,
    taxonomy_threshold_sensitivity,
)

# REUSE the analyze_campaign renderer (no duplicate markdown logic). scripts/ is on sys.path when this
# file runs as a script or when the tests insert it; fall back to a path insert for direct module use.
try:  # pragma: no cover - import plumbing
    from analyze_campaign import reward_taxonomy_markdown
except ImportError:  # pragma: no cover - standalone invocation
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from analyze_campaign import reward_taxonomy_markdown

__all__ = ["build", "collect_sources", "main"]


def collect_sources(root: str | Path) -> dict[str, dict[str, str]]:
    """``{arm: {candidate_dir: reward source}}`` over ``<root>/<arm>/<candidate>/reward.py`` leaves.

    Deterministic sorted walk. A first-level directory with no ``reward.py``-bearing candidate subdirs
    (e.g. the prototype's ``tables/``) is simply not an arm and is skipped; loose files (``COMPLETE``,
    ``llm_calls.jsonl``) are ignored. Sources are read as UTF-8 text — unparseable content is NOT
    filtered here; the taxonomy module excludes and counts it (P7c).
    """
    root = Path(root)
    sources_by_arm: dict[str, dict[str, str]] = {}
    if not root.is_dir():
        return sources_by_arm
    for arm_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        per_arm: dict[str, str] = {}
        for cand_dir in sorted(p for p in arm_dir.iterdir() if p.is_dir()):
            reward_path = cand_dir / "reward.py"
            if reward_path.is_file():
                per_arm[cand_dir.name] = reward_path.read_text(encoding="utf-8")
        if per_arm:
            sources_by_arm[arm_dir.name] = per_arm
    return sources_by_arm


def build(
    root: str | Path, *, depth: int = 4, sim_threshold: float = 0.6
) -> dict[str, Any]:
    """Pooled taxonomy + per-arm composition + threshold sensitivity for one archive root."""
    sources_by_arm = collect_sources(root)
    if not sources_by_arm:
        return {
            "status": "no_data",
            "reason": f"no <arm>/<candidate>/reward.py found under {Path(root).as_posix()}",
        }
    result = taxonomy_by_arm(sources_by_arm, depth=depth, sim_threshold=sim_threshold)
    result["sensitivity"] = taxonomy_threshold_sensitivity(
        pool_sources(sources_by_arm), depth=depth
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Induce the reward-program taxonomy from an <arm>/<candidate>/reward.py archive.",
    )
    parser.add_argument("--root", default="outputs/prototype")
    parser.add_argument("--out-dir", default="outputs/tables")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--sim-threshold", type=float, default=0.6)
    args = parser.parse_args()

    result = build(args.root, depth=args.depth, sim_threshold=args.sim_threshold)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"reward_taxonomy_{Path(args.root).name}"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    md_path.write_text(reward_taxonomy_markdown(result), encoding="utf-8")

    print(f"[build_taxonomy] {args.root} -> {json_path} + {md_path}")
    if result.get("status") != "ok":
        print(f"  status={result.get('status')}: {result.get('reason')}")
        return
    pooled = result["pooled"]
    print(
        f"  {pooled['n_kinds']} kind(s) over {pooled['n_programs']} program(s) "
        f"(threshold={pooled['sim_threshold']:g}, depth={pooled['depth']}, "
        f"unparseable={pooled['n_unparseable']}, singletons={pooled['n_singletons']})"
    )
    for kind in pooled["kinds"]:
        arms = ", ".join(result["kind_arms"].get(kind["kind_id"], []))
        print(f"    {kind['kind_id']}: size={kind['size']:>3}  [{kind['label']}]  arms: {arms}")
    # Console output is ASCII-only: a Windows console may decode as cp1251/cp437, where a Greek tau or
    # an em-dash raises UnicodeEncodeError. The UTF-8 report FILES keep the typographic symbols.
    for arm, e in result["per_arm"].items():
        ent = e["entropy_bits"]
        print(
            f"  {arm:>14}: n={e['n_programs']:>3} kinds={e['n_kinds_present']} "
            f"entropy={'n/a' if ent is None else f'{ent:.3f}'} bits unparseable={e['n_unparseable']}"
        )
    sens = result.get("sensitivity") or {}
    if sens.get("status") == "ok":
        ks = ", ".join(f"t={r['threshold']:g}:{r['n_kinds']}" for r in sens["by_threshold"])
        st = ", ".join(
            f"{r['pair']} RI={'n/a' if r['rand_index'] is None else format(r['rand_index'], '.3f')}"
            for r in sens["adjacent_stability"]
        )
        print(f"  sensitivity: n_kinds {ks}; stability {st}")


if __name__ == "__main__":
    main()
