"""Unit tests for the crash-rehearsal comparison core (scripts/crash_rehearsal.py).

The live kill->resume->prove-equality loop is a pre-freeze gauntlet SCRIPT (runbook §3e-bis); these
pin the pure pieces it stands on: volatile-field canonicalization and the tree diff verdict.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import crash_rehearsal as CR  # noqa: E402


def _rec(root: Path, rel: str, *, fitness: float, wall: float = 1.0, env: str = "e1") -> None:
    d = root / rel
    d.mkdir(parents=True, exist_ok=True)
    (d / "record.json").write_text(json.dumps({
        "run_id": rel.split("/")[-1], "metrics": {"val_fitness": fitness},
        "wall_clock": wall, "env_fingerprint": env,
    }), encoding="utf-8")


def test_canonical_records_drop_volatile_fields(tmp_path: Path) -> None:
    _rec(tmp_path, "arm/c0", fitness=0.1, wall=1.0, env="a")
    recs = CR.canonical_records(tmp_path)
    assert set(recs) == {"arm/c0"}
    assert "wall_clock" not in recs["arm/c0"] and "env_fingerprint" not in recs["arm/c0"]
    assert recs["arm/c0"]["metrics"]["val_fitness"] == 0.1


def test_compare_trees_equal_when_only_volatile_differs(tmp_path: Path) -> None:
    ref, other = tmp_path / "ref", tmp_path / "other"
    _rec(ref, "arm/c0", fitness=0.1, wall=1.0, env="a")
    _rec(other, "arm/c0", fitness=0.1, wall=99.0, env="b")  # only timing/env differ
    assert CR.compare_trees(ref, other) == []


def test_compare_trees_reports_missing_extra_and_differing(tmp_path: Path) -> None:
    ref, other = tmp_path / "ref", tmp_path / "other"
    _rec(ref, "arm/c0", fitness=0.1)
    _rec(ref, "arm/c1", fitness=0.2)
    _rec(other, "arm/c1", fitness=0.9)   # differs
    _rec(other, "arm/c2", fitness=0.3)   # extra
    diffs = CR.compare_trees(ref, other)
    assert any(d.startswith("MISSING") and "c0" in d for d in diffs)
    assert any(d.startswith("EXTRA") and "c2" in d for d in diffs)
    assert any(d.startswith("DIFFERS") and "c1" in d for d in diffs)
