"""Regression tests for the D49/D50 loader fix (shipped 2026-08-18).

Pre-fix behaviour, measured live before the change and recorded in CHANGELOG
[2026-08-18h]:

* the default walk pooled the ten ``*_leg_*`` replication subtrees into the core
  record set; every line reuses the core run_id vocabulary (``distributional-s0``,
  ...), so ``_seed_scores`` raised ``ValueError`` on the cross-line conflict and
  ``analyze()`` died before ``write_report()`` -- twice, ~34 minutes each, writing
  nothing;
* a doubly-nested stray copy (``<cand>/<cand>/record.json``, byte-identical) loaded
  TWICE under the (directory, run_id) key: measured 2 copies of
  ``placebo_shuffled-g3-c4`` on the real run-4 ``search_leg_glm_5_2`` line, 1 after
  the fix.

If either skip is reverted these tests fail.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import analyze_campaign as ac  # noqa: E402


def _write_record(run_dir: Path, run_id: str, arm: str, marker: float) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "run_id": run_id,
        "arm": arm,
        "seed": 0,
        "fold": "test",
        "candidate_id": "c0",
        "generation": 0,
        "reward_source_hash": "0" * 64,
        "feedback_block": "",
        "metrics": {"marker": marker},
        "wall_clock": 1.0,
        "env_fingerprint": "test-fp",
    }
    (run_dir / "record.json").write_text(json.dumps(record), encoding="utf-8")


def test_leg_subtrees_are_excluded_from_the_default_walk(tmp_path: Path) -> None:
    """A ``*_leg_*`` line sharing the core run_id vocabulary must not pool into the core."""
    _write_record(tmp_path / "test" / "distributional" / "distributional-s0",
                  "distributional-s0", "distributional", marker=1.0)
    for line in ("test_leg_glm_5_2", "search_leg_glm_5_2"):
        _write_record(tmp_path / line / "distributional" / "distributional-s0",
                      "distributional-s0", "distributional", marker=2.0)

    records = ac.load_campaign_records(tmp_path)

    assert len(records) == 1, "leg subtrees pooled into the core walk (D49 regression)"
    assert records[0]["metrics"]["marker"] == 1.0, "a LEG record displaced the core record"


def test_h3_singleshot_and_dot_dirs_stay_excluded(tmp_path: Path) -> None:
    """The two pre-existing exclusions must survive the D49 edit."""
    _write_record(tmp_path / "test" / "distributional" / "distributional-s0",
                  "distributional-s0", "distributional", marker=1.0)
    _write_record(tmp_path / "test_h3_singleshot" / "distributional" / "distributional-s1",
                  "distributional-s1", "distributional", marker=2.0)
    _write_record(tmp_path / ".pull_tmp.123" / "distributional" / "distributional-s2",
                  "distributional-s2", "distributional", marker=3.0)

    records = ac.load_campaign_records(tmp_path)

    assert [r["run_id"] for r in records] == ["distributional-s0"]


def test_nested_record_under_a_leaf_loads_once_and_warns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A ``<cand>/<cand>/record.json`` stray must not double-count its candidate (D50)."""
    leaf = tmp_path / "search" / "placebo_shuffled" / "placebo_shuffled-g3-c4"
    _write_record(leaf, "placebo_shuffled-g3-c4", "placebo_shuffled", marker=1.0)
    _write_record(leaf / "placebo_shuffled-g3-c4",
                  "placebo_shuffled-g3-c4", "placebo_shuffled", marker=1.0)

    records = ac.load_campaign_records(tmp_path)

    ids = [r["run_id"] for r in records]
    assert ids.count("placebo_shuffled-g3-c4") == 1, (
        "the doubly-nested stray copy entered the record set twice (D50 regression)"
    )
    err = capsys.readouterr().err
    assert "nested record-bearing dir" in err, "the leaf skip must be LOUD, never silent (D51)"
