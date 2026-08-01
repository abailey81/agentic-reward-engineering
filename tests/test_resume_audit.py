"""Resume-audit tests: the pre-relaunch plan + integrity verdict over a local archive (no cluster)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.resume_audit import audit_resume  # noqa: E402


def _write(root: Path, stage: str, arm: str, rid: str) -> None:
    from src.io.results import write_run

    write_run(
        {"run_id": rid, "arm": arm, "seed": 0, "fold": 0, "candidate_id": rid, "generation": 0,
         "reward_source_hash": "h", "feedback_block": "", "wall_clock": 0.0, "env_fingerprint": "x",
         "metrics": {"val_fitness": 0.5}},
        str(root / stage / arm),
    )


def test_audit_reports_exact_missing_seeds_and_search_completeness(tmp_path):
    # SEARCH: distributional has 2 archived candidates == candidates(2)*k(1) -> complete + winner frozen
    _write(tmp_path, "search", "distributional", "distributional-g0-c0")
    _write(tmp_path, "search", "distributional", "distributional-g0-c1")
    (tmp_path / "frozen" / "distributional").mkdir(parents=True)
    (tmp_path / "frozen" / "distributional" / "record.json").write_text(
        json.dumps({"run_id": "w", "candidate_id": "distributional-g0-c1"}), encoding="utf-8")
    # TEST: seeds 0,1 present of {0,1,2} -> s2 missing; baseline has only s0 of {0,1,2}
    _write(tmp_path, "test", "distributional", "distributional-s0")
    _write(tmp_path, "test", "distributional", "distributional-s1")
    _write(tmp_path, "test", "baseline_differential_sharpe", "baseline_differential_sharpe-s0")

    rep = audit_resume(tmp_path, arms=["distributional"], test_seeds=[0, 1, 2], candidates=2,
                       baselines=["differential_sharpe"])
    assert rep["test"]["distributional"]["missing"] == ["distributional-s2"]
    assert rep["test"]["baseline_differential_sharpe"]["n_missing"] == 2
    assert rep["verdict"]["remaining_test_units"] == 3  # 1 (dist) + 2 (baseline)
    assert rep["search"]["distributional"]["complete"] is True
    assert rep["winners_frozen"]["distributional"] is True
    assert rep["verdict"]["integrity_ok"] is True
    assert rep["verdict"]["search_all_complete"] is True


def test_audit_flags_corrupt_records_as_integrity_failure(tmp_path):
    _write(tmp_path, "test", "scalar", "scalar-s0")
    bad = tmp_path / "test" / "scalar" / "scalar-s1" / "record.json"
    bad.parent.mkdir(parents=True)
    bad.write_text("{ this is not valid json", encoding="utf-8")  # torn/corrupt record
    rep = audit_resume(tmp_path, arms=["scalar"], test_seeds=[0, 1], candidates=1)
    assert rep["corrupt_records"] == 1 and rep["verdict"]["integrity_ok"] is False
    # the corrupt slot is NOT counted done -> s1 is still on the resume plan
    assert "scalar-s1" in rep["test"]["scalar"]["missing"]


def test_audit_mirror_behind_is_benign_but_local_loss_is_flagged(tmp_path):
    # (a) mirror BEHIND (local ahead) = benign 6-hourly lag → integrity stays OK, mirror_behind noted
    archive, mirror = tmp_path / "a", tmp_path / "m"
    _write(archive, "test", "scalar", "scalar-s0")
    _write(archive, "test", "scalar", "scalar-s1")
    _write(mirror, "test", "scalar", "scalar-s0")  # s1 not yet mirrored (normal lag)
    rep = audit_resume(archive, arms=["scalar"], test_seeds=[0, 1], candidates=1, mirror_root=mirror)
    assert rep["mirror"]["scalar"]["mirror_behind"] == 1
    assert rep["verdict"]["integrity_ok"] is True       # benign lag must NOT cry wolf
    assert rep["verdict"]["mirror_current"] is False

    # (b) LOCAL lost a sealed record the mirror still holds = the REAL alarm → integrity FAILS
    archive2, mirror2 = tmp_path / "a2", tmp_path / "m2"
    _write(archive2, "test", "scalar", "scalar-s0")
    _write(mirror2, "test", "scalar", "scalar-s0")
    _write(mirror2, "test", "scalar", "scalar-s1")  # mirror has s1; local lost it
    rep2 = audit_resume(archive2, arms=["scalar"], test_seeds=[0, 1], candidates=1, mirror_root=mirror2)
    assert rep2["mirror"]["scalar"]["local_lost"] == 1
    assert "scalar-s1" in rep2["mirror"]["scalar"]["recover_from_mirror"]
    assert rep2["verdict"]["integrity_ok"] is False


def test_audit_cli_exit_code_and_json(tmp_path, capsys):
    from scripts.resume_audit import main

    _write(tmp_path, "test", "scalar", "scalar-s0")
    rc = main([str(tmp_path), "--arms", "scalar", "--seeds", "0", "--candidates", "1", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"]["remaining_test_units"] == 0
