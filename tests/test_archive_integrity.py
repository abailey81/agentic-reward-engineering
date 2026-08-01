"""Tests for the content-addressed RESULT-ARCHIVE integrity seal (scripts/archive_integrity.py).

The archive is the one irreplaceable artifact (results replay from it). These pin the tamper-evidence:
seal-then-verify is OK; ANY modify / add / remove / corrupt changes the root and is reported precisely;
the digest is line-ending invariant (Windows-authored <-> Linux-analysed).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import archive_integrity as AI  # noqa: E402


def _write_record(root: Path, run_id: str, val: float) -> None:
    d = root / "arm" / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "record.json").write_text(
        json.dumps({"run_id": run_id, "metrics": {"val_fitness": val}}), encoding="utf-8")


def test_seal_then_verify_ok(tmp_path: Path) -> None:
    for i in range(4):
        _write_record(tmp_path, f"c{i}", 0.1 * i)
    AI.write_manifest(tmp_path)
    result = AI.verify_manifest(tmp_path)
    assert result.ok is True
    assert result.n_verified == 4
    assert not (result.added or result.removed or result.changed)
    assert result.expected_root == result.actual_root


def test_verify_detects_a_modified_record(tmp_path: Path) -> None:
    for i in range(3):
        _write_record(tmp_path, f"c{i}", 0.1 * i)
    AI.write_manifest(tmp_path)
    # silently CHANGE one record's bytes after sealing
    _write_record(tmp_path, "c1", 0.999)
    result = AI.verify_manifest(tmp_path)
    assert result.ok is False
    assert result.changed == ["c1"]
    assert result.actual_root != result.expected_root


def test_verify_detects_an_added_and_a_removed_record(tmp_path: Path) -> None:
    for i in range(3):
        _write_record(tmp_path, f"c{i}", 0.1 * i)
    AI.write_manifest(tmp_path)
    # remove one, add another
    (tmp_path / "arm" / "c2" / "record.json").unlink()
    _write_record(tmp_path, "c9", 0.5)
    result = AI.verify_manifest(tmp_path)
    assert result.ok is False
    assert result.removed == ["c2"]
    assert result.added == ["c9"]


def test_root_is_order_independent_and_deterministic(tmp_path: Path, tmp_path_factory) -> None:
    # two archives with the SAME records written in a DIFFERENT order have the SAME root.
    a = tmp_path
    b = tmp_path_factory.mktemp("b")
    for i in (0, 1, 2):
        _write_record(a, f"c{i}", 0.1 * i)
    for i in (2, 0, 1):
        _write_record(b, f"c{i}", 0.1 * i)
    assert AI.merkle_root(AI.record_digests(a)) == AI.merkle_root(AI.record_digests(b))


def test_digest_is_line_ending_invariant(tmp_path: Path) -> None:
    # The SAME multi-line record content, written once with LF and once with CRLF line endings, must
    # produce the SAME digest (a Windows-authored archive verified on Linux, matching the freeze-hash
    # convention). Only the line endings differ — the logical content is byte-identical after \r\n->\n.
    d = tmp_path / "arm" / "c0"
    d.mkdir(parents=True)
    lines = ['{', '  "run_id": "c0",', '  "metrics": {"val_fitness": 0.5}', '}']
    (d / "record.json").write_bytes("\n".join(lines).encode("utf-8"))
    lf_root = AI.merkle_root(AI.record_digests(tmp_path))
    (d / "record.json").write_bytes("\r\n".join(lines).encode("utf-8"))
    crlf_root = AI.merkle_root(AI.record_digests(tmp_path))
    assert lf_root == crlf_root


def test_unreadable_record_perturbs_the_root(tmp_path: Path) -> None:
    _write_record(tmp_path, "c0", 0.1)
    good_root = AI.merkle_root(AI.record_digests(tmp_path))
    # a record that is not valid UTF-8 JSON still gets a stable digest (over its bytes), so it is
    # sealed, not silently dropped
    d = tmp_path / "arm" / "c1"
    d.mkdir(parents=True)
    (d / "record.json").write_bytes(b"\xff\xfe not json")
    assert AI.merkle_root(AI.record_digests(tmp_path)) != good_root


def test_verify_mirror_mode_tolerates_added_but_fails_changed(tmp_path: Path) -> None:
    """2026-07-06 A5: verify-mirror = the BACKUP check — records ADDED after the seal are lawful
    (a mid-campaign mirror carries newer work); a sealed record that changed/vanished is corruption."""
    root = tmp_path / "mirror"
    _write_record(root, "r1", 0.1)
    _write_record(root, "r2", 0.2)
    assert AI.main(["write", str(root)]) == 0
    # a record added AFTER the seal: strict verify fails, verify-mirror tolerates
    _write_record(root, "r3", 0.3)
    assert AI.main(["verify", str(root)]) == 1
    assert AI.main(["verify-mirror", str(root)]) == 0
    # a SEALED record mutated: both modes fail
    _write_record(root, "r1", 999.0)
    assert AI.main(["verify-mirror", str(root)]) == 1
