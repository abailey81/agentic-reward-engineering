"""Manifest relpath separator regression (data_pipeline/src/data/vault.py).

Before the 2026-07-02 fix, ``freeze`` recorded ``str(path.relative_to(ROOT))`` — on
Windows that writes backslash relpaths (``data\\gold\\...``) into manifest.jsonl,
checksums.txt, and (via ``FrozenArtifact.relpath``) lineage.jsonl, breaking exact-relpath
matching for univ5/univ4-era entries; consumers survived only via the name fallback in
``find_entry``. These tests pin:

  * a freshly frozen artifact records POSIX relpaths (forward slashes) in ALL three
    ledgers plus the returned ``FrozenArtifact`` — even on Windows;
  * historical backslash lines are NOT rewritten (write-once ledger) and still resolve
    through ``find_entry``/``read_verified``'s name fallback.

Run isolated (the repo-root conftest binds the ENGINE's ``src``; here ``src`` is data_pipeline's):
    PYTHONPATH=data_pipeline python -m pytest data_pipeline/tests/test_vault_relpath.py --noconftest
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_DP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_DP))
for _m in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
    del sys.modules[_m]

from src.data import vault  # noqa: E402


def _isolated_vault(tmp_path, monkeypatch) -> None:
    # Redirect the module-level ROOT so layers + all three ledgers land under tmp_path;
    # config values are relative paths, so nothing touches the real vault.
    monkeypatch.setattr(vault, "ROOT", tmp_path)


def _frame() -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-02", periods=4)
    return pd.DataFrame({"r": [0.01, -0.02, 0.0, 0.005]}, index=idx)


def test_fresh_freeze_records_posix_relpaths_in_all_ledgers(tmp_path, monkeypatch) -> None:
    _isolated_vault(tmp_path, monkeypatch)
    art = vault.freeze(_frame(), "gold", "posix_check.csv", {"vendor": "test"})

    assert art.relpath == "data/gold/posix_check.csv"
    assert "\\" not in art.relpath

    entry = vault.find_entry("data/gold/posix_check.csv")     # exact POSIX match, no fallback
    assert entry is not None and entry["relpath"] == "data/gold/posix_check.csv"

    checksum_line = (tmp_path / "data/manifest/checksums.txt").read_text().splitlines()[-1]
    assert "data/gold/posix_check.csv" in checksum_line and "\\" not in checksum_line

    vault.record_lineage(art, [art], "test.stage")
    edge = json.loads((tmp_path / "data/manifest/lineage.jsonl").read_text().splitlines()[-1])
    assert edge["output"]["relpath"] == "data/gold/posix_check.csv"


def test_historical_backslash_entries_still_resolve_via_name_fallback(tmp_path, monkeypatch) -> None:
    _isolated_vault(tmp_path, monkeypatch)
    # Simulate a univ5-era Windows-written line: backslash relpath, otherwise well-formed.
    payload = _frame().to_csv().encode()
    legacy_path = tmp_path / "data" / "raw" / "legacy.csv"
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_bytes(payload)
    legacy = {
        "relpath": "data\\raw\\legacy.csv", "layer": "raw", "name": "legacy.csv",
        "format": "csv", "sha256": vault.sha256_bytes(payload), "rows": 4, "cols": 1,
        "row_hash": "x", "frozen_utc": "2026-07-01T00:00:00+00:00",
        "run_id": "historic", "vendor": "test",
    }
    manifest = tmp_path / "data" / "manifest" / "manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(legacy, sort_keys=True) + "\n")

    # Name fallback (the read path historical consumers rely on) must keep working …
    df = vault.read_verified("legacy.csv")
    assert df.shape == (4, 1)
    # … and freezing a NEW artifact must not rewrite the historical line.
    vault.freeze(_frame(), "gold", "new.csv", {"vendor": "test"})
    lines = manifest.read_text().splitlines()
    assert json.loads(lines[0])["relpath"] == "data\\raw\\legacy.csv"
    assert json.loads(lines[1])["relpath"] == "data/gold/new.csv"
