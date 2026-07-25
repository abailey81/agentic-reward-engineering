"""Tests for the pre-registration deposit bundle (scripts/make_prereg_bundle.py)."""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import make_prereg_bundle as mpb  # noqa: E402
from scripts.freeze import canonical_hash  # noqa: E402


def test_bundle_contains_exactly_the_hash_bound_files(tmp_path: Path) -> None:
    out = tmp_path / "bundle.zip"
    path = mpb.make_bundle(out=out)
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        expected = set(mpb.bundle_members()) | {"MANIFEST.json", "DEPOSIT_README.md"}
        assert names == expected

        manifest = json.loads(zf.read("MANIFEST.json"))
        # The embedded canonical hash must equal a fresh recompute (same definition as freeze --check).
        assert manifest["canonical_sha256"] == canonical_hash(mpb.REPO)
        assert set(manifest["files"]) == set(mpb.bundle_members())
        readme = zf.read("DEPOSIT_README.md").decode("utf-8")
        assert manifest["canonical_sha256"] in readme
        # Pre-freeze, the dry-run banner must be present so a premature deposit is impossible to miss.
        if not manifest["frozen"]:
            assert "PRE-FREEZE DRY-RUN" in readme
