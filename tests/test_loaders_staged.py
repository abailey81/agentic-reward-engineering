"""Hermetic tests for the node-staged gold-dir hook (``LLM_RP_GOLD_STAGED_DIR``, V7 / PLAN §14.2).

No licensed data needed: a tiny synthetic gold fixture is built per test (so unlike
``test_loaders.py`` these do NOT skip when the frozen panel is absent). The hook is what makes
cluster jobs read gold at all (a node has no repo ``data/gold/``) AND makes ``$TMPDIR`` staging
load-bearing, so its three safety properties are pinned here:

1. per-file precedence with canonical fallback;
2. suffix-in-filename anti-masquerade (wrong-panel staging can never be silently read);
3. staged bytes are verified against the SAME frozen manifest SHA-256 (basename matching).
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.data.loaders as loaders
from src.data.loaders import gold_suffix, load_gold_panel

RICS = ["AAA.N", "BBB.N", "CCC.N"]


def _mini_gold(root: Path, sfx: str) -> None:
    """The minimal 3-artifact gold set the loader reads (40 sessions, 3 names, dev phase)."""
    root.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range("2016-01-04", periods=40)
    rng = np.random.default_rng(7)
    rets = pd.DataFrame(rng.normal(0.0, 0.01, (40, len(RICS))), index=dates, columns=RICS)
    rets.to_parquet(root / f"returns_panel_{sfx}.parquet")
    pd.DataFrame({"vix": np.full(40, 0.15)}, index=dates).to_parquet(
        root / f"cash_features_{sfx}.parquet"
    )
    pd.DataFrame(
        {"phase": ["development"], "window_start": [dates[0]], "selection": [str(RICS)]}
    ).to_parquet(root / f"top30_selection_{sfx}.parquet")


def test_staged_dir_takes_precedence_per_file_with_canonical_fallback(tmp_path, monkeypatch):
    sfx = gold_suffix()
    canonical, staged = tmp_path / "gold", tmp_path / "tmpfs"
    _mini_gold(canonical, sfx)
    staged.mkdir()
    # returns panel ONLY in staged; cash/top30 ONLY in canonical → proves per-file resolution
    shutil.move(
        str(canonical / f"returns_panel_{sfx}.parquet"),
        str(staged / f"returns_panel_{sfx}.parquet"),
    )
    monkeypatch.setenv("LLM_RP_GOLD_STAGED_DIR", str(staged))
    res = load_gold_panel(gold_dir=canonical)
    assert res.panel.returns.shape == (40, len(RICS))
    assert np.isfinite(res.panel.returns).all()
    # the env var was LOAD-BEARING: without it the canonical dir lacks the returns panel
    monkeypatch.delenv("LLM_RP_GOLD_STAGED_DIR")
    with pytest.raises(FileNotFoundError, match="returns_panel"):
        load_gold_panel(gold_dir=canonical)


def test_wrong_suffix_staging_cannot_masquerade(tmp_path, monkeypatch):
    """Staging a DIFFERENT panel (e.g. univ3 while univ5 is active) must be a filename miss →
    the canonical panel is read; the staged impostor is never touched."""
    sfx = gold_suffix()
    canonical, staged = tmp_path / "gold", tmp_path / "tmpfs"
    _mini_gold(canonical, sfx)
    _mini_gold(staged, "WRONGSFX")  # a full artifact set under a different suffix
    baseline = load_gold_panel(gold_dir=canonical).panel.returns
    monkeypatch.setenv("LLM_RP_GOLD_STAGED_DIR", str(staged))
    got = load_gold_panel(gold_dir=canonical).panel.returns
    assert np.array_equal(got, baseline)


def test_staged_bytes_are_verified_against_the_frozen_manifest(tmp_path, monkeypatch):
    sfx = gold_suffix()
    canonical, staged = tmp_path / "gold", tmp_path / "tmpfs"
    _mini_gold(canonical, sfx)
    # the frozen manifest carries the TRUE bytes' shas; staged paths resolve by BASENAME
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        json.dumps(
            {"relpath": f"data/gold/{p.name}", "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        )
        for p in sorted(canonical.glob("*.parquet"))
    ]
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(loaders, "_MANIFEST", manifest)

    staged.mkdir()
    fname = f"returns_panel_{sfx}.parquet"
    shutil.copy2(canonical / fname, staged / fname)
    monkeypatch.setenv("LLM_RP_GOLD_STAGED_DIR", str(staged))
    # true staged bytes → verification PASSES on the staged copy
    load_gold_panel(gold_dir=canonical, verify_checksum=True)
    # corrupt the staged copy → the SAME frozen sha fails LOUD (no silent wrong-bytes read)
    with (staged / fname).open("ab") as fh:
        fh.write(b"CORRUPT")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_gold_panel(gold_dir=canonical, verify_checksum=True)
