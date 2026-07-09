"""Opt-in SHA-256 freeze verification in the gold loader (``src/data/loaders.py``).

``config/data.yaml`` declares ``freeze: {checksum: sha256}`` and forbids redistribution; the gold
artifacts are frozen and SHA-256-manifested in ``data/manifest/manifest.jsonl``. ``load_gold_panel``
gained an opt-in ``verify_checksum`` flag that compares each parquet's hash to that frozen value.

These tests are SELF-CONTAINED (a tiny temp parquet + temp manifest) so they run without the
licensed gold panel: they verify the three contractual behaviours —
    * match  -> passes (no raise),
    * tamper -> raises ValueError naming a checksum mismatch,
    * absent (no manifest entry / no manifest) -> SKIPS gracefully (no raise).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data import loaders


def _write_parquet(path: Path, payload: int = 0) -> str:
    """Write a tiny deterministic parquet and return its SHA-256 hex digest."""
    df = pd.DataFrame({"a": [payload, payload + 1], "b": [payload + 2, payload + 3]})
    df.to_parquet(path)
    return loaders._file_sha256(path)


def _write_manifest(path: Path, relpath: str, sha256: str) -> None:
    path.write_text(
        json.dumps({"relpath": relpath, "name": Path(relpath).name, "sha256": sha256}) + "\n",
        encoding="utf-8",
    )


def test_file_sha256_matches_hashlib(tmp_path: Path) -> None:
    """The streaming digest equals a one-shot hashlib digest (chunking is correct)."""
    import hashlib

    p = tmp_path / "x.parquet"
    _write_parquet(p)
    assert loaders._file_sha256(p) == hashlib.sha256(p.read_bytes()).hexdigest()


def test_expected_sha256_returns_none_without_manifest(tmp_path: Path) -> None:
    p = tmp_path / "returns_panel_univ3.parquet"
    _write_parquet(p)
    assert loaders._expected_sha256(p, manifest_path=tmp_path / "missing.jsonl") is None


def test_expected_sha256_matches_by_name(tmp_path: Path) -> None:
    p = tmp_path / "returns_panel_univ3.parquet"
    digest = _write_parquet(p)
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, "data/gold/returns_panel_univ3.parquet", digest)
    assert loaders._expected_sha256(p, manifest_path=manifest) == digest


def test_verify_passes_on_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A file whose hash matches the manifest verifies without raising."""
    p = tmp_path / "returns_panel_univ3.parquet"
    digest = _write_parquet(p)
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, "data/gold/returns_panel_univ3.parquet", digest)
    monkeypatch.setattr(loaders, "_MANIFEST", manifest)
    loaders._verify_checksum(p)  # must not raise


def test_verify_raises_on_tamper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A file whose bytes changed after the manifest was frozen must raise."""
    p = tmp_path / "returns_panel_univ3.parquet"
    frozen_digest = _write_parquet(p, payload=0)
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, "data/gold/returns_panel_univ3.parquet", frozen_digest)
    monkeypatch.setattr(loaders, "_MANIFEST", manifest)
    _write_parquet(p, payload=99)  # tamper: rewrite with different content
    with pytest.raises(ValueError, match="checksum mismatch"):
        loaders._verify_checksum(p)


def test_verify_raises_when_absent_from_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No manifest entry for the file -> FAIL LOUD (C2): verification was requested but cannot be proven."""
    p = tmp_path / "returns_panel_univ3.parquet"
    _write_parquet(p)
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, "data/gold/some_other_artifact.parquet", "deadbeef")
    monkeypatch.setattr(loaders, "_MANIFEST", manifest)
    with pytest.raises(ValueError, match="no manifest entry"):
        loaders._verify_checksum(p)  # absent -> raise (C2: silent-skip on the headline panel is the bug)


def test_verify_raises_when_no_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No manifest at all -> FAIL LOUD (C2), not a silent skip."""
    p = tmp_path / "returns_panel_univ3.parquet"
    _write_parquet(p)
    monkeypatch.setattr(loaders, "_MANIFEST", tmp_path / "nope.jsonl")
    with pytest.raises(ValueError, match="no manifest entry"):
        loaders._verify_checksum(p)  # no manifest -> raise (C2)


def test_expected_sha256_exact_relpath_branch_fires(tmp_path: Path) -> None:
    """The EXACT repo-root-relative relpath branch must resolve (Rank 18: parents[1]->parents[2]).

    The manifest convention keys artifacts by a repo-root-relative POSIX path; the repo root is the
    manifest's ``parents[2]`` (``<root>/data/manifest/manifest.jsonl``). The previous ``parents[1]``
    produced a ``data/``-relative key (``gold/…``) that never matched the ``data/``-prefixed relpaths,
    so ONLY the basename fallback fired. Here we lay out ``<tmp>/data/gold/`` + ``<tmp>/data/manifest/``
    and give the entry a DIFFERENT basename from the file, so a match can come ONLY from the exact
    relpath — proving the branch is live.
    """
    gold = tmp_path / "data" / "gold"
    man_dir = tmp_path / "data" / "manifest"
    gold.mkdir(parents=True)
    man_dir.mkdir(parents=True)
    p = gold / "returns_panel_univ3.parquet"
    digest = _write_parquet(p)
    manifest = man_dir / "manifest.jsonl"
    # Basename in the manifest deliberately DIFFERS from the file's basename; only the exact relpath
    # ("data/gold/returns_panel_univ3.parquet") can match it.
    manifest.write_text(
        json.dumps({"relpath": "data/gold/returns_panel_univ3.parquet",
                    "name": "DIFFERENT_BASENAME.parquet", "sha256": digest}) + "\n",
        encoding="utf-8",
    )
    assert loaders._expected_sha256(p, manifest_path=manifest) == digest


def test_expected_sha256_basename_fallback_when_outside_root(tmp_path: Path) -> None:
    """Fallback still works when the file lives OUTSIDE the manifest's repo root (relpath can't match)."""
    p = tmp_path / "elsewhere" / "returns_panel_univ3.parquet"
    p.parent.mkdir(parents=True)
    digest = _write_parquet(p)
    manifest = tmp_path / "mani" / "manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    _write_manifest(manifest, "data/gold/returns_panel_univ3.parquet", digest)  # relpath won't match
    assert loaders._expected_sha256(p, manifest_path=manifest) == digest  # basename match


# --------------------------------------------------------------------------- #
# C1 — gold_suffix() precedence: config PRIMARY, env var EXPLICIT override      #
# --------------------------------------------------------------------------- #
def _write_data_yaml(path: Path, suffix: str | None) -> None:
    body = "period: {start: 2005-01-01, end: 2025-12-31}\n"
    if suffix is not None:
        body += f"gold:\n  suffix: {suffix}\n"
    path.write_text(body, encoding="utf-8")


def test_config_gold_suffix_reads_data_yaml(tmp_path: Path) -> None:
    """``_config_gold_suffix`` returns config/data.yaml's ``gold.suffix`` (stripped of a leading _)."""
    y = tmp_path / "data.yaml"
    _write_data_yaml(y, "_univ9")
    assert loaders._config_gold_suffix(y) == "univ9"


def test_config_gold_suffix_none_when_absent(tmp_path: Path) -> None:
    """No ``gold.suffix`` key -> None (caller falls back to the default)."""
    y = tmp_path / "data.yaml"
    _write_data_yaml(y, None)
    assert loaders._config_gold_suffix(y) is None
    assert loaders._config_gold_suffix(tmp_path / "missing.yaml") is None


def test_gold_suffix_config_is_primary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no env var set, gold_suffix() uses config/data.yaml's ``gold.suffix`` (C1 PRIMARY source)."""
    y = tmp_path / "data.yaml"
    _write_data_yaml(y, "univ3")
    monkeypatch.setattr(loaders, "_DATA_YAML", y)
    monkeypatch.delenv("LLM_RP_GOLD_SUFFIX", raising=False)
    assert loaders.gold_suffix() == "univ3"


def test_gold_suffix_env_overrides_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``LLM_RP_GOLD_SUFFIX`` is an EXPLICIT override that wins over the config value (C1)."""
    y = tmp_path / "data.yaml"
    _write_data_yaml(y, "univ3")
    monkeypatch.setattr(loaders, "_DATA_YAML", y)
    monkeypatch.setenv("LLM_RP_GOLD_SUFFIX", "univ4")
    assert loaders.gold_suffix() == "univ4"


def test_gold_suffix_falls_back_to_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No env var AND no config key -> the univ5 default (minimal/synthetic install; Split C, ADR-044/051)."""
    y = tmp_path / "data.yaml"
    _write_data_yaml(y, None)
    monkeypatch.setattr(loaders, "_DATA_YAML", y)
    monkeypatch.delenv("LLM_RP_GOLD_SUFFIX", raising=False)
    assert loaders.gold_suffix() == loaders._DEFAULT_SUFFIX == "univ5"


def test_read_verify_checksum_tamper_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``_read(..., verify_checksum=True)`` enforces the freeze on the actual read path."""
    name = "returns_panel"
    p = tmp_path / f"{name}_{loaders._SUFFIX}.parquet"
    frozen_digest = _write_parquet(p, payload=1)
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, f"data/gold/{name}_{loaders._SUFFIX}.parquet", frozen_digest)
    monkeypatch.setattr(loaders, "_MANIFEST", manifest)

    # match -> reads fine
    out = loaders._read(name, tmp_path, verify_checksum=True)
    assert list(out.columns) == ["a", "b"]

    # tamper -> raises
    _write_parquet(p, payload=7)
    with pytest.raises(ValueError, match="checksum mismatch"):
        loaders._read(name, tmp_path, verify_checksum=True)

    # opt-out (default) -> never verifies, reads the tampered file without raising
    out2 = loaders._read(name, tmp_path)
    assert list(out2.columns) == ["a", "b"]
