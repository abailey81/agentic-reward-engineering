"""Tests for the point-in-time, survivorship-free data pipeline (FINAL_PLAN F.15)."""

from __future__ import annotations


import numpy as np

from src.data.panel import Panel
from src.data.pipeline import GoldPipeline
from src.utils.config import load_config


def _pipeline(seed: int = 12345) -> GoldPipeline:
    cfg = load_config("data")
    return GoldPipeline(cfg, n_assets=30, top_n=8, n_days=600, seed=seed)


def test_run_returns_valid_panel_of_configured_shape() -> None:
    """pipeline.run() returns a valid Panel of the configured shape."""
    panel, manifest = _pipeline().run()
    assert isinstance(panel, Panel)
    assert panel.N == 8
    assert panel.T == 600
    assert np.isfinite(panel.returns).all()


def test_top_n_selection_picks_assets_by_market_cap() -> None:
    """top-N selection picks the N highest-market-cap assets (PIT, first day)."""
    pipe = _pipeline()
    pipe.run()
    caps_first_day = pipe._caps[0]
    expected = np.sort(np.argsort(caps_first_day)[::-1][: pipe.top_n])
    np.testing.assert_array_equal(pipe._selected, expected)
    # the selected assets must be the top-N by cap
    threshold = np.sort(caps_first_day)[::-1][pipe.top_n - 1]
    assert (caps_first_day[pipe._selected] >= threshold).all()


def test_split_indices_contiguous_nonoverlapping_with_embargo() -> None:
    """Split indices are contiguous, non-overlapping and embargo-separated."""
    _, manifest = _pipeline().run()
    splits = manifest["splits"]
    # The purge moved OUT of the (now homogeneous) splits dict to manifest['embargo_days'] = the APPLIED
    # gap max(embargo_days, lookback_days) (R18; audit 2026-06-20) — read it from the new location.
    purge = manifest["embargo_days"]
    train, val, test = splits["train"], splits["val"], splits["test"]

    # within each split: start <= end
    for s in (train, val, test):
        assert s["start"] <= s["end"]

    # ordered, non-overlapping, separated by the FULL purge — which must cover the 60-day feature lookback,
    # not just the 21-day embargo (else a later split's first (lookback-embargo) windows read prior returns).
    assert purge >= 60  # == max(embargo_days=21, lookback_days=60); regression-locks R18 against an embargo-only revert
    assert val["start"] >= train["end"] + purge
    assert test["start"] >= val["end"] + purge


def test_manifest_has_checksum_per_stage_and_is_deterministic() -> None:
    """The manifest has a checksum per stage and is deterministic for a fixed seed."""
    _, m1 = _pipeline(seed=999).run()
    _, m2 = _pipeline(seed=999).run()

    stage_keys = [k for k in m1 if k[:2].isdigit()]
    assert len(stage_keys) == 13  # one checksum per stage
    for k in stage_keys:
        assert isinstance(m1[k], str) and len(m1[k]) == 64  # sha256 hexdigest
        assert m1[k] == m2[k]  # deterministic

    # different seed => different data checksums
    _, m3 = _pipeline(seed=1).run()
    assert m3["02_ingest_prices_returns"] != m1["02_ingest_prices_returns"]


def test_delisting_path_exercised() -> None:
    """The delisting-correction path runs (synthetic source has no real delistings)."""
    pipe = _pipeline()
    pipe.run()
    # documented: synthetic data carries no delistings, so the event table is empty
    assert pipe._delisting == {"delisted_assets": [], "delisting_returns": {}}
    assert "04_ingest_corporate_actions" in pipe.manifest
    assert "07_correct_delisting_returns" in pipe.manifest
