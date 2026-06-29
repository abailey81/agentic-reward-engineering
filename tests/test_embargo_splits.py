"""Embargo at the EXECUTED search-split boundary (PREREGISTRATION §7; Rank 18).

The prototype orchestrators (``scripts/run_prototype.py::_load_panel_and_windows`` and
``src/orchestration/parallel.py::_panel_and_windows``) used to ABUT train/val (``val_start ==
train_end + 1``), with NO purge+embargo — violating the frozen pre-registration. They now resolve the
val start through ``src.data.loaders.embargoed_val_start(lookback=...)``, returning ``max(materialized
boundary, first_post_train + max(embargo, lookback))`` (R18). The materialized boundary is the embargo
FLOOR and is honoured only when it sits after the train end — SPLIT-C NOTE (ADR-044): the frozen tables
(incl. ``splits_univ5``) still record the PRE-Split-C dev boundary (2015-02-03), which predates the
Split-C train_end (2016-12-31) and is therefore INERT; the +21 embargo floor would be ~2017-02-02, but
the production lookback=60 purge DOMINATES, so the EXECUTED val start is 2017-03-30 (train_end + 60
sessions) — a byte-match to a materialized boundary holds only at ``lookback=0`` with a post-train
boundary (the per-unit tests below). It (2) FALLS BACK to advancing ``max(embargo, lookback)`` trading
sessions when the split table omits the boundary.

These tests are fast: the unit tests build a synthetic ascending date index + a temp splits parquet (no
gold load); one end-to-end test exercises the real ``_load_panel_and_windows`` and is skipped when the
licensed gold panel is absent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.loaders import embargoed_val_start, gold_suffix

_GOLD_DIR = Path(__file__).resolve().parents[1] / "data" / "gold"
_GOLD = _GOLD_DIR / f"returns_panel_{gold_suffix()}.parquet"  # the ACTIVE panel (univ5 post-Split-C)
_SPLITS_UNIV3 = _GOLD_DIR / "splits_univ3.parquet"  # the FROZEN pre-Split-C reference table
_SPLITS_ACTIVE = _GOLD_DIR / f"splits_{gold_suffix()}.parquet"  # the table the loader actually reads


def _business_dates(start: str, n: int) -> np.ndarray:
    """``n`` ascending business-day sessions as ``datetime64[ns]`` (a stand-in trading calendar)."""
    return pd.bdate_range(start=start, periods=n).to_numpy()


def _write_splits(gold_dir: Path, val_post_embargo_start: str, embargo: int = 21) -> None:
    """Write a SPLIT-C-shaped split table under the loader's ``splits_<gold_suffix()>`` convention."""
    payload = {
        "development": {"train": ["2005-01-03", "2016-12-31"],
                        "validation_post_embargo": [val_post_embargo_start, "2019-12-31"]},
        "embargo_trading_days": embargo,
    }
    path = gold_dir / f"splits_{gold_suffix()}.parquet"
    pd.DataFrame({"splits_json": [json.dumps(payload)]}).to_parquet(path)


def test_materialized_boundary_is_honoured(tmp_path: Path) -> None:
    """When the split table carries a POST-train boundary, val_start lands EXACTLY on it (byte-match).

    Uses the Split-C dates: train_end 2016-12-31 with a rematerialized +21 boundary 2017-02-02 (the
    embargo floor ADR-044 implies; the FROZEN on-disk tables still carry the stale pre-Split-C one).
    """
    dates = _business_dates("2016-12-01", 80)  # spans the 2016->2017 boundary
    _write_splits(tmp_path, "2017-02-02", embargo=21)
    idx = embargoed_val_start(dates, "2016-12-31", phase="development", gold_dir=tmp_path)
    assert pd.Timestamp(dates[idx]) == pd.Timestamp("2017-02-02")


def test_embargo_is_non_empty_vs_abutting_start(tmp_path: Path) -> None:
    """The embargoed val_start sits strictly AFTER the old abutting (first post-train) start."""
    dates = _business_dates("2016-12-01", 80)
    _write_splits(tmp_path, "2017-02-02", embargo=21)
    # side='right': the first post-train position even when train_end is a non-session date
    # (2016-12-31 is a Saturday) — the same convention embargoed_val_start applies internally.
    abut = int(np.searchsorted(dates, np.datetime64(pd.Timestamp("2016-12-31")), side="right"))
    idx = embargoed_val_start(dates, "2016-12-31", phase="development", gold_dir=tmp_path)
    assert idx > abut  # a real purge gap, not abutting


def test_fallback_purges_embargo_trading_days(tmp_path: Path) -> None:
    """No split table -> purge exactly ``embargo_days`` SESSIONS (not calendar days) after train_end."""
    dates = _business_dates("2015-01-01", 60)  # bdate skips weekends -> a stand-in trading calendar
    train_end = str(pd.Timestamp(dates[10]))  # an in-range session to measure the embargo from
    train_idx = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(train_end))))
    abut = train_idx + 1  # first post-train session (the OLD abutting val start)
    # gold_dir points at an EMPTY dir => no splits parquet => fallback path.
    idx = embargoed_val_start(dates, train_end, phase="development", embargo_days=21, gold_dir=tmp_path)
    assert idx - abut == 21  # the dropped [abut, idx) sessions number exactly 21


def test_lookback_purge_covers_feature_window(tmp_path: Path) -> None:
    """R18 (leakage fix): with a feature ``lookback``, the purge is max(embargo, lookback), so the
    downstream window's first observation cannot read a return from the prior split. ``lookback=0``
    (the default) preserves the legacy embargo-only gap."""
    dates = _business_dates("2015-01-01", 200)
    train_end = str(pd.Timestamp(dates[40]))
    abut = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(train_end)))) + 1
    g0 = embargoed_val_start(dates, train_end, embargo_days=21, lookback=0, gold_dir=tmp_path)
    g60 = embargoed_val_start(dates, train_end, embargo_days=21, lookback=60, gold_dir=tmp_path)
    assert g0 - abut == 21  # legacy embargo-only behaviour preserved
    assert g60 - abut >= 60  # the purge now covers the 60-day feature lookback
    assert g60 - 60 >= abut  # the first val feature window does NOT reach train


def test_fallback_default_embargo_is_21(tmp_path: Path) -> None:
    """The fallback default (no arg) purges 21 trading sessions, matching config/data.yaml::embargo_days."""
    dates = _business_dates("2015-01-01", 60)
    train_end = str(pd.Timestamp(dates[5]))
    abut = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(train_end)))) + 1
    idx = embargoed_val_start(dates, train_end, phase="development", gold_dir=tmp_path)
    assert idx - abut == 21


def test_fallback_when_boundary_absent_but_embargo_present(tmp_path: Path) -> None:
    """A split table WITHOUT validation_post_embargo still yields a 21-session purge from its embargo field."""
    payload = {"development": {"train": ["2005-01-03", "2016-12-31"]}, "embargo_trading_days": 21}
    pd.DataFrame({"splits_json": [json.dumps(payload)]}).to_parquet(
        tmp_path / f"splits_{gold_suffix()}.parquet"
    )
    dates = _business_dates("2015-01-01", 60)
    train_end = str(pd.Timestamp(dates[5]))
    abut = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(train_end)))) + 1
    idx = embargoed_val_start(dates, train_end, phase="development", gold_dir=tmp_path)
    assert idx - abut == 21


@pytest.mark.skipif(not _SPLITS_UNIV3.exists(), reason="frozen splits_univ3.parquet not present (licensed data)")
def test_frozen_splits_carry_dev_post_embargo_boundary() -> None:
    """PINS the FROZEN pre-Split-C univ3 table: dev boundary 2015-02-03, 21-trading-day embargo.

    Historical reference only — the ACTIVE Split-C behaviour of that (stale) boundary is asserted in
    ``test_active_splits_stale_dev_boundary_is_inert_under_split_c`` below."""
    cell = pd.read_parquet(_SPLITS_UNIV3)["splits_json"].iloc[0]
    splits = json.loads(cell)
    assert splits["embargo_trading_days"] == 21
    assert splits["development"]["validation_post_embargo"][0] == "2015-02-03"


@pytest.mark.skipif(not _SPLITS_ACTIVE.exists(), reason="active splits table not present (licensed data)")
def test_active_splits_stale_dev_boundary_is_inert_under_split_c() -> None:
    """The ACTIVE table's dev boundary predates the Split-C train end -> inert; the purge governs.

    ``splits_univ5.parquet`` was rebuilt with the panel but still records the PRE-Split-C dev
    boundary (2015-02-03). Under the Split-C train_end (2016-12-31) that floor sits BEFORE the
    train end, so ``embargoed_val_start`` must ignore it and execute the R18 lookback purge:
    val_start = first post-train session + 60 (the ratified 2017-03-30 on the real calendar).
    """
    cell = pd.read_parquet(_SPLITS_ACTIVE)["splits_json"].iloc[0]
    splits = json.loads(cell)
    boundary = pd.Timestamp(splits["development"]["validation_post_embargo"][0])
    assert boundary < pd.Timestamp("2016-12-31")  # stale (pre-Split-C) -> must be inert

    dates = _business_dates("2016-10-03", 200)  # a stand-in calendar spanning the Split-C boundary
    abut = int(np.searchsorted(dates, np.datetime64(pd.Timestamp("2016-12-31")), side="right"))
    idx = embargoed_val_start(
        dates, "2016-12-31", phase="development", embargo_days=21, lookback=60, gold_dir=_GOLD_DIR
    )
    assert idx - abut == 60  # the lookback purge, NOT the stale materialized boundary


@pytest.mark.skipif(not _GOLD.exists(), reason="frozen gold panel not present (licensed data)")
def test_executed_prototype_val_start_respects_embargo() -> None:
    """END-TO-END: the EXECUTED prototype window purges the FEATURE LOOKBACK at the train/val boundary.

    Drives the real ``_load_panel_and_windows`` against the frozen gold panel and asserts the val
    window starts at least ``lookback`` (60) trading sessions after the last TRAIN session, so the
    val window's first observation's 60-day feature lookback does NOT reach the train window (R18
    leakage fix, 2026-06-20: the prior 21-session embargo left 39 contaminated observations).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_prototype  # noqa: WPS433 - test-time import of the script module

    data_cfg = {"phase": "development", "train_end": "2016-12-31", "val_end": "2019-12-31",
                "embargo_days": 21, "on_missing": "liquidate_to_cash"}
    panel, (t_lo, t_hi), (v_lo, v_hi) = run_prototype._load_panel_and_windows(
        synthetic=False, data_cfg=data_cfg, lookback=60
    )
    dates = np.asarray(panel.dates)
    # side='right' = the first post-train session even though the Split-C train_end (2016-12-31) is a
    # Saturday — the same half-open convention the executed paths + resolve_windows now share.
    abut = int(np.searchsorted(dates, np.datetime64(pd.Timestamp("2016-12-31")), side="right"))
    lookback = 60
    assert t_hi == abut, "the train window must END at the split boundary (no val-side session leaks in)"
    assert v_lo > abut, "executed train/val windows must NOT abut (embargo required)"
    # R18 (2026-06-20 leakage fix): the PURGE must cover the FEATURE LOOKBACK (60), not merely the
    # embargo (21). Each observation reads returns[t-lookback:t]; with only a 21-session gap the val
    # window's first (lookback - embargo) = 39 observations read TRAIN returns (returns[v_lo-lookback:
    # v_lo] reaches back across the boundary) — a López de Prado purge-insufficiency. The effective
    # purge is now max(embargo, lookback) = 60, and CRUCIALLY the first val feature window clears train.
    assert v_lo - abut >= lookback, "the purge must cover the feature lookback (>= 60 sessions)"
    assert v_lo - lookback >= abut, "the val window's first lookback must NOT reach the train window"
    # ADR-044 ratified EXECUTED start: 2017-03-30 = train_end + 60 sessions (the stale materialized
    # boundary 2015-02-03 is inert; matches resolve_windows' recorded univ5 val start, index 3081).
    assert str(dates[v_lo])[:10] == "2017-03-30"
    assert v_hi == panel.T and v_lo < v_hi  # non-empty validation window


@pytest.mark.skipif(not _GOLD.exists(), reason="frozen gold panel not present (licensed data)")
def test_both_window_resolvers_agree_with_the_univ5_registry() -> None:
    """CROSS-RESOLVER PIN (review batch-4 #6): two independent purge-arithmetic implementations
    exist — the prototype/σ-pilot path (``run_prototype._load_panel_and_windows`` →
    ``loaders.embargoed_val_start``) and the headline campaign path (``run_campaign.resolve_windows``).
    They agree on univ5 today (both derive val_start=3081); this pins BOTH to the recorded
    ``config/inference.yaml: splits.expected_windows.univ5`` so a future edit to EITHER resolver fails
    HERE instead of silently desyncing the σ-pilot's train/val windows from the campaign's.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_campaign  # noqa: WPS433 - test-time import of the script module
    import run_prototype  # noqa: WPS433

    from src.data.loaders import load_gold_panel
    from src.utils.config import load_config

    splits = load_config("inference")["splits"]
    expected = splits["expected_windows"]["univ5"]
    want_train = [int(x) for x in expected["train"]]
    want_val = [int(x) for x in expected["val"]]
    want_test = [int(x) for x in expected["test"]]
    lookback = int(load_config("environment")["state"]["lookback_days"])
    embargo = int(load_config("data").get("embargo_days", 21))

    # Headline resolver: the full-span (test-inclusive) panel + the frozen calendar splits — the
    # exact load run_campaign's headline path performs (checksum-verified).
    ev_end = str(splits["evaluation"]["span"][1])
    gold = load_gold_panel(phase="development", end=ev_end, verify_checksum=True)
    tw, vw, xw = run_campaign.resolve_windows(gold.panel, lookback, splits, embargo=embargo)
    assert [list(tw), list(vw), list(xw)] == [want_train, want_val, want_test]

    # Prototype/σ-pilot resolver: the dev-span panel; its train/val must equal the registry's dev
    # windows (its panel ends at val_end, so the val window's end == panel.T == the registry's).
    data_cfg = {
        "phase": "development",
        "train_end": str(splits["development"]["train"][1]),
        "val_end": str(splits["development"]["validation"][1]),
        "embargo_days": embargo,
        "on_missing": "liquidate_to_cash",
    }
    panel, t_win, v_win = run_prototype._load_panel_and_windows(
        synthetic=False, data_cfg=data_cfg, lookback=lookback
    )
    assert list(t_win) == want_train
    assert list(v_win) == want_val
