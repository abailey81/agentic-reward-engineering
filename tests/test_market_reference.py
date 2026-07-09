"""Tests for the reference-data loaders (risk-free, market proxy, Fama-French factors; block B11/B13).

Behaviour: correct unit conversion, alignment to a date axis, no-future-leak forward fill, and graceful
degradation to an empty/zero series with ``available=False`` on a synthetic-only install (files absent).
The against-real-gold checks skip when the licensed files are not present.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.market_reference import (
    TRADING_DAYS_PER_YEAR,
    load_ff_factors,
    load_market_proxy_returns,
    load_risk_free_daily,
    load_spx_total_return,
)

from src.data.loaders import gold_suffix

_FRED = Path("data/raw/fred_macro.csv")
_MKT = Path(f"data/gold/market_proxy_{gold_suffix()}.parquet")  # ACTIVE suffix (Split C: univ5)
_FF = Path("data/raw/french_F-F_Research_Data_Factors_daily.csv")


# --------------------------------------------------------------------------- #
# graceful degradation (synthetic-only install: files absent)                 #
# --------------------------------------------------------------------------- #
def test_risk_free_absent_is_zero_and_flagged(tmp_path: Path) -> None:
    dates = pd.bdate_range("2020-01-01", periods=50).to_numpy()
    rf = load_risk_free_daily(dates, raw_dir=tmp_path)
    assert rf.available is False
    assert rf.daily.shape == (50,) and np.all(rf.daily == 0.0)


def test_market_proxy_absent_is_zero_and_flagged(tmp_path: Path) -> None:
    dates = pd.bdate_range("2020-01-01", periods=40).to_numpy()
    mp = load_market_proxy_returns(dates, gold_dir=tmp_path)
    assert mp.available is False
    assert mp.returns.shape == (40,) and np.all(mp.returns == 0.0)


def test_ff_absent_is_empty_and_flagged(tmp_path: Path) -> None:
    dates = pd.bdate_range("2020-01-01", periods=40).to_numpy()
    ff = load_ff_factors(dates, raw_dir=tmp_path)
    assert ff.available is False and ff.factors == {}


# --------------------------------------------------------------------------- #
# unit conversion + alignment + no-future-leak (synthetic fixtures)           #
# --------------------------------------------------------------------------- #
def test_risk_free_converts_annual_pct_to_daily_decimal(tmp_path: Path) -> None:
    # a constant 2.52% annualised yield -> per-session (1+0.0252)**(1/252)-1
    dates = pd.bdate_range("2021-01-04", periods=10)
    df = pd.DataFrame({"observation_date": dates.strftime("%Y-%m-%d"), "DGS3MO": [2.52] * 10})
    df.to_csv(tmp_path / "fred_macro.csv", index=False)
    rf = load_risk_free_daily(dates.to_numpy(), raw_dir=tmp_path)
    assert rf.available is True
    expected = (1.0 + 2.52 / 100.0) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    assert rf.daily == pytest.approx(expected, rel=1e-9)
    assert rf.annual_pct_mean == pytest.approx(2.52, rel=1e-9)


def test_risk_free_forward_fills_gaps_never_future(tmp_path: Path) -> None:
    # publish a yield only on day 0 and day 5; days 1-4 must read day-0 (the LAST KNOWN), not day-5.
    dates = pd.bdate_range("2021-01-04", periods=6)
    yields = [1.00, np.nan, np.nan, np.nan, np.nan, 5.00]
    pd.DataFrame({"observation_date": dates.strftime("%Y-%m-%d"), "DGS3MO": yields}).to_csv(
        tmp_path / "fred_macro.csv", index=False
    )
    rf = load_risk_free_daily(dates.to_numpy(), raw_dir=tmp_path)
    d1 = (1.0 + 1.00 / 100.0) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    assert rf.daily[1] == pytest.approx(d1, rel=1e-9)  # forward-filled from day 0, NOT day 5
    assert rf.daily[3] == pytest.approx(d1, rel=1e-9)


def test_raw_path_prefers_refreshed_artifact(tmp_path: Path) -> None:
    """ADR-051: readers prefer the versioned extension refresh (x26) when present, else canonical."""
    from src.data.market_reference import _FF_CSV, _FRED_CSV, _raw_path

    # neither present -> canonical path returned (caller handles absence)
    assert _raw_path(tmp_path, _FRED_CSV).name == _FRED_CSV
    # refresh present -> preferred
    (tmp_path / "fred_macro_x26.csv").write_text("observation_date,DGS3MO\n2026-06-30,4.0\n")
    assert _raw_path(tmp_path, _FRED_CSV).name == "fred_macro_x26.csv"
    (tmp_path / "french_ff3_daily_x26.csv").write_text("Date,Mkt-RF,SMB,HML,RF\n2026-05-29,0.001,0.0,0.0,0.0\n")
    assert _raw_path(tmp_path, _FF_CSV).name == "french_ff3_daily_x26.csv"


def test_ff_factors_load_from_refreshed_file(tmp_path: Path) -> None:
    """The x26 ff3 layout (Date-indexed, decimal factors) loads through load_ff_factors unchanged."""
    idx = pd.bdate_range("2026-05-26", periods=4)
    rows = "\n".join(f"{d.date()},0.001,0.0002,-0.0003,0.0001" for d in idx)
    (tmp_path / "french_ff3_daily_x26.csv").write_text("Date,Mkt-RF,SMB,HML,RF\n" + rows + "\n")
    res = load_ff_factors(idx.to_numpy(), raw_dir=tmp_path)
    assert res.available is True
    assert res.factors["Mkt-RF"] == pytest.approx([0.001] * 4)


def test_market_proxy_aligns_and_zero_fills_outside(tmp_path: Path) -> None:
    idx = pd.bdate_range("2021-01-04", periods=5)
    pd.DataFrame({"market_ew": [0.01, -0.02, 0.0, 0.005, -0.001]}, index=idx).to_parquet(
        tmp_path / f"market_proxy_{gold_suffix()}.parquet"
    )
    mp = load_market_proxy_returns(idx.to_numpy(), gold_dir=tmp_path)
    assert mp.available is True
    assert mp.returns == pytest.approx([0.01, -0.02, 0.0, 0.005, -0.001])


# --------------------------------------------------------------------------- #
# against the real frozen reference data (skip if licensed files absent)      #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _FRED.exists(), reason="frozen FRED macro not present")
def test_real_risk_free_is_plausible() -> None:
    dates = pd.bdate_range("2005-01-03", "2017-12-29").to_numpy()
    rf = load_risk_free_daily(dates)
    assert rf.available and np.isfinite(rf.daily).all()
    assert 0.0 <= rf.daily.mean() * TRADING_DAYS_PER_YEAR < 0.10  # ~1-2%/yr over 2005-2017 (incl. ZIRP)


@pytest.mark.skipif(not _MKT.exists(), reason="frozen market proxy not present")
def test_real_market_proxy_is_a_real_series() -> None:
    dates = pd.bdate_range("2005-01-03", "2017-12-29").to_numpy()
    mp = load_market_proxy_returns(dates)
    assert mp.available and np.isfinite(mp.returns).all()
    assert mp.returns.std() > 0.0  # a genuine market line, not a constant 1/N stand-in


@pytest.mark.skipif(not _FF.exists(), reason="frozen Fama-French factors not present")
def test_real_ff_factors_present() -> None:
    dates = pd.bdate_range("2005-01-03", "2017-12-29").to_numpy()
    ff = load_ff_factors(dates)
    assert ff.available and "Mkt-RF" in ff.factors
    assert np.isfinite(ff.factors["Mkt-RF"]).all()


def test_forward_fill_past_the_source_end_is_COUNTED_and_WARNED_not_silent(tmp_path, caplog) -> None:
    """A stale raw pull must be detectable, not silently extrapolated (deep review #53, loop 80).

    The forward-fill that correctly bridges a publication gap also extends the LAST value past the end
    of the source file for as long as the target axis runs. That is not hypothetical: this module's
    own ``_REFRESHED_RAW`` note records it happening once (the Momentum refresh had no mapping, so
    attribution "silently used the canonical file ending 2026-04-30 and forward-filled the test
    window's tail with a constant"). A mapping was added; no DETECTOR was, so the condition recurred —
    MEASURED on the live repo, the French dailies end 2026-05-29 against a frozen test window running
    to 2026-06-30, leaving 21 of 1631 sessions (1.3%) of the factor ladder on repeated values with
    ``available=True``. The values are deliberately unchanged (inventing factor data would be worse);
    this pins that the condition is now COUNTED and LOGGED."""
    import logging

    import pandas as pd

    from src.data.market_reference import load_ff_factors, load_risk_free_daily

    # a source that stops 5 sessions before the target axis ends
    src_dates = pd.bdate_range("2024-01-01", periods=20)
    tgt_dates = pd.bdate_range("2024-01-01", periods=25).values

    raw = tmp_path
    pd.DataFrame({"observation_date": src_dates, "DGS3MO": np.linspace(4.0, 5.0, 20)}).to_csv(
        raw / "fred_macro.csv", index=False
    )
    with caplog.at_level(logging.WARNING):
        rf = load_risk_free_daily(tgt_dates, raw_dir=raw)
    assert rf.available is True
    assert rf.n_extrapolated == 5, f"expected 5 extrapolated sessions, got {rf.n_extrapolated}"
    assert rf.last_observation == str(src_dates[-1])[:10]
    assert "market_reference_EXTRAPOLATED" in caplog.text
    # the carried value really is the last real one, repeated (no future read, no invention)
    assert rf.daily[-1] == rf.daily[-5]

    # FULLY-COVERED source -> silent, zero count (the guard must not cry wolf)
    caplog.clear()
    pd.DataFrame({"observation_date": pd.bdate_range("2024-01-01", periods=25),
                  "DGS3MO": np.linspace(4.0, 5.0, 25)}).to_csv(raw / "fred_macro.csv", index=False)
    with caplog.at_level(logging.WARNING):
        rf_ok = load_risk_free_daily(tgt_dates, raw_dir=raw)
    assert rf_ok.n_extrapolated == 0
    assert "market_reference_EXTRAPOLATED" not in caplog.text

    # a factor set is only as fresh as its STALEST column
    ff_dates = pd.bdate_range("2024-01-01", periods=20)
    ff = pd.DataFrame({"Date": ff_dates, "Mkt-RF": 0.001, "SMB": 0.001, "HML": 0.001})
    ff.loc[ff.index[-4:], "SMB"] = np.nan          # SMB stops 4 sessions earlier than the rest
    ff.to_csv(raw / "french_F-F_Research_Data_Factors_daily.csv", index=False)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        fr = load_ff_factors(tgt_dates, raw_dir=raw)
    assert fr.available is True
    assert fr.last_observation == str(ff_dates[-5])[:10], "the stalest column must govern"
    assert fr.n_extrapolated == 9, f"5 beyond the file + 4 stale SMB rows; got {fr.n_extrapolated}"


# --------------------------------------------------------------------------- #
# .SPXTR — the CAP-WEIGHTED market line (added 2026-07-30)                     #
#                                                                             #
# The data had been pulled, frozen with provenance on 2026-07-01 and left      #
# UNLOADED for a month, while the module docstring called a cap-weighted index #
# "a documented limitation". These tests exist so that cannot recur silently.  #
# --------------------------------------------------------------------------- #
def _write_spxtr(raw: Path, dates: list[str], levels: list[float], name: str) -> None:
    """Write a minimal Refinitiv-shaped .SPXTR csv (Date + the TRDPRC_1 level column)."""
    pd.DataFrame({"Date": dates, "TRDPRC_1": levels}).to_csv(raw / name, index=False)


def test_spxtr_absent_is_zero_and_flagged(tmp_path: Path) -> None:
    """A synthetic-only install has no licensed pull: degrade, never crash."""
    res = load_spx_total_return(np.array(["2020-01-02", "2020-01-03"], dtype="datetime64[D]"),
                               raw_dir=tmp_path)
    assert res.available is False
    assert res.returns.tolist() == [0.0, 0.0]


def test_spxtr_concatenates_the_base_and_the_2026_extension(tmp_path: Path) -> None:
    """The base pull ends 2025-12-31 and `_x26` carries 2026 — the sealed window needs BOTH.

    Reading either file alone silently truncates the test window, which is exactly the class of
    error that produced the section-36 benchmark-window retraction.
    """
    _write_spxtr(tmp_path, ["2025-12-30", "2025-12-31"], [100.0, 101.0], "rf_spxtr.csv")
    _write_spxtr(tmp_path, ["2026-01-02", "2026-01-05"], [102.0, 103.0], "rf_spxtr_x26.csv")
    dates = np.array(["2025-12-31", "2026-01-02", "2026-01-05"], dtype="datetime64[D]")
    res = load_spx_total_return(dates, raw_dir=tmp_path)
    assert res.available is True
    assert res.last_observation == "2026-01-05", "the extension must extend the history"
    assert res.n_extrapolated == 0
    # 101 -> 102 -> 103
    assert res.returns[1] == pytest.approx(102.0 / 101.0 - 1.0)
    assert res.returns[2] == pytest.approx(103.0 / 102.0 - 1.0)


def test_spxtr_differences_the_ALIGNED_level_not_the_source_level(tmp_path: Path) -> None:
    """THE ORDER-OF-OPERATIONS TEST — the one that matters.

    The files store a LEVEL. If a future edit differences on the SOURCE axis and then forward-fills
    the RETURNS, a session the index did not publish would REPEAT the previous return, booking the
    same market move twice. Forward-filling the LEVEL first and differencing second makes a
    non-publication session correctly 0.0.

    Here 2026-01-06 is missing from the source, so: 100 -> 110 (+10%), then a flat session (0.0),
    then 110 -> 121 (+10%). A repeat-the-return implementation would give +10% on the flat session
    and fail this test.
    """
    _write_spxtr(tmp_path, ["2026-01-05", "2026-01-07", "2026-01-08"], [100.0, 110.0, 121.0],
                 "rf_spxtr.csv")
    dates = np.array(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"], dtype="datetime64[D]")
    res = load_spx_total_return(dates, raw_dir=tmp_path)
    assert res.returns[0] == pytest.approx(0.0), "leading gap: no prior level, so 0.0"
    assert res.returns[1] == pytest.approx(0.0), "a non-publication session must NOT repeat a return"
    assert res.returns[2] == pytest.approx(0.10)
    assert res.returns[3] == pytest.approx(121.0 / 110.0 - 1.0)


def test_spxtr_never_reads_the_future_and_counts_the_extrapolated_tail(tmp_path: Path) -> None:
    """Sessions beyond the last real observation carry a constant level (=> 0.0 return) and are COUNTED.

    Same provenance contract as the RF/FF/market loaders (deep review #53): a forward-filled tail is
    not data, and the caller must be able to see how much of it there is.
    """
    _write_spxtr(tmp_path, ["2026-01-05", "2026-01-06"], [100.0, 105.0], "rf_spxtr.csv")
    dates = np.array(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"], dtype="datetime64[D]")
    res = load_spx_total_return(dates, raw_dir=tmp_path)
    assert res.last_observation == "2026-01-06"
    assert res.n_extrapolated == 2, "two target sessions fall beyond the last real observation"
    assert res.returns[2] == pytest.approx(0.0)
    assert res.returns[3] == pytest.approx(0.0)


def test_spxtr_prefers_the_last_reading_on_an_overlapping_boundary(tmp_path: Path) -> None:
    """The base file and the extension can both carry the boundary session; keep the LAST."""
    _write_spxtr(tmp_path, ["2025-12-31"], [100.0], "rf_spxtr.csv")
    _write_spxtr(tmp_path, ["2025-12-31", "2026-01-02"], [200.0, 220.0], "rf_spxtr_x26.csv")
    dates = np.array(["2025-12-31", "2026-01-02"], dtype="datetime64[D]")
    res = load_spx_total_return(dates, raw_dir=tmp_path)
    assert res.returns[1] == pytest.approx(0.10), "220/200 - 1, i.e. the extension's reading won"


@pytest.mark.skipif(not Path("data/raw/rf_spxtr.csv").exists(),
                    reason="licensed .SPXTR pull not present (synthetic-only install)")
def test_spxtr_real_pull_covers_the_sealed_window() -> None:
    """On the real install the pull must span the sealed window 2020-03-30 -> 2026-06-30."""
    dates = pd.bdate_range("2020-03-30", "2026-06-30").to_numpy()
    res = load_spx_total_return(dates)
    assert res.available is True
    assert res.n_extrapolated == 0, (
        f"the .SPXTR pull stops at {res.last_observation}, leaving {res.n_extrapolated} "
        "forward-filled sessions in the sealed window — re-pull before reporting it")
