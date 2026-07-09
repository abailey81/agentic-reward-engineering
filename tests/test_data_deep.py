"""DEEP, advanced tests for the data layer (synthetic + pure-logic + PIT/leakage invariants).

These complement — and deliberately do NOT duplicate — the existing data tests
(``test_loaders.py``, ``test_loaders_checksum.py``, ``test_market_reference.py``,
``test_membership_shumway.py``, ``test_data_pipeline.py``, ``test_embargo_splits.py``).

Focus, per the dissertation's survivorship-free, point-in-time (PIT) licensed-equity premise:
look-ahead / leakage would invalidate everything, so the no-future-read and anonymisation
invariants are pinned here with property-based (Hypothesis, ``derandomize=True``), metamorphic,
adversarial, and boundary tests. Licensed-data reads are guarded behind availability checks; the
synthetic + pure-logic paths carry the weight.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.market_reference import (
    TRADING_DAYS_PER_YEAR,
    load_ff_factors,
    load_market_proxy_returns,
    load_risk_free_daily,
)
from src.data.panel import Panel
from src.data.synthetic import make_synthetic_panel

hyp = pytest.importorskip("hypothesis")
from hypothesis import given  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]


# ===========================================================================
# synthetic.py — determinism, shape/dtype, finiteness, stylised facts, bounds
# ===========================================================================
def test_synthetic_shape_dtype_and_components() -> None:
    """The generated panel has exactly the requested (n_days, n_assets) shape, float64 returns,
    aligned vix/dates/asset_ids, and PIT market caps of matching shape."""
    p = make_synthetic_panel(n_assets=11, n_days=137, seed=3)
    assert p.returns.shape == (137, 11)
    assert p.returns.dtype == np.float64
    assert p.vix.shape == (137,) and p.vix.dtype == np.float64
    assert p.dates.shape == (137,) and p.dates.dtype == np.dtype("datetime64[D]")
    assert p.asset_ids.shape == (11,) and p.asset_ids.dtype.kind in "iu"
    assert p.market_caps is not None and p.market_caps.shape == (137, 11)
    assert p.vix_prelagged is False  # synthetic convention: env lags it


def test_synthetic_is_bit_for_bit_deterministic_given_seed() -> None:
    """Two calls with the same seed are byte-identical across EVERY array (provenance rests on this)."""
    a = make_synthetic_panel(n_assets=7, n_days=200, seed=42)
    b = make_synthetic_panel(n_assets=7, n_days=200, seed=42)
    np.testing.assert_array_equal(a.returns, b.returns)
    np.testing.assert_array_equal(a.vix, b.vix)
    np.testing.assert_array_equal(a.dates, b.dates)
    np.testing.assert_array_equal(a.asset_ids, b.asset_ids)
    assert a.market_caps is not None and b.market_caps is not None
    np.testing.assert_array_equal(a.market_caps, b.market_caps)


def test_synthetic_different_seed_yields_different_returns() -> None:
    """Distinct seeds drive distinct draws (the RNG is actually seeded, not constant)."""
    a = make_synthetic_panel(n_assets=7, n_days=200, seed=1)
    b = make_synthetic_panel(n_assets=7, n_days=200, seed=2)
    assert not np.array_equal(a.returns, b.returns)


def test_synthetic_all_arrays_finite() -> None:
    """No NaN/inf anywhere — Panel would reject returns, and downstream consumers assume finiteness."""
    p = make_synthetic_panel(n_assets=12, n_days=400, seed=9)
    assert np.isfinite(p.returns).all()
    assert np.isfinite(p.vix).all()
    assert p.market_caps is not None and np.isfinite(p.market_caps).all()


def test_synthetic_market_caps_and_vix_strictly_positive() -> None:
    """Caps are exp(log-cap) so strictly > 0; the VIX-like index (annualised dispersion) is >= 0."""
    p = make_synthetic_panel(n_assets=10, n_days=300, seed=5)
    assert p.market_caps is not None and (p.market_caps > 0.0).all()
    assert (p.vix >= 0.0).all()


def test_synthetic_asset_ids_are_anonymised_range() -> None:
    """asset_ids carry NO semantic meaning — exactly 0..N-1 integers, never tickers/strings."""
    p = make_synthetic_panel(n_assets=9, n_days=120, seed=0)
    np.testing.assert_array_equal(p.asset_ids, np.arange(9))
    assert p.asset_ids.dtype.kind in "iu"


def test_synthetic_dates_are_strictly_increasing_daily() -> None:
    """The date axis is ascending and unique (splits/regimes/embargo logic relies on monotonicity)."""
    p = make_synthetic_panel(n_assets=6, n_days=250, seed=7)
    d = p.dates.astype("datetime64[D]")
    assert (np.diff(d).astype("timedelta64[D]").astype(int) > 0).all()
    assert len(np.unique(d)) == d.size


def test_synthetic_rejects_nonstationary_garch() -> None:
    """alpha + beta >= 1 is non-stationary variance — must raise, not silently emit exploding vol."""
    with pytest.raises(ValueError, match="stationary"):
        make_synthetic_panel(n_assets=4, n_days=50, garch_alpha=0.5, garch_beta=0.6)


@pytest.mark.parametrize("bad_df", [2.0, 1.5, 0.0, -3.0])
def test_synthetic_rejects_infinite_variance_df(bad_df: float) -> None:
    """df <= 2 makes the unit-variance rescale sqrt(df/(df-2)) divide-by-zero / sqrt-of-negative;
    the generator must reject it up front rather than emit a misleading later finiteness error."""
    with pytest.raises(ValueError, match="df must be > 2"):
        make_synthetic_panel(n_assets=4, n_days=50, df=bad_df)


def test_synthetic_heavier_tails_with_smaller_df() -> None:
    """METAMORPHIC stylised fact: smaller Student-t df => heavier tails => more extreme excess
    kurtosis in the cross-section of returns (Cont 2001, the dissertation's tail premise)."""
    light = make_synthetic_panel(n_assets=20, n_days=3000, seed=11, df=30.0)
    heavy = make_synthetic_panel(n_assets=20, n_days=3000, seed=11, df=3.0)

    def _excess_kurt(x: np.ndarray) -> float:
        z = (x - x.mean()) / x.std()
        return float((z**4).mean() - 3.0)

    assert _excess_kurt(heavy.returns.ravel()) > _excess_kurt(light.returns.ravel())


def test_synthetic_volatility_clustering_present() -> None:
    """Stylised fact: |returns| are positively autocorrelated (GARCH clustering), unlike raw returns
    which are ~uncorrelated. Lag-1 autocorrelation of |r| should be clearly positive."""
    p = make_synthetic_panel(n_assets=1, n_days=4000, seed=21, garch_alpha=0.1, garch_beta=0.88)
    r = p.returns[:, 0]
    abs_r = np.abs(r) - np.abs(r).mean()
    acf1 = float((abs_r[:-1] * abs_r[1:]).sum() / (abs_r**2).sum())
    assert acf1 > 0.05  # persistent volatility, not iid


@given(
    n_assets=st.integers(min_value=1, max_value=6),
    n_days=st.integers(min_value=3, max_value=40),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_synthetic_shape_invariant_property(n_assets: int, n_days: int, seed: int) -> None:
    """PROPERTY: for any small valid (n_assets, n_days, seed), the panel is well-formed and finite."""
    p = make_synthetic_panel(n_assets=n_assets, n_days=n_days, seed=seed)
    assert p.T == n_days and p.N == n_assets
    assert np.isfinite(p.returns).all()
    assert p.returns.shape == (n_days, n_assets)


# ===========================================================================
# panel.py — construction contract, shape validation, slice semantics,
#            anonymisation, frozen/copy semantics, vix_prelagged propagation
# ===========================================================================
def _toy_panel(t: int = 10, n: int = 3, *, caps: bool = True, prelagged: bool = False) -> Panel:
    rng = np.random.default_rng(0)
    returns = 0.01 * rng.standard_normal((t, n))
    return Panel(
        returns=returns,
        vix=np.linspace(10.0, 20.0, t),
        dates=np.arange("2010-01-01", t, dtype="datetime64[D]"),
        asset_ids=np.arange(n, dtype=np.int64),
        market_caps=(rng.uniform(1e9, 1e12, (t, n)) if caps else None),
        vix_prelagged=prelagged,
    )


def test_panel_rejects_non_2d_returns() -> None:
    with pytest.raises(ValueError, match="2-D"):
        Panel(
            returns=np.zeros(5),
            vix=np.zeros(5),
            dates=np.arange("2010-01-01", 5, dtype="datetime64[D]"),
            asset_ids=np.zeros(1, dtype=int),
        )


@pytest.mark.parametrize("field", ["vix", "dates", "asset_ids", "market_caps"])
def test_panel_rejects_misaligned_field(field: str) -> None:
    """Each constructor shape guard fires independently with a field-naming message."""
    t, n = 8, 4
    kwargs = dict(
        returns=np.zeros((t, n)),
        vix=np.zeros(t),
        dates=np.arange("2010-01-01", t, dtype="datetime64[D]"),
        asset_ids=np.arange(n, dtype=int),
        market_caps=np.zeros((t, n)),
    )
    if field == "vix":
        kwargs["vix"] = np.zeros(t + 1)
    elif field == "dates":
        kwargs["dates"] = np.arange("2010-01-01", t + 1, dtype="datetime64[D]")
    elif field == "asset_ids":
        kwargs["asset_ids"] = np.arange(n + 1, dtype=int)
    elif field == "market_caps":
        kwargs["market_caps"] = np.zeros((t, n + 1))
    with pytest.raises(ValueError, match=field):
        Panel(**kwargs)


def test_panel_rejects_non_finite_returns() -> None:
    """The finiteness contract (env rejects non-finite rows) is enforced at construction."""
    t, n = 6, 2
    r = np.zeros((t, n))
    r[3, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        Panel(
            returns=r,
            vix=np.zeros(t),
            dates=np.arange("2010-01-01", t, dtype="datetime64[D]"),
            asset_ids=np.arange(n, dtype=int),
        )


def test_panel_T_and_N_match_returns_shape() -> None:
    p = _toy_panel(t=15, n=5)
    assert (p.T, p.N) == (15, 5)
    assert p.returns.shape == (p.T, p.N)


def test_panel_is_frozen_immutable() -> None:
    """@dataclass(frozen=True): reassigning a field must raise — the panel is a frozen artifact."""
    p = _toy_panel()
    with pytest.raises(FrozenInstanceError):
        p.returns = np.zeros_like(p.returns)  # type: ignore[misc]


def test_panel_slice_semantics_half_open() -> None:
    """slice(start, end) returns the contiguous [start, end) rows on every time-indexed array,
    leaves asset_ids untouched, and preserves T==end-start."""
    p = _toy_panel(t=20, n=4)
    s = p.slice(5, 12)
    assert s.T == 7 and s.N == 4
    np.testing.assert_array_equal(s.returns, p.returns[5:12])
    np.testing.assert_array_equal(s.vix, p.vix[5:12])
    np.testing.assert_array_equal(s.dates, p.dates[5:12])
    np.testing.assert_array_equal(s.asset_ids, p.asset_ids)  # asset axis unchanged
    assert s.market_caps is not None and p.market_caps is not None
    np.testing.assert_array_equal(s.market_caps, p.market_caps[5:12])


def test_panel_slice_returns_new_panel_not_self() -> None:
    p = _toy_panel(t=10, n=2)
    s = p.slice(0, 10)
    assert s is not p and isinstance(s, Panel)


def test_panel_slice_propagates_prelag_flag() -> None:
    """A sliced GOLD panel must KEEP vix_prelagged=True — the re-audit regression where a slice
    reverted to False and the env then double-lagged an already-lagged vix to a t-2 close."""
    p = _toy_panel(t=12, n=3, prelagged=True)
    assert p.slice(2, 8).vix_prelagged is True
    p2 = _toy_panel(t=12, n=3, prelagged=False)
    assert p2.slice(2, 8).vix_prelagged is False


def test_panel_slice_rejects_windows_numpy_would_silently_reinterpret() -> None:
    """An illegal window must RAISE -- NumPy would otherwise silently REINTERPRET it.

    A negative ``start`` is an offset from the END, so ``slice(-5, 20)`` on a T=20 panel returned the
    LAST five rows -- FUTURE data -- instead of failing; inverted and past-end windows silently
    produced an EMPTY panel, which ``__post_init__`` accepts because T=0 is a valid shape. Both are
    look-ahead-shaped failures on the very object the no-look-ahead proof slices
    (tests/test_env_nolookahead.py), so an out-of-contract window has to fail loudly.
    """
    p = _toy_panel(t=20, n=3)
    for start, end in [(-5, 20), (10, 5), (0, 999), (-1, -1), (21, 21)]:
        with pytest.raises(ValueError, match="slice window"):
            p.slice(start, end)

    # Legal boundaries still work: full span, a deliberately EMPTY window, and the final row.
    assert p.slice(0, p.T).T == 20
    assert p.slice(7, 7).T == 0
    assert p.slice(19, 20).T == 1


def test_panel_slice_without_market_caps_stays_none() -> None:
    p = _toy_panel(t=10, n=2, caps=False)
    assert p.market_caps is None
    assert p.slice(1, 5).market_caps is None


def test_panel_carries_no_string_identities() -> None:
    """ANONYMISATION (N3 contamination defence): no ticker/RIC/string anywhere in the Panel — the
    only id channel is integer asset_ids, and dates are datetime64 (numeric), never reward inputs."""
    p = make_synthetic_panel(n_assets=8, n_days=60, seed=0)
    assert p.asset_ids.dtype.kind in "iu"
    assert p.returns.dtype.kind == "f" and p.vix.dtype.kind == "f"
    assert p.dates.dtype.kind == "M"  # datetime64, not object/string
    assert p.market_caps is not None and p.market_caps.dtype.kind == "f"


def test_panel_slice_is_metamorphic_composable() -> None:
    """METAMORPHIC: slicing [a,c) then [0, b-a) == slicing [a, b) directly (sub-slice composition)."""
    p = _toy_panel(t=30, n=3)
    a, b, c = 4, 10, 20
    direct = p.slice(a, b)
    composed = p.slice(a, c).slice(0, b - a)
    np.testing.assert_array_equal(direct.returns, composed.returns)
    np.testing.assert_array_equal(direct.dates, composed.dates)


# ===========================================================================
# market_reference.py — rf/market/FF alignment, no-look-ahead, metamorphic
# ===========================================================================
def _write_fred(tmp: Path, dates: pd.DatetimeIndex, yields) -> None:
    pd.DataFrame({"observation_date": dates.strftime("%Y-%m-%d"), "DGS3MO": yields}).to_csv(
        tmp / "fred_macro.csv", index=False
    )


def test_rf_value_at_t_uses_data_known_at_or_before_t() -> None:
    """PIT / NO-LOOK-AHEAD: the rf value aligned to date t reflects ONLY observations <= t. We publish
    a yield that JUMPS on the last day; earlier sessions must still read the old (last-known) yield, never
    the future jump."""
    tmp = Path(pytest.importorskip("tempfile").mkdtemp())
    dates = pd.bdate_range("2021-03-01", periods=8)
    yields = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 9.0]  # jump only on the final session
    _write_fred(tmp, dates, yields)
    rf = load_risk_free_daily(dates.to_numpy(), raw_dir=tmp)
    d_old = (1.0 + 2.0 / 100.0) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    d_new = (1.0 + 9.0 / 100.0) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    assert rf.daily[:-1] == pytest.approx([d_old] * 7)  # no future leak into earlier sessions
    assert rf.daily[-1] == pytest.approx(d_new)


def test_rf_leading_gap_falls_back_to_zero() -> None:
    """A target date BEFORE the first published observation has no past value -> 0.0 (never bfill the
    future). Here the panel starts a week before the first FRED print."""
    tmp = Path(pytest.importorskip("tempfile").mkdtemp())
    fred_dates = pd.bdate_range("2021-03-08", periods=5)
    _write_fred(tmp, fred_dates, [3.0] * 5)
    target = pd.bdate_range("2021-03-01", periods=10).to_numpy()  # first 5 precede any FRED obs
    rf = load_risk_free_daily(target, raw_dir=tmp)
    assert rf.available is True
    assert rf.daily[0] == 0.0  # leading gap -> 0, not a backward-pulled future yield


def test_rf_metamorphic_index_shift_shifts_output() -> None:
    """METAMORPHIC: shifting EVERY target date forward by one published step shifts the aligned rf
    series consistently (alignment is a pure function of the request axis, no hidden global state)."""
    tmp = Path(pytest.importorskip("tempfile").mkdtemp())
    fred_dates = pd.bdate_range("2021-01-04", periods=20)
    yields = list(np.linspace(1.0, 3.0, 20))
    _write_fred(tmp, fred_dates, yields)
    a = load_risk_free_daily(fred_dates[2:8].to_numpy(), raw_dir=tmp)
    b = load_risk_free_daily(fred_dates[3:9].to_numpy(), raw_dir=tmp)
    # b is a's window advanced by one session: a[1:] should equal b[:-1] (same underlying yields).
    assert a.daily[1:] == pytest.approx(b.daily[:-1])


def test_rf_constant_yield_is_scale_correct() -> None:
    """A flat annual yield maps to a constant per-session geometric decimal everywhere it is known."""
    tmp = Path(pytest.importorskip("tempfile").mkdtemp())
    dates = pd.bdate_range("2021-01-04", periods=12)
    _write_fred(tmp, dates, [4.0] * 12)
    rf = load_risk_free_daily(dates.to_numpy(), raw_dir=tmp)
    expected = (1.0 + 4.0 / 100.0) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    assert np.allclose(rf.daily, expected, atol=1e-15)
    assert rf.annual_pct_mean == pytest.approx(4.0, abs=1e-9)


def test_rf_missing_source_column_degrades() -> None:
    """If the FRED file exists but lacks the requested source column, degrade to zero/unavailable."""
    tmp = Path(pytest.importorskip("tempfile").mkdtemp())
    pd.DataFrame({"observation_date": ["2021-01-04"], "SOMETHING_ELSE": [1.0]}).to_csv(
        tmp / "fred_macro.csv", index=False
    )
    dates = pd.bdate_range("2021-01-04", periods=4).to_numpy()
    rf = load_risk_free_daily(dates, raw_dir=tmp)
    assert rf.available is False and np.all(rf.daily == 0.0)
    assert rf.daily.shape == (4,)


def test_market_proxy_forward_fills_interior_gap_not_future() -> None:
    """A bond-holiday interior gap reads the LAST KNOWN market return (ffill), never a future one.
    We omit a session from the parquet but request it: it must inherit the prior value."""
    from src.data.loaders import gold_suffix

    tmp = Path(pytest.importorskip("tempfile").mkdtemp())
    full = pd.bdate_range("2021-02-01", periods=6)
    kept = full.delete(3)  # drop the 4th session from the stored parquet
    pd.DataFrame({"market_ew": [0.01, -0.02, 0.03, -0.04, 0.05]}, index=kept).to_parquet(
        tmp / f"market_proxy_{gold_suffix()}.parquet"
    )
    mp = load_market_proxy_returns(full.to_numpy(), gold_dir=tmp)
    assert mp.available is True
    # index 3 was missing -> carries index-2's stored value (0.03), the last KNOWN return.
    assert mp.returns[3] == pytest.approx(0.03)
    assert mp.returns[2] == pytest.approx(0.03)


def test_market_proxy_unnamed_index_column0_heuristic() -> None:
    """The parquet stores dates in an UNNAMED DatetimeIndex; the loader must read dates from the
    index (not misread column 0 as values, the documented _aligned_series pitfall it sidesteps)."""
    tmp = Path(pytest.importorskip("tempfile").mkdtemp())
    idx = pd.bdate_range("2021-02-01", periods=4)
    from src.data.loaders import gold_suffix

    pd.DataFrame({"market_ew": [0.1, 0.2, 0.3, 0.4]}, index=idx).to_parquet(
        tmp / f"market_proxy_{gold_suffix()}.parquet"
    )
    mp = load_market_proxy_returns(idx.to_numpy(), gold_dir=tmp)
    assert mp.returns == pytest.approx([0.1, 0.2, 0.3, 0.4])


def test_market_proxy_honours_gold_suffix_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """suffix=None resolves to gold_suffix(): under LLM_RP_GOLD_SUFFIX the proxy tracks the SAME
    universe as the traded panel. An override to a suffix with no parquet degrades (available=False)."""
    tmp = Path(pytest.importorskip("tempfile").mkdtemp())
    idx = pd.bdate_range("2021-02-01", periods=3)
    pd.DataFrame({"market_ew": [0.01, 0.02, 0.03]}, index=idx).to_parquet(
        tmp / "market_proxy_univ4.parquet"  # only the univ4 file exists
    )
    monkeypatch.setenv("LLM_RP_GOLD_SUFFIX", "univ4")
    mp = load_market_proxy_returns(idx.to_numpy(), gold_dir=tmp)  # suffix=None -> univ4
    assert mp.available is True and mp.returns == pytest.approx([0.01, 0.02, 0.03])
    monkeypatch.setenv("LLM_RP_GOLD_SUFFIX", "univ3")  # univ3 parquet absent -> degrade
    mp3 = load_market_proxy_returns(idx.to_numpy(), gold_dir=tmp)
    assert mp3.available is False and np.all(mp3.returns == 0.0)


def test_ff_factors_alignment_and_shape(tmp_path: Path) -> None:
    """FF loader returns the three factor columns as float arrays aligned 1:1 to the request axis,
    forward-filled with no future read; absent columns are simply omitted."""
    idx = pd.bdate_range("2021-01-04", periods=6)
    pd.DataFrame(
        {
            "Date": idx.strftime("%Y-%m-%d"),
            "Mkt-RF": [0.001, 0.002, np.nan, 0.004, 0.005, 0.006],
            "SMB": [0.0001] * 6,
            # HML deliberately ABSENT
        }
    ).to_csv(tmp_path / "french_F-F_Research_Data_Factors_daily.csv", index=False)
    ff = load_ff_factors(idx.to_numpy(), raw_dir=tmp_path)
    assert ff.available is True
    assert set(ff.factors) == {"Mkt-RF", "SMB"}  # HML omitted, not faked
    for col, arr in ff.factors.items():
        assert arr.shape == (6,) and arr.dtype.kind == "f" and np.isfinite(arr).all()
    # the day-2 gap forward-fills day-1's 0.002 (last known), not the future 0.004.
    assert ff.factors["Mkt-RF"][2] == pytest.approx(0.002)


# ===========================================================================
# loaders.py — gold_suffix env contract + clear-error on a missing panel
# ===========================================================================
def test_gold_suffix_unset_defaults_to_univ3(monkeypatch: pytest.MonkeyPatch) -> None:
    """No LLM_RP_GOLD_SUFFIX => the ACTIVE headline panel univ5 (SPLIT C, ADR-044/051; the campaign
    runs with the env var UNSET, so config/data.yaml's gold.suffix governs)."""
    from src.data.loaders import gold_suffix

    monkeypatch.delenv("LLM_RP_GOLD_SUFFIX", raising=False)
    assert gold_suffix() == "univ5"


def test_gold_suffix_strips_leading_underscore_and_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    """The selector tolerates a leading '_' and surrounding whitespace; blank falls back to default."""
    from src.data.loaders import gold_suffix

    monkeypatch.setenv("LLM_RP_GOLD_SUFFIX", "_univ4")
    assert gold_suffix() == "univ4"
    monkeypatch.setenv("LLM_RP_GOLD_SUFFIX", "  univ4  ")
    assert gold_suffix() == "univ4"
    monkeypatch.setenv("LLM_RP_GOLD_SUFFIX", "__univ3")
    assert gold_suffix() == "univ3"  # only ONE leading underscore is stripped... but '_univ3'->'univ3'
    monkeypatch.setenv("LLM_RP_GOLD_SUFFIX", "   ")
    assert gold_suffix() == "univ5"  # blank override falls back to the ACTIVE config suffix (Split C)


def test_gold_suffix_read_live_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """gold_suffix() is evaluated PER CALL so a sensitivity-band sweep can switch panels via the env
    without a code edit / re-import (it must not be cached at import)."""
    from src.data.loaders import gold_suffix

    monkeypatch.setenv("LLM_RP_GOLD_SUFFIX", "univ4")
    assert gold_suffix() == "univ4"
    monkeypatch.delenv("LLM_RP_GOLD_SUFFIX", raising=False)
    assert gold_suffix() == "univ5"  # the very next call reflects the change (ACTIVE = univ5, Split C)


def test_load_gold_panel_missing_artifact_raises_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pointed at an EMPTY gold dir, the loader fails loudly with a FileNotFoundError that names the
    missing artifact AND the active suffix — never a silent empty panel (a synthetic-only install)."""
    from src.data.loaders import load_gold_panel

    monkeypatch.setenv("LLM_RP_GOLD_SUFFIX", "univ3")
    with pytest.raises(FileNotFoundError, match=r"returns_panel_univ3\.parquet"):
        load_gold_panel("development", gold_dir=tmp_path)


def test_read_missing_artifact_names_active_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The not-found error surfaces the active suffix so a mis-switch is diagnosable, not silent."""
    from src.data import loaders

    monkeypatch.setenv("LLM_RP_GOLD_SUFFIX", "univ7")
    with pytest.raises(FileNotFoundError, match=r"(univ7|LLM_RP_GOLD_SUFFIX)"):
        loaders._read("cash_features", tmp_path)


def test_seed_leading_vix_never_pulls_future_in_window() -> None:
    """LEAKAGE-CRITICAL (covers the no-future-read rule on a fresh, independent case): a leading
    in-window VIX NaN is seeded from the genuine PRE-start session, NOT bfill'd from a later in-window
    close. Built without gold data (no real window has a leading NaN + genuine prior data)."""
    from src.data.loaders import _seed_leading_vix

    idx = pd.to_datetime(["2019-06-03", "2019-06-04", "2019-06-05", "2019-06-06", "2019-06-07"])
    cash = pd.DataFrame({"vix": [0.20, 0.21, 0.22, np.nan, 0.30]}, index=idx)
    start = pd.Timestamp("2019-06-06")  # in-window leading NaN at start; 06-05 (=0.22) precedes it
    out = _seed_leading_vix(cash.loc[start:, "vix"].copy(), cash, start)
    assert out.iloc[0] == pytest.approx(0.22)  # the genuine t-1 close, not the future in-window 0.30
    assert not out.isna().any()


# ===========================================================================
# loaders.py against the REAL frozen gold panel — guarded (licensed data)
# ===========================================================================
# Gate on the ACTIVE panel (batch-6 M7, 2026-07-03): was hardcoded to univ3 while the live suffix is
# univ5 (ADR-051) — if univ3 were pruned these tests would wrongly SKIP though the active panel is present.
# Track gold_suffix() like the sibling gold-gated test files (test_loaders/test_embargo_splits/test_viz_eda).
from src.data.loaders import gold_suffix as _gold_suffix  # noqa: E402

_GOLD = _REPO / "data" / "gold" / f"returns_panel_{_gold_suffix()}.parquet"
gold_only = pytest.mark.skipif(not _GOLD.exists(), reason="frozen gold panel not present (licensed data)")


@gold_only
def test_gold_panel_dates_strictly_ascending_and_prelagged() -> None:
    """The loaded gold panel's session axis is strictly ascending (splits/embargo rely on it) and the
    panel is vix_prelagged=True (the gold cash_features.vix is already shift(1)-lagged at build)."""
    from src.data.loaders import load_gold_panel

    p = load_gold_panel("development").panel
    d = np.asarray(p.dates).astype("datetime64[ns]")
    assert (np.diff(d).astype("int64") > 0).all()
    assert p.vix_prelagged is True


@gold_only
def test_gold_panel_slice_preserves_finiteness_and_prelag() -> None:
    """A time-slice of the real gold panel stays finite and KEEPS the prelag flag (no double-lag)."""
    from src.data.loaders import load_gold_panel

    p = load_gold_panel("development").panel
    s = p.slice(0, min(50, p.T))
    assert np.isfinite(s.returns).all() and np.isfinite(s.vix).all()
    assert s.vix_prelagged is True
