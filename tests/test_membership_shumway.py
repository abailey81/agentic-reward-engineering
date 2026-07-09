"""Direct unit + tail-preservation tests for the Shumway-STYLE delisting correction
(``data_pipeline/src/data/membership.py::apply_shumway_corrections``) and its wiring.

These are FAST and need NO licensed data — every frame is synthetic. They pin the two
behaviours the M3/M4 adversarial review demands of the ``_univ4`` panel (ADR-024):

  1. the correction BOOKS the crash return onto the dead name's LAST VALID session,
     COMPOUNDED MULTIPLICATIVELY ``(1+r)(1+dl)-1`` (never additive — additive can breach
     −100%, OpenSourceAP #49; never onto the all-NaN nominal-delist row, which KeyErrored
     off-grid / planted a phantom row the env's ``liquidate_to_cash`` would then zero-fill);
  2. that correction PRESERVES the left tail — CVaR_05 of the corrected panel is strictly
     MORE NEGATIVE than the ``liquidate_to_cash`` (zero-fill) panel reads.

Import isolation: ``data_pipeline/src`` and the live A-line ``src`` are BOTH named ``src``.
To call the B-line ``apply_shumway_corrections`` without shadowing the A-line ``src`` that
the rest of the suite imports, membership.py is loaded under a PRIVATE ``dp_src.*``
namespace (no top-level ``src`` is registered), so test order (pytest-randomly) is safe.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.inference.bootstrap import cvar  # A-line canonical CVaR (reused, not re-derived)

_DP = Path(__file__).resolve().parents[1] / "data_pipeline"


def _load_dp_membership():
    """Load ``data_pipeline/src/data/membership.py`` under a private ``dp_src.*`` package
    so its ``from ..config import get`` resolves WITHOUT registering a top-level ``src``
    (which would shadow the A-line ``src`` the rest of the suite imports)."""
    if "dp_src.data.membership" in sys.modules:
        return sys.modules["dp_src.data.membership"]
    src_dir = _DP / "src"
    if "dp_src" not in sys.modules:
        pkg = types.ModuleType("dp_src")
        pkg.__path__ = [str(src_dir)]
        sys.modules["dp_src"] = pkg
        data_pkg = types.ModuleType("dp_src.data")
        data_pkg.__path__ = [str(src_dir / "data")]
        sys.modules["dp_src.data"] = data_pkg

    def _load(modname: str, filepath: Path):
        spec = importlib.util.spec_from_file_location(modname, filepath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[modname] = mod
        spec.loader.exec_module(mod)
        return mod

    _load("dp_src.config", src_dir / "config.py")
    return _load("dp_src.data.membership", src_dir / "data" / "membership.py")


_membership = _load_dp_membership()
apply_shumway_corrections = _membership.apply_shumway_corrections


# Config-pinned surcharges (data_pipeline/config/data.yaml: series.delisting_corrections).
NYSE_AMEX_DL = -0.30
NASDAQ_DL = -0.55


def _panel_with_dead_name() -> pd.DataFrame:
    """A 6-session, 2-name frame. ``DEAD.N`` trades for 3 sessions then NaNs out (a dead
    ^RIC's last traded session is session 2); ``ALIVE.O`` trades throughout. The nominal
    delist date is session 4 — POST the last valid DEAD.N row (session 2)."""
    idx = pd.bdate_range("2009-01-05", periods=6)
    df = pd.DataFrame(
        {
            "DEAD.N": [0.01, -0.02, 0.005, np.nan, np.nan, np.nan],
            "ALIVE.O": [0.003, 0.001, -0.004, 0.002, 0.0, -0.001],
        },
        index=idx,
    )
    return df, idx


# ----------------------------------------------------------------- landmine / unit tests
def test_correction_lands_on_last_valid_row_not_nan_delist_row() -> None:
    """The −30% books onto DEAD.N's LAST VALID session (session 2), NOT the all-NaN
    nominal-delist row (session 4) — and never KeyErrors. (THE LANDMINE FIX.)"""
    df, idx = _panel_with_dead_name()
    delisted = {"DEAD.N": {"date": idx[4], "exchange": "nyse_amex",
                           "vendor_terminal_return": None}}
    out, log = apply_shumway_corrections(df, delisted)

    last_valid = idx[2]
    prior = df.at[last_valid, "DEAD.N"]            # 0.005, the real terminal return
    expected = (1.0 + prior) * (1.0 + NYSE_AMEX_DL) - 1.0
    assert out.at[last_valid, "DEAD.N"] == pytest.approx(expected)
    # the all-NaN nominal-delist row is untouched (still NaN — no phantom value planted)
    assert pd.isna(out.at[idx[4], "DEAD.N"])
    # the alive name and the dead name's earlier rows are untouched (no mutation)
    pd.testing.assert_series_equal(out["ALIVE.O"], df["ALIVE.O"])
    assert out.at[idx[0], "DEAD.N"] == pytest.approx(0.01)
    # input frame is NEVER mutated
    assert pd.isna(df.at[idx[4], "DEAD.N"]) and df.at[last_valid, "DEAD.N"] == pytest.approx(prior)
    # audit row records the booked-on label + the multiplicative components
    row = log[log["ric"] == "DEAD.N"].iloc[0]
    assert row["action"] == "shumway_correction_applied"
    assert pd.Timestamp(row["booked_on"]) == last_valid
    assert row["delisting_return"] == pytest.approx(NYSE_AMEX_DL)
    assert row["value"] == pytest.approx(expected)


def test_multiplicative_compounding_cannot_breach_minus_one() -> None:
    """Multiplicative ``(1+r)(1+dl)-1`` stays > −1 even when the terminal day was already
    very negative; the (wrong) additive ``r + dl`` would breach −100%."""
    idx = pd.bdate_range("2009-01-05", periods=4)
    df = pd.DataFrame({"X.O": [0.0, -0.80, np.nan, np.nan]}, index=idx)   # −80% last day
    delisted = {"X.O": {"date": idx[2], "exchange": "nasdaq", "vendor_terminal_return": None}}
    out, _ = apply_shumway_corrections(df, delisted)
    booked = out.at[idx[1], "X.O"]
    assert booked == pytest.approx((1.0 - 0.80) * (1.0 + NASDAQ_DL) - 1.0)  # = -0.91
    assert booked > -1.0                                   # multiplicative never breaches
    assert booked != pytest.approx(-0.80 + NASDAQ_DL)      # NOT additive (= -1.35)


def test_surcharge_applies_by_exchange() -> None:
    """−30% for nyse_amex, −55% for nasdaq, keyed off the per-name exchange bucket."""
    idx = pd.bdate_range("2009-01-05", periods=3)
    df = pd.DataFrame({"N.N": [0.0, 0.0, np.nan], "Q.O": [0.0, 0.0, np.nan]}, index=idx)
    delisted = {
        "N.N": {"date": idx[2], "exchange": "nyse_amex", "vendor_terminal_return": None},
        "Q.O": {"date": idx[2], "exchange": "nasdaq", "vendor_terminal_return": None},
    }
    out, log = apply_shumway_corrections(df, delisted)
    # prior terminal return is 0.0 for both, so the booked value == the surcharge
    assert out.at[idx[1], "N.N"] == pytest.approx(NYSE_AMEX_DL)
    assert out.at[idx[1], "Q.O"] == pytest.approx(NASDAQ_DL)
    by_ric = log.set_index("ric")["delisting_return"]
    assert by_ric["N.N"] == pytest.approx(NYSE_AMEX_DL)
    assert by_ric["Q.O"] == pytest.approx(NASDAQ_DL)


def test_vendor_terminal_is_preferred_when_present() -> None:
    """When the vendor terminal return is present it is KEPT (logged ``vendor_terminal_kept``)
    and the frame is NOT overwritten by the fixed surcharge."""
    df, idx = _panel_with_dead_name()
    before = df["DEAD.N"].copy()
    delisted = {"DEAD.N": {"date": idx[4], "exchange": "nyse_amex",
                           "vendor_terminal_return": -0.42}}
    out, log = apply_shumway_corrections(df, delisted)
    pd.testing.assert_series_equal(out["DEAD.N"], before)   # frame untouched — vendor kept
    row = log[log["ric"] == "DEAD.N"].iloc[0]
    assert row["action"] == "vendor_terminal_kept"
    assert row["value"] == pytest.approx(-0.42)
    # no surcharge row was emitted for this name
    assert (log["action"] == "shumway_correction_applied").sum() == 0


def test_offgrid_delist_date_does_not_keyerror() -> None:
    """A delist date that is NOT a session in the index (weekend / beyond last obs) books
    onto the last valid row instead of raising the old KeyError. (THE LANDMINE.)"""
    df, idx = _panel_with_dead_name()
    off_grid = idx[2] + pd.Timedelta(days=3)                # Sat 2009-01-10 — not a session
    assert off_grid.weekday() >= 5 and off_grid not in df.index
    delisted = {"DEAD.N": {"date": off_grid, "exchange": "nyse_amex",
                           "vendor_terminal_return": None}}
    out, log = apply_shumway_corrections(df, delisted)       # must not raise
    expected = (1.0 + 0.005) * (1.0 + NYSE_AMEX_DL) - 1.0
    assert out.at[idx[2], "DEAD.N"] == pytest.approx(expected)
    assert log.iloc[0]["action"] == "shumway_correction_applied"


def test_all_nan_column_is_skipped_not_raised() -> None:
    """A name with NO valid observation in the window is logged as a skip (no row to
    compound into) rather than raising or fabricating a phantom value (R4)."""
    idx = pd.bdate_range("2009-01-05", periods=3)
    df = pd.DataFrame({"GHOST.N": [np.nan, np.nan, np.nan], "OK.O": [0.0, 0.0, 0.0]}, index=idx)
    delisted = {"GHOST.N": {"date": idx[1], "exchange": "nyse_amex",
                            "vendor_terminal_return": None}}
    out, log = apply_shumway_corrections(df, delisted)
    assert out["GHOST.N"].isna().all()                      # nothing planted
    assert log.iloc[0]["action"] == "shumway_skipped_no_obs"


def test_unconfigured_exchange_raises_keyerror() -> None:
    """An exchange with no configured surcharge is a misconfiguration — raise loudly
    (never silently skip the integrity correction)."""
    idx = pd.bdate_range("2009-01-05", periods=3)
    df = pd.DataFrame({"Z.L": [0.0, 0.01, np.nan]}, index=idx)
    delisted = {"Z.L": {"date": idx[2], "exchange": "london", "vendor_terminal_return": None}}
    with pytest.raises(KeyError, match="no delisting correction configured"):
        apply_shumway_corrections(df, delisted)


def test_input_frame_never_mutated() -> None:
    """The corrector returns a COPY; the caller's frame is byte-identical afterwards."""
    df, idx = _panel_with_dead_name()
    snapshot = df.copy()
    delisted = {"DEAD.N": {"date": idx[4], "exchange": "nyse_amex",
                           "vendor_terminal_return": None}}
    apply_shumway_corrections(df, delisted)
    pd.testing.assert_frame_equal(df, snapshot)


# --------------------------------------------------------- tail-preservation regression
def _liquidate_to_cash(returns: pd.DataFrame) -> pd.DataFrame:
    """The env's default delisting policy (src/data/loaders.load_gold_panel,
    on_missing='liquidate_to_cash'): post-event / missing returns become 0.0."""
    return returns.fillna(0.0)


def test_correction_makes_cvar_strictly_more_negative_than_liquidate_to_cash() -> None:
    """TAIL-PRESERVATION (the headline invariant): on a synthetic panel of dead names whose
    terminal day is the crash, the Shumway-corrected equal-weight portfolio's CVaR_05 is
    strictly MORE NEGATIVE than the liquidate_to_cash version — the correction puts the
    crash return BACK INTO the left tail that the H2 measures read, instead of zero-filling
    it (the M3/M4 invalidation of any univ3 headline tail number)."""
    rng = np.random.default_rng(12345)
    n_days, n_names = 260, 40
    idx = pd.bdate_range("2008-06-02", periods=n_days)
    base = 0.0004 + 0.01 * rng.standard_normal((n_days, n_names))
    df = pd.DataFrame(base, index=idx, columns=[f"D{i:02d}.N" for i in range(n_names)])

    # Half the names die mid-window: their terminal traded day is a deep crash, then NaN.
    delisted: dict[str, dict] = {}
    for i in range(0, n_names, 2):
        name = df.columns[i]
        death_pos = int(rng.integers(80, n_days - 20))
        df.iloc[death_pos, i] = -0.10                       # crash on the last traded day
        df.iloc[death_pos + 1:, i] = np.nan                 # dead thereafter
        delisted[name] = {"date": idx[min(death_pos + 5, n_days - 1)],
                          "exchange": "nyse_amex", "vendor_terminal_return": None}

    corrected, log = apply_shumway_corrections(df, delisted)
    assert (log["action"] == "shumway_correction_applied").all()
    assert len(log) == len(delisted)                        # every dead name was corrected

    # Equal-weight portfolio return per session under each policy.
    liq_port = _liquidate_to_cash(df).mean(axis=1).to_numpy()
    cor_port = _liquidate_to_cash(corrected).mean(axis=1).to_numpy()  # fill the OTHER NaNs

    cvar_liq = cvar(liq_port, 0.05)
    cvar_cor = cvar(cor_port, 0.05)
    assert cvar_cor < cvar_liq                              # corrected tail is heavier
    # and the gap is economically meaningful (the −30% surcharges, not float noise)
    assert cvar_liq - cvar_cor > 1e-4


# --------------------------------------------------------------- loaders suffix switch
def test_gold_suffix_defaults_to_univ3_and_switches(monkeypatch) -> None:
    """The live default is ``univ5`` (the ACTIVE Split-C panel, ADR-044/051; config gold.suffix
    governs); ``LLM_RP_GOLD_SUFFIX`` switches it (leading ``_`` stripped) WITHOUT a code edit."""
    from src.data import loaders

    monkeypatch.delenv("LLM_RP_GOLD_SUFFIX", raising=False)
    assert loaders.gold_suffix() == "univ5"
    monkeypatch.setenv("LLM_RP_GOLD_SUFFIX", "univ4")
    assert loaders.gold_suffix() == "univ4"
    monkeypatch.setenv("LLM_RP_GOLD_SUFFIX", "_univ4")     # tolerant of a leading underscore
    assert loaders.gold_suffix() == "univ4"
    monkeypatch.setenv("LLM_RP_GOLD_SUFFIX", "  ")          # blank -> default
    assert loaders.gold_suffix() == "univ5"


def test_read_path_follows_active_suffix(monkeypatch, tmp_path) -> None:
    """``_read`` resolves ``<name>_<suffix>.parquet`` from the ACTIVE suffix, and reports
    that suffix in the not-found error so a mis-switch fails loudly (not silently)."""
    from src.data import loaders

    monkeypatch.setenv("LLM_RP_GOLD_SUFFIX", "univ4")
    with pytest.raises(FileNotFoundError, match=r"returns_panel_univ4\.parquet"):
        loaders._read("returns_panel", tmp_path)
