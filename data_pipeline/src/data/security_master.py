"""Security master & symbology (stage 3): RIC <-> ticker mapping WITH HISTORY.

Every downstream join keys on `security_id` from this master — never on raw vendor
tickers. Rationale: tickers are reused (e.g. post-bankruptcy reissues), renamed
(FB->META 2022-06), split into share classes (Google 2014-04: GOOG kept by the new
class C, class A became GOOGL), and dead RICs carry the '^' suffix convention
(LEH.N^I08 = base RIC LEH.N, delisted September 2008; month letters A..L = Jan..Dec).

The master is BUILT from vendor symbology pulls plus a small curated override table
(documented anomalies the vendors themselves disagree on). Overrides are repo code —
reviewed, cited, versioned — not silent data edits (R4: raw stays untouched; the
master is a derived, staged artifact)."""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

_MONTH_CODES = "ABCDEFGHIJKL"          # RIC delisting suffix: A=Jan .. L=Dec
_DEAD_RE = re.compile(r"^(?P<base>.+)\^(?P<m>[A-L])(?P<yy>\d{2})$")

#: Curated symbology anomalies (extend as the universe build surfaces more; each row
#: cites its public corporate-action source in `note`).
CURATED_OVERRIDES: list[dict] = [
    {"ric": "GOOGL.O", "ticker": "GOOGL", "note": "Class A; pre-2014-04-03 history lives under GOOG (share-class split)"},
    {"ric": "GOOG.O", "ticker": "GOOG", "note": "Class C created 2014-04-03; do not concatenate naively with GOOGL"},
    {"ric": "META.O", "ticker": "META", "note": "Renamed from FB 2022-06-09; FB.O history is the same security"},
]


@dataclass(frozen=True)
class DeadRic:
    base: str
    year: int
    month: int


def parse_dead_ric(ric: str) -> DeadRic | None:
    """LEH.N^I08 -> (LEH.N, 2008, 9); None when the RIC carries no death suffix.
    Two-digit years pivot at 70 (>=70 -> 19xx) — adequate for a 2005-2025 panel."""
    m = _DEAD_RE.match(ric)
    if not m:
        return None
    yy = int(m.group("yy"))
    year = 1900 + yy if yy >= 70 else 2000 + yy
    return DeadRic(m.group("base"), year, _MONTH_CODES.index(m.group("m")) + 1)


def ric_to_yf_ticker(ric: str) -> str:
    """Best-effort RIC -> yfinance symbol: strip death suffix and exchange code,
    map share-class dots (BRK.B style arrives as BRKb in RICs -> BRK-B)."""
    dead = parse_dead_ric(ric)
    base = dead.base if dead else ric
    base = base.split(".")[0]
    base = re.sub(r"([a-z])$", lambda m: "-" + m.group(1).upper(), base)
    return base.upper() if "-" not in base else base


def build_security_master(symbology_frames: list[pd.DataFrame] | None = None) -> pd.DataFrame:
    """Assemble the master from vendor symbology pulls (columns at minimum
    ric/ticker/name; optional isin/exchange/first_date/last_date) + curated overrides.

    Returns one row per (security_id, ric) with: security_id (stable: base RIC),
    ric, ticker, yf_ticker, isin, name, exchange, dead (bool), death_year/month,
    valid_from, valid_to, note. Joins downstream go through `resolve`."""
    rows: list[dict] = []
    for frame in symbology_frames or []:
        for _, r in frame.iterrows():
            ric = str(r.get("ric", "")).strip()
            if not ric:
                continue
            dead = parse_dead_ric(ric)
            rows.append({
                "ric": ric,
                "ticker": str(r.get("ticker", "")) or ric_to_yf_ticker(ric),
                "isin": r.get("isin"),
                "name": r.get("name"),
                "exchange": r.get("exchange"),
                "valid_from": r.get("first_date"),
                "valid_to": r.get("last_date"),
                "dead": dead is not None,
                "death_year": dead.year if dead else None,
                "death_month": dead.month if dead else None,
                "note": r.get("note", ""),
            })
    for o in CURATED_OVERRIDES:
        rows.append({"ric": o["ric"], "ticker": o["ticker"], "isin": None, "name": None,
                     "exchange": None, "valid_from": None, "valid_to": None,
                     "dead": False, "death_year": None, "death_month": None,
                     "note": o["note"]})
    master = pd.DataFrame(rows).drop_duplicates(subset=["ric"], keep="last")
    if master.empty:
        return master.assign(security_id=pd.Series(dtype=str), yf_ticker=pd.Series(dtype=str))
    master["security_id"] = master["ric"].map(lambda x: (parse_dead_ric(x) or DeadRic(x, 0, 0)).base)
    master["yf_ticker"] = master["ric"].map(ric_to_yf_ticker)
    return master.reset_index(drop=True)


def resolve(master: pd.DataFrame, ric_or_ticker: str) -> str:
    """RIC or ticker -> security_id; raises (never guesses) on unknown symbols."""
    hit = master[(master["ric"] == ric_or_ticker) | (master["ticker"] == ric_or_ticker)
                 | (master["yf_ticker"] == ric_or_ticker)]
    if hit.empty:
        raise KeyError(f"'{ric_or_ticker}' not in security master — extend symbology pulls "
                       "or CURATED_OVERRIDES; downstream joins must not bypass the master")
    ids = hit["security_id"].unique()
    if len(ids) > 1:
        raise ValueError(f"'{ric_or_ticker}' is ambiguous across securities {sorted(ids)} — "
                         "qualify with the full RIC")
    return str(ids[0])
