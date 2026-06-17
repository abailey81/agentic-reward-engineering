"""Universe acquisition orchestrator (A1-A5) — `make pull-universe` (ADR-019).

RUNS ONLY ONCE ENTITLEMENT LANDS. `acquire_universe()` opens a Refinitiv/DSWS session
(raising EntitlementError, captured into the run sidecar, if scopes are still empty) and
drives the EXISTING journaled engine (`acquire.journaled_pull`: rate-governed, chunked,
resumable, provenance-manifested) over the A1-A5 bill-of-materials in `config/data.yaml:
universe_pull`. No new acquisition machinery — this is wiring + vendor-shape PARSERS.

Design split (so tests are network-free):
  * PURE PARSERS (parse_*): vendor-shaped frame -> the tidy format the pipeline already
    consumes (membership -> normalize_membership rows; delisting -> apply_shumway_corrections
    dict; sector/cap -> tidy panels). Unit-tested on synthetic fixtures.
  * ORCHESTRATOR (acquire_universe): opens the session, calls journaled_pull with the real
    fetchers. Never invoked by tests; only by `make pull-universe` with live entitlement.

IDENTIFICATION UNTOUCHED (ADR-019): A1/A2/A3 feed the pre-registered panel + top-30
selection (PREREG §5); A4/A5 serve cost calibration, capacity defence, integrity, and EDA
attribution ONLY — none enters the state vector or the reward search (R1/R3 intact).

Refinitiv mnemonics marked VERIFY in config must be confirmed by the entitled probe before
the first live pull; the parsers are written against the DOCUMENTED column shapes and are
schema-tolerant (they locate columns by role, not by fixed position).
"""
from __future__ import annotations

import pandas as pd

from ..config import get


# --------------------------------------------------------------------------- parsers
def _first_col(df: pd.DataFrame, candidates: list[str], contains: list[str]) -> str | None:
    """Locate a column by exact name (candidates) then by case-insensitive substring
    (contains) — tolerant to the exact header the entitled feed returns."""
    for c in candidates:
        if c in df.columns:
            return c
    low = {str(c).lower(): c for c in df.columns}
    for token in contains:
        for lc, orig in low.items():
            if token in lc:
                return orig
    return None


def parse_refinitiv_constituents(df: pd.DataFrame, month_end: str | pd.Timestamp) -> pd.DataFrame:
    """TR.IndexConstituentRIC result -> tidy {month_end, ric}. The result carries one row
    per constituent with a RIC-bearing column (header varies: 'Constituent RIC' /
    'TR.IndexConstituentRIC' / 'Instrument')."""
    ric_col = _first_col(df, ["Constituent RIC", "TR.IndexConstituentRIC", "RIC", "Instrument"],
                         ["constituent", "ric", "instrument"])
    if ric_col is None:
        raise ValueError(f"no RIC column in constituent frame; columns={list(df.columns)}")
    rics = df[ric_col].dropna().astype(str).str.strip()
    rics = rics[rics != ""]
    return pd.DataFrame({"month_end": pd.Timestamp(month_end), "ric": rics.values})


def parse_datastream_list(df: pd.DataFrame, month_end: str | pd.Timestamp) -> pd.DataFrame:
    """Datastream LS&PCOMP{MMYY} static request -> tidy {month_end, ric}. DSWS returns the
    list members with an instrument/symbol column."""
    sym_col = _first_col(df, ["Instrument", "Symbol", "DS Mnemonic", "RIC"],
                         ["instrument", "symbol", "mnemonic", "ric"])
    if sym_col is None:
        raise ValueError(f"no symbol column in Datastream list; columns={list(df.columns)}")
    syms = df[sym_col].dropna().astype(str).str.strip()
    syms = syms[syms != ""]
    return pd.DataFrame({"month_end": pd.Timestamp(month_end), "ric": syms.values})


def parse_delisting_metadata(df: pd.DataFrame) -> dict[str, dict]:
    """Delisting fields -> {ric: {date, exchange, reason, vendor_terminal_return}} consumable
    by membership.apply_shumway_corrections. Exchange string is mapped to the Shumway bucket
    ('nyse_amex' | 'nasdaq'); an unmapped exchange leaves bucket None so the caller decides."""
    ric_col = _first_col(df, ["Instrument", "RIC"], ["instrument", "ric"])
    date_col = _first_col(df, ["Delisting Date", "Date"], ["delist", "date"])
    exch_col = _first_col(df, ["Exchange Name", "TR.ExchangeName"], ["exchange"])
    reason_col = _first_col(df, ["Reason", "Description"], ["reason", "descrip"])
    ret_col = _first_col(df, ["Terminal Return", "TR.TotalReturn"], ["terminal", "totalreturn", "return"])
    if ric_col is None or date_col is None:
        raise ValueError("delisting frame needs at least RIC and date columns")
    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        ric = str(row[ric_col]).strip()
        if not ric or ric.lower() == "nan":
            continue
        exch = str(row[exch_col]).lower() if exch_col and pd.notna(row.get(exch_col)) else ""
        bucket = ("nasdaq" if "nasdaq" in exch
                  else "nyse_amex" if any(x in exch for x in ("new york", "nyse", "amex", "american"))
                  else None)
        out[ric] = {
            "date": pd.Timestamp(row[date_col]) if pd.notna(row[date_col]) else None,
            "exchange": bucket,
            "reason": (str(row[reason_col]) if reason_col and pd.notna(row.get(reason_col)) else None),
            "vendor_terminal_return": (float(row[ret_col])
                                       if ret_col and pd.notna(row.get(ret_col)) else None),
        }
    return out


def parse_panel_field(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Wide-or-long vendor history -> (date-indexed, RIC-columned) panel. Refinitiv
    get_history returns a Date index with one column per instrument already; a long
    (Instrument, Date, value) shape is pivoted. Pure reshape — no value fabrication."""
    if isinstance(df.index, pd.DatetimeIndex) or _first_col(df, ["Date"], ["date"]) is None:
        out = df.copy()
        out.index = pd.to_datetime(out.index)
        return out.sort_index()
    date_col = _first_col(df, ["Date"], ["date"])
    inst_col = _first_col(df, ["Instrument", "RIC"], ["instrument", "ric"])
    val_col = _first_col(df, [value_name], [value_name.lower().split(".")[-1]]) or df.columns[-1]
    wide = df.pivot_table(index=date_col, columns=inst_col, values=val_col, aggfunc="last")
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def parse_jl_events(df: pd.DataFrame) -> pd.DataFrame:
    """Joiners/leavers event frame (TR.IndexJL* fields) -> tidy {date, ric} sorted
    ascending. Dead ^RICs are preserved verbatim — they are exactly what the A2
    history pull needs."""
    ric_col = _first_col(df, ["Constituent RIC"], ["constituent", "ric"])
    date_col = _first_col(df, ["Date"], ["date"])
    if ric_col is None or date_col is None:
        raise ValueError(f"JL frame needs constituent-RIC and date columns; got {list(df.columns)}")
    out = pd.DataFrame({"date": pd.to_datetime(df[date_col]), "ric": df[ric_col].astype(str).str.strip()})
    out = out[(out["ric"] != "") & out["date"].notna() & (out["ric"].str.lower() != "nan")]
    return out.sort_values("date").reset_index(drop=True)


def reconstruct_membership(
    current_rics: list[str],
    joiners: pd.DataFrame,
    leavers: pd.DataFrame,
    month_ends: list[pd.Timestamp],
) -> pd.DataFrame:
    """Point-in-time membership by REVERSE EVENT REPLAY (the method this feed actually
    supports: snapshot/dated-chain queries silently return the CURRENT chain — a
    survivorship trap caught by content validation, ADR-020).

    Semantics: membership(m) reflects all events dated <= m. Walking month-ends
    NEWEST -> OLDEST from today's chain, every event dated AFTER m is UNDONE
    (undo join = remove; undo leave = add). Pure function; synthetic-fixture tested
    (multi-episode names, boundary-dated events, count conservation)."""
    events = pd.concat([
        parse_jl_events(joiners).assign(kind="J"),
        parse_jl_events(leavers).assign(kind="L"),
    ]).sort_values("date", ascending=False).reset_index(drop=True)

    state = set(current_rics)
    rows: list[dict] = []
    ev_i = 0
    for m in sorted(month_ends, reverse=True):
        while ev_i < len(events) and events.loc[ev_i, "date"] > m:
            ric, kind = events.loc[ev_i, "ric"], events.loc[ev_i, "kind"]
            if kind == "J":
                state.discard(ric)               # undo a future join
            else:
                state.add(ric)                   # undo a future leave
            ev_i += 1
        rows += [{"month_end": m, "ric": r} for r in sorted(state)]
    out = pd.DataFrame(rows).sort_values(["month_end", "ric"]).reset_index(drop=True)
    return out


def validate_membership(pit: pd.DataFrame, lo: int = 495, hi: int = 510) -> dict:
    """Content gates the reconstruction must pass (counts ~500-505 every month; known
    joiner/leaver truths). Returns the validation dict; RAISES on a count breach —
    a drifting count means a missed/duplicated event and the artifact must not ship."""
    counts = pit.groupby("month_end")["ric"].count()
    breaches = counts[(counts < lo) | (counts > hi)]
    checks = {
        "months": int(counts.shape[0]),
        "count_min": int(counts.min()), "count_max": int(counts.max()),
        "lehman_in_2008H1": bool(pit[(pit["month_end"] == pd.Timestamp("2008-06-30"))
                                     & pit["ric"].str.startswith("LEH")].shape[0]),
        "lehman_out_2008Q4": not bool(pit[(pit["month_end"] == pd.Timestamp("2008-10-31"))
                                          & pit["ric"].str.startswith("LEH")].shape[0]),
        "tesla_out_2019": not bool(pit[(pit["month_end"] == pd.Timestamp("2019-12-31"))
                                       & pit["ric"].str.startswith("TSLA")].shape[0]),
        "tesla_in_2021": bool(pit[(pit["month_end"] == pd.Timestamp("2021-06-30"))
                                  & pit["ric"].str.startswith("TSLA")].shape[0]),
    }
    if len(breaches):
        raise RuntimeError(f"membership count breach (missed/duplicate events?): "
                           f"{breaches.head(5).to_dict()} — artifact must not ship")
    if not all(v for k, v in checks.items() if isinstance(v, bool)):
        raise RuntimeError(f"membership known-truth spot-check FAILED: {checks}")
    return checks


# ----------------------------------------------------------------------- orchestrator
def month_end_grid(start: str, end: str) -> list[pd.Timestamp]:
    """Month-end snapshot grid for the membership reconstruction."""
    return list(pd.date_range(start, end, freq="ME"))


def acquire_universe(dry_run: bool = True) -> dict:
    """Drive A1-A5 through the journaled engine. With `dry_run=True` (default) this only
    REPORTS the plan (chunk counts, fields, journal id) without opening a session — safe to
    call anywhere. With `dry_run=False` it opens the entitled session and pulls; that path is
    reached only by `make pull-universe` once scopes exist (it raises EntitlementError, which
    the CLI records into the run sidecar, if they do not).
    """
    from . import acquire  # local import: keeps module import-light and test-isolatable

    cfg = get("data.universe_pull")
    window = get("data.window")
    months = month_end_grid(window["start"], window["end"])
    plan = {
        "A1_membership_refinitiv_requests": len(months),
        "A1_membership_datastream_lists": len(months),
        "A2_total_return_field": cfg["total_return"]["refinitiv_field"],
        "A3_market_cap_field": cfg["market_cap"]["refinitiv_field"],
        "A4_delisting_fields": cfg["delisting"]["refinitiv_fields"],
        "A5_hardening_roles": {k: v.get("role") for k, v in cfg["hardening"].items()},
        "window": [window["start"], window["end"]],
        "chunking": get("data.platform.chunking"),
    }
    if dry_run:
        plan["mode"] = "dry_run (no session opened; plan only)"
        return plan

    from .vault import freeze, manifest_entries, read_verified, record_lineage

    rd = acquire.open_refinitiv_session()             # EntitlementError if scopes empty
    journal = acquire.PullJournal(pull_id="universe_refinitiv")
    governor = acquire.RateGovernor()
    frozen: list = []
    window = get("data.window")
    today = pd.Timestamp.today().date().isoformat()
    buffer_months = int(get("data.universe_pull.selection_buffer_months", 0))
    buffered_start = (pd.Timestamp(window["start"])
                      - pd.offsets.MonthBegin(buffer_months + 1)).date().isoformat()
    months = month_end_grid(buffered_start, window["end"])      # incl. the pre-window buffer

    # ---- A1: PIT membership by EVENT REPLAY (ADR-020). Snapshot/dated-chain queries
    # silently return the CURRENT chain on this route (survivorship trap — the 92
    # rf_members_* v1 artifacts are INVALIDATED, see data/manifest/invalidated.jsonl).
    # Three requests: today's chain anchor + the full joiner and leaver event streams.
    jl_fields = ["TR.IndexJLConstituentRIC", "TR.IndexJLConstituentName",
                 "TR.IndexJLConstituentChangeDate"]
    a1_chunks = [
        ("chain_current", {"universe": "0#.SPX", "fields": ["TR.CommonName"], "parameters": None}),
        ("jl_joiners", {"universe": ".SPX", "fields": jl_fields,
                        "parameters": {"SDate": window["start"], "EDate": today, "IC": "J"}}),
        ("jl_leavers", {"universe": ".SPX", "fields": jl_fields,
                        "parameters": {"SDate": window["start"], "EDate": today, "IC": "L"}}),
    ]
    frozen += acquire.journaled_pull(
        "refinitiv", "membership_event_replay", a1_chunks,
        fetch=lambda p: acquire.fetch_refinitiv_data(rd, p["universe"], p["fields"], p["parameters"]),
        journal=journal, name_for=lambda label, p: f"rf_{label}.csv",
        governor=governor)

    # ---- reconstruct + VALIDATE membership; freeze as a STAGED derived artifact
    # (versioned by month-grid span: extending the buffer yields a new artifact;
    # build_universe consumes the LATEST pit_membership* manifest entry)
    chain = read_verified("rf_chain_current.csv")
    joiners = read_verified("rf_jl_joiners.csv")
    leavers = read_verified("rf_jl_leavers.csv")
    current_rics = sorted(set(chain["Instrument"].dropna().astype(str)))
    pit = reconstruct_membership(current_rics, joiners, leavers, months)
    checks = validate_membership(pit)                 # raises on count/known-truth breach
    pit_name = f"pit_membership_{months[0].strftime('%Y%m')}_{months[-1].strftime('%Y%m')}.parquet"
    existing = {e["name"] for e in manifest_entries()}
    if pit_name not in existing:
        art = freeze(pit, "staged", pit_name,
                     {"vendor": "derived", "stage": "pull_universe.reconstruct_membership",
                      "method": "reverse event replay (ADR-020)",
                      "selection_buffer_months": buffer_months, "validation": checks},
                     fmt="parquet", run_id="universe_refinitiv")
        inputs = [e for e in manifest_entries() if e["name"] in
                  ("rf_chain_current.csv", "rf_jl_joiners.csv", "rf_jl_leavers.csv")]
        record_lineage(art, inputs, "membership_event_replay", checks)

    # ---- A2+ UNION: everyone who was EVER a member in-window
    union_rics = sorted(set(pit["ric"]))
    if not union_rics:
        raise RuntimeError("A1 produced no members — inspect rf_chain/rf_jl artifacts before A2-A5")

    # ---- A2: DAILY TOTAL RETURNS via datagrid LONG form (content-validated 2026-06-12:
    # `TR.TotalReturn` through get_history returns empty/NaN on this route — those 39
    # rf_tr_* artifacts are invalidated; the long form with Frq=D returns real percent
    # series incl. dead ^RICs, e.g. Lehman daily through 2008-09-12, worst day -44.9%).
    # Chunked 25 names x ~2 years to stay far below datagrid row caps.
    tr_field = cfg["total_return"]["refinitiv_field"]
    def _trd_chunks():
        for i, names in enumerate(acquire.chunk_tickers(union_rics)):
            for j, (a, b) in enumerate(acquire.chunk_dates(window["start"], window["end"], days=730)):
                yield (f"{i:03d}_{j:02d}",
                       {"universe": names, "fields": [f"{tr_field}.date", tr_field],
                        "parameters": {"SDate": a, "EDate": b, "Frq": "D"}})
    frozen += acquire.journaled_pull(
        "refinitiv", f"datagrid.{tr_field}.D", _trd_chunks(),
        fetch=lambda p: acquire.fetch_refinitiv_data(rd, p["universe"], p["fields"], p["parameters"]),
        journal=journal, name_for=lambda label, p: f"rf_trd_{label}.csv",
        governor=governor)

    # ---- A3: MONTHLY market cap (long form, Frq=M — selection needs caps at window
    # starts; the strictly-prior top-30 rule consumes month-end caps, PREREG §5)
    mc_field = cfg["market_cap"]["refinitiv_field"]
    def _mcap_chunks():
        for i, names in enumerate(acquire.chunk_tickers(union_rics)):
            yield (f"{i:03d}", {"universe": names, "fields": [f"{mc_field}.date", mc_field],
                                "parameters": {"SDate": buffered_start, "EDate": window["end"],
                                               "Frq": "M"}})   # buffered: first selection is PIT-clean
    mcap_stamp = buffered_start[:7].replace("-", "")    # span-stamped: re-pulls with a
    frozen += acquire.journaled_pull(                   # different SDate version cleanly
        "refinitiv", f"datagrid.{mc_field}.M", _mcap_chunks(),
        fetch=lambda p: acquire.fetch_refinitiv_data(rd, p["universe"], p["fields"], p["parameters"]),
        journal=journal, name_for=lambda label, p: f"rf_mcapm_{label}_{mcap_stamp}.csv",
        governor=governor)

    # ---- A5: price/bid/ask/volume via no-fields get_history (content-validated:
    # returns TRDPRC_1/BID/ASK/ACVOL_UNS together). MultiIndex response is split into
    # per-field flat artifacts before freezing (lossless CSV round-trip). PARALLEL:
    # launches stay governor-spaced; heavy responses overlap across workers.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    a5_fields = {"TRDPRC_1": "px", "BID": "bid", "ASK": "ask", "ACVOL_UNS": "vol"}

    def _ohlc_chunk(i: int, names: list[str]) -> list:
        cid = acquire.chunk_id("refinitiv", "get_history.ohlc", {"universe": names})
        if journal.done(cid):
            return []
        governor.wait()
        requested = pd.Timestamp.utcnow().isoformat()
        try:
            h = acquire.with_backoff(lambda n=names: rd.get_history(
                universe=n, start=window["start"], end=window["end"]))
        except RuntimeError as e:
            journal.mark(cid, "failed", label=f"ohlc_{i:03d}", error=str(e)[:300])
            raise
        if h is None or len(h) == 0:
            journal.mark(cid, "skipped_empty", label=f"ohlc_{i:03d}")
            return []
        arts, relpaths = [], []
        for fld, tag in a5_fields.items():
            if h.columns.nlevels == 2:
                cols = [c for c in h.columns if c[1] == fld]
                sub = h[cols]
                sub.columns = [c[0] for c in cols]
            else:                                   # single instrument: flat columns
                if fld not in h.columns:
                    continue
                sub = h[[fld]]
                sub.columns = names[:1]
            if sub.dropna(how="all").empty:
                continue
            art = freeze(sub, "raw", f"rf_{tag}_{i:03d}.csv",
                         acquire.provenance("refinitiv", "get_history.ohlc",
                                            {"universe": names, "field": fld}, requested),
                         run_id="universe_refinitiv")
            relpaths.append(art.relpath)
            arts.append(art)
        journal.mark(cid, "frozen", label=f"ohlc_{i:03d}", relpaths=relpaths)
        return arts

    n_workers = int(get("data.platform.rate_limit.workers", 1))
    ohlc_failures: list[BaseException] = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(_ohlc_chunk, i, names)
                for i, names in enumerate(acquire.chunk_tickers(union_rics))]
        for fut in as_completed(futs):
            try:
                frozen += fut.result()
            except BaseException as e:              # noqa: BLE001 — collected, re-raised
                ohlc_failures.append(e)
    if ohlc_failures:
        raise ohlc_failures[0]

    # ---- A4 (+A5 sector): static metadata over the union (flat frame, chunked)
    meta_fields = cfg["delisting"]["refinitiv_fields"]
    def _meta_chunks():
        for i, chunk in enumerate(acquire.chunk_tickers(union_rics, per_request=200)):
            yield (f"{i:02d}", {"universe": chunk, "fields": meta_fields})
    frozen += acquire.journaled_pull(
        "refinitiv", "get_data.delisting_meta", _meta_chunks(),
        fetch=lambda p: acquire.fetch_refinitiv_data(rd, p["universe"], p["fields"]),
        journal=journal, name_for=lambda label, p: f"rf_meta_{label}.csv",
        governor=governor)

    # ---- A5 benchmark: S&P 500 TR index (.SPXTR); VIX deliberately absent (not licensed,
    # config index_parity.vix_ric: null — FRED VIXCLS is the VIX source)
    spx_ric = cfg["hardening"]["index_parity"]["sp500_tr_index"]
    frozen += acquire.journaled_pull(
        "refinitiv", "get_history.spxtr",
        [("spxtr", {"universe": [spx_ric], "fields": None,
                    "start": window["start"], "end": window["end"]})],
        fetch=lambda p: rd.get_history(universe=p["universe"], start=p["start"], end=p["end"]),
        journal=journal, name_for=lambda label, p: "rf_spxtr.csv",
        governor=governor)

    try:
        import refinitiv.data as rdlib
        rdlib.close_session()
    except Exception:
        pass
    return {"n_union_rics": len(union_rics), "frozen_this_run": len(frozen),
            "journal": journal.summary(), "plan": plan}
