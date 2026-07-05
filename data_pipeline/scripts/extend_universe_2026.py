#!/usr/bin/env python3
"""Forward-extension pull to the 2026-06-30 settled cutoff (ADR-051; the ULTRAPLAN P1 rebuild).

WHY A DEDICATED DRIVER (not a config-span re-run of ``acquire_universe``): the journal keys
chunks by ``sha256(vendor, query, params)``, so bumping ``period.end`` re-keys EVERY chunk and
would re-pull the whole 21-year vault; leaving the config alone and re-running skips everything
as done. This driver therefore pulls ONLY the extension, under ``_x26``-labelled artifact names
(the vault is write-once; no frozen name is ever re-frozen), in its OWN journal
(``universe_refinitiv_x26``), reusing the pipeline's engine end-to-end (RateGovernor, backoff,
journaled_pull, freeze/provenance, parsers, reconstruct/validate_membership).

WHAT IT PULLS (all consumed automatically by ``build_universe`` via its ``rf_trd_``/``rf_mcapm_``
prefix collection and the LATEST ``pit_membership_*`` staged artifact):
  A1' chain+joiner/leaver EVENT streams, FULL span to today (fresh reverse-replay inputs)
      -> reconstruct over the month grid extended to 2026-06
      -> **HARD-FAIL overlap check**: every month-end present in the FROZEN pit must carry an
         IDENTICAL member set in the fresh reconstruction (a vendor event-history revision must
         STOP the rebuild for investigation, never silently land — ADR-051 acceptance gate 1)
      -> freeze the new span-stamped ``pit_membership_<first>_<last>.parquet``.
  A2' daily total returns: OLD-union names over 2026-01-01..2026-06-30; BRAND-NEW 2026 joiners
      over the FULL window (their pre-listing emptiness journals as skipped_empty).
  A3' monthly market caps: old names 2026 month-ends; new joiners the full buffered span
      (the strictly-prior top-30 rule needs their caps wherever they are selectable).
  A4' delisting metadata over the new union (cheap; catches names delisted IN 2026).
  A5' px/bid/ask/vol via no-fields get_history over the extension window (feeds the report-only
      spread/impact cost model for the 2026 test months).
  SPXTR extension (index-parity integrity series).

RUN (PowerShell + .venv-lseg ONLY — ADR-048; the Bash sandbox cannot resolve api.refinitiv.com):
    $env:PYTHONPATH = "<repo>\\data_pipeline"
    & "<repo>\\.venv-lseg\\Scripts\\python.exe" data_pipeline\\scripts\\extend_universe_2026.py            # DRY-RUN (plan only)
    & "<repo>\\.venv-lseg\\Scripts\\python.exe" data_pipeline\\scripts\\extend_universe_2026.py --live     # the real pull

Idempotent: the x26 journal resumes; frozen x26 chunks are never re-pulled. ASCII-only output
(the Windows console is cp1251).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
_DP = _REPO / "data_pipeline"
sys.path.insert(0, str(_DP))

from src.config import get  # noqa: E402
from src.data import acquire  # noqa: E402
from src.data.pull_universe import (  # noqa: E402
    month_end_grid,
    reconstruct_membership,
    validate_membership,
)
from src.data.vault import freeze, manifest_entries, read_verified, record_lineage  # noqa: E402

EXT_START = "2026-01-01"   # first session after the frozen univ3 end (2025-12-31)
EXT_END = "2026-06-30"     # ADR-051: the settled month-end cutoff
RUN_ID = "universe_refinitiv_x26"


def _latest_frozen_pit_name() -> str:
    """The newest FROZEN pit_membership artifact name (the reconstruction to overlap-check against)."""
    pits = sorted(e["name"] for e in manifest_entries()
                  if str(e.get("name", "")).startswith("pit_membership_"))
    if not pits:
        raise FileNotFoundError("no frozen pit_membership_* artifact — the base pull must exist first")
    return pits[-1]


#: Fresh-replay-vs-frozen overlap differences VERIFIED IMMATERIAL and allowlisted (ADR-051 addendum,
#: 2026-07-02). EVHC.N^L16: the vendor BACKFILLED the Dec-2016 leaver event between the frozen pull
#: (2026-06-12) and this one; its join counterpart is missing/re-keyed, so reverse replay over-extends
#: membership back to the grid start — provably wrong (old-EVHC IPO'd Aug-2013; merged into AMSURG
#: 2016-12-01, NYSE delisting 2016-12-13 = the ^L16 suffix; SEC Form 25-NSE) and provably immaterial
#: (peak cap ~$7B — never remotely a top-30 mega-cap, so selection is invariant either way). The
#: FROZEN pit therefore stays the authoritative record for its own span; anything NOT in this
#: allowlist still hard-fails.
_OVERLAP_ALLOWLIST: frozenset[str] = frozenset({"EVHC.N^L16"})


def overlap_check(old_pit: pd.DataFrame, new_pit: pd.DataFrame) -> dict:
    """ADR-051 gate 1 (diagnostic form): compare member sets on every common month.

    The SPLICE rule (ADR-051 addendum) makes the FROZEN pit authoritative for its own span, so
    overlap differences do not enter the spliced artifact — but they are still diagnosed, and any
    ric outside ``_OVERLAP_ALLOWLIST`` HARD-FAILS the run (an unexplained vendor revision must stop
    the rebuild, never be silently spliced around).

    Returns ``{"months_checked", "mismatches", "unexplained"}``.
    """
    old_by_m = {m: frozenset(g["ric"]) for m, g in old_pit.groupby("month_end")}
    new_by_m = {m: frozenset(g["ric"]) for m, g in new_pit.groupby("month_end")}
    mismatches: list[dict] = []
    unexplained: set[str] = set()
    common = sorted(set(old_by_m) & set(new_by_m))
    for m in common:
        if old_by_m[m] != new_by_m[m]:
            gone = sorted(old_by_m[m] - new_by_m[m])
            added = sorted(new_by_m[m] - old_by_m[m])
            unexplained |= (set(gone) | set(added)) - _OVERLAP_ALLOWLIST
            mismatches.append({"month_end": str(m), "missing_in_new": gone[:10],
                               "extra_in_new": added[:10],
                               "n_missing": len(gone), "n_extra": len(added)})
    return {"months_checked": len(common), "mismatches": mismatches,
            "unexplained": sorted(unexplained)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Forward-extension pull to 2026-06-30 (ADR-051).")
    ap.add_argument("--live", action="store_true",
                    help="open the Refinitiv session and PULL (default: dry-run plan only)")
    args = ap.parse_args(argv)

    window = get("data.window")
    cfg = get("data.universe_pull")
    buffer_months = int(get("data.universe_pull.selection_buffer_months", 0))
    buffered_start = (pd.Timestamp(window["start"])
                      - pd.offsets.MonthBegin(buffer_months + 1)).date().isoformat()
    months = month_end_grid(buffered_start, EXT_END)
    today = pd.Timestamp.today().date().isoformat()

    old_pit_name = _latest_frozen_pit_name()
    old_pit = read_verified(old_pit_name)
    old_union = sorted(set(old_pit["ric"].astype(str)))
    tr_field = cfg["total_return"]["refinitiv_field"]
    mc_field = cfg["market_cap"]["refinitiv_field"]
    meta_fields = cfg["delisting"]["refinitiv_fields"]
    spx_ric = cfg["hardening"]["index_parity"]["sp500_tr_index"]

    n_tick = len(acquire.chunk_tickers(old_union))
    print(f"[x26] base pit: {old_pit_name}  (union={len(old_union)} rics)")
    print(f"[x26] month grid: {months[0].date()} .. {months[-1].date()}  ({len(months)} month-ends)")
    print(f"[x26] extension: {EXT_START} .. {EXT_END}; A1' events {window['start']} .. {today}")
    print(f"[x26] plan: A1' 3 req | A2' ~{n_tick} chunks (+new-ric full-window) | "
          f"A3' ~{n_tick} | A4' ~{(len(old_union) // 200) + 1} | A5' ~{n_tick} | SPXTR 1")

    if not args.live:
        print("[x26] DRY-RUN complete (no session opened). Re-run with --live to pull.")
        return 0

    rd = acquire.open_refinitiv_session()
    journal = acquire.PullJournal(pull_id=RUN_ID)
    governor = acquire.RateGovernor()
    frozen: list = []

    # ---- A1': fresh chain + FULL event streams (x26 names; write-once safe) -------------
    jl_fields = ["TR.IndexJLConstituentRIC", "TR.IndexJLConstituentName",
                 "TR.IndexJLConstituentChangeDate"]
    a1_chunks = [
        ("chain_current_x26", {"universe": "0#.SPX", "fields": ["TR.CommonName"], "parameters": None}),
        ("jl_joiners_x26", {"universe": ".SPX", "fields": jl_fields,
                            "parameters": {"SDate": window["start"], "EDate": today, "IC": "J"}}),
        ("jl_leavers_x26", {"universe": ".SPX", "fields": jl_fields,
                            "parameters": {"SDate": window["start"], "EDate": today, "IC": "L"}}),
    ]
    frozen += acquire.journaled_pull(
        "refinitiv", "membership_event_replay_x26", a1_chunks,
        fetch=lambda p: acquire.fetch_refinitiv_data(rd, p["universe"], p["fields"], p["parameters"]),
        journal=journal, name_for=lambda label, p: f"rf_{label}.csv",
        governor=governor)

    chain = read_verified("rf_chain_current_x26.csv")
    joiners = read_verified("rf_jl_joiners_x26.csv")
    leavers = read_verified("rf_jl_leavers_x26.csv")
    current_rics = sorted(set(chain["Instrument"].dropna().astype(str)))
    pit = reconstruct_membership(current_rics, joiners, leavers, months)
    checks = validate_membership(pit)
    print(f"[x26] fresh reconstruction: {len(pit)} rows; validation: {checks}")

    ov = overlap_check(old_pit, pit)
    if ov["unexplained"]:
        for mm in ov["mismatches"][:6]:
            print(f"[x26] OVERLAP MISMATCH {mm['month_end']}: "
                  f"-{mm['n_missing']} {mm['missing_in_new']} / +{mm['n_extra']} {mm['extra_in_new']}")
        raise RuntimeError(
            f"membership overlap check FAILED: UNEXPLAINED ric(s) {ov['unexplained'][:8]} across "
            f"{len(ov['mismatches'])}/{ov['months_checked']} months — a vendor event-history revision "
            "outside the verified allowlist; STOP and investigate before any rebuild (ADR-051 gate 1)")
    if ov["mismatches"]:
        print(f"[x26] overlap: {len(ov['mismatches'])}/{ov['months_checked']} months differ, ALL within "
              f"the verified-immaterial allowlist {sorted(_OVERLAP_ALLOWLIST)} — the SPLICE rule keeps "
              "the frozen record authoritative for its span (ADR-051 addendum)")
    else:
        print(f"[x26] overlap check PASSED: {ov['months_checked']} common month-ends identical")

    # SPLICE (ADR-051 addendum): the FROZEN pit is the pre-registered membership record through its
    # own last month; the fresh replay contributes ONLY the months beyond it (2026 — the region where
    # reverse-replay-from-today's-chain is at its most reliable). No vendor revision can silently
    # rewrite frozen history; the extension is append-only, mirroring the panel byte-diff contract.
    old_last = pd.Timestamp(old_pit["month_end"].max())
    fresh_tail = pit[pit["month_end"] > old_last]
    new_months = sorted(fresh_tail["month_end"].unique())
    if len(new_months) != 6 or pd.Timestamp(max(new_months)) != pd.Timestamp(EXT_END):
        raise RuntimeError(
            f"extension months look wrong: expected 6 month-ends ending {EXT_END}, got "
            f"{[str(pd.Timestamp(m).date()) for m in new_months]} — investigate the fresh replay")
    counts = fresh_tail.groupby("month_end")["ric"].nunique()
    if not ((counts >= 495) & (counts <= 510)).all():
        raise RuntimeError(f"2026 member counts out of bounds: {counts.to_dict()}")
    pit_spliced = pd.concat([old_pit, fresh_tail], ignore_index=True)

    pit_name = f"pit_membership_{months[0].strftime('%Y%m')}_{months[-1].strftime('%Y%m')}.parquet"
    if pit_name not in {e["name"] for e in manifest_entries()}:
        art = freeze(pit_spliced, "staged", pit_name,
                     {"vendor": "derived", "stage": "extend_universe_2026.reconstruct_membership",
                      "method": "SPLICE (ADR-051 addendum): frozen pit authoritative through "
                                f"{old_last.date()}; fresh reverse event replay (ADR-020) for "
                                f"{[str(pd.Timestamp(m).date()) for m in new_months]}",
                      "selection_buffer_months": buffer_months, "validation": checks,
                      "overlap_diagnostic": {"months_checked": ov["months_checked"],
                                             "months_differing": len(ov["mismatches"]),
                                             "allowlisted": sorted(_OVERLAP_ALLOWLIST),
                                             "unexplained": ov["unexplained"]}},
                     fmt="parquet", run_id=RUN_ID)
        inputs = [e for e in manifest_entries() if e["name"] in
                  ("rf_chain_current_x26.csv", "rf_jl_joiners_x26.csv", "rf_jl_leavers_x26.csv")]
        record_lineage(art, inputs, "membership_event_replay_x26_splice", checks)
        print(f"[x26] froze {pit_name} (spliced: frozen span + {len(new_months)} new month-ends)")
    else:
        print(f"[x26] {pit_name} already frozen (resume)")

    union_new = sorted(set(pit_spliced["ric"].astype(str)))
    new_rics = sorted(set(union_new) - set(old_union))
    print(f"[x26] union: {len(union_new)} rics; brand-new vs old union: {len(new_rics)} {new_rics[:8]}")

    # ---- A2': daily total returns -------------------------------------------------------
    def _trd_ext_chunks():
        for i, names in enumerate(acquire.chunk_tickers(old_union)):
            yield (f"x26_{i:03d}_00",
                   {"universe": names, "fields": [f"{tr_field}.date", tr_field],
                    "parameters": {"SDate": EXT_START, "EDate": EXT_END, "Frq": "D"}})
        for i, names in enumerate(acquire.chunk_tickers(new_rics)):
            for j, (a, b) in enumerate(acquire.chunk_dates(window["start"], EXT_END, days=730)):
                yield (f"x26n_{i:03d}_{j:02d}",
                       {"universe": names, "fields": [f"{tr_field}.date", tr_field],
                        "parameters": {"SDate": a, "EDate": b, "Frq": "D"}})
    frozen += acquire.journaled_pull(
        "refinitiv", f"datagrid.{tr_field}.D.x26", _trd_ext_chunks(),
        fetch=lambda p: acquire.fetch_refinitiv_data(rd, p["universe"], p["fields"], p["parameters"]),
        journal=journal, name_for=lambda label, p: f"rf_trd_{label}.csv",
        governor=governor)

    # ---- A3': monthly market caps -------------------------------------------------------
    def _mcap_ext_chunks():
        for i, names in enumerate(acquire.chunk_tickers(old_union)):
            yield (f"x26_{i:03d}", {"universe": names, "fields": [f"{mc_field}.date", mc_field],
                                    "parameters": {"SDate": EXT_START, "EDate": EXT_END, "Frq": "M"}})
        for i, names in enumerate(acquire.chunk_tickers(new_rics)):
            yield (f"x26n_{i:03d}", {"universe": names, "fields": [f"{mc_field}.date", mc_field],
                                     "parameters": {"SDate": buffered_start, "EDate": EXT_END,
                                                    "Frq": "M"}})
    frozen += acquire.journaled_pull(
        "refinitiv", f"datagrid.{mc_field}.M.x26", _mcap_ext_chunks(),
        fetch=lambda p: acquire.fetch_refinitiv_data(rd, p["universe"], p["fields"], p["parameters"]),
        journal=journal, name_for=lambda label, p: f"rf_mcapm_{label}_202601.csv",
        governor=governor)

    # ---- A4': delisting metadata over the NEW union (catches 2026 delistings) -----------
    def _meta_ext_chunks():
        for i, chunk in enumerate(acquire.chunk_tickers(union_new, per_request=200)):
            yield (f"x26_{i:02d}", {"universe": chunk, "fields": meta_fields})
    frozen += acquire.journaled_pull(
        "refinitiv", "get_data.delisting_meta.x26", _meta_ext_chunks(),
        fetch=lambda p: acquire.fetch_refinitiv_data(rd, p["universe"], p["fields"]),
        journal=journal, name_for=lambda label, p: f"rf_meta_{label}.csv",
        governor=governor)

    # ---- A5': px/bid/ask/vol over the extension window (mirrors the base _ohlc_chunk) ---
    a5_fields = {"TRDPRC_1": "px", "BID": "bid", "ASK": "ask", "ACVOL_UNS": "vol"}
    for i, names in enumerate(acquire.chunk_tickers(union_new)):
        cid = acquire.chunk_id("refinitiv", "get_history.ohlc.x26", {"universe": names})
        if journal.done(cid):
            continue
        governor.wait()
        requested = pd.Timestamp.utcnow().isoformat()
        try:
            h = acquire.with_backoff(lambda n=names: rd.get_history(
                universe=n, start=EXT_START, end=EXT_END))
        except RuntimeError as e:
            journal.mark(cid, "failed", label=f"ohlc_x26_{i:03d}", error=str(e)[:300])
            raise
        if h is None or len(h) == 0:
            journal.mark(cid, "skipped_empty", label=f"ohlc_x26_{i:03d}")
            continue
        relpaths = []
        for fld, tag in a5_fields.items():
            if h.columns.nlevels == 2:
                cols = [c for c in h.columns if c[1] == fld]
                sub = h[cols]
                sub.columns = [c[0] for c in cols]
            else:
                if fld not in h.columns:
                    continue
                sub = h[[fld]]
                sub.columns = names[:1]
            if sub.dropna(how="all").empty:
                continue
            art = freeze(sub, "raw", f"rf_{tag}_x26_{i:03d}.csv",
                         acquire.provenance("refinitiv", "get_history.ohlc.x26",
                                            {"universe": names, "field": fld}, requested),
                         run_id=RUN_ID)
            relpaths.append(art.relpath)
            frozen.append(art)
        journal.mark(cid, "frozen", label=f"ohlc_x26_{i:03d}", relpaths=relpaths)

    # ---- SPXTR extension (index-parity integrity series) --------------------------------
    frozen += acquire.journaled_pull(
        "refinitiv", "get_history.spxtr.x26",
        [("spxtr_x26", {"universe": [spx_ric], "fields": None,
                        "start": EXT_START, "end": EXT_END})],
        fetch=lambda p: rd.get_history(universe=p["universe"], start=p["start"], end=p["end"]),
        journal=journal, name_for=lambda label, p: "rf_spxtr_x26.csv",
        governor=governor)

    try:
        import refinitiv.data as rdlib
        rdlib.close_session()
    except Exception:  # noqa: BLE001 - session teardown must never mask a completed pull
        pass

    print(f"[x26] DONE: froze {len(frozen)} artifacts this run; journal: {journal.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
