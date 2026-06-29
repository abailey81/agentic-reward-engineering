"""Data-platform CLI (stage 13):

    python -m src.data.cli probe   [--offline]
    python -m src.data.cli pull    [--pilot] [--vendors yfinance,fred,french,refinitiv]
    python -m src.data.cli build   [--suffix _pilot]
    python -m src.data.cli validate
    python -m src.data.cli reconcile
    python -m src.data.cli eda
    python -m src.data.cli status

Every stage writes a run sidecar (runs/data/<stage>_<runid>.json: config hashes,
inputs, outputs, counts, wall-clock) — the project's compute-reporting discipline.
Stages degrade gracefully: a missing vendor/entitlement yields an explicit SKIP with
its reason in the sidecar, never a fabricated artifact (R4).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import CONFIG_DIR, get
from . import acquire, eda, integrity, probes, quality, reconcile_full, validate
from .panel import build_gold
from .vault import find_entry, freeze, manifest_entries, read_verified, record_lineage

ROOT = Path(__file__).resolve().parent.parent.parent


# ----------------------------------------------------------------------- run infra
def _run_id(stage: str) -> str:
    return f"{stage}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _config_hashes() -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            for p in sorted(CONFIG_DIR.glob("*.yaml"))}


def _sidecar(stage: str, run_id: str, payload: dict, t0: float) -> Path:
    out_dir = ROOT / get("data.platform.runs_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "run_id": run_id, "config_sha256_16": _config_hashes(),
               "wall_clock_s": round(time.monotonic() - t0, 2),
               "utc": datetime.now(timezone.utc).isoformat(), **payload}
    path = out_dir / f"{run_id}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[{stage}] sidecar -> {path.relative_to(ROOT)}")
    return path


def _say(stage: str, msg: str) -> None:
    print(f"[{stage}] {msg}")


# -------------------------------------------------------------------------- probe
def cmd_probe(args) -> int:
    t0 = time.monotonic()
    rid = _run_id("probe")
    results = probes.run_probes(skip_live=args.offline)
    report = probes.write_report(results)
    for r in results:
        _say("probe", f"{r.probe_id} {r.status:8s} {r.title}" + (f" — {r.error[:90]}" if r.error else ""))
    _say("probe", f"report -> {report.relative_to(ROOT)}")
    _sidecar("probe", rid, {"results": [r.status for r in results],
                            "report": str(report.relative_to(ROOT))}, t0)
    blocked = sum(r.status in ("BLOCKED", "FAIL") for r in results)
    return 0 if blocked == 0 else 2          # 2 = degraded-mode signal, not a crash


# --------------------------------------------------------------------------- pull
_YF_FIELDS = {"Adj Close": "adjclose", "Close": "close", "Volume": "volume",
              "Dividends": "dividends", "Stock Splits": "splits"}


def _pull_yfinance(tickers: list[str], start: str, end: str, rid: str,
                   journal: acquire.PullJournal, governor: acquire.RateGovernor) -> list:
    frozen = []
    for chunk in acquire.chunk_tickers(tickers):
        params = {"tickers": chunk, "start": start, "end": end}
        cid = acquire.chunk_id("yfinance", "download_ohlcv_actions", params)
        if journal.done(cid):
            continue
        governor.wait()
        requested = datetime.now(timezone.utc).isoformat()
        df = acquire.with_backoff(lambda p=params: acquire.fetch_yfinance(p))
        if df is None or len(df) == 0:
            journal.mark(cid, "skipped_empty", label=",".join(chunk))
            continue
        relpaths = []
        label = hashlib.sha256(",".join(chunk).encode()).hexdigest()[:8]
        for field, suffix in _YF_FIELDS.items():
            if field not in df.columns.get_level_values(0):
                continue
            sub = df[field]
            if isinstance(sub, pd.Series):
                sub = sub.to_frame(chunk[0])
            art = freeze(sub, "raw", f"yf_{suffix}_{label}.csv",
                         acquire.provenance("yfinance", "download_ohlcv_actions",
                                            {**params, "field": field}, requested),
                         run_id=rid)
            relpaths.append(art.relpath)
            frozen.append(art)
        journal.mark(cid, "frozen", label=",".join(chunk), relpaths=relpaths)
    return frozen


def _pull_refinitiv(tickers: list[str], start: str, end: str, rid: str,
                    journal: acquire.PullJournal, governor: acquire.RateGovernor) -> list:
    rd = acquire.open_refinitiv_session()        # raises EntitlementError when blocked
    frozen = []
    for chunk in acquire.chunk_tickers(tickers):
        params = {"universe": chunk, "fields": ["TR.TotalReturn"], "start": start, "end": end}
        cid = acquire.chunk_id("refinitiv", "get_history.TR.TotalReturn", params)
        if journal.done(cid):
            continue
        governor.wait()
        requested = datetime.now(timezone.utc).isoformat()
        df = acquire.with_backoff(
            lambda p=params: acquire.fetch_refinitiv_history(rd, p["universe"], p["fields"],
                                                             p["start"], p["end"]))
        if df is None or len(df) == 0:
            journal.mark(cid, "skipped_empty", label=",".join(chunk))
            continue
        label = hashlib.sha256(",".join(chunk).encode()).hexdigest()[:8]
        art = freeze(df, "raw", f"rf_totalreturn_{label}.csv",
                     acquire.provenance("refinitiv", "get_history.TR.TotalReturn",
                                        params, requested), run_id=rid)
        journal.mark(cid, "frozen", label=",".join(chunk), relpaths=[art.relpath])
        frozen.append(art)
    return frozen


def cmd_pull(args) -> int:
    t0 = time.monotonic()
    rid = _run_id("pull")
    window = get("data.window")
    start, end = window["start"], window["end"]
    tickers = get("data.pilot.tickers") if args.pilot else \
        (args.tickers.split(",") if args.tickers else get("data.pilot.tickers"))
    vendors = args.vendors.split(",")
    journal = acquire.PullJournal(pull_id=f"{'pilot' if args.pilot else 'custom'}_{'_'.join(sorted(vendors))}")
    governor = acquire.RateGovernor()
    outcome: dict = {"tickers": tickers, "window": [start, end], "vendors": {}}

    if "yfinance" in vendors:
        try:
            arts = _pull_yfinance(tickers, start, end, rid, journal, governor)
            outcome["vendors"]["yfinance"] = {"frozen": len(arts)}
        except Exception as e:                                       # noqa: BLE001 — vendor boundary
            outcome["vendors"]["yfinance"] = {"error": f"{type(e).__name__}: {e}"}
    if "fred" in vendors:
        try:
            params = {"series": get("data.macro_fred"), "start": start, "end": end}
            df = acquire.with_backoff(lambda: acquire.fetch_fred(params))
            art = freeze(df, "raw", "fred_macro.csv",
                         acquire.provenance("fred", "fred_series", params,
                                            datetime.now(timezone.utc).isoformat()), run_id=rid)
            outcome["vendors"]["fred"] = {"frozen": 1, "rows": art.rows}
        except Exception as e:                                       # noqa: BLE001
            outcome["vendors"]["fred"] = {"error": f"{type(e).__name__}: {e}"}
    if "french" in vendors:
        ok, errs = 0, []
        for dataset in get("data.factors_french"):
            try:
                params = {"dataset": dataset, "start": start}
                df = acquire.with_backoff(lambda p=params: acquire.fetch_french(p))
                freeze(df, "raw", f"french_{dataset.replace('/', '_')}.csv",
                       acquire.provenance("kenfrench", "famafrench", params,
                                          datetime.now(timezone.utc).isoformat()), run_id=rid)
                ok += 1
            except Exception as e:                                   # noqa: BLE001
                errs.append(f"{dataset}: {type(e).__name__}: {e}")
        outcome["vendors"]["french"] = {"frozen": ok, "errors": errs}
    if "refinitiv" in vendors:
        try:
            arts = _pull_refinitiv(tickers, start, end, rid, journal, governor)
            outcome["vendors"]["refinitiv"] = {"frozen": len(arts)}
        except acquire.EntitlementError as e:
            outcome["vendors"]["refinitiv"] = {"blocked": str(e)}
            _say("pull", f"refinitiv BLOCKED — {e}")
    outcome["journal"] = journal.summary()
    _sidecar("pull", rid, outcome, t0)
    return 0


# -------------------------------------------------------------- staging primitives
def _collect_raw(prefix: str) -> pd.DataFrame | None:
    """Concatenate all raw artifacts whose name starts with `prefix` (column-wise),
    each read CHECKSUM-VERIFIED (R4)."""
    frames = []
    for e in manifest_entries():
        if e["layer"] == "raw" and e["name"].startswith(prefix):
            df = read_verified(e["relpath"])
            df.index = pd.to_datetime(df.index)
            frames.append(df)
    if not frames:
        return None
    out = pd.concat(frames, axis=1)
    return out.loc[:, ~out.columns.duplicated()].sort_index()


def cmd_build(args) -> int:
    t0 = time.monotonic()
    rid = _run_id("build")
    suffix = args.suffix
    window = get("data.window")
    counts: dict = {}

    adj = _collect_raw("yf_adjclose_")
    if adj is None:
        _say("build", "no raw yfinance pulls found — run `pull` first")
        return 1
    sessions = validate.trading_sessions(window["start"],
                                         min(pd.Timestamp(window["end"]),
                                             adj.index.max()).date().isoformat())

    # ---- staged: returns (shadow vendor), aligned + validated
    adj_aligned, align_rep = validate.align_to_calendar(adj, sessions)
    rets_yf = adj_aligned.pct_change(fill_method=None)
    schema = validate.TableSchema("staged_returns", {c: validate.ColumnSpec("float")
                                                     for c in rets_yf.columns})
    vrep = validate.validate_frame(rets_yf, schema)
    flags, missing_rep = validate.classify_missing(rets_yf, sessions, raw_index=adj.index)
    staged_inputs = [e for e in manifest_entries() if e["layer"] == "raw"]
    art_stage = freeze(rets_yf, "staged", f"staged_returns_yfinance{suffix}.parquet",
                       {"vendor": "yfinance", "stage": "cli.build/staging",
                        "alignment": align_rep, "validation": vrep.violations,
                        "missing": missing_rep.as_dict()}, fmt="parquet", run_id=rid)
    record_lineage(art_stage, staged_inputs, "staging", {"calendar": "XNYS"})
    counts["staged"] = {"rows": art_stage.rows, "cols": art_stage.cols,
                        "violations": vrep.violations, "missing": missing_rep.as_dict()}

    # ---- staged: refinitiv returns when pulled (authority path)
    rf = _collect_raw("rf_totalreturn_")
    frames_by_vendor = {"yfinance": rets_yf}
    if rf is not None:
        rf_aligned, _ = validate.align_to_calendar(rf.apply(pd.to_numeric, errors="coerce"), sessions)
        rf_rets = rf_aligned / 100.0 if rf_aligned.abs().median().median() > 1 else rf_aligned
        frames_by_vendor["refinitiv"] = rf_rets
        freeze(rf_rets, "staged", f"staged_returns_refinitiv{suffix}.parquet",
               {"vendor": "refinitiv", "stage": "cli.build/staging"}, fmt="parquet", run_id=rid)

    # ---- integrity flags + quarantine (flag-only; values untouched)
    # Ince-Porter price screens need ACTUAL traded prices. Yahoo's Close is
    # split-adjusted even with auto_adjust=False, so the traded price is
    # RECONSTRUCTED as close x prod(future split ratios) — see
    # integrity.reconstruct_unadjusted_close (vendor subtlety, data chapter).
    close_raw = _collect_raw("yf_close_")
    splits_raw = _collect_raw("yf_splits_")
    if close_raw is not None:
        traded = pd.DataFrame({
            c: integrity.reconstruct_unadjusted_close(
                close_raw[c], splits_raw[c] if splits_raw is not None and c in splits_raw.columns else None)
            for c in close_raw.columns})
        screen_aligned, _ = validate.align_to_calendar(traded, sessions)
    else:
        screen_aligned = adj_aligned
    ip_frames = {f"ince_porter:{c}": integrity.ince_porter_flags(screen_aligned[c], rets_yf[c])
                 for c in rets_yf.columns if c in screen_aligned.columns}
    ip_frames["extreme_context"] = integrity.classify_extreme_days(rets_yf)
    divs = _collect_raw("yf_dividends_")
    for c in rets_yf.columns:
        s_col = splits_raw[c] if splits_raw is not None and c in splits_raw.columns else None
        ip_frames[f"split_artifacts:{c}"] = integrity.split_artifact_flags(rets_yf[c], s_col)
    quarantine = integrity.assemble_quarantine(ip_frames)
    counts["integrity"] = {"quarantine_rows": int(len(quarantine)),
                           "real_tails_preserved": int((ip_frames["extreme_context"]["class"] == "REAL_TAIL").sum())
                           if not ip_frames["extreme_context"].empty else 0}
    if not quarantine.empty:
        freeze(quarantine, "clean", f"quarantine{suffix}.csv",
               {"vendor": "derived", "stage": "cli.build/integrity",
                "note": "reason-coded exclusion queue; original values preserved"},
               run_id=rid)

    # ---- reconciliation + authority merge -> clean
    recon_summary = None
    if "refinitiv" in frames_by_vendor:
        rep = reconcile_full.reconcile_returns(frames_by_vendor["refinitiv"], rets_yf,
                                               dividends=divs, splits=splits_raw)
        (ROOT / "reports" / f"vendor_reconciliation_full{suffix}.md").write_text(
            "# Full reconciliation\n\n" + rep.to_markdown() + "\n")
        recon_summary = [r.__dict__ for r in rep.rows]
    merged, decision = reconcile_full.authority_merge(frames_by_vendor, "total_return")
    art_clean = freeze(merged, "clean", f"clean_returns{suffix}.parquet",
                       {"vendor": "merged", "stage": "cli.build/authority_merge",
                        "decision": decision}, fmt="parquet", run_id=rid)
    record_lineage(art_clean, [art_stage], "authority_merge", decision)
    counts["clean"] = {"rows": art_clean.rows, "cols": art_clean.cols, "authority": decision}

    # ---- gold
    fred = _collect_raw("fred_macro")
    if fred is None or "VIXCLS" not in fred.columns:
        _say("build", "no FRED VIXCLS in raw — gold needs the macro pull; stopping after clean")
        _sidecar("build", rid, {"counts": counts, "stopped": "missing VIX"}, t0)
        return 1
    vix = fred["VIXCLS"].astype(float)
    gold = build_gold(merged, vix, sessions, run_id=rid,
                      input_artifacts=[art_clean], suffix=suffix)
    counts["gold"] = {k: a.relpath for k, a in gold["artifacts"].items()}
    counts["splits"] = {"dev_val_sessions": gold["splits"]["development"]["n_validation_sessions"],
                        "walk_forward_folds": len(gold["splits"]["walk_forward"])}
    if recon_summary:
        counts["reconciliation"] = recon_summary
    _sidecar("build", rid, {"counts": counts}, t0)
    _say("build", f"clean {art_clean.rows}x{art_clean.cols}; gold artifacts: {sorted(gold['artifacts'])}")
    return 0


# ---------------------------------------------------------------- validate / recon
def cmd_validate(args) -> int:
    t0 = time.monotonic()
    rid = _run_id("validate")
    reports = []
    for e in manifest_entries():
        if e["layer"] != "raw":
            continue
        try:
            read_verified(e["relpath"])
            reports.append({"artifact": e["relpath"], "checksum": "OK"})
        except ValueError as err:
            reports.append({"artifact": e["relpath"], "checksum": f"FAIL: {err}"})
    bad = [r for r in reports if r["checksum"] != "OK"]
    _say("validate", f"{len(reports)} raw artifacts verified, {len(bad)} failures")
    _sidecar("validate", rid, {"verified": len(reports), "failures": bad}, t0)
    return 0 if not bad else 1


def cmd_reconcile(args) -> int:
    t0 = time.monotonic()
    rid = _run_id("reconcile")
    yf_e = find_entry(f"staged_returns_yfinance{args.suffix}.parquet")
    rf_e = find_entry(f"staged_returns_refinitiv{args.suffix}.parquet")
    if not (yf_e and rf_e):
        _say("reconcile", "need BOTH staged vendor panels — refinitiv pending entitlement "
                          "(yfinance-only build proceeds via authority fallback)")
        _sidecar("reconcile", rid, {"skipped": "second vendor missing"}, t0)
        return 2
    rep = reconcile_full.reconcile_returns(read_verified(rf_e["relpath"]),
                                           read_verified(yf_e["relpath"]))
    out = ROOT / "reports" / f"vendor_reconciliation_full{args.suffix}.md"
    out.write_text("# Full reconciliation\n\n" + rep.to_markdown() + "\n")
    _say("reconcile", f"-> {out.relative_to(ROOT)}")
    _sidecar("reconcile", rid, {"rows": len(rep.rows),
                                "quarantined": 0 if rep.quarantine is None else len(rep.quarantine)}, t0)
    return 0


def cmd_eda(args) -> int:
    t0 = time.monotonic()
    rid = _run_id("eda")
    e = find_entry(f"clean_returns{args.suffix}.parquet")
    if e is None:
        _say("eda", "no clean returns artifact — run `build` first")
        return 1
    rets = read_verified(e["relpath"])
    rets.index = pd.to_datetime(rets.index)
    fred_e = find_entry("fred_macro.csv")
    vix = None
    if fred_e:
        fred = read_verified(fred_e["relpath"])
        fred.index = pd.to_datetime(fred.index)
        vix = fred.get("VIXCLS")
    out = eda.run_eda(rets, rets.mean(axis=1), vix)
    cov = quality.coverage_matrix(rets)
    score = quality.scoreboard(cov)
    gold_paths = [x["relpath"] for x in manifest_entries() if x["layer"] == "gold"]
    qpaths = quality.write_quality_reports(cov, score, gold_paths, numbers={
        "eda": f"median excess kurtosis {np.median([m for m in eda.moments_table(rets)['excess_kurtosis']]):.1f}"
               if len(rets.columns) else "⟨TBD⟩"})
    _say("eda", f"report -> {out.relative_to(ROOT)}; quality -> {qpaths['scoreboard'].relative_to(ROOT)}")
    _sidecar("eda", rid, {"eda_report": str(out.relative_to(ROOT)),
                          "quality_outputs": {k: str(v) for k, v in qpaths.items()}}, t0)
    return 0


def cmd_status(args) -> int:
    entries = manifest_entries()
    by_layer: dict[str, list] = {}
    for e in entries:
        by_layer.setdefault(e["layer"], []).append(e)
    print("=== data platform status ===")
    for layer in ("raw", "staged", "clean", "gold"):
        arts = by_layer.get(layer, [])
        rows = sum(a["rows"] for a in arts)
        print(f"{layer:7s} {len(arts):3d} artifacts  {rows:>10,} rows")
        for a in arts[-3:]:
            print(f"        … {a['name']}  {a['sha256'][:12]}  {a['rows']}x{a['cols']}")
    jdir = ROOT / get("data.platform.journal_dir")
    if jdir.exists():
        for j in sorted(jdir.glob("*.jsonl")):
            journal = acquire.PullJournal(pull_id=j.stem)
            print(f"journal {j.stem}: {journal.summary()}")
    ev = ROOT / get("data.platform.evidence_dir") / "entitlement_report.md"
    print(f"entitlement report: {'present' if ev.exists() else 'NOT RUN'}")
    return 0


def cmd_pull_universe(args) -> int:
    """A1-A5 universe acquisition (ADR-019). Default --dry-run reports the plan WITHOUT
    opening a session; --live attempts the entitled pull and records EntitlementError into
    the sidecar if scopes are still empty (never fabricates)."""
    t0 = time.monotonic()
    rid = _run_id("pull_universe")
    from .pull_universe import acquire_universe
    try:
        out = acquire_universe(dry_run=not args.live)
        outcome = {"result": out}
    except acquire.EntitlementError as e:
        outcome = {"blocked": str(e)}
        _say("pull-universe", f"BLOCKED — {e}")
    _sidecar("pull_universe", rid, outcome, t0)
    if "result" in outcome and outcome["result"].get("mode", "").startswith("dry"):
        _say("pull-universe", f"DRY RUN plan: {outcome['result']['A1_membership_refinitiv_requests']} "
                              "monthly membership requests + A2/A3/A4/A5 fields "
                              "(run `--live` once entitlement lands)")
    return 0


def cmd_build_universe(args) -> int:
    """Assemble the entitled pulls into the PIT research panel (suffix _univ)."""
    t0 = time.monotonic()
    rid = _run_id("build_universe")
    from .build_universe import build_universe
    try:
        out = build_universe(suffix=args.suffix)
        _say("build-universe", f"clean {out['n_sessions']}x{out['n_names']}; gold: {sorted(out['gold'])}")
        _sidecar("build_universe", rid, {"result": out}, t0)
        return 0
    except FileNotFoundError as e:
        _say("build-universe", f"BLOCKED — {e}")
        _sidecar("build_universe", rid, {"blocked": str(e)}, t0)
        return 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="src.data.cli", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("probe")
    sp.add_argument("--offline", action="store_true")
    sp = sub.add_parser("pull")
    sp.add_argument("--pilot", action="store_true")
    sp.add_argument("--tickers", default="")
    sp.add_argument("--vendors", default="yfinance,fred,french,refinitiv")
    for name in ("build", "reconcile", "eda"):
        sp = sub.add_parser(name)
        sp.add_argument("--suffix", default="_pilot")
    sp = sub.add_parser("pull-universe")
    sp.add_argument("--live", action="store_true", help="attempt the entitled pull (default: dry-run plan)")
    sp = sub.add_parser("build-universe")
    sp.add_argument("--suffix", default="_univ")
    sub.add_parser("validate")
    sub.add_parser("status")
    args = p.parse_args(argv)
    return {"probe": cmd_probe, "pull": cmd_pull, "build": cmd_build,
            "validate": cmd_validate, "reconcile": cmd_reconcile,
            "eda": cmd_eda, "status": cmd_status,
            "pull-universe": cmd_pull_universe,
            "build-universe": cmd_build_universe}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
