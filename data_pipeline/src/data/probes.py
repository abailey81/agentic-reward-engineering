"""Automated entitlement probes (plan W3) — runs the vendor-entitlement
checklist live and writes docs/evidence/entitlement_report.md.

Probe design: every probe is guarded (one failing entitlement never blocks the
others), timed, and captures EVIDENCE (row counts, date ranges, tiny samples — never
bulk data, never secrets). Failures record the exact exception text: a with its
reason is the deliverable, not an embarrassment (the degradation path is designed —
yfinance shadow + quarantined gap + escalation email).

Dead-RIC convention probed: '^' suffix + month-letter/year, e.g. LEH.N^I08 = Lehman
delisted September 2008 (RIC month codes A..L = Jan..Dec).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..config import get
from .acquire import (
    EntitlementError,
    capture_field_definitions,
    load_env,
    open_datastream,
    open_refinitiv_session,
)

ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class ProbeResult:
    probe_id: str
    title: str
    status: str                      # PASS | FAIL | BLOCKED | MANUAL
    latency_s: float = 0.0
    evidence: dict = field(default_factory=dict)
    error: str = ""


def _run(probe_id: str, title: str, fn) -> ProbeResult:
    t0 = time.monotonic()
    try:
        evidence = fn()
        return ProbeResult(probe_id, title, "PASS", round(time.monotonic() - t0, 2),
                           evidence or {})
    except EntitlementError as e:
        return ProbeResult(probe_id, title, "BLOCKED", round(time.monotonic() - t0, 2),
                           {}, str(e))
    except Exception as e:                                          # noqa: BLE001 — probe boundary
        return ProbeResult(probe_id, title, "FAIL", round(time.monotonic() - t0, 2),
                           {}, f"{type(e).__name__}: {e}")


def _frame_evidence(df, sample_cols: int = 3, sample_rows: int = 3) -> dict:
    import pandas as pd
    if df is None:
        return {"rows": 0}
    if isinstance(df, pd.Series):
        df = df.to_frame()
    ev = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
    try:
        sub = df.iloc[:sample_rows, :sample_cols].astype(str)
        ev["sample"] = {str(c): {str(k): v for k, v in col.items()}
                        for c, col in sub.to_dict().items()}        # JSON-safe keys
        if hasattr(df.index, "min") and len(df) > 0:
            ev["index_range"] = [str(df.index.min()), str(df.index.max())]
    except Exception:
        pass
    return ev


def run_probes(skip_live: bool = False) -> list[ProbeResult]:
    """Execute the checklist. `skip_live=True` produces the offline skeleton
    (every live probe BLOCKED with the environment reason) — used by tests."""
    results: list[ProbeResult] = []
    env_loaded = load_env()
    pre = {"env_file": (ROOT / ".env").exists() or (ROOT.parent / ".env").exists(),
           "env_keys_loaded": sorted(env_loaded)}

    rd = None
    if skip_live:
        results.append(ProbeResult("P0", "Session preflight", "BLOCKED", 0.0, pre,
                                   "skip_live requested (offline mode)"))
    else:
        def _open():
            nonlocal rd
            rd = open_refinitiv_session()
            return {**pre, "session": "open (platform grant or desktop proxy)"}
        results.append(_run("P0", "Refinitiv session (platform/desktop)", _open))

    live = rd is not None

    # 1. chain resolves
    results.append(_run("P1", "0#.SPX chain resolves (~503 rows)", (
        lambda: _frame_evidence(rd.get_data("0#.SPX", ["TR.CommonName"]))) if live
        else (lambda: (_ for _ in ()).throw(EntitlementError("no session")))))
    # 2/3. PIT membership via JOINER/LEAVER EVENTS — content-validated against known
    # truth (ADR-020: snapshot/SDate forms silently return the CURRENT chain on this
    # route; a probe that only counts rows would false-PASS, so these assert content).
    _JL = ["TR.IndexJLConstituentRIC", "TR.IndexJLConstituentName",
           "TR.IndexJLConstituentChangeDate"]

    def _jl_probe(sdate: str, edate: str, must_contain_prefix: str):
        df = rd.get_data(".SPX", _JL, {"SDate": sdate, "EDate": edate, "IC": "L"})
        ev = _frame_evidence(df)
        rics = df[[c for c in df.columns if "RIC" in str(c)][0]].dropna().astype(str)
        if not rics.str.startswith(must_contain_prefix).any():
            raise RuntimeError(f"JL events {sdate}..{edate} lack known leaver "
                               f"'{must_contain_prefix}*' — content validation FAILED")
        ev["known_truth"] = f"{must_contain_prefix}* leaver present"
        return ev
    results.append(_run("P2", "PIT membership events 2018 (content-validated)", (
        lambda: _jl_probe("2018-01-01", "2018-12-31", "GGP")) if live
        else (lambda: (_ for _ in ()).throw(EntitlementError("no session")))))
    results.append(_run("P3", "PIT membership events PRE-2016 (content: Lehman leaver 2008)", (
        lambda: _jl_probe("2008-01-01", "2008-12-31", "LEH")) if live
        else (lambda: (_ for _ in ()).throw(EntitlementError("no session")))))
    # 4. Datastream LS&PCOMP list
    def _dsws_probe():
        ds = open_datastream()
        df = ds.get_data(tickers="LS&PCOMP0110", fields=["NAME"], kind=0)
        return _frame_evidence(df)
    results.append(_run("P4", "Datastream list LS&PCOMP0110 (DSWS)", _dsws_probe))
    # 5. RI continuity on the GE index-exit window
    exit_date = str(get("data.pilot.ge_index_exit"))
    results.append(_run("P5", f"GE total-return continuity around {exit_date}", (
        lambda: _frame_evidence(rd.get_history("GE", ["TR.TotalReturn"],
                                               start="2018-06-01", end="2018-07-15"))) if live
        else (lambda: (_ for _ in ()).throw(EntitlementError("no session")))))
    # 6. dead-RIC probe (Lehman, delisted Sep-2008 -> ^I08)
    results.append(_run("P6", "Dead RIC LEH.N^I08 history (delisted coverage)", (
        lambda: _frame_evidence(rd.get_history("LEH.N^I08", ["TR.PriceClose"],
                                               start="2008-01-01", end="2008-12-31"))) if live
        else (lambda: (_ for _ in ()).throw(EntitlementError("no session")))))
    # 7. field definitions: RI day-count + currency
    results.append(_run("P7", "Field definitions: RI day-count & currency",
                        lambda: capture_field_definitions(rd)))
    if results[-1].status == "PASS" and \
            results[-1].evidence.get("RI_day_count", {}).get("status") == "REQUIRES_MANUAL_CONFIRMATION":
        results[-1].status = "MANUAL"
    # 8. scope census across product families — distinguishes "wrong endpoint" from
    # "token has no scopes" (empty set across ALL families = app-key permissions or
    # seat licence issue; evidence for the LSEG/UCL conversation)
    def _scope_census():
        if rd is None:
            raise EntitlementError("no session")
        out = {}
        probes_by_family = {
            "search": lambda: rd.discovery.search(query="IBM", top=1),
            "news": lambda: rd.news.get_headlines("Apple", count=1),
            "pricing_snapshot": lambda: rd.get_data(universe=["IBM"], fields=["BID"]),
        }
        for family, fn in probes_by_family.items():
            try:
                fn()
                out[family] = "IN SCOPE"
            except Exception as e:                                  # noqa: BLE001 — census boundary
                msg = str(e)
                out[family] = ("EMPTY_SCOPE_SET" if "Available scopes: {}" in msg
                               else f"{type(e).__name__}: {msg[:120]}")
        if all(v == "EMPTY_SCOPE_SET" for v in out.values()):
            raise EntitlementError(
                "token carries ZERO scopes for every product family — app key minted "
                "without API permissions, or the seat has no server-side RDP data licence. "
                "App-free fix attempt: mint a new key at the WEB App Key Generator "
                "(apps.cp.thomsonreuters.com/apps/AppkeyGenerator) with the 'EDP API' box "
                f"ticked, put it in .env as REFINITIV_APP_KEY, re-run probe. census={out}"
            )
        return out
    results.append(_run("P8", "RDP scope census (search/news/pricing)", _scope_census))

    if rd is not None:
        try:
            import refinitiv.data as rdlib
            rdlib.close_session()
        except Exception:
            pass
    return results


ESCALATION_EMAIL = """Subject: Datastream/DSWS service enablement for MSc dissertation (supervisor: Dr R. Okhrati)

Dear Library Data Services — for my IFTE0008 dissertation I require Datastream constituent lists
(`LS&PCOMP` + monthly `LS&PCOMPmmyy`, 2005–2016) and the `RI` total-return datatype for delisted/exited
S&P 500 members. My existing LSEG credentials AUTHENTICATE against the Datastream Web Service but the
account is "not entitled to ClientApi service" — could that service flag (DSWS/ClientApi) be enabled on
my existing account, or access granted via the Datastream Excel add-in? Separately, my LSEG platform
token carries no API scopes — if RDP data scopes (datagrid / historical-pricing) can be attached, that
would also unblock the build. Could you also confirm whether UCL provides WRDS/CRSP access (MSP500LIST)?
Timeline is tight (data build starts mid-June). Thank you — Tamer Atesyakar (MSc B&DF, IFT)."""


def write_report(results: list[ProbeResult], out_path: Path | None = None) -> Path:
    out_dir = ROOT / get("data.platform.evidence_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_path or out_dir / "entitlement_report.md"
    icon = {"PASS": "[PASS]", "FAIL": "[FAIL]", "BLOCKED": "[BLOCKED]", "MANUAL": "[MANUAL]"}
    pre2016 = next((r for r in results if r.probe_id == "P3"), None)
    dsws = next((r for r in results if r.probe_id == "P4"), None)
    escalate = (pre2016 is None or pre2016.status != "PASS") and \
               (dsws is None or dsws.status != "PASS")

    lines = [
        "# Entitlement probe report (automated — plan W3)",
        f"Generated {datetime.now(timezone.utc).isoformat()} by `python -m src.data.cli probe`.",
        "Vendor-entitlement checklist. Secrets are never recorded here.",
        "",
        "| # | Probe | Status | Latency | Evidence / error |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        detail = json.dumps(r.evidence, default=str)[:300] if r.evidence else r.error[:300]
        lines.append(f"| {r.probe_id} | {r.title} | {icon.get(r.status, '?')} {r.status} "
                     f"| {r.latency_s}s | `{detail}` |")
    lines += ["", "## Interpretation", ""]
    if escalate:
        lines += [
            "**Pre-2016 PIT membership is NOT currently provable on this machine** (P3 and P4 "
            "both non-PASS). Per the degradation protocol: the build continues on the entitled "
            "span plus the yfinance shadow; the 2005–2016 membership gap is QUARANTINED (no "
            "fabrication, R4); the pipeline back-fills cleanly when access lands (membership "
            "loaders are vendor-agnostic over the vault).",
            "",
            "### Escalation email (send now)",
            "```text", ESCALATION_EMAIL, "```",
        ]
    else:
        lines += ["Pre-2016 membership path verified — proceed with the full PIT build."]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "entitlement_probes.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2, default=str), encoding="utf-8")
    return out_path
