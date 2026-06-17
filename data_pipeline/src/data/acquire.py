"""Acquisition engine (stage 1): rate-limit governor, exponential backoff with
jitter, idempotent resumable journaled pulls, chunking, provenance capture, and
vendor adapters (Refinitiv / Datastream-DSWS / yfinance shadow / FRED / Ken French).

Engineering contract
    * Every network call passes through `RateGovernor.wait()` (vendor courtesy /
      rate-limit compliance) and `with_backoff` (exponential, jittered, capped).
    * Every pull is journaled per chunk (`PullJournal`): a dropped session RESUMES —
      frozen chunks are never re-pulled, never re-frozen (write-once, R4).
    * Every frozen artifact carries provenance: vendor, exact query/params, request &
      receive timestamps, row hash, library versions (vault sidecar).
    * Vendor fetchers are INJECTABLE callables — unit tests run the full engine with
      fakes; nothing in tests touches the network.

Secrets: keys come from `.env` (gitignored) via `load_env`; they are never logged,
never embedded in provenance, never frozen.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

import pandas as pd

from ..config import get
from .vault import FrozenArtifact, freeze

ROOT = Path(__file__).resolve().parent.parent.parent


# ------------------------------------------------------------------ env & provenance
def load_env(path: Path | None = None) -> dict[str, str]:
    """Minimal .env loader (no extra dependency): KEY=VALUE lines, # comments.
    Loads into os.environ WITHOUT overriding pre-set variables. Values never logged.

    Post-unification (ADR-022) the gitignored .env lives at the UNIFIED repo root
    (the parent of data_pipeline/), not inside data_pipeline/. Search ROOT then ROOT.parent
    so the probe/pulls find the creds either way (the desktop-fallback bug, 2026-06-19)."""
    if path is None:
        path = next((c for c in (ROOT / ".env", ROOT.parent / ".env") if c.exists()),
                    ROOT / ".env")
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded[key] = "<set>"
    return loaded


def _lib_versions() -> dict[str, str]:
    out = {}
    for mod in ("pandas", "yfinance", "refinitiv.data", "DatastreamPy"):
        try:
            m = __import__(mod.split(".")[0])
            out[mod.split(".")[0]] = getattr(m, "__version__", "?")
        except ImportError:
            pass
    return out


def provenance(vendor: str, query: str, params: dict, requested_utc: str) -> dict:
    return {
        "vendor": vendor, "query": query, "params": params,
        "requested_utc": requested_utc,
        "received_utc": datetime.now(timezone.utc).isoformat(),
        "library_versions": _lib_versions(),
    }


# --------------------------------------------------------------- governor & backoff
@dataclass
class RateGovernor:
    """Spaces LAUNCHES to at most `requests_per_minute` — thread-safe: with N parallel
    workers, requests still leave the box at the configured global rate (the lock
    serialises the spacing sleep), while RESPONSES overlap. Injectable clock for tests."""

    requests_per_minute: float | None = None
    clock: Callable[[], float] = time.monotonic
    sleeper: Callable[[float], None] = time.sleep
    _last: float = field(default=-1e9, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.requests_per_minute is None:
            self.requests_per_minute = float(get("data.platform.rate_limit.requests_per_minute"))

    @property
    def min_interval(self) -> float:
        return 60.0 / float(self.requests_per_minute)

    def wait(self) -> float:
        with self._lock:                       # one launch at a time, globally spaced
            now = self.clock()
            delay = max(0.0, self._last + self.min_interval - now)
            if delay > 0:
                self.sleeper(delay)
            self._last = self.clock()
            return delay


def with_backoff(
    fn: Callable[[], object],
    max_retries: int | None = None,
    base_s: float | None = None,
    cap_s: float | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
) -> object:
    """Run `fn` with exponential backoff + full jitter (delay ~ U[0, min(cap, base*2^k)]).
    Re-raises the final failure — acquisition never swallows errors silently."""
    cfg = get("data.platform.rate_limit")
    max_retries = int(max_retries if max_retries is not None else cfg["max_retries"])
    base_s = float(base_s if base_s is not None else cfg["backoff_base_s"])
    cap_s = float(cap_s if cap_s is not None else cfg["backoff_cap_s"])
    rng = rng or random.Random()
    last: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except retry_on as e:                      # noqa: PERF203 — retry loop by design
            last = e
            if attempt == max_retries:
                break
            sleeper(rng.uniform(0.0, min(cap_s, base_s * (2.0 ** attempt))))
    raise RuntimeError(f"acquisition failed after {max_retries + 1} attempts: {last}") from last


# ------------------------------------------------------------------------- chunking
def chunk_tickers(tickers: Sequence[str], per_request: int | None = None) -> list[list[str]]:
    n = int(per_request if per_request is not None else get("data.platform.chunking.tickers_per_request"))
    return [list(tickers[i: i + n]) for i in range(0, len(tickers), n)]


def chunk_dates(start: str, end: str, days: int | None = None) -> list[tuple[str, str]]:
    n = int(days if days is not None else get("data.platform.chunking.days_per_request"))
    spans, cur, stop = [], pd.Timestamp(start), pd.Timestamp(end)
    while cur <= stop:
        nxt = min(cur + pd.Timedelta(days=n - 1), stop)
        spans.append((cur.date().isoformat(), nxt.date().isoformat()))
        cur = nxt + pd.Timedelta(days=1)
    return spans


def chunk_id(vendor: str, query: str, params: dict) -> str:
    blob = json.dumps({"vendor": vendor, "query": query, "params": params},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# -------------------------------------------------------------------------- journal
@dataclass
class PullJournal:
    """Append-only per-pull chunk ledger -> resumable, idempotent acquisition."""

    pull_id: str
    journal_dir: Path | None = None

    def __post_init__(self) -> None:
        self.journal_dir = Path(self.journal_dir or ROOT / get("data.platform.journal_dir"))
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.journal_dir / f"{self.pull_id}.jsonl"
        self._lock = threading.Lock()          # parallel workers append concurrently

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def done(self, cid: str) -> bool:
        return any(e["chunk_id"] == cid and e["status"] == "frozen" for e in self.entries())

    def mark(self, cid: str, status: str, **extra) -> None:
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"chunk_id": cid, "status": status,
                                "ts_utc": datetime.now(timezone.utc).isoformat(),
                                **extra}, sort_keys=True, default=str) + "\n")

    def summary(self) -> dict:
        ent = self.entries()
        return {"pull_id": self.pull_id,
                "frozen": sum(e["status"] == "frozen" for e in ent),
                "failed": sum(e["status"] == "failed" for e in ent),
                "skipped_empty": sum(e["status"] == "skipped_empty" for e in ent)}


def journaled_pull(
    vendor: str,
    query: str,
    chunks: Iterable[tuple[str, dict]],
    fetch: Callable[[dict], pd.DataFrame],
    journal: PullJournal,
    name_for: Callable[[str, dict], str],
    layer: str = "raw",
    fmt: str = "csv",
    governor: RateGovernor | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    workers: int | None = None,
) -> list[FrozenArtifact]:
    """The acquisition core: for each (label, params) chunk — skip if journaled
    frozen; otherwise govern -> fetch with backoff -> freeze with provenance ->
    journal. Empty frames journal as `skipped_empty` (counted, never fabricated).

    `workers` > 1 runs chunks on a thread pool: launches stay globally spaced by the
    (thread-safe) governor — the configured requests_per_minute is RESPECTED — while
    response latency overlaps. Vault freezes and journal appends are lock-serialised.
    Failures journal per-chunk; the first failure is re-raised after in-flight chunks
    settle (resume re-attempts exactly the non-frozen chunks)."""
    governor = governor or RateGovernor(sleeper=sleeper)
    if workers is None:
        workers = int(get("data.platform.rate_limit.workers", 1))
    todo = [(label, params) for label, params in chunks
            if not journal.done(chunk_id(vendor, query, params))]
    frozen: list[FrozenArtifact] = []
    failures: list[BaseException] = []

    def _one(label: str, params: dict) -> FrozenArtifact | None:
        cid = chunk_id(vendor, query, params)
        governor.wait()
        requested = datetime.now(timezone.utc).isoformat()
        try:
            df = with_backoff(lambda p=params: fetch(p), sleeper=sleeper)
        except RuntimeError as e:
            journal.mark(cid, "failed", label=label, error=str(e)[:500])
            raise
        if df is None or len(df) == 0:
            journal.mark(cid, "skipped_empty", label=label)
            return None
        art = freeze(df, layer, name_for(label, params),
                     provenance(vendor, query, params, requested),
                     fmt=fmt, run_id=journal.pull_id)
        journal.mark(cid, "frozen", label=label, relpath=art.relpath, sha256=art.sha256)
        return art

    if workers <= 1:
        for label, params in todo:
            art = _one(label, params)
            if art is not None:
                frozen.append(art)
        return frozen

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, label, params): label for label, params in todo}
        for fut in as_completed(futures):
            try:
                art = fut.result()
                if art is not None:
                    frozen.append(art)
            except BaseException as e:          # noqa: BLE001 — collected, re-raised below
                failures.append(e)
    if failures:
        raise failures[0]
    return frozen


# ------------------------------------------------------------------ vendor fetchers
class EntitlementError(RuntimeError):
    """Raised when a vendor session/entitlement is unavailable — callers degrade
    gracefully (probe report + quarantined gap), never fabricate (R4)."""


def open_refinitiv_session():
    """Open a refinitiv.data session, preferring the RDP PLATFORM grant when
    REFINITIV_{USERNAME,PASSWORD,APP_KEY} are in .env (a "Platform Access" app key —
    no Workspace login required), else the Workspace DESKTOP session via
    EIKON_APP_KEY. The session's open_state is VERIFIED — a lazily-pending session
    must not masquerade as entitled access."""
    load_env()
    try:
        import refinitiv.data as rd
    except ImportError as e:
        raise EntitlementError("refinitiv-data not installed (requirements.txt / ADR-013)") from e
    user = os.environ.get("REFINITIV_USERNAME")
    password = os.environ.get("REFINITIV_PASSWORD")
    platform_key = os.environ.get("REFINITIV_APP_KEY")
    desktop_key = os.environ.get("EIKON_APP_KEY") or os.environ.get("RD_APP_KEY")
    try:
        if user and password and platform_key:
            session = rd.session.platform.Definition(
                app_key=platform_key,
                grant=rd.session.platform.GrantPassword(username=user, password=password),
                signon_control=True,
            ).get_session()
            session.open()
            rd.session.set_default(session)
        elif desktop_key:
            session = rd.session.desktop.Definition(app_key=desktop_key).get_session()
            session.open()
            rd.session.set_default(session)
        else:
            rd.open_session()                      # falls back to refinitiv-data config file
        state = rd.session.get_default().open_state
        if "Opened" not in str(state):             # Pending/Closed = NOT entitled-live
            raise RuntimeError(
                f"session open_state={state} (credentials rejected? Workspace not "
                "logged in? API proxy port not listening?)"
            )
        return rd
    except Exception as e:
        raise EntitlementError(
            f"Refinitiv session failed ({type(e).__name__}: {e}) — check .env "
            "credentials (platform) or Workspace login (desktop)"
        ) from e


def fetch_refinitiv_history(rd, universe: list[str], fields: list[str],
                            start: str, end: str) -> pd.DataFrame:
    return rd.get_history(universe=universe, fields=fields, start=start, end=end)


def fetch_refinitiv_data(rd, universe: list[str] | str, fields: list[str],
                         parameters: dict | None = None) -> pd.DataFrame:
    df = rd.get_data(universe=universe, fields=fields, parameters=parameters)
    return df


def open_datastream():
    """DSWS session (separate entitlement from Workspace; see DATA_ENTITLEMENTS)."""
    load_env()
    try:
        import DatastreamPy as dsweb
    except ImportError as e:
        raise EntitlementError("DatastreamPy not installed") from e
    user, pw = os.environ.get("DSWS_USERNAME"), os.environ.get("DSWS_PASSWORD")
    if not (user and pw):
        # Single best-effort fallback: some LSEG accounts share RDP credentials with
        # DSWS. ONE attempt only (probe-time); a rejection is evidence, not retried.
        user, pw = os.environ.get("REFINITIV_USERNAME"), os.environ.get("REFINITIV_PASSWORD")
    if not (user and pw):
        raise EntitlementError("DSWS_USERNAME/DSWS_PASSWORD not in .env — DSWS is a "
                               "separate entitlement; escalate to the data desk")
    try:
        return dsweb.DataClient(None, user, pw)        # DatastreamPy >=2 client class
    except Exception as e:
        raise EntitlementError(f"DSWS login failed: {type(e).__name__}: {e}") from e


def fetch_yfinance(params: dict) -> pd.DataFrame:
    """Shadow-vendor fetch: OHLCV + actions, NON-adjusted (raw fidelity) — the
    adjusted view is derivable; the reverse is not."""
    import yfinance as yf
    df = yf.download(params["tickers"], start=params["start"], end=params["end"],
                     auto_adjust=False, actions=True, progress=False, group_by="column")
    return df


def fetch_fred(params: dict) -> pd.DataFrame:
    load_env()
    key = os.environ.get("FRED_API_KEY")
    if key:
        from fredapi import Fred
        fred = Fred(api_key=key)
        return pd.DataFrame({s: fred.get_series(s, params["start"], params["end"])
                             for s in params["series"]})
    # Keyless path: FRED's public fredgraph CSV endpoint, chunked by date span —
    # full 2005-2025 requests 504 at FRED's gateway, ~5y spans return in ~1s each
    # (pandas-datareader's combined request hits the same gateway timeout).
    import io
    import urllib.request
    frames = {}
    for s in params["series"]:
        parts = []
        for a, b in chunk_dates(params["start"], params["end"], days=1825):
            url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}"
                   f"&cosd={a}&coed={b}")
            with urllib.request.urlopen(url, timeout=60) as r:
                body = r.read().decode()
            part = pd.read_csv(io.StringIO(body), index_col=0, parse_dates=True, na_values=".")
            parts.append(part.iloc[:, 0])
        series = pd.concat(parts)
        frames[s] = series[~series.index.duplicated(keep="first")]
    return pd.DataFrame(frames)


def fetch_french(params: dict) -> pd.DataFrame:
    from pandas_datareader import data as pdr
    ds = pdr.DataReader(params["dataset"], "famafrench", start=params["start"])
    return ds[0] / 100.0                                            # percent -> decimal


# ----------------------------------------------------------- field-definition probe
def capture_field_definitions(rd=None) -> dict:
    """Capture vendor field DEFINITIONS as evidence (stage-1 requirement): the RI
    day-count convention (Datastream computes RI with dividends reinvested using an
    annualised dividend yield on a fixed day-count — 260 vs 365 must be confirmed on
    OUR feed) and the currency of each requested field. What cannot be confirmed
    programmatically is recorded as requiring manual confirmation — never assumed."""
    out: dict = {
        "RI_day_count": {
            "status": "REQUIRES_MANUAL_CONFIRMATION",
            "note": "Datastream Navigator > RI definition on the entitled feed; "
                    "harvest missing-piece #5 (entitlement test 7)",
        },
        "currency_check": {"status": "PENDING"},
    }
    if rd is not None:
        try:
            cur = fetch_refinitiv_data(rd, "AAPL.O", ["TR.PriceClose.currency"])
            out["currency_check"] = {"status": "OK", "sample": cur.to_dict("records")[:3]}
        except Exception as e:
            out["currency_check"] = {"status": f"FAILED: {type(e).__name__}: {e}"}
    return out
