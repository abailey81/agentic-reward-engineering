"""Live dashboard for a (possibly backgrounded / remote-headless) prototype/campaign run.

Reads ``<run_dir>/progress.json`` (the atomically-rewritten state from :class:`src.utils.monitoring.RunMonitor`)
and tails ``events.jsonl`` / ``anomalies.jsonl`` to render a precise, real-time picture: multi-level progress
(arms > candidates > training steps), the latest training metrics, GPU/CPU/RAM telemetry, ETA, an error
tracker (anomalies grouped by category), live LLM token-spend + estimated cost, a **silent-hang detector**
(loud STALE banner when ``progress.json`` stops being rewritten while the run is still "running"), and any
anomalies — so anything even slightly off is visible at a glance, **without attaching to the training
process**. This watcher is strictly READ-ONLY: it parses on-disk artefacts and never touches the run, so a
crash here can never affect the frozen campaign or its byte-identical replay.

Usage::

    python scripts/monitor.py outputs/prototype                       # live dashboard (~2 Hz refresh)
    python scripts/monitor.py outputs/prototype --once                # one plain-text snapshot (scriptable)
    python scripts/monitor.py outputs/prototype --interval 1
    python scripts/monitor.py outputs/prototype --stale-secs 600      # tune the silent-hang threshold
    # Remote, unattended (rented GPU over SSH): push a phone alert on done/error/stall. OFF by default;
    # stdlib urllib only (no new dep); a pure observation side-channel — sends run STATUS, never data.
    python scripts/monitor.py outputs/campaign --notify https://ntfy.sh/<your-private-topic>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

# Default silence threshold: ``RunMonitor._flush`` rewrites progress.json on every candidate AND every idle
# pump tick (~0.3 s in the parallel path), so no update for minutes means the run has genuinely hung
# (deadlocked worker, wedged CUDA call, OOM-killed driver) rather than merely being between candidates.
_DEFAULT_STALE_SECS = 300.0

# LLM list price per 1M tokens (Anthropic, Jan 2026) — matched by substring against the logged model id, so
# the dashboard shows a running USD estimate for the reward-authoring spend. Report-only; over-estimates
# slightly because it ignores prompt-cache reads (which bill ~0.1x). Update if Anthropic re-prices.
_PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "fable-5": (10.0, 50.0),
    "opus-5": (5.0, 25.0),    # confirmatory author (R102, 2026-07-25); same $5/$25 as 4.8
    "opus-4-8": (5.0, 25.0),  # retained: M2 generation-pair partner + legacy
    "opus-4-7": (5.0, 25.0),
    "opus-4-6": (5.0, 25.0),
    "sonnet-5": (2.0, 10.0),  # seated leg (R90/R92), ANTHROPIC-billed. Introductory pricing through
                              # 2026-08-31 (covers the campaign window), standard $3/$15 after — mirrors
                              # config/legs.yaml. ADDED 2026-07-26 (deep review): it was ABSENT
                              # from both Anthropic price tables, so every sonnet-5 call booked $0.00.
    "sonnet-4-6": (3.0, 15.0),  # legacy: the sonnet-4.6 LEG was removed by R92; kept for archived calls
    "haiku-4-5": (1.0, 5.0),
}

# Rotating-circle spinner frames (user-requested "circles") — advanced through on each refresh tick.
_SPINNER = "|/-\\"

# ---- small read-only cache: re-parse a JSONL only when its (mtime,size) changes ------------------------- #
_CACHE: dict[str, tuple[tuple[float, int], list[dict[str, Any]]]] = {}


def read_state(run_dir: Path) -> dict[str, Any] | None:
    p = run_dir / "progress.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None  # mid-write race; the next tick re-reads the atomic replacement


def _read_jsonl_cached(path: Path) -> list[dict[str, Any]]:
    """Parse a JSONL file, caching by (mtime, size) so the ~2 Hz refresh doesn't re-read an unchanged file."""
    if not path.exists():
        return []
    try:
        stt = path.stat()
        key = (stt.st_mtime, stt.st_size)
    except Exception:
        return []
    hit = _CACHE.get(str(path))
    if hit is not None and hit[0] == key:
        return hit[1]
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    _CACHE[str(path)] = (key, out)
    return out


def tail_jsonl(path: Path, n: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()[-n:]:
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


# ---- liveness / silent-hang detection ------------------------------------------------------------------ #
def state_age_seconds(
    st: dict[str, Any] | None, now_epoch: float, *, mtime_epoch: float | None = None
) -> float | None:
    """Seconds since ``progress.json`` was last rewritten, or None if unknown/unparseable.

    Uses the monitor's own ``updated`` wall-clock stamp (``%Y-%m-%dT%H:%M:%S``, local time), so it measures
    *writer* liveness independent of this watcher's clock drift within the same host. m11 (ops audit
    2026-07-02): when the file's ``mtime_epoch`` is supplied, the freshest of the two evidences of a
    write — ``max(parsed stamp, mtime)`` — is used, so a skewed/garbled writer clock (DST shift, a stamp
    the parser rejects) cannot fabricate a false STALE banner while the file is demonstrably being
    rewritten (and vice versa a weird filesystem mtime cannot mask a genuinely old stamp).
    """
    if not st:
        return None
    wrote: float | None = None
    ts = st.get("updated")
    if isinstance(ts, str):
        try:
            wrote = time.mktime(time.strptime(ts, "%Y-%m-%dT%H:%M:%S"))
        except Exception:
            wrote = None
    candidates = [t for t in (wrote, mtime_epoch) if t is not None]
    if not candidates:
        return None
    return max(0.0, now_epoch - max(candidates))


def is_stale(
    st: dict[str, Any] | None, now_epoch: float, threshold: float, *, mtime_epoch: float | None = None
) -> bool:
    """True when the run *should* be advancing but ``progress.json`` has gone silent past ``threshold``.

    A terminal phase (``done``/``error``) is never stale — the writer has intentionally stopped.
    """
    if not st or st.get("phase") in ("done", "error", "starting"):
        return False
    age = state_age_seconds(st, now_epoch, mtime_epoch=mtime_epoch)
    return age is not None and age > threshold


def _progress_mtime(run_dir: Path) -> float | None:
    """mtime of ``<run_dir>/progress.json`` (None when absent) — the m11 clock-skew fallback input."""
    try:
        return (run_dir / "progress.json").stat().st_mtime
    except OSError:
        return None


# ---- error tracker + token/cost accounting (read-only over the structured logs) ------------------------- #
def anomaly_counts(run_dir: Path) -> dict[str, int]:
    """Anomalies grouped by ``kind`` (critic_explosion, ram_pressure_*, failure_wave, …) — the error tracker."""
    c: Counter[str] = Counter()
    for rec in _read_jsonl_cached(run_dir / "anomalies.jsonl"):
        c[str(rec.get("kind", "?"))] += 1
    return dict(c.most_common())


def _resolve_price(model: str | None) -> tuple[float, float] | None:
    if not model:
        return None
    m = model.lower()
    for key, price in _PRICES_PER_MTOK.items():
        if key in m:
            return price
    return None


def token_spend(run_dir: Path) -> dict[str, Any]:
    """Aggregate LLM token usage + estimated USD cost from the ``llm_call`` records in ``events.jsonl``.

    Returns ``{calls, in_tok, out_tok, usd, costed_calls}``. ``usd`` sums only calls whose model id matches
    the price table (``costed_calls``); list price, cache reads ignored ⇒ a conservative upper bound.
    """
    calls = in_tok = out_tok = costed = 0
    usd = 0.0
    for rec in _read_jsonl_cached(run_dir / "events.jsonl"):
        if rec.get("event") != "llm_call":
            continue
        calls += 1
        it = rec.get("in_tok")
        ot = rec.get("out_tok")
        it = int(it) if isinstance(it, (int, float)) else 0
        ot = int(ot) if isinstance(ot, (int, float)) else 0
        in_tok += it
        out_tok += ot
        price = _resolve_price(rec.get("model"))
        if price is not None:
            usd += it / 1e6 * price[0] + ot / 1e6 * price[1]
            costed += 1
    return {"calls": calls, "in_tok": in_tok, "out_tok": out_tok, "usd": round(usd, 4), "costed_calls": costed}


# ---- remote alert (opt-in, fail-safe, stdlib only; a pure side-channel) --------------------------------- #
def build_alert(st: dict[str, Any] | None, reason: str, run_dir: Path) -> str:
    """One-line status message for a push notification. Carries run STATUS only — never data/returns/code."""
    if not st:
        if reason == "state_lost":  # the file was READ before and is now gone — not a slow start
            return f"[{run_dir.name}] STATE_LOST: progress.json VANISHED mid-run (was readable earlier) — check the run/disk"
        return f"[{run_dir.name}] {reason}: no progress.json yet"
    a, c = st.get("arms", {}), st.get("candidates", {})
    an = st.get("anomalies", {})
    tok = token_spend(run_dir)
    return (
        f"[{st.get('title', run_dir.name)}] {reason.upper()} | phase={st.get('phase')} "
        f"arms {a.get('done', '?')}/{a.get('total', '?')} cand {c.get('run_done', '?')}/{c.get('run_total', '?')} "
        f"anomalies={an.get('count', 0)} llm_usd≈{tok['usd']}"
    )


def post_alert(url: str, message: str, *, timeout: float = 10.0) -> bool:
    """POST ``message`` to ``url`` (ntfy.sh-compatible). Best-effort: swallows every error, never raises."""
    try:
        req = urllib.request.Request(
            url, data=message.encode("utf-8"),
            headers={"Content-Type": "text/plain", "Title": "campaign-monitor"},
        )
        urllib.request.urlopen(req, timeout=timeout).read()  # noqa: S310 - operator-supplied alert URL
        return True
    except Exception:
        return False


# ---- notification lifecycle + campaign-follow (pure logic, unit-tested; M5a ops audit 2026-07-02) ------- #
def alert_reason(
    st: dict[str, Any] | None,
    now_epoch: float,
    stale_secs: float,
    *,
    mtime_epoch: float | None = None,
    disk_free_gb: float | None = None,
    min_disk_gb: float = 10.0,
    anomaly_delta: int | None = None,
    anomaly_delta_limit: int = 20,
    state_seen: bool = False,
) -> str | None:
    """The alert-worthy condition right now (pure; 2026-07-05 rule additions).

    ``"error"`` / ``"done"`` (terminal) take precedence; then ``"disk_low"`` (the run drive under
    ``min_disk_gb`` — the pagefile + Windows Update live on C:, and a full drive kills the run in ways
    the stall detector only reports after the fact); then ``"anomaly_surge"`` (the anomalies.jsonl
    grew by >= ``anomaly_delta_limit`` lines within ONE poll interval — a critic-explosion /
    ram-pressure / failure-wave cascade worth a phone push the moment it starts, not at the daily
    read); then ``"stall"``. ``None`` = healthy. Extra signals default to ``None`` = rule inactive,
    so existing callers/tests are untouched.

    ``state_seen`` (LATCH-ONCE-SEEN, 2026-07-26) disambiguates the two meanings of a MISSING state.
    Before the run writes its first ``progress.json``, ``st is None`` means "not started yet" and is
    correctly silent. AFTER a state has once been read, ``None`` means the file VANISHED — deleted,
    truncated, or a write that failed on a full disk — and the watcher would otherwise report the run
    healthy forever while every later stall/error/done alert is silently lost, exactly when the
    operator is asleep and relying on the push. With the latch set, that case raises ``"state_lost"``.
    """
    if st is None:
        return "state_lost" if state_seen else None
    phase = st.get("phase")
    if phase == "error":
        return "error"
    if phase == "done":
        return "done"
    if disk_free_gb is not None and disk_free_gb < float(min_disk_gb):
        return "disk_low"
    if anomaly_delta is not None and int(anomaly_delta) >= int(anomaly_delta_limit):
        return "anomaly_surge"
    if is_stale(st, now_epoch, stale_secs, mtime_epoch=mtime_epoch):
        return "stall"
    return None


def process_notification(reason: str | None, sent: set[str], send: Any) -> bool:
    """Drive the alert dedupe set for one poll tick; returns True iff an alert was posted.

    Fixes two silent-alert-loss bugs (M5a ii/iii): (ii) the ``sent`` set is RESET when the run is
    healthy again (``reason is None``), so after a campaign phase transition (arm N ``done`` -> arm
    N+1 running) or a recovered stall, the NEXT done/error/stall alerts fire instead of being deduped
    forever by the first arm's; (iii) a reason enters ``sent`` only AFTER ``send`` succeeded — a
    failed POST (network blip at exactly the wrong moment) is retried on the next poll rather than
    being permanently swallowed by an optimistic ``sent.add``.
    """
    if reason is None:
        if sent:
            sent.clear()
        return False
    if reason in sent:
        return False
    if send(reason):
        sent.add(reason)
        return True
    return False


def campaign_done(
    sentinel_mtime: float | None, started_epoch: float, exit_code: int | None = 0
) -> bool:
    """--follow-campaign exit test: a FRESH sentinel from a TERMINAL pass.

    ``run_campaign`` writes ``campaign_summary.json`` at the end of EVERY pass — including a
    resumable exit-3 (``EXIT_INCOMPLETE``) pass that the supervisor immediately relaunches with
    ``--resume``. The old mtime-only test therefore killed the dashboard/ntfy watcher on the FIRST
    such pass and the remaining (possibly week-long) passes ran unwatched (2026-07-05 fix). Exit now
    requires BOTH: (a) the sentinel was (re)written after this watcher started (the ``>=
    started_epoch`` guard keeps a stale summary from a previous run from terminating immediately),
    and (b) the pass was TERMINAL — ``exit_code == 0`` (full success, or an operator-initiated
    graceful shutdown; the supervisor stops in both cases). ``exit_code=None`` (an unreadable /
    mid-write summary) is NOT done — the next tick re-reads it.
    """
    if sentinel_mtime is None or sentinel_mtime < started_epoch:
        return False
    return exit_code is not None and int(exit_code) == 0


def _sentinel_exit_code(run_dir: Path) -> int | None:
    """``exit_code`` recorded in the campaign summary (0 for legacy summaries without the key).

    Returns ``None`` when the file exists but cannot be parsed (mid-write) — the watcher then keeps
    watching and re-reads on the next tick. Missing file -> 0 (the mtime guard alone decides)."""
    for p in (run_dir / "campaign_summary.json", run_dir.parent / "campaign_summary.json"):
        if not p.is_file():
            continue
        try:
            summary = json.loads(p.read_text(encoding="utf-8"))
            return int(summary.get("exit_code", 0) or 0)
        except (OSError, ValueError):
            return None
    return 0


def _sentinel_mtime(run_dir: Path) -> float | None:
    """mtime of the campaign summary sentinel, checking ``run_dir`` then its parent.

    The campaign's per-arm search progress lives under ``<output>/search/progress.json`` while the
    summary is written to ``<output>/campaign_summary.json`` — so when the watcher is pointed at the
    search dir (the documented usage) the sentinel is one level up; checking both keeps a watcher
    pointed at ``<output>`` itself working too.
    """
    for p in (run_dir / "campaign_summary.json", run_dir.parent / "campaign_summary.json"):
        try:
            return p.stat().st_mtime
        except OSError:
            continue
    return None


def sentinel_summary_line(run_dir: Path) -> str | None:
    """ONE line of health verdict from the SENTINEL's invariant checks (B7 unified view).

    The dashboard shows what is HAPPENING; this line says whether it is HEALTHY — worst severity +
    the top offending checks, or coverage/ETA when everything is green (the operator's #1 question).
    Guarded/best-effort and read-only: any failure returns ``None`` and the dashboard renders
    without it. Recomputed every ~15 ticks by the watch loop (it walks the archive), not per frame.
    """
    try:
        import sys as _sys

        _here = str(Path(__file__).resolve().parent)
        if _here not in _sys.path:
            _sys.path.insert(0, _here)
        import sentinel as _sentinel

        report = _sentinel.evaluate_health(_sentinel.gather_inputs(run_dir))
        bad = [c for c in report.checks
               if c.severity in (_sentinel.WARN, _sentinel.CRITICAL, _sentinel.UNKNOWN)]
        if not bad:
            cov = [c.detail for c in report.checks if c.name.startswith("coverage_")]
            return f"SENTINEL OK — {'; '.join(cov) if cov else 'all checks OK'}"
        bad.sort(key=lambda c: -_sentinel._RANK.get(c.severity, 0))
        tops = "; ".join(f"{c.name}: {c.detail}" for c in bad[:3])
        return f"SENTINEL {report.severity} — {tops}"
    except Exception:  # noqa: BLE001 — the health line must never break the dashboard
        return None


def _bar(done: int, total: int, width: int = 30) -> str:
    # ASCII fill so the plain-text snapshot prints on ANY console codepage (e.g. Windows cp1251);
    # rich renders these fine too.
    total = max(int(total or 0), 1)
    frac = min(max(done / total, 0.0), 1.0)
    fill = int(frac * width)
    return f"[{'#' * fill}{'-' * (width - fill)}] {done}/{total} ({frac * 100:4.1f}%)"


def _fmt_secs(s: float | None) -> str:
    if not s or s <= 0:
        return "?"
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{sec:02d}s"


def snapshot_text(st: dict[str, Any], run_dir: Path | None = None, *, stale_secs: float = _DEFAULT_STALE_SECS) -> str:
    a, c, tr, r = st["arms"], st["candidates"], st.get("training", {}), st.get("resources", {})
    an = st.get("anomalies", {})
    gpu = f"GPU {r.get('gpu_util', '?')}% {r.get('gpu_mem_mib', ['?', '?'])[0]}/{r.get('gpu_mem_mib', ['?', '?'])[1]}MiB {r.get('gpu_temp', '?')}C"
    mt = _progress_mtime(run_dir) if run_dir is not None else None  # m11 clock-skew tolerance
    age = state_age_seconds(st, time.time(), mtime_epoch=mt)
    lines = []
    if is_stale(st, time.time(), stale_secs, mtime_epoch=mt):
        lines.append(f"  !!! STALE — no progress.json update for {_fmt_secs(age)} (silent hang? check the run)")
    lines += [
        f"{st.get('title')} [{st.get('model')}]  phase={st.get('phase')}  elapsed={_fmt_secs(st.get('elapsed_s'))}  ETA={_fmt_secs(st.get('eta_s'))}  age={_fmt_secs(age)}",
        f"  Arms       {_bar(a['done'], a['total'])}  current={a.get('current')}",
        f"  Candidates {_bar(c['run_done'], c['run_total'])}  in-arm {c['in_arm_done']}/{c['in_arm_total']}",
        f"  Training   {_bar(tr.get('step', 0), tr.get('total', 1))}  "
        f"fps={_g(tr.get('fps'))} critic={_g(tr.get('critic_loss'))} actor={_g(tr.get('actor_loss'))} ent={_g(tr.get('ent_coef'))} rew={_g(tr.get('ep_rew_mean'))}",
        f"  {gpu}  CPU {r.get('cpu_pct', '?')}%  RAM {r.get('ram_pct', '?')}%  RSS {r.get('proc_rss_mib', '?')}MiB",
        f"  best_fitness={st.get('best_fitness')}",
        f"  ANOMALIES={an.get('count', 0)}",
    ]
    if run_dir is not None:
        by_kind = anomaly_counts(run_dir)
        if by_kind:
            lines.append("    by kind: " + ", ".join(f"{k}×{v}" for k, v in by_kind.items()))
        tok = token_spend(run_dir)
        if tok["calls"]:
            lines.append(
                f"  LLM        {tok['calls']} calls  in={tok['in_tok']:,} out={tok['out_tok']:,} tok  est≈${tok['usd']}"
            )
    for rec in an.get("recent", []):
        lines.append(f"    ! {rec.get('kind')}: {rec.get('detail')}")
    if run_dir is not None:
        health = sentinel_summary_line(run_dir)
        if health:
            lines.append(f"  {health}")
    return "\n".join(lines)


def _g(v: Any) -> str:
    return f"{v:.4g}" if isinstance(v, (int, float)) and v == v else "?"


def render(st: dict[str, Any] | None, run_dir: Path, *, stale_secs: float = _DEFAULT_STALE_SECS,
           tick: int = 0, sentinel_line: str | None = None) -> Any:
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    if st is None:
        return Panel(Text("waiting for progress.json …", style="yellow"), title=str(run_dir))

    now = time.time()
    mt = _progress_mtime(run_dir)  # m11 clock-skew tolerance: freshest of parsed stamp vs file mtime
    stale = is_stale(st, now, stale_secs, mtime_epoch=mt)
    age = state_age_seconds(st, now, mtime_epoch=mt)
    a, c, tr, r = st["arms"], st["candidates"], st.get("training", {}), st.get("resources", {})
    an = st.get("anomalies", {})
    t = Table.grid(padding=(0, 1))
    t.add_column(justify="right", style="bold cyan")
    t.add_column()
    if stale:
        t.add_row("STALE", Text(f"no update for {_fmt_secs(age)} — silent hang? the run may be wedged", style="bold red"))
    t.add_row("phase", f"{st.get('phase')}   elapsed {_fmt_secs(st.get('elapsed_s'))}   ETA {_fmt_secs(st.get('eta_s'))}   age {_fmt_secs(age)}")
    t.add_row("Arms", _bar(a["done"], a["total"]) + f"   {a.get('current') or ''}")
    t.add_row("Candidates", _bar(c["run_done"], c["run_total"]) + f"   in-arm {c['in_arm_done']}/{c['in_arm_total']} gen{c.get('gen', 0)}")
    t.add_row("Training", _bar(tr.get("step", 0), tr.get("total", 1)))
    t.add_row("", f"fps={_g(tr.get('fps'))}  critic={_g(tr.get('critic_loss'))}  actor={_g(tr.get('actor_loss'))}  ent={_g(tr.get('ent_coef'))}  rew={_g(tr.get('ep_rew_mean'))}")
    gpu_mem = r.get("gpu_mem_mib", ["?", "?"])
    t.add_row("GPU", f"{r.get('gpu_util', '?')}%  {gpu_mem[0]}/{gpu_mem[1]} MiB  {r.get('gpu_temp', '?')}C")
    t.add_row("Host", f"CPU {r.get('cpu_pct', '?')}%  RAM {r.get('ram_pct', '?')}%  RSS {r.get('proc_rss_mib', '?')} MiB")
    t.add_row("best", str(st.get("best_fitness")))
    # LLM token-spend + estimated USD (reward-authoring cost) — read from events.jsonl.
    tok = token_spend(run_dir)
    if tok["calls"]:
        t.add_row("LLM", f"{tok['calls']} calls  {tok['in_tok']:,} in / {tok['out_tok']:,} out tok  est≈${tok['usd']}")
    # Candidate-parallel: a row per concurrently-training candidate.
    active = st.get("active") or []
    if active:
        t.add_row("active", f"{len(active)} training concurrently")
        for w in active[:8]:
            t.add_row(
                "  > " + str(w.get("arm", ""))[:13],
                f"{w.get('cand', '')} {_bar(w.get('step', 0), w.get('total', 1), 16)} "
                f"fps={_g(w.get('fps'))} critic={_g(w.get('critic_loss'))}",
            )

    nan = int(an.get("count", 0))
    anbody = Text(f"anomalies: {nan}", style="bold red" if nan else "green")
    by_kind = anomaly_counts(run_dir)
    if by_kind:
        anbody.append("  [" + ", ".join(f"{k}×{v}" for k, v in by_kind.items()) + "]", style="red")
    for rec in an.get("recent", []):
        anbody.append(f"\n ! {rec.get('kind')}: {rec.get('detail')}", style="red")

    ev = Text("recent events", style="dim")
    for e in tail_jsonl(run_dir / "events.jsonl", 8):
        style = "red" if e.get("event") == "ANOMALY" else ("yellow" if e.get("level") == "WARNING" else "dim")
        ev.append(f"\n {e.get('ts', '')[-8:]} {e.get('event')} {e.get('msg', '')[:80]}", style=style)
    if sentinel_line:
        # B7 unified view: the sentinel's one-line health verdict beneath the event tail.
        ev.append(
            f"\n {sentinel_line}",
            style="bold red" if "CRITICAL" in sentinel_line
            else ("yellow" if ("WARN" in sentinel_line or "UNKNOWN" in sentinel_line) else "green"),
        )

    spin = _SPINNER[tick % len(_SPINNER)] if st.get("phase") not in ("done", "error") else "*"
    title = f"{spin} {st.get('title')} [{st.get('model')}]  pid {st.get('pid')}"
    border = "red" if (nan or stale) else ("green" if st.get("phase") == "done" else "cyan")
    return Panel(Group(t, anbody, ev), title=title, border_style=border)


def main() -> None:
    ap = argparse.ArgumentParser(description="Live, read-only dashboard for a prototype/campaign run.")
    ap.add_argument("run_dir", help="the run output dir (contains progress.json)")
    ap.add_argument("--once", action="store_true", help="print one plain-text snapshot and exit")
    ap.add_argument("--interval", type=float, default=2.0, help="refresh seconds (default 2)")
    ap.add_argument("--stale-secs", type=float, default=_DEFAULT_STALE_SECS,
                    help=f"silent-hang threshold in seconds (default {_DEFAULT_STALE_SECS:g})")
    ap.add_argument("--notify", metavar="URL", default=None,
                    help="opt-in push URL (e.g. https://ntfy.sh/<topic>): POST a STATUS line on done/error/stall. "
                         "OFF by default; stdlib only; a read-only side-channel that never touches the run.")
    ap.add_argument("--heartbeat", metavar="URL", default=None,
                    help="opt-in DEADMAN heartbeat URL (e.g. a healthchecks.io ping): POST a compact health "
                         "snapshot every --heartbeat-every seconds UNCONDITIONALLY. The external service "
                         "alarms on the ABSENCE of pings — the only way to detect host death (power loss, "
                         "kernel panic) that no on-host monitor can report. Best-effort; never touches the run.")
    ap.add_argument("--heartbeat-every", type=float, default=600.0,
                    help="seconds between deadman heartbeats (default 600; set the external service's "
                         "grace period to ~2-3x this)")
    ap.add_argument("--follow-campaign", action="store_true",
                    help="do NOT exit when one phase reports done/error — the campaign rewrites the SAME "
                         "progress.json once per arm, so a plain watch dies after the FIRST arm. Keeps "
                         "polling (re-arming alerts between arms) until <run_dir>/../campaign_summary.json "
                         "is (re)written — the overall-campaign sentinel — or Ctrl-C. Point run_dir at "
                         "outputs/campaign/search for the headline run (M5a, ops audit 2026-07-02).")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]  # safe unicode on any console codepage
    except Exception:
        pass
    run_dir = Path(args.run_dir)

    if args.once:
        st = read_state(run_dir)
        print(snapshot_text(st, run_dir, stale_secs=args.stale_secs) if st else f"no progress.json under {run_dir} yet")
        return

    try:  # beautiful, locals-capturing tracebacks if the dashboard itself errors (advanced error tracking)
        from rich.traceback import install as _install_tb

        _install_tb(show_locals=False)
    except Exception:
        pass

    from rich.console import Console
    from rich.live import Live

    console = Console()
    # Dedupe push notifications: one per (done|error|stall) PER unhealthy episode — the set is reset
    # by process_notification when the run turns healthy again (M5a ii), so a multi-arm campaign's
    # later transitions still alert; entries are added only AFTER a successful POST (M5a iii).
    sent: set[str] = set()
    started_epoch = time.time()  # --follow-campaign: only a sentinel written AFTER this counts

    _last_anomaly_total = [None]  # per-tick anomaly-count memory for the surge rule (2026-07-05)
    _state_seen = [False]  # LATCH: once progress.json has been read, a later disappearance is an ALERT

    def _maybe_notify(st: dict[str, Any] | None) -> None:
        if not args.notify:
            return
        if st is not None:
            _state_seen[0] = True  # latch BEFORE the check: a first read is never itself "lost"
        # Extra precision signals (2026-07-05): free disk on the run drive + the per-tick anomaly
        # GROWTH (a cascade pushes a phone alert the moment it starts). Both read-only and best-effort
        # — a probe failure silently deactivates its rule for this tick, never the watcher.
        disk_free_gb = None
        try:
            import shutil as _sh

            disk_free_gb = _sh.disk_usage(str(run_dir)).free / 1e9
        except OSError:
            pass
        anomaly_delta = None
        try:
            total = sum(anomaly_counts(run_dir).values())
            if _last_anomaly_total[0] is not None:
                anomaly_delta = total - int(_last_anomaly_total[0])
            _last_anomaly_total[0] = total
        except Exception:  # noqa: BLE001 - a parse hiccup must never kill the watcher tick
            pass
        reason = alert_reason(
            st, time.time(), args.stale_secs, mtime_epoch=_progress_mtime(run_dir),
            disk_free_gb=disk_free_gb, anomaly_delta=anomaly_delta, state_seen=_state_seen[0],
        )
        process_notification(
            reason, sent, lambda r: post_alert(args.notify, build_alert(st, r, run_dir))
        )

    tick = 0
    _health_line: list[str | None] = [None]  # B7: recomputed every ~15 ticks (it walks the archive)
    _last_beat = [0.0]  # B6: last deadman heartbeat epoch

    def _maybe_heartbeat(st: dict[str, Any] | None) -> None:
        if not args.heartbeat:
            return
        now = time.time()
        if now - _last_beat[0] < max(30.0, float(args.heartbeat_every)):
            return
        line = build_alert(st, "heartbeat", run_dir) if st else f"[{run_dir}] heartbeat (no state yet)"
        if _health_line[0]:
            line = f"{line} | {_health_line[0]}"
        # UNCONDITIONAL send (no dedupe): the external service alarms on ABSENCE, so every interval
        # must ping. _last_beat advances even on failure — a dead network must not tight-loop POSTs.
        post_alert(args.heartbeat, line)
        _last_beat[0] = now

    with Live(render(read_state(run_dir), run_dir, stale_secs=args.stale_secs), console=console,
              refresh_per_second=2, screen=False) as live:
        while True:
            try:
                tick += 1
                if tick % 15 == 1:
                    _health_line[0] = sentinel_summary_line(run_dir)
                st = read_state(run_dir)
                live.update(render(st, run_dir, stale_secs=args.stale_secs, tick=tick,
                                   sentinel_line=_health_line[0]))
                _maybe_notify(st)
                _maybe_heartbeat(st)
                if args.follow_campaign:
                    # M5a(i): a per-arm 'done'/'error' is NOT the end of the campaign — the next arm
                    # rewrites the same progress.json. Exit only on the overall-campaign sentinel,
                    # and only when its pass was TERMINAL (exit_code==0) — an exit-3 resumable pass
                    # is relaunched by the supervisor and must stay watched (2026-07-05).
                    if campaign_done(_sentinel_mtime(run_dir), started_epoch,
                                     exit_code=_sentinel_exit_code(run_dir)):
                        live.update(render(read_state(run_dir), run_dir, stale_secs=args.stale_secs, tick=tick))
                        break
                elif st is not None and st.get("phase") in ("done", "error"):
                    time.sleep(args.interval)
                    live.update(render(read_state(run_dir), run_dir, stale_secs=args.stale_secs, tick=tick))
                    break
                time.sleep(args.interval)
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()
