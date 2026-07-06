"""Read-side journal over a run's ``events.jsonl`` — the timestamped state-transition ledger.

The WRITE side already exists and is deliberately not duplicated here (one ledger, no drift):
``src.utils.logging.attach_run_logging`` hooks the ROOT logger, so every ``log_event`` in the driver
process appends one timestamped JSON line to ``<run_dir>/events.jsonl`` (``run_start``,
``candidate_done``, ``seed_done``, ``test_leg_stall``, sentinel transitions, ...). This module is the
tolerant READER + the aggregations the monitoring layer builds on:

* :func:`read_events` — parse the ledger, skipping torn/corrupt lines (a crash mid-append must never
  poison monitoring or resume tooling);
* :func:`completions` — the (epoch-seconds, event) completion stream for progress-rate / ETA math;
* :func:`cadence_stats` — robust (median/MAD) inter-completion cadence, the yardstick for
  multi-granularity stall detection (a pool that normally completes a training every ~28 min and has
  been silent for 3 h is wedged even though every process is alive);
* :func:`error_taxonomy` — errors clustered by kind with counts, first/last timestamps, and the
  affected arms (the triage panel's data).

Timestamps: the ``_JsonlFormatter`` stamps ``ts`` as LOCAL time ``%Y-%m-%dT%H:%M:%S`` (no zone
marker). :func:`parse_ts` therefore converts via ``time.mktime`` (local), which is consistent with
comparing against ``time.time()`` on the SAME machine — the only consumer (the sentinel runs where
the campaign runs). Report-only infrastructure: nothing here touches a result or the frozen design.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

__all__ = [
    "read_events",
    "parse_ts",
    "completions",
    "cadence_stats",
    "error_taxonomy",
]

#: Event kinds that mark ONE unit of campaign work durably finished (search candidate / TEST seed).
COMPLETION_EVENTS: tuple[str, ...] = ("candidate_done", "seed_done")

#: Event kinds that mark a unit FAILING (streamed failure rows + captured worker exceptions).
FAILURE_EVENTS: tuple[str, ...] = ("seed_failed",)


def read_events(path: str | Path, *, kinds: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    """Parse ``events.jsonl`` tolerantly: skip torn/corrupt lines, optionally filter by event kind.

    Returns ``[]`` for a missing file — monitoring must degrade, never crash, on a not-yet-started
    run. A line that fails to parse (the torn last line of a crash) is silently skipped: the ledger
    is a monitoring side-channel, and the durable source of truth for WORK is the archive itself.
    """
    p = Path(path)
    if not p.is_file():
        return []
    out: list[dict[str, Any]] = []
    try:
        raw = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(ev, dict):
            continue
        if kinds is not None and ev.get("event") not in kinds:
            continue
        out.append(ev)
    return out


def parse_ts(ev: dict[str, Any]) -> float | None:
    """Event ``ts`` -> local epoch seconds; ``None`` when absent/unparseable (skip, don't guess)."""
    ts = ev.get("ts")
    if not isinstance(ts, str):
        return None
    try:
        return time.mktime(time.strptime(ts, "%Y-%m-%dT%H:%M:%S"))
    except (ValueError, OverflowError):
        return None


def completions(
    events: list[dict[str, Any]], *, kinds: tuple[str, ...] = COMPLETION_EVENTS
) -> list[tuple[float, dict[str, Any]]]:
    """The timestamped unit-completion stream, sorted by time (rate / ETA / stall math input)."""
    out: list[tuple[float, dict[str, Any]]] = []
    for ev in events:
        if ev.get("event") not in kinds:
            continue
        t = parse_ts(ev)
        if t is not None:
            out.append((t, ev))
    out.sort(key=lambda te: te[0])
    return out


def cadence_stats(times: list[float]) -> dict[str, float] | None:
    """Robust inter-completion cadence: ``{median_gap_s, mad_gap_s, last_completion_s}``.

    ``None`` with fewer than 3 completions (2 gaps) — too little signal to set a stall yardstick;
    callers fall back to their static thresholds. Median + MAD (not mean/std) so one long thermal
    pause does not inflate the yardstick and mask a later genuine wedge.
    """
    ts = sorted(float(t) for t in times)
    if len(ts) < 3:
        return None
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    gaps_sorted = sorted(gaps)
    n = len(gaps_sorted)
    median = (
        gaps_sorted[n // 2] if n % 2 == 1 else 0.5 * (gaps_sorted[n // 2 - 1] + gaps_sorted[n // 2])
    )
    abs_dev = sorted(abs(g - median) for g in gaps)
    mad = abs_dev[n // 2] if n % 2 == 1 else 0.5 * (abs_dev[n // 2 - 1] + abs_dev[n // 2])
    return {"median_gap_s": float(median), "mad_gap_s": float(mad), "last_completion_s": float(ts[-1])}


#: Taxonomy needles, in priority order (FIRST match wins — keep the specific before the generic).
#: Matched on LETTER boundaries, not as bare substrings (deep review 2026-07-26, #65). A plain
#: ``needle in text`` misfires on ordinary words that happen to contain a needle, and it did:
#: MEASURED — "pip install failed" classified as ``stall`` (in-STALL-ed), "cluster maintenance
#: window" as ``nan`` (mainte-NAN-ce), and "reading capital allocation table" as ``api_error``
#: (c-API-tal). The triage panel would then show a NUMERIC-DIVERGENCE row for a maintenance notice,
#: which at 03:00 sends the operator hunting a science problem that does not exist.
#:
#: The boundary is LETTER-only (``(?<![a-z])…(?![a-z])``), deliberately NOT ``\b``: event kinds are
#: snake_case, so ``\bstall\b`` would FAIL to match the real ``test_leg_stall`` event (``_`` is a
#: word character) while still matching nothing useful. A letter boundary matches ``test_leg_stall``
#: and ``api_error`` while refusing ``install`` and ``capital``.
#: ``(needle, kind, whole_word)``. The LEADING boundary always applies — it is what blocks
#: "in-STALL-ed", "c-API-tal" and "mainte-NAN-ce". The TRAILING boundary is opt-in because most
#: needles are STEMS that must still match their inflections: requiring it everywhere regressed
#: "loss diverged to inf" to ``other`` (caught by testing the fix against the legitimate cases, not
#: only against the misfires). It is applied only to the short, high-collision needles where a bare
#: prefix would over-match — ``api`` (apiary/apiece) and ``nan`` (nanometer).
_TAXONOMY: tuple[tuple[str, str, bool], ...] = (
    ("out of memory", "oom", False),
    ("cuda", "cuda_error", False),
    ("sandbox", "sandbox_reject", False),      # sandboxed / sandboxing
    ("overloaded", "api_error", False),
    ("rate limit", "api_error", False),        # rate limited / rate limiting
    ("api", "api_error", True),                # NOT capital / therapist / apiary
    ("timeout", "timeout", False),             # timeouts / timeout_s
    ("diverge", "divergence", False),          # diverged / divergence / diverging
    ("nan", "nan", True),                      # NOT maintenance / nanometer
    ("stall", "stall", False),                 # stalled / stalling — but NOT install (leading bound)
)

_TAXONOMY_RE: tuple[tuple[Any, str], ...] = tuple(
    (re.compile(r"(?<![a-z])" + re.escape(needle) + (r"(?![a-z])" if whole else "")), kind)
    for needle, kind, whole in _TAXONOMY
)


def _classify(ev: dict[str, Any]) -> str:
    """Cluster an error-ish event into a stable taxonomy kind (triage panel rows)."""
    text = " ".join(
        str(ev.get(k, "")) for k in ("error", "msg", "kind", "event")
    ).lower()
    for pattern, kind in _TAXONOMY_RE:
        if pattern.search(text):
            return kind
    return "other"


def error_taxonomy(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Cluster failures/anomalies by kind -> ``{count, first_ts, last_ts, arms}`` (B5 triage data).

    Feeds on the WARNING+ events and the explicit failure kinds; INFO progress events are ignored.
    """
    out: dict[str, dict[str, Any]] = {}
    for ev in events:
        is_failure = ev.get("event") in FAILURE_EVENTS
        level = str(ev.get("level", "")).upper()
        if not is_failure and level not in ("WARNING", "ERROR", "CRITICAL"):
            continue
        kind = _classify(ev)
        row = out.setdefault(
            kind, {"count": 0, "first_ts": None, "last_ts": None, "arms": set()}
        )
        row["count"] += 1
        t = parse_ts(ev)
        if t is not None:
            row["first_ts"] = t if row["first_ts"] is None else min(row["first_ts"], t)
            row["last_ts"] = t if row["last_ts"] is None else max(row["last_ts"], t)
        arm = ev.get("arm")
        if arm:
            row["arms"].add(str(arm))
    for row in out.values():
        row["arms"] = sorted(row["arms"])
    return out
