"""Administrative-kill detection and GRADUATED RETREAT for the Myriad campaign.

WHY THIS EXISTS (2026-07-26). The campaign's throughput comes from holding a large, continuously
fed queue of small CPU jobs (measured: 636 concurrent cores; see
the Myriad scheduler dossier). UCL's fair-use rule (R8) is a *staff judgment
call* — "staff may kill jobs impairing shared resources" — so the realistic adverse event is not a
crash but an ADMINISTRATOR deliberately killing our jobs.

The driver's existing failure handling is exactly wrong for that case: it treats task death as
transport/node failure and REQUEUES (``ledger.MAX_RETRIES``). Blind resubmission immediately after
an administrative kill is the behaviour that escalates "your jobs were killed" into "your account
was suspended" — and the standing instruction is that **preserving Myriad access outranks
throughput**. So the system must be able to tell the two apart and RETREAT rather than fight.

THE DISCRIMINATOR. The three ways our tasks die have different fingerprints, and the epilogue
ledger the jobscript already writes (``{"task","host","gpu","rc","secs"}`` per task) carries
exactly the fields needed to separate them:

======================  =========================================================================
node failure            deaths concentrated on ONE host (the node fell over); ``-r y`` requeue is
                        correct and costs nobody anything.
walltime kill           ``rc`` 137/143 with ``secs`` ~= the requested ``h_rt``; expected, per-task,
                        uncorrelated across hosts. Fix the sizing, not the footprint.
**administrative kill** MANY deaths in a SHORT window spread across MANY DISTINCT hosts, none of
                        them walltime-proximate. No node failure looks like this: a human ran
                        ``qdel`` over our job list.
======================  =========================================================================

THE RESPONSE IS ASYMMETRIC BY DESIGN. False-positive cost (we pause a healthy campaign and ask a
human) is a few hours of throughput on a run that has ~7 days of calendar slack. False-negative
cost (we hammer the scheduler straight after an admin kill) is the account. So the detector is
deliberately tuned to retreat on ambiguous evidence, and the concurrency cap it sets is
**MONOTONE NON-INCREASING**: nothing in this module ever raises the cap back up. Only a human
clearing the incident file does that.

NOTE: RETREAT IS *NOT* DEPRIORITISATION. Reducing our FOOTPRINT (submitting fewer concurrent jobs) is
a courtesy lever entirely under our control. Lowering our SGE *priority* (``qalter -p <negative>``)
is FORBIDDEN, always, by the project's standing order — and this module never does it. The
two are different things and must not be confused.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

_LOG = logging.getLogger(__name__)

__all__ = [
    "KillVerdict",
    "classify_task_deaths",
    "retreat_cap",
    "plan_footprint",
    "write_incident",
    "incident_blocks_submission",
    "clear_incident",
    "INCIDENT_FILENAME",
    "ABSOLUTE_CORE_CEILING",
]

#: RUNAWAY BACKSTOP — deliberately NOT a policy limit (revised 2026-07-26: *"if it would
#: offer us say 2000, why would we reject?"*).
#:
#: THE POLICY IS: **take whatever the scheduler grants, minus a reserve for other users.** SGE's
#: fair-share IS the allocation mechanism — if it dispatches our jobs it has already arbitrated our
#: entitlement against every other user, so declining granted capacity is pure loss with no
#: principled justification. The earlier 640 (then 2560) hard caps were conservatism, not policy;
#: they cost campaign speed to buy nothing, because the two constraints that actually protect our
#: access are :data:`FREE_CORE_RESERVE` (we can never be the reason someone waits for a CPU slot)
#: and the administrative-kill retreat below.
#:
#: Note the arithmetic that makes this uncontroversial: the campaign's total work is FIXED
#: (1,800 + 71n trainings — corrected 2026-07-26 from a stale "1,740 + 69n"; see
#: :func:`lanes.total_trainings`), so total core-hours are the same however fast we take them — 636 cores
#: x 23.4 d ~= 337k core-h vs 2,560 x 5.8 d ~= 356k. Going faster does not consume more of Myriad;
#: it consumes the same amount in a shorter window, and UCL's own criterion explicitly permits
#: impact that is *"not of long duration"*. What the rate buys is schedule robustness: at 636 cores
#: the 31-day GO->Aug-27 window holds only ~7 days of slack.
#:
#: This value therefore exists ONLY to stop a bug (a loop that submits 100k tasks) from running
#: away. It is set far above any rate the scheduler has ever granted us — the observed plateau is
#: ~75 concurrent jobs — so in normal operation it never binds.
ABSOLUTE_CORE_CEILING = 8192

#: THE COURTESY GUARANTEE, and the reason a high ceiling is defensible: we never consume the last
#: cores. Whatever else the arithmetic says, leave at least this many cores free cluster-wide, so
#: our campaign can never be the reason another user waits for a plain CPU slot. This binds BEFORE
#: the ceiling — on a busy cluster it drives the target to zero on its own.
FREE_CORE_RESERVE = 1000

INCIDENT_FILENAME = "MYRIAD_KILL_INCIDENT.json"

#: Deaths must be at least this many, within :data:`WINDOW_SECS`, across at least
#: :data:`MIN_DISTINCT_HOSTS` hosts, before we call it an administrative kill. Tuned to retreat
#: readily: 8 INFRASTRUCTURE tasks dying across 4 nodes inside 5 minutes has no benign explanation.
#: (2026-07-28: "infrastructure" is load-bearing in that sentence — see :data:`_APPLICATION_EXIT_RC`.
#: There IS a benign explanation for 8 *application* exits in that window, and it is a registered
#: deliverable of this study.)
MIN_DEATHS = 8
MIN_DISTINCT_HOSTS = 4
WINDOW_SECS = 300.0

#: A death is "walltime-proximate" (i.e. an expected h_rt kill, not an admin kill) if the task ran
#: for at least this fraction of its requested walltime.
WALLTIME_FRACTION = 0.9

#: Exit codes that are an APPLICATION verdict, not an infrastructure death — excluded from the
#: administrative-kill burst test.
#:
#: ``run_one.main`` ends with ``return 0 if n_ok == len(rows) else 1``, and its own comment records
#: that a sandbox/contract reject "surfaced as a bare rc=1". An LLM writing a reward that fails the
#: sandbox is THE PHENOMENON UNDER STUDY — ``qwen3.5-9b``'s measured ~17 % gate-pass rate makes such
#: rejects frequent BY DESIGN — and a batch of them lands fast, on many hosts at once, which is
#: precisely the shape this detector calls an administrative qdel.
#:
#: MEASURED on the RUN 1 archive: the worst 300 s burst of rc=1 deaths was **7 across 6 hosts** —
#: ONE death under ``MIN_DEATHS`` and already past ``MIN_DISTINCT_HOSTS``. And only **61 %** of
#: RUN 1's candidate slots ever reached a node — 621 records + 59 genuine rejects = 680 of ~1,119 —
#: because the cross-line reject collision suppressed **439 (39 %)** UNSUBMITTED and unjudged. With
#: that defect fixed the flow is ~**1.65x** denser, so the threshold would be crossed — writing
#: ``MYRIAD_KILL_INCIDENT.json`` and hard-blocking submission on all twelve lines until a human
#: clears it. Fixing one defect unmasked the next.
#:
#: Excluding rc=1 cannot weaken admin-kill detection, and that is the point rather than a trade-off:
#: an administrative ``qdel`` terminates by SIGNAL (rc 137/143) or the job never starts (rc 126/127);
#: a clean ``1`` can only be produced by our own ``return 1``. It therefore carries no evidence about
#: administrative action in either direction. ``ledger.host_task_counts`` already reached the same
#: conclusion for the bad-node detector (``_NOT_A_NODE_FAULT_RC``) and deferred rc=1 to "the
#: killswitch" — which was counting it as an unexplained death. Neither owner treated it as what it
#: is. Transient application failures remain fully handled elsewhere: the driver retries them and
#: ledgers ``retries_exhausted``, and permanent ones leave a ``_rejects`` marker.
_APPLICATION_EXIT_RC: frozenset[int] = frozenset({1})


@dataclass(frozen=True)
class KillVerdict:
    """The classification plus the action the driver must take.

    Attributes
    ----------
    classification : {"ok", "node_failure", "walltime", "admin_kill"}
    action : {"continue", "requeue", "retreat"}
        ``"retreat"`` means: STOP submitting, do NOT requeue the dead tasks, apply
        ``new_core_cap``, write the incident file, and alert a human.
    reason : str
        Human-readable evidence — goes straight into the incident file and the alert.
    n_deaths, n_hosts : int
        The evidence counts behind the verdict.
    new_core_cap : int or None
        The reduced self-imposed concurrency cap (``None`` when no change is required).
    n_undated : int
        Rows excluded because they carry no usable ``ts`` and so cannot be placed in the burst
        window (#56). Non-zero means the admin-kill detector ran DEGRADED on this input — surfaced
        rather than hidden, because under-detecting is the error that costs Myriad access.
    """

    classification: str
    action: str
    reason: str
    n_deaths: int = 0
    n_hosts: int = 0
    new_core_cap: int | None = None
    n_undated: int = 0


def _recent(
    events: Iterable[dict[str, Any]], now: float, window: float
) -> tuple[list[dict[str, Any]], int]:
    """``(rows inside the window, n_undated)`` — an UNDATED row is never called recent.

    NOTE: This previously admitted a row whose ``ts`` was missing (``ts is None or ...``), which made the
    window INERT on real data: the jobscript's epilogue line emitted ``task/host/gpu/rc/secs`` and no
    timestamp at all, so EVERY death ever written counted as "within 300s". MEASURED (deep review
    2026-07-26, #56): 12 ordinary failures scattered over 20 DAYS across 6 hosts were classified
    ``admin_kill`` -> ``retreat``, which halves the core cap AND writes the incident file that blocks
    all submission until a human clears it. Over a 23-day campaign that false positive was close to
    certain.

    Burstiness is inherently temporal, so a row we cannot place in time is not evidence of a burst
    and is EXCLUDED. But excluding it silently would trade a false positive for a false NEGATIVE on
    the one guard protecting Myriad access, so the count is returned and reported by the caller
    (same discipline as the loop-80 extrapolation counter: never fabricate, always make the gap
    visible). The emitter now stamps ``ts``; this keeps a legacy or torn ledger degrading LOUDLY.
    """
    out: list[dict[str, Any]] = []
    n_undated = 0
    for e in events:
        ts = e.get("ts")
        if ts is None:
            n_undated += 1
            continue
        try:
            fresh = float(ts) >= now - window
        except (TypeError, ValueError):
            n_undated += 1
            continue
        if fresh:
            out.append(e)
    return out, n_undated


def classify_task_deaths(
    events: Sequence[dict[str, Any]],
    *,
    now: float | None = None,
    h_rt_secs: float | None = None,
    current_core_cap: int | None = None,
    window_secs: float = WINDOW_SECS,
    min_deaths: int = MIN_DEATHS,
    min_hosts: int = MIN_DISTINCT_HOSTS,
) -> KillVerdict:
    """Classify a burst of task deaths from the epilogue-ledger rows.

    Parameters
    ----------
    events : sequence of dict
        Epilogue rows, as the jobscript emits them: ``{"host", "rc", "secs"}`` plus an optional
        ``"ts"`` (epoch seconds). Rows with ``rc == 0`` are ignored — they are successes.
    now : float, optional
        Clock override for tests (default :func:`time.time`).
    h_rt_secs : float, optional
        The batch's requested walltime. When given, deaths that ran >= ``WALLTIME_FRACTION`` of it
        are classified as expected walltime kills and EXCLUDED from the admin-kill evidence — this
        is what stops a badly-sized ``h_rt`` from masquerading as an administrative action.
    current_core_cap : int, optional
        The footprint cap in force. On a ``retreat`` verdict the returned ``new_core_cap`` is
        :func:`retreat_cap` of it.

    Returns
    -------
    KillVerdict

    Notes
    -----
    Ordering matters. Walltime deaths are removed FIRST (they are benign and per-task), then
    single-host concentration is checked (a node failure, where ``-r y``/requeue is correct), and
    only a multi-host burst that survives both filters is called an administrative kill.
    """
    now = time.time() if now is None else float(now)
    in_window, n_undated = _recent(events, now, window_secs)
    if n_undated:
        # Loud, because the alternative to a false positive here is a false NEGATIVE on the guard
        # that protects Myriad access (#56). An undated ledger means the burst detector is BLIND.
        _LOG.warning(
            "killswitch_UNDATED_EPILOGUE_ROWS: %d of %d rows carry no usable 'ts' and were EXCLUDED "
            "from the %.0fs burst window — the admin-kill detector is degraded for those rows. "
            "Ensure the jobscript epilogue stamps 'ts' (it does since 2026-07-26); a legacy ledger "
            "written before that will under-detect.",
            n_undated, len(list(events)) if not isinstance(events, Sequence) else len(events),
            window_secs,
        )
    deaths = [e for e in in_window if int(e.get("rc", 0) or 0) != 0]

    # 2026-07-28: drop APPLICATION exits before the burst test. A sandbox/contract reject is our own
    # `run_one` returning 1 — the LLM-authoring failure this study MEASURES, not a cluster event —
    # and a batch of them is fast, multi-host and bursty, i.e. shaped exactly like an administrative
    # qdel. See `_APPLICATION_EXIT_RC` for the measurement (RUN 1's worst rc=1 burst: 7 deaths / 6
    # hosts, one under the threshold, with 88 % of candidates never even submitted).
    app_exits = [e for e in deaths if int(e.get("rc", 0) or 0) in _APPLICATION_EXIT_RC]
    if app_exits:
        deaths = [e for e in deaths if int(e.get("rc", 0) or 0) not in _APPLICATION_EXIT_RC]
        # Excluded, never silent: a mass self-inflicted failure must stay VISIBLE even though it is
        # not an admin kill. This is the one signal that would otherwise vanish between two
        # detectors, since ledger.host_task_counts already ignores rc=1 as "not a node fault".
        _LOG.warning(
            "killswitch_APPLICATION_EXITS_EXCLUDED: %d task(s) in the %.0fs window exited rc in %s "
            "across %d host(s) — these are OUR OWN application verdicts (sandbox/contract rejects: "
            "run_one returns 1 when a row fails), NOT infrastructure deaths, so they are excluded "
            "from the administrative-kill test. Expected and frequent by design on the weaker legs. "
            "If this count is large AND records are not accumulating, the problem is the authoring "
            "path, not the cluster.",
            len(app_exits), window_secs, sorted(_APPLICATION_EXIT_RC),
            len({e.get("host") for e in app_exits}),
        )

    if not deaths:
        return KillVerdict(
            "ok", "continue",
            "no infrastructure task deaths in the window"
            + (f" ({len(app_exits)} application exit(s) excluded)" if app_exits else "")
            + (f" ({n_undated} undated rows excluded — detector degraded)" if n_undated else ""),
            n_undated=n_undated,
        )

    if h_rt_secs:
        non_walltime = [e for e in deaths
                        if float(e.get("secs", 0) or 0) < WALLTIME_FRACTION * float(h_rt_secs)]
        n_wall = len(deaths) - len(non_walltime)
        if not non_walltime:
            return KillVerdict(
                "walltime", "requeue",
                f"all {n_wall} deaths ran >= {WALLTIME_FRACTION:.0%} of h_rt "
                f"({h_rt_secs:.0f}s) — expected walltime kills, size h_rt up",
                n_deaths=len(deaths), n_hosts=len({e.get("host") for e in deaths}))
        deaths = non_walltime

    hosts = {str(e.get("host") or "?") for e in deaths}
    if len(hosts) == 1:
        return KillVerdict(
            "node_failure", "requeue",
            f"{len(deaths)} deaths all on ONE host ({next(iter(hosts))}) — a node failure; "
            "`-r y`/driver requeue is the correct response",
            n_deaths=len(deaths), n_hosts=1)

    if len(deaths) >= min_deaths and len(hosts) >= min_hosts:
        return KillVerdict(
            "admin_kill", "retreat",
            f"{len(deaths)} task deaths across {len(hosts)} DISTINCT hosts within "
            f"{window_secs:.0f}s, none walltime-proximate — no node failure has this shape; "
            "treating as an administrative qdel. STOP submitting, do NOT requeue, alert a human.",
            n_deaths=len(deaths), n_hosts=len(hosts),
            new_core_cap=retreat_cap(current_core_cap))

    return KillVerdict(
        "node_failure", "requeue",
        f"{len(deaths)} deaths across {len(hosts)} hosts — below the admin-kill threshold "
        f"({min_deaths} deaths / {min_hosts} hosts); treating as ordinary failures",
        n_deaths=len(deaths), n_hosts=len(hosts))


def retreat_cap(current: int | None, *, floor: int = 64) -> int | None:
    """Halve the self-imposed concurrency cap, never below ``floor``; MONOTONE NON-INCREASING.

    Returns ``None`` when there is no cap to reduce. The floor exists so a retreat parks the
    campaign at a genuinely small, unobtrusive footprint rather than at zero — a stopped campaign
    that nobody notices is its own failure mode. Nothing here ever raises the cap: only a human
    clearing the incident file does that.
    """
    if not current or current <= 0:
        return None
    return max(int(floor), int(current) // 2)


def plan_footprint(
    *,
    free_cores: int,
    pending_jobs: int,
    retreat_cap_cores: int | None = None,
    ceiling: int = ABSOLUTE_CORE_CEILING,
) -> tuple[int, str]:
    """How many cores to hold RIGHT NOW, given live cluster pressure. Returns ``(cores, why)``.

    This is the PROACTIVE half of the system, and it is what makes "go as fast as Myriad allows"
    and "never jeopardise the account" the same policy rather than opposed ones.

    The insight from the 2026-07-26 capacity probe: **~5,000 cores sat FREE while ~2,000 jobs
    pended** — the pending queue wants GPUs and large memory, not plain cores. So the courteous
    move and the fast move coincide: take plain CPU cores when they are idle, and stand down when
    they are contested. A fixed footprint cannot do that; this scales with what is actually spare.

    **Take what the scheduler grants.** The share is deliberately AGGRESSIVE — the protection is
    the reserve floor below, not a timid share — because declining capacity SGE has already
    arbitrated in our favour costs campaign speed and buys nothing:

    ========================  ==========  ==================================================
    condition                 share       rationale
    ========================  ==========  ==================================================
    quiet (< 500 pending)     90%         nobody is waiting; idle cores are pure waste
    normal                    70%         plenty spare beyond the reserve
    busy (>= 2000 pending)    50%         still take half the genuine slack
    ========================  ==========  ==================================================

    In practice the SCHEDULER, not this function, is usually the binding constraint: the measured
    plateau is ~75 concurrent jobs however deep the queue, and 32-core requests do not place at all.

    Four clamps then apply, TIGHTEST WINS, in increasing order of authority:

    1. the pressure-scaled share above;
    2. :data:`FREE_CORE_RESERVE` — **never consume the last cores.** This is the courtesy guarantee
       that makes a high ceiling defensible: our campaign can never be the reason someone else
       waits for a plain CPU slot. On a tight cluster this alone drives the target to zero;
    3. the absolute self-imposed ``ceiling``;
    4. any standing ``retreat_cap_cores`` — **always wins**, because it encodes a decision from a
       kill incident that a human has not yet reversed.
    """
    free = max(0, int(free_cores))
    pending = max(0, int(pending_jobs))
    if pending < 500:
        share, regime = 0.90, "quiet"
    elif pending < 2000:
        share, regime = 0.70, "normal"
    else:
        share, regime = 0.50, "busy"
    target = int(free * share)
    why = f"{regime} cluster ({pending} pending, {free} free cores) -> {share:.0%} share = {target}"
    headroom = free - FREE_CORE_RESERVE
    if target > headroom:
        target = max(0, headroom)
        why += f"; cut to {target} to leave the {FREE_CORE_RESERVE}-core reserve free for others"
    if target > ceiling:
        target, why = ceiling, why + f"; clamped to the self-imposed ceiling {ceiling}"
    if retreat_cap_cores is not None and target > retreat_cap_cores:
        target = int(retreat_cap_cores)
        why += f"; OVERRIDDEN by the standing retreat cap {retreat_cap_cores} (uncleared incident)"
    return max(0, target), why


def write_incident(root: str | Path, verdict: KillVerdict, *, extra: dict[str, Any] | None = None) -> Path:
    """Record a retreat durably next to the archive, so a driver restart cannot un-know it.

    The file is the SUBMISSION GATE (:func:`incident_blocks_submission`) — the same
    human-in-the-loop pattern as the tier-1 review gate: a machine may stop the campaign, but only
    a person may restart it.
    """
    p = Path(root) / INCIDENT_FILENAME
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": time.time(),
        "classification": verdict.classification,
        "action": verdict.action,
        "reason": verdict.reason,
        "n_deaths": verdict.n_deaths,
        "n_hosts": verdict.n_hosts,
        "new_core_cap": verdict.new_core_cap,
        "cleared": False,
        "what_to_do": (
            "1) Do NOT resubmit. 2) Check for mail from rc-support@ucl.ac.uk. 3) If UCL made "
            "contact, reply and STOP until it is resolved — do not negotiate via the scheduler. "
            "4) If it was not UCL, relaunch with --resume at the reduced cap. 5) Clear this file "
            "with clear_incident() only after a human decision."
        ),
    }
    if extra:
        payload.update(extra)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(p)
    return p


def incident_blocks_submission(root: str | Path) -> tuple[bool, str]:
    """``(blocked, reason)`` — True while an uncleared incident sits next to the archive.

    Fail-SAFE: an unreadable/corrupt incident file BLOCKS. The whole point is that the expensive
    error is submitting when we should not, so ambiguity resolves to "do not submit".
    """
    p = Path(root) / INCIDENT_FILENAME
    if not p.is_file():
        return False, ""
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — unreadable => block (fail-safe)
        return True, f"{INCIDENT_FILENAME} is present but unreadable ({exc}); refusing to submit"
    if data.get("cleared") is True:
        return False, ""
    return True, (
        f"Myriad kill-incident OPEN ({data.get('classification')}): {data.get('reason')} "
        f"— submission is blocked until a human clears {p}."
    )


def clear_incident(root: str | Path, *, who: str, note: str = "") -> Path:
    """Mark an incident cleared (the human-in-the-loop release). ``who`` is recorded."""
    p = Path(root) / INCIDENT_FILENAME
    data = json.loads(p.read_text(encoding="utf-8"))
    data.update({"cleared": True, "cleared_by": str(who), "cleared_ts": time.time(),
                 "cleared_note": str(note)})
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p
