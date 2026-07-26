"""Myriad live telemetry — the SENSOR half of the adaptive allocation system (2026-07-24).

Collects one structured snapshot of everything the allocation advisor needs, in ONE ssh
round-trip: per-pool free GPUs, cluster contention, our own jobs' states/priorities, and the
U/V access-probe verdict. Read-only against the cluster; NEVER touches priorities.

Design: the PARSERS are pure functions over command text (fully unit-tested offline); the
collector is a thin ssh shell. The dossier is the
ground truth for the semantics encoded here (pools, dispatch mechanics, the two-regime doctrine).
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

#: Node-prefix -> pool letter map (dossier §2). EF is ONE scheduling pool (`-ac allow=EF`).
_PREFIX_POOL = (("node-e", "EF"), ("node-f", "EF"), ("node-l", "L"),
                ("node-u", "U"), ("node-v", "V"))

#: Registered per-training speed factors vs V100 (dossier §2; canary re-measures).
POOL_SPEED = {"EF": 1.0, "L": 1.9, "U": 2.1, "V": 2.1}

#: The U/V access probes submitted 2026-07-24 (runbook §10 best-hardware protocol).
PROBE_JOBS = {"10293": "U", "10294": "V", "10295": "EF-control"}

#: Contention threshold (eligible `qw` jobs cluster-wide) separating the two chunking regimes
#: (dossier §3b). Calibration: the 2026-07-23/24 jam measured ~3,000 qw; quiet nights run <500.
CONTENDED_QW = 1500


@dataclass
class Snapshot:
    """One timestamped telemetry frame (all fields safe to archive/print — no secrets)."""

    ts: str
    pool_free: dict[str, int]            # pool -> free GPU count (from `qhost -F gpu`)
    cluster_qw: int                      # eligible pending jobs cluster-wide
    cluster_users: int                   # distinct users with pending jobs
    our_jobs: list[dict]                 # [{id, prior, state}] for our user
    probe_states: dict[str, str]         # probe job id -> qstat state ("r"/"qw"/"done"/"absent")
    errors: list[str] = field(default_factory=list)
    # 2026-07-26 CPU lane: free CORES per node-type letter ("d", "b", "t", ...). The CPU lane is
    # sized in cores, not GPUs, so pool_free (a GPU count) cannot answer it. Defaulted so every
    # existing caller/archived frame stays valid.
    cpu_free: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# --------------------------------------------------------------------------------------- #
# Pure parsers (unit-tested offline)                                                       #
# --------------------------------------------------------------------------------------- #
def parse_qhost_gpu(text: str) -> dict[str, int]:
    """``qhost -F gpu`` -> {pool: free_gpu_count}.

    The listing alternates host lines (``node-xNNa-…``) with resource lines
    (``Host Resource(s):      hc:gpu=N``); a missing resource line means 0 free.
    """
    free: dict[str, int] = {}
    current_pool: str | None = None
    for line in text.splitlines():
        s = line.strip()
        m = re.match(r"(node-[a-z]\d+a?-?\d*)", s)
        if m:
            current_pool = None
            for prefix, pool in _PREFIX_POOL:
                if m.group(1).startswith(prefix):
                    current_pool = pool
                    break
            continue
        g = re.search(r"hc:gpu=(\d+(?:\.\d+)?)", s)
        if g and current_pool:
            free[current_pool] = free.get(current_pool, 0) + int(float(g.group(1)))
            current_pool = None
    return free


def parse_cpu_free(text: str) -> dict[str, int]:
    """``"<type> <free_cores>"`` lines -> ``{node_type: free_cores}``.

    Tolerant by design (telemetry must degrade, never raise): unparseable lines are skipped and a
    missing/empty section yields ``{}``, which the CPU-lane advisory reads as "unknown" rather than
    "zero free" — the safe direction, since a false zero would silently stall the lane.
    """
    free: dict[str, int] = {}
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        pool, n = parts
        if len(pool) != 1 or not pool.isalpha():
            continue
        try:
            free[pool] = int(n)
        except ValueError:
            continue
    return free


def parse_cluster_pending(text: str) -> tuple[int, int]:
    """``qstat -u '*' -s p`` -> (eligible qw job count, distinct users)."""
    qw = 0
    users: set[str] = set()
    for line in text.splitlines()[2:]:
        cols = line.split()
        if len(cols) < 5:
            continue
        # Audit m9: a job NAME containing spaces shifts the whitespace columns — detect the
        # eligible state by TOKEN (a standalone " qw ", never hqw/Eqw), not by position.
        if re.search(r"\s(qw)\s", line) and not re.search(r"\s\w*[hE]\w*qw\s", line):
            qw += 1
        users.add(cols[3])
    return qw, len(users)


def parse_our_jobs(text: str) -> list[dict]:
    """``qstat`` (our user) -> [{id, prior, state, slots}] (one row per line, arrays included).

    ``slots`` added 2026-07-26 so the GO-day ACCUMULATION CURVE is reconstructable from the
    archived telemetry (runbook §11.5): without it the log records job COUNT but not CORES, and the
    one projection the campaign plan still rests on — that ~8.5 h tasks accumulate to ~2,000–3,000
    cores rather than churning at ~636 — could not be checked against reality.

    NOTE: THE COLUMN POSITION DIFFERS BY STATE, which is the trap: a RUNNING row carries a queue name
    (``Queue@node-…``) that a PENDING row does not, so slots sit at index 8 when running and 7 when
    pending. Reading a fixed index silently yields the date/time field for half the rows.
    """
    out: list[dict] = []
    for line in text.splitlines()[2:]:
        cols = line.split()
        if len(cols) >= 5 and cols[0].isdigit():
            state = cols[4]
            idx = 8 if state and state[0] in "rRtsS" else 7
            slots = 0
            if len(cols) > idx:
                try:
                    slots = int(cols[idx])
                except ValueError:
                    slots = 0
            out.append({"id": cols[0], "prior": float(cols[1]), "state": state, "slots": slots})
    return out


def running_slots(our_jobs: list[dict]) -> tuple[int, int]:
    """``(running_job_count, running_core_count)`` — the accumulation-curve datapoint."""
    jobs = [j for j in our_jobs if str(j.get("state", ""))[:1] in ("r", "R", "t")]
    return len(jobs), sum(int(j.get("slots", 0) or 0) for j in jobs)


def accumulation_report(log_path: str | Path | None = None, *, hours: float = 3.0) -> dict:
    """Has our concurrency PLATEAUED, or is it still climbing? — the GO-day canary check.

    The campaign plan's one un-measured projection is that ~8.5 h tasks ACCUMULATE (the ~75-job
    plateau seen with 20-min probe jobs was a flow equilibrium, `concurrent = dispatch_rate ×
    duration`). This reads the archived telemetry and answers it from data instead of hope:
    compares the last third of the window against the first third and calls ``climbing`` /
    ``plateaued`` / ``declining``. Feed the observed plateau back into
    ``lanes.plan_lanes(cpu_cores=…)`` to re-forecast the reachable rung.
    """
    import calendar as _calendar
    import json as _json

    path = Path(log_path) if log_path else _LOG_PATH
    if not path.is_file():
        return {"status": "no-data", "reason": f"{path} absent"}
    cutoff = time.time() - hours * 3600.0
    pts: list[tuple[float, int]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = _json.loads(line)
        except Exception:  # noqa: BLE001 — a truncated tail line must not kill the report
            continue
        jobs = row.get("our_jobs") or []
        if not any("slots" in j for j in jobs):
            continue                      # frames written before slots were recorded
        try:
            # UTC in, UTC out (deep review 2026-07-26, #59). Frames are stamped with
            # ``time.gmtime()`` and the trailing ``Z`` says so, but ``time.mktime`` interprets its
            # struct as LOCAL time — so every frame read back one UTC-offset OLDER than it is.
            # MEASURED under BST (the campaign runs Jul-Aug, DST active): a −3600 s shift silently
            # truncated the window by an hour, so a 3.0 h watch of 18 frames reported n=12 and an
            # ``early_mean_cores`` of 250 against a true 150 — the climb ratio read 1.64 instead of
            # 2.73. Near the 1.15/0.85 thresholds that can FLIP the GO-day verdict, and
            # ``early_mean_cores`` is a number the operator re-forecasts the rung from.
            ts = _calendar.timegm(time.strptime(str(row.get("ts", "")), "%Y-%m-%dT%H:%M:%SZ"))
        except Exception:  # noqa: BLE001
            continue
        if ts >= cutoff:
            pts.append((ts, running_slots(jobs)[1]))
    if len(pts) < 6:
        return {"status": "insufficient", "n": len(pts),
                "reason": "need >=6 slot-bearing frames; keep --watch running"}
    pts.sort()
    third = max(1, len(pts) // 3)
    early = sum(c for _, c in pts[:third]) / third
    late = sum(c for _, c in pts[-third:]) / third
    peak = max(c for _, c in pts)
    if late > early * 1.15:
        status = "climbing"
    elif late < early * 0.85:
        status = "declining"
    else:
        status = "plateaued"
    return {"status": status, "n": len(pts), "early_mean_cores": round(early),
            "late_mean_cores": round(late), "peak_cores": peak,
            "advice": ("still accumulating — do NOT re-forecast the rung yet"
                       if status == "climbing" else
                       f"re-forecast from ~{round(late)} cores via lanes.plan_lanes")}


#: The one verdict that admits a pool. Kept as a module constant so the ADMISSION TEST lives beside
#: the vocabulary it tests (deep review 2026-07-26, #60) — see :func:`pool_is_usable`.
USABLE_VERDICT = "RUNNING (USABLE)"


def pool_is_usable(verdict: str | None) -> bool:
    """True only for the ONE verdict that admits a pool — a POSITIVE test, never a word scan.

    ``allocation.usable_pools`` previously admitted U/V on ``"USABLE" in verdict``, a case-sensitive
    substring of a human-readable sentence written in THIS module. That is fail-OPEN by
    construction: the negative verdicts happen to spell it lowercase ("NOT usable"), so the gate is
    correct today, but REPRODUCED 2026-07-26 — a verdict reworded to ``"NOT USABLE"`` or
    ``"UNUSABLE"`` (the natural way to add emphasis) makes the pool JOIN. That would schedule the
    campaign onto a pool that never grants, so the jobs pend forever, or onto one we were told we may
    not use — the exact hazard Audit M1 added the "absent is NOT auto-usable" rule for.

    Matching the positive explicitly cannot fail open on a reworded negation: a new or edited
    negative verdict simply is not this string, so it is refused by default.
    """
    return (verdict or "").strip() == USABLE_VERDICT


def probe_verdicts(our_jobs: list[dict], *, pending_hours: float,
                   restricted_after_h: float = 48.0) -> dict[str, str]:
    """The U/V experiment verdict per probe (runbook §10 branch).

    ``run``/``done`` (absent after having been submitted) => the pool answered => USABLE.
    still pending past ``restricted_after_h`` => effectively RESTRICTED. Else PENDING.
    """
    states = {j["id"]: j["state"] for j in our_jobs}
    verdicts: dict[str, str] = {}
    for jid, pool in PROBE_JOBS.items():
        st = states.get(jid)
        if st is None:
            # Audit M1 (2026-07-24): absent must NEVER read as usable — a qdel'd/aged-out probe
            # would silently admit a pool that never grants (CRN blocks pending forever). A
            # positive confirmation (qacct/archived hostname) is required to flip this to usable.
            verdicts[pool] = "GONE from qstat — verify via qacct before enabling (NOT auto-usable)"
        elif st.lower().startswith("e"):
            # Audit m6: Eqw = OUR probe errored; a diagnosis, not a pool restriction.
            verdicts[pool] = f"probe ERROR ({st}) — investigate/resubmit; NOT usable"
        elif st.lower().startswith("r"):
            # Audit m6: case-insensitive so Rr/Rt (running-after-restart) still count as running.
            verdicts[pool] = USABLE_VERDICT   # the ONE admitting verdict (see pool_is_usable, #60)
        elif pending_hours >= restricted_after_h:
            verdicts[pool] = "pending>%dh (effectively RESTRICTED)" % int(restricted_after_h)
        else:
            verdicts[pool] = f"pending ({st})"
    return verdicts


# --------------------------------------------------------------------------------------- #
# The collector (thin ssh shell — one round trip)                                          #
# --------------------------------------------------------------------------------------- #
_REMOTE = (
    # Audit M3: each section emits its OWN rc sentinel — a failed middle command can no longer
    # parse as "legitimately empty" (rc was previously only the LAST command's).
    "echo '#QHOST'; qhost -F gpu 2>/dev/null; echo \"#RC:QHOST=$?\"; "
    "echo '#PENDING'; qstat -u '*' -s p 2>/dev/null; echo \"#RC:PENDING=$?\"; "
    "echo '#MINE'; qstat 2>/dev/null; echo \"#RC:MINE=$?\"; "
    # CPU FREE CORES per node type (2026-07-26). `qhost` gives NCPU per host but no used count,
    # and `qstat -f` gives used slots per QUEUE INSTANCE — a host appears in ~40 queues, so the
    # per-host used total is the SUM over its instances. Hence the join. Hostnames must be
    # normalised: qhost prints `node-d00a-001`, qstat prints `Queue@node-d00a-001.data.priv` —
    # the un-normalised join silently matches nothing and reports every core as free (a real bug
    # hit while measuring on 2026-07-26).
    "echo '#CPUFREE'; { qhost | awk 'NR>3 && $1 ~ /^node-/ {h=$1; sub(/\\..*/,\"\",h); print h, $3}' "
    "| sort > /tmp/_llmrp_n.txt; qstat -f -q '*' 2>/dev/null "
    "| awk '/@node-/ {split($1,q,\"@\"); h=q[2]; sub(/\\..*/,\"\",h); split($3,r,\"/\"); u[h]+=r[2]} "
    "END{for(h in u) print h, u[h]}' | sort > /tmp/_llmrp_u.txt; "
    "join /tmp/_llmrp_n.txt /tmp/_llmrp_u.txt "
    "| awk '{f=$2-$3; if(f<0)f=0; split($1,a,\"-\"); t=substr(a[2],1,1); tot[t]+=f} "
    "END{for(k in tot) print k, tot[k]}' | sort; "
    "rm -f /tmp/_llmrp_n.txt /tmp/_llmrp_u.txt; } 2>/dev/null; echo \"#RC:CPUFREE=$?\""
)


def collect(host: str = "myriad", *, probe_age_hours: float = 0.0,
            timeout: int = 45) -> Snapshot:
    """One ssh round trip -> a parsed :class:`Snapshot` (errors captured, never raised)."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    errors: list[str] = []
    sections = {"QHOST": "", "PENDING": "", "MINE": "", "CPUFREE": ""}
    try:
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=20", host, _REMOTE],
                           capture_output=True, text=True, timeout=timeout)
        current = None
        section_rc: dict[str, int] = {}
        for line in (r.stdout or "").splitlines():
            m_rc = re.match(r"#RC:(\w+)=(\d+)", line)
            if m_rc:
                section_rc[m_rc.group(1)] = int(m_rc.group(2))
                continue
            if line.startswith("#") and line[1:] in sections:
                current = line[1:]
                continue
            if current:
                sections[current] += line + "\n"
        if r.returncode != 0:
            errors.append(f"ssh rc={r.returncode}: {(r.stderr or '')[-200:]}")
        for name in sections:
            rc = section_rc.get(name)
            if rc is None:
                errors.append(f"section {name}: no rc sentinel (transport truncated?)")
            elif rc not in (0, 1):
                # qstat exits 1 with no jobs on some builds — rc 1 with empty output is legal;
                # anything else is a genuine section failure (audit M3).
                errors.append(f"section {name}: rc={rc}")
    except Exception as exc:  # noqa: BLE001 — telemetry must degrade, never crash a caller
        errors.append(f"{type(exc).__name__}: {exc}")

    qw, users = parse_cluster_pending(sections["PENDING"])
    mine = parse_our_jobs(sections["MINE"])
    return Snapshot(
        ts=ts,
        pool_free=parse_qhost_gpu(sections["QHOST"]),
        cluster_qw=qw,
        cluster_users=users,
        our_jobs=mine,
        probe_states=probe_verdicts(mine, pending_hours=probe_age_hours),
        errors=errors,
        cpu_free=parse_cpu_free(sections["CPUFREE"]),
    )


def append_log(snap: Snapshot, path: str | Path | None = None) -> None:
    """Append one frame to the local telemetry log (atomic-enough single write)."""
    p = Path(path) if path is not None else _LOG_PATH  # audit m7: repo-anchored default
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(snap)) + "\n")


# Audit m7: anchor ALL persistence to the repo root — a CWD-relative path silently forked the
# state/log into stray outputs/ dirs (losing hysteresis + trend) when run from elsewhere.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_STATE_PATH = _REPO_ROOT / "outputs" / "allocation_state.json"
_LOG_PATH = _REPO_ROOT / "outputs" / "myriad_telemetry.jsonl"


def load_state(path: str | Path = _STATE_PATH) -> dict:
    """The advisor's persisted memory (prev regime + last plan fields). {} when absent/corrupt —
    a broken state file must never break the advisor (it just loses hysteresis for one cycle)."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(state: dict, path: str | Path = _STATE_PATH) -> None:
    """Persist the advisor's memory (atomic replace so a mid-write kill can't corrupt it)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(p)


# --------------------------------------------------------------------------------------- #
# Self-measurement (the SMART half: the system computes its own facts from the archives)   #
# --------------------------------------------------------------------------------------- #
def measure_rate(archive_root: str | Path, *, window_hours: float = 24.0) -> tuple[float, int]:
    """(trainings/day, records-in-window) measured from ``record.json`` mtimes under a root.

    The rate the ETA math uses is OUR OWN realized throughput — never an assumption. Returns
    (0.0, 0) when nothing has landed yet (the advisor then refuses to invent ETAs).
    """
    root = Path(archive_root)
    if not root.is_dir():
        return 0.0, 0
    now = time.time()
    cutoff = now - window_hours * 3600
    times = [m for pth in root.rglob("record.json")
             if not any(part.startswith(".pull_tmp") for part in pth.parts)
             and (m := pth.stat().st_mtime) >= cutoff]
    if not times:
        return 0.0, 0
    # Audit m5: a DEAD pipeline (records exist in-window but nothing recent) must read as rate 0
    # — a healthy-looking extrapolation from stale records would mask a stall in the ETAs.
    if (now - max(times)) > 3.0 * 3600:
        return 0.0, len(times)
    span_h = max((now - min(times)) / 3600.0, 0.25)   # floor: a burst never divides by ~zero
    return len(times) * 24.0 / span_h, len(times)


def observed_gpus(archive_root: str | Path, *, sample: int = 40) -> dict[str, int]:
    """Which GPU models the cluster ACTUALLY granted us, from archived records (defensive parse:
    any string field mentioning a known card name counts). Resolves e.g. the V100 16G-vs-32G
    question empirically once the first records land."""
    root = Path(archive_root)
    counts: dict[str, int] = {}
    if not root.is_dir():
        return counts
    seen = 0
    for pth in root.rglob("record.json"):
        if any(part.startswith(".pull_tmp") for part in pth.parts):
            continue
        seen += 1
        if seen > sample:
            break
        try:
            blob = pth.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for card in ("V100", "A100", "H100"):
            if card in blob:
                key = card
                m = re.search(card + r'[^"]{0,20}?(\d\d)GB', blob)
                if m:
                    key = f"{card}-{m.group(1)}G"
                counts[key] = counts.get(key, 0) + 1
                break
    return counts


def contention_trend(log_path: str | Path | None = None,
                     *, frames: int = 12) -> str:
    """Rising/falling/flat verdict from the telemetry log's recent cluster_qw values."""
    p = Path(log_path) if log_path is not None else _LOG_PATH  # audit m7: repo-anchored
    if not p.is_file():
        return "no-history"
    rows = p.read_text(encoding="utf-8").strip().splitlines()[-frames:]
    qws: list[int] = []
    for r in rows:
        try:
            qws.append(int(json.loads(r).get("cluster_qw", 0)))
        except (ValueError, KeyError):
            continue
    if len(qws) < 3:
        return "insufficient-history"
    first, last = qws[0], qws[-1]
    if last > first * 1.15:
        return f"RISING ({first}->{last})"
    if last < first * 0.85:
        return f"FALLING ({first}->{last})"
    return f"flat (~{last})"
