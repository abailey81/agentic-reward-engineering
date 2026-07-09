"""Failure forensics from Grid Engine accounting + the epilogue ledger (§14.1/§14.4).

``qacct -j <job_id>`` is the scheduler's own truth (exit_status, failed-category, maxvmem,
wallclock per task) — independent of our records and of the epilogue file, so a task that died
before writing ANYTHING still shows up here. ``parse_qacct`` turns the record-block format into
rows; ``failures`` filters the ones needing attention; ``requeue_specs`` maps them back to specs
(bounded retries; 3rd strike = permanent ledger row, same JSON-lines shape as failures.jsonl).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["parse_qacct", "failures", "requeue_specs", "read_epilogue", "host_task_counts"]

_RETRY_KEY = "_cluster_retries"
MAX_RETRIES = 2


def parse_qacct(text: str) -> list[dict[str, Any]]:
    """Parse ``qacct -j`` output (blocks separated by ==== lines) into per-task dicts."""
    rows: list[dict[str, Any]] = []
    block: dict[str, Any] = {}
    for line in text.splitlines():
        if line.startswith("====") :
            if block:
                rows.append(block)
            block = {}
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            block[parts[0]] = parts[1].strip()
    if block:
        rows.append(block)
    return rows


def failures(qacct_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows with a nonzero exit_status or a nonzero Grid Engine 'failed' category."""
    bad: list[dict[str, Any]] = []
    for r in qacct_rows:
        exit_status = str(r.get("exit_status", "0")).split()[0]
        failed = str(r.get("failed", "0")).split()[0]
        if exit_status not in ("0", "") or failed not in ("0", ""):
            bad.append(r)
    return bad


def requeue_specs(
    failed_task_ids: list[int],
    task_specs: dict[int, dict[str, Any]],
    permanent_ledger: str | Path,
) -> list[dict[str, Any]]:
    """Specs to resubmit (retry count bumped); exhausted ones go to the permanent ledger.

    The ledger file is append-only JSON-lines (one row per abandoned task) so the bank-gate
    accounting and the matched-budget guard can consume it exactly like failures.jsonl.
    """
    retry: list[dict[str, Any]] = []
    ledger_path = Path(permanent_ledger)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    for tid in failed_task_ids:
        spec = dict(task_specs[tid])
        n = int(spec.get(_RETRY_KEY, 0))
        if n < MAX_RETRIES:
            spec[_RETRY_KEY] = n + 1
            retry.append(spec)
        else:
            with ledger_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"task": tid, "spec": spec, "reason": "retries_exhausted"},
                                    sort_keys=True, default=str) + "\n")
    return retry


def read_epilogue(path: str | Path) -> list[dict[str, Any]]:
    """Tolerant reader for the per-array epilogue JSONL (§14.4) — skips torn lines."""
    p = Path(path)
    if not p.is_file():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


#: Exit codes that are NOT evidence of a broken NODE, and so must not be attributed to one.
#:
#: The bad-node detector asks one question: *is this host eating work that would have SUCCEEDED
#: somewhere else?* Two large classes of non-zero exit answer "no", and counting them produced a
#: false accusation on 2026-07-28 (``node-d00a-229`` flagged at 3/5 while it had in fact completed
#: three full trainings of 11,000-12,400 s):
#:
#: * ``1`` — APPLICATION level. The dominant cause by far is the sandbox rejecting LLM-authored
#:   reward code (measured: failures at 5 s and 20 s on that same host, against 40 such rejects
#:   campaign-wide). That code fails identically on every node; it is the authoring-reliability
#:   FINDING, not infrastructure.
#: * ``126``/``137``/``143``/``152`` — KILL signatures (SIGKILL/SIGTERM/SIGXCPU and the shell's
#:   "cannot execute"). These are admin kills, walltime kills and our own ``qdel``\ s, and they are
#:   already owned and classified by ``killswitch.classify_task_deaths``. Double-counting them here
#:   turns one cluster event into a fleet of phantom bad nodes.
#:
#: Everything else still counts — crucially ``127``, the measured signature of the real case this
#: detector was built for (``node-d00a-230`` had no ``apptainer`` and returned rc=127 in ~1 s).
#: Acting on a false positive is not harmless: the remedy is to EXCLUDE the host, which would have
#: removed a healthy 36-core node from a capacity-bound campaign.
_NOT_A_NODE_FAULT_RC: frozenset[int] = frozenset({1, 126, 137, 143, 152})


def host_task_counts(epilogue_rows: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    """``(attempts_by_host, failures_by_host)`` from epilogue rows — the bad-node detector's input.

    A node can be individually broken while the cluster is fine (measured 2026-07-26: node-d00a-230
    had no ``apptainer``, so every task routed to it returned rc=127 in ~1 s). A GLOBAL failure rate
    cannot see that — it averages one dead host into thousands of healthy tasks — so attribution by
    host is what makes it detectable. Rows without a usable host are skipped rather than bucketed
    under a fake key, which would invent a phantom bad node.
    """
    attempts: dict[str, int] = {}
    failed: dict[str, int] = {}
    for row in epilogue_rows:
        host = str(row.get("host") or "").strip()
        if not host:
            continue
        attempts[host] = attempts.get(host, 0) + 1
        try:
            rc = int(row.get("rc"))
        except (TypeError, ValueError):
            continue          # unparseable rc is unknown, NOT a failure
        if rc != 0 and rc not in _NOT_A_NODE_FAULT_RC:
            failed[host] = failed.get(host, 0) + 1
    return attempts, failed
