"""Spawn-side bootstrap for ``validate_once``'s candidate child — STDLIB-ONLY at module level.

WHY THIS MODULE EXISTS (2026-07-18 forensics, pre-launch): the validation child used to be
spawned directly on ``executor._candidate_child`` with numpy arrays in its args, so the child's
*bootstrap* (unpickling the target module + the fixture) imported numpy/MKL BEFORE any signal
reached the parent — and the parent's single ``timeout_s`` (2.0 s) clock covered that whole
startup. On a commit-starved box (the ArmouryCrate leak left ~0.37 GB of system commit charge;
the numpy DLL load then stalls for minutes while Windows grows the pagefile) or a contended
cluster node (the p6ext800 x0.5 incident class), startup alone exceeds 2 s and a PERFECTLY GOOD
reward is rejected as "exceeded the validation timeout" — a false negative that would discard a
paid candidate at authoring or fail a seed on the sealed test leg.

The fix is a three-phase handshake, and it only works if this module imports NOTHING heavy:

    1. ``ready``  — put on the queue before any heavy import (proves the child spawned);
    2. ``armed``  — put after numpy + the executor module + the fixture unpickle are done
                    (proves startup finished; the user-code clock starts HERE);
    3. verdict    — ``("ok", "")`` / ``("error", msg)`` from the unchanged validation body.

The fixture travels as ``pickle.dumps`` BYTES so unpickling it (which imports numpy) happens
inside phase 2, not during process bootstrap. The parent (``executor.validate_once``) grants
generous grace for phases 1-2 (environment noise, never the candidate's fault) and applies the
strict ``timeout_s`` ONLY to phase 3 — the candidate's own code. Security is unchanged: the
child is killable, the AST gate ran in the parent, and the user-code wall-clock cap is intact.
"""
from __future__ import annotations

from typing import Any

__all__ = ["boot_candidate_child"]


def boot_candidate_child(src: str, fixture_blob: bytes, q: Any) -> None:  # pragma: no cover - spawned-child body; exercised via validate_once integration tests
    """Handshake wrapper around ``executor._candidate_child`` (see module docstring)."""
    q.put(("ready", ""))
    try:
        import pickle  # noqa: S403 - the blob is produced by OUR parent process, not untrusted input

        from src.sandbox.executor import _candidate_child  # heavy: numpy + contract imports

        fixture = pickle.loads(fixture_blob)  # noqa: S301 - same-process-tree payload (see above)
    except Exception as exc:  # noqa: BLE001 - any startup failure must reach the parent loudly
        q.put(("error", f"validation child crashed during startup: {exc!r}"))
        return
    q.put(("armed", ""))
    _candidate_child(src, fixture, q)
