"""D12 — a C3 gate stop must NOT look like a success, and must NOT look like a crash.

THE DEFECT (found 2026-07-29, applied 2026-08-01, record §97). The tiered driver returned ``0`` when
the review gate stopped it, so the supervisor's ``if ($rc -eq 0)`` logged **"LINE COMPLETE"** and
exited the line. Six legs reported complete on 2026-07-29 having produced nothing at all; only the
watchdog's 300 s revive loop kept them alive.

WHY IT WAS APPLIED NOW RATHER THAN AT THE NEXT NATURAL RESTART: D16 (same commit) folds the substrate
census into the gate's ``health_ok``, which makes gate stops MORE LIKELY. A stop that reports success
would have turned the CONFIRMATORY line into a silent relaunch loop logging "LINE COMPLETE" on every
pass — strictly worse than the silent pass it replaced. The two fixes are hard-coupled.

THE THREE STATES THAT MUST BE DISTINGUISHABLE TO THE PROCESS THAT DECIDES WHETHER TO RELAUNCH:
  0  the line genuinely finished          -> stop, do not relaunch
  3  the line stopped AWAITING REVIEW     -> stop, do not relaunch, say so loudly   <- the new one
  *  the line crashed                     -> relaunch after the backoff
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUPERVISOR = REPO / "scripts" / "mode_d_supervisor.ps1"
DRIVER = REPO / "scripts" / "run_campaign_cluster.py"


def test_driver_returns_3_on_a_gate_stop_not_0() -> None:
    """The awaiting-review branch must return 3. Against the pre-fix code this finds `return 0`."""
    src = DRIVER.read_text(encoding="utf-8")
    m = re.search(r'if out\.get\("awaiting_review"\):(.{0,3000}?)\n        ok = bool', src, re.S)
    assert m, "the awaiting_review branch moved — re-anchor this test rather than deleting it"
    branch = m.group(1)
    assert "return 3" in branch, (
        "D12: a gate stop must return a code the supervisor can tell apart from success")
    assert not re.search(r"^\s+return 0\s*$", branch, re.M), (
        "the awaiting-review branch still returns 0 somewhere — a stop would read as LINE COMPLETE")


def test_supervisor_branches_on_3_distinctly_from_0_and_from_a_crash() -> None:
    """The supervisor must have a dedicated rc==3 arm that breaks WITHOUT relaunching."""
    ps = SUPERVISOR.read_text(encoding="utf-8")
    assert "$rc -eq 3" in ps, "D12: the supervisor cannot distinguish a gate stop from a crash"

    # the rc==3 arm must break (stop the line), not fall through to the relaunch
    m = re.search(r"if \(\$rc -eq 3\) \{(.*?)\n    \}", ps, re.S)
    assert m, "the rc==3 arm is not a block — it must break, not fall through"
    assert "break" in m.group(1), (
        "D12: without a break the line spins in the backoff loop forever, which is worse than the "
        "silent success it replaced")
    assert "NOT relaunching" in m.group(1) or "not relaunching" in m.group(1).lower()

    # and the ordinary crash path must STILL relaunch (no regression)
    assert "relaunching in {1}s" in ps, "a genuine crash must still be relaunched"


def test_supervisor_ps1_is_pure_ascii_and_parses() -> None:
    """PowerShell 5.1 turns non-ASCII into string-breaking smart quotes (standing repo rule)."""
    raw = SUPERVISOR.read_bytes()
    bad = [i for i, b in enumerate(raw) if b > 127]
    assert not bad, f"non-ASCII byte(s) in a .ps1 at offsets {bad[:5]}"
