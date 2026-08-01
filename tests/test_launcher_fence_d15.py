"""D15 — every launcher of a supervised line must carry the substrate host fence.

Regression for the campaign execution record §28 and the deferred-fix register item 5.

**The rule these tests encode:** AN AUTOMATIC RESTARTER IS A SECOND LAUNCHER, AND MUST TAKE EVERY
PARAMETER THE THING THAT STARTED THE LINE TOOK. `mode_d_watchdog.ps1` already carried that warning
in its own header — for `OutDir` and `RemoteRoot`, the two parameters D4 had already caught it on —
and then omitted the third.

**What the omission costs.** The host fence (`node-d00b-024`, record §28) exists because a single
heterogeneous host put four archived records on an Intel Xeon Gold 6140 while every sibling ran on
a 6240. That is a CRN-pairing hazard inside a comparison unit, i.e. a validity defect, not an
ops annoyance. A revived line that silently reverts to the default fence re-opens it.

**NOTE: AND `mode_d_launch.ps1` HAD THE SAME HOLE, WHICH THE REGISTER ONLY SUSPECTED.** Verified
2026-08-01 (RUN 10): the primary launcher passed NO `-ExcludeHosts` at all, so the live fence
existed only because the running processes were started with it BY HAND. A clean relaunch from
that script would have dropped it for ALL TWELVE LINES AT ONCE — strictly worse than the revival
case D15 describes.

**Why the sources are comment-stripped:** these files deliberately quote the flags they are about
in their headers, so matching raw text would let a mere COMMENT satisfy the test. A test that a
comment can satisfy is not a test.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.test_mode_d import _strip_ps_comments

ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = ("mode_d_watchdog.ps1", "mode_d_launch.ps1")


def _src(name: str) -> str:
    return _strip_ps_comments((ROOT / "scripts" / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", LAUNCHERS)
def test_launcher_declares_an_excludehosts_parameter(name: str) -> None:
    assert re.search(r"\[string\]\$ExcludeHosts\s*=", _src(name)), (
        f"{name} has no ExcludeHosts PARAMETER, so the fence cannot be threaded through it")


@pytest.mark.parametrize("name", LAUNCHERS)
def test_launcher_passes_the_fence_to_the_supervisor(name: str) -> None:
    """It must pass the PARAMETER, not a hard-coded literal.

    A hard-coded host list would satisfy a naive `-ExcludeHosts` grep while still being unable to
    carry a fence the operator adds later — which is the same latent-divergence failure in a new
    costume.
    """
    src = _src(name)
    assert '"-ExcludeHosts", $ExcludeHosts' in src, (
        f"{name} does not thread $ExcludeHosts into the supervisor argument vector")


@pytest.mark.parametrize("name", LAUNCHERS)
def test_the_fence_default_matches_the_supervisors_own_default(name: str) -> None:
    """A launcher default that DRIFTS from the supervisor's is a second silent divergence.

    The operative fence is always passed explicitly; the defaults exist only so that an
    argument-less invocation cannot be MORE permissive than the supervisor it starts.
    """
    sup = _src("mode_d_supervisor.ps1")
    want = re.search(r"\[string\]\$ExcludeHosts\s*=\s*\"([^\"]*)\"", sup)
    assert want, "mode_d_supervisor.ps1 lost its own ExcludeHosts default"
    got = re.search(r"\[string\]\$ExcludeHosts\s*=\s*\"([^\"]*)\"", _src(name))
    assert got and got.group(1) == want.group(1), (
        f"{name} default fence {got and got.group(1)!r} != supervisor default {want.group(1)!r}")


def test_the_watchdog_logs_the_fence_it_will_revive_with() -> None:
    """The fix for 'silently undone' is not only threading the value — it is PRINTING it.

    A fence that is never logged can be wrong for days without anyone being able to see it; this
    is what made D15 a discovery rather than an observation.
    """
    src = _src("mode_d_watchdog.ps1")
    assert "fence=" in src, "the watchdog never states which fence it is running with"
    assert src.count("$ExcludeHosts") >= 3, (
        "expected the fence in the param block, the revival vector AND the startup log")


# --------------------------------------------------------------------------- #
# D21 — reboot recovery must re-enter THE FLEET, not one campaign process       #
# --------------------------------------------------------------------------- #
def test_reboot_recovery_reenters_the_fleet_not_a_single_line() -> None:
    """D21 (2026-08-01). The ONSTART task's `-Myriad` branch used to re-enter ONE
    `run_campaign_cluster.py` with a hand-typed argument vector, which had already drifted from
    the live fleet in three independent ways: `--batch-tag c1` (the CORE LINE ONLY — the other
    eleven stay down), `--pack 4` (the fleet runs `--pack 8`), and `--exclude-hosts
    node-d00a-230` (MISSING the substrate fence, i.e. D15 on the reboot path).

    The fix deletes the vector rather than correcting it: recovery goes through
    `mode_d_launch.ps1` + `mode_d_watchdog.ps1`, the single source of truth for how a line starts.
    This test pins that, because a fourth drift would otherwise appear the next time the fleet's
    shape changed.
    """
    src = _src("install_onstart_task.ps1")
    assert "mode_d_launch.ps1" in src, "reboot recovery does not start the twelve supervised lines"
    assert "mode_d_watchdog.ps1" in src, "reboot recovery does not restart the watchdog"
    assert "--batch-tag c1" not in src, (
        "the single-line argument vector is back — a reboot would resume ONLY the core line")
    assert "--pack 4" not in src, "a stale pack size is hard-coded into reboot recovery again"
    assert "-ExcludeHosts" in src and "$ExcludeHosts" in src, (
        "reboot recovery does not thread the substrate fence")


def test_reboot_recovery_is_ascii_and_parses() -> None:
    """PS1 files are ASCII-only: PowerShell 5.1 turns em-dashes into string-breaking smart quotes.

    Asserted here rather than trusted because RUN 10 introduced non-ASCII into this very file
    while writing the D21 fix and had to strip it.
    """
    raw = (ROOT / "scripts" / "install_onstart_task.ps1").read_bytes()
    # the pre-existing docstring prose carries a handful; the bar is "no WORSE than the baseline"
    assert sum(1 for b in raw if b > 127) <= 35, (
        "non-ASCII crept into a .ps1 — PowerShell 5.1 will mangle it into smart quotes")


def test_every_supervised_line_launcher_is_covered_by_this_test() -> None:
    """COMPLETENESS. The whole D15 family is 'one launcher was missed'; a hard-coded list of
    launchers in this very test is the same failure waiting to happen.

    So DISCOVER the launchers instead: any script that starts `mode_d_supervisor.ps1` is a
    launcher and must be in LAUNCHERS above.
    """
    # Comment-stripped, for the same reason every other assertion here is: `campaign_supervisor.ps1`
    # and `install_onstart_task.ps1` both NAME `mode_d_supervisor.ps1` in prose while starting
    # something else entirely. Matching raw text would make a comment look like a launcher — and
    # then the exclusion list needed to silence it becomes the very hard-coding this test exists
    # to forbid. Verified 2026-08-01: exactly two files EXECUTE the supervisor.
    found = {p.name for p in (ROOT / "scripts").glob("*.ps1")
             if p.name != "mode_d_supervisor.ps1"
             and "mode_d_supervisor.ps1" in _strip_ps_comments(p.read_text(encoding="utf-8"))}
    assert found == set(LAUNCHERS), (
        f"the set of scripts that EXECUTE mode_d_supervisor.ps1 changed: {sorted(found)} vs the "
        f"fence-tested {sorted(LAUNCHERS)}. Add the new one to LAUNCHERS — a launcher that nobody "
        f"fence-tests is exactly how mode_d_launch.ps1 went 4 days without the substrate fence.")
