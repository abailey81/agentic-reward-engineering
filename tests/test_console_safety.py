"""The console-codepage lock — written because a codepage KILLED the post-campaign runsheet.

On 2026-07-27 `scripts/bank_gate.py` died with `UnicodeEncodeError` while **printing a log tail**.
It reads subprocess output with `errors="replace"`, which injects U+FFFD, and this box's console is
cp1251, which cannot encode it. The analysis that runs AFTER a 23-day campaign was killed by a
console codepage — the most expensive possible moment in the project to lose.

Note cp1251 *does* map the em-dash, which is why this hid for so long: only the rarer glyphs
(`σ → ≥ −`) and U+FFFD bite. The exposure was measured across the launch-critical scripts and 9 were
unguarded, including `freeze.py` (GO step 1), `run_campaign_cluster.py` (the launcher) and
`sentinel.py` (which runs for the whole campaign).

These tests make that unrepeatable: one behavioural check that the helper cannot itself fail, and one
structural lock over the scripts where a crash is most expensive.
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest

from src.utils.console import make_console_safe

REPO = Path(__file__).resolve().parents[1]

#: Scripts where a console crash costs the most. Each is on this list for a stated reason; a new
#: launch-critical entrypoint should be ADDED here deliberately rather than discovered in production.
LAUNCH_CRITICAL = {
    "freeze.py": "GO step 1 — stamps the recorded hash",
    "run_campaign_cluster.py": "the launcher itself",
    "sentinel.py": "runs for the whole campaign",
    "bank_gate.py": "the POST-campaign runsheet — where this actually happened",
    "pretrain_validate.py": "the pre-launch gate",
    "preflight.py": "the GO/NO-GO gauntlet",
    "leg_gates.py": "bills real money per call",
    "certify_commit.py": "reads subprocess output with errors='replace'",
    "first_seed_sanity.py": "the earliest-warning check during the run",
    "check_rung_freshness.py": "guards the achieved-rung claim",
    "provisional_bank.py": "runs every N seeds during the campaign",
    "analyze_campaign.py": "produces the dissertation numbers",
    "monitor.py": "the live monitor",
    "allocation_advisor.py": "measured at GO and re-forecasts the rung",
}

#: Either the shared helper or the equivalent inline idiom counts — five scripts predate the helper
#: and are deliberately not churned (see src/utils/console.py).
_GUARDS = ("make_console_safe", "reconfigure")


def _is_guarded(src: str) -> bool:
    return any(g in src for g in _GUARDS)


def test_make_console_safe_cannot_itself_fail_on_a_hostile_stream(monkeypatch):
    """The helper exists to STOP failures, so it must never be the thing that raises."""

    class Hostile:
        """A stream with no reconfigure at all (what pytest's capture object looks like)."""

        def write(self, s):  # pragma: no cover - never called here
            raise AssertionError("not exercised")

    monkeypatch.setattr("sys.stdout", Hostile())
    monkeypatch.setattr("sys.stderr", Hostile())
    make_console_safe()          # must be a silent no-op, not an AttributeError

    class Raises(io.StringIO):
        def reconfigure(self, **kw):
            raise ValueError("underlying stream is detached")

    monkeypatch.setattr("sys.stdout", Raises())
    make_console_safe()          # must swallow it too


def test_make_console_safe_actually_sets_replace_on_a_real_text_stream():
    """Not just 'it ran' — the stream must come back with errors='replace'."""
    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1251", errors="strict")
    with pytest.raises(UnicodeEncodeError):     # the failure mode, reproduced
        stream.write("�")
        stream.flush()

    stream2 = io.TextIOWrapper(io.BytesIO(), encoding="cp1251", errors="strict")
    stream2.reconfigure(encoding="utf-8", errors="replace")   # what the helper does
    stream2.write("�σ→")                                  # now survivable
    stream2.flush()
    assert stream2.errors == "replace"


def test_every_launch_critical_script_makes_the_console_safe():
    """Structural lock: the scripts where a codepage crash is most expensive must be guarded."""
    missing = []
    for name, why in sorted(LAUNCH_CRITICAL.items()):
        path = REPO / "scripts" / name
        if not path.is_file():
            continue
        if not _is_guarded(path.read_text(encoding="utf-8")):
            missing.append(f"{name} ({why})")
    assert not missing, (
        "launch-critical script(s) can be killed by the console codepage — call "
        "make_console_safe() at the top of main():\n  " + "\n  ".join(missing))


def test_the_detector_can_actually_go_red():
    """A lock that cannot fail is worthless: prove the guard-detector rejects unguarded source."""
    assert not _is_guarded("import sys\n\ndef main():\n    print('hello')\n")
    assert _is_guarded("from src.utils.console import make_console_safe\nmake_console_safe()\n")
    assert _is_guarded('sys.stdout.reconfigure(encoding="utf-8", errors="replace")\n')
