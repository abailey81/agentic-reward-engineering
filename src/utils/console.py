"""Never let the console codepage kill a run.

WHY THIS EXISTS (2026-07-27 — observed, not hypothesised). ``scripts/bank_gate.py`` died with
``UnicodeEncodeError`` while **printing a log tail** — not while analysing anything. It reads
subprocess output with ``errors="replace"``, which injects U+FFFD, and this box's console is
**cp1251**, which has no mapping for that character. So the POST-CAMPAIGN analysis runsheet was
killed by a console codepage: the most expensive possible moment in the whole project to lose.

The exposure is systemic, not incidental. Launch-critical scripts print status decorated with
characters cp1251 cannot encode (``→ σ ≥ −``, plus U+FFFD from any replace-read), and a crash in
``freeze.py`` (GO step 1), ``run_campaign_cluster.py`` (the launcher) or ``sentinel.py`` (the
whole-campaign monitor) costs far more than the cosmetic problem it looks like. Note cp1251 *does*
map the em-dash, which is why this stayed hidden for so long — only the rarer glyphs bite.

**Call from ``main()``, never at import.** Reconfiguring a global stream as an import side effect
would surprise every importer and every test that captures output — and several of these scripts
are imported by ``run_campaign.py`` and by the suite.

The encoding choice matches the idiom already used inline by ``analyze_campaign``/``monitor``/
``build_paper``/``allocation_advisor``/``resume_brief``; those five predate this helper and are
deliberately left alone rather than churned.
"""
from __future__ import annotations

import sys

__all__ = ["make_console_safe"]


def make_console_safe() -> None:
    """Make stdout/stderr tolerate un-encodable characters instead of raising.

    Idempotent, and a no-op when the streams are pytest capture objects or already UTF-8 — the
    point is that it can NEVER be the thing that fails, since it exists to stop failures.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001 — non-reconfigurable capture object, or already UTF-8
            pass
