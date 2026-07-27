"""Load the gitignored ``.env`` into ``os.environ`` — called at every real entry point.

Why a dedicated helper (ADR-038)
--------------------------------
The LLM API key lives in a gitignored ``.env`` and is never committed. Nothing in the project
loaded it, so a Pass-B run died with "ANTHROPIC_API_KEY not found" despite ``.env`` holding it. We load it
HERE, at each real entry point (``run_prototype``/``run_campaign`` ``main`` and the parallel worker), rather
than inside ``src.llm.client.build_transport`` — keeping that factory PURE so its no-key error path stays
unit-testable. Windows ``spawn`` workers inherit ``os.environ`` from the parent that already called this, and
call it again defensively. ``load_dotenv`` never overrides an already-set environment variable (shell wins).
"""

from __future__ import annotations


def load_env() -> None:
    """Load ``.env`` into ``os.environ`` if ``python-dotenv`` is importable (idempotent; shell vars win)."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is a declared dep; degrade gracefully if somehow absent
        return
    load_dotenv()
