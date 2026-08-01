"""Behaviour tests for the ADVISORY spend ledger (R81 softened by R83 — warns, NEVER refuses)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm.spend_ledger import (  # noqa: E402
    DEFAULT_CEILING_USD,
    planning_ceiling,
    record_spend,
    spend_summary,
)


def test_accumulates_and_summarizes(tmp_path: Path) -> None:
    led = tmp_path / "spend.jsonl"
    t1 = record_spend(led, provider="anthropic", model="claude-opus-4-8", cost_usd=1.5,
                      tokens_in=500, tokens_out=1200)
    t2 = record_spend(led, provider="openrouter", model="z-ai/glm-5.2", cost_usd=0.25)
    assert t1 == pytest.approx(1.5)
    assert t2 == pytest.approx(1.75)
    s = spend_summary(led)
    assert s["total_usd"] == pytest.approx(1.75)
    assert s["by_provider"] == {"anthropic": 1.5, "openrouter": 0.25}
    assert s["by_model"]["z-ai/glm-5.2"] == pytest.approx(0.25)
    assert s["n_calls"] == 2


def test_never_refuses_past_the_ceiling(tmp_path: Path) -> None:
    """R83: crossing (even far exceeding) the advisory ceiling records fine — no exception."""
    led = tmp_path / "spend.jsonl"
    total = record_spend(led, provider="anthropic", model="m", cost_usd=DEFAULT_CEILING_USD * 3)
    assert total == pytest.approx(DEFAULT_CEILING_USD * 3)  # accepted, not refused
    # and further spend still records
    assert record_spend(led, provider="anthropic", model="m", cost_usd=1.0) > total


def test_warns_once_per_threshold_crossing(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    led = tmp_path / "spend.jsonl"
    ceiling = planning_ceiling()
    with caplog.at_level(logging.WARNING, logger="src.llm.spend_ledger"):
        record_spend(led, provider="p", model="m", cost_usd=0.79 * ceiling)   # below 80%
        assert not [r for r in caplog.records if "ADVISORY" in r.getMessage()]
        record_spend(led, provider="p", model="m", cost_usd=0.02 * ceiling)   # crosses 80%
        crossings = [r for r in caplog.records if "80%" in r.getMessage()]
        assert len(crossings) == 1
        record_spend(led, provider="p", model="m", cost_usd=0.30 * ceiling)   # crosses 100%
        assert [r for r in caplog.records if "100%" in r.getMessage()]


def test_torn_final_line_is_skipped_not_fatal(tmp_path: Path) -> None:
    led = tmp_path / "spend.jsonl"
    record_spend(led, provider="p", model="m", cost_usd=2.0)
    with led.open("a", encoding="utf-8") as fh:
        fh.write('{"provider": "p", "cost_usd": 5.0')  # torn append (no closing brace/newline)
    s = spend_summary(led)
    assert s["total_usd"] == pytest.approx(2.0)  # torn line skipped
    assert s["n_calls"] == 1


def test_negative_cost_rejected_loudly(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cost_usd"):
        record_spend(tmp_path / "l.jsonl", provider="p", model="m", cost_usd=-0.01)


def test_ceiling_reads_from_config() -> None:
    """The advisory ceiling comes from the live registration mirror (30 per R81/R83)."""
    assert planning_ceiling() == pytest.approx(30.0)


def test_summary_of_missing_ledger_is_zero(tmp_path: Path) -> None:
    s = spend_summary(tmp_path / "absent.jsonl")
    assert s["total_usd"] == 0.0 and s["n_calls"] == 0
