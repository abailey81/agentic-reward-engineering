"""Provisional rung banking — legitimate interim looks, and the record that makes them defensible.

Repeated looks inflate the false-positive rate only when the sample size can respond to what is
seen. Here the stopping rule is exogenous (throughput against a fixed calendar date), so interim
banking costs nothing — PROVIDED nothing about data collection ever moves because of it. These tests
pin the two things that keep that true: every look is logged, and every payload carries the
attestation so the condition cannot be separated from the numbers later.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.provisional_bank import (
    EXOGENEITY_ATTESTATION,
    achieved_rung,
    already_banked,
    bank,
    due_rung,
)
from src.io.results import write_run


def _rec(arm: str, seed: int):
    r = np.random.default_rng(seed).standard_normal(50) * 0.01
    return {"run_id": f"{arm}-s{seed}", "arm": arm, "seed": seed, "fold": 0,
            "candidate_id": f"{arm}-w", "generation": 0, "reward_source_hash": "h" * 64,
            "feedback_block": "", "wall_clock": 10.0, "env_fingerprint": "e",
            "metrics": {"test_returns": [float(x) for x in r]}}


def _archive(root: Path, arms=("distributional", "scalar"), seeds=range(12)):
    for a in arms:
        for s in seeds:
            write_run(_rec(a, s), root / "test" / a)


def test_the_rung_is_the_COMMON_depth_not_the_deepest_arm(tmp_path: Path):
    """A paired contrast can only use seeds present in EVERY arm, so the honest rung is the
    minimum. Reporting the maximum would overstate what is actually bankable."""
    _archive(tmp_path, arms=("distributional",), seeds=range(20))
    _archive(tmp_path, arms=("scalar",), seeds=range(7))
    rung, counts = achieved_rung(tmp_path)
    assert rung == 7 and max(counts.values()) == 20


def test_banking_is_due_only_on_a_multiple_and_only_once():
    assert due_rung(23, 10, set()) == 20
    assert due_rung(23, 10, {20}) is None      # already recorded — do not clutter the log
    assert due_rung(7, 10, set()) is None      # not yet reached
    assert due_rung(40, 10, {10, 20, 30}) == 40


def test_every_look_is_LOGGED_which_is_what_makes_it_defensible(tmp_path: Path):
    """Undisclosed peeking is an integrity problem; disclosed, exogenous, logged monitoring is
    ordinary practice. The log is the entire difference, so it must always be written."""
    _archive(tmp_path)
    bank(tmp_path, 10, counts={"test/distributional": 12}, run_analysis=False)
    log = tmp_path / "banked_provisional" / "look_log.jsonl"
    rows = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(rows) == 1
    assert rows[0]["rung"] == 10
    assert "EXOGENOUS" in rows[0]["attestation"]
    assert already_banked(tmp_path) == {10}


def test_the_payload_is_stamped_PROVISIONAL_and_carries_the_attestation(tmp_path: Path):
    """The condition must travel WITH the numbers — a provisional figure separated from its caveat
    is how an interim number becomes a claim."""
    _archive(tmp_path)
    payload = bank(tmp_path, 10, counts={}, run_analysis=False)
    assert payload["PROVISIONAL"] is True and payload["confirmatory"] is False
    assert "never by any observed effect" in payload["attestation"]
    assert "PROVISIONAL" in Path(payload["path"]).name


def test_repeated_banking_accumulates_an_HONEST_history(tmp_path: Path):
    """An examiner asking 'did you peek?' should get a complete dated answer, not a reconstruction."""
    _archive(tmp_path)
    for rung in (10, 20, 30):
        bank(tmp_path, rung, counts={}, run_analysis=False)
    assert already_banked(tmp_path) == {10, 20, 30}


def test_an_empty_archive_reports_rung_zero_rather_than_guessing(tmp_path: Path):
    assert achieved_rung(tmp_path) == (0, {})


def test_the_attestation_names_every_thing_that_must_not_move():
    """If the text ever loosens, the protection loosens with it."""
    for term in ("rung", "seeds", "arms", "stop date", "EXOGENOUS", "ONCE"):
        assert term in EXOGENEITY_ATTESTATION
