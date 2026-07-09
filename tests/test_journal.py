"""Tests for ``src.utils.journal`` — the tolerant read-side over a run's ``events.jsonl``.

The write side is the existing root-attached run logging (one ledger, no drift); these tests pin the
reader's crash-tolerance (torn last line), the completion-stream extraction, the robust cadence
yardstick (median/MAD), and the error-taxonomy clustering the sentinel panels build on.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from src.utils.journal import (
    cadence_stats,
    completions,
    error_taxonomy,
    parse_ts,
    read_events,
)


def _line(event: str, ts: str = "2026-07-06T10:00:00", level: str = "INFO", **fields) -> str:
    return json.dumps({"ts": ts, "level": level, "logger": "t", "event": event, "msg": event, **fields})


def test_read_events_tolerates_torn_and_garbage_lines(tmp_path: Path) -> None:
    p = tmp_path / "events.jsonl"
    p.write_text(
        _line("seed_done", run_id="a-s0") + "\n"
        + "not json at all\n"
        + _line("candidate_done", arm="scalar") + "\n"
        + '{"ts": "2026-07-06T10:03:00", "event": "seed_done", "run_id": "a-s1"'  # torn last line
        , encoding="utf-8",
    )
    evs = read_events(p)
    assert [e["event"] for e in evs] == ["seed_done", "candidate_done"]
    # kind filter
    assert [e["event"] for e in read_events(p, kinds=("seed_done",))] == ["seed_done"]


def test_read_events_missing_file_returns_empty(tmp_path: Path) -> None:
    assert read_events(tmp_path / "absent.jsonl") == []


def test_parse_ts_roundtrips_local_time() -> None:
    ev = {"ts": "2026-07-06T10:00:00"}
    t = parse_ts(ev)
    assert t is not None
    assert time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t)) == "2026-07-06T10:00:00"
    assert parse_ts({"ts": "garbage"}) is None
    assert parse_ts({}) is None


def test_completions_sorted_and_filtered(tmp_path: Path) -> None:
    p = tmp_path / "events.jsonl"
    p.write_text(
        _line("seed_done", ts="2026-07-06T10:02:00", run_id="b") + "\n"
        + _line("run_start", ts="2026-07-06T09:00:00") + "\n"
        + _line("candidate_done", ts="2026-07-06T10:01:00", arm="scalar") + "\n",
        encoding="utf-8",
    )
    comp = completions(read_events(p))
    assert [e["event"] for _t, e in comp] == ["candidate_done", "seed_done"]  # sorted by ts
    assert comp[0][0] < comp[1][0]


def test_cadence_stats_median_mad_and_minimum_n() -> None:
    assert cadence_stats([1.0, 2.0]) is None  # <3 completions -> no yardstick
    # gaps: 10, 10, 10, 100 (one thermal pause) -> median 10, robust to the outlier
    stats = cadence_stats([0.0, 10.0, 20.0, 30.0, 130.0])
    assert stats is not None
    assert stats["median_gap_s"] == 10.0
    assert stats["mad_gap_s"] == 0.0  # majority gaps identical
    assert stats["last_completion_s"] == 130.0


def test_error_taxonomy_clusters_and_ignores_info_noise() -> None:
    evs = [
        {"ts": "2026-07-06T10:00:00", "level": "WARNING", "event": "anomaly",
         "msg": "CUDA out of memory", "arm": "scalar"},
        {"ts": "2026-07-06T10:05:00", "level": "WARNING", "event": "anomaly",
         "msg": "CUDA out of memory", "arm": "distributional"},
        {"ts": "2026-07-06T10:06:00", "level": "WARNING", "event": "test_leg_stall",
         "since_last_completion_s": 5400},
        {"ts": "2026-07-06T10:07:00", "level": "INFO", "event": "candidate_done", "arm": "scalar"},
        {"ts": "2026-07-06T10:08:00", "level": "WARNING", "event": "seed_failed",
         "error": "sandbox: rejected import", "arm": "placebo"},
    ]
    tax = error_taxonomy(evs)
    assert tax["oom"]["count"] == 2
    assert tax["oom"]["arms"] == ["distributional", "scalar"]
    assert tax["oom"]["first_ts"] is not None and tax["oom"]["first_ts"] < tax["oom"]["last_ts"]
    assert tax["stall"]["count"] == 1
    assert tax["sandbox_reject"]["count"] == 1
    assert "other" not in tax  # the INFO progress event was ignored


def test_error_taxonomy_matches_on_word_boundaries_not_bare_substrings():
    """Ordinary words that CONTAIN a needle must not be classified as that failure (#65).

    ``_classify`` used ``needle in text``, which misfires on real driver-log messages. MEASURED
    before the fix: "pip install failed for pyarrow" -> ``stall`` (in-STALL-ed), "cluster maintenance
    window" -> ``nan`` (mainte-NAN-ce), "reading capital allocation table" -> ``api_error``
    (c-API-tal). The triage panel would then show a NUMERIC-DIVERGENCE row for a maintenance notice,
    which at 03:00 sends the operator hunting a science problem that does not exist.

    The boundary is LETTER-only and mostly LEADING-only, which matters in both directions:
      * ``\b`` would break ``test_leg_stall`` (event kinds are snake_case, ``_`` is a word char);
      * requiring a TRAILING boundary everywhere regressed "loss diverged to inf" to ``other`` —
        the needles are STEMS and must still match their inflections. Trailing is opt-in, used only
        for the short high-collision needles (``api``, ``nan``)."""
    from src.utils.journal import _classify, error_taxonomy

    # real failures still classify, INCLUDING inflected forms
    for msg, expected in (
        ("CUDA out of memory", "oom"),
        ("sandbox rejected the reward source", "sandbox_reject"),
        ("sandboxed execution refused", "sandbox_reject"),
        ("provider overloaded, retrying", "api_error"),
        ("rate limited by openrouter", "api_error"),
        ("api_error from the provider", "api_error"),
        ("request timeouts observed", "timeout"),
        ("loss diverged to inf", "divergence"),
        ("divergence detected in the critic", "divergence"),
        ("reward returned nan at step 1200", "nan"),
        ("test_leg_stall: no completion in 3h", "stall"),
        ("training stalled for 3h", "stall"),
    ):
        assert _classify({"msg": msg}) == expected, f"{msg!r} misclassified"

    # ordinary words that merely CONTAIN a needle must fall through to "other"
    for msg in ("pip install failed for pyarrow", "cluster maintenance window on Sunday",
                "reading capital allocation table", "nanometer scale", "therapist",
                "scheduled maintenance: no action needed"):
        assert _classify({"msg": msg}) == "other", f"{msg!r} still misfires"

    # and the misfire cannot reach the triage panel either
    events = [
        {"event": "seed_failed", "msg": "cluster maintenance window", "level": "WARNING"},
        {"event": "seed_failed", "msg": "reward returned nan", "level": "WARNING", "arm": "scalar"},
    ]
    tax = error_taxonomy(events)
    assert set(tax) == {"other", "nan"}, tax
    assert tax["nan"]["count"] == 1 and tax["nan"]["arms"] == ["scalar"]
