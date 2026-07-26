"""Tests for the adaptive allocation system (telemetry parsers + the advisor's doctrine)."""
from __future__ import annotations

from src.cluster.allocation import (advise, chunking, pick_search_pool, recommend_pack,
                                    stripe, usable_pools)
from src.cluster.telemetry import (Snapshot, parse_cluster_pending, parse_our_jobs,
                                   parse_qhost_gpu, probe_verdicts)

_QHOST = """HOSTNAME  ARCH  NCPU
----------------------------------------------------------------------------------------------
node-e00a-008           lx-amd64       36    2   36   36     -  188.4G       -    1.8T       -
    Host Resource(s):      hc:gpu=2.000000
node-e00a-014           lx-amd64       36    2   36   36 10.16  188.4G   25.2G    1.8T    9.2M
    Host Resource(s):      hc:gpu=1.000000
node-l00a-006           lx-amd64       36    2   36   36  0.34  188.4G   24.4G    1.8T   16.6M
    Host Resource(s):      hc:gpu=3.000000
node-u00a-001           lx-amd64       48    2   48   48  3.28  251.4G   34.6G    1.7T    1.0G
    Host Resource(s):      hc:gpu=4.000000
node-d00a-001           lx-amd64       36    2   36   36  1.00  188.4G   10.0G    1.8T    1.0M
"""

_PENDING = """job-ID  prior   name   user   state submit
------------------------------------------------------------
 100 3.4 a userA qw 07/24
 101 3.3 b userA qw 07/24
 102 0.0 c userB hqw 07/24
 103 3.2 d userC qw 07/24
"""

_MINE = """job-ID  prior   name   user   state submit
------------------------------------------------------------
 8323 3.18533 mr ucestes qw 07/23
 8323 0.00000 mr ucestes hqw 07/23
 10293 1.10000 probe_u ucestes qw 07/24
"""


def _snap(qw: int = 3000, probes: dict | None = None, pools: dict | None = None) -> Snapshot:
    return Snapshot(ts="t", pool_free=pools or {"EF": 3, "L": 3},
                    cluster_qw=qw, cluster_users=100,
                    our_jobs=[], probe_states=probes or {})


def test_parse_qhost_pools_and_counts() -> None:
    free = parse_qhost_gpu(_QHOST)
    assert free == {"EF": 3, "L": 3, "U": 4}  # d-node ignored (no pool, and no gpu line counted


def test_parse_pending_counts_eligible_and_users() -> None:
    qw, users = parse_cluster_pending(_PENDING)
    assert qw == 3 and users == 3  # hqw not counted as eligible; userB still a user


def test_parse_our_jobs_rows() -> None:
    rows = parse_our_jobs(_MINE)
    assert [r["state"] for r in rows] == ["qw", "hqw", "qw"]
    assert rows[0]["prior"] == 3.18533


def test_probe_verdicts_running_pending_restricted() -> None:
    jobs = [{"id": "10293", "prior": 1.0, "state": "r"},
            {"id": "10294", "prior": 0.0, "state": "qw"}]
    v = probe_verdicts(jobs, pending_hours=1.0)
    assert "USABLE" in v["U"] and v["V"].startswith("pending")
    v48 = probe_verdicts(jobs, pending_hours=49.0)
    assert "RESTRICTED" in v48["V"]


def test_chunking_two_regimes() -> None:
    assert chunking(_snap(qw=3000))[0] == "CONTENDED"
    assert chunking(_snap(qw=200)) == ("QUIET", 1)


def test_usable_pools_gates_uv_on_probe_verdict() -> None:
    s = _snap(pools={"EF": 2, "L": 1, "U": 4}, probes={"U": "pending (qw)"})
    assert "U" not in usable_pools(s)
    s2 = _snap(pools={"EF": 2, "L": 1, "U": 4}, probes={"U": "RUNNING (USABLE)"})
    assert usable_pools(s2)["U"] == 4


def test_search_pool_prefers_fastest_with_headroom() -> None:
    assert pick_search_pool({"EF": 5, "L": 2}) == "L"
    assert pick_search_pool({"EF": 5, "L": 1}) == "EF"   # L lacks the 2-GPU headroom bar
    assert pick_search_pool({"EF": 0, "L": 0}) == "EF"   # fallback


def test_stripe_blocks_are_contiguous_pool_homogeneous_and_cover() -> None:
    order, blocks = stripe([(0, 29), (30, 99)], {"EF": 2, "L": 2})
    assert order[0] == "L"  # faster pool first at equal free counts
    seen: list[tuple[int, int]] = []
    for part in blocks.split(","):
        pool, rng = part.split(":")
        a, b = (int(x) for x in rng.split("-"))
        assert pool in ("EF", "L") and a <= b
        seen.append((a, b))
    # full coverage, no overlap, contiguity within each segment
    covered = sorted(seen)
    assert covered[0][0] == 0 and covered[-1][1] == 99
    for (a1, b1), (a2, b2) in zip(covered, covered[1:]):
        assert a2 == b1 + 1


def test_recommend_pack_gated_on_measurement() -> None:
    pools = {"EF": 2, "L": 2, "U": 4}
    unmeasured, note = recommend_pack(pools, measured_vram_per_training_gb=None)
    assert set(unmeasured.values()) == {5} and "pending" in note
    measured, _ = recommend_pack(pools, measured_vram_per_training_gb=2.8)
    assert measured["EF"] == 5 and measured["L"] == 8 and measured["U"] == 10  # capped by table


def test_advise_end_to_end_plan_shape() -> None:
    s = _snap(qw=200, pools={"EF": 3, "L": 4}, probes={"U": "pending (qw)", "V": "pending (qw)"})
    plan = advise(s, measured_trainings_per_day=800.0,
                  remaining_trainings={"tier403": 6800})
    assert plan.regime == "QUIET" and plan.chunk_tasks == 1
    assert plan.search_pool == "L"
    assert plan.eta_days == {"tier403": 8.5}
    assert "priorities untouched" in plan.notes[-1]
    # The rule: the plan can never render a priority ACTION — no `-p <n>` flag, no qalter.
    import re

    rendered = plan.render()
    assert "qalter" not in rendered
    assert not re.search(r"(?:^|\s)-p\s+-?\d", rendered)


def test_chunking_hysteresis_no_flap() -> None:
    """Near the boundary the regime sticks to its previous state (enter 1.2x, exit 0.8x)."""
    near = _snap(qw=1500)  # exactly at the base threshold
    assert chunking(near, prev_regime="CONTENDED")[0] == "CONTENDED"  # 1500 > 1200 exit bar
    assert chunking(near, prev_regime="QUIET")[0] == "QUIET"          # 1500 < 1800 enter bar
    assert chunking(_snap(qw=1000), prev_regime="CONTENDED")[0] == "QUIET"   # true exit
    assert chunking(_snap(qw=2000), prev_regime="QUIET")[0] == "CONTENDED"   # true enter


def test_measure_rate_from_record_mtimes(tmp_path) -> None:
    import os
    import time as _t

    from src.cluster.telemetry import measure_rate
    root = tmp_path / "archive"
    now = _t.time()
    for i, age_h in enumerate([1.0, 2.0, 3.0]):
        d = root / f"u{i}"
        d.mkdir(parents=True)
        f = d / "record.json"
        f.write_text("{}", encoding="utf-8")
        os.utime(f, (now - age_h * 3600, now - age_h * 3600))
    rate, n = measure_rate(root, window_hours=24.0)
    assert n == 3
    assert 20.0 <= rate <= 28.0  # 3 records over ~3h -> ~24/day
    assert measure_rate(tmp_path / "absent") == (0.0, 0)


def test_observed_gpus_defensive_parse(tmp_path) -> None:
    from src.cluster.telemetry import observed_gpus
    d = tmp_path / "a"
    d.mkdir()
    (d / "record.json").write_text('{"gpu": "Tesla V100-SXM2-32GB, 550.54"}', encoding="utf-8")
    e = tmp_path / "b"
    e.mkdir()
    (e / "record.json").write_text('{"gpu": "NVIDIA A100 80GB PCIe"}', encoding="utf-8")
    got = observed_gpus(tmp_path)
    assert got.get("V100-32G") == 1 and sum(got.values()) == 2


def test_contention_trend(tmp_path) -> None:
    import json as _json

    from src.cluster.telemetry import contention_trend
    log = tmp_path / "tele.jsonl"
    rows = [{"cluster_qw": q} for q in (1000, 1200, 1500, 2000)]
    log.write_text("\n".join(_json.dumps(r) for r in rows), encoding="utf-8")
    assert contention_trend(log).startswith("RISING")
    assert contention_trend(tmp_path / "absent.jsonl") == "no-history"


def test_plan_delta_alerts_on_actionable_changes_only() -> None:
    from src.cluster.allocation import plan_delta
    base = {"regime": "QUIET", "search_pool": "L", "stripe_pools": ["L", "EF"],
            "chunk_tasks": 1}
    assert plan_delta(None, base)[0].startswith("INITIAL PLAN")
    assert plan_delta(base, dict(base)) == []                       # no change -> silence
    flip = dict(base, regime="CONTENDED", chunk_tasks=25)
    assert any("REGIME FLIP" in a for a in plan_delta(base, flip))
    unlock = dict(base, stripe_pools=["U", "L", "EF"])
    assert any("STRIPE POOLS CHANGED" in a for a in plan_delta(base, unlock))


def test_state_round_trip_and_corruption_tolerance(tmp_path) -> None:
    from src.cluster.telemetry import load_state, save_state
    p = tmp_path / "state.json"
    save_state({"prev_regime": "CONTENDED", "last_plan": {"regime": "CONTENDED"}}, p)
    assert load_state(p)["prev_regime"] == "CONTENDED"
    p.write_text("{corrupt", encoding="utf-8")
    assert load_state(p) == {}                                      # never raises
    assert load_state(tmp_path / "absent.json") == {}


def test_absent_probe_is_never_usable() -> None:
    """Audit M1 regression: a probe missing from qstat must NOT admit its pool."""
    v = probe_verdicts([], pending_hours=100.0)
    assert all("USABLE" not in s for s in v.values())
    s = _snap(pools={"EF": 2, "L": 1, "U": 4}, probes=v)
    assert "U" not in usable_pools(s) and "V" not in usable_pools(s)


def test_pool_admission_is_a_POSITIVE_test_and_cannot_fail_open_on_a_reworded_negative() -> None:
    """U/V admission must not hinge on a word appearing inside a human-readable sentence (#60).

    ``usable_pools`` gated on ``"USABLE" in verdict`` — a CASE-SENSITIVE substring of prose written
    in ``telemetry.probe_verdicts``. The live negatives spell it lowercase ("NOT usable"), so the
    gate was correct in practice, but it is fail-OPEN by construction: REPRODUCED 2026-07-26, a
    verdict reworded to ``"NOT USABLE"`` or ``"UNUSABLE"`` — the natural way to add emphasis —
    ADMITTED the pool. That would schedule the campaign onto a pool that never grants (jobs pend
    forever) or onto one we were told not to use, which is exactly what Audit M1's "absent is NOT
    auto-usable" rule exists to prevent.

    The admission test is now positive and lives beside the vocabulary it tests, so an edited or
    newly-added negative is refused by default rather than accidentally matching."""
    from src.cluster.telemetry import USABLE_VERDICT, pool_is_usable

    def _joins(verdict: str | None) -> bool:
        return "U" in usable_pools(_snap(pools={"EF": 2, "L": 1, "U": 4}, probes={"U": verdict}))

    # the ONE admitting verdict, and it is the same string probe_verdicts actually emits
    assert pool_is_usable(USABLE_VERDICT) and _joins(USABLE_VERDICT)
    assert probe_verdicts([{"id": "10293", "state": "r"}], pending_hours=1.0)["U"] == USABLE_VERDICT

    # every negation is refused — including the case variants that previously admitted
    for verdict in ("NOT USABLE", "UNUSABLE", "NOT usable", "not auto-usable",
                    "usable maybe", "RUNNING (USABLE) but drained", "", None):
        assert not pool_is_usable(verdict), f"pool_is_usable admitted {verdict!r}"
        assert not _joins(verdict), f"usable_pools admitted {verdict!r}"

    # the real verdicts for every probe state still behave as designed
    for state, expected in ((None, False), ("Eqw", False), ("r", True), ("qw", False)):
        jobs = [] if state is None else [{"id": "10293", "state": state}]
        verdict = probe_verdicts(jobs, pending_hours=1.0)["U"]
        assert _joins(verdict) is expected, f"state={state!r} verdict={verdict!r}"

    # EF/L are always ours and never depend on a probe verdict
    always = usable_pools(_snap(pools={"EF": 2, "L": 1}, probes={"U": "NOT USABLE"}))
    assert set(always) == {"EF", "L"}


def test_a_FIRST_read_with_degraded_telemetry_defaults_to_the_safe_regime() -> None:
    """Audit M3 freezes the regime only when a PRIOR one exists; a first read fell through (#61).

    With no ``prev_regime``, a failed PENDING section parses as ``cluster_qw=0`` and the doctrine
    reads that phantom-empty queue as QUIET — the AGGRESSIVE chunk-1 flood. Wrong direction under
    unknown pressure: the constraint chunking actually protects is ``max_u_jobs = 1000``, chunk-1
    with pipelined rungs can enqueue ~1,200 arrays, and a cap hit classes as a transport error.
    CONTENDED/chunk-25 cannot breach it.

    Not hypothetical: the runbook has the advisor run AT LAUNCH — both a first read and the moment
    ssh/qstat is most likely to hiccup."""
    degraded = _snap(qw=0, probes={})
    degraded.errors = ["section PENDING: rc=2"]
    plan = advise(degraded, prev_regime=None)
    assert plan.regime == "CONTENDED" and plan.chunk_tasks == 25, (
        f"a degraded first read advised {plan.regime}/chunk-{plan.chunk_tasks}"
    )
    assert any("FIRST read" in n for n in plan.notes)
    assert any("telemetry degraded" in n for n in plan.notes)

    # a CLEAN empty queue is still legitimately QUIET — the fix must not blunt a real measurement
    clean = _snap(qw=0, probes={})
    assert advise(clean, prev_regime=None).regime == "QUIET"

    # and the M3 freeze is untouched where a prior regime exists
    for prev in ("QUIET", "CONTENDED"):
        assert advise(degraded, prev_regime=prev).regime == prev
