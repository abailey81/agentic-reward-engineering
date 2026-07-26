"""The CPU-lane advisory — closing the gap that left the 2026-07-26 capacity findings manual.

`advise_cpu_lane` composes the two modules that own the halves (killswitch = how much to take;
lanes = what the makespan then is) and adds the measured job shape, so GO day reads a number
instead of a human re-deriving it from the dossier.
"""
from __future__ import annotations

from src.cluster.allocation import advise_cpu_lane, chunking
from src.cluster.telemetry import Snapshot, parse_cpu_free


def _snap(*, cpu_free: dict[str, int] | None = None, qw: int = 1000) -> Snapshot:
    return Snapshot(ts="t", pool_free={}, cluster_qw=qw, cluster_users=50, our_jobs=[],
                    probe_states={}, cpu_free=(cpu_free if cpu_free is not None else {}))


# --- parsing -------------------------------------------------------------------------------

def test_parse_cpu_free_reads_the_type_core_lines():
    assert parse_cpu_free("d 3668\nb 292\nt 77\n") == {"d": 3668, "b": 292, "t": 77}


def test_parse_cpu_free_is_tolerant_and_never_raises():
    """Telemetry must DEGRADE, never crash a caller — junk lines are skipped, not fatal."""
    assert parse_cpu_free("") == {}
    assert parse_cpu_free("garbage\nd notanumber\ndd 5\nd 100\n") == {"d": 100}
    assert parse_cpu_free(None) == {}  # type: ignore[arg-type]


# --- the advisory --------------------------------------------------------------------------

def test_advisory_uses_only_the_confirmatory_pools():
    """t is EXCLUDED (AMD vs Intel -> CRN bit-exactness) and GPU-node cores are never harvested,
    so a free-capacity table must not leak them into the target."""
    out = advise_cpu_lane(_snap(cpu_free={"d": 3000, "b": 200, "t": 400, "e": 500}))
    assert out["free_by_pool"] == {"d": 3000, "b": 200}
    assert out["free_cores"] == 3200
    assert set(out["pools"]) == {"d", "b"}


def test_unknown_cpu_free_reports_None_not_a_silent_stall():
    """An empty telemetry section means UNKNOWN. Reporting 0 would recommend holding no cores at
    all — a silent stall dressed as a recommendation."""
    out = advise_cpu_lane(_snap(cpu_free={}))
    assert out["target_cores"] is None
    assert "unknown" in out["why"]


def test_advisory_carries_the_measured_job_shape_and_thread_split():
    out = advise_cpu_lane(_snap(cpu_free={"d": 3000, "b": 200}))
    assert "smp 8" in out["job_shape"] and "NEVER smp 36" in out["job_shape"]
    assert out["flood_threads"] == 1      # throughput lane
    assert out["chain_threads"] == 8      # latency lane
    assert out["makespan_days"] > 0 and out["binding"] in ("throughput", "critical_chain")


def test_advisory_respects_live_pressure_and_the_courtesy_reserve():
    """Busier cluster -> smaller target; and the 1000-core reserve is never eaten into."""
    busy = advise_cpu_lane(_snap(cpu_free={"d": 4000}, qw=3000))["target_cores"]
    quiet = advise_cpu_lane(_snap(cpu_free={"d": 4000}, qw=100))["target_cores"]
    assert busy < quiet
    assert quiet <= 4000 - 1000          # FREE_CORE_RESERVE intact


def test_a_standing_retreat_cap_flows_through_to_the_advisory():
    """An uncleared kill incident must dominate any live recommendation."""
    out = advise_cpu_lane(_snap(cpu_free={"d": 4000}, qw=100), retreat_cap_cores=160)
    assert out["target_cores"] == 160
    assert "retreat cap" in out["why"]


# --- the refuted-doctrine correction --------------------------------------------------------

def test_chunking_behaviour_is_unchanged_but_no_longer_rests_on_the_refuted_doctrine():
    """The ticket-concentration doctrine is REFUTED (dossier §0-PRE M5), but chunk-25 SURVIVES on a
    different real constraint: max_u_jobs=1000. Behaviour must not change; the stated reason must."""
    import inspect

    from src.cluster import allocation

    assert chunking(_snap(qw=3000)) == ("CONTENDED", 25)
    assert chunking(_snap(qw=200)) == ("QUIET", 1)

    src = inspect.getsource(allocation.chunking)
    assert "REFUTED" in src, "the refutation must be recorded at the decision point"
    assert "max_u_jobs" in src, "the SURVIVING justification must be stated"

    # The OPERATIVE justification is the inline comment on the CONTENDED return itself. (Scanning
    # the whole function is wrong: the prose above legitimately QUOTES the refuted claim in order
    # to refute it — documentation must be free to name what it is overturning.)
    operative = next(ln for ln in src.splitlines() if 'return "CONTENDED"' in ln)
    assert "max_u_jobs" in operative
    assert "priority" not in operative.lower() or "NOT for priority" in operative
