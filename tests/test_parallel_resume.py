"""Search-replay (resume) cache for the PARALLEL campaign path (``parallel._drive_llm_arm``).

The parallel LLM driver, on ``--resume``, must REPLAY already-archived candidates from disk instead
of re-calling the (paid, non-deterministic) LLM author and re-training every candidate from scratch —
which could newly SUCCEED a previously-rejected candidate and silently change the candidate set /
winner. This mirrors the SERIAL path's resume cache (src/llm/loop.py) but is adapted to the parallel
archive scheme (bare ``cid`` run-ids under ``<archive_root>/<arm>/``; LIVE reflection-block rebuild
from the generation's BEST; a sibling ``failures.jsonl`` ledger).

These tests use the stub-designer transport (``pass_mode="A"``) and a FAKE DevicePool whose ``submit``
returns a deterministic accepted result WITHOUT any real SAC training — so the whole search/replay
contract is exercised with no torch and no LLM. We monkeypatch the stub transport with a COUNTING
wrapper to assert exactly how many ``llm.complete`` calls each run makes.
"""
from __future__ import annotations

import sys
from concurrent.futures import Future
from pathlib import Path

# build_parallel_opts lives in scripts/run_prototype.py (the SHARED opts builder).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_prototype import build_parallel_opts  # type: ignore[import-not-found]  # noqa: E402

from src.llm.stub_designer import StubDesignerTransport as _RealStubTransport  # noqa: E402
from src.orchestration import parallel  # noqa: E402
from src.utils.config import load_config  # noqa: E402


class _FakePool:
    """A DevicePool stand-in: ``submit(spec)`` returns an immediately-resolved Future carrying a
    deterministic accepted result keyed off the candidate id — NO real SAC training. ``train_candidate``
    is never called. Fitness is a stable function of the candidate index so the winner is predictable
    and identical across a fresh run and its replay. Carries the DevicePool capacity attributes the
    search driver's sliding-window scheduler reads (2026-07-06)."""

    n_gpu = 0
    n_cpu = 2

    def __init__(self) -> None:
        self.submitted: list[str] = []

    def submit(self, spec: dict) -> Future:
        cid = str(spec["candidate_id"])
        self.submitted.append(cid)
        # Deterministic fitness from the trailing per-generation index ``c<k>`` so different k -> a
        # distinct, ordered fitness (the LAST candidate of each gen is the gen-best here).
        k = int(cid.rsplit("-c", 1)[-1])
        fut: Future = Future()
        fut.set_result({
            "ok": True,
            "candidate_id": cid,
            "arm": spec["arm"],
            "fitness": 0.10 + 0.01 * k,
            "val_returns": [0.001, -0.002, 0.003],
            "tail_stats": {
                "cvar_05": -0.02, "cvar_10": -0.015, "cvar_25": -0.01, "cvar_01": -0.03,
                "left_tail_mass": 0.1, "robust_skew": -0.2,
            },
            "reward_source": spec["reward"],
            "reward_hash": "deadbeef",
            "popart_scale": 1.0,
        })
        return fut


class _CountingTransport:
    """Wraps the real StubDesignerTransport and COUNTS every ``llm.complete`` (== transport call)."""

    def __init__(self, seed: int = 0) -> None:
        # Bind the REAL (pre-patch) transport so the counter never recurses into its own factory.
        self._inner = _RealStubTransport(seed=int(seed))
        self.n_calls = 0

    def __call__(self, system: str, user: str) -> str:
        self.n_calls += 1
        return self._inner(system, user)

    def advance(self, k: int = 1) -> None:
        """Forward the stream-faithful replay advance (2026-07-06) — hiding it would silently
        no-op the invariant in every driver-level resume test."""
        self._inner.advance(k)


def _opts(seed: int = 3, *, resume: bool = False) -> dict:
    structural = {
        "agent": {"batch_size": 256, "normalize_obs": True, "device": "cpu"},
        "reward_family": {"cvar_alpha": 0.05, "window": 20},
        "data": {},
    }
    return build_parallel_opts(
        structural,
        load_config("environment"),
        llm_block={},
        train_steps=10,
        n_trials=4,
        synthetic=True,
        seed=seed,
        candidates=4,
        generations=2,
        pass_mode="A",
        provider="stub",
        resume=resume,
    )


def _install_counter(monkeypatch) -> dict:
    """Replace StubDesignerTransport (imported inside _drive_llm_arm) with the counting wrapper and
    return a holder whose ``["t"]`` is the most-recently constructed counting transport."""
    holder: dict = {}

    def _factory(seed: int = 0):
        holder["t"] = _CountingTransport(seed=seed)
        return holder["t"]

    monkeypatch.setattr("src.llm.stub_designer.StubDesignerTransport", _factory)
    return holder


def test_resume_opts_threaded_through_builder():
    """``build_parallel_opts(resume=...)`` surfaces a ``resume`` key; default is False (fresh = unchanged)."""
    assert _opts(resume=False)["resume"] is False
    assert _opts(resume=True)["resume"] is True
    # back-compat: omitting the kwarg defaults to False (a fresh run is byte-unchanged)
    base = build_parallel_opts(
        {"agent": {}, "reward_family": {}, "data": {}}, load_config("environment"),
        llm_block={}, train_steps=10, n_trials=4, synthetic=True, seed=0,
        candidates=2, generations=1, pass_mode="A", provider="stub",
    )
    assert base["resume"] is False


def test_full_replay_makes_zero_llm_calls_and_same_winner(tmp_path, monkeypatch):
    """(i) populate an archive; (ii) resume over the FULLY-archived arm -> ZERO llm.complete calls,
    ZERO training submissions, and the SAME winner_fitness."""
    holder = _install_counter(monkeypatch)
    arm = "distributional"

    # (i) Fresh run populates the archive.
    pool1 = _FakePool()
    summary1 = parallel._drive_llm_arm(arm, pool1, _opts(resume=False), str(tmp_path))
    assert holder["t"].n_calls == 4  # 2 generations x 2 candidates/gen, all freshly generated
    assert len(pool1.submitted) == 4  # all four trained in the pool
    assert summary1["n_candidates"] == 4 and summary1["n_failed"] == 0
    assert summary1["winner_fitness"] is not None

    # (ii) Resume over the fully-archived arm: a NEW transport (call count starts at 0) + a fresh pool.
    pool2 = _FakePool()
    summary2 = parallel._drive_llm_arm(arm, pool2, _opts(resume=True), str(tmp_path))
    assert holder["t"].n_calls == 0  # NOTHING re-generated -> no paid LLM call
    assert pool2.submitted == []  # NOTHING re-trained
    assert summary2["n_candidates"] == 4 and summary2["n_failed"] == 0
    # The replay reproduces the EXACT same winner.
    assert summary2["winner_fitness"] == summary1["winner_fitness"]


def test_fresh_run_still_calls_transport(tmp_path, monkeypatch):
    """A fresh (resume=False) run over an EMPTY archive still drives the transport (no silent skip)."""
    holder = _install_counter(monkeypatch)
    pool = _FakePool()
    parallel._drive_llm_arm("scalar", pool, _opts(resume=False), str(tmp_path))
    assert holder["t"].n_calls == 4 and len(pool.submitted) == 4


def test_partial_replay_only_generates_missing(tmp_path, monkeypatch):
    """A PARTIALLY-archived arm replays the done candidates and generates ONLY the missing ones.

    We seed the archive with the first generation's candidates by running ONE generation, then resume
    a TWO-generation run: gen-0 (2 cands) replays at zero cost; gen-1 (2 cands) is generated fresh.
    """
    holder = _install_counter(monkeypatch)
    arm = "scalar_cvar5"

    # Archive ONLY generation 0 (candidates=2, generations=1 -> ids ...-g0-c0, ...-g0-c1).
    one_gen = _opts(resume=False)
    one_gen["candidates"], one_gen["generations"] = 2, 1
    parallel._drive_llm_arm(arm, _FakePool(), one_gen, str(tmp_path))
    assert holder["t"].n_calls == 2

    # Resume a 2-gen / 4-candidate run: gen-0's two ids (...-g0-c0/c1) already exist -> replayed;
    # gen-1's two (...-g1-c0/c1) are missing -> generated.
    holder2 = _install_counter(monkeypatch)
    two_gen = _opts(resume=True)  # candidates=4, generations=2 by default
    pool = _FakePool()
    summary = parallel._drive_llm_arm(arm, pool, two_gen, str(tmp_path))
    assert holder2["t"].n_calls == 2  # ONLY the missing generation-1 pair
    assert sorted(pool.submitted) == [f"{arm}-g1-c0", f"{arm}-g1-c1"]
    assert summary["n_candidates"] == 4 and summary["n_failed"] == 0


def test_cached_failure_is_not_regenerated(tmp_path, monkeypatch):
    """A previously-archived sandbox FAILURE replays from failures.jsonl and is NOT regenerated."""
    import json

    holder = _install_counter(monkeypatch)
    arm = "placebo"
    arm_dir = tmp_path / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    # Pre-seed: one accepted candidate on disk + one failed candidate in the ledger, for gen 0.
    from src.io.results import write_run

    write_run(
        {
            "run_id": f"{arm}-g0-c0", "arm": arm, "seed": 3, "fold": 0,
            "candidate_id": f"{arm}-g0-c0", "generation": 0, "prompt": "p",
            "reward_source": "def reward(*a, **k):\n    return 0.0\n", "reward_source_hash": "h",
            "feedback_block": "", "metrics": {"val_fitness": 0.5, "val_returns": [0.0], "tail_stats": None},
            "wall_clock": 0.0, "env_fingerprint": "x",
        },
        str(arm_dir),
    )
    (arm_dir / "failures.jsonl").write_text(
        json.dumps({"candidate_id": f"{arm}-g0-c1", "prompt": "p", "reward_source": "bad",
                    "error": "sandbox: nope"}) + "\n",
        encoding="utf-8",
    )

    # Resume a single generation of 2 candidates: c0 replays (accepted), c1 replays (failed) -> zero
    # generation; the rewritten ledger preserves the failure.
    opts = _opts(resume=True)
    opts["candidates"], opts["generations"] = 2, 1
    pool = _FakePool()
    summary = parallel._drive_llm_arm(arm, pool, opts, str(tmp_path))
    assert holder["t"].n_calls == 0  # neither candidate regenerated
    assert pool.submitted == []
    assert summary["n_candidates"] == 1 and summary["n_failed"] == 1
    # The failure survives the end-of-arm ledger rewrite.
    rows = [json.loads(ln) for ln in (arm_dir / "failures.jsonl").read_text().splitlines() if ln.strip()]
    assert [r["candidate_id"] for r in rows] == [f"{arm}-g0-c1"]


def test_torn_failure_ledger_does_not_crash_resume(tmp_path, monkeypatch):
    """A truncated last line in failures.jsonl is tolerated (that candidate just regenerates)."""
    holder = _install_counter(monkeypatch)
    arm = "placebo"
    arm_dir = tmp_path / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    # A complete failure line followed by a TORN one.
    (arm_dir / "failures.jsonl").write_text(
        '{"candidate_id": "placebo-g0-c0", "error": "boom"}\n{"candidate_id": "placebo-g0-c1", "err',
        encoding="utf-8",
    )
    opts = _opts(resume=True)
    opts["candidates"], opts["generations"] = 2, 1
    pool = _FakePool()
    # c0 is a cached failure (replayed); c1's line was torn -> NOT cached -> regenerated (1 call).
    summary = parallel._drive_llm_arm(arm, pool, opts, str(tmp_path))
    assert holder["t"].n_calls == 1
    assert pool.submitted == [f"{arm}-g0-c1"]
    assert summary["n_failed"] == 1  # the one cached failure (c0); c1 succeeded via the fake pool


# --------------------------------------------------------------------------- #
# 2026-07-06 S21: the SEARCH arms' parallel resume (random_search / bayes_opt) #
# --------------------------------------------------------------------------- #
class _HashPool(_FakePool):
    """_FakePool whose results carry the REAL sha256 of the materialized source, so the resume
    cache's hash verification sees exactly what a genuine worker would have archived."""

    def submit(self, spec: dict) -> Future:
        import hashlib as _hl

        fut = super().submit(spec)
        r = dict(fut.result())
        if spec["reward_kind"] == "coeffs":
            from src.baselines.reward_family import params_to_source

            src = params_to_source(
                list(spec["reward"]), cvar_alpha=float(spec.get("cvar_alpha", 0.05)),
                window=int(spec.get("window", 20)),
            )
        else:
            src = str(spec["reward"])
        r["reward_source"] = src
        r["reward_hash"] = _hl.sha256(src.encode("utf-8")).hexdigest()
        out: Future = Future()
        out.set_result(r)
        return out


def test_parallel_random_search_replays_archived_candidates_on_resume(tmp_path) -> None:
    """A resumed parallel random_search arm must SKIP archived candidates (zero re-training),
    reproduce the identical summary, and re-train EXACTLY the missing ones (2026-07-06 S21 —
    previously it re-trained all n and silently overwrote the records)."""
    import shutil

    opts = _opts(resume=False)
    pool1 = _HashPool()
    s1 = parallel._drive_search_arm("random_search", pool1, opts, str(tmp_path))
    assert len(pool1.submitted) == 4 and s1["n_candidates"] == 4

    # full resume: every candidate archived -> ZERO submissions, identical summary
    pool2 = _HashPool()
    s2 = parallel._drive_search_arm("random_search", pool2, _opts(resume=True), str(tmp_path))
    assert pool2.submitted == []
    assert s2["winner_fitness"] == s1["winner_fitness"] and s2["n_candidates"] == 4

    # partial resume: drop one record -> exactly that candidate re-trains
    shutil.rmtree(tmp_path / "random_search" / "random_search-c2")
    pool3 = _HashPool()
    s3 = parallel._drive_search_arm("random_search", pool3, _opts(resume=True), str(tmp_path))
    assert pool3.submitted == ["random_search-c2"]
    assert s3["n_candidates"] == 4 and s3["winner_fitness"] == s1["winner_fitness"]

    # resume OFF stays byte-unchanged behavior: everything re-trains
    pool4 = _HashPool()
    parallel._drive_search_arm("random_search", pool4, _opts(resume=False), str(tmp_path))
    assert len(pool4.submitted) == 4


def test_parallel_bayes_opt_replays_archived_evaluations_on_resume(tmp_path) -> None:
    """Same contract for the sequential-BO arm: archived evaluations replay (the GP trajectory is
    reconstructed from identical (x, y) history), zero re-training on a full resume."""
    opts = _opts(resume=False)
    pool1 = _HashPool()
    s1 = parallel._drive_search_arm("bayes_opt", pool1, opts, str(tmp_path))
    assert len(pool1.submitted) == 4 and s1["n_candidates"] == 4

    pool2 = _HashPool()
    s2 = parallel._drive_search_arm("bayes_opt", pool2, _opts(resume=True), str(tmp_path))
    assert pool2.submitted == []
    assert s2["winner_fitness"] == s1["winner_fitness"] and s2["n_candidates"] == 4


def test_parallel_search_resume_hash_mismatch_fails_loud(tmp_path) -> None:
    """An archived candidate whose source hash mismatches this run's draw at that index is a
    config/seed DRIFT — the resume must refuse loudly, never silently mix archives."""
    import json

    import pytest

    opts = _opts(resume=False)
    parallel._drive_search_arm("random_search", _HashPool(), opts, str(tmp_path))
    rec_path = tmp_path / "random_search" / "random_search-c1" / "record.json"
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    rec["reward_source_hash"] = "0" * 64  # forge a drifted archive
    rec_path.write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises((RuntimeError, ValueError), match="MISMATCH|CORRUPT"):
        parallel._drive_search_arm("random_search", _HashPool(), _opts(resume=True), str(tmp_path))


def test_advance_author_stream_forwards_and_noops(tmp_path) -> None:
    """2026-07-06: LLMClient.advance_author_stream forwards to a POSITIONAL transport's advance and
    silently no-ops on transports without one (real LLMs) — the duck-typed invariant, unit-pinned
    so a rename/logic slip cannot degrade to an all-tests-pass no-op."""
    from src.llm.client import LLMClient

    stub = _RealStubTransport(seed=9)
    client = LLMClient({"model": "stub"}, transport=stub)
    first = stub("s", "u")          # position 0 consumed directly
    client.advance_author_stream()  # consume position 1 by replay-advance
    third = client.complete("s", "u")  # must equal the uninterrupted run's position-2 output
    fresh = _RealStubTransport(seed=9)
    expected = [fresh("s", "u") for _ in range(3)]
    assert first == expected[0] and third == expected[2]
    # a transport WITHOUT advance: no-op, no raise
    client2 = LLMClient({"model": "x"}, transport=lambda s, u: "def reward(*a): return 0.0, {}, None")
    client2.advance_author_stream()  # must not raise


def test_partial_resume_authors_fresh_candidates_at_uninterrupted_positions(tmp_path, monkeypatch) -> None:
    """The stream-faithfulness END invariant at the driver level: after replaying gen-0 from the
    archive, the fresh gen-1 candidates must be BYTE-IDENTICAL to the ones an uninterrupted
    two-generation run authors (the crash-rehearsal catch, pinned as a fast unit test)."""
    holder = _install_counter(monkeypatch)
    arm = "placebo"

    # Uninterrupted reference: 2 generations x 2 candidates.
    ref_root = tmp_path / "ref"
    parallel._drive_llm_arm(arm, _FakePool(), _opts(resume=False), str(ref_root))
    from src.io.results import load_all
    ref_sources = {r["run_id"]: r["reward_source"] for r in load_all(ref_root / arm)}

    # Crash simulation: archive ONLY generation 0, then resume a full 2-gen run.
    crash_root = tmp_path / "crash"
    one_gen = _opts(resume=False)
    one_gen["candidates"], one_gen["generations"] = 2, 1
    parallel._drive_llm_arm(arm, _FakePool(), one_gen, str(crash_root))
    parallel._drive_llm_arm(arm, _FakePool(), _opts(resume=True), str(crash_root))
    res_sources = {r["run_id"]: r["reward_source"] for r in load_all(crash_root / arm)}

    assert set(res_sources) == set(ref_sources)
    assert res_sources == ref_sources  # gen-1 authored at positions 2,3 — not restarted at 0,1
    assert holder["t"].n_calls == 2  # the resume authored ONLY the two missing gen-1 candidates
