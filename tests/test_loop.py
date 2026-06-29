"""Tests for the LLM reward-discovery loop (FINAL_PLAN F.8, B.3).

Everything heavy is injected via fakes (no real LLM, training, or torch):
    - FakeLLM     : returns a fixed VALID numpy reward source and records prompts.
    - FakeEnv     : built per reward; yields synthetic train/val returns.
    - fake_trainer: returns a fake policy (carries a per-candidate fitness seed).

Covers: reloadable archive (load_run round-trips), reflection-prompt arm block
(distributional carries tail-stat lines; scalar does not), sandbox-gate failures
logged + skipped, single-shot budget, and best-fitness winner selection.
"""

from __future__ import annotations


import numpy as np

from src.io.results import load_run
from src.llm import loop
from src.selection.fitness import held_out_fitness


# A valid reward conforming to the contract; passes the sandbox AST gate. Shape-faithful to the
# production call (fixture-parity fix 2026-07-03): weights carries the cash slot (N+1) while returns
# is the risky leg (N) — run_loop's validation fixture now mirrors that, so the old
# `np.dot(weights, returns)` (equal-length assumption) would be REJECTED at validate_once.
_VALID_REWARD_SRC = (
    "def reward(weights, returns, prev_weights, port_ret, info):\n"
    "    total = float(np.dot(weights[:-1], returns))\n"
    "    return total, {'pnl': total}, None\n"
)

# Fails the AST gate (forbidden import).
_INVALID_REWARD_SRC = (
    "import os\n"
    "def reward(weights, returns, prev_weights, port_ret, info):\n"
    "    return 0.0, {}, None\n"
)


class FakeLLM:
    """Injected LLM stub: returns a fixed source and records every prompt."""

    def __init__(self, source: str, cfg: dict | None = None) -> None:
        self.source = source
        self.cfg = cfg or {"model": "fake-model-2026"}
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.prompts.append((system, user))
        return self.source


class FakePolicy:
    """A fake trained policy carrying a deterministic return level."""

    def __init__(self, level: float) -> None:
        self.level = level


class FakeEnv:
    """Per-reward fake env exposing train/val realized returns."""

    _counter = 0

    def __init__(self, reward_fn) -> None:
        self.reward_fn = reward_fn
        # Distinct level per construction so successive candidates differ.
        FakeEnv._counter += 1
        self.level = float(FakeEnv._counter)

    def train_env(self):
        return self

    def train_returns(self, policy) -> np.ndarray:
        rng = np.random.default_rng(int(self.level) * 7 + 1)
        # Negative-mean, heavy-ish tail so tail_stats are well-defined.
        return -0.001 + 0.01 * rng.standard_t(5, size=512)

    def val_returns(self, policy) -> np.ndarray:
        # Higher policy.level -> higher mean -> higher fitness (deterministic).
        rng = np.random.default_rng(int(self.level) * 13 + 3)
        return policy.level * 0.001 + 0.01 * rng.standard_normal(512)


def _make_trainer():
    """Trainer assigning ascending policy levels so winner selection is checkable."""
    state = {"n": 0}

    def _train(train_env) -> FakePolicy:
        state["n"] += 1
        return FakePolicy(level=float(state["n"]))

    return _train


def _measurement_factory():
    from src.feedback.measurement import ReturnDistribution

    return ReturnDistribution()


def _run(arm: str, source: str, cfg: dict, archive_root):
    """Run the loop with the standard fakes and return (archive, llm)."""
    FakeEnv._counter = 0
    llm = FakeLLM(source)
    archive = loop.run_loop(
        arm=arm,
        env_builder=FakeEnv,
        llm=llm,
        agent_trainer=_make_trainer(),
        measurement=_measurement_factory,
        fitness_fn=held_out_fitness,
        cfg=cfg,
        archive_root=archive_root,
    )
    return archive, llm


def test_two_candidate_one_generation_archives(tmp_path) -> None:
    """A 2-candidate / 1-generation run archives reloadable artifacts."""
    cfg = {"generations": 1, "candidates_per_gen": 2, "budget": 100, "n_trials": 5}
    archive, _ = _run("distributional", _VALID_REWARD_SRC, cfg, tmp_path)

    assert len(archive.candidates) == 2
    for cand in archive.candidates:
        record = load_run(f"run-{cand.candidate_id}", tmp_path)
        assert record["arm"] == "distributional"
        assert record["reward_source_hash"] == cand.reward_hash
        # reward.py is reattached on load -> replay, not regenerate (C-2).
        assert "reward_source" in record
        assert record["reward_source"] == _VALID_REWARD_SRC
        assert record["metrics"]["val_fitness"] == cand.val_fitness


def test_reflection_prompt_distributional_has_tail_lines_scalar_does_not(tmp_path) -> None:
    """Distributional reflection prompt carries tail-stat lines; scalar does not."""
    cfg = {"generations": 2, "candidates_per_gen": 1, "budget": 100, "n_trials": 5}

    dist_archive, dist_llm = _run("distributional", _VALID_REWARD_SRC, cfg, tmp_path / "d")
    # Gen 0 prompt is the initial prompt; gen 1 is the reflection prompt.
    gen1_user = dist_llm.prompts[1][1]
    assert "tail diagnostics" in gen1_user
    assert "CVaR 5%" in gen1_user
    assert "left-tail" in gen1_user

    scalar_archive, scalar_llm = _run("scalar", _VALID_REWARD_SRC, cfg, tmp_path / "s")
    scalar_gen1_user = scalar_llm.prompts[1][1]
    assert "CVaR" not in scalar_gen1_user
    assert "tail diagnostics" not in scalar_gen1_user
    # Both produced one accepted candidate per generation.
    assert len(dist_archive.candidates) == 2
    assert len(scalar_archive.candidates) == 2


def test_validation_failure_logged_and_skipped(tmp_path) -> None:
    """A candidate failing the sandbox gate is logged and skipped, not fatal."""
    cfg = {"generations": 1, "candidates_per_gen": 2, "budget": 100, "n_trials": 5}
    archive, _ = _run("scalar", _INVALID_REWARD_SRC, cfg, tmp_path)

    # No candidate accepted; both draws failed and were recorded.
    assert archive.candidates == []
    assert len(archive.failures) == 2
    assert all("error" in f for f in archive.failures)
    assert archive.winner() is None


def test_single_shot_draws_full_budget_in_one_generation(tmp_path) -> None:
    """A single-shot arm spends the whole matched budget in one generation — and ``budget_spent``
    records the ACTUAL draws (F15: candidates_per_gen per generation), not the even-split PLAN of the
    cfg ``budget`` (which over-reported whenever the two disagreed)."""
    cfg = {"generations": 1, "candidates_per_gen": 3, "budget": 3, "n_trials": 5}
    archive, llm = _run("scalar", _VALID_REWARD_SRC, cfg, tmp_path)

    assert archive.meta["generations"] == 1
    assert archive.meta["budget_spent"] == 3  # actual draws == the whole matched budget, in gen 0
    # One generation -> every draw uses the INITIAL prompt (reflection never engages).
    users = [u for _, u in llm.prompts]
    assert len(users) == 3 and len(set(users)) == 1
    assert all(c.generation == 0 for c in archive.candidates)


def test_winner_is_best_fitness_candidate(tmp_path) -> None:
    """The winner is the accepted candidate with the highest validation fitness."""
    cfg = {"generations": 1, "candidates_per_gen": 4, "budget": 100, "n_trials": 5}
    archive, _ = _run("distributional", _VALID_REWARD_SRC, cfg, tmp_path)

    assert len(archive.candidates) == 4
    winner = archive.winner()
    best = max(c.val_fitness for c in archive.candidates)
    assert winner is not None
    assert winner.val_fitness == best


# --------------------------------------------------------------------------- #
# Prompt-variation diversity (temperature-free within-generation variety)      #
# --------------------------------------------------------------------------- #
def test_diversity_prompt_variation_makes_candidate_prompts_distinct(tmp_path) -> None:
    """With diversity ON the K candidates get DISTINCT prompts (per-candidate directive); with it
    OFF they're identical. This is the mechanism that lets a temperature-rejecting author
    (Claude Opus 5) still get within-generation variety. Uniform across arms -> no H2 confound."""
    cfg_off = {"generations": 1, "candidates_per_gen": 3, "budget": 99, "n_trials": 5}
    _, llm_off = _run("scalar", _VALID_REWARD_SRC, cfg_off, tmp_path / "off")
    prompts_off = [u for _, u in llm_off.prompts]
    assert len(prompts_off) == 3
    assert len(set(prompts_off)) == 1  # identical without diversity

    cfg_on = {**cfg_off, "diversity_prompt_variation": True}
    _, llm_on = _run("scalar", _VALID_REWARD_SRC, cfg_on, tmp_path / "on")
    prompts_on = [u for _, u in llm_on.prompts]
    assert len(prompts_on) == 3
    assert len(set(prompts_on)) == 3  # all distinct
    assert all("Exploration directive" in p for p in prompts_on)
    # Each candidate prompt is the shared base prompt + a distinct per-candidate suffix.
    base = prompts_off[0]
    assert all(p.startswith(base) for p in prompts_on)


def test_diversity_skipped_for_single_candidate(tmp_path) -> None:
    """No directive when candidates_per_gen == 1 (nothing to diversify within the generation)."""
    cfg = {
        "generations": 1, "candidates_per_gen": 1, "budget": 99, "n_trials": 5,
        "diversity_prompt_variation": True,
    }
    _, llm = _run("scalar", _VALID_REWARD_SRC, cfg, tmp_path)
    assert "Exploration directive" not in llm.prompts[0][1]


def test_diversity_directive_uniform_across_arms() -> None:
    """The directive depends only on (cidx, n) — identical for every arm, so it cannot bias the
    distributional-vs-scalar contrast (only the arm feedback block differs)."""
    assert loop._diversity_directive(0, 5) == loop._diversity_directive(0, 5)
    assert loop._diversity_directive(0, 5) != loop._diversity_directive(1, 5)


# --------------------------------------------------------------------------- #
# Search-replay cache (resume): a mid-search crash REPLAYS archived candidates  #
# instead of re-calling the paid, non-deterministic LLM + retraining. This both #
# saves cost/GPU time AND makes resume byte-faithful (same candidates -> winner)#
# --------------------------------------------------------------------------- #
class _RaisingLLM:
    """An LLM stub that fails if called — proves a full cache hit touches no API."""

    cfg = {"model": "fake-model-2026"}

    def complete(self, system: str, user: str) -> str:  # noqa: ARG002
        raise AssertionError("LLM.complete called on a full-cache resume (should have replayed)")


def _raising_trainer(train_env):  # noqa: ARG001
    raise AssertionError("trainer called on a full-cache resume (should have replayed)")


class _CountingLLM(FakeLLM):
    """Counts real completions so a PARTIAL resume can assert exactly the misses regenerate."""


def _resume_loop(arm, archive_root, *, llm, trainer, cfg):
    return loop.run_loop(
        arm=arm, env_builder=FakeEnv, llm=llm, agent_trainer=trainer,
        measurement=_measurement_factory, fitness_fn=held_out_fitness,
        cfg=dict(cfg, resume=True), archive_root=archive_root,
    )


def test_resume_replays_all_candidates_without_llm_or_trainer(tmp_path) -> None:
    """A resume over a complete archive replays every candidate — no LLM, no training — and
    reproduces the identical winner and per-candidate fitnesses (byte-faithful search replay)."""
    cfg = {"generations": 2, "candidates_per_gen": 2, "budget": 100, "n_trials": 5}
    first, llm1 = _run("distributional", _VALID_REWARD_SRC, cfg, tmp_path)
    assert len(first.candidates) == 4 and len(llm1.prompts) == 4  # 4 real calls the first time

    again = _resume_loop("distributional", tmp_path, llm=_RaisingLLM(), trainer=_raising_trainer, cfg=cfg)
    assert len(again.candidates) == 4
    assert again.winner().candidate_id == first.winner().candidate_id
    assert [c.val_fitness for c in again.candidates] == [c.val_fitness for c in first.candidates]
    assert [c.reward_hash for c in again.candidates] == [c.reward_hash for c in first.candidates]


def test_resume_replays_archived_failures_without_llm(tmp_path) -> None:
    """A sandbox-rejected candidate is replayed from the failures-ledger on resume, so it is NOT
    regenerated against the non-deterministic author (which could newly succeed and change the set)."""
    cfg = {"generations": 1, "candidates_per_gen": 2, "budget": 100, "n_trials": 5}
    first, llm1 = _run("scalar", _INVALID_REWARD_SRC, cfg, tmp_path)
    assert first.candidates == [] and len(first.failures) == 2 and len(llm1.prompts) == 2

    again = _resume_loop("scalar", tmp_path, llm=_RaisingLLM(), trainer=_raising_trainer, cfg=cfg)
    assert again.candidates == []
    assert len(again.failures) == 2  # both replayed from the ledger, no API call


def test_llm_api_failure_skips_candidate_without_crashing(tmp_path) -> None:
    """A candidate whose LLM call raises (e.g. API exhaustion after the client's retries) is logged
    and SKIPPED — the arm survives, and it is NOT cached as a sandbox-failure (so --resume re-attempts
    it). This is the graceful-degradation guard that keeps a 2-week unattended run alive."""

    class _BoomThenOkLLM:
        cfg = {"model": "fake-model-2026"}

        def __init__(self, source: str) -> None:
            self.source = source
            self.calls = 0

        def complete(self, system: str, user: str) -> str:  # noqa: ARG002
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("simulated API exhaustion after bounded retries")
            return self.source

    cfg = {"generations": 1, "candidates_per_gen": 2, "budget": 100, "n_trials": 5}
    FakeEnv._counter = 0
    llm = _BoomThenOkLLM(_VALID_REWARD_SRC)
    archive = loop.run_loop(
        arm="scalar", env_builder=FakeEnv, llm=llm, agent_trainer=_make_trainer(),
        measurement=_measurement_factory, fitness_fn=held_out_fitness, cfg=cfg, archive_root=tmp_path,
    )
    assert llm.calls == 2  # both candidates attempted; the arm did NOT crash on the first failure
    assert len(archive.candidates) == 1  # the second candidate succeeded
    assert len(archive.failures) == 0  # an API error is not a (cached) sandbox failure
    assert not (tmp_path / "run-scalar.failures.jsonl").exists()  # nothing written to the ledger
    # C3c (ops audit 2026-07-02): the skip is COUNTED in the run meta so the campaign's C3b
    # winner-selection floor can see a resumably-short pool (accepted+failed+skips == budget drawn).
    assert archive.meta["llm_error_skips"] == 1


def test_llm_error_skips_counted_and_emitted_as_monitor_anomaly(tmp_path) -> None:
    """C3c: EVERY llm_error skip increments the meta counter AND emits a monitor anomaly of kind
    'llm_error' (loud on the dashboard/anomalies.jsonl, not just a transient log line); a healthy
    run keeps the counter at 0 and emits nothing."""

    class _AlwaysExplodingLLM:
        def complete(self, system: str, user: str) -> str:  # noqa: ARG002
            raise RuntimeError("RateLimitError: simulated sustained outage")

    class _SpyMonitor:
        """Records anomaly + candidate_done; every other RunMonitor method is a no-op."""

        def __init__(self) -> None:
            self.anomalies: list[tuple[str, str]] = []
            self.statuses: list[str] = []

        def anomaly(self, kind: str, detail: str, **_fields) -> None:
            self.anomalies.append((kind, detail))

        def candidate_done(self, _arm, _cand, *, fitness, status, secs=None) -> None:  # noqa: ANN001, ARG002
            self.statuses.append(status)

        def __getattr__(self, _name: str):
            return lambda *a, **k: None

    spy = _SpyMonitor()
    cfg = {"generations": 1, "candidates_per_gen": 2, "budget": 100, "n_trials": 5, "monitor": spy}
    FakeEnv._counter = 0
    archive = loop.run_loop(
        arm="scalar", env_builder=FakeEnv, llm=_AlwaysExplodingLLM(), agent_trainer=_make_trainer(),
        measurement=_measurement_factory, fitness_fn=held_out_fitness, cfg=cfg, archive_root=tmp_path,
    )
    assert archive.meta["llm_error_skips"] == 2                 # every budgeted slot skipped
    assert archive.meta["accepted"] == 0 and archive.meta["failed"] == 0
    assert [k for k, _d in spy.anomalies] == ["llm_error", "llm_error"]
    assert all("--resume re-attempts" in d for _k, d in spy.anomalies)  # the recovery is NAMED
    assert spy.statuses == ["llm_error", "llm_error"]

    # Healthy run: counter stays 0, no anomalies (the counter never fires on success paths).
    spy2 = _SpyMonitor()
    ok_cfg = {"generations": 1, "candidates_per_gen": 2, "budget": 100, "n_trials": 5, "monitor": spy2}
    ok_archive, _ = _run("scalar", _VALID_REWARD_SRC, ok_cfg, tmp_path / "ok")
    assert ok_archive.meta["llm_error_skips"] == 0
    assert spy2.anomalies == []


def test_resume_regenerates_only_the_missing_candidate(tmp_path) -> None:
    """Partial resume: with one archived record removed, ONLY that candidate is regenerated
    (one LLM call), the rest replay — the crash-recovery guarantee."""
    import shutil

    cfg = {"generations": 1, "candidates_per_gen": 3, "budget": 100, "n_trials": 5}
    first, _ = _run("distributional", _VALID_REWARD_SRC, cfg, tmp_path)
    assert len(first.candidates) == 3

    # Simulate a crash that lost exactly one in-flight candidate's archive.
    shutil.rmtree(tmp_path / "run-distributional-g0-c1")

    FakeEnv._counter = 0
    llm = _CountingLLM(_VALID_REWARD_SRC)
    again = _resume_loop("distributional", tmp_path, llm=llm, trainer=_make_trainer(), cfg=cfg)
    assert len(llm.prompts) == 1  # exactly the missing candidate was regenerated
    assert len(again.candidates) == 3  # the other two replayed from the archive
