"""Tests for the σ_D pilot archive GENERATOR (scripts/run_sigma_pilot_train.py).

The generator trains the two FIXED rewards (differential_sharpe, return_minus_cvar) across a shared
CRN seed set and archives per-seed TEST records in the exact schema ``scripts/sigma_seed_pilot.py``
reads. These tests pin the load-bearing GENERATE -> ANALYZE contract WITHOUT torch by stubbing the
per-cell training (``train_one``) — the same pattern ``tests/test_learning_curve.py`` uses:

  - the archive LAYOUT is ``<out-dir>/<reward>/<reward>-s{seed}/record.json`` (the layout
    ``analyze_campaign.load_campaign_records`` walks), one record per (reward, seed);
  - each record carries the analyzer's required schema: ``arm`` == the reward key, ``seed``, and
    ``metrics['test_returns']`` (via the shared ``build_test_record``);
  - the analyzer (``sigma_seed_pilot.run_pilot``) reads the generated archive back and returns a
    real σ_D/ρ result with ``status='ok'`` for the Sharpe leg (NOT skipped) + the evidence flag;
  - ``--resume`` skips (reward, seed) cells already on disk;
  - a failed cell is reported, never aborts the sweep.

The real-training path (make_agent_trainer + SAC) is exercised by the CLI smoke on the synthetic
panel (documented in the module docstring); it is not run here so the suite stays torch-free + fast.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from scripts import run_sigma_pilot_train as gen  # noqa: E402


# --------------------------------------------------------------------------- #
# A deterministic synthetic archive (no torch): distinct per-seed test paths    #
# so σ_a, σ_b > 0 and ρ is defined, exercising the analyzer's status='ok' path. #
# --------------------------------------------------------------------------- #
def _fake_test_returns(reward_key: str, seed: int) -> np.ndarray:
    """A short, seed-varying, per-reward test-return path (deterministic; finite; length 40)."""
    # Salt-free, deterministic per-(reward, seed) RNG (Python's hash() is PYTHONHASHSEED-salted).
    arm_offset = 1000 if reward_key == "return_minus_cvar" else 0
    rng = np.random.default_rng(arm_offset + int(seed))
    base = 0.001 * (1 + seed)  # per-seed mean shift -> across-seed variance in the score
    return (base + 0.01 * rng.standard_normal(40)).astype(float)


def _install_fake_train_one(monkeypatch, *, fail_seeds: set[int] | None = None) -> None:
    """Patch ``train_one`` to build a real ``build_test_record`` from a synthetic path (no training)."""
    from src.orchestration.test_leg import build_test_record

    fail_seeds = fail_seeds or set()

    def _stub(reward_key, seed, *, test_window, **_kw):  # noqa: ANN001, ANN003
        if int(seed) in fail_seeds:
            return {"record": None, "ok": False, "error": "ValueError: injected", "seconds": 0.0}
        tr = _fake_test_returns(reward_key, int(seed))
        winner = {
            "arm": reward_key,
            "candidate_id": f"{reward_key}-pilot",
            "generation": 0,
            "reward_source": f"# sigma_pilot:{reward_key}\n",
            "reward_source_hash": "0" * 64,
            "feedback_block": "",
            "metrics": {"val_fitness": float("nan")},
        }
        rec = build_test_record(
            winner=winner, arm=reward_key, seed=int(seed), reward_hash="0" * 64,
            env_fp=f"sigma_pilot:{reward_key}", test_returns=tr,
        )
        return {"record": rec, "ok": True, "error": None, "seconds": 0.01}

    # Avoid the real panel/window/agent resolution (which loads config + would build torch envs).
    def _fake_resolve(synthetic, end, embargo, lookback):  # noqa: ANN001
        return object(), {"synthetic": True}, (60, 330), (390, 450), (510, 600)

    def _fake_agent_cfg(budget, device):  # noqa: ANN001
        return {"train_steps_per_candidate": int(budget), "device": device}

    monkeypatch.setattr(gen, "train_one", _stub)
    monkeypatch.setattr(gen, "_resolve_panel_and_windows", _fake_resolve)
    monkeypatch.setattr(gen, "_resolve_agent_cfg", _fake_agent_cfg)


# --------------------------------------------------------------------------- #
# Layout + schema                                                              #
# --------------------------------------------------------------------------- #
def test_generates_archive_layout_and_schema(tmp_path, monkeypatch) -> None:
    _install_fake_train_one(monkeypatch)
    summary = gen.run_pilot_training(
        budget=500, n_seeds=3, out_dir=tmp_path, synthetic=True, device="cpu", end="x",
    )
    # Two arms x 3 seeds written, layout <out>/<reward>/<reward>-s{seed}/record.json.
    for reward in gen.PILOT_REWARDS:
        assert summary["per_arm"][reward]["written"] == 3
        for seed in range(3):
            rec_path = tmp_path / reward / f"{reward}-s{seed}" / "record.json"
            assert rec_path.is_file(), rec_path

    # The analyzer's required per-record fields (via src.io.results.load_all).
    from src.io.results import load_all

    recs = load_all(tmp_path / "differential_sharpe")
    assert len(recs) == 3
    r = recs[0]
    assert r["arm"] == "differential_sharpe"
    assert isinstance(r["seed"], int)
    assert isinstance(r["metrics"]["test_returns"], list) and len(r["metrics"]["test_returns"]) == 40


# --------------------------------------------------------------------------- #
# The GENERATE -> ANALYZE round-trip (the load-bearing contract)               #
# --------------------------------------------------------------------------- #
def test_generated_archive_feeds_sigma_seed_pilot_status_ok(tmp_path, monkeypatch) -> None:
    """sigma_seed_pilot reads the generated archive back and returns σ_D/ρ with status='ok'."""
    _install_fake_train_one(monkeypatch)
    gen.run_pilot_training(
        budget=500, n_seeds=4, out_dir=tmp_path, synthetic=True, device="cpu", end="x",
    )

    import sigma_seed_pilot as ssp  # scripts/ is on sys.path
    from analyze_campaign import load_campaign_records

    records = load_campaign_records(str(tmp_path))
    result = ssp.run_pilot(records, arm_a="differential_sharpe", arm_b="return_minus_cvar")

    sharpe = result["per_statistic"]["sharpe"]["stats"]
    assert sharpe["status"] == "ok"  # NOT skipped -> the freeze-blocker evidence exists
    assert sharpe["n_shared"] == 4
    assert sharpe["sigma_seed"] is not None and sharpe["sigma_seed"] >= 0.0
    assert sharpe["rho"] is not None  # both arms have across-seed variance -> ρ defined
    assert result["sigma_seed_pilot"] is True  # the flag determine_design reads to flip n_seeds


# --------------------------------------------------------------------------- #
# Resume + graceful failure                                                    #
# --------------------------------------------------------------------------- #
def test_resume_skips_already_archived(tmp_path, monkeypatch) -> None:
    _install_fake_train_one(monkeypatch)
    gen.run_pilot_training(budget=500, n_seeds=3, out_dir=tmp_path, synthetic=True, device="cpu", end="x")

    # On resume every cell is already on disk -> nothing re-written.
    summary2 = gen.run_pilot_training(
        budget=500, n_seeds=3, out_dir=tmp_path, synthetic=True, device="cpu", end="x", resume=True,
    )
    for reward in gen.PILOT_REWARDS:
        assert summary2["per_arm"][reward]["written"] == 0


def test_failed_cell_reported_not_raised(tmp_path, monkeypatch) -> None:
    _install_fake_train_one(monkeypatch, fail_seeds={1})
    summary = gen.run_pilot_training(
        budget=500, n_seeds=3, out_dir=tmp_path, synthetic=True, device="cpu", end="x",
    )
    for reward in gen.PILOT_REWARDS:
        assert summary["per_arm"][reward]["written"] == 2  # seeds 0, 2
        assert summary["per_arm"][reward]["failed"] == 1   # seed 1
        assert summary["per_arm"][reward]["errors"]


def test_build_parser_defaults() -> None:
    args = gen.build_parser().parse_args([])
    assert args.budget == 50000
    assert args.n_seeds == 15
    assert args.out_dir == "outputs/sigma_pilot"
    assert args.device == "auto"
    assert args.gpu == 1  # farm mode is strictly opt-in; the default is the serial path


# --------------------------------------------------------------------------- #
# Farm mode (--gpu N>=2): specs, worker, N==1 parity, resume, aggregation       #
# --------------------------------------------------------------------------- #
def _specs_kwargs(tmp_path: Path) -> dict:
    """The keyword bundle build_cell_specs shares with the farm driver (synthetic-window shaped)."""
    return dict(
        out_dir=tmp_path,
        panel_descriptor={"synthetic": True},
        env_cfg={"state": {"lookback_days": 60}},
        agent_cfg={"train_steps_per_candidate": 500, "device": "auto"},
        train_window=(60, 330), val_window=(390, 450), test_window=(510, 600),
        embargo=21, lookback=60,
    )


def _inproc_runner_factory(ran: list, *, reverse: bool = False):
    """A no-spawn run_recycling stand-in: maps the worker over specs in-process (test_leg pattern).

    ``reverse=True`` EXECUTES the cells backwards (still returning results in submission order, the
    run_recycling contract) — proving the archive/summary do not depend on execution order.
    """

    def _runner(specs, *, worker, n_gpu, n_cpu, recycle_every):  # noqa: ANN001, ARG001
        order = list(reversed(specs)) if reverse else list(specs)
        results = {}
        for s in order:
            ran.append(s["run_id"])
            results[s["run_id"]] = worker({**s, "device": "cpu"})  # the pool's device-token injection
        return [results[s["run_id"]] for s in specs]

    return _runner


def test_build_cell_specs_matches_serial_iteration_order_and_content(tmp_path) -> None:
    """(a) The spec list IS the serial nest: reward-major, seed-minor, same ids/paths/configs."""
    seeds = [0, 1, 2]
    specs = gen.build_cell_specs(gen.PILOT_REWARDS, seeds, **_specs_kwargs(tmp_path))
    assert [(s["arm"], s["seed"]) for s in specs] == [
        (r, sd) for r in gen.PILOT_REWARDS for sd in seeds
    ]
    # Same resume key + the same per-arm output root the serial loop writes under.
    assert specs[0]["run_id"] == f"{gen.PILOT_REWARDS[0]}-s0"
    assert specs[0]["arm_root"] == str(tmp_path / gen.PILOT_REWARDS[0])
    # Every cell carries the SAME configs/windows the serial loop hands train_one.
    for s in specs:
        assert s["agent_cfg"]["train_steps_per_candidate"] == 500
        assert (tuple(s["train_window"]), tuple(s["val_window"]), tuple(s["test_window"])) == (
            (60, 330), (390, 450), (510, 600)
        )
        assert s["embargo"] == 21 and s["lookback"] == 60
        assert s["panel_descriptor"] == {"synthetic": True}
    # done_ids drops exactly the done cells and preserves the remaining serial order (the resume key).
    done = {f"{gen.PILOT_REWARDS[0]}-s1"}
    kept = gen.build_cell_specs(gen.PILOT_REWARDS, seeds, done_ids=done, **_specs_kwargs(tmp_path))
    assert [(s["arm"], s["seed"]) for s in kept] == [
        (r, sd) for r in gen.PILOT_REWARDS for sd in seeds if f"{r}-s{sd}" not in done
    ]


def test_gpu_1_keeps_serial_call_shape_and_never_farms(tmp_path, monkeypatch) -> None:
    """(b) --gpu 1 (the default) is the pre-farm serial path verbatim: same train_one call sequence,
    and the farm machinery (specs/pool/preflight) is never touched."""
    _install_fake_train_one(monkeypatch)
    fake = gen.train_one  # the stub _install_fake_train_one installed
    calls: list[tuple[str, int]] = []

    def _recording(reward_key, seed, **kw):  # noqa: ANN001, ANN003
        calls.append((reward_key, int(seed)))
        return fake(reward_key, seed, **kw)

    monkeypatch.setattr(gen, "train_one", _recording)

    def _boom(*_a, **_k):  # noqa: ANN002, ANN003
        raise AssertionError("farm machinery must not run at --gpu 1")

    monkeypatch.setattr(gen, "_run_farm", _boom)
    monkeypatch.setattr(gen, "_preflight_farm_ram", _boom)

    summary = gen.run_pilot_training(
        budget=500, n_seeds=2, out_dir=tmp_path, synthetic=True, device="cpu", end="x", gpu=1,
    )
    # The serial nest, in order, one call per (reward, seed) — and the serial archive layout.
    assert calls == [(r, s) for r in gen.PILOT_REWARDS for s in (0, 1)]
    for reward in gen.PILOT_REWARDS:
        assert summary["per_arm"][reward]["written"] == 2
        assert (tmp_path / reward / f"{reward}-s0" / "record.json").is_file()


def test_worker_is_picklable_and_seeds_before_training(tmp_path, monkeypatch) -> None:
    """(c) The farm worker pickles by reference (spawn requirement) and its cell re-seeds the FULL
    stack BEFORE the env build / training (the CRN property farming must preserve)."""
    import pickle

    assert pickle.loads(pickle.dumps(gen._pilot_cell_worker)) is gen._pilot_cell_worker

    events: list[str] = []

    def _fake_seed(seed, *, deterministic_torch=False):  # noqa: ANN001
        events.append(f"seed:{int(seed)}:{bool(deterministic_torch)}")
        return int(seed)

    class _Bundle:
        def train_env(self):
            events.append("train_env")
            return "env"

        def test_returns(self, policy):  # noqa: ANN001, ARG002
            return np.array([0.01, -0.02, 0.03, 0.0, -0.01])

    def _fake_builder(panel, env_cfg, tw, vw, **kw):  # noqa: ANN001, ANN003, ARG001
        events.append("env_build")
        return lambda reward_fn: _Bundle()

    def _fake_trainer_factory(cfg, seed):  # noqa: ANN001, ARG001
        def _train(env):  # noqa: ANN001, ARG001
            events.append("train")
            return "policy"

        return _train

    # The REAL train_one with its stack stubbed at the source modules (all torch-free module tops);
    # train_one's late `from X import Y` binds whatever is patched at call time.
    monkeypatch.setattr("src.utils.seeding.set_global_seed", _fake_seed)
    monkeypatch.setattr("src.env.runner.make_env_builder", _fake_builder)
    monkeypatch.setattr("src.agents.trainer.make_agent_trainer", _fake_trainer_factory)
    monkeypatch.setattr("src.orchestration.test_leg._load_test_panel", lambda d: "panel")

    spec = gen.build_cell_specs(("differential_sharpe",), [3], **_specs_kwargs(tmp_path))[0]
    out = gen._pilot_cell_worker({**spec, "device": "cpu"})

    assert out["ok"] is True, out
    assert out["arm"] == "differential_sharpe" and out["seed"] == 3
    # Seeding happens FIRST — before the env is even built, and strictly before training.
    assert events == ["seed:3:True", "env_build", "train_env", "train"]
    # ...and the record landed at the SAME location the serial loop writes.
    assert (tmp_path / "differential_sharpe" / "differential_sharpe-s3" / "record.json").is_file()


def test_farm_resume_skips_existing_records(tmp_path, monkeypatch) -> None:
    """(d) Idempotence: a farmed re-run with --resume submits NOTHING for cells already archived."""
    _install_fake_train_one(monkeypatch)
    monkeypatch.setattr(gen, "_preflight_farm_ram", lambda n: None)  # unit test: no host-RAM gate

    ran: list[str] = []
    runner = _inproc_runner_factory(ran)
    summary = gen.run_pilot_training(
        budget=500, n_seeds=3, out_dir=tmp_path, synthetic=True, device="cpu", end="x",
        gpu=2, runner=runner,
    )
    assert len(ran) == 6
    for reward in gen.PILOT_REWARDS:
        assert summary["per_arm"][reward]["written"] == 3
        for seed in range(3):
            assert (tmp_path / reward / f"{reward}-s{seed}" / "record.json").is_file()

    ran.clear()
    summary2 = gen.run_pilot_training(
        budget=500, n_seeds=3, out_dir=tmp_path, synthetic=True, device="cpu", end="x",
        gpu=2, runner=runner, resume=True,
    )
    assert ran == []  # every record already on disk -> zero specs reach the pool
    for reward in gen.PILOT_REWARDS:
        assert summary2["per_arm"][reward]["written"] == 0


def test_farm_archive_byte_identical_to_serial_even_out_of_order(tmp_path, monkeypatch) -> None:
    """Farmed records == serial records BYTE-for-byte, even when cells execute in reversed order —
    the archive (what sigma_seed_pilot reads) is independent of scheduling."""
    _install_fake_train_one(monkeypatch)
    monkeypatch.setattr(gen, "_preflight_farm_ram", lambda n: None)

    gen.run_pilot_training(
        budget=500, n_seeds=3, out_dir=tmp_path / "serial", synthetic=True, device="cpu", end="x",
    )
    ran: list[str] = []
    gen.run_pilot_training(
        budget=500, n_seeds=3, out_dir=tmp_path / "farm", synthetic=True, device="cpu", end="x",
        gpu=3, runner=_inproc_runner_factory(ran, reverse=True),
    )
    assert ran == list(reversed([f"{r}-s{s}" for r in gen.PILOT_REWARDS for s in range(3)]))
    for reward in gen.PILOT_REWARDS:
        for seed in range(3):
            rel = Path(reward) / f"{reward}-s{seed}" / "record.json"
            a = (tmp_path / "serial" / rel).read_bytes()
            b = (tmp_path / "farm" / rel).read_bytes()
            assert a == b, f"farmed record differs from serial: {rel}"


def test_farm_summary_is_order_independent(monkeypatch) -> None:
    """(e) The per-arm summary folds identically under ANY result order (shuffle -> same summary)."""
    import random

    seeds = list(range(4))
    results = []
    for reward in gen.PILOT_REWARDS:
        for seed in seeds:
            ok = not (reward == gen.PILOT_REWARDS[0] and seed == 2)
            results.append({
                "ok": ok, "run_id": f"{reward}-s{seed}", "arm": reward, "seed": seed,
                "error": None if ok else "ValueError: injected", "seconds": float(1 + seed),
            })
    baseline = gen._summarize_farm(results, gen.PILOT_REWARDS, seeds)
    for k in range(5):
        shuffled = list(results)
        random.Random(k).shuffle(shuffled)
        assert gen._summarize_farm(shuffled, gen.PILOT_REWARDS, seeds) == baseline
    # The folded fields mirror the serial bookkeeping (counts, seed-ordered errors, ok-cell median).
    a0 = baseline[gen.PILOT_REWARDS[0]]
    assert a0["written"] == 3 and a0["failed"] == 1 and a0["n_seeds"] == 4
    assert a0["errors"] == ["seed 2: ValueError: injected"]
    assert a0["seconds_median"] == 2.0  # median over ok seconds {1, 2, 4}
    assert baseline[gen.PILOT_REWARDS[1]]["seconds_median"] == 2.5  # all ok: {1, 2, 3, 4}


def test_farm_ram_preflight_blocks_with_clear_message(monkeypatch) -> None:
    """The preflight gate refuses to spawn when free RAM < ~2.5 GB/worker, naming both numbers."""

    class _Low:
        available = 4 * 2**30  # 4 GB free < the ~7.5 GB three workers need

    monkeypatch.setattr("psutil.virtual_memory", lambda: _Low)
    with pytest.raises(RuntimeError, match=r"needs ~7\.5 GB"):
        gen._preflight_farm_ram(3)

    class _Enough:
        available = 8 * 2**30

    monkeypatch.setattr("psutil.virtual_memory", lambda: _Enough)
    gen._preflight_farm_ram(3)  # no raise
