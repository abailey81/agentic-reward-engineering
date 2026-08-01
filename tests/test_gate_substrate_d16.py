"""D16 — the C3 review gate must SEE a substrate mix, not just a device-label mix.

THE DEFECT (found 2026-07-30, applied 2026-08-01, record §97). ``write_integrity_report``'s
``health_ok`` is the ONLY field the review gate reads, and its stop message claims it fires on
"device inhomogeneity". It was computed from ``crn_pair_device_consistent``, which keys on the
cpu/cuda LABEL — so on a CPU-only lane every record looks identical to it. The one inhomogeneity
that actually occurred in RUN 4 — four ``baseline_volatility_scaled_return`` records on an Intel
Xeon Gold 6140 against 26 on a Gold 6240 — passed the gate silently, while the ADVISORY sentinel
(``check_substrate_fields``) rated the same thing CRITICAL.

WHY THESE TESTS FAIL AGAINST THE PRE-FIX CODE (the requirement that makes them worth having):
``test_substrate_mix_at_a_shared_seed_fails_the_gate`` builds two units whose records at the SAME
seed carry DIFFERENT ``cpu.model_name`` and IDENTICAL device labels. Pre-fix, ``health_ok`` is True
because nothing consults ``env.json``'s cpu block; post-fix it is False. The three companions pin
that the fix did not simply make the gate stricter about everything: a homogeneous leg still passes,
a device-only mix still fails as it did before (no regression), and a missing ``env.json`` is a
WILDCARD rather than a failure — because a capture gap must not stop a line.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.cluster.integrity import write_integrity_report

XEON_6240 = "Intel(R) Xeon(R) Gold 6240 CPU @ 2.60GHz"
XEON_6140 = "Intel(R) Xeon(R) Gold 6140 CPU @ 2.30GHz"


class _Run:
    """The minimal ClusterRun surface write_integrity_report touches."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self.read_root = str(root)

    def search_read(self) -> Path:
        return self._root / "search"

    def test_read(self) -> Path:
        return self._root / "test"

    def line_tag(self) -> str:
        return "unit_test"


def _write(root: Path, arm: str, seed: int, *, cpu: str | None, device: str = "cpu",
           gpu: str | None = None) -> None:
    """One archived test record + its env.json sibling.

    NOTE: The record's ``env_fingerprint.env_json_sha256`` must be the REAL digest of the env.json
    written beside it: ``src/io/results.load_run`` verifies the two against each other and raises
    on a mismatch (provenance tamper-evidence). The fixture computes it with the SAME
    ``sha256_obj`` canonicalisation the writer uses, rather than faking a value — a fixture that
    sidestepped the check would also sidestep the thing being tested.
    """
    from src.utils.provenance import sha256_obj

    env_obj = None
    if cpu is not None:
        env_obj = {
            "cpu": {"model_name": cpu, "logical_cores": 36},
            "determinism_env": {"OMP_NUM_THREADS": "1"},
            "torch_cuda": {"num_threads": 1, "cuda_available": False},
        }
        # NOTE: `crn_pair_device_consistent` reads `_record_device`, which parses
        # `env.json -> nvidia_smi.gpus[0]` — NOT `metrics.device`. Planting the device violation on
        # `metrics.device` (my first attempt) exercises only the INFORMATIONAL census and leaves the
        # gate untouched. Read the predicate before planting the violation.
        if gpu is not None:
            env_obj["nvidia_smi"] = {"gpus": [f"550.127.05, {gpu}"]}
    env_sha = sha256_obj(env_obj) if env_obj is not None else f"sha-{arm}-{seed}"
    d = root / "test" / arm / f"{arm}-s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "record.json").write_text(json.dumps({
        # the full src/io/results.REQUIRED_FIELDS schema — the loader fails LOUD on any omission,
        # which is correct behaviour and cost me one iteration to respect rather than work around
        "run_id": f"{arm}-s{seed}", "arm": arm, "seed": seed, "fold": 0,
        "candidate_id": f"{arm}-winner", "generation": None,
        "reward_source_hash": "hash-one", "feedback_block": "", "wall_clock": 1.0,
        "env_fingerprint": {"env_json_sha256": env_sha, "label": f"x|dev={device}"},
        "metrics": {"device": device, "popart_scale": {"rms": 1.0},
                    "train_safe_default_count": 0, "test_sharpe": 0.1},
    }), encoding="utf-8")
    if env_obj is not None:
        (d / "env.json").write_text(json.dumps(env_obj), encoding="utf-8")


def _search(root: Path, arm: str) -> None:
    """A complete search census so `all_units_complete` is not what fails the gate."""
    d = root / "search" / arm / f"{arm}-g0-c0"
    d.mkdir(parents=True, exist_ok=True)
    (d / "record.json").write_text(json.dumps({
        "run_id": f"{arm}-g0-c0", "arm": arm, "candidate_id": f"{arm}-g0-c0",
        "generation": 0, "seed": 0, "fold": 0, "reward_source_hash": "hash-one",
        "feedback_block": "", "wall_clock": 1.0,
        "env_fingerprint": {"env_json_sha256": "sha-s", "label": "x|dev=cpu"},
        "metrics": {"val_fitness": 0.1},
    }), encoding="utf-8")


def _report(tmp_path: Path, arms: list[str], seeds: list[int]) -> dict:
    for a in arms:
        _search(tmp_path, a)
    rep, _, _ = write_integrity_report(
        _Run(tmp_path), arms=arms, h2_arms=arms, baseline_names=[],
        core_seeds=seeds, opts_for=lambda _a: {"candidates": 1, "search_seeds_per_candidate": 1},
        out_dir=tmp_path / "out",
    )
    return rep


def test_substrate_mix_at_a_shared_seed_fails_the_gate(tmp_path: Path) -> None:
    """THE D16 CASE. Same seed, same device label, DIFFERENT CPU model -> the gate must stop.

    Pre-fix this returns health_ok=True: nothing in the verdict reads env.json's cpu block.
    """
    _write(tmp_path, "armA", 0, cpu=XEON_6240)
    _write(tmp_path, "armB", 0, cpu=XEON_6140)      # <- the mix, invisible to the device label
    rep = _report(tmp_path, ["armA", "armB"], [0])

    assert rep["verdict"]["crn_pair_device_consistent"] is True, (
        "the device labels are identical by construction — if this fails the fixture is wrong, "
        "not the gate")
    assert rep["verdict"]["crn_pair_substrate_consistent"] is False
    assert "0" in rep["verdict"]["crn_substrate_violations"]
    assert rep["verdict"]["health_ok"] is False, (
        "D16: a CPU-model mix at a shared seed confounds the paired contrast and MUST stop the gate")


def test_homogeneous_substrate_passes(tmp_path: Path) -> None:
    """The control: identical hardware everywhere must still pass, or the fix is just a brake."""
    _write(tmp_path, "armA", 0, cpu=XEON_6240)
    _write(tmp_path, "armB", 0, cpu=XEON_6240)
    rep = _report(tmp_path, ["armA", "armB"], [0])
    assert rep["verdict"]["crn_pair_substrate_consistent"] is True
    assert rep["verdict"]["health_ok"] is True


def test_missing_env_json_is_a_wildcard_not_a_failure(tmp_path: Path) -> None:
    """A capture gap must not stop a line — same rule the device check already applies."""
    _write(tmp_path, "armA", 0, cpu=XEON_6240)
    _write(tmp_path, "armB", 0, cpu=None)           # no env.json sibling at all
    rep = _report(tmp_path, ["armA", "armB"], [0])
    assert rep["verdict"]["crn_pair_substrate_consistent"] is True
    assert rep["verdict"]["health_ok"] is True


def test_device_mix_still_fails_no_regression(tmp_path: Path) -> None:
    """The pre-existing DEVICE invariant must be untouched by D16 (no regression).

    Planted on the field the predicate actually reads — ``env.json -> nvidia_smi.gpus[0]``, via
    ``_record_device`` — not on ``metrics.device``, which feeds only the informational census.
    """
    _write(tmp_path, "armA", 0, cpu=XEON_6240, gpu="Tesla V100-PCIE-32GB")
    _write(tmp_path, "armB", 0, cpu=XEON_6240, gpu="NVIDIA A100-SXM4-80GB")
    rep = _report(tmp_path, ["armA", "armB"], [0])
    assert rep["verdict"]["crn_pair_device_consistent"] is False
    assert rep["verdict"]["crn_pair_substrate_consistent"] is True, (
        "the CPU substrate is identical here — only the GPU differs, so D16's check must stay quiet "
        "and the DEVICE check must be the one that fires")
    assert rep["verdict"]["health_ok"] is False
