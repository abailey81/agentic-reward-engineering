"""END-TO-END §15 pack-path certification — the one path no unit test had ever EXECUTED.

Two REAL specs (built by the production ``parallel._spec``) go through ``run_task(pack=2)``:
a real DevicePool spawn (2 'cuda' tokens on the one physical GPU — exactly the Myriad packing
shape), real SAC construction on the synthetic panel (keyless), and real atomic archive
commits that the poll layer's completion truth + compacted-resume diff then read back.

Runtime ~2–4 min (process spawn + torch import dominates). 300 steps sits below the trainer's
1000-step ``learning_starts`` floor, so this certifies the PLUMBING — pack concurrency,
device-token routing, archiving, run_id idempotency — not learning (that is the campaign's
job). Deselect with ``-m "not slow"``.
"""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.integration]


def test_pack_path_end_to_end_two_real_specs(tmp_path):
    from src.cluster.poll import completed_run_ids, pending_specs
    from src.cluster.run_one import run_task
    from src.orchestration.parallel import _spec

    opts = {
        "train_steps": 300,
        "batch_size": 64,
        "normalize_obs": True,
        "n_trials": 1,
        "synthetic": True,
        "data": {},
        "cvar_alpha": 0.05,
        "window": 20,
        "seed": 123,
    }
    specs = []
    for i in range(2):
        s = _spec("scalar", "baseline", "raw_return", f"packtest-c{i}", opts)
        s["archive_root"] = str(tmp_path)
        specs.append(s)

    rows = run_task(specs, pack=2)

    assert len(rows) == 2 and all(r.get("ok") for r in rows), rows
    # the poll layer's completion truth reads the packed job's commits
    assert completed_run_ids(tmp_path) == {"packtest-c0", "packtest-c1"}
    # run_id idempotency: the compacted-resume diff over this archive is now EMPTY
    assert pending_specs(specs, tmp_path) == []
