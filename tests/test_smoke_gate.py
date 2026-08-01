"""Behaviour guard for the Phase-0 GATE roster (``scripts/smoke_test.py``).

A gate must never report a PASS having verified nothing. Two defects, both REPRODUCED before being
fixed (deep review 2026-07-26), motivate this file:

* an EMPTY roster made the GREEN length-equalities vacuously true (``0 == 0``), so ``--algos ""``
  printed ``STATUS: GREEN`` and exited 0 without training a single agent; and
* ``_train_one`` dispatched with a bare ``else``, so ANY name other than ``"sac"`` silently built
  TQC -- ``--algos foo`` reported ``[FOO] OK`` GREEN while the HEADLINE SAC agent was never
  constructed (and ``--algos SAC`` in capitals silently meant TQC for the same reason).

The gate is the thing that decides whether the campaign may proceed, so "it passed" has to mean the
agents it names were actually built and trained.
"""
from __future__ import annotations

import sys

import pytest

from src.data.panel import Panel


@pytest.mark.parametrize("bad", ["", "   ", ",", ",,", "foo", "sac,foo", "tqc,bogus"])
def test_gate_rejects_a_roster_it_cannot_honour(bad: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty or unrecognised ``--algos`` roster exits 2 (argparse usage error), never GREEN/0.

    Validation deliberately runs BEFORE the torch import and the panel load, so this stays cheap --
    and that ORDERING is itself part of the contract being locked down here: a usage error must not
    require loading the gold panel to be discovered.
    """
    import scripts.smoke_test as st

    monkeypatch.setattr(sys, "argv", ["smoke_test.py", "--synthetic", "--algos", bad])
    with pytest.raises(SystemExit) as exc:
        st.main()
    assert exc.value.code == 2, f"roster {bad!r} must be REJECTED, not silently run"


def test_train_one_refuses_an_unknown_algo_rather_than_mislabelling_tqc(
    synthetic_panel: Panel,
) -> None:
    """An unrecognised algo must FAIL the leg, not build TQC and report it under the wrong name."""
    import scripts.smoke_test as st
    from src.utils.config import load_config

    out = st._train_one("bogus", synthetic_panel, load_config("environment"), steps=1, device="cpu")

    assert out["ok"] is False, "an unknown algo must not report a passing leg"
    assert "bogus" in str(out["error"]), f"the error must name the offending algo; got {out['error']!r}"


def test_planning_number_m_excludes_sac_warmup(synthetic_panel: Panel) -> None:
    """``m`` (min/50k) must be the STEADY-STATE cost, not the whole-run average.

    ``m`` is the Phase-0 planning number the operator records in DECISION_LOG. SAC's warmup is a
    random-action rollout with NO gradient update — measured ~5407 steps/s vs ~30 steps/s training —
    and ``learning_starts = min(1000, steps//3)`` makes a THIRD of the default 3000-step run warmup.
    Timing the whole run therefore reported m = 16.12 min/50k where the true steady state was 24.54:
    a 34% UNDERSTATEMENT of the number the campaign is sized from (deep review 2026-07-26).
    """
    import scripts.smoke_test as st
    from src.utils.config import load_config

    out = st._train_one("sac", synthetic_panel, load_config("environment"), steps=40, device="cpu")
    assert out["ok"] is True, out.get("error")

    steady, raw = out["steady_steps_per_sec"], out["steps_per_sec"]
    assert steady is not None and out["warmup_steps"], "the warmup boundary must be observed"

    # m MUST be derived from the steady window, never from the whole-run rate. The tolerance absorbs
    # the 1-dp rounding of the REPORTED rate (m itself is computed from the unrounded value).
    assert out["minutes_per_50k"] == pytest.approx((50000 / steady) / 60, rel=0.01)
    assert out["minutes_per_50k"] != pytest.approx((50000 / raw) / 60, rel=0.01), (
        "m still tracks the WHOLE-RUN rate — the warmup/setup window is not being excluded"
    )
    # NOTE deliberately NOT asserted: `steady <= raw`. That holds at realistic step counts, where
    # cheap warmup inflates the whole-run rate (verified live at --steps 600: raw 51.1 vs steady
    # 34.1), but INVERTS at tiny counts like this one, where the fixed cost inside `learn()` — the
    # ~0.76 GB replay allocation and model setup — dominates instead (observed raw 19.0 vs steady
    # 57.1). Both regimes are reasons the steady window is the right basis: it excludes warmup AND
    # one-off setup, and neither of those scales to a 50k-step training.


def test_every_supported_algo_is_actually_dispatchable() -> None:
    """The advertised roster and the dispatch must not drift apart.

    If a name is added to ``_SUPPORTED_ALGOS`` without a matching dispatch branch, it would pass
    validation and then fail every leg -- the mirror image of the bug fixed above.
    """
    import inspect

    import scripts.smoke_test as st

    source = inspect.getsource(st._train_one)
    for algo in st._SUPPORTED_ALGOS:
        assert f'"{algo}"' in source, f"{algo!r} is advertised but has no dispatch branch"
