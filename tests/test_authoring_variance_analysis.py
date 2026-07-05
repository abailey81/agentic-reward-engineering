"""The authoring-variance estimator's own selftest, wired into the suite.

WHY THIS TEST EXISTS -- a measured gap, 2026-08-24. `scripts/analyse_authoring_variance.py` carries
a thorough `--selftest` (it recovers known variance components from synthetic data and holds seven
falsifiers), and NOTHING RAN IT. A grep of `tests/` for the module or any of its functions returned
no files, so every regression guard in that script fired only when a human happened to type the flag.

That mattered on the day it was found. Three real defects were live in the estimator at once, and
each is now covered by a falsifier inside that selftest:

* the seed cut kept each candidate's OWN lowest seeds rather than the seeds COMMON to a unit, so two
  candidates were scored on different common-random-number draws and the difference leaked into
  sigma2_candidate -- the quantity the whole script exists to estimate;
* the arm contrast pooled every score of each arm regardless of seed, so it compared two arms over
  whichever seeds each had happened to reach;
* the contrast's interval resampled CONTRASTS, treating a model's two authoring chains as two
  independent observations when they share the model, which made it about 30% too narrow.

The test drives `selftest()` itself rather than re-asserting any of its checks here. Re-implementing
them would double the maintenance and, worse, would pass against a `selftest()` that had been
gutted -- which is the failure mode `test_line_balance_held_out.py` records from its own history.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyse_authoring_variance.py"


def _load():
    """Import the script by path; it lives in `scripts/`, which is not an importable package."""
    if not _SCRIPT.is_file():
        pytest.skip(f"{_SCRIPT} is absent")
    # The module imports `run_authoring_variance` as a sibling, so `scripts/` must be importable.
    scripts_dir = str(_SCRIPT.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("analyse_authoring_variance", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_selftest_passes() -> None:
    """Every check in the estimator's own battery, including its falsifiers."""
    assert _load().selftest() == 0


def test_the_battery_is_not_empty() -> None:
    """A selftest that asserts nothing would pass this file forever.

    `selftest()` returns 0 both when every check passes and when there are no checks at all, so the
    return code alone cannot tell the two apart. Count what it printed instead.
    """
    mod = _load()
    import io as _io
    from contextlib import redirect_stdout

    buf = _io.StringIO()
    with redirect_stdout(buf):
        mod.selftest()
    out = buf.getvalue()
    n_pass = out.count("PASS")
    assert n_pass >= 15, f"only {n_pass} checks ran; the battery has been gutted:\n{out}"
    assert "FAIL" not in out


def test_seed_cut_is_common_not_per_candidate() -> None:
    """The 2026-08-24 defect, asserted here as well because it is the load-bearing one.

    Both candidates hold five seeds, so a rule that counts seeds calls this unit balanced. Their
    seeds are not the SAME five, and seeds are common random numbers.
    """
    mod = _load()
    rows = ([mod.Record("L", 1, "scalar", 0, s, 1.0) for s in (0, 1, 2, 3, 4, 7, 8, 9)]
            + [mod.Record("L", 1, "scalar", 1, s, 2.0) for s in (0, 1, 7, 8, 9)])
    kept, _ = mod.truncate_to_seed_floor(rows, 5)
    assert {r.seed for r in kept} == {0, 1, 7, 8, 9}
    # Every surviving candidate must stand on the identical seed set, not merely the same count.
    per_cand = {}
    for r in kept:
        per_cand.setdefault(r.candidate, set()).add(r.seed)
    assert len(set(map(frozenset, per_cand.values()))) == 1
