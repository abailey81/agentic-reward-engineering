"""P23 (2026-07-04): lock the MANIPULATED-VARIABLE CVaR levels against silent drift.

The fed tail vector's CVaR levels ``{0.01, 0.05, 0.10, 0.25}`` are the *manipulated variable* of the whole
study, and they live in THREE places that were never reconciled by a test:
  (a) the code — ``ReturnDistribution.tail_stats`` emits keys ``cvar_01/05/10/25`` (levels hardcoded);
  (b) ``config/inference.yaml: distributional_feedback.alpha_grid``;
  (c) ``config/preregistration.yaml: tail_diagnostic_set.cvar_levels``.
A divergence of any one from the others would change (or misdescribe) the fed feedback with nothing to catch
it. This test asserts all three agree as a set. Read-only; adds no scope.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from src.feedback.measurement import ReturnDistribution

_ROOT = Path(__file__).resolve().parents[1]


def _levels_from_cvar_keys(keys) -> set[float]:  # type: ignore[no-untyped-def]
    """Parse {0.01, 0.05, ...} from the ``cvar_NN`` field names (NN = tail percentage)."""
    return {round(int(k.split("_")[1]) / 100.0, 6) for k in keys if k.startswith("cvar_")}


def _yaml(name: str) -> dict:
    return yaml.safe_load((_ROOT / "config" / f"{name}.yaml").read_text(encoding="utf-8"))


def test_fed_cvar_levels_reconcile_code_vs_both_configs() -> None:
    rng = np.random.default_rng(0)
    stats_out = ReturnDistribution().fit(rng.normal(0.0, 0.01, size=4000)).tail_stats()
    code_levels = _levels_from_cvar_keys(stats_out.keys())
    assert code_levels == {0.01, 0.05, 0.10, 0.25}, f"unexpected fed CVaR levels in code: {sorted(code_levels)}"

    alpha_grid = {round(float(x), 6) for x in _yaml("inference")["distributional_feedback"]["alpha_grid"]}
    cvar_levels = {round(float(x), 6) for x in _yaml("preregistration")["tail_diagnostic_set"]["cvar_levels"]}

    assert code_levels == alpha_grid == cvar_levels, (
        "fed CVaR levels drifted across surfaces: "
        f"code={sorted(code_levels)}, inference.alpha_grid={sorted(alpha_grid)}, "
        f"prereg.cvar_levels={sorted(cvar_levels)}"
    )
