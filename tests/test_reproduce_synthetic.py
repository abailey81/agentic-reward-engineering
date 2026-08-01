"""Guards the golden SYNTHETIC reproduction (`scripts/reproduce_synthetic.py`) — reproducibility that
ACTUALLY holds. The FAST tests pin the bit-exact anchors (the synthetic panel; the committed golden's
shape) with NO training; the SLOW test runs the whole keyless pipeline (synthetic panel -> stub author ->
gate -> SAC -> measure -> held-out fitness -> select) and asserts it reproduces the committed golden.
"""
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import pytest

from scripts.reproduce_synthetic import (
    N_ASSETS, N_DAYS, PANEL_SEED, GOLDEN, run_reproduction, _compare,
)


def _golden() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def test_golden_exists_and_well_formed():
    g = _golden()
    assert g["schema"] == 1
    assert g["n_records"] == len(g["records"]) == g["candidates"]
    shas = [r["reward_sha"] for r in g["records"]]
    assert len(set(shas)) == len(shas), "the stub must author DISTINCT rewards per candidate"
    winner = max(g["records"], key=lambda r: r["val_fitness"])
    assert g["winner_candidate_id"] == winner["candidate_id"], "winner = argmax(val_fitness)"


def test_synthetic_panel_is_bit_stable():
    # The synthetic panel is a BIT-EXACT reproducibility anchor (no training involved): a change to
    # make_synthetic_panel or numpy's Generator that moved the panel would silently break every
    # downstream reproduction, so pin it here (fast, deterministic).
    from src.data.synthetic import make_synthetic_panel

    panel = make_synthetic_panel(n_assets=N_ASSETS, n_days=N_DAYS, seed=PANEL_SEED)
    sha16 = hashlib.sha256(panel.returns.tobytes()).hexdigest()[:16]
    assert sha16 == _golden()["panel"]["sha16"]


@pytest.mark.slow
def test_reproduce_synthetic_matches_golden():
    # Full end-to-end (~2 min on CPU): run the REAL keyless pipeline and assert it reproduces the golden.
    # Excluded from the fast suite (marked slow); the CI `reproduce` job runs it. Verified determinism:
    # two independent runs both matched the golden (panel + reward hashes exact, fitness within rel_tol).
    tmp = Path(tempfile.mkdtemp(prefix="test_repro_"))
    try:
        summary = run_reproduction(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    diffs = _compare(summary, _golden())
    assert diffs == [], "reproduction drifted from the golden:\n" + "\n".join(diffs)
