"""M2 survey runner (protocol v1) — item determinism, answer-key correctness, archival resume,
and the hard spend cap. Hermetic: stub transport, synthetic archive (no key, no GPU)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m2_survey as m2  # noqa: E402


def test_items_deterministic_and_protocol_shaped(tmp_path):
    """Same seed -> byte-identical item set (cross-model pairing rests on this); families and
    counts match protocol §3 (~40 P-A, 10 P-B, 8 P-C + 2 placebo, +20% self-consistency reps)."""
    a = m2.build_items(tmp_path, seed=7)
    b = m2.build_items(tmp_path, seed=7)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    base = [it for it in a if it.get("rep") != 1]
    reps = [it for it in a if it.get("rep") == 1]
    fams = {}
    for it in base:
        fams[it["family"]] = fams.get(it["family"], 0) + 1
    assert fams["P-A"] == 40 and fams["P-B"] == 10
    assert fams["P-C"] == 8 and fams["P-C-placebo"] == 2
    assert len(reps) == (fams["P-A"] + fams["P-B"]) // 5  # the registered 20% self-consistency
    # a DIFFERENT seed changes the items (the seed is registered, not decorative)
    c = m2.build_items(tmp_path, seed=8)
    assert json.dumps(a, sort_keys=True) != json.dumps(c, sort_keys=True)


def test_pa_answer_key_is_the_more_negative_cvar(tmp_path):
    """The P-A key must point at the WORSE (more negative) CVaR — a key inversion would silently
    invert the whole psychophysics curve."""
    items = [it for it in m2.build_items(tmp_path, seed=7) if it["family"] == "P-A"]
    import re

    checked = 0
    for it in items:
        if it["format"] == "basis_points":
            continue  # bp rendering is exercised by the generator; key checked on the other formats
        nums = []
        for line in it["prompt"].splitlines():
            m = re.match(r"Candidate [AB]: (-?\d+\.\d+)", line)
            if m:
                nums.append(float(m.group(1)))  # the candidate VALUE, never its CI annotation
        if len(nums) == 2 and abs(nums[0] - nums[1]) > 1e-9:
            worse_letter = "A" if nums[0] < nums[1] else "B"
            assert it["answer"] == worse_letter, it["id"]
            checked += 1
    assert checked >= 20  # the key check must actually bite


def test_run_model_archives_resumes_and_caps(tmp_path):
    items = m2.build_items(tmp_path, seed=7)
    out = tmp_path / "stub"
    budget = [1000]
    t1 = m2.run_model("stub", m2._stub_transport, items, out, budget=budget)
    assert t1["answered"] == len(items) and t1["errors"] == 0
    rows = [json.loads(x) for x in (out / "responses.jsonl").read_text().splitlines()]
    assert len(rows) == len(items)
    scored = [r for r in rows if "correct" in r]
    assert scored, "auto-scoring produced no verdicts"
    # resume: nothing re-asked, nothing double-spent
    budget2 = [1000]
    t2 = m2.run_model("stub", m2._stub_transport, items, out, budget=budget2)
    assert t2["resumed_skips"] == len(items) and t2["answered"] == 0
    assert budget2[0] == 1000  # zero calls consumed on a full resume
    # the HARD cap stops a fresh model early and is resumable
    out2 = tmp_path / "capped"
    t3 = m2.run_model("capped", m2._stub_transport, items, out2, budget=[5])
    assert t3["answered"] == 5


def test_transport_errors_are_archived_not_fatal(tmp_path):
    def _dying(_s, _u):
        raise RuntimeError("provider 500")

    items = m2.build_items(tmp_path, seed=7)[:3]
    t = m2.run_model("dying", _dying, items, tmp_path / "dying", budget=[10])
    assert t["errors"] == 3 and t["answered"] == 0
    rows = [json.loads(x) for x in (tmp_path / "dying" / "responses.jsonl").read_text().splitlines()]
    assert all("provider 500" in r["error"] for r in rows)
