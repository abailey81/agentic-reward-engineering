"""The registered analysis-output key set must be MACHINE-DEFINED, not remembered.

Why this exists
---------------
Four registered outputs — ``benchmark_floor`` (the DeMiguel 1/N floor, already wired into the PDF),
``attribution``, ``h2_rf_robustness``, ``regime_stratified`` — would have been silently absent from
the final analysis because one file, ``campaign_summary.json``, does not exist for RUN 4. The only
signal was a print that named ONE of the four, and the analysis exited 0 regardless.

An earlier scope note pinned the enumeration to "the 35 ``out[...]`` keys of
scripts/analyze_campaign.py". Measured: 32 subscript assignments and 7 keys in the annotated
``out: dict[str, Any] = {...}`` initialiser — 39 in the union, and 35 matches none of the three
numbers. A remembered count is exactly how a quantity escapes an enumeration that is supposed to be
exhaustive, so the registry is pinned to the source here instead.

The test derives the key set from the AST by BOTH routes. Counting only subscript assignments is
the wrong-denominator error in its purest form: it would silently shrink the registry by seven.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import analyze_campaign as ac  # noqa: E402

SRC = REPO / "scripts" / "analyze_campaign.py"


def _keys_from_source() -> set[str]:
    """Every constant key that can enter ``analyze()``'s ``out`` dict, by either route."""
    tree = ast.parse(SRC.read_text(encoding="utf-8"), filename=str(SRC))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "analyze")
    keys: set[str] = set()
    for node in ast.walk(fn):
        # route 1 — the annotated initialiser `out: dict[str, Any] = {...}`
        if (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                and node.target.id == "out" and isinstance(node.value, ast.Dict)):
            keys |= {k.value for k in node.value.keys
                     if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        # route 2 — `out["key"] = ...`
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == "out" and isinstance(tgt.slice, ast.Constant)
                        and isinstance(tgt.slice.value, str)):
                    keys.add(tgt.slice.value)
    return keys


def test_registry_matches_the_source_exactly() -> None:
    derived = _keys_from_source()
    registered = set(ac.REGISTERED_OUTPUT_KEYS)
    assert derived == registered, (
        f"the key registry has drifted from analyze():\n"
        f"  in the source but NOT registered: {sorted(derived - registered)}\n"
        f"  registered but NOT in the source: {sorted(registered - derived)}\n"
        f"Add the new result to REGISTERED_OUTPUT_KEYS (and to CONDITIONAL_OUTPUT_KEYS if it is "
        f"legitimately absent without a precondition), so it cannot escape the enumeration."
    )


def test_no_dynamic_keys_can_escape_the_registry() -> None:
    """A computed key would be invisible to the AST walk, so the registry could not be exhaustive."""
    tree = ast.parse(SRC.read_text(encoding="utf-8"), filename=str(SRC))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "analyze")
    dynamic = [
        ast.dump(tgt.slice)[:120]
        for node in ast.walk(fn) if isinstance(node, ast.Assign)
        for tgt in node.targets
        if (isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name)
            and tgt.value.id == "out"
            and not (isinstance(tgt.slice, ast.Constant) and isinstance(tgt.slice.value, str)))
    ]
    mutators = [
        ast.unparse(node)[:120]
        for node in ast.walk(fn)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"update", "setdefault"}
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "out")
    ]
    assert not dynamic, f"computed out[...] key(s) escape the registry: {dynamic}"
    assert not mutators, f"out.update()/setdefault() bypass(es) the registry: {mutators}"


def test_the_count_claimed_in_claude_md_is_the_measured_one() -> None:
    """39, by measurement. Pinned so the number in the contract can be corrected against evidence."""
    assert len(ac.REGISTERED_OUTPUT_KEYS) == 39


def test_conditional_keys_are_all_registered() -> None:
    """A precondition attached to a key that no longer exists would silently excuse nothing."""
    unknown = set(ac.CONDITIONAL_OUTPUT_KEYS) - set(ac.REGISTERED_OUTPUT_KEYS)
    assert not unknown, f"CONDITIONAL_OUTPUT_KEYS names non-existent output(s): {sorted(unknown)}"


def test_missing_keys_split_into_explained_and_unexplained() -> None:
    # A complete result: nothing absent either way.
    full = dict.fromkeys(ac.REGISTERED_OUTPUT_KEYS, {})
    assert ac.missing_output_keys(full) == ({}, [])

    # The live RUN 4 shape: the summary is absent, so the four panel-dependants and `variance` go.
    live = {k: {} for k in ac.REGISTERED_OUTPUT_KEYS
            if k not in ac.CONDITIONAL_OUTPUT_KEYS}
    explained, unexplained = ac.missing_output_keys(live)
    assert set(explained) == set(ac.CONDITIONAL_OUTPUT_KEYS)
    assert unexplained == [], "a conditional absence must never be reported as a defect"

    # A genuine defect: a key with no stated precondition simply not produced.
    broken = {k: {} for k in ac.REGISTERED_OUTPUT_KEYS if k != "mediation"}
    explained, unexplained = ac.missing_output_keys(broken)
    assert unexplained == ["mediation"] and explained == {}
