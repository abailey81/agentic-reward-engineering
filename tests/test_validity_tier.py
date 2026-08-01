"""R105: bind the graphical-multiplicity VALIDITY TIER (A3) so its structure is VERIFIED, not asserted.

Checks the registered `inference.validity_tier` is a well-formed Bretz-Maurer-Brannath-Posch (2009)
alpha-graph: all alpha starts on the two H2 co-primaries (the tier costs the headline zero power); every
node's out-edges sum to <= 1; H1 (the snooped beat-human comparison) is EXCLUDED; every tier node is
reachable from the headline; the block is ratification-pending (NOT a silent override of the R31 stance);
and the 2026-07-26 content-over-format upgrade (N5 = placebo_shuffled) is present.
"""
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _tier() -> dict:
    prereg = yaml.safe_load((ROOT / "config" / "preregistration.yaml").read_text(encoding="utf-8"))
    return prereg["inference"]["validity_tier"]


def test_tier_ratification_is_RECORDED_not_silent():
    """R108 (2026-07-26): the tier was signed off in supervision — exactly the precondition
    the `status` field named. The guard is NOT removed, it is RE-POINTED: a ratified tier must
    still carry WHO ratified it and WHEN, so activation can never be a silent edit. It still
    declares what it supersedes (R31), and it is still not frozen."""
    t = _tier()
    assert t["status"] == "ratified"
    assert set(t["ratified_by"]) == {"tamer", "okhrati"}, "activation requires a NAMED sign-off"
    assert t["ratified_utc"], "a ratification without a date is not auditable"
    assert t["supersedes_on_ratification"] == "cross_hypothesis_multiplicity"   # R31 now superseded
    assert t["ratification_pending"] == [], "nothing may remain pending once status is ratified"
    assert len(t["ratification_completed"]) >= 7, "the completed set must record what was signed"


def test_all_initial_alpha_starts_on_the_headline():
    # activate-on-upstream-success => the H3/H4/N5 tier costs the H2 headline ZERO power
    w = _tier()["initial_weights"]
    assert w["N1_h2_tail"] == 0.5 and w["N2_h2_ra"] == 0.5
    assert all(v == 0.0 for n, v in w.items() if n not in ("N1_h2_tail", "N2_h2_ra"))
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_every_node_out_edges_sum_to_at_most_one():
    for src, outs in _tier()["edges"].items():
        assert sum(outs.values()) <= 1.0 + 1e-9, (src, outs)


def test_snooped_h1_excluded_but_h1_is_confirmatory_node_n6_via_iut():
    # 2026-07-26 (nothing-frozen + "make it smart + sound"): H1 PROMOTED to a confirmatory node via a
    # SNOOP-FREE intersection-union test over the full canon (beat EVERY member one-sided at alpha == beat the
    # best human, since best = max), SUPERSEDING the val-select framing (dead code: the campaign archives
    # val_fitness=NaN, so beat_human_baseline already fell back to the test-snoop) — the IUT selects no
    # comparator, so there is nothing to snoop and no fragile baseline val-roll is needed.
    t = _tier()
    assert "H1_snooped_test_max" in t["excludes"]   # the max-over-the-sealed-TEST snoop has no valid p-value
    assert "N6_h1" in t["nodes"]                     # H1 IS a confirmatory node (snoop-free IUT)
    n6 = t["nodes"]["N6_h1"]
    assert n6["test"] == "llm_beats_best_human_reward"
    assert n6["method"] == "intersection_union_over_canon"       # berger1982iut: beat-all <=> beat-best, no selection
    assert n6["comparator"] == "full_11name_hand_reward_canon"   # the FULL canon (not a 4-subset)
    assert "selection" not in n6                                 # NO comparator selected -> nothing to snoop (dissolved)
    assert n6["endpoint"] == "sharpe_annualized", (
        "N6 must register the endpoint the code actually computes. It previously registered "
        "deflated_sharpe while scripts/analyze_campaign.py built the IUT legs from annualised "
        "Sharpe; a DSR endpoint would score the winner against an E[max SR] benchmark of ~0.83 "
        "annualised (n_trials=30) and each hand reward against 0.0 (n_trials=1), i.e. different "
        "nulls per arm, and the winner would lose every leg even at an equal true Sharpe "
        "(deep review 2026-07-26, loop 5)."
    )


def test_every_tier_node_is_reachable_from_the_headline():
    t = _tier()
    edges = t["edges"]
    reached = {"N1_h2_tail", "N2_h2_ra"}   # the headline nodes hold the initial alpha
    changed = True
    while changed:
        changed = False
        for src in list(reached):
            for dst in edges.get(src, {}):
                if dst not in reached:
                    reached.add(dst)
                    changed = True
    assert set(t["nodes"]) <= reached, f"unreachable nodes: {set(t['nodes']) - reached}"


def test_n5_is_the_content_over_format_upgrade():
    n5 = _tier()["nodes"]["N5_structure"]
    assert n5["arm_b"] == "placebo_shuffled"
    assert n5["direction"] == "one_sided_content_over_format"


# --- #99: the ANALYSIS must not contradict the ratified config about its own node status ---------

def test_analysis_does_not_report_N6_as_ratification_pending_when_the_config_says_ratified():
    """A stale status string in the analysis lands in the ARCHIVED result JSON, not just a comment.

    `scripts/analyze_campaign.py` emits an `iut_block` for node N6 (H1 dominates the hand-reward
    canon) carrying a `status` field. That field read
    ``"registered_pending_supervisor_ratification"`` for a day after **R108 (2026-07-26)** recorded
    ``inference.validity_tier.status: ratified`` with ``ratification_pending: []`` and
    ``n6_h1_confirmatory_node`` in ``ratification_completed``, signed off in supervision.

    Nothing branches on the field, so no computed number was wrong — but a replay-only campaign
    would have archived a result that describes its own CONFIRMATORY node as un-ratified, and the
    write-up reads that archive. Same stale-fact class as #84, which was reconciled across `paper/`
    and `PREREGISTRATION.md` but not across `scripts/`.

    Guards the AGREEMENT, not a literal string, so ratification state may change freely provided the
    analysis follows it.
    """
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "config" / "preregistration.yaml").read_text(encoding="utf-8"))
    tier_status = str(cfg["inference"]["validity_tier"].get("status", "")).strip().lower()

    src = (root / "scripts" / "analyze_campaign.py").read_text(encoding="utf-8")
    # Only STATUS ASSIGNMENTS count; prose that quotes the old value as history is fine.
    emitted = [
        ln.strip() for ln in src.splitlines()
        if '"status"' in ln and ":" in ln and not ln.strip().startswith("#")
    ]
    stale = [ln for ln in emitted if "pending" in ln.lower() and "ratifi" in ln.lower()]

    if tier_status == "ratified":
        assert not stale, (
            "config/preregistration.yaml says inference.validity_tier.status: ratified, but "
            f"analyze_campaign.py still EMITS a pending-ratification status: {stale} — the archived "
            "result would describe a ratified confirmatory node as un-ratified"
        )
