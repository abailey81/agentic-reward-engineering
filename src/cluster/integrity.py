"""The EFFECT-BLIND integrity report for the C3 review gate (PLAN §13: "counts-only monitoring;
single-look discipline"; control cascade §13.3 "effect-blind by construction").

The review checkpoint between the all-arms n=30 design floor (C3) and the uniform-n sweep (C4)
must NOT look at results — conditioning continuation on observed effects is optional stopping and
would invalidate the pre-registered single-look inference. What it CAN (and should) verify, very
carefully, is EXECUTION: did every unit train, complete, and archive exactly as designed?

This module writes that report: completeness counts (search candidates vs the matched budget,
k-seed integrity, test seeds vs the core set), failure censuses (F5 ledgers), and homogeneity
censuses (device + env-fingerprint labels across the sealed leg). It NEVER reads ``val_fitness``,
``test_sharpe``, ``test_cvar05``, ``test_returns``, or ``tail_stats`` values — only presence.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

__all__ = ["write_integrity_report", "record_env_label", "env_label_census"]

_BLIND_HEADER = (
    "EFFECT-BLIND EXECUTION-INTEGRITY REPORT — contains NO performance statistics.\n"
    "Reviewing results before the sweep would be optional stopping (a forking path); this gate\n"
    "verifies EXECUTION only. The single confirmatory look stays at the pre-declared date.\n"
)


def _count_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip())


def _search_census(arm_root: Path, expected_candidates: int, k_seeds: int) -> dict[str, Any]:
    """Counts for one arm's SEARCH sub-root: records vs budget×k + the F5 failure ledger."""
    from src.io.results import load_all

    records = load_all(str(arm_root)) if arm_root.is_dir() else []
    n_records = len(records)
    n_failures = _count_lines(arm_root / "failures.jsonl")
    expected_records = expected_candidates * max(1, k_seeds)
    # Reflection-archival census (2026-07-12 upgrade; instrument (g) dies silently without it):
    # the funnel content analysis codes the designer's archived COMPLETIONS, so the gate counts
    # llm_calls rows and how many carry an EMPTY response — counts only, never the text itself
    # (effect-blind). A tail-fed arm with archived candidates but zero/empty completions is an
    # archival defect to fix before C4, not at the bank gate.
    llm_calls = 0
    empty_completions = 0
    calls_path = arm_root / "llm_calls.jsonl"
    if calls_path.is_file():
        for line in calls_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            llm_calls += 1
            try:
                if not str(json.loads(line).get("response", "")).strip():
                    empty_completions += 1
            except ValueError:
                empty_completions += 1  # a torn/unparseable row cannot be coded either
    # P17/A3-F8 (2026-07-13 audit): the budget check was `>=` — an OVERSHOOT (duplicate
    # candidates from a reused root or a doubled generation) passed as "matched budget", the
    # exact contamination class the matched-budget defense exists to catch. EXACT candidate-
    # level accounting instead: every expected candidate is either RESOLVED (>=1 record) or
    # ledgered-failed (an F5 row with NO record); a recovered resubmit (record + a stale F5
    # row) counts ONCE, and anything != expected — short OR overshoot — fails the gate.
    def _cand(cid: str) -> str:
        return re.sub(r"-s\d+$", "", cid) if k_seeds > 1 else cid

    resolved = {_cand(str(r.get("candidate_id") or r.get("run_id"))) for r in records}
    failed_cids: set[str] = set()
    fpath = arm_root / "failures.jsonl"
    if fpath.is_file():
        for line in fpath.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                cid = json.loads(line).get("candidate_id")
            except ValueError:
                continue
            if cid:
                failed_cids.add(str(cid))
    accounted = len(resolved) + len(failed_cids - resolved)
    return {
        "records": n_records,
        "ledgered_failures": n_failures,
        "expected_records": expected_records,
        "expected_candidates": expected_candidates,
        "accounted_candidates": accounted,
        "matched_budget_ok": accounted == expected_candidates,
        "llm_calls_archived": llm_calls,
        "empty_completions": empty_completions,
        "reflection_archive_ok": llm_calls == 0 or empty_completions == 0,
    }


def _record_device(arm_root: Path, run_id: str) -> str:
    """The GPU model that trained this record, from its replayable ``env.json`` (S6).

    Source: ``env.json → nvidia_smi.gpus[0]`` (e.g. ``'550.127.05, Tesla V100-PCIE-32GB'`` →
    ``'Tesla V100-PCIE-32GB'``), else ``'<absent>'`` — which the CRN-consistency check treats as a
    WILDCARD (laptop/legacy records without a GPU capture must not fail the gate)."""
    try:
        env = json.loads((arm_root / run_id / "env.json").read_text(encoding="utf-8"))
        gpus = (env.get("nvidia_smi") or {}).get("gpus") or []
        if gpus:
            return str(gpus[0]).split(",", 1)[-1].strip()
    except (OSError, ValueError):
        pass
    return "<absent>"


def _record_substrate(arm_root: Path, run_id: str) -> str:
    """The CPU-level SUBSTRATE that trained this record, from its replayable ``env.json`` (D16).

    The sibling of :func:`_record_device`, and the reason D16 existed: ``_record_device`` reads the
    GPU model and ``per_seed_device`` reads the cpu/cuda LABEL, so on a CPU-only lane every record
    looks identical to both — while the properties that actually fix floating-point reduction order
    (**CPU model**, OMP threads, torch threads) differ silently. RUN 4 proved it: four
    ``baseline_volatility_scaled_return`` records landed on an ``Intel Xeon Gold 6140`` while the
    other 26 ran on a ``6240``, and every device-keyed check passed.

    The signature is deliberately IDENTICAL in shape to :func:`substrate_field_census`'s, so the
    blocking gate and the advisory sentinel can never disagree about what counts as one substrate.

    ``'<absent>'`` when no ``env.json`` can be read — treated as a WILDCARD by the pair check, on the
    same reasoning as ``_record_device``: a capture gap must not fail the gate, and the *absence* of
    env.json is separately visible as ``"<no env.json>"`` in the leg-wide census.
    """
    try:
        env = json.loads((arm_root / run_id / "env.json").read_text(encoding="utf-8"))
        if not isinstance(env, dict):
            return "<absent>"
        cpu = env.get("cpu") if isinstance(env.get("cpu"), dict) else {}
        det = (env.get("determinism_env")
               if isinstance(env.get("determinism_env"), dict) else {})
        tc = env.get("torch_cuda") if isinstance(env.get("torch_cuda"), dict) else {}
        return (f"cpu={cpu.get('model_name')}"
                f" | omp={det.get('OMP_NUM_THREADS')}"
                f" | torch_threads={tc.get('num_threads')}"
                f" | cuda={tc.get('cuda_available')}")
    except (OSError, ValueError):
        pass
    return "<absent>"


def record_env_label(record: dict[str, Any]) -> str:
    """The record's environment-fingerprint LABEL, normalised — the substrate-homogeneity key.

    Extracted so the post-hoc S6 gate (:func:`_test_census`) and the LIVE campaign check
    (:func:`src.cluster.campaign_health.check_determinism_homogeneity`) key on the SAME string. If
    they normalised differently, one of them would lie about homogeneity — and the one that lies is
    the one that decides whether a paired contrast is valid.

    ``env_fingerprint`` is the driver's ``{label, env_json_sha256}`` dict; older/degraded records may
    carry a bare string. A non-str label is coerced deterministically rather than crashing the gate
    (defense-in-depth for the 2026-07-13 run_one nested-dict bug, fixed at source).
    """
    fp = record.get("env_fingerprint")
    label = fp.get("label", "<dict>") if isinstance(fp, dict) else str(fp)[:60]
    if not isinstance(label, str):
        label = json.dumps(label, sort_keys=True, default=str)[:60]
    return label


def substrate_field_census(arm_roots: list[Path] | list[str]) -> dict[str, int]:
    """Leg-wide ``{substrate signature: count}`` read from the per-run ``env.json`` siblings.

    WHY THIS EXISTS AS WELL AS :func:`env_label_census` (2026-07-28). The env LABEL carries only the
    DEVICE (``|dev=cpu``). The properties that actually fix float reduction order — **CPU model**,
    thread counts, BLAS settings — live in ``env.json`` and are invisible to the label. They are
    also invisible to ``env_json_sha256``, because that digest includes the run's ``seed``, so two
    records on identical hardware hash differently and two records on DIFFERENT hardware cannot be
    told apart by it either. The hash is provenance; it is not a substrate key.

    That gap is not hypothetical. On 2026-07-28 the SEARCH leg held 116 records on an
    ``Intel Xeon Gold 6240`` and **1 on a Gold 6140** — different microarchitectures, so
    ``-ac allow=d`` confines placement to the d-class WITHOUT guaranteeing one CPU model. On the
    search leg that is a tolerable nuisance (it changes only WHICH candidate is selected). On the
    SCORED leg it would silently confound every paired contrast, and nothing was watching for it.

    Missing/unreadable ``env.json`` is counted under ``"<no env.json>"`` rather than skipped: a
    record whose substrate cannot be established is a real gap and must stay visible.
    """
    counts: dict[str, int] = {}
    for root in arm_roots:
        p = Path(root)
        if not p.is_dir():
            continue
        for rec_path in p.rglob("record.json"):
            if any(part.startswith(".pull_tmp") for part in rec_path.parts):
                continue
            env_path = rec_path.parent / "env.json"
            try:
                env = json.loads(env_path.read_text(encoding="utf-8"))
                if not isinstance(env, dict):
                    raise ValueError("env.json is not an object")
            except (OSError, ValueError):
                counts["<no env.json>"] = counts.get("<no env.json>", 0) + 1
                continue
            cpu = env.get("cpu") if isinstance(env.get("cpu"), dict) else {}
            det = (env.get("determinism_env")
                   if isinstance(env.get("determinism_env"), dict) else {})
            tc = env.get("torch_cuda") if isinstance(env.get("torch_cuda"), dict) else {}
            sig = (f"cpu={cpu.get('model_name')}"
                   f" | omp={det.get('OMP_NUM_THREADS')}"
                   f" | torch_threads={tc.get('num_threads')}"
                   f" | cuda={tc.get('cuda_available')}")
            counts[sig] = counts.get(sig, 0) + 1
    return counts


def record_substrate_key(record: dict[str, Any]) -> str:
    """The SUBSTRATE portion of a record's env label — the leg-wide homogeneity key.

    :func:`record_env_label` returns the label whole, and the label carries IDENTITY as well as
    substrate::

        campaign:baseline_return_minus_cvar:test[3835,5406)|dev=cpu     (test leg)
        dev=cpu                                                          (search leg)

    Keying a LEG-WIDE census on that string makes every distinct ARM look like a distinct
    substrate. Found live 2026-07-28: the sentinel reported ``test leg is NOT substrate-homogeneous:
    8 distinct environments`` when all 107 scored records were ``dev=cpu`` on one CPU model — the
    "8 environments" were 8 arm NAMES. That is not merely noisy: the check is CRITICAL and its
    remedy line reads "quarantine the minority substrate and re-run it", so acting on it would have
    destroyed perfectly good scored records. It also fires by CONSTRUCTION on any multi-arm leg,
    i.e. always, which is the cry-wolf failure that trains an operator to ignore the real one.

    ``_test_census`` is unaffected because it censuses ONE arm at a time, where the identity prefix
    is constant — which is exactly why this survived until the leg-wide census met a second arm.

    A label with no ``dev=`` token keys as ``dev=<unrecorded>`` rather than being treated as
    equivalent to a stamped one: missing provenance is a real difference and must stay visible.
    A ``capture-failed`` label is preserved whole, since it is a capture fault, not a substrate.
    """
    label = record_env_label(record)
    if label.startswith("capture-failed"):
        return label
    tail = label.rsplit("|", 1)[-1].strip()
    return tail if tail.startswith("dev=") else "dev=<unrecorded>"


def env_label_census(arm_roots: list[Path] | list[str]) -> dict[str, int]:
    """Leg-wide ``{env label: count}`` across archive roots — the LIVE homogeneity census.

    ``_test_census`` censuses ONE unit at report time; this censuses the whole scored leg WHILE the
    campaign runs, which is the only point at which a device/thread mix can still be re-run.

    Deliberately reads ``record.json`` directly instead of going through ``load_all``: that reader is
    ALL-OR-NOTHING by design (it raises on the first schema-invalid record so the GATE fails loud),
    which is right for the gate and wrong here — a single half-written record mid-campaign would
    take the live monitor DARK exactly when something is going wrong. A record we cannot parse is
    skipped, never guessed at; the labels themselves still come from :func:`record_env_label`, so the
    live census and the gate can never disagree about what counts as the same substrate.
    """
    counts: dict[str, int] = {}
    for root in arm_roots:
        p = Path(root)
        if not p.is_dir():
            continue
        for rec_path in p.rglob("record.json"):
            if any(part.startswith(".pull_tmp") for part in rec_path.parts):
                continue                      # in-flight staging is not archived truth
            try:
                rec = json.loads(rec_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(rec, dict):
                continue
            lbl = record_substrate_key(rec)   # SUBSTRATE, not identity — see record_substrate_key
            counts[lbl] = counts.get(lbl, 0) + 1
    return counts


def _test_census(arm_root: Path, arm: str, seeds: list[int]) -> dict[str, Any]:
    """Presence census for one unit's sealed-leg records at the given seeds (+ homogeneity fields)."""
    from src.io.results import load_all

    records = load_all(str(arm_root)) if arm_root.is_dir() else []
    have = {str(r.get("run_id")) for r in records}
    missing = [s for s in seeds if f"{arm}-s{s}" not in have]
    devices: dict[str, int] = {}
    env_labels: dict[str, int] = {}
    per_seed_device: dict[str, str] = {}
    # D16 (2026-08-01): the SUBSTRATE per seed, not just the device LABEL. `per_seed_device` carries
    # only cpu/cuda; the properties that actually fix float reduction order — CPU MODEL, thread
    # counts, BLAS settings — live in env.json and are invisible to it. See substrate_field_census.
    per_seed_substrate: dict[str, str] = {}
    reward_hashes: dict[str, int] = {}
    popart_present = 0
    safe_default_total = 0
    for r in records:
        # P5 (2026-07-13 audit): a mid-unit WINNER SWAP (resume after config drift re-freezes a
        # different winner; remaining seeds train the new reward) was undetectable — census the
        # distinct reward_source_hash per unit; >1 hash = a mixed-winner unit = gate RED.
        rh = str(r.get("reward_source_hash", "<absent>"))
        reward_hashes[rh] = reward_hashes.get(rh, 0) + 1
        m = r.get("metrics", {}) or {}
        dev = str(m.get("device", "<absent>"))
        devices[dev] = devices.get(dev, 0) + 1
        label = record_env_label(r)   # shared with the live check so the two can never disagree
        env_labels[label] = env_labels.get(label, 0) + 1
        if r.get("seed") is not None:
            per_seed_device[str(r["seed"])] = _record_device(arm_root, str(r.get("run_id")))
            per_seed_substrate[str(r["seed"])] = _record_substrate(arm_root, str(r.get("run_id")))
        if m.get("popart_scale") is not None:
            popart_present += 1
        if m.get("train_safe_default_count") is not None:
            safe_default_total += int(m["train_safe_default_count"])
    return {
        "present": len(seeds) - len(missing),
        "expected": len(seeds),
        "missing_seeds": missing[:25],
        "device_census": devices,
        "device_homogeneous": len([d for d in devices if d != "<absent>"]) <= 1,
        "per_seed_device": per_seed_device,
        "per_seed_substrate": per_seed_substrate,
        "env_label_census": env_labels,
        "reward_hash_census": reward_hashes,
        "winner_hash_consistent": len(reward_hashes) <= 1,
        "popart_scale_present": popart_present,
        "train_safe_default_total": safe_default_total,
    }


def write_integrity_report(
    run: Any,
    *,
    arms: list[str],
    h2_arms: list[str],
    baseline_names: list[str],
    core_seeds: list[int],
    opts_for: Callable[[str], dict],
    winners: dict[str, dict] | None = None,
    out_dir: str | Path | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    """Write the effect-blind integrity report (JSON + MD) under ``out_dir`` (default: read_root).

    2026-07-28: the filenames are scoped by ``run.line_tag()``. ``read_root`` is shared by all
    twelve supervised lines, so the previous unqualified ``tier1_integrity.json`` was ONE file that
    every line overwrote — the report a reviewer read was whichever line wrote last, and the review
    gate's staleness check (approval mtime vs report mtime) raced against other lines' writes. The
    RUN 1 archive confirms it: a single report existed for all twelve lines.

    Returns ``(report, json_path, md_path)``. Every GATE check is a COUNT or a CENSUS — never a
    statistic of a performance metric — so the auto-proceed decision is effect-blind. When
    ``winners`` is given, an additional **SEALED-SAFE SELECTION** section carries each arm's chosen
    reward CODE (the authored source — the mechanism headline "does the fed tail change the code?"),
    which is sealed from the confirmatory test leg; it is DESCRIPTIVE and does NOT gate continuation.
    NO ``val_fitness``/``test_sharpe``/``test_cvar``/``tail_stats`` VALUE is ever written.
    """
    out_root = Path(out_dir) if out_dir is not None else Path(run.read_root)
    out_root.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {"blind": True, "core_n": len(core_seeds), "search": {}, "test": {}}
    for arm in arms:
        opts = opts_for(arm)
        k = max(1, int(opts.get("search_seeds_per_candidate", 1)))
        report["search"][arm] = _search_census(
            run.search_read() / arm, int(opts.get("candidates", 0)), k
        )
        report["test"][arm] = _test_census(run.test_read() / arm, arm, core_seeds)
    for nm in baseline_names:
        unit = f"baseline_{nm}"
        report["test"][unit] = _test_census(run.test_read() / unit, unit, core_seeds)

    all_complete = all(
        t["present"] == t["expected"] for t in report["test"].values()
    ) and all(s["matched_budget_ok"] for s in report["search"].values())
    all_homogeneous = all(t["device_homogeneous"] for t in report["test"].values())
    # CRN-pair device consistency (2026-07-12; implements the device-stratified seed-block
    # ratification, CHANGELOG [2026-07-11c]). Under seed-pool blocks a unit legitimately spans
    # devices, so per-UNIT homogeneity is the WRONG invariant — the paired inference needs every
    # unit at seed s on the SAME device class (the device then cancels in D_s). '<absent>' is a
    # wildcard. Single-pool runs satisfy this trivially, so the gate is strictly more correct.
    seed_devices: dict[str, set[str]] = {}
    for t in report["test"].values():
        for s, dev in t.get("per_seed_device", {}).items():
            if dev != "<absent>":
                seed_devices.setdefault(s, set()).add(dev)
    crn_violations = {s: sorted(devs) for s, devs in seed_devices.items() if len(devs) > 1}
    crn_consistent = not crn_violations
    # ── D16 (2026-08-01) ──────────────────────────────────────────────────────────────────────────
    # THE GATE PROMISED A CHECK IT DID NOT PERFORM. Its stop message claims it fires on "device
    # inhomogeneity", but `crn_consistent` keys on the cpu/cuda LABEL only — so the ONE inhomogeneity
    # that actually occurred in RUN 4 (four `baseline_volatility_scaled_return` records on an Intel
    # Xeon Gold 6140 against 26 on a 6240) passed silently. `check_substrate_fields` already rates
    # exactly this CRITICAL, but it is wired into the ADVISORY sentinel, not the BLOCKING gate.
    #
    # NOTE: IMPLEMENTED AS A PER-SEED PAIR INVARIANT, **NOT** the leg-wide census the deferred-fix note
    # sketched — and the difference is scientific, not cosmetic. Per-UNIT (or per-leg) homogeneity is
    # the WRONG invariant for the same reason it is wrong for the device check twelve lines above:
    # under seed-pool blocks a unit may legitimately span substrates, while what the PAIRED inference
    # actually needs is that at each seed s every unit shares one substrate, so the substrate cancels
    # in the difference D_s. A leg-wide census would red-flag a legitimate stratified run and would
    # still miss a mix that happened to balance across units. This form is strictly more correct and
    # exactly mirrors the device check, so the two can never disagree about what "inhomogeneous"
    # means. '<absent>' is a wildcard, as it is there.
    seed_substrates: dict[str, set[str]] = {}
    for t in report["test"].values():
        for s, sub in t.get("per_seed_substrate", {}).items():
            if sub != "<absent>":
                seed_substrates.setdefault(s, set()).add(sub)
    substrate_violations = {s: sorted(v) for s, v in seed_substrates.items() if len(v) > 1}
    substrate_consistent = not substrate_violations
    # P5 (2026-07-13): every test unit must carry exactly ONE reward hash (a mixed-winner unit is
    # scientifically meaningless — some seeds trained a different reward). Effect-blind: a HASH
    # census, never a performance value.
    mixed_winner_units = [u for u, t in report["test"].items()
                          if not t.get("winner_hash_consistent", True)]
    report["verdict"] = {
        "all_units_complete": all_complete,
        "device_homogeneous_everywhere": all_homogeneous,  # informational under seed-pool blocks
        "crn_pair_device_consistent": crn_consistent,
        "crn_device_violations": dict(list(crn_violations.items())[:10]),
        # D16: the SUBSTRATE analogue of the two lines above — CPU model / thread regime, the things
        # that actually fix float reduction order and that the device label cannot see.
        "crn_pair_substrate_consistent": substrate_consistent,
        "crn_substrate_violations": dict(list(substrate_violations.items())[:10]),
        "winner_hash_consistent_everywhere": not mixed_winner_units,
        "mixed_winner_units": mixed_winner_units[:10],
        # the GATE reads ONLY this — execution health, never a performance statistic
        "health_ok": bool(all_complete and crn_consistent and substrate_consistent
                          and not mixed_winner_units),
        "h2_arms": h2_arms,
    }

    # SEALED-SAFE SELECTION (descriptive; NOT a gate input) — each arm's authored winner reward CODE,
    # which is sealed from the confirmatory test leg. This is the mechanism headline the reviewer
    # actually wants to see; it carries NO performance number.
    if winners:
        report["selection"] = {
            arm: {"winner_id": w.get("candidate_id"),
                  "reward_source": str(w.get("reward_source", ""))[:4000]}
            for arm, w in winners.items()
        }

    json_path = out_root / f"tier1_integrity_{run.line_tag()}.json"
    json_path.write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")

    lines = ["# Tier-1 (C3 design floor) integrity report", "", _BLIND_HEADER, ""]
    lines += [f"- core n = {len(core_seeds)} seeds; arms = {', '.join(arms)}",
              f"- ALL UNITS COMPLETE: **{all_complete}**",
              f"- DEVICE HOMOGENEOUS EVERYWHERE: **{all_homogeneous}**", "", "## Search", ""]
    lines += ["| arm | records | ledgered failures | expected | matched budget |",
              "|---|---|---|---|---|"]
    for arm, s in report["search"].items():
        lines.append(f"| {arm} | {s['records']} | {s['ledgered_failures']} | "
                     f"{s['expected_records']} | {'OK' if s['matched_budget_ok'] else 'SHORT'} |")
    lines += ["", "## Sealed test leg", "",
              "| unit | present/expected | missing | devices | popart | safe-defaults |",
              "|---|---|---|---|---|---|"]
    for unit, t in report["test"].items():
        lines.append(
            f"| {unit} | {t['present']}/{t['expected']} | {len(t['missing_seeds'])} | "
            f"{t['device_census']} | {t['popart_scale_present']} | {t['train_safe_default_total']} |")
    if winners:
        lines += ["", "## Selection (SEALED-SAFE — authored winner reward code; NOT a gate input)", ""]
        for arm, sel in report["selection"].items():
            lines += [f"### {arm} — winner `{sel['winner_id']}`", "", "```python",
                      sel["reward_source"], "```", ""]
    lines += ["", "Gate proceeds AUTOMATICALLY on green health (no manual wait). It stops only on a "
              "real execution problem, or when `--hold-at-gate` is set — then create `TIER1_APPROVED` "
              "next to this report and re-run with `--resume`."]
    md_path = out_root / f"tier1_integrity_{run.line_tag()}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report, json_path, md_path
