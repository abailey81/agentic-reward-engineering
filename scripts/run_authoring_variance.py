"""Authoring-variance sub-experiment — REPORT-ONLY, DISJOINT from the frozen campaign.

WHY THIS EXISTS. Dr Okhrati, 2026-08-20, on the submitted draft:

    "The 102 seeds mostly measure randomness in training the RL agent. They do not adequately
    measure randomness in: what the LLM happens to write; which ideas it produces in a particular
    run; which program happens to become the winner; whether another run of the same LLM would
    produce a different conclusion etc. ... I think the ideal solution is to repeat the entire LLM
    authoring and selection exercise several times for each model and condition."

The campaign estimates a difference between two SELECTED programs and resamples the training seed
alone, so writing variance sits outside every interval by construction (CH6 Table 5.3 says so). This
sub-experiment supplies the missing component by doing exactly what he asked: it repeats the whole
authoring and selection exercise, and it trains the WHOLE final generation rather than only its
winner, so the distribution of what a model produces is observed and not inferred.

WHAT IT RUNS. Eleven authoring lines x five feedback arms x two independent chains. Each chain is a
fresh two-generation search of five candidates per generation, byte-identical in machinery to the
campaign's own loop (``run_search_arm`` with ``candidates=10, generations=2`` gives ``cpg=5``). All
five final-generation candidates are then trained on the sealed test window at the seed ladder.

    11 lines x 5 arms x 2 chains x 10 draws            = 1,100 authored programs
    11 lines x 5 arms x 2 chains x 5 final candidates  =   550 programs trained on test
    550 x 30 seeds                                     = 16,500 sealed trainings

TWO CONSTRAINTS THAT SHAPE THE LAYOUT, both verified before this file was written:

* ``src.feedback.schema.build_block`` accepts ONLY the five canonical arm names and raises
  ``ValueError`` on anything else. A chain therefore cannot be encoded in the arm name during
  SEARCH. Each (line, chain) gets its own archive root instead.
* ``build_test_specs`` uses the arm string purely as a label and never calls ``build_block``, so the
  per-candidate labels ``<arm>__c<k>`` are safe on the TEST side, where they keep the 25 programs of
  a chain from colliding at ``<label>-s<seed>``.

MAXIMISING CORES. Dispatch rate is set by fair share and is not ours to move (``src/cluster/driver``
:func:`chunk_specs`, measured 2026-08-06: 78 D-pool hosts held free slots while we won zero
dispatches in two hours). ``cores = dispatch_rate x duration x slots``, so DURATION is the only free
variable: pass a long ``--h-rt`` and a ``--specs-per-task`` sized to fill it, which holds cores for
several waves per job. Defaults here mirror the campaign's measured throughput shape.

NOT MAXIMISING SSH. Twelve concurrent driver lines once stampeded login12 and earned the account a
UCL usage penalty (measured 2026-08-03). ``--max-parallel-units`` bounds
the number of concurrent submission/polling streams; parallelism belongs on the compute nodes, in
array width, not on the login node.

NOTHING WAITS. Units are independent and pipelined: a unit submits its test array the moment its own
search finishes, while other units are still authoring. Within a unit the five arms search
concurrently on the ``ClusterRun``'s shared throttled puller and authoring lock.

Usage
-----
    # wiring only: no ssh, no submit, no spend
    python scripts/run_authoring_variance.py --dry-run --synthetic

    # the internal consistency battery
    python scripts/run_authoring_variance.py --selftest

    # one cell end-to-end on the real cluster, before anything scales
    python scripts/run_authoring_variance.py --lines opus-5 --chains 1 --arms distributional \
        --seeds 0-1 --max-parallel-units 1

    # the real run
    python scripts/run_authoring_variance.py --seeds 0-29 --resume
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_LOG = logging.getLogger("authoring_variance")

#: The eleven authoring lines. ``opus-5`` is the campaign's own author (config/campaign.yaml llm
#: block); the other ten are v2 replication legs resolved from config/legs.yaml.
CORE_LINE = "opus-5"
LEG_LINES: tuple[str, ...] = (
    "deepseek-v4-pro", "gemini-2.5-flash", "glm-5.2", "gpt-5.6-luna", "haiku-4.5",
    "kimi-k3", "nemotron-3-super", "qwen3.5-9b", "qwen3.6-27b", "sonnet-5",
)
ALL_LINES: tuple[str, ...] = (CORE_LINE,) + LEG_LINES

#: The five feedback arms, exactly as ``schema.build_block`` names them. Any other string raises.
ARMS: tuple[str, ...] = (
    "distributional", "scalar", "scalar_cvar5", "placebo", "placebo_shuffled",
)

GENERATIONS = 2          # g0 (identical prompts across arms) then g1 (the arm's feedback is live)
CANDIDATES = 10          # run_search_arm computes cpg = candidates // generations = 5
CANDIDATES_PER_GEN = CANDIDATES // GENERATIONS
FINAL_GEN = GENERATIONS - 1   # two generations are g0 and g1; the final one is g1

#: SGE ``-p`` for every array this driver submits. **ZERO, and it must stay zero.**
#:
#: ``src/cluster/campaign.py`` states the rule on PRIORITY_CORE: *"0 = full fair-share standing.
#: This is R101's requirement and the absolute standing rule: never lower the SGE priority of
#: any of our jobs, ever."* The campaign ran its core, Stage-1 and rung work at 0.
#:
#: This driver first used the runbook's -200 report-only ladder value, and it cost real wall-clock:
#: Myriad's scheduler weights are ``weight_priority 4.0`` against ``weight_ticket 1.5`` and
#: ``weight_urgency 0.0``, so ``-p`` is the single heaviest term in the ranking and -200 threw away
#: more than the whole ticket advantage. The ladder exists to stop report-only legs starving
#: CONFIRMATORY work — and there is none: the campaign is complete and holds zero running jobs, so
#: there is nothing to yield to and nothing to gain by yielding.
SUBMIT_PRIORITY = 0

DEFAULT_OUTPUT = "outputs/authoring_variance"


# --------------------------------------------------------------------------- #
# AUTHOR REPLAY — never pay twice for a program we already hold                #
# --------------------------------------------------------------------------- #
#
# WHY THIS EXISTS. A candidate's authored source is written into its staged task spec the
# moment it is submitted, but it only reaches the ARCHIVE when its training finishes. The
# campaign's own resume path (``campaign._archived_source``) reads the archive, so a driver
# restarted mid-training re-authors — and pays for — every candidate whose training had not yet
# returned. Measured 2026-08-21: 527 authored programs sat in ``_batches/*/task_*.json`` with a
# fresh $20 of provider credit standing behind them.
#
# The staged spec is a COMPLETE record of the draw: ``reward`` is the extracted source and
# ``prompt`` is the exact user message that produced it. Serving that pair back to the author
# reproduces the draw byte-for-byte, so the restart is scientifically identical to letting the
# original run continue — the same eleven models made the same 527 choices, and the original API
# calls remain recorded in each arm's ``llm_calls.jsonl``.
#
# NOTE: THE PROMPT MUST MATCH EXACTLY OR WE AUTHOR FRESH. A generation-1 prompt is built from
# generation 0's best candidate, so if selection lands differently the cached draw belongs to a
# context that no longer exists and replaying it would fabricate a chain that never happened.
# Matching on the prompt makes that impossible: a changed context misses, and the model is asked
# again. This is the one place where paying is the correct behaviour.
_REPLAY_CTX = threading.local()
_REPLAY_STATS: dict[str, int] = {"hits": 0, "context_misses": 0, "fresh": 0}
_REPLAY_STATS_LOCK = threading.Lock()


def build_replay_cache(output_dir: str | Path) -> dict[tuple[str, int, str, str], tuple[str, str]]:
    """``(line_slug, chain, arm, candidate_id) -> (reward_source, prompt)`` from staged task specs.

    Reads only what this driver itself staged, so there is no path by which a foreign program can
    enter. Test-leg specs share the directory but use a different candidate-id shape
    (``<arm>__c<k>-s<j>`` rather than ``<arm>-g<gen>-c<i>``), so they simply never match a lookup.
    """
    cache: dict[tuple[str, int, str, str], tuple[str, str]] = {}
    base = Path(output_dir)
    for path in base.glob("*/r*/_batches/*/task_*.json"):
        rel = path.relative_to(base).parts
        if len(rel) < 4:
            continue
        line_slug, chain_dir = rel[0], rel[1]
        if not chain_dir.startswith("r") or not chain_dir[1:].isdigit():
            continue
        chain = int(chain_dir[1:])
        try:
            specs = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue  # a half-written spec is not evidence of anything; author fresh
        if not isinstance(specs, list):
            continue
        for spec in specs:
            if not isinstance(spec, dict):
                continue
            arm, cid = spec.get("arm"), spec.get("candidate_id")
            source, prompt = spec.get("reward"), spec.get("prompt")
            if not (arm and cid and isinstance(source, str) and source.strip()
                    and isinstance(prompt, str) and prompt.strip()):
                continue
            cache[(str(line_slug), chain, str(arm), str(cid))] = (source, prompt)
    return cache


def install_author_replay(cache: dict[tuple[str, int, str, str], tuple[str, str]]) -> None:
    """Serve cached draws from ``campaign._complete_with_outage_tolerance``, the single author call.

    That function is the one place ``run_search_arm`` reaches the provider, and it is the only one
    that carries the candidate id (as ``label``). Patching it — rather than the transport — keeps
    every downstream gate untouched: the returned source still runs through
    ``extract_reward_source`` (a no-op on valid Python, byte-identical by its own fast path), then
    ``ast_gate`` and ``defines_reward``, exactly as a fresh completion would.
    """
    from src.cluster import campaign

    original = campaign._complete_with_outage_tolerance

    def _replaying(llm, system, user, *, label="", **kwargs):  # noqa: ANN001, ANN202
        key_prefix = getattr(_REPLAY_CTX, "unit", None)
        if key_prefix is not None and label:
            hit = cache.get((key_prefix[0], key_prefix[1], key_prefix[2], str(label)))
            if hit is not None:
                source, prompt = hit
                if prompt == user:
                    with _REPLAY_STATS_LOCK:
                        _REPLAY_STATS["hits"] += 1
                    return source
                with _REPLAY_STATS_LOCK:
                    _REPLAY_STATS["context_misses"] += 1
                _LOG.info("[replay:%s] cached draw discarded — the prompt changed, authoring fresh",
                          label)
        with _REPLAY_STATS_LOCK:
            _REPLAY_STATS["fresh"] += 1
        return original(llm, system, user, label=label, **kwargs)

    campaign._complete_with_outage_tolerance = _replaying


# --------------------------------------------------------------------------- #
# naming — one place, so the analysis can parse what the driver wrote          #
# --------------------------------------------------------------------------- #
def unit_root(output_dir: str | Path, line: str, chain: int) -> Path:
    """Archive root for one (line, chain). Chains are SEPARATE ROOTS, never renamed arms."""
    return Path(output_dir) / _slug(line) / f"r{int(chain)}"


def test_label(arm: str, cand_index: int) -> str:
    """The TEST-side label for one candidate. Safe: the test path never calls ``build_block``."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    return f"{arm}__c{int(cand_index)}"


def parse_test_label(label: str) -> tuple[str, int]:
    """Inverse of :func:`test_label`. Raises on anything this driver did not write."""
    m = re.fullmatch(r"([a-z0-9_]+)__c(\d+)", str(label))
    if not m or m.group(1) not in ARMS:
        raise ValueError(f"not an authoring-variance test label: {label!r}")
    return m.group(1), int(m.group(2))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", str(text).lower())


def _parse_seeds(spec: str) -> list[int]:
    """Comma list and/or inclusive ``a-b`` ranges. Rejects silent malformations."""
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = (t.strip() for t in part.split("-", 1))
            if not (a.isdigit() and b.isdigit()) or int(b) < int(a):
                raise SystemExit(f"--seeds range {part!r} must be 'A-B' with A <= B, non-negative")
            out.extend(range(int(a), int(b) + 1))
        else:
            if not part.isdigit():
                raise SystemExit(f"--seeds entry {part!r} must be a non-negative integer")
            out.append(int(part))
    seen: set[int] = set()
    ordered = [s for s in out if not (s in seen or seen.add(s))]
    if not ordered:
        raise SystemExit("--seeds resolved to an empty set")
    return ordered


# --------------------------------------------------------------------------- #
# candidate collection                                                        #
# --------------------------------------------------------------------------- #
def collect_final_generation(search_root: Path, arm: str) -> list[tuple[str, dict[str, Any]]]:
    """The final generation's candidates for one arm, as ``(test_label, record)`` pairs.

    Reads the archive, which is the only truth (``src/cluster/driver`` design law). Keeps only
    candidates whose ``candidate_id`` is ``<arm>-g<FINAL_GEN>-c<k>`` and that carry a usable reward
    source, ordered by ``k``. A rejected or crashed draw simply is not there — that is the campaign's
    own no-replacement rule, and its absence is itself the measurement (see
    ``scripts/pretrain_validate.check_matched_budget``).
    """
    from src.io.results import load_all

    arm_root = Path(search_root) / arm
    if not arm_root.is_dir():
        return []
    # The generation is recorded TWICE: as the required ``generation`` field and inside
    # ``candidate_id``. We require them to AGREE rather than trusting either, so an archive whose
    # two provenances disagree stops the unit loudly instead of quietly testing the wrong
    # generation. ``load_run`` reattaches ``reward_source`` from the reward.py sidecar and verifies
    # it byte-for-byte, so a record that still lacks a source genuinely produced no program.
    pattern = re.compile(rf"^{re.escape(arm)}-g(\d+)-c(\d+)$")
    out: list[tuple[int, str, dict[str, Any]]] = []
    for rec in load_all(arm_root):
        cid = str(rec.get("candidate_id") or rec.get("run_id") or "")
        m = pattern.match(cid)
        if not m:
            continue
        gen_from_id = int(m.group(1))
        gen_field = rec.get("generation")
        if gen_field is not None and int(gen_field) != gen_from_id:
            raise ValueError(
                f"{arm_root / cid}: generation field {gen_field!r} disagrees with candidate_id "
                f"{cid!r} (g{gen_from_id}) — refusing to guess which generation this is"
            )
        if gen_from_id != FINAL_GEN:
            continue
        if not rec.get("reward_source"):
            continue  # a draw that produced no executable program: absent by design, not an error
        k = int(m.group(2))
        out.append((k, test_label(arm, k), rec))
    out.sort(key=lambda t: t[0])
    return [(label, rec) for _k, label, rec in out]


# --------------------------------------------------------------------------- #
# one unit = one (line, chain)                                                #
# --------------------------------------------------------------------------- #
def run_unit(
    *,
    line: str,
    chain: int,
    arms: tuple[str, ...],
    seeds: list[int],
    output_dir: str | Path,
    cluster_kwargs: dict[str, Any],
    assemble_kwargs: dict[str, Any],
    resume: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """SEARCH all arms of one (line, chain), then TEST every final-generation candidate.

    Returns a result dict; never raises for an arm that produced nothing (that arm contributes no
    test records and is reported, which is the honest outcome for a low-yield model line).
    """
    from run_campaign_cluster import assemble_cluster_inputs, resolve_leg_override

    from src.utils.config import cfg_get, load_config

    root = unit_root(output_dir, line, chain)
    root.mkdir(parents=True, exist_ok=True)

    # The line's author block.
    #
    # NOTE: llm_cfg=None does NOT mean "the campaign's author". ``assemble_cluster_inputs`` falls back
    # to PROTOTYPE.yaml's llm block, which is a different, cheaper model — the exact defect the
    # 2026-07-13 pre-spend audit caught on the campaign runner itself
    # (scripts/run_campaign_cluster.py, --llm-from). The core line must therefore pass campaign.yaml's
    # block EXPLICITLY; a leg passes its pinned transport from config/legs.yaml, through the same
    # translation the pre-launch gates use.
    llm_cfg: dict[str, Any]
    if line == CORE_LINE:
        llm_cfg = dict(cfg_get(load_config("campaign"), "llm", {}) or {})
        if not llm_cfg.get("model_snapshot"):
            raise SystemExit("campaign.yaml llm.model_snapshot missing — the core author is unresolved")
        provider = str(llm_cfg.get("provider") or "anthropic")
    else:
        llm_cfg, provider, _suffix = resolve_leg_override(line, None)
    expected_model = str(llm_cfg["model_snapshot"])

    assembled = assemble_cluster_inputs(
        arms=list(arms),
        seeds=list(seeds),
        output_dir=str(root),
        candidates=CANDIDATES,
        generations=GENERATIONS,
        # Chains must differ. The LLM is stochastic (verified: two identical prompts to opus-5
        # returned 12.2%-similar programs), and the search seed additionally separates the
        # training RNG so the two chains share nothing.
        search_seed=int(chain) - 1,
        llm_cfg=llm_cfg,
        provider=provider,
        resume=resume,
        **assemble_kwargs,
    )
    opts = assembled["opts"]
    test_leg_kwargs = assembled["test_leg_kwargs"]

    # The guard that makes the wrong-author defect an impossible state rather than a silent one.
    # Every paid call and every training downstream depends on this being right, and nothing else
    # in the pipeline would notice: the archive would simply record a different model.
    actual_model = str(opts.get("model") or "")
    if actual_model != expected_model:
        raise SystemExit(
            f"[{line} r{chain}] AUTHOR MISMATCH: assembled model {actual_model!r} != expected "
            f"{expected_model!r}. Refusing to spend — the archive would record the wrong author."
        )
    actual_steps = int(assembled["agent_cfg"].get("train_steps_per_candidate") or 0)
    _prereg_bstar = int(cfg_get(load_config("preregistration"), "train_steps_per_candidate", 0) or 0)
    if assemble_kwargs.get("train_steps") is None and _prereg_bstar and actual_steps != _prereg_bstar:
        raise SystemExit(
            f"[{line} r{chain}] B* MISMATCH: assembled {actual_steps} != pre-registered "
            f"{_prereg_bstar}. Refusing to run a non-comparable training budget."
        )

    if dry_run:
        return {
            "line": line, "chain": chain, "root": str(root), "dry_run": True,
            "arms": list(arms), "seeds": len(seeds), "provider": provider,
            "model": actual_model, "train_steps": actual_steps,
            "windows": {"train": list(assembled["windows"][0]),
                        "val": list(assembled["windows"][1]),
                        "test": list(assembled["windows"][2])},
            "candidates_per_gen": max(1, int(opts["candidates"]) // int(opts["generations"])),
            "generations": int(opts["generations"]),
        }

    from src.cluster import build_cluster_run, run_search_arm, run_test_leg

    # NOTE: EACH UNIT GETS ITS OWN COMPLETE REMOTE ROOT, and this is a CORRECTNESS requirement, not
    # tidiness. The driver's pull mirrors the WHOLE remote outputs tree into the unit's local
    # archive. Sharing the campaign's root (~/Scratch/llmrp/outputs) drags its `search/<arm>/...`
    # records — and a `_quarantined_precampaign_...` tree — into our archive, where
    # ``collect_final_generation`` would read old campaign candidates as if this run had authored
    # them and send them to the sealed window. Caught by the 2026-08-21 prototype, whose pull
    # failed on exactly that quarantine directory.
    #
    # It is the ROOT that is split, not just the outputs sub-root: ``submit.prepare_remote`` creates
    # ``<root>/specs``, ``<root>/ledger``, ``<root>/outputs`` and ``<root>/logs/<batch>``, so a
    # custom outputs root alone is never created by anything and the first pull fails on a missing
    # directory (the prototype's second failure). Splitting the root keeps the standard layout.
    unit_kwargs = dict(cluster_kwargs)
    base_remote = str(unit_kwargs.pop("remote_root")).rstrip("/")
    unit_tag = f"av{int(chain)}{_slug(line)}"
    remote_unit_root = f"{base_remote}/av/{_slug(line)}_r{int(chain)}"
    from src.cluster.submit import prepare_remote, ssh_runner
    prepare_remote(remote_unit_root, [], ssh_runner(str(unit_kwargs.get("host") or "myriad")))
    # IMPORTANT: batch_tag IS MANDATORY HERE, and it is a CORRECTNESS requirement.
    #
    # ``run_search_arm`` names its array ``<arm>_g<gen>`` with no unit qualifier, and the driver's
    # double-submit guard matches queued jobs by NAME across the user's WHOLE queue. Without a tag
    # all 22 units compete for the same five names, and campaign.py's own 2026-07-11d bug-fix note
    # spells out the consequence: "the later run silently 'adopts' the earlier run's queued array
    # and polls an archive that job will never write to."
    #
    # Measured on 2026-08-21: launched untagged, the run produced SIX submissions in 4.4 hours
    # across 110 arms and left nothing running, because the units were serialising on five shared
    # names. Same-run resume adoption still works, because a unit's tag is deterministic.
    run = build_cluster_run(
        local_archive_root=str(root),
        local_batch_root=str(root / "_batches"),
        remote_root=remote_unit_root,
        remote_outputs_root=f"{remote_unit_root}/outputs",
        batch_tag=unit_tag,
        **unit_kwargs,
    )

    # THE AUTHORING LOCK IS THE RAMP BOTTLENECK, AND WIDENING IT IS SAFE HERE.
    #
    # build_cluster_run installs a plain threading.Lock (campaign.py:371, "arm-serial API"), so a
    # unit authors its five arms ONE AT A TIME: 25 sequential provider calls per generation. With 22
    # units that is only 22 concurrent calls, and it starves the cluster of work — measured
    # 2026-08-21, ten minutes in: 22 of 110 arms had submitted anything, 111 jobs existed, and 50 of
    # them were ALREADY RUNNING. Dispatch was not the constraint; we simply had not submitted enough.
    #
    # The lock guards exactly two things: the provider call, and ``author_guard`` — a hard spend cap
    # that is ``lambda: None`` unless ``max_author_calls`` is set (campaign.py:374). We do not set
    # it, so there is no shared counter and nothing to race. What remains is rate-limit courtesy, so
    # a SEMAPHORE sized to the arm count keeps that courtesy while letting a unit's five arms author
    # at once: 22 x 5 = 110 concurrent calls spread over ELEVEN different providers, about ten each,
    # and the client already carries bounded retry with backoff.
    run.author_lock = threading.Semaphore(len(arms))

    # SEARCH AND TEST NEED DIFFERENT JOB SHAPES, AND run_batch CANNOT OVERRIDE specs_per_task
    # PER CALL (it takes `pack` but not `specs_per_task`, campaign.py:288), so they need separate
    # ClusterRuns over the SAME archive roots.
    #
    # Why it matters, measured 2026-08-21: a search batch is five candidates. With
    # specs_per_task=32 all five land in ONE task, and `--search-pack 1` runs a task's specs one at
    # a time — 5 x 3.8 h = 19 h per generation, 38 h for the search alone, on 110 jobs instead of
    # 550. The campaign left specs_per_task UNSET for search precisely so each candidate became its
    # own job. The test flood wants the opposite: 32 specs per task is what holds cores for 68 h per
    # dispatch, and dispatch is the scarce thing.
    search_kwargs = dict(unit_kwargs)
    search_kwargs["specs_per_task"] = 1      # one candidate per job -> the five run in parallel
    run_search = build_cluster_run(
        local_archive_root=str(root),
        local_batch_root=str(root / "_batches"),
        remote_root=remote_unit_root,
        remote_outputs_root=f"{remote_unit_root}/outputs",
        batch_tag=unit_tag,
        **search_kwargs,
    )
    run_search.author_lock = run.author_lock  # one semaphore per unit, shared across both runs

    # ---- SEARCH -> TEST, PER ARM, WITH NO BARRIER BETWEEN THEM --------------------------------
    #
    # NOTE: NOTHING WAITS FOR ANYTHING IT DOES NOT DEPEND ON. An arm's test leg depends only on THAT
    # arm's own final generation, so it is submitted the instant that arm's search returns — while
    # the other four arms are still authoring. Batching all five into one test submission (the
    # earlier shape) left a finished arm's 150 trainings idle behind its slowest sibling, which on
    # a low-yield line like qwen3.5-9b could be hours of dead cores.
    #
    # What genuinely cannot be parallelised is INSIDE a chain: generation 1's prompt is built from
    # generation 0's BEST candidate, so the five candidates of g0 must all return before g1 can be
    # authored. That is reflect-on-best, the mechanism under test, not an implementation choice.
    searched: dict[str, Any] = {}
    errored: dict[str, str] = {}
    per_arm_counts: dict[str, int] = {}
    tests: dict[str, Any] = {}
    tested_units: list[tuple[str, dict[str, Any]]] = []

    def _search_then_test(arm: str) -> tuple[str, Any, list[tuple[str, dict[str, Any]]], Any]:
        # The replay cache is keyed by unit AND arm, because a candidate id repeats across all 22
        # units. `run_search_arm` authors synchronously in this thread, so a thread-local carries
        # the identity down to the patched author call without threading it through campaign.py.
        _REPLAY_CTX.unit = (_slug(line), int(chain), arm)
        res = run_search_arm(arm, opts, run_search, resume=resume, priority=SUBMIT_PRIORITY)
        found = collect_final_generation(run_search.search_read(), arm)
        if len(found) < CANDIDATES_PER_GEN:
            _LOG.warning("[%s r%d %s] %d of %d final-generation candidates survived the gate",
                         line, chain, arm, len(found), CANDIDATES_PER_GEN)
        t: Any = None
        if found:
            t = run_test_leg(
                found, list(seeds), run,
                name=f"{_slug(line)}_r{chain}_{arm}_test",
                resume=resume, priority=SUBMIT_PRIORITY, interleave=True,
                **test_leg_kwargs,
            )
        return arm, res, found, t

    with ThreadPoolExecutor(max_workers=len(arms)) as pool:
        futs = {pool.submit(_search_then_test, arm): arm for arm in arms}
        for fut in as_completed(futs):
            arm = futs[fut]
            try:
                _a, res, found, t = fut.result()
                searched[arm] = res
                per_arm_counts[arm] = len(found)
                tested_units.extend(found)
                if t is not None:
                    tests[arm] = t
            except Exception as exc:  # noqa: BLE001 — one arm must not kill the unit
                _LOG.error("[%s r%d %s] search/test failed: %s: %s",
                           line, chain, arm, type(exc).__name__, exc)
                errored[arm] = f"{type(exc).__name__}: {exc}"
                searched[arm] = {"ok": False, "error": errored[arm]}

    # NOTE: A BROKEN RUN AND AN UNPRODUCTIVE MODEL MUST NEVER LOOK THE SAME. A wiring fault makes every
    # arm yield nothing, which is exactly the shape of the genuine low-yield finding this study
    # reports for qwen3.5-9b (86% of its draws fail the gate). Reading one as the other would put a
    # fabricated capability result in the dissertation. So an arm whose search RAISED is reported as
    # a defect, never as a zero yield, and the unit refuses to continue on partial machinery.
    # Caught 2026-08-21 by the prototype, where an unexpanded '~' in apptainer_sif produced
    # "0 of 5 final-generation candidates survived the gate" on a model that had authored fine.
    if errored:
        return {"line": line, "chain": chain, "root": str(root), "ok": False,
                "reason": "search_error", "errors": errored, "search": searched,
                "note": "MACHINERY FAULT, NOT A YIELD MEASUREMENT — do not read these arms as "
                        "low-yield; nothing from this unit is usable until the fault is fixed."}

    if not tested_units:
        return {"line": line, "chain": chain, "root": str(root), "ok": False,
                "reason": "no_candidates", "search": searched, "counts": per_arm_counts}

    return {
        "line": line, "chain": chain, "root": str(root), "ok": True,
        "search": searched, "counts": per_arm_counts,
        "programs_tested": len(tested_units), "seeds": len(seeds), "tests": tests,
    }


# --------------------------------------------------------------------------- #
# selftest                                                                    #
# --------------------------------------------------------------------------- #
def selftest() -> int:
    """Internal consistency, with a falsifier for every claim that can be checked offline."""
    import tempfile

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    # 1. The arm vocabulary is exactly what build_block accepts, and nothing else is.
    from src.feedback import schema
    # The six fed statistics, from schema._DIST_FIELDS — the tail-carrying arms REQUIRE them.
    tail_stats = {"cvar_05": -0.019, "cvar_10": -0.014, "cvar_25": -0.008,
                  "cvar_01": -0.031, "left_tail_mass": 0.021, "robust_skew": 0.21}
    ok_all = True
    for arm in ARMS:
        try:
            schema.build_block(arm, 0.5, tail_stats,
                               shuffle_seed=0 if arm == "placebo_shuffled" else None)
        except Exception as exc:  # noqa: BLE001
            ok_all = False
            check(f"build_block accepts {arm}", False, f"{type(exc).__name__}: {exc}")
    check("build_block accepts all five arms", ok_all)
    bad_rejected = False
    try:
        schema.build_block("distributional__c0", 0.5, None)
    except ValueError:
        bad_rejected = True
    check("build_block REJECTS a test label (falsifier)", bad_rejected,
          "a renamed arm must never reach the feedback builder")

    # 2. Label round-trip.
    rt = all(parse_test_label(test_label(a, k)) == (a, k) for a in ARMS for k in range(5))
    check("test label round-trips", rt)
    bad_label = False
    try:
        parse_test_label("distributional-g1-c0")
    except ValueError:
        bad_label = True
    check("parse_test_label REJECTS a candidate id (falsifier)", bad_label)

    # 3. Chains never share a root.
    roots = {str(unit_root("out", ln, c)) for ln in ALL_LINES for c in (1, 2)}
    check("22 units, 22 distinct roots", len(roots) == len(ALL_LINES) * 2, f"got {len(roots)}")

    # 3b. Batch tags must be unique per unit, or the driver's name-matched double-submit guard makes
    #     units adopt each other's arrays (campaign.py 2026-07-11d). This is the check that would
    #     have caught the 2026-08-21 launch defect before it burned 4.4 hours.
    tags = [f"av{c}{_slug(ln)}" for ln in ALL_LINES for c in (1, 2)]
    check("22 units, 22 distinct batch tags", len(set(tags)) == len(tags),
          f"{len(set(tags))} unique of {len(tags)}")
    check("batch tags are SGE-name-safe", all(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", t) for t in tags))

    # 4. Seed parsing.
    check("seed range parses", _parse_seeds("0-29") == list(range(30)))
    check("seed list parses", _parse_seeds("0,5,7") == [0, 5, 7])
    rev = False
    try:
        _parse_seeds("29-0")
    except SystemExit:
        rev = True
    check("reversed seed range rejected (falsifier)", rev)

    # 5. cpg arithmetic matches run_search_arm's own floor division.
    check("candidates_per_gen == 5", max(1, CANDIDATES // GENERATIONS) == 5)
    check("final generation is g1", FINAL_GEN == 1)

    # 6. Candidate collection: only the final generation, only executable draws, ordered.
    with tempfile.TemporaryDirectory() as td:
        from src.io.results import write_run
        arm_root = Path(td) / "distributional"
        import hashlib
        made = [
            ("distributional-g0-c0", 0, "def r(): pass"),        # earlier generation -> excluded
            ("distributional-g1-c2", 1, "def r(): return 2"),
            ("distributional-g1-c0", 1, "def r(): return 0"),
            ("distributional-g1-c1", 1, None),                   # no program -> excluded
        ]
        for cid, gen, src in made:
            rec: dict[str, Any] = {
                "run_id": cid, "arm": "distributional", "seed": 0, "fold": "val",
                "candidate_id": cid, "generation": gen,
                "reward_source_hash": hashlib.sha256((src or "").encode()).hexdigest(),
                "feedback_block": "", "metrics": {}, "wall_clock": 1.0,
                "env_fingerprint": "selftest", "val_fitness": 0.1,
            }
            if src is not None:
                rec["reward_source"] = src
            write_run(rec, arm_root)
        got = collect_final_generation(Path(td), "distributional")
        labels = [lab for lab, _ in got]
        check("collects ONLY final-generation, source-bearing candidates, in index order",
              labels == ["distributional__c0", "distributional__c2"], f"got {labels}")

        # Falsifier: a record whose two generation provenances disagree must STOP the unit.
        bad_cid = "distributional-g1-c4"
        bad = {
            "run_id": bad_cid, "arm": "distributional", "seed": 0, "fold": "val",
            "candidate_id": bad_cid, "generation": 0,  # disagrees with the id's g1
            "reward_source": "def r(): return 4",
            "reward_source_hash": hashlib.sha256(b"def r(): return 4").hexdigest(),
            "feedback_block": "", "metrics": {}, "wall_clock": 1.0,
            "env_fingerprint": "selftest",
        }
        write_run(bad, arm_root)
        raised = False
        try:
            collect_final_generation(Path(td), "distributional")
        except ValueError:
            raised = True
        check("generation-provenance mismatch RAISES (falsifier)", raised,
              "a disagreeing archive must never be silently tested")

    # 7. Author replay reproduces the draw exactly, and refuses to when the context has moved.
    with tempfile.TemporaryDirectory() as td:
        src_text = "def reward(weights, returns, prev_weights, port_ret, info):\n    return 0.0\n"
        batch = Path(td) / "opus_5" / "r2" / "_batches" / "av2opus_5_distributional_g0"
        batch.mkdir(parents=True)
        (batch / "task_1.json").write_text(json.dumps([
            {"arm": "distributional", "candidate_id": "distributional-g0-c0",
             "reward": src_text, "prompt": "PROMPT-A", "generation": 0},
            {"arm": "distributional", "candidate_id": "distributional-g0-c1",
             "reward": "", "prompt": "PROMPT-B", "generation": 0},  # empty source: not a draw
        ]), encoding="utf-8")
        cache = build_replay_cache(td)
        check("replay cache reads staged draws, keyed by unit+arm+candidate",
              list(cache) == [("opus_5", 2, "distributional", "distributional-g0-c0")],
              f"got {list(cache)}")

        # The gate the source must survive is the same one a fresh completion faces, and
        # extract_reward_source promises a byte-identical no-op on text that already parses.
        from src.sandbox.executor import ast_gate, defines_reward, extract_reward_source
        replayed = extract_reward_source(src_text)
        check("a replayed draw survives extraction BYTE-IDENTICALLY",
              replayed == src_text, f"{len(replayed)} vs {len(src_text)} chars")
        check("a replayed draw passes the author gate", ast_gate(replayed) and defines_reward(replayed))

        from src.cluster import campaign
        saved = campaign._complete_with_outage_tolerance
        try:
            calls: list[str] = []

            def _counting(llm, system, user, *, label="", **kwargs):  # noqa: ANN001, ANN202
                calls.append(label)
                return "def reward(w, r, p, pr, i):\n    return 1.0\n"

            campaign._complete_with_outage_tolerance = _counting
            install_author_replay(cache)
            patched = campaign._complete_with_outage_tolerance
            _REPLAY_CTX.unit = ("opus_5", 2, "distributional")

            got = patched(None, "SYS", "PROMPT-A", label="distributional-g0-c0")
            check("a matching prompt is served from disk, with NO provider call",
                  got == src_text and calls == [], f"calls={calls}")

            # Falsifier: generation 1's prompt depends on generation 0's winner. If that prompt is
            # not the one the cached draw answered, the draw belongs to a chain that never
            # happened, and paying again is the only correct behaviour.
            got2 = patched(None, "SYS", "A DIFFERENT PROMPT", label="distributional-g0-c0")
            check("a CHANGED prompt refuses the cache and authors fresh (falsifier)",
                  got2 != src_text and calls == ["distributional-g0-c0"], f"calls={calls}")

            # Falsifier: candidate ids repeat across all 22 units, so the unit must be part of the key.
            _REPLAY_CTX.unit = ("sonnet_5", 2, "distributional")
            got3 = patched(None, "SYS", "PROMPT-A", label="distributional-g0-c0")
            check("another LINE never receives this line's draw (falsifier)",
                  got3 != src_text and len(calls) == 2, f"calls={calls}")
        finally:
            campaign._complete_with_outage_tolerance = saved
            _REPLAY_CTX.unit = None

    # 8. Every job shape we submit must fit Myriad's 48-hour walltime ceiling.
    #
    # This check exists because the ceiling is invisible until submission: the driver builds a
    # jobscript happily, `qsub` refuses it, and the run reports a queue-op failure that looks like
    # SSH trouble. On 2026-08-21 the default --h-rt of 90:00:00 meant every test leg failed this
    # way, seven times over, while the search legs (submitted at 48h) ran normally.
    MYRIAD_H_RT_CEILING_H = 48.0        # probed: 48:00:00 accepted, 60:00:00 refused
    CORE_HOURS_PER_TRAINING = 8.39      # measured, the executed campaign §39.3, at one thread
    defaults = build_parser().parse_args([])

    def _hours(hms: str) -> float:
        h, m, s = (int(x) for x in str(hms).split(":"))
        return h + m / 60.0 + s / 3600.0

    for label, h_rt in (("--h-rt", defaults.h_rt), ("--search-h-rt", defaults.search_h_rt)):
        check(f"{label} is inside Myriad's 48h ceiling",
              _hours(h_rt) <= MYRIAD_H_RT_CEILING_H, f"{h_rt} = {_hours(h_rt):.1f} h")

    waves = -(-int(defaults.specs_per_task) // int(defaults.pack))   # ceiling division
    test_wall = waves * CORE_HOURS_PER_TRAINING / max(1, int(defaults.cores_per_training))
    check("a test job's work fits its own walltime, with margin",
          test_wall <= _hours(defaults.h_rt) / 1.3,
          f"{waves} waves x {CORE_HOURS_PER_TRAINING} h = {test_wall:.1f} h "
          f"against {_hours(defaults.h_rt):.0f} h")
    # A task archives NOTHING until all but its last wave has finished (DevicePool.submit_with
    # blocks on a full pool, parallel.py:608), so waves are SILENCE. This run exists to give the
    # 5/10/15/20/25/30-seed readouts, and a truncated run reports its largest COMPLETED rung — so
    # records that arrive late are records that may never count. Two waves is the documented shape.
    check("a test job archives within two waves, so early seed rungs survive truncation",
          waves <= 2, f"{waves} waves = {(waves - 1) * CORE_HOURS_PER_TRAINING:.1f} h of silence "
                      f"before the first record")
    # Memory is the placement discriminator (campaign record section 38, re-measured 2026-08-21:
    # 4 cores at 2G placed 15/15 at the first pass, 8 cores at 2G placed 0/15).
    job_mem_gb = int(defaults.pack) * 2
    check("a test job's memory request stays in the 8GB tier that places",
          job_mem_gb <= 8, f"--pack {defaults.pack} x 2G = {job_mem_gb} GB per job")
    search_wall = CORE_HOURS_PER_TRAINING / 1.75    # one training at --search-threads 8
    check("a search job's work fits its own walltime, with margin",
          search_wall <= _hours(defaults.search_h_rt) / 1.3,
          f"{search_wall:.1f} h against {_hours(defaults.search_h_rt):.0f} h")

    # 9. Scale arithmetic matches the design.
    n_units = len(ALL_LINES) * 2
    draws = n_units * len(ARMS) * CANDIDATES
    progs = n_units * len(ARMS) * CANDIDATES_PER_GEN
    check("1,100 authored draws", draws == 1100, f"got {draws}")
    check("550 programs on test", progs == 550, f"got {progs}")
    check("16,500 sealed trainings at 30 seeds", progs * 30 == 16500)

    width = max(len(n) for n, _, _ in checks)
    failed = 0
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
        failed += 0 if ok else 1
    print(f"\n{len(checks) - failed} of {len(checks)} checks passed")
    return 1 if failed else 0


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Authoring-variance sub-experiment (report-only, disjoint).")
    p.add_argument("--lines", nargs="+", default=None,
                   help=f"Authoring lines (default: all eleven). Known: {', '.join(ALL_LINES)}")
    p.add_argument("--arms", nargs="+", default=None,
                   help=f"Feedback arms (default: all five). Known: {', '.join(ARMS)}")
    p.add_argument("--chains", nargs="+", type=int, default=[1, 2],
                   help="Independent authoring chains per cell (default: 1 2).")
    p.add_argument("--seeds", default="0-29",
                   help="Test seeds, comma list and/or a-b ranges (default 0-29). Seeds run in "
                        "registered order, so ANY completed contiguous prefix is a whole study — "
                        "that is what gives the 5/10/15/20/25/30 readouts.")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--replay-authored", action="store_true",
                   help="Serve a candidate's authored source from its staged task spec instead of "
                        "paying the provider again, but ONLY when the prompt matches exactly. Use "
                        "whenever a restart follows a run that had already submitted work: the "
                        "draws are identical and the provider is not charged twice.")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate wiring and print the plan; no ssh, no submit, no spend.")
    p.add_argument("--synthetic", action="store_true", help="Synthetic panel (dry-run only).")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--max-parallel-units", type=int, default=22,
                   help="Concurrent (line, chain) streams. ALL 22 by default, because this number "
                        "SERIALISES THE SEARCH: a unit's two generations cost about 17 h whatever "
                        "else is running, so 6 at a time turns a 17 h authoring phase into 68 h. "
                        "The units are independent and nothing waits on anything.")
    p.add_argument("--stagger-secs", type=float, default=20.0,
                   help="Delay between unit STARTS. Full concurrency is safe; a synchronised burst "
                        "is not. Twelve driver lines resuming AT ONCE stampeded login12 in 2026-08 "
                        "(qacct at 298.9%% CPU) and the account was capped for 30 minutes. Staggering "
                        "de-phases the poll and drain cycles so the same total work never lands on "
                        "the login node in one instant. 22 units x 20 s = under 8 minutes of ramp "
                        "against a multi-day run.")
    # Cluster wiring (defaults mirror the campaign's measured throughput shape)
    p.add_argument("--host", default="myriad")
    p.add_argument("--remote-root", default="~/Scratch/llmrp")
    p.add_argument("--gold-dir", default="~/ACFS/gold",
                   help="Staged licensed gold on the cluster. The campaign runner's default "
                        "(~/Scratch/llmrp/inputs) is EMPTY on this account; the real univ5 panel "
                        "lives on ACFS and its three parquets hash-match the local copies and the "
                        "campaign archive's recorded manifest.")
    p.add_argument("--pool", default="db",
                   help="CPU node classes. d+b are the ONLY ones open to us and this is the "
                        "measured ceiling: t is AMD EPYC and would break CRN bit-exactness; "
                        "e/f/l/u/v are GPU nodes whose idle cores we refuse by construction "
                        "because taking them blocks other users' GPU jobs; d97 is PAID and "
                        "gated. Measured 2026-08-21 at pack 4: d=1,992 placeable cores, "
                        "b=48. See src/cluster/lanes.EXCLUDED_CPU_POOLS.")
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--pack", type=int, default=4,
                   help="Trainings running AT ONCE inside one job, one core each (packing is not "
                        "threading), so the job asks for --pack cores and --pack x 2G of memory. "
                        "FOUR, not the C4 campaign's eight, because memory placement is a property "
                        "of the DAY and eight does not place on this one. Measured 2026-08-21 with "
                        "15 sleep-only canaries per shape, submitted together: 4 cores at 2G (8GB "
                        "per job) placed 15 of 15 at the first scheduling pass, while 8 cores at 2G "
                        "(16GB) placed 0 of 15. The campaign's own pack-8 note predicts exactly "
                        "this — it records pool d giving '3 jobs = 24 cores' at pack 8 because 82%% "
                        "of its usable hosts hold under 16G free — and it adopted eight on a day "
                        "when the wider job still placed.")
    p.add_argument("--cores-per-training", type=int, default=1,
                   help="1 is throughput-optimal: 8.39 core-hours per training against 38.41 at "
                        "8 threads (measured, the executed campaign §39.3).")
    p.add_argument("--specs-per-task", type=int, default=8,
                   help="Trainings per TASK in total. EIGHT is two waves at --pack 4, which is what "
                        "the paragraph below actually prescribes; the previous default of 32 was "
                        "eight waves, contradicted its own help text, and at 8.39 h a wave needed "
                        "67 h of walltime that Myriad refuses (the cap is 48 h). "
                        "Larger than --pack means several waves in one "
                        "job, which HOLDS CORES LONGER at the same dispatch rate — duration is the "
                        "only lever we control. TWO waves (2 x --pack) is deliberate, not timid: "
                        "DevicePool.submit_with BLOCKS on a full pool (parallel.py:608) and run_one "
                        "materialises every submission before as_completed, so a task archives "
                        "NOTHING until all but the last wave has finished. At 5 waves that is about "
                        "34 h of silence, which would destroy the 5/10/15-seed readouts this run "
                        "exists to provide. Two waves doubles core-holding and still lands records "
                        "at about 8.5 h. Raise it only if you no longer need early rungs.")
    p.add_argument("--h-rt", default="48:00:00",
                   help="Wallclock per job. 48:00:00 IS THE CEILING, not a preference: probed "
                        "directly on 2026-08-21, Myriad accepts 48:00:00 and refuses 60:00:00 and "
                        "above with 'Rejected by policyjsv Reason: Unable to find a place to run "
                        "this job'. The refusal is on walltime alone — 4, 8 and 16 cores are all "
                        "accepted at 48h. The previous 90:00:00 default meant every test leg failed "
                        "to submit. Sized for --specs-per-task / --pack waves at the measured 8.39 "
                        "core-hours per training (the executed campaign §39.3) plus headroom: "
                        "4 x 8.39 = 33.6h, a 1.43x margin.")
    p.add_argument("--search-h-rt", default="48:00:00",
                   help="Wallclock for SEARCH jobs, which ask for more cores than test jobs and are "
                        "therefore rejected at the walltime the test flood uses. 48:00:00 is the "
                        "measured-good value; the default --h-rt is refused by Myriad policy here.")
    p.add_argument("--search-threads", type=int, default=8,
                   help="Threads per SEARCH training. The search chain is LATENCY-bound (a "
                        "generation cannot start until the previous one returns), where 8 threads "
                        "buys 1.75x less latency; the test ladder is throughput-bound and uses 1.")
    p.add_argument("--chunk-tasks", type=int, default=1,
                   help="Tasks per submitted ARRAY. ONE, and this is hard-won: Myriad SERIALISES "
                        "array tasks by policy — tasks 2..n sit in hqw while task 1 runs, and "
                        "pending tails have twice been PURGED OUTRIGHT (mode_d_supervisor.ps1:119). "
                        "An array of 5 therefore runs at one-fifth speed and can silently lose its "
                        "tail. One job per task, no tail. This is what lets 4-core jobs place in "
                        "about 6 minutes and 1,000 jobs x 4 reach the 4,000-core ceiling that "
                        "max_u_jobs allows.")
    p.add_argument("--search-pack", type=int, default=1,
                   help="Trainings per SEARCH job. One, because the search lane runs at 8 threads "
                        "(latency-bound: a generation cannot start until the previous returns), "
                        "while the test flood is throughput work at 1 core each.")
    p.add_argument("--train-steps", type=int, default=None,
                   help="SMOKE-TEST ONLY. Leave UNSET for any real run: unset resolves B* from "
                        "campaign.yaml and hard-asserts it equals the pre-registered value, which "
                        "is the guard that stops a campaign training at 1/16th of B*. Setting it "
                        "BYPASSES that assert, so a run with this flag is a wiring rehearsal and "
                        "its records are not comparable to anything.")
    p.add_argument("--apptainer-sif", default="~/python311.sif")
    p.add_argument("--poll-secs", type=float, default=180.0,
                   help="Burst cadence for the test flood (campaign value).")
    p.add_argument("--search-poll-secs", type=float, default=90.0,
                   help="Chain cadence. Short on purpose: every generation handoff pays up to "
                        "one poll of notice, and the search chain is the latency-bound phase.")
    p.add_argument("--exclude-hosts", default=None)
    return p


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass
    args = build_parser().parse_args()

    if args.selftest:
        return selftest()

    lines = tuple(args.lines) if args.lines else ALL_LINES
    unknown = [ln for ln in lines if ln not in ALL_LINES]
    if unknown:
        raise SystemExit(f"unknown line(s) {unknown}; known: {', '.join(ALL_LINES)}")
    arms = tuple(args.arms) if args.arms else ARMS
    bad_arms = [a for a in arms if a not in ARMS]
    if bad_arms:
        raise SystemExit(f"unknown arm(s) {bad_arms}; build_block accepts only {', '.join(ARMS)}")
    chains = [int(c) for c in args.chains]
    seeds = _parse_seeds(args.seeds)

    from src.utils.env import load_env
    load_env()

    n_units = len(lines) * len(chains)
    print(f"lines {len(lines)}  arms {len(arms)}  chains {len(chains)}  units {n_units}")
    print(f"seeds {len(seeds)} ({seeds[0]}..{seeds[-1]})   generations {GENERATIONS} x "
          f"{CANDIDATES_PER_GEN} candidates")
    print(f"authored draws {n_units * len(arms) * CANDIDATES}   "
          f"programs on test {n_units * len(arms) * CANDIDATES_PER_GEN}   "
          f"sealed trainings {n_units * len(arms) * CANDIDATES_PER_GEN * len(seeds)}")
    print(f"output {args.output_dir}   parallel units {args.max_parallel_units}")

    if args.replay_authored and not args.dry_run:
        cache = build_replay_cache(args.output_dir)
        install_author_replay(cache)
        print(f"author replay ARMED: {len(cache)} staged draws recovered from disk "
              f"(a cached draw is served only when its prompt matches exactly)")
    print()

    if args.train_steps is not None:
        print(f"!! SMOKE TEST: train_steps forced to {args.train_steps}, which BYPASSES the "
              f"pre-registered B* assert. These records are a wiring rehearsal, not results.\n")
    assemble_kwargs = dict(
        synthetic=bool(args.synthetic),
        train_steps=(int(args.train_steps) if args.train_steps is not None else None),
        n_trials=0,
        embargo=0,
        pass_mode="B",
    )
    # SGE directive lines expand NOTHING: a leading '~' in remote_root lands verbatim in '#$ -wd'
    # and the job dies. jobscript.render_jobscript rejects a non-absolute remote_root outright, so
    # resolve the real remote home ONCE over ssh and expand, exactly as the campaign runner does.
    if not args.dry_run:
        from src.cluster.submit import expand_remote, remote_home, ssh_runner
        _home = remote_home(ssh_runner(args.host))
        args.remote_root = expand_remote(args.remote_root, _home)
        args.gold_dir = expand_remote(args.gold_dir, _home)
        # apptainer_sif too: render_jobscript rejects a literal '~' in it for the same reason.
        args.apptainer_sif = expand_remote(args.apptainer_sif, _home)
        print(f"remote root {args.remote_root}\ngold        {args.gold_dir}\n"
              f"apptainer   {args.apptainer_sif}\n")

    # NOTE: remote_outputs_root is deliberately ABSENT here — run_unit sets a PER-UNIT one, so no
    # unit can ever mirror another unit's or the campaign's records into its own archive.
    cluster_kwargs = dict(
        remote_root=args.remote_root,
        gold_dir=args.gold_dir,
        host=args.host,
        pool_confirmatory=args.pool,
        pool_report_only=args.pool,
        pack=int(args.pack),
        chunk_tasks=int(args.chunk_tasks),
        specs_per_task=int(args.specs_per_task),
        cores_per_training=int(args.cores_per_training),
        h_rt=args.h_rt,
        # NOTE: A SEARCH JOB MUST ASK FOR LESS WALLTIME THAN A TEST JOB, and this is a HARD gate, not
        # a preference. Myriad's policy filter rejects at submission: measured 2026-08-21,
        # ``qsub`` on a 90-hour 8-core request answered "Rejected by policyjsv Reason: Unable to
        # find a place to run this job 3.8days, 8 cores" and NOTHING entered the queue. The test
        # flood asks for 90 hours at 4 cores and is accepted; the same 90 hours at the search leg's
        # 8 cores is not. Left unset, ``search_h_rt`` falls through to ``h_rt`` and every search
        # submission fails — which is exactly what happened on the first relaunch.
        #
        # 48 hours is the value that demonstrably placed: 105 search jobs entered the queue under it
        # earlier the same day, 59 of them running. A search job at ``specs_per_task=1`` runs ONE
        # ~3.8-hour training, so this is still a 12x over-request, and a shorter request is only
        # ever easier to place — but 48 is the measured-good number and a walltime kill costs far
        # more than a slightly slower placement.
        search_h_rt=args.search_h_rt,
        search_threads=int(args.search_threads),
        search_pack=(int(args.search_pack) if args.search_pack else None),
        apptainer_sif=args.apptainer_sif,
        poll_secs=float(args.poll_secs),
        min_pull_interval=120.0,
        search_poll_secs=float(args.search_poll_secs),
        device=args.device,
        exclude_hosts=([h.strip() for h in args.exclude_hosts.split(",")]
                       if args.exclude_hosts else None),
    )

    units = [(ln, ch) for ln in lines for ch in chains]
    results: list[dict[str, Any]] = []
    lock = threading.Lock()
    t0 = time.time()

    _start_gate = threading.Lock()
    _next_start = [0.0]

    def one(line: str, chain: int) -> dict[str, Any]:
        # De-phase the unit starts (see --stagger-secs). Taken here rather than at submit time so a
        # unit's whole poll/drain cycle inherits the offset, not just its first ssh call.
        if float(args.stagger_secs) > 0 and not args.dry_run:
            with _start_gate:
                wait = max(0.0, _next_start[0] - time.monotonic())
                _next_start[0] = time.monotonic() + wait + float(args.stagger_secs)
            if wait > 0:
                time.sleep(wait)
        try:
            r = run_unit(
                line=line, chain=chain, arms=arms, seeds=seeds,
                output_dir=args.output_dir, cluster_kwargs=cluster_kwargs,
                assemble_kwargs=assemble_kwargs, resume=bool(args.resume),
                dry_run=bool(args.dry_run),
            )
        except Exception as exc:  # noqa: BLE001 — one unit must not kill the run
            _LOG.exception("[%s r%d] unit failed", line, chain)
            r = {"line": line, "chain": chain, "ok": False,
                 "error": f"{type(exc).__name__}: {exc}"}
        with lock:
            results.append(r)
            done = len(results)
            print(f"[{done}/{len(units)}  {(time.time()-t0)/60:.1f} min] "
                  f"{line} r{chain}: {'ok' if r.get('ok') or r.get('dry_run') else r.get('reason') or r.get('error')}",
                  flush=True)
        return r

    workers = max(1, min(int(args.max_parallel_units), len(units)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(one, ln, ch) for ln, ch in units]
        for f in as_completed(futs):
            f.result()

    summary_path = Path(args.output_dir) / "authoring_variance_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({
        "lines": list(lines), "arms": list(arms), "chains": chains,
        "seeds": seeds, "generations": GENERATIONS,
        "candidates_per_gen": CANDIDATES_PER_GEN,
        "dry_run": bool(args.dry_run),
        "elapsed_min": round((time.time() - t0) / 60.0, 2),
        # Provenance, directive 6: a replayed draw made no provider call in THIS process, so the
        # count of replays is part of the record of how the run was produced.
        "author_replay": dict(_REPLAY_STATS),
        "units": results,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {summary_path}")
    if _REPLAY_STATS["hits"] or _REPLAY_STATS["context_misses"]:
        print(f"author replay: {_REPLAY_STATS['hits']} served from disk, "
              f"{_REPLAY_STATS['context_misses']} discarded on a changed prompt, "
              f"{_REPLAY_STATS['fresh']} authored fresh")
    failed = [r for r in results if not (r.get("ok") or r.get("dry_run"))]
    if failed:
        print(f"{len(failed)} of {len(results)} units did not complete")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
