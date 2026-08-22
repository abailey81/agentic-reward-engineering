#!/usr/bin/env python3
"""Chain 3 = the campaign's OWN authoring draw, sent through the sealed window unchanged.

WHY THIS EXISTS
---------------
``scripts/run_authoring_variance.py`` answers Dr Okhrati's point 1 by re-running the whole authoring
and selection exercise twice per model (chains 1 and 2), so the spread between chains measures how
much the LLM's *writing* moves the result. Two chains give that spread one degree of freedom, and
they leave the most interesting question implicit: where does the draw that actually produced the
dissertation's headline sit inside it?

The campaign already contains a third, independent draw of exactly the same procedure. This script
takes the campaign's generation-1 programs and runs them through the SAME sealed window, on the SAME
seeds, with the SAME training pipeline as chains 1 and 2. Nothing is authored, nothing is searched
and no provider is called: the programs exist on disk, so the only cost is cluster time.

WHAT WAS VERIFIED BEFORE THIS WAS WRITTEN (2026-08-22, first-hand)
------------------------------------------------------------------
* The campaign's generation-0 prompt for haiku-4.5/distributional is BYTE-IDENTICAL to chain 1's
  (2,602 characters, sha256 39cda898f55cbe1c...). The two runs pose the model the same question.
* The three draws produced DIFFERENT programs. Nine candidates checked across three arms: campaign,
  chain 1 and chain 2 differ in every one. That is what makes this a third draw rather than a copy.
* Every line has the campaign's five arms, five candidates per generation, six generations.
* 221 of a possible 275 generation-1 programs exist and every one carries readable source.

TWO PROPERTIES THAT MUST BE CARRIED INTO THE WRITE-UP
-----------------------------------------------------
1. THIS CHAIN IS NOT BLIND. It is the run that produced the headline result. It therefore belongs in
   the paper as a placement ("the headline draw sits here among three") and NOT as a third
   observation quietly pooled into a variance estimate, which would be circular.
2. THIS CHAIN IS NOT BALANCED. 221 of 275, and qwen3.5-9b contributes 1 of 25 because 86% of its
   campaign draws failed the gate. An unbalanced nest cannot enter the balanced nested mean-squares
   estimator in ``scripts/analyse_authoring_variance.py``.

Both properties are why the records land in their OWN tree, ``outputs/campaign_chain3``, and not
under ``outputs/authoring_variance/<line>/r3``. The analysis script globs ``*/r*/test/...``; a
directory named ``r3`` there would be swept into the balanced estimator silently, which is precisely
the class of accident that put 2,000-step prototype records into this study's archive on 2026-08-21.
Reading chain 3 has to be a deliberate act.

WHY IT WILL NOT SLOW THE MAIN RUN
---------------------------------
Myriad caps one user at 1,000 jobs (``max_u_jobs``), and on 2026-08-22 the authoring-variance run
sat at 994 with 651 tasks already queued and unable to place. Submitting chain 3 into that would
displace primary work. So every submission passes a HEADROOM GATE: a shared, politely-polled job
count, and a worker waits until the queue is below ``--max-jobs`` before it submits. Chain 3 eats
leftovers and nothing else.

Run::

    python scripts/run_campaign_chain3.py --selftest
    python scripts/run_campaign_chain3.py --dry-run
    python scripts/run_campaign_chain3.py --resume
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_LOG = logging.getLogger("campaign_chain3")

#: The chain number this run publishes under. Chains 1 and 2 are the two fresh re-runs; the
#: campaign is the third draw in time order and the only one that is not blind.
CHAIN = 3

#: Where the campaign's search archives live, relative to the repo root.
CAMPAIGN_ROOT = "outputs/campaign_cluster_run4"

DEFAULT_OUTPUT = "outputs/campaign_chain3"

#: Wait this long between job-count polls. The login node is shared and this account has been
#: penalised once for a usage spike, so the gate polls ONCE for all workers and does so slowly.
POLL_SECS = 180.0

#: After this many consecutive failed polls the gate proceeds ungated and says so. A gate that can
#: deadlock the run it is protecting is worse than the crowding it prevents; a submission refused at
#: the cap merely returns exit 25 and is retried by the driver.
MAX_BLIND_POLLS = 10


# --------------------------------------------------------------------------- #
# campaign archive resolution                                                 #
# --------------------------------------------------------------------------- #
def campaign_search_root(line: str, campaign_root: str | Path = CAMPAIGN_ROOT) -> Path:
    """The campaign's search archive for one model line.

    The core line's archive is ``search/`` (it also holds the four optimiser baselines, which we
    never select because ``collect_final_generation`` filters on the arm name). Every other line is
    a leg with its own ``search_leg_<slug>/`` directory.
    """
    from run_authoring_variance import CORE_LINE, _slug

    base = Path(campaign_root)
    return base / "search" if line == CORE_LINE else base / f"search_leg_{_slug(line)}"


def collect_campaign_chain(line: str, arms: tuple[str, ...],
                           campaign_root: str | Path = CAMPAIGN_ROOT
                           ) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    """``{arm: [(test_label, record), ...]}`` for the campaign's FINAL-generation programs.

    Reuses ``collect_final_generation`` unchanged, which is the point: chain 3 is admitted through
    exactly the gate chains 1 and 2 passed — the generation recorded in ``candidate_id`` must agree
    with the ``generation`` field, and a record without a verified reward source is absent by design.
    """
    from run_authoring_variance import FINAL_GEN, collect_final_generation

    if int(FINAL_GEN) != 1:
        raise SystemExit(
            f"FINAL_GEN is {FINAL_GEN}, not 1. This script takes the campaign's g1 because that is "
            f"the same DEPTH as the authoring-variance run's final generation. If the study's depth "
            f"changed, chain 3 must be re-cut to match or it is not comparable."
        )
    root = campaign_search_root(line, campaign_root)
    out: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for arm in arms:
        found = collect_final_generation(root, arm)
        if found:
            out[arm] = found
    return out


# --------------------------------------------------------------------------- #
# the headroom gate                                                           #
# --------------------------------------------------------------------------- #
class HeadroomGate:
    """Shared, politely-polled view of how many jobs we hold, so chain 3 only fills leftovers.

    One thread polls; every worker reads the cached number. ``wait()`` blocks while the queue is at
    or above ``max_jobs``. Reads are best-effort: after ``MAX_BLIND_POLLS`` consecutive failures the
    gate opens and logs, because a stuck gate stalls the run while a refused submission does not.
    """

    def __init__(self, host: str, max_jobs: int, *, poll_secs: float = POLL_SECS,
                 enabled: bool = True) -> None:
        self.host = str(host)
        self.max_jobs = int(max_jobs)
        self.poll_secs = float(poll_secs)
        self.enabled = bool(enabled)
        self._count: int | None = None
        self._blind = 0
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    # -- the measurement ---------------------------------------------------- #
    def read_count(self) -> int | None:
        """Our current job (task) count, or None if the queue could not be read."""
        try:
            proc = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
                 self.host, "qstat -u ucestes | tail -n +3 | wc -l"],
                capture_output=True, text=True, timeout=120, check=False)
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode != 0:
            return None
        text = (proc.stdout or "").strip().splitlines()
        for line in reversed(text):
            line = line.strip()
            if line.isdigit():
                return int(line)
        return None

    # -- the loop ----------------------------------------------------------- #
    def _run(self) -> None:
        while not self._stop.is_set():
            n = self.read_count()
            with self._lock:
                if n is None:
                    self._blind += 1
                    if self._blind == MAX_BLIND_POLLS:
                        _LOG.warning(
                            "headroom gate has failed to read the queue %d times running; opening "
                            "the gate. A submission refused at the cap returns exit 25 and is "
                            "retried, so proceeding is safe; a stuck gate is not.", self._blind)
                else:
                    self._blind = 0
                    self._count = n
            self._stop.wait(self.poll_secs)

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="headroom", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # -- the decision (pure, unit-tested) ----------------------------------- #
    def is_open(self) -> bool:
        """True when a submission may proceed."""
        if not self.enabled:
            return True
        with self._lock:
            if self._blind >= MAX_BLIND_POLLS:
                return True          # bounded fail-open, announced above
            if self._count is None:
                return False         # nothing measured yet: wait rather than crowd
            return self._count < self.max_jobs

    def wait(self, label: str = "") -> None:
        """Block until the queue has room. Logs once when it starts waiting, once when it clears."""
        if self.is_open():
            return
        with self._lock:
            seen = self._count
        _LOG.info("[%s] waiting for queue headroom (%s of %d jobs held)",
                  label, "?" if seen is None else seen, self.max_jobs)
        t0 = time.time()
        while not self.is_open():
            if self._stop.wait(15.0):
                return
        _LOG.info("[%s] headroom after %.1f min; submitting", label, (time.time() - t0) / 60.0)


# --------------------------------------------------------------------------- #
# one line = 5 arms of already-written programs                               #
# --------------------------------------------------------------------------- #
def run_line(*, line: str, arms: tuple[str, ...], seeds: list[int], output_dir: str | Path,
             cluster_kwargs: dict[str, Any], assemble_kwargs: dict[str, Any],
             gate: HeadroomGate, resume: bool, dry_run: bool,
             campaign_root: str | Path = CAMPAIGN_ROOT) -> dict[str, Any]:
    """Send one line's campaign programs through the sealed window. No search, no provider call."""
    from run_authoring_variance import CORE_LINE, _slug
    from run_campaign_cluster import assemble_cluster_inputs, resolve_leg_override

    from src.utils.config import cfg_get, load_config

    root = Path(output_dir) / _slug(line)
    root.mkdir(parents=True, exist_ok=True)

    found_by_arm = collect_campaign_chain(line, arms, campaign_root)
    n_progs = sum(len(v) for v in found_by_arm.values())
    if not n_progs:
        return {"line": line, "chain": CHAIN, "root": str(root), "ok": False,
                "reason": "no_campaign_programs",
                "note": "the campaign archive holds no final-generation program for this line"}

    # The author block is resolved exactly as the authoring-variance driver resolves it. Nothing
    # here calls the provider — assemble_cluster_inputs builds the data, windows and agent config —
    # but the model identity still has to match, because it is written into every record.
    if line == CORE_LINE:
        llm_cfg = dict(cfg_get(load_config("campaign"), "llm", {}) or {})
        if not llm_cfg.get("model_snapshot"):
            raise SystemExit("campaign.yaml llm.model_snapshot missing — the author is unresolved")
        provider = str(llm_cfg.get("provider") or "anthropic")
    else:
        llm_cfg, provider, _suffix = resolve_leg_override(line, None)
    expected_model = str(llm_cfg["model_snapshot"])

    from run_authoring_variance import CANDIDATES, GENERATIONS

    assembled = assemble_cluster_inputs(
        arms=list(arms), seeds=list(seeds), output_dir=str(root),
        candidates=CANDIDATES, generations=GENERATIONS,
        # Chain 3 keeps the chain->seed convention (chain 1 -> 0, chain 2 -> 1), so nothing about
        # this run shares a training RNG stream with either fresh chain. The SEALED seeds are the
        # explicit list below and are identical across all three chains, which is what makes the
        # three comparable.
        search_seed=CHAIN - 1,
        llm_cfg=llm_cfg, provider=provider, resume=resume, **assemble_kwargs,
    )
    opts = assembled["opts"]
    test_leg_kwargs = assembled["test_leg_kwargs"]

    actual_model = str(opts.get("model") or "")
    if actual_model != expected_model:
        raise SystemExit(
            f"[{line} r{CHAIN}] AUTHOR MISMATCH: assembled model {actual_model!r} != expected "
            f"{expected_model!r}. Refusing to run — the archive would record the wrong author.")
    actual_steps = int(assembled["agent_cfg"].get("train_steps_per_candidate") or 0)
    _prereg = int(cfg_get(load_config("preregistration"), "train_steps_per_candidate", 0) or 0)
    if assemble_kwargs.get("train_steps") is None and _prereg and actual_steps != _prereg:
        raise SystemExit(
            f"[{line} r{CHAIN}] B* MISMATCH: assembled {actual_steps} != pre-registered {_prereg}. "
            f"Refusing to run a training budget that is not comparable with chains 1 and 2.")

    if dry_run:
        return {"line": line, "chain": CHAIN, "root": str(root), "dry_run": True,
                "programs": n_progs, "per_arm": {a: len(v) for a, v in found_by_arm.items()},
                "seeds": len(seeds), "trainings": n_progs * len(seeds),
                "model": actual_model, "train_steps": actual_steps,
                "windows": {"test": list(assembled["windows"][2])}}

    from src.cluster import build_cluster_run, run_test_leg
    from src.cluster.submit import prepare_remote, ssh_runner

    unit_kwargs = dict(cluster_kwargs)
    base_remote = str(unit_kwargs.pop("remote_root")).rstrip("/")
    # A REMOTE ROOT OF ITS OWN, for the same reason each authoring-variance unit has one: the pull
    # mirrors the whole remote outputs tree into the local archive, so a shared root would drag
    # other units' records in. ``av3`` is a sibling of ``av`` and cannot collide with any chain.
    remote_unit_root = f"{base_remote}/av3/{_slug(line)}"
    prepare_remote(remote_unit_root, [], ssh_runner(str(unit_kwargs.get("host") or "myriad")))
    # The tag namespaces every array name. Without it, ``<line>_r3_<arm>_test`` could collide with a
    # queued array from another run and be silently adopted (campaign.py:291-296).
    unit_tag = f"av{CHAIN}{_slug(line)}"

    # Search-only knobs are meaningless here and must not be passed: this run never searches.
    for k in ("search_h_rt", "search_threads", "search_pack", "search_poll_secs"):
        unit_kwargs.pop(k, None)

    run = build_cluster_run(
        local_archive_root=str(root), local_batch_root=str(root / "_batches"),
        remote_root=remote_unit_root, remote_outputs_root=f"{remote_unit_root}/outputs",
        batch_tag=unit_tag, **unit_kwargs,
    )

    # THE FIVE ARMS RUN AT ONCE, exactly as they do in the authoring-variance driver, which shares
    # one ClusterRun across its arm threads for the same reason. ``run_test_leg`` blocks until its
    # leg drains, so running the arms in sequence would put a finished arm's cores behind its
    # slowest sibling for hours. The headroom gate — not serialisation — is what paces this run, and
    # extra threads waiting at a closed gate cost nothing but fill the queue faster when it opens.
    tests: dict[str, Any] = {}
    errored: dict[str, str] = {}

    def _one_arm(arm: str) -> tuple[str, Any]:
        label = f"{_slug(line)}_r{CHAIN}_{arm}"
        gate.wait(label)
        return arm, run_test_leg(
            found_by_arm[arm], list(seeds), run, name=f"{label}_test",
            resume=resume, priority=0, interleave=True, **test_leg_kwargs,
        )

    with ThreadPoolExecutor(max_workers=max(1, len(found_by_arm))) as arm_pool:
        futs = {arm_pool.submit(_one_arm, a): a for a in sorted(found_by_arm)}
        for fut in as_completed(futs):
            arm = futs[fut]
            try:
                _a, t = fut.result()
                tests[arm] = t
            except Exception as exc:  # noqa: BLE001 — one arm must not kill the line
                _LOG.exception("[%s r%d %s] test leg failed", line, CHAIN, arm)
                errored[arm] = f"{type(exc).__name__}: {exc}"

    return {"line": line, "chain": CHAIN, "root": str(root), "ok": not errored,
            "programs": n_progs, "per_arm": {a: len(v) for a, v in found_by_arm.items()},
            "seeds": len(seeds), "trainings": n_progs * len(seeds),
            "errors": errored, "tests": tests}


# --------------------------------------------------------------------------- #
# selftest                                                                    #
# --------------------------------------------------------------------------- #
def selftest() -> int:
    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {name:<70} {detail}")

    from run_authoring_variance import (ALL_LINES, ARMS, CORE_LINE, FINAL_GEN, _parse_seeds,
                                        parse_test_label, _slug)

    print("campaign chain-3 selftest\n")

    # 1. Depth. Taking the campaign's g1 is only correct while the study's final generation IS g1.
    check("the study's final generation is g1", int(FINAL_GEN) == 1, f"FINAL_GEN={FINAL_GEN}")

    # 2. Every line resolves to a real campaign archive.
    missing = [ln for ln in ALL_LINES if not campaign_search_root(ln).is_dir()]
    check("every line has a campaign search archive", not missing, f"missing: {missing}")
    check("the core line maps to search/, not a leg",
          campaign_search_root(CORE_LINE).name == "search",
          str(campaign_search_root(CORE_LINE)))

    # 3. The programs are really there, and the count is the one we quoted before committing.
    per_line = {}
    total = 0
    for ln in ALL_LINES:
        got = collect_campaign_chain(ln, ARMS)
        n = sum(len(v) for v in got.values())
        per_line[ln] = n
        total += n
    check("221 campaign generation-1 programs are available", total == 221, f"got {total}")
    check("qwen3.5-9b is the known low-yield line", per_line.get("qwen3.5-9b", 0) <= 2,
          f"qwen3.5-9b={per_line.get('qwen3.5-9b')}")
    check("every other line yields at least 15", all(v >= 15 for k, v in per_line.items()
                                                     if k != "qwen3.5-9b"),
          ", ".join(f"{k}={v}" for k, v in per_line.items()))

    # 4. FALSIFIER: nothing from a generation other than the final one may be selected.
    sample = collect_campaign_chain("haiku-4.5", ("distributional",))
    gens = set()
    for _label, rec in sample.get("distributional", []):
        m = re.match(r"^distributional-g(\d+)-c\d+$", str(rec.get("candidate_id") or ""))
        if m:
            gens.add(int(m.group(1)))
    check("only generation 1 is selected (falsifier)", gens == {1}, f"generations seen: {sorted(gens)}")

    # 5. FALSIFIER: every selected record carries a real, executable-looking program.
    bad = [lbl for lbl, rec in sample.get("distributional", [])
           if len(str(rec.get("reward_source") or "")) < 100 or "def " not in str(rec.get("reward_source"))]
    check("every selected record carries a reward source (falsifier)", not bad, f"bad: {bad}")

    # 6. Labels round-trip through the study's own parser, so the analysis can read these records.
    labels = [lbl for lbl, _ in sample.get("distributional", [])]
    try:
        parsed = [parse_test_label(lbl) for lbl in labels]
        rt = all(a == "distributional" for a, _k in parsed)
    except Exception as exc:  # noqa: BLE001
        rt = False
        parsed = [f"{type(exc).__name__}: {exc}"]  # type: ignore[list-item]
    check("test labels parse as authoring-variance labels", rt and bool(labels), f"{labels[:2]}")

    # 7. THE ISOLATION FALSIFIER. The authoring-variance analysis globs '*/r*/test/*/*/record.json'.
    #    Chain 3 must be invisible to it, or an unbalanced, non-blind chain joins a balanced
    #    estimator silently. This asserts the output path cannot match that glob.
    from fnmatch import fnmatch
    probe = f"{DEFAULT_OUTPUT}/haiku_4_5/test/x/y/record.json".replace("\\", "/")
    av_glob_hit = fnmatch(probe, "outputs/authoring_variance/*/r*/test/*/*/record.json")
    check("chain-3 records cannot be swept up by the AV analysis glob (falsifier)",
          not av_glob_hit and not DEFAULT_OUTPUT.startswith("outputs/authoring_variance"), probe)

    # 8. Remote roots are siblings, never nested, so no pull can mirror one chain into another.
    r3 = f"/base/av3/{_slug('haiku-4.5')}"
    r1 = f"/base/av/{_slug('haiku-4.5')}_r1"
    check("chain-3 remote root is disjoint from the AV roots (falsifier)",
          not r3.startswith(r1) and not r1.startswith(r3) and r3 != r1, f"{r3} vs {r1}")

    # 9. The headroom gate is a pure decision and it actually blocks.
    g = HeadroomGate("nowhere", 900)
    g._count = 950
    check("gate CLOSED above the cap", not g.is_open(), "count=950 max=900")
    g._count = 800
    check("gate OPEN below the cap", g.is_open(), "count=800 max=900")
    g._count = None
    check("gate CLOSED before any measurement (falsifier)", not g.is_open(), "count=None")
    g._blind = MAX_BLIND_POLLS
    check("gate fails OPEN after repeated unreadable polls", g.is_open(), f"blind={g._blind}")
    g2 = HeadroomGate("nowhere", 900, enabled=False)
    check("gate disabled by --no-gate is always open", g2.is_open(), "enabled=False")

    # 10. Seeds resolve to the same sealed set the other chains use.
    seeds = _parse_seeds("0-29")
    check("seeds 0-29 resolve to 30 paired seeds", seeds == list(range(30)), f"n={len(seeds)}")

    # 11. The arithmetic we quoted.
    check("221 programs x 30 seeds = 6,630 sealed trainings", total * len(seeds) == 6630,
          f"{total} x {len(seeds)} = {total * len(seeds)}")

    # 12. This script must never be able to author. Nothing may reach the search or replay path.
    #     The needles are BUILT at runtime rather than written out, because a literal would appear
    #     in this very file and the check would match itself — which is exactly what the first
    #     version of this assertion did.
    src = Path(__file__).read_text(encoding="utf-8")
    body = src.split('"""', 2)[-1]          # skip the module docstring
    needles = ["run_search" + "_arm(", "install_author" + "_replay(", "_complete_with" + "_outage_tolerance"]
    hit = [n for n in needles if n in body]
    check("no search or authoring call anywhere in the body (falsifier)", not hit,
          f"chain 3 replays nothing and authors nothing; hits={hit}")

    print(f"\n{'ALL CHECKS PASSED' if ok else 'FAILURES ABOVE'}")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    from run_authoring_variance import ALL_LINES, ARMS

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lines", nargs="+", default=None,
                   help=f"Model lines. Default all: {', '.join(ALL_LINES)}")
    p.add_argument("--arms", nargs="+", default=None, help=f"Default all five: {', '.join(ARMS)}")
    p.add_argument("--seeds", default="0-29",
                   help="Sealed seeds. MUST match the authoring-variance run or the chains are not "
                        "comparable. Default 0-29.")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT,
                   help="Deliberately OUTSIDE outputs/authoring_variance so this non-blind, "
                        "unbalanced chain can never be pooled into the balanced estimator by a glob.")
    p.add_argument("--campaign-root", default=CAMPAIGN_ROOT)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Resolve and count; submit nothing.")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--max-parallel-lines", type=int, default=11,
                   help="Lines worked at once; each works its five arms in parallel, so the default "
                        "puts all 55 test legs in flight. The headroom gate, not this number, is "
                        "what paces submission.")
    p.add_argument("--max-jobs", type=int, default=900,
                   help="Submit only while we hold FEWER than this many jobs. Myriad's per-user cap "
                        "is 1000; 900 leaves the primary run a 100-job cushion it never has to "
                        "compete for.")
    p.add_argument("--no-gate", action="store_true",
                   help="Disable the headroom gate so chain 3 climbs the seed ladder ALONGSIDE "
                        "chains 1 and 2 instead of after them. This is the ratified default "
                        "(2026-08-22) run_test_leg sorts its taskfile SEED-MAJOR "
                        "(campaign.py:1293), so every chain passes through 5, 10, 15, 20, 25 and 30 "
                        "seeds holding THE SAME seeds. Running all three together therefore means "
                        "that at every moment there is a complete, balanced three-chain result. "
                        "Gating chain 3 to the end instead risks finishing with two chains and "
                        "nothing at all for the third. Total finish time is identical either way, "
                        "because the cluster's throughput is fixed; only the order changes.")
    p.add_argument("--poll-gate-secs", type=float, default=POLL_SECS)
    # Cluster shape. These MUST mirror the authoring-variance run: the whole point is that only the
    # authoring differs between the three chains.
    p.add_argument("--host", default="myriad")
    p.add_argument("--remote-root", default="~/Scratch/llmrp")
    p.add_argument("--gold-dir", default="~/ACFS/gold")
    p.add_argument("--pool", default="db")
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--pack", type=int, default=4,
                   help="4 renders 8 GB per job. Memory is the placement discriminator and 8 GB is "
                        "the measured sweet spot; pack 8 (16 GB) placed 0 of 15 canaries.")
    p.add_argument("--cores-per-training", type=int, default=1)
    p.add_argument("--specs-per-task", type=int, default=8)
    p.add_argument("--h-rt", default="48:00:00",
                   help="48:00:00 IS THE CEILING. 60, 72 and 90 hours are refused by policyjsv on "
                        "walltime alone.")
    p.add_argument("--chunk-tasks", type=int, default=1)
    p.add_argument("--apptainer-sif", default="~/python311.sif")
    p.add_argument("--poll-secs", type=float, default=180.0)
    p.add_argument("--exclude-hosts", default=None)
    p.add_argument("--train-steps", type=int, default=None,
                   help="SMOKE TEST ONLY. Bypasses the pre-registered B* assert and produces "
                        "records that are a wiring rehearsal, not results.")
    p.add_argument("--synthetic", action="store_true", help="Synthetic panel (dry-run only).")
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

    from run_authoring_variance import ALL_LINES, ARMS, _parse_seeds

    lines = tuple(args.lines) if args.lines else ALL_LINES
    unknown = [ln for ln in lines if ln not in ALL_LINES]
    if unknown:
        raise SystemExit(f"unknown line(s) {unknown}; known: {', '.join(ALL_LINES)}")
    arms = tuple(args.arms) if args.arms else ARMS
    bad = [a for a in arms if a not in ARMS]
    if bad:
        raise SystemExit(f"unknown arm(s) {bad}; expected {', '.join(ARMS)}")
    seeds = _parse_seeds(args.seeds)

    from src.utils.env import load_env
    load_env()

    counts = {ln: sum(len(v) for v in collect_campaign_chain(ln, arms, args.campaign_root).values())
              for ln in lines}
    total = sum(counts.values())
    print("CHAIN 3 = the campaign's own draw. No search, no provider call, no spend.")
    print(f"lines {len(lines)}  arms {len(arms)}  seeds {len(seeds)} ({seeds[0]}..{seeds[-1]})")
    print(f"campaign generation-1 programs {total}   sealed trainings {total * len(seeds):,}")
    for ln in lines:
        print(f"    {ln:<20} {counts[ln]:>3} programs")
    print(f"output {args.output_dir}   (deliberately outside outputs/authoring_variance)")
    if args.no_gate:
        print("HEADROOM GATE OFF — chain 3 climbs the seed ladder ALONGSIDE chains 1 and 2. The "
              "taskfile is sorted seed-major, so all three pass through 5, 10, 15, 20, 25 and 30 "
              "seeds holding the same seeds: at every moment there is a complete, balanced "
              "three-chain result. Total finish time is unchanged; only the order is.")
    else:
        print(f"headroom gate ON: submits only while we hold < {args.max_jobs} jobs")
    if args.train_steps is not None:
        print(f"!! SMOKE TEST: train_steps forced to {args.train_steps}. Not results.")
    print()

    assemble_kwargs = dict(
        synthetic=bool(args.synthetic),
        train_steps=(int(args.train_steps) if args.train_steps is not None else None),
        n_trials=0, embargo=0, pass_mode="B",
    )
    if not args.dry_run:
        from src.cluster.submit import expand_remote, remote_home, ssh_runner
        _home = remote_home(ssh_runner(args.host))
        args.remote_root = expand_remote(args.remote_root, _home)
        args.gold_dir = expand_remote(args.gold_dir, _home)
        args.apptainer_sif = expand_remote(args.apptainer_sif, _home)
        print(f"remote root {args.remote_root}\ngold        {args.gold_dir}\n"
              f"apptainer   {args.apptainer_sif}\n")

    cluster_kwargs = dict(
        remote_root=args.remote_root, gold_dir=args.gold_dir, host=args.host,
        pool_confirmatory=args.pool, pool_report_only=args.pool,
        pack=int(args.pack), chunk_tasks=int(args.chunk_tasks),
        specs_per_task=int(args.specs_per_task),
        cores_per_training=int(args.cores_per_training),
        h_rt=args.h_rt, apptainer_sif=args.apptainer_sif,
        poll_secs=float(args.poll_secs), min_pull_interval=120.0, device=args.device,
        exclude_hosts=([h.strip() for h in args.exclude_hosts.split(",")]
                       if args.exclude_hosts else None),
    )

    gate = HeadroomGate(args.host, int(args.max_jobs), poll_secs=float(args.poll_gate_secs),
                        enabled=not args.no_gate)
    if not args.dry_run:
        gate.start()

    results: list[dict[str, Any]] = []
    lock = threading.Lock()
    t0 = time.time()

    def one(line: str) -> dict[str, Any]:
        try:
            r = run_line(line=line, arms=arms, seeds=seeds, output_dir=args.output_dir,
                         cluster_kwargs=cluster_kwargs, assemble_kwargs=assemble_kwargs,
                         gate=gate, resume=bool(args.resume), dry_run=bool(args.dry_run),
                         campaign_root=args.campaign_root)
        except Exception as exc:  # noqa: BLE001 — one line must not kill the run
            _LOG.exception("[%s r%d] line failed", line, CHAIN)
            r = {"line": line, "chain": CHAIN, "ok": False,
                 "error": f"{type(exc).__name__}: {exc}"}
        with lock:
            results.append(r)
            print(f"[{len(results)}/{len(lines)}  {(time.time()-t0)/60:.1f} min] {line}: "
                  f"{'ok' if r.get('ok') or r.get('dry_run') else r.get('reason') or r.get('error')}",
                  flush=True)
        return r

    workers = max(1, min(int(args.max_parallel_lines), len(lines)))
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(one, ln) for ln in lines]
            for f in as_completed(futs):
                f.result()
    finally:
        gate.stop()

    summary = Path(args.output_dir) / "campaign_chain3_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps({
        "chain": CHAIN, "source": str(args.campaign_root),
        "not_blind": "This chain produced the dissertation's headline result. Report it as a "
                     "PLACEMENT of that draw among three, never as a third observation pooled "
                     "into a variance estimate.",
        "not_balanced": "221 of 275 programs; qwen3.5-9b contributes 1 of 25. Excluded from the "
                        "balanced nested estimator by construction.",
        "lines": list(lines), "arms": list(arms), "seeds": seeds,
        "programs": total, "trainings": total * len(seeds),
        "dry_run": bool(args.dry_run),
        "elapsed_min": round((time.time() - t0) / 60.0, 2),
        "units": results,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
