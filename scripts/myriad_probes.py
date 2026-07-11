"""P1 + P4 Myriad probes — packing factor F + cross-node determinism.

Submitted ALONGSIDE P0/P6 rather than after them: queue latency dominates wall-clock on a saturated
cluster, so every probe should be aging in the queue at once (fix-and-resubmit on a path bug costs
minutes; serialized queue waits cost days). The standalone sustained-C probe (old P2) is DROPPED as
redundant: the live arrays' epilogue ledgers + qacct timestamps ARE the placement experiment.

* **P1 — packing factor F.** Three one-task batches, pack ∈ {2, 3, 5}: each task trains `pack`
  authored-winner candidates CONCURRENTLY on one V100 (``run_one --pack N`` — G0 proved cgroup
  device-exclusivity makes this safe). Per-training rate = train_steps / record wall_clock;
  F(pack) = pack x rate_packed / rate_solo against the measured G1 solo anchor (102.2 steps/s).
  Cores = pack x 1 (the rehearsal finding: a training uses <1 core, and CORES gate placement).
* **P4 — cross-node determinism.** One two-task array: the IDENTICAL (reward, seed=777, 50k) spec
  under two candidate_ids. SGE typically lands the tasks on different nodes (checked at pull from
  the epilogue host field; resubmit one task if co-located). PASS = byte-equal ``val_returns`` +
  equal ``val_fitness`` — the replay-from-archive determinism claim, live on the cluster.

Rails: the campaign's own ``assemble_cluster_inputs`` + ``_search_spec`` + certified on-node
``train_candidate`` (parity — nothing reimplemented); real gold; train/val only; keyless.

Usage:
    python scripts/myriad_probes.py --submit | --build-only | --pull
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: `scripts.` + `src.` importable

STEPS = 50_000
PACKS = [2, 3, 5]
DET_SEED = 777
SOLO_STEPS_PER_SEC = 102.2  # G1 anchor (job 764154, V100-PCIE-32GB, pack=1)
REWARD_PATH = "outputs/prototype/distributional/distributional-g3-c0/reward.py"


def _opts(local_out: str) -> dict[str, Any]:
    from scripts.run_campaign_cluster import assemble_cluster_inputs

    inputs = assemble_cluster_inputs(
        arms=["distributional"], seeds=[0], output_dir=local_out, synthetic=False,
        train_steps=STEPS, n_trials=30, candidates=1, generations=1, search_seed=0,
        embargo=21, pass_mode="A", provider="stub", llm_cfg=None, resume=False,
    )
    return inputs["opts"]


def build_batches(remote_root: str, gold_dir: str, local_out: str, *, pool: str,
                  include_det: bool = True, cores_total: int | None = None,
                  suffix: str = "") -> list[Path]:
    from src.cluster.campaign import _search_spec
    from src.cluster.jobscript import render_jobscript, write_jobscript
    from src.cluster.spec_io import write_specs

    source = Path(REWARD_PATH).read_text(encoding="utf-8")
    opts = _opts(local_out)
    archive = f"{remote_root}/outputs/search"
    dirs: list[Path] = []

    def spec(cid: str, seed: int) -> dict[str, Any]:
        o = dict(opts)
        o["seed"] = int(seed)
        return _search_spec(cid.split("-")[0], source, cid, o, f"probe ({REWARD_PATH})", 0, archive)

    for pack in PACKS:  # P1: one task = `pack` concurrent trainings
        name = f"p1pack{pack}{suffix}"
        batch = Path(local_out) / "batches" / name
        task = [spec(f"{name}-k{i}", 200 + i) for i in range(pack)]
        write_specs([task], batch)
        # cores_total discriminates CPU-starvation vs GPU-time-slicing (2026-07-11: the default
        # cores=pack ladder CONFOUNDS them — F(2)@2cores ≈ 0.7; if F(2)@4cores recovers, the CPU
        # side was the binder and the campaign packs with cores=2×pack; if not, it's GPU slicing
        # and MPS is the lever).
        js = render_jobscript(name, 1, remote_root, gold_dir, pool=pool, pack=pack,
                              cores=(cores_total or pack), apptainer_sif="$HOME/python311.sif")
        write_jobscript(js, batch / f"{name}.sh")
        dirs.append(batch)

    if include_det:
        name = f"p4det{suffix}"  # P4: two tasks, identical (reward, seed), distinct cids
        batch = Path(local_out) / "batches" / name
        write_specs([spec(f"{name}-t1", DET_SEED), spec(f"{name}-t2", DET_SEED)], batch)
        js = render_jobscript(name, 2, remote_root, gold_dir, pool=pool, pack=1,
                              cores=1, apptainer_sif="$HOME/python311.sif")
        write_jobscript(js, batch / f"{name}.sh")
        dirs.append(batch)

    print(f"[probes] built {len(dirs)} batches: {[d.name for d in dirs]}")
    return dirs


def submit(remote_root: str, gold_dir: str, local_out: str, *, host: str, pool: str,
           include_det: bool = True, cores_total: int | None = None, suffix: str = "") -> None:
    from src.cluster.submit import prepare_remote, push_batch, qsub, ssh_runner

    runner = ssh_runner(host)
    dirs = build_batches(remote_root, gold_dir, local_out, pool=pool, include_det=include_det,
                         cores_total=cores_total, suffix=suffix)
    prepare_remote(remote_root, [d.name for d in dirs], runner)
    for d in dirs:
        push_batch(d, f"{remote_root.rstrip('/')}/specs", host=host)
        job = qsub(f"{remote_root.rstrip('/')}/specs/{d.name}/{d.name}.sh", runner)
        print(f"[probes] submitted {d.name} as job {job}")


def pull_and_summarize(remote_root: str, local_out: str, *, host: str) -> None:
    from src.cluster.poll import pull_archive

    local_root = Path(local_out)
    pull_archive(f"{remote_root.rstrip('/')}/outputs", local_root, host=host)
    recs: dict[str, dict[str, Any]] = {}
    for rec_path in sorted(local_root.glob("search/**/record.json")):
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        recs[str(rec.get("candidate_id", ""))] = rec

    print("== P1 packing factor (solo anchor 102.2 steps/s) ==")
    for pack in PACKS:
        walls = [recs[c]["wall_clock"] for c in recs if c.startswith(f"p1pack{pack}-") and recs[c].get("wall_clock")]
        if not walls:
            print(f"  pack={pack}: no records yet")
            continue
        rate = STEPS / (sum(walls) / len(walls))  # per-training steps/s under the pack
        f_factor = pack * rate / SOLO_STEPS_PER_SEC
        print(f"  pack={pack}: per-training {rate:6.1f} steps/s  ->  F = {f_factor:.2f}")

    print("== P4 cross-node determinism ==")
    t1, t2 = recs.get("p4det-t1"), recs.get("p4det-t2")
    if t1 and t2:
        vf_eq = t1["metrics"]["val_fitness"] == t2["metrics"]["val_fitness"]
        vr_eq = t1["metrics"]["val_returns"] == t2["metrics"]["val_returns"]
        print(f"  val_fitness equal: {vf_eq} ({t1['metrics']['val_fitness']} vs {t2['metrics']['val_fitness']})")
        print(f"  val_returns byte-equal: {vr_eq}")
        print(f"  hosts: {t1.get('env_fingerprint')} | {t2.get('env_fingerprint')}")
        print("  VERDICT:", "DETERMINISM HOLDS" if (vf_eq and vr_eq) else "DRIFT — investigate before campaign")
    else:
        print(f"  records present: t1={bool(t1)} t2={bool(t2)} (not both done yet)")


def main() -> int:
    p = argparse.ArgumentParser(description="P1 packing + P4 determinism probes on Myriad.")
    p.add_argument("--submit", action="store_true")
    p.add_argument("--build-only", action="store_true")
    p.add_argument("--pull", action="store_true")
    p.add_argument("--host", default="myriad")
    p.add_argument("--remote-root", default="~/Scratch/probes")
    p.add_argument("--gold-dir", default="/acfs/users/ucestes/gold")
    p.add_argument("--output-dir", default="outputs/probes")
    p.add_argument("--pool", default="EF")
    p.add_argument("--packs", default=None,
                   help="Comma list of pack sizes to probe (default: the module PACKS). "
                        "e.g. --packs 8 extends the F ceiling once 2/3/5 are queued.")
    p.add_argument("--cores-total", type=int, default=None,
                   help="TOTAL cores for each pack task (default: pack x 1). Use with --name-suffix "
                        "to disambiguate, e.g. --packs 2 --cores-total 4 --name-suffix c4.")
    p.add_argument("--name-suffix", default="",
                   help="Suffix for the batch/candidate names (a re-probe must not collide).")
    p.add_argument("--only-det", action="store_true",
                   help="Submit ONLY a P4 determinism pair (evidence replicate; use with "
                        "--name-suffix and optionally --det-seed).")
    p.add_argument("--det-seed", type=int, default=None,
                   help="Override the determinism pair's seed (a replicate pair at a new seed).")
    global PACKS, DET_SEED
    args = p.parse_args()
    if args.det_seed is not None:
        DET_SEED = int(args.det_seed)
    if args.packs:
        PACKS = [int(x) for x in str(args.packs).split(",") if x.strip()]

    from src.utils.preload import preload
    preload(strict=True)

    remote_root = args.remote_root
    if remote_root.startswith("~"):
        if args.build_only:
            from src.cluster.submit import expand_remote
            remote_root = expand_remote(remote_root, "/home/USER")  # documented stub (no ssh)
        else:
            from src.cluster.submit import expand_remote, remote_home, ssh_runner
            remote_root = expand_remote(remote_root, remote_home(ssh_runner(args.host)))

    det = args.packs is None  # a custom --packs run extends P1 only; never resubmit the P4 pair
    if args.only_det:
        PACKS, det = [], True
        if not args.name_suffix:
            p.error("--only-det requires --name-suffix (a replicate must not collide with p4det)")
    extra = {"cores_total": args.cores_total, "suffix": args.name_suffix}
    if args.build_only:
        build_batches(remote_root, args.gold_dir, args.output_dir, pool=args.pool,
                      include_det=det, **extra)
        return 0
    if args.submit:
        submit(remote_root, args.gold_dir, args.output_dir, host=args.host, pool=args.pool,
               include_det=det, **extra)
        return 0
    if args.pull:
        pull_and_summarize(remote_root, args.output_dir, host=args.host)
        return 0
    p.error("pass one of --submit / --build-only / --pull")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
