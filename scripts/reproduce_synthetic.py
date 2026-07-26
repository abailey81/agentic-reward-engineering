#!/usr/bin/env python
"""Golden SYNTHETIC reproduction — proves the FULL machinery runs end-to-end, keyless + deterministic.

    git clone ... && pip install -r requirements.lock
    python scripts/reproduce_synthetic.py --check        # or: make reproduce

On a FIXED seed this generates the shape-identical synthetic panel, then runs the REAL campaign loop
(author[stub] -> AST-gate + sandbox-validate -> train the fixed SB3-SAC -> measure the realized-return
distribution -> score held-out fitness -> archive -> select the winner) and asserts the resulting summary
MATCHES the committed golden (`tests/golden/synthetic_summary.json`). It converts "the method is
verifiable end-to-end" from a README claim into a one-command, CI-guarded FACT — with **no API key and no
licensed Refinitiv data** (the keyless `StubDesignerTransport` stands in for the LLM author; swapping it
for the real transport is the only change to the headline run).

Determinism model (honest): the synthetic panel and the stub-authored reward SOURCES are **bit-exact**
(hashed below); the trained `val_fitness` / tail metrics are bit-exact on a fixed device (CPU, fixed
torch) and only *near*-exact across a CPU<->GPU / different-torch boundary (FP non-associativity), so the
check asserts hash-equality on the exact parts and a tight relative tolerance on the trained numbers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SEED = 12345
ARM = "distributional"          # one canonical tail-fed arm exercises author->gate->train->score->select
CANDIDATES = 4                  # 4 stub archetypes: return-only, turnover, rolling-variance, rolling-CVaR
TRAIN_STEPS = 1500              # > learning_starts (1000) so the SAC actually takes gradient steps
N_ASSETS, N_DAYS, PANEL_SEED = 30, 600, 0
GOLDEN = REPO / "tests" / "golden" / "synthetic_summary.json"
RTOL = 1e-4                     # cross-device tolerance on trained numbers (hashes are exact)


def _sha16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def run_reproduction(out_dir: Path) -> dict:
    """Run the real keyless loop on the seeded synthetic panel; return the summary dict."""
    from src.utils.preload import preload
    from src.utils.seeding import set_global_seed
    from src.utils.config import load_config
    from src.data.synthetic import make_synthetic_panel
    from src.env.runner import make_env_builder
    from src.agents.trainer import make_agent_trainer
    from src.feedback.measurement import ReturnDistribution
    from src.llm.prompts import build_prompt_set
    from src.llm.client import LLMClient
    from src.llm.stub_designer import StubDesignerTransport
    from src.llm.loop import run_loop
    from src.selection.fitness import held_out_fitness
    from src.io.results import load_all

    preload(strict=False)                                  # pyarrow before torch; synthetic needs no gold
    set_global_seed(SEED, deterministic_torch=True)
    env_cfg = load_config("environment")
    lookback = int(env_cfg["state"]["lookback_days"])

    panel = make_synthetic_panel(n_assets=N_ASSETS, n_days=N_DAYS, seed=PANEL_SEED)
    panel_sha = hashlib.sha256(panel.returns.tobytes()).hexdigest()[:16]
    # Disjoint, PURGED splits (R18): val must start at train_end + max(embargo, lookback).
    embargo, train_end = 21, 380
    purge = max(embargo, lookback)
    train_window, val_window = (lookback, train_end), (train_end + purge, panel.T)
    env_builder = make_env_builder(
        panel, env_cfg, train_window, val_window, embargo=embargo, lookback=lookback)
    agent_cfg = {"train_steps_per_candidate": TRAIN_STEPS, "buffer_size": TRAIN_STEPS,
                 "batch_size": 256, "learning_rate": 3e-4, "gamma": 0.99}
    agent_trainer = make_agent_trainer(agent_cfg, SEED)
    prompts = build_prompt_set(env_cfg, panel.N)

    transport = StubDesignerTransport(seed=SEED)
    llm = LLMClient({"model": "stub-designer"}, transport=transport)
    loop_cfg = {
        "generations": 1, "candidates_per_gen": CANDIDATES, "budget": CANDIDATES,
        "seed": SEED, "n_trials": CANDIDATES, "model": "stub-designer",
        "run_prefix": f"repro-{ARM}", "fold": 0, "env_fingerprint": f"repro:{ARM}",
        "prompts": {"system": prompts.system, "initial": prompts.initial},
        "diversity_prompt_variation": False, "legible_render": False, "extra_record_fields": {},
    }
    run_loop(ARM, env_builder, llm, agent_trainer, ReturnDistribution, held_out_fitness,
             loop_cfg, out_dir / ARM)

    recs = []
    for rec in load_all(out_dir / ARM):
        m = rec.get("metrics") or {}
        ts = m.get("tail_stats") or {}
        recs.append({
            "candidate_id": rec.get("candidate_id"),
            "generation": rec.get("generation"),
            "reward_sha": (rec.get("reward_source_hash") or _sha16(rec.get("reward_source", "")))[:16],
            "val_fitness": float(m["val_fitness"]),
            "cvar_05": float(ts.get("cvar_05", 0.0)),
        })
    recs.sort(key=lambda r: str(r["candidate_id"]))
    winner = max(recs, key=lambda r: r["val_fitness"]) if recs else None
    return {
        "schema": 1, "seed": SEED, "arm": ARM, "candidates": CANDIDATES, "train_steps": TRAIN_STEPS,
        "panel": {"n_assets": N_ASSETS, "n_days": N_DAYS, "seed": PANEL_SEED, "sha16": panel_sha},
        "n_records": len(recs), "records": recs,
        "winner_candidate_id": winner["candidate_id"] if winner else None,
    }


def _compare(got: dict, golden: dict) -> list[str]:
    """Return a list of human-readable diffs (empty => reproduction matches the golden)."""
    diffs: list[str] = []
    # exact structural / bit-exact fields
    for k in ("seed", "arm", "candidates", "train_steps", "n_records", "winner_candidate_id"):
        if got.get(k) != golden.get(k):
            diffs.append(f"{k}: got {got.get(k)!r} != golden {golden.get(k)!r}")
    if got.get("panel", {}).get("sha16") != golden.get("panel", {}).get("sha16"):
        diffs.append(f"panel.sha16: got {got['panel']['sha16']} != golden {golden['panel']['sha16']} "
                     "(synthetic panel not bit-identical — a generator or numpy change)")
    g_recs = {str(r["candidate_id"]): r for r in golden.get("records", [])}
    for r in got.get("records", []):
        cid = str(r["candidate_id"])
        gr = g_recs.get(cid)
        if gr is None:
            diffs.append(f"record {cid}: present in run, absent in golden")
            continue
        if r["reward_sha"] != gr["reward_sha"]:
            diffs.append(f"record {cid} reward_sha: {r['reward_sha']} != {gr['reward_sha']} "
                         "(stub author not bit-identical)")
        for num in ("val_fitness", "cvar_05"):
            a, b = float(r[num]), float(gr[num])
            if not math.isclose(a, b, rel_tol=RTOL, abs_tol=1e-9):
                diffs.append(f"record {cid} {num}: {a:.10g} != {b:.10g} (rel_tol {RTOL})")
    return diffs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Golden synthetic reproduction (keyless, deterministic).")
    ap.add_argument("--check", action="store_true", help="run + assert the summary matches the golden")
    ap.add_argument("--update-golden", action="store_true", help="run + (re)write the committed golden")
    ap.add_argument("--out", default=None, help="output dir (default: a temp dir, auto-cleaned)")
    args = ap.parse_args(argv)

    tmp = None
    out_dir = Path(args.out) if args.out else Path(tmp := tempfile.mkdtemp(prefix="repro_synth_"))
    try:
        summary = run_reproduction(out_dir)
    finally:
        if tmp:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    if args.update_golden:
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[reproduce_synthetic] wrote golden -> {GOLDEN.relative_to(REPO)}")
        return 0

    if args.check:
        if not GOLDEN.is_file():
            print(f"[reproduce_synthetic] FAIL: no golden at {GOLDEN} (run --update-golden once)")
            return 1
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        diffs = _compare(summary, golden)
        if diffs:
            print("[reproduce_synthetic] MISMATCH vs golden:")
            for d in diffs:
                print("  -", d)
            return 1
        print(f"[reproduce_synthetic] OK: {summary['n_records']} records reproduce the golden "
              f"(panel {summary['panel']['sha16']}, winner {summary['winner_candidate_id']}).")
        return 0

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
