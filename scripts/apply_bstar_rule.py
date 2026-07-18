"""Apply the PRE-COMMITTED extended B* decision rule (EVIDENCE_LEDGER_2026-07-12.md, written
2026-07-13 BEFORE the curve data existed) to the completed 30-point same-protocol curve.

THE RULE (verbatim semantics): for each winner separately, on the Myriad val-DSR rows with
CRN-paired seeds: if the paired mean ascent  mean_s[eval(b_hi) − eval(200k)] > 2 × SE_s(paired
diff)  for ANY b_hi ∈ {400k, 800k, 1.6M}, a B* amendment PROPOSAL goes for review pre-freeze.

Pairing is BY SEED FIELD read from each record (never from sorted display order). SE uses the
sample sd (ddof=1) over the n=3 paired diffs. Output: the full grid, every paired contrast,
and the per-winner verdict — written to outputs/tables/bstar_rule_verdict.json for the record.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "outputs" / "p6ladder" / "search"
BUDGETS = [100_000, 200_000, 400_000, 800_000, 1_600_000]
B_BASE = 200_000
B_HIGHS = [400_000, 800_000, 1_600_000]
WINNERS = {"p6dist": "distributional winner", "p6scal": "scalar winner"}


def load_grid() -> dict[str, dict[int, dict[int, float]]]:
    """{winner: {budget: {seed: val_fitness}}} — seed read from the RECORD, never inferred."""
    grid: dict[str, dict[int, dict[int, float]]] = {w: {b: {} for b in BUDGETS} for w in WINNERS}
    for rec_path in ARCHIVE.glob("*/*/record.json"):
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        cid = str(rec.get("candidate_id", ""))
        for w in WINNERS:
            if cid.startswith(w + "-b"):
                budget = int(cid.split("-b")[1].split("-s")[0])
                seed = int(rec["seed"])
                vf = float(rec["metrics"]["val_fitness"])
                if seed in grid[w].get(budget, {}):
                    raise SystemExit(f"DUPLICATE cell {w} b={budget} s={seed} — archive corrupt?")
                grid[w].setdefault(budget, {})[seed] = vf
    for w, per_b in grid.items():
        for b in BUDGETS:
            if sorted(per_b.get(b, {})) != [0, 1, 2]:
                raise SystemExit(f"INCOMPLETE cell {w} b={b}: seeds {sorted(per_b.get(b, {}))} "
                                 f"— the rule requires the complete 3-seed rung")
    return grid


def main() -> int:
    grid = load_grid()
    out: dict[str, object] = {"rule": "mean_s[eval(b_hi)-eval(200k)] > 2*SE_s, per winner, "
                                      "any b_hi in {400k,800k,1.6M}; n=3 CRN-paired seeds",
                              "winners": {}}
    print("=" * 76)
    print("PRE-COMMITTED B* RULE — applied", flush=True)
    fired_any = False
    for w, label in WINNERS.items():
        print(f"\n{label} ({w}):")
        print(f"  {'budget':>10} " + " ".join(f"s{s}={grid[w][b][s]:+.4f}"
                                              for b in [B_BASE] for s in [0, 1, 2]))
        wres = {}
        for bh in B_HIGHS:
            diffs = [grid[w][bh][s] - grid[w][B_BASE][s] for s in (0, 1, 2)]
            mean = sum(diffs) / 3
            sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / 2)  # ddof=1
            se = sd / math.sqrt(3)
            fires = mean > 2 * se
            fired_any |= fires
            wres[str(bh)] = {"paired_diffs_s0_s1_s2": diffs, "mean": mean, "se": se,
                             "ratio_mean_over_se": (mean / se if se > 0 else float("inf")),
                             "fires": fires}
            print(f"  {bh:>10,} vs 200k: diffs = "
                  + ", ".join(f"{d:+.4f}" for d in diffs)
                  + f" | mean {mean:+.4f}  SE {se:.4f}  mean/SE {mean/se if se>0 else float('inf'):.2f}"
                  + f"  -> {'FIRES' if fires else 'no'}")
        out["winners"][w] = wres  # type: ignore[index]
    out["verdict"] = ("RULE FIRES — a B* amendment PROPOSAL goes for review pre-freeze"
                      if fired_any else
                      "no CI-separated ascent — B*=200k stands; claim 8 returns to A")
    print("\nVERDICT:", out["verdict"])
    dest = ROOT / "outputs" / "tables" / "bstar_rule_verdict.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[rule] written {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
