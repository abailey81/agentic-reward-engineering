"""Instrument (h) — the fed-delta SNR + responsiveness-attenuation exhibit (PREREG §2a(h), R76).

Report-only, archive-computed, validation/train data only, outside every confirmatory family.
Answers, mechanically, the two statistical objections a probabilist raises at the SQ1 null:

1. *"Were the fed deltas even resolvable signal?"* — the distribution of |Δ fed component| across
   consecutive candidates per arm-chain, against the PAIRED sampling-noise floor (common market
   path ⇒ paired differences are far better determined than marginal CIs suggest).
2. *"Isn't the responsiveness estimate attenuated by measurement noise?"* — the errors-in-variables
   attenuation factor λ_att = Var(signal)/(Var(signal)+Var(noise)); a disattenuated SQ1 sensitivity
   (NEVER the primary), plus the A5 fingerprint row input: the |Δ| split into high-SNR vs sub-noise
   subsets for SNR-conditioned responsiveness.

Noise-floor modes (honest about provenance, printed in the output):
* ``--paired-se <float>``   — an externally supplied paired SE per component (e.g. from a replay-
  based bootstrap; the exact mode once campaign replays exist).
* default CALIBRATED mode  — the 2026-07-11 measured anchors on the univ5 train window (EW-30,
  stationary block bootstrap): paired diff-SE ≈ 1e-4 (sibling books) … 8e-4 (structurally distant
  books). The default threshold uses the CONSERVATIVE (distant, 8e-4) floor for cvar components so
  "resolvable" is never overclaimed; left_tail_mass/robust_skew use scaled floors (their units
  differ) and are reported but flagged approximate.

Usage:
    python scripts/fed_delta_snr.py --archive outputs/prototype --arms distributional scalar_cvar5
    python scripts/fed_delta_snr.py --archive <campaign_root> --paired-se 0.00013 --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

COMPONENTS = ["cvar_01", "cvar_05", "cvar_10", "cvar_25", "left_tail_mass", "robust_skew"]
# Conservative paired noise floors (see module docstring). cvar_* share the measured return-scale
# floor; the two shape stats get order-of-magnitude floors and are flagged approximate.
CALIBRATED_FLOOR = {
    "cvar_01": 8e-4, "cvar_05": 8e-4, "cvar_10": 8e-4, "cvar_25": 8e-4,
    "left_tail_mass": 5e-3, "robust_skew": 2e-2,
}
APPROX_FLOOR = {"left_tail_mass", "robust_skew"}


def load_chain(archive: Path, arm: str) -> list[dict[str, Any]]:
    """The arm's candidates in chain order (generation, then candidate index) with tail_stats.

    2026-07-13 audit fixes: (a) the cluster mirror nests search records under ``<root>/search/``
    — try both layouts; (b) candidate order is now NUMERIC (``…-c10`` sorted after ``…-c2``): the
    old lexicographic sort scrambled consecutive-delta pairs in any generation with ≥10 candidates,
    biasing the |Δ| stimulus distribution the exhibit exists to measure."""
    import re

    root = archive / arm
    if not root.is_dir() and (archive / "search" / arm).is_dir():
        root = archive / "search" / arm
    rows: list[dict[str, Any]] = []
    for rec_path in sorted(root.glob(f"{arm}-*/record.json")):
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        ts = (rec.get("metrics") or {}).get("tail_stats")
        if ts:
            rows.append({"cid": rec["candidate_id"], "gen": int(rec.get("generation", 0)), "ts": ts})

    def _key(r: dict[str, Any]) -> tuple[int, int, str]:
        m = re.search(r"-g(\d+)-c(\d+)", str(r["cid"]))
        if m:
            return (int(m.group(1)), int(m.group(2)), "")
        return (r["gen"], 10**9, str(r["cid"]))  # non-standard ids: stable, after numbered ones
    return sorted(rows, key=_key)


def consecutive_deltas(chain: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """|Δ component| between consecutive candidates in the chain (the designer's actual stimuli)."""
    out: dict[str, list[float]] = {c: [] for c in COMPONENTS}
    for a, b in zip(chain, chain[1:]):
        for c in COMPONENTS:
            va, vb = a["ts"].get(c), b["ts"].get(c)
            if va is not None and vb is not None:
                out[c].append(abs(float(vb) - float(va)))
    return {c: np.asarray(v, dtype=float) for c, v in out.items()}


def analyze_arm(deltas: dict[str, np.ndarray], floors: dict[str, float]) -> dict[str, Any]:
    res: dict[str, Any] = {}
    for c, d in deltas.items():
        if d.size == 0:
            continue
        floor = floors[c]
        resolvable = d > 2.0 * floor
        var_total = float(np.var(d, ddof=1)) if d.size > 1 else 0.0
        var_noise = float(floor**2)  # noise variance of a paired delta at the floor SE
        lam = var_total / (var_total + var_noise) if (var_total + var_noise) > 0 else float("nan")
        res[c] = {
            "n_deltas": int(d.size),
            "median_abs_delta": float(np.median(d)),
            "q10": float(np.quantile(d, 0.10)), "q90": float(np.quantile(d, 0.90)),
            "paired_se_floor": floor,
            "approx_floor": c in APPROX_FLOOR,
            "resolvable_share": float(resolvable.mean()),
            "lambda_att": lam,  # attenuation multiplier on a correlation against these deltas
            "high_snr_idx": [int(i) for i in np.flatnonzero(resolvable)],  # for SNR-conditioned SQ1
        }
    return res


def main() -> int:
    p = argparse.ArgumentParser(description="Instrument (h): fed-delta SNR / attenuation exhibit.")
    p.add_argument("--archive", required=True, help="Archive root holding <arm>/<cid>/record.json.")
    p.add_argument("--arms", nargs="+", default=["distributional", "scalar_cvar5", "placebo_shuffled"])
    p.add_argument("--paired-se", type=float, default=None,
                   help="Override the cvar-component paired SE (e.g. replay-bootstrap exact value).")
    p.add_argument("--json", default=None, help="Also write the full result to this path.")
    args = p.parse_args()

    floors = dict(CALIBRATED_FLOOR)
    mode = "calibrated-conservative (2026-07-11 EW-30 block-bootstrap anchors; distant-book floor)"
    if args.paired_se is not None:
        for c in ("cvar_01", "cvar_05", "cvar_10", "cvar_25"):
            floors[c] = float(args.paired_se)
        mode = f"externally supplied paired SE = {args.paired_se:g}"

    archive = Path(args.archive)
    out: dict[str, Any] = {"mode": mode, "arms": {}}
    print(f"[fed_delta_snr] noise-floor mode: {mode}")
    for arm in args.arms:
        chain = load_chain(archive, arm)
        if len(chain) < 2:
            print(f"  {arm}: <2 candidates with tail_stats — skipped")
            continue
        res = analyze_arm(consecutive_deltas(chain), floors)
        out["arms"][arm] = res
        print(f"  {arm} ({len(chain)} candidates):")
        for c in COMPONENTS:
            if c not in res:
                continue
            r = res[c]
            approx = " (floor approx.)" if r["approx_floor"] else ""
            print(f"    {c:15} n={r['n_deltas']:>3}  median|d|={r['median_abs_delta']:.5f}  "
                  f"resolvable>{2*r['paired_se_floor']:.4g}: {r['resolvable_share']:5.1%}  "
                  f"lambda_att={r['lambda_att']:.3f}{approx}")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[fed_delta_snr] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
