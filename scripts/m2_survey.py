"""M2 — the cross-LLM numeracy + responsiveness survey FLEET RUNNER (protocol v1, 2026-07-12).

Executes survey protocol v1 exactly: three pre-registered probe
families per model — P-A numeracy discrimination (difficulty anchored to the EMPIRICAL fed-delta
|Δ| quantiles of an archived search chain), P-B ordering (4 archived candidates' 6-vectors),
P-C responsiveness-in-code (the Stage-1 reflection block schema VERBATIM via
``src.feedback.schema.build_block`` with controlled fed-vector deltas between turns, plus
placebo-shuffled control turns). Descriptive, OUTSIDE every confirmatory family.

Design contract (mirrors the campaign's engineering discipline):
- **Deterministic items**: one seeded generation, identical across models (paired cross-model
  comparison); the item set is archived once (``items.json``) with answer keys.
- **Same plumbing**: every call goes through ``src.llm.client.build_transport`` — nothing
  re-implemented; temperature 0 requested (providers that reject it fall back, disclosed in the
  per-call record).
- **Archival + resume**: one JSONL row per (model, item) under ``<out>/<model>/responses.jsonl``,
  appended atomically; a re-run skips archived item ids (idempotent; no double spend).
- **Spend cap**: ``--max-calls`` is a HARD ceiling across the whole invocation (0 = forbid).
- **Self-consistency**: a deterministic 20% sample of P-A/P-B is re-asked once (rep=1 rows).

The runner only COLLECTS (archives prompts + completions + parsed answers where trivially
parseable); the pre-named analyses (§4: psychophysics curve, responsiveness score, the Spearman
headline, the A5 cross-read, the reject table) run post-bank on the archive. A prototype-anchored
run is DIRECTIONAL-ONLY (protocol §5) and never enters the dissertation.

Usage::

    # harness validation, zero spend (stub transport):
    python scripts/m2_survey.py --archive outputs/prototype --models stub --out outputs/m2_dryrun
    # the real survey (post-bank; models pinned in config/m2_models.yaml):
    python scripts/m2_survey.py --archive <campaign>/search --models config/m2_models.yaml \
        --out outputs/m2_survey --max-calls 900
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: The six frozen fed components (schema order) — cvar_05 is the headline difficulty axis.
COMPONENTS = ["cvar_05", "cvar_10", "cvar_25", "cvar_01", "left_tail_mass", "robust_skew"]
#: Protocol §3 P-A: difficulty rungs = these |Δ| quantiles of the empirical fed-delta distribution.
RUNG_QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
#: P-A format variants (M3 legibility lever). "rank" is P-B territory; CI-annotated feeds the A5 read.
FORMATS = ["raw", "basis_points", "ci_annotated"]
#: A realistic base tail vector (prototype-magnitude; the ANCHOR values are archive-derived).
BASE_TAIL = {"cvar_05": -0.0305, "cvar_10": -0.0242, "cvar_25": -0.0154,
             "cvar_01": -0.0548, "left_tail_mass": 0.048, "robust_skew": 0.21}


# --------------------------------------------------------------------------- #
# Empirical anchoring (instrument (h) reuse: chain deltas -> |Δ| quantiles)     #
# --------------------------------------------------------------------------- #
def empirical_delta_rungs(archive_root: str | Path, component: str = "cvar_05") -> list[float]:
    """|Δ| difficulty rungs from the archived search chains' successive tail vectors.

    Mirrors ``fed_delta_snr``'s chain construction (records in (gen, cand) numeric order per
    arm; successive absolute deltas of ``component``). Falls back to calibrated synthetic rungs
    (the measured prototype floor scale) when the archive has <8 usable deltas — loudly."""
    import re

    import numpy as np

    deltas: list[float] = []
    root = Path(archive_root)
    for arm_dir in sorted(p for p in root.glob("*") if p.is_dir()):
        rows: list[tuple[int, int, float]] = []
        for rec_path in arm_dir.glob("*/record.json"):
            try:
                rec = json.loads(rec_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            ts = (rec.get("metrics") or {}).get("tail_stats") or {}
            if component not in ts:
                continue
            m = re.search(r"-g(\d+)-c(\d+)", str(rec.get("candidate_id", "")))
            if not m:
                continue
            rows.append((int(m.group(1)), int(m.group(2)), float(ts[component])))
        rows.sort()
        vals = [v for _, _, v in rows]
        deltas.extend(abs(b - a) for a, b in zip(vals, vals[1:]))
    if len(deltas) >= 8:
        qs = [float(np.quantile(deltas, q)) for q in RUNG_QUANTILES]
        return [max(q, 1e-6) for q in qs]
    print(f"[m2] WARNING: only {len(deltas)} archived deltas for {component} — "
          "using calibrated synthetic rungs (harness-validation mode; directional only)", flush=True)
    return [0.0001, 0.0004, 0.0010, 0.0026, 0.0060]


# --------------------------------------------------------------------------- #
# Deterministic item generation (identical across models; archived with keys)   #
# --------------------------------------------------------------------------- #
def _fmt(value: float, fmt: str, rung_delta: float) -> str:
    if fmt == "raw":
        return f"{value:.4f}"
    if fmt == "basis_points":
        return f"{value * 10_000:.1f} bp"
    # ci_annotated: the paired sampling floor as a ±CI (instrument (h)'s calibrated floor scale)
    return f"{value:.4f} (95% CI ±{max(rung_delta * 0.5, 1e-4):.4f})"


def build_items(archive_root: str | Path, *, seed: int = 20260712) -> list[dict[str, Any]]:
    """The full deterministic item set: P-A (~40) + P-B (10) + P-C (8 + 2 placebo) per protocol §3."""
    import numpy as np

    rng = np.random.default_rng(seed)
    rungs = empirical_delta_rungs(archive_root)
    items: list[dict[str, Any]] = []

    # ---- P-A: pairwise discrimination, 5 rungs x 3 formats x 2 base draws = 30 core items,
    # + 10 extra raw-format items across rungs (protocol "~40 items/model") ------------------- #
    def _pa_item(idx: int, rung_i: int, fmt: str) -> dict[str, Any]:
        base = float(BASE_TAIL["cvar_05"] + rng.normal(0, 0.004))
        delta = rungs[rung_i]
        worse_first = bool(rng.integers(0, 2))
        worse, better = base - delta, base
        a, b = (worse, better) if worse_first else (better, worse)
        return {
            "id": f"PA-{idx:03d}", "family": "P-A", "rung": rung_i, "format": fmt,
            "delta": delta,
            "prompt": (
                "Two candidate reward functions produced these realized CVaR-5% values on the "
                f"same market path.\nCandidate A: {_fmt(a, fmt, delta)}\n"
                f"Candidate B: {_fmt(b, fmt, delta)}\n"
                "Which candidate indicates WORSE tail risk? Reply with exactly one letter: A or B."
            ),
            "answer": "A" if worse_first else "B",
        }

    idx = 0
    for rung_i in range(len(rungs)):
        for fmt in FORMATS:
            for _ in range(2):
                items.append(_pa_item(idx, rung_i, fmt))
                idx += 1
    for rung_i in (0, 1, 2, 3, 4, 0, 2, 4, 1, 3):
        items.append(_pa_item(idx, rung_i, "raw"))
        idx += 1

    # ---- P-B: rank 4 candidates by the stated criterion from their 6-vectors (10 items) ----- #
    for j in range(10):
        vecs = []
        for k in range(4):
            vecs.append({c: float(BASE_TAIL[c] + rng.normal(0, abs(BASE_TAIL[c]) * 0.15 + 1e-4))
                         for c in COMPONENTS})
        order = sorted(range(4), key=lambda k: vecs[k]["cvar_05"])  # most negative = worst first
        lines = []
        for k, v in enumerate(vecs):
            lines.append(f"Candidate {chr(65 + k)}: " +
                         ", ".join(f"{c}={v[c]:.4f}" for c in COMPONENTS))
        items.append({
            "id": f"PB-{j:03d}", "family": "P-B",
            "prompt": ("Rank these four candidates from WORST tail risk to BEST, judging by "
                       "CVaR-5% (cvar_05; more negative = worse).\n" + "\n".join(lines) +
                       "\nReply with exactly four letters separated by '>' (worst first), "
                       "e.g. C>A>D>B."),
            "answer": ">".join(chr(65 + k) for k in order),
        })

    # ---- P-C: responsiveness-in-code — the Stage-1 block schema VERBATIM ------------------- #
    from src.feedback import schema

    pc_deltas = [+rungs[3], -rungs[3], +rungs[4], -rungs[4], +rungs[2], -rungs[2],
                 +rungs[4], -rungs[3]]
    tail = dict(BASE_TAIL)
    for t, d in enumerate(pc_deltas):
        tail = {**tail, "cvar_05": float(tail["cvar_05"] + d)}
        block = schema.build_block("distributional", 0.85, tail)
        items.append({
            "id": f"PC-{t:03d}", "family": "P-C", "delta": float(d),
            "prompt": ("You previously wrote a Python reward function for a portfolio RL agent. "
                       "Here is the realized feedback for your last candidate:\n\n" + block +
                       "\n\nWrite an improved reward function (a single Python function named "
                       "`reward`). Respond with only the code."),
            "answer": None,  # scored post-hoc by the SQ1 code-feature extractor
        })
    for t in range(2):  # placebo-shuffled control turns (within-survey control)
        block = schema.build_block("placebo_shuffled", 0.85, dict(BASE_TAIL),
                                   shuffle_seed=1000 + t)
        items.append({
            "id": f"PCX-{t:03d}", "family": "P-C-placebo",
            "prompt": ("You previously wrote a Python reward function for a portfolio RL agent. "
                       "Here is the realized feedback for your last candidate:\n\n" + block +
                       "\n\nWrite an improved reward function (a single Python function named "
                       "`reward`). Respond with only the code."),
            "answer": None,
        })

    # ---- self-consistency: a deterministic 20% sample of P-A/P-B re-asked once -------------- #
    scored = [it for it in items if it["family"] in ("P-A", "P-B")]
    sample = rng.choice(len(scored), size=max(1, len(scored) // 5), replace=False)
    for s in sorted(int(x) for x in sample):
        rep = dict(scored[s])
        rep["id"] = rep["id"] + "-rep1"
        rep["rep"] = 1
        items.append(rep)
    return items


# --------------------------------------------------------------------------- #
# The fleet runner (archival-resume; hard spend cap; stub-injectable)           #
# --------------------------------------------------------------------------- #
def _parse_answer(family: str, text: str) -> str | None:
    """Trivial parse for P-A/P-B (single letter / letter chain); None = unparseable (kept raw)."""
    t = text.strip().upper()
    if family == "P-A":
        for ch in t:
            if ch in "AB":
                return ch
        return None
    if family == "P-B":
        letters = [c for c in t if c in "ABCD"]
        return ">".join(letters[:4]) if len(letters) >= 4 else None
    return None


def run_model(label: str, transport: Callable[[str, str], str], items: list[dict[str, Any]],
              out_dir: Path, *, budget: list[int]) -> dict[str, Any]:
    """Run one model over the item set with archival resume; returns the per-model tally."""
    import time

    out_dir.mkdir(parents=True, exist_ok=True)
    resp_path = out_dir / "responses.jsonl"
    done: set[str] = set()
    if resp_path.is_file():
        for line in resp_path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["id"])
            except (ValueError, KeyError):
                continue
    n_ok = n_err = n_skip = 0
    system = ("You are participating in a calibrated survey about quantitative risk feedback. "
              "Answer precisely and follow the requested output format exactly.")
    for it in items:
        if it["id"] in done:
            n_skip += 1
            continue
        if budget[0] <= 0:
            print(f"[m2:{label}] SPEND CAP reached — stopping (resume continues later)", flush=True)
            break
        budget[0] -= 1
        row: dict[str, Any] = {"id": it["id"], "family": it["family"], "model": label,
                               "ts": time.time()}
        try:
            t0 = time.perf_counter()
            text = transport(system, it["prompt"])
            row["response"] = str(text)
            row["secs"] = round(time.perf_counter() - t0, 2)
            row["parsed"] = _parse_answer(it["family"], str(text))
            if it.get("answer") is not None and row["parsed"] is not None:
                row["correct"] = bool(row["parsed"] == it["answer"])
            n_ok += 1
        except Exception as exc:  # noqa: BLE001 — one model/item failure never sinks the fleet
            row["error"] = f"{type(exc).__name__}: {exc}"
            n_err += 1
        with resp_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    tally = {"model": label, "answered": n_ok, "errors": n_err, "resumed_skips": n_skip}
    (out_dir / "tally.json").write_text(json.dumps(tally, indent=2), encoding="utf-8")
    return tally


def _stub_transport(_system: str, user: str) -> str:
    """Keyless harness-validation transport: answers P-A/P-B deterministically-plausibly."""
    if "one letter: A or B" in user:
        return "A"
    if "four letters" in user:
        return "A>B>C>D"
    return "def reward(returns, weights, **kw):\n    return returns.mean()\n"


def load_roster(spec: str, include_extras: bool = False) -> list[dict[str, str]]:
    """``stub`` -> the keyless harness roster; else a YAML roster file.

    Accepts BOTH schemas: the v1 flat ``models: [...]`` list and the v2 (R80, 2026-07-20)
    ``core: [...]`` + ``extras_budget_permitting: [...]`` split — the extras are appended only
    when ``include_extras`` is set (they run only if the R81 $30 ceiling has headroom; the
    budget-ordered execution is the caller's responsibility). ``excluded_by_design`` entries are
    never loaded. Final ids pinned at execution — the Qwen snapshot pattern."""
    if spec == "stub":
        return [{"label": "stub-a", "provider": "stub", "model": "stub"},
                {"label": "stub-b", "provider": "stub", "model": "stub"}]
    import yaml

    cfg = yaml.safe_load(Path(spec).read_text(encoding="utf-8")) or {}
    models = list(cfg.get("models") or cfg.get("core") or [])
    if include_extras:
        models += list(cfg.get("extras_budget_permitting") or [])
    if not models:
        raise SystemExit(f"no models in {spec} (expected 'models:' or 'core:')")
    return [dict(m) for m in models]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="M2 cross-LLM numeracy+responsiveness survey runner.")
    p.add_argument("--archive", required=True,
                   help="Search archive root whose chains anchor the P-A difficulty rungs "
                        "(campaign search/ post-bank; a prototype root = directional dry run).")
    p.add_argument("--models", required=True,
                   help="'stub' (keyless harness validation) or a YAML roster file.")
    p.add_argument("--include-extras", action="store_true",
                   help="Also load the roster's extras_budget_permitting tier (R81: run only "
                        "with $30-ceiling headroom).")
    p.add_argument("--out", required=True, help="Output root (per-model subdirs).")
    p.add_argument("--max-calls", type=int, default=1000,
                   help="HARD spend cap across the whole invocation (protocol ~700-800).")
    p.add_argument("--seed", type=int, default=20260712, help="Item-generation seed (registered).")
    args = p.parse_args(argv)

    from src.utils.env import load_env

    load_env()
    items = build_items(args.archive, seed=args.seed)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "items.json").write_text(
        json.dumps({"seed": args.seed, "archive": str(args.archive), "items": items},
                   indent=1, sort_keys=True), encoding="utf-8")
    n_scored = sum(1 for it in items if it.get("answer") is not None)
    print(f"[m2] {len(items)} items generated ({n_scored} auto-scored; seed {args.seed})")

    budget = [int(args.max_calls)]
    tallies = []
    for m in load_roster(args.models, include_extras=bool(getattr(args, "include_extras", False))):
        label = str(m["label"])
        if m.get("provider") == "stub":
            transport: Callable[[str, str], str] = _stub_transport
        else:
            from src.llm.client import build_transport

            transport = build_transport(str(m["provider"]), str(m["model"]),
                                        m.get("api_key_env"), temperature=0.0, max_tokens=1024)
        print(f"[m2] running {label} ({m.get('provider')}/{m.get('model')})", flush=True)
        tallies.append(run_model(label, transport, items, out_root / label, budget=budget))
    (out_root / "fleet_summary.json").write_text(json.dumps(tallies, indent=2), encoding="utf-8")
    print(f"[m2] fleet complete: {tallies}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
