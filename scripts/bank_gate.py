"""The BANK-GATE runsheet as ONE command (OVERALL_DESIGN §6 "crisp-up"; PLAN §3).

The single-look hour is the highest-stakes step of the campaign; this wrapper turns it from tribal
knowledge into a replayed procedure: each step is an EXISTING certified tool, run in the registered
order, fail-fast, with every step's stdout captured to a timestamped log directory. Nothing here
computes science — it only SEQUENCES the tools (so rehearsing it on prototype data exercises the
identical procedure the sealed look will use).

Order (the registered runsheet):
  1. archive_integrity.py      — the archive is complete/uncorrupted (counts + homogeneity)
  2. resume_audit.py           — read-only: nothing pending, mirror parity
  3. analyze_campaign.py       — THE single confirmatory look (only once, on the real gate)
  4. variance_decomposition.py — D3 depth bundle
  5. fed_delta_snr.py          — instrument (h) exhibit (R76)
  6. make_prereg_bundle.py     — the frozen-design results bundle

Usage:
    python scripts/bank_gate.py --archive <campaign_root> --rehearsal   # prototype dress rehearsal
    python scripts/bank_gate.py --archive <campaign_root>              # the real gate (operator-run)

``--rehearsal`` stamps every log line + the output dir REHEARSAL (directional data only; no
dissertation number) and tolerates step-skips where a tool needs campaign-only inputs. The REAL gate
runs all steps and stops on the first failure.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable

def step(cmd: list[str], name: str, log_dir: Path, *, tolerate: bool) -> bool:
    print(f"\n=== [bank_gate] {name} ===\n    $ {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding="utf-8",
                         errors="replace")
    (log_dir / f"{name}.log").write_text(
        f"$ {' '.join(cmd)}\nexit={res.returncode}\n\n--- stdout ---\n{res.stdout}\n"
        f"--- stderr ---\n{res.stderr}", encoding="utf-8")
    tail = "\n".join((res.stdout or "").splitlines()[-5:])
    print(tail)
    if res.returncode != 0:
        print(f"[bank_gate] {name} FAILED (exit {res.returncode}) — log: {log_dir / (name + '.log')}")
        # 2026-07-13 audit fix: a WIRING error (argparse exit 2 / usage text) is NEVER tolerable —
        # the 2026-07-12 version tolerated everything in rehearsal mode, so a dress run would have
        # printed COMPLETE with every step dead (a false-green certification of a broken runsheet).
        if res.returncode == 2 or "unrecognized arguments" in (res.stderr or ""):
            print("[bank_gate] WIRING ERROR (argparse) — never tolerated, even in rehearsal.")
            return False
        if not tolerate:
            return False
        print("[bank_gate] REHEARSAL mode: tolerated (missing campaign-only inputs), continuing.")
    return True


def main() -> int:
    from src.utils.config import cfg_get, load_config
    from src.utils.console import make_console_safe

    # NOTE: OBSERVED 2026-07-27: this runsheet died with UnicodeEncodeError while PRINTING a log tail —
    # ``step`` reads subprocess output with errors="replace" (U+FFFD) and the console here is cp1251.
    # The POST-CAMPAIGN analysis, killed by a codepage. See src/utils/console.py.
    make_console_safe()

    p = argparse.ArgumentParser(description="The bank-gate runsheet, sequenced fail-fast.")
    p.add_argument("--archive", required=True, help="Campaign archive root (local mirror).")
    p.add_argument("--seeds", default=None,
                   help="The REALIZED seed rung (e.g. 0-402). REQUIRED for the real gate — the "
                        "resume audit must check the stopped rung, not the full 568 ladder "
                        "(exogenous truncation is a valid stop, not missing work).")
    p.add_argument("--arms", nargs="+", default=None,
                   help="Unit roster (default: config/campaign.yaml arms).")
    p.add_argument("--baselines", nargs="*", default=None,
                   help="H1 baselines (default: config/campaign.yaml h1_baselines).")
    p.add_argument("--single-shot-root", default=None,
                   help="H3 single-shot archive root (threads to analyze_campaign).")
    p.add_argument("--rehearsal", action="store_true",
                   help="Dress-rehearsal mode (prototype data): label everything REHEARSAL and "
                        "tolerate missing campaign-only inputs (never wiring errors).")
    p.add_argument("--skip", nargs="*", default=[], help="Step names to skip (documented reason!).")
    args = p.parse_args()

    camp = load_config("campaign")
    arms = args.arms or list(cfg_get(camp, "arms", []) or [])
    baselines = args.baselines if args.baselines is not None else list(
        cfg_get(camp, "h1_baselines", []) or [])
    seeds = args.seeds
    if seeds is None:
        if not args.rehearsal:
            raise SystemExit("--seeds (the realized rung, e.g. 0-402) is REQUIRED for the real "
                             "gate — auditing the full ladder against a valid exogenous stop "
                             "reports phantom missing seeds.")
        seeds = "0-0"  # rehearsal stub

    tag = "REHEARSAL" if args.rehearsal else "GATE"
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = REPO / "outputs" / "bank_gate_logs" / f"{tag}_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[bank_gate] mode={tag}  archive={args.archive}  logs={log_dir}")
    print(f"[bank_gate] arms={arms}  baselines={baselines}  seeds={seeds}")

    # Every invocation below matches the target script's ACTUAL argparse contract (2026-07-13
    # audit: four of six steps previously used a uniform --archive flag none of them defined).
    analyze_cmd = [PY, "scripts/analyze_campaign.py", "--root", args.archive]
    if args.single_shot_root:
        analyze_cmd += ["--single-shot-root", args.single_shot_root]
    steps: list[tuple[str, list[str]]] = [
        # write-then-verify: the cluster path does not seal the mirror on completion, so the gate
        # seals what was pulled, then verifies the seal (tamper/corruption evidence from here on).
        ("archive_integrity_write", [PY, "scripts/archive_integrity.py", "write", args.archive]),
        ("archive_integrity_verify", [PY, "scripts/archive_integrity.py", "verify", args.archive]),
        ("resume_audit", [PY, "scripts/resume_audit.py", args.archive,
                          "--arms", *arms, *(["--baselines", *baselines] if baselines else []),
                          "--seeds", seeds]),
        ("analyze_campaign", analyze_cmd),
        ("variance_decomposition", [PY, "scripts/variance_decomposition.py", "--root", args.archive]),
        # instrument (h): the cluster mirror nests search records under <root>/search/<arm>/
        ("fed_delta_snr", [PY, "scripts/fed_delta_snr.py",
                           "--archive", str(Path(args.archive) / "search"),
                           "--json", str(log_dir / "fed_delta_snr.json")]),
        ("make_prereg_bundle", [PY, "scripts/make_prereg_bundle.py"]),
    ]
    for name, cmd in steps:
        if name in args.skip:
            print(f"=== [bank_gate] {name} SKIPPED (--skip) ===")
            continue
        if not step(cmd, name, log_dir, tolerate=args.rehearsal):
            print(f"[bank_gate] STOPPED at {name}. Fix, then re-run — do NOT proceed past a red step.")
            return 1
    print(f"\n[bank_gate] {tag} COMPLETE — logs in {log_dir}")
    if args.rehearsal:
        print("[bank_gate] REHEARSAL data is directional-only; no number enters the dissertation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
