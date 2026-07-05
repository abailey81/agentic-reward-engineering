"""Deterministic, reproducible mutation-testing probe (test-rigor exhibit).

WHY THIS EXISTS. Mutation testing answers "do the tests actually CATCH faults, or just execute lines?".
The off-the-shelf tool (`mutmut`) is installed (requirements-test.txt) but its emoji/console I/O is broken on
a Windows cp1251 console (UnicodeEncode on the summary / UnicodeDecode on the captured pytest output). This
probe is a small, dependency-free, ASCII-only, fully reproducible alternative: it applies a fixed catalogue of
targeted source mutations to ONE module, runs that module's real test suite per mutant, and reports the kill
rate. A mutant is KILLED iff the test run fails (non-zero exit); a SURVIVING mutant marks a gap in the tests.

SAFETY. The target's pristine bytes are read once into memory; every mutant is written then restored in a
``finally``; an ``atexit`` hook and SIGINT/SIGTERM handlers also restore; and the caller additionally keeps an
out-of-tree backup. After the run the file is asserted byte-identical to the pristine original.

USAGE:
    python scripts/mutation_probe.py            # default: src/backtest/metrics.py + its 3 test files
Exit code 0 iff every applied mutant was killed (100% kill rate) AND the source was restored intact.
"""
from __future__ import annotations

import argparse
import atexit
import signal
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# (description, find, replace). `find` must be a substring that occurs in the target; a mutation whose `find`
# is absent is reported NOT-APPLICABLE and excluded from the rate (keeps the catalogue robust to refactors).
# Each mutation is a SEMANTIC fault a competent test suite should detect.
_METRICS_MUTATIONS: list[tuple[str, str, str]] = [
    ("epsilon guard neutralised", "_EPS = 1e-12", "_EPS = 0.0"),
    ("annualisation factor wrong", "PERIODS_PER_YEAR: int = 252", "PERIODS_PER_YEAR: int = 200"),
    ("hit-rate boundary >0 -> >=0", "float(np.mean(r > 0.0))", "float(np.mean(r >= 0.0))"),
    ("win/loss split boundary", "wins, loss = r[r > 0.0], r[r < 0.0]", "wins, loss = r[r >= 0.0], r[r <= 0.0]"),
    ("profit_factor guard reverted (denormal bug)",
     "if (loss.size and abs(loss.sum()) > _EPS)", "if (loss.size and loss.sum() != 0)"),
    ("gain_loss guard reverted (denormal bug)",
     "if (loss.size and wins.size and abs(loss.mean()) > _EPS)", "if (loss.size and wins.size)"),
    ("drawdown sign flipped", "wealth / safe_peak - 1.0", "wealth / safe_peak + 1.0"),
    ("underwater comparison flipped", "underwater = dd < -_EPS", "underwater = dd > -_EPS"),
    ("best/worst period swapped (best)", 'out["best_period"] = float(r.max())',
     'out["best_period"] = float(r.min())'),
    ("best/worst period swapped (worst)", 'out["worst_period"] = float(r.min())',
     'out["worst_period"] = float(r.max())'),
    ("MAR threshold shifted", "_mar = 0.0", "_mar = 1.0"),
    ("omega gains->minimum", "gains = float(np.maximum(r - _mar, 0.0).sum())",
     "gains = float(np.minimum(r - _mar, 0.0).sum())"),
    ("sortino downside guard flipped", "if dd_dev > _EPS else 0.0", "if dd_dev < _EPS else 0.0"),
    ("calmar sign", 'float(out["cagr"] / dd["max_drawdown"])', 'float(-out["cagr"] / dd["max_drawdown"])'),
]

#: Tail estimator (the project's core contribution) — empirical body + EVT tail + the WS5 uncertainty methods.
_MEASUREMENT_MUTATIONS: list[tuple[str, str, str]] = [
    ("empirical CVaR takes BEST not worst tail", "worst = arr[:n]", "worst = arr[-n:]"),
    ("EVT alpha>F_u fallback boundary flipped", "if alpha > fu:  # POT", "if alpha < fu:  # POT"),
    ("EVT xi>=1 (infinite-mean) guard neutralised", "if xi >= 1.0:", "if xi >= 100.0:"),
    ("EVT xi<=-0.5 (non-regular) guard neutralised", "if xi <= -0.5:", "if xi <= -100.0:"),
    ("EVT alpha-cutoff routing widened", "EVT_ALPHA_CUTOFF = 0.05", "EVT_ALPHA_CUTOFF = 0.5"),
    ("left_tail_mass comparison flipped", "np.mean(arr < -LEFT_TAIL_K * std)", "np.mean(arr > -LEFT_TAIL_K * std)"),
    ("left_tail_mass k zeroed", "LEFT_TAIL_K = 2.0", "LEFT_TAIL_K = 0.0"),
    ("robust_skew sign inverted", "robust_skew = ((q95 - q50) - (q50 - q05))", "robust_skew = ((q50 - q05) - (q95 - q50))"),
    ("exceedance_count floor not ceil", "int(max(1, math.ceil(float(alpha) * self.T)))", "int(max(1, math.floor(float(alpha) * self.T)))"),
    ("reliability HIGH tier loosened", 'if n_exc > 30:', 'if n_exc > 3:'),
    ("reliability MED tier loosened", "if n_exc >= 7:", "if n_exc >= 1:"),
    ("cvar_ci percentile bounds swapped", "lo, hi = np.quantile(boots, [lo_q, 1.0 - lo_q])", "lo, hi = np.quantile(boots, [1.0 - lo_q, lo_q])"),
    ("block-bootstrap restart prob inverted", "cur = int(rng.integers(0, t)) if rng.random() < p else (cur + 1) % t",
     "cur = int(rng.integers(0, t)) if rng.random() > p else (cur + 1) % t"),
]

#: Validation-DSR fitness selector (selection/fitness.py) — patterns matched against the real source.
_FITNESS_MUTATIONS: list[tuple[str, str, str]] = [
    ("validation split guard inverted", 'if split != "val":', 'if split == "val":'),
    ("CVaR penalty sign flipped (dsr+penalty)", "return float(dsr - penalty)", "return float(dsr + penalty)"),
    ("non-finite CVaR guard inverted", "lam * abs(c) if np.isfinite(c) else 0.0",
     "lam * abs(c) if not np.isfinite(c) else 0.0"),
    ("lam==0 fast-path inverted", "if lam == 0.0:", "if lam != 0.0:"),
]

#: Deflated/Probabilistic Sharpe (inference/deflated_sharpe.py) — patterns matched against the real source.
_DEFLATED_SHARPE_MUTATIONS: list[tuple[str, str, str]] = [
    ("PSR small-sample sqrt(n-1) -> sqrt(n+1)",
     "(sr - sr_star) * math.sqrt(max(n - 1, 0))", "(sr - sr_star) * math.sqrt(max(n + 1, 0))"),
    ("PSR upper-tail flipped (1 - cdf)", "z = (sr - sr_star)", "z = -(sr - sr_star)"),
    ("PSR denominator skew-sign flipped",
     "denom_var = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr",
     "denom_var = 1.0 + skew * sr + (kurt - 1.0) / 4.0 * sr * sr"),
    ("expected-max Euler g-term sign flipped",
     "+ g * norm.ppf(1.0 - 1.0 / (n * e))", "- g * norm.ppf(1.0 - 1.0 / (n * e))"),
    ("expected-max Euler-Mascheroni constant zeroed", "g = _EULER_MASCHERONI", "g = 0.0"),
    ("DSR within-series var proxy n-1 -> n+1", "var_sr = denom_var / (n - 1)", "var_sr = denom_var / (n + 1)"),
    ("MinTRL never-significant guard inverted", "if sr <= sr_star:", "if sr > sr_star:"),
    ("legacy alias loud-reject inverted", "if sr_benchmark != 0.0:", "if sr_benchmark == 0.0:"),
]

#: Registry: module path -> (default test files, mutation catalogue). ``--module`` selects the entry; a
#: catalogue mutation whose ``find`` is absent is reported NA (so the registry tolerates source drift).
_CATALOGUES: dict[str, tuple[list[str], list[tuple[str, str, str]]]] = {
    "src/backtest/metrics.py": (
        ["tests/test_backtest_metrics.py", "tests/test_backtest_metrics_deep.py",
         "tests/test_metrics_denormal_guard.py"],
        _METRICS_MUTATIONS,
    ),
    "src/feedback/measurement.py": (
        ["tests/test_measurement.py", "tests/test_measurement_uncertainty.py", "tests/test_properties.py"],
        _MEASUREMENT_MUTATIONS,
    ),
    "src/selection/fitness.py": (
        ["tests/test_fitness.py", "tests/test_rewards_deep.py"],
        _FITNESS_MUTATIONS,
    ),
    "src/inference/deflated_sharpe.py": (
        ["tests/test_inference.py", "tests/test_inference_deep.py"],
        _DEFLATED_SHARPE_MUTATIONS,
    ),
}


def _run() -> int:
    ap = argparse.ArgumentParser(description="Deterministic mutation probe (ASCII, reproducible).")
    ap.add_argument("--module", default="src/backtest/metrics.py", choices=sorted(_CATALOGUES))
    ap.add_argument("--tests", nargs="+", default=None,
                    help="override the module's default test files (else the registry default is used)")
    args = ap.parse_args()
    default_tests, _MUTATIONS = _CATALOGUES[args.module]
    if args.tests is None:
        args.tests = default_tests

    target = REPO / args.module
    pristine = target.read_bytes()

    def restore() -> None:
        if target.read_bytes() != pristine:
            target.write_bytes(pristine)

    atexit.register(restore)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda *_: (restore(), sys.exit(130)))
        except (ValueError, OSError):
            pass

    pristine_text = pristine.decode("utf-8")
    runner = [sys.executable, "-m", "pytest", *args.tests, "-x", "-q", "-p", "no:randomly", "--no-header"]

    print(f"[mutation-probe] target={args.module}  tests={' '.join(args.tests)}")
    print(f"[mutation-probe] catalogue: {len(_MUTATIONS)} candidate mutations\n")

    killed = survived = na = 0
    survivors: list[str] = []
    try:
        for desc, find, repl in _MUTATIONS:
            if find not in pristine_text:
                print(f"  NA   {desc}  (pattern absent)")
                na += 1
                continue
            mutated = pristine_text.replace(find, repl, 1)
            if mutated == pristine_text:
                print(f"  NA   {desc}  (no-op replacement)")
                na += 1
                continue
            target.write_text(mutated, encoding="utf-8")
            try:
                rc = subprocess.run(runner, cwd=REPO, capture_output=True).returncode
            finally:
                target.write_bytes(pristine)
            if rc != 0:
                killed += 1
                print(f"  KILL {desc}")
            else:
                survived += 1
                survivors.append(desc)
                print(f"  SURV {desc}  <-- tests did NOT catch this mutation")
    finally:
        restore()

    applied = killed + survived
    rate = (100.0 * killed / applied) if applied else 0.0
    print(f"\n[mutation-probe] applied={applied}  killed={killed}  survived={survived}  n/a={na}")
    print(f"[mutation-probe] KILL RATE = {rate:.1f}%")
    if survivors:
        print("[mutation-probe] SURVIVORS (test gaps):")
        for s in survivors:
            print(f"    - {s}")
    assert target.read_bytes() == pristine, "FATAL: target not restored to pristine bytes"
    print("[mutation-probe] source restored intact.")
    return 0 if survived == 0 and applied > 0 else 1


if __name__ == "__main__":
    raise SystemExit(_run())
