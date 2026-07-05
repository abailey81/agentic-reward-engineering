"""Consolidated reproducibility audit — one command that scores the repo against its own repro contract.

Reproducibility is a graded dimension and a standing priority, but its evidence is scattered (a lockfile
here, a determinism setting there, a freeze hash in a config, a data checksum in a manifest). This script
gathers it into a single PASS/WARN/FAIL report so a marker — or a future self — can verify the claims in
the project's reproducibility checklist in one shot.

Most checks are pure text/presence checks of the repo root. TWO pillars do REAL verification, reusing the
project's OWN primitives through GUARDED imports so the audit still runs in a bare environment — degrading
to WARN, **never a silent PASS**, if the optional dependency is absent:

  * **pre-registration freeze** — when ``frozen: true``, the recorded ``freeze_hash`` is re-derived from the
    design artifacts via ``scripts/freeze.py::canonical_hash`` (the SINGLE source of the canonical byte
    order — the same function ``scripts/freeze.py --check`` runs; this module never re-implements it) and
    compared. A MISMATCH is a **FAIL** — the drift ``--check`` catches (an edited hash-bound file) — not a
    PASS. ``frozen: false`` is the project's legitimate current state (WARN). Needs PyYAML; import absent → WARN.
  * **data provenance** — the ACTIVE-suffix headline gold parquet's SHA-256 is recomputed with stdlib
    ``hashlib`` (reusing ``src/data/loaders.py::_file_sha256``) and compared to the frozen manifest entry
    (``loaders._expected_sha256`` reading ``data/manifest/manifest.jsonl``). A MISMATCH is a **FAIL** (tamper
    / wrong build — what ``loaders._verify_checksum`` raises on at load time). The licensed panel is
    legitimately out-of-repo, so an ABSENT parquet is a WARN. Needs numpy/pandas (to import the loader);
    import absent → WARN.

Each check returns ``{"name", "status", "detail"}``; ``status`` is ``pass`` / ``warn`` / ``fail``. ``warn``
is for legitimately-pending or unverifiable-here state (pre-freeze; an out-of-repo licensed panel; an
optional dependency absent); ``fail`` is a genuine repro defect (a missing lockfile / determinism setting,
a freeze-hash DRIFT, a data-checksum MISMATCH). ``ok = (n_fail == 0)`` — so a real regression in either
verified pillar fails the audit and its non-zero exit is meaningful. The tool NEVER fabricates: a check whose
evidence is absent, or which it cannot verify here, says so rather than passing.

NOTE — CI: this consolidated report is NOT wired into CI. ``.github/workflows/ci.yml`` enforces the
design-hash drift-guard directly (``scripts/freeze.py --check``) and runs the test suite; this script is a
local / on-demand roll-up. Run it by hand, or add it to CI and gate on ``--strict``, to enforce it there.

Usage::

    python scripts/audit_reproducibility.py                 # human report, exit 0 unless a hard FAIL
    python scripts/audit_reproducibility.py --json          # machine-readable
    python scripts/audit_reproducibility.py --strict        # exit non-zero on WARN too
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

PASS, WARN, FAIL = "pass", "warn", "fail"

#: This file's own location in the REAL install (NOT the audited ``root``, which may be a synthetic test
#: repo). The two REAL-verification pillars import the project's primitives from HERE, then apply them to
#: whatever ``root`` is being audited, so ``freeze.canonical_hash``/``loaders._file_sha256`` remain the
#: single source and the checks stay testable against a synthetic root.
_HERE = Path(__file__).resolve()
_SCRIPTS_DIR = _HERE.parent
_REPO = _HERE.parents[1]


def _read(root: Path, *parts: str) -> str:
    """Return a file's text, or ``""`` if it is absent/unreadable (checks degrade, never crash)."""
    try:
        return (root.joinpath(*parts)).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _import_freeze() -> Any | None:
    """GUARDED import of ``scripts/freeze.py`` — the single source of the canonical design hash.

    Returns the module, or ``None`` if it (or its PyYAML dependency) cannot be imported, so the freeze
    check degrades to WARN instead of crashing the audit in a bare environment. Importing ``freeze`` runs
    that module's own ``sys.path`` bootstrap (it adds the real repo root and pulls ``src.utils.provenance``),
    so the returned ``canonical_hash`` is exactly the function ``scripts/freeze.py --check`` uses — this
    module never re-derives the canonical byte order. The broad ``except`` is deliberate: ANY import-time
    failure must degrade to WARN, never propagate (mirrors ``_read``'s degrade-never-crash contract).
    """
    try:
        if str(_SCRIPTS_DIR) not in sys.path:
            sys.path.insert(0, str(_SCRIPTS_DIR))
        import freeze  # type: ignore[import-not-found]

        return freeze
    except Exception:
        return None


def _import_loaders() -> Any | None:
    """GUARDED import of ``src.data.loaders`` — reused only for its stdlib checksum primitives.

    ``loaders._file_sha256`` streams a file to a SHA-256; ``loaders._expected_sha256`` looks a file up in
    the frozen manifest. Both are stdlib-pure, but the module imports numpy/pandas at top level, so the
    import is GUARDED: ``None`` when those deps are absent, letting the data-provenance check degrade to
    WARN (never a silent PASS). The broad ``except`` is deliberate (see :func:`_import_freeze`).
    """
    try:
        if str(_REPO) not in sys.path:
            sys.path.insert(0, str(_REPO))
        from src.data import loaders  # type: ignore[import-not-found]

        return loaders
    except Exception:
        return None


def check_python_version(root: Path) -> dict[str, str]:
    txt = _read(root, ".python-version").strip()
    if not txt:
        return {"name": "python-version pin", "status": FAIL, "detail": ".python-version is missing"}
    pyproject = _read(root, "pyproject.toml")
    consistent = bool(re.search(r"requires-python", pyproject))
    detail = f"pinned to {txt}" + ("; pyproject declares requires-python" if consistent else "")
    return {"name": "python-version pin", "status": PASS, "detail": detail}


def check_lockfile(root: Path) -> dict[str, str]:
    pinless: list[str] = []
    for name in ("requirements.lock", "requirements.txt", "uv.lock", "poetry.lock"):
        txt = _read(root, name)
        if not txt.strip():
            continue
        n = sum(1 for line_v in txt.splitlines() if line_v.strip() and not line_v.lstrip().startswith("#"))
        # Deep review 2026-07-26: this PASSED on the first non-empty candidate even when it pinned
        # NOTHING — it computed ``n``, reported "(0 pinned lines)", and called that a pass. A
        # comment-only lockfile certifies exactly as much reproducibility as no lockfile at all, so
        # per this module's own "never a silent PASS" contract it must not pass. The early return
        # also let a pin-less requirements.lock MASK a genuine uv.lock/poetry.lock later in the tuple.
        if n:
            return {"name": "dependency lockfile", "status": PASS, "detail": f"{name} ({n} pinned lines)"}
        pinless.append(name)
    detail = (f"lockfile(s) present but pin NOTHING: {', '.join(pinless)}" if pinless
              else "no lockfile (requirements.lock/uv.lock/…)")
    return {"name": "dependency lockfile", "status": FAIL, "detail": detail}


def check_version_pins(root: Path) -> dict[str, str]:
    pyproject = _read(root, "pyproject.toml")
    if not pyproject:
        return {"name": "version pins", "status": FAIL, "detail": "pyproject.toml missing"}
    pins = [p for p in ("torch", "numpy", "stable-baselines3", "scipy") if re.search(rf"\b{re.escape(p)}\b", pyproject)]
    if "torch" in pins and "numpy" in pins:
        return {"name": "version pins", "status": PASS, "detail": "pyproject pins " + ", ".join(pins)}
    return {"name": "version pins", "status": WARN, "detail": f"only found pins for {pins or 'none'}"}


def check_determinism_settings(root: Path) -> dict[str, str]:
    txt = _read(root, "src", "utils", "seeding.py")
    if not txt:
        return {"name": "determinism settings", "status": FAIL, "detail": "src/utils/seeding.py missing"}
    needles = {
        "CUBLAS_WORKSPACE_CONFIG": "CUBLAS_WORKSPACE_CONFIG" in txt,
        "use_deterministic_algorithms": "use_deterministic_algorithms" in txt,
        "cudnn.deterministic": "cudnn.deterministic" in txt or "deterministic = True" in txt,
        "PYTHONHASHSEED": "PYTHONHASHSEED" in txt,
    }
    missing = [k for k, ok in needles.items() if not ok]
    if not missing:
        return {"name": "determinism settings", "status": PASS, "detail": "all 4 determinism knobs set in seeding.py"}
    status = FAIL if len(missing) >= 2 else WARN
    return {"name": "determinism settings", "status": status, "detail": "missing: " + ", ".join(missing)}


def check_seed_recording(root: Path) -> dict[str, str]:
    for name in ("campaign.yaml", "preregistration.yaml"):
        txt = _read(root, "config", name)
        if re.search(r"seed", txt, re.IGNORECASE):
            return {"name": "seed management", "status": PASS, "detail": f"seeds declared in config/{name}"}
    if re.search(r"def set_global_seed", _read(root, "src", "utils", "seeding.py")):
        return {"name": "seed management", "status": WARN, "detail": "set_global_seed present; no seed count in config"}
    return {"name": "seed management", "status": FAIL, "detail": "no seed declaration found"}


_NULLISH = ("null", "~", "''", '""', "")


def check_freeze_state(root: Path) -> dict[str, str]:
    """Verify the pre-registration freeze by RE-COMPUTING the design hash — not a text scan.

    Pre-freeze (``frozen: false``) is this project's legitimate current state → WARN. Once frozen, the
    recorded ``freeze_hash`` is re-derived from the design artifacts via ``scripts/freeze.py::canonical_hash``
    (the single source of the canonical byte order — the same digest ``scripts/freeze.py --check`` recomputes)
    and compared: a match is PASS; a MISMATCH — an edited hash-bound file, the drift ``--check`` catches — is
    a **FAIL**. If the freeze module cannot be imported (PyYAML absent) or its hash cannot be recomputed, the
    hash is NOT re-verified → WARN, never a silent PASS.
    """
    txt = _read(root, "config", "preregistration.yaml")
    if not txt:
        return {"name": "pre-registration freeze", "status": FAIL, "detail": "config/preregistration.yaml missing"}
    frozen_m = re.search(r"frozen:\s*(true|false)", txt, re.IGNORECASE)
    hash_m = re.search(r"freeze_hash:\s*([^\s#]+)", txt)
    frozen_val = (frozen_m.group(1).lower() == "true") if frozen_m else False
    hash_val = hash_m.group(1).strip() if hash_m else "null"

    if not frozen_val:
        return {"name": "pre-registration freeze", "status": WARN,
                "detail": "pre-freeze (frozen=false; hash not yet bound) — legitimate by design"}
    # frozen: true — the design hash must be recorded AND must re-derive from the artifacts.
    if hash_val in _NULLISH:
        return {"name": "pre-registration freeze", "status": FAIL,
                "detail": "frozen: true but freeze_hash is null/blank — the design hash was never recorded"}
    freeze = _import_freeze()
    if freeze is None:
        return {"name": "pre-registration freeze", "status": WARN,
                "detail": f"FROZEN (freeze_hash={hash_val[:12]}…) but hash NOT re-verified (freeze deps absent)"}
    try:
        recomputed = freeze.canonical_hash(root)
    except Exception as exc:  # artifacts unreadable / malformed — cannot verify, so do NOT fabricate a PASS
        return {"name": "pre-registration freeze", "status": WARN,
                "detail": f"FROZEN but canonical hash not re-computable ({type(exc).__name__}: {exc})"}
    if recomputed == hash_val:
        return {"name": "pre-registration freeze", "status": PASS,
                "detail": f"FROZEN — canonical SHA-256 re-verified ({recomputed[:12]}…)"}
    return {"name": "pre-registration freeze", "status": FAIL,
            "detail": f"freeze_hash DRIFT — recorded {hash_val[:12]}… != canonical {recomputed[:12]}… "
                      "(a hash-bound design file changed; run scripts/freeze.py --check)"}


def check_archive_replay(root: Path) -> dict[str, str]:
    txt = _read(root, "config", "llm.yaml")
    if re.search(r"archive:\s*true", txt, re.IGNORECASE):
        return {"name": "LLM archive-replay", "status": PASS, "detail": "config/llm.yaml archive: true (non-det LLM replayed)"}
    if txt:
        return {"name": "LLM archive-replay", "status": WARN, "detail": "config/llm.yaml present but archive flag not 'true'"}
    return {"name": "LLM archive-replay", "status": WARN, "detail": "config/llm.yaml not found"}


def _active_gold_suffix(root: Path) -> str:
    """The active gold suffix from ``config/data.yaml``'s ``gold.suffix`` (stdlib regex, no PyYAML).

    Mirrors ``loaders.gold_suffix``'s PRIMARY source (the frozen config key) so the provenance check can
    locate the headline panel even in a bare environment. A leading ``_`` is stripped (loader convention);
    falls back to ``univ5`` (the loader's ``_DEFAULT_SUFFIX``) when the key is absent.
    """
    m = re.search(r"(?m)^\s*suffix:\s*_?([A-Za-z0-9]+)", _read(root, "config", "data.yaml"))
    return m.group(1) if m else "univ5"


def check_data_provenance(root: Path) -> dict[str, str]:
    """Verify the frozen data panel against its manifest checksum — a REAL SHA-256 comparison.

    ``data/manifest/manifest.jsonl`` (the shipped provenance record) must exist and be non-empty, else FAIL.
    For the ACTIVE-suffix headline gold parquet, if it is present locally its SHA-256 is recomputed with
    stdlib ``hashlib`` (reusing ``loaders._file_sha256``) and compared to the frozen manifest entry
    (``loaders._expected_sha256``): a match is PASS; a MISMATCH — tamper / wrong build, what
    ``loaders._verify_checksum`` raises on at load time — is a **FAIL**. The licensed panel is legitimately
    out-of-repo, so an ABSENT parquet is a WARN, not a FAIL. If the checksum primitives cannot be imported
    (numpy/pandas absent) the checksum is not recomputed → WARN, never a silent PASS.
    """
    manifest = root / "data" / "manifest" / "manifest.jsonl"
    if not manifest.is_file():
        return {"name": "data provenance", "status": FAIL,
                "detail": "data/manifest/manifest.jsonl missing (the shipped checksum record is absent)"}
    n_entries = sum(1 for ln in manifest.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip())
    if n_entries == 0:
        return {"name": "data provenance", "status": FAIL,
                "detail": "data/manifest/manifest.jsonl is empty (no checksums recorded)"}

    suffix = _active_gold_suffix(root)
    parquet = root / "data" / "gold" / f"returns_panel_{suffix}.parquet"
    if not parquet.is_file():
        return {"name": "data provenance", "status": WARN,
                "detail": f"manifest OK ({n_entries} entries); {parquet.name} out-of-repo "
                          "(licensed panel not shipped) — checksum verifiable on the build host"}

    loaders = _import_loaders()
    if loaders is None:
        return {"name": "data provenance", "status": WARN,
                "detail": f"manifest OK ({n_entries} entries); {parquet.name} present but checksum NOT "
                          "re-verified (data deps absent)"}
    expected = loaders._expected_sha256(parquet, manifest)
    if expected is None:
        return {"name": "data provenance", "status": WARN,
                "detail": f"{parquet.name} present but has no manifest entry — integrity not verifiable"}
    actual = loaders._file_sha256(parquet)
    if actual == expected:
        return {"name": "data provenance", "status": PASS,
                "detail": f"{parquet.name} SHA-256 re-verified against manifest ({expected[:12]}…)"}
    return {"name": "data provenance", "status": FAIL,
            "detail": f"checksum MISMATCH for {parquet.name}: manifest {expected[:12]}… != "
                      f"on-disk {actual[:12]}… (tamper / wrong build)"}


CHECKS: tuple[Callable[[Path], dict[str, str]], ...] = (
    check_python_version,
    check_lockfile,
    check_version_pins,
    check_determinism_settings,
    check_seed_recording,
    check_freeze_state,
    check_archive_replay,
    check_data_provenance,
)


def audit_reproducibility(root: Path) -> dict[str, Any]:
    """Run every check against ``root``; return a structured report with pass/warn/fail tallies and ``ok``."""
    rows = [c(root) for c in CHECKS]
    n_pass = sum(r["status"] == PASS for r in rows)
    n_warn = sum(r["status"] == WARN for r in rows)
    n_fail = sum(r["status"] == FAIL for r in rows)
    return {
        "root": str(root),
        "checks": rows,
        "n_pass": n_pass,
        "n_warn": n_warn,
        "n_fail": n_fail,
        "ok": n_fail == 0,  # WARN does not fail the audit (pre-freeze etc. are legitimate)
    }


def render_report(result: dict[str, Any]) -> str:
    glyph = {PASS: "PASS", WARN: "WARN", FAIL: "FAIL"}
    lines = [
        "Reproducibility audit",
        f"  root: {result['root']}",
        "  " + "-" * 72,
    ]
    for r in result["checks"]:
        lines.append(f"  [{glyph[r['status']]}] {r['name']:<26} {r['detail']}")
    lines.append("  " + "-" * 72)
    lines.append(
        f"  {result['n_pass']} pass / {result['n_warn']} warn / {result['n_fail']} fail "
        f"=> {'OK' if result['ok'] else 'FAILED'}"
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit the repository's reproducibility contract.")
    ap.add_argument("--root", default=".", help="repo root (default: cwd)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--strict", action="store_true", help="exit non-zero on WARN as well as FAIL")
    args = ap.parse_args()

    result = audit_reproducibility(Path(args.root).resolve())
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(render_report(result))
    if not result["ok"]:
        return 1
    if args.strict and result["n_warn"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
