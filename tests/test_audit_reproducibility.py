"""Behaviour tests for the consolidated reproducibility audit (scripts/audit_reproducibility.py).

Checks each rule on a synthetic repo (pass/warn/fail discrimination) and that the audit runs on the REAL
repo root without crashing. The two REAL-verification pillars are exercised end-to-end:

  * pre-registration freeze — a frozen synthetic repo whose recorded ``freeze_hash`` MATCHES the recomputed
    ``freeze.canonical_hash`` PASSes; a MISMATCHED hash FAILs; a pre-freeze repo WARNs.
  * data provenance — a gold parquet whose SHA-256 matches its manifest entry PASSes; a corrupted parquet
    (checksum MISMATCH) FAILs; an out-of-repo (absent) panel WARNs; a missing manifest FAILs.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_reproducibility as A  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

_SEEDING = (
    "import os\nos.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'\nos.environ['PYTHONHASHSEED']='0'\n"
    "def set_global_seed(s):\n    torch.use_deterministic_algorithms(True)\n"
    "    torch.backends.cudnn.deterministic = True\n"
)


def _make_repo(tmp: Path, *, frozen: bool = False, freeze_hash: str = "null") -> Path:
    """A synthetic repo with every repro pillar present.

    Writes a real data manifest + a matching local gold parquet (so the data-provenance pillar is a REAL
    checksum match, not a keyword hit) and a minimal ``PREREGISTRATION.md`` + ``config/preregistration.yaml``
    (so ``freeze.canonical_hash`` can re-derive the design hash). ``freeze_hash`` is written verbatim into
    the yaml — tests set it to a matching / mismatching value to drive the freeze pillar.
    """
    (tmp / "config").mkdir()
    (tmp / "src" / "utils").mkdir(parents=True)
    (tmp / ".python-version").write_text("3.11.9\n", encoding="utf-8")
    (tmp / "requirements.lock").write_text("torch==2.6.0+cu124\nnumpy==1.26.4\n", encoding="utf-8")
    (tmp / "pyproject.toml").write_text(
        'requires-python = ">=3.11,<3.13"\ntorch = ">=2.6"\nnumpy = ">=1.26"\n', encoding="utf-8"
    )
    (tmp / "src" / "utils" / "seeding.py").write_text(_SEEDING, encoding="utf-8")
    (tmp / "config" / "campaign.yaml").write_text("winner_seeds: 30\nseed: 0\n", encoding="utf-8")
    (tmp / "config" / "llm.yaml").write_text("archive: true\n", encoding="utf-8")
    (tmp / "config" / "data.yaml").write_text(
        "gold:\n  suffix: univ5\nfreeze: {checksum: sha256}\nlicensing: redistribution_prohibited\n",
        encoding="utf-8",
    )
    (tmp / "docs").mkdir()
    (tmp / "docs" / "DATASHEET_v1.md").write_text("# datasheet\n", encoding="utf-8")

    # A shipped manifest + a matching local gold parquet -> data provenance is a REAL SHA-256 match.
    gold = tmp / "data" / "gold"
    gold.mkdir(parents=True)
    parquet = gold / "returns_panel_univ5.parquet"
    parquet.write_bytes(b"synthetic-gold-panel-bytes")
    sha = hashlib.sha256(parquet.read_bytes()).hexdigest()
    man_dir = tmp / "data" / "manifest"
    man_dir.mkdir(parents=True)
    (man_dir / "manifest.jsonl").write_text(
        json.dumps(
            {
                "relpath": "data/gold/returns_panel_univ5.parquet",
                "name": "returns_panel_univ5.parquet",
                "sha256": sha,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    # Pre-registration prose + yaml. PREREGISTRATION.md is required by freeze.canonical_hash.
    (tmp / "PREREGISTRATION.md").write_text(
        "# Pre-registration\nSynthetic design record for the reproducibility-audit tests.\n", encoding="utf-8"
    )
    (tmp / "config" / "preregistration.yaml").write_text(
        f"frozen: {'true' if frozen else 'false'}\nfreeze_hash: {freeze_hash}\n", encoding="utf-8"
    )
    return tmp


def _set_freeze_hash(repo: Path, value: str) -> None:
    """Rewrite the ``freeze_hash:`` line in the synthetic prereg yaml (leaves everything else intact)."""
    path = repo / "config" / "preregistration.yaml"
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"(?m)^(freeze_hash:\s*).*$", rf"\g<1>{value}", text, count=1)
    path.write_text(text, encoding="utf-8")


def _statuses(res: dict) -> dict[str, str]:
    return {r["name"]: r["status"] for r in res["checks"]}


# --------------------------------------------------------------------------- #
# Whole-repo discrimination                                                    #
# --------------------------------------------------------------------------- #
def test_complete_prefreeze_repo_passes_with_one_freeze_warn(tmp_path: Path) -> None:
    res = A.audit_reproducibility(_make_repo(tmp_path, frozen=False))
    assert res["ok"] is True              # no hard fails
    assert res["n_fail"] == 0
    st = _statuses(res)
    assert st["pre-registration freeze"] == A.WARN     # pre-freeze is a WARN, by design
    assert st["determinism settings"] == A.PASS
    assert st["dependency lockfile"] == A.PASS


def test_pinless_lockfile_does_not_pass_and_does_not_mask_a_real_one(tmp_path: Path) -> None:
    """A lockfile that pins NOTHING certifies nothing (deep review 2026-07-26).

    The check used to PASS on the first NON-EMPTY candidate: a comment-only ``requirements.lock``
    computed ``n = 0``, reported "(0 pinned lines)", and still returned PASS — a reproducibility claim
    made on a file pinning nothing, against this module's own "never a silent PASS" contract. The
    early return also let that pin-less file MASK a genuine ``uv.lock`` later in the search order.
    """
    (tmp_path / "requirements.lock").write_text("# pins go here\n# TODO: regenerate\n", encoding="utf-8")
    res = A.check_lockfile(tmp_path)
    assert res["status"] == A.FAIL, res
    assert "pin NOTHING" in res["detail"]

    # A real lockfile further down the search order must still be found, not shadowed.
    (tmp_path / "uv.lock").write_text("numpy==1.26.4\ntorch==2.3.1\n", encoding="utf-8")
    res2 = A.check_lockfile(tmp_path)
    assert res2["status"] == A.PASS and "uv.lock (2 pinned lines)" in res2["detail"], res2


def test_missing_pillars_fail(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    res = A.audit_reproducibility(tmp_path)  # almost-empty repo
    assert res["ok"] is False
    assert res["n_fail"] >= 2
    names_failed = {r["name"] for r in res["checks"] if r["status"] == A.FAIL}
    assert "python-version pin" in names_failed
    assert "dependency lockfile" in names_failed
    assert "data provenance" in names_failed   # no manifest -> FAIL (not a silent WARN)


def test_determinism_partial_is_warn_not_fail(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    # drop exactly ONE determinism knob -> WARN (one missing), not FAIL (>=2 missing)
    (repo / "src" / "utils" / "seeding.py").write_text(
        "import os\nos.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'\nos.environ['PYTHONHASHSEED']='0'\n"
        "torch.use_deterministic_algorithms(True)\n",  # cudnn.deterministic absent
        encoding="utf-8",
    )
    det = next(r for r in A.audit_reproducibility(repo)["checks"] if r["name"] == "determinism settings")
    assert det["status"] == A.WARN


# --------------------------------------------------------------------------- #
# Pillar 1 — pre-registration freeze: REAL hash re-computation                 #
# --------------------------------------------------------------------------- #
def test_prefreeze_freeze_check_warns(tmp_path: Path) -> None:
    freeze = next(
        r for r in A.audit_reproducibility(_make_repo(tmp_path, frozen=False))["checks"]
        if r["name"] == "pre-registration freeze"
    )
    assert freeze["status"] == A.WARN
    assert "pre-freeze" in freeze["detail"]


def test_frozen_matching_hash_passes(tmp_path: Path) -> None:
    if A._import_freeze() is None:
        pytest.skip("freeze module (PyYAML) unavailable")
    repo = _make_repo(tmp_path, frozen=True)
    # The canonical hash strips the freeze_hash line before hashing, so recording it back cannot change it.
    real = A._import_freeze().canonical_hash(repo)
    _set_freeze_hash(repo, real)
    res = A.audit_reproducibility(repo)
    freeze = next(r for r in res["checks"] if r["name"] == "pre-registration freeze")
    assert freeze["status"] == A.PASS, freeze["detail"]
    assert res["ok"] is True


def test_frozen_mismatched_hash_fails(tmp_path: Path) -> None:
    if A._import_freeze() is None:
        pytest.skip("freeze module (PyYAML) unavailable")
    # A frozen repo whose recorded hash matches NOTHING must now FAIL (was a bogus PASS before).
    repo = _make_repo(tmp_path, frozen=True, freeze_hash="0" * 64)
    res = A.audit_reproducibility(repo)
    freeze = next(r for r in res["checks"] if r["name"] == "pre-registration freeze")
    assert freeze["status"] == A.FAIL, freeze["detail"]
    assert "DRIFT" in freeze["detail"]
    assert res["ok"] is False


def test_frozen_null_hash_fails(tmp_path: Path) -> None:
    # frozen: true but no hash recorded is a broken freeze, not a legitimate pending state -> FAIL.
    repo = _make_repo(tmp_path, frozen=True, freeze_hash="null")
    freeze = next(
        r for r in A.audit_reproducibility(repo)["checks"] if r["name"] == "pre-registration freeze"
    )
    assert freeze["status"] == A.FAIL


# --------------------------------------------------------------------------- #
# Pillar 2 — data provenance: REAL SHA-256 checksum comparison                 #
# --------------------------------------------------------------------------- #
def test_data_provenance_matches_manifest_passes(tmp_path: Path) -> None:
    if A._import_loaders() is None:
        pytest.skip("data loader (numpy/pandas) unavailable")
    dp = next(
        r for r in A.audit_reproducibility(_make_repo(tmp_path))["checks"] if r["name"] == "data provenance"
    )
    assert dp["status"] == A.PASS, dp["detail"]
    assert "re-verified" in dp["detail"]


def test_data_provenance_mismatch_fails(tmp_path: Path) -> None:
    if A._import_loaders() is None:
        pytest.skip("data loader (numpy/pandas) unavailable")
    repo = _make_repo(tmp_path)
    # Corrupt the gold parquet so its SHA-256 no longer matches the frozen manifest entry -> FAIL.
    (repo / "data" / "gold" / "returns_panel_univ5.parquet").write_bytes(b"tampered-bytes")
    res = A.audit_reproducibility(repo)
    dp = next(r for r in res["checks"] if r["name"] == "data provenance")
    assert dp["status"] == A.FAIL, dp["detail"]
    assert "MISMATCH" in dp["detail"]
    assert res["ok"] is False


def test_data_provenance_out_of_repo_panel_warns(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    # The licensed panel is legitimately out-of-repo: manifest present, parquet absent -> WARN, not FAIL.
    (repo / "data" / "gold" / "returns_panel_univ5.parquet").unlink()
    res = A.audit_reproducibility(repo)
    dp = next(r for r in res["checks"] if r["name"] == "data provenance")
    assert dp["status"] == A.WARN
    assert "out-of-repo" in dp["detail"]
    assert res["ok"] is True   # a WARN must not fail the audit


def test_data_provenance_missing_manifest_fails(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    (repo / "data" / "manifest" / "manifest.jsonl").unlink()
    dp = next(
        r for r in A.audit_reproducibility(repo)["checks"] if r["name"] == "data provenance"
    )
    assert dp["status"] == A.FAIL


# --------------------------------------------------------------------------- #
# Real repo                                                                    #
# --------------------------------------------------------------------------- #
def test_runs_on_the_real_repo_without_crashing() -> None:
    res = A.audit_reproducibility(REPO)
    assert set(res) >= {"checks", "n_pass", "n_warn", "n_fail", "ok"}
    assert len(res["checks"]) == len(A.CHECKS)
    st = _statuses(res)
    # Core pillars present + earned.
    assert st["determinism settings"] == A.PASS
    assert st["dependency lockfile"] == A.PASS
    # STATE-ADAPTIVE (ADR-059): PASS when frozen-and-matching, WARN when pre-freeze — both honest
    # non-FAIL states of the live repo; a FAIL (frozen-but-drifted) is the only wrong answer.
    assert st["pre-registration freeze"] in (A.PASS, A.WARN)
    assert st["data provenance"] in (A.PASS, A.WARN)   # PASS when the panel is on disk; WARN on a bare clone
    assert res["ok"] is True
    assert A.render_report(res).startswith("Reproducibility audit")
