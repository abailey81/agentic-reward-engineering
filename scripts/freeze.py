"""Pre-registration freeze (FINAL_PLAN Phase 1.E; punch-list Rank 9).

Purpose
-------
Lock the experimental design before any results are seen. The freeze is a GATE:
``PREREGISTRATION.md`` (the human-readable prose record) and
``config/preregistration.yaml`` (its machine-readable mirror) must agree on every
freeze-relevant field, Phase 0 must have passed, and only then is a single canonical
SHA-256 over the pre-registration artifacts AND the executed load-bearing configs
recorded as the immutable design hash.

The canonical hash
------------------
SHA-256 over the **LF-normalized UTF-8 bytes** of the hashed artifacts concatenated in a
FIXED, documented order (``norm`` = strip-BOM + CRLF/CR -> LF; ``strip_state`` = blank the
two MUTABLE ``frozen``/``freeze_hash`` lines so the hash is INVARIANT to the freeze act)::

    canonical_bytes = norm(PREREGISTRATION.md)
                      + b"\n" + strip_state(norm(config/preregistration.yaml))
                      + b"\n" + norm(config/inference.yaml)        # if present  ┐ _BOUND_CONFIGS
                      + b"\n" + norm(config/environment.yaml)      # if present  │ (executed knobs)
                      + b"\n" + norm(config/data.yaml)             # if present  ┘
                      + b"\n" + norm(config/arms.yaml)             # if present  ┐ _BOUND_TREATMENT
                      + b"\n" + norm(prompts/system.txt)           # if present  │ (the manipulated
                      + b"\n" + norm(prompts/initial_generation.txt) # if present│  variable itself)
                      + b"\n" + norm(src/feedback/schema.py)       # if present  ┘  (#97/R114: the
                      #   RENDERER. The designer never sees a float, it sees a STRING, so the
                      #   rendered precision IS part of the stimulus and must be bound with it.

i.e. **prose, THEN the freeze-state-stripped prereg yaml, THEN the three bound configs, THEN the four
bound treatment files** (see ``_BOUND_CONFIGS`` + ``_BOUND_TREATMENT``) — each joined by a single ``\n``
(LF) record separator. The bound configs are the EXECUTED knobs the campaign reads (splits/embargo/
lookback/family); the bound treatment is the per-arm feedback spec + the two loaded prompts (the
manipulated variable's text, R62). Binding both stops the frozen DESIGN drifting from the IMPLEMENTED
design (critical-review 2026-06-20; treatment added 2026-06-28). A bound file absent on a minimal root is
skipped, so a two-file (prose + prereg-yaml) root still hashes — this module is the single definition of
that order; nothing else may re-define it.

The prose<->YAML gate (the assertion)
-------------------------------------
Both files are parsed and asserted to agree on the freeze-relevant fields:

  - seed count                         -> ``len(yaml['seeds'])`` vs the prose "5->30 / [0..29]"
  - ``inference.testing_family.m``     -> the union size (6) vs the enumerated member rows; the two
                                          co-primary IUT sub-families (H2-RA m=3, H2-Tail m=3; R25) must
                                          partition it, and every prose "m = <N>" must be a declared size
                                          ({3, 6}) so a stray "m = 9" still fails
  - ``inference.difference_tests``     -> each label has its prose anchor (R11 relabel)
  - ``inference.sesoi``                -> vs the prose "SESOI = 0.05 validation-DSR units"
  - ``inference.equivalence_margin``   -> vs the prose "symmetric TOST ... +/-0.05 DSR"
  - top-level ``cost_sweep.grid_bps``  -> vs the prose "grid_bps = [0, 5, 10, 25, 50]"

Any mismatch raises :class:`FreezeConsistencyError` with a clear message naming the field.

Phase-0 precondition
--------------------
The freeze REFUSES unless ``phase0_smoke_passed_log_id`` is set in the YAML (the smoke
GATE must have gone GREEN/AMBER first). A missing/blank marker raises
:class:`FreezePreconditionError`.

Modes
-----
  --check   Recompute the hash + re-run every assertion WITHOUT writing anything (CI/drift
            guard). Exit code 0 == consistent + Phase-0 met; non-zero on any drift/failure.
  (default) The real freeze (the write path): set ``frozen: true`` + ``freeze_hash`` in the
            YAML, append the hash + UTC + git SHA to ``provenance/DECISION_LOG.md`` (the ADR-005
            slot), create a signed git tag ``prereg-v1.0`` (best-effort), and OpenTimestamp
            the hash (best-effort). This path is implemented but MUST be run only by the user.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# scripts/ is not a package; make ``src`` importable when run as a file.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.provenance import git_commit, sha256_bytes  # noqa: E402

# --------------------------------------------------------------------------- #
# Paths (repo-root relative; resolved robustly)                                #
# --------------------------------------------------------------------------- #
PREREG_MD = "PREREGISTRATION.md"
PREREG_YAML = "config/preregistration.yaml"
#: The EXECUTED load-bearing config the campaign actually reads — bound into the freeze hash so the frozen
#: DESIGN cannot silently drift from the IMPLEMENTED knobs (splits/embargo/lookback/family live HERE, not in
#: the prereg yaml; the hash previously covered only the prereg, so 'nothing frozen can drift' was false at
#: the config layer — critical-review 2026-06-20). Missing files (a minimal test root) are skipped.
_BOUND_CONFIGS: tuple[str, ...] = (
    "config/inference.yaml",   # splits, embargo, testing_family (m=6), multiplicity, sesoi, dsr
    "config/environment.yaml", # lookback, vol windows, action projection/bound, costs, cash_daily_rate
    "config/data.yaml",        # splits, embargo_days
)

#: The DESIGN-DEFINING treatment text — the per-arm feedback spec (``config/arms.yaml``) and the human-
#: authored prompt templates LOADED at run time (``prompts/system.txt`` + ``prompts/initial_generation
#: .txt``). These literally constitute the manipulated variable: distributional-vs-scalar feedback is
#: rendered through them, yet the original canonical hash bound NEITHER — so the treatment could be
#: edited post-freeze without tripping ``--check`` (deep-audit 2026-06-28, amendment R62). Binding their
#: CONTENT closes that gap. ``arms.yaml`` is ALSO roster-checked (``_ARM_ROSTER_CONFIGS``); hashing its
#: bytes here is independent of, and complementary to, that key-level guard. ``prompts/reflection.txt`` is
#: deliberately EXCLUDED: it is dead (no runtime path loads it — the reflection turn is built in-code from
#: ``src/llm/prompts._REFLECTION_PREAMBLE`` + ``src/feedback/schema.build_block``) and is archived (R63).
#: The IN-CODE treatment surface (``schema.build_block``, the preamble) is pinned by the git SHA recorded
#: at the freeze, not by this content hash. A file absent on a minimal test root is skipped.
#:
#: IMPORTANT: 2026-07-27 (deep review, finding #97) — THAT LAST SENTENCE OVERSTATES THE PROTECTION, and the
#: gap it hides is the same class as the R62 one the paragraph above celebrates closing. VERIFIED by
#: reading every use: ``git_commit()`` is called EXACTLY ONCE in this file (line ~1334) and only to
#: WRITE the decision-log entry. **No check anywhere compares a recorded SHA against HEAD** — grep for
#: an assert on it returns nothing, and none of the 23 gate checks is a SHA check. So the git SHA is
#: RECORDED, never VERIFIED: `--check` cannot detect a post-freeze edit to ``src/feedback/schema.py``.
#: The pin is therefore ARCHIVAL (you could look the SHA up afterwards), not ENFORCED.
#:
#: Why this specifically matters: ``schema.build_block`` RENDERS the fed text, so it IS the manipulated
#: variable, not a wrapper around it. Finding #87 (same review) proved the point empirically — changing
#: ONE format string in that file decided whether the scalar arm received a usable signal at all
#: (`{metric:.2f}` made 55 % of real rendered headers read literally "0.00"). Pre-freeze that was a
#: legitimate fix; POST-freeze the identical edit would pass this gate silently. ``arms.yaml`` binds
#: WHICH block each arm gets; nothing binds HOW its numbers are rendered.
#:
#: NOT fixed here, deliberately: adding ``src/feedback/schema.py`` to ``_BOUND_TREATMENT`` is a one-line
#: change that closes it completely and costs nothing while ``frozen: false`` (the hash moves freely
#: pre-freeze) — but it WIDENS THE FREEZE ENVELOPE, which is a design decision, not a tidying-up.
#: What IS fixed here is the false claim of protection: a comment
#: asserting a guard that does not exist is worse than no comment, because it stops the next reader looking.
_BOUND_TREATMENT: tuple[str, ...] = (
    "config/arms.yaml",               # the per-arm feedback spec (the manipulated variable's wiring)
    "prompts/system.txt",             # the reward-design contract shown to every arm
    "prompts/initial_generation.txt", # the generation-0 instruction shown to every arm
    # #97 CLOSED 2026-07-27 under the close-all-gaps instruction (previously left open
    # because widening the envelope was his call). ``schema.build_block`` RENDERS the fed text, so it
    # IS the manipulated variable, not a wrapper around it — ``arms.yaml`` binds WHICH block each arm
    # gets, and until now NOTHING bound HOW its numbers are rendered. The git-SHA "pin" the comment
    # above once relied on is ARCHIVAL, never verified: no gate check compares a recorded SHA to HEAD.
    # Finding #87 proved the stakes empirically — changing ONE format string in this file decided
    # whether the scalar arm received a usable signal at all (`{metric:.2f}` made 55 % of real
    # rendered headers read literally "0.00"). Pre-freeze that was a legitimate fix; POST-freeze the
    # identical edit would have passed the gate silently. Binding it costs nothing while
    # ``frozen: false`` (the hash moves freely) and makes the treatment surface tamper-evident.
    "src/feedback/schema.py",         # RENDERS the fed numbers — the treatment surface itself (#97)
)

#: Executed configs whose ``arms`` roster must AGREE with the frozen prereg roster (V1 cross-file guard,
#: 2026-06-26). These are deliberately NOT bound into the canonical hash — operational compute knobs in
#: ``campaign.yaml`` (gpu, budget, resume, ...) must stay amendable post-freeze without a design amendment —
#: but their ARM LIST is asserted equal to the frozen design's ``arms`` so the campaign can never silently
#: run a DIFFERENT roster than the one frozen (the ``placebo_shuffled`` 6-vs-7 drift, DEEP_AUDIT V1/V2).
#: ``arms.yaml`` carries the roster as the KEYS of its ``arms`` mapping; ``campaign.yaml`` as a list. Either
#: file absent on a minimal test root is skipped (the prereg-only fixture still verifies).
_ARM_ROSTER_CONFIGS: tuple[str, ...] = (
    "config/campaign.yaml",    # the executed campaign roster (list under `arms:`)
    "config/arms.yaml",        # the per-arm feedback spec (roster = keys of the `arms:` mapping)
)
DECISION_LOG = "provenance/DECISION_LOG.md"

#: The annotated/signed git tag stamped at the freeze.
#: The git tag + OpenTimestamps proof name for THIS freeze.
#:
#: NOTE: **v2.0, changed 2026-07-27 at the launch gate, and the change is load-bearing.** The value was
#: ``prereg-v1.0``, and that tag ALREADY EXISTS in this repository from the 2026-07-18 v1.0 freeze
#: that R78/ADR-059 later lifted. Two silent failures followed from that, both verified:
#:
#: 1. ``_git_tag_signed`` is best-effort by design, so ``git tag -s`` AND its ``git tag -a``
#:    fallback would both have failed with "tag already exists" and the freeze would have reported
#:    ``"tag SKIPPED"`` — i.e. the v2 design would have had **no git provenance anchor at all**,
#:    while the freeze receipt in ``DECISION_LOG.md`` still named a tag pointing at the OLD design.
#: 2. ``_ots_stamp`` writes ``provenance/<FREEZE_TAG>.sha256``, so it would have **OVERWRITTEN**
#:    ``provenance/prereg-v1.0.sha256`` — a committed artifact holding v1.0's digest ``ce5db62c…`` — with
#:    the v2 digest, under the v1 name. The one file whose entire job is to prove which bytes were
#:    registered when would have been made to attest the wrong thing.
#:
#: A pre-registration proof that is absent, or that quietly re-labels a superseded one, is worse
#: than no proof: it is the reproducibility claim (Stefan's criterion 3) failing at exactly the
#: point it is supposed to hold. The v1.0 tag and its ``.sha256`` stay untouched as the historical
#: record of a freeze that was lifted pre-data.
#:
#: 2026-07-28 → ``prereg-v2.1``. The v2.0 freeze (``4f90ecc4…``, 00:05Z) was LIFTED PRE-DATA the same
#: day to register the R115 winner-eligibility execution floor (ADR-062), so the design re-freezes
#: under a NEW tag. Bumping this is not bookkeeping: leaving it at ``prereg-v2.0`` reproduces the
#: exact failure the block above documents — the tag already exists, so the freeze would report
#: "tag SKIPPED" leaving v2.1 with NO provenance anchor, and ``_ots_stamp`` would OVERWRITE
#: ``provenance/prereg-v2.0.sha256`` (which holds v2.0's digest, the attestation for the bytes RUN 1 and
#: RUN 2 actually executed) with the v2.1 digest under the v2.0 name. The guard in
#: ``tests/test_launch_gate_regressions.py::test_the_freeze_tag_is_not_one_that_already_exists``
#: caught precisely this before the freeze ran. v2.0's tag and .sha256 stay untouched as history.
FREEZE_TAG = "prereg-v2.1"

#: The marker line in ``provenance/DECISION_LOG.md`` whose body the freeze fills (ADR-005 slot).
_DECISION_LOG_FREEZE_HEADER = "### FREEZE — pre-registration content hash"

#: Written when the log is absent, so the receipt always has an append-only home.
_DECISION_LOG_SEED = (
    "# Decision log\n\n"
    "Append-only record of the pre-registration freeze and every dated post-freeze\n"
    "amendment. One entry per decision; newest at the bottom.\n\n"
    "<!-- amendments appended below this line -->\n"
)


def repo_root() -> Path:
    """Repository root (the directory holding ``config/`` + ``pyproject.toml``)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("could not locate repo root (no parent has both config/ and pyproject.toml)")


# --------------------------------------------------------------------------- #
# Errors                                                                       #
# --------------------------------------------------------------------------- #
class FreezeError(RuntimeError):
    """Base class for any freeze-gate failure."""


class FreezePreconditionError(FreezeError):
    """A precondition for freezing is not met (e.g. Phase-0 not recorded)."""


class FreezeConsistencyError(FreezeError):
    """PREREGISTRATION.md prose and preregistration.yaml disagree on a frozen field."""


# --------------------------------------------------------------------------- #
# Canonical bytes + hash                                                       #
# --------------------------------------------------------------------------- #
def _normalize_bytes(raw: bytes) -> bytes:
    """LF-normalize UTF-8 bytes: strip a leading BOM, rewrite CRLF/CR -> LF.

    Makes the content hash invariant to OS line endings / a BOM-prefixed checkout, so the
    same logical artifact hashes identically on Windows-dev and the Linux campaign box.
    """
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    # Collapse ANY run of CR(s) + an optional trailing LF to a single LF — invariant not only to CRLF/CR
    # but to a doubled `\r\r\n` (a CRLF checkout re-rewritten as CRLF, e.g. the line-ending test). Identical
    # output to the prior replace-chain for real LF/CRLF files (canonical hash UNCHANGED), and idempotent.
    return re.sub(rb"\r+\n?", b"\n", raw)


def _strip_freeze_state(yml_text: str) -> str:
    """Blank out the two MUTABLE freeze-state fields so the canonical hash is invariant to the act of
    freezing. ``_set_yaml_frozen`` flips ``frozen: false->true`` and ``freeze_hash: null-><digest>`` IN the
    hashed file; hashing those raw bytes made ``--check`` report DRIFT forever post-freeze (it recorded the
    PRE-flip digest then re-hashed the POST-flip file). We hash the frozen DESIGN content, not the
    freeze-state bookkeeping, by normalising both lines to a fixed placeholder (critical-review 2026-06-20).
    """
    import re

    out = re.sub(r"(?m)^(frozen:\s*).*$", r"\1<FREEZE_STATE>", yml_text, count=1)
    out = re.sub(r"(?m)^(freeze_hash:\s*).*$", r"\1<FREEZE_STATE>", out, count=1)
    return out


def canonical_bytes(root: Path | None = None) -> bytes:
    """Return the canonical byte string the freeze hashes (FIXED order; see the module docstring).

    Order: ``norm(PREREGISTRATION.md)``, then the freeze-state-stripped prereg yaml (the two MUTABLE
    ``frozen``/``freeze_hash`` fields blanked so the hash is INVARIANT to the freeze act), then each of
    ``_BOUND_CONFIGS`` (inference/environment/data), then each of ``_BOUND_TREATMENT`` (arms.yaml +
    the two loaded prompts) that is present — joined by a single ``\\n`` record separator so each boundary
    is unambiguous and order-sensitive (bound configs added 2026-06-20; bound treatment added 2026-06-28).
    """
    root = root or repo_root()
    md = _normalize_bytes((root / PREREG_MD).read_bytes())
    yml_text = _strip_freeze_state(_normalize_bytes((root / PREREG_YAML).read_bytes()).decode("utf-8"))
    parts: list[bytes] = [md, yml_text.encode("utf-8")]
    # Bind the EXECUTED load-bearing config (skip any absent on a minimal root, so the prereg-only hash
    # is preserved for tests that supply just the two prereg files).
    for rel in _BOUND_CONFIGS:
        p = root / rel
        if p.exists():
            parts.append(_normalize_bytes(p.read_bytes()))
    # Then bind the treatment-defining text (per-arm feedback spec + the two loaded prompts) in a fixed,
    # documented order AFTER the configs, so the frozen DESIGN includes the manipulated variable itself
    # (R62). Absent on a minimal root -> skipped, preserving the prereg-only hash for the freeze tests.
    for rel in _BOUND_TREATMENT:
        p = root / rel
        if p.exists():
            parts.append(_normalize_bytes(p.read_bytes()))
    return b"\n".join(parts)


def canonical_hash(root: Path | None = None) -> str:
    """SHA-256 hex digest of :func:`canonical_bytes` (the frozen design hash).

    Covers up to NINE files in fixed order — prose, freeze-state-stripped prereg yaml, the
    three bound configs (inference/environment/data), then the four bound treatment files
    (arms.yaml + prompts/system.txt + prompts/initial_generation.txt + src/feedback/schema.py),
    each only if present; see the module docstring, ``_BOUND_CONFIGS`` and ``_BOUND_TREATMENT``
    (R62, 2026-06-28; widened EIGHT -> NINE by #97 on 2026-07-27, which bound the fed-text
    RENDERER — ``arms.yaml`` binds WHICH block an arm gets, this binds HOW its numbers are shown).
    """
    return sha256_bytes(canonical_bytes(root))


# --------------------------------------------------------------------------- #
# Parsing                                                                      #
# --------------------------------------------------------------------------- #
def load_yaml(root: Path | None = None) -> dict[str, Any]:
    """Parse ``config/preregistration.yaml`` (LF-normalized) into a plain dict."""
    root = root or repo_root()
    text = _normalize_bytes((root / PREREG_YAML).read_bytes()).decode("utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise FreezeConsistencyError(f"{PREREG_YAML} did not parse to a mapping")
    return data


def load_prose(root: Path | None = None) -> str:
    """Read ``PREREGISTRATION.md`` (LF-normalized UTF-8 text)."""
    root = root or repo_root()
    return _normalize_bytes((root / PREREG_MD).read_bytes()).decode("utf-8")


def _prose_floats(prose: str, pattern: str) -> list[float]:
    """Every float captured by ``pattern`` (one capturing group) in ``prose``."""
    return [float(m) for m in re.findall(pattern, prose)]


def _prose_grid(prose: str) -> list[int] | None:
    """The cost-sweep bps grid as written in the prose: ``grid_bps = [0, 5, 10, 25, 50]``."""
    m = re.search(r"grid_bps\s*=\s*\[([0-9,\s]+)\]", prose)
    if not m:
        return None
    return [int(x) for x in re.findall(r"\d+", m.group(1))]


# --------------------------------------------------------------------------- #
# The gate: prose <-> yaml consistency on the freeze-relevant fields           #
# --------------------------------------------------------------------------- #
def _require(condition: object, message: str) -> None:
    if not condition:
        raise FreezeConsistencyError(message)


def assert_prose_matches_yaml(yml: dict[str, Any], prose: str, root: Path) -> list[str]:
    """Assert the PROSE record agrees with the YAML mirror on every frozen field.

    ``root`` anchors the cross-config reads (the data_panel check reads ``root/config/data.yaml``, batch-6
    M2, 2026-07-03 — the old bare ``Path("config/data.yaml")`` resolved against the CWD, so the guard was
    correct only when run from the repo root and could not be tested hermetically from a mini-repo root).

    Returns a list of human-readable "checked + agreed" lines (for the report); raises
    :class:`FreezeConsistencyError` naming the first field that disagrees.
    """
    checked: list[str] = []
    inf = yml.get("inference") or {}

    # 1) Seed count -------------------------------------------------------- #
    # B-A3: 'seeds' may be a bare list OR a schema ({mode: uniform/tiered, …}). Resolve it to the flat
    # [0..N-1] set the campaign + freeze both bind on (resolve_seeds is the SINGLE source both use, so
    # they cannot disagree on n); a malformed schema fails the freeze LOUD, not silently.
    from src.utils.seeds import SeedSchemaError, resolve_seeds

    try:
        seeds = resolve_seeds(yml.get("seeds"))
    except SeedSchemaError as exc:
        raise FreezeConsistencyError(f"yaml 'seeds' is not a valid seed list/schema: {exc}") from exc
    n_seeds = len(seeds)
    _require(
        seeds == list(range(n_seeds)),  # guaranteed by resolve_seeds; belt-and-suspenders
        f"resolved seeds must be the contiguous [0..{n_seeds - 1}] block, got {seeds[:5]}…",
    )
    # Prose binds the headline count via amendment D2 ("5->30", "[0..29]", "headline seed count 30").
    prose_seed_counts = {
        int(x)
        for x in re.findall(r"5\s*(?:->|→)\s*(\d+)", prose)
        + re.findall(r"headline seed count\s+(\d+)", prose)
        + re.findall(r"target\s+\*\*(\d+)\*\*", prose)
    }
    prose_high = max(prose_seed_counts) if prose_seed_counts else None
    _require(
        prose_high == n_seeds,
        f"seed count mismatch: yaml has {n_seeds} seeds ([0..{n_seeds - 1}]) but the prose "
        f"amendment states a headline count of {prose_high}",
    )
    # The "[0..N-1]" literal is also written in the prose; check it agrees.
    prose_ranges = re.findall(r"\[0\.\.(\d+)\]", prose)
    if prose_ranges:
        _require(
            all(int(hi) == n_seeds - 1 for hi in prose_ranges),
            f"seed range mismatch: yaml is [0..{n_seeds - 1}] but prose writes [0..{prose_ranges}]",
        )
    checked.append(f"seeds: yaml [0..{n_seeds - 1}] (n={n_seeds}) == prose 5->{prose_high} / [0..{n_seeds - 1}]")

    # 1b) Arm roster: every frozen arm is named in the prose, and §3's count word matches len(arms).
    # (V1 reconcile 2026-06-26: the frozen roster silently dropped `placebo_shuffled` — the headline
    # structure-vs-content control — while the campaign ran 7 arms; this guard makes that drift fail loud.)
    arms = yml.get("arms")
    if not isinstance(arms, list) or not arms:
        raise FreezeConsistencyError("yaml 'arms' must be a non-empty list")
    n_arms = len(arms)
    missing_arms = [a for a in arms if str(a) not in prose]
    _require(
        not missing_arms,
        f"arm roster mismatch: yaml lists {arms} but the prose never names {missing_arms}",
    )
    _ARM_WORD = {5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
    arm_word = _ARM_WORD.get(n_arms)
    _require(
        arm_word is not None and re.search(rf"##\s*3\.\s*The\s+{arm_word}\s+arms", prose) is not None,
        f"arm count mismatch: yaml has {n_arms} arms but §3 prose does not read 'The {arm_word} arms'",
    )
    checked.append(f"arms: yaml n={n_arms} ({list(arms)}) all named in prose; §3 == 'The {arm_word} arms'")

    # 2) inference.testing_family: the m=6 union + the two co-primary IUT sub-families (R25) ----- #
    fam = inf.get("testing_family") or {}
    m_yaml = fam.get("m")
    members = fam.get("members") or []
    if not isinstance(m_yaml, int):
        raise FreezeConsistencyError("yaml 'inference.testing_family.m' must be an int")
    _require(
        len(members) == m_yaml,
        f"yaml testing_family.m={m_yaml} but {len(members)} members are enumerated",
    )
    # R25: the headline gate is TWO co-primary intersection-union tests (H2-RA on Sharpe, H2-Tail on
    # CVaR-5%). Validate the `families` block: each sub-family's m equals its member count, the two are
    # DISJOINT, and together they partition the m=6 union. `declared_sizes` is the set of family sizes
    # the prose may legitimately reference (each sub-family's m + the union m).
    families = fam.get("families") if isinstance(fam.get("families"), dict) else {}
    declared_sizes = {m_yaml}
    if families:
        def _member_key(mem: dict[str, Any]) -> tuple[str, str, str, object]:
            return (str(mem["arm_a"]), str(mem["arm_b"]), str(mem["metric"]), mem.get("level"))

        union_keys = {_member_key(mem) for mem in members}
        sub_union: set[tuple[str, str, str, object]] = set()
        for sub_name, sub in families.items():
            sub_m = sub.get("m")
            sub_members = sub.get("members") or []
            if not isinstance(sub_m, int):
                raise FreezeConsistencyError(
                    f"yaml 'inference.testing_family.families.{sub_name}.m' must be an int"
                )
            _require(
                len(sub_members) == sub_m,
                f"yaml testing_family.families.{sub_name}.m={sub_m} but {len(sub_members)} members enumerated",
            )
            keys = {_member_key(mem) for mem in sub_members}
            _require(
                sub_union.isdisjoint(keys),
                f"testing_family co-primary sub-families overlap at {sorted(map(str, sub_union & keys))} "
                "(the two IUTs must be DISJOINT; R25)",
            )
            sub_union |= keys
            declared_sizes.add(sub_m)
        _require(
            sub_union == union_keys,
            "testing_family.families do not partition the m=6 union (R25): "
            f"sub-only {sorted(map(str, sub_union - union_keys))}, "
            f"union-only {sorted(map(str, union_keys - sub_union))}",
        )
        _require(
            sum(int(families[s]["m"]) for s in families) == m_yaml,
            f"testing_family sub-family sizes {[families[s]['m'] for s in families]} do not sum to m={m_yaml}",
        )
    prose_m = re.findall(r"\bm\s*=\s*\*{0,2}(\d+)\*{0,2}", prose)
    _require(bool(prose_m), "prose does not state the testing-family size 'm = <N>'")
    # Every prose 'm = <N>' must be one of the DECLARED family sizes (the union 6 + each IUT's 3); a stray
    # 'm = 9' (or any undeclared size) still fails the gate, exactly as the single-m check did.
    _require(
        all(int(v) in declared_sizes for v in prose_m),
        f"testing_family m mismatch: prose states m={set(prose_m)} but the declared family sizes are "
        f"{sorted(declared_sizes)} (union m={m_yaml}; R25 co-primary IUTs)",
    )
    _require(
        m_yaml in {int(v) for v in prose_m},
        f"testing_family union size m={m_yaml} is not stated in the prose (found m={set(prose_m)})",
    )
    checked.append(
        f"inference.testing_family.m: yaml union {m_yaml} (== {len(members)} members) + IUT sub-families "
        f"{sorted(declared_sizes - {m_yaml})} == prose m={sorted(set(int(v) for v in prose_m))}"
    )

    # 3) inference.difference_tests (R11 relabel) -------------------------- #
    diff = inf.get("difference_tests")
    if not isinstance(diff, list) or not diff:
        raise FreezeConsistencyError("yaml 'inference.difference_tests' must be a non-empty list")
    # Each frozen label must have its prose anchor (so a silent relabel in one file is caught).
    diff_anchors = {
        "sharpe_recentred_bootstrap": r"re-?centred basic",
        "cvar_difference": r"CVaR-?difference",
    }
    for label in diff:
        anchor = diff_anchors.get(label)
        if anchor is None:
            # Unknown label: require it to appear verbatim somewhere in the prose.
            _require(
                label.replace("_", " ") in prose or label in prose,
                f"difference_tests label '{label}' has no anchor in the prose record",
            )
        else:
            _require(
                re.search(anchor, prose) is not None,
                f"difference_tests label '{label}' present in yaml but its prose anchor "
                f"/{anchor}/ is absent (a silent relabel?)",
            )
    checked.append(f"inference.difference_tests: {diff} — each label anchored in prose")

    # 4) inference.sesoi --------------------------------------------------- #
    sesoi = inf.get("sesoi")
    if not isinstance(sesoi, (int, float)):
        raise FreezeConsistencyError("yaml 'inference.sesoi' must be numeric")
    prose_sesoi = _prose_floats(prose, r"SESOI\s*=\s*\*{0,2}([0-9]*\.?[0-9]+)")
    _require(bool(prose_sesoi), "prose does not state 'SESOI = <x>'")
    _require(
        all(abs(v - float(sesoi)) < 1e-12 for v in prose_sesoi),
        f"SESOI mismatch: yaml sesoi={sesoi} but prose states SESOI={prose_sesoi}",
    )
    checked.append(f"inference.sesoi: yaml {sesoi} == prose SESOI={prose_sesoi[0]}")

    # 5) inference.equivalence_margin (symmetric TOST +/- margin) ---------- #
    margin = inf.get("equivalence_margin")
    if not isinstance(margin, (int, float)):
        raise FreezeConsistencyError("yaml 'inference.equivalence_margin' must be numeric")
    prose_margin = _prose_floats(prose, r"(?:±|\+/-|\+-)\s*\*{0,2}([0-9]*\.?[0-9]+)\s*DSR")
    _require(bool(prose_margin), "prose does not state the symmetric TOST margin '±<x> DSR'")
    _require(
        all(abs(v - float(margin)) < 1e-12 for v in prose_margin),
        f"equivalence_margin mismatch: yaml equivalence_margin={margin} but prose states "
        f"±{prose_margin} DSR",
    )
    checked.append(f"inference.equivalence_margin: yaml {margin} == prose ±{prose_margin[0]} DSR")

    # 6) cost_sweep.grid_bps (top-level) ----------------------------------- #
    cost_sweep = yml.get("cost_sweep") or {}
    grid_yaml = cost_sweep.get("grid_bps")
    if not isinstance(grid_yaml, list) or not grid_yaml:
        raise FreezeConsistencyError("yaml 'cost_sweep.grid_bps' must be a non-empty list")
    grid_prose = _prose_grid(prose)
    _require(grid_prose is not None, "prose does not state 'grid_bps = [...]'")
    _require(
        [int(x) for x in grid_yaml] == grid_prose,
        f"cost_sweep.grid_bps mismatch: yaml {grid_yaml} vs prose {grid_prose}",
    )
    checked.append(f"cost_sweep.grid_bps: yaml {grid_yaml} == prose {grid_prose}")

    # 7) 2026-06-24 amendments: lambda_cvar (λ=0), agent_numerics.tf32, search.reflect_protocol ---- #
    # The prose for these is verbose, so we assert the YAML carries the frozen VALUES + that the prose
    # NAMES each amendment (robust value-side checks), extending the gate to the three newest frozen items
    # without a fuzzy free-text regex that could false-fail the freeze.
    lam = (yml.get("fitness") or {}).get("lambda_cvar")
    _require(
        isinstance(lam, (int, float)) and abs(float(lam)) < 1e-12,
        f"fitness.lambda_cvar must be 0.0 (the frozen lambda=0 selection, amendment R22); yaml has {lam!r}",
    )
    _require(("lambda" in prose.lower()) or ("λ" in prose), "prose does not name the lambda=0 amendment (R22)")
    checked.append(f"fitness.lambda_cvar: yaml {lam} == prose lambda=0 (R22)")

    tf32 = (yml.get("agent_numerics") or {}).get("tf32")
    _require(tf32 is True, f"agent_numerics.tf32 must be true (frozen uniform precision, R23); yaml has {tf32!r}")
    _require("tf32" in prose.lower(), "prose does not name the TF32 amendment (R23)")
    checked.append(f"agent_numerics.tf32: yaml {tf32} == prose §11 (R23)")

    reflect = (yml.get("search") or {}).get("reflect_protocol_default")
    _require(
        bool(reflect),
        "search.reflect_protocol_default must be set (the headline reflection-protocol record, R21)",
    )
    _require(
        ("reflect-on-best" in prose.lower()) or ("reflect-on-last" in prose.lower()),
        "prose does not name the reflect-protocol amendment (R21)",
    )
    checked.append(f"search.reflect_protocol_default: yaml {reflect!r} present (R21)")

    # 8) 2026-07-02 (pre-freeze audit): data_panel.headline must equal the EXECUTED panel ------- #
    # The gate previously never looked at data_panel — R73's univ3->univ5 flip left the yaml mirror
    # stale (headline: univ3) and would have FROZEN silently. Bind the yaml record to the engine
    # config (config/data.yaml gold.suffix, itself hash-bound) so record and execution cannot drift.
    panel_yaml = (yml.get("data_panel") or {}).get("headline")
    _require(
        bool(panel_yaml),
        "data_panel.headline must be set (the frozen headline-panel record, R44/R73)",
    )
    try:
        data_cfg_suffix = (
            (yaml.safe_load((root / "config" / "data.yaml").read_text(encoding="utf-8")) or {})
            .get("gold", {})
            .get("suffix")
        )
    except OSError as exc:
        raise FreezeConsistencyError(f"cannot read config/data.yaml for the panel cross-check: {exc}") from exc
    _require(
        panel_yaml == data_cfg_suffix,
        f"data_panel.headline mismatch: preregistration.yaml says {panel_yaml!r} but the executed "
        f"config/data.yaml gold.suffix is {data_cfg_suffix!r} (R73 drift — fix the mirror before freezing)",
    )
    _require(
        str(panel_yaml) in prose,
        f"prose never names the frozen headline panel {panel_yaml!r}",
    )
    checked.append(f"data_panel.headline: yaml {panel_yaml!r} == config/data.yaml gold.suffix == prose (R73)")

    # 9) 2026-07-03 (batch-6 M5): the frozen tail-diagnostic set (§4) must be named in the prose ------ #
    # The tail set is hash-bound (in preregistration.yaml), so POST-freeze drift is caught by the hash,
    # but a PRE-freeze yaml<->prose §4 contradiction had no guard (unlike sesoi/m/grid). Bind each frozen
    # CVaR level + extra to a prose mention so the yaml mirror and §4 cannot freeze a contradiction.
    tail = yml.get("tail_diagnostic_set") or {}
    cvar_levels = tail.get("cvar_levels") or []
    extras = tail.get("extras") or []
    _require(
        bool(cvar_levels),
        "tail_diagnostic_set.cvar_levels must be a non-empty list (the frozen tail set, §4)",
    )
    for lv in cvar_levels:
        # yaml stores the level (0.25); prose §4 names the component by token (cvar_25). round() avoids the
        # 0.10*100 == 10.000...2 float artifact that int() would truncate to 9.
        tok = f"cvar_{round(float(lv) * 100):02d}"
        _require(
            tok in prose,
            f"prose §4 never names the frozen tail component {tok!r} (yaml cvar level {lv})",
        )
    for ex in extras:
        _require(
            str(ex) in prose,
            f"prose §4 never names the frozen tail extra {str(ex)!r}",
        )
    checked.append(
        f"tail_diagnostic_set: cvar_levels {cvar_levels} + extras {list(extras)} all named in prose §4"
    )

    return checked


def assert_phase0_recorded(yml: dict[str, Any]) -> str:
    """Refuse the freeze unless the Phase-0 pass marker is recorded.

    Returns the marker string; raises :class:`FreezePreconditionError` if absent/blank.
    """
    marker = yml.get("phase0_smoke_passed_log_id")
    if not isinstance(marker, str) or not marker.strip():
        raise FreezePreconditionError(
            "Phase-0 precondition NOT met: config/preregistration.yaml "
            "'phase0_smoke_passed_log_id' is missing/blank. The smoke GATE "
            "(scripts/smoke_test.py) must go GREEN/AMBER and be recorded before freezing."
        )
    return marker


def _config_arm_roster(rel: str, text: str) -> set[str]:
    """The arm roster declared by an executed config (``_ARM_ROSTER_CONFIGS`` member).

    ``arms.yaml`` declares the roster as the KEYS of its ``arms:`` mapping; ``campaign.yaml`` as a
    list under ``arms:``. Raise :class:`FreezeConsistencyError` if neither shape is present.
    """
    data = yaml.safe_load(text)
    arms_field = (data or {}).get("arms") if isinstance(data, dict) else None
    if isinstance(arms_field, dict):
        return {str(k) for k in arms_field}
    if isinstance(arms_field, list) and arms_field:
        return {str(a) for a in arms_field}
    raise FreezeConsistencyError(
        f"{rel} declares no usable 'arms' roster (a non-empty list or mapping); got "
        f"{type(arms_field).__name__}"
    )


def assert_executed_arms_match(yml: dict[str, Any], root: Path) -> str | None:
    """Cross-file V1 guard: every executed config's arm roster must equal the FROZEN prereg roster.

    The frozen design's ``arms`` (``config/preregistration.yaml``, already in the canonical hash and
    cross-checked against the §3 prose by :func:`assert_prose_matches_yaml`) is the source of truth. This
    asserts the EXECUTED configs the campaign actually reads (``_ARM_ROSTER_CONFIGS``) declare exactly that
    roster, so the campaign cannot silently run a different arm set than the one frozen — the ``placebo_
    shuffled`` 6-vs-7 drift the DEEP_AUDIT (V1/V2) flagged. These configs are NOT in the hash (compute knobs
    must stay amendable), so this assertion — not the hash — is what binds their roster.

    Returns a "checked + agreed" line naming the configs verified, or ``None`` when none is present (a
    minimal prereg-only test root). Raises :class:`FreezeConsistencyError` naming the first config that
    disagrees.
    """
    prereg_arms = yml.get("arms")
    if not isinstance(prereg_arms, list) or not prereg_arms:
        raise FreezeConsistencyError("yaml 'arms' must be a non-empty list")
    frozen = {str(a) for a in prereg_arms}
    checked: list[str] = []
    for rel in _ARM_ROSTER_CONFIGS:
        p = root / rel
        if not p.exists():
            continue
        roster = _config_arm_roster(rel, _normalize_bytes(p.read_bytes()).decode("utf-8"))
        _require(
            roster == frozen,
            f"executed arm roster in {rel} {sorted(roster)} != frozen prereg arms {sorted(frozen)} "
            f"(missing {sorted(frozen - roster)}, extra {sorted(roster - frozen)}); the campaign must "
            "run EXACTLY the frozen roster (V1: the placebo_shuffled 6-vs-7 drift)",
        )
        checked.append(rel)
    if not checked:
        return None
    return f"executed arms: {checked} rosters == frozen prereg arms (n={len(frozen)}: {sorted(frozen)})"


def assert_h1_baselines_match(yml: dict[str, Any], root: Path) -> str | None:
    """Cross-file guard (audit H-L2, 2026-07-02): the executed H1 family must equal the FROZEN prereg family.

    The H1 "beat-the-human" baseline family (PREREGISTRATION.md §1 (H1) + the amendment register) is frozen as the machine-readable
    ``h1_baselines`` list in ``config/preregistration.yaml`` (hash-bound). ``config/campaign.yaml`` — which
    the baseline TEST stage actually reads — is NOT hash-bound (its compute knobs must stay amendable), so,
    exactly like the arm-roster guard above, THIS assertion (not the hash) is what binds the executed
    family. Returns the checked-summary line, or ``None`` when the prereg yaml carries no ``h1_baselines``
    key (pre-migration checkouts) or campaign.yaml is absent (minimal test roots).
    """
    frozen_field = yml.get("h1_baselines")
    if frozen_field is None:
        return None
    frozen = {str(x) for x in frozen_field}
    camp_path = root / "config" / "campaign.yaml"
    if not camp_path.exists():
        return None
    camp = yaml.safe_load(camp_path.read_text(encoding="utf-8")) or {}
    executed_field = camp.get("h1_baselines")
    _require(
        isinstance(executed_field, list) and executed_field,
        "config/campaign.yaml declares no usable 'h1_baselines' list (the frozen H1 family, PREREGISTRATION §1 + the amendment register)",
    )
    executed = {str(x) for x in executed_field}
    _require(
        executed == frozen,
        f"executed h1_baselines in config/campaign.yaml {sorted(executed)} != frozen prereg family "
        f"{sorted(frozen)} (missing {sorted(frozen - executed)}, extra {sorted(executed - frozen)}); "
        "the H1 beat-the-human family (PREREGISTRATION §1 + register) may only change by dated amendment",
    )
    return f"h1_baselines: campaign.yaml == frozen prereg family (n={len(frozen)}: {sorted(frozen)})"


#: Executed configs that carry the per-candidate step budget B* at the TOP LEVEL (the value
#: run_campaign.py actually reads). ``config/preregistration.yaml`` is the FROZEN source of truth (in the
#: hash); these two are compute knobs kept OUT of the hash, so — exactly like the arm roster + H1 family —
#: an assertion (not the hash) is what binds them to the frozen B*.
_TRAIN_STEPS_CONFIGS: tuple[str, ...] = (
    "config/campaign.yaml",    # what the campaign EXECUTES (run_campaign.py reads camp["train_steps_per_candidate"])
    "config/algos.yaml",       # the mirror the preflight budget-mirror guard pairs with campaign.yaml
)



def assert_confirmatory_author_match(yml: dict[str, Any], root: Path) -> str | None:
    """Cross-file guard: the EXECUTED reward-author must equal the FROZEN ``confirmatory_author``.

    Same not-in-the-hash-so-assert pattern as the arm-roster / H1-family / B* / seeds guards, and added
    for the same reason (deep review 2026-07-26, loop 12). ``model_suite.confirmatory_author`` is inside
    the hash, but what the campaign actually CALLS is ``config/llm.yaml: model_snapshot`` (mirrored in
    ``config/campaign.yaml: llm.model_snapshot``) -- and **config/llm.yaml is not one of the bound
    configs**, so it is not hashed. ``scripts/preflight.py`` already checks the two EXECUTED mirrors
    against EACH OTHER, but nothing compared either to the REGISTERED value, so both could drift together
    and leave ``--check`` green. The author is common to every arm (not a treatment), so a drift does not
    break identification -- it silently changes WHICH MODEL the reported result generalises to, which is
    precisely what the registration exists to pin (cf. R102's own migration note).

    Returns the checked-summary line, or ``None`` when the prereg yaml declares no
    ``confirmatory_author`` (pre-migration checkouts) or ``config/llm.yaml`` is absent (minimal roots).
    """
    registered = (yml.get("model_suite") or {}).get("confirmatory_author")
    if registered is None:
        return None
    llm_path = root / "config" / "llm.yaml"
    if not llm_path.exists():
        return None
    llm = yaml.safe_load(llm_path.read_text(encoding="utf-8")) or {}
    executed = llm.get("model_snapshot")
    _require(
        isinstance(executed, str) and executed,
        "config/llm.yaml declares no usable 'model_snapshot' (the executed reward-author the campaign calls)",
    )
    _require(
        str(executed) == str(registered),
        f"executed reward-author config/llm.yaml model_snapshot={executed!r} != frozen prereg "
        f"model_suite.confirmatory_author={registered!r}; the confirmatory author may only change by a "
        "dated amendment (cf. R102), never by an un-hashed config edit",
    )
    # The campaign.yaml mirror is optional, but if present it must agree too (preflight checks this pair
    # as well; binding it HERE means a drift is caught at the gate, not only at pre-flight).
    camp_path = root / "config" / "campaign.yaml"
    if camp_path.exists():
        camp = yaml.safe_load(camp_path.read_text(encoding="utf-8")) or {}
        mirror = (camp.get("llm") or {}).get("model_snapshot")
        if mirror is not None:
            _require(
                str(mirror) == str(registered),
                f"config/campaign.yaml llm.model_snapshot={mirror!r} != frozen prereg "
                f"model_suite.confirmatory_author={registered!r}",
            )
    return f"confirmatory_author: config/llm.yaml (+campaign.yaml mirror) == frozen prereg ({registered})"

def assert_train_steps_match(yml: dict[str, Any], root: Path) -> str | None:
    """Cross-file guard (batch-6 M1, 2026-07-03): the EXECUTED B* must equal the FROZEN prereg B*.

    ``train_steps_per_candidate`` (B* = 200,000, R74) is the single most important executed number, yet it
    was the ONLY headline frozen quantity with no freeze-time executed<->frozen check: the frozen copy in
    ``config/preregistration.yaml`` is hash-bound, but ``campaign.yaml``/``algos.yaml`` (which the run reads)
    are deliberately un-hashed compute knobs, and ``preflight.check_budget_mirror`` compares only
    campaign<->algos — never the prereg. A coordinated post-freeze edit of BOTH campaign+algos (e.g. to 250k)
    would pass budget_mirror, leave the hashed prereg at 200k, and run undetected. This ties the executed
    top-level budget back to the frozen prereg (the same not-in-the-hash-so-assert pattern as
    :func:`assert_executed_arms_match` / :func:`assert_h1_baselines_match`).

    Returns the checked-summary line, or ``None`` when the prereg yaml carries no ``train_steps_per_candidate``
    (pre-migration checkouts) or none of the executed configs is present (a minimal prereg-only test root).
    Raises :class:`FreezeConsistencyError` naming the first config that disagrees.
    """
    frozen_field = yml.get("train_steps_per_candidate")
    if frozen_field is None:
        return None
    frozen = int(frozen_field)
    checked: list[str] = []
    for rel in _TRAIN_STEPS_CONFIGS:
        p = root / rel
        if not p.exists():
            continue
        cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        executed_field = cfg.get("train_steps_per_candidate")
        _require(
            executed_field is not None,
            f"{rel} declares no top-level 'train_steps_per_candidate' (the executed B*, R74)",
        )
        _require(
            int(executed_field) == frozen,
            f"executed train_steps_per_candidate in {rel} ({int(executed_field)}) != frozen prereg B* "
            f"({frozen}); the per-candidate budget B* (R74) may only change by dated amendment that also "
            "updates config/preregistration.yaml (the budget-mirror guard alone does NOT bind the prereg)",
        )
        checked.append(rel)
    if not checked:
        return None
    return f"train_steps_per_candidate: {checked} == frozen prereg B* ({frozen})"


def assert_seeds_match(yml: dict[str, Any], root: Path) -> str | None:
    """Cross-file guard (2026-07-04, DEEP_SWEEP E-F1): the EXECUTED seed list must equal the FROZEN prereg seeds.

    The seed set is the knob the power / equivalence headline hinges on, and a post-freeze edit of
    ``config/campaign.yaml`` ``seeds`` (30 -> 50, or a shift to a cherry-picked non-contiguous block) is the
    textbook forking path. The frozen copy in ``config/preregistration.yaml`` is hash-bound, but
    ``campaign.yaml`` (which run_campaign.py reads) is a deliberately un-hashed compute knob that nothing else
    tied to the prereg — the same not-in-the-hash-so-assert gap the arm-roster / H1 / B* guards close. Any
    change to the seed set may only land by a dated amendment that also updates the prereg.

    Returns the checked-summary line, or ``None`` when the prereg carries no ``seeds`` or ``campaign.yaml`` is
    absent (a minimal prereg-only test root). Raises :class:`FreezeConsistencyError` on disagreement.
    """
    frozen_field = yml.get("seeds")
    if frozen_field is None:
        return None
    # B-A3: resolve BOTH the prereg and campaign seed fields through the SAME resolver (list or schema)
    # so a uniform/tiered schema is compared on its FLAT set — the two files can express the seed set
    # differently only if they resolve to the identical [0..N-1] block.
    from src.utils.seeds import SeedSchemaError, resolve_seeds

    try:
        frozen = resolve_seeds(frozen_field)
    except SeedSchemaError as exc:
        raise FreezeConsistencyError(f"prereg 'seeds' is not a valid seed list/schema: {exc}") from exc
    camp_path = root / "config" / "campaign.yaml"
    if not camp_path.exists():
        return None
    camp = yaml.safe_load(camp_path.read_text(encoding="utf-8")) or {}
    executed_field = camp.get("seeds")
    _require(
        executed_field is not None,
        "config/campaign.yaml declares no 'seeds' (the frozen seed set the campaign executes)",
    )
    try:
        executed = resolve_seeds(executed_field)
    except SeedSchemaError as exc:
        raise FreezeConsistencyError(f"campaign 'seeds' is not a valid seed list/schema: {exc}") from exc
    _require(
        executed == frozen,
        f"executed seeds in config/campaign.yaml (n={len(executed)}) != frozen prereg seeds "
        f"(n={len(frozen)}); the seed set (PREREGISTRATION §6) may only change by a dated amendment that "
        "also updates config/preregistration.yaml — a post-freeze seed edit is a forking path",
    )
    return f"seeds: config/campaign.yaml == frozen prereg seeds (n={len(frozen)})"


def assert_matched_budget_match(yml: dict[str, Any], root: Path) -> str | None:
    """Cross-file guard (2026-07-04, DEEP_SWEEP E-F1): executed candidates-per-arm == FROZEN matched budget.

    The frozen ``matched_budget`` (``config/preregistration.yaml``, hash-bound) is the number of candidates
    each arm may author — the search multiplicity the DSR / PBO machinery corrects for. ``config/campaign.yaml``
    executes it as ``candidates_per_arm`` and is an un-hashed compute knob, so this assertion (not the hash)
    binds it, in the same pattern as the arm-roster / H1 / B* / seeds guards. A post-freeze change to the
    per-arm budget may only land by a dated amendment that also updates the prereg.

    Returns the checked-summary line, or ``None`` when the prereg carries no ``matched_budget`` or
    ``campaign.yaml`` is absent. Raises :class:`FreezeConsistencyError` on disagreement.
    """
    frozen_field = yml.get("matched_budget")
    if frozen_field is None:
        return None
    frozen = int(frozen_field)
    camp_path = root / "config" / "campaign.yaml"
    if not camp_path.exists():
        return None
    camp = yaml.safe_load(camp_path.read_text(encoding="utf-8")) or {}
    executed_field = camp.get("candidates_per_arm")
    _require(
        executed_field is not None,
        "config/campaign.yaml declares no top-level 'candidates_per_arm' (the executed matched budget)",
    )
    _require(
        int(executed_field) == frozen,
        f"executed candidates_per_arm in config/campaign.yaml ({int(executed_field)}) != frozen prereg "
        f"matched_budget ({frozen}); the per-arm candidate budget (PREREGISTRATION §6) may only change by a "
        "dated amendment that also updates config/preregistration.yaml",
    )
    return f"matched_budget: config/campaign.yaml candidates_per_arm == frozen prereg matched_budget ({frozen})"


#: Vocabulary that must NEVER appear in the two hash-bound base prompts (case-insensitive substrings).
#: R38 de-seeded the shared prompts so ONLY the distributional arm's FEEDBACK introduces the tail —
#: the construct-validity premise of the whole H2 contrast. Until 2026-07-05 nothing GUARDED that
#: property: a pre-freeze wording edit re-adding tail vocabulary would have been frozen unnoticed and
#: silently collapsed the placebo/scalar contrast (every arm pre-seeded with the tail again).
#: "tail"/"var" are matched as whole WORDS (plural included), NOT substrings — bare substrings would
#: false-positive on benign prompt wording (detail/retail/entail; variance/varying) and block a
#: legitimate pre-freeze edit (2026-07-06 audit MINOR). The rest are compound-safe as substrings.
#: 2026-07-06 completeness: "skew"/"kurtosis" added — the fed vector includes robust_skew, so a
#: prompt edit re-seeding a tail MOMENT must trip the gate exactly like a CVaR mention would.
_PROMPT_TAIL_VOCAB: tuple[str, ...] = (
    "cvar", "drawdown", "downside", "quantile", "shortfall", "value-at-risk", "value at risk",
    "skew", "kurtosis",
)
_PROMPT_TAIL_WORD_RE = re.compile(r"\btails?\b|\bvar\b")


def assert_prompt_tail_neutrality(root: Path) -> str | None:
    """The two hash-bound base prompts must stay TAIL-NEUTRAL (construct validity, R38).

    Returns the checked-summary line, or ``None`` when neither prompt exists (a minimal test root).
    Raises :class:`FreezeConsistencyError` naming the file + token on any hit.
    """
    prompt_paths = [root / rel for rel in _BOUND_TREATMENT if rel.startswith("prompts/")]
    present = [p for p in prompt_paths if p.exists()]
    if not present:
        return None
    for p in present:
        text = p.read_text(encoding="utf-8").lower()
        word_hit = _PROMPT_TAIL_WORD_RE.search(text)
        hit = next(
            (tok for tok in _PROMPT_TAIL_VOCAB if tok in text),
            word_hit.group(0) if word_hit else None,
        )
        _require(
            hit is None,
            f"{p.relative_to(root)} contains the tail token {hit!r} — the base prompts must stay "
            "tail-NEUTRAL (R38 de-seeding: only the distributional arm's FEEDBACK may introduce the "
            "tail, else the placebo/scalar construct-validity contrast collapses)",
        )
    n_tokens = len(_PROMPT_TAIL_VOCAB) + 2  # + the word-boundary "tail(s)" and "var" patterns
    return f"prompt tail-neutrality: {len(present)} base prompt(s) carry none of {n_tokens} tail tokens (R38)"


def assert_executed_tf32_matches_frozen(root: Path) -> str | None:
    """The EXECUTED float32 matmul precision must equal the frozen ``agent_numerics.tf32`` (R23).

    The R23 gate above asserts the PREREG's own ``agent_numerics.tf32`` and that the prose names the
    amendment — both on the frozen side. Nothing checked the side that actually runs:
    ``config/prototype.yaml`` is NOT in ``_BOUND_CONFIGS``, yet ``run_campaign`` builds the campaign's
    ``agent_cfg`` from it (``run_prototype._agent_cfg``, verified 2026-07-26 to carry the ``tf32`` key
    through), and ``train_agent`` applies it to ``torch.backends.cuda.matmul.allow_tf32`` +
    ``cudnn.allow_tf32`` for EVERY training leg (serial / SEARCH / TEST). So the executed float32 matmul
    arithmetic could be flipped post-freeze without moving the canonical hash and without tripping
    ``--check`` — and TF32 is recorded in NO per-record provenance (``env_fingerprint`` carries python /
    platform / git / packages only), so the divergence would also be invisible to audit. That is exactly
    the "a pin nobody can verify is FICTIONAL" failure the determinism envelope forbids, and exactly the
    not-in-the-hash-so-assert gap :func:`assert_search_splits_match` closes for the split windows.

    Returns the checked-summary line, or ``None`` when either side is absent (a minimal test root).
    """
    proto_path = root / "config" / "prototype.yaml"
    prereg_path = root / "config" / "preregistration.yaml"
    if not (proto_path.exists() and prereg_path.exists()):
        return None
    frozen = ((yaml.safe_load(prereg_path.read_text(encoding="utf-8")) or {})
              .get("agent_numerics") or {}).get("tf32")
    executed = ((yaml.safe_load(proto_path.read_text(encoding="utf-8")) or {})
                .get("agent") or {}).get("tf32")
    if frozen is None or executed is None:
        return None
    _require(
        bool(executed) is bool(frozen),
        f"config/prototype.yaml agent.tf32 ({executed!r}) != the frozen agent_numerics.tf32 "
        f"({frozen!r}). The campaign's agent_cfg is built from prototype.yaml, so the EXECUTED float32 "
        "matmul precision would differ from the frozen R23 design while the canonical hash still "
        "matched — prototype.yaml is deliberately NOT in _BOUND_CONFIGS. Reconcile before freezing.",
    )
    return f"agent_numerics.tf32 EXECUTED mirror: config/prototype.yaml agent.tf32 {executed} == frozen {frozen} (R23)"


def assert_search_splits_match(root: Path) -> str | None:
    """The executed SEARCH/sub-experiment split boundaries must equal the frozen data.yaml splits.

    ``config/prototype.yaml`` (the serial search driver's data window) and ``config/subexperiment.yaml``
    are UN-hashed, yet their ``data.val_end``/``data.train_end`` decide which window the executed search
    actually runs — and this drift class already bit once (subexperiment.yaml shipped a stale
    pre-Split-C ``2017-12-31`` until batch-6 M3). Same not-in-the-hash-so-assert pattern as the
    roster/B*/seeds guards (2026-07-05, map P23).

    Returns the checked-summary line, or ``None`` when data.yaml (or both consumers) are absent.
    """
    data_path = root / "config" / "data.yaml"
    if not data_path.exists():
        return None
    data_cfg = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    splits = data_cfg.get("splits") or {}
    frozen_val_end = str((splits.get("val") or {}).get("end", "")).strip()
    frozen_train_end = str((splits.get("train") or {}).get("end", "")).strip()
    if not frozen_val_end:
        return None
    checked: list[str] = []
    for rel, keys in (
        ("config/prototype.yaml", {"val_end": frozen_val_end, "train_end": frozen_train_end}),
        ("config/subexperiment.yaml", {"val_end": frozen_val_end}),
    ):
        path = root / rel
        if not path.exists():
            continue
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        block = cfg.get("data") or {}
        for key, frozen_value in keys.items():
            if not frozen_value:
                continue
            executed = block.get(key)
            if executed is None:
                continue
            _require(
                str(executed).strip() == frozen_value,
                f"{rel} data.{key} ({str(executed).strip()}) != the frozen config/data.yaml splits "
                f"boundary ({frozen_value}) — the executed search window has drifted from the frozen "
                "design (Split C, R73); reconcile before freezing",
            )
        checked.append(rel)
    if not checked:
        return None
    return f"search splits: {' + '.join(checked)} val/train boundaries == frozen data.yaml splits"


def assert_bound_files_exist(root: Path) -> str | None:
    """On a REAL repo root, every hash-bound file must EXIST before the hash is trusted.

    ``canonical_bytes`` deliberately skips absent bound files so minimal test roots can exercise the
    machinery — but on the production root that affordance meant a deleted/renamed prompt or config
    would freeze a silently-smaller hash with the gate fully green (2026-07-05, map finding). A root
    is REAL iff it carries ``.git`` (the mini-repo fixtures create ``pyproject.toml`` but never a git
    dir); minimal fixtures return ``None`` (skipped).
    """
    if not (root / ".git").exists():
        return None
    bound = [PREREG_MD, PREREG_YAML, *_BOUND_CONFIGS, *_BOUND_TREATMENT]
    missing = [rel for rel in bound if not (root / rel).exists()]
    _require(
        not missing,
        f"hash-bound file(s) MISSING on a real repo root: {missing} — the canonical hash would silently "
        "bind fewer files than the design declares; restore them (or record a dated amendment) before "
        "freezing",
    )
    return f"bound-file existence: all {len(bound)} hash-bound files present on the real root"


def assert_leg_roster_match(yml: dict, root: Path) -> str | None:
    """v2 (R80/R82): the EXECUTED leg config (``config/legs.yaml``) must match the registered
    ``model_suite`` — the same not-in-the-hash-so-assert pattern as the arm-roster/B* guards
    (legs.yaml is an executed knob; ``model_suite`` lives inside the hashed prereg yaml).

    Binds: label set AND queue order, per-leg model id, provider-pin membership, quantization,
    output caps where the registration specifies them, reasoning-pin presence, and the rolling-
    alias ban. Skipped (``None``) when the registration carries no ``model_suite`` (pre-v2
    fixtures) or when ``legs.yaml`` is absent on a NON-real root (minimal test fixtures — the
    same ``.git`` realness test as :func:`assert_bound_files_exist`).
    """
    suite = yml.get("model_suite")
    if not suite:
        return None
    legs_path = root / "config" / "legs.yaml"
    if not legs_path.exists():
        _require(
            not (root / ".git").exists(),
            "config/legs.yaml is MISSING on a real repo root while the registration declares a "
            "model_suite — the executed leg roster is unverifiable; restore it before freezing",
        )
        return None
    executed = (yaml.safe_load(legs_path.read_text(encoding="utf-8")) or {}).get("legs") or []
    exe_by_label = {str(leg.get("label")): leg for leg in executed}
    _require(
        len(exe_by_label) == len(executed),
        "duplicate leg labels in config/legs.yaml (labels must be unique)",
    )
    frozen_legs = suite.get("legs") or []
    frozen_labels = [str(leg.get("label")) for leg in frozen_legs]
    queue = [str(x) for x in (suite.get("queue_order") or [])]
    # Exact three-way equality: executed order == frozen queue == frozen leg list. Catches
    # extras, omissions, AND reordering in one comparison (dict preserves insertion order).
    _require(
        list(exe_by_label) == queue == frozen_labels,
        f"leg roster/order drift: executed {list(exe_by_label)} vs frozen queue {queue} "
        f"vs frozen legs {frozen_labels}",
    )
    for fleg in frozen_legs:
        label = str(fleg.get("label"))
        eleg = exe_by_label[label]
        _require(
            str(eleg.get("model")) == str(fleg.get("id")),
            f"leg {label!r}: executed model {eleg.get('model')!r} != frozen id {fleg.get('id')!r}",
        )
        model = str(eleg.get("model"))
        _require(
            "~" not in model and not model.endswith("-latest"),
            f"leg {label!r}: executed model {model!r} is a rolling alias (banned, R80)",
        )
        if fleg.get("provider_pin"):
            only = ((eleg.get("provider_pin") or {}).get("only")) or []
            _require(
                str(fleg["provider_pin"]) in [str(x) for x in only],
                f"leg {label!r}: frozen provider pin {fleg['provider_pin']!r} not in the "
                f"executed only-list {only!r}",
            )
        if fleg.get("quantization"):
            quants = [str(q) for q in (eleg.get("quantizations") or [])]
            _require(
                str(fleg["quantization"]) in quants,
                f"leg {label!r}: frozen quantization {fleg['quantization']!r} not in the "
                f"executed list {quants!r}",
            )
        if fleg.get("output_cap_tokens"):
            _require(
                int(eleg.get("max_tokens", -1)) == int(fleg["output_cap_tokens"]),
                f"leg {label!r}: executed max_tokens {eleg.get('max_tokens')!r} != frozen "
                f"output cap {fleg['output_cap_tokens']!r}",
            )
        if fleg.get("reasoning_pin"):
            _require(
                bool(eleg.get("reasoning")),
                f"leg {label!r}: the registration pins reasoning "
                f"({fleg['reasoning_pin']!r}) but the executed leg carries no reasoning block",
            )
        # R103 (2026-07-25, repro-audit HOLE 5): bind the hf_pin COMMIT HASH — the reproducibility
        # PERMANENCE anchor. model_suite.hf_pins_recorded carries "repo@commit" per filled open leg
        # (legs.yaml mirrors it as hf_pin: {repo, commit}); previously only pin PRESENCE was bound, so
        # a post-freeze commit drift of the exact weights release passed --check UNDETECTED. (The
        # reasoning VALUE is deliberately NOT static-bound here — a human label can't cleanly match the
        # API dict; the R103 leg-gate round-trip verifies the pin FUNCTIONED, a stronger guarantee.)
        recorded_pin = (suite.get("hf_pins_recorded") or {}).get(label)
        if recorded_pin:
            hp = eleg.get("hf_pin") or {}
            executed_pin = f"{hp.get('repo')}@{hp.get('commit')}" if hp else "<none>"
            _require(
                executed_pin == str(recorded_pin),
                f"leg {label!r}: executed hf_pin {executed_pin!r} != registered "
                f"hf_pins_recorded {recorded_pin!r} — the weights-hash permanence anchor must not drift",
            )
    pending = pending_hf_pins(root)
    suffix = (
        f"; R85 hf_pins PENDING for {len(pending)} open leg(s) — filled at gate time, "
        "the real freeze REFUSES until then" if pending else "; R85 hf_pins filled"
    )
    return (
        f"leg roster (v2): config/legs.yaml == frozen model_suite "
        f"(n={len(frozen_legs)}, order + ids + pins verified{suffix})"
    )


def pending_hf_pins(root: Path) -> list[str]:
    """Open-weight legs whose R85 hf_pin (repo+commit) is absent or still a placeholder.

    The weights hash is the permanence anchor behind the open-replication reproducibility claim
    (MODEL_SWEEP §4.1; model_suite.hf_weights_pins): pre-freeze ``--check`` REPORTS pending pins;
    :func:`do_freeze` REFUSES while any remain (a registered pin must be real before it is hashed).
    """
    legs_path = root / "config" / "legs.yaml"
    if not legs_path.exists():
        return []
    executed = (yaml.safe_load(legs_path.read_text(encoding="utf-8")) or {}).get("legs") or []
    pending: list[str] = []
    for leg in executed:
        if "hf_pin" not in leg:
            continue  # closed legs carry no weights pin by design
        hf = leg.get("hf_pin") or {}
        repo, commit = str(hf.get("repo") or ""), str(hf.get("commit") or "")
        if not repo or not commit or "TO-VERIFY" in repo or "TO-VERIFY" in commit:
            pending.append(str(leg.get("label")))
    return pending


# --------------------------------------------------------------------------- #
# Verification (shared by --check and the real freeze)                         #
# --------------------------------------------------------------------------- #
@dataclass
class FreezeStatus:
    """The result of verifying the pre-registration (no side effects)."""

    hash: str
    phase0_marker: str
    checks: list[str]
    already_frozen: bool
    recorded_hash: str | None


def verify(root: Path | None = None) -> FreezeStatus:
    """Run every freeze precondition + the prose<->yaml gate; compute the hash. No writes.

    Raises :class:`FreezePreconditionError` / :class:`FreezeConsistencyError` on any failure.
    """
    root = root or repo_root()
    yml = load_yaml(root)
    prose = load_prose(root)

    phase0_marker = assert_phase0_recorded(yml)
    checks = assert_prose_matches_yaml(yml, prose, root)
    # V1 cross-file guard: the EXECUTED configs (campaign.yaml/arms.yaml) must declare exactly the frozen
    # roster. Not part of the hash (compute knobs stay amendable); this assertion is what binds their roster.
    arms_check = assert_executed_arms_match(yml, root)
    if arms_check is not None:
        checks.append(arms_check)
    # H1-family guard (audit H-L2): same not-in-the-hash-so-assert pattern as the roster guard.
    h1_check = assert_h1_baselines_match(yml, root)
    if h1_check is not None:
        checks.append(h1_check)
    # Confirmatory-author guard (deep review 2026-07-26, loop 12): config/llm.yaml is NOT hashed, so the
    # EXECUTED reward-author could drift from the registered one with --check still green.
    author_check = assert_confirmatory_author_match(yml, root)
    if author_check is not None:
        checks.append(author_check)
    # B* guard (batch-6 M1): the executed per-candidate budget must equal the frozen prereg B*; same
    # not-in-the-hash-so-assert pattern (campaign.yaml/algos.yaml are un-hashed compute knobs).
    steps_check = assert_train_steps_match(yml, root)
    if steps_check is not None:
        checks.append(steps_check)
    # Seed-set + matched-budget guards (2026-07-04, DEEP_SWEEP E-F1): same not-in-the-hash-so-assert pattern.
    # The seed count is the knob the power/equivalence headline hinges on; bind the executed campaign.yaml
    # seeds + candidates_per_arm back to the frozen prereg so a post-freeze edit cannot drift undetected.
    seeds_check = assert_seeds_match(yml, root)
    if seeds_check is not None:
        checks.append(seeds_check)
    budget_check = assert_matched_budget_match(yml, root)
    if budget_check is not None:
        checks.append(budget_check)
    # 2026-07-05 hardening (map P23 + construct-validity + existence): three further not-in-the-hash
    # guards — the executed search-split boundaries, the R38 tail-neutrality of the two base prompts,
    # and (real roots only) the existence of every hash-bound file before the hash is trusted.
    tf32_mirror_check = assert_executed_tf32_matches_frozen(root)
    if tf32_mirror_check:
        checks.append(tf32_mirror_check)
    splits_check = assert_search_splits_match(root)
    if splits_check is not None:
        checks.append(splits_check)
    neutrality_check = assert_prompt_tail_neutrality(root)
    if neutrality_check is not None:
        checks.append(neutrality_check)
    existence_check = assert_bound_files_exist(root)
    if existence_check is not None:
        checks.append(existence_check)
    # v2 (R80/R82): the executed leg roster must match the registered model_suite — same
    # not-in-the-hash-so-assert pattern (legs.yaml is an executed knob, model_suite is hashed).
    legs_check = assert_leg_roster_match(yml, root)
    if legs_check is not None:
        checks.append(legs_check)
    digest = canonical_hash(root)

    return FreezeStatus(
        hash=digest,
        phase0_marker=phase0_marker,
        checks=checks,
        already_frozen=bool(yml.get("frozen")),
        recorded_hash=yml.get("freeze_hash"),
    )


# --------------------------------------------------------------------------- #
# The WRITE path (real freeze only — never run by an agent / by --check)        #
# --------------------------------------------------------------------------- #
def _set_yaml_frozen(root: Path, digest: str) -> None:
    """Flip ``frozen: false -> true`` and ``freeze_hash: null -> <digest>`` IN PLACE.

    A line-level edit (not a YAML round-trip) so every comment, amendment note, and the
    exact byte layout of the mirror are preserved — only the two scalar values change.
    """
    path = root / PREREG_YAML
    text = _normalize_bytes(path.read_bytes()).decode("utf-8")

    new_text, n_frozen = re.subn(
        r"(?m)^(frozen:\s*)false\b", r"\g<1>true", text, count=1
    )
    if n_frozen != 1:
        raise FreezeError(f"could not flip 'frozen: false' in {PREREG_YAML} (already frozen?)")

    new_text, n_hash = re.subn(
        r"(?m)^(freeze_hash:\s*)null\b", rf"\g<1>{digest}", new_text, count=1
    )
    if n_hash != 1:
        raise FreezeError(f"could not set 'freeze_hash: null' in {PREREG_YAML} (already set?)")

    path.write_text(new_text, encoding="utf-8", newline="\n")


def _utc_now_iso() -> str:
    """UTC timestamp, ISO-8601, second precision, 'Z' suffix."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_decision_log(root: Path, digest: str, when: str, git_sha: str | None) -> None:
    """Record the hash + UTC + git SHA into ``provenance/DECISION_LOG.md`` (the ADR-005 slot).

    Appends a dated completion entry beneath the ``<!-- amendments appended ... -->``
    marker, so the append-only audit log stays append-only. The log is a RECEIPT, not an
    input: a fresh clone carries none, so it is seeded here rather than required.
    """
    path = root / DECISION_LOG
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_DECISION_LOG_SEED, encoding="utf-8", newline="\n")
    text = _normalize_bytes(path.read_bytes()).decode("utf-8")
    sha = git_sha or "(no git SHA)"
    entry = (
        f"\n### FREEZE-DONE — pre-registration content hash recorded ({when[:10]})\n"
        f"**Decision:** PREREGISTRATION.md + config/preregistration.yaml FROZEN.\n"
        f"- **Content hash (SHA-256):** `{digest}`\n"
        f"- **Frozen at (UTC):** {when}\n"
        f"- **Git commit:** `{sha}`\n"
        f"- **Phase-0 precondition:** met (see the PHASE-0 GREEN entry above).\n"
        f"- **Tag:** `{FREEZE_TAG}` (signed; OpenTimestamps proof beside it if `ots` was present).\n"
        f"**Status:** frozen; any post-freeze change requires a dated, user-approved amendment.\n"
    )
    marker = "<!-- amendments appended below this line -->"
    if marker in text:
        text = text.replace(marker, marker + "\n" + entry, 1)
    else:  # pragma: no cover - the marker is present in the live file
        text = text.rstrip() + "\n" + entry
    path.write_text(text, encoding="utf-8", newline="\n")


def _git_tag_signed(root: Path, digest: str, when: str) -> str:
    """Create the freeze tag. Try a SIGNED tag first; fall back to annotated + warn.

    Returns a short status string for the report. Best-effort: never raises.
    """
    message = f"Pre-registration freeze {FREEZE_TAG}\nSHA-256 {digest}\nUTC {when}"
    try:
        subprocess.run(
            ["git", "tag", "-s", FREEZE_TAG, "-m", message],
            cwd=root, capture_output=True, text=True, check=True,
        )
        return f"signed tag {FREEZE_TAG} created"
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        print(f"[freeze] WARN: signed tag failed ({detail.strip()}); falling back to annotated tag.")
    try:
        subprocess.run(
            ["git", "tag", "-a", FREEZE_TAG, "-m", message],
            cwd=root, capture_output=True, text=True, check=True,
        )
        return f"annotated tag {FREEZE_TAG} created (UNSIGNED — `-s` unavailable)"
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:  # pragma: no cover
        detail = getattr(exc, "stderr", "") or str(exc)
        print(f"[freeze] WARN: annotated tag also failed ({detail.strip()}); skipping tag.")
        return "tag SKIPPED (git tag failed)"


def _ots_stamp(root: Path, digest: str) -> str:
    """OpenTimestamp the hash (best-effort). Skip + warn if `ots` is absent. Never raises."""
    proof = root / "provenance" / f"{FREEZE_TAG}.sha256"
    try:
        proof.write_text(f"{digest}  PREREGISTRATION+yaml ({FREEZE_TAG})\n", encoding="utf-8", newline="\n")
        subprocess.run(
            ["ots", "stamp", str(proof)],
            cwd=root, capture_output=True, text=True, check=True,
        )
        return f"ots stamped {proof.name} (.ots proof beside it)"
    except FileNotFoundError:
        print("[freeze] WARN: `ots` (opentimestamps-client) not found — skipping the timestamp proof.")
        return "ots SKIPPED (client absent)"
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        print(f"[freeze] WARN: `ots stamp` failed ({(exc.stderr or '').strip()}); skipping.")
        return "ots SKIPPED (stamp failed)"


def do_freeze(root: Path | None = None) -> FreezeStatus:
    """The REAL freeze (the write path). Verifies, then writes the hash, log, tag, ots.

    NB: this MUTATES the repo (YAML + DECISION_LOG + a git tag). It must be run ONLY by the
    user at the Phase-1 freeze, never by an agent or in CI. ``--check`` never calls this.
    """
    root = root or repo_root()
    status = verify(root)
    if status.already_frozen:
        raise FreezeError(
            f"{PREREG_YAML} is already frozen (frozen: true, freeze_hash={status.recorded_hash}). "
            "Re-freezing is forbidden; post-freeze changes go through a dated amendment."
        )
    # R85 (2026-07-21): a registered weights pin must be REAL before it is hashed — refuse the
    # freeze while any open leg still carries a TO-VERIFY-AT-GATE placeholder.
    _pending = pending_hf_pins(root)
    if _pending:
        raise FreezeError(
            f"R85 hf_pin placeholders remain for open leg(s) {_pending} — fill repo+commit from "
            "each official HF card (gate step) before freezing."
        )

    when = _utc_now_iso()
    git_sha = git_commit()

    # Order: stamp the design into the YAML, then record it, then tag, then timestamp.
    _set_yaml_frozen(root, status.hash)
    # Recompute the hash AFTER flipping frozen/freeze_hash would change it — so the RECORDED
    # hash is the PRE-flip canonical hash (the design content), which is what `verify` returns
    # and what `--check` re-derives. The two scalar-state flips are deliberately NOT part of
    # the hashed design (see ADR-005 freeze-record note).
    _append_decision_log(root, status.hash, when, git_sha)
    tag_status = _git_tag_signed(root, status.hash, when)
    ots_status = _ots_stamp(root, status.hash)

    print(f"[freeze] FROZEN. SHA-256 {status.hash}")
    print(f"[freeze]   UTC {when}  git {git_sha or '(none)'}")
    print(f"[freeze]   {tag_status}")
    print(f"[freeze]   {ots_status}")
    return status


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze / verify the pre-registration (FINAL_PLAN Phase 1.E; Rank 9).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify prose<->yaml + Phase-0 + recompute the hash; NO writes. Non-zero exit on drift.",
    )
    return parser


def _print_check(status: FreezeStatus) -> None:
    print("[freeze] --check — verifying the pre-registration (no writes):")
    print(f"  Phase-0 precondition MET: {status.phase0_marker}")
    print("  prose<->yaml consistency:")
    for line in status.checks:
        print(f"    OK  {line}")
    print(f"  canonical SHA-256: {status.hash}")
    if status.recorded_hash:
        match = "MATCHES" if status.recorded_hash == status.hash else "DRIFT!"
        print(f"  recorded freeze_hash: {status.recorded_hash}  [{match}]")
    else:
        print("  recorded freeze_hash: null (not yet frozen — expected pre-freeze)")


def main(argv: list[str] | None = None) -> int:
    from src.utils.console import make_console_safe
    make_console_safe()   # a console codepage must never kill GO step 1 (src/utils/console.py)
    args = build_parser().parse_args(argv)
    root = repo_root()

    if args.check:
        try:
            status = verify(root)
        except FreezeError as exc:
            print(f"[freeze] --check FAILED: {exc}", file=sys.stderr)
            return 1
        _print_check(status)
        # On a frozen repo, --check is also a DRIFT guard: the recorded hash must match.
        if status.recorded_hash and status.recorded_hash != status.hash:
            print(
                f"[freeze] --check FAILED: recorded freeze_hash {status.recorded_hash} != "
                f"current {status.hash} (the frozen artifacts changed).",
                file=sys.stderr,
            )
            return 1
        return 0

    # Default == the REAL freeze (write path). Implemented; gated to the user.
    status = do_freeze(root)
    return 0 if status else 1


if __name__ == "__main__":
    raise SystemExit(main())
