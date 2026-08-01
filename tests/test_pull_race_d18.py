"""D18 — the pull must never nest a record inside itself, even when it loses a race.

Root cause found 2026-08-01 (RUN 10, record §100). §44.6 and §65.4 correctly identified the
DEFECT — one record at two paths, `<candidate>/<candidate>/record.json` — and left the MECHANISM
unstated, so it was carried for two sessions as "a destination computed as `<dest>/<run_id>`".

**The real mechanism.** `shutil.move(src, dst)` where `dst` is an EXISTING DIRECTORY does not
fail — it moves `src` INSIDE it. The `if dest.exists(): continue` guard cannot close that, because
`local_root` (`read_root`) is SHARED BY ALL TWELVE SUPERVISED LINES: another line's driver can
commit the same record in the window between our check and our move.

**Three independent facts confirm it, and none of them is an inference:**
  * the remote side is FLAT — verified on the node, no nesting exists there, so the doubling is
    created locally by the pull;
  * both live instances sit on LEG lines, i.e. the many-concurrent-drivers case;
  * the count grew 1 -> 2 as concurrency rose (1,025 records -> 1,449 -> 1,597).

`os.rename` is the fix because it CANNOT nest: renaming onto an existing directory raises rather
than descending into it. Same filesystem by construction (staging is a child of the mirror), so it
is atomic too.

**Impact was bounded and is stated honestly:** both duplicates are byte-identical, both sit on
report-only legs, and — verified first-hand rather than assumed — NEITHER the C3 gate nor the
completeness census can be misled by one, because the gate counts DISTINCT substrate signatures
(a duplicate adds to an existing one) and `_test_census` builds a SET of `run_id`s.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.cluster.poll import pull_archive
from tests.test_cluster_submit_poll_ledger import _fake_find_runner


def _racing_fetch(loser_rels: set[str], mirror: Path):
    """A fetch that ALSO commits the record straight into the mirror — the concurrent driver.

    The remote-vs-local DIFF has already run by the time fetch is called, so the relpath is still
    in the transfer list; the rival copy appears only afterwards. That is the real ordering.
    """
    def _fetch(relpaths, staging):
        for rel in relpaths:
            d = staging / rel
            d.mkdir(parents=True, exist_ok=True)
            (d / "record.json").write_text('{"run_id": "x"}', encoding="utf-8")
            if rel in loser_rels:                       # the other driver wins the race
                committed = mirror / rel
                committed.mkdir(parents=True, exist_ok=True)
                (committed / "record.json").write_text('{"run_id": "x"}', encoding="utf-8")

    return _fetch


@pytest.fixture
def blind_guard(monkeypatch: pytest.MonkeyPatch):
    """Open the TOCTOU window the defect actually lives in.

    NOTE: THIS FIXTURE IS THE WHOLE TEST, AND IT WAS ADDED AFTER THE FIRST VERSION OF THIS FILE FAILED
    TO FALSIFY. Without it, `_racing_fetch` commits the rival copy BEFORE `if dest.exists()` runs,
    the guard catches it, `shutil.move` is never reached, and the tests pass against the PRE-FIX
    code — proving nothing. The defect is not "the destination exists"; the guard handles that.
    The defect is "the destination did not exist WHEN WE LOOKED, and does by the time we move".

    So make the guard's `exists()` answer False EXACTLY ONCE for the contested path, while the
    filesystem genuinely holds the rival copy. Everything else is real.
    """
    import pathlib

    real_exists = pathlib.Path.exists
    state: dict[str, int] = {}

    def install(blind_path: Path) -> None:
        def fake_exists(self, *a, **k):  # noqa: ANN001, ANN202
            if Path(self) == Path(blind_path) and state.get("n", 0) == 0:
                state["n"] = 1
                return False
            return real_exists(self, *a, **k)

        monkeypatch.setattr(pathlib.Path, "exists", fake_exists)

    return install


def test_losing_the_commit_race_never_nests_the_record(tmp_path: Path, blind_guard) -> None:
    """THE DEFECT. Pre-fix this produced `search/c1/c1/record.json` — one record at two paths."""
    rel = "search/c1"
    blind_guard(tmp_path / "search" / "c1")
    pull_archive("/r/out", tmp_path,
                 runner=_fake_find_runner("/r/out", [rel]),
                 fetch=_racing_fetch({rel}, tmp_path))

    committed = tmp_path / "search" / "c1" / "record.json"
    nested = tmp_path / "search" / "c1" / "c1" / "record.json"
    assert committed.is_file(), "the committed copy must survive"
    assert not nested.exists(), (
        "the pull nested the record inside itself — this is D18, and it is the shutil.move "
        "into-an-existing-directory behaviour")
    # and exactly ONE record.json exists under that candidate
    assert len(list((tmp_path / "search" / "c1").rglob("record.json"))) == 1


def test_the_uncontended_path_is_unchanged(tmp_path: Path) -> None:
    """NO REGRESSION: with no race, the record lands exactly where it always did."""
    rel = "search/c2"
    n = pull_archive("/r/out", tmp_path,
                     runner=_fake_find_runner("/r/out", [rel]),
                     fetch=_racing_fetch(set(), tmp_path))
    assert n == 1
    assert (tmp_path / "search" / "c2" / "record.json").is_file()
    assert not (tmp_path / ".pull_tmp").exists()


def test_a_lost_race_leaves_no_staging_debris(tmp_path: Path, blind_guard) -> None:
    """The staged loser must be discarded, not left to be swept as a phantom record.

    A `.pull_tmp` leftover carrying a record.json is itself a known duplicate source (§86.2's
    stale `.pull_tmp` byte-identical duplicate), so losing the race must not create one.
    """
    rel = "search/c3"
    blind_guard(tmp_path / "search" / "c3")
    pull_archive("/r/out", tmp_path,
                 runner=_fake_find_runner("/r/out", [rel]),
                 fetch=_racing_fetch({rel}, tmp_path))
    assert not list(tmp_path.glob(".pull_tmp*")), "staging debris survived a lost race"


def test_the_guard_still_catches_the_race_it_could_already_see(tmp_path: Path) -> None:
    """NO REGRESSION on the EASY case: when the rival copy is visible at check time, the pull
    skips it exactly as before. Without this the fix could be 'always rename', which would
    overwrite a committed record instead of keeping it."""
    rel = "search/c5"
    pull_archive("/r/out", tmp_path,
                 runner=_fake_find_runner("/r/out", [rel]),
                 fetch=_racing_fetch({rel}, tmp_path))
    assert len(list((tmp_path / rel).rglob("record.json"))) == 1
