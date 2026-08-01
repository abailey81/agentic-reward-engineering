"""The junction-safety lock — written because the unsafe version DESTROYED the data directory.

On 2026-07-27 `certify_commit` junctioned the real `data/` into a throwaway worktree and tore the
worktree down without severing the link. A Windows junction is not a symlink, the teardown followed
it, and the REAL directory was deleted: 1,179 tracked files (restored from git) and ~1.2 GB of
untracked data including the gold panel, which is not in git.

NOTE: The incident note blamed `shutil.rmtree`. Measured here (Python 3.11.9 + Git for Windows), that is
WRONG: CPython treats a junction as a link and will not descend it, whether you rmtree the junction
itself or a parent containing one. `git worktree remove --force` DOES follow it, and that is what
destroyed the data — see `test_git_worktree_remove_follows_a_junction_and_destroys_the_target`,
which reproduces the loss so the guard's rationale matches the hazard it actually guards.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.certify_commit import _is_reparse_point, _link, unlink_reparse_points

pytestmark = pytest.mark.skipif(os.name != "nt", reason="junction semantics are Windows-specific")


def _junction(link: Path, target: Path) -> bool:
    r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                       capture_output=True)
    return r.returncode == 0


def test_a_junction_is_recognised_even_though_it_is_not_a_symlink(tmp_path):
    """The root cause: Path.is_symlink() returns False for a junction, so the old guard missed it."""
    target = tmp_path / "real"
    target.mkdir()
    (target / "precious.txt").write_text("do not delete", encoding="utf-8")
    link = tmp_path / "link"
    if not _junction(link, target):
        pytest.skip("mklink unavailable")
    assert _is_reparse_point(link) is True
    assert link.is_symlink() is False          # exactly why the naive check failed


def test_severing_a_link_NEVER_touches_the_target(tmp_path):
    """os.rmdir on a junction removes the link only. This is the property the fix depends on."""
    target = tmp_path / "real"
    target.mkdir()
    (target / "precious.txt").write_text("survive", encoding="utf-8")
    holder = tmp_path / "holder"
    holder.mkdir()
    if not _junction(holder / "data", target):
        pytest.skip("mklink unavailable")

    removed = unlink_reparse_points(holder)
    assert removed and not (holder / "data").exists()
    assert (target / "precious.txt").read_text(encoding="utf-8") == "survive"


def test_after_severing_a_recursive_delete_cannot_reach_the_target(tmp_path):
    """The end-to-end guarantee: sever first, THEN delete, and the real data survives."""


    target = tmp_path / "real"
    target.mkdir()
    (target / "gold.parquet").write_bytes(b"expensive")
    holder = tmp_path / "holder"
    holder.mkdir()
    (holder / "code.py").write_text("x = 1", encoding="utf-8")
    if not _junction(holder / "data", target):
        pytest.skip("mklink unavailable")

    unlink_reparse_points(holder)              # the step whose absence caused the loss
    shutil.rmtree(holder, ignore_errors=True)
    assert not holder.exists()
    assert (target / "gold.parquet").read_bytes() == b"expensive"


def test_relinking_an_existing_junction_does_not_delete_the_target(tmp_path):
    """``_link`` on a re-run, when ``dst`` is already the junction the first run created.

    Its guard read ``dst.exists() and not dst.is_symlink()``, which cannot see a junction. MEASURED:
    that was NOT destructive (CPython's rmtree will not descend a junction) — it merely left a stale
    link and returned True. So this is a LOCK on the invariant, not a reproduction of a live loss;
    the destructive path is ``git worktree remove`` below.
    """
    target = tmp_path / "real"
    target.mkdir()
    (target / "gold.parquet").write_bytes(b"expensive")
    worktree = tmp_path / "worktree"
    worktree.mkdir()

    assert _link(target, worktree / "data") is True
    if not _is_reparse_point(worktree / "data"):
        pytest.skip("mklink unavailable")
    assert _link(target, worktree / "data") is True          # the re-run over an existing junction

    assert (target / "gold.parquet").read_bytes() == b"expensive"
    assert (worktree / "data" / "gold.parquet").read_bytes() == b"expensive"


def _repo_with_junctioned_worktree(tmp_path):
    """A REAL git repo + real linked worktree whose ``data/`` is a junction to the repo's own."""
    main = tmp_path / "main"
    main.mkdir()
    run = lambda *a: subprocess.run(a, cwd=main, capture_output=True, text=True)  # noqa: E731
    run("git", "init", "-q")
    run("git", "config", "user.email", "t@e.com")
    run("git", "config", "user.name", "T")
    (main / "data").mkdir()
    (main / "data" / "tracked.txt").write_text("tracked", encoding="utf-8")
    run("git", "add", "-A")
    run("git", "commit", "-qm", "init")
    (main / "data" / "untracked_gold.parquet").write_bytes(b"expensive")   # gitignored-equivalent
    wt = tmp_path / "wt"
    if run("git", "worktree", "add", "-q", str(wt), "HEAD").returncode != 0:
        pytest.skip("git worktree unavailable")
    shutil.rmtree(wt / "data")
    if not _junction(wt / "data", main / "data"):
        pytest.skip("mklink unavailable")
    return main, wt


def test_git_worktree_remove_follows_a_junction_and_destroys_the_target(tmp_path):
    """THE ACTUAL MECHANISM of the 2026-07-27 loss, reproduced.

    The incident was recorded as ``shutil.rmtree`` walking through the junction. It is not: CPython
    refuses to descend one. GIT's recursive removal does follow it. This test pins the real hazard,
    so the guarantee below is known to be guarding the thing that actually bites.
    """
    main, wt = _repo_with_junctioned_worktree(tmp_path)
    subprocess.run(["git", "worktree", "remove", "--force", str(wt)],
                   cwd=main, capture_output=True)
    assert not (main / "data" / "untracked_gold.parquet").exists(), (
        "git worktree remove no longer follows junctions on this platform — if this fails, the "
        "hazard changed and the incident note should be revisited (the severing guard stays either "
        "way, but its rationale must match reality)")


def test_destroy_worktree_severs_first_so_the_real_data_survives(tmp_path):
    """The end-to-end guarantee against the mechanism proven above."""
    from scripts.certify_commit import destroy_worktree

    main, wt = _repo_with_junctioned_worktree(tmp_path)
    cwd = Path.cwd()
    try:
        os.chdir(main)                     # destroy_worktree shells out to git in the CWD repo
        destroy_worktree(wt)
    finally:
        os.chdir(cwd)
    assert not wt.exists()
    assert (main / "data" / "untracked_gold.parquet").read_bytes() == b"expensive"
    assert (main / "data" / "tracked.txt").exists()
