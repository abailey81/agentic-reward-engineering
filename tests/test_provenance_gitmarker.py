"""The deployed-archive GIT_COMMIT fallback (2026-07-18 publication-readiness catch): cluster
records must carry the producing code version even though `git archive | tar` deploys a
non-work-tree — the marker file supplies it, prefixed so it can't masquerade as a live tree."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_git_commit_prefers_real_tree_here():
    from src.utils.provenance import git_commit

    sha = git_commit()
    assert sha and not sha.startswith("deployed-archive:")  # we ARE in a work-tree locally


def test_marker_fallback_shape(tmp_path, monkeypatch):
    # simulate the deployed tree: no git available + a marker at the resolved root

    import src.utils.provenance as prov

    def _no_git(*a, **k):
        raise FileNotFoundError("git absent on node")

    monkeypatch.setattr(prov.subprocess, "run", _no_git)
    # point the marker path at a temp root by monkeypatching Path resolution is invasive;
    # instead validate the parsing contract directly on a real marker file
    marker = Path(prov.__file__).resolve().parents[2] / "GIT_COMMIT"
    existed = marker.exists()
    original = marker.read_bytes() if existed else None
    try:
        marker.write_text("abc123def456789\n", encoding="utf-8")
        out = prov.git_commit()
        assert out == "deployed-archive:abc123def456789"
        assert prov.git_commit(short=True) == "deployed-archive:abc123def456"
    finally:
        # Restore the tree exactly: rewrite the original content if the marker pre-existed
        # (never silently corrupt a genuine deployed marker), else remove the temp one.
        if existed:
            marker.write_bytes(original)
        else:
            marker.unlink(missing_ok=True)
