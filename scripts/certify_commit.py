#!/usr/bin/env python3
"""CERTIFY A COMMIT — prove the exact code you will deploy is green, without stopping anyone.

THE PROBLEM THIS DISSOLVES
--------------------------
A test run certifies a TREE STATE, not a folder. Four concurrent sessions share this working
directory (95 commits landed on 2026-07-26 alone), so a suite run against the live tree is
invalidated the moment anyone saves a file — and the only remedy anyone had was social: ask every
session to stop, hope they do, and hope nothing lands during the hour the suite takes. That is
fragile, it wastes everyone's time, and it has to be repeated for every re-certification.

Git already provides immutable tree states: commits. And the launch itself deploys a COMMIT
(``git archive HEAD | ssh myriad``), so a commit is not merely a convenient unit to certify — it is
the CORRECT one. Certifying the working directory would certify something that is never deployed.

So this checks out the target commit into an ISOLATED worktree and runs the gauntlet there. Other
sessions keep committing to the main tree and cannot affect the result; the certificate names the
exact SHA it holds for, and ``--check`` says whether that SHA is still what you are about to launch.

WHY THE OUTPUT GOES TO FILES
----------------------------
Every RC is read from a log file, never from a pipe. A ``| tail`` once masked pytest's exit code in
this repo and produced a FALSE GREEN, which is the worst possible failure for a launch gate.

Usage::

    python scripts/certify_commit.py                 # certify HEAD
    python scripts/certify_commit.py --commit <sha>
    python scripts/certify_commit.py --check         # is the stored certificate still valid?
    python scripts/certify_commit.py --quick         # gates only, skip the full suite
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
CERT_PATH = REPO / "outputs" / "commit_certificate.json"
#: Gitignored directories the suite needs but a worktree does not carry. Linked, never copied.
_LINK_DIRS = ("data", "outputs")


def git(*args: str, cwd: Path | None = None, check: bool = True) -> str:
    r = subprocess.run(["git", *args], cwd=str(cwd or REPO), capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed:\n{r.stderr}")
    return (r.stdout or "").strip()


#: Artifacts that MUST resolve inside the worktree, or data-dependent tests fail for infrastructure
#: reasons and the whole certification becomes untrustworthy. Verified, never assumed.
_DATA_PROBES = ("data/raw/fred_macro_x26.csv",)


def _link(src: Path, dst: Path) -> bool:
    """Point the worktree's ``dst`` at the real ``src`` directory, REPLACING any checked-out copy.

    NOTE: THE BUG THIS FIXES (2026-07-27), found when the first certification produced a false red.
    ``data/`` is TRACKED (1,179 files), so ``git worktree add`` creates it — but ~1.2 GB of the
    content underneath is GITIGNORED and therefore absent. The previous version returned True
    whenever ``dst.exists()``, so it reported ``linked: ['data']`` while the files tests actually
    need were still missing: a function claiming success without verifying the thing it claims,
    which is the exact failure class this repo keeps finding. Every data-dependent test then failed,
    and those failures looked like code defects.

    Replacing the directory is correct because the MAIN repo holds both the tracked and the ignored
    files, so the junction is complete where a copy would be 1.2 GB. The tradeoff is deliberate and
    small: data files are taken from the working tree rather than from the certified commit, which
    matters only if the data itself changed — and this gate certifies CODE.
    """
    if not src.is_dir():
        return False
    try:
        # This read ``dst.exists() and not dst.is_symlink()``, which cannot see a junction (that is
        # the whole lesson below). MEASURED 2026-07-27: it was not destructive — CPython 3.11's
        # rmtree refuses to descend a junction, so the call failed and ``ignore_errors`` swallowed
        # it — but it then left a stale link in place and returned True, i.e. reported success
        # without relinking. ``os.rmdir`` severs the LINK deterministically instead of relying on
        # rmtree's version-dependent junction handling.
        if _is_reparse_point(dst):
            os.rmdir(dst)
        elif dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        if dst.exists():
            return True
        if os.name == "nt":
            subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(src)],
                           capture_output=True, check=True)
        else:
            dst.symlink_to(src, target_is_directory=True)
        return True
    except Exception:  # noqa: BLE001 — reported by the probe below, never silently assumed
        return False


def verify_data(worktree: Path) -> list[str]:
    """Which required artifacts are NOT reachable in the worktree? Empty means good to run.

    Without this the harness cannot tell an INFRASTRUCTURE failure from a CODE failure, and a
    missing data link would be reported as a red suite — the most misleading thing a launch gate
    can do.
    """
    return [p for p in _DATA_PROBES if not (worktree / p).exists()]


_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


def _is_reparse_point(p: Path) -> bool:
    """True for a junction or symlink. ``is_symlink()`` alone MISSES Windows junctions."""
    try:
        st = os.lstat(p)
    except OSError:
        return False
    if getattr(st, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT:
        return True
    return p.is_symlink()


def unlink_reparse_points(root: Path) -> list[str]:
    """Remove every junction/symlink under ``root`` LINK-ONLY, before any recursive delete.

    NOTE: THIS EXISTS BECAUSE IT ALREADY HAPPENED (2026-07-27). A previous version junctioned the real
    ``data/`` into the worktree and then cleaned up with ``shutil.rmtree(worktree)``. On Windows a
    junction is NOT a symlink, and the cleanup DELETED THE REAL DATA DIRECTORY — 1,179 tracked files
    (restored from git) and ~1.2 GB of untracked data including the gold panel (not in git).

    NOTE: WHICH CALL DID IT — corrected 2026-07-27 after reproducing all three on this box (Python
    3.11.9, Git for Windows). The original incident note blamed ``shutil.rmtree``; that is WRONG,
    and the distinction matters because believing it leaves the real hazard unguarded:

      * ``shutil.rmtree`` — on the junction ITSELF or on a PARENT containing one: SAFE here.
        CPython treats a name-surrogate reparse point as a link and removes it without descending.
      * ``git worktree remove --force`` — **DESTRUCTIVE**. Git's own recursive removal follows the
        junction and deletes the target. This is what destroyed the data, and it is why severing
        must happen BEFORE the git call, not merely before the Python one.

    ``os.rmdir`` on a junction removes the LINK and never touches the target, so every link is
    severed before anything recursive runs — git or Python. Locked by
    ``tests/test_certify_worktree_safety.py``, which reproduces the loss against a REAL worktree.
    """
    removed: list[str] = []
    if not root.exists():
        return removed
    for dirpath, dirnames, _files in os.walk(root, topdown=True):
        for d in list(dirnames):
            p = Path(dirpath) / d
            if _is_reparse_point(p):
                try:
                    os.rmdir(p)          # LINK ONLY — never recursive, never touches the target
                    removed.append(str(p))
                except OSError:
                    pass
                dirnames.remove(d)       # do NOT descend into it under any circumstances
    return removed


def destroy_worktree(path: Path) -> None:
    """Remove a worktree SAFELY: sever links first, then delete. Order is not optional."""
    unlink_reparse_points(path)
    git("worktree", "remove", "--force", str(path), check=False)
    unlink_reparse_points(path)          # git may have recreated/left one
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def make_worktree(commit: str, path: Path) -> Path:
    if path.exists():
        destroy_worktree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    git("worktree", "add", "--detach", str(path), commit)
    linked = [d for d in _LINK_DIRS if _link(REPO / d, path / d)]
    missing = verify_data(path)
    print(f"[certify] worktree at {path} ({commit[:12]}); linked: {linked or 'none'}")
    if missing:
        # Refuse rather than run: a suite that fails because its DATA is unreachable produces a red
        # that looks exactly like a code defect, and acting on that red is worse than not running.
        raise SystemExit(
            f"[certify] ABORT — required data is not reachable in the worktree: {missing}. "
            "The suite would fail for infrastructure reasons and the reds would be indistinguishable "
            "from code defects.")
    print(f"[certify] data verified reachable: {list(_DATA_PROBES)}")
    return path


def run_step(name: str, argv: list[str], cwd: Path, log_dir: Path,
             timeout: int = 5400) -> dict[str, Any]:
    """Run one gate, RC read from the LOG (never a pipe — a pipe once produced a false green)."""
    log = log_dir / f"{name}.log"
    t0 = time.time()
    with log.open("w", encoding="utf-8", errors="replace") as fh:
        try:
            rc = subprocess.run(argv, cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT,
                                timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = 124
    raw = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-3:]
    # The console here is cp1251; a tool's output can carry characters it cannot encode, and a
    # crash while REPORTING a result would lose the result itself. Keep the log verbatim on disk
    # and print an ASCII-safe rendering.
    tail = [ln.encode("ascii", "replace").decode("ascii") for ln in raw]
    out = {"name": name, "returncode": rc, "seconds": round(time.time() - t0, 1),
           "log": str(log), "tail": tail}
    print(f"[certify] {name:<22} rc={rc:<4} {out['seconds']:>7.1f}s  {tail[-1][:70] if tail else ''}")
    return out


def certify(commit: str, *, quick: bool = False) -> dict[str, Any]:
    sha = git("rev-parse", commit)
    wt = Path(os.environ.get("TMP", "/tmp")) / f"certify_{sha[:12]}"
    log_dir = REPO / "outputs" / "certification" / sha[:12]
    log_dir.mkdir(parents=True, exist_ok=True)
    make_worktree(sha, wt)
    py = str(REPO / ".venv" / "Scripts" / "python.exe")
    if not Path(py).exists():
        py = sys.executable

    steps: list[dict[str, Any]] = []
    try:
        steps.append(run_step("freeze_check", [py, "scripts/freeze.py", "--check"], wt, log_dir))
        steps.append(run_step("ruff", [py, "-m", "ruff", "check", "src", "scripts"], wt, log_dir))
        steps.append(run_step("pretrain_validate",
                              [py, "scripts/pretrain_validate.py", "--self-test"], wt, log_dir))
        if not quick:
            # ONE sequential suite run. Deliberately not parallel: concurrent pytest on this box
            # produced WinError 1455 and spurious CUDA failures — false reds are worse than slow.
            steps.append(run_step("pytest_full", [py, "-m", "pytest", "tests", "-q",
                                                  "-p", "no:warnings", "--no-header"],
                                  wt, log_dir, timeout=10800))
    finally:
        # NEVER a bare rmtree here: the worktree contains junctions to the REAL data, and both
        # rmtree and `git worktree remove` follow them on Windows. This exact cleanup destroyed
        # the data directory once already (2026-07-27).
        destroy_worktree(wt)

    failed = [s["name"] for s in steps if s["returncode"] != 0]
    cert = {
        "commit": sha,
        "commit_short": sha[:12],
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "certified": not failed,
        "failed_steps": failed,
        "steps": steps,
        "quick": quick,
        "note": ("This certificate holds for THIS COMMIT ONLY. It says nothing about the working "
                 "directory, and nothing about any later commit. Re-certify after any change you "
                 "intend to deploy."),
    }
    CERT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CERT_PATH.write_text(json.dumps(cert, indent=2), encoding="utf-8")
    return cert


def check(commit: str = "HEAD") -> dict[str, Any]:
    """Is the stored certificate still valid for what we are about to deploy?"""
    head = git("rev-parse", commit)
    if not CERT_PATH.is_file():
        return {"valid": False, "reason": "no certificate has ever been produced", "head": head}
    cert = json.loads(CERT_PATH.read_text(encoding="utf-8"))
    if not cert.get("certified"):
        return {"valid": False, "head": head, "certified_commit": cert.get("commit"),
                "reason": f"the certificate FAILED: {cert.get('failed_steps')}"}
    if cert.get("commit") != head:
        behind = git("rev-list", "--count", f"{cert['commit']}..{head}", check=False) or "?"
        return {"valid": False, "head": head, "certified_commit": cert.get("commit"),
                "reason": f"the certificate is for {cert['commit'][:12]}, but HEAD is "
                          f"{head[:12]} — {behind} commit(s) later. Re-certify before launching."}
    return {"valid": True, "head": head, "certified_commit": cert["commit"],
            "utc": cert.get("utc"), "quick": cert.get("quick", False)}


def main(argv: list[str] | None = None) -> int:
    from src.utils.console import make_console_safe
    make_console_safe()   # src/utils/console.py — it reads subprocess output with errors="replace"
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commit", default="HEAD")
    ap.add_argument("--check", action="store_true", help="report certificate validity and exit")
    ap.add_argument("--quick", action="store_true", help="gates only; skip the full suite")
    args = ap.parse_args(argv)

    if args.check:
        v = check(args.commit)
        print(json.dumps(v, indent=2))
        return 0 if v.get("valid") else 1

    cert = certify(args.commit, quick=args.quick)
    print("\n" + "=" * 78)
    print(f"  commit    : {cert['commit_short']}")
    print(f"  CERTIFIED : {cert['certified']}"
          + ("" if cert["certified"] else f"   FAILED: {cert['failed_steps']}"))
    print(f"  written   : {CERT_PATH}")
    print("=" * 78)
    return 0 if cert["certified"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
