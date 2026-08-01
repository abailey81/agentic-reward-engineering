"""D20 — the driver lock must test pid IDENTITY, not pid EXISTENCE.

Regression for the campaign execution record §59 and §100, and the deferred-fix register
item 13. **This defect stranded a live line twice:**

  2026-07-31  `h3`   — Windows recycled a dead driver's pid onto `OpenConsole.exe`
  2026-08-01  `leg4` — pid 34216 recycled onto `backgroundTaskHost.exe`, stranding a line that
                       was ALREADY AT C4 for ~14 h. Every relaunch died 12 s in. Every guard green.

`psutil.pid_exists(pid)` answers a weaker question than the lock needs. The identity test is the
owner's process CREATE-TIME, recorded beside the pid: a recycled pid necessarily carries a later
create-time than the one written.

**The asymmetry is the design, and it is tested explicitly.** Breaking a LIVE owner's lock permits
two drivers on one batch — double requeue rounds and corrupted retry accounting, unrecoverable.
Failing to break a dead owner's lock stalls a line — recoverable, and the monitoring cycle now
auto-reaps it. So every ambiguity must resolve to "owned", and `test_ambiguity_resolves_to_owned`
is the test that pins that direction.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from src.cluster.driver import _acquire_driver_lock, _lock_owner_is_live_driver


@pytest.fixture
def live_pid():
    """A real, live, NON-driver process we can name in a lock file."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        yield proc.pid
    finally:
        proc.kill()
        proc.wait(timeout=10)


@pytest.fixture
def live_driver_pid():
    """A live process whose COMMAND LINE looks like a driver (for the legacy-lock path)."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)  # run_campaign_cluster"])
    try:
        yield proc.pid
    finally:
        proc.kill()
        proc.wait(timeout=10)


def _write_lock(path: Path, pid: int, create_time: float | None) -> None:
    path.write_text(json.dumps({"pid": pid, "create_time": create_time, "ts": time.time()}),
                    encoding="utf-8")


# --------------------------------------------------------------------------- #
# THE DEFECT — a RECYCLED pid must not hold a line hostage                     #
# --------------------------------------------------------------------------- #
def test_recycled_pid_lock_is_broken_and_reacquired(tmp_path: Path, live_pid: int) -> None:
    """The live incident, reproduced: pid alive, but it is NOT the process that wrote the lock.

    Pre-fix this raised the refuse-to-double-drive RuntimeError forever -- which is exactly how
    `leg4` sat dead for ~14 h.
    """
    lock = tmp_path / "batch.driver.lock"
    real_ct = psutil.Process(live_pid).create_time()
    _write_lock(lock, live_pid, real_ct - 5000.0)      # a DIFFERENT process, same pid
    _acquire_driver_lock(lock)                          # must NOT raise
    assert json.loads(lock.read_text(encoding="utf-8"))["pid"] == os.getpid()


def test_legacy_lock_whose_pid_is_now_a_non_driver_is_broken(tmp_path: Path,
                                                             live_pid: int) -> None:
    """Locks written BEFORE this fix carry no create_time. They must still self-heal.

    Falls back to identity-by-command-line, which is still strictly stronger than existence.
    This is the case that matters during the deploy itself: in-flight locks are legacy-shaped.
    """
    lock = tmp_path / "batch.driver.lock"
    _write_lock(lock, live_pid, None)
    _acquire_driver_lock(lock)
    assert json.loads(lock.read_text(encoding="utf-8"))["pid"] == os.getpid()


# --------------------------------------------------------------------------- #
# NO REGRESSION — the lock must still do its actual job                        #
# --------------------------------------------------------------------------- #
def test_a_genuinely_live_owner_still_refuses_to_double_drive(tmp_path: Path,
                                                              live_pid: int) -> None:
    """pid AND create_time both match a live process -> the lock holds. The safety property."""
    lock = tmp_path / "batch.driver.lock"
    _write_lock(lock, live_pid, psutil.Process(live_pid).create_time())
    with pytest.raises(RuntimeError, match="refusing to double-drive"):
        _acquire_driver_lock(lock)


def test_legacy_lock_held_by_a_live_driver_still_refuses(tmp_path: Path,
                                                         live_driver_pid: int) -> None:
    """Legacy shape, owner's cmdline looks like a driver -> still refuses. No regression."""
    lock = tmp_path / "batch.driver.lock"
    _write_lock(lock, live_driver_pid, None)
    with pytest.raises(RuntimeError, match="refusing to double-drive"):
        _acquire_driver_lock(lock)


def test_dead_owner_lock_is_still_broken(tmp_path: Path) -> None:
    """The pre-existing self-heal must survive the change (crash-resume stays one-command)."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=10)
    lock = tmp_path / "batch.driver.lock"
    _write_lock(lock, proc.pid, None)
    _acquire_driver_lock(lock)
    assert json.loads(lock.read_text(encoding="utf-8"))["pid"] == os.getpid()


def test_torn_lock_is_broken(tmp_path: Path) -> None:
    lock = tmp_path / "batch.driver.lock"
    lock.write_text("{not json", encoding="utf-8")
    _acquire_driver_lock(lock)
    assert json.loads(lock.read_text(encoding="utf-8"))["pid"] == os.getpid()


# --------------------------------------------------------------------------- #
# THE WRITE SIDE — identity must actually be RECORDED, or none of it works     #
# --------------------------------------------------------------------------- #
def test_the_lock_records_the_owner_create_time(tmp_path: Path) -> None:
    """A fix that reads an identity nobody writes is a fix that does nothing."""
    lock = tmp_path / "batch.driver.lock"
    _acquire_driver_lock(lock)
    payload = json.loads(lock.read_text(encoding="utf-8"))
    assert payload["pid"] == os.getpid()
    assert payload["create_time"] == pytest.approx(psutil.Process(os.getpid()).create_time())


# --------------------------------------------------------------------------- #
# THE DIRECTION OF THE ERROR — ambiguity must resolve to "owned"             #
# --------------------------------------------------------------------------- #
def test_ambiguity_resolves_to_owned(monkeypatch: pytest.MonkeyPatch, live_pid: int) -> None:
    """If the owner cannot be inspected, the lock is treated as HELD -- never as stale.

    Breaking a live owner's lock corrupts the run; refusing to break a dead one stalls a line and
    is now auto-reaped by the monitoring cycle. The two errors are not symmetric, so the predicate
    must not be either.
    """
    class _Denied(psutil.Process):  # type: ignore[misc]
        def create_time(self):  # noqa: ANN201
            raise psutil.AccessDenied(self.pid)

        def cmdline(self):  # noqa: ANN201
            raise psutil.AccessDenied(self.pid)

    monkeypatch.setattr(psutil, "Process", _Denied)
    real_ct = 1.0
    assert _lock_owner_is_live_driver(live_pid, real_ct) is True     # create_time unreadable
    assert _lock_owner_is_live_driver(live_pid, None) is True        # cmdline unreadable


def test_empty_cmdline_resolves_to_owned(monkeypatch: pytest.MonkeyPatch, live_pid: int) -> None:
    """An EMPTY cmdline trivially "does not contain run_campaign_cluster".

    Reading that as "not a driver" would delete a live driver's lock -- the exact hole the
    monitoring-side detector had before the reaper was written. Must be OWNED.
    """
    class _Empty(psutil.Process):  # type: ignore[misc]
        def cmdline(self):  # noqa: ANN201
            return []

    monkeypatch.setattr(psutil, "Process", _Empty)
    assert _lock_owner_is_live_driver(live_pid, None) is True


def test_a_dead_pid_is_never_owned() -> None:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=10)
    assert _lock_owner_is_live_driver(proc.pid, None) is False
    assert _lock_owner_is_live_driver(-1, None) is False
