"""Tests for the canonical results IO loader (FINAL_PLAN F.13, audit C-1)."""

from __future__ import annotations

import json

import pytest

import src.io.results as results_mod
from src.io.results import REQUIRED_FIELDS, load_all, load_run, write_run


def _record(run_id: str = "run-0001", arm: str = "scalar", **overrides) -> dict:
    rec = {
        "run_id": run_id,
        "arm": arm,
        "seed": 12345,
        "fold": 0,
        "candidate_id": "cand-42",
        "generation": 3,
        "reward_source_hash": "deadbeef",
        "feedback_block": {"type": arm, "text": "tail stats here"},
        "metrics": {"sharpe": 1.23, "cvar_95": -0.04},
        "wall_clock": 12.5,
        "env_fingerprint": {"python": "3.11.0", "numpy": "2.3.5"},
    }
    rec.update(overrides)
    return rec


def test_write_read_round_trip(results_dir) -> None:
    """write_run then load_run round-trips a record without loss."""
    rec = _record()
    path = write_run(rec, results_dir)
    assert path.is_file()
    loaded = load_run(rec["run_id"], results_dir)
    for k, v in rec.items():
        assert loaded[k] == v


@pytest.mark.parametrize("missing", list(REQUIRED_FIELDS))
def test_loader_fails_on_missing_required_field(results_dir, missing) -> None:
    """write_run raises KeyError naming a missing required field."""
    rec = _record()
    del rec[missing]
    with pytest.raises(KeyError) as exc:
        write_run(rec, results_dir)
    assert missing in str(exc.value)


def test_load_all_returns_all_and_respects_filter(results_dir) -> None:
    """load_all returns every written run and honours a simple filter."""
    write_run(_record(run_id="r1", arm="scalar"), results_dir)
    write_run(_record(run_id="r2", arm="distributional"), results_dir)
    write_run(_record(run_id="r3", arm="scalar"), results_dir)

    everything = load_all(results_dir)
    assert {r["run_id"] for r in everything} == {"r1", "r2", "r3"}

    scalars = load_all(results_dir, filter={"arm": "scalar"})
    assert {r["run_id"] for r in scalars} == {"r1", "r3"}


def test_reward_source_written_and_reloadable(results_dir) -> None:
    """reward.py is written when reward_source is present and reloadable."""
    source = "def reward(weights, returns, prev_weights, port_ret, info):\n    return port_ret, {}, None\n"
    rec = _record(run_id="rw", reward_source=source)
    write_run(rec, results_dir)
    assert (results_dir / "rw" / "reward.py").read_text() == source

    loaded = load_run("rw", results_dir)
    assert loaded["reward_source"] == source


# --------------------------------------------------------------------------- #
# Atomic record write (campaign-robustness §F): a kill mid-write must never     #
# leave a truncated record.json that crashes --resume.                         #
# --------------------------------------------------------------------------- #
def test_failed_midwrite_leaves_prior_record_intact(results_dir, monkeypatch) -> None:
    """A simulated crash mid-``json.dump`` leaves the PRIOR record.json intact and parseable.

    Because ``write_run`` dumps to a ``.json.tmp`` sibling and only ``os.replace``s on success, a
    failure during the dump can corrupt at most the temp file — never the live ``record.json``. So a
    re-write that fails mid-flight cannot clobber a previously-good record, and ``--resume`` (which
    parses ``record.json``) still reads the old one.
    """
    rec = _record(run_id="atomic")
    write_run(rec, results_dir)  # a first, GOOD write
    good_path = results_dir / "atomic" / "record.json"
    original_bytes = good_path.read_bytes()

    # Now make json.dump blow up PART-WAY through the next write (after some bytes may have hit the
    # temp file) and confirm the live record.json is untouched.
    real_dump = json.dump

    def _exploding_dump(obj, fp, **kw):  # noqa: ANN001
        fp.write('{"run_id": "atomic", "partial": true')  # write a truncated prefix into the temp file
        raise OSError("simulated power loss mid-write")

    monkeypatch.setattr(results_mod.json, "dump", _exploding_dump)
    with pytest.raises(OSError, match="simulated power loss"):
        write_run({**rec, "metrics": {"changed": 1}}, results_dir)
    monkeypatch.setattr(results_mod.json, "dump", real_dump)

    # The live record.json is byte-identical to the original good write (os.replace never ran).
    assert good_path.read_bytes() == original_bytes
    # load_run still parses it (no truncation reached the canonical path).
    assert load_run("atomic", results_dir)["run_id"] == "atomic"


def test_no_truncated_record_is_ever_read_after_midwrite(results_dir, monkeypatch) -> None:
    """``load_all`` ignores a stray ``.json.tmp`` and never surfaces a half-written record.

    Even for a BRAND-NEW run whose first write fails mid-dump, the directory may contain only a
    ``.json.tmp`` — ``load_all`` keys off exactly ``record.json``, so it skips the dir entirely rather
    than handing a corrupt record to the resume path. This is the property that keeps ``--resume`` from
    crashing on a record that was being written when the process was killed.
    """
    def _exploding_dump(obj, fp, **kw):  # noqa: ANN001
        fp.write('{"truncated":')  # only a fragment lands in the temp file
        raise OSError("killed mid-dump")

    monkeypatch.setattr(results_mod.json, "dump", _exploding_dump)
    with pytest.raises(OSError):
        write_run(_record(run_id="newdir"), results_dir)

    run_dir = results_dir / "newdir"
    # No canonical record.json was produced (only possibly a .json.tmp fragment).
    assert not (run_dir / "record.json").is_file()
    # load_all skips the dir -> no corrupt record reaches a resume.
    assert load_all(results_dir) == []
    with pytest.raises(FileNotFoundError):
        load_run("newdir", results_dir)


def test_load_all_corrupt_record_fails_loud_with_recovery_guidance(tmp_path) -> None:
    """2026-07-06 A4: a corrupt record.json must fail LOUD (never silently regenerate — an LLM-arm
    record is irreplaceable) with an ACTIONABLE message naming the path + the mirror recovery route."""
    import json as _json

    import pytest as _pytest

    from src.io.results import load_all, write_run

    rec = {
        "run_id": "a-s0", "arm": "a", "seed": 0, "fold": 0, "candidate_id": "c0",
        "generation": 0, "reward_source_hash": "h", "feedback_block": "",
        "metrics": {"val_fitness": 0.1}, "wall_clock": 1.0, "env_fingerprint": "t",
    }
    write_run(rec, tmp_path)
    # corrupt it: drop a REQUIRED field (arm) so load_run's schema validation raises
    p = tmp_path / "a-s0" / "record.json"
    broken = _json.loads(p.read_text(encoding="utf-8"))
    broken.pop("arm")
    p.write_text(_json.dumps(broken), encoding="utf-8")
    with _pytest.raises(ValueError, match="CORRUPT archive record.*mirror"):
        load_all(tmp_path)
