"""DEEP platform/util-layer tests targeting the reproducibility-critical, low-coverage modules.

Scope (NO overlap with tests/test_utils.py, test_results_io.py, test_regimes.py,
test_monitoring_resources.py — those are skimmed and their cases are NOT repeated):

  * src/utils/provenance.py   — hashing determinism + collision-sensitivity + LF/CRLF + git never-raises
  * src/io/results.py         — exact write→read round-trip, schema enforcement, deterministic ordering,
                                sidecar/env.json integrity (tamper detection), optional-field survival
  * src/utils/config.py       — typed structure, nested-override merges without sibling clobber, unknown raises
  * src/utils/env.py + preload — public helpers are safe + idempotent + crash-free on missing-optional
  * src/regimes/definition.py — labels PARTITION the timeline; determinism; boundary dates; metamorphic monotone
  * src/utils/monitoring.py + logging — setup is idempotent + isolated (no cross-test logger-level leakage)

Property tests use the conftest 'deterministic' Hypothesis profile (derandomize=True): a failure is a
stable, replayable counter-example, not a flake. IO uses tmp_path; env uses monkeypatch; numerics use tight
atol; RNG is seeded.
"""

from __future__ import annotations

import json
import logging
import string
from pathlib import Path

import numpy as np
import pytest
import yaml

import src.io.results as results_mod
import src.utils.logging as logmod
import src.utils.provenance as pv
from src.data.panel import Panel
from src.io.results import load_all, load_run, write_run
from src.regimes.definition import (
    CALM,
    NORMAL,
    STRESS,
    independent_regime_count,
    label_regimes,
)
from src.utils.config import DotDict, cfg_get, config_dir, load_all as cfg_load_all, load_config, repo_root
from src.utils.env import load_env
from src.utils.monitoring import RunMonitor
from src.utils.preload import preload

hyp = pytest.importorskip("hypothesis")
from hypothesis import given  # noqa: E402
from hypothesis import strategies as st  # noqa: E402


# ============================================================================ #
# provenance.py — content hashing is the reproducibility root of trust         #
# ============================================================================ #
class TestProvenanceHashing:
    @given(st.binary(max_size=512))
    def test_sha256_bytes_deterministic_and_64_hex(self, data: bytes) -> None:
        """sha256_bytes is a pure function: same bytes → same 64-char lowercase-hex digest."""
        d1 = pv.sha256_bytes(data)
        d2 = pv.sha256_bytes(data)
        assert d1 == d2
        assert len(d1) == 64
        assert all(c in string.hexdigits for c in d1)
        assert d1 == d1.lower()

    @given(st.binary(min_size=1, max_size=256), st.integers(min_value=0))
    def test_one_byte_flip_changes_digest(self, data: bytes, idx: int) -> None:
        """Adversarial: flipping a SINGLE bit of one byte must change the digest (collision-sensitive)."""
        i = idx % len(data)
        mutated = bytearray(data)
        mutated[i] ^= 0x01  # flip the low bit of byte i
        assert pv.sha256_bytes(bytes(mutated)) != pv.sha256_bytes(data)

    def test_sha256_bytes_matches_hashlib_reference(self) -> None:
        """Known-answer vector: empty input is the canonical SHA-256 of zero bytes."""
        assert pv.sha256_bytes(b"") == (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )

    def test_sha256_text_is_utf8_of_bytes(self) -> None:
        """sha256_text(s) == sha256_bytes(s.encode('utf-8')) for non-ASCII too (round-trip via encoding)."""
        s = "café—λ—Ω"
        assert pv.sha256_text(s) == pv.sha256_bytes(s.encode("utf-8"))

    def test_lf_crlf_differ_textually_as_documented(self) -> None:
        """CRLF vs LF are DIFFERENT byte streams → different digests (no newline normalisation)."""
        lf = "a\nb\nc"
        crlf = "a\r\nb\r\nc"
        assert lf != crlf
        assert pv.sha256_text(lf) != pv.sha256_text(crlf)
        # but each is internally consistent with its own bytes
        assert pv.sha256_text(crlf) == pv.sha256_bytes(b"a\r\nb\r\nc")

    def test_sha256_obj_key_order_invariant_but_value_sensitive(self) -> None:
        """Canonical-JSON hash: key order is irrelevant; nested-value change flips the digest."""
        a = {"x": {"b": 2, "a": 1}, "y": [1, 2, 3]}
        b = {"y": [1, 2, 3], "x": {"a": 1, "b": 2}}
        assert pv.sha256_obj(a) == pv.sha256_obj(b)
        c = {"x": {"a": 1, "b": 2}, "y": [1, 2, 4]}  # one element changed
        assert pv.sha256_obj(a) != pv.sha256_obj(c)

    def test_sha256_obj_distinguishes_int_from_string(self) -> None:
        """Metamorphic: 1 and '1' canonicalise differently (type is part of the content)."""
        assert pv.sha256_obj({"k": 1}) != pv.sha256_obj({"k": "1"})

    def test_sha256_file_streams_equal_inmemory(self, tmp_path: Path) -> None:
        """Streaming file hash == in-memory hash of the same bytes, across a chunk boundary."""
        payload = bytes(range(256)) * 4096  # 1 MiB == chunk size, exercises the iter loop boundary
        f = tmp_path / "blob.bin"
        f.write_bytes(payload)
        assert pv.sha256_file(f) == pv.sha256_bytes(payload)
        assert pv.sha256_file(f, chunk=7) == pv.sha256_bytes(payload)  # tiny chunk → same digest


class TestProvenanceGit:
    def test_git_commit_returns_str_or_none_and_never_raises(self) -> None:
        for short in (False, True):
            out = pv.git_commit(short=short)
            assert out is None or isinstance(out, str)

    def test_git_dirty_returns_bool_or_none(self) -> None:
        out = pv.git_dirty()
        assert out is None or isinstance(out, bool)

    def test_git_commit_returns_none_when_git_binary_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Adversarial: if `git` cannot be invoked, git_commit/git_dirty swallow it and return None."""
        def _boom(*a, **k):  # noqa: ANN002, ANN003
            raise FileNotFoundError("no git on PATH")

        monkeypatch.setattr(pv.subprocess, "run", _boom)
        assert pv.git_commit() is None
        assert pv.git_commit(short=True) is None
        assert pv.git_dirty() is None

    def test_git_commit_swallows_called_process_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A non-zero git exit (not a work-tree) → None, not an exception."""
        import subprocess as _sp

        def _fail(*a, **k):  # noqa: ANN002, ANN003
            raise _sp.CalledProcessError(128, "git")

        monkeypatch.setattr(pv.subprocess, "run", _fail)
        assert pv.git_commit() is None
        assert pv.git_dirty() is None

    def test_env_fingerprint_has_all_keys_and_jsonable(self) -> None:
        """The fingerprint is fully serialisable (it is embedded in every record) and self-consistent."""
        fp = pv.env_fingerprint()
        assert {"python", "platform", "git_commit", "git_dirty", "packages"} <= set(fp)
        # every tracked package is present (installed or the literal "not-installed")
        for pkg in pv._TRACKED_PACKAGES:
            assert pkg in fp["packages"]
            assert isinstance(fp["packages"][pkg], str)
        json.dumps(fp)  # must not raise — record.json embeds this verbatim


# ============================================================================ #
# io/results.py — the ONLY sanctioned results read/write path                  #
# ============================================================================ #
def _record(run_id: str = "run-0001", arm: str = "scalar", **overrides: object) -> dict:
    rec: dict = {
        "run_id": run_id,
        "arm": arm,
        "seed": 12345,
        "fold": 0,
        "candidate_id": "cand-42",
        "generation": 3,
        "reward_source_hash": "deadbeef",
        "feedback_block": {"type": arm, "text": "tail stats here", "nums": [1, 2, 3]},
        "metrics": {"sharpe": 1.23, "cvar_95": -0.04, "val_returns": [0.01, -0.02, 0.003]},
        "wall_clock": 12.5,
        "env_fingerprint": {"python": "3.11.0", "numpy": "2.3.5"},
    }
    rec.update(overrides)
    return rec


class TestResultsRoundTrip:
    def test_round_trip_is_value_identical(self, results_dir: Path) -> None:
        """A full record survives write→load Value-identical for every key (deep structures included)."""
        rec = _record()
        write_run(rec, results_dir)
        loaded = load_run(rec["run_id"], results_dir)
        for k, v in rec.items():
            assert loaded[k] == v, f"field {k!r} did not round-trip"
        # nested containers preserved exactly (not stringified)
        assert loaded["metrics"]["val_returns"] == [0.01, -0.02, 0.003]
        assert loaded["feedback_block"]["nums"] == [1, 2, 3]

    def test_on_disk_record_is_canonical_json_sorted_keys(self, results_dir: Path) -> None:
        """write_run persists sorted-key JSON → byte-deterministic on disk (re-write yields same bytes)."""
        rec = _record(run_id="canon")
        p1 = write_run(rec, results_dir)
        first = p1.read_bytes()
        # reorder the dict's insertion order — sorted-key dump must produce identical bytes
        reordered = {k: rec[k] for k in reversed(list(rec))}
        p2 = write_run(reordered, results_dir)
        assert p2.read_bytes() == first
        top = json.loads(first.decode("utf-8"))
        assert list(top.keys()) == sorted(top.keys())  # deterministic ordering

    def test_optional_fields_survive_round_trip(self, results_dir: Path) -> None:
        """Optional provenance fields (frozen/test_returns/per_period_pnl) round-trip when present."""
        rec = _record(
            run_id="opt",
            frozen=True,
            test_returns=[0.001, -0.002, 0.0],
            per_period_pnl=[10.0, -5.0, 0.0],
        )
        write_run(rec, results_dir)
        loaded = load_run("opt", results_dir)
        assert loaded["frozen"] is True
        assert loaded["test_returns"] == [0.001, -0.002, 0.0]
        assert loaded["per_period_pnl"] == [10.0, -5.0, 0.0]

    def test_load_all_orders_by_run_id_deterministically(self, results_dir: Path) -> None:
        """load_all returns records sorted by run_id regardless of write order."""
        for rid in ("r-c", "r-a", "r-b"):
            write_run(_record(run_id=rid), results_dir)
        ids = [r["run_id"] for r in load_all(results_dir)]
        assert ids == ["r-a", "r-b", "r-c"]

    def test_load_all_empty_or_missing_root(self, tmp_path: Path) -> None:
        """A missing root → [] (no crash); a dir with no record.json is skipped."""
        assert load_all(tmp_path / "does-not-exist") == []
        empty = tmp_path / "empty"
        (empty / "junkdir").mkdir(parents=True)
        (empty / "junkdir" / "notes.txt").write_text("hi", encoding="utf-8")
        assert load_all(empty) == []


class TestResultsSchemaEnforcement:
    def test_write_rejects_missing_required_field_naming_it(self, results_dir: Path) -> None:
        """Every required field is enforced on WRITE and the error names the offender."""
        for field in results_mod.REQUIRED_FIELDS:
            rec = _record(run_id=f"miss-{field}")
            del rec[field]
            with pytest.raises(KeyError) as exc:
                write_run(rec, results_dir)
            assert field in str(exc.value)

    def test_load_rejects_record_with_required_field_removed(self, results_dir: Path) -> None:
        """A hand-corrupted record.json missing a required field fails loudly on LOAD too."""
        rec = _record(run_id="corrupt")
        write_run(rec, results_dir)
        rp = results_dir / "corrupt" / "record.json"
        data = json.loads(rp.read_text(encoding="utf-8"))
        del data["metrics"]
        rp.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(KeyError, match="metrics"):
            load_run("corrupt", results_dir)

    def test_load_missing_run_raises_filenotfound(self, results_dir: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_run("ghost", results_dir)


class TestResultsSidecarIntegrity:
    def test_reward_sidecar_written_and_verified(self, results_dir: Path) -> None:
        src = "def reward(weights, returns, prev_weights, port_ret, info):\n    return port_ret, {}, None\n"
        write_run(_record(run_id="rw", reward_source=src), results_dir)
        assert (results_dir / "rw" / "reward.py").read_text(encoding="utf-8") == src
        assert load_run("rw", results_dir)["reward_source"] == src

    def test_tampered_reward_sidecar_raises_valueerror(self, results_dir: Path) -> None:
        """audit C-2: a reward.py edited to differ from the embedded copy is caught on the read path."""
        src = "def reward(w, r, p, pr, info):\n    return 0.0, {}, None\n"
        write_run(_record(run_id="tamper", reward_source=src), results_dir)
        (results_dir / "tamper" / "reward.py").write_text(src + "# injected\n", encoding="utf-8")
        with pytest.raises(ValueError, match="reward.py sidecar mismatch"):
            load_run("tamper", results_dir)

    def test_prompt_sidecar_written_and_tamper_detected(self, results_dir: Path) -> None:
        """The prompt.txt sidecar round-trips and a mismatch raises (symmetric with reward.py)."""
        write_run(_record(run_id="pr", prompt="design a tail-aware reward"), results_dir)
        assert (results_dir / "pr" / "prompt.txt").read_text(encoding="utf-8") == "design a tail-aware reward"
        assert load_run("pr", results_dir)["prompt"] == "design a tail-aware reward"
        (results_dir / "pr" / "prompt.txt").write_text("DIFFERENT", encoding="utf-8")
        with pytest.raises(ValueError, match="prompt.txt sidecar mismatch"):
            load_run("pr", results_dir)

    def test_empty_prompt_writes_no_sidecar(self, results_dir: Path) -> None:
        """An empty-string prompt is treated as absent — no prompt.txt sidecar is written."""
        write_run(_record(run_id="emptyp", prompt=""), results_dir)
        assert not (results_dir / "emptyp" / "prompt.txt").exists()

    def test_reward_sidecar_reattached_when_record_lacks_embedded_copy(self, results_dir: Path) -> None:
        """A hand-placed reward.py with no embedded reward_source is reattached on load."""
        write_run(_record(run_id="reattach"), results_dir)
        side = "def reward(w, r, p, pr, info):\n    return 1.0, {}, None\n"
        (results_dir / "reattach" / "reward.py").write_text(side, encoding="utf-8")
        loaded = load_run("reattach", results_dir)
        assert loaded["reward_source"] == side

    def test_env_json_integrity_verified_and_tamper_detected(self, results_dir: Path) -> None:
        """final-audit #37: a matching env.json reattaches; a tampered one raises ValueError."""
        env_obj = {"python": "3.11.0", "schema": "capture_env/1", "seed": 7}
        digest = pv.sha256_obj(env_obj)
        rec = _record(run_id="env", env_fingerprint={"python": "3.11.0", "env_json_sha256": digest})
        write_run(rec, results_dir)
        ep = results_dir / "env" / "env.json"
        with ep.open("w", encoding="utf-8") as fh:
            json.dump(env_obj, fh)
        loaded = load_run("env", results_dir)
        assert loaded["env"] == env_obj  # reattached on a clean match
        # tamper: rewrite env.json with a different content → digest no longer matches
        with ep.open("w", encoding="utf-8") as fh:
            json.dump({**env_obj, "seed": 999}, fh)
        with pytest.raises(ValueError, match="env.json sha256 mismatch"):
            load_run("env", results_dir)


# ============================================================================ #
# config.py — config/*.yaml is the single source of truth                      #
# ============================================================================ #
class TestConfigLoading:
    def test_every_yaml_loads_to_dotdict(self) -> None:
        """Each config/*.yaml loads into a DotDict (typed, attribute-accessible) and is non-empty."""
        for path in sorted(config_dir().glob("*.yaml")):
            cfg = load_config(path.stem)
            assert isinstance(cfg, DotDict)
            assert len(cfg) > 0, f"{path.stem}.yaml loaded empty"

    def test_load_config_suffix_optional_and_cached(self) -> None:
        """'regimes' and 'regimes.yaml' resolve identically; lru_cache returns the SAME object.

        ``cache_clear()`` first (added 2026-07-26, deep review): without it the two suffix spellings
        are SEPARATE cache keys, so if ``config/regimes.yaml`` changes on disk between the two calls
        — which really happened when a config was edited mid-suite — one key holds the pre-edit parse
        and the other the post-edit parse, and ``a == b`` fails for a reason unrelated to what this
        test is about.
        """
        load_config.cache_clear()
        a = load_config("regimes")
        b = load_config("regimes.yaml")
        assert a == b
        assert load_config("regimes") is a  # memoised

    def test_every_config_loads_as_utf8_not_the_platform_locale(self) -> None:
        """Every config/*.yaml must load EXACTLY as its on-disk UTF-8 bytes parse.

        Regression lock for the CRITICAL defect found by the 2026-07-26 deep review:
        ``load_config`` used ``path.open()``, i.e. the platform LOCALE codec. On the Windows
        development box (locale cp1251) that silently mis-decoded every non-ASCII byte, so 30+
        registered ``config/preregistration.yaml::model_suite`` values (and
        ``config/m2_models.yaml::core`` / ``excluded_by_design``) came back with U+2014 "—"
        mojibake'd to "вЂ”" — the LOADED design of record differed from the file on disk AND
        differed between machines, which breaks the protocol layer of the reproducibility claim.
        Bytes undefined in the locale codec (e.g. U+2605) raised UnicodeDecodeError outright.

        This assertion fails on ANY machine whose locale codec is not UTF-8 if the explicit
        ``encoding="utf-8"`` is ever dropped again.

        ``cache_clear()`` first: ``load_config`` is ``lru_cache``d, so an entry memoised BEFORE a
        config file changed on disk would make this compare a stale parse against current bytes and
        fail for a reason that has nothing to do with encoding. (Observed for real: editing configs
        while the suite was running turned both this test and the pre-existing
        ``test_load_config_suffix_optional_and_cached`` RED. Clearing keeps the test measuring the
        DECODE path, which is what it is for.)
        """
        load_config.cache_clear()
        for path in sorted(config_dir().glob("*.yaml")):
            on_disk = yaml.safe_load(path.read_bytes().decode("utf-8")) or {}
            assert dict(load_config(path.stem)) == on_disk, (
                f"{path.name} did not load as UTF-8 — load_config must pass encoding='utf-8' "
                "explicitly, never the platform locale codec"
            )

    def test_load_config_preserves_non_ascii_characters_verbatim(self) -> None:
        """A known non-ASCII config VALUE survives loading byte-for-byte (companion to the sweep above).

        Explicit and self-describing so the failure message names the actual corruption: the em-dash
        must be U+2014, never its cp1251 mis-decode. Uses a registered ``model_suite`` string, i.e.
        part of the design of record, not an incidental comment.

        ``cache_clear()`` for the same reason as the sweep above: measure the decode, not the cache.
        """
        load_config.cache_clear()
        note = load_config("preregistration")["model_suite"]["served_variant_disclosure"]
        assert "—" in note, "the em-dash in a registered model_suite value was not preserved"
        assert "вЂ”" not in note, "cp1251 mis-decode of U+2014 leaked into a loaded config value"

    def test_unknown_config_raises_filenotfound_listing_available(self) -> None:
        with pytest.raises(FileNotFoundError) as exc:
            load_config("definitely_not_a_config")
        assert "available" in str(exc.value)

    def test_required_typed_keys_environment(self) -> None:
        """environment.yaml carries the MDP structure with the expected types (single source of truth)."""
        env = load_config("environment")
        assert int(env.require("universe").require("n_assets")) == 30
        assert isinstance(env.require("universe").require("include_cash"), bool)
        assert isinstance(env.require("state").require("lookback_days"), int)
        costs = env.require("costs")
        assert isinstance(costs.require("grid_bps"), list)
        assert all(isinstance(b, int) for b in costs["grid_bps"])

    def test_required_typed_keys_data_splits(self) -> None:
        """data.yaml splits expose train/val/test each with start+end (the pre-registered windows)."""
        splits = load_config("data").require("splits")
        for leg in ("train", "val", "test"):
            seg = splits.require(leg)
            assert "start" in seg and "end" in seg

    def test_regimes_thresholds_typed(self) -> None:
        thr = load_config("regimes").require("vix_thresholds")
        assert float(thr["calm"]) <= float(thr["stress"])

    def test_repo_root_contains_config_and_pyproject(self) -> None:
        root = repo_root()
        assert (root / "config").is_dir()
        assert (root / "pyproject.toml").is_file()

    def test_cfg_load_all_keys_match_files(self) -> None:
        """load_all() is keyed by stem and covers exactly the *.yaml on disk."""
        loaded = cfg_load_all()
        stems = {p.stem for p in config_dir().glob("*.yaml")}
        assert set(loaded) == stems


class TestDotDictMerge:
    def test_nested_override_does_not_clobber_siblings(self) -> None:
        """A deep override touching one leaf must leave sibling keys at every level intact."""
        base = {"a": {"x": 1, "y": 2, "deep": {"p": 10, "q": 20}}, "b": 99}
        d = DotDict(base)
        # emulate a config override merge on a copy: change only a.deep.p
        merged = {**base, "a": {**base["a"], "deep": {**base["a"]["deep"], "p": 11}}}
        m = DotDict(merged)
        assert m.require("a").require("deep").require("p") == 11
        assert m.require("a").require("deep").require("q") == 20  # sibling untouched
        assert m.require("a").require("x") == 1                   # sibling at parent level untouched
        assert m.require("b") == 99                               # top-level sibling untouched
        # the original DotDict is unmutated (override worked on a copy)
        assert d.require("a").require("deep").require("p") == 10

    def test_nested_attribute_access_wraps_dicts(self) -> None:
        d = DotDict({"splits": {"train": {"start": "2005-01-01"}}})
        assert isinstance(d.splits, DotDict)
        assert d.splits.train.start == "2005-01-01"

    def test_require_distinguishes_missing_from_null(self) -> None:
        d = DotDict({"present": 0, "nullish": None})
        assert d.require("present") == 0  # falsy-but-present is fine
        with pytest.raises(KeyError, match="missing"):
            d.require("absent")
        with pytest.raises(ValueError, match="null"):
            d.require("nullish")

    def test_getattr_raises_attributeerror_for_missing(self) -> None:
        d = DotDict({"a": 1})
        with pytest.raises(AttributeError):
            _ = d.nope


class TestCfgGet:
    def test_cfg_get_dict_dotdict_object_and_none(self) -> None:
        """cfg_get reads dicts, DotDicts, attribute objects, and tolerates None / missing keys."""
        assert cfg_get({"k": 5}, "k") == 5
        assert cfg_get(DotDict({"k": 6}), "k") == 6
        assert cfg_get(None, "k", default="d") == "d"
        assert cfg_get({"k": 5}, "missing", default=42) == 42

        class _O:
            attr = "v"

        assert cfg_get(_O(), "attr") == "v"
        assert cfg_get(_O(), "absent", default="fallback") == "fallback"


# ============================================================================ #
# env.py + preload.py — entry-point helpers, must be safe + idempotent         #
# ============================================================================ #
class TestEnvAndPreload:
    def test_load_env_is_idempotent_and_never_raises(self) -> None:
        load_env()
        load_env()  # second call must be a clean no-op (idempotent)

    def test_load_env_degrades_when_dotenv_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If python-dotenv is somehow unimportable, load_env returns silently (no crash)."""
        import builtins

        real_import = builtins.__import__

        def _no_dotenv(name, *a, **k):  # noqa: ANN001, ANN002, ANN003
            if name == "dotenv":
                raise ImportError("simulated missing dotenv")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", _no_dotenv)
        load_env()  # must not raise

    def test_load_env_does_not_override_shell_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Documented contract: load_dotenv never overrides an already-set env var (shell wins)."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "shell-wins-sentinel")
        load_env()
        import os

        assert os.environ["ANTHROPIC_API_KEY"] == "shell-wins-sentinel"

    def test_preload_is_safe_and_idempotent(self) -> None:
        """preload imports pyarrow-before-torch; safe to call repeatedly and never raises."""
        preload()
        preload()

    def test_preload_strict_raises_when_pyarrow_absent(self, monkeypatch) -> None:
        """H2: strict=True FAILS LOUD on an ImportError (production precondition), while the default
        tolerant path swallows it (test/conftest env)."""
        import builtins

        real_import = builtins.__import__

        def _no_pyarrow(name, *args, **kwargs):
            if name == "pyarrow" or name.startswith("pyarrow."):
                raise ImportError("simulated: pyarrow unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_pyarrow)
        # tolerant default: swallows the failure (keeps the suite collectable in a minimal env)
        preload()
        # production contract: raises so a real run cannot proceed to a torch-first SIGSEGV
        with pytest.raises(ImportError, match="pyarrow"):
            preload(strict=True)


# ============================================================================ #
# regimes/definition.py — labels must PARTITION the timeline                   #
# ============================================================================ #
def _vix_panel(vix: np.ndarray) -> Panel:
    t = int(vix.size)
    returns = np.zeros((t, 2), dtype=np.float64)
    dates = np.arange("2010-01-04", t, dtype="datetime64[D]")
    return Panel(returns=returns, vix=vix.astype(np.float64), dates=dates, asset_ids=np.arange(2, dtype=np.int64))


class TestRegimePartition:
    @given(
        st.lists(st.floats(min_value=0.0, max_value=120.0, allow_nan=False), min_size=1, max_size=120),
    )
    def test_every_date_gets_exactly_one_valid_label(self, vix_list: list[float]) -> None:
        """PARTITION property: |labels| == T, each label in {0,1,2}, exactly one per date."""
        vix = np.asarray(vix_list)
        cfg = load_config("regimes")
        labels = label_regimes(_vix_panel(vix), cfg)
        assert labels.shape == (vix.size,)
        assert set(np.unique(labels)).issubset({CALM, NORMAL, STRESS})
        # exactly-one: each label is a single scalar int per index (trivially true for a 1-D array,
        # asserted via dtype + no NaN sentinel possible in an int array)
        assert labels.dtype == np.int64

    def test_boundary_dates_map_to_documented_regime(self) -> None:
        """Boundaries: strict < calm = CALM; == calm and == stress = NORMAL (inclusive); > stress = STRESS."""
        cfg = load_config("regimes")  # calm=15, stress=25
        calm = float(cfg["vix_thresholds"]["calm"])
        stress = float(cfg["vix_thresholds"]["stress"])
        eps = 1e-9
        vix = np.array([calm - eps, calm, (calm + stress) / 2, stress, stress + eps])
        labels = label_regimes(_vix_panel(vix), cfg)
        np.testing.assert_array_equal(labels, np.array([CALM, NORMAL, NORMAL, NORMAL, STRESS]))

    def test_label_regimes_deterministic(self) -> None:
        cfg = load_config("regimes")
        vix = np.linspace(5.0, 40.0, 200)
        a = label_regimes(_vix_panel(vix), cfg)
        b = label_regimes(_vix_panel(vix), cfg)
        np.testing.assert_array_equal(a, b)

    @given(st.lists(st.floats(min_value=0.0, max_value=120.0, allow_nan=False), min_size=2, max_size=80))
    def test_metamorphic_monotone_in_vix(self, vix_list: list[float]) -> None:
        """Metamorphic: labels are MONOTONE non-decreasing in VIX (higher VIX never → calmer regime)."""
        vix = np.asarray(vix_list)
        cfg = load_config("regimes")
        labels = label_regimes(_vix_panel(vix), cfg)
        order = np.argsort(vix, kind="stable")
        sorted_labels = labels[order]
        assert np.all(np.diff(sorted_labels) >= 0), "regime label decreased as VIX increased"

    def test_calm_greater_than_stress_raises(self) -> None:
        """Adversarial config: calm > stress is an invalid definition → ValueError."""
        cfg = dict(load_config("regimes"))
        cfg["vix_thresholds"] = {"calm": 30, "stress": 10}
        with pytest.raises(ValueError, match="calm threshold"):
            label_regimes(_vix_panel(np.array([5.0, 20.0])), cfg)

    def test_unknown_definition_raises_valueerror(self) -> None:
        cfg = dict(load_config("regimes"))
        cfg["definition"] = "kmeans_made_up"
        with pytest.raises(ValueError, match="unknown regime definition"):
            label_regimes(_vix_panel(np.array([5.0, 20.0])), cfg)

    @given(st.lists(st.integers(min_value=0, max_value=2), max_size=200))
    def test_independent_count_equals_groupby_blocks(self, labels_list: list[int]) -> None:
        """Property: independent_regime_count == number of itertools.groupby runs."""
        import itertools

        labels = np.asarray(labels_list, dtype=np.int64)
        expected = sum(1 for _ in itertools.groupby(labels_list))
        assert independent_regime_count(labels) == expected
        if labels.size == 0:
            assert expected == 0


# ============================================================================ #
# monitoring + logging — setup idempotent + isolated (no level/handler leak)   #
# ============================================================================ #
class TestLoggingIsolation:
    def test_attach_run_logging_idempotent_per_dir(self, tmp_path: Path) -> None:
        """Re-attaching the SAME run dir must not add duplicate file handlers (idempotent per dir)."""
        root = logging.getLogger()
        saved_handlers = root.handlers[:]
        saved_configured = logmod._configured
        saved_attached = set(logmod._run_attached)
        try:
            run_dir = tmp_path / "run"
            p1 = logmod.attach_run_logging(run_dir)
            n_after_first = len(root.handlers)
            p2 = logmod.attach_run_logging(run_dir)  # second attach: no new handlers
            assert p1 == p2
            assert len(root.handlers) == n_after_first
            assert p1["log"].name == "run.log"
            assert p1["events"].name == "events.jsonl"
        finally:
            root.handlers[:] = saved_handlers
            logmod._configured = saved_configured
            logmod._run_attached.clear()
            logmod._run_attached.update(saved_attached)

    def test_log_event_writes_jsonl_with_fields(self, tmp_path: Path) -> None:
        """log_event emits a parseable JSONL record carrying the structured fields to events.jsonl."""
        root = logging.getLogger()
        saved_handlers = root.handlers[:]
        saved_configured = logmod._configured
        saved_attached = set(logmod._run_attached)
        saved_level = root.level
        try:
            run_dir = tmp_path / "evrun"
            paths = logmod.attach_run_logging(run_dir, level=logging.INFO)
            log = logmod.get_logger("deeptest")
            logmod.log_event(log, "candidate_done", arm="distributional", cand=3, fitness=0.0125)
            for h in root.handlers:
                h.flush()
            lines = paths["events"].read_text(encoding="utf-8").strip().splitlines()
            recs = [json.loads(ln) for ln in lines]
            mine = [r for r in recs if r.get("event") == "candidate_done"]
            assert mine, "no candidate_done event written"
            r = mine[-1]
            assert r["arm"] == "distributional"
            assert r["cand"] == 3
            assert r["fitness"] == 0.0125
            assert r["level"] == "INFO"
        finally:
            root.handlers[:] = saved_handlers
            logmod._configured = saved_configured
            root.setLevel(saved_level)
            logmod._run_attached.clear()
            logmod._run_attached.update(saved_attached)

    def test_configure_logging_does_not_leak_level_across_calls(self) -> None:
        """A re-configure changes the level but never duplicates handlers (idempotency + isolation)."""
        root = logging.getLogger()
        saved_handlers = root.handlers[:]
        saved_configured = logmod._configured
        saved_level = root.level
        try:
            root.handlers.clear()
            logmod._configured = False
            logmod.configure_logging(logging.WARNING)
            assert root.level == logging.WARNING
            assert len(root.handlers) == 1
            logmod.configure_logging(logging.DEBUG)  # idempotent: only adjusts level
            assert root.level == logging.DEBUG
            assert len(root.handlers) == 1
        finally:
            root.handlers[:] = saved_handlers
            logmod._configured = saved_configured
            root.setLevel(saved_level)


class TestMonitorStateFile:
    def _isolate(self):  # noqa: ANN202
        """Snapshot/restore root logging so a RunMonitor's attach_run_logging doesn't leak."""
        root = logging.getLogger()
        return (root, root.handlers[:], logmod._configured, set(logmod._run_attached))

    def test_progress_json_is_atomic_and_valid(self, tmp_path: Path) -> None:
        """RunMonitor writes a parseable progress.json snapshot with the documented top-level keys."""
        root, sh, sc, sa = self._isolate()
        try:
            mon = RunMonitor(
                tmp_path / "mrun", title="deep", total_arms=2, candidates_per_arm=3,
                train_steps=100, model="stub",
            )
            assert mon.state_path.name == "progress.json"
            assert mon.state_path.is_file()
            snap = json.loads(mon.state_path.read_text(encoding="utf-8"))
            assert snap["title"] == "deep"
            assert snap["arms"]["total"] == 2
            assert snap["candidates"]["run_total"] == 2 * 3
            assert snap["phase"] == "starting"
            mon.close(status="done")
            final = json.loads(mon.state_path.read_text(encoding="utf-8"))
            assert final["phase"] == "done"
        finally:
            root.handlers[:] = sh
            logmod._configured = sc
            logmod._run_attached.clear()
            logmod._run_attached.update(sa)

    def test_lifecycle_updates_counters_and_best(self, tmp_path: Path) -> None:
        """arm_start/candidate_done track run_done and per-arm best fitness in the snapshot."""
        root, sh, sc, sa = self._isolate()
        try:
            mon = RunMonitor(
                tmp_path / "lc", title="lc", total_arms=1, candidates_per_arm=2,
                train_steps=10, model="stub",
            )
            mon.arm_start("scalar", 0)
            mon.candidate_done("scalar", 0, fitness=0.5, secs=1.0)
            mon.candidate_done("scalar", 1, fitness=0.9, secs=1.0)  # higher → new best
            snap = json.loads(mon.state_path.read_text(encoding="utf-8"))
            assert snap["candidates"]["run_done"] == 2
            assert snap["best_fitness"]["scalar"] == 0.9
            mon.close()
        finally:
            root.handlers[:] = sh
            logmod._configured = sc
            logmod._run_attached.clear()
            logmod._run_attached.update(sa)

    def test_nan_fitness_flags_an_anomaly(self, tmp_path: Path) -> None:
        """A NaN candidate fitness is caught as a 'fitness_nan' anomaly (early-warning, not silent)."""
        root, sh, sc, sa = self._isolate()
        try:
            mon = RunMonitor(
                tmp_path / "nan", title="n", total_arms=1, candidates_per_arm=1,
                train_steps=10, model="stub",
            )
            mon.candidate_done("scalar", 0, fitness=float("nan"), secs=0.1)
            assert any(a["kind"] == "fitness_nan" for a in mon._anomalies)
            mon.close()
        finally:
            root.handlers[:] = sh
            logmod._configured = sc
            logmod._run_attached.clear()
            logmod._run_attached.update(sa)
