"""Coverage tests for the directly-callable, in-process paths of src/sandbox/executor.py that the spawn-based
test_sandbox suite does not line-cover: the `_validate_inline` fallback (each contract-rejection branch),
`_safe_import`'s non-numpy rejection, and the `ast_gate` Name-dunder / format-field edges + `extract_reward_source`
prose-salvage. (The POSIX-rlimit / spawned-child / GPU paths are `# pragma: no cover` with rationale.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.reward.contract import SAFE_DEFAULT  # noqa: E402,F401
from src.sandbox import executor as ex  # noqa: E402

# Reward-contract fixture: (weights, returns, prev_weights, port_ret, info) — anonymised arrays only.
_FIX = (np.full(3, 1.0 / 3.0), np.zeros(2), np.full(3, 1.0 / 3.0), 0.0, {})


# --------------------------------------------------------------------------- #
# _safe_import: non-numpy, not-already-loaded import is rejected                #
# --------------------------------------------------------------------------- #
def test_safe_import_rejects_non_numpy() -> None:
    # A name that is neither numpy-rooted NOR already in sys.modules must be refused (the defence-in-depth
    # restricted __import__). `os`/`socket` would pass (already loaded), so use a guaranteed-absent name.
    assert "zzz_blocked_import_xyz" not in sys.modules
    with pytest.raises(ImportError, match="not permitted"):
        ex._safe_import("zzz_blocked_import_xyz")


def test_safe_import_allows_numpy() -> None:
    assert ex._safe_import("numpy") is not None  # numpy-rooted -> delegated to the real importer


# --------------------------------------------------------------------------- #
# _validate_inline: success + every contract-rejection branch                   #
# --------------------------------------------------------------------------- #
def test_validate_inline_accepts_a_valid_reward() -> None:
    fn = ex._validate_inline("def reward(w, r, pw, pr, info):\n    return float(pr), {}, None\n", _FIX)
    assert callable(fn)


@pytest.mark.parametrize(
    "src, match",
    [
        ("def reward(w, r, pw, pr, info):\n    return (1.0,\n", "compile/exec"),  # SyntaxError
        ("x = 1\n", "no callable"),                                                # no `reward`
        ("def reward(*a):\n    raise ValueError('boom')\n", "crashed"),            # runtime crash
        ("def reward(*a):\n    return 1.0\n", "unpackable"),                       # not a 3-element unpackable
        ("def reward(*a):\n    return float('nan'), {}, None\n", "non-finite"),    # non-finite total
        ("def reward(*a):\n    return 1.0, 7, None\n", "components is not a dict"),# components not dict
    ],
)
def test_validate_inline_rejects_contract_violations(src: str, match: str) -> None:
    with pytest.raises(ex.SandboxError, match=match):
        ex._validate_inline(src, _FIX)


# --------------------------------------------------------------------------- #
# ast_gate edge branches: bare dunder Name + format-string field access          #
# --------------------------------------------------------------------------- #
def test_ast_gate_rejects_bare_dunder_name() -> None:
    # A bare reference to a dunder NAME (not attribute) — e.g. __builtins__ — must be rejected.
    assert ex.ast_gate("def reward(*a):\n    x = __builtins__\n    return 0.0, {}, None\n") is False


def test_ast_gate_rejects_format_field_dunder_walk() -> None:
    # A string LITERAL containing a replacement field with attribute access (the str.format dunder-walk
    # the AST attribute walk cannot see) is caught by the defence-in-depth regex.
    src = "def reward(*a):\n    s = '{0.__class__.__mro__[1]}'\n    return 0.0, {}, None\n"
    assert ex.ast_gate(src) is False


def test_ast_gate_accepts_clean_numeric_reward() -> None:
    assert ex.ast_gate("def reward(w, r, pw, pr, info):\n    return float(np.mean(r)), {}, None\n") is True


# --------------------------------------------------------------------------- #
# extract_reward_source: prose-preamble salvage (the non-fenced slice path)      #
# --------------------------------------------------------------------------- #
def test_extract_reward_source_salvages_prose_preamble() -> None:
    raw = "Here is my reward function:\ndef reward(w, r, pw, pr, info):\n    return float(pr), {}, None\n"
    out = ex.extract_reward_source(raw)
    assert ex._parses_as_python(out) and "def reward" in out


def test_extract_reward_source_noop_on_clean_code() -> None:
    clean = "def reward(w, r, pw, pr, info):\n    return float(pr), {}, None\n"
    assert ex.extract_reward_source(clean) == clean  # already valid -> returned byte-identical
