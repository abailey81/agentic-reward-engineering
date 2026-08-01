"""Tests for src/sandbox/executor.py — two-stage untrusted execution (F.5, A-5)."""

from __future__ import annotations

import ast
import importlib.util
import multiprocessing as mp

import numpy as np
import pytest

from src.reward.contract import SAFE_DEFAULT
from src.sandbox import executor
from src.sandbox.executor import (
    _ALLOWED_ATTRS,
    SandboxError,
    ast_gate,
    candidate_failed,
    extract_reward_source,
    reset_failure_flag,
    safe_call,
    safe_call_count,
    safe_default_count,
    validate_once,
)

# A minimal, contract-conforming numpy-only reward used across tests.
VALID_SRC = """
def reward(weights, returns, prev_weights, port_ret, info):
    total = float(np.sum(weights * returns))
    return total, {"raw": total}, None
"""


def _fixture(rng: np.random.Generator) -> tuple:
    """A contract fixture of anonymized numpy arrays only."""
    n = 5
    weights = np.full(n, 1.0 / n)
    returns = rng.standard_normal(n) * 0.01
    prev_weights = np.full(n, 1.0 / n)
    port_ret = float(np.sum(weights * returns))
    info: dict = {}
    return (weights, returns, prev_weights, port_ret, info)


def test_os_import_rejected_at_gate() -> None:
    """ast_gate rejects `import os` (imports outside the numpy allowlist)."""
    assert ast_gate("import os\n") is False
    assert ast_gate("import numpy as np\n") is True
    assert ast_gate("from os import path\n") is False


def test_from_numpy_import_rce_rejected_at_gate() -> None:
    """RCE fix (2026-06-25): `from numpy import load` passed a root-module-only ImportFrom check
    (numpy allowlisted), then `load(...)` was a BARE ast.Name the _BANNED_ATTRS attribute-allowlist
    never inspected -> np.load pickle-RCE. ALL `from ... import` (incl. wildcard) are now rejected;
    `import numpy as np` stays the only legitimate import."""
    assert ast_gate("from numpy import load\n") is False
    assert ast_gate("from numpy import load as l\n") is False
    assert ast_gate("from numpy import *\n") is False
    assert ast_gate("from numpy.lib import npyio\n") is False
    assert ast_gate("import numpy as np\n") is True
    # End-to-end: validate_once must also refuse the bypass.
    with pytest.raises(SandboxError):
        validate_once(
            "def reward(weights, returns, prev_weights, port_ret, info):\n"
            "    from numpy import load\n"
            "    return 0.0, {}, None\n",
            _fixture(np.random.default_rng(0)),
        )


def test_file_read_rejected_at_gate() -> None:
    """ast_gate rejects open()/eval()/exec() attempts in the reward source."""
    assert ast_gate("def reward(*a):\n    open('/etc/passwd')\n") is False
    assert ast_gate("def reward(*a):\n    eval('1+1')\n") is False
    assert ast_gate("def reward(*a):\n    exec('x=1')\n") is False
    # Dunder attribute escape (().__class__...) must also be rejected.
    assert ast_gate("def reward(*a):\n    return ().__class__\n") is False
    assert ast_gate("def reward(*a):\n    return __import__('os')\n") is False


def test_str_format_dunder_escape_rejected() -> None:
    """str.format can walk dunders INSIDE a string literal (invisible to the Attribute AST walk):
    '{0.__class__.__mro__[1].__subclasses__}'.format(x) -> object.__subclasses__ (RCE). The gate must
    reject it via (a) 'format' no longer allowlisted and (b) the format-field-access literal scan."""
    assert ast_gate(
        "def reward(w,r,pw,pr,info):\n"
        "    s = '{0.__class__.__mro__[1].__subclasses__}'.format(w)\n    return 0.0,{},None"
    ) is False
    assert ast_gate("def reward(*a):\n    x = '{}'.format(3)\n    return 0.0,{},None") is False  # format gone
    assert ast_gate("def reward(*a):\n    s = '{0[0]}'.format([1])\n    return 0.0,{},None") is False
    # a legit numeric reward (no string formatting) still passes
    assert ast_gate(
        "import numpy as np\ndef reward(w,r,pw,pr,info):\n    return float(pr-0.5*float(np.var(r))),{},None"
    ) is True


def test_docstring_is_exempt_from_the_format_field_scan() -> None:
    """A DOCSTRING documenting the components dict must not cost the candidate (2026-07-26 review).

    The format-field literal scan is defence-in-depth against a format TEMPLATE smuggling attribute
    access inside a string. A docstring can never be that template — reaching one needs ``__doc__``,
    a dunder the gate already rejects — so scanning docstrings was pure false-positive cost. It was an
    EXPENSIVE false positive: ``prompts/initial_generation.txt`` shows the author the literal example
    ``{"port_ret": float(port_ret)}``, so a reward whose docstring documents its own components dict
    matched ``[.\\[]`` inside braces and the whole PAID candidate was discarded for its prose.
    """
    # The exact shape the initial-generation prompt primes the model to produce.
    assert ast_gate(
        'def reward(weights, returns, prev_weights, port_ret, info):\n'
        '    """Return (total, {"cvar_05": -0.03, "ret": 0.1}, state)."""\n'
        '    return float(port_ret), {"ret": float(port_ret)}, None\n'
    ) is True
    # Indexing and decimals inside docstring braces are likewise harmless prose.
    assert ast_gate(
        'def reward(w, r, pw, pr, info):\n    """Uses {r[0]} and the grid {0.1, 0.2}."""\n'
        '    return 0.0, {}, None\n'
    ) is True
    # Module-level and class docstrings get the same exemption.
    assert ast_gate('"""Module note {a.b}."""\ndef reward(*a):\n    return 0.0, {}, None\n') is True


def test_format_field_scan_still_blocks_non_docstring_strings() -> None:
    """The docstring exemption must NOT widen the hole: every real template is still rejected."""
    # A bare string statement is NOT a docstring (it is not the first statement of the body).
    assert ast_gate(
        'def reward(w, r, pw, pr, info):\n    x = 1\n    """{0.__class__}"""\n    return 0.0, {}, None\n'
    ) is False
    # An assigned template, the classic escape, and the format() builtin all still reject.
    assert ast_gate(
        'def reward(w, r, pw, pr, info):\n    t = "{0.__class__.__mro__[1]}"\n    return 0.0, {}, None\n'
    ) is False
    assert ast_gate(
        'def reward(w, r, pw, pr, info):\n    return "{0.__class__}".format(w), {}, None\n'
    ) is False
    assert ast_gate(
        'def reward(w, r, pw, pr, info):\n    return format(w, "{0.__class__}"), {}, None\n'
    ) is False
    # A docstring exemption must not leak to a SECOND string literal in the same function.
    assert ast_gate(
        'def reward(w, r, pw, pr, info):\n    """Fine {a.b} prose."""\n'
        '    t = "{0.__class__}"\n    return 0.0, {}, None\n'
    ) is False


def test_date_reference_rejected() -> None:
    """Source importing datetime is rejected (anonymized arrays only, no dates)."""
    assert ast_gate("import datetime\n") is False
    assert ast_gate("from datetime import date\n") is False


def test_infinite_loop_killed_at_validate_once(rng: np.random.Generator) -> None:
    """validate_once kills a non-terminating reward at its timeout (stage 1)."""
    src = """
def reward(weights, returns, prev_weights, port_ret, info):
    while True:
        pass
    return 0.0, {}, None
"""
    with pytest.raises(SandboxError):
        validate_once(src, _fixture(rng), timeout_s=1.0)


def test_valid_reward_passes_gate_validates_and_runs(rng: np.random.Generator) -> None:
    """A valid numpy reward passes the gate, validates, and safe_call returns its value."""
    assert ast_gate(VALID_SRC) is True
    fixture = _fixture(rng)
    fn = validate_once(VALID_SRC, fixture, timeout_s=2.0)
    reset_failure_flag()
    total, components, state = safe_call(fn, *fixture)
    expected = float(np.sum(fixture[0] * fixture[1]))
    assert np.isclose(total, expected)
    assert components == {"raw": expected}
    assert state is None
    assert candidate_failed() is False


def test_in_process_error_caught_flagged_and_loop_continues(
    rng: np.random.Generator,
) -> None:
    """safe_call catches an in-process error, substitutes SAFE_DEFAULT, flags, continues."""

    def boom(weights, returns, prev_weights, port_ret, info):  # noqa: ANN001
        raise RuntimeError("kaboom")

    reset_failure_flag()
    fixture = _fixture(rng)
    total, components, state = safe_call(boom, *fixture)
    assert total == SAFE_DEFAULT
    assert components == {}
    assert state is None
    assert candidate_failed() is True

    # A NaN-returning reward is also caught.
    def nanny(weights, returns, prev_weights, port_ret, info):  # noqa: ANN001
        return float("nan"), {}, None

    reset_failure_flag()
    total, _, _ = safe_call(nanny, *fixture)
    assert total == SAFE_DEFAULT
    assert candidate_failed() is True

    # The loop continues: a subsequent good reward succeeds and clears the flag.
    def good(weights, returns, prev_weights, port_ret, info):  # noqa: ANN001
        return float(port_ret), {"r": float(port_ret)}, None

    total, _, _ = safe_call(good, *fixture)
    assert np.isclose(total, fixture[3])
    assert candidate_failed() is False


def test_safe_default_count_accumulates_over_window(rng: np.random.Generator) -> None:
    """R66: ``safe_default_count`` accumulates EVERY substitution in a window (not last-call only), so an
    intermittent reward that recovers on the final step is still quantified — while ``candidate_failed``
    keeps its documented last-call semantics (a later success clears the boolean)."""
    fixture = _fixture(rng)

    def boom(weights, returns, prev_weights, port_ret, info):  # noqa: ANN001
        raise RuntimeError("boom")

    def good(weights, returns, prev_weights, port_ret, info):  # noqa: ANN001
        return float(port_ret), {}, None

    reset_failure_flag()
    assert safe_default_count() == 0 and safe_call_count() == 0
    safe_call(boom, *fixture)  # fail -> count 1
    safe_call(boom, *fixture)  # fail -> count 2
    safe_call(good, *fixture)  # succeed on the LAST call -> count stays 2
    assert safe_call_count() == 3
    assert safe_default_count() == 2  # accumulates across the whole window
    assert candidate_failed() is False  # last-call contract preserved (a later success clears the bool)
    reset_failure_flag()
    assert safe_default_count() == 0 and safe_call_count() == 0


def test_reward_never_receives_tickers_or_dates(rng: np.random.Generator) -> None:
    """A reward is only ever passed anonymized arrays — no tickers, no dates."""
    captured: dict = {}

    def spy(weights, returns, prev_weights, port_ret, info):  # noqa: ANN001
        captured["weights"] = weights
        captured["returns"] = returns
        captured["prev_weights"] = prev_weights
        captured["port_ret"] = port_ret
        return float(port_ret), {}, None

    fixture = _fixture(rng)
    safe_call(spy, *fixture)
    for key in ("weights", "returns", "prev_weights"):
        assert isinstance(captured[key], np.ndarray)
        assert np.issubdtype(captured[key].dtype, np.number)
    assert isinstance(captured["port_ret"], float)


def test_failure_flag_module_level() -> None:
    """The failure flag is module-level so an orchestrator can mark a candidate."""
    reset_failure_flag()
    assert executor.candidate_failed() is False


# Regression: ndarray METHODS (.mean()/.var()) and other numpy ops trigger numpy's
# lazy submodule imports, which resolve __import__ from the sandbox namespace. With a
# restricted-builtins namespace lacking __import__ these crashed with KeyError, both at
# validation and SILENTLY during training via safe_call (every such candidate flagged
# "failed"). The fix is the controlled _safe_import in SAFE_BUILTINS.
@pytest.mark.parametrize(
    "body",
    [
        "    return float(port_ret - 0.5 * returns.var()), {'m': float(returns.mean())}, None",
        "    q = np.quantile(returns, 0.05); return float(port_ret + q), {'q': float(q)}, None",
        "    s = np.sort(returns); return float(port_ret + float(s[0])), {}, None",
        "    xs = [float(x) for x in returns]; return float(port_ret * len(xs)), {'n': float(len(xs))}, None",
    ],
)
def test_numpy_method_rewards_validate_and_run_without_failing(
    body: str, rng: np.random.Generator
) -> None:
    """Legitimate numpy reward code (methods, np funcs, comprehensions) must validate AND
    run via safe_call WITHOUT being flagged failed — guards the __import__ sandbox fix."""
    src = "def reward(weights, returns, prev_weights, port_ret, info):\n" + body
    assert ast_gate(src) is True
    fixture = _fixture(rng)
    fn = validate_once(src, fixture, timeout_s=2.0)
    reset_failure_flag()
    total, _components, _state = safe_call(fn, *fixture)
    assert np.isfinite(total)
    assert candidate_failed() is False  # the silent-failure bug would set this True


def test_safe_import_denies_fresh_non_numpy_module() -> None:
    """Defence-in-depth: the sandbox __import__ denies a brand-new non-numpy import."""
    with pytest.raises(ImportError):
        executor._safe_import("a_module_that_is_definitely_not_loaded_zzz")


# --- ADR-008: numpy IO/FFI attribute denylist at the gate ---------------------------

# Each source is well-formed numpy that the OLD gate (dunder-only) would have PASSED but
# whose attribute is a filesystem / pickle-RCE / FFI vector. The hardened gate must
# REJECT every one of them (np.load with allow_pickle=True is arbitrary code execution).
@pytest.mark.parametrize(
    "src",
    [
        "def reward(*a):\n    return float(np.load('x.npy', allow_pickle=True).sum()), {}, None\n",
        "def reward(*a):\n    np.save('out.npy', a[0]); return 0.0, {}, None\n",
        "def reward(*a):\n    return float(np.fromfile('x.bin').sum()), {}, None\n",
        "def reward(*a):\n    return float(np.genfromtxt('x.csv').sum()), {}, None\n",
        "def reward(*a):\n    return float(np.loadtxt('x.csv').sum()), {}, None\n",
        "def reward(*a):\n    m = np.memmap('x.bin', dtype='f8'); return float(m[0]), {}, None\n",
        "def reward(*a):\n    return float(np.DataSource().open('x').read()), {}, None\n",
        "def reward(*a):\n    a[0].tofile('leak.bin'); return 0.0, {}, None\n",
        "def reward(*a):\n    return np.lib.format.read_magic, {}, None\n",
        "def reward(*a):\n    return np.ctypeslib.as_array, {}, None\n",
        # final-acceptance-audit 2026-06-19 P1 READ-escapes (genfromtxt aliases + file-first read):
        "def reward(*a):\n    return float(np.recfromtxt('x.csv').sum()), {}, None\n",
        "def reward(*a):\n    return float(np.recfromcsv('x.csv').sum()), {}, None\n",
        "def reward(*a):\n    return float(np.fromregex('x', '.', float).sum()), {}, None\n",
        # final-acceptance-audit 2026-06-19 P1 WRITE-escapes (ndarray.dump/dumps pickle to a path):
        "def reward(*a):\n    a[0].dump('leak.pkl'); return 0.0, {}, None\n",
        "def reward(*a):\n    return float(len(a[0].dumps())), {}, None\n",
    ],
)
def test_gate_rejects_numpy_io_and_ffi_attributes(src: str) -> None:
    """ast_gate REJECTS numpy's file/FFI/loader surface (ADR-008 attribute denylist)."""
    assert ast_gate(src) is False


def test_gate_rejects_mro_object_model_escape() -> None:
    """The .mro object-model escape route is rejected as a banned attribute (ADR-008)."""
    assert ast_gate("def reward(*a):\n    return type(a).mro(), {}, None\n") is False


def test_gate_rejects_np_seterr_global_state_leak() -> None:
    """np.seterr sets numpy's PROCESS-GLOBAL float-error mode and does not restore it, so an
    untrusted reward calling it would leak that mode into every LATER candidate -> order-dependent,
    non-deterministic campaign (V15b). It must be rejected at the gate (``seterr`` de-allowlisted)."""
    assert ast_gate("def reward(*a):\n    np.seterr(all='ignore'); return 0.0, {}, None\n") is False
    assert ast_gate("def reward(*a):\n    np.seterr(divide='raise'); return 0.0, {}, None\n") is False
    # End-to-end: validate_once must also refuse a seterr reward (gate runs first).
    with pytest.raises(SandboxError):
        validate_once(
            "def reward(weights, returns, prev_weights, port_ret, info):\n"
            "    np.seterr(all='ignore')\n"
            "    return float(port_ret), {}, None\n",
            _fixture(np.random.default_rng(0)),
        )
    # The leak-FREE alternatives stay allowed: errstate (a context manager that RESTORES the prior
    # mode on exit) and geterr (read-only) are not global-state leaks, so a reward may still use them.
    assert ast_gate(
        "def reward(*a):\n"
        "    with np.errstate(divide='ignore'):\n"
        "        x = np.geterr()\n"
        "    return 0.0, {}, None\n"
    ) is True


def test_legitimate_reward_math_still_passes_gate(rng: np.random.Generator) -> None:
    """POSITIVE control: real reward math (reductions, indexing, np.abs/where/clip/dot,
    arithmetic) STILL PASSES — the IO/FFI denylist must not over-block (ADR-008)."""
    src = (
        "def reward(weights, returns, prev_weights, port_ret, info):\n"
        "    pnl = np.sum(weights * returns)\n"
        "    mu = np.mean(returns)\n"
        "    sd = np.std(returns)\n"
        "    v = np.var(returns)\n"
        "    dd = np.dot(weights, returns)\n"
        "    turnover = np.sum(np.abs(weights - prev_weights))\n"
        "    downside = np.where(returns < 0.0, returns, 0.0)\n"
        "    worst = returns[0] - np.clip(sd, 0.0, 1.0)\n"
        "    total = float(pnl + mu - 0.5 * v + dd - 0.01 * turnover"
        " + float(np.sum(downside)) + float(worst))\n"
        "    return total, {'pnl': float(pnl), 'sd': float(sd)}, None\n"
    )
    # Passes the static gate ...
    assert ast_gate(src) is True
    # ... and actually validates + runs without being flagged failed.
    fixture = _fixture(rng)
    fn = validate_once(src, fixture, timeout_s=2.0)
    reset_failure_flag()
    total, components, _state = safe_call(fn, *fixture)
    assert np.isfinite(total)
    assert set(components) == {"pnl", "sd"}
    assert candidate_failed() is False


# The address-space (RLIMIT_AS) cap is POSIX-only; on Windows `resource` is absent and
# the wall-clock timeout is the backstop, so skip there (ADR-008 documented gap).
_HAS_RESOURCE = importlib.util.find_spec("resource") is not None


@pytest.mark.skipif(not _HAS_RESOURCE, reason="POSIX 'resource' module unavailable (Windows)")
def test_memory_bomb_rejected_at_validate_once(rng: np.random.Generator) -> None:
    """A reward that allocates far past RLIMIT_AS is killed/rejected at validate_once.

    The child caps address space at ~2 GiB (clamped to the hard limit); a multi-GiB
    allocation must raise MemoryError in the child -> reported as a SandboxError, not an
    OOM of the host. The source itself is gate-clean (only arithmetic + np.ones)."""
    src = (
        "def reward(weights, returns, prev_weights, port_ret, info):\n"
        "    big = np.ones((8, 1024, 1024, 1024), dtype=np.float64)\n"  # ~64 GiB
        "    return float(big.sum()), {}, None\n"
    )
    assert ast_gate(src) is True
    with pytest.raises(SandboxError):
        validate_once(src, _fixture(rng), timeout_s=10.0)


# =====================================================================================
# ADDED (this task): strict security/determinism net for the AST gate + sandbox.
# Each test cites the threat it locks; every test EXTENDS coverage above (no duplicates).
# =====================================================================================


def test_v15b_seterr_absent_from_allowlist_determinism_rationale() -> None:
    """V15b REGRESSION-LOCK (determinism). The reason ``np.seterr`` is gated is that it mutates
    numpy's PROCESS-GLOBAL float-error mode without restoring it, leaking that mode into every
    LATER candidate and making the campaign order-dependent / non-deterministic. The mechanism
    that enforces this is ``seterr`` being absent from ``_ALLOWED_ATTRS`` (the allowlist) while the
    leak-free alternatives stay present. Assert that invariant directly so a future allowlist edit
    that re-adds ``seterr`` fails LOUDLY here, independent of any payload-level gate test."""
    assert "seterr" not in _ALLOWED_ATTRS  # de-allowlisted on purpose (determinism)
    # The leak-free alternatives MUST remain allowlisted (errstate restores on exit; geterr is read-only).
    assert "errstate" in _ALLOWED_ATTRS
    assert "geterr" in _ALLOWED_ATTRS


def test_v15b_errstate_geterr_reward_validates_and_runs(rng: np.random.Generator) -> None:
    """V15b (positive half). A reward using the leak-FREE float-error API — ``with np.errstate(...)``
    (restores the prior mode on exit) and ``np.geterr()`` (read-only) — must not only pass the gate
    but actually VALIDATE and run finite via safe_call, proving the determinism carve-out is usable."""
    src = (
        "def reward(weights, returns, prev_weights, port_ret, info):\n"
        "    with np.errstate(divide='ignore', invalid='ignore'):\n"
        "        x = np.log1p(np.abs(returns))\n"
        "        _mode = np.geterr()\n"
        "    return float(port_ret + float(np.sum(x))), {'mode_keys': float(len(_mode))}, None\n"
    )
    assert ast_gate(src) is True
    fixture = _fixture(rng)
    fn = validate_once(src, fixture, timeout_s=2.0)
    reset_failure_flag()
    total, _components, _state = safe_call(fn, *fixture)
    assert np.isfinite(total)
    assert candidate_failed() is False


# RCE / sandbox-ESCAPE payloads. Each is a known object-model / FFI / IO / dynamic-eval escape that
# MUST be stopped — either statically (ast_gate -> False) or end-to-end (validate_once -> SandboxError).
# The threat: the gate is an allowlist net; any payload reaching os/builtins/pickle/FFI is an RCE.
# Some overlap with focused tests above is intentional — this is the consolidated belt-and-braces sweep.
_RCE_ESCAPE_PAYLOADS: list[str] = [
    # --- import-surface escapes ---
    "def reward(*a):\n    from numpy import *\n    return 0.0, {}, None\n",
    "def reward(*a):\n    from numpy import load\n    return 0.0, {}, None\n",
    "def reward(*a):\n    return __import__('os').getcwd(), {}, None\n",
    # --- object-model / dunder-walk escapes ---
    "def reward(*a):\n    return ().__class__.__mro__, {}, None\n",
    "def reward(*a):\n    return type(a).mro(), {}, None\n",
    "def reward(*a):\n    return '{0.__class__}'.format(a), {}, None\n",
    # --- numpy submodule traversal to os/builtins (the _pytesttester chain) ---
    "def reward(*a):\n    return np._pytesttester, {}, None\n",
    # --- dynamic-attribute fetch (getattr is a forbidden call) ---
    "def reward(*a):\n    return getattr(np, 'load'), {}, None\n",
    # --- numpy IO / pickle-RCE / FFI surface ---
    "def reward(*a):\n    return np.load('x.npy'), {}, None\n",
    "def reward(*a):\n    np.save('x.npy', a[0]); return 0.0, {}, None\n",
    "def reward(*a):\n    return np.fromfile('x.bin'), {}, None\n",
    "def reward(*a):\n    m = np.memmap('x.bin', dtype='f8'); return float(m[0]), {}, None\n",
    "def reward(*a):\n    a[0].dump('leak.pkl'); return 0.0, {}, None\n",
    "def reward(*a):\n    return np.ctypeslib.as_array, {}, None\n",
    "def reward(*a):\n    return a[0].ctypes, {}, None\n",
    "def reward(*a):\n    return a[0].data, {}, None\n",
    # --- dynamic-eval / IO builtins ---
    "def reward(*a):\n    return eval('1+1'), {}, None\n",
    "def reward(*a):\n    exec('x=1'); return 0.0, {}, None\n",
    "def reward(*a):\n    return compile('1', '<s>', 'eval'), {}, None\n",
    "def reward(*a):\n    return open('/etc/passwd').read(), {}, None\n",
]


@pytest.mark.parametrize("payload", _RCE_ESCAPE_PAYLOADS)
def test_rce_and_escape_payloads_all_blocked(payload: str, rng: np.random.Generator) -> None:
    """EVERY RCE / sandbox-escape payload is blocked — gated statically (ast_gate False) OR refused
    end-to-end (validate_once raises SandboxError). A payload that passes BOTH would be a live escape
    to os/builtins/pickle/FFI: a security finding, so the assertion is hard and the test is permanent."""
    gated = ast_gate(payload) is False
    if gated:
        return  # blocked at the static gate — the cheapest, preferred stop
    # If somehow gate-clean, the runtime stage MUST still reject it (else it is a real escape).
    with pytest.raises(SandboxError):
        validate_once(payload, _fixture(rng), timeout_s=5.0)


def test_benign_pure_numpy_reward_validates_and_runs_finite(rng: np.random.Generator) -> None:
    """POSITIVE control: a pure-numpy reward using mean/std/clip/percentile/where/log1p/errstate must
    validate and safe_call must return a finite (total, dict, state). Guards against the allowlist
    over-blocking real risk-sensitive reward arithmetic (the gate is a net, not a wall)."""
    src = (
        "def reward(weights, returns, prev_weights, port_ret, info):\n"
        "    mu = np.mean(returns)\n"
        "    sd = np.std(returns)\n"
        "    var5 = np.percentile(returns, 5.0)\n"
        "    downside = np.where(returns < 0.0, returns, 0.0)\n"
        "    with np.errstate(divide='ignore', invalid='ignore'):\n"
        "        damp = np.log1p(np.abs(np.clip(returns, -1.0, 1.0)))\n"
        "    total = float(port_ret + float(mu) - 0.5 * float(sd) + float(var5)"
        " + float(np.sum(downside)) + float(np.sum(damp)))\n"
        "    return total, {'mu': float(mu), 'var5': float(var5)}, None\n"
    )
    assert ast_gate(src) is True
    fixture = _fixture(rng)
    fn = validate_once(src, fixture, timeout_s=2.0)
    reset_failure_flag()
    total, components, state = safe_call(fn, *fixture)
    assert isinstance(total, float) and np.isfinite(total)
    assert isinstance(components, dict) and set(components) == {"mu", "var5"}
    assert state is None
    assert candidate_failed() is False


# --- FENCE / PROSE SALVAGE: extract_reward_source recovers runnable code from raw LLM output -------

_CLEAN_REWARD = (
    "import numpy as np\n"
    "def reward(weights, returns, prev_weights, port_ret, info):\n"
    "    return float(port_ret), {}, None"
)


def test_extract_reward_source_strips_python_fence() -> None:
    """FENCE SALVAGE: an LLM wraps the function in a ```python ... ``` fence despite the
    'return only the function' instruction; ast.parse(whole) would SyntaxError and starve the arm.
    extract_reward_source must recover the runnable function from the fence."""
    wrapped = "```python\n" + _CLEAN_REWARD + "\n```"
    out = extract_reward_source(wrapped)
    assert "def reward(" in out
    assert ast_gate(out) is True


def test_extract_reward_source_strips_prose_preamble() -> None:
    """PROSE SALVAGE: a prose preamble + trailing prose around the code (no fence). The shim slices
    from the first code-start line and trims to the longest parsing suffix."""
    completion = (
        "Sure! Here is a risk-sensitive reward function for you:\n\n"
        + _CLEAN_REWARD
        + "\n\nThis penalizes downside variance. Let me know if you want changes."
    )
    out = extract_reward_source(completion)
    assert "def reward(" in out
    assert ast_gate(out) is True


def test_extract_reward_source_multi_fence_picks_parsing_reward_block() -> None:
    """MULTI-FENCE SALVAGE: a 'first attempt / corrected version' completion has two fences; the first
    is syntactically broken. extract_reward_source must pick a block that PARSES and defines reward,
    not commit to the broken first block (audit fix: recoverable candidate must not be starved)."""
    broken = "```python\ndef reward(weights, returns,   # oops truncated\n```"
    good = "```python\n" + _CLEAN_REWARD + "\n```"
    completion = "First attempt:\n" + broken + "\n\nCorrected version:\n" + good
    out = extract_reward_source(completion)
    assert "def reward(" in out
    assert ast_gate(out) is True


def test_extract_reward_source_clean_code_is_byte_identical() -> None:
    """NO-OP path: already-clean code that parses is returned BYTE-IDENTICAL (exact bytes) so the
    archived source and its hash stay stable (re-audit invariant)."""
    assert extract_reward_source(_CLEAN_REWARD) == _CLEAN_REWARD
    # Even clean code that merely CONTAINS a ``` inside a string literal still parses -> unchanged.
    with_ticks = (
        "def reward(weights, returns, prev_weights, port_ret, info):\n"
        "    note = 'see ```docs```'\n"
        "    return float(port_ret), {}, None"
    )
    assert extract_reward_source(with_ticks) == with_ticks


# --- safe_call CONTAINMENT: the in-process training path never propagates a candidate failure ------


def test_safe_call_contains_wrong_arity_return(rng: np.random.Generator) -> None:
    """CONTAINMENT: a reward returning the WRONG ARITY (not a 3-tuple) is contained by safe_call ->
    SAFE_DEFAULT, flag flips True, no exception escapes into the training loop; reset clears it."""
    fixture = _fixture(rng)

    def two_tuple(weights, returns, prev_weights, port_ret, info):  # noqa: ANN001
        return float(port_ret), {}  # wrong arity (2-tuple): unpack to 3 raises in safe_call

    reset_failure_flag()
    total, components, state = safe_call(two_tuple, *fixture)
    assert total == SAFE_DEFAULT
    assert components == {} and state is None
    assert candidate_failed() is True

    # A non-tuple return is likewise contained (cannot unpack a scalar to 3).
    def scalar_only(weights, returns, prev_weights, port_ret, info):  # noqa: ANN001
        return 1.0

    reset_failure_flag()
    total, _, _ = safe_call(scalar_only, *fixture)
    assert total == SAFE_DEFAULT
    assert candidate_failed() is True

    # reset_failure_flag clears the flag for the next candidate.
    reset_failure_flag()
    assert candidate_failed() is False


def test_safe_call_never_raises_into_training_path(rng: np.random.Generator) -> None:
    """CONTAINMENT: whatever a malicious/buggy reward does (raise, +inf, -inf, wrong arity),
    safe_call MUST return a 3-tuple with a finite total and never let the exception propagate —
    the training rollout depends on this for forward progress."""
    fixture = _fixture(rng)

    def raiser(weights, returns, prev_weights, port_ret, info):  # noqa: ANN001
        raise ValueError("boom")

    def pos_inf(weights, returns, prev_weights, port_ret, info):  # noqa: ANN001
        return float("inf"), {}, None

    def neg_inf(weights, returns, prev_weights, port_ret, info):  # noqa: ANN001
        return float("-inf"), {}, None

    for fn in (raiser, pos_inf, neg_inf):
        reset_failure_flag()
        out = safe_call(fn, *fixture)  # must NOT raise
        assert isinstance(out, tuple) and len(out) == 3
        total, components, state = out
        assert total == SAFE_DEFAULT and np.isfinite(total)
        assert components == {} and state is None
        assert candidate_failed() is True


def test_safe_call_clamps_astronomical_finite_reward(rng: np.random.Generator) -> None:
    """row 30e: a finite-but-astronomical total (a decayed-denominator ratio can emit 1e15+)
    is treated exactly like non-finite — SAFE_DEFAULT + candidate-failure — because SAC has no
    other magnitude guard in the popart=False ablation. The 1e6 contract bound is inclusive:
    a large-but-legal total passes."""
    fixture = _fixture(rng)

    def huge(weights, returns, prev_weights, port_ret, info):  # noqa: ANN001
        return 1.0e200, {}, None

    def legal(weights, returns, prev_weights, port_ret, info):  # noqa: ANN001
        return 9.9e5, {"x": 9.9e5}, None

    reset_failure_flag()
    total, components, state = safe_call(huge, *fixture)
    assert total == SAFE_DEFAULT and components == {} and state is None
    assert candidate_failed() is True

    reset_failure_flag()
    total2, _, _ = safe_call(legal, *fixture)
    assert total2 == 9.9e5
    assert candidate_failed() is False


# --- POSIX-guarded: a resource bomb is killed at validate_once and the PARENT survives -------------


def _parent_survives_marker() -> str:
    """A trivial computation the parent runs AFTER a kill, proving it is still alive and usable."""
    return str(int(np.sum(np.arange(10))))


@pytest.mark.skipif(not _HAS_RESOURCE, reason="POSIX 'resource' module unavailable (Windows)")
def test_resource_bomb_killed_and_parent_survives(rng: np.random.Generator) -> None:
    """CONTAINMENT (POSIX): a gate-clean reward that allocates far past RLIMIT_AS is killed in the
    KILLABLE child and surfaces as a SandboxError — and crucially the PARENT (test) process survives
    and keeps running. This is the cross-platform-timeout / rlimit boundary (C2 / ADR-008)."""
    src = (
        "def reward(weights, returns, prev_weights, port_ret, info):\n"
        "    big = np.ones((8, 1024, 1024, 1024), dtype=np.float64)\n"  # ~64 GiB > 2 GiB cap
        "    return float(big.sum()), {}, None\n"
    )
    assert ast_gate(src) is True
    # validate_once uses the 'spawn' mp context; a runaway/oversized child must not take us down.
    assert mp.get_context("spawn") is not None  # the context the module itself uses (spawn)
    with pytest.raises(SandboxError):
        validate_once(src, _fixture(rng), timeout_s=10.0)
    # PARENT SURVIVES: we are still executing and can still compute.
    assert _parent_survives_marker() == "45"


@pytest.mark.skipif(not _HAS_RESOURCE, reason="POSIX 'resource' module unavailable (Windows)")
def test_infinite_loop_killed_and_parent_survives(rng: np.random.Generator) -> None:
    """CONTAINMENT (POSIX): a non-terminating reward is killed at the validate_once wall-clock
    timeout (the child is terminated) and the PARENT survives — the runaway cannot hang the run."""
    src = (
        "def reward(weights, returns, prev_weights, port_ret, info):\n"
        "    while True:\n"
        "        pass\n"
        "    return 0.0, {}, None\n"
    )
    assert ast_gate(src) is True
    with pytest.raises(SandboxError):
        validate_once(src, _fixture(rng), timeout_s=1.0)
    assert _parent_survives_marker() == "45"


# --------------------------------------------------------------------------------------------- #
# 2026-07-18 validation-handshake regression lock (commit-starvation forensics): timeout_s must  #
# clock ONLY the candidate's code — spawn + numpy import get environment grace; a starved spawn  #
# environment raises a DISTINCT environment error, never a candidate rejection.                  #
# --------------------------------------------------------------------------------------------- #
def test_handshake_verdict_timeout_names_the_candidate(rng: np.random.Generator) -> None:
    """A while-True reward is rejected with the CANDIDATE timeout message (phase 3), and the
    rejection arrives promptly (the graces gate earlier phases, not the verdict clock)."""
    import time

    src = """
def reward(weights, returns, prev_weights, port_ret, info):
    while True:
        pass
"""
    t0 = time.perf_counter()
    with pytest.raises(SandboxError, match="exceeded the 1.0s validation timeout"):
        validate_once(src, _fixture(rng), timeout_s=1.0)
    # ready+armed are fast on a healthy box; the whole call must be far below the graces
    assert time.perf_counter() - t0 < 30.0


def test_handshake_startup_crash_is_reported_not_misdiagnosed() -> None:
    """A startup failure inside the boot shim reaches the parent as a loud 'error' verdict
    (exercised in-process: a garbage fixture blob cannot unpickle)."""
    import queue as _queue

    from src.sandbox._child_boot import boot_candidate_child

    q: _queue.Queue = _queue.Queue()
    boot_candidate_child("def reward():\n    pass\n", b"not-a-pickle", q)
    status, message = q.get_nowait()  # phase 1
    assert status == "ready"
    status, message = q.get_nowait()  # early terminal
    assert status == "error"
    assert "crashed during startup" in message


def test_handshake_boot_module_is_stdlib_only_at_import() -> None:
    """The boot shim must import NOTHING heavy at module level — that is the whole point
    (numpy loads only AFTER 'ready'); a numpy/torch/module-level regression re-opens the
    startup-counted-against-the-candidate bug."""
    import ast
    from pathlib import Path

    tree = ast.parse(Path("src/sandbox/_child_boot.py").read_text(encoding="utf-8"))
    top_level_imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = {a.name for n in top_level_imports if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module for n in top_level_imports if isinstance(n, ast.ImportFrom)}
    assert names <= {"__future__", "typing"}, f"heavy module-level import(s) crept in: {names}"


def test_handshake_fixture_roundtrips_through_the_blob(rng: np.random.Generator) -> None:
    """Arrays must arrive in the child bit-intact through the pickle-bytes handoff: the
    validated reward's value on the fixture equals the in-process computation."""
    fixture = _fixture(rng)
    fn = validate_once(VALID_SRC, fixture, timeout_s=5.0)
    reset_failure_flag()
    total, _components, _state = safe_call(fn, *fixture)
    assert np.isclose(total, float(np.sum(fixture[0] * fixture[1])))


# ============================================================================ #
# The SandboxEnvironmentError CONTRACT, enforced repo-wide (deep review 2026-07-26, loop 2)
# ============================================================================ #
def _except_handler_names(handler: ast.ExceptHandler) -> set[str]:
    """Bare exception NAMES caught by one ``except`` clause (``Name`` or ``Tuple`` of ``Name``)."""
    t = handler.type
    if t is None:
        return {"<bare>"}
    nodes = t.elts if isinstance(t, ast.Tuple) else [t]
    out: set[str] = set()
    for n in nodes:
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
    return out


def test_sandbox_environment_error_is_caught_before_sandbox_error_everywhere():
    """EVERY ``except SandboxError`` must be preceded, in the SAME try, by ``SandboxEnvironmentError``.

    ``SandboxEnvironmentError`` subclasses ``SandboxError`` (src/sandbox/executor.py) precisely so that
    legacy handlers keep working — but its docstring makes the contract explicit: *callers that
    permanently ledger a rejection must catch this FIRST*, because a starved spawn environment is an
    ENVIRONMENT failure, not a candidate defect.

    The 2026-07-26 deep review found the contract documented but VIOLATED at two of the three ledgering
    call sites: ``src/orchestration/parallel.py::train_candidate`` set ``failed_validation = True`` —
    poisoning a good, PAID candidate into the frozen reject set that ``--resume`` replays — and
    ``scripts/run_campaign.py::_reinstantiate_frozen_winner`` raised ``ValueError``, which turns a
    transient starvation into a DETERMINISTIC exit-3 on every resume of the sealed test leg, blaming
    the frozen winner. ``src/llm/loop.py`` alone had it right.

    This is a whole-repo structural lock rather than one test per call site, so a NEW handler added
    later cannot reintroduce the defect silently.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []
    for path in sorted([*(root / "src").rglob("*.py"), *(root / "scripts").rglob("*.py")]):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a syntactically broken file fails elsewhere
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            seen_env = False
            for h in node.handlers:
                names = _except_handler_names(h)
                if "SandboxEnvironmentError" in names:
                    seen_env = True
                    continue
                if "SandboxError" in names and not seen_env:
                    offenders.append(f"{path.relative_to(root).as_posix()}:{h.lineno}")
    assert not offenders, (
        "these `except SandboxError` handlers are not preceded by `except SandboxEnvironmentError` "
        "in the same try, so a starved spawn environment would be permanently ledgered as a "
        f"candidate rejection: {offenders}"
    )


def test_inline_fallback_counter_starts_at_zero_and_is_readable():
    """The no-timeout inline fallback is COUNTED, not silent (deep review 2026-07-26, loop 2).

    ``validate_once`` degrades to ``_validate_inline`` — which takes no timeout at all — whenever a
    killable child cannot be spawned. That degradation used to be completely silent, so on a
    commit-/handle-starved box the sandbox could drop its only wall-clock timeout with no log line,
    no counter and no field on the record. The counter is the auditable signal; this locks in that it
    exists and is exported.
    """
    from src.sandbox.executor import inline_fallback_count

    n = inline_fallback_count()
    assert isinstance(n, int) and n >= 0


def test_defines_reward_separates_SAFE_from_USABLE() -> None:
    """``ast_gate`` proves SAFE; ``defines_reward`` proves USABLE — an empty completion is both safe
    and useless (deep review 2026-07-26).

    ``ast_gate("")`` is True (nothing dangerous is present in nothing), so an empty / whitespace /
    comment-only / no-reward completion — the canonical refusal & ``content_filter`` output — passed
    the campaign's pre-ship author gate and was SHIPPED to a cluster node. That is exactly the
    truncated/refused case the P8 gate (``cluster/campaign.py``) exists to stop; only the
    syntactically-invalid subset (prose) was actually being caught.
    """
    from src.sandbox.executor import ast_gate, defines_reward

    for src in ["", "   \n\n ", "# I cannot provide this", "import numpy as np\nx = 1"]:
        assert ast_gate(src) is True, f"the SAFETY gate alone accepts {src!r} — that is the point"
        assert defines_reward(src) is False, f"{src!r} must not count as a usable reward"

    # Both forms the executor accepts (`namespace.get("reward")` + `callable`) must still pass: a
    # def-only check would be STRICTER than validation and would discard real candidates.
    assert defines_reward("def reward(w, r, wp, pr, info):\n    return 0.0, {}, None")
    assert defines_reward("reward = lambda w, r, wp, pr, info: (0.0, {}, None)")
    assert defines_reward("reward: object = lambda w, r, wp, pr, i: (0.0, {}, None)")

    # A NESTED def never reaches the exec namespace, so it must NOT count — this mirrors the
    # executor's own "source defines no callable named 'reward'" rejection.
    assert not defines_reward(
        "def outer():\n    def reward(a, b, c, d, e):\n        return 0.0, {}, None"
    )
    # Unparseable stays rejected (fail-closed), as ast_gate already is.
    assert not defines_reward("def reward(:")
