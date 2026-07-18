"""Two-stage sandbox for untrusted LLM reward code (F.5, audit A-5).

LLM-authored reward code is untrusted. Per audit A-5 it is executed in two stages:

  1. VALIDATE ONCE per candidate (`validate_once`): AST-gate the source, then exec
     it once in a restricted namespace `{"np": numpy}` on a small fixture under a
     timeout. Reject crashers, timeouts, and contract violators here. This is the
     only stage that can be slow / use a timeout.
  2. DURING TRAINING (`safe_call`): the already-compiled function runs IN-PROCESS
     (fast) wrapped in a cheap per-call try/except. On NaN / inf / exception it
     substitutes SAFE_DEFAULT and flags the candidate. NO subprocess, NO per-step
     timeout (that would dominate the training cost).

The AST gate (`ast_gate`) rejects:
  - imports outside the allowlist (ALLOWED_IMPORTS = numpy only);
  - dunder attribute access (`x.__class__`, etc.);
  - banned numpy IO/FFI attributes (`np.load` pickle-RCE, `np.save`, `np.fromfile`,
    `np.genfromtxt`, `np.memmap`, `np.DataSource`, ... — see `_BANNED_ATTRS`; ADR-008);
  - calls to open / exec / eval / __import__.

The candidate child (`_candidate_child`) additionally applies best-effort POSIX
resource caps (address space, CPU, open files, file size) before exec so an LLM-written
reward cannot OOM the rented GPU box (ADR-008); on Windows those caps are absent and the
wall-clock timeout is the backstop.

Anonymized arrays only — no tickers, no dates — ever reach a reward (enforced by
the contract and exercised by the tests).

Audit refs: A-5 (gate-once then in-process), B-4 (state round-tripped via info).

Tests (test_sandbox.py):
  - os-import / file-read / date-reference source is rejected at the gate.
  - an infinite loop is killed at validate_once (timeout).
  - an in-process error is caught, SAFE_DEFAULT substituted, candidate flagged, and
    the training loop continues.
  - a reward never receives tickers or dates (anonymized arrays only).
"""

from __future__ import annotations

import ast
import builtins
import logging
import math
import re
import sys
from typing import Any

import numpy as np

from src.reward.contract import ALLOWED_IMPORTS, RewardFn, SAFE_DEFAULT

# Names that may never be called from untrusted reward source. The introspection/hook builtins
# (vars/globals/locals/dir/help/breakpoint) are ALSO runtime-blocked by absence from SAFE_BUILTINS,
# but listing them here makes the gate namespace-INDEPENDENT defense-in-depth (audit 2026-07-19):
# breakpoint() can drop into pdb (arbitrary-code path) and globals()/vars() expose the exec namespace.
_FORBIDDEN_CALLS: frozenset[str] = frozenset(
    {
        "open", "exec", "eval", "__import__", "compile", "getattr", "setattr", "input",
        "vars", "globals", "locals", "dir", "help", "breakpoint",
    }
)

# Attribute names rejected ANYWHERE in the source (ADR-008). Covers numpy's
# file/FFI/loader surface — np.load is a pickle-RCE vector; np.save/savez/savetxt/
# tofile + ndarray.dump/dumps write the filesystem (dump/dumps pickle to an arbitrary
# path — the tofile-equivalent write); np.fromfile/genfromtxt/loadtxt/recfromtxt/
# recfromcsv/fromregex/memmap/frombuffer read it or map it (recfromtxt/recfromcsv are
# genfromtxt aliases; fromregex takes a file-first arg); np.DataSource/lib/ctypeslib/
# f2py/testing expose IO/FFI/dynamic-eval paths — plus generic object-model escape
# routes (.mro, .open). A reward needs NONE of these for portfolio arithmetic; the
# static gate makes their use immediate, free, and loggable rather than relying on the
# namespace alone. (Ported from archive/pre_merge_repo_B/src_flat/sandbox.py; the
# recfromtxt/recfromcsv/fromregex + dump/dumps additions close the final-audit
# 2026-06-19 P1 read/write escapes.)
_BANNED_ATTRS: frozenset[str] = frozenset(
    {
        "load",
        "loads",
        "save",
        "savez",
        "savez_compressed",
        "savetxt",
        "loadtxt",
        "genfromtxt",
        "recfromtxt",         # genfromtxt alias (final-audit P1 read-escape)
        "recfromcsv",         # genfromtxt alias (final-audit P1 read-escape)
        "fromregex",          # file-first read (final-audit P1 read-escape)
        "fromfile",
        "tofile",
        "dump",               # ndarray.dump(path) pickles to an arbitrary path (final-audit P1 write-escape)
        "dumps",              # ndarray.dumps() pickles (final-audit P1 write-escape)
        "memmap",
        "frombuffer",
        "DataSource",
        "lib",
        "ctypeslib",
        "f2py",
        "testing",
        "mro",
        "open",
        "ctypes",              # ndarray.ctypes -> .data_as(...) is a libc FFI pointer vector (allowlist-RCE audit)
        "data",                # ndarray.data exposes the raw buffer pointer (allowlist-RCE audit)
    }
)

# ALLOWLIST of attribute names a reward may access (final-audit 2026-06-19 P1 RCE fix). The denylist
# above is unsound on its own: numpy's public object graph reaches os/builtins/pickle via gate-legal,
# non-dunder, non-banned submodule chains (e.g. ``np._pytesttester.os.system`` — verified end-to-end).
# The robust fix is an ALLOWLIST: every ``ast.Attribute`` must name a known-safe numeric/array/container
# operation. This is sound because the DANGEROUS LEAVES (system, popen, environ, getcwd, exec, socket,
# fork, ...) are not numeric ops, so they are absent here and NO chain — however deep — can reach them
# (the final hop would need its name in this set). Generous on purpose: a numeric name omitted here only
# costs a (logged, skipped) candidate, never a security hole. Pair with the dunder + denylist checks.
_ALLOWED_ATTRS: frozenset[str] = frozenset(
    {
        # --- reductions / statistics ---
        "sum", "prod", "mean", "std", "var", "min", "max", "ptp", "median", "average",
        "percentile", "quantile", "cumsum", "cumprod", "nancumsum", "nancumprod",
        "nanmean", "nanstd", "nanvar", "nansum", "nanprod", "nanmin", "nanmax",
        "nanmedian", "nanpercentile", "nanquantile", "cov", "corrcoef", "histogram",
        "bincount", "digitize", "count_nonzero", "argmin", "argmax", "nanargmin", "nanargmax",
        # --- elementwise math / ufuncs ---
        "abs", "absolute", "fabs", "sqrt", "cbrt", "square", "exp", "exp2", "expm1",
        "log", "log2", "log10", "log1p", "logaddexp", "logaddexp2", "sign", "signbit",
        "power", "float_power", "reciprocal", "negative", "positive", "add", "subtract",
        "multiply", "divide", "true_divide", "floor_divide", "mod", "remainder", "fmod",
        "maximum", "minimum", "fmax", "fmin", "clip", "round", "around", "rint", "round_",
        "floor", "ceil", "trunc", "modf", "frexp", "ldexp", "copysign", "nextafter",
        "sin", "cos", "tan", "arcsin", "arccos", "arctan", "arctan2", "sinh", "cosh",
        "tanh", "arcsinh", "arccosh", "arctanh", "deg2rad", "rad2deg", "hypot",
        "nan_to_num", "real", "imag", "conj", "conjugate", "angle", "heaviside",
        # --- logic / comparison ---
        "where", "isfinite", "isnan", "isinf", "isneginf", "isposinf", "isclose",
        "allclose", "array_equal", "array_equiv", "equal", "not_equal", "greater",
        "greater_equal", "less", "less_equal", "logical_and", "logical_or", "logical_not",
        "logical_xor", "all", "any", "isin", "in1d", "nonzero", "flatnonzero", "argwhere",
        "maximum_sctype", "isreal", "iscomplex",
        # --- array construction / shape / manipulation ---
        "array", "asarray", "asanyarray", "ascontiguousarray", "asfarray", "atleast_1d",
        "atleast_2d", "zeros", "ones", "full", "empty", "zeros_like", "ones_like",
        "full_like", "empty_like", "arange", "linspace", "logspace", "geomspace", "eye",
        "identity", "diag", "diagflat", "diagonal", "tri", "tril", "triu", "concatenate",
        "stack", "vstack", "hstack", "dstack", "column_stack", "row_stack", "append",
        "insert", "delete", "repeat", "tile", "pad", "reshape", "ravel", "flatten",
        "transpose", "swapaxes", "moveaxis", "rollaxis", "expand_dims", "squeeze", "sort",
        "argsort", "searchsorted", "partition", "argpartition", "unique", "flip", "fliplr",
        "flipud", "roll", "rot90", "diff", "ediff1d", "gradient", "trapz", "cross",
        "dot", "vdot", "inner", "outer", "matmul", "tensordot", "einsum", "kron", "trace",
        "take", "compress", "choose", "select", "extract", "place", "copyto", "broadcast_to",
        "broadcast_arrays", "meshgrid", "indices", "ix_", "fill", "fill_diagonal", "put",
        "copy", "astype", "tolist", "item", "view", "ravel_multi_index", "unravel_index",
        # --- linear algebra (submodule + its numeric leaves; dangerous leaves still excluded) ---
        "linalg", "norm", "inv", "pinv", "det", "solve", "lstsq", "eig", "eigh", "eigvals",
        "eigvalsh", "svd", "qr", "cholesky", "matrix_power", "matrix_rank", "slogdet", "cond",
        # --- dtypes / type predicates / constants ---
        "float64", "float32", "float16", "float128", "float_", "double", "longdouble",
        "int64", "int32", "int16", "int8", "int_", "intp", "uintp", "uint64", "uint32",
        "uint16", "uint8", "bool_", "complex64", "complex128", "complex_", "number",
        "integer", "floating", "complexfloating", "signedinteger", "unsignedinteger",
        "inexact", "generic", "dtype", "ndarray", "issubdtype", "can_cast", "result_type",
        "promote_types", "finfo", "iinfo", "nan", "inf", "NINF", "PINF", "NAN", "NaN",
        "Inf", "Infinity", "infty", "pi", "e", "euler_gamma", "newaxis",
        # --- ufunc methods (e.g. np.maximum.accumulate for running max) ---
        # NB ``seterr`` is DELIBERATELY OMITTED (V15b): it sets numpy's PROCESS-GLOBAL float-error
        # mode and does NOT restore it, so an untrusted reward calling ``np.seterr(all="ignore")``
        # would leak that mode into every later candidate, making the campaign order-dependent and
        # non-deterministic. ``errstate`` (a context manager that RESTORES the prior mode on exit)
        # and ``geterr`` (read-only) are leak-free and stay allowlisted.
        "reduce", "accumulate", "reduceat", "at", "errstate", "geterr",
        # --- ndarray attributes / numeric properties ---
        "shape", "size", "ndim", "T", "flat", "itemsize", "nbytes", "strides", "flags",
        # --- safe builtin container / scalar methods reward code uses ---
        "get", "keys", "values", "items", "setdefault", "pop", "popitem", "update",
        "extend", "index", "count", "remove", "insert", "discard", "add", "clear",
        "is_integer", "bit_length", "join", "split", "strip", "lower", "upper",
    }
)
# ``str.format`` / ``format_map`` are DELIBERATELY NOT allowlisted: a format string can do attribute +
# index access INSIDE a string LITERAL (e.g. ``'{0.__class__.__mro__[1].__subclasses__}'.format(x)``),
# which the AST gate cannot see because it inspects Attribute NODE names, not string contents — a proven
# dunder-walking RCE/info-disclosure escape (critical-review 2026-06-20). Reward code is pure numeric and
# never needs string formatting; ``_FORMAT_FIELD_RE`` below is a defence-in-depth scan of string literals.
_FORMAT_FIELD_RE = re.compile(r"\{[^{}]*[.\[][^{}]*\}")  # a replacement field with attr/index access

def _safe_import(
    name: str,
    globals: dict[str, Any] | None = None,  # noqa: A002 - matches builtins.__import__ signature
    locals: dict[str, Any] | None = None,  # noqa: A002
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    """A restricted ``__import__`` for the reward sandbox.

    NumPy lazily imports its own submodules (e.g. on the first ``ndarray.mean()`` /
    ``ndarray.var()`` call), and those imports resolve ``__builtins__['__import__']``
    from the *calling frame* — i.e. this sandbox namespace. Omitting ``__import__``
    therefore makes legitimate numpy reward code crash with ``KeyError('__import__')``
    both at validation and (worse) silently during training via ``safe_call``.

    The untrusted reward source itself can never reach this function: the AST gate
    rejects ``import`` statements, references to the ``__import__`` name (a dunder),
    and calls to ``__import__``. So the only callers are trusted numpy/Python
    internals. As defence-in-depth we still permit only numpy-rooted modules and
    modules already loaded at process start (numpy's transitive deps), denying any
    brand-new non-numpy import.
    """
    root = name.split(".")[0]
    if root == "numpy" or name in sys.modules:
        return builtins.__import__(name, globals, locals, fromlist, level)
    raise ImportError(f"import of {name!r} is not permitted in the reward sandbox")


# Builtins handed to exec'd reward source. Excludes open/exec/eval/compile/getattr/
# setattr/input and every IO primitive (the AST gate also statically rejects them).
# ``__import__`` is the *restricted* ``_safe_import`` above — required so numpy's lazy
# submodule loads work for legitimate reward code (e.g. ``returns.mean()``).
SAFE_BUILTINS: dict[str, Any] = {
    "__import__": _safe_import,
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "all": all,
    "any": any,
    "sorted": sorted,
    "reversed": reversed,
    "float": float,
    "int": int,
    "bool": bool,
    "str": str,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "round": round,
    "pow": pow,
    "divmod": divmod,
    "slice": slice,
    "isinstance": isinstance,
    "True": True,
    "False": False,
    "None": None,
}

# Module-level failure telemetry. The boolean is a LAST-call diagnostic (set on a substitution,
# cleared by the next success) consumed by tests via candidate_failed(); it has NO production caller —
# the loop does NOT gate candidates on it (docstring corrected 2026-07-03; the old text promised a
# mark-failed-and-move-on mechanism that never existed).
_LAST_CALL_FAILED: bool = False
# Accumulating counters across one WINDOW (reset by reset_failure_flag). The boolean above is
# LAST-call only (a later success "clears" it, by design — test_sandbox); these COUNT every substitution
# so a reward that fails on SOME steps but succeeds on the final one is still surfaced and QUANTIFIED
# (the silent partial-degradation gap — R66, audit 2026-06-28). The ACTUAL visibility mechanism
# (2026-07-03): each consumer resets the counters at its window start and reads them at its window end —
# * TRAINING: src/agents/trainer.py::train_agent resets before model.learn, then WARNs with the
#   substitution fraction and attaches train_safe_default_count/train_safe_call_count to the returned
#   policy, which the loop / parallel worker archive into the candidate record's metrics;
# * EVALUATION ROLLOUTS: src/env/runner.py resets before each rollout and WARNs on the count after it.
# SEQUENTIAL-USE INVARIANT (batch-4 audit, 2026-07-03): reset-at-window-start / read-at-window-end is
# sound ONLY because each process runs ONE reward window at a time (train, then rollouts, serially;
# parallel workers are separate processes with their own module globals). Concurrent rollouts inside
# one process would interleave the counts — scope these per-thread/per-env before ever doing that.
_SAFE_DEFAULT_COUNT: int = 0
_LOG = logging.getLogger(__name__)

_CALL_COUNT: int = 0

#: How many times ``validate_once`` had to DEGRADE to the no-timeout inline path because a killable
#: child could not be spawned. Previously that degradation was completely silent (deep-review
#: 2026-07-26, loop 2): on a commit-/handle-starved box ``Process.start()`` raises OSError, the
#: fallback fires, and the candidate is then validated with NO wall-clock timeout at all — an
#: infinite-loop reward hangs the worker forever, with no log line, no monitor anomaly and no field
#: on the record to say the timeout was skipped. It is now counted, logged at ERROR, and stamped on
#: the returned callable as ``fn.validated_inline``. Read via :func:`inline_fallback_count`.
_INLINE_FALLBACK_COUNT: int = 0


def inline_fallback_count() -> int:
    """Number of no-timeout inline validation fallbacks taken in this process (see the module note).

    A nonzero value means at least one candidate was validated WITHOUT the wall-clock timeout, which
    a campaign audit should surface rather than discover from a hung worker.
    """
    return _INLINE_FALLBACK_COUNT


class SandboxError(Exception):
    """Raised by validate_once when a candidate reward is rejected."""


class SandboxEnvironmentError(SandboxError):
    """Raised when validation could not RUN because the spawn ENVIRONMENT is starved — NOT a
    candidate defect (2026-07-19 audit). Subclasses ``SandboxError`` so existing ``except
    SandboxError`` sites still catch it, but callers that permanently ledger a rejection must
    catch this FIRST and treat it as retryable/abort-and-resume: a good candidate hit by a
    transient commit-/CPU-starved box during PAID authoring must never be poisoned into the
    frozen reject set that ``--resume`` replays."""


# Resource ceilings for the candidate child (ADR-008). Address space caps OOM blast
# radius on the rented GPU box; the rest bound CPU burn, open files, and file size.
_RLIMIT_AS_BYTES: int = 2 * 1024 * 1024 * 1024  # ~2 GiB address space
_RLIMIT_CPU_SECONDS: int = 15  # CPU-seconds ceiling (wall-clock timeout is the primary stop)
_RLIMIT_NOFILE: int = 64  # open file descriptors
_RLIMIT_FSIZE_BYTES: int = 1 * 1024 * 1024  # ~1 MiB max written-file size

# Environment grace for the validation child's NON-candidate phases (handshake, 2026-07-18):
# spawn+interpreter start ("ready") and numpy/MKL import + fixture unpickle ("armed"). Sized
# from forensics — a commit-starved box completed the numpy DLL load in ~103 s worst-case; the
# grace must absorb a starved-but-recovering environment without hiding a truly wedged one.
_STARTUP_GRACE_S: float = 45.0
_IMPORT_GRACE_S: float = 120.0


def _apply_resource_limits() -> None:  # pragma: no cover - called only from the spawned _candidate_child (POSIX rlimits); never in-process
    """Best-effort POSIX resource caps for the candidate child, applied before exec.

    Each limit is set per-resource, clamped to the existing HARD limit (never raised),
    and skipped if the platform refuses it (macOS rejects some combos) — exactly the
    pattern ported from archive/pre_merge_repo_B/src_flat/sandbox.py::_limit.

    On Windows the ``resource`` module does not exist: this is a SILENT no-op there and
    the parent's wall-clock timeout in ``validate_once`` is the only backstop. The
    Linux/4090 campaign box enforces every cap. No psutil RSS watchdog is added for the
    Windows gap because psutil is deliberately not a project dependency (ADR-008).
    """
    try:
        import resource
    except ImportError:  # non-POSIX (Windows): wall-clock timeout remains the backstop
        return

    def _set(name: str, soft: int) -> None:  # pragma: no cover - POSIX-only; Windows returns above; enforced on the Linux/4090 box (ADR-008)
        res = getattr(resource, name, None)
        if res is None:
            return
        try:
            _, hard = resource.getrlimit(res)  # type: ignore[attr-defined]  # POSIX-only; no Windows stub
            tgt = soft if hard == resource.RLIM_INFINITY else min(soft, hard)  # type: ignore[attr-defined]
            resource.setrlimit(res, (tgt, tgt))  # type: ignore[attr-defined]
        except (ValueError, OSError):
            pass  # best-effort per-resource (Darwin rejects some combinations)

    _set("RLIMIT_AS", _RLIMIT_AS_BYTES)        # pragma: no cover - POSIX-only (see _set)
    _set("RLIMIT_CPU", _RLIMIT_CPU_SECONDS)    # pragma: no cover - POSIX-only (see _set)
    _set("RLIMIT_NOFILE", _RLIMIT_NOFILE)      # pragma: no cover - POSIX-only (see _set)
    _set("RLIMIT_FSIZE", _RLIMIT_FSIZE_BYTES)  # pragma: no cover - POSIX-only (see _set)


def _candidate_child(src: str, fixture: Any, q: Any) -> None:  # pragma: no cover - spawned-child body; runs only in a `validate_once` subprocess (integration-tested), never in-process line-tracked
    """Child-process target: exec the gated source, call it once, contract-check it.

    Runs in a SEPARATE process so a runaway reward (e.g. ``while True``) can be
    *killed* by the parent on timeout — cross-platform, unlike ``signal.SIGALRM``
    (Unix-main-thread only; a silent no-op on Windows). Reports the verdict on ``q``:
    ``("ok", "")`` or ``("error", message)``. A hang reports nothing and is caught by
    the parent's queue-get timeout. (C2 / ADR-028.)

    Before exec, best-effort POSIX resource caps are applied so an LLM-written reward
    cannot OOM the rented GPU box or exhaust file handles (ADR-008). On Windows the
    ``resource`` module is absent: the caps are SILENTLY SKIPPED and the parent's
    wall-clock timeout (``validate_once`` queue-get -> ``terminate``) is the only
    backstop there. The Linux/4090 experiment box — the box that actually runs the
    campaign — enforces all of them. (psutil is not a project dependency, so no RSS
    watchdog is added on Windows; ADR-008 records this as a documented platform gap.)
    """
    _apply_resource_limits()
    try:
        namespace: dict[str, Any] = {"np": np, "__builtins__": SAFE_BUILTINS}
        exec(compile(src, "<candidate_reward>", "exec"), namespace)  # noqa: S102
        fn = namespace.get("reward")
        if not callable(fn):
            q.put(("error", "source defines no callable named 'reward'"))
            return
        out = fn(*fixture)
        # Mirror production safe_call (executor.py:699), which unpacks ANY 3-element iterable, not tuples
        # only: a reward returning a 3-element LIST trains fine yet was falsely rejected here (audit
        # 2026-07-19). Validation must reject exactly what production rejects — no stricter.
        try:
            total, components, _state = out
        except (TypeError, ValueError):
            q.put(("error", "reward did not return an unpackable (total, components, state)"))
            return
        try:
            total_f = float(total)
        except (TypeError, ValueError):
            q.put(("error", "reward total is not coercible to float"))
            return
        if not math.isfinite(total_f):
            q.put(("error", "reward returned a non-finite total"))
            return
        if not isinstance(components, dict):
            q.put(("error", "reward components is not a dict"))
            return
        q.put(("ok", ""))
    except Exception as exc:  # noqa: BLE001 — any failure is a candidate rejection
        q.put(("error", f"reward crashed during validation: {exc!r}"))


def _validate_inline(src: str, fixture: Any) -> RewardFn:
    """Fallback validation WITHOUT a hard timeout (crash/contract checks still apply).

    Used only when a killable child process cannot be spawned (e.g. inside a daemonic
    worker). The orchestrator runs candidates in NON-daemon workers, so this is rare.
    """
    namespace: dict[str, Any] = {"np": np, "__builtins__": SAFE_BUILTINS}
    try:
        exec(compile(src, "<candidate_reward>", "exec"), namespace)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        raise SandboxError(f"reward source failed to compile/exec: {exc!r}") from exc
    fn = namespace.get("reward")
    if not callable(fn):
        raise SandboxError("source defines no callable named 'reward'")
    try:
        out = fn(*fixture)
    except Exception as exc:  # noqa: BLE001
        raise SandboxError(f"reward crashed during validation: {exc!r}") from exc
    # Mirror production safe_call (executor.py:699): unpack ANY 3-element iterable, not tuples only
    # (audit 2026-07-19 — a 3-element list trains fine but was falsely rejected here).
    try:
        total, components, _state = out
    except (TypeError, ValueError) as exc:
        raise SandboxError("reward did not return an unpackable (total, components, state)") from exc
    try:
        total_f = float(total)
    except (TypeError, ValueError) as exc:
        raise SandboxError("reward total is not coercible to float") from exc
    if not math.isfinite(total_f):
        raise SandboxError("reward returned a non-finite total")
    if not isinstance(components, dict):
        raise SandboxError("reward components is not a dict")
    return fn  # type: ignore[return-value]


_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_CODE_START_RE = re.compile(r"^\s*(?:def |async def |import |from |@|class )")


def _parses_as_python(src: str) -> bool:
    """True iff ``src`` is syntactically valid Python (the precondition ast_gate needs)."""
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False


def defines_reward(src: str) -> bool:
    """True iff ``src`` BINDS a top-level name ``reward`` (deep review 2026-07-26).

    ``ast_gate`` answers "is this source SAFE", which is a different question from "is this a REWARD":
    ``ast_gate("")`` is ``True`` because nothing dangerous is present in nothing. So an EMPTY,
    whitespace-only, comment-only, or code-without-a-reward completion — the canonical refusal /
    ``content_filter`` output (``src/llm/client.py`` ``_INCOMPLETE_STOP_REASONS``) — passes a safety
    gate untouched. Callers deciding whether a completion is USABLE (not merely harmless) need both.

    Deliberately mirrors what the executor actually accepts (``namespace.get("reward")`` +
    ``callable``), so ``reward = lambda ...`` counts: a check restricted to ``def`` would be STRICTER
    than validation and could reject a valid candidate. The bias is intentional — a false negative
    would discard real science, a false positive only wastes the node slot the caller was already
    prepared to spend.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    for node in tree.body:  # top level only: a nested def never reaches the exec namespace
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "reward":
            return True
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "reward" for t in node.targets
        ):
            return True
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "reward"
        ):
            return True
    return False


def extract_reward_source(text: str) -> str:
    """Recover runnable reward source from a raw LLM completion (final-audit 2026-06-19 P0 fix).

    A real reward-author (Claude Opus 5 with adaptive thinking, Gemini 3.5 Flash, ...) frequently
    wraps the function in a markdown fence (```python ... ```) or adds a prose preamble/epilogue
    despite the 'return only the function' instruction. ``ast_gate`` parses the WHOLE string, so any
    such wrapper makes ``ast.parse`` raise SyntaxError and a perfectly good reward is rejected for
    FORMATTING, not logic — which, at campaign scale, can starve every arm of candidates (the stub
    returns bare code, so the fast tests never see this). This shim strips the wrapper:

      1. if a fenced block is present, take the first that defines ``reward`` (else the first);
      2. else if the text already parses, return it unchanged (the stub / clean-code path: no-op);
      3. else drop a prose preamble — slice from the first code-start line and trim any trailing
         prose to the longest suffix that still parses.

    Deliberately conservative: on anything it cannot confidently extract it returns the stripped
    text and lets ``ast_gate`` reject it honestly (no silent acceptance of garbage).
    """
    if not text or not text.strip():
        return text
    # Fast path: if the WHOLE text is already valid Python, return it UNCHANGED, preserving exact
    # bytes (trailing newline, etc.) so the archived source + its hash are stable. A fenced / prose-
    # wrapped completion does NOT parse (the fences/prose are syntax errors), so it correctly falls
    # through to extraction; clean code that merely CONTAINS a ``` inside a string still parses and is
    # returned byte-identical (re-audit: the prior `"```" not in text` guard broke that no-op case).
    if _parses_as_python(text):
        return text
    s = text.strip()
    blocks = _FENCE_RE.findall(s)
    if blocks:
        # Prefer a block that BOTH parses AND defines reward, then any block that merely
        # parses, then fall through to the original conservative choice (first containing
        # "def reward", else the first block). Selecting purely on the "def reward"
        # SUBSTRING committed to a syntactically broken first block even when a later block
        # held the clean reward — starving the arm of a recoverable candidate (audit fix:
        # multi-fence "first attempt ... corrected version" completions).
        stripped = [b.strip() for b in blocks]
        parsable = [b for b in stripped if _parses_as_python(b)]
        s = next(
            (b for b in parsable if "def reward" in b),
            parsable[0]
            if parsable
            else next((b for b in stripped if "def reward" in b), stripped[0]),
        )
    if _parses_as_python(s):
        return s
    lines = s.splitlines()
    start = next((i for i, ln in enumerate(lines) if _CODE_START_RE.match(ln)), None)
    if start is not None:
        for end in range(len(lines), start, -1):
            candidate = "\n".join(lines[start:end]).strip()
            if candidate and _parses_as_python(candidate):
                return candidate
    return s


def ast_gate(src: str) -> bool:
    """Statically reject unsafe reward source before any execution.

    Algorithm (FINAL_PLAN F.5; ADR-008 attribute denylist):
        Parse `src` and walk the AST. Reject if any node is:
          - an Import / ImportFrom outside ALLOWED_IMPORTS;
          - an Attribute whose name starts with "__" (dunder access) OR is in
            _BANNED_ATTRS (numpy IO/FFI surface: load/save/fromfile/genfromtxt/
            memmap/DataSource/lib/ctypeslib/... and .mro/.open);
          - a Name starting with "__" (dunder name reference);
          - a Call to one of _FORBIDDEN_CALLS (open, exec, eval, __import__, ...).
        Otherwise accept.

    Args:
        src: The candidate reward source code.

    Returns:
        True iff the source passes every gate check.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False

    # DOCSTRINGS are exempt from the format-field scan in check 5 below (2026-07-26 review). The scan
    # is defence-in-depth against a format TEMPLATE smuggling attribute access inside a string literal,
    # but a docstring can never BE that template: reaching one requires ``fn.__doc__`` / ``__doc__``,
    # both dunders already rejected by checks 2-3. Scanning them was pure false-positive cost, and an
    # expensive one — the initial-generation prompt shows the author the literal example
    # ``{"port_ret": float(port_ret)}``, so a reward whose docstring documents its own components dict
    # (e.g. ``Returns (total, {"cvar_05": -0.03}, state)``) matched the regex and the whole PAID
    # candidate was discarded for its prose, not its logic. Every non-docstring string is still scanned.
    _docstrings: set[int] = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }

    for node in ast.walk(tree):
        # 1. Imports must be inside the numpy allowlist.
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    return False
        elif isinstance(node, ast.ImportFrom):
            # SECURITY (RCE fix, 2026-06-25): a root-module-only check let `from numpy import load`
            # through (numpy is allowlisted), after which `load(...)` is a BARE ast.Name that the
            # _BANNED_ATTRS attribute-allowlist never inspects -> the np.load pickle-RCE the banlist
            # exists to stop (confirmed end-to-end). `from numpy import *` is the same hole. Reward code
            # only ever needs `import numpy as np`, so forbid `from ... import` (and wildcard) entirely.
            return False

        # 2. No dunder attribute access (e.g. ().__class__, x.__globals__), and no
        #    banned numpy IO/FFI or object-model attributes (np.load pickle-RCE,
        #    np.save/fromfile/genfromtxt/memmap/DataSource, .mro, ...). (ADR-008.)
        elif isinstance(node, ast.Attribute):
            # Store/Del context = attribute MUTATION (`np.mean = ...`, `del np.mean`,
            # `np.pi += 1`). numpy is a process-global singleton and sandbox workers are
            # REUSED across candidates, so an allowlisted-name assignment would poison every
            # later candidate in that worker (cross-candidate contamination + determinism
            # break). Reward code only ever READS attributes.
            if not isinstance(node.ctx, ast.Load):
                return False
            if node.attr.startswith("__"):
                return False
            if node.attr in _BANNED_ATTRS:
                return False
            # ALLOWLIST (final-audit 2026-06-19 P1 RCE fix): the attribute must name a known-safe
            # numeric / array / container op. This blocks numpy-submodule traversal to
            # os/sys/builtins/pickle (e.g. np._pytesttester.os.system) because the chain's final
            # hop would need its name in _ALLOWED_ATTRS, and the dangerous leaves are absent.
            if node.attr not in _ALLOWED_ATTRS:
                return False

        # 3. No dunder *name* references (e.g. __import__, __builtins__).
        elif isinstance(node, ast.Name):
            if node.id.startswith("__"):
                return False

        # 4. No calls to dangerous builtins.
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _FORBIDDEN_CALLS:
                return False
            if isinstance(func, ast.Attribute) and func.attr in _FORBIDDEN_CALLS:
                return False

        # 5. No format-string field access (defence-in-depth for the str.format dunder-walk escape: a
        #    string LITERAL with a replacement field that does attribute/index access, which the Attribute
        #    walk above cannot see — e.g. '{0.__class__.__mro__[1].__subclasses__}'.format(x)).
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in _docstrings and _FORMAT_FIELD_RE.search(node.value):
                return False

    return True


def validate_once(src: str, fixture: Any, timeout_s: float = 2.0) -> RewardFn:
    """Gate the source, then run it exactly once on a fixture under a timeout.

    Algorithm (stage 1 of audit A-5):
        Assert ast_gate(src); compile/exec the source in the restricted namespace
        {"np": numpy}; invoke the resulting `reward` once on `fixture` under a
        wall-clock timeout. Reject (raise) on gate failure, crash, timeout, or a
        contract violation. Return the compiled callable on success.

    Args:
        src: The candidate reward source code.
        fixture: A small anonymized fixture matching the reward contract args.
        timeout_s: Wall-clock timeout for the single validation run.

    Returns:
        The validated, compiled reward callable.

    Raises:
        SandboxError: on gate failure, exec/compile error, missing/non-callable
            ``reward``, runtime crash, timeout, or a contract violation.

    Notes
    -----
    The raw ``src`` is first run through :func:`extract_reward_source` (a no-op on already-clean
    code) so a markdown-fenced / prose-wrapped LLM completion is salvaged before the gate. The
    orchestrators ALSO extract at the LLM boundary so the *archived* source is clean; this is the
    single-choke-point safety net that protects every current and future caller.

    Notes
    -----
    The wall-clock timeout runs the candidate in a KILLABLE child process, so it is
    enforced on EVERY platform (C2 / ADR-028 — ``signal.SIGALRM`` is Unix-main-thread
    only and was a silent no-op on Windows, letting a ``while True`` reward hang the
    whole run). The orchestrator must run candidates in NON-daemon workers so the child
    can be spawned; if it cannot (e.g. inside a daemonic worker), validation falls back
    to inline crash/contract checks without a hard timeout.

    Handshake (2026-07-18): the child boots via ``src/sandbox/_child_boot.py`` and
    reports ``ready`` (spawned) then ``armed`` (imports + fixture done) before the
    verdict, so ``timeout_s`` clocks ONLY the candidate's own code — spawn and
    numpy/MKL import time (minutes on a commit-starved box or a contended node) get
    separate environment grace (``_STARTUP_GRACE_S``/``_IMPORT_GRACE_S``) and their
    exhaustion raises a DISTINCT environment error, never a candidate rejection.
    """
    src = extract_reward_source(src)  # salvage fenced / prose-wrapped LLM output (final-audit P0)
    if not ast_gate(src):
        raise SandboxError("ast_gate rejected the candidate source")

    # Runtime validation in a killable child process (cross-platform timeout, C2/ADR-028),
    # via the three-phase handshake (ready -> armed -> verdict; src/sandbox/_child_boot.py).
    # 2026-07-18 (commit-starvation forensics, pre-launch): ``timeout_s`` used to clock the
    # child's WHOLE lifetime, so spawn + numpy DLL load counted against the candidate — on a
    # commit-starved box or a contended cluster node startup alone exceeds 2 s and a GOOD
    # reward was rejected (a paid candidate discarded at authoring, or a sealed-leg seed
    # failed on re-instantiation). Phases 1-2 (spawn, imports, fixture unpickle) now get
    # generous ENVIRONMENT grace; the strict ``timeout_s`` clocks ONLY the candidate's code.
    import multiprocessing as mp
    import pickle
    import queue as _queue

    from src.sandbox._child_boot import boot_candidate_child

    ctx = mp.get_context("spawn")  # Windows-safe; uniform across platforms
    q: Any = ctx.Queue()
    proc = ctx.Process(target=boot_candidate_child,
                       args=(src, pickle.dumps(fixture), q), daemon=True)
    try:
        proc.start()
    except (AssertionError, OSError, RuntimeError) as _spawn_exc:  # pragma: no cover - spawn-failure fallback (only inside a daemonic worker); the inline path is covered directly via _validate_inline tests
        # Cannot spawn a killable child (inside a daemonic worker, or the box is commit-/handle-
        # starved) -> inline fallback, which has NO wall-clock timeout: an infinite-loop reward on
        # this path hangs the worker forever. The degradation is legitimate (a daemonic worker
        # genuinely cannot spawn) but was previously SILENT — no log, no counter, no field on the
        # record — so a starved box could quietly drop the sandbox's only real timeout
        # (deep-review 2026-07-26, loop 2). It is now counted, logged at ERROR, and stamped on the
        # returned callable so a campaign audit can find it instead of a hung worker.
        global _INLINE_FALLBACK_COUNT
        _INLINE_FALLBACK_COUNT += 1
        _LOG.error(
            "sandbox validation DEGRADED to the inline path (NO wall-clock timeout): could not "
            "spawn a killable child (%s: %s). This is expected inside a daemonic worker, but on a "
            "commit-/handle-starved box it means this candidate was validated without a timeout. "
            "Inline fallbacks so far in this process: %d.",
            type(_spawn_exc).__name__, _spawn_exc, _INLINE_FALLBACK_COUNT,
        )
        _fn = _validate_inline(src, fixture)
        try:
            _fn.validated_inline = True  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - a non-writable callable must not break validation
            pass
        return _fn

    timed_out_phase: str | None = None
    status, message = "error", "validation child ended without a verdict"
    try:
        for phase, grace in (("ready", _STARTUP_GRACE_S), ("armed", _IMPORT_GRACE_S),
                             ("verdict", float(timeout_s))):
            try:
                status, message = q.get(timeout=grace)
            except _queue.Empty:
                timed_out_phase = phase
                break
            if status in ("ok", "error"):  # terminal verdict (possibly early, e.g. startup crash)
                break
    finally:
        # Graceful first: a child that just delivered its verdict is exiting on its own —
        # killing it mid-teardown is what the forensics flagged as needless violence.
        proc.join(timeout=1.0)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2.0)

    if timed_out_phase == "verdict":
        raise SandboxError(f"reward exceeded the {timeout_s}s validation timeout")
    if timed_out_phase is not None:
        raise SandboxEnvironmentError(
            f"validation child did not reach phase {timed_out_phase!r} within its "
            f"environment grace — the SPAWN ENVIRONMENT is starved (system commit "
            f"charge / CPU contention), NOT a candidate defect; free resources and retry")
    if status == "error":
        raise SandboxError(str(message))

    # Runtime cleared in the child; compile in THIS process to return the callable
    # (defining the function never hangs — only CALLING it can, which the child cleared).
    namespace: dict[str, Any] = {"np": np, "__builtins__": SAFE_BUILTINS}
    try:
        exec(compile(src, "<candidate_reward>", "exec"), namespace)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        raise SandboxError(f"reward source failed to compile/exec: {exc!r}") from exc
    return namespace["reward"]  # type: ignore[return-value]


def safe_call(fn: RewardFn, *args: Any) -> tuple[float, dict[str, float], object]:
    """Invoke a validated reward in-process, substituting SAFE_DEFAULT on failure.

    Algorithm (stage 2 of audit A-5):
        try: total, components, state = fn(*args); require np.isfinite(total);
             return (float(total), components, state).
        except (or non-finite total): return (SAFE_DEFAULT, {}, None) and flag the
             candidate. No subprocess, no per-step timeout.

    CONTAINMENT BOUNDARY (final-audit #12; Windows truth + residuals recorded 2026-07-03, batch-4
    audit): this in-process training path is deliberately fast and is NOT a security boundary — it
    applies no rlimits and no timeout. The boundary is (a) the static ``ast_gate`` ALLOWLIST, which
    denies the reward any reach to os/subprocess/socket/FFI, and (b) ``validate_once``, which runs the
    reward once in a killable child under a PARENT-ENFORCED wall-clock timeout (real on every platform
    incl. the Windows laptop the campaign actually runs on — ADR-028/ADR-040; the POSIX RLIMIT_AS/CPU
    caps are a silent no-op there, superseding the old "Linux campaign box" framing) on a
    PRODUCTION-SHAPE fixture, so shape-proportional allocation bombs and single-step crashes surface
    at validation. Accepted residuals under the non-adversarial threat model (the authors are
    Opus/Qwen writing genuine reward code): a reward whose COST DEPENDS ON INPUT VALUES (cheap on the
    fixture, explosive on a real return) or whose ``reward_state`` accumulates unboundedly across
    ~200k in-process steps can still hang/OOM its training worker. Recovery is OPERATIONAL, not
    in-process: watcher stall alert -> kill -> supervisor ``--resume`` (the un-archived slot
    regenerates and the matched budget holds; CAMPAIGN_RUNBOOK §6). A reward that is gate-clean and
    finite on the fixture is trusted to run in-process.

    Args:
        fn: A reward callable already cleared by `validate_once`.
        *args: The contract arguments (weights, returns, prev_weights, port_ret, info).

    Returns:
        (total, components, reward_state); the safe-default triple on failure.
    """
    global _LAST_CALL_FAILED, _SAFE_DEFAULT_COUNT, _CALL_COUNT
    _CALL_COUNT += 1
    try:
        out = fn(*args)
        total, components, state = out
        total_f = float(total)
        if not math.isfinite(total_f):
            raise ValueError("non-finite total")
        # row 30e (audit 2026-07-22): a finite-but-astronomical reward (e.g. 1e200 from a
        # decayed-denominator ratio) previously passed straight into SAC — the only magnitude
        # guard was PopArt's sigma cap, absent in the popart=False ablation. Treat it exactly
        # like non-finite: candidate-failure semantics, SAFE_DEFAULT substitution, counted.
        if abs(total_f) > 1.0e6:
            raise ValueError(f"reward magnitude {total_f!r} exceeds the 1e6 contract bound")
    except Exception:  # noqa: BLE001 — any failure is a candidate failure, not fatal
        _LAST_CALL_FAILED = True
        _SAFE_DEFAULT_COUNT += 1  # accumulate across the window (R66); the bool is last-call only
        return (SAFE_DEFAULT, {}, None)

    _LAST_CALL_FAILED = False
    return (total_f, components, state)


def reset_failure_flag() -> None:
    """Clear the candidate-failed flag AND the substitution counters before a rollout (P0-2)."""
    global _LAST_CALL_FAILED, _SAFE_DEFAULT_COUNT, _CALL_COUNT
    _LAST_CALL_FAILED = False
    _SAFE_DEFAULT_COUNT = 0
    _CALL_COUNT = 0


def candidate_failed() -> bool:
    """True if the MOST RECENT ``safe_call`` substituted SAFE_DEFAULT (last-call flag; P0-2).

    NB: a subsequent successful call clears this by design (test_sandbox). To detect whether ANY call
    failed during a rollout window — or HOW MANY — use :func:`safe_default_count` (the rollout warns on
    that count, so an intermittent reward that recovers on the final step is still surfaced; R66).
    """
    return _LAST_CALL_FAILED


def safe_default_count() -> int:
    """Number of SAFE_DEFAULT substitutions since the last :func:`reset_failure_flag` (R66).

    Unlike :func:`candidate_failed` (last-call only), this accumulates over the whole window, so a
    reward that silently fails on a FRACTION of steps is quantifiable rather than invisible.
    """
    return _SAFE_DEFAULT_COUNT


def safe_call_count() -> int:
    """Total ``safe_call`` invocations since the last :func:`reset_failure_flag` (denominator for the
    SAFE_DEFAULT substitution FRACTION; R66)."""
    return _CALL_COUNT
