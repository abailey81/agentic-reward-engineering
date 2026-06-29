"""Keyless stub reward-designer transport (MASTER_EXECUTION_PLAN P4, Pass A).

A deterministic, offline ``transport`` (``(system, user) -> source``) that emits **valid,
varied** reward-function code so the ENTIRE LLM-arm pipeline (sandbox-validate → train the
fixed SAC → measure the realized-return distribution → score held-out fitness → archive)
can be exercised end-to-end on real GPU/CPU + data **without an API key**. It is wired into
the loop exactly like a real LLM client (``LLMClient(cfg, transport=StubDesignerTransport(...))``);
swapping it for ``make_openai_transport`` / ``make_anthropic_transport`` turns Pass A into the
real headline run (Pass B) with no other change.

**This is NOT an LLM and NOT the scientific result.** It is a machinery-validation harness:
the generated rewards are drawn from a fixed library of archetypes spanning the H4 reward
family and tail-aware designs (return-only, turnover-penalised, rolling-variance, rolling-CVaR,
drawdown-penalised, online differential-Sharpe — Moody-Saffell 2001), with coefficients varied
deterministically by call index. Every archetype is numpy-only, passes the AST gate, conforms to
the reward contract, and is finite on the validation fixture (verified in tests/test_stub_designer.py).
"""
from __future__ import annotations

import numpy as np

__all__ = ["StubDesignerTransport", "ARCHETYPES"]

# Placeholder tokens (NOT str.format, because the code bodies contain literal ``{...}`` dicts).
# Each archetype is filled by ``str.replace`` of __C1__/__WINDOW__/__ALPHA__/__ETA__.

_RETURN_ONLY = '''
def reward(weights, returns, prev_weights, port_ret, info):
    total = float(port_ret)
    return total, {"port_ret": total}, None
'''

_TURNOVER = '''
def reward(weights, returns, prev_weights, port_ret, info):
    turnover = float(np.sum(np.abs(weights - prev_weights)))
    total = float(port_ret) - __C1__ * turnover
    return total, {"port_ret": float(port_ret), "turnover": turnover}, None
'''

_ROLLING_VARIANCE = '''
def reward(weights, returns, prev_weights, port_ret, info):
    state = info.get("reward_state")
    hist = list(state) if state is not None else []
    hist.append(float(port_ret))
    if len(hist) > __WINDOW__:
        hist = hist[-__WINDOW__:]
    arr = np.asarray(hist, dtype=float)
    var = float(np.var(arr)) if arr.size > 1 else 0.0
    total = float(port_ret) - __C1__ * var
    return total, {"port_ret": float(port_ret), "variance": var}, hist
'''

_ROLLING_CVAR = '''
def reward(weights, returns, prev_weights, port_ret, info):
    state = info.get("reward_state")
    hist = list(state) if state is not None else []
    hist.append(float(port_ret))
    if len(hist) > __WINDOW__:
        hist = hist[-__WINDOW__:]
    arr = np.sort(np.asarray(hist, dtype=float))
    k = max(1, int(np.ceil(__ALPHA__ * arr.size)))
    cvar = float(np.mean(arr[:k]))
    total = float(port_ret) - __C1__ * max(0.0, -cvar)
    return total, {"port_ret": float(port_ret), "cvar": cvar}, hist
'''

_DRAWDOWN = '''
def reward(weights, returns, prev_weights, port_ret, info):
    state = info.get("reward_state")
    cum = float(state[0]) if state is not None else 0.0
    peak = float(state[1]) if state is not None else 0.0
    cum = cum + float(np.log1p(port_ret))
    peak = max(peak, cum)
    drawdown = peak - cum
    total = float(port_ret) - __C1__ * drawdown
    return total, {"port_ret": float(port_ret), "drawdown": float(drawdown)}, (cum, peak)
'''

_DIFFERENTIAL_SHARPE = '''
def reward(weights, returns, prev_weights, port_ret, info):
    state = info.get("reward_state")
    eta = __ETA__
    r = float(port_ret)
    a = float(state[0]) if state is not None else 0.0
    b = float(state[1]) if state is not None else 0.0
    d_a = r - a
    d_b = r * r - b
    denom = (b - a * a) ** 1.5
    dsr = (b * d_a - 0.5 * a * d_b) / denom if denom > 1e-12 else 0.0
    a = a + eta * d_a
    b = b + eta * d_b
    return float(dsr), {"dsr": float(dsr), "port_ret": r}, (a, b)
'''

#: The fixed archetype library (return-only first so the very first candidate is trivial-valid).
ARCHETYPES: tuple[str, ...] = (
    _RETURN_ONLY,
    _TURNOVER,
    _ROLLING_VARIANCE,
    _ROLLING_CVAR,
    _DRAWDOWN,
    _DIFFERENTIAL_SHARPE,
)

_C1_GRID = (0.05, 0.1, 0.5, 1.0, 2.0)
_WINDOW_GRID = (20, 60)
_ALPHA_GRID = (0.01, 0.05, 0.10)
_ETA_GRID = (0.01, 0.05, 0.1)


class StubDesignerTransport:
    """A deterministic ``(system, user) -> reward_source`` transport for keyless Pass A.

    Cycles through :data:`ARCHETYPES`, filling each with coefficients from a PER-CALL-INDEX
    child generator ``default_rng((seed, n))`` — call ``n`` depends ONLY on ``(seed, n)``, never
    on the draw history (2026-07-06, crash-rehearsal catch: the old single sequential stream
    meant a resume that REPLAYED archived candidates without consuming their draws shifted every
    later candidate's coefficients, so a killed+resumed Pass-A run was not byte-faithful to an
    uninterrupted one). Replay paths keep the index aligned via :meth:`advance` — one position
    per replayed (authored) candidate, mirroring the search arms' draw-unconditionally rule.
    It records every call like ``FakeTransport`` so tests can assert on what was sent.

    Parameters
    ----------
    seed : int
        Seeds every per-index child generator (the candidate at index ``n`` is a pure function
        of ``(seed, n)``).
    """

    def __init__(self, seed: int = 0) -> None:
        self._seed = int(seed)
        self._n = 0
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        n = self._n
        self._n += 1
        template = ARCHETYPES[n % len(ARCHETYPES)]
        rng = np.random.default_rng((self._seed, n))  # pure f(seed, index): any-subset replay-safe
        c1 = float(rng.choice(_C1_GRID))
        window = int(rng.choice(_WINDOW_GRID))
        alpha = float(rng.choice(_ALPHA_GRID))
        eta = float(rng.choice(_ETA_GRID))
        return (
            template.replace("__C1__", repr(c1))
            .replace("__WINDOW__", str(window))
            .replace("__ALPHA__", repr(alpha))
            .replace("__ETA__", repr(eta))
        )

    def advance(self, k: int = 1) -> None:
        """Consume ``k`` author-stream positions WITHOUT generating (resume replay).

        A replayed candidate — archived success or ledgered sandbox failure — was AUTHORED in the
        original run, so it owns one stream position; skipping it here keeps every post-resume
        fresh candidate at the same index (hence the same output) it would have had uninterrupted.
        Real LLM transports deliberately have no ``advance`` (a paid, non-deterministic call
        cannot be replayed-by-position); callers duck-type on the attribute."""
        self._n += max(0, int(k))

    def reset(self) -> None:
        """Reset the call-index counter to 0."""
        self._n = 0
