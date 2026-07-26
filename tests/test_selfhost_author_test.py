"""A5 authoring-reliability harness: the scorer (executable-code detection reusing the executor's
extractor + thinking-off round-trip) and the aggregator (reliability + thinking-off rates). Off-GPU:
a fake transport stands in for the served endpoint."""
from scripts.selfhost_author_test import run_selfhost_author_test, score_response

_GOOD = "Here you go:\n```python\nimport numpy as np\ndef reward(returns):\n    return float(np.mean(returns))\n```\n"
_THINKING = "<think>I should reason first</think>\n```python\ndef reward(returns):\n    return 0.0\n```\n"
_EMPTY = "I'm sorry, I can't produce that."


def test_score_response_flags_executable_code_and_thinking_off() -> None:
    s = score_response(_GOOD)
    assert s["has_code"] and s["defines_reward"] and s["no_thinking"] and s["compliant"]


def test_score_response_catches_a_leaked_thinking_block() -> None:
    s = score_response(_THINKING)
    assert s["compliant"] is True          # the code is still there...
    assert s["no_thinking"] is False       # ...but thinking leaked -> the R103 round-trip FAILS


def test_score_response_non_authoring_is_not_compliant() -> None:
    s = score_response(_EMPTY)
    # extract_reward_source is lenient (like leg_gates), so has_code may be True; the real
    # "no reward function authored" signal is defines_reward -> not compliant.
    assert s["defines_reward"] is False and s["compliant"] is False and s["no_thinking"] is True


class _Fake:
    """A stand-in transport: returns canned author responses in order."""
    def __init__(self, responses):
        self._r = list(responses)
        self._i = 0

    def __call__(self, system: str, user: str) -> str:
        out = self._r[self._i % len(self._r)]
        self._i += 1
        return out


def test_run_aggregates_reliability_and_thinking_off_rates() -> None:
    t = _Fake([_GOOD, _GOOD, _EMPTY, _THINKING])
    r = run_selfhost_author_test(t, "sys", "user", 4)
    assert r["n"] == 4
    assert abs(r["compliance_rate"] - 0.75) < 1e-9      # GOOD, GOOD, THINKING compliant; EMPTY not
    assert abs(r["thinking_off_rate"] - 0.75) < 1e-9    # only THINKING leaks a <think> block


def test_run_prefers_archived_reasoning_tokens_when_available() -> None:
    t = _Fake([_GOOD, _GOOD])
    # a served reasoning_tokens>0 overrides the no-<think> heuristic -> thinking NOT off
    r = run_selfhost_author_test(t, "s", "u", 2, reasoning_tokens_of=lambda: 5)
    assert r["thinking_off_rate"] == 0.0
    r2 = run_selfhost_author_test(t, "s", "u", 2, reasoning_tokens_of=lambda: 0)
    assert r2["thinking_off_rate"] == 1.0
