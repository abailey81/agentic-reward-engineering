"""Behaviour tests for the v2 leg LAUNCH wiring (R80/R82) + the R83 per-call author ledger.

What must hold:
* ``resolve_leg_override`` builds the llm-block override from the SAME ``transport_kwargs``
  translation the pre-launch gates use (pins survive; anthropic legs carry no extra_body), forces
  the sanitized ``leg_<label>`` namespace, and refuses a conflicting explicit suffix.
* ``build_parallel_opts`` threads ``extra_body`` + ``spend_ledger`` from the llm block to opts —
  without this a leg's registered provider/quantization/reasoning pins were SILENTLY DROPPED.
* ``_build_cluster_author`` hands ``extra_body`` to ``build_transport`` (pin survival end-to-end).
* ``LLMClient`` records per-call spend to the R83 advisory ledger ONLY when the transport surfaces
  real metadata (realized cost, else tokens×planning-prices estimate); fakes/stubs stay
  ledger-silent; a ledger failure NEVER breaks a paid call.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_campaign_cluster as RCC  # noqa: E402
from run_prototype import build_parallel_opts  # noqa: E402

from src.llm.client import LLMClient  # noqa: E402
from src.llm.spend_ledger import estimate_cost_usd  # noqa: E402


# ---- resolve_leg_override --------------------------------------------------------------------------- #
def test_openrouter_leg_carries_pins_and_forced_suffix() -> None:
    llm_cfg, provider, suffix = RCC.resolve_leg_override("qwen3.6-27b", None)
    assert provider == "openrouter"
    assert llm_cfg["model_snapshot"] == "qwen/qwen3.6-27b"
    assert llm_cfg["pass"] == "B"
    assert llm_cfg["temperature"] == 1.0        # R85 uniform decoding pin on OpenRouter legs
    assert llm_cfg["diversity_prompt_variation"] is True
    eb = llm_cfg["extra_body"]
    assert eb["provider"] == {"only": ["siliconflow"], "allow_fallbacks": False,
                              "quantizations": ["fp8"]}
    assert eb["usage"] == {"include": True}                      # the R83 cost request rides along
    assert suffix == "leg_qwen3_6_27b"                            # sanitized to the suffix grammar


def test_anthropic_leg_has_no_extra_body() -> None:
    llm_cfg, provider, suffix = RCC.resolve_leg_override("haiku-4.5", None)
    assert provider == "anthropic"
    assert "extra_body" not in llm_cfg                            # native transport: id + caps only
    assert llm_cfg["temperature"] is None                          # R85: Anthropic legs carry no temp pin
    assert llm_cfg["model_snapshot"] == "claude-haiku-4-5-20251001"
    assert suffix == "leg_haiku_4_5"


def test_unknown_leg_fails_loud_and_conflicting_suffix_refused() -> None:
    with pytest.raises(KeyError, match="known legs"):
        RCC.resolve_leg_override("gpt-99-imaginary", None)
    with pytest.raises(SystemExit, match="conflicts"):
        RCC.resolve_leg_override("glm-5.2", "my_custom_suffix")
    # A MATCHING explicit suffix is fine (idempotent relaunch lines).
    _, _, suffix = RCC.resolve_leg_override("glm-5.2", "leg_glm_5_2")
    assert suffix == "leg_glm_5_2"


# ---- opts threading --------------------------------------------------------------------------------- #
def test_build_parallel_opts_threads_extra_body_and_ledger() -> None:
    blk = {"model_snapshot": "x/y", "api_key_env": "K", "extra_body": {"reasoning": {"effort": "low"}}}
    opts = build_parallel_opts({}, {}, llm_block=blk, train_steps=100, n_trials=1, synthetic=True,
                               seed=0, candidates=5, generations=1, pass_mode="B",
                               provider="openrouter", resume=False)
    assert opts["extra_body"] == {"reasoning": {"effort": "low"}}
    assert opts["spend_ledger"] == "outputs/spend_ledger.jsonl"   # default ON for campaign paths
    # v1 blocks without the key stay byte-identical in behaviour: extra_body None.
    opts_v1 = build_parallel_opts({}, {}, llm_block={"model_snapshot": "m"}, train_steps=100,
                                  n_trials=1, synthetic=True, seed=0, candidates=5, generations=1,
                                  pass_mode="B", provider="anthropic", resume=False)
    assert opts_v1["extra_body"] is None


def test_cluster_author_passes_extra_body_to_build_transport(monkeypatch, tmp_path) -> None:
    from src.cluster import campaign as CC

    received: dict = {}

    def fake_build_transport(provider, model, key_env=None, **kw):
        received.update({"provider": provider, "model": model, **kw})
        return lambda s, u: "def reward(): pass"

    monkeypatch.setattr("src.llm.client.build_transport", fake_build_transport)
    monkeypatch.setattr("src.llm.prompts.build_prompt_set", lambda env, n: {"system": "s"})
    opts = {"pass_mode": "B", "provider": "openrouter", "model": "qwen/qwen3.6-27b",
            "api_key_env": "OPENROUTER_API_KEY", "temperature": None, "max_tokens": 4096,
            "max_retries": 6, "seed": 0, "diversity_prompt_variation": True,
            "extra_body": {"provider": {"only": ["siliconflow"]}}, "env_cfg": {}, "n_assets": 30,
            "spend_ledger": None}
    CC._build_cluster_author("distributional", opts, tmp_path)
    assert received["extra_body"] == {"provider": {"only": ["siliconflow"]}}  # pins SURVIVE


# ---- the R83 per-call author ledger ----------------------------------------------------------------- #
class _MetaTransport:
    """A fake surfacing the real-transport metadata attributes."""

    def __init__(self, cost=None, usage=None):
        self.last_cost_usd = cost
        self.last_usage = usage

    def __call__(self, system, user):
        return "ok"


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def test_realized_cost_recorded_per_call(tmp_path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    llm = LLMClient({"model": "qwen/qwen3.6-27b", "provider": "openrouter",
                     "spend_ledger": str(ledger)},
                    transport=_MetaTransport(cost=0.0123, usage={"input_tokens": 10,
                                                                 "output_tokens": 20}))
    llm.complete("s", "u")
    (row,) = _rows(ledger)
    assert row["cost_usd"] == 0.0123 and row["note"] == "realized"
    assert row["provider"] == "openrouter" and row["tokens_out"] == 20


def test_estimated_cost_when_provider_returns_none(tmp_path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    llm = LLMClient({"model": "claude-opus-4-8", "provider": "anthropic",
                     "spend_ledger": str(ledger)},
                    transport=_MetaTransport(cost=None, usage={"input_tokens": 1_000_000,
                                                               "output_tokens": 0}))
    llm.complete("s", "u")
    (row,) = _rows(ledger)
    assert row["note"] == "estimated-from-planning-prices"
    assert row["cost_usd"] == pytest.approx(5.00)                 # $5/MTok in (legs.yaml planning row)


def test_fake_transports_stay_ledger_silent(tmp_path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    llm = LLMClient({"model": "m", "spend_ledger": str(ledger)},
                    transport=lambda s, u: "ok")                  # no metadata attributes at all
    llm.complete("s", "u")
    assert _rows(ledger) == []                                    # nothing truthful to record
    # And with NO ledger configured, a metadata transport records nowhere (bare-client default).
    llm2 = LLMClient({"model": "m"}, transport=_MetaTransport(cost=1.0))
    llm2.complete("s", "u")
    assert _rows(ledger) == []


def test_ledger_failure_never_breaks_the_call(tmp_path, monkeypatch) -> None:
    llm = LLMClient({"model": "m", "spend_ledger": str(tmp_path / "l.jsonl")},
                    transport=_MetaTransport(cost=1.0))
    monkeypatch.setattr("src.llm.spend_ledger.record_spend",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    assert llm.complete("s", "u") == "ok"                         # the paid call survives


# ---- the author call sites must THREAD the provider, not just be capable of recording it -------------- #
# 2026-07-28, found by reading RUNs 1-3's ledgers rather than the code: every one of the 1,361 rows was
# stamped `provider: "anthropic"`, including rows whose `model` is plainly an OpenRouter id
# (`deepseek/deepseek-v4-pro`, `google/gemini-2.5-flash`, `qwen/qwen3.5-9b`, `z-ai/glm-5.2`, ...).
#
# ROUTING WAS NEVER WRONG — `build_transport` is called with the real `opts["provider"]`, so the legs
# genuinely reached OpenRouter and no recorded result is affected. What was wrong is the LABEL: both
# production authors constructed `LLMClient({"model": ..., "spend_ledger": ...})` with no `provider`
# key, so `client.py`'s `cfg_get(cfg, "provider", "anthropic")` DEFAULT was stamped onto every row.
#
# The defect survived because `test_realized_cost_recorded_per_call` above passes `provider` in
# explicitly: it proves the client records whatever it is GIVEN, while production gave it nothing.
# That is the gap these two tests close — one behavioural, one structural across both call sites.
def test_cluster_author_stamps_the_real_provider_not_the_anthropic_default(monkeypatch, tmp_path) -> None:
    """`_build_cluster_author` must hand the LEG's provider to the client, not leave the default."""
    from src.cluster import campaign as CC

    monkeypatch.setattr("src.llm.client.build_transport",
                        lambda provider, model, key_env=None, **kw: _MetaTransport(
                            cost=0.5, usage={"input_tokens": 3, "output_tokens": 4}))
    monkeypatch.setattr("src.llm.prompts.build_prompt_set", lambda env, n: {"system": "s"})

    ledger = tmp_path / "spend.jsonl"
    opts = {"pass_mode": "B", "provider": "openrouter", "model": "deepseek/deepseek-v4-pro",
            "api_key_env": "OPENROUTER_API_KEY", "temperature": None, "max_tokens": 4096,
            "max_retries": 6, "seed": 0, "diversity_prompt_variation": False,
            "extra_body": None, "env_cfg": {}, "n_assets": 30, "spend_ledger": str(ledger)}
    llm, _, _ = CC._build_cluster_author("distributional", opts, tmp_path)

    assert llm.provider == "openrouter", (
        f"cluster author stamped provider={llm.provider!r} for an OpenRouter leg — the "
        "spend ledger will mis-attribute the cost to Anthropic"
    )
    llm.complete("s", "u")
    (row,) = _rows(ledger)
    assert row["provider"] == "openrouter" and row["model"] == "deepseek/deepseek-v4-pro"


def test_both_production_authors_pass_provider_into_the_client_cfg() -> None:
    """Structural lock across BOTH author call sites.

    The behavioural test above can only reach one of them; `parallel._drive_llm_arm` runs a whole
    arm and is far too heavy to instantiate here. So assert on the AST instead: every `LLMClient(`
    construction in the two production authors must pass a cfg mapping containing "provider".
    Checking the parsed syntax rather than a source substring keeps it robust to formatting.
    """
    import ast

    repo = Path(__file__).resolve().parents[1]
    sites = [repo / "src" / "cluster" / "campaign.py",
             repo / "src" / "orchestration" / "parallel.py"]

    checked = 0
    for path in sites:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "LLMClient"):
                continue
            assert node.args, f"{path.name}:{node.lineno}: LLMClient called with no cfg argument"
            cfg = node.args[0]
            assert isinstance(cfg, ast.Dict), (
                f"{path.name}:{node.lineno}: expected a literal cfg dict to inspect"
            )
            keys = {k.value for k in cfg.keys if isinstance(k, ast.Constant)}
            assert "provider" in keys, (
                f"{path.name}:{node.lineno}: LLMClient cfg omits 'provider' {sorted(keys)} — "
                "client.py falls back to its 'anthropic' default and every ledger row for a "
                "non-Anthropic leg is mis-attributed"
            )
            checked += 1
    assert checked == 2, f"expected exactly 2 production LLMClient sites, found {checked}"


def test_estimate_cost_math_and_unpriced_none() -> None:
    assert estimate_cost_usd("claude-opus-4-8", 100_000, 10_000) == pytest.approx(
        0.1 * 5.00 + 0.01 * 25.00)
    assert estimate_cost_usd("claude-opus-5", 100_000, 10_000) == pytest.approx(  # R102: confirmatory author priced
        0.1 * 5.00 + 0.01 * 25.00)
    assert estimate_cost_usd("never/priced-model", 100, 100) is None
    assert estimate_cost_usd("claude-opus-4-8", None, None) is None
