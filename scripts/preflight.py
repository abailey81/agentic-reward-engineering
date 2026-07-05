#!/usr/bin/env python3
"""Pre-flight gauntlet — read-only GO/NO-GO checks before committing to the ~2-week campaign.

Each probe returns a ``Check(name, status, detail)``. A hard ``FAIL`` makes the script exit non-zero so a
supervisor or operator gates the run; ``WARN`` is advisory; ``PASS`` is green. NOTHING here mutates state —
the optional ``--probe`` is opt-in and the ONLY network call/spend (a REAL 1-token Anthropic liveness call,
M9: a dead key or empty credit must fail at t0, not at hour 3). The decision logic lives in pure functions
(unit-tested in ``tests/test_preflight.py``); ``main`` only gathers the measured inputs.

Run::

    python scripts/preflight.py --gpu 2            # checks for an n_gpu=2 campaign
    python scripts/preflight.py --gpu 3 --min-disk-gb 5 --probe
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"

# Per-worker RAM (replay buffer ~0.76 GB at a 50k cap + ~1.4 GB torch/SB3/panel/CUDA base) and the per-worker
# VRAM CUDA-context+model footprint — calibrated to the measured 2026-06-20 ceiling (parallel.py::max_power_config).
_PER_WORKER_GB = 2.2
_VRAM_CTX_GB = 1.4
_OS_RESERVE_GB = 3.0


@dataclass(frozen=True)
class Check:
    name: str
    status: str  # PASS | WARN | FAIL
    detail: str


# ── Pure decision logic (unit-tested) ─────────────────────────────────────────────────────────────────
def check_disk(free_gb: float, min_gb: float) -> Check:
    if free_gb < min_gb:
        return Check("disk", FAIL, f"only {free_gb:.1f} GB free; need >= {min_gb:.1f} GB")
    return Check("disk", PASS, f"{free_gb:.1f} GB free (>= {min_gb:.1f})")


def check_ram(total_gb: float, avail_gb: float, n_gpu: int) -> Check:
    need = n_gpu * _PER_WORKER_GB + _OS_RESERVE_GB
    if avail_gb < need:
        return Check("ram", FAIL, f"{avail_gb:.1f} GB available; n_gpu={n_gpu} needs ~{need:.1f} GB "
                                  f"({n_gpu}x{_PER_WORKER_GB}+{_OS_RESERVE_GB} reserve)")
    if total_gb < 15.0 and n_gpu >= 3:
        return Check("ram", WARN, f"{total_gb:.1f} GB total + n_gpu=3 is the measured creep-to-OOM band; "
                                  f"prefer n_gpu=2 or arm recycling")
    return Check("ram", PASS, f"{avail_gb:.1f} GB available, n_gpu={n_gpu} needs ~{need:.1f} GB")


def check_commit_headroom(avail_gb: float | None, n_gpu: int | None = None,
                          min_gb: float | None = None) -> Check:
    """Windows COMMIT-CHARGE headroom (2026-07-18 forensics): with system commit exhausted
    (the 8-day ArmouryCrate.UserSessionHelper leak held 7.6 GB; avail fell to 0.37 GB), every
    spawned process stalls for MINUTES loading the numpy/MKL DLLs while Windows grows the
    pagefile — validation children then miss even generous graces and the driver wedges.
    RAM-available does NOT catch this (commit is RAM+pagefile, capped by C: free space).

    CALIBRATED 2026-07-27 to what the development laptop can give. The floor was a
    HARDCODED 6.0 regardless of workload, while the sibling :func:`check_ram` scales as
    ``n_gpu * 2.2 + 3.0``. That mattered once the campaign moved to Myriad: on a cluster run the
    laptop TRAINS NOTHING — it only drives (submit/poll/pull) and authors — so RAM correctly relaxed
    to 3.0 GB while commit still demanded 6.0, failing the gauntlet on a workload that does not exist.

    MEASURED on this box, sampling the whole process tree at 20 Hz from outside:
      * cluster driver (dry-run, core AND a leg): peak **0.54 GB**, 2 processes
      * spawned validation child (the class the incident actually hurt): peak **1.33 GB**, 4 processes
      * freeze-gate child: 0.02 GB
    So the laptop's driver role needs ~1.3 GB at peak, and 6.0 was ~4.5x that.

    The floor now uses the SAME shape as ``check_ram`` so the two can never disagree about the
    workload again. Note this makes the check **STRICTER where the laptop really does train**
    (n_gpu=2 -> 7.4 GB, up from the flat 6.0) and honest where it does not (n_gpu=0 -> 3.0 GB, still
    a 2.25x margin over the measured 1.33 GB peak). ``min_gb`` remains overridable for callers that
    want the legacy flat floor.
    """
    if avail_gb is None:
        return Check("commit", WARN, "commit-charge telemetry unavailable (non-Windows?)")
    if min_gb is None:
        min_gb = (n_gpu * _PER_WORKER_GB + _OS_RESERVE_GB) if n_gpu is not None else 6.0
    if avail_gb < min_gb:
        return Check("commit", FAIL,
                     f"only {avail_gb:.1f} GB system commit available (< {min_gb:.1f} for "
                     f"n_gpu={n_gpu}); child spawns will stall — close other apps, find the leaker "
                     f"(Get-Process | sort PrivateMemorySize64; the known one is "
                     f"ArmouryCrate.UserSessionHelper), and/or grow the pagefile")
    return Check("commit", PASS,
                 f"{avail_gb:.1f} GB system commit available (>= {min_gb:.1f} for n_gpu={n_gpu}; "
                 "measured driver peak 1.3 GB on the cluster path)")


def check_vram(free_mb: float | None, n_gpu: int) -> Check:
    if free_mb is None:
        return Check("vram", WARN, "no CUDA / NVML telemetry — cannot verify VRAM headroom (CPU run?)")
    need_gb = n_gpu * _VRAM_CTX_GB
    free_gb = free_mb / 1024.0
    if n_gpu >= 4:
        return Check("vram", FAIL, f"n_gpu={n_gpu} OOMs the 6 GB RTX-4050 (measured); cap at 3")
    if free_gb < need_gb:
        return Check("vram", FAIL, f"{free_gb:.1f} GB VRAM free; n_gpu={n_gpu} needs ~{need_gb:.1f} GB")
    return Check("vram", PASS, f"{free_gb:.1f} GB VRAM free, n_gpu={n_gpu} needs ~{need_gb:.1f} GB")


def check_api_key(present: bool, key_name: str, *, required: bool) -> Check:
    if not required:
        return Check("api_key", PASS, "stub/offline provider — no API key required")
    if not present:
        return Check("api_key", FAIL, f"environment variable {key_name} is unset/empty (paid provider)")
    return Check("api_key", PASS, f"{key_name} is set")


def check_retry_layer(installed: bool) -> Check:
    """C1 (ops audit 2026-07-02): tenacity is the SINGLE retry/backoff layer for every LLM call.

    ``src/llm/client.py::_make_retrying`` degrades GRACEFULLY when tenacity is not importable — the
    transport still works, but every API call becomes SINGLE-ATTEMPT with no backoff, so the first
    transient 429/5xx/timeout kills a candidate and a sustained blip kills the arm: the #1 run-killer
    for a multi-week unattended run. That graceful degradation is right for the library and WRONG for
    launch day, so the gauntlet turns it into a hard FAIL here.
    """
    if not installed:
        return Check("retry_layer", FAIL,
                     "tenacity is NOT importable — every API call would be single-attempt (no backoff "
                     "on 429/5xx/timeouts), the #1 run-killer; `pip install tenacity` into the venv")
    return Check("retry_layer", PASS, "tenacity importable — bounded exponential backoff active (ADR-034)")


def check_windows_update(pending_reboot: bool | None, updates_paused: bool | None) -> Check:
    """C2 (ops audit 2026-07-02): Windows Update is the biggest EXOGENOUS interrupt of a 2-3-week run.

    ``pending_reboot=True`` -> FAIL: the OS already has a reboot queued, and it WILL take the run down
    at an arbitrary point — reboot deliberately BEFORE launch (then re-run this gauntlet). Updates not
    verifiably PAUSED -> WARN: a forced patch cycle mid-run reboots the host; pause updates for the
    full run window (~5 weeks) and register the ONSTART task (scripts/install_onstart_task.ps1) so an
    unavoidable reboot re-enters via --resume. ``None`` inputs mean the registry probe was unavailable
    (non-Windows host / access failure) -> advisory WARN, never a hard fail on a probe gap.
    """
    if pending_reboot is True:
        return Check("windows_update", FAIL,
                     "a REBOOT IS PENDING (RebootRequired/RebootPending registry keys) — the OS will "
                     "restart mid-run; reboot NOW, then re-run preflight")
    if pending_reboot is None and updates_paused is None:
        return Check("windows_update", WARN, "probe unavailable (non-Windows host or registry access failed)")
    if updates_paused is not True:
        state = "NOT paused" if updates_paused is False else "pause state unknown"
        return Check("windows_update", WARN,
                     f"no reboot pending, but Windows Updates are {state} — a forced patch reboot can "
                     f"interrupt the run; pause updates ~5 weeks + register the ONSTART re-entry task")
    return Check("windows_update", PASS, "no reboot pending; updates paused past a future expiry")


def check_api_probe(outcome: str, detail: str) -> Check:
    """M9 (ops audit 2026-07-02): classify the LIVE 1-token probe — a dead key must fail at t0, not hour 3.

    ``outcome``: ``ok`` -> PASS; ``auth_error`` (401/403 — expired/revoked key) and ``client_error``
    (any other deterministic 4xx — e.g. an exhausted credit balance's 400, a wrong model id's 404;
    EVERY campaign call would fail identically) -> FAIL; ``transient`` (connection/timeout/429/5xx —
    tenacity absorbs these at runtime) and ``unavailable`` (SDK/config missing) -> advisory WARN.
    """
    if outcome == "ok":
        return Check("api_probe", PASS, f"live 1-token call succeeded — {detail}")
    if outcome == "auth_error":
        return Check("api_probe", FAIL,
                     f"AUTH failure — the key is invalid/expired/revoked; fix .env before launch ({detail})")
    if outcome == "client_error":
        return Check("api_probe", FAIL,
                     f"deterministic 4xx — every campaign call would fail the same way "
                     f"(empty credit? wrong model id?) ({detail})")
    if outcome == "transient":
        return Check("api_probe", WARN, f"transient API failure (retried at runtime by tenacity): {detail}")
    return Check("api_probe", WARN, f"probe unavailable: {detail}")


def check_freeze(frozen: bool, hash_match: bool | None) -> Check:
    if not frozen:
        return Check("freeze", FAIL, "preregistration.yaml frozen=false — freeze the design before the run")
    if hash_match is False:
        return Check("freeze", FAIL, "recorded freeze hash != freshly recomputed hash (design drifted)")
    if hash_match is None:
        return Check("freeze", WARN, "could not recompute freeze hash to cross-check (frozen flag is true)")
    return Check("freeze", PASS, "design frozen and hash matches")


def check_data(path_exists: bool, checksum_ok: bool | None, path: str) -> Check:
    if not path_exists:
        return Check("data", FAIL, f"gold panel not found at {path}")
    if checksum_ok is False:
        return Check("data", FAIL, f"gold panel checksum mismatch at {path} (corrupted / wrong file)")
    if checksum_ok is None:
        return Check("data", WARN, f"gold panel present at {path}; checksum not verified")
    return Check("data", PASS, f"gold panel present and checksum verified ({path})")


def verdict(checks: list[Check]) -> str:
    """GO/NO-GO roll-up: any FAIL -> NO-GO; any WARN -> GO-WITH-WARNINGS; else GO."""
    if any(c.status == FAIL for c in checks):
        return "NO-GO"
    if any(c.status == WARN for c in checks):
        return "GO-WITH-WARNINGS"
    return "GO"


def check_budget_mirror(campaign_steps: object, algos_steps: object) -> Check:
    """Audit H-M1 (2026-07-02): ``train_steps_per_candidate`` is authored in BOTH config/campaign.yaml (the
    executed source of truth — run_campaign reads ONLY this) and config/algos.yaml (consulted by the design
    tooling: determine_design / smoke_test / the factory fallback). A B* amendment that edits one file but
    not the other leaves a stale mirror the tooling would silently trust — FAIL so the operator reconciles
    both in the same amendment."""
    if campaign_steps is None or algos_steps is None:
        return Check("budget_mirror", WARN,
                     "could not read train_steps_per_candidate from both campaign.yaml and algos.yaml")
    if int(campaign_steps) != int(algos_steps):  # type: ignore[arg-type]
        return Check("budget_mirror", FAIL,
                     f"train_steps_per_candidate mismatch: campaign.yaml={campaign_steps} != "
                     f"algos.yaml={algos_steps} — update BOTH in the B* amendment "
                     "(campaign.yaml is what the campaign executes)")
    return Check("budget_mirror", PASS,
                 f"train_steps_per_candidate consistent across campaign/algos: {campaign_steps}")


def check_model_consistency(campaign_model: object, llm_model: object) -> Check:
    """Batch-6 M4 (2026-07-03): the campaign reward-author model is authored TWICE — config/campaign.yaml
    ``llm.model_snapshot`` (the executed snapshot run_campaign reads) and config/llm.yaml
    ``model_snapshot`` (the standalone mirror). Both are common-mode across arms (not a treatment → the
    arm-contrast identification is untouched) and the served snapshot is archived at first live call as the
    reproducibility anchor, so the model is deliberately NOT hash-bound. But a partial edit (swap one file,
    forget the other) would leave a stale mirror the design tooling trusts — exactly the budget-mirror
    failure mode. FAIL so the operator reconciles both; this is infra consistency, not a frozen commitment."""
    if not campaign_model or not llm_model:
        return Check("model_mirror", WARN,
                     "could not read model_snapshot from both campaign.yaml (llm block) and llm.yaml")
    if str(campaign_model) != str(llm_model):
        return Check("model_mirror", FAIL,
                     f"model_snapshot mismatch: campaign.yaml llm.model_snapshot={campaign_model!r} != "
                     f"llm.yaml model_snapshot={llm_model!r} — reconcile BOTH (campaign.yaml is what the "
                     "campaign executes; llm.yaml is the standalone mirror)")
    return Check("model_mirror", PASS,
                 f"model_snapshot consistent across campaign/llm: {campaign_model!r}")


def check_author_pin_mirror(campaign_llm: dict, llm_yaml: dict) -> Check:
    """R106 (2026-07-27): the confirmatory author's CAP and REASONING PIN must be present and agree.

    Found the hard way. ``config/campaign.yaml``'s ``llm`` block is what the campaign EXECUTES
    (``run_campaign_cluster.py`` reads it into ``llm_cfg``); ``config/llm.yaml`` has no code consumer
    for the author. R102 raised the Opus cap 4096 -> 8192 and recorded it **only in llm.yaml**, so
    ``campaign.py`` fell back to its 4096 default and the confirmatory author ran at a value the
    registration did not state — silently, since ``check_model_consistency`` mirrors the model id and
    nothing mirrored the cap. Same shape as the budget-mirror failure that check already guards.

    ABSENCE is a FAIL, not a warning: an absent key does not mean "default", it means the executed
    value is invisible to the registration. The pin is likewise required — R106 is uniform
    reasoning-off across all 11 models, and the author is the 11th.
    """
    c_tok, l_tok = campaign_llm.get("max_tokens"), llm_yaml.get("max_tokens")
    c_think, l_think = campaign_llm.get("thinking"), llm_yaml.get("thinking")
    if c_tok is None:
        return Check("author_pin_mirror", FAIL,
                     "campaign.yaml llm.max_tokens is ABSENT — the author silently runs on the code "
                     "default (4096), not on any registered value; set it (R106: 8192)")
    if c_think is None:
        return Check("author_pin_mirror", FAIL,
                     "campaign.yaml llm.thinking is ABSENT — the confirmatory author runs on the "
                     "vendor default, unpinned (R106 requires {type: disabled} on all 11 models)")
    if l_tok is not None and int(c_tok) != int(l_tok):
        return Check("author_pin_mirror", FAIL,
                     f"author max_tokens mismatch: campaign.yaml={c_tok!r} != llm.yaml={l_tok!r} — "
                     "reconcile BOTH (campaign.yaml is what executes)")
    if l_think is not None and dict(c_think) != dict(l_think):
        return Check("author_pin_mirror", FAIL,
                     f"author thinking mismatch: campaign.yaml={c_think!r} != llm.yaml={l_think!r}")
    return Check("author_pin_mirror", PASS,
                 f"author cap+pin registered and mirrored: max_tokens={c_tok}, thinking={c_think}")


def check_generations_mirror(
    campaign_generations: object, llm_generations: object, h3_generations: object
) -> Check:
    """2026-07-05 (map M04): the reflection depth is DESIGN-DEFINING (PREREG R30 pins generations=6 and
    the H3 iterative-vs-single-shot contrast IS 6-vs-1) yet lives only in un-hashed compute configs,
    authored twice (campaign.yaml ``llm.generations`` — what run_campaign executes — and llm.yaml
    ``generations``). A key-drop silently runs EVERY LLM arm single-shot (``cfg_get`` default 1) and
    destroys both the iterative protocol and H3. Same infra-mirror idiom as the model/budget guards;
    the frozen prereg-mirror key rides the seed-ratification amendment (one batched hash move)."""
    if campaign_generations is None or llm_generations is None:
        return Check("generations_mirror", FAIL,
                     "generations missing from campaign.yaml (llm block) and/or llm.yaml — a dropped key "
                     "silently runs every LLM arm single-shot (cfg_get default 1); restore generations: 6")
    if int(campaign_generations) != int(llm_generations):
        return Check("generations_mirror", FAIL,
                     f"generations mismatch: campaign.yaml llm.generations={campaign_generations!r} != "
                     f"llm.yaml generations={llm_generations!r} — reconcile BOTH")
    if h3_generations is not None and int(h3_generations) != 1:
        return Check("generations_mirror", FAIL,
                     f"h3_singleshot_generations={h3_generations!r} != 1 — the H3 control's whole point "
                     "is a single-generation best-of-N search (DEEP_H3 §1)")
    return Check("generations_mirror", PASS,
                 f"generations consistent across campaign/llm ({int(campaign_generations)}); "
                 f"h3_singleshot_generations={h3_generations!r}")


#: FIX-class agent-key typo guard (2026-07-03). Every key the training stack actually CONSUMES from a
#: config ``agent:`` block — enumerated from the REAL consumers, not from the configs. ``cfg_get``
#: returns a silent default for an unknown key, so a typo'd frozen key (e.g. ``learning_start:``) would
#: freeze into the hash while every run silently trains on the default; this list makes that a hard FAIL.
#: Consumers (keep in sync when one grows a key):
#:   * src/agents/trainer.py::resolve_agent_kwargs — train_steps_per_candidate, train_steps,
#:     buffer_size, batch_size, learning_starts, learning_rate, gamma, ent_coef, verbose, device;
#:   * src/agents/trainer.py::train_agent — tf32, normalize_obs, clip_obs; ::_make_governor —
#:     thermal_guardian; src/agents/popart.py::wrap_popart — popart, popart_beta, popart_min_scale,
#:     popart_warmup;
#:   * src/agents/factory.py::_policy_kwargs — policy, seed, policy_kwargs (net_arch etc. live INSIDE
#:     policy_kwargs, where the SB3 constructor fails loud on an unknown key — so net_arch is
#:     deliberately NOT allowed at the agent-block level, where it would be silently ignored);
#:   * src/agents/factory.py::make_distributional_agent (TQC secondary) —
#:     top_quantiles_to_drop_per_net, n_quantiles, n_critics;
#:   * ``algo`` — a documented CHOICE marker in prototype.yaml read by no code (config/algos.yaml
#:     ``headline_agent`` is the executed choice); allowed so the existing config stays green.
ALLOWED_AGENT_KEYS: frozenset[str] = frozenset({
    "algo",
    "batch_size",
    "buffer_size",
    "clip_obs",
    "device",
    "ent_coef",
    "gamma",
    "learning_rate",
    "learning_starts",
    "n_critics",
    "n_quantiles",
    "normalize_obs",
    "policy",
    "policy_kwargs",
    "popart",
    "popart_beta",
    "popart_min_scale",
    "popart_warmup",
    "seed",
    "thermal_guardian",
    "tf32",
    "top_quantiles_to_drop_per_net",
    "train_steps",
    "train_steps_per_candidate",
    "verbose",
})


def check_agent_keys(blocks: dict[str, object]) -> Check:
    """FAIL on any ``agent:`` key no consumer reads (the cfg_get silent-default typo trap, 2026-07-03).

    ``blocks`` maps a config label (e.g. ``"campaign.yaml"``) to its parsed ``agent`` block. BOTH the
    campaign and prototype blocks matter to the campaign: run_campaign bases ``agent_cfg`` on the
    prototype block and overlays the campaign block verbatim (run_campaign.py ~L1363-1366), so a typo
    in EITHER silently defaults into the frozen run. A missing/None block is fine (nothing to typo);
    a non-mapping block FAILs (a scalar ``agent:`` value means the YAML structure itself is broken).
    """
    unknown: list[str] = []
    malformed: list[str] = []
    for name, block in blocks.items():
        if block is None:
            continue
        if not isinstance(block, dict):
            malformed.append(f"{name} (agent block is {type(block).__name__}, expected mapping)")
            continue
        bad = sorted(str(k) for k in block if str(k) not in ALLOWED_AGENT_KEYS)
        if bad:
            unknown.append(f"{name}: {', '.join(bad)}")
    if malformed:
        return Check("agent_keys", FAIL, "malformed agent block — " + "; ".join(malformed))
    if unknown:
        return Check("agent_keys", FAIL,
                     "unknown agent key(s) — cfg_get would silently default them while the typo "
                     "freezes into the hash: " + "; ".join(unknown)
                     + " (allowed = the keys resolve_agent_kwargs/factory/train_agent consume; "
                       "see ALLOWED_AGENT_KEYS)")
    return Check("agent_keys", PASS,
                 f"agent blocks carry only consumed keys ({', '.join(sorted(blocks)) or 'none found'})")


# ── Measurement + driver (thin, guarded) ──────────────────────────────────────────────────────────────
def _pause_expiry_is_future(value: str, now_epoch: float) -> bool | None:
    """Parse the registry ``PauseUpdatesExpiryTime`` (ISO-8601, usually ``...Z``): paused iff it is in
    the future. ``None`` on an unparseable value (probe gap, not proof either way). Pure (unit-tested)."""
    from datetime import datetime, timezone

    try:
        dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp() > now_epoch


def _windows_update_status() -> tuple[bool | None, bool | None]:
    """Measured ``(pending_reboot, updates_paused)`` from the registry; ``None`` = probe unavailable (C2).

    Pending reboot = existence of either well-known key (Windows Update's ``RebootRequired`` or CBS's
    ``RebootPending``); paused = ``PauseUpdatesExpiryTime`` under ``WindowsUpdate\\UX\\Settings`` holding
    a FUTURE datetime (an absent value is definitive: updates are NOT paused). stdlib ``winreg`` only —
    a non-Windows host returns ``(None, None)`` and the check WARNs instead of guessing.
    """
    try:
        import winreg
    except ImportError:  # non-Windows host
        return None, None

    def _key_exists(path: str) -> bool:
        try:
            winreg.CloseKey(winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path))
            return True
        except OSError:
            return False

    pending: bool | None
    try:
        pending = (
            _key_exists(r"SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired")
            or _key_exists(r"SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending")
        )
    except Exception:  # noqa: BLE001 - a probe failure must not abort the gauntlet
        pending = None

    paused: bool | None
    try:
        import time as _time

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\WindowsUpdate\UX\Settings")
        try:
            value, _kind = winreg.QueryValueEx(key, "PauseUpdatesExpiryTime")
        finally:
            winreg.CloseKey(key)
        paused = _pause_expiry_is_future(str(value), _time.time())
    except OSError:
        paused = False  # key/value absent on a Windows host => updates are NOT paused (definitive)
    except Exception:  # noqa: BLE001 - a probe failure must not abort the gauntlet
        paused = None
    return pending, paused


def _probe_api(model: str, key_env: str) -> tuple[str, str]:
    """LIVE 1-token Anthropic liveness call (M9) — runs ONLY under --probe; NEVER called from tests.

    Bills ~1 output token against the pinned campaign model. ``max_retries=0`` so the RAW first-attempt
    failure surfaces (the gauntlet wants the unvarnished key/credit state, not a retried success).
    Classified by exception class name + ``status_code`` (mirrors ``src/llm/client.py::
    _is_transient_api_error``'s name-based approach) so classification needs no SDK import of its own.
    """
    import os

    try:
        from anthropic import Anthropic
    except ImportError as exc:
        return "unavailable", f"anthropic SDK not importable: {exc}"
    try:
        client = Anthropic(api_key=os.environ.get(key_env, "").strip(), max_retries=0)
        msg = client.messages.create(
            model=model, max_tokens=1, messages=[{"role": "user", "content": "ping"}]
        )
        return "ok", f"model {model} answered (stop_reason={getattr(msg, 'stop_reason', None)})"
    except Exception as exc:  # noqa: BLE001 - every failure mode is CLASSIFIED, never raised
        name = type(exc).__name__
        status = getattr(exc, "status_code", None)
        detail = f"{name} (status={status}): {str(exc)[:160]}"
        if name in ("AuthenticationError", "PermissionDeniedError") or status in (401, 403):
            return "auth_error", detail
        if (
            name in ("APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError")
            or status == 429
            or (isinstance(status, int) and status >= 500)
        ):
            return "transient", detail
        if isinstance(status, int) and 400 <= status < 500:
            return "client_error", detail
        return "transient", detail  # unknown network-ish failure -> advisory, not a hard NO-GO


def _free_disk_gb(path: str) -> float:
    import shutil
    return shutil.disk_usage(path).free / 2**30


def _ram_gb() -> tuple[float, float]:
    import psutil
    vm = psutil.virtual_memory()
    return vm.total / 2**30, vm.available / 2**30


def _commit_avail_gb() -> float | None:
    """Available Windows commit charge (RAM + pagefile headroom) in GB; None off-Windows."""
    try:
        import ctypes

        class _MSX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        m = _MSX()
        m.dwLength = ctypes.sizeof(m)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return None
        return m.ullAvailPageFile / 2**30
    except (AttributeError, OSError):
        return None


def _free_vram_mb() -> float | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free_b, _total_b = torch.cuda.mem_get_info()
        return free_b / 2**20
    except Exception:
        return None


def _repo_root() -> str:
    from pathlib import Path
    return str(Path(__file__).resolve().parents[1])


def _gather_and_check(n_gpu: int, min_disk_gb: float, *, provider: str, api_probe: bool) -> list[Check]:
    from pathlib import Path

    root = Path(_repo_root())
    checks: list[Check] = []

    # M9: load the gitignored .env FIRST (the campaign does the same via ADR-038) so the
    # ANTHROPIC_API_KEY the run will actually see is visible to the key check + live probe below —
    # previously the gauntlet FAILed api_key on a machine whose key lives only in .env.
    try:
        import sys as _sys

        _sys.path.insert(0, str(root))
        from src.utils.env import load_env  # type: ignore[import-not-found]

        load_env()
    except Exception:  # noqa: BLE001 - a probe failure must not abort the gauntlet
        pass

    # disk / ram / vram
    try:
        checks.append(check_disk(_free_disk_gb(str(root)), min_disk_gb))
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("disk", WARN, f"disk probe failed: {exc}"))
    try:
        total, avail = _ram_gb()
        checks.append(check_ram(total, avail, n_gpu))
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("ram", WARN, f"ram probe failed (psutil?): {exc}"))
    checks.append(check_commit_headroom(_commit_avail_gb(), n_gpu))
    checks.append(check_vram(_free_vram_mb(), n_gpu))

    # retry layer (C1): tenacity must be importable or every API call is single-attempt.
    try:
        import importlib.util as _ilu

        checks.append(check_retry_layer(_ilu.find_spec("tenacity") is not None))
    except Exception as exc:  # noqa: BLE001 - a probe failure must not abort the gauntlet
        checks.append(Check("retry_layer", WARN, f"retry-layer probe failed: {exc}"))

    # Windows Update (C2): FAIL on a queued reboot, WARN when updates are not verifiably paused.
    try:
        pending_reboot, updates_paused = _windows_update_status()
        checks.append(check_windows_update(pending_reboot, updates_paused))
    except Exception as exc:  # noqa: BLE001 - a probe failure must not abort the gauntlet
        checks.append(Check("windows_update", WARN, f"windows-update probe failed: {exc}"))

    # api key (only required for a real paid provider)
    required = provider.lower() not in ("stub", "offline", "fake")
    key_name = "ANTHROPIC_API_KEY"
    present = bool(os.environ.get(key_name, "").strip())
    checks.append(check_api_key(present, key_name, required=required))
    if api_probe and required and present:
        # M9: REAL 1-token liveness call (opt-in via --probe ONLY — never runs in tests/CI). Model +
        # key-env are read from config/campaign.yaml: llm (the pinned snapshot the run will use), so
        # the probe exercises EXACTLY the campaign's model/key pair. An expired key or an exhausted
        # credit balance must surface here, at t0, not at hour 3 of the search leg.
        try:
            import yaml

            camp_llm = (
                yaml.safe_load((root / "config" / "campaign.yaml").read_text(encoding="utf-8")) or {}
            ).get("llm", {}) or {}
            model = str(camp_llm.get("model_snapshot", "") or "")
            probe_key_env = str(camp_llm.get("api_key_env", key_name) or key_name)
            if not model:
                outcome, detail = "unavailable", "llm.model_snapshot missing from config/campaign.yaml"
            else:
                outcome, detail = _probe_api(model, probe_key_env)
        except Exception as exc:  # noqa: BLE001 - a probe failure must not abort the gauntlet
            outcome, detail = "unavailable", f"probe setup failed: {exc}"
        checks.append(check_api_probe(outcome, detail))

    # freeze
    try:
        import yaml
        prereg = yaml.safe_load((root / "config" / "preregistration.yaml").read_text(encoding="utf-8"))
        frozen = bool(prereg.get("frozen", False))
        hash_match: bool | None = None
        try:
            sys.path.insert(0, str(root))
            from scripts.freeze import canonical_hash  # type: ignore

            recorded = str(prereg.get("freeze_hash", "") or "")
            fresh = str(canonical_hash(root))
            hash_match = bool(recorded) and recorded == fresh
        except Exception:
            hash_match = None
        checks.append(check_freeze(frozen, hash_match))
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("freeze", WARN, f"freeze probe failed: {exc}"))

    # budget mirror (H-M1): the two authored copies of train_steps_per_candidate must agree.
    try:
        import yaml

        camp = yaml.safe_load((root / "config" / "campaign.yaml").read_text(encoding="utf-8")) or {}
        alg = yaml.safe_load((root / "config" / "algos.yaml").read_text(encoding="utf-8")) or {}
        checks.append(check_budget_mirror(
            camp.get("train_steps_per_candidate"), alg.get("train_steps_per_candidate")))
    except Exception as exc:  # noqa: BLE001 - a probe failure must not abort the gauntlet
        checks.append(Check("budget_mirror", WARN, f"budget-mirror probe failed: {exc}"))

    # model mirror (batch-6 M4): the reward-author model_snapshot is authored in campaign.yaml's llm block
    # (executed) AND llm.yaml (standalone mirror); a partial edit must FAIL so the two cannot silently drift.
    try:
        import yaml

        camp = yaml.safe_load((root / "config" / "campaign.yaml").read_text(encoding="utf-8")) or {}
        llm_cfg = yaml.safe_load((root / "config" / "llm.yaml").read_text(encoding="utf-8")) or {}
        checks.append(check_model_consistency(
            (camp.get("llm") or {}).get("model_snapshot"), llm_cfg.get("model_snapshot")))
        # generations mirror (2026-07-05, map M04): the design-defining reflection depth (6) + the H3
        # single-shot control (1) — a key-drop would silently run every LLM arm single-shot.
        checks.append(check_generations_mirror(
            (camp.get("llm") or {}).get("generations"), llm_cfg.get("generations"),
            (camp.get("llm") or {}).get("h3_singleshot_generations")))
        # R106 (2026-07-27): the author's CAP + REASONING PIN. R102's 8192 lived only in llm.yaml,
        # which no code reads for the author, so the confirmatory run silently used campaign.py's
        # 4096 default — the same drift this block already guards for the model id and generations.
        checks.append(check_author_pin_mirror(camp.get("llm") or {}, llm_cfg))
    except Exception as exc:  # noqa: BLE001 - a probe failure must not abort the gauntlet
        checks.append(Check("model_mirror", WARN, f"model-mirror probe failed: {exc}"))

    # agent-key typo guard (2026-07-03): campaign.yaml AND prototype.yaml agent blocks — the campaign
    # consumes BOTH (run_campaign bases agent_cfg on prototype's block, then overlays campaign's), so a
    # typo'd key in either would freeze into the hash while cfg_get silently trains on the default.
    try:
        import yaml

        blocks: dict[str, object] = {}
        for fname in ("campaign.yaml", "prototype.yaml"):
            cfg = yaml.safe_load((root / "config" / fname).read_text(encoding="utf-8")) or {}
            blocks[fname] = cfg.get("agent")
        checks.append(check_agent_keys(blocks))
    except Exception as exc:  # noqa: BLE001 - a probe failure must not abort the gauntlet
        checks.append(Check("agent_keys", WARN, f"agent-key probe failed: {exc}"))

    # data — resolve the ACTIVE panel via gold_suffix() (C3: no hardcoded univ3) and verify its
    # bytes against the frozen manifest so `checksum_ok` is REAL (was always None -> WARN).
    try:
        sys.path.insert(0, str(root))
        from src.data.loaders import (  # type: ignore[import-not-found]
            _expected_sha256,
            _file_sha256,
            _MANIFEST,
            gold_suffix,
        )

        suffix = gold_suffix()
        gold = root / "data" / "gold" / f"returns_panel_{suffix}.parquet"
        checksum_ok: bool | None
        if not gold.is_file():
            checksum_ok = None  # existence handled by check_data
        else:
            expected = _expected_sha256(gold, _MANIFEST)
            # None -> no manifest entry for the panel; surface as WARN (checksum_ok=None) here rather
            # than raising, so the GO/NO-GO gauntlet reports it advisorily (the hard fail-loud lives in
            # the production LOAD path, loaders._verify_checksum, C2).
            checksum_ok = (expected == _file_sha256(gold)) if expected is not None else None
        checks.append(check_data(gold.is_file(), checksum_ok, str(gold)))
    except Exception as exc:  # noqa: BLE001 - a probe failure must not abort the gauntlet
        gold = root / "data" / "gold" / "returns_panel_univ3.parquet"
        checks.append(check_data(gold.is_file(), None, f"{gold} (checksum probe failed: {exc})"))

    return checks


def main(argv: list[str] | None = None) -> int:
    from src.utils.console import make_console_safe
    make_console_safe()   # src/utils/console.py — a console codepage must never kill the GO gate
    ap = argparse.ArgumentParser(description="Pre-flight GO/NO-GO gauntlet for the campaign run.")
    ap.add_argument("--gpu", type=int, default=2, help="planned concurrent GPU workers (n_gpu)")
    ap.add_argument("--min-disk-gb", type=float, default=5.0)
    ap.add_argument("--provider", default="anthropic", help="LLM provider (stub/offline -> no API key needed)")
    ap.add_argument("--probe", action="store_true",
                    help="opt-in LIVE 1-token API call against config/campaign.yaml llm.model_snapshot "
                         "(M9: an expired key / empty credit fails HERE, not at hour 3; bills ~1 token)")
    args = ap.parse_args(argv)

    checks = _gather_and_check(args.gpu, args.min_disk_gb, provider=args.provider, api_probe=args.probe)
    print("=" * 64)
    print("PRE-FLIGHT GAUNTLET")
    print("=" * 64)
    for c in checks:
        mark = {PASS: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]"}[c.status]
        print(f"{mark} {c.name:10s} {c.detail}")
    v = verdict(checks)
    print("-" * 64)
    print(f"VERDICT: {v}")
    return 0 if v != "NO-GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
