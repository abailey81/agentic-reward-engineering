# MODE-D LAUNCHER (2026-07-21) - the maximum-parallel campaign: TWELVE supervised driver lines
# at once (the Opus core + the H3 floor unit + all 10 replication legs), each with its own log.
#
# WHAT MODE D IS (runbook s.10): every line submits from L+0; the SGE priority ladder - core
# search/floor/tier-100 highest, legs -200..-290 in the registered queue order, tier-189+ blocks
# from -300 - makes the scheduler enforce the registered unified queue natively, so completion
# and truncation order are EXACTLY the pre-declared ones while the eligible backlog stays deep
# enough to harvest every idle window. Search waves run pack-2 (latency lane); bursts pack-5
# (throughput lane); C4 rungs are pipelined. All ops-only: no registered quantity changes (R88).
#
# USAGE (after the v2 FREEZE + on an explicit LAUNCH decision - never before):
#   powershell -ExecutionPolicy Bypass -File scripts\mode_d_launch.ps1
# Stop everything: create outputs\campaign_cluster\STOP_CAMPAIGN (all lines check it).
# Monitoring: bash scripts/campaign_monitor.sh + the sentinel watch the shared mirror as usual;
# per-line logs at outputs\campaign_cluster\supervisor_<line>.log.

param(
    # RUN GENERATION (2026-07-28). RUN 1 was invalidated by the cross-line reject-marker collision.
    # A clean re-execution needs a fresh archive on
    # BOTH sides - the local mirror AND the Scratch working root, because the halted run's jobs
    # keep archiving remotely for hours and archive-truth resume would adopt their records.
    # Defaults reproduce RUN 1's paths exactly, so an unparameterised call is unchanged.
    # RUN 2 is launched as:
    #   powershell -ExecutionPolicy Bypass -File scripts\mode_d_launch.ps1 `
    #     -OutDir outputs\campaign_cluster_run2 -RemoteRoot ~/Scratch/llmrp2
    [string]$OutDir = "outputs\campaign_cluster",
    [string]$RemoteRoot = "~/Scratch/llmrp",
    # 2026-08-01 (RUN 10), found while applying D15: the register asked whether this launcher had
    # the same omission as the watchdog. IT DID - this file launched every supervised line with NO
    # -ExcludeHosts at all, so the substrate fence that protects the run existed ONLY because the
    # live processes happened to be started with it BY HAND. A clean relaunch from this script
    # would have silently dropped it, which is the same failure D15 describes for revival, at the
    # more dangerous end: it would apply to ALL TWELVE lines at once rather than one revived line.
    [string]$ExcludeHosts = "node-d00a-230"
)

$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

# Queue order (must match config/preregistration.yaml model_suite.queue_order).
$lines = @(
  "core", "h3",
  "deepseek-v4-pro", "glm-5.2", "qwen3.6-27b", "qwen3.5-9b",
  "haiku-4.5", "gpt-5.6-luna", "nemotron-3-super", "sonnet-5", "gemini-2.5-flash", "kimi-k3"
)

# 2026-07-27: the priority ladder this header describes is RETIRED BY R101 (Okhrati's seed-parity
# directive: all 11 full-loop models run at EQUAL standing, climbing ONE common ladder in lockstep).
# No line passes --priority any more; the default 0 is full fair-share standing, which is also
# The absolute never-deprioritise rule applies. The substrate is the CPU lane, not the GPU pools - see
# the rewritten header of mode_d_supervisor.ps1 for the five defects that change closed.

$i = 0
foreach ($line in $lines) {
    $stagger = if ($line -eq "core") { 0 } else { 3600 + $i * 20 }   # CANARY SHIELD: legs start
    # ~1h after the core so most path breakage the C0 canary exists to catch surfaces before
    # any leg authoring is billed (bounded anyway; ~1h is negligible vs the legs' 3-4 days).
    # The +20s spacing spreads the poll phases (login-node kindness).
    Write-Host ("mode-D: starting supervised line '{0}' (stagger {1}s)" -f $line, $stagger)
    Start-Process powershell -ArgumentList @(
        "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $repo "scripts\mode_d_supervisor.ps1"),
        "-Line", $line, "-StaggerSecs", [string]$stagger,
        "-ExcludeHosts", $ExcludeHosts,
        "-OutDir", $OutDir, "-RemoteRoot", $RemoteRoot
    )
    $i += 1
}
Write-Host ("mode-D: {0} supervised lines started (out={1}, remote={2}). STOP file: {3}" -f `
    $lines.Count, $OutDir, $RemoteRoot, (Join-Path $OutDir "STOP_CAMPAIGN"))
