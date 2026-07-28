# MODE-D WATCHDOG - keeps all twelve supervised lines ALIVE for the length of the campaign.
#
# WHY THIS EXISTS (2026-07-28, written during the launch itself).
# mode_d_supervisor relaunches its driver on any NONZERO exit. It does NOT relaunch on exit 0,
# because 0 means "line complete" -- and that is correct for a finished line. But the tiered path
# ALSO returns 0 from the C3 review-gate stop:
#
#     if out.get("awaiting_review"): ... return 0
#
# so a line that stops at the gate is indistinguishable, to the supervisor, from a line that
# finished. The supervisor logs "LINE COMPLETE" and exits, and that line is then dead for the rest
# of the campaign. It happened within the first hour: five leg lines started while a
# MYRIAD_KILL_INCIDENT was blocking submission, their units were therefore incomplete, the gate
# correctly read RED-execution-health, and all five lines exited permanently. Under R101 every model
# must climb the SAME ladder in lockstep, so a silently dead leg is not a delay -- it is a hole in
# the design.
#
# The watchdog restarts any line whose supervisor is gone. It deliberately does NOT clear a kill
# incident: that is a human decision by construction, and it is SAFE to restart lines while an
# incident stands, because the incident blocks SUBMISSION, not process startup. A restarted line
# under a live incident simply backs off again, which is exactly the intended behaviour.
#
# USAGE (background it and forget it):
#   powershell -ExecutionPolicy Bypass -File scripts\mode_d_watchdog.ps1
# Stop it the same way everything else stops: create outputs\campaign_cluster\STOP_CAMPAIGN.

param(
    [int]$IntervalSecs = 300,
    # RUN GENERATION (2026-07-28) - these MUST be passed whenever the campaign runs on a non-default
    # root, and they must match mode_d_launch.ps1 exactly. Before this parameter existed the
    # watchdog restarted every dead line with the supervisor's DEFAULTS, so under a fresh-root run
    # a single restart would have silently pointed that line back at the PREVIOUS run's local mirror
    # and Scratch root - mixing two runs' archives, which is the same class of silent cross-run
    # contamination that invalidated RUN 1.
    [string]$OutDir = "outputs\campaign_cluster",
    [string]$RemoteRoot = "~/Scratch/llmrp",
    # D15 (found 2026-07-30, worked around live, APPLIED 2026-08-01 by RUN 10). This parameter did
    # not exist, so a revived line fell back to the supervisor's DEFAULT fence and SILENTLY UNDID
    # the substrate fence applied to all twelve supervisors (record s.28) - re-opening the very
    # CPU-model inhomogeneity the fence was added to close, which had already cost four archived
    # records on an Intel Xeon Gold 6140.
    #
    # This is EXACTLY the D4 shape, one parameter later, and the comment above already warns about
    # it for the other two: AN AUTOMATIC RESTARTER IS A SECOND LAUNCHER AND MUST TAKE EVERY
    # PARAMETER THE THING THAT STARTED THE LINE TOOK. Anything the supervisor accepts and this file
    # omits will be silently reset on the next revival.
    #
    # The default deliberately MATCHES mode_d_supervisor.ps1's own default rather than the current
    # live fence: a default that drifts from the supervisor's is a second silent divergence. The
    # operative fence is passed explicitly at launch, and the startup log below PRINTS it so a
    # mismatch is visible rather than latent.
    [string]$ExcludeHosts = "node-d00a-230"
)

$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$outDir   = $OutDir
$stopFile = Join-Path $outDir "STOP_CAMPAIGN"
$log      = Join-Path $outDir "watchdog.log"
New-Item -ItemType Directory -Force $outDir | Out-Null

# Must match mode_d_launch.ps1 exactly.
$lines = @(
  "core", "h3",
  "deepseek-v4-pro", "glm-5.2", "qwen3.6-27b", "qwen3.5-9b",
  "haiku-4.5", "gpt-5.6-luna", "nemotron-3-super", "sonnet-5", "gemini-2.5-flash", "kimi-k3"
)

function WLog([string]$m) {
    $l = "{0} | watchdog | {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m
    Write-Host $l
    Add-Content -Path $log -Value $l
}

WLog ("started; watching {0} lines every {1}s (out={2}, remote={3}, fence={4})" -f `
    $lines.Count, $IntervalSecs, $OutDir, $RemoteRoot, $ExcludeHosts)

while ($true) {
    if (Test-Path $stopFile) { WLog "STOP_CAMPAIGN present - watchdog exiting."; break }

    $alive = @()
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match 'mode_d_supervisor' } |
        ForEach-Object { if ($_.CommandLine -match '-Line\s+(\S+)') { $alive += $Matches[1] } }

    $dead = $lines | Where-Object { $alive -notcontains $_ }
    if ($dead.Count -gt 0) {
        WLog ("DEAD lines: {0}" -f ($dead -join ", "))
        foreach ($d in $dead) {
            Start-Process powershell -ArgumentList @(
                "-ExecutionPolicy", "Bypass",
                "-File", (Join-Path $repo "scripts\mode_d_supervisor.ps1"),
                "-Line", $d, "-StaggerSecs", "0",
                "-ExcludeHosts", $ExcludeHosts,
                "-OutDir", $OutDir, "-RemoteRoot", $RemoteRoot
            )
            WLog ("  restarted {0} (fence={1})" -f $d, $ExcludeHosts)
            Start-Sleep -Seconds 3
        }
    }
    Start-Sleep -Seconds $IntervalSecs
}
WLog "watchdog exiting."
