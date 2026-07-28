<#
.SYNOPSIS
    Register a Task Scheduler ONSTART task that re-enters the campaign after ANY reboot (ops audit C2).

.DESCRIPTION
    A 2-3-week unattended laptop run WILL see a reboot the operator did not schedule (Windows Update,
    power loss, a driver bluescreen). Without re-entry, the run silently stays down until a human
    notices — potentially days of lost wall-clock. This script registers a scheduled task that fires
    AT SYSTEM STARTUP and relaunches `scripts\supervisor.py -- <campaign args> --resume`:

      * the supervisor runs preflight, then keeps the campaign alive (bounded restarts + backoff);
      * `--resume` (also auto-appended by the supervisor itself, M7) makes re-entry idempotent —
        archived candidates/seeds replay from disk, so a reboot never re-bills the paid author;
      * stdout/stderr are appended to outputs\campaign\onstart_task.log for post-hoc forensics.

    OPERATOR-RUN ONLY — nothing in the campaign invokes this. Run it ONCE from an elevated PowerShell
    on run day, and remove it with scripts\uninstall_onstart_task.ps1
    when the campaign completes (otherwise every future boot relaunches the supervisor; that relaunch
    is harmless — a finished campaign resumes to a no-op and exits 0 — but it is noise you don't want).

    The task runs as the CURRENT user (the .env key + the venv live in this profile) with highest
    privileges, whether or not the user is logged on (S4U logon: no password stored), with a 2-minute
    startup delay so CUDA/network services are up before training starts.

.PARAMETER TaskName
    Scheduled-task name (default: LLMRewardCampaignResume).

.PARAMETER CampaignArgs
    Argument string passed to run_campaign.py AFTER the supervisor's `--` separator. Default is the
    runbook's laptop headline invocation. `--resume` is appended by the supervisor regardless.

.PARAMETER SupervisorGpu
    n_gpu handed to the supervisor's preflight gate (default 2).

.PARAMETER Myriad
    Register the CLUSTER-driver resume task instead of the laptop one: the supervisor child becomes the
    Myriad driver (scripts\run_campaign_cluster.py), the laptop GPU clock lock is SKIPPED (the laptop
    GPU is idle — training is 100% on Myriad), and the supervisor's laptop-GPU preflight is bypassed
    (--no-preflight; the cluster's own G1 cert is the preflight). Use this for the 100%-Myriad campaign.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install_onstart_task.ps1                 # laptop
    powershell -ExecutionPolicy Bypass -File scripts\install_onstart_task.ps1 -Myriad         # cluster
    powershell -ExecutionPolicy Bypass -File scripts\uninstall_onstart_task.ps1   # after the run
#>
param(
    [string]$TaskName = "LLMRewardCampaignResume",
    # SERIAL reflect-on-best headline (ratified 2026-07-01, corrected 2026-07-02): --search-gpu 0 is the
    # default, so the flag is simply OMITTED. (The previous default here still said "--search-gpu 2" —
    # the SUPERSEDED R24 parallel protocol; a reboot would have resumed the campaign on the WRONG
    # protocol. Found + fixed in the 2026-07-02 engineering review.)
    [string]$CampaignArgs = "--gpu 3 --h3-singleshot --no-shutdown",
    # ---- D21 (found 2026-08-01, RUN 10) --------------------------------------------------------
    # MODE-D recovery parameters. Under -Myriad this task no longer re-enters ONE campaign process
    # with a hand-typed argument vector; it re-enters THE FLEET through mode_d_launch.ps1 and
    # mode_d_watchdog.ps1, which are the single source of truth for how this campaign starts.
    # See the -Myriad branch below for the three drifts that duplication had already produced.
    [string]$OutDir = "outputs\campaign_cluster_run4",
    [string]$RemoteRoot = "~/Scratch/llmrp4",
    [string]$ExcludeHosts = "node-d00a-230,node-d00b-024",
    # Best-effort restart of the two detached monitoring loops (bash). Never blocks the campaign
    # recovery: if bash is absent the loops are skipped and the fleet still comes back.
    [string]$BashExe = "C:\Program Files\Git\bin\bash.exe",
    [int]$SupervisorGpu = 3,
    [switch]$Myriad
)

$ErrorActionPreference = "Stop"

# Resolve everything to ABSOLUTE paths (an ONSTART task has no useful working directory).
$repoRoot     = Split-Path -Parent $PSScriptRoot
$pythonExe    = Join-Path $repoRoot ".venv\Scripts\python.exe"
$supervisor   = Join-Path $repoRoot "scripts\supervisor.py"
$clusterEntry = Join-Path $repoRoot "scripts\run_campaign_cluster.py"
$logDir       = Join-Path $repoRoot ($(if ($Myriad) { "outputs\campaign_cluster" } else { "outputs\campaign" }))
$logFile      = Join-Path $logDir "onstart_task.log"

if (-not (Test-Path $pythonExe))  { throw "venv python not found at $pythonExe - create the venv first." }
if (-not (Test-Path $supervisor)) { throw "supervisor not found at $supervisor" }
if ($Myriad -and -not (Test-Path $clusterEntry)) { throw "cluster entry not found at $clusterEntry" }
New-Item -ItemType Directory -Force $logDir | Out-Null

# cmd.exe /c wrapper so stdout+stderr append to the log (Task Scheduler actions cannot redirect).
if ($Myriad) {
    # CLUSTER driver resume: the child is the Myriad driver (over the UCL VPN). No laptop GPU lock (the
    # laptop GPU is idle), and the supervisor's laptop preflight is bypassed. The default cluster args
    # are the tiered C-ladder over the frozen 7-arm roster; override with -CampaignArgs. The supervisor
    # (and, redundantly, the entry) append --resume so re-entry replays the archive — never re-bills.
    # == D21 (2026-08-01, RUN 10): RE-ENTER THE FLEET, NOT ONE CAMPAIGN PROCESS ==================
    #
    # THE GAP. `Get-ScheduledTask` filtered on this repo returned NOTHING on 2026-08-01: this task
    # was never registered for RUN 4. A Windows Update reboot -- over a 26-day run, not a remote
    # possibility -- would have killed all 12 supervisors, 24 drivers, the watchdog, the cycle loop,
    # the publisher and the backup, and NONE of them would have come back, while the Myriad arrays
    # kept running and finishing with nobody polling, pulling or submitting the next generation.
    #
    # AND THE BRANCH BELOW WOULD NOT HAVE FIXED IT. It re-entered ONE `run_campaign_cluster.py`
    # with a hand-typed argument vector, which had already drifted from the live fleet in THREE
    # independent ways:
    #   * `--batch-tag c1`            -> the CORE LINE ONLY; the other ELEVEN lines stay down;
    #   * `--pack 4`                  -> the live fleet runs `--pack 8` (record s.58);
    #   * `--exclude-hosts node-d00a-230` -> MISSING `node-d00b-024`, i.e. the substrate fence.
    #     That is the D15 defect on the reboot path, and it is the worst of the three: it silently
    #     re-opens the CPU-model inhomogeneity that already cost four archived records.
    #
    # THE FIX IS NOT TO CORRECT THOSE THREE LITERALS -- IT IS TO DELETE THEM. Duplicating the
    # launch vector is what produced all three drifts, and a fourth would appear the next time the
    # fleet's shape changed. `mode_d_launch.ps1` starts the twelve supervised lines and returns;
    # `mode_d_watchdog.ps1` then keeps them alive and blocks, which is exactly what a Task
    # Scheduler action should hold open. Both now carry `-ExcludeHosts` (D15, same commit), so the
    # fence survives a reboot for the first time.
    $mdLaunch   = Join-Path $repoRoot "scripts\mode_d_launch.ps1"
    $mdWatchdog = Join-Path $repoRoot "scripts\mode_d_watchdog.ps1"
    foreach ($f in @($mdLaunch, $mdWatchdog)) {
        if (-not (Test-Path $f)) { throw "mode-D recovery script not found at $f" }
    }
    $psExe = "powershell.exe"
    $fleet = ("`"$psExe`" -NoProfile -ExecutionPolicy Bypass -File `"$mdLaunch`" " +
              "-OutDir `"$OutDir`" -RemoteRoot `"$RemoteRoot`" -ExcludeHosts `"$ExcludeHosts`"")
    $watch = ("`"$psExe`" -NoProfile -ExecutionPolicy Bypass -File `"$mdWatchdog`" " +
              "-OutDir `"$OutDir`" -RemoteRoot `"$RemoteRoot`" -ExcludeHosts `"$ExcludeHosts`"")
    $mydInner = "$fleet & $watch"

    # !! `-CampaignArgs` NO LONGER APPLIES UNDER `-Myriad`. The hand-typed vector it used to carry is
    # DELETED rather than corrected (see the block above): every line's arguments now come from
    # `mode_d_supervisor.ps1`, reached through `mode_d_launch.ps1`, so there is exactly one place
    # that knows how a line starts. The frozen roster and the 11-member H1 canon are still resolved
    # from config by the launcher (`resolve_cluster_arms` / `resolve_cluster_baselines`), which
    # REFUSES a partial list before ssh - so a reboot-recovery still cannot resume a subset of the
    # design, and now it also cannot resume a subset of the FLEET.
    if ($PSBoundParameters.ContainsKey('CampaignArgs')) {
        Write-Warning ("-CampaignArgs is ignored under -Myriad since 2026-08-01 (D21): reboot " +
                       "recovery re-enters the fleet via mode_d_launch.ps1, not a single campaign " +
                       "process. Pass -OutDir / -RemoteRoot / -ExcludeHosts instead.")
    }
    $inner = $mydInner
} else {
    # LAPTOP campaign. The GPU clock lock (nvidia-smi -lgc) RESETS on every reboot — re-apply it BEFORE
    # re-entering so a crash-reboot resumes under the SAME uniform conditions the pilots calibrated
    # (2026-07-02 engineering review; Turbo self-heals via Armoury Crate, the lock does not). The `&`
    # chain means a lock failure logs but never blocks the resume itself.
    $inner = "nvidia-smi -lgc 2200,2560 & `"$pythonExe`" `"$supervisor`" --gpu $SupervisorGpu -- $CampaignArgs --resume"
}
$cmdArg = "/c `"$inner >> `"$logFile`" 2>&1`""

$action    = New-ScheduledTaskAction -Execute "cmd.exe" -Argument $cmdArg -WorkingDirectory $repoRoot
$trigger   = New-ScheduledTaskTrigger -AtStartup
$trigger.Delay = "PT2M"   # let CUDA/network services settle before training re-enters
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Highest
# Long-running unattended compute: no execution time limit, keep going on battery, never park it.
$settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings -Force | Out-Null

Write-Host "Registered ONSTART task '$TaskName':"
Write-Host "  cmd.exe $cmdArg"
Write-Host "  log -> $logFile"
Write-Host "Verify with:  Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo"
Write-Host "Remove after the campaign with:  scripts\uninstall_onstart_task.ps1"
