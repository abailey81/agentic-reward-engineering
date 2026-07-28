# CAMPAIGN SUPERVISOR (2026-07-18) — makes any driver death SELF-HEALING.
#
# WHY: the 2026-07-17 VPN outage killed both laptop drivers after their bounded ride-out
# (loud + resumable BY DESIGN — but a human had to notice). During the real campaign a
# night-time outage would stall authoring/requeues until morning. This wrapper relaunches
# the driver on ANY nonzero exit: the driver is idempotent (archive-truth resume; the P12
# lock auto-breaks for a dead owner; ledger-resubmit never re-bills authored candidates),
# so relaunching is always safe. Running arrays on Myriad are NEVER affected by a driver
# death — only the laptop-side loop stalls, and this closes that gap.
#
# USAGE (from the repo root, after the freeze — the P19 gate applies on every relaunch):
#   powershell -ExecutionPolicy Bypass -File scripts\campaign_supervisor.ps1
# Stop deliberately: create the file outputs\campaign_cluster\STOP_CAMPAIGN (checked
# between attempts), or Ctrl+C the supervisor itself.

$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$outDir   = "outputs\campaign_cluster"
$stopFile = Join-Path $outDir "STOP_CAMPAIGN"
$logFile  = Join-Path $outDir "supervisor.log"
New-Item -ItemType Directory -Force $outDir | Out-Null

# THE LAUNCH LINE — must stay in lockstep with the campaign-day runbook.
$py = Join-Path $repo ".venv\Scripts\python.exe"   # ABSOLUTE (2026-07-18 drill: a relative path can fail to resolve and PS tries module-autoload)
# NO --arms and NO --baselines: under --tiered the launcher RESOLVES both from the frozen config
# (resolve_cluster_arms / resolve_cluster_baselines) and refuses a partial list before ssh. This
# line hand-typed SEVEN arms and was still doing so after R108 took the roster to NINE, so it would
# have silently dropped cma_es and tpe and made confirmatory node N4 unsatisfiable - the same drift
# that hit --baselines when the H1 canon went 4 -> 11. Never hand-type a frozen list here again.
#
# CPU LANE (2026-07-27): this line was GPU-only (--pool EF plus a GPU --seed-pool-blocks stripe),
# which the launcher now REFUSES together with --device cpu. Flags mirror mode_d_supervisor.ps1's
# $cpuLane; read that file's header for the justification of each one.
$args = @(
  "scripts/run_campaign_cluster.py", "--tiered",
  "--pass-mode", "B", "--llm-from", "campaign",
  "--device", "cpu", "--pool", "d",
  "--pack", "4", "--cores-per-training", "1",
  "--search-pack", "1", "--search-threads", "8",
  "--chunk-tasks", "1", "--exclude-hosts", "node-d00a-230",
  "--gold-dir", "/acfs/users/ucestes/gold",
  "--batch-tag", "c1", "--poll-secs", "180", "--search-poll-secs", "45",
  "--output-dir", $outDir, "--resume"
)

$attempt = 0
$maxAttempts = 1000
$backoffSecs = 600

function Log([string]$msg) {
    $line = "{0} | supervisor | {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $logFile -Value $line
}

Log "supervisor started (MSYS_NO_PATHCONV irrelevant here: native PowerShell, no path mangling)"
while ($attempt -lt $maxAttempts) {
    if (Test-Path $stopFile) { Log "STOP_CAMPAIGN present - deliberate stop."; break }
    $attempt += 1
    Log ("attempt {0}: launching the campaign driver" -f $attempt)
    # 2026-07-18 drill catch: if the invocation FAILS TO START, $LASTEXITCODE is never set and a
    # STALE zero from a prior iteration would read as "CAMPAIGN COMPLETE". Default rc to -1 and
    # only trust $LASTEXITCODE when the call actually ran.
    $rc = -1
    try {
        & $py @args
        if ($null -ne $LASTEXITCODE) { $rc = $LASTEXITCODE }
    } catch {
        Log ("driver INVOCATION failed before start: {0}" -f $_.Exception.Message)
        $rc = -1
    }
    if ($rc -eq 0) { Log "driver exited 0 - CAMPAIGN COMPLETE (or gate stop handled). Supervisor done."; break }
    Log ("driver exited {0} (outage/crash class) - relaunching in {1}s; running arrays on Myriad are unaffected" -f $rc, $backoffSecs)
    Start-Sleep -Seconds $backoffSecs
}
Log "supervisor exiting."
