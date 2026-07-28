<#
.SYNOPSIS
    Dead-man's-switch heartbeat for the unattended campaign (ops audit M5b).

.DESCRIPTION
    Every alert channel that runs ON the laptop (monitor.py --notify, the supervisor's console) dies
    WITH the laptop — a power cut, a hard hang, or a dead network produces exactly zero alerts. The
    only alert that survives host death is an EXTERNAL service that pages when an expected heartbeat
    STOPS arriving. Setup (once, free):

      1. Register a check at a healthchecks-style service (e.g. https://healthchecks.io) with an
         expected period of 15 minutes and a grace period of ~10 minutes; copy its ping URL.
      2. Run this script alongside the campaign — either as a loop in its own terminal:
             powershell -ExecutionPolicy Bypass -File scripts\deadman_ping.ps1 -Url https://hc-ping.com/<uuid>
         or as a Task Scheduler job firing the one-shot form every 15 minutes:
             ... -File scripts\deadman_ping.ps1 -Url https://hc-ping.com/<uuid> -Once
      3. The service now emails/pushes when pings STOP — i.e. when the host lost power, hung, or
         dropped off the network. (It does NOT know about run health; pair it with
         `monitor.py --follow-campaign --notify` for done/error/stall alerts.)

    OPERATOR-RUN ONLY - nothing in the campaign invokes this. Read-only side-channel: it sends an
    empty GET carrying no run data. Failures are swallowed (the next interval retries); a dead ping
    loop is exactly the condition the external service exists to catch.

.PARAMETER Url
    The registered check's ping URL (required).

.PARAMETER IntervalSeconds
    Loop period (default 900 = 15 min; match the check's expected period).

.PARAMETER Once
    Send a single ping and exit (Task-Scheduler-friendly).
#>
param(
    [Parameter(Mandatory = $true)][string]$Url,
    [int]$IntervalSeconds = 900,
    [switch]$Once
)

while ($true) {
    try {
        Invoke-RestMethod -Uri $Url -Method Get -TimeoutSec 20 | Out-Null
        Write-Host "$(Get-Date -Format s) ping ok"
    } catch {
        Write-Host "$(Get-Date -Format s) ping FAILED: $($_.Exception.Message)"
    }
    if ($Once) { break }
    Start-Sleep -Seconds $IntervalSeconds
}
