<#
.SYNOPSIS
    Remove the campaign's ONSTART re-entry task (counterpart of scripts\install_onstart_task.ps1).

.DESCRIPTION
    Run this once the campaign has fully completed (ops audit C2). Leaving the task registered is not
    dangerous — a finished campaign resumes to a no-op and exits 0 — but every boot would relaunch the
    supervisor (preflight + a resume pass), which is pointless noise once the run is banked.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\uninstall_onstart_task.ps1
#>
param(
    [string]$TaskName = "LLMRewardCampaignResume"
)

$ErrorActionPreference = "Stop"

$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($null -eq $task) {
    Write-Host "No scheduled task named '$TaskName' found - nothing to remove."
} else {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed scheduled task '$TaskName'."
}
