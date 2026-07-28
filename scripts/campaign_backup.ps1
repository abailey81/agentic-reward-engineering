# CAMPAIGN BACKUP - a second copy of the irreplaceable campaign archive, plus the node-side
# authoring evidence that lives ONLY on purge-eligible cluster Scratch.
#
# WHY THIS EXISTS (2026-07-28, written mid-campaign).
# The driver supports mirroring behind a `--mirror` flag, and the twelve MODE-D lines were launched
# WITHOUT it. So `D:\llm_rp_archive_mirror` sat 540 h stale while the confirmatory campaign wrote
# 186+ records that cannot be regenerated: LLM calls are non-deterministic, so a lost record is lost
# science, not a re-run. The sentinel reported the staleness correctly; nothing acted on it.
#
# Restarting twelve running lines just to add a flag is the larger risk, so this backs the archive
# up from OUTSIDE the campaign instead. It touches nothing the campaign writes.
#
# APPEND-ONLY, DELIBERATELY. Robocopy `/MIR` would propagate DELETIONS into the backup - which is
# precisely the failure this guards against. On 2026-07-27 a recursive delete removed the licensed
# gold panel and 1,085 raw files; had a `/MIR` job run after it, the backup would have been wiped
# in the same stroke. `/E` without purge means the mirror only ever GROWS: a deletion upstream
# leaves the copy intact, which is the entire point of a backup.
#
# USAGE (background it and forget it):
#   powershell -ExecutionPolicy Bypass -File scripts\campaign_backup.ps1
# Stop it the same way everything else stops: create outputs\campaign_cluster\STOP_CAMPAIGN.

param(
    [int]$IntervalSecs = 900,
    [string]$MirrorRoot = "",
    [string]$Remote = "myriad",
    # RUN GENERATION (2026-07-28). Both roots were hardcoded, which would have silently broken the
    # RUN 2 relaunch three ways: the mirror would have kept copying the HALTED run's tree and never
    # touched RUN 2 (leaving the irreplaceable new archive with no off-machine copy at all), the
    # node-log harvest would have read RUN 1's Scratch, and the harvest OVERWRITES its output file,
    # so RUN 2's rows would have destroyed the RUN 1 evidence that the post-mortem in
    # the campaign execution record rests on. The evidence filename is now root-scoped so
    # the two runs cannot collide. Defaults reproduce RUN 1's paths exactly.
    [string]$SrcRoot = "outputs\campaign_cluster",
    [string]$RemoteRoot = "~/Scratch/llmrp"
)

$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$src      = Join-Path $repo $SrcRoot
$runTag   = Split-Path $SrcRoot -Leaf
if (-not $MirrorRoot) { $MirrorRoot = Join-Path "D:\llm_rp_archive_mirror" $runTag }
$stopFile = Join-Path $src "STOP_CAMPAIGN"
$log      = Join-Path $src "backup.log"
$evidence = Join-Path $repo "docs\evidence"

New-Item -ItemType Directory -Force $MirrorRoot | Out-Null
New-Item -ItemType Directory -Force $evidence   | Out-Null

function BLog([string]$m) {
    $l = "{0} | backup | {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m
    Write-Host $l
    Add-Content -Path $log -Value $l
}

BLog ("started; src={0} -> {1} every {2}s (append-only)" -f $src, $MirrorRoot, $IntervalSecs)

while ($true) {
    if (Test-Path $stopFile) { BLog "STOP_CAMPAIGN present - backup exiting."; break }

    # 1) ARCHIVE -> local mirror. /E recurse, NO /MIR (never propagate deletions), /XO skip older,
    #    /R:1 /W:1 so a transiently-locked in-flight record costs a second, not the whole cycle.
    if (Test-Path $src) {
        $rc = 0
        try {
            # /XD ".pull_tmp*": a pull STAGES into `.pull_tmp.<pid>` and then MOVES the records into
            # the real root, so those files are open and being written while we copy. They are not
            # archived truth (every other instrument excludes them for the same reason), and trying
            # to copy them was the sole source of ERROR 32 sharing violations on the first pass.
            robocopy $src $MirrorRoot /E /XO /R:1 /W:1 /NFL /NDL /NJH /NJS /MT:8 /XD ".pull_tmp*" | Out-Null
            $rc = $LASTEXITCODE
        } catch { BLog ("robocopy threw: {0}" -f $_.Exception.Message) }
        # Robocopy exit codes are a BITMASK: 0-7 are success (0=nothing to do, 1=copied, 2=extras,
        # 4=mismatched). >=8 means a genuine copy FAILURE. Treating any non-zero as an error here
        # would make every successful cycle look broken.
        if ($rc -ge 8) { BLog ("robocopy FAILED rc={0}" -f $rc) }
        else {
            $n = (Get-ChildItem -Path $MirrorRoot -Recurse -Filter record.json -ErrorAction SilentlyContinue).Count
            BLog ("archive mirrored (rc={0}); {1} record(s) in the backup" -f $rc, $n)
            # Stamp the marker the SENTINEL stats (`_MIRROR_ROOT/mirror.log`). Without this the
            # `mirror` check keeps reporting "541 h stale" while the backup is in fact running every
            # cycle - a monitor that disagrees with reality is the failure mode this whole session
            # has been closing, so the freshness signal must be driven by the actual copy.
            $marker = Join-Path (Split-Path $MirrorRoot -Parent) "mirror.log"
            Add-Content -Path $marker -Encoding utf8 -Value (
                "{0} | {1} records mirrored (rc={2})" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $n, $rc)
        }
    }

    # 2) NODE-SIDE AUTHORING EVIDENCE. The per-model authoring-reliability table is a REGISTERED
    #    deliverable, and the only record of WHY a candidate was rejected ("reward crashed during
    #    validation: NameError(...)") lives in the node logs on Scratch, which is purge-eligible.
    #    The ledger row degrades to a generic message because the reject marker is mirrored back by
    #    a LATER pull, so these logs are not a duplicate of anything we already hold.
    try {
        $out = Join-Path $evidence ("node_authoring_rejects_{0}.jsonl" -f $runTag)
        $cmd = "grep -h '\`"failed\`"' {0}/logs/*/*.o 2>/dev/null | sort -u" -f $RemoteRoot
        $rows = & ssh $Remote $cmd 2>$null
        if ($rows -and $rows.Count -gt 0) {
            Set-Content -Path $out -Value $rows -Encoding utf8
            BLog ("harvested {0} node-side reject row(s) -> {1}" -f $rows.Count, (Split-Path $out -Leaf))
        }
    } catch { BLog ("node-log harvest skipped: {0}" -f $_.Exception.Message) }

    Start-Sleep -Seconds $IntervalSecs
}
BLog "backup exiting."
