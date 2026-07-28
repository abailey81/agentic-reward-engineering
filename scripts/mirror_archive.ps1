<#
.SYNOPSIS
    Mirror the irreplaceable run archives to a second physical drive (engineering review 2026-07-02).

.DESCRIPTION
    The campaign archive (LLM-authored reward code, prompts, reflections, per-seed records) is the ONE
    artifact in this project that cannot be regenerated: results REPLAY from the archive
    because LLM calls are non-deterministic. Everything else — panels, configs,
    code — is reconstructible; the archive is not. Until 2026-07-02 it lived on a single drive with no
    redundancy: a C: failure mid-campaign would have destroyed the experiment.

    This script robocopy-mirrors the archive roots to D:\llm_rp_archive_mirror\ (a different physical
    disk). /MIR keeps an exact replica (adds+updates+deletes); /R:2 /W:5 bounds retries so a locked
    in-flight file never stalls the mirror (it lands on the next pass); /XD excludes the transient
    driver_status (heartbeats) + batches (reconstructible specs) dirs. Run it:
      * manually after any pilot completes;
      * during the campaign: every 6 h via the watcher loop or a scheduled task —
        schtasks /Create /SC HOURLY /MO 6 /TN LLMRewardArchiveMirror
          /TR "powershell -ExecutionPolicy Bypass -File <repo>\scripts\mirror_archive.ps1"
    I/O-light (incremental after the first pass); safe to run beside training (reads only), but the
    2026-07-02 exclusive-phase discipline still prefers the recycle boundaries.

.PARAMETER MirrorRoot
    Destination root on the second drive (default D:\llm_rp_archive_mirror).

.PARAMETER Force
    Override the shrink guard (m8). By default a source whose record.json count is SMALLER than the
    existing mirror's is SKIPPED (a shrunken/emptied source would otherwise propagate the loss into the
    backup via /MIR). Pass -Force to mirror it anyway — e.g. after a deliberate, verified prune.
#>
param(
    [string]$MirrorRoot = "D:\llm_rp_archive_mirror",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

# The irreplaceable set: run archives + the frozen-design record + the manifest ledgers.
# NOTE (2026-07-08): campaign_cluster is the MYRIAD driver's local_archive_root (the pulled Scratch
# mirror) — it MUST be mirrored too, or the 3-site grade-security guarantee (Myriad->laptop->D:->Mac)
# does not hold for the cluster run. driver_status/ (transient heartbeats) and batches/ (reconstructible
# specs/jobscripts) are EXCLUDED via /XD below: they are not irreplaceable and would churn every pass.
$sources = @(
    @{ src = Join-Path $repoRoot "outputs\campaign";         dst = Join-Path $MirrorRoot "campaign" },
    @{ src = Join-Path $repoRoot "outputs\campaign_cluster"; dst = Join-Path $MirrorRoot "campaign_cluster" },
    @{ src = Join-Path $repoRoot "outputs\sigma_pilot";      dst = Join-Path $MirrorRoot "sigma_pilot" },
    @{ src = Join-Path $repoRoot "outputs\prototype";        dst = Join-Path $MirrorRoot "prototype" },
    @{ src = Join-Path $repoRoot "outputs\tables";           dst = Join-Path $MirrorRoot "tables" },
    @{ src = Join-Path $repoRoot "data\manifest";            dst = Join-Path $MirrorRoot "manifest" }
)

New-Item -ItemType Directory -Force $MirrorRoot | Out-Null
$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$log = Join-Path $MirrorRoot "mirror.log"
Add-Content $log "=== mirror pass $stamp ==="

foreach ($pair in $sources) {
    if (-not (Test-Path $pair.src)) { Add-Content $log "SKIP (absent): $($pair.src)"; continue }
    # Shrink guard (m8): /MIR propagates DELETIONS, so a shrunken or emptied SOURCE (a bad prune, a
    # half-synced drive, a corrupted run dir) would wipe archived history from the BACKUP in a single
    # pass. Test-Path above only catches a FULLY-absent source. Count record.json (the per-run archive
    # marker) recursively in the source vs the existing mirror; if the mirror holds MORE than the source,
    # the source has lost archives -> SKIP loudly rather than mirror the loss, unless -Force is passed.
    # PS 5.1-safe: @(...) forces an array so .Count is valid for 0/1 matches; no ternary / no && used.
    $srcCount = @(Get-ChildItem -LiteralPath $pair.src -Recurse -Filter record.json -File -ErrorAction SilentlyContinue).Count
    $dstCount = 0
    if (Test-Path $pair.dst) {
        $dstCount = @(Get-ChildItem -LiteralPath $pair.dst -Recurse -Filter record.json -File -ErrorAction SilentlyContinue).Count
    }
    if (($dstCount -gt $srcCount) -and (-not $Force)) {
        $skipMsg = "SKIP (shrink guard): $($pair.src) has $srcCount record.json but mirror $($pair.dst) has $dstCount -- refusing to mirror history loss (pass -Force to override)"
        Add-Content $log $skipMsg
        Write-Warning $skipMsg
        continue
    }
    # robocopy exit codes 0-7 = success variants; >=8 = failure. Never let one root abort the rest.
    # /XD skips transient, reconstructible dirs (heartbeats + spec/jobscript batches) so the mirror holds
    # only the irreplaceable records and never churns on monitoring state.
    robocopy $pair.src $pair.dst /MIR /R:2 /W:5 /NP /NDL /NFL /NJH /XD driver_status batches | Out-Null
    $code = $LASTEXITCODE
    $verdict = if ($code -ge 8) { "FAIL($code)" } else { "ok($code)" }
    Add-Content $log "$verdict : $($pair.src) -> $($pair.dst)"
    if ($code -ge 8) { Write-Warning "mirror FAILED for $($pair.src) (robocopy $code)"; $anyFail = $true }
}

$total = (Get-ChildItem $MirrorRoot -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
Add-Content $log ("mirror size: {0:N1} MB" -f $total)

# VERIFIED mirror (2026-07-06 A5): after the copy, re-hash the mirror against its sealed
# archive-integrity manifest so a silently-corrupted BACKUP is caught at mirror time, not at the
# disaster-recovery moment. verify-mirror mode: sealed records must be intact (removed/changed fail);
# records ADDED after the seal are tolerated (a mid-campaign mirror lawfully carries newer work).
# Skipped quietly when no manifest exists yet (the seal is written at campaign end). Best-effort:
# a verify failure marks the pass FAILED (exit 9) but never blocks the copy itself.
$py = Join-Path $repoRoot ".venv\Scripts\python.exe"
$verifyFail = $false
foreach ($pair in $sources) {
    $manifest = Join-Path $pair.dst "archive_integrity.json"
    if ((Test-Path $manifest) -and (Test-Path $py)) {
        & $py (Join-Path $repoRoot "scripts\archive_integrity.py") verify-mirror $pair.dst
        if ($LASTEXITCODE -ne 0) {
            $msg = "MIRROR VERIFY FAILED: $($pair.dst) does not match its sealed manifest (backup corrupt?)"
            Add-Content $log $msg
            Write-Warning $msg
            $verifyFail = $true
        } else {
            Add-Content $log "verify ok: $($pair.dst)"
        }
    }
}

Write-Host ("mirror pass done -> {0}  ({1:N1} MB; log: {2})" -f $MirrorRoot, $total, $log)
# robocopy's success code (1 = files copied) would otherwise leak as the script exit code and read as a
# failure to any caller. Map the 0-7 robocopy-success band to a clean exit 0; reserve non-zero for a real
# >=8 mirror failure (2026-07-05) and exit 9 for a backup-integrity verify failure (2026-07-06).
if ($anyFail) { exit 8 } elseif ($verifyFail) { exit 9 } else { exit 0 }
