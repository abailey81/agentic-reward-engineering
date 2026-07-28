# MODE-D LINE SUPERVISOR - one self-healing driver line (core, h3, or one replication leg).
#
# The mode-D campaign runs TWELVE driver lines concurrently (the Opus core + the H3 floor unit +
# 10 replication legs). Each line gets its own
# supervisor instance (this script): relaunch on any nonzero exit, because the driver is idempotent
# (archive-truth resume, P12 lock auto-break, no authoring re-billed) and running arrays on Myriad
# are never affected by a laptop-side death.
#
# USAGE (normally via mode_d_launch.ps1, which spawns all twelve):
#   powershell -ExecutionPolicy Bypass -File scripts\mode_d_supervisor.ps1 -Line core
#   powershell -ExecutionPolicy Bypass -File scripts\mode_d_supervisor.ps1 -Line deepseek-v4-pro
# Stop everything deliberately: create outputs\campaign_cluster\STOP_CAMPAIGN (shared, checked
# between attempts by every line), or Ctrl+C individual windows.
#
# =====================================================================================
# REWRITTEN 2026-07-27 (the launch gate). The previous version could not have run this
# campaign. Every item below was verified against the code before being changed:
#
#  1. IT WAS GPU-ONLY. Every line passed "--pool EF" and a GPU "--seed-pool-blocks EF:...,L:..."
#     stripe. The confirmatory lane is now CPU (R107/R108), and run_campaign_cluster REFUSES
#     "--device cpu" together with "--seed-pool-blocks" by design - a CPU job pins no pool, so the
#     stripe would assert a device stratification the run does not have. The launcher could not
#     start on the lane it was meant to launch.
#
#  2. THE CORE LINE HAND-TYPED 7 OF THE FROZEN 9 ARMS. cma_es and tpe - two of the four
#     comparators of CONFIRMATORY node N4 - were missing, exactly the drift that hit --baselines
#     when the H1 canon went 4 -> 11. The roster is no longer typed here at all: omitting --arms
#     under --tiered makes the launcher resolve the frozen config roster (resolve_cluster_arms),
#     and passing a partial list is now refused before ssh.
#
#  3. EVERY LEG LINE WOULD HAVE DIED AT ARGV PARSING. They passed "--priority -200".."-290", and
#     finding #96 made a negative priority a hard SystemExit unless --allow-deprioritise is given.
#     All ten lines would have exited non-zero and this supervisor would have relaunched them
#     forever at 600s backoff. The ladder is also RETIRED BY R101 (all 11 full-loop models run at
#     EQUAL standing), and the standing rule is that our jobs never sit below full fair-share.
#     So no line passes --priority at all now: the default is 0, which is what R101 requires.
#
#  4. R101 LOCKSTEP WAS NEVER IMPLEMENTED. Legs were pinned at "--seeds 0-29", the leg_seed_tier:30
#     floor that R101 explicitly retired. Every line now climbs the SAME registered ladder
#     [30,100,189,279,340,403,568]: the core and the legs via --tiered, H3 across the full seed
#     range in one line instead of the old manual "re-run later at 0-567" step.
#
#  5. THE GOLD PANEL WAS NOT WHERE THE LAUNCHER LOOKS. --gold-dir defaults to
#     ~/Scratch/llmrp/inputs, which exists on Myriad and is EMPTY; the licensed panel is on ACFS.
#     Passed explicitly below and verified sha256-against-the-frozen-manifest at launch.
# =====================================================================================

param(
    [Parameter(Mandatory = $true)][string]$Line,
    [int]$StaggerSecs = 0,
    # The ACFS path holding the licensed gold. Verified at launch by assert_remote_gold(): the
    # panel must be present AND its sha256 must equal the frozen manifest, or the run refuses.
    [string]$GoldDir = "/acfs/users/ucestes/gold",
    # Nodes fenced off. A node that fails a job in SECONDS is always free, so the scheduler keeps
    # feeding it work: a job vacuum. node-d00a-230 has no apptainer and ate 13 ladder cells.
    # EXTEND THIS if the sentinel's host_failure_concentration check fires, then restart the line.
    [string]$ExcludeHosts = "node-d00a-230",
    # RUN GENERATION (added 2026-07-28, after RUN 1 was invalidated by the cross-line reject-marker
    # collision). A clean re-execution needs a fresh
    # archive on BOTH sides, and BOTH were hardcoded here:
    #   OutDir     - the LOCAL mirror. Also the root every line's driver_status/ and ledger/ live
    #                under, and the root the killswitch reads, so a fresh one also means the
    #                previous run's task deaths are invisible to the new run's admin-kill detector.
    #   RemoteRoot - the SCRATCH working root. This one is not optional: the previous run's jobs
    #                keep archiving into it for hours after a halt, and archive-truth resume would
    #                ADOPT their records - the exact hazard the 2026-07-27 launch gate caught with
    #                8 foreign probe records in the confirmatory search root.
    # The deployed code tree (~/llmrp) and the licensed gold (-GoldDir, on ACFS) do NOT move, so
    # provenance stamps and the gold sha256 are unchanged: same code, same data, same frozen hash.
    [string]$OutDir = "outputs\campaign_cluster",
    [string]$RemoteRoot = "~/Scratch/llmrp",
    # THE DURATION LEVER (2026-08-06). Trainings per TASK, independent of --pack (trainings at
    # ONCE). N>pack runs multiple WAVES in one job, so the job holds its cores ~N/pack times longer
    # at the SAME dispatch rate. That is the only free variable left: measured 2026-08-06, 78
    # pool-d hosts held >=8 free slots while we won ZERO dispatches in two hours, because dispatch
    # order is decided entirely by ntckts and ours is the lowest of any major user but one.
    # 0 = unset = the behaviour this campaign has run with all along, byte for byte.
    [int]$SpecsPerTask = 0,
    # MUST be raised with -SpecsPerTask: run_one warns about extra waves but sizes nothing. At
    # pack 8, N=16 is 2 waves and ~18.2h of work; today's utilisation is 61% of a 15h limit, so
    # 30:0:0 holds the same wall-kill margin. Empty = leave the autosized value alone.
    [string]$HRt = ""
)

$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$outDir   = $OutDir
$stopFile = Join-Path $outDir "STOP_CAMPAIGN"
New-Item -ItemType Directory -Force $outDir | Out-Null
$safe     = ($Line -replace "[^a-zA-Z0-9_-]", "_")
$logFile  = Join-Path $outDir ("supervisor_{0}.log" -f $safe)
# The DRIVER's own stdout/stderr, kept separate from the supervisor's few lines so a 23-day
# driver log never buries the relaunch history that says whether the line is healthy.
$driverLog = Join-Path $outDir ("driver_{0}.log" -f $safe)

$py = Join-Path $repo ".venv\Scripts\python.exe"   # ABSOLUTE (2026-07-18 drill)

# Queue order - the ORDER only (R101 retired the priority ladder; these are all equal now).
# Must match config/preregistration.yaml model_suite.queue_order exactly; tests/test_mode_d.py
# locks the labels across files.
$legTag = [ordered]@{
  "deepseek-v4-pro" = "leg1"; "glm-5.2" = "leg2"; "qwen3.6-27b" = "leg3"; "qwen3.5-9b" = "leg4";
  "haiku-4.5" = "leg5"; "gpt-5.6-luna" = "leg6";
  "nemotron-3-super" = "leg7"; "sonnet-5" = "leg8"; "gemini-2.5-flash" = "leg9"; "kimi-k3" = "leg10"
}

# ---------------------------------------------------------------------------------------------
# THE CPU-LANE SUBSTRATE FLAGS, shared by EVERY line so no line can drift from another.
# Each one is backed by a measurement taken 2026-07-27
# s.4 and the R107 register.
#
#   --device cpu            the confirmatory lane. GPUs are not reachable at our priority floor
#                           (a GPU probe sat queued 42 min with 24 advertised free).
#   --pool d                294 nodes x 36 = 10,584 Intel Xeon cores. The t pool (AMD EPYC) is
#                           excluded by lanes.EXCLUDED_CPU_POOLS: different oneDNN kernels change
#                           float reduction order and would break CRN bit-exactness.
#   --pack 4                4 INDEPENDENT trainings per job, ONE CORE EACH, so 100% efficient
#                           (packing is not threading). 4-core jobs place in ~6 min; 1,000 jobs x 4
#                           = the 4,000-core ceiling that max_u_jobs allows.
#   --cores-per-training 1  the test flood is throughput work: 8 separate 1-thread trainings give
#                           ~104 steps/s aggregate against ~35 for one 8-thread training.
#   --search-pack 1         the SEARCH lane runs one training per job so that, at 8 threads, the
#   --search-threads 8      job asks for max(1,8) x 1 = 8 cores (places in ~19 min) instead of
#                           8 x 4 = 32 (past the placement cliff). 8 threads is the REGISTERED
#                           chain_thread_count (R107, ratified; NEVER exceed 8, 16 is slower), and
#                           it is worth ~2.7x on the two things that gate everything: the 6-step
#                           reflection chain and bayes_opt's 25-step serial GP chain.
#   --chunk-tasks 1         arrays are SERIALISED by policy (tasks 2..n sit in hqw) and pending
#                           tails have twice been PURGED outright. One job per task, no tail.
#   --poll-secs 180         burst cadence.
#   --search-poll-secs 45   chain cadence: every chain handoff pays up to one poll of notice, and
#                           the BO chain has 25 of them.
#   NO --priority           default 0 = full fair-share standing (R101 + the absolute rule).
#   NO --seed-pool-blocks   refused on the CPU lane, and correctly so.
#   NO --arms / --baselines resolved from the FROZEN config; a hand-typed list drifts.
# ---------------------------------------------------------------------------------------------
# --pack 8 (2026-07-31, DEFERRED_FIXES 11 / record sections 50 and 58). C4 ONLY - INERT during
# search, which uses --search-pack 1 (run_search_arm takes pack=(run.search_pack or run.pack)), so
# changing it does not touch the running search phase. At C4 run_test_leg takes run.pack, giving 8
# trainings per 8-slot job instead of 4 per 4-slot job: the SAME job width and therefore the same
# placement profile, for twice the trainings per placed job. Memory renders 2G/slot = 16 GB/job
# (verified against the renderer), so 500 concurrent jobs reserve 7.8 TB against ~12 TB free.
# Pack is OUTSIDE the determinism envelope -- pack-mates are separate spawned processes with
# OMP=1 (330 packed CPU baselines in this run prove the path) -- so this buys wall-clock without
# touching arithmetic. Measured justification: at the cores held on 2026-07-31 rung 568 lands
# 08-30, MISSING the 08-27 exogenous stop; pack 8 roughly halves the C4 makespan.
# ---------------------------------------------------------------------------------------------
# *** PREPARED AND DELIBERATELY NOT APPLIED - POOL WIDENING d -> db (2026-08-02, RUN 14, D30).
#
# TO APPLY, CHANGE ONE TOKEN BELOW:   "--pool", "d",   ->   "--pool", "db",
# then restart one supervisor (nemotron first) and let the fenced watchdog revive it on this script.
#
# WHY IT IS NOT ALREADY APPLIED. A supervisor builds this array ONCE at launch, so the change is
# INERT until a supervisor is RESTARTED - and process termination is BLOCKED for the agent (both
# taskkill and Stop-Process were refused by the harness classifier on 2026-08-02). Committing the
# widened value would therefore have left drift=1 against RUNNING_SHA indefinitely, on an invariant
# whose whole value is that it never changes, while buying no capacity at all until someone restarts
# a line. So the measurement and the safety proof are banked HERE, at the decision point, and the
# one-token edit is left for whoever can restart a supervisor. See DEFERRED_FIXES_RUN4 D30.
#
# POOL WIDENING d -> d+b is re-opened on the 2026-07-31 decline's OWN stated condition ("re-open only
# if pool d's own capacity becomes the binding constraint"). It now is, and the numbers below are
# measured on the day, not inherited:
#
#   PLACEABLE at pack 8, after excluding PAID and disabled hosts and gating on memory (2G/slot):
#     pool d (d00a+d00b) ..... 3 jobs =  24 cores   <- 82% of its usable >=8-slot hosts have <16G free
#     pool b (b00a) .......... 11 jobs = 88 cores   <- 0% memory-blocked; b00a hosts carry 1.5T RAM
#
# So pool b can give us ~3.7x what pool d itself can still give us. (The earlier +592-core figure
# came from `pool_capacity_compare.py`, which reads free slots from `qhost` host counters and so
# counts DISABLED hosts and ignores both fragmentation and memory
# is the correction and reports 88, not 216, for pool b.)
#
# SAFETY - THE ONE FACT THIS RESTS ON, VERIFIED FIRST-HAND RATHER THAN INHERITED. The C3 gate
# enforces per-seed substrate homogeneity, where the substrate string is `cpu model | omp | threads
# | cuda`. A probe job run on node-b00a-013 on 2026-08-02 reported:
#     Intel(R) Xeon(R) Gold 6240 CPU @ 2.60GHz, 2 sockets, 36 cores, avx512f=1
# which is the SAME model string pool d reports, so widening cannot make a comparison unit
# heterogeneous. This is the determinism envelope's requirement (a substrate change is a
# DESIGN decision) satisfied by identity, not by tolerance: same silicon, same reduction order.
#
# WHY "db" AND NOT "d,b": the site JSV maps the `allow=` context onto a wildcard PE. Verified live -
# `qsub -ac allow=db` is granted PE `smp-[BD]*` (spanning both pools), whereas `-pe smp-B` is
# rejected outright by policyjsv. Pool e is REFUSED on measurement, not on theory: four real
# submissions with `allow=e` were rejected with "Unable to find a place to run this job", so e00a's
# apparent 328 cores are not reachable by us and must not be counted.
#
# ROLLOUT IS GRADUAL BY CONSTRUCTION, WHICH IS WHY IT IS SAFE. `run_batch` submits only when the
# batch has NO live job in the queue (driver.py, the alive-names guard), so a restarted line does
# not resubmit its in-flight work; already-queued jobs keep `allow=d` and drain normally, and only
# the NEXT batch boundary picks up the wider pool.
$cpuLane = @(
  "--device", "cpu", "--pool", "d",
  "--pack", "8", "--cores-per-training", "1",
  "--search-pack", "1", "--search-threads", "8",
  "--chunk-tasks", "1",
  "--exclude-hosts", $ExcludeHosts,
  "--gold-dir", $GoldDir,
  "--remote-root", $RemoteRoot,
  "--poll-secs", "180", "--search-poll-secs", "45",
  "--output-dir", $outDir, "--resume"
)

# THE CANARY SEAM. Both default to off, so an unmodified launch produces a BYTE-IDENTICAL argument
# vector and this file cannot change any line that does not explicitly ask for it. That property is
# load-bearing rather than tidy: watchdog_fenced.ps1 may relaunch ANY supervisor at any moment,
# including the core line that carries the entire reported result.
if ($SpecsPerTask -gt 0) { $cpuLane += @("--specs-per-task", "$SpecsPerTask") }
if ($HRt -ne "")         { $cpuLane += @("--h-rt", $HRt) }

if ($Line -eq "h3") {
    # THE H3 FLOOR UNIT: the pre-registered single-shot control (generations=1, no reflection),
    # same candidate budget and agent config. Its own invocation, archived to disjoint
    # *_h3_singleshot/ roots. --tiered is refused here by design (C5 vs C1-C4), so R101 lockstep is
    # expressed as the FULL seed range in ONE line rather than the old "0-29 now, 0-567 later"
    # two-step, which was a manual dependency on every rung bank.
    $driverArgs = @(
      "scripts/run_campaign_cluster.py", "--h3-singleshot",
      "--seeds", "0-567", "--pass-mode", "B", "--llm-from", "campaign",
      "--batch-tag", "h3ss"
    ) + $cpuLane
} elseif ($Line -eq "core") {
    # THE CORE LINE - the frozen 9-arm roster (RESOLVED, never typed) + the frozen 11-name H1 canon
    # (also resolved), climbing the registered assurance ladder with the rungs pipelined.
    $driverArgs = @(
      "scripts/run_campaign_cluster.py", "--tiered",
      "--pass-mode", "B", "--llm-from", "campaign", "--pipeline-rungs",
      "--batch-tag", "c1"
    ) + $cpuLane
} elseif ($legTag.Contains($Line)) {
    # A REPLICATION LEG - the identical five LLM feedback arms authored by this leg's pinned model,
    # archived under leg_<label>/ (forced by --leg). It carries NO H1 canon (those eleven rewards
    # are hand-designed and model-INDEPENDENT, so they belong to the core line exactly once) and no
    # DFO arm. Under R101 it climbs the SAME ladder as the core, at the SAME priority.
    #
    # 2026-08-01 (RUN 11) - --pipeline-rungs ADDED. It was on the core line only, so the comment
    # above ("the SAME ladder as the core") was true of WHAT the legs run and false of HOW they
    # submit it: eleven of twelve lines drained each C4 assurance block before submitting the next.
    # src/cluster/campaign.py says what that costs, in its own words: pipelining "keeps the eligible
    # backlog deep at every instant (idle-window harvesting - the sequential path FORFEITS CAPACITY
    # during every block's drain)".
    #
    # WHY IT MATTERS NOW AND NOT BEFORE, measured 2026-08-01: the campaign is 79% through the SEARCH
    # phase (1,417 of 1,800 trainings), which is intrinsically narrow because generations are
    # sequential - so the current ~870-slot holding is largely structural, not a defect. But the C4
    # sweep is 40,328 of the 42,128 trainings, it is embarrassingly parallel, it is 2.6% done, and
    # it is IMMINENT. Entering the wide phase on the sequential path is exactly when the drain
    # bubbles cost most. Measured the same day: 4,497 slots free cluster-wide, 3,366 of them in
    # pool D which we already use, against our 870 held and 0-2 queued.
    #
    # IT CHANGES NO REGISTERED QUANTITY. Submission order and timing only - same arms, same seeds,
    # same specs, same arithmetic, same priority (0). P17 banking is preserved where it binds: a
    # rung banks only when it AND every rung below it are complete, so a broken lower level can
    # never carry results. It also REMOVES an asymmetry rather than adding one - the core line has
    # pipelined all along, so the legs were the odd ones out on a ladder R101 requires them to climb
    # in lockstep with it.
    $driverArgs = @(
      "scripts/run_campaign_cluster.py", "--leg", $Line, "--tiered",
      "--pass-mode", "B", "--pipeline-rungs",
      "--batch-tag", $legTag[$Line]
    ) + $cpuLane
} else {
    Write-Host ("unknown line '{0}' - use 'core', 'h3', or a leg label from config/legs.yaml" -f $Line)
    exit 2
}

function Log([string]$msg) {
    $l = "{0} | supervisor[{1}] | {2}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Line, $msg
    Write-Host $l
    Add-Content -Path $logFile -Value $l
}

if ($StaggerSecs -gt 0) {
    # Poll staggering: twelve lines polling qstat/rsync in lockstep would burst the login node -
    # offset each line's start so the poll phases spread out.
    Log ("staggering start by {0}s" -f $StaggerSecs)
    Start-Sleep -Seconds $StaggerSecs
}

$attempt = 0
$maxAttempts = 1000
$backoffSecs = 600

Log ("line supervisor started: {0}" -f ($driverArgs -join " "))
while ($attempt -lt $maxAttempts) {
    if (Test-Path $stopFile) { Log "STOP_CAMPAIGN present - deliberate stop."; break }
    $attempt += 1
    Log ("attempt {0}: launching the driver" -f $attempt)
    $rc = -1
    try {
        # CAPTURE THE DRIVER'S OWN OUTPUT (2026-07-28, added during the launch itself).
        # Without this the driver's stdout/stderr goes only to the spawned console window, so its
        # WARNINGs are unreadable to anyone not sitting at the machine. That gap bit immediately:
        # within the first hour the heartbeat reported "N consecutive pull/ops failures" and the
        # driver had logged the exact exception behind it -- to a window nobody can read. For a
        # 23-day unattended run the diagnosis has to survive in a file. Tee, not redirect, so the
        # console still shows it live.
        # Out-File -Encoding utf8, NOT Tee-Object: PowerShell 5.1's Tee writes UTF-16LE, which
        # makes every grep/Select-String over the log return nothing (each character separated by a
        # NUL byte). A log you cannot search is not a log. The console copy is not worth having --
        # nobody reads the spawned window, which is the whole reason this redirect exists.
        & $py @driverArgs 2>&1 | Out-File -FilePath $driverLog -Encoding utf8 -Append
        if ($null -ne $LASTEXITCODE) { $rc = $LASTEXITCODE }
    } catch {
        Log ("driver INVOCATION failed before start: {0}" -f $_.Exception.Message)
        $rc = -1
    }
    if ($rc -eq 0) { Log "driver exited 0 - LINE COMPLETE."; break }
    # D12 (applied 2026-08-01, record s.97). A GATE STOP IS NOT A SUCCESS AND IS NOT A CRASH.
    # Before this branch existed the driver returned 0 on a gate stop, so the line above logged
    # "LINE COMPLETE" and exited - six legs reported complete on 2026-07-29 having produced nothing.
    # The driver now returns 3 (EXIT_AWAITING_REVIEW), and WITHOUT this branch that would fall
    # through to the relaunch line below and spin the line in a backoff loop forever.
    #
    # URGENT BECAUSE OF D16 (same commit): folding the substrate census into the gate's health_ok
    # makes stops MORE likely, so an unhandled stop code would have turned the CONFIRMATORY line
    # into a silent relaunch loop. The two fixes are hard-coupled.
    #
    # Release: review the effect-blind report, then create TIER1_APPROVED_<line_tag> under the read
    # root and re-run with --approve-tier1 --resume. The watchdog decides "dead line" by process
    # ABSENCE, not by exit code (verified in mode_d_watchdog.ps1),
    # so breaking here does NOT trigger an automatic revival of this supervisor.
    if ($rc -eq 3) {
        Log "driver exited 3 - STOPPED AT THE REVIEW GATE, awaiting approval. NOT relaunching."
        break
    }
    Log ("driver exited {0} - relaunching in {1}s; Myriad arrays unaffected" -f $rc, $backoffSecs)
    Start-Sleep -Seconds $backoffSecs
}
Log "line supervisor exiting."
