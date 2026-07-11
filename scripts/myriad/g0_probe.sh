#!/bin/bash -l
# G0 PROBE — runs ON a Myriad login node.
# Read-only reconnaissance: answers every VERIFY-AT-G0 unknown in one pass.
# Usage (from the laptop, once key auth works):  ssh myriad 'bash -s' < scripts/myriad/g0_probe.sh
set -u
echo "==== G0 PROBE $(date) on $(hostname) ===="

echo "--- [1] identity + shell ---"
whoami; echo "HOME=$HOME"

echo "--- [2] quotas (lquota preferred per Data_Storage docs; gquota fallback) + inode/file count ---"
lquota 2>&1 | head -10 || gquota 2>&1 | head -10 || echo "quota cmd not found"

echo "--- [3] outbound HTTPS from the LOGIN node (Architecture A'/B decision) ---"
for url in https://api.anthropic.com https://pypi.org; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$url" 2>&1) && \
    echo "curl $url -> HTTP $code" || echo "curl $url -> FAILED (no outbound)"
done

echo "--- [4] scheduler present + queue snapshot ---"
which qsub qstat 2>&1
qstat -g c 2>&1 | head -15 || true

echo "--- [5] GPU node types visible to the scheduler ---"
qhost -l gpu=1 2>/dev/null | head -8 || echo "(qhost gpu listing not available; will use probe jobs)"

echo "--- [6] modules: newest python + cuda on offer ---"
module avail python 2>&1 | tail -5
module avail cuda 2>&1 | tail -5

echo "--- [7] apptainer/singularity (the container fallback) ---"
which apptainer singularity 2>&1; apptainer --version 2>&1 || singularity --version 2>&1 || true

echo "--- [8] ACFS present? ---"
ls -d /acfs/users/$(whoami) 2>&1 || echo "(no ACFS dir yet — may need requesting)"

echo "--- [9] scratch layout ---"
ls -d ~/Scratch 2>&1; df -h ~ 2>&1 | tail -2

echo "--- [10] GPU-node probe: driver + CGROUP ISOLATION (the §15 packing dependency) + outbound ---"
# THE load-bearing check: under -l gpu=1 the device cgroup (UCL-wide since 2022-08-10) must expose
# EXACTLY ONE GPU as CUDA_VISIBLE_DEVICES index 0. If so, packing N processes on it is safe (no other
# job can land on our GPU) and cuda:0 is correct. n>1 or an empty CUDA_VISIBLE_DEVICES = investigate.
timeout 300 qrsh -l gpu=1,h_rt=0:10:0 -ac allow=EF -now n \
  'hostname; echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; nvidia-smi -L; \
   n=$(nvidia-smi -L | grep -c GPU); echo "GPUs VISIBLE under gpu=1: $n (EXPECT 1 => cgroup isolation => packing safe)"; \
   nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; \
   curl -s -o /dev/null -w "compute-node outbound: HTTP %{http_code}\n" --max-time 8 https://api.anthropic.com || echo "compute-node outbound: NONE"' \
  2>&1 || echo "(qrsh probe skipped/queued-out — rerun later or submit as a batch probe)"

echo "--- [11] (Stage-2 only) A100 node: MIG status — reveals if hardware GPU partitioning is enabled ---"
# nvidia-smi -L on an -ac allow=L (A100) job lists MIG instances IF MIG mode is on (near-certainly OFF on
# a shared research cluster; then Stage-2 packs the full cgroup GPU via time-slice/MPS). See the
# advanced-capabilities assessment of the cluster.
timeout 300 qrsh -l gpu=1,h_rt=0:5:0 -ac allow=L -now n \
  'nvidia-smi -L; nvidia-smi --query-gpu=name,mig.mode.current --format=csv,noheader 2>/dev/null || true' \
  2>&1 || echo "(A100 MIG probe skipped/queued-out — optional; Stage-1 uses the V100 pool)"

echo "==== G0 PROBE COMPLETE ===="
