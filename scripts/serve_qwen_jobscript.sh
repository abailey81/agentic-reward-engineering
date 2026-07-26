#!/bin/bash -l
# =====================================================================================
# A5 self-host SERVING jobscript (UCL Myriad / SGE) -- serve the pinned bf16 Qwen leg.
#
# WHAT: one LONG-RUNNING single GPU task that runs scripts/serve_qwen_selfhost.py, which
# launches `vllm serve` with --revision <hf_commit> (ENFORCES the reproducibility pin),
# --dtype bfloat16, and thinking-off, and writes served-manifest-*.json (serving-node
# provenance). It is NOT the training array (src/cluster/jobscript.py, run_one): a vLLM
# server holds a PORT for the whole campaign, so it needs its own single-task template.
#
# USAGE (set the runtime env, override the resource directives on the qsub line for the
# CURRENT Myriad allocation, then submit):
#   qsub -l h_rt=48:00:00 -ac allow=<A100_pool> \
#        -v VLLM_LEG=qwen3.5-9b,VLLM_PORT=8000,VLLM_API_KEY=<dummy>,VLLM_SIF=<image.sif> \
#        scripts/serve_qwen_jobscript.sh
# Then, for the campaign driver on the head node:
#   export VLLM_BASE_URL=http://<served-node>:8000/v1   VLLM_API_KEY=<dummy>
# The served node:port is written to serve-endpoint-<JOB_ID>.txt for discovery.
#
# NOTE: The SGE resource directives below (h_rt / mem / pool) are Myriad-allocation-specific --
# CONFIRM against the live pool state and override
# on the qsub line as needed. The A100-80G U/V pools are the campaign-speed default.
# =====================================================================================
#$ -N qwen_selfhost_serve
#$ -l gpu=1
#$ -l h_rt=48:00:00
#$ -l mem=32G
#$ -cwd
#$ -j y
#$ -o serve_qwen_$JOB_ID.log

set -euo pipefail

: "${VLLM_PORT:=8000}"
: "${VLLM_LEG:=qwen3.5-9b}"
: "${VLLM_API_KEY:?set VLLM_API_KEY (any non-empty dummy string; the client + vLLM auth)}"

# OFFLINE WEIGHTS (2026-07-26). Myriad compute nodes have NO internet, so `vllm serve --revision`
# must NOT try to fetch: the pinned revision is pre-staged on a LOGIN node into a SHARED HF_HOME
# (`python -m scripts.serve_qwen_selfhost --prestage --leg <leg>`), and the serve runs offline.
# Without this the job dies AFTER the scarce GPU allocation is granted, with a network error that
# blames vLLM rather than the missing stage. serve_qwen_selfhost.py preflights and refuses early.
: "${HF_HOME:=$HOME/Scratch/hf}"
export HF_HOME
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
echo "[serve_qwen] HF_HOME=${HF_HOME} HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"

NODE="$(hostname -f)"
ENDPOINT="http://${NODE}:${VLLM_PORT}/v1"
echo "[serve_qwen] node=${NODE} port=${VLLM_PORT} leg=${VLLM_LEG} job=${JOB_ID:-local}"
echo "${ENDPOINT}" > "serve-endpoint-${JOB_ID:-local}.txt"
echo "[serve_qwen] driver: export VLLM_BASE_URL=${ENDPOINT}"

# Serve inside the pinned Apptainer image if provided (--nv exposes the GPU); else host venv.
if [ -n "${VLLM_SIF:-}" ]; then
  # APPTAINER-PRESENCE GUARD (2026-07-26), mirroring src/cluster/jobscript.py. /usr/bin/apptainer is
  # MISSING on some nodes (measured: node-d00a-230), and the venv python lives INSIDE the .sif, so a
  # missing container burns the granted GPU slot with a bare rc=127 and no diagnosis. Fail NAMED.
  command -v apptainer >/dev/null 2>&1 || {
    echo "FATAL apptainer missing on $(hostname) - cannot start the vLLM container" >&2; exit 127; }
  [ -f "${VLLM_SIF}" ] || {
    echo "FATAL VLLM_SIF not found: ${VLLM_SIF} (stage the image on a shared path first)" >&2
    exit 127; }
  exec apptainer exec --nv "${VLLM_SIF}" \
    python -m scripts.serve_qwen_selfhost --leg "${VLLM_LEG}" --port "${VLLM_PORT}"
else
  exec python -m scripts.serve_qwen_selfhost --leg "${VLLM_LEG}" --port "${VLLM_PORT}"
fi
