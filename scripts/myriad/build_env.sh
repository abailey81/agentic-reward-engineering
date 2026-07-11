#!/bin/bash -l
# Build the Myriad runtime venv for llm-reward-portfolio (PLAN §7 Phase B, R12).
#
# WHY a venv (not the central modules): Myriad's central stack is OLD (CUDA 11.2/11.3, torch 1.11)
# — far below the validated build (torch 2.6.0+cu124, ADR-030/032). We therefore bundle our own
# runtime: a fresh venv + the cu124 wheels (which vendor the CUDA runtime, so only a recent-enough
# NVIDIA *driver* is needed on the compute node, checked by the --smoke step below).
#
# USAGE (on a login node — pip install is well under the 1 CPU-hour login limit):
#   ssh myriad 'bash -l' < scripts/myriad/build_env.sh              # build the venv
#   ssh myriad 'bash -l -s' -- --smoke < scripts/myriad/build_env.sh   # + import/CUDA smoke via qrsh
#
# The repo itself must already be on the cluster (PLAN §12 B-B step 4:
#   git archive HEAD | ssh myriad "mkdir -p ~/llmrp && tar -x -C ~/llmrp"), with requirements.lock.
#
# FALLBACK (R12): if the node NVIDIA driver is too old for the cu124 wheels (the --smoke CUDA check
# fails), do NOT fight it — build the Plan-B Apptainer image instead (it vendors a matching driver
# userspace) and run jobs with `apptainer exec --nv`. This script prints that instruction on failure.
set -euo pipefail

REPO="${LLMRP_REPO:-$HOME/llmrp}"
VENV="${LLMRP_VENV:-$HOME/venvs/llmrp}"
TORCH_VERSION="2.6.0"
CU_INDEX="https://download.pytorch.org/whl/cu124"
SMOKE=0
[ "${1:-}" = "--smoke" ] && SMOKE=1

echo "=== llm-reward-portfolio :: Myriad env build ==="
echo "repo=$REPO  venv=$VENV  torch=$TORCH_VERSION (cu124)"

# --- 1. a clean, recent Python (purge the old central modules; try known-good names) --------------
module purge 2>/dev/null || true
_loaded_py=""
for m in python3/3.11 python/3.11 python3/3.12 python/3.12 python3/recommended python3; do
  if module load "$m" 2>/dev/null; then _loaded_py="$m"; break; fi
done
echo "--- python module: ${_loaded_py:-<none loaded; using PATH python3>} ---"
PY="$(command -v python3 || true)"
[ -n "$PY" ] || { echo "FATAL: no python3 on PATH after module load"; exit 2; }
"$PY" --version
case "$("$PY" -c 'import sys;print(f"{sys.version_info[0]}.{sys.version_info[1]}")')" in
  3.11|3.12) : ;;
  *) echo "WARNING: python is not 3.11/3.12 (requires-python = >=3.11,<3.13) — the venv may mis-resolve"; ;;
esac

# --- 2. the venv (idempotent: reuse if already valid) --------------------------------------------
if [ -x "$VENV/bin/python" ]; then
  echo "--- venv exists at $VENV — reusing (delete it to rebuild) ---"
else
  echo "--- creating venv at $VENV ---"
  mkdir -p "$(dirname "$VENV")"
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel >/dev/null

# --- 3. torch first (pinned cu124 build), then the locked deps -----------------------------------
echo "--- installing torch $TORCH_VERSION (+cu124) ---"
pip install "torch==$TORCH_VERSION" --index-url "$CU_INDEX"
if [ -f "$REPO/requirements.lock" ]; then
  echo "--- installing requirements.lock (torch already satisfied) ---"
  pip install -r "$REPO/requirements.lock"
else
  echo "WARNING: $REPO/requirements.lock absent — install the repo deps manually (pip install -e $REPO)"
fi
# Install the repo itself so `import src` works from ANY cwd (belt-and-suspenders: the jobscript
# also exports PYTHONPATH={repo_root}, so jobs import even if this editable install is skipped).
if [ -f "$REPO/pyproject.toml" ]; then
  echo "--- pip install -e the repo (src importable venv-wide) ---"
  pip install -e "$REPO" --no-deps
fi
echo "--- installed torch: $(python -c 'import torch;print(torch.__version__)') ---"

# --- 4. optional GPU smoke (must run on a COMPUTE node — login nodes have no GPU) -----------------
if [ "$SMOKE" = "1" ]; then
  echo "--- GPU smoke via qrsh (1 GPU, 10 min) — proves the node driver drives the cu124 build ---"
  SMOKE_PY="import torch; assert torch.cuda.is_available(), 'CUDA not available on this node'; \
print('cuda', torch.version.cuda, 'device', torch.cuda.get_device_name(0)); \
x=torch.randn(1024,1024,device='cuda'); print('matmul ok', float((x@x).sum()))"
  if qrsh -l gpu=1,h_rt=0:10:0 -ac allow=EF \
       "source $VENV/bin/activate && python -c \"$SMOKE_PY\""; then
    echo "=== SMOKE PASSED: the cu124 wheels drive this node's GPU ==="
  else
    echo "=== SMOKE FAILED ==="
    echo "The node NVIDIA driver is likely too old for the cu124 wheels (R12)."
    echo "FALLBACK: run jobs in the Plan-B Apptainer image (it vendors a matching driver userspace):"
    echo "  1. Get the Plan-B image ONTO Myriad (a LOCALLY-built docker image is NOT reachable via"
    echo "     docker:// here). Either push docker/Dockerfile.planb to a registry, or transfer it:"
    echo "       (on a docker host) docker build -f docker/Dockerfile.planb -t llm-rp-planb ."
    echo "       docker save llm-rp-planb | gzip > planb.tar.gz  &&  scp planb.tar.gz myriad:~/"
    echo "  2. Build the .sif on a LOCAL filesystem — Apptainer CANNOT build on home/Scratch:"
    echo "       export APPTAINER_TMPDIR=\"\$TMPDIR\"; cd \"\$TMPDIR\""
    echo "       apptainer build \"\$TMPDIR/llmrp.sif\" docker-archive://\$HOME/planb.tar.gz  # or docker://<registry>/img"
    echo "       mv \"\$TMPDIR/llmrp.sif\" ~/llmrp.sif"
    echo "  3. Run: render_jobscript(..., apptainer_sif='~/llmrp.sif')  # jobscript uses 'apptainer exec --nv'"
    exit 3
  fi
fi

echo "=== env build complete: $VENV ==="
echo "next: a keyless dry-run through src.cluster (G1 cert), then the fps/queue benchmark."
