#!/usr/bin/env bash
# Sourced by every script. Override any variable before running, e.g.  WORK=/scratch/$USER/realtwin bash setup.sh
_B="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BUNDLE="$_B"
export WORK="${WORK:-$_B/work}"                 # Cosmos code, Python env, model cache (~100 GB)
export RESULTS="${RESULTS:-$_B/results}"        # everything to send back
export COSMOS="$WORK/cosmos-transfer2.5"
export COSMOS_COMMIT=2ff49d0717af02057ae79bc75c00fbff9da1b4e7   # the commit tested on the source workstation
export UV_VERSION=0.13.4
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORK/uv_cache}" UV_PYTHON_INSTALL_DIR="$WORK/uv_python"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$WORK/hf_cache}" # model downloads; your HF login/token location is left untouched
export PATH="$WORK/uv:$WORK/bin:$PATH"
export MALLOC_MMAP_THRESHOLD_=131072
export COSMOS_EXPERIMENTAL_CHECKPOINTS=1        # registers the distilled edge model (used by one small job); default checkpoints unchanged
export GIT_LFS_SKIP_SMUDGE=1                    # Cosmos example assets are not needed
_NVRTC="$COSMOS/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc"
[ -d "$_NVRTC" ] && export CUDA_HOME="$_NVRTC"  # same as the tested setup (lets torch find libnvrtc without a system CUDA toolkit)
export PY="${PY:-$COSMOS/.venv/bin/python}"
export COSMOS_CACHE_DIR="${COSMOS_CACHE_DIR:-$WORK/cosmos_cache}"  # Cosmos's own helper-model cache (on-the-fly depth weights)
export SIF="${SIF:-$WORK/ubuntu-22.04.sif}"                     # container image used on hosts with an old C library (see below)

# NVIDIA's Transformer Engine wheel (pinned by Cosmos) needs glibc >= 2.29. RHEL/Rocky-8-family hosts, including ASU Sol, have 2.28.
# There, the GPU scripts re-run themselves inside an Ubuntu 22.04 Apptainer image (glibc 2.35) with the GPU passed through;
# the Python environment, models and results stay where they are. USE_CONTAINER=1 forces this, USE_CONTAINER=0 disables it.
needs_container() {
  [ -n "${APPTAINER_CONTAINER:-}${SINGULARITY_CONTAINER:-}" ] && return 1      # already inside
  case "${USE_CONTAINER:-auto}" in 0) return 1 ;; 1) return 0 ;; esac
  local v; v=$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$')
  [ -n "$v" ] && [ "$(printf '%s\n2.29\n' "$v" | sort -V | head -1)" != "2.29" ]
}
run_in_container() {   # usage: run_in_container /path/to/script.sh [args...]   (does not return)
  local A; A=$(command -v apptainer || command -v singularity) || { echo "ERROR: glibc here is older than 2.29 and apptainer is not available on this node"; exit 1; }
  [ -e "$SIF" ] || { echo "ERROR: container image $SIF not found. Run setup.sh first (it creates it), or set SIF=/path/to/ubuntu-22.04.sif"; exit 1; }
  mkdir -p "$RESULTS"
  local binds="" d
  for d in "$BUNDLE" "$WORK" "$RESULTS" "$HF_HUB_CACHE" /scratch; do
    [ -d "$d" ] || continue
    case "$d/" in "$HOME"/*) ;; *) binds="$binds -B $d" ;; esac             # home is bound by default
  done
  echo "[info] glibc $(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9.]+$') on this host: running inside $SIF"
  exec "$A" exec --nv $binds "$SIF" bash "$@"
}
has_internet() { "$PY" -c "import urllib.request; urllib.request.urlopen('https://huggingface.co', timeout=8)" > /dev/null 2>&1; }
