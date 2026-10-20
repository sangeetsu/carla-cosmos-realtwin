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
