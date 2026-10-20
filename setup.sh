#!/usr/bin/env bash
# One-time setup. No GPU and no root needed (can run on a login node). ~30-60 min, mostly downloads (~60 GB of models).
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$WORK/bin" "$RESULTS/logs"
log(){ echo "[setup $(date '+%F %T')] $*" | tee -a "$RESULTS/logs/setup.log"; }
command -v git >/dev/null || { log "ERROR: git is required"; exit 1; }

# 1. uv (Python package manager), installed inside $WORK
if ! command -v uv >/dev/null; then
  log "installing uv $UV_VERSION into $WORK/uv"
  if command -v curl >/dev/null; then curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_INSTALL_DIR="$WORK/uv" UV_NO_MODIFY_PATH=1 sh
  else wget -qO- "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_INSTALL_DIR="$WORK/uv" UV_NO_MODIFY_PATH=1 sh; fi
fi
log "uv: $(uv --version)"

# 2. Cosmos-Transfer2.5 code at the tested commit
if [ ! -d "$COSMOS/.git" ]; then log "cloning cosmos-transfer2.5"; git clone -q https://github.com/nvidia-cosmos/cosmos-transfer2.5.git "$COSMOS"; fi
git -C "$COSMOS" fetch -q origin || true
git -C "$COSMOS" checkout -q "$COSMOS_COMMIT"
log "cosmos-transfer2.5 at $(git -C "$COSMOS" rev-parse --short HEAD)"

# 3. Python 3.10 environment from the repo's lock file (the flash-attn CUDA 12.8 wheel exists for Python 3.10)
log "building the Python environment (uv sync --extra=cu128)"
(cd "$COSMOS" && uv python install 3.10 && uv sync --extra=cu128 --python 3.10) 2>&1 | tail -5 | tee -a "$RESULTS/logs/setup.log"

# 4. ffmpeg: use the system one, else the copy bundled with the imageio-ffmpeg wheel
if ! command -v ffmpeg >/dev/null; then
  ln -sf "$("$PY" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" "$WORK/bin/ffmpeg"; log "ffmpeg: using imageio-ffmpeg's binary"
fi

# 5. Low-RAM text-encoder loader: only on machines with < 64 GB RAM (or LOWRAM_PATCH=1). Same weights; it only streams them to the GPU.
MEM_GB=$(awk '/MemTotal/{print int($2/1048576)}' /proc/meminfo)
if [ "${LOWRAM_PATCH:-0}" = 1 ] || [ "$MEM_GB" -lt 64 ]; then
  if git -C "$COSMOS" apply --check "$BUNDLE/patches/text_encoder_lowram.patch" 2>/dev/null; then git -C "$COSMOS" apply "$BUNDLE/patches/text_encoder_lowram.patch"; log "applied low-RAM text-encoder patch (RAM ${MEM_GB} GB)"; fi
fi

# 6. Hugging Face access + model downloads (exact revisions used by this Cosmos commit)
"$PY" "$BUNDLE/tools/predownload.py" 2>&1 | tee -a "$RESULTS/logs/setup.log"
log "SETUP DONE. Next, on the GPU node:  bash check.sh"
