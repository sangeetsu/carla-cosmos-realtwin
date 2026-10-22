#!/usr/bin/env bash
# One-time setup. No GPU and no root needed (can run on a login node). ~30-60 min, mostly downloads (~60 GB of models).
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$WORK/bin" "$RESULTS/logs"
log(){ echo "[setup $(date '+%F %T')] $*" | tee -a "$RESULTS/logs/setup.log"; }
command -v git >/dev/null || { log "ERROR: git is required"; exit 1; }

# 1. uv (Python package manager), always our own pinned copy inside $WORK (it must also be visible inside the container)
if [ ! -x "$WORK/uv/uv" ]; then
  log "installing uv $UV_VERSION into $WORK/uv"
  if command -v curl >/dev/null; then curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_INSTALL_DIR="$WORK/uv" UV_NO_MODIFY_PATH=1 sh
  else wget -qO- "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_INSTALL_DIR="$WORK/uv" UV_NO_MODIFY_PATH=1 sh; fi
fi
log "uv: $("$WORK/uv/uv" --version)"

# 2. Cosmos-Transfer2.5 code at the tested commit
if [ ! -d "$COSMOS/.git" ]; then log "cloning cosmos-transfer2.5"; git clone -q https://github.com/nvidia-cosmos/cosmos-transfer2.5.git "$COSMOS"; fi
git -C "$COSMOS" fetch -q origin || true
git -C "$COSMOS" checkout -q "$COSMOS_COMMIT"
log "cosmos-transfer2.5 at $(git -C "$COSMOS" rev-parse --short HEAD)"

# 3. Python 3.10 environment from the repo's lock file (the flash-attn CUDA 12.8 wheel exists for Python 3.10)
log "building the Python environment (uv sync --extra=cu128)"
(cd "$COSMOS" && uv python install 3.10 && uv sync --extra=cu128 --python 3.10) 2>&1 | tail -5 | tee -a "$RESULTS/logs/setup.log"

# 3b. OpenCV: keep only the headless build. The lock installs both opencv-python and opencv-python-headless into one cv2 folder,
#     and whichever lands last wins; the desktop build needs X11/OpenGL/GLib libraries (e.g. libgthread-2.0) that minimal
#     systems and the container image lack. Same version, same functions, no GUI.
OCV=$("$PY" -c "import importlib.metadata as m; print(m.version('opencv-python-headless'))")
"$WORK/uv/uv" pip uninstall --python "$PY" opencv-python > /dev/null 2>&1 || true
"$WORK/uv/uv" pip install --python "$PY" --reinstall --no-deps "opencv-python-headless==$OCV" 2>&1 | tail -1 | tee -a "$RESULTS/logs/setup.log"
"$PY" -c "import cv2" && log "OpenCV $OCV (headless) OK"

# 4. ffmpeg: the static binary bundled with the imageio-ffmpeg wheel (works on the host and inside the container)
ln -sf "$("$PY" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" "$WORK/bin/ffmpeg"; log "ffmpeg: $("$WORK/bin/ffmpeg" -version | head -1)"

# 4b. Hosts with glibc < 2.29 (e.g. ASU Sol): an Ubuntu 22.04 image for the GPU runs (see env.sh)
if needs_container; then
  if [ -f "$SIF" ]; then log "container image present: $SIF"
  else
    A=$(command -v apptainer || command -v singularity) || { log "ERROR: glibc < 2.29 here and apptainer is not on this node. Run setup.sh on a compute node (e.g. the lightwork partition), or set SIF to an existing ubuntu-22.04.sif"; exit 1; }
    log "glibc < 2.29: pulling an Ubuntu 22.04 container image to $SIF"; "$A" pull "$SIF" docker://ubuntu:22.04
  fi
fi

# 5. Low-RAM text-encoder loader: streams the text-encoder weights to the GPU instead of staging all of them in host RAM.
#    Same weights; ~18 GB peak host RAM per generation instead of > 22 GB. Applied everywhere: it is the tested configuration.
if git -C "$COSMOS" apply --check "$BUNDLE/patches/text_encoder_lowram.patch" 2>/dev/null; then git -C "$COSMOS" apply "$BUNDLE/patches/text_encoder_lowram.patch"; log "applied low-RAM text-encoder patch"
else log "low-RAM text-encoder patch already in place"; fi

# 6. Hugging Face access + model downloads (exact revisions used by this Cosmos commit)
"$PY" "$BUNDLE/tools/predownload.py" 2>&1 | tee -a "$RESULTS/logs/setup.log"
# 7. Inside the container (hosts with glibc < 2.29): the libraries that need system files must import there too
if needs_container; then
  A=$(command -v apptainer || command -v singularity)
  "$A" exec $(for d in "$BUNDLE" "$WORK"; do case "$d/" in "$HOME"/*) ;; *) printf -- "-B %s " "$d";; esac; done) "$SIF" \
    "$PY" -c "import cv2, av, decord, imageio_ffmpeg, numpy, PIL; print('container imports OK')" 2>&1 | tail -1 | tee -a "$RESULTS/logs/setup.log"
fi
log "SETUP DONE. Next: on Sol, bash slurm/submit_chain.sh 4 ; on a workstation, bash check.sh then bash run_overnight.sh"
