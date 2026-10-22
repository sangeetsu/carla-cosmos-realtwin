#!/usr/bin/env bash
# GPU pre-flight, ~5-15 min: two real one-frame generations (edge only; edge + depth + segmentation) and a worker-slot choice.
# On Sol the first chained job runs it automatically; elsewhere run it on the GPU machine and wait for "CHECK PASSED".
set -uo pipefail
source "$(dirname "$0")/env.sh"
MOCK=0; case " $* " in *" --mock "*) MOCK=1; export RESULTS="$BUNDLE/results_mock" GEN_SCRATCH="$WORK/mock";; esac   # dry runs never touch the real results
mkdir -p "$RESULTS/logs"
# GPU nodes without internet: use the caches filled by setup.sh
if ! curl -sI --max-time 8 https://huggingface.co > /dev/null 2>&1; then export HF_HUB_OFFLINE=1 UV_OFFLINE=1; echo "[info] no internet on this node: offline mode (models from the setup.sh cache)"; fi
[ -x "$PY" ] || { echo "Run setup.sh first ($PY not found)"; exit 1; }
"$PY" "$BUNDLE/tools/gen.py" --check "$@"
