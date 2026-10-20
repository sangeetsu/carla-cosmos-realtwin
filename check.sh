#!/usr/bin/env bash
# GPU pre-flight, ~5-15 min: two real one-frame generations (edge only; edge + depth + segmentation) and a worker-slot choice.
# Run it on the GPU node and wait for "CHECK PASSED" before leaving the overnight run.
set -uo pipefail
source "$(dirname "$0")/env.sh"
MOCK=0; case " $* " in *" --mock "*) MOCK=1; export RESULTS="$BUNDLE/results_mock" GEN_SCRATCH="$WORK/mock";; esac   # dry runs never touch the real results
mkdir -p "$RESULTS/logs"
[ -x "$PY" ] || { echo "Run setup.sh first ($PY not found)"; exit 1; }
"$PY" "$BUNDLE/tools/gen.py" --check "$@"
