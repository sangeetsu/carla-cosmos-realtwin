#!/usr/bin/env bash
# The overnight run. Resumable: run the same command again to continue where it stopped.
#   optional:  WORKERS=2 bash run_overnight.sh      (override the slot count chosen by check.sh)
#              JOBS=J3 bash run_overnight.sh         (only some jobs: J1 repro, J4 video, J2 multi-control, J3 bulk, J5 seed replicate)
set -uo pipefail
source "$(dirname "$0")/env.sh"
MOCK=0; case " $* " in *" --mock "*) MOCK=1; export RESULTS="$BUNDLE/results_mock" GEN_SCRATCH="$WORK/mock";; esac   # dry runs never touch the real results
mkdir -p "$RESULTS/logs"
[ -f "$RESULTS/logs/check.json" ] || bash "$BUNDLE/check.sh" $([ $MOCK = 1 ] && echo --mock) || { echo "check.sh failed - see $RESULTS/logs/gen.log"; exit 1; }
"$PY" "$BUNDLE/tools/gen.py" --run ${JOBS:+--only "$JOBS"} ${WORKERS:+--workers "$WORKERS"} "$@"
"$PY" "$BUNDLE/tools/summarize.py" > "$RESULTS/logs/summary.log" 2>&1 || true
[ $MOCK = 1 ] && { echo "mock run finished: see $RESULTS"; exit 0; }
bash "$BUNDLE/pack_results.sh"
