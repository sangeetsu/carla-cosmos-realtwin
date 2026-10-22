#!/usr/bin/env bash
# The run. Resumable: run the same command again (or let the next chained Slurm job run) to continue where it stopped.
#   optional:  WORKERS=2 bash run_overnight.sh      (override the slot count chosen by check.sh)
#              JOBS=J3 bash run_overnight.sh         (only some jobs: J1 repro, J2 comparison, J4 video, J3 bulk; J5 seed replicate only when listed)
set -uo pipefail
source "$(dirname "$0")/env.sh"
MOCK=0; case " $* " in *" --mock "*) MOCK=1; export RESULTS="$BUNDLE/results_mock" GEN_SCRATCH="$WORK/mock";; esac   # dry runs never touch the real results
mkdir -p "$RESULTS/logs"
# GPU nodes without internet: use the caches filled by setup.sh
if ! curl -sI --max-time 8 https://huggingface.co > /dev/null 2>&1; then export HF_HUB_OFFLINE=1 UV_OFFLINE=1; echo "[info] no internet on this node: offline mode (models from the setup.sh cache)"; fi
[ -f "$RESULTS/logs/check.json" ] || bash "$BUNDLE/check.sh" $([ $MOCK = 1 ] && echo --mock) || { echo "check.sh failed - see $RESULTS/logs/gen.log"; exit 1; }
"$PY" "$BUNDLE/tools/gen.py" --run ${JOBS:+--only "$JOBS"} ${WORKERS:+--workers "$WORKERS"} "$@"; RC=$?
"$PY" "$BUNDLE/tools/summarize.py" > "$RESULTS/logs/summary.log" 2>&1 || true
[ $MOCK = 1 ] && { echo "mock run finished (status $RC): see $RESULTS"; exit 0; }
if [ $RC = 0 ] && [ ! -f "$RESULTS/.packed" ]; then bash "$BUNDLE/pack_results.sh" && touch "$RESULTS/.packed"   # everything done: pack once
elif [ $RC = 3 ]; then echo "Not finished yet: run this again (or let the next chained job run) to continue. 'bash pack_results.sh' packs what exists now."; fi
exit $RC
