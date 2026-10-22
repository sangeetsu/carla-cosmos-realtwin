#!/usr/bin/env bash
# Submit N chained Sol jobs (default 4); extra arguments are passed to sbatch. Each starts when the previous one ends, even at its time limit, and continues where it stopped;
# finished frames are never redone. Extra jobs after the work is done exit within minutes. Cancel the rest with:  scancel -u $USER -n realtwin
set -euo pipefail
cd "$(dirname "$0")/.."
N=${1:-4}; shift || true            # any further arguments go to sbatch, e.g.  -G a100:2 --mem=128G --time=16:00:00
prev=""
for i in $(seq 1 "$N"); do
  id=$(sbatch --parsable ${prev:+--dependency=afterany:$prev} "$@" slurm/sol.sbatch)
  echo "submitted job $id${prev:+ (starts after $prev)}"; prev=$id
done
echo "Watch:  squeue -u \$USER     and     tail -f results/logs/gen.log"
