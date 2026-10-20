#!/usr/bin/env bash
# Packs results/ (generated frames, videos, logs, summary) into one .tar to send back. Safe to run any time, also mid-run.
set -uo pipefail
source "$(dirname "$0")/env.sh"
[ -f "$RESULTS/summary.json" ] || "$PY" "$BUNDLE/tools/summarize.py" > /dev/null 2>&1 || true
OUT="$BUNDLE/realtwin_results_$(hostname -s)_$(date +%Y%m%d-%H%M).tar"
tar -cf "$OUT" -C "$(dirname "$RESULTS")" "$(basename "$RESULTS")"
echo "packed: $OUT ($(du -h "$OUT" | cut -f1))"; sha256sum "$OUT" | tee "$OUT.sha256"
echo "Send this .tar file back (any file-sharing link is fine)."
