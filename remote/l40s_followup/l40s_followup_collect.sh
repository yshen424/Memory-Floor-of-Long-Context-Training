#!/bin/bash
# Pack one L40S follow-up run for sending back (summary, JSONs, per-arm stdout/stderr,
# and the snapshot arm's cuda_snapshot_rank*.pkl, which names the +0.148 MiB offline).
#   ./l40s_followup_collect.sh <jobid>
set -euo pipefail
cd "$(dirname "$0")"
JOB=${1:?usage: ./l40s_followup_collect.sh <jobid>}
DIR=l40s_followup_out/$JOB
[ -d "$DIR" ] || { echo "no $DIR"; exit 1; }
LOGS=""; for f in l40s_followup_${JOB}.log l40s_followup_${JOB}.err; do [ -f "$f" ] && LOGS="$LOGS $f"; done
TAR=l40s_followup_results_${JOB}.tar.gz
tar czf "$TAR" --exclude='torch_trace_rank*' "$DIR" $LOGS
echo "wrote $TAR ($(du -h "$TAR" | cut -f1)) -- send this file back"
