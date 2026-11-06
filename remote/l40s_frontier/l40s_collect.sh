#!/bin/bash
# Pack one L40S frontier run for sending back.
#   ./l40s_collect.sh <jobid>          summary, predictions, JSONs, per-arm stdout/stderr (a few MB)
#   ./l40s_collect.sh <jobid> traces   the two profiled arms' chrome traces too (send only if asked)
set -euo pipefail
cd "$(dirname "$0")"
JOB=${1:?usage: ./l40s_collect.sh <jobid> [traces]}
DIR=l40s_out/$JOB
[ -d "$DIR" ] || { echo "no $DIR"; exit 1; }
LOGS=""; for f in l40s_frontier_${JOB}.log l40s_frontier_${JOB}.err l40s_v100smoke_${JOB#v100smoke_}.log; do [ -f "$f" ] && LOGS="$LOGS $f"; done
if [ "${2:-}" = "traces" ]; then
    TAR=l40s_frontier_traces_${JOB}.tar.gz; tar czf "$TAR" "$DIR"/P_*/torch_trace_rank*
else
    TAR=l40s_frontier_results_${JOB}.tar.gz; tar czf "$TAR" --exclude='torch_trace_rank*' --exclude='*.pickle' "$DIR" $LOGS
fi
echo "wrote $TAR ($(du -h "$TAR" | cut -f1)) -- send this file back"
