#!/bin/bash
# Pack one A30 run for sending back.
#   ./a30_collect.sh <jobid>          summary, predictions, JSONs, per-arm stdout/stderr (a few MB)
#   ./a30_collect.sh <jobid> traces   the two profiled arms' chrome traces too (send only if asked)
set -euo pipefail
cd "$(dirname "$0")"
JOB=${1:?usage: ./a30_collect.sh <jobid> [traces]}
DIR=a30_out/$JOB
[ -d "$DIR" ] || { echo "no $DIR"; exit 1; }
LOGS=""; for f in a30_frontier_${JOB}.log a30_frontier_${JOB}.err a30_v100smoke_${JOB#v100smoke_}.log; do [ -f "$f" ] && LOGS="$LOGS $f"; done
if [ "${2:-}" = "traces" ]; then
    TAR=a30_traces_${JOB}.tar.gz; tar czf "$TAR" "$DIR"/P_*/torch_trace_rank*
else
    TAR=a30_results_${JOB}.tar.gz; tar czf "$TAR" --exclude='torch_trace_rank*' --exclude='*.pickle' "$DIR" $LOGS
fi
echo "wrote $TAR ($(du -h "$TAR" | cut -f1)) -- send this file back"
