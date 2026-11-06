#!/bin/bash
# Pack one link-test run for sending back.
#   ./l40s_collect.sh <jobid>          small archive: summary, JSONs, logs (~a few MB)
#   ./l40s_collect.sh <jobid> traces   chrome traces too (~75 MB per rank per arm;
#                                      send only if asked -- the summary already
#                                      contains the trace analysis)
set -euo pipefail
cd "$(dirname "$0")"
JOB=${1:?usage: ./l40s_collect.sh <jobid> [traces]}
DIR=l40s_out/$JOB
[ -d "$DIR" ] || { echo "no $DIR"; exit 1; }
LOGS=""; for f in l40s_linktest_${JOB}.log l40s_linktest_${JOB}.err; do [ -f "$f" ] && LOGS="$LOGS $f"; done
if [ "${2:-}" = "traces" ]; then
    TAR=l40s_traces_${JOB}.tar.gz
    tar czf "$TAR" "$DIR"/T*/torch_trace_rank*
else
    TAR=l40s_results_${JOB}.tar.gz
    tar czf "$TAR" --exclude='torch_trace_rank*' "$DIR" $LOGS
fi
echo "wrote $TAR ($(du -h "$TAR" | cut -f1)) -- send this file back"
