#!/bin/bash
# Pack one A100 run for sending back.
#   ./a100_collect.sh <jobid>          summary, predictions, bandwidth, decompositions, step JSONs, logs (small)
#   ./a100_collect.sh <jobid> traces   rank-0 chrome traces of the profiled arms too (large; only if asked)
set -euo pipefail
cd "$(dirname "$0")"
JOB=${1:?usage: ./a100_collect.sh <jobid> [traces]}
DIR=a100_out/$JOB
[ -d "$DIR" ] || { echo "no $DIR"; exit 1; }
LOGS=""; for f in a100_regime_${JOB}.log a100_regime_${JOB}.err; do [ -f "$f" ] && LOGS="$LOGS $f"; done
if [ "${2:-}" = "traces" ]; then
    TAR=a100_traces_${JOB}.tar.gz; tar czf "$TAR" "$DIR"/P_*/torch_trace_rank0
else
    TAR=a100_results_${JOB}.tar.gz; tar czf "$TAR" --exclude='torch_trace_rank*' "$DIR" $LOGS
fi
echo "wrote $TAR ($(du -h "$TAR" | cut -f1)) -- send this file back"
