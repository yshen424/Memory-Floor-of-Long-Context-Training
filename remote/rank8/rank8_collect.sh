#!/bin/bash
# Pack one rank-8 run for sending back:  ./rank8_collect.sh <jobid>
set -euo pipefail
cd "$(dirname "$0")"
JOB=${1:?usage: ./rank8_collect.sh <jobid>}
DIR=rank8_out/$JOB
[ -d "$DIR" ] || { echo "no $DIR"; exit 1; }
LOGS=""; for f in a30_rank8_${JOB}.log a30_rank8_${JOB}.err a100_rank8_${JOB}.log a100_rank8_${JOB}.err; do [ -f "$f" ] && LOGS="$LOGS $f"; done
TAR=rank8_results_${JOB}.tar.gz
tar czf "$TAR" --exclude='torch_trace_rank*' --exclude='cuda_snapshot_*' "$DIR" $LOGS
echo "wrote $TAR ($(du -h "$TAR" | cut -f1)) -- send this file back"
