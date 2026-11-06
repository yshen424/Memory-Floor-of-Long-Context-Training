#!/bin/bash
# On ClusterA after a job:  ./a30f_collect.sh tax|native <jobid> [full]
# Packs the summary, every arm's json/csv/stdout/stderr and the flash live sets.
# Traces and raw allocator snapshots stay unless 'full' is given.
set -euo pipefail
cd "$(dirname "$0")"
KIND=$1; JOB=$2; DIR=a30f_out/${KIND}_$JOB
[ -d "$DIR" ] || { echo "no $DIR"; exit 1; }
EXC=(); [ "${3:-}" = full ] || EXC=(--exclude='torch_trace_rank*' --exclude='*.pkl')
tar czf "a30f_${KIND}_results_$JOB.tar.gz" "${EXC[@]}" "$DIR" a30f_${KIND}_$JOB.log a30f_${KIND}_$JOB.err 2>/dev/null || \
  tar czf "a30f_${KIND}_results_$JOB.tar.gz" "${EXC[@]}" "$DIR"
echo "packed a30f_${KIND}_results_$JOB.tar.gz ($(du -h a30f_${KIND}_results_$JOB.tar.gz | cut -f1))"
