#!/bin/bash
# ALLOCATION RECORDS OF THE PAPER'S REMOTE JOBS (whether nodes were exclusive).
# Run on a login node of each cluster; no allocation needed.
# A job held its node exclusively when AllocTRES shows the node's full CPU count
# (64 on ClusterA, 128 on a DGX A100) and its full memory.
#   ClusterA:        ./job_records.sh 633912 642029 642032 630188 634078 644212
#   ClusterB:  ./job_records.sh 8039176 8051891
set -u
[ $# -ge 1 ] || { echo "usage: $0 <jobid>..."; exit 2; }
OUT=job_records_$(hostname -s)_$(date +%Y%m%d).txt
sacct -X -j "$(IFS=,; echo "$*")" \
    -o JobID,JobName%22,Partition%16,NodeList%14,AllocTRES%80,Start,Elapsed,State \
    > "$OUT" 2>&1
cat "$OUT"; echo "wrote $OUT"
