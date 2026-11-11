#!/bin/bash
# Least-traffic schedules at depths 4 and 16, each solved at its mode's minimum peak (feasible)
# and 0.001 MiB below it (infeasible). Depth 32 and 64 are in paper_run.py.
export TMPDIR="$(cd "$(dirname "$0")/.." && pwd)/tmp"; mkdir -p "$TMPDIR"
LOG=../logs/census_traffic.jsonl
for L in 4 16; do
  for cfg in "0 inclass 37.8203125" "0 wide_replay 34.5546875" "1 wide_replay 20.0" "1 wide 20.0" "0 static $(python3 -c "print(36.3203125+1.5*$L)")"; do
    set -- $cfg
    for d in 0.000001 -0.001; do
      B=$(python3 -c "print($3+$d)")
      timeout 2400 ../.venv/bin/python sparse_poet.py --L $L --gran census --spread $1 --mode $2 --method check --budget $B --threads 2 --tlimit 1200 >> $LOG 2>/dev/null
    done
  done
done
