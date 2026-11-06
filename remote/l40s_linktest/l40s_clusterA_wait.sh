#!/bin/bash
# When WOULD a 1-node, 4-GPU, exclusive job start on ClusterA's A30 and L40S
# partitions right now? Uses sbatch --test-only: NOTHING is submitted.
#
#   ./l40s_clusterA_wait.sh            # 1 h job (default)
#   ./l40s_clusterA_wait.sh 08:00:00   # any wall time up to 24:00:00
#
# Estimates come from the scheduler's current backfill plan and move with the
# queue; re-run before deciding.
T=${1:-01:00:00}
echo "now: $(date '+%F %T')   wall time requested: $T"
for g in a30 l40s; do
    out=$(sbatch --test-only -A <account> -p <partition> --qos=<qos> \
          --nodes=1 --exclusive --gres=gpu:${g}:4 --time="$T" --wrap="true" 2>&1)
    printf "%-5s %s\n" "$g" "$(echo "$out" | sed 's/^sbatch: //')"
done
