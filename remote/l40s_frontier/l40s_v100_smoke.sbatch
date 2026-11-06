#!/bin/bash
#SBATCH --job-name=l40s_v100smoke
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=l40s_v100smoke_%j.log
#SBATCH --error=l40s_v100smoke_%j.err
#
# Run on the LOCAL V100 cluster, from this directory:  sbatch l40s_v100_smoke.sbatch
# Runs l40s_frontier.sbatch's smoke branch from the package's own code/ snapshot:
# four nodes x one V100 (world 4, srun + c10d rendezvous), the cluster's python.
# Output: l40s_out/v100smoke_<jobid>/l40s_summary.txt. Expected ~45-60 min.
#
# PURPOSE: a CODE CHECK of the L40S package. The methods (bytes, timecal, capped
# ladders, native prediction) already passed in the A30 smoke (7762); what is new
# here is the Llama-3.2-3B geometry, its tag, its slope verdict and the RS/RO
# ladder path. The full 3B (L=28) does not fit a V100, so it runs at L=4.
# S-P1 BYTES: every tier-A B/T arm on its line within 0.05 MiB (7762 reproduced).
# S-P2 3B GEOMETRY AT L=4: the driver builds h 3072, 24/8 heads (head_dim 128),
#      v 128256 and runs; harness lines (V100): static x=8 7794.33, x=16 9095.26
#      (second replay); offload x=8 7794.33, x=16 8999.27. R-P1 prints HELD.
#      A miss here is the first data point of the out-of-sample test, not a bug.
# S-P3 RS ladder at L=4 closes near x=55 (V100 native capacity) on a step-1
#      backward OOM of exactly 16x MiB; the reader prints the bracket section.
# S-P4 CA [34, 35) as in 7760/7762; capcal writes a native prediction file.
# S-P5 timecal writes l40s_pred_time.txt; attn_recompute_share's self-check ok.
export L40S_SMOKE=1
exec bash l40s_frontier.sbatch
