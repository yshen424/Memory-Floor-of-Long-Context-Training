#!/bin/bash
#SBATCH --job-name=a100_v100smoke
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=a100_v100smoke_%j.log
#SBATCH --error=a100_v100smoke_%j.err
#
# Run on the LOCAL V100 cluster, from this directory:  sbatch a100_v100_smoke.sbatch
# A CODE CHECK of a100_regime.sbatch from the package's own code/ snapshot, world 4
# (four nodes x one V100), ~60-80 min. Output a100_out/v100smoke_<jobid>/.
# The node, preflight and bandwidth blocks run on the batch node's one V100; the
# NVSwitch check is skipped (single-GPU nodes). Pre-registered:
# S-P1 B, T and H arms on the V100 lines with common offset 0.00 +- 0.05 MiB (the a100
#      platform carries the V100 workspace): H-B1's reader path.
# S-P2 timecal writes a100_pred_time.txt from the x=16/32/64 pairs (V100 kappa ~0.03 s/x).
# S-P3 H arms record arm_env.txt and complete; across nodes NCCL uses NET either way, so
#      the reader's transport table shows NET/* for every block (H-N1's reader path)
#      and the HOST-PATH table prints one row at x=16.
# S-P4 G: offload x=24 + 1877.905 MiB ballast completes; the MATCHED FOOTPRINT table
#      prints a row (its T partners do not exist at x=24: nan, expected).
# S-P5 P x=16: attn_share.json, duty.json and decomp_x16.json written; the
#      decomposition identity span = C + X + N + I holds.
# S-P6 host_copy_bw prints solo and concurrent rates for one GPU (V100: ~6-12 GB/s)
#      and the reader quotes a100_bw.json; gpu.csv sampled in every arm.
export A100_SMOKE=1
bash a100_regime.sbatch
