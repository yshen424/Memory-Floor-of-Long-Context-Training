#!/bin/bash
#SBATCH --job-name=a30_v100smoke
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=a30_v100smoke_%j.log
#SBATCH --error=a30_v100smoke_%j.err
#
# Run on the LOCAL V100 cluster, from this directory:  sbatch a30_v100_smoke.sbatch
# Runs a30_frontier.sbatch's smoke branch from the package's own code/ snapshot:
# four nodes x one V100 (world 4, srun + c10d rendezvous), the cluster's python.
# Output: a30_out/v100smoke_<jobid>/a30_summary.txt. Expected ~1.5 h (CC dominates).
#
# ── PURPOSE: every code path of the A30 job end to end, and the two in-job ──
# ── predictors checked where the answers are already measured.            ──
# S-P1 BYTES: every B arm on its V100 line within 0.05 MiB, common offset 0.00
#      (the snapshot computes the bytes the tree does).
# S-P2 REFUSALS: the bf16 mem-efficient arm and the flash+native-GQA arm fail in
#      seconds with the sm_70 refusal in stderr.txt (they must not fall back).
# S-P3 CONTEXT: outside_alloc_mb in [420, 560] MiB, centre ~488 -- the context
#      the harness's uncapped C0 implies if the capped and uncapped reserves
#      are one constant. Outside that range the native-bracket method is
#      re-derived before the A30 job is submitted.
# S-P4 CAPPED: CA [34, 35) and CC [110, 112), 7760 reproduced; every failure a
#      step-1 backward OOM on exactly 4x MiB.
# S-P5 NATIVE PREDICTION: capcal, from CA + CC + the measured context, writes a
#      largest static fit of 104 or 106 (measured [104, 108), 7743/7744). This is
#      the test of the method the A30 job uses for its native ladder.
# S-P6 TIME (one repeat, context only): kappa in [0.020, 0.045] s/x (7758: 0.032);
#      the written tax at x=100 in [1.0, 2.5]% (7758: 1.72%).
# S-P7 PROFILE: attn_recompute_share's kernel-count self-check ok; duty analysis runs.
export A30_SMOKE=1
exec bash a30_frontier.sbatch
