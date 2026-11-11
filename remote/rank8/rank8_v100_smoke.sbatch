#!/bin/bash
#SBATCH --job-name=rank8_v100smoke
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=rank8_v100smoke_%j.log
#SBATCH --error=rank8_v100smoke_%j.err
#
# Run on the LOCAL V100 cluster, from this directory, after ./rank8_build.sh:
#   sbatch rank8_v100_smoke.sbatch
# A CODE CHECK of both job scripts from the package's own code/ snapshot, world 4 (four nodes x
# one V100, the multi-node srun launcher with one GPU per node), ~45-60 min. Shapes are cut to
# fit 16 GB: 3B at depth 4, 7B at depth 4. Output rank8_out/v100smoke_{a30,a100}_<jobid>/.
# Pre-registered:
# S-P1 node block: facts per node, PREFLIGHT ok on all four, the 4-rank NCCL test prints a bus
#      bandwidth and NET/* channels (one GPU per node: every channel crosses the network).
# S-P2 bytes: every B/BR/BR4/BZ/T peak on its harness line (V100 platform constants differ from
#      the a30/a100 profiles only in the workspace: common offset 0.000 +- 0.05 MiB); the arm tags
#      carry _w4 and the shape, and the reader picks the right line for each.
# S-P3 natpred writes a prediction file with CENTRE and RUNGS lines (values meaningless on V100).
# S-P4 timecal writes the time prediction from x=4/8/16 before the x=24 arm; read prints every
#      section without a traceback.
export R8_V100=1
# R8_ONLY=a30 or a100 runs one half (the shipped smokes: the A30 half 7943, the A100 half 7944):
#   sbatch --export=ALL,R8_ONLY=a100 rank8_v100_smoke.sbatch
[ "${R8_ONLY:-a30}" = a30 ] && bash a30_rank8.sbatch
[ "${R8_ONLY:-a100}" = a100 ] && bash a100_rank8.sbatch
exit 0
