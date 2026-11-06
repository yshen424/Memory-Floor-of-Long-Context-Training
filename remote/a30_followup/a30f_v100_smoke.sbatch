#!/bin/bash
#SBATCH --job-name=a30f_v100smoke
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=a30f_v100smoke_%j.log
#SBATCH --error=a30f_v100smoke_%j.err
#
# Run on the LOCAL V100 cluster, from this directory:  sbatch a30f_v100_smoke.sbatch
# A CODE CHECK of both follow-up jobs from the package's own code/ snapshot, world 4
# (four nodes x one V100). ~75-100 min. Outputs a30f_out/v100smoke_{tax,native}_<jobid>/.
# S-P1 the new per-step allocator counters appear in every arm and the reader prints
#      per-step cudaMalloc/cudaFree rates; gpu.csv has >= 1 sample per second of arm.
# S-P2 E arms record alloc_conf '' (expandable segments OFF) and complete at x=24.
# S-P3 G: offload x=24 + 1877.9 MiB ballast completes and its peak is exactly the
#      plain offload peak + 1877.9 (+-0.05 MiB): the ballast is idle and counted.
# S-P4 Q: both profiled arms yield attn_share.json and the reader prints the H4 split.
# S-P5 M: the mem-efficient snapshot arm writes cuda_snapshot_rank0.pkl and a live set.
# S-P6 native ladder logic on 7760's capped ladder C (6 GiB offload): starts at 110
#      (fits), climbs to 112 (fails on exactly 448 MiB in backward): bracket
#      [110, 112) = 7760/7762. The host watchdog runs (min MemAvailable recorded) and
#      does not fire. NO-P1/P3/P4 print MISS here by design (A30 numbers).
# S-P7 the tax reader's R1/R2 lines are absent (no x>=80 in the smoke): expected.
export A30F_SMOKE=1
bash a30f_tax.sbatch
bash a30f_native.sbatch
