#!/bin/bash -l
#SBATCH --job-name=a30_rank8
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=a30_rank8_%j.log
#SBATCH --error=a30_rank8_%j.err
#
# Submit from this directory; the GPU request is ON THE COMMAND LINE ONLY (typed GRES, per node):
#   smoke (~20 min):  sbatch --gres=gpu:a30:4 --time=01:00:00 --export=ALL,R8_PLAN=smoke a30_rank8.sbatch
#   full  (~5-6 h):   sbatch --gres=gpu:a30:4 a30_rank8.sbatch
# Venv: ~/a30_venv (A30_VENV overrides), launcher-checked before any arm.

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  EIGHT RANKS ACROSS TWO A30 NODES. Two exclusive A30 nodes, 4 GPUs      ║
# ║  each, joined by 200 Gb/s InfiniBand (NDR, ConnectX-7).                  ║
# ║                                                                          ║
# ║  0   NODES: facts per node (GPUs, topology, InfiniBand ports), a GPU     ║
# ║      PREFLIGHT on both (exit 3), an 8-rank NCCL all-reduce across the    ║
# ║      two nodes and a 4-rank one inside node 0 (transport + bus           ║
# ║      bandwidth; exit 5 if the launcher fails), host-copy bandwidth.      ║
# ║  B   THE MULTI-NODE POINT: base shape (Table 2), world 8 across both     ║
# ║      nodes: static x=24/48, stock offload x=24/48/96, 3 steps.           ║
# ║  BR  THE WORKED EXAMPLE'S BYTES: Llama-3.2-3B geometry (h 3072, I 8192,  ║
# ║      24/8 heads, L=28, v=128256, untied), world 8: static x=16/32,       ║
# ║      offload x=16/48, 3 steps.                                           ║
# ║  KS8, KO8  ITS FRONTIER ON TWO NODES: native 24 GB brackets, static and  ║
# ║      stock offload, world 8, rungs written in-job (reader natpred)       ║
# ║      before each ladder, 2 steps per rung.                               ║
# ║  BR4, KS4, KO4  THE SAME MODEL ON ONE NODE (world 4, node 0 only): one   ║
# ║      byte point per mechanism (x=16), then both brackets.                ║
# ║  T   ITS TIME ON TWO NODES: static/offload pairs, ABBA, 2 repeats at     ║
# ║      x=8/16/32 -> IN-JOB CALIBRATION writes a30_pred_time.txt -> 2       ║
# ║      repeats at x=44 (static's frontier is 49 here). GPU clocks at 1 Hz. ║
# ║                                                                          ║
# ║  All: fp16 sdpa_mem_efficient (regime B), whole-layer, chunked CE 256,   ║
# ║  fused-input, use_cache off, b=1, seed 1234, --disable_hbm_drift, FSDP   ║
# ║  full shard, CPU offload OFF.                                            ║
# ║  WALL CLOCK ~5-6 h (0 0.2, B 0.3, BR 0.3, KS8 0.4, KO8 1.0, BR4/KS4/KO4  ║
# ║  0.8, T 1.3 h, plus up to ~10 min of NCCL teardown per closing rung).    ║
# ║  No arm starts after 11 h; 12 h requested.                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── PRE-REGISTERED 2026-11-08 from harness/ (platform a30: V100 workspace 16.2559  ──
# ── carried; the A30 measured 16.2546, i.e. every arm ~-0.001 MiB). MiB, x = 1024 tokens ──
# Lines (derived; the peak is the largest instant, its name in brackets):
#   base L=64 world 8 static    830.3424 + 132.320312x   x=24: 4006.03  x=48: 7181.72
#   base L=64 world 8 offload  1037.3930 +  37.820312x   x=24: 1945.08  x=48: 2852.77  x=96: 4668.14
#     (world 4: 1476.7967 / 1866.8919; sharding moves the intercepts by -646.4543 and
#      -829.4989 MiB and no slope)
#   3B  L=28 world 8 static    7862.2857 + 312.617188x   x=16: 13152.18  x=32: 18058.06  [second replay,
#                                                         8246.3091 + 306.617188x, binds below x=64]
#   3B  L=28 world 8 offload   9446.3824 + 150.617188x   x=16: 11856.26  x=48: 16676.01
#   3B  L=28 world 4 static   13021.7862 + 312.617188x   x=16: 18311.83 [deep replay]
#   3B  L=28 world 4 offload  15805.9561 + 150.617188x   x=16: 18215.83
#
# R8-B1 BYTES ACROSS NODES: every B, BR, BR4 and T arm on its line plus ONE common offset,
#       all within 0.05 MiB of each other; lean offset -0.001 (the A30's workspace).
#       FALSIFIER: arms whose offsets differ by more -- a byte term moved with the rank
#       count or across the network; name it from the arm.
# R8-B2 SHARDING MOVES ONLY THE INTERCEPT: B's fitted slopes 132.3203 and 37.8203 (world 4's,
#       to 0.0001) and its intercepts 830.34 / 1037.39 (to 0.05). This is the multi-node point.
# R8-B3 (ADDENDUM 2026-11-10, registered before the full job, from smoke 805321 and A100 8548619):
#       the base-shape world-8 arms sit +0.0452 (static) and +0.0601 (offload) MiB above their lines,
#       the allocator's 512-byte rounding of world-8 fp32 shards; the 3B arms carry
#       none (their shards are multiples of 512 B). R8-B1 is read with this term.
# R8-N1 TRANSPORT: the 8-rank all-reduce and every world-8 arm's inter-node channels run
#       via NET/IB; inside a node P2P or SHM. FALSIFIER: NET/Socket -> NCCL did not use the
#       InfiniBand fabric (bytes unaffected; T's times then carry a socket fabric; recorded).
#
# THE WORKED EXAMPLE: Llama-3.2-3B geometry on A30 24 GB nodes.
# Native brackets are written IN-JOB (reader natpred) from the Ampere capacity reserve the
# A30 measured (633912: delta 59.5 MiB, alpha 0.093, feasible 0-0.186), this job's card total
# and the context it measures at the same rank count. Header values below use the A30's
# card total 24164.25 MiB and context 618.88 MiB (633912); the in-job file is the registration.
# R8-K1 TWO NODES (world 8): static largest fit 49 (range 49-49 over alpha), stock offload 92
#       (91-93). Registered tolerance +-2 rungs (a new shape on this card, the paper's
#       cross-shape rule). FALSIFIER: a measured largest fit outside [47, 51] / [89, 95].
# R8-K2 ONE NODE (world 4): static 32 (32-32), offload 50 (50-50); +-2 rungs.
# R8-K3 WHAT A SECOND NODE BUYS: static 32 -> 49 (1.5x), offload 50 -> 92 (1.8x), because the
#       sharded state is a larger share of offload's smaller peak. Spans (offload/static):
#       world 8 1.88, world 4 1.56 (structural cap 2.08 at this shape).
# R8-K4 FAILURES: every closing rung is a backward OOM on the replayed MLP activation, 16x MiB
#       (PyTorch prints GiB to 2 decimals: the reader allows +-0.005 GiB).
# x_host (computed, not run): stock offload pins 168 MiB per x per rank, four ranks share a
#       node's 515 GB: host memory binds at x ~ 785 (harness). With the node's 503.4 GiB, which
#       the harness now uses, x ~ 767.
#
# TIME (T; time ratios: THIS SETUP ONLY -- two A30 nodes over InfiniBand, 3B geometry, world 8).
# R8-T1 TWO NAMED OUTCOMES for stock offload's tax = (offload - static) / static:
#       (a) LEAN, compute outgrows the copy: per layer and token the 3B moves 4x the base shape's
#           boundary but computes >= 14x its dense work and 4x its attention, so the tax sits
#           below the base shape's A30 tax (1.5% at x=16, world 4): tax(16) <= 1.0%, x*(2%) <= 16.
#       (b) the network shares the copy path (GPU and NIC under one PCIe switch, GPUDirect RDMA):
#           tax(16) in (1%, 8%], x*(2%) in (16, 48].
#       Neither, i.e. tax(44) > 2%: x*(2%) lies at or beyond static's own frontier (49), so
#       offloading is required for length anyway on two nodes and costs > 2% near static's longest length.
# R8-T2 the in-job prediction a30_pred_time.txt (T(x) and kappa from x=8/16/32) is written BEFORE
#       x=44 runs. FALSIFIER: tax at 44 outside +-30% of the file (the A30 failed this at world 4).
#
# FAILURE CLASSES: (0) a GPU that will not open -> exit 3, host named, resubmit --exclude;
# (1) the 8-rank NCCL test fails -> exit 5 (launcher or fabric), nothing else runs; (2) an OOM
# outside a ladder's closing rung is a defect (the largest byte or time arm, 3B world 8 static
# x=44 in T at 21.2 GiB, sits 1.6 GiB under the predicted capacity).
set -uo pipefail
PLATFORM=a30; PREFIX=a30; VENV_VAR=A30_VENV; VENV_DEFAULT=$HOME/a30_venv; HOURS=12
source "$SLURM_SUBMIT_DIR/rank8_common.sh"
r8_setup
if [ -z "${R8_V100:-}" ] && [ "$NN" != 2 -o "$GPN" != 4 ]; then
    echo "THIS JOB NEEDS 2 NODES x 4 A30 (have $NN x $GPN): submit with --nodes=2 --gres=gpu:a30:4"; exit 2
fi
if [ -n "${R8_V100:-}" ]; then
    R8_NCCL_WORLDS="4"; W8=4; W4=4; L3=4                      # V100 smoke: 4 nodes x 1 GPU, 3B at depth 4
else
    R8_NCCL_WORLDS="8 4"; W8=8; W4=4; L3=28
fi
r8_nodes

if [ "${R8_PLAN:-full}" = smoke ] || [ -n "${R8_V100:-}" ]; then
    # ── SMOKE: every code path at small x (~20 min on ClusterA) ──
    arm B static  tier_a 64 8 3 0 $W8; arm B offload tier_a 64 8 3 0 $W8
    arm BR static llama3b $L3 4 3 0 $W8
    arm BR4 static llama3b $L3 4 3 0 $W4
    reader bytes
    reader natpred --pred "$OUT/${PREFIX}_pred_native_KS8.txt" --ladder KS8 --shape llama3b --L $L3 --world $W8 --mode static --step 1
    pairs T 1 llama3b $L3 $W8 4:4 8:4 16:4
    reader timecal --pred "$OUT/${PREFIX}_pred_time.txt" --xs 4,8,16 --at 24
    arm T static llama3b $L3 24 3 1 $W8
else
    # ── B. the multi-node point ──
    arm B static  tier_a 64 24 3 0 8; arm B static  tier_a 64 48 3 0 8
    arm B offload tier_a 64 24 3 0 8; arm B offload tier_a 64 48 3 0 8; arm B offload tier_a 64 96 3 0 8
    # ── BR, KS8, KO8. the worked example on two nodes ──
    arm BR static  llama3b 28 16 3 0 8; arm BR static  llama3b 28 32 3 0 8
    arm BR offload llama3b 28 16 3 0 8; arm BR offload llama3b 28 48 3 0 8
    reader bytes
    native KS8 static  llama3b 28 8 1 "48 49 50 51 52 53"
    native KO8 offload llama3b 28 8 1 "90 91 92 93 94 95 96"
    # ── BR4, KS4, KO4. the same model on one node (node 0) ──
    arm BR4 static llama3b 28 16 3 0 4; arm BR4 offload llama3b 28 16 3 0 4
    native KS4 static  llama3b 28 4 1 "31 32 33 34 35 36"
    native KO4 offload llama3b 28 4 1 "49 50 51 52 53 54"
    # ── T. its time on two nodes ──
    pairs T 2 llama3b 28 8 8:8 16:6 32:5
    reader timecal --pred "$OUT/${PREFIX}_pred_time.txt" --xs 8,16,32 --at 44
    pairs T 2 llama3b 28 8 44:4
fi
reader read
log "=== DONE in $((SECONDS / 60)) min: ./rank8_collect.sh $SLURM_JOB_ID ==="
