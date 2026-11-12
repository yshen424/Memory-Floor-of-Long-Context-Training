#!/bin/bash -l
#SBATCH --job-name=a100_rank8
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --constraint=dgx-A100
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=a100_rank8_%j.log
#SBATCH --error=a100_rank8_%j.err
#
# Submit from this directory; the GPU request is ON THE COMMAND LINE ONLY (typed GRES):
#   sbatch --gres=gpu:a100:8 a100_rank8.sbatch
# --constraint=dgx-A100 is REQUIRED (<partition> also holds hpe-A100 nodes of unknown fabric);
# the job checks the topology itself and stops with exit 4 if any pair of its 8 GPUs is not NVLink.
# Venv: ~/a100_venv (A100_VENV overrides; module from its .module), launcher-checked before any arm.

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  EIGHT RANKS ON ONE DGX A100 NODE, AND A 7B-LIKE MODEL. One exclusive    ║
# ║  DGX A100 node, all 8 GPUs (NVSwitch), world 8; blocks run most-         ║
# ║  decisive first. Eight ranks in one node and a 7B-like model without    ║
# ║  CPU offload need eight GPUs of >= 40 GB.                                ║
# ║                                                                          ║
# ║  0   NODE: facts, GPU PREFLIGHT (exit 3), NVSWITCH CHECK on all 8 (exit  ║
# ║      4), an 8-rank NCCL all-reduce (exit 5), host-copy bandwidth (two    ║
# ║      GPUs share each PCIe switch here, unlike the 4-GPU A100 visits).    ║
# ║  B   base shape (Table 2), world 8: static x=24/48, offload x=24/48/96.  ║
# ║  BZ  7B GQA geometry (h 4096, I 14336, 32/8 heads, L=32, v=32000,        ║
# ║      untied), world 8: static x=24/48, offload x=8/16/32, 3 steps.       ║
# ║  T   its time: static/offload pairs, ABBA, 2 repeats at x=16/32/64 ->    ║
# ║      IN-JOB CALIBRATION writes a100_pred_time.txt -> 2 repeats at x=100. ║
# ║  ZS, ZO  its native 80 GB brackets, static then stock offload, rungs     ║
# ║      written in-job before each ladder, 2 steps per rung.                ║
# ║                                                                          ║
# ║  All: fp16 sdpa_mem_efficient (regime B), whole-layer, chunked CE 256,   ║
# ║  fused-input, use_cache off, b=1, seed 1234, --disable_hbm_drift, FSDP   ║
# ║  full shard, CPU offload OFF.                                            ║
# ║  WALL CLOCK ~8-10 h (0 0.2, B 0.2, BZ 0.3, T 2.5, ZS 1.2, ZO 3.5-4.5 h:  ║
# ║  a 7B step at x=264 is estimated at ~25 min). No arm after 23 h.         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── PRE-REGISTERED 2026-11-08 from harness/ (platform a100: V100 workspace 16.2559 carried; ──
# ── the A100 measured -0.001 MiB on every arm, 8039176/8051891). MiB, x = 1024 tokens ──
#   base L=64 world 8 static    830.3424 + 132.320312x   x=24: 4006.03  x=48: 7181.72
#   base L=64 world 8 offload  1037.3930 +  37.820312x   x=24: 1945.08  x=48: 2852.77  x=96: 4668.14
#   7B  L=32 world 8 static   12069.6758 + 488.648438x   [second replay, 12901.7071 + 480.648438x,
#                                                         binds below x=104] x=24: 24437.27  x=48: 35972.83
#   7B  L=32 world 8 offload  15917.8204 + 240.648438x   x=8: 17843.01  x=16: 19768.20  x=32: 23618.57
#   A boundary is 8 MiB per x per layer here (1.5 at the base shape) against a non-migratable
#   232.65 MiB/x: the structural cap (static slope over offload slope) is 2.0305 (3.4987 base).
#
# R8-B1 BYTES: every B, BZ and T arm on its line plus ONE common offset, all within 0.05 MiB of
#       each other; lean -0.001. FALSIFIER: offsets that differ by more -- name the term.
# R8-B2 B's slopes 132.3203 / 37.8203 and intercepts 830.34 / 1037.39 (the A30 job's B, on NVSwitch).
# R8-N1 TRANSPORT: the 8-rank all-reduce and the arms' channels run via P2P/NVLink or NVLS.
#
# THE 7B FRONTIER (written IN-JOB by reader natpred before each ladder, from the Ampere reserve
# the A30 measured -- delta 59.5 MiB, alpha 0.093, feasible 0-0.186 -- this card's total and the
# context measured at world 8). Header values: card total 81152.75 MiB (8051891), context 700 MiB:
# R8-Z1 static largest fit 138 (138-138, rung step 2), stock offload 264 (260-264, step 4);
#       +-2 rungs (a new shape on a card whose capacity was never bracketed). FALSIFIER: a
#       measured largest fit outside [134, 142] / [252, 272].
# R8-Z2 span (offload/static) 1.91, against the structural cap 2.03: at 7B the floor still holds
#       but buys about half the base shape's span (3.42 on V100).
# R8-Z3 FAILURES: backward OOMs on the replayed MLP activation, 28x MiB.
# x_host (computed): stock offload pins 256 MiB per x per rank and EIGHT ranks share the node's
#       2015 GiB here: host memory binds at x ~ 1008 (harness, --ranks_per_node 8).
#
# TIME (THIS SETUP ONLY: DGX A100, 8 GPUs over NVSwitch, two per PCIe switch, 7B, world 8).
# R8-T1 TWO NAMED OUTCOMES for stock offload's tax:
#       (a) LEAN, compute outgrows the copy: per layer and token the 7B moves 5.3x the base shape's
#           boundary but computes ~31x its dense work and 5.3x its attention, so its tax sits well
#           under the base shape's A100 tax (6.2% at x=16, 2.6% at x=64): tax(16) <= 1.5%,
#           x*(2%) <= 16.
#       (b) the copies dominate as at the base shape (two GPUs per PCIe switch halve each GPU's
#           copy rate): tax(16) in (1.5%, 6%], x*(2%) in (16, 64].
# R8-T2 the in-job prediction a100_pred_time.txt (from x=16/32/64) is written BEFORE x=100 runs.
#       FALSIFIER: tax at 100 outside +-30% of it (both earlier A100 runs missed low here).
#
# FAILURE CLASSES: (0) GPU fault -> exit 3; (1) not NVSwitch -> exit 4; (2) the NCCL test fails ->
# exit 5; (3) an OOM outside a ladder's closing rung is a defect (largest non-ladder arm: 7B static
# x=100 in T at 59.5 GiB on an 80 GB card).
set -uo pipefail
PLATFORM=a100; PREFIX=a100; VENV_VAR=A100_VENV; VENV_DEFAULT=$HOME/a100_venv; HOURS=24
source "$SLURM_SUBMIT_DIR/rank8_common.sh"
r8_setup
if [ -n "${R8_V100:-}" ]; then
    R8_NCCL_WORLDS="4"; W=4; L7=4                             # V100 smoke: 4 nodes x 1 GPU, 7B at depth 4
    # (not 2: at L=2 the replay instants have no prefetched next layer and the gradient pile goes
    #  negative, so the lines are not defined there)
else
    if [ "$GPN" != 8 ]; then echo "THIS JOB NEEDS ALL 8 GPUs OF A DGX A100 (have $GPN): --gres=gpu:a100:8"; exit 2; fi
    R8_NCCL_WORLDS="8"; W=8; L7=32
    # NVSWITCH CHECK: every pair among the 8 GPUs must be NVLink (NV#) in `nvidia-smi topo -m`
    FAB=$(nvidia-smi topo -m | sed 's/\x1b\[[0-9;]*m//g' | awk -v use="0,1,2,3,4,5,6,7" '
        BEGIN { n = split(use, u, ","); for (i = 1; i <= n; i++) want["GPU" u[i]] = 1 }
        NR == 1 { for (i = 1; i <= NF; i++) col[i] = $i; next }
        ($1 in want) && !($1 in seen) { seen[$1] = 1; rows++
            for (i = 2; i <= NF; i++) if ((col[i-1] in want) && col[i-1] != $1 && $i !~ /^NV[0-9]/) bad = bad " " $1 "-" col[i-1] ":" $i }
        END { if (rows != n) print "NOT CHECKED: " rows " of " n " GPU rows found"; else print (bad == "" ? "ok" : "NOT NVLINK:" bad) }')
    log "NVSWITCH CHECK (GPUs 0-7): $FAB"
    if [ "${FAB#NOT NVLINK}" != "$FAB" ]; then log "Not an NVSwitch node. Resubmit with --constraint=dgx-A100 --exclude=$(hostname -s)"; exit 4; fi
fi
r8_nodes

if [ -n "${R8_V100:-}" ] || [ "${R8_PLAN:-full}" = smoke ]; then
    # ── SMOKE: every code path at small x ──
    arm B static tier_a 64 8 3 0 $W; arm B offload tier_a 64 8 3 0 $W
    arm BZ static mistral7b $L7 8 3 0 $W; arm BZ offload mistral7b $L7 8 3 0 $W
    reader bytes
    reader natpred --pred "$OUT/${PREFIX}_pred_native_ZS.txt" --ladder ZS --shape mistral7b --L $L7 --world $W --mode static --step 2
    pairs T 1 mistral7b $L7 $W 4:4 8:4 16:4
    reader timecal --pred "$OUT/${PREFIX}_pred_time.txt" --xs 4,8,16 --at 24
    arm T static mistral7b $L7 24 3 1 $W
else
    # ── B. the base shape at eight ranks ──
    arm B static  tier_a 64 24 3 0 8; arm B static  tier_a 64 48 3 0 8
    arm B offload tier_a 64 24 3 0 8; arm B offload tier_a 64 48 3 0 8; arm B offload tier_a 64 96 3 0 8
    # ── BZ. the 7B lines ──
    arm BZ static  mistral7b 32 24 3 0 8; arm BZ static  mistral7b 32 48 3 0 8
    arm BZ offload mistral7b 32 8 3 0 8;  arm BZ offload mistral7b 32 16 3 0 8; arm BZ offload mistral7b 32 32 3 0 8
    reader bytes
    # ── T. its time ──
    pairs T 2 mistral7b 32 8 16:6 32:5 64:4
    reader timecal --pred "$OUT/${PREFIX}_pred_time.txt" --xs 16,32,64 --at 100
    pairs T 2 mistral7b 32 8 100:3
    # ── ZS, ZO. its frontier ──
    native ZS static  mistral7b 32 8 2 "134 136 138 140 142 144 146"
    ARM_TMO=14400 native ZO offload mistral7b 32 8 4 "256 260 264 268 272 276 280"
fi
reader read
log "=== DONE in $((SECONDS / 60)) min: ./rank8_collect.sh $SLURM_JOB_ID ==="
