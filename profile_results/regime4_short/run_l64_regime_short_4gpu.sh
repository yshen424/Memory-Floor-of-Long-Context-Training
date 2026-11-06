#!/bin/bash
#SBATCH --job-name=fsdp_l64_rshort4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1        # 1 process per node (1 GPU per node)
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/rshort4_%j.log
#SBATCH --error=logs/rshort4_%j.err

# ── JOB: THE SHORT END OF THE REGIME CURVE. WORLD 4, V100 ───────────────────
# 7758 measured stock offload's extra step time as linear in x from x=24 up
# (kappa ~ 0.032 s per x) but could not resolve x<=16: step-time noise
# (sd ~0.2 s on 2-8 s steps, 7 steps per arm) exceeded the effect. This job
# resolves it with 40 steps per arm and two repeats in ABBA order, so slow
# drift cancels.
#
#  S (28 arms): at each x in {2,4,6,8,10,12,16}: static, offload, offload,
#      static (ABBA), 40 steps each. Steps 0-1 excluded from every mean.
#
#  Same configuration as 7758: L=64, lever 3 c=256, regime B fp16 /
#  sdpa_mem_efficient, fused-input, cache off, b=1, seed 1234. No engine
#  arm (7758 covers its behaviour below the ceiling).
#  Wall clock ~2 h (~80 min of steps, ~35 min of per-arm init).

# ── THE PREDICTION. PRE-REGISTERED 2026-10-09 ──────────────────────────────
# Inputs (measured before this job): static step time
#   T(x) = 0.02123 x^2 + 0.0397 x + 2.066 s; 7758's per-x cost of stock
#   offload at x>=24, kappa = 0.032 s per x.
# Two hypotheses, stated so the data can decide between them:
#   H-LIN   the per-x cost is the same at short x: delta(x) = kappa * x
#             x        2     4     6     8     10    12    16
#             delta s  0.064 0.128 0.192 0.256 0.320 0.384 0.512
#             tax %    2.87  4.99  6.26  6.84  6.98  6.86  6.29
#           -> the tax RISES to a peak near x = sqrt(c/a) = 9.9 (~7.0%) and
#              falls after it.
#   H-SHORT the per-x cost is smaller at short x (7758's short arms read
#           0.008-0.022 s per x, inside their noise): delta(x)/x rises toward
#           0.032 by x~24, and the tax is flatter, peaking at ~4-5% near x=16-24.
# S-P1  delta(x) increases monotonically with x (both hypotheses).
# S-P2  THE DECIDER: the mean of delta(x)/x over x = 2..12. Within ±25% of
#       0.032 (0.024-0.040) supports H-LIN; below 0.024 supports H-SHORT.
# S-P3  The tax at x=2 is LOWER than at x=8 (both hypotheses: the rise
#       exists). A tax at x=2 at or above x=8's falsifies "rise then fall" and
#       makes the tax monotone over the whole measured range.
# S-P4  Precision: the standard error of each delta is <= 0.03 s (40 steps,
#       sd ~0.2 s, two repeats). If it is not, the arm's spikes are reported
#       (median and trimmed mean beside the mean) rather than averaged away.
# S-P5  Peaks: every arm on its harness line (static and floor W=0); step-0
#       losses bit-identical between static and offload at each x.

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29900
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ATTEMPT=${SLURM_RESTART_COUNT:-0}
if [ "$ATTEMPT" = "0" ]; then ASUF=""; else ASUF="r${ATTEMPT}"; fi
echo "NODES: $SLURM_JOB_NODELIST  WORLD_SIZE: $SLURM_NTASKS  ATTEMPT: $ATTEMPT"

if [ "$SLURM_NTASKS" != "4" ]; then
    echo "WORLD SIZE IS ${SLURM_NTASKS}, NOT 4. The static fit and kappa are world 4."
    exit 2
fi

OUTROOT=profile_results/regime4_short
mkdir -p logs "$OUTROOT"

MIN_AVAIL_GIB=100
AVAIL=$(srun --ntasks-per-node=1 bash -c \
    'echo "$(hostname) $(awk "/MemAvailable/{printf \"%d\", \$2/1048576}" /proc/meminfo)"' \
    < /dev/null)
echo "MemAvailable per node (GiB):"; echo "$AVAIL" | sed 's/^/    /'
if echo "$AVAIL" | awk -v m=$MIN_AVAIL_GIB 'NF<2 || $2<m {bad=1} END {exit !bad}'; then
    echo "PREFLIGHT FAILED: a node has under ${MIN_AVAIL_GIB} GiB MemAvailable, or did not answer."
    exit 3
fi

# rep tag a/b keeps the two repeats apart; ABBA order within each x.
ARMS=""
for X in 2 4 6 8 10 12 16; do
    ARMS="$ARMS $X:static:a $X:offload:a $X:offload:b $X:static:b"
done

HID=768; INTER=2048; NH=12; NKV=4; NL=64; BS=1; VOC=32000; CHUNK=256
DT=fp16; ATTN=sdpa_mem_efficient; NORM=input; SEED=1234; NSTEPS=40
PORT_BASE=${MASTER_PORT}
IDX=0
NARM=$(echo $ARMS | wc -w)

for ARM in $ARMS; do
    OLDIFS=$IFS; IFS=:; set -- $ARM; IFS=$OLDIFS
    X=$1; MODE=$2; REP=$3; SQ=$((X * 1024))
    case "$MODE" in
        static)  MODE_FLAGS="--layer_checkpoint" ;;
        offload) MODE_FLAGS="--layer_checkpoint --activation_offload" ;;
    esac
    TAG=h${HID}L${NL}_S_${MODE}_x${X}_rep${REP}_n${NSTEPS}_c${CHUNK}_norminput_cacheoff
    OUTDIR=${OUTROOT}/${SLURM_JOB_ID}${ASUF}-${TAG}_b${BS}_v${VOC}_${DT}_mem_efficient_world4
    mkdir -p "$OUTDIR"
    IDX=$((IDX + 1))
    echo "ARM ${IDX}/${NARM}: x=${X} mode=${MODE} rep=${REP} steps=${NSTEPS} -> ${OUTDIR}"
    srun torchrun \
        --nnodes=$SLURM_NNODES --nproc_per_node=1 \
        --rdzv_id=${SLURM_JOB_ID}${ASUF}_a${IDX} --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$((PORT_BASE + IDX)) \
        train_fsdp_profile.py \
            --hidden_size ${HID} --intermediate_size ${INTER} --num_layers ${NL} \
            --num_heads ${NH} --num_kv_heads ${NKV} --vocab_size ${VOC} \
            --dtype ${DT} --attn_impl ${ATTN} \
            --seq_len ${SQ} --batch_size ${BS} --steps ${NSTEPS} \
            --chunked_ce ${CHUNK} ${MODE_FLAGS} \
            --fused_rmsnorm ${NORM} --use_cache off --seed ${SEED} \
            --disable_hbm_drift --output_dir "$OUTDIR" \
            < /dev/null \
        || echo "ARM ${IDX} (x=${X}, ${MODE}, rep ${REP}) DID NOT COMPLETE. Every shape here is far below static's ceiling (x=104), so this is NOT a capacity result: check the .err for a node failure or socket timeout and re-run the arm."
    sleep 20
done

echo "RUN COMPLETE: ran ${IDX} of ${NARM} arms."
echo "READ IT WITH:  python3 tools/regime_short.py ${OUTROOT} --job ${SLURM_JOB_ID}${ASUF}"
echo "AFTER THE JOB: append rows to profile_results/runs_4gpu.csv (back it up first) and"
echo "move this script to ${OUTROOT}/."
