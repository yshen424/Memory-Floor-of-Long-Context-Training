#!/bin/bash
#SBATCH --job-name=fsdp_l64_regime4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1        # 1 process per node (1 GPU per node)
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/regime4_%j.log
#SBATCH --error=logs/regime4_%j.err

# ── JOB: THE REGIME CURVE. WORLD 4, V100 ────────────────────────────────────
# Measures stock offload's time cost over static whole-layer checkpointing
# ("the tax") across x = 2..100, against a formula pre-registered below, and
# locates x*, the sequence length above which moving every boundary is cheap
# relative to compute.
#
#  R (33 arms) THE CURVE: at each x in {2,4,8,12,16,24,32,48,64,100,104},
#      static -> stock offload -> engine (gated, the "planner") on the
#      same 4 nodes, back to back. Steps: 8 at x<=16, 6 at 24-48, 4 at
#      64-104 (104 = static's largest fit). Tax = mean(steps 1+) of
#      offload / static - 1.
#  P (2 arms) PROFILE: stock offload at x=16 and x=64, --torch_profile,
#      3 steps — copy durations and achieved bandwidth from the trace,
#      read with tools/analyze_nccl_duty.py, to split the tax into
#      bandwidth and exposure.
#  D (2 arms) DRIFT CONTROL: static at x=16 and x=64 again at the end.
#      If either moves by more than the measured tax at its x, the curve
#      is read from adjacent pairs only.
#
#  All regime B (fp16 / sdpa_mem_efficient), fused-input, use_cache off,
#  b=1, v=32000, c=256, L=64, seed 1234, W=0, budget 14.9769, LOOSE. Every
#  shape fits under static's own ceiling (x=104), so the gated engine
#  should evict NOTHING and its tax is its decision cost.
#  Wall clock ~3.3 h (static fit below; ~45 min of per-arm init).

# ── THE PREDICTION. PRE-REGISTERED 2026-10-08, NOT FITTED AFTERWARDS ───────
# Inputs, all measured at world 4 before this job:
#   static step time  T(x) = 0.02123 x^2 + 0.0397 x + 2.066  s   (fit to 7744
#       block S x=8/16/24/48/104 + 7747 M1 x=100; residuals <= 0.17 s; these
#       means include step 0, which for static is within noise of steady)
#   link rates  D2H 6.54 GiB/s, H2D 7.73 GiB/s  (7744 x=356: 65.7 GiB each way
#       per two steps in 10.05 s / 8.50 s of copy time)
#   bytes moved by stock offload: 64 boundaries x 1.50x MiB each way
#   => copy time with NO hiding  k*x,  k = 0.02646 s per x
#   exposed fraction h = 0.805, calibrated at ONE point: 7747 x=100 steps 1+
#       (offload 220.04 s vs static 217.91 s)
# Model:  tax(x) = h * k * x / T(x)
#
#   x       2     4     8     12    16    24    32    48    64    100
#   h=0.805 1.91  3.32  4.55  4.56  4.19  3.35  2.72  1.93  1.49  0.98  (%)
#   h=1.0   2.37  4.13  5.66  5.67  5.20  4.17  3.38  2.40  1.85  1.21  (%)
#
# R-P1 SHAPE. The tax is NOT monotone in x: it RISES from x=2, peaks between
#      x=4 and x=24, and falls as ~1/x beyond (T(x) has a ~2 s constant, so at
#      short x the step is not yet attention-dominated). A monotone decline
#      from x=2 falsifies the model's form.
# R-P2 LONG END. For x >= 48 the measured tax lies within ±30% of the h-band
#      [h=0.805, h=1.0] at every x, and x=100 reproduces 7747's +0.98% within
#      the arms' own step-time spread (P3 of 7747 carried over).
# R-P3 SHORT END. For x <= 24 the measured tax is AT LEAST the h=0.805 curve
#      and at most 2.5x the h=1.0 curve. Reason for the asymmetric band: small
#      copies run below the bulk link rate (per-copy latency), and the
#      instrument's forced W=0 arm at x=16 (7744 block L) paid +11.2% for 44
#      boundaries, ~2.3x the bandwidth model. A tax above 2.5x falsifies the
#      bandwidth-only model at short x (an intercept term is then owed).
# R-P4 CROSSINGS. PRIMARY: x*(2%) in [40, 65] (model 46.2-58.8). The 1%
#      crossing (model 97.5-122) sits next to the calibration point and may
#      lie beyond x=104, the largest shape static can run; if so it is
#      reported as "beyond the measured range", which is consistent with the
#      model, not a failure.
# R-P5 THE PLANNER. The gated engine evicts 0 boundaries at every x
#      (evict_decisions 0), its peak equals static's to 0.01 MiB, and its tax
#      is flat in x at the decision cost (7747: +0.17% at x=100), with no
#      rise at short x. Any eviction below x=104 is a gate defect.
# R-P6 PEAKS. Static and offload peaks land on their own world-4 lines:
#      static 1476.80 + 132.3201x; offload 1866.89 + 37.820312x (step 1+)
#      to 0.01 MiB at every x >= 16 (x=8 and below may sit on the +12 MiB
#      x=8 class; record, do not fail). Step-0 losses bit-identical across
#      the three arms at each x.
# R-P7 PROFILE. At x=64 the offload trace's memcpy spans achieve within ±20%
#      of the bulk rates above; at x=16 they achieve less (latency-dominated).
#
# WHAT EACH OUTCOME MEANS: R-P1..P4 hold -> the regime boundary is a
# measured, formula-predicted quantity on V100 and the remote platforms test
# its shift. R-P3 fails high -> the model gains an intercept (per-copy
# latency) term before any remote prediction is made. R-P2 fails -> the h
# calibration does not transfer across x and the formula is not yet usable.

# --- Environment: byte-for-byte 7744/7747/7756's -----------------------------
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29900
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ATTEMPT=${SLURM_RESTART_COUNT:-0}
if [ "$ATTEMPT" = "0" ]; then ASUF=""; else ASUF="r${ATTEMPT}"; fi

echo "MASTER_ADDR  : $MASTER_ADDR"
echo "NODES        : $SLURM_JOB_NODELIST"
echo "WORLD_SIZE   : $SLURM_NTASKS"
echo "ATTEMPT      : $ATTEMPT  (output suffix '${ASUF}')"

if [ "$SLURM_NTASKS" != "4" ]; then
    echo "WORLD SIZE IS ${SLURM_NTASKS}, NOT 4. The static fit and link rates are world 4."
    exit 2
fi

OUTROOT=profile_results/regime4
mkdir -p logs "$OUTROOT"

# --- Preflight: every node must have its RAM back ----------------------------
MIN_AVAIL_GIB=100
AVAIL=$(srun --ntasks-per-node=1 bash -c \
    'echo "$(hostname) $(awk "/MemAvailable/{printf \"%d\", \$2/1048576}" /proc/meminfo)"' \
    < /dev/null)
echo "MemAvailable per node (GiB):"
echo "$AVAIL" | sed 's/^/    /'
if echo "$AVAIL" | awk -v m=$MIN_AVAIL_GIB 'NF<2 || $2<m {bad=1} END {exit !bad}'; then
    echo "PREFLIGHT FAILED: a node has under ${MIN_AVAIL_GIB} GiB MemAvailable, or did not answer."
    exit 3
fi

# --- The arms: block:x:mode:steps ---------------------------------------------
# `for` over a word list, never `while read` (srun drains stdin).
# Within each x the three arms run back to back on the same nodes, so the tax
# is a paired difference. Short x first: they are cheap and they carry R-P1.
ARMS=""
for X in 2 4 8 12 16 24 32 48 64 100 104; do
    if   [ $X -le 16 ]; then N=8
    elif [ $X -le 48 ]; then N=6
    else                     N=4; fi
    ARMS="$ARMS R:$X:static:$N R:$X:offload:$N R:$X:engine:$N"
done
ARMS="$ARMS P:16:offload_profile:3 P:64:offload_profile:3 D:16:static:8 D:64:static:4"

HID=768; INTER=2048; NH=12; NKV=4; NL=64; BS=1; VOC=32000; CHUNK=256
DT=fp16; ATTN=sdpa_mem_efficient; NORM=input; SEED=1234; BUDGET=14.9769; WIN=0
PORT_BASE=${MASTER_PORT}
IDX=0
NARM=$(echo $ARMS | wc -w)

run_arm() {
    SQ=$((X * 1024))
    case "$MODE" in
        static)          MODE_FLAGS="--layer_checkpoint" ;;
        offload)         MODE_FLAGS="--layer_checkpoint --activation_offload" ;;
        offload_profile) MODE_FLAGS="--layer_checkpoint --activation_offload --torch_profile" ;;
        engine)          MODE_FLAGS="--layer_checkpoint --placement_engine --engine_live --hbm_budget_gb ${BUDGET} --prefetch_window_layers ${WIN} --evict_layout_census" ;;
        *) echo "UNKNOWN MODE ${MODE}"; return 2 ;;
    esac
    TAG=h${HID}L${NL}_${BLOCK}_${MODE}_x${X}_n${NSTEPS}_c${CHUNK}_norminput_cacheoff
    OUTDIR=${OUTROOT}/${SLURM_JOB_ID}${ASUF}-${TAG}_b${BS}_v${VOC}_${DT}_mem_efficient_world4
    mkdir -p "$OUTDIR"
    IDX=$((IDX + 1))
    echo "════════════════════════════════════════════════════════════════════"
    echo "ARM ${IDX}/${NARM} [block ${BLOCK}] x=${X} (seq ${SQ}) mode=${MODE} steps=${NSTEPS}"
    echo "  mode flags: ${MODE_FLAGS}"
    echo "  -> ${OUTDIR}"
    echo "════════════════════════════════════════════════════════════════════"
    srun torchrun \
        --nnodes=$SLURM_NNODES \
        --nproc_per_node=1 \
        --rdzv_id=${SLURM_JOB_ID}${ASUF}_a${IDX} \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$((PORT_BASE + IDX)) \
        train_fsdp_profile.py \
            --hidden_size ${HID} --intermediate_size ${INTER} --num_layers ${NL} \
            --num_heads ${NH} --num_kv_heads ${NKV} \
            --vocab_size ${VOC} \
            --dtype ${DT} \
            --attn_impl ${ATTN} \
            --seq_len ${SQ} \
            --batch_size ${BS} --steps ${NSTEPS} \
            --chunked_ce ${CHUNK} \
            ${MODE_FLAGS} \
            --fused_rmsnorm ${NORM} \
            --use_cache off \
            --seed ${SEED} \
            --disable_hbm_drift \
            --output_dir "$OUTDIR" \
            < /dev/null \
        || echo "ARM ${IDX} (block ${BLOCK}, mode ${MODE}, x=${X}) DID NOT COMPLETE. Every shape here is BELOW static's own ceiling (x=104 fits at 15238.11 MiB), so ANY failure is NOT a capacity result: (0) node failure / socket timeout / 'Task launch failed' -> re-run the arm; (1) a CUDA OOM at x<=100 means the build or the nodes moved — stop reading the curve; (2) a cudaHostAlloc / pinned-memory failure on an offload arm is a host limit (96x MiB per rank, 9.4 GiB at x=100) and should not happen on 112 GB nodes; (3) a P-arm failure is most likely the profiler's own memory — the R curve stands without it."

    if [ "$BLOCK" = "P" ]; then
        TRACE=$(ls -t "$OUTDIR"/torch_trace_rank0/*.pt.trace.json* 2>/dev/null | head -n 1)
        if [ -n "$TRACE" ]; then
            python3 tools/analyze_nccl_duty.py "$TRACE" > "$OUTDIR/memcpy_analysis.txt" 2>&1 \
                || echo "P x=${X}: trace analysis failed — the trace is kept."
            tail -n 25 "$OUTDIR/memcpy_analysis.txt"
        fi
    fi
}

for ARM in $ARMS; do
    OLDIFS=$IFS; IFS=:; set -- $ARM; IFS=$OLDIFS
    BLOCK=$1; X=$2; MODE=$3; NSTEPS=$4
    run_arm
    sleep 20   # let NCCL tear down and the rdzv port clear before the next arm
done

echo "RUN COMPLETE: ran ${IDX} of ${NARM} arms."
echo ""
echo "READ IT WITH:  python3 tools/regime_curve.py ${OUTROOT} --job ${SLURM_JOB_ID}${ASUF}"
echo "  (prints the paired tax per x against the pre-registered band, the crossings,"
echo "   the planner's evictions, and the peak checks R-P6)"
echo ""
echo "AFTER THE JOB: append one row per arm to profile_results/runs_4gpu.csv (back it"
echo "up first) and move this script to ${OUTROOT}/."
