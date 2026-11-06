#!/bin/bash
#SBATCH --job-name=fsdp_l64_w12_4g
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1        # 1 process per node (1 GPU per node)
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --output=logs/l64w12_4g_%j.log
#SBATCH --error=logs/l64w12_4g_%j.err

# ── JOB: THE W=2 AND W=1 LADDERS AT WORLD 4 (V100). L=64, LEVER 3, c=256 ────
# A floor line's INTERCEPT is world-dependent (FSDP shards the fp32 master
# copy, AdamW state and the relocated gradient pile across ranks), so W=2
# and W=1 are measured at world 4. W=0 and the static bracket are in the
# headline job, not here.
#
# IT IS A PREDICTION TEST, NOT A SEARCH. Every SLOPE is rank-invariant —
# each rank holds the same activations at b=1 — and every INTERCEPT shift is
# derived below from parameter counts alone.
#
#  B2 (W=2)  BRACKET SEARCH starting at the predicted largest fit x=328.
#  B1 (W=1)  BRACKET SEARCH starting at the predicted largest fit x=340.
#  Z  (W=1)  TIME TWIN at W=2's measured largest fit — same shape, one
#            parameter changed — so "restore depth costs nothing in time"
#            is measured at world 4.
#
#  All regime B (fp16 / sdpa_mem_efficient), fused-input, use_cache off,
#  b=1, v=32000, c=256, WORLD 4, seed 1234, L=64, budget 14.9769, loose,
#  2 steps.

# ── WHY LOOSE ───────────────────────────────────────────────────────────────
# The gate over-reports at the pick cap, so a strict arm measures the gate's
# arithmetic and never reaches the hardware.
#
# THE THREE POST-HOC CHECKS THAT REPLACE --strict_evict, on every rung:
#     evict_realization.realized_fraction == 1.0
#     evict_realization.denied_ops == 0  and  pool.denials_total == 0
#     relief == picks x one full boundary (1.5000x MiB), ZERO residual
# A rung failing any of these has not measured capacity and must not be quoted.

# ── THE WORLD-4 LINES, DERIVED 2026-09-29 FROM PARAMETER COUNTS ────────────
# Parameters: 6,292,992 per layer x 64 + root unit 49,152,768 (embed + untied
# head + final norm) = 451,904,256. Sharded per rank:
#   AdamW state     8 B/param / world
#   fp32 master     4 B/param / world : per-layer fixed term 12 B/param / world
#   gradient shard  U4 = 6.001658 MiB (an earlier measured shard, rescaled
#                   to four ranks)
# The root unit's un-resharded fp16 copy (2 B/param) is NOT sharded and does
# not move. Intercepts at step 0 (fp32 master only) and step 1 (fp32
# master + AdamW):
#
#   static step-1 line   1476.80 + 132.320279x
#                        -> largest fit x=104 (15238.11), x=108 fails (15767.39)
#                        -> THE SPAN DENOMINATOR IS PREDICTED AT 104.
#                           Measured in the headline job, NOT here.
#   pile at step 1       66 U4 at W=2 (396.11), 67 U4 at W=1 (402.11)
#   pile at step 0       61 U4 at W=2 (366.10), 62 U4 at W=1 (372.10)
#
#   W=2 floor   step 0   980.96 + 40.820279x    step 1  1872.91 + 40.820279x
#   W=1 floor   step 0   986.96 + 39.320279x    step 1  1878.91 + 39.320279x
#   (W=0, for the headline job:  step 1  1878.91 + 37.820312x)
#
# THE ONE UNPINNED ASSUMPTION: everything else in the intercept (all-gather
# buffers, the root's fp16 copy, rotary tables, allocator context) is
# world-independent. If the measured step-1 intercept misses by more than
# ~2 MiB, that assumption is what failed, and the miss is itself the
# finding — it names a world-dependent term the byte model does not contain.

# ── THE LADDER, COMPUTED ───────────────────────────────────────────────────
# Reference capacity on this card (RUNG
# PLACEMENT ONLY — a bracket is never carried across configs):
#   W=2 [15352.08 achieved, ~15416.3 refused]   W=0 [15231.77, ~15373.4)
#   overlap of all three depths [15352.08, ~15373.4)
#
#     x    seq      W=2 step1   W=1 step1   verdict
#   324  331776     15098.68        -       W=2 FIT (descend target)
#   328  335872     15261.96        -       W=2 FIT, 90 MiB under ref  <-- B2 START
#   332  339968     15425.24        -       W=2 FAIL, 9 MiB over W=2's ref ceiling  (THIN)
#   336  344064         -       15090.52    W=1 FIT (descend target)
#   340  348160         -       15247.80    W=1 FIT, 104 MiB under ref <-- B1 START
#   344  352256         -       15405.08    W=1 INSIDE THE REFERENCE BAND — coin flip
#   348  356352         -       15562.37    W=1 FAIL
#
# Predicted spans against static's predicted x=104:
#   W=2 328/104 = 3.154    W=1 340/104 = 3.269 (344 -> 3.308)    [W=0 356/104 = 3.423]
# The ORDER must hold (W=2 < W=1 < W=0).

# ── THE SEARCH RULE (per depth) ────────────────────────────────────────────
# Start at the predicted largest fit. If it fits, climb +4 until a rung fails.
# If it fails, descend -4 until a rung fits. At most MAX_RUNGS rungs per depth.
# The result is always a fit/fail pair 4 x-units apart, at 2 rungs in the
# best case, 3 typically. Each rung is ~85 min at these shapes (two ~40 min
# steps + init). A failing rung dies in step 1's backward and costs ~65 min.
# Expected total: 2-3 rungs x 2 depths + 1 twin = 5-7 rungs, ~7.5-10 h.

# ── PRE-REGISTERED PREDICTIONS ─────────────────────────────────────────────
#
# EVERY COMPLETING RUNG, BOTH DEPTHS:
#   A-P1  PICKS = L-W-1: 61 at W=2, 62 at W=1, in BOTH steps. The pick cap is
#         structural and world-independent. Any other count outranks
#         everything below.
#   A-P2  STEP-1 PEAK ON THE WORLD-4 FLOOR LINE ABOVE to within 2 MiB. After
#         the FIRST rung calibrates the intercept, every later rung at that
#         depth must land on it to < 0.01 MiB. Record the first rung's
#         residual against the derived intercept separately: THAT is the
#         world-scaling test.
#   A-P3  STEP-0 PEAK on 980.96 / 986.96 + slope, same tolerances.
#   A-P4  STEP-1 minus STEP-0 intercept = AdamW(4) + 5 U4 = 861.94 + 30.01 =
#         891.95 MiB at both depths. This isolates the sharded optimizer term
#         from everything else; if it holds and A-P2 misses, the miss is in
#         the step-0 (non-optimizer) intercept.
#   A-P5  RELOCATED PEAK at post_bwd_layer_2 (W=2) and post_bwd_layer_1
#         (W=1), phi reading 0.0 against it with relief FULL — the known
#         relocated-peak trap, not a defect. Read realized relief against the
#         unrelieved line, never peak_coverage_fraction.
#   A-P6  census ONE class, layer/torch.float16 at the boundary size
#         (1.5 x MiB: e.g. 515899392 B at x=328, 534773760 at x=340);
#         noncontiguous 0 / offset_nonzero 0 / storage_larger 0. No NaN.
#
# THE FAILING RUNG:
#   F-P1  CUDA OOM IN BACKWARD IN STEP 1, all four ranks. The failing request
#         is 4x MiB — 2 B x tokens x intermediate, the replayed feed-forward
#         activation: 1328.00 at x=332, 1344.00 at 336, 1376.00 at 344,
#         1392.00 at 348. A different size is a different mechanism and the
#         rung below is not a bracket.
#   F-P2  Allocated-when-refused + request lands inside or above the
#         reference band. Record it: it is this config's capacity upper
#         reference, and it belongs to THIS config only.
#
# THE TIME TWIN (arm Z, W=1 at W=2's largest fit):
#   Z-P1  Same peak line as B1 (it is a W=1 rung), so it must land on the
#         B1-calibrated line to < 0.01 MiB.
#   Z-P2  Step time within run-to-run spread of the W=2 rung at the same x.
#         Expect ~2440 s/step at x=328 (an earlier quadratic step-time fit,
#         scaled ~0.9 for the four-rank collectives), not an exact number.
#
# RIDER — the gate's anchor split (predicted_anchor_mb per step, anchor_live_
# bytes on every projecting decision) is recorded on every arm at no cost.
# Prediction: the 3.265625 MiB/x anchor excess and the W-invariant
# over-report slope 3.992 MiB/x carry over; the intercept does NOT (it is
# world-dependent). Recording only — no decision reads it.

# ── WHAT THIS JOB DOES NOT DELIVER ─────────────────────────────────────────
# The static bracket and the span DENOMINATOR (headline job). W=0 (headline
# job). Losslessness (headline job's 10-step control). The gate's pick
# sequence below the cap (headline job's W=0 ladder).

# --- Environment ---
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Same allocator config as every fit; changing it changes the fragmentation
# component of the peak and the capacity being measured.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MAX_RUNGS=${MAX_RUNGS:-4}      # per depth; a search needing more has missed its prediction badly
RUN_TWIN=${RUN_TWIN:-1}        # arm Z; set 0 to skip if the night is short

echo "MASTER_ADDR  : $MASTER_ADDR"
echo "NODES        : $SLURM_JOB_NODELIST"
echo "WORLD_SIZE   : $SLURM_NTASKS"
echo "MAX_RUNGS    : $MAX_RUNGS"
echo "RUN_TWIN     : $RUN_TWIN"

if [ "$SLURM_NTASKS" != "4" ]; then
    echo "WORLD SIZE IS ${SLURM_NTASKS}, NOT 4. Every prediction above is derived for world 4"
    echo "  and the intercepts move with world size. Refusing to run."
    exit 1
fi

mkdir -p logs profile_results

HID=768
INTER=2048
NH=12
NKV=4
BS=1
VOC=32000
DT=fp16
ATTN=sdpa_mem_efficient
NORM=input
SEED=1234
NL=64
CHUNK=256
BUDGET=14.9769
NSTEPS=2
PORT_BASE=${MASTER_PORT}
IDX=0

# One rung, start to finish. Args: BLOCK W X. Sets OUTDIR; returns 0 iff the
# rung completed (a completed run writes steps_rank0.json; a run that died in
# a step does not).
run_rung() {
    local BLOCK=$1 WIN=$2 XVAL=$3
    local SQ=$((XVAL * 1024))
    local MODE=engine_loose
    # THE INSTRUMENT, LOOSE. The gate still projects, books infeasible steps
    # and picks every bundle it can reach; it simply does not raise.
    local MODE_FLAGS="--layer_checkpoint --placement_engine --engine_live --hbm_budget_gb ${BUDGET} --prefetch_window_layers ${WIN} --evict_layout_census"
    local TAG=h${HID}L${NL}_${BLOCK}_${MODE}_b${BUDGET}_w${WIN}_c${CHUNK}_s${SQ}_n${NSTEPS}_norminput_cacheoff
    OUTDIR=profile_results/l64w12_4gpu/${SLURM_JOB_ID}-${TAG}_b${BS}_v${VOC}_${DT}_mem_efficient_world4
    mkdir -p "$OUTDIR"
    IDX=$((IDX + 1))
    echo "════════════════════════════════════════════════════════════════════"
    echo "RUNG ${IDX} [block ${BLOCK}]: W=${WIN} x=${XVAL}"
    echo "  layers=${NL}  seq_len=${SQ}  chunked_ce=${CHUNK}  world=${SLURM_NTASKS}"
    echo "  hidden=${HID} inter=${INTER} heads=${NH}/${NKV} b=${BS} v=${VOC} ${DT}/${ATTN}"
    echo "  norm=${NORM}  use_cache=off  steps=${NSTEPS}  seed=${SEED}"
    echo "  mode flags: ${MODE_FLAGS}"
    echo "  -> ${OUTDIR}"
    echo "════════════════════════════════════════════════════════════════════"
    srun torchrun \
        --nnodes=$SLURM_NNODES \
        --nproc_per_node=1 \
        --rdzv_id=${SLURM_JOB_ID}_a${IDX} \
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
        || echo "RUNG ${IDX} (block ${BLOCK}, W=${WIN}, x=${XVAL}) DID NOT COMPLETE — READ THE FAILURE CLASS BEFORE THE SEARCH MOVES ON. (1) CUDA OOM IN BACKWARD IN STEP 1 is the designed stop (F-P1): record allocated/reserved and the FAILING REQUEST SIZE, which must be 4x MiB ($((XVAL * 4)).00 here); a different size is a different mechanism and this rung is not a bracket. It does NOT bound capacity from above. (2) CUDA OOM IN FORWARD is a diagnostically different class: the shape is far above the ceiling, the rung below is a LOOSE bracket and one intermediate rung is owed. (3) A GATE REFUSAL RAISING must not happen (--strict_evict is off); if it does the arm was built wrong — check strict_evict in the config block. (4) A POOL DENIAL (cpu_dram_full) is silent: check pool.denials_total and evict_realization.denied_ops on EVERY completing rung; a denial at a size other than this x's boundary ($((XVAL * 1572864)) B) is a Gate-0 bundle-mismatch defect and voids the job. (5) cudaHostAlloc failure at init is the pool asking for ~34-35 GiB of host DRAM: an implementation limit, never a finding; do NOT touch --pool_buffers_per_bucket. (6) NaN loss or a census class other than the single fp16 residual boundary: read evict_layout_census first. (7) A HANG or wall-clock kill: check whether step 0 printed before assuming capacity."
    sleep 20   # let NCCL tear down and the rdzv port clear before the next rung
    [ -f "$OUTDIR/steps_rank0.json" ]
}

# Bracket search for one depth. Args: BLOCK W START. Sets FIT_X and FAIL_X
# (empty if not found within MAX_RUNGS).
search() {
    local BLOCK=$1 WIN=$2 X=$3 N=0 DIR=""
    FIT_X=""; FAIL_X=""
    while [ $N -lt $MAX_RUNGS ]; do
        N=$((N + 1))
        if run_rung "$BLOCK" "$WIN" "$X"; then
            FIT_X=$X
            [ -n "$FAIL_X" ] && break          # descended onto a fit: bracket closed
            DIR=up; X=$((X + 4))
        else
            FAIL_X=$X
            [ -n "$FIT_X" ] && break           # climbed into a fail: bracket closed
            DIR=down; X=$((X - 4))
        fi
    done
    echo "SEARCH W=${WIN}: largest fit x=${FIT_X:-NONE}  smallest fail x=${FAIL_X:-NONE}  (${N} rungs)"
    if [ -z "$FIT_X" ] || [ -z "$FAIL_X" ]; then
        echo "  BRACKET NOT CLOSED within MAX_RUNGS=${MAX_RUNGS}. The derived intercept missed by"
        echo "  more than $((MAX_RUNGS * 4)) x-units (~$((MAX_RUNGS * 160)) MiB) — that miss is the finding;"
        echo "  read A-P2's residual on the rungs that did run before re-submitting."
    fi
}

# ── B2: W=2, start at predicted largest fit 328 ────────────────────────────
search B2 2 328
W2_FIT=$FIT_X; W2_FAIL=$FAIL_X

# ── B1: W=1, start at predicted largest fit 340 ────────────────────────────
search B1 1 340
W1_FIT=$FIT_X; W1_FAIL=$FAIL_X

# ── Z: W=1 time twin at W=2's largest fit (skip if B1 already ran that x) ──
if [ "$RUN_TWIN" = "1" ] && [ -n "$W2_FIT" ]; then
    if ls -d profile_results/l64w12_4gpu/${SLURM_JOB_ID}-h${HID}L${NL}_B1_*_w1_*_s$((W2_FIT * 1024))_* >/dev/null 2>&1; then
        echo "TWIN SKIPPED: B1 already ran W=1 at x=${W2_FIT}; that rung is the twin."
    else
        run_rung Z 1 "$W2_FIT" || echo "TWIN at x=${W2_FIT} DID NOT COMPLETE. It sits ~150 MiB below W=1's own largest fit, so this is NOT a capacity result — read the failure class."
    fi
fi

echo "RUN COMPLETE: ran ${IDX} rungs."
echo "  W=2 : largest fit x=${W2_FIT:-NONE}  smallest fail x=${W2_FAIL:-NONE}   (predicted 328 / 332)"
echo "  W=1 : largest fit x=${W1_FIT:-NONE}  smallest fail x=${W1_FAIL:-NONE}   (predicted 340 / 344-or-348)"
echo "Lines every rung is read against (world 4, derived 2026-09-29):"
echo "  W=2 floor   step0  980.96 + 40.820279x   step1 1872.91 + 40.820279x"
echo "  W=1 floor   step0  986.96 + 39.320279x   step1 1878.91 + 39.320279x"
echo "  step1 - step0 intercept = 891.95 at both depths (AdamW 861.94 + 5 x U4 30.01)"
echo "  U4 = 6.001658 MiB"
echo "  picks = L-W-1: 61 at W=2, 62 at W=1"
echo "  spans: largest_fit_x / static largest fit from the headline job"

# ── VALIDITY CHECKS, IN ORDER ──────────────────────────────────────────────
#   1. ARM IDENTITY on every rung before any number is read:
#        world_size == 4               fused_rmsnorm_replaced == 129
#        config.use_cache == false     layer_checkpoint_wrapped == 64
#        chunked_ce_tokens == 256      (NOT config.chunked_ce, always None)
#        hbm_budget_gb == 14.9769      prefetch_window_layers == the rung's W
#        engine_live == true           strict_evict == false
#      `seed` and `evict_layout_census` are absent from the config block
#      (a recording gap) — read them off THIS script.
#   2. A-P2 ON THE FIRST RUNG OF EACH DEPTH: residual of the step-1 peak
#      against the DERIVED intercept. This is the world-scaling test and the
#      input the headline job's W=0 and static predictions depend on.
#   3. A-P4: step-1 minus step-0 intercept = 891.95.
#   4. The three checks that replace --strict_evict, on EVERY completing rung.
#   5. A-P1 picks, A-P5 peak location, A-P6 census.
#   6. The failure class at each depth's failing rung (F-P1).
#   7. Z-P2: step time at the shared x, W=2 against W=1.
#   8. The spans are NOT computed here — the denominator is the headline job's
#      static bracket at world 4. Quote each as largest_fit_x / static_fit_x
#      with W, lever state, kernel, vocabulary, chunk size AND WORLD SIZE.
#   9. Record every rung in profile_results/runs_4gpu.csv.
