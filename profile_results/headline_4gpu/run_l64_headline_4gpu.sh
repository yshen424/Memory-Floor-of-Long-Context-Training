#!/bin/bash
#SBATCH --job-name=fsdp_l64_head_4g
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1        # 1 process per node (1 GPU per node)
#SBATCH --gres=gpu:1
#SBATCH --time=14:00:00
#SBATCH --output=logs/l64head_4g_%j.log
#SBATCH --error=logs/l64head_4g_%j.err

# ── JOB: THE HEADLINE AT WORLD 4 (V100) ─────────────────────────────────────
# The world-dependent numbers not covered by the W=2/W=1 job
# (run_l64_w12_4gpu.sh).
#
#  S (L=64) STATIC LINE + BRACKET — the span's DENOMINATOR. 4 line points,
#           then a bracket search from the predicted largest fit x=104.
#  E (L=64) THE ENGINE AT W=0 — the span's NUMERATOR. Below-cap rungs at
#           static's failing x, 140 and 280 (gate pick sequence, relief
#           reconstruction), the SHARED rung for the W=0/W=1/W=2 time
#           comparison, then a bracket search from x=356.
#  L (L=64) LOSSLESSNESS — 10 steps, engine vs static twin at x=16. Also
#           the sub-ceiling time cost.
#  U (L=64) SLOPE INVARIANCE CHECK — whole-layer, UNCHUNKED, x=2/4/8.
#  C (L=32) CHUNKED-CE COST PAIR — c=1024 vs unchunked, 10 steps, x=2.
#  T (L=32) THE ANCHOR LINE + BRACKET — 4 line points, then a bracket
#           search from x=168.
#
#  All regime B (fp16 / sdpa_mem_efficient), fused-input, use_cache off,
#  b=1, v=32000, WORLD 4, seed 1234. c=256 unless the block says otherwise.
#
#  QUEUE IT BEHIND run_l64_w12_4gpu.sh WITH A DEPENDENCY:
#    sbatch --dependency=afterany:<w12 job id> run_l64_headline_4gpu.sh
#  SHARED_X (the shared rung) is read at RUN TIME from that job's completed
#  W=2 rungs; SHARED_X=<x> overrides; 328 is the fallback.

# ── THE WORLD-4 LINES, DERIVED 2026-09-29 FROM PARAMETER COUNTS ────────────
# Same derivation as run_l64_w12_4gpu.sh. Slopes are rank-invariant (each
# rank holds the same activations at b=1). Intercepts carry the four-rank
# sharded state: fp32 master 4 B/param and AdamW 8 B/param sharded per rank; the root unit's fp16 copy
# is not sharded. L=64 has 451,904,256 params, L=32 250,528,512.
# U4 (one fp32 gradient shard) = 6.001658 MiB.
#
#   S  static L=64 c=256   1476.80 + 132.320279x
#   E  W=0 floor, step 1   1878.90 + 37.820312x
#   T  static L=32 c=256    900.66 + 84.320299x
#   U  static L=64 unchunked 1414.92 + 536.77x
#
# THE ONE UNPINNED ASSUMPTION: every other intercept term is
# world-independent. run_l64_w12_4gpu.sh tests it first; if that job
# reported an intercept miss, ADD THE MISS to every intercept here before
# reading any verdict — the rung placement has 2-3 x-units of slack for it.
# Each config below gets its OWN bracket; none is carried across configs.

# ── PRE-REGISTERED PREDICTIONS ─────────────────────────────────────────────
#
# BLOCK S — static L=64, c=256, 3 steps (a rung counts only once AdamW is
# co-resident; a failure can land in step 2, not step 1).
#   S-P1  LINE POINTS (step max): x=8 2547.36 | 16 3593.93 | 24 4652.48 |
#         48 7828.17 MiB. x=8 is expected ~12 MiB above its own line and
#         is quoted separately or not at all. SLOPE over 16/24/48 must be
#         132.320279 to < 0.001 — THE WORLD-INVARIANCE TEST FOR LEVER 3.
#         Step 0 at x=16: 2731.98.
#   S-P2  BRACKET: x=104 FITS at 15238.11; x=108 FAILS (line 15767.39).
#         THE DENOMINATOR IS 104.
#         Failing request 4x MiB = 432.00 at x=108, in backward, in step 1 or
#         2 — the replayed feed-forward activation of one layer.
#
# BLOCK E — W=0, L=64, c=256, loose, budget 14.9769, 2 steps.
#   E-P1  BELOW THE CAP (x = static's failing x, 140, 280): picks rise with
#         the breach, ~5 / ~25 / ~58 (+/-2; the gate over-picks by 2, a known
#         structural offset), phi = 1.0, peak at post_bwd_layer_63, relief =
#         picks x 1.5x MiB exactly.
#   E-P2  RELIEF RECONSTRUCTION: on every below-cap rung, step peak + realized
#         relief lands on block S's MEASURED line to < 0.01 MiB. This is the
#         decomposition predicting the engine from the static line; it
#         outranks the bracket if it fails.
#   E-P3  AT AND ABOVE THE SHARED RUNG: 63 picks (L-1, the W=0 cap), peak
#         relocated to post_bwd_layer_1, phi reading 0.0 against it with relief
#         FULL (the relocated-peak trap, not a defect), step-1 peak on the
#         floor 1878.90 + 37.820312x: 328 -> 14283.97 | 352 -> 15191.65 |
#         356 -> 15342.93 | 360 -> 15494.21.
#   E-P4  BRACKET: x=352 fits with margin; x=356 IS INSIDE W=0's REFERENCE
#         CAPACITY BAND on this card (15342.93 vs [15231.77, 15373.4)) — a
#         genuine coin flip; x=360 fails. Failing request 4x MiB (1424.00 at
#         356, 1440.00 at 360), step-1 backward, all four ranks.
#         PREDICTED SPAN: 352/104 = 3.385 or 356/104 = 3.423.
#   E-P5  TIME: transfer share of step time 0.3-0.4% and FLAT across 140 ->
#         SHARED_X -> top rung, exposed PCIe stall 0.0. Step time at SHARED_X
#         within run-to-run spread of the W=2 and W=1 rungs at the same x
#         from run_l64_w12_4gpu.sh.
#   E-P6  census one class at every rung; pool denials 0.
#
# BLOCK L — x=16, 10 steps, seed 1234. Static twin vs engine STRICT at budget
# 2.5 GiB, W=0 (every restore is synchronous, the stronger losslessness
# test).
#   L-P1  STEP 0 EVICTS: static step 0 is 2731.98 > 2560 budget. All 10 steps
#         evict; ~10 picks at step 0, ~45 thereafter (+/-2). No refusal raises.
#   L-P2  max |loss_engine - loss_static| <= 1e-3 over 10 steps, SIGN-CHANGING,
#         not monotone. A monotone drift is the pre-registered falsifier.
#   L-P3  Step-time cost +7% to +15% (~45 picks). Quote it with the relief it
#         bought.
#
# BLOCK U — L=64 whole-layer, UNCHUNKED (--chunked_ce 0), 3 steps.
#   U-P1  2488.45 / 3561.99 / 5709.07 at x=2/4/8. SLOPE 536.77 to < 0.01.
#
# BLOCK C — L=32, x=2, 10 steps, static ckpt: c=1024 vs unchunked.
#   C-P1  Peaks 1366.20 (c=1024) and 1816.31 (unchunked) (same derivation
#         as U-P1).
#   C-P2  Step-time ratio ~1.04.
#   C-P3  Loss agreement within 6e-4 at every step, sign-changing.
#
# BLOCK T — static L=32, c=256, 3 steps.
#   T-P1  LINE: x=16 2249.78 | 24 2924.34 | 32 3598.91 | 40 4273.47; slope
#         84.320299 to < 0.001; remainder 84.320299 - 1.5 x 32 = 36.3203,
#         the same as L=64's — depth-independence re-checked at world 4.
#   T-P2  BRACKET: x=168 fits (15066.47); x=172 (15403.75) sits above the
#         reference largest fit on this card and below its failure's
#         allocated+requested — open.

# W=2 and W=1 are run_l64_w12_4gpu.sh's.

# --- Environment ---
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Same allocator config as every fit.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# SHARED_X: run_l64_w12_4gpu.sh's measured W=2 largest fit. Read AT RUN TIME
# from that job's output (the newest job's completed B2 rungs), so this job
# can be queued before the other one finishes. An explicit SHARED_X wins;
# 328 (the W=2 prediction) is the last resort.
if [ -z "$SHARED_X" ]; then
    SHARED_X=$(python3 - <<'PY'
import glob, os, re
runs = {}
for d in glob.glob('profile_results/l64w12_4gpu/*-h768L64_B2_*_w2_*'):
    m = re.match(r'(\d+)-.*_s(\d+)_', os.path.basename(d))
    if m and os.path.isfile(os.path.join(d, 'steps_rank0.json')):
        runs.setdefault(int(m.group(1)), []).append(int(m.group(2)) // 1024)
print(max(runs[max(runs)]) if runs else '')
PY
)
    if [ -n "$SHARED_X" ]; then
        echo "SHARED_X read from run_l64_w12_4gpu.sh output: W=2 largest fit x=${SHARED_X}"
    else
        SHARED_X=328
        echo "SHARED_X: no completed W=2 rung found — FALLING BACK TO THE PREDICTION 328."
        echo "  The shared rung is then only shared if the W=2 job also landed on 328."
    fi
fi
MAX_RUNGS=${MAX_RUNGS:-4}      # per bracket search
BLOCKS=${BLOCKS:-"S E L U C T"}  # subset to re-run a block, e.g. BLOCKS="E"

echo "MASTER_ADDR  : $MASTER_ADDR"
echo "NODES        : $SLURM_JOB_NODELIST"
echo "WORLD_SIZE   : $SLURM_NTASKS"
echo "SHARED_X     : $SHARED_X"
echo "MAX_RUNGS    : $MAX_RUNGS"
echo "BLOCKS       : $BLOCKS"

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
PORT_BASE=${MASTER_PORT}
IDX=0
OUTROOT=profile_results/headline_4gpu

want() { case " $BLOCKS " in *" $1 "*) return 0 ;; esac; return 1; }

# One arm. Args: BLOCK NL X CHUNK MODE NSTEPS [WIN]. Returns 0 iff the arm
# completed (steps_rank0.json is written only by a completed run).
run_arm() {
    local BLOCK=$1 NL=$2 XVAL=$3 CHUNK=$4 MODE=$5 NSTEPS=$6 WIN=${7:-0}
    local SQ=$((XVAL * 1024)) BUDGET=0 MODE_FLAGS
    case "$MODE" in
        static)
            # Whole-layer checkpointing alone: the baseline every span is
            # quoted against. No engine, so no budget and no W.
            MODE_FLAGS="--layer_checkpoint" ;;
        engine_loose)
            # THE INSTRUMENT, LOOSE: the gate over-reports at the pick cap.
            BUDGET=14.9769
            MODE_FLAGS="--layer_checkpoint --placement_engine --engine_live --hbm_budget_gb ${BUDGET} --prefetch_window_layers ${WIN} --evict_layout_census" ;;
        engine_strict)
            # Block L only: below the cap, strict is safe, and it is what
            # makes a silent retention impossible in a fidelity test.
            BUDGET=2.5
            MODE_FLAGS="--layer_checkpoint --placement_engine --engine_live --hbm_budget_gb ${BUDGET} --prefetch_window_layers ${WIN} --evict_layout_census --strict_evict" ;;
        *) echo "UNKNOWN MODE ${MODE}"; return 1 ;;
    esac
    local TAG=h${HID}L${NL}_${BLOCK}_${MODE}_b${BUDGET}_w${WIN}_c${CHUNK}_s${SQ}_n${NSTEPS}_norminput_cacheoff
    OUTDIR=${OUTROOT}/${SLURM_JOB_ID}-${TAG}_b${BS}_v${VOC}_${DT}_mem_efficient_world4
    mkdir -p "$OUTDIR"
    IDX=$((IDX + 1))
    echo "════════════════════════════════════════════════════════════════════"
    echo "ARM ${IDX} [block ${BLOCK}]: ${MODE} L=${NL} x=${XVAL} W=${WIN} steps=${NSTEPS}"
    echo "  seq_len=${SQ}  chunked_ce=${CHUNK}  world=${SLURM_NTASKS}"
    echo "  hidden=${HID} inter=${INTER} heads=${NH}/${NKV} b=${BS} v=${VOC} ${DT}/${ATTN}"
    echo "  norm=${NORM}  use_cache=off  seed=${SEED}"
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
        || echo "ARM ${IDX} (block ${BLOCK}, ${MODE}, L=${NL}, x=${XVAL}, W=${WIN}) DID NOT COMPLETE — READ THE FAILURE CLASS. (0) NODE FAILURE / socket timeout / 'Task launch ... failed on node' is NOT a result: check the .err for slurm_receive_msgs first; it imitates every class below and the arm must be RE-RUN. (1) CUDA OOM IN BACKWARD in a bracket search is the designed stop: record allocated/reserved and the FAILING REQUEST SIZE, which must be 4x MiB ($((XVAL * 4)).00 here) — the replayed feed-forward activation; a different size is a different mechanism and the rung below is not a bracket. It does NOT bound capacity from above. (2) CUDA OOM IN FORWARD is a different class: far above the ceiling, the rung below is a LOOSE bracket and one intermediate rung is owed. (3) ANY failure on a LINE POINT (S/T x<=48, U, C) or a BELOW-CAP E rung is NOT a capacity result — those shapes sit thousands of MiB under capacity; the build moved or the world-4 derivation is badly wrong. (4) A GATE REFUSAL RAISING in block L means strict refused a shape at budget 2.5 — a gate finding, not a fidelity one; re-run L loose before reading L-P2. On any E arm it must not happen (loose); check strict_evict in the config block. (5) POOL DENIAL (cpu_dram_full) is silent: check pool.denials_total and evict_realization.denied_ops on EVERY completing engine arm; a denial at a size other than this x's boundary ($((XVAL * 1572864)) B) is a Gate-0 bundle-mismatch defect and voids the E block. (6) cudaHostAlloc failure at init: an implementation limit, never a finding; do NOT touch --pool_buffers_per_bucket. (7) NaN loss or a census class other than the single fp16 residual boundary: read evict_layout_census first. (8) A HANG or wall-clock kill at W=0: every restore is synchronous; check whether step 0 printed before assuming capacity."
    sleep 20   # let NCCL tear down and the rdzv port clear before the next arm
    [ -f "$OUTDIR/steps_rank0.json" ]
}

# Bracket search. Args: BLOCK NL START CHUNK MODE NSTEPS [WIN]. Climb +4 from
# a fit, descend -4 from a fail, stop when a fit/fail pair 4 x-units apart
# exists or after MAX_RUNGS. Sets FIT_X and FAIL_X.
search() {
    local BLOCK=$1 NL=$2 X=$3 CHUNK=$4 MODE=$5 NSTEPS=$6 WIN=${7:-0} N=0
    FIT_X=""; FAIL_X=""
    while [ $N -lt $MAX_RUNGS ]; do
        N=$((N + 1))
        if run_arm "$BLOCK" "$NL" "$X" "$CHUNK" "$MODE" "$NSTEPS" "$WIN"; then
            FIT_X=$X; [ -n "$FAIL_X" ] && break; X=$((X + 4))
        else
            FAIL_X=$X; [ -n "$FIT_X" ] && break; X=$((X - 4))
        fi
    done
    echo "SEARCH ${BLOCK} (${MODE}, L=${NL}, W=${WIN}): largest fit x=${FIT_X:-NONE}  smallest fail x=${FAIL_X:-NONE}  (${N} rungs)"
    if [ -z "$FIT_X" ] || [ -z "$FAIL_X" ]; then
        echo "  BRACKET NOT CLOSED within MAX_RUNGS=${MAX_RUNGS} — the derived intercept missed by more"
        echo "  than $((MAX_RUNGS * 4)) x-units. Read the line points' residuals before re-submitting."
    fi
}

S_FIT=""; S_FAIL=""; E_FIT=""; E_FAIL=""; T_FIT=""; T_FAIL=""

# ── S: static L=64 line + bracket (THE DENOMINATOR) ────────────────────────
if want S; then
    for X in 8 16 24 48; do run_arm S 64 $X 256 static 3; done
    search S 64 104 256 static 3
    S_FIT=$FIT_X; S_FAIL=$FAIL_X
fi

# ── E: W=0 ladder (THE NUMERATOR) ──────────────────────────────────────────
if want E; then
    # Below the cap. The first rung is the shape the baseline fails at.
    for X in ${S_FAIL:-108} 140 280; do run_arm E 64 $X 256 engine_loose 2 0; done
    # The shared rung: same x as the W=2 and W=1 rungs of the other job.
    run_arm E 64 $SHARED_X 256 engine_loose 2 0 \
        || echo "SHARED RUNG x=${SHARED_X} DID NOT COMPLETE at W=0. It is predicted ~1060 MiB under W=0's line at 356, so this is NOT a capacity result — read the failure class before trusting the bracket below."
    search E 64 356 256 engine_loose 2 0
    E_FIT=$FIT_X; E_FAIL=$FAIL_X
fi

# ── L: losslessness, 10 steps, static twin vs engine strict ────────────────
if want L; then
    run_arm L 64 16 256 static 10
    run_arm L 64 16 256 engine_strict 10 0
fi

# ── U: slope invariance (whole-layer, unchunked) ───────────────────────────
if want U; then
    for X in 2 4 8; do run_arm U 64 $X 0 static 3; done
fi

# ── C: chunked-CE cost pair, L=32, 10 steps ────────────────────────────────
if want C; then
    run_arm C 32 2 1024 static 10
    run_arm C 32 2 0 static 10
fi

# ── T: static L=32 line + bracket ──────────────────────────────────────────
if want T; then
    for X in 16 24 32 40; do run_arm T 32 $X 256 static 3; done
    search T 32 168 256 static 3
    T_FIT=$FIT_X; T_FAIL=$FAIL_X
fi

echo "RUN COMPLETE: ran ${IDX} arms."
echo "  S static L=64 : largest fit x=${S_FIT:-?}  smallest fail x=${S_FAIL:-?}   (predicted 104 / 108)"
echo "  E W=0    L=64 : largest fit x=${E_FIT:-?}  smallest fail x=${E_FAIL:-?}   (predicted 352-or-356 / 356-or-360)"
if [ -n "$S_FIT" ] && [ -n "$E_FIT" ]; then
    echo "  HEADLINE SPAN  : ${E_FIT}/${S_FIT} = $(python3 -c "print(f'{${E_FIT}/${S_FIT}:.3f}')")"
    echo "    quote with: W=0, whole-layer unit, chunked CE c=256, mem-efficient kernel, fp16, v=32000, b=1, WORLD 4"
fi
echo "  T static L=32 : largest fit x=${T_FIT:-?}  smallest fail x=${T_FAIL:-?}   (predicted 168 / 172-or-176)"
echo "Lines every arm is read against (world 4, derived 2026-09-29):"
echo "  S  1476.80 + 132.320279x    E(W=0, step1)  1878.90 + 37.820312x"
echo "  T   900.66 + 84.320299x     U  1414.92 + 536.77x"

# ── VALIDITY CHECKS, IN ORDER ──────────────────────────────────────────────
#   1. ARM IDENTITY on every arm: world_size == 4, fused_rmsnorm_replaced ==
#      2L+1 (129 / 65), config.use_cache == false, layer_checkpoint_wrapped ==
#      L, chunked_ce_tokens == the arm's chunk (NOT config.chunked_ce),
#      prefetch_window_layers == 0 on engine arms, strict_evict true ONLY on
#      block L's engine arm. seed / evict_layout_census: read off this script.
#   2. S-P1 AND U-P1 FIRST: the slopes. They are the world-invariance claim
#      the world-4 lines rest on. A slope miss outranks every bracket in the
#      job.
#   3. S-P1's intercept residual against 1476.80, compared with the W=2/W=1
#      job's residual: the same miss in both = one world-dependent term.
#   4. The three checks that replace --strict_evict on every loose E rung:
#      realized_fraction == 1.0, denied_ops == 0 and pool.denials_total == 0,
#      relief == picks x 1.5x MiB with zero residual.
#   5. E-P2 relief reconstruction; E-P3 picks/location/floor; E-P4 failure
#      class and request size.
#   6. THE SPAN: E_FIT / S_FIT, bracketed by the two fails, with W, levers,
#      kernel, vocabulary, chunk size AND WORLD SIZE named.
#   7. L-P2 losslessness; L-P3 and C-P2 step-time costs; E-P5 time share.
#   8. Record one row per arm in profile_results/runs_4gpu.csv.
