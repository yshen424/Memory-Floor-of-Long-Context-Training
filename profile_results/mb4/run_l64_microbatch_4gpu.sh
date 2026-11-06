#!/bin/bash
#SBATCH --job-name=fsdp_l64_mb4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1        # 1 process per node (1 GPU per node)
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/l64mb4_%j.log
#SBATCH --error=logs/l64mb4_%j.err

# ── JOB: THE MICRO-BATCHING BASELINE. L=64, LEVER 3, c=256, W=0, WORLD 4 ────
# THE WORKLOAD IS FIXED: two samples of sequence length s per optimizer
# step. Two ways to train it are compared, same tokens, same FLOPs:
#   ENGINE   one b=2 step, the engine relieving what does not fit
#   ACCUM    two b=1 micro-batches with gradient accumulation, static
#            whole-layer checkpointing (--batch_size 1 --grad_accum 2)
# The accumulated arm draws the b=2 arm's exact tokens and reproduces its
# loss and gradients (tests/test_grad_accum.py, 7/7, incl. a negative
# control), so any difference is execution strategy, not workload.
#
# WHAT THIS CAN AND CANNOT SHOW. On REACH accumulation wins by construction
# — it halves per-step activations — which is why the evaluation fixes b=1,
# where accumulation has nothing left to give and the engine extends the
# frontier: the two COMPOSE. The open question is TIME at a shape both can
# run: accumulation re-pays every layer's FSDP all-gather and reduce-scatter
# per micro-batch, and on this non-GDR fabric those cross the contended host
# link; the engine pays transfers only for the boundaries it picks. That is
# the one comparison migration might WIN on time. It can also lose. Both are
# results.
#
#  C (2 arms) THE CONTROL, s=49152 (x=48/sample, 96 per step). Static b=2
#             FITS here, so static b=2 vs ACCUM isolates what batching
#             itself is worth — fewer collective rounds, larger kernels —
#             with NO migration in either arm.
#  G (6 arms) THE COMPARISON at s=57344 / 73728 / 98304 (x=56/72/96 per
#             sample, 112/144/192 per step): static b=2 cannot fit any of
#             them, ACCUM fits all, the engine at b=2 picks ~7/25/41.
#  S (1 arm)  STATIC b=2 at s=57344 — the "why not just run b=2" row. It
#             must OOM; it runs LAST because it is the cheapest to lose.
#
#  All regime B (fp16 / sdpa_mem_efficient), fused-input, use_cache off,
#  v=32000, c=256, seed 1234, L=64, W=0, budget 14.9769, LOOSE, 4 steps
#  (steps 1+ are read; step 0 pays one-time allocation). Byte-for-byte
#  7744/7747's environment.
#
#  WALL CLOCK ~3 h (step times scaled off 7744/7747: ~0.022 x^2 s/step at
#  b=1, so ~101 / 138 / 228 / 405 s per b=2 step at x=48/56/72/96).

# ── THE PREDICTIONS. PRE-REGISTERED, NOT FITTED AFTERWARDS ─────────────────
# Inputs, world 4, L=64 lever 3 c=256 regime B, x = tokens/1024, MiB:
#   static whole-layer (7744)      1476.80 + 132.3201x     largest fit x=104
#   engine W=0 below the cap       static line − picks × 1.50x (7744, to 0.01)
#   gate over-report below cap     +0.727x (safe direction)
#   budget                         14.9769 GiB = 15336.3 MiB
#   sharded fp32 grad pile         ~431 MiB per rank (P = 452M params / 4 × 4 B)
#   H-B1: (b=2, s) == (b=1, 2s) to 0.014%
#
# MB-P1 SAME TOKENS, SAME LOSS. Every pair at one s reports the same step-0
#       loss: ACCUM's is the mean of its micro losses, identical in exact
#       arithmetic (test_grad_accum.py). Predict |Δloss0| < 1e-3 (fp16 kernel
#       batching may differ in the last bits). A larger gap voids the pair.
# MB-P2 ACCUM PEAKS = the static line at the PER-SAMPLE x, plus the sharded
#       gradient pile the first micro-batch leaves resident through the
#       second: predicted 7828.16 / 8886.73 / 11003.85 / 14179.53 + [384, 431]
#       at x=48/56/72/96. Every ACCUM arm FITS (top ~14610 < 15238.11).
#       A peak ON the bare line means FSDP does not hold the pile there —
#       a recording, not a defect.
# MB-P3 ENGINE b=2 FOLLOWS H-B1: it is the b=1 engine at x = 2s/1024.
#       picks (bracket, the gate's 0.727x over-report included):
#         x=112 -> 7 [5, 9]    peak 15120.7
#         x=144 -> 25 [23, 27] peak 15130.9   (7744's b=1 engine took 25 at 140)
#         x=192 -> 41 [39, 43] peak 15074.3
#       peak = static(x) − picks × 1.50x, to 0.01 MiB, as 7744 measured at b=1.
#       A pick count outside its bracket means the engine's model does not
#       carry to b=2. Read the census.
# MB-P4 C1 (static b=2, x=96) fits at 14179.53 (H-B1 on the static line).
#       S1 (static b=2, x=112) OOMs in step-1 backward, request 448.00 = 4x,
#       predicted peak 16296.66 against a 15238.11 largest fit.
# MB-P5 THE QUESTION — TIME, steps 1+, ENGINE vs ACCUM at each G shape.
#       Let D_C = t(ACCUM, s=48K) − t(static b=2, s=48K): what batching is
#       worth with no migration anywhere. PRIMARY: D_C > 0 (accumulation pays
#       one extra round of all-gathers/reduce-scatters per step on a contended
#       link), and at each G shape t(ACCUM) − t(ENGINE) ≈ D_C scaled by
#       compute, minus the engine's transfer cost (picks × 1.50x MiB each
#       way; ~0.1–0.4% of step). So the engine is predicted FASTER at x=112
#       and x=144, with the margin shrinking at x=192 as picks grow.
#       FALSIFIER: at any G shape ENGINE is slower than ACCUM by more than
#       0.5% while D_C > 0 — the transfers cost more than batching saves.
#       ALTERNATIVE: D_C ≈ 0 within noise (FSDP's prefetch hides the extra
#       collectives); then the comparison carries no time claim either way.

# --- Environment: BYTE-FOR-BYTE 7744/7747's --------------------------------
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=30000
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Attempt-stamped outputs: a requeue can neither overwrite nor impersonate a
# prior attempt.
ATTEMPT=${SLURM_RESTART_COUNT:-0}
if [ "$ATTEMPT" = "0" ]; then ASUF=""; else ASUF="r${ATTEMPT}"; fi

echo "MASTER_ADDR  : $MASTER_ADDR"
echo "NODES        : $SLURM_JOB_NODELIST"
echo "WORLD_SIZE   : $SLURM_NTASKS"
echo "ATTEMPT      : $ATTEMPT  (output suffix '${ASUF}')"

if [ "$SLURM_NTASKS" != "4" ]; then
    echo "WORLD SIZE IS ${SLURM_NTASKS}, NOT 4. Every reference line here is world 4;"
    echo "  intercepts do not carry across worlds. Refusing to run."
    exit 2
fi

mkdir -p logs profile_results/mb4

# --- Preflight: every node must have its RAM back ----------------------------
MIN_AVAIL_GIB=100
AVAIL=$(srun --ntasks-per-node=1 bash -c \
    'echo "$(hostname) $(awk "/MemAvailable/{printf \"%d\", \$2/1048576}" /proc/meminfo)"' \
    < /dev/null)
echo "MemAvailable per node (GiB):"
echo "$AVAIL" | sed 's/^/    /'
if echo "$AVAIL" | awk -v m=$MIN_AVAIL_GIB 'NF<2 || $2<m {bad=1} END {exit !bad}'; then
    echo "PREFLIGHT FAILED: a node has under ${MIN_AVAIL_GIB} GiB MemAvailable, or did not"
    echo "  answer. Ask the admins to clean it; do not run arms on it."
    exit 3
fi

# --- The arms: block:seq:mode:batch:accum:steps -------------------------------
# `for` over a word list, never `while read` from a pipe (srun drains
# stdin); field split via `set --`; `< /dev/null` on srun.
# ORDER: the control first (cheapest, and the comparison is unreadable without
# it), then G ascending with ACCUM before ENGINE at each shape, then S.
# No arm closes another: these are pairs, not a ladder.
ARMS="
C:49152:static:2:1:4
C:49152:static:1:2:4
G:57344:static:1:2:4
G:57344:engine:2:1:4
G:73728:static:1:2:4
G:73728:engine:2:1:4
G:98304:static:1:2:4
G:98304:engine:2:1:4
S:57344:static:2:1:4
"

NL=64; HID=768; INTER=2048; NH=12; NKV=4; VOC=32000; DT=fp16
ATTN=sdpa_mem_efficient; NORM=input; SEED=1234; CHUNK=256
BUDGET=14.9769; WIN=0
PORT_BASE=${MASTER_PORT}
IDX=0
NARM=$(echo $ARMS | wc -w)
echo "ARMS TO RUN  : $NARM"

for ARM in $ARMS; do
    OLDIFS=$IFS; IFS=:; set -- $ARM; IFS=$OLDIFS
    BLOCK=$1; SQ=$2; MODE=$3; BS=$4; ACC=$5; NSTEPS=$6
    XS=$((SQ / 1024)); XSTEP=$((SQ * BS * ACC / 1024))

    case "$MODE" in
        static)
            # Stock whole-layer checkpointing. With ACC=2 it is the
            # micro-batching arm; with BS=2 ACC=1 it is static b=2.
            MODE_FLAGS="--layer_checkpoint"
            ;;
        engine)
            # The 7744/7747 instrument, unchanged, at b=2. LOOSE: the gate
            # projects and books but never raises.
            MODE_FLAGS="--layer_checkpoint --placement_engine --engine_live --hbm_budget_gb ${BUDGET} --prefetch_window_layers ${WIN} --evict_layout_census"
            ;;
        *) echo "UNKNOWN MODE ${MODE} — skipping"; continue ;;
    esac

    TAG=h${HID}L${NL}_${BLOCK}_${MODE}_b${BS}a${ACC}_w${WIN}_c${CHUNK}_s${SQ}_n${NSTEPS}_norminput_cacheoff
    OUTDIR=profile_results/mb4/${SLURM_JOB_ID}${ASUF}-${TAG}_v${VOC}_${DT}_mem_efficient
    mkdir -p "$OUTDIR"
    IDX=$((IDX + 1))
    echo "════════════════════════════════════════════════════════════════════"
    echo "ARM ${IDX}/${NARM} [block ${BLOCK}] mode=${MODE} batch=${BS} grad_accum=${ACC}"
    echo "  seq_len=${SQ} (x=${XS} per sample, ${XSTEP} per optimizer step)  steps=${NSTEPS}"
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
            --batch_size ${BS} --grad_accum ${ACC} --steps ${NSTEPS} \
            --chunked_ce ${CHUNK} \
            ${MODE_FLAGS} \
            --fused_rmsnorm ${NORM} \
            --use_cache off \
            --seed ${SEED} \
            --disable_hbm_drift \
            --output_dir "$OUTDIR" \
            < /dev/null \
        || echo "ARM ${IDX} (block ${BLOCK}, mode ${MODE}, b=${BS}, accum=${ACC}, s=${SQ}) DID NOT COMPLETE — READ THE CLASS. (0) NODE FAILURE / 'Task launch failed' / socket timeout is NOT A RESULT: check the .err first; re-run the arm. (1) S1 (static b=2, s=57344) OOMing in step-1 backward on a 448.00 MiB request is THE DESIGNED RESULT (MB-P4), not a failure. (2) C1 (static b=2, s=49152) or ANY ACCUM ARM failing is NOT designed: ACCUM's top predicted peak is ~14610 against a 15238.11 largest fit, and C1 sits 1059 MiB under it. If an ACCUM arm OOMs, record the request size — a 4x(per-sample) request means the gradient pile is larger than the 431 MiB bracket; anything else is a different mechanism. (3) An ENGINE arm OOMing at b=2: read evict_layout_census (one class, the fp16 boundary, at 2 x s x 768 x 2 B), pool.denials_total, and the pick count against MB-P3's bracket before concluding anything about capacity. (4) A driver raise naming --grad_accum on an engine arm means the mode string is wrong — accumulation is refused with the engine by design. (5) A step-0 loss differing across a pair by more than 1e-3 voids that pair (MB-P1). (6) A HANG on an engine arm is the W=0 synchronous-restore class; check whether step 0 printed."
    sleep 20   # let NCCL tear down and the rdzv port clear
done

echo "RUN COMPLETE: ran ${IDX} of ${NARM} arms."
echo ""
echo "WHAT TO READ, IN ORDER:"
echo "  1. MB-P1: step-0 loss per pair (same s) — equal to < 1e-3, or the pair is void."
echo "  2. C: D_C = t(ACCUM) − t(static b=2) at s=49152, steps 1+. Sign and size first:"
echo "     it is what batching is worth with no migration anywhere."
echo "  3. G: t(ACCUM) − t(ENGINE) at each s, steps 1+, beside D_C, and the engine's"
echo "     pick count and transfer bytes (evict_realization) against MB-P3."
echo "  4. MB-P2 / MB-P3 peaks against their predicted lines."
echo "  5. S1 must OOM on 448.00 MiB."
echo "AFTER THE JOB: append one row per arm to profile_results/runs_4gpu.csv, never overwrite."
