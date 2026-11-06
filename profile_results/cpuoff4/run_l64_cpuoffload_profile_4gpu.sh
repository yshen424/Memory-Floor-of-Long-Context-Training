#!/bin/bash
#SBATCH --job-name=fsdp_l64_cpuoff4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1        # 1 process per node (1 GPU per node)
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/cpuoff4_%j.log
#SBATCH --error=logs/cpuoff4_%j.err

# ── JOB: CPU-OFFLOAD BASELINE AND ATTENTION-REPLAY PROFILE, WORLD 4 (V100) ──
# Two questions, one job, 7744/7747's environment byte for byte.
#
# (1) DOES "OFFLOAD EVERYTHING" OUT-REACH THE ENGINE? Stock activation offload
#     ties the engine to the byte (7747, x=356). This job adds FSDP
#     CPUOffload(offload_params=True): fp32 master, AdamW state and gradient
#     shards leave HBM, the optimizer steps on CPU. That removes a FIXED pot
#     (exactly master + AdamW) from the intercept and touches no slope — so
#     it should buy ~+35 x for ANY mechanism, at a per-step cost (~1 s) that
#     vanishes against ~3000 s frontier steps. If true, stock
#     offload+CPUOffload beats engine-alone by ~10% on reach; block E tests
#     whether the engine composes with CPUOffload.
# (2) HOW MUCH OF A STEP IS THE ATTENTION REPLAY? Under the whole-layer unit
#     attention's forward runs twice per step. Saving q/k/v/out/lse to host
#     instead of replaying attention trades linear bytes for quadratic FLOPs;
#     this measures the FLOPs side.
#
#  M (4 arms) MATCHED SHAPE x=100, 4 steps: offload (IDENTITY CONTROL vs
#             7747), offload+cpu, force-evict+cpu (COMPOSITION SMOKE — the
#             engine under CPUOffload), static+cpu. Reach-free shape: this is
#             the time row.
#  P (3 arms) PROFILE, 3 steps, --torch_profile, stock offload at x=48,
#             100, 200. One active profiled step (step 2). Analyzer runs
#             in-job: tools/attn_recompute_share.py -> attn_share.json.
#  S (3 arms) STATIC+CPU LADDER x=112, 116, 120 — the fair denominator
#             once both sides offload their intercept.
#  O (4 arms) OFFLOAD+CPU LADDER x=200 (slope), 388, 392, 396.
#  E (3 arms) ENGINE(loose)+CPU LADDER x=388, 392, 396 — does the engine
#             compose with CPUOffload to the same frontier?
#
#  All regime B (fp16 / sdpa_mem_efficient), fused-input, use_cache off,
#  b=1, v=32000, c=256 (levers 1-3 on), WORLD 4, seed 1234, L=64, W=0,
#  budget 14.9769, LOOSE. Wall clock ~15-16 h (scaled off 7744/7747 step
#  times).

# ── HOST MEMORY ────────────────────────────────────────────────────────────
# Stock offload pins 96x MiB per rank outside any pool: 37.1 GiB at x=396.
# CPUOffload adds the sharded fp32 params, grads and AdamW state on the host:
# 113M params/rank x 16 B ~ 1.7 GiB. Engine arms (M3, E) add the pinned pool
# (~36-40 GiB at the frontier, guarded at 60% of MemAvailable). All well inside
# 112 GB; the preflight still requires >= 100 GiB MemAvailable on every node.

# ── THE PREDICTIONS. PRE-REGISTERED, NOT FITTED AFTERWARDS ─────────────────
# x = tokens/1024, MiB. Reference lines MEASURED at world 4 (7744/7747):
#   static L=64                 1476.80 + 132.3201x   fit x=104 (15238.11),
#                               fail x=108 (pred 15767.4, req 432.00)
#   engine W=0 / stock offload  1866.89 + 37.820312x  fit x=356 (15330.92),
#                               fail x=360 (pred 15482.20, req 1440.00)
#
# THE REMOVED POT, DERIVED BEFORE THE RUN. P = sharded param bytes at 1 B/param
# = 430.97 MiB. CPUOffload removes the fp32 master (4 B/param/world) and
# AdamW (8 B): 12 P / 4 = 3P = **1292.91 MiB at world 4**. The ONE open term
# is the step-1 gradient pile (3 fp32 shards = 18.005 MiB): CPUOffload copies
# each reduced grad shard to host, so the pile may leave too. Two-sided,
# decided by M2:
#   PRIMARY      pile leaves:  offload+cpu line  555.98 + 37.820312x
#   ALTERNATIVE  pile stays :  offload+cpu line  573.98 + 37.820312x
#   static+cpu (peak at post_bwd_layer_63, no pile)  183.89 + 132.3201x
#
# M-P1  M1 offload x=100 = 5648.92 to 0.01 (7747: 5648.9219). IDENTITY
#       CONTROL — a miss voids every prediction below.
# M-P2  M2 offload+cpu x=100 = 4338.01 (primary) or 4356.01 (alternative),
#       to 0.05. Any other value is a miss in the removed-pot derivation —
#       read peak_location and the allocator delta before using block O.
# M-P3  M3 force-evict+cpu = M2 to 0.01, 64/64 evicted, realized 1.0, denials
#       0, census one class (the fp16 boundary). This is the 7747 M2=M3
#       identity under CPUOffload. A failure here is a COMPOSITION DEFECT
#       (engine x CPUOffload), never a capacity result, and block E is void.
# M-P4  M4 static+cpu x=100 = 13415.92 to 1.
# M-P5  Step-0 loss: M1, M3 bit-identical to 7747's world-4 value; M2/M3/M4
#       (CPU optimizer) equal M1 at STEP 0 (the step precedes any optimizer
#       update); later steps may differ in the last bits (CPU vs GPU AdamW).
# M-T   TIME at x=100 (steps 1+): M2 - M1 in [0, +1.5%] of M1 (CPUOffload
#       expected to cost a fixed ~1 s on a ~220 s step). > +3% falsifies
#       "CPUOffload is free at long context".
#
# P-P1  The analyzer's count self-check passes: 128 attention-forward and 64
#       attention-backward launches in the one active step (forward + replay
#       under the whole-layer unit). A failure prints the real kernel names;
#       the arm still produced a trace, so it is re-analysable offline.
# P-P2  ATTENTION REPLAY SHARE of the step's GPU span (attn_replay_share_of_
#       span): x=100 in [10%, 25%]; x=200 in [12%, 25%]; non-decreasing over
#       x=48 -> 100 -> 200 (attention is quadratic, everything else linear).
#       Back-of-envelope: ~1e15 FLOP of attention forward per pass at x=100,
#       ~30 TFLOP/s -> ~34 s of a ~218 s step = ~16%.
# P-P3  attn_bwd_over_one_fwd in [2.0, 3.5].
# P-DECISION  If the replay share at x=200 is < 8%, the save-instead-of-
#       recompute idea is DROPPED (after host-DRAM limits it could return
#       < ~5%). If >= 12%, it goes to a design proposal.
#
# S-P1  static+cpu: x=112 = 15003.74 FITS; x=116 = 15533.02 is the DECIDER
#       (between static's largest fit 15238.11 and its failing 15767.4);
#       x=120 = 16062.30, predicted closer, request 480.00 (464.00 at 116).
#       Largest fit 112 or 116.
# O-P1  offload+cpu rides its line (intercept from M2) to 0.01: x=200 =
#       8120.04 | 8138.04.
# O-P2  x=388 = 15230.26 | 15248.26 FITS. x=392 = 15381.54 | 15399.54 is the
#       DECIDER (inside offload's own bracket [15330.92, 15482.20)). x=396 =
#       15532.82 | 15550.82, predicted closer, step-1 backward, request
#       1584.00 (1568.00 if 392 closes). Largest fit 388 or 392 -> span vs
#       static (104) 3.731 or 3.769: OFFLOAD+CPU OUT-REACHES ENGINE-ALONE
#       (356) by 32-36 x. This is the hostile result and it is the PRIMARY.
# O-P3  The FAIR span — intercept offloaded on both sides — is O_fit/S_fit in
#       [388/116, 392/112] = [3.34, 3.50]: the decomposition says CPUOffload
#       moves both intercepts and no slope, so the span barely moves.
# E-P1  engine+cpu lands on the offload+cpu line to 0.01 on every completing
#       rung and closes at the same rung (picks 63, realized 1.0, census one
#       class). The gate over-projects by an extra ~1293 MiB (its byte model
#       books params on HBM) — LOOSE, so it books infeasible steps and does
#       not raise. A closure one rung below O is the allocation-pattern
#       class, not a defect.

# --- Environment: 7744/7747's, byte for byte ---------------------------------
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29900
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STOP_ON_FAIL=${STOP_ON_FAIL:-1}

ATTEMPT=${SLURM_RESTART_COUNT:-0}
if [ "$ATTEMPT" = "0" ]; then ASUF=""; else ASUF="r${ATTEMPT}"; fi

echo "MASTER_ADDR  : $MASTER_ADDR"
echo "NODES        : $SLURM_JOB_NODELIST"
echo "WORLD_SIZE   : $SLURM_NTASKS"
echo "STOP_ON_FAIL : $STOP_ON_FAIL"
echo "ATTEMPT      : $ATTEMPT  (output suffix '${ASUF}')"

if [ "$SLURM_NTASKS" != "4" ]; then
    echo "WORLD SIZE IS ${SLURM_NTASKS}, NOT 4. Every reference line here is world 4;"
    echo "  intercepts, brackets and the removed pot (3P) do not carry. Refusing to run."
    exit 2
fi

OUTROOT=profile_results/cpuoff4
mkdir -p logs "$OUTROOT"

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

# --- The arms: block:layers:seq:chunk:mode:budget:window:steps ----------------
# `for` over a word list, never `while read` (srun drains stdin).
# ORDER: M (control + composition smoke) -> P (cheap) -> S -> O -> E.
# Within O the slope rung runs first. Ladders close independently.
ARMS="
M:64:102400:256:offload:14.9769:0:4
M:64:102400:256:offload_cpu:14.9769:0:4
M:64:102400:256:engine_force_evict_cpu:14.9769:0:4
M:64:102400:256:static_cpu:14.9769:0:4
P:64:49152:256:offload_profile:14.9769:0:3
P:64:102400:256:offload_profile:14.9769:0:3
P:64:204800:256:offload_profile:14.9769:0:3
S:64:114688:256:static_cpu:14.9769:0:2
S:64:118784:256:static_cpu:14.9769:0:2
S:64:122880:256:static_cpu:14.9769:0:2
O:64:204800:256:offload_cpu:14.9769:0:2
O:64:397312:256:offload_cpu:14.9769:0:2
O:64:401408:256:offload_cpu:14.9769:0:2
O:64:405504:256:offload_cpu:14.9769:0:2
E:64:397312:256:engine_loose_cpu:14.9769:0:2
E:64:401408:256:engine_loose_cpu:14.9769:0:2
E:64:405504:256:engine_loose_cpu:14.9769:0:2
"

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
NARM=$(echo $ARMS | wc -w)

run_arm() {
    TOK=$((SQ * BS))
    XVAL=$((TOK / 1024))
    ENGINE="--layer_checkpoint --placement_engine --engine_live --hbm_budget_gb ${BUDGET} --prefetch_window_layers ${WIN} --evict_layout_census"

    case "$MODE" in
        offload)
            # 7747's stock baseline, unchanged: offload_wrapper OUTSIDE the
            # checkpoint wrapper, which moves exactly the boundary.
            MODE_FLAGS="--layer_checkpoint --activation_offload" ;;
        offload_profile)
            # Same arm, traced. Kernel durations are read from the trace, never
            # this arm's wall clock (the profiler's CPU bookkeeping perturbs it).
            MODE_FLAGS="--layer_checkpoint --activation_offload --torch_profile" ;;
        offload_cpu)
            # THE HOSTILE BASELINE: stock activation offload + FSDP CPUOffload.
            MODE_FLAGS="--layer_checkpoint --activation_offload --cpu_offload" ;;
        static_cpu)
            MODE_FLAGS="--layer_checkpoint --cpu_offload" ;;
        engine_force_evict_cpu)
            MODE_FLAGS="${ENGINE} --force_action evict --cpu_offload" ;;
        engine_loose_cpu)
            MODE_FLAGS="${ENGINE} --cpu_offload" ;;
        *)
            echo "UNKNOWN MODE ${MODE} — skipping"; return 2 ;;
    esac

    case "$BLOCK" in
        M) LABEL="MATCHED SHAPE x=100 — identity control, CPUOffload time cost, composition smoke" ;;
        P) LABEL="PROFILE — attention replay share of the step" ;;
        S) LABEL="STATIC+CPU LADDER — the fair denominator; x=116 is the decider" ;;
        O) LABEL="OFFLOAD+CPU LADDER — the hostile baseline; x=392 is the decider" ;;
        E) LABEL="ENGINE+CPU LADDER — does the engine compose with CPUOffload to O's frontier?" ;;
        *) echo "UNKNOWN BLOCK ${BLOCK} — skipping"; return 2 ;;
    esac

    TAG=h${HID}L${NL}_${BLOCK}_${MODE}_b${BUDGET}_w${WIN}_c${CHUNK}_s${SQ}_n${NSTEPS}_norminput_cacheoff
    OUTDIR=${OUTROOT}/${SLURM_JOB_ID}${ASUF}-${TAG}_b${BS}_v${VOC}_${DT}_mem_efficient_world4
    mkdir -p "$OUTDIR"
    IDX=$((IDX + 1))
    echo "════════════════════════════════════════════════════════════════════"
    echo "ARM ${IDX}/${NARM} [block ${BLOCK}]: ${LABEL}"
    echo "  layers=${NL}  seq_len=${SQ} (${TOK} tokens, x=${XVAL})  chunked_ce=${CHUNK}"
    echo "  hidden=${HID} inter=${INTER} heads=${NH}/${NKV} b=${BS} v=${VOC} ${DT}/${ATTN}"
    echo "  norm=${NORM}  use_cache=off  steps=${NSTEPS}  seed=${SEED}  W=${WIN}  world=${SLURM_NTASKS}"
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
        || echo "ARM ${IDX} (block ${BLOCK}, mode ${MODE}, x=${XVAL}) DID NOT COMPLETE — READ THE FAILURE CLASS, IT IS THE FINDING, AND THE CLASSES MUST NEVER BE SUMMED. (0) A NODE FAILURE, SOCKET TIMEOUT OR 'Task launch ... failed on node' IS NOT A RESULT — check the .err for slurm_receive_msgs first; the arm must be RE-RUN. (1) A FAILURE IN M1 (stock offload x=100, which 7747 completed at 5648.92) means the build or the nodes moved: every prediction is void. (2) A FAILURE IN M2/M4 AT x=100 IS NOT A CAPACITY RESULT — x=100 sits >9 GiB under capacity with CPUOffload on; it is a CPUOffload x (fp16 mixed precision, chunked CE, fused norm, checkpoint, offload_wrapper) incompatibility: record the traceback, block O (M2) or block S (M4) is void. (3) A FAILURE IN M3 IS A COMPOSITION DEFECT (engine x CPUOffload): read evict_layout_census and the traceback; block E is void, block O stands. (4) A CUDA OOM IN STEP-1 BACKWARD IN S, O OR E IS THE DESIGNED STOP — the failing request must be 4x MiB, the replayed feed-forward activation (464.00/480.00 in S; 1568.00/1584.00 in O and E). A DIFFERENT SIZE IS A DIFFERENT MECHANISM and the rung below is not a bracket; this OOM does NOT bound capacity from above. (5) A STEP-2 OOM after a clean step 1 is the optimizer-co-residency class — on CPUOffload arms the optimizer state is on the host, so a step-2 OOM there is NEW and must be named, not folded into (4). (6) A cudaHostAlloc / pinned-memory failure is a host-footprint limit of the arm (offload pins 96x MiB/rank, ~37 GiB at x=396, plus ~1.7 GiB of CPUOffload state): a table row, not a capacity result. (7) A '[PinnedPool] refusing to pin' RuntimeError on M3/E is the host-memory guard: a pool-sizing defect, never capacity. (8) A STEP-0 LOSS DIFFERENT FROM M1'S voids that arm on fidelity. (9) A P-ARM FAILURE that is not class (0) most likely means the profiler's own device memory tipped it (CUPTI buffers): x=200 is far under capacity, so record it and re-analyse nothing. Continuing per STOP_ON_FAIL."

    # The profile analysis runs in-job so the verdict is in the log. Its exit
    # code is the count self-check; a failure prints the real kernel names.
    if [ "$BLOCK" = "P" ]; then
        TRACE=$(ls -t "$OUTDIR"/torch_trace_rank0/*.pt.trace.json* 2>/dev/null | head -n 1)
        if [ -n "$TRACE" ]; then
            python3 tools/attn_recompute_share.py "$TRACE" --layers ${NL} --steps 1 \
                --json "$OUTDIR/attn_share.json" \
                || echo "P x=${XVAL}: ANALYZER SELF-CHECK FAILED — the trace is kept; read the kernel names above."
        else
            echo "P x=${XVAL}: NO TRACE WRITTEN — see failure class (9)."
        fi
    fi
}

CLOSED=""

for ARM in $ARMS; do
    OLDIFS=$IFS; IFS=:; set -- $ARM; IFS=$OLDIFS
    BLOCK=$1; NL=$2; SQ=$3; CHUNK=$4; MODE=$5; BUDGET=$6; WIN=$7; NSTEPS=$8

    case " $CLOSED " in
        *" $BLOCK "*)
            echo "SKIP block ${BLOCK} rung seq_len=${SQ}: the ladder already closed below it."
            continue ;;
    esac

    run_arm

    # The predicate is the JSON, not srun's exit code. M and P are not ladders.
    if [ ! -f "$OUTDIR/steps_rank0.json" ]; then
        if [ "$STOP_ON_FAIL" = "1" ] && [ "$BLOCK" != "M" ] && [ "$BLOCK" != "P" ]; then
            echo "LADDER ${BLOCK} CLOSED at x=${XVAL}. The rung BELOW is this block's largest"
            echo "  measured fit; the pair is its own bracket and belongs to no other block."
            echo "  CHECK FAILURE CLASS (0) FIRST: a node failure closes nothing."
            CLOSED="$CLOSED $BLOCK"
        fi
        # A composition failure at x=100 voids block E before it costs ~5 h.
        if [ "$BLOCK" = "M" ] && [ "$MODE" = "engine_force_evict_cpu" ]; then
            echo "M3 (engine x CPUOffload) DID NOT COMPLETE: block E is VOID and will be skipped."
            CLOSED="$CLOSED E"
        fi
    fi
    sleep 20   # let NCCL tear down and the rdzv port clear before the next arm
done

echo "RUN COMPLETE: ran ${IDX} of ${NARM} arms."
echo ""
echo "LINES EVERY BLOCK IS READ AGAINST (L=64, c=256, regime B, fused-input, cache off, b=1, v=32000, WORLD 4):"
echo "  static L=64 (7744)            : 1476.80 + 132.3201x   fit x=104"
echo "  engine W=0 = stock offload    : 1866.89 + 37.820312x  fit x=356  (span 3.423)"
echo "  removed by CPUOffload (3P)    : 1292.91 MiB, + 18.005 if the grad pile leaves"
echo "  offload+cpu (pred.)           : 555.98 | 573.98 + 37.820312x"
echo "  static+cpu  (pred.)           : 183.89 + 132.3201x"
echo ""
echo "WHAT TO READ: M2 first (decides the intercept), then O's largest fit — if it is"
echo "  388 or 392, stock offload+CPUOffload out-reaches engine-alone and the engine's"
echo "  answer is E. Then P's attn_share.json files (attn_replay_share_of_span)."
echo ""
echo "AFTER THE JOB: append one row per arm to profile_results/runs_4gpu.csv, never"
echo "overwrite; move this script to ${OUTROOT}/."
