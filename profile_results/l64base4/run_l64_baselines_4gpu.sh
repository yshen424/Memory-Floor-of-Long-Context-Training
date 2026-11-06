#!/bin/bash
#SBATCH --job-name=fsdp_l64_base4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1        # 1 process per node (1 GPU per node)
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/l64base4_%j.log
#SBATCH --error=logs/l64base4_%j.err

# ── JOB: L=64 BASELINES AT WORLD 4 (V100) ───────────────────────────────────
# The four-policy matched-shape table and three baseline ladders, all at
# world 4, in 7744's environment.
#
#  M (4 arms) MATCHED SHAPE, x=100, 4 steps. static / stock-offload /
#             engine-force-evict / engine-policy. The time, overhead,
#             bandwidth and host-cost row at ONE shape on ONE hardware.
#             M1 doubles as the IDENTITY CONTROL against 7744's static
#             line; a miss voids the job and the loop stops on its own.
#  V (6 arms) VANILLA LADDER, x=2..9. No checkpointing at all.
#  F (1 arm)  FORCE-EVICT AT THE ENGINE'S FAILING RUNG, x=360: is the
#             engine's closure pattern-dependent?
#  O (4 arms) OFFLOAD LADDER, x=200, 356, 360, 364. Tests whether stock
#             offloading rides the engine's own line and brackets
#             offload's own ceiling.
#  P (3 arms) PER-MODULE LADDER, x=32..34. LAST ON PURPOSE (host memory,
#             below). Runs ~20 minutes.
#
#  All regime B (fp16 / sdpa_mem_efficient), fused-input, use_cache off,
#  b=1, v=32000, c=256, WORLD 4, seed 1234, L=64, W=0, budget 14.9769,
#  loose. Identical to 7744's arms except the policy under test.
#  Wall clock ~8-10 h (scaled off 7744's measured step times).

# ── HOST MEMORY ─────────────────────────────────────────────────────────────
# Pinned pages are unreclaimable; a pool sized for layers that never stage
# can starve a 112 GB node. Three guards:
#   (a) The pool is funded only for layers that can reach it. Forced-recompute
#       arms (block P) pin ~5.0 GiB instead of ~102-108. Whole-layer engine
#       arms pin ~10-37 GiB.
#   (b) The pool REFUSES to pin more than 60% of the node's MemAvailable per
#       rank and raises BEFORE allocating — failure class (13), a clean arm
#       failure instead of a node hang.
#   (c) The preflight below requires >= 100 GiB MemAvailable on every node
#       before any arm runs.
# Stock offload (block O) pins 96x MiB per rank outside the pool: 18.8 GiB at
# x=200, 33.4 at x=356, 34.1 at x=364 — all well inside 112 GB. P runs LAST
# so that even a surprise cannot cost M, V, F or O.

# ── THE PREDICTIONS. PRE-REGISTERED, NOT FITTED AFTERWARDS ─────────────────
# Reference lines, MEASURED at world 4 (7743/7744) except vanilla and
# per-module, derived as noted. x = tokens/1024, MiB.
#
#   static L=64 (7744)          1476.80 + 132.3201x    largest fit x=104 at
#                               15238.11, x=108 fails (request 432.00)
#   engine W=0 step1 (7744)     1866.89 + 37.820312x   largest fit x=356 at
#                               15330.92, x=360 fails (request 1440.00)
#   engine W=0 bracket          [15330.92, ~15438)     ENGINE'S OWN pattern
#   vanilla                     1555.79 + 1800.33x
#   per-module x=32             14983.80
#   one boundary                1.5000x MiB per layer = 2 B x hidden x tokens
#
# M-P1  M1 static x=100 lands on 1476.80 + 13232.01 = 14708.81 MiB to 1 MiB,
#       peak at post_bwd_layer_63, 4 steps. THE IDENTITY CONTROL: a miss
#       means the build moved and every prediction here is void.
# M-P2  M2 stock offload and M3 force-evict BOTH land on the engine's step-1
#       line: 5648.92 MiB at x=100, equal to each other to ~0.01 MiB, peak at
#       post_bwd_layer_1.
# M-P3  M3 evicts 64 of 64 (not 63): under force the refused candidate IS
#       staged and relieves nothing at the relocated peak, so the peak does
#       not move. realized_fraction 1.0, denials 0.
# M-P4  M4 engine loose at x=100 authorizes ZERO evictions — the static peak
#       14708.81 sits under the 15336.35 MiB budget, the gate finds no breach
#       — and is bit-identical to M1 on peak, within noise on step time.
#       M4 vs M2 is the ADAPTIVITY row: same seam, policy on vs policy off.
# M-P5  All four step-0 losses are bit-identical to M1's.
# V-P1  Vanilla rungs land on 1555.79 + 1800.33x: 5156.5 / 8757.1 / 12357.8 /
#       14158.1 at x=2/4/6/7. x=8 predicts 15958.4 (+~12 outlier class) and
#       is the DECIDER — 417 MiB above the largest peak any world-4 pattern
#       has achieved, so the primary call is FAIL IN FORWARD with largest fit
#       x=7; if it fits, world-4 vanilla capacity >= 15958 and x=9 (17758.8)
#       is the guaranteed closer. Fail class is FORWARD OOM (class 4) —
#       vanilla has no replay; reading a backward class into V is a category
#       error.
# F-P1  TWO-SIDED. PRIMARY: x=360 COMPLETES at 15482.20 MiB — exactly the
#       engine's own floor at x=360 — with 64 of 64 evicted, realized_fraction
#       1.0, zero denials. That makes the engine's x=360 OOM pattern-dependent
#       at world 4, and the engine has a rung in hand. ALTERNATIVE: OOM in
#       step-1 backward, request exactly 1440.00 MiB, which makes the world-4
#       cap REAL at this shape. Either is a result; 15482.20 sits 44 MiB above
#       the engine-pattern bracket top (~15438), so this arm is genuinely
#       undetermined — which is why it runs.
# O-P1  THE OFFLOAD LINE IS THE ENGINE'S OWN LINE: 1866.89 + 37.820312x.
#       x=200 -> 9430.95 and x=356 -> 15330.92, each to ~0.01 MiB. A slope of
#       132.32 means the wrapping order inverted (interior offloaded) and
#       voids blocks O and M2. 36.32 would mean offload beats the engine by
#       one boundary — a result, not a defect.
# O-P2  THE LADDER: x=356 FITS (the engine's largest fit, by identity),
#       x=360 is the DECIDER at 15482.20 (the engine OOM'd here; F's result
#       is the sibling datum), x=364 (15633.48) is reached only if 360 fits
#       and is the predicted closer. Block O brackets ITSELF — the engine's
#       bracket is a guide only, because capacity is allocation-pattern-
#       dependent (stock save_on_cpu pins fresh buffers, no pool).
# O-P3  Offload step time at the matched rungs is within ~2% of the engine's
#       at the same shape — the time axis does not separate the mechanisms;
#       adaptivity (M4 vs M2) is what does.
# P-P1  x=32 lands on 14983.80 MiB. The slope is UNMEASURED; using a derived
#       slope ~424 only as a guide, x=33 projects ~15408, AT the bracket edge:
#       the decider. x=34 (~15832) is reached only if x=33 fits and is the
#       predicted closer. Pool on these arms is ~5.0 GiB (guard (a)); a
#       '[PinnedPool] refusing to pin' raise is class (13), a sizing defect
#       and never a capacity result.
#
# POST-HOC CHECKS THAT REPLACE --strict_evict, on every ENGINE arm (M3, M4, F):
#     evict_realization.realized_fraction == 1.0
#     evict_realization.denied_ops == 0  and  pool.denials_total == 0
# Block O arms are stock PyTorch with no telemetry; their check is the slope.
# Loose: the gate over-reports at the pick cap (x=356: +969 MiB) and a strict
# arm up there measures the gate's arithmetic, never the card.

# --- Environment: BYTE-FOR-BYTE 7744's, and not one variable more ----------
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29900
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STOP_ON_FAIL=${STOP_ON_FAIL:-1}

# Attempt suffix: a Slurm requeue reuses the job id, and the completion
# predicate is steps_rank0.json — stamping the attempt into the path keeps the
# predicate attempt-local.
ATTEMPT=${SLURM_RESTART_COUNT:-0}
if [ "$ATTEMPT" = "0" ]; then ASUF=""; else ASUF="r${ATTEMPT}"; fi

echo "MASTER_ADDR  : $MASTER_ADDR"
echo "NODES        : $SLURM_JOB_NODELIST"
echo "WORLD_SIZE   : $SLURM_NTASKS"
echo "STOP_ON_FAIL : $STOP_ON_FAIL"
echo "ATTEMPT      : $ATTEMPT  (output suffix '${ASUF}')"

if [ "$SLURM_NTASKS" != "4" ]; then
    echo "WORLD SIZE IS ${SLURM_NTASKS}, NOT 4. Every reference line in this job is"
    echo "  world 4, and FSDP shards parameters across ranks, so intercepts,"
    echo "  brackets and spans do not carry. Refusing to run."
    exit 2
fi

mkdir -p logs profile_results/l64base4

# --- Preflight: every node must have its RAM back ----------------------------
# A node still holding a hung process's pinned pages accepts an arm and then
# dies in it, which reads as a failure of the arm.
MIN_AVAIL_GIB=100
AVAIL=$(srun --ntasks-per-node=1 bash -c \
    'echo "$(hostname) $(awk "/MemAvailable/{printf \"%d\", \$2/1048576}" /proc/meminfo)"' \
    < /dev/null)
echo "MemAvailable per node (GiB):"
echo "$AVAIL" | sed 's/^/    /'
if echo "$AVAIL" | awk -v m=$MIN_AVAIL_GIB 'NF<2 || $2<m {bad=1} END {exit !bad}'; then
    echo "PREFLIGHT FAILED: a node has under ${MIN_AVAIL_GIB} GiB MemAvailable, or did not"
    echo "  answer. It is probably still holding a hung process's pinned memory."
    echo "  Ask the admins to reboot or clean it; do not run arms on it."
    exit 3
fi

# --- The arms: block:layers:seq:chunk:mode:budget:window:steps ----------------
# `for` over a word list, never `while read` from a pipe (srun drains stdin);
# field split via `set --`; `< /dev/null` on srun as second defence.
#
# ORDER: M FIRST (the table and the identity control that validates
# everything else), then V (cheap row), F, O, and P LAST — the per-module arm
# class cannot cost the others even if the pool surprises on hardware.
#
# WITHIN BLOCK O THE SLOPE RUNG RUNS FIRST: if x=356 fails, x=200 is what
# keeps the block readable (a second point and therefore a slope).
#
# EACH LADDER BLOCK CLOSES INDEPENDENTLY — a failed rung in O does not stop P.
ARMS="
M:64:102400:256:static:14.9769:0:4
M:64:102400:256:offload:14.9769:0:4
M:64:102400:256:engine_force_evict:14.9769:0:4
M:64:102400:256:engine_loose:14.9769:0:4
V:64:2048:256:vanilla:14.9769:0:2
V:64:4096:256:vanilla:14.9769:0:2
V:64:6144:256:vanilla:14.9769:0:2
V:64:7168:256:vanilla:14.9769:0:2
V:64:8192:256:vanilla:14.9769:0:2
V:64:9216:256:vanilla:14.9769:0:2
F:64:368640:256:engine_force_evict:14.9769:0:2
O:64:204800:256:offload:14.9769:0:2
O:64:364544:256:offload:14.9769:0:2
O:64:368640:256:offload:14.9769:0:2
O:64:372736:256:offload:14.9769:0:2
P:64:32768:256:permodule:14.9769:0:2
P:64:33792:256:permodule:14.9769:0:2
P:64:34816:256:permodule:14.9769:0:2
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

    case "$MODE" in
        static)
            # THE PAPER'S DENOMINATOR. Stock PyTorch non-reentrant whole-layer
            # checkpointing, nothing attached. Must reproduce 7744's line.
            MODE_FLAGS="--layer_checkpoint"
            ;;
        vanilla)
            # NO checkpointing and NO engine. Levers 1 and 3 stay in force —
            # every arm carries them or no span is comparable — and lever 2
            # is exactly what this arm removes.
            MODE_FLAGS=""
            ;;
        permodule)
            # THE CONTRAST UNIT. --layer_checkpoint is ABSENT on purpose: with
            # it the driver raises (forcing recompute under the whole-layer
            # unit is a no-op); without it, forcing recompute on the attention
            # and mlp bundles IS per-module checkpointing.
            MODE_FLAGS="--placement_engine --engine_live --hbm_budget_gb ${BUDGET} --prefetch_window_layers ${WIN} --evict_layout_census --force_action recompute"
            ;;
        offload)
            # THE STOCK BASELINE. offload_wrapper OUTSIDE the checkpoint
            # wrapper — the only order that offloads the BOUNDARY, not the
            # interior. CPU-verified with a negative control on the inverted
            # order. No criterion, no gate, no pinned pool, no telemetry.
            MODE_FLAGS="--layer_checkpoint --activation_offload"
            ;;
        engine_force_evict)
            # OUR MECHANISM WITH THE POLICY REMOVED. The force path takes all
            # 64 candidates; what it asks at the frontier is whether the
            # engine's closure is the card or the pattern.
            MODE_FLAGS="--layer_checkpoint --placement_engine --engine_live --hbm_budget_gb ${BUDGET} --prefetch_window_layers ${WIN} --evict_layout_census --force_action evict"
            ;;
        engine_loose)
            # THE 7744 INSTRUMENT, UNCHANGED. The gate still projects, still
            # books infeasible steps and still picks every bundle it can
            # reach — it simply does not raise.
            MODE_FLAGS="--layer_checkpoint --placement_engine --engine_live --hbm_budget_gb ${BUDGET} --prefetch_window_layers ${WIN} --evict_layout_census"
            ;;
        *)
            echo "UNKNOWN MODE ${MODE} — skipping"; return 2
            ;;
    esac

    MODE_FLAGS="${MODE_FLAGS}${EXTRA_FLAGS:+ ${EXTRA_FLAGS}}"

    case "$BLOCK" in
        M) LABEL="MATCHED SHAPE x=100 — four policies, one shape, one hardware, WORLD 4; the table's time/overhead/bandwidth row" ;;
        V) LABEL="VANILLA LADDER rung — no checkpointing at all; x=8 is the decider" ;;
        F) LABEL="FORCE-EVICT AT THE ENGINE'S FAILING RUNG x=360 — is the world-4 closure the card or the pattern?" ;;
        O) LABEL="OFFLOAD LADDER rung — x=200 fixes the slope, x=356 is the fit side, x=360 is the decider" ;;
        P) LABEL="PER-MODULE LADDER rung — x=33 is the decider and the block's only slope point" ;;
        *) echo "UNKNOWN BLOCK ${BLOCK} — skipping"; return 2 ;;
    esac

    TAG=h${HID}L${NL}_${BLOCK}_${MODE}_b${BUDGET}_w${WIN}_c${CHUNK}_s${SQ}_n${NSTEPS}_norminput_cacheoff
    OUTDIR=profile_results/l64base4/${SLURM_JOB_ID}${ASUF}-${TAG}_b${BS}_v${VOC}_${DT}_mem_efficient
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
        || echo "ARM ${IDX} (block ${BLOCK}, mode ${MODE}, x=${XVAL}) DID NOT COMPLETE — READ THE FAILURE CLASS, IT IS THE FINDING, AND THE CLASSES MUST NEVER BE SUMMED. (0) A NODE FAILURE, A SOCKET TIMEOUT OR A 'Task launch ... failed on node' IS NOT A RESULT OF ANY KIND — check the .err file for slurm_receive_msgs before reading any other class, because this one can imitate all of them; the arm must be RE-RUN. (1) A FAILURE IN M1 (static x=100) IS NOT A CAPACITY RESULT: x=100 sits 529 MiB under a shape 7744 completed, on a five-point line with max residual 0.006 — a miss means the build moved and every prediction here is void. (2) A cudaHostAlloc / PINNED-MEMORY FAILURE IN AN OFFLOAD ARM IS AN IMPLEMENTATION LIMIT OF THE STOCK BASELINE AND IS ITSELF A TABLE ROW, NOT A CAPACITY RESULT — host footprint is 96x MiB per rank (9.4 GiB at x=100, 18.8 at x=200, 33.4 at x=356, 34.1 at x=364) against 112 GB; record the host figure, never retry with a smaller pool, because there is no pool. (3) A CUDA OOM IN BACKWARD IN STEP 1 IN BLOCK O, F OR M IS THE DESIGNED STOP and closes that block's axis. Record reserved/allocated AND THE FAILING REQUEST SIZE — it must be 4x MiB, the replayed feed-forward activation, non-migratable by construction: 400.00 at x=100, 800.00 at x=200, 1424.00 at x=356, 1440.00 at x=360, 1456.00 at x=364. A DIFFERENT SIZE IS A DIFFERENT MECHANISM and the rung below is not a bracket; this OOM does NOT bound capacity from above. (4) A CUDA OOM IN FORWARD IN BLOCK V IS THE EXPECTED CLASS — vanilla has no replay; reading class (3) into V is a category error. In O, F or M a forward OOM means far above the ceiling: the rung below is a LOOSE bracket and one intermediate rung is owed. (5) AN OFFLOAD ARM AT SLOPE 132.32 HAS THE WRAPPING ORDER INVERTED (interior offloaded) and voids blocks O and M2. 37.82 is the EXPECTED reading (on the engine's line); 36.32 means one boundary more absent than the engine manages — a result, not a defect. (6) AN ARM WHOSE STEP-0 LOSS DIFFERS FROM M1'S voids that arm on fidelity: every path here is lossless by construction and by test. (7) A GATE REFUSAL RAISING MUST NOT HAPPEN — --strict_evict is off everywhere; the gate BOOKING infeasible steps on rungs that complete is expected. (8) A POOL DENIAL ('cpu_dram_full') is SILENT — check pool.denials_total and evict_realization.denied_ops on M3, M4 and F even when they complete; a denial at this x's boundary size is a pool-sizing defect, at any other size a Gate-0 bundle-mismatch defect. Block O has no pool and cannot produce this class. (9) A DRIVER RAISE IN BLOCK P naming --force_action recompute as a no-op means --layer_checkpoint leaked into the permodule mode string — the mode is wrong, not the shape. (10) A NaN LOSS or a census class other than the single fp16 residual boundary on an engine arm is a restore-layout defect: read evict_layout_census first. (11) A HANG OR WALL-CLOCK KILL is the W=0 class — every restore is synchronous; check whether step 0 printed before assuming capacity. (12) A '[PinnedPool] refusing to pin' RuntimeError is the host-memory guard: a POOL-SIZING DEFECT, never a capacity result — the arm is void and the fix is in _derive_pool_buckets, never a smaller ladder. Continuing per STOP_ON_FAIL."
}

# Ladder blocks close INDEPENDENTLY. Block M is not a ladder: its four arms
# are one table row and a failure there stops nothing downstream by itself —
# but an M1 identity miss voids the job's predictions (class 1), so read M
# before trusting anything after it.
CLOSED=""

for ARM in $ARMS; do
    OLDIFS=$IFS; IFS=:; set -- $ARM; IFS=$OLDIFS
    BLOCK=$1; NL=$2; SQ=$3; CHUNK=$4; MODE=$5; BUDGET=$6; WIN=$7; NSTEPS=$8
    EXTRA_FLAGS=""

    case " $CLOSED " in
        *" $BLOCK "*)
            echo "SKIP block ${BLOCK} rung seq_len=${SQ}: the ladder already closed below it."
            continue
            ;;
    esac

    run_arm

    # The predicate is the JSON, not srun's exit code: an OOM can tear down
    # cleanly and a hang can be killed with status zero. OUTDIR carries the
    # attempt number, so a requeue cannot read a prior attempt's file.
    if [ ! -f "$OUTDIR/steps_rank0.json" ]; then
        if [ "$STOP_ON_FAIL" = "1" ] && [ "$BLOCK" != "M" ]; then
            echo "LADDER ${BLOCK} CLOSED at x=${XVAL}: no steps_rank0.json, so this rung"
            echo "  did not complete. The rung BELOW is this block's largest measured fit"
            echo "  and the pair is its capacity bracket — which belongs to THIS block's"
            echo "  allocation pattern and to no other."
            echo "  CHECK FAILURE CLASS (0) FIRST: a node failure closes nothing and the"
            echo "  rung is owed a re-run."
            echo "  Re-run with STOP_ON_FAIL=0 to climb through it and test monotonicity."
            CLOSED="$CLOSED $BLOCK"
        fi
    fi
    sleep 20   # let NCCL tear down and the rdzv port clear before the next arm
done

echo "RUN COMPLETE: ran ${IDX} of ${NARM} arms."
echo ""
echo "LINES EVERY BLOCK IS READ AGAINST (L=64, c=256, regime B, fused-input, cache off, b=1, v=32000, WORLD 4):"
echo "  static L=64 (7744)          : 1476.80 + 132.3201x    largest fit x=104 at 15238.11, x=108 fails"
echo "  engine W=0 step1 (7744)     : 1866.89 + 37.820312x   largest fit x=356 at 15330.92, x=360 OOM'd"
echo "  THE ENGINE'S SPAN           : 356/104 = 3.423, ratio of ceilings [3.296, 3.462), W=0, WORLD 4"
echo "  one boundary                : 1.5000x MiB per layer = 2 B x hidden x tokens"
echo ""
echo "WHAT TO READ, PER BLOCK:"
echo "  M : the four-policy table row — peak, step time, PCIe traffic, host bytes, planner ms."
echo "      M4 vs M2 is the ADAPTIVITY argument: policy-on finds no breach and moves ~zero;"
echo "      policy-off moves 9600 MiB each way per step at a shape with 629 MiB of headroom."
echo "  V : largest fit x only; rungs ~1800 MiB apart do not fit a quotable slope."
echo "  F : pick count (expect 64), realized_fraction (1.0), then the verdict — fit at"
echo "      15482.20 makes the engine's closure pattern-dependent at world 4 (it has a rung"
echo "      in hand); an OOM at request 1440.00 makes the world-4 cap real at this shape."
echo "  O : THE SLOPE FIRST (x=200 and x=356): 37.82 = on the engine's line, identity carried"
echo "      to world 4. Then the largest fit — block O's own bracket and no one else's."
echo "  P : x=32 against 14983.80, then x=33 — the block's only slope point at"
echo "      world 4."
echo ""
echo "AFTER THE JOB: append one row per arm to profile_results/runs_4gpu.csv, never overwrite;"
echo "snapshot under profile_results/l64base4/. These arms are the four-rank"
echo "baseline comparison."