#!/bin/bash
#SBATCH --job-name=fsdp_l64_cap4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1        # 1 process per node (1 GPU per node)
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/cap4_%j.log
#SBATCH --error=logs/cap4_%j.err

# ── JOB: CAPPED BRACKETS ON V100, WORLD 4 ───────────────────────────────────
# Tests whether a bracket measured under a capped caching allocator
# (--hbm_cap_gb -> set_per_process_memory_fraction) is PREDICTABLE, on the
# card where every line is already known.
#
# Four climbing ladders, 2 steps per rung (the step-1 peak carries AdamW
# state), each stopping at its first failure:
#   A  cap 6 GiB, static         x = 30..37 step 1
#   B  cap 9 GiB, static         x = 53..60 step 1
#   C  cap 6 GiB, stock offload  x = 96..112 step 2
#   D  cap 9 GiB, stock offload  x = 176..192 step 2
# Static and offload fail on different request sizes (4x MiB: ~130 vs
# ~400-750 MiB), so the four brackets constrain both the reserve and the
# request dependence, at two caps.
#
# L=64, lever 3 c=256, regime B fp16/sdpa_mem_efficient, fused-input,
# cache off, b=1, v=32000, seed 1234. Wall clock ~4-5 h (D dominates:
# ~25 min per rung).

# ── THE PREDICTION. PRE-REGISTERED 2026-10-09, from harness/ ───────────────
# Lines (harness, derived; MiB, x = tokens/1024):
#   static         1476.7966 + 132.320312x
#   stock offload  1866.8919 +  37.820312x
# Uncapped V100 capacity: fits iff peak <= C0 - alpha*request,
#   C0 = 15670.85 MiB, alpha = 0.198, request = 4x MiB (the replayed MLP
#   activation). Uncapped, C0 = (card total 16149 MiB) - (CUDA context) - delta,
#   where delta is the allocator's reserve; the context share is not known, so
#   delta is bounded only to [~60, 478] MiB.
# Under a cap the context lies OUTSIDE the cap, so the model becomes
#     fits  iff  peak <= cap_MiB - delta - alpha * 4x
# with the SAME alpha and an unknown delta. This job measures delta.
#
# C-P1 PEAKS: every completing rung on its harness line to 0.05 MiB (a cap
#      changes no byte).
# C-P2 FAILURES: every failure an OOM in step 1's backward on a request of
#      exactly 4x MiB (static and offload alike).
# C-P3 BRACKETS (largest fit), from delta in [60, 478]:
#        A  x in [31.5, 34.6]      B  x in [54.5, 57.7]
#        C  x in [98.4, 109.2]     D  x in [178.0, 188.8]
# C-P4 ONE RESERVE: with alpha = 0.198, the four brackets admit a common delta
#      (rung resolution: 133 MiB on A/B, 77 MiB on C/D). FALSIFIER: the delta
#      intervals of the two caps are disjoint (the reserve is not a constant),
#      or the static and offload intervals at one cap are disjoint (alpha
#      differs under a cap).
# C-P5 If C-P4 holds, the capped model C(req) = cap - delta - 0.198*req is the
#      method for the remote cards, pre-registered with this job's delta.

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
    echo "WORLD SIZE IS ${SLURM_NTASKS}, NOT 4. The lines and capacity model are world 4."
    exit 2
fi

OUTROOT=profile_results/capped4
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

# block:cap_gib:mode:x — cheap ladders first.
ARMS=""
for X in 30 31 32 33 34 35 36 37;                  do ARMS="$ARMS A:6:static:$X"; done
for X in 53 54 55 56 57 58 59 60;                  do ARMS="$ARMS B:9:static:$X"; done
for X in 96 98 100 102 104 106 108 110 112;        do ARMS="$ARMS C:6:offload:$X"; done
for X in 176 178 180 182 184 186 188 190 192;      do ARMS="$ARMS D:9:offload:$X"; done

HID=768; INTER=2048; NH=12; NKV=4; NL=64; BS=1; VOC=32000; CHUNK=256
DT=fp16; ATTN=sdpa_mem_efficient; NORM=input; SEED=1234; NSTEPS=2
PORT_BASE=${MASTER_PORT}
IDX=0
NARM=$(echo $ARMS | wc -w)
CLOSED=""

for ARM in $ARMS; do
    OLDIFS=$IFS; IFS=:; set -- $ARM; IFS=$OLDIFS
    BLOCK=$1; CAP=$2; MODE=$3; X=$4; SQ=$((X * 1024))
    case " $CLOSED " in *" $BLOCK "*) echo "SKIP ${BLOCK} x=${X}: ladder closed below"; continue ;; esac
    case "$MODE" in
        static)  MODE_FLAGS="--layer_checkpoint" ;;
        offload) MODE_FLAGS="--layer_checkpoint --activation_offload" ;;
    esac
    TAG=h${HID}L${NL}_${BLOCK}_cap${CAP}_${MODE}_x${X}_n${NSTEPS}_c${CHUNK}_norminput_cacheoff
    OUTDIR=${OUTROOT}/${SLURM_JOB_ID}${ASUF}-${TAG}_b${BS}_v${VOC}_${DT}_mem_efficient_world4
    mkdir -p "$OUTDIR"
    IDX=$((IDX + 1))
    echo "ARM ${IDX}/${NARM} [${BLOCK}] cap ${CAP} GiB, ${MODE}, x=${X} -> ${OUTDIR}"
    srun torchrun \
        --nnodes=$SLURM_NNODES --nproc_per_node=1 \
        --rdzv_id=${SLURM_JOB_ID}${ASUF}_a${IDX} --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$((PORT_BASE + IDX)) \
        train_fsdp_profile.py \
            --hidden_size ${HID} --intermediate_size ${INTER} --num_layers ${NL} \
            --num_heads ${NH} --num_kv_heads ${NKV} --vocab_size ${VOC} \
            --dtype ${DT} --attn_impl ${ATTN} \
            --seq_len ${SQ} --batch_size ${BS} --steps ${NSTEPS} \
            --chunked_ce ${CHUNK} ${MODE_FLAGS} --hbm_cap_gb ${CAP} \
            --fused_rmsnorm ${NORM} --use_cache off --seed ${SEED} \
            --disable_hbm_drift --output_dir "$OUTDIR" \
            < /dev/null 2> >(tee "$OUTDIR/stderr.txt" >&2) \
        || echo "ARM ${IDX} (${BLOCK}, cap ${CAP}, ${MODE}, x=${X}) DID NOT COMPLETE. Read its stderr.txt: (0) node failure / socket timeout -> re-run, it closes nothing; (1) CUDA OOM in step-1 backward with 'Tried to allocate' exactly 4x MiB -> the DESIGNED closure (C-P2); (2) an OOM on a different size, or in forward -> a different mechanism, record it and do not read this ladder's bracket as one; (3) 'exceeds this card' -> the cap flag is wrong."
    if [ ! -f "$OUTDIR/steps_rank0.json" ]; then
        echo "LADDER ${BLOCK} CLOSED at x=${X} (check failure class 0 first)."
        CLOSED="$CLOSED $BLOCK"
    fi
    sleep 20
done

echo "RUN COMPLETE: ran ${IDX} of ${NARM} arms."
echo "READ IT WITH:  python3 tools/capped_brackets.py ${OUTROOT} --job ${SLURM_JOB_ID}${ASUF}"
echo "AFTER THE JOB: append rows to profile_results/runs_4gpu.csv (back it up first) and"
echo "move this script to ${OUTROOT}/."
