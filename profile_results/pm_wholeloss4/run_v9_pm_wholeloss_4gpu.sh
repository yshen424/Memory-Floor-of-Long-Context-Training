#!/bin/bash
#SBATCH --job-name=fsdp_v9_pmwl
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1        # 1 process per node (1 GPU per node)
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/v9pmwl_%j.log
#SBATCH --error=logs/v9pmwl_%j.err

# ── JOB: PER-MODULE LINE POINTS AND THE WHOLE-LOSS BRACKET. V100, WORLD 4 ───
# Block A — the per-module line `1485.80 + 421.81x` rests on two lengths,
# x = 32 and 33 (7747 block P); x = 16 and 24 give it four points over 2x.
# Block B — the whole-loss whole-layer line (7744 block U, `1414.92 + 536.77x`,
# fitted at x = 2, 4, 8) gets line points far beyond the fitted range plus
# its own bracket, x = 20, 24..28, against the chunked static bracket
# [104, 108) (7747) on the same nodes; the ratio tests "chunking the loss
# buys about four times the context".
#
# Tier-A L=64 (h 768, I 2048, 12/4 heads, v 32000), regime B
# fp16/sdpa_mem_efficient, fused-input, cache off, b=1, seed 1234,
# 3 steps per arm (failures can land in step 2).
#   A  per-module, chunked CE c=256: x = 16, 24   (7747 block P flags:
#      engine with --force_action recompute, NO --layer_checkpoint)
#   B  whole-layer static, whole loss (no --chunked_ce): x = 20, 24, 25,
#      26, 27, 28 — every rung runs; a failure does not stop the ladder.
# Wall clock ~1.2 h (8 arms; failing rungs die in step 1's backward).

# ── THE PREDICTIONS. PRE-REGISTERED 2026-10-21 ───────────────────────────────
# Lines are the measured four-rank lines (peak = max over steps 1+, MiB).
#   A  per-module (7747 P: 14983.7998 @ x=32, 15405.6123 @ x=33):
#        step1+  1485.7998 + 421.8125x   step0  623.8598 + 421.8125x
#        x=16: step1+ 8234.80   step0 7372.86
#        x=24: step1+ 11609.30  step0 10747.36
#   PM-P1: both points on the step1+ line to ±0.05 MiB (slope 421.81 then rests
#          on four points, x = 16-33). PM-P2: step1+ − step0 = 861.94 (AdamW).
#          A miss above 0.05 MiB names an instant the two-point line hid; the
#          per-module derived numbers (289.49, 97.49, 1.84x, the 1.997 limit)
#          are then re-derived from the four-point fit.
#   B  whole loss (7744 U: 3561.99 @ x=4, 5709.07 @ x=8, step1+):
#        step1+  1414.91 + 536.77x
#        x=20: 12150.31  x=24: 14297.39  x=25: 14834.16  x=26: 15370.93
#        x=27: 15907.70  x=28: 16444.47
#        failing request, if a rung fails: 125.00x MiB (fp32 logits gradient,
#        tokens x 32000 x 4 B) in loss.backward(), step 1 — 3250.00 @ 26,
#        3375.00 @ 27, 3500.00 @ 28.
#   WL-P1: every completing rung on the line to ±0.05 MiB.
#   WL-P2 (bracket, harness capacity C0 15627.64 − 0.163 × request):
#          registered [25, 26) — x=25 fits by 284 MiB, x=26 fails by 273.
#          SECOND OUTCOME, registered with it: [26, 27) or [27, 28). The
#          model's α was calibrated on chunked-loss requests ≤ 1.5 GiB and is
#          known to under-predict capacity when the failing request is a
#          single 3.2-3.5 GiB block. A one- or two-rung high bracket repeats
#          that known miss; a LOW bracket ([24, 25) or below) is a new miss.
#   WL-P3 (context ratio against chunked static [104, 108)):
#          [25,26) -> 104/25 = 4.16, true ratio in (4.00, 4.32);
#          [26,27) -> 4.00 in (3.85, 4.15); [27,28) -> 3.85 in (3.71, 4.00).
#          Every outcome supports "about four times"; the measured ratio is
#          quoted with its interval.
# Failure classes: (0) node failure / socket timeout / rendezvous — re-run,
#   closes nothing; (1) OOM in step 1 or 2 backward on the 125x request — the
#   designed stop; (2) OOM in forward — far above the ceiling, the rung below
#   is a loose bracket; (3) any other request size is a different mechanism;
#   (4) a '[PinnedPool] refusing to pin' error on block A is a pool-sizing
#   defect, never a result (7747 pinned ~5 GiB per node at these settings).

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29950
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ATTEMPT=${SLURM_RESTART_COUNT:-0}
if [ "$ATTEMPT" = "0" ]; then ASUF=""; else ASUF="r${ATTEMPT}"; fi
echo "NODES: $SLURM_JOB_NODELIST  NNODES: $SLURM_NNODES  ATTEMPT: $ATTEMPT"
if [ "$SLURM_NNODES" != "4" ]; then
    echo "ALLOCATION IS ${SLURM_NNODES} NODES, NOT 4."
    exit 2
fi

OUTROOT=profile_results/pm_wholeloss4
mkdir -p logs "$OUTROOT"

MIN_AVAIL_GIB=60
AVAIL=$(srun --ntasks-per-node=1 bash -c \
    'echo "$(hostname) $(awk "/MemAvailable/{printf \"%d\", \$2/1048576}" /proc/meminfo)"' \
    < /dev/null)
echo "MemAvailable per node (GiB):"; echo "$AVAIL" | sed 's/^/    /'
if echo "$AVAIL" | awk -v m=$MIN_AVAIL_GIB 'NF<2 || $2<m {bad=1} END {exit !bad}'; then
    echo "PREFLIGHT FAILED: a node has under ${MIN_AVAIL_GIB} GiB MemAvailable, or did not answer."
    exit 3
fi

# Private copy of the driver taken at job start, so a mid-run edit cannot reach a running arm.
CODE=${OUTROOT}/code_${SLURM_JOB_ID}${ASUF}
mkdir -p "$CODE"
cp -r train_fsdp_profile.py fsdp_profiler.py chunked_ce.py fused_rmsnorm.py memory_manager "$CODE"/
echo "code snapshot: $CODE ($(sha1sum train_fsdp_profile.py | cut -c1-12))"

# block:mode:x
ARMS="A:permodule:16 A:permodule:24 B:wholeloss:20 B:wholeloss:24 B:wholeloss:25 B:wholeloss:26 B:wholeloss:27 B:wholeloss:28"

WS=4; NL=64; HID=768; INTER=2048; NH=12; NKV=4; VOC=32000
DT=fp16; ATTN=sdpa_mem_efficient; NORM=input; SEED=1234; NSTEPS=3; BS=1
IDX=0
NARM=$(echo $ARMS | wc -w)

for ARM in $ARMS; do
    OLDIFS=$IFS; IFS=:; set -- $ARM; IFS=$OLDIFS
    BLOCK=$1; MODE=$2; X=$3; SQ=$((X * 1024))
    case "$MODE" in
        permodule) MODE_FLAGS="--chunked_ce 256 --placement_engine --engine_live --hbm_budget_gb 14.9769 --prefetch_window_layers 0 --evict_layout_census --force_action recompute"; C=256 ;;
        wholeloss) MODE_FLAGS="--layer_checkpoint"; C=0 ;;
    esac
    TAG=V9${BLOCK}_L${NL}_h${HID}_${MODE}_x${X}_n${NSTEPS}_c${C}
    OUTDIR=${OUTROOT}/${SLURM_JOB_ID}${ASUF}-${TAG}_b${BS}_${DT}_mem_efficient_world${WS}
    mkdir -p "$OUTDIR"
    IDX=$((IDX + 1))
    echo "ARM ${IDX}/${NARM} block ${BLOCK} ${MODE} x=${X} -> ${OUTDIR}"
    echo "  mode flags: ${MODE_FLAGS}"
    srun --nodes=${WS} --ntasks=${WS} --ntasks-per-node=1 torchrun \
        --nnodes=${WS} --nproc_per_node=1 \
        --rdzv_id=${SLURM_JOB_ID}${ASUF}_a${IDX} --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$((MASTER_PORT + IDX)) \
        "$CODE"/train_fsdp_profile.py \
            --hidden_size ${HID} --intermediate_size ${INTER} --num_layers ${NL} \
            --num_heads ${NH} --num_kv_heads ${NKV} --vocab_size ${VOC} \
            --dtype ${DT} --attn_impl ${ATTN} \
            --seq_len ${SQ} --batch_size ${BS} --steps ${NSTEPS} \
            ${MODE_FLAGS} \
            --fused_rmsnorm ${NORM} --use_cache off --seed ${SEED} \
            --disable_hbm_drift --output_dir "$OUTDIR" \
            < /dev/null 2> >(tee "$OUTDIR/stderr.txt" >&2) \
        || echo "ARM ${IDX} (block ${BLOCK}, ${MODE}, x=${X}) DID NOT COMPLETE — read the failure class in the header; record the step, the phase and the failing request."
    sleep 20
done

echo "RUN COMPLETE: ran ${IDX} of ${NARM} arms."
# ── Reader: measured peaks against the registered lines ─────────────────────
python3 - "$OUTROOT" "${SLURM_JOB_ID}${ASUF}" <<'PY'
import glob, json, os, re, sys
root, job = sys.argv[1], sys.argv[2]
LINES = {'permodule': (1485.7998, 421.8125, 623.8598), 'wholeloss': (1414.91, 536.77, None)}
for d in sorted(glob.glob(os.path.join(root, job + '-V9*'))):
    m = re.search(r'_(permodule|wholeloss)_x(\d+)_', d)
    mode, x = m[1], int(m[2])
    a, b, a0 = LINES[mode]
    f = os.path.join(d, 'steps_rank0.json')
    if not os.path.exists(f):
        err = open(os.path.join(d, 'stderr.txt')).read() if os.path.exists(os.path.join(d, 'stderr.txt')) else ''
        req = re.findall(r'Tried to allocate ([\d.]+ [GM]iB)', err)
        print(f"{mode:9} x={x:<3d} NO STEPS FILE  request {req[:1]}  predicted request {125*x:.2f} MiB" if mode == 'wholeloss' else f"{mode:9} x={x:<3d} NO STEPS FILE")
        continue
    st = json.load(open(f))['steps']
    p1 = max(s['peak_mb'] for s in st[1:]) if len(st) > 1 else float('nan')
    line = f"{mode:9} x={x:<3d} steps {len(st)}  step1+ {p1:.3f} pred {a + b*x:.3f} diff {p1 - a - b*x:+.3f}"
    if a0 is not None:
        line += f"  step0 {st[0]['peak_mb']:.3f} pred {a0 + b*x:.3f} diff {st[0]['peak_mb'] - a0 - b*x:+.3f}"
    print(line)
PY
echo "AFTER THE JOB: append rows to profile_results/runs_4gpu.csv (back it up as"
echo "runs_4gpu.csv.bak_pre<job> first) and move this script to"
echo "${OUTROOT}/."
