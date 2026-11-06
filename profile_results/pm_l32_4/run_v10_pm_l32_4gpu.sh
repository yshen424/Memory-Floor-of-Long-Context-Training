#!/bin/bash
#SBATCH --job-name=fsdp_v10_pml32
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1        # 1 process per node (1 GPU per node)
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/v10pml32_%j.log
#SBATCH --error=logs/v10pml32_%j.err

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  V10 — PER-MODULE MASS PER LAYER AT THE PEAK, FROM TWO DEPTHS, AND THE   ║
# ║  CHUNKED LOSS'S BYTES PER CHUNK TOKEN. V100, world 4.                    ║
# ║                                                                          ║
# ║  Under the per-module unit each layer holds p MiB per 1024 tokens at the ║
# ║  peak instant; the paper quotes p = 6.008 and builds the per-module span ║
# ║  (n + p·L)/(n + 3.008·L) on it. One depth cannot separate p from the     ║
# ║  non-layer slope n, so this job measures the per-module line at L = 32   ║
# ║  and reads p against the L = 64 line (7747, 7793) re-measured here.      ║
# ║  Block K measures the chunked loss's 12 B per chunk token per vocabulary ║
# ║  entry from a chunk-size ladder.                                         ║
# ║                                                                          ║
# ║  Tier-A (h 768, I 2048, 12/4 heads, v 32000), chunked CE c=256 (block K  ║
# ║  varies c), regime B fp16/sdpa_mem_efficient, fused-input, cache off,    ║
# ║  b=1, seed 1234, 3 steps per arm (failures can land in step 2).          ║
# ║    C  whole-layer static, L=32: x = 24, 48   identity control            ║
# ║    P  per-module, L=32: x = 32, 40, 48, 56, then 16, 20, 24              ║
# ║    R  per-module, L=64: x = 32               repeat of 7747's point      ║
# ║    K  whole-layer static, L=32, x = 8: c = 1024, 2048, 4096, 8192        ║
# ║  Per-module = the instrument with --force_action recompute and NO        ║
# ║  --layer_checkpoint (7747's flags).                                      ║
# ║  WALL CLOCK ~1 h (14 arms).                                              ║
# ║  SUBMIT WITH:  sbatch run_v10_pm_l32_4gpu.sh                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── THE PREDICTIONS. PRE-REGISTERED 2026-11-05 ───────────────────────────────
# Peak = max over steps 1+ (MiB); step 0 lacks AdamW's state.
#   C  harness static line, L=32: 900.6560 + 84.320312x
#        x=24: 2924.3435   x=48: 4948.0310
#      C-P1: both within ±0.05 MiB. A miss means the build moved: the job is
#            void and nothing below is read.
#   R  7747 measured 14983.7998 (step 1+), 14121.8608 (step 0).
#      R-P1: within ±0.05 MiB at both steps, so 7747/7793's L=64 lines hold
#            on this build.
#   P  L=64 per-module lines, two instants inside layer 63's backward
#      (7747 x=32/33, 7793 x=16/24):
#        high  1485.7998 + 421.8125x   (binds at x >= 27)
#        low   1566.0504 + 418.8086x
#      Carried to L=32 with p = 6.0078 (the paper's 6.008) and the
#      whole-layer intercept step per layer (576.1406 MiB for 32 layers,
#      harness arithmetic):
#        high  909.6592 + 229.5629x    low  989.9098 + 226.5590x
#        crossing at x ≈ 26.7, so x = 32-56 sit on high, x = 16-24 on low:
#        x=16: 4614.85  x=20: 5521.09  x=24: 6427.33  (low)
#        x=32: 8255.67  x=40: 10092.18 x=48: 11928.68 x=56: 13765.18 (high)
#        step 1+ − step 0 = 477.85 at every x (AdamW at L=32).
#   P-P1 (THE DECIDER): the slope s of the x = 32-56 points gives
#        p = (421.8125 − s)/32. HOLDS if p is within 6.0078 ± 0.01, i.e.
#        s in [229.2429, 229.8829]. The alternative, a non-layer slope
#        shared with the whole-layer unit (n = 36.3203), gives p = 6.0233
#        and s = 229.0664, outside the window. The four points must lie on
#        one line to 0.05 MiB, or the instant changes inside the range and
#        p is not read from this job.
#   P-P2: x = 16-24 on the low line to 0.05 MiB (slope 3.00 below high).
#   P-P3: intercepts within ±1 MiB of 909.66 / 989.91 (the intercept step
#        is carried from the whole-layer unit, untested here; a miss here
#        does not touch p).
#   P-P4: step 1+ − step 0 = 477.85 ± 0.05 at every completing arm.
#   K  chunk-size ladder at x = 8: with the chunk's backward the binding
#      instant, each doubling of c adds 12 B × Δc × 32000:
#        +375.00 (1024→2048)  +750.00 (2048→4096)  +1500.00 (4096→8192) MiB.
#      K-P1 (the paper's 12 B per chunk token per vocabulary entry): the
#        2048→4096 and 4096→8192 steps within ±0.05 MiB of 750.00 and 1500.00.
#      K-P2: the 1024→2048 step is 375.00 if c = 1024 lies above the knee
#        (where the loss instant overtakes the last layer's replay); a
#        smaller step puts the knee above 1024 and does not touch K-P1.
#      Absolute peaks are not registered: no four-rank chunk-size line exists.
#   IF P-P1 FAILS: the paper's 6.008, the per-module span formula, its
#        1.84x at L=64 and its limit p/(p − 3.008) are re-derived from the
#        measured p; the frontier slope 421.81 and the 289.49 / 97.49
#        contrasts are measured at L=64 and do not change.
# Failure classes: (0) node failure / socket timeout / rendezvous — re-run,
#   closes nothing; (1) any OOM — no arm here is near capacity (largest
#   predicted peak 14983.80 at R, a completed shape), so an OOM is a defect,
#   not a bracket; (2) a '[PinnedPool] refusing to pin' error is a
#   pool-sizing defect, never a result; (3) a NaN loss, or a step-0 loss
#   differing between arms of the same depth, voids that arm.

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

OUTROOT=profile_results/pm_l32_4
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

# Private copy of the driver taken at job start, so an edit during the run cannot reach it.
CODE=${OUTROOT}/code_${SLURM_JOB_ID}${ASUF}
mkdir -p "$CODE"
cp -r train_fsdp_profile.py fsdp_profiler.py chunked_ce.py fused_rmsnorm.py memory_manager "$CODE"/
echo "code snapshot: $CODE ($(sha1sum train_fsdp_profile.py | cut -c1-12))"

# block:mode:layers:x[:chunk tokens, default 256]
ARMS="C:static:32:24 P:permodule:32:32 P:permodule:32:40 P:permodule:32:48 P:permodule:32:56 K:static:32:8:1024 K:static:32:8:2048 K:static:32:8:4096 K:static:32:8:8192 P:permodule:32:16 P:permodule:32:20 P:permodule:32:24 R:permodule:64:32 C:static:32:48"

WS=4; HID=768; INTER=2048; NH=12; NKV=4; VOC=32000
DT=fp16; ATTN=sdpa_mem_efficient; NORM=input; SEED=1234; NSTEPS=3; BS=1
IDX=0
NARM=$(echo $ARMS | wc -w)

for ARM in $ARMS; do
    OLDIFS=$IFS; IFS=:; set -- $ARM; IFS=$OLDIFS
    BLOCK=$1; MODE=$2; NL=$3; X=$4; C=${5:-256}; SQ=$((X * 1024))
    case "$MODE" in
        permodule) MODE_FLAGS="--placement_engine --engine_live --hbm_budget_gb 14.9769 --prefetch_window_layers 0 --evict_layout_census --force_action recompute" ;;
        static) MODE_FLAGS="--layer_checkpoint" ;;
    esac
    TAG=V10${BLOCK}_L${NL}_h${HID}_${MODE}_x${X}_n${NSTEPS}_c${C}
    OUTDIR=${OUTROOT}/${SLURM_JOB_ID}${ASUF}-${TAG}_b${BS}_${DT}_mem_efficient_world${WS}
    mkdir -p "$OUTDIR"
    IDX=$((IDX + 1))
    echo "ARM ${IDX}/${NARM} block ${BLOCK} ${MODE} L=${NL} x=${X} c=${C} -> ${OUTDIR}"
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
            --chunked_ce ${C} ${MODE_FLAGS} \
            --fused_rmsnorm ${NORM} --use_cache off --seed ${SEED} \
            --disable_hbm_drift --output_dir "$OUTDIR" \
            < /dev/null 2> >(tee "$OUTDIR/stderr.txt" >&2) \
        || echo "ARM ${IDX} (block ${BLOCK}, ${MODE}, L=${NL}, x=${X}, c=${C}) DID NOT COMPLETE — read the failure class in the header."
    sleep 20
done

echo "RUN COMPLETE: ran ${IDX} of ${NARM} arms."
# ── Reader: measured peaks against the registered lines, then the decider ────
python3 - "$OUTROOT" "${SLURM_JOB_ID}${ASUF}" <<'PY'
import glob, json, os, re, sys
root, job = sys.argv[1], sys.argv[2]
ADAM = {32: 477.8452, 64: 861.9390}
def pred(block, L, x):
    if block == 'C':
        return 900.6560 + 84.320312 * x
    if L == 64:
        return max(1485.7998 + 421.8125 * x, 1566.0504 + 418.8086 * x)
    return max(909.6592 + 229.5629 * x, 989.9098 + 226.5590 * x)
pts, kpk = [], {}
for d in sorted(glob.glob(os.path.join(root, job + '-V10*'))):
    m = re.search(r'-V10([CPRK])_L(\d+)_h\d+_(\w+?)_x(\d+)_n\d+_c(\d+)_', d)
    block, L, x, c = m[1], int(m[2]), int(m[4]), int(m[5])
    f = os.path.join(d, 'steps_rank0.json')
    if not os.path.exists(f):
        print(f"{block} L={L} x={x:<3d} NO STEPS FILE — read stderr.txt")
        continue
    st = json.load(open(f))['steps']
    if len(st) < 2:
        print(f"{block} L={L} x={x:<3d} only {len(st)} step(s)")
        continue
    p1, p0 = max(s['peak_mb'] for s in st[1:]), st[0]['peak_mb']
    if block == 'K':
        kpk[c] = p1
        print(f"K L={L:<3d} x={x:<3d} c={c:<5d} step1+ {p1:10.4f}   step1+ - step0 {p1 - p0:8.4f}")
        continue
    pr = pred(block, L, x)
    print(f"{block} L={L:<3d} x={x:<3d} step1+ {p1:10.4f} pred {pr:10.4f} diff {p1 - pr:+8.4f}   "
          f"step1+ - step0 {p1 - p0:8.4f} (AdamW {ADAM[L]:.4f}, diff {p1 - p0 - ADAM[L]:+.4f})")
    if block == 'P' and x >= 32:
        pts.append((x, p1))
for lo, hi in ((1024, 2048), (2048, 4096), (4096, 8192)):
    if lo in kpk and hi in kpk:
        d = kpk[hi] - kpk[lo]; want = 375.0 * (hi - lo) / 1024
        print(f"K step c={lo}->{hi}: {d:+.4f} MiB (12 B/token/vocab predicts {want:+.2f}; "
              f"{d * 1048576 / ((hi - lo) * 32000):.4f} B/token/vocab)"
              + ("" if lo == 1024 else ("  K-P1 ok" if abs(d - want) <= 0.05 else "  K-P1 MISS")))
if len(pts) >= 2:
    n = len(pts); mx = sum(x for x, _ in pts) / n; my = sum(y for _, y in pts) / n
    s = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)
    a = my - s * mx
    res = max(abs(y - a - s * x) for x, y in pts)
    p = (421.8125 - s) / 32
    print(f"\nP-P1: x=32-56 fit {a:.4f} + {s:.4f}x over {n} points, max residual {res:.4f} MiB")
    print(f"      p = (421.8125 - {s:.4f})/32 = {p:.4f} MiB/layer/x  ->  "
          + ("HOLDS" if abs(p - 6.0078) <= 0.01 and res <= 0.05 else "FAILS (read the header's IF P-P1 FAILS)"))
PY
echo "AFTER THE JOB: append one row per arm to profile_results/runs_4gpu.csv"
echo "(back it up as runs_4gpu.csv.bak_pre<job> first) and move this"
echo "script to ${OUTROOT}/."
