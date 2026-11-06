#!/bin/bash -l
#SBATCH --job-name=a30_frontier
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:a30:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=a30_frontier_%j.log
#SBATCH --error=a30_frontier_%j.err
#
# Submit from this directory (the directives above carry the same flags):
#   sbatch -A <account> -p <partition> --qos=<qos> --nodes=1 \
#          --exclusive --gres=gpu:a30:4 --time=24:00:00 a30_frontier.sbatch
# GPU request is typed --gres only; never add --gpus-per-node (they combine).
# Venv: the first of a30_venv, ../l64s_linktest/l64s_venv, ../modern_gpu_probe/venv
# that passes a launcher check (torchrun imports).

# ── JOB: BYTES, KERNELS, TIME AND CAPACITY ON A30. ONE EXCLUSIVE A30 NODE ──────
# ── (Ampere sm80, 24 GB), WORLD 4. Hypotheses R-A1..R-A5.                  ──
# R-A1 bytes need a second architecture; R-A2 bf16 memory-efficient and R-A3
# FlashAttention have no kernel on sm_70; R-A4 capacity and R-A5 time are
# per-platform by definition.
#
#  0  NODE facts: GPUs, topology, NVLink, P2P, NUMA, host RAM, software, card
#     total -> a30_node.json.
#  K  CAP CHECK: --hbm_cap_gb bounds the allocator at 6 and 9 GiB. If it
#     fails, blocks C and N are void and skipped; everything else runs.
#  B  R-A1 BYTES (fp16, mem-efficient, 3 steps): L=32 static x=24/48;
#     L=64 static, offload x=24/48; static+CPU, offload+CPU, force-evict
#     x=48; offload x=100 (7747's matched point).
#  D  R-A2 bf16 mem-efficient: L=64 static x=24/48, offload x=48.
#  F  R-A3 FlashAttention: fp16 static x=24/48, offload x=48; bf16 static
#     x=48; native GQA (k/v at kv width) static x=24/48, offload x=48.
#  T  R-A5 TIME: static/offload pairs, two repeats in ABBA order, at
#     x=16/32/64 -> IN-JOB CALIBRATION writes the tax and crossing
#     predictions -> x=100/150 run against them. P: one profiled static
#     and one profiled offload arm at x=64 (attention-replay share, copy
#     bandwidth, NCCL duty).
#  C  R-A4 CAPPED LADDERS, 2 steps, V100's four (7760): static at 6 and
#     9 GiB, offload at 6 and 9 GiB; each climbs to its first failure.
#  N  NATIVE 24 GB STATIC LADDER: rungs chosen IN-JOB from C's calibrated
#     reserve plus the measured context, written before the ladder runs.
#     (Native offload is not climbed: x~580, ~6 h.)
#
#  All: L=64 unless named, hidden 768, inter 2048, heads 12/4, v=32000,
#  chunked CE 256, whole-layer, fused-input, use_cache off, b=1, seed 1234,
#  --disable_hbm_drift. Each arm saves its own results.
#  WALL CLOCK: ~12 h if the A30 were only as fast as the V100 (B 0.7, D+F
#  0.6, T 5.2, P 0.3, C 3.0, N 2.1 h); expected 5-8 h. No arm starts after
#  22.5 h. 24 h requested (the wait does not depend on it).

# ── PRE-REGISTERED 2026-10-10 from harness/ (platform a30 = V100 constants  ──
# ── carried; MiB, x = tokens/1024). Byte statements are hardware-free; the  ──
# ── capacity and time statements are written by the job itself (files).    ──
# Frozen after the shape sweep (7761) and the V100 smoke (7762). The shape
# sweep's two new terms are zero for every arm here: the shape term vanishes
# at tier-A by construction, and the second-replay class needs static x < 16.0.
# Lines (derived):
#   L=32 static          900.6560 + 84.320312x      x=24: 2924.34   x=48: 4948.03
#   L=64 static         1476.7966 + 132.320312x     x=24: 4652.48   x=48: 7828.17
#   L=64 stock offload  1866.8919 + 37.820312x      x=24: 2774.58   x=48: 3682.27  x=100: 5648.92
#   L=64 force-evict    on the offload line (the cap binds every boundary mechanism)
#   L=64 static+CPU      183.8882 + 132.320312x     x=48: 6535.26   (exactly 3P = 1292.91 lower)
#   L=64 offload+CPU     573.9834 + 37.820312x      x=48: 2389.36
#   L=64 native GQA static 1476.7966 + 130.320312x  x=24: 4604.48   x=48: 7732.17
#   L=64 native GQA offload 1866.8919 + 35.820312x  x=48: 3586.27
#
# A-P1 BYTES CARRY TO AMPERE: every B arm sits on its line plus ONE common offset,
#      all offsets within 0.05 MiB of each other. FALSIFIER: two B arms whose
#      offsets differ by more -- a byte term moved on Ampere (kernel saved set,
#      allocator rounding); name it from the arm that moved.
# A-P2 THE OFFSET IS 0.00 (lean): the library workspace is the V100's 31.26 MiB,
#      as the L40S static arm (621880) showed to 0.00 MiB. A nonzero common
#      offset is a platform constant, recorded, not a miss of A-P1.
# A-P3 bf16 = fp16 BYTE FOR BYTE: every D arm equals its fp16 B/F twin within
#      0.05 MiB (activations are 2 B either way); step-0 losses agree within 2%.
# A-P4 FLASH = MEM-EFFICIENT: under HF's path k/v are expanded before the kernel
#      and both kernels save an fp32 log-sum-exp per head per token, so every
#      flash arm equals its mem-efficient twin within 0.05 MiB.
# A-P5 NATIVE GQA: -2.0000 MiB/x on BOTH static and offload (k and v saved at
#      256 channels instead of 768: 2 planes x 2 B x 512 = 2048 B/token).
#      Two-outcome prediction: 0.0000 MiB/x if flash expands k/v internally
#      (the CPU math kernel does; tests/test_attn_modes.py). Anything else is
#      a third plane, named from the twin difference.
# A-P6 force-evict x=48 lands on the offload line (3682.27 + offset).
# A-P7 CPU offload lowers each line by exactly 3P = 1292.91 MiB, slope unchanged.
#
# T-P1 STEP TIME IS QUADRATIC, THE OFFLOAD COST LINEAR: delta(64)/delta(32)
#      within 2.0 +- 0.6 (linear in x), static T(64)/T(32) in [2.5, 4.0].
# T-P2 REPEATS: the two repeats of each delta agree within 0.1 s or 15%,
#      whichever is larger (uncertainty is run-to-run).
# T-P3/T-P4 written by the job to a30_pred_time.txt BEFORE x=100/150 run
#      (tax and crossing from T(x) and kappa fitted here; the V100 values,
#      kappa 0.032 s/x and x*(2%) ~ 79, are context, NOT the prediction).
#      FALSIFIER: the measured tax at x=100 or 150 outside +-30% of the file.
#
# C-P1 CAPPED BRACKETS = V100's (7760), if delta and alpha are allocator
#      properties (CUDA context lies outside the cap): largest fit
#      CA (6 GiB static) 34, CB (9 GiB static) 57, CC (6 GiB offload) 110,
#      CD (9 GiB offload) 190 (margins 113, 127, 12, 6 MiB: CC and CD are
#      +-1 rung). Every failure: step-1 backward OOM on exactly 4x MiB.
#      FALSIFIER: a shift of 2+ rungs -- the A30 allocator's reserve differs;
#      capcal then re-measures it and the miss is recorded.
# C-P2 one (alpha, delta) fits the four A30 ladders (capcal prints it).
# N-P1/N-P2 written by the job to a30_pred_native.txt BEFORE the native ladder:
#      C0 = card total - outside-allocator bytes (measured) - delta (C-P2).
#      FALSIFIER: the native bracket off its prediction by 2+ rungs.
#
# FAILURE CLASSES (read stderr.txt in the arm's directory): (0) node/socket
# failure -> not a result, re-run; (1) step-1 backward OOM on 4x MiB in a C or
# N rung -> the designed closure; (2) any OOM in a B/D/F/T arm -> these sit far
# under capacity: a defect, record it; (3) a D/F arm failing in the attention
# preflight -> that kernel mode does not dispatch on this build: the arm is void,
# the rest of the job stands; (4) cudaHostAlloc failure -> a host limit, not
# capacity.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
SMOKE=${A30_SMOKE:-}
if [ -z "$SMOKE" ]; then
    # Each candidate must pass the launcher check, not merely exist: a venv without
    # importlib_metadata (python 3.9) has a torchrun that cannot import, and every
    # arm then dies in 1 s.
    venv_ok() (
        [ -f "$1/bin/activate" ] || exit 1
        if [ -f "$1/.module" ]; then module load "$(cat "$1/.module")" 2>/dev/null || true; fi
        source "$1/bin/activate" &&
        python3 -c "import torch.distributed.run, transformers" >/dev/null 2>&1 &&
        torchrun --help >/dev/null 2>&1
    )
    VENV=${A30_VENV:-}
    if [ -n "$VENV" ]; then
        venv_ok "$VENV" || { echo "A30_VENV=$VENV FAILS the launcher check (torchrun / torch.distributed.run import)"; exit 1; }
    else
        for v in a30_venv ../l64s_linktest/l64s_venv ../modern_gpu_probe/venv; do
            if venv_ok "$v"; then VENV=$v; break; fi
            [ -f "$v/bin/activate" ] && echo "venv $v skipped: fails the launcher check"
        done
    fi
    [ -z "$VENV" ] && { echo "no working venv: run ./a30_setup_env.sh once"; exit 1; }
    if [ -f "$VENV/.module" ]; then module load "$(cat "$VENV/.module")" 2>/dev/null || true; fi
    source "$VENV/bin/activate"
    echo "venv: $(cd "$VENV" && pwd)"
    OUT=a30_out/${SLURM_JOB_ID}
    NGPU=$(nvidia-smi -L | wc -l)
    if [ "$NGPU" != "4" ]; then echo "THIS NODE HAS $NGPU GPUs, NOT 4: everything here is world 4."; exit 2; fi
else
    OUT=a30_out/v100smoke_${SLURM_JOB_ID}
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    if [ "${SLURM_NTASKS:-0}" != "4" ]; then echo "SMOKE NEEDS 4 NODES x 1 GPU (world 4)"; exit 2; fi
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TF_CPP_MIN_LOG_LEVEL=3
export OMP_NUM_THREADS=8
mkdir -p "$OUT"
SUM="$OUT/a30_summary.txt"
DEADLINE=$((22 * 3600 + 1800))
PORT=29800
log() { echo "$@" | tee -a "$SUM"; }

# ── 0. NODE ─────────────────────────────────────────────────────────────────
{
echo "=== 0. NODE ==="
echo "host: $(hostname)   job: $SLURM_JOB_ID   date: $(date -Is)   smoke: ${SMOKE:-no}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max,pci.bus_id --format=csv
echo "--- topology ---";          nvidia-smi topo -m
echo "--- NVLink ---";            nvidia-smi nvlink -s 2>&1 | head -20
echo "--- P2P read support ---";  nvidia-smi topo -p2p r
echo "--- NUMA / RAM ---";        (numactl -H 2>/dev/null || lscpu | grep -i numa) | head -12; free -g | head -2
} >> "$SUM" 2>&1
python3 - "$OUT/a30_node.json" <<'PY' 2>&1 | tee -a "$SUM"
import json, sys, torch, transformers
p = torch.cuda.get_device_properties(0)
d = {'device': torch.cuda.get_device_name(0), 'capability': list(torch.cuda.get_device_capability(0)),
     'total_mib': p.total_memory / 2**20, 'torch': torch.__version__, 'cuda': torch.version.cuda,
     'nccl': '.'.join(map(str, torch.cuda.nccl.version())), 'transformers': transformers.__version__,
     'flash_sdp_available': torch.backends.cuda.is_flash_attention_available()}
json.dump(d, open(sys.argv[1], 'w'), indent=1)
print(json.dumps(d))
PY

# ── K. CAP CHECK ────────────────────────────────────────────────────────────
CAP_OK=1
for CAPG in 6 9; do
    python3 - "$CAPG" <<'PY' 2>&1 | tee -a "$SUM" || CAP_OK=0
import sys, torch
cap = float(sys.argv[1]); total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(cap * 2**30 / total, 0)
keep = [torch.empty(256 * 2**20, dtype=torch.uint8, device='cuda') for _ in range(int((cap - 0.5) * 4))]
try:
    keep.append(torch.empty(2**30, dtype=torch.uint8, device='cuda'))
    print(f"K CAP CHECK {cap:g} GiB FAILED: allocated past the cap"); sys.exit(1)
except torch.cuda.OutOfMemoryError:
    print(f"K CAP CHECK {cap:g} GiB ok: {cap - 0.5:g} GiB allocated, +1 GiB refused")
PY
done
[ "$CAP_OK" = "1" ] || log "K FAILED: the cap does not bound the allocator; blocks C and N are void and skipped."

# ── arm runner ──────────────────────────────────────────────────────────────
# arm BLK MODE DT ATTN GQA L X CAP N REP [extra flags...]   -> sets ARM_OK
arm() {
    local BLK=$1 MODE=$2 DT=$3 ATTN=$4 GQA=$5 L=$6 X=$7 CAP=$8 N=$9 REP=${10}; shift 10
    ARM_OK=0
    if [ "$SECONDS" -gt "$DEADLINE" ]; then log "SKIP $BLK $MODE x=$X: past the 22.5 h guard"; return; fi
    local TAG="${BLK}_${MODE}_${DT}_${ATTN}$( [ "$GQA" = 1 ] && echo _gqa )_L${L}_x${X}$( [ "$CAP" != 0 ] && echo _cap${CAP} )_n${N}$( [ "$REP" != 0 ] && echo _r${REP} )"
    local D="$OUT/$TAG"; mkdir -p "$D"
    local F="--layer_checkpoint"
    case $MODE in
        offload)    F="$F --activation_offload" ;;
        staticcpu)  F="$F --cpu_offload" ;;
        offloadcpu) F="$F --activation_offload --cpu_offload" ;;
        evict)      F="$F --placement_engine --engine_live --hbm_budget_gb 14.9769 --prefetch_window_layers 0 --evict_layout_census --force_action evict" ;;
    esac
    [ "$GQA" = 1 ] && F="$F --native_gqa"
    [ "$CAP" != 0 ] && F="$F --hbm_cap_gb $CAP"
    local A=sdpa_mem_efficient; [ "$ATTN" = flash ] && A=sdpa_flash
    PORT=$((PORT + 1))
    local T0=$SECONDS
    if [ -z "$SMOKE" ]; then
        LAUNCH=(torchrun --standalone --nproc_per_node=4 --master_port="$PORT")
    else
        LAUNCH=(srun torchrun --nnodes=4 --nproc_per_node=1 --rdzv_id="${SLURM_JOB_ID}_$PORT"
                --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$PORT")
    fi
    "${LAUNCH[@]}" code/train_fsdp_profile.py \
        --hidden_size 768 --intermediate_size 2048 --num_layers "$L" --num_heads 12 --num_kv_heads 4 \
        --vocab_size 32000 --dtype "$DT" --attn_impl "$A" --seq_len $((X * 1024)) --batch_size 1 \
        --steps "$N" --chunked_ce 256 $F --fused_rmsnorm input --use_cache off --seed 1234 \
        --disable_hbm_drift --output_dir "$D" "$@" < /dev/null > "$D/stdout.txt" 2> "$D/stderr.txt"
    if [ -f "$D/steps_rank0.json" ]; then ARM_OK=1; fi
    log "ARM $TAG: $( [ $ARM_OK = 1 ] && echo done || echo 'DID NOT COMPLETE (read stderr.txt)' ) in $((SECONDS - T0)) s"
}
# ladder BLK MODE CAP X... : climbs, stops at the first failure
ladder() {
    local BLK=$1 MODE=$2 CAP=$3; shift 3
    for X in "$@"; do
        arm "$BLK" "$MODE" fp16 mem 0 64 "$X" "$CAP" 2 0
        [ "$ARM_OK" = 1 ] || { log "LADDER $BLK CLOSED at x=$X"; break; }
    done
}
reader() { python3 code/tools/a30_frontier.py "$@" "$OUT" 2>&1 | tee -a "$SUM"; }

if [ -z "$SMOKE" ]; then
    # ── B. R-A1 bytes ──
    arm B static  fp16 mem 0 32 24 0 3 0;  arm B static  fp16 mem 0 32 48 0 3 0
    arm B static  fp16 mem 0 64 24 0 3 0;  arm B static  fp16 mem 0 64 48 0 3 0
    arm B offload fp16 mem 0 64 24 0 3 0;  arm B offload fp16 mem 0 64 48 0 3 0
    arm B staticcpu  fp16 mem 0 64 48 0 3 0
    arm B offloadcpu fp16 mem 0 64 48 0 3 0
    arm B evict   fp16 mem 0 64 48 0 3 0
    arm B offload fp16 mem 0 64 100 0 3 0
    reader bytes
    # ── D. R-A2 bf16 ──
    arm D static  bf16 mem 0 64 24 0 3 0;  arm D static  bf16 mem 0 64 48 0 3 0
    arm D offload bf16 mem 0 64 48 0 3 0
    # ── F. R-A3 flash, native GQA ──
    arm F static  fp16 flash 0 64 24 0 3 0;  arm F static  fp16 flash 0 64 48 0 3 0
    arm F offload fp16 flash 0 64 48 0 3 0
    arm F static  bf16 flash 0 64 48 0 3 0
    arm F static  fp16 flash 1 64 24 0 3 0;  arm F static  fp16 flash 1 64 48 0 3 0
    arm F offload fp16 flash 1 64 48 0 3 0
    reader bytes
    # ── T. R-A5 time: calibrate on 16/32/64, predict, then 100/150 ──
    for XN in 16:20 32:10 64:6; do
        X=${XN%:*}; N=${XN#*:}
        arm T static  fp16 mem 0 64 $X 0 $N 1; arm T offload fp16 mem 0 64 $X 0 $N 1
        arm T offload fp16 mem 0 64 $X 0 $N 2; arm T static  fp16 mem 0 64 $X 0 $N 2
    done
    reader timecal --pred "$OUT/a30_pred_time.txt"
    for XN in 100:5 150:4; do
        X=${XN%:*}; N=${XN#*:}
        arm T static  fp16 mem 0 64 $X 0 $N 1; arm T offload fp16 mem 0 64 $X 0 $N 1
        arm T offload fp16 mem 0 64 $X 0 $N 2; arm T static  fp16 mem 0 64 $X 0 $N 2
    done
    # ── P. profiled arms ──
    arm P static  fp16 mem 0 64 64 0 3 0 --torch_profile
    arm P offload fp16 mem 0 64 64 0 3 0 --torch_profile
    # ── C. R-A4 capped ladders, then N. the native static ladder ──
    if [ "$CAP_OK" = 1 ]; then
        ladder CA static  6 31 32 33 34 35 36 37
        ladder CB static  9 54 55 56 57 58 59 60
        ladder CC offload 6 104 106 108 110 112 114 116
        ladder CD offload 9 184 186 188 190 192 194 196
        reader capcal --pred "$OUT/a30_pred_native.txt"
        RUNGS=$(grep '^RUNGS:' "$OUT/a30_pred_native.txt" 2>/dev/null | cut -d: -f2)
        if [ -n "$RUNGS" ]; then
            ladder N static 0 $RUNGS
            FIRST=$(echo $RUNGS | cut -d' ' -f1)
            if [ ! -f "$OUT/N_static_fp16_mem_L64_x${FIRST}_n2/steps_rank0.json" ]; then
                log "N: the first rung x=$FIRST failed -- the prediction was high; descending"
                for X in $((FIRST - 2)) $((FIRST - 4)) $((FIRST - 6)) $((FIRST - 8)); do
                    arm N static fp16 mem 0 64 "$X" 0 2 0; [ "$ARM_OK" = 1 ] && break
                done
            fi
        else log "N skipped: capcal wrote no rungs"; fi
    fi
else
    # ── V100 SMOKE (a30_v100_smoke.sbatch): every code path, small and fast ──
    arm B static  fp16 mem 0 32 24 0 3 0
    arm B static  fp16 mem 0 64 24 0 3 0;  arm B offload fp16 mem 0 64 24 0 3 0
    arm B staticcpu fp16 mem 0 64 24 0 3 0; arm B evict fp16 mem 0 64 24 0 3 0
    reader bytes
    arm D static bf16 mem 0 64 24 0 3 0                         # must refuse on sm_70
    arm F static fp16 flash 1 64 24 0 3 0                       # must refuse on sm_70
    for XN in 16:6 32:4 64:4; do
        X=${XN%:*}; N=${XN#*:}
        arm T static fp16 mem 0 64 $X 0 $N 1; arm T offload fp16 mem 0 64 $X 0 $N 1
    done
    reader timecal --pred "$OUT/a30_pred_time.txt"
    arm P offload fp16 mem 0 64 16 0 3 0 --torch_profile
    if [ "$CAP_OK" = 1 ]; then
        ladder CA static  6 33 34 35 36
        ladder CC offload 6 108 110 112 114
        reader capcal --pred "$OUT/a30_pred_native.txt"
    fi
fi

# ── profiled-arm analysis ──
for d in "$OUT"/P_*/; do
    [ -d "$d" ] || continue
    TR=$(ls -t "$d"torch_trace_rank0/*.pt.trace.json* 2>/dev/null | head -n 1)
    [ -n "$TR" ] || continue
    log "=== P: $(basename "$d") ==="
    python3 code/tools/attn_recompute_share.py "$TR" --layers 64 --steps 1 --json "${d}attn_share.json" 2>&1 | tail -15 | tee -a "$SUM"
    python3 code/tools/analyze_nccl_duty.py "$d"torch_trace_rank*/*.pt.trace.json* --json_out "${d}duty.json" 2>&1 | tail -25 | tee -a "$SUM"
done
reader read
log "=== DONE in $((SECONDS / 60)) min: ./a30_collect.sh $SLURM_JOB_ID ==="
