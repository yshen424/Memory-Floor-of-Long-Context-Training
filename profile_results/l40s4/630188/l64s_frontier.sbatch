#!/bin/bash -l
#SBATCH --job-name=l64s_frontier
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=l64s_frontier_%j.log
#SBATCH --error=l64s_frontier_%j.err
#
# Submit from this directory (the directives above carry the same flags):
#   sbatch -A <account> -p <partition> --qos=<qos> --nodes=1 \
#          --exclusive --gres=gpu:l40s:4 --time=24:00:00 l64s_frontier.sbatch
# GPU request is typed --gres only; never add --gpus-per-node (they combine).
# Venv: reuses ../l64s_linktest/l64s_venv (621880 ran on it) or ../modern_gpu_probe/venv.

# ── JOB: BYTES, TIME AND CAPACITY ON L40S. ONE EXCLUSIVE L40S NODE ───────────
# ── (Ada sm89, 48 GB, PCIe-only), WORLD 4. Hypotheses R-L1..R-L3; also the ──
# ── first out-of-sample test of the shape sweep's (7761) post hoc terms.   ──
#
# R-L1 a third architecture (Volta, Ampere, Ada); R-L2 a PCIe-only node whose
# NCCL is host-staged (621880: every GPU pair crosses sockets, NCCL picks SHM),
# so offload's copies share the host link with collectives; R-L3 a
# Llama-3.2-3B shape that needs ~17 GB of sharded optimizer state per rank (no
# fit on 16 or 24 GB). The bf16 and FlashAttention blocks of the A30 job
# (D, F) are NOT here: kernel questions the A30 answers.
#
#  0  NODE facts: GPUs, topology, P2P, NUMA, host RAM, software, card
#     total -> l64s_node.json.
#  K  CAP CHECK at 6 and 9 GiB. If it fails, C and N are void and skipped.
#  B  R-L1 BYTES (tier-A, fp16 mem-efficient, 3 steps): L=32 static
#     x=24/48; L=64 static, offload x=24/48; static+CPU, offload+CPU,
#     force-evict x=48; offload x=100.
#  T  R-L1/R-L2 TIME: static/offload pairs, two repeats in ABBA order, at
#     x=16/32/64 -> IN-JOB CALIBRATION writes the tax and crossing
#     predictions -> x=100/150 run against them. P: profiled static and
#     offload at x=64 (attention-replay share, copy bandwidth, NCCL duty).
#  R  R-L3 LLAMA-3.2-3B SLOPE POINTS (h 3072, I 8192, 24/8 heads, L=28,
#     v=128256, untied): static x=16/32/48/96, offload x=16/48/96, 3 steps.
#  RS, RO  R-L3 NATIVE 48 GB BRACKETS, 2 steps, no cap: static x=100..106,
#     stock offload x=189..198; each climbs to its first failure.
#  C  CAPPED LADDERS (tier-A, V100's four, 7760): the Ada reserve.
#  N  NATIVE 48 GB TIER-A STATIC LADDER: rungs written in-job from C.
#
#  All tier-A arms: L=64 unless named, hidden 768, inter 2048, heads 12/4,
#  v=32000. Every arm: chunked CE 256, whole-layer, fused-input, use_cache
#  off, b=1, seed 1234, fp16 sdpa_mem_efficient, --disable_hbm_drift.
#  WALL CLOCK expected 5-9 h (B 0.2, T 1.5-2, P 0.1, R 0.5, RS 0.5,
#  RO 1-1.5, C 1, N 2-2.5 h, plus ~10 min of NCCL watchdog per closing
#  rung). No arm starts after 22.5 h; 24 h requested.

# ── PRE-REGISTERED 2026-10-10 from harness/ (platform l40s: V100 workspace  ──
# ── 16.2559 MiB carried, which 621880 reproduced to 0.001 MiB at L=32 x=8;  ──
# ── MiB, x = tokens/1024). Capacity and time statements beyond these are   ──
# ── written by the job itself, each before the arms it predicts.           ──
# Tier-A lines (identical to the A30 job's):
#   L=32 static          900.6560 + 84.320312x      x=24: 2924.34   x=48: 4948.03
#   L=64 static         1476.7966 + 132.320312x     x=24: 4652.48   x=48: 7828.17
#   L=64 stock offload  1866.8919 + 37.820312x      x=24: 2774.58   x=48: 3682.27  x=100: 5648.92
#   L=64 force-evict    on the offload line;  static+CPU x=48: 6535.26;  offload+CPU x=48: 2389.36
# Llama-3.2-3B lines (p_layer 100,669,440; P 3.607 B):
#   static, first replay   13021.7862 + 312.617188x
#   static, second replay  13405.8096 + 306.617188x   (binds below x = 2 p_layer / h = 64.0)
#   stock offload          15805.9561 + 150.617188x
#   optimizer step         17214.5909 (+ 0.0078x)      (binds only below x ~ 8)
#   backward end (root)    17532.4  (7764 smoke, POST HOC from one point at L=4:
#                          12 B P/W + 4 B L p_layer/W + 6 B p_root + workspace;
#                          binds below x ~ 13.5 static, ~ 11.5 offload; no R arm in reach)
#   points: static x=16 18311.69  x=32 23217.56  x=48 28123.44 (second replay)  x=96 43033.04 (first)
#           offload x=16 18215.83  x=48 23035.58  x=96 30265.21
#
# L-P1 BYTES CARRY TO ADA (the shared reader prints it as A-P1/A-P2): every B arm
#      on its line plus ONE common offset, all
#      within 0.05 MiB of each other; lean offset 0.00 (621880). FALSIFIER: two
#      B arms whose offsets differ by more.
# L-P2 force-evict x=48 on the offload line; CPU offload lowers each line by 3P.
# R-P1 OUT-OF-SAMPLE (the shape-sweep terms' first test): every R arm on its 3B line plus
#      the B common offset, within 0.05 MiB -- including the three static points
#      in the SECOND-REPLAY class and the shape term (2 B p_layer + 2 B h I =
#      240.0 MiB here, against 15.0 at tier-A). FALSIFIER: a residual > 0.05 MiB;
#      a residual equal on static and offload points the shape term; one on
#      static x<=48 only the second-replay term.
# R-P2 NATIVE BRACKETS (V100 reserve delta 33.1, alpha 0.163 carried; card total
#      46068 MiB from 621880; context estimated 400-1000 MiB, measured in-job):
#      static largest fit 103 (fails 104; range 101-103 over context and alpha),
#      margin at 103 ~51 MiB; stock offload largest fit 194 (fails 195; range
#      190-195), margin at 194 ~9 MiB, so 193 is equally likely. Ladder span
#      194/103 = 1.883 (slope ratio 2.076). Each failure: step-1 backward OOM on
#      exactly 16x MiB (2 I bytes per token). FALSIFIER: 2+ rungs off.
#      The reader re-states the prediction with the measured context.
# T-P0 (lean, from 621880 + byte arithmetic, context for the in-job file): offload
#      moves 63 boundaries x 1.5 MiB x 2 ways = 189 MiB per x per step; at ~21 GB/s
#      and 621880's ~80% exposure, kappa ~ 0.0075 s/x, in [0.005, 0.012]. This is
#      a PCIe-only, host-staged-NCCL number, not a hardware-free constant.
# T-P1 delta(64)/delta(32) within 2.0 +- 0.6; static T(64)/T(32) in [2.5, 4.0].
# T-P2 the two repeats of each delta agree within 0.1 s or 15%.
# T-P3/T-P4 written to l64s_pred_time.txt BEFORE x=100/150 run. FALSIFIER: the
#      measured tax at x=100 or 150 outside +-30% of the file.
# C-P1 capped brackets = V100's (7760): CA 34, CB 57, CC 110, CD 190 (+-1 rung
#      for CC, CD). C-P2 one (alpha, delta) fits all four (capcal prints it).
# N-P1/N-P2 written to l64s_pred_native.txt BEFORE the native tier-A ladder.
#
# FAILURE CLASSES (read stderr.txt in the arm's directory): (0) node/socket
# failure -> not a result, re-run; (1) step-1 backward OOM on 4x MiB (tier-A) or
# 16x MiB (3B) in a C, N, RS or RO rung -> the designed closure; (2) any OOM in a
# B, T or R arm -> these sit far under capacity: a defect, record it; (3)
# cudaHostAlloc failure -> a host limit, not capacity (RO pins ~33 GiB/rank).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
SMOKE=${L64S_SMOKE:-}
if [ -z "$SMOKE" ]; then
    VENV=${L64S_VENV:-}
    [ -z "$VENV" ] && for v in l64s_venv ../l64s_linktest/l64s_venv ../modern_gpu_probe/venv ../a30_frontier/a30_venv; do
        [ -f "$v/bin/activate" ] && { VENV=$v; break; }; done
    [ -z "$VENV" ] && { echo "no venv: run ./l64s_setup_env.sh once"; exit 1; }
    if [ -f "$VENV/.module" ]; then module load "$(cat "$VENV/.module")" 2>/dev/null || true; fi
    source "$VENV/bin/activate"
    echo "venv: $(cd "$VENV" && pwd)"
    OUT=l64s_out/${SLURM_JOB_ID}
    NGPU=$(nvidia-smi -L | wc -l)
    if [ "$NGPU" != "4" ]; then echo "THIS NODE HAS $NGPU GPUs, NOT 4: everything here is world 4."; exit 2; fi
else
    OUT=l64s_out/v100smoke_${SLURM_JOB_ID}
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    if [ "${SLURM_NTASKS:-0}" != "4" ]; then echo "SMOKE NEEDS 4 NODES x 1 GPU (world 4)"; exit 2; fi
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TF_CPP_MIN_LOG_LEVEL=3
export OMP_NUM_THREADS=8
mkdir -p "$OUT"
SUM="$OUT/l64s_summary.txt"
DEADLINE=$((22 * 3600 + 1800))
PORT=29900
log() { echo "$@" | tee -a "$SUM"; }

# ── 0. NODE ─────────────────────────────────────────────────────────────────
{
echo "=== 0. NODE ==="
echo "host: $(hostname)   job: $SLURM_JOB_ID   date: $(date -Is)   smoke: ${SMOKE:-no}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max,pci.bus_id --format=csv
echo "--- topology ---";          nvidia-smi topo -m
echo "--- P2P read support ---";  nvidia-smi topo -p2p r
echo "--- NUMA / RAM ---";        (numactl -H 2>/dev/null || lscpu | grep -i numa) | head -12; free -g | head -2
} >> "$SUM" 2>&1
python3 - "$OUT/l64s_node.json" <<'PY' 2>&1 | tee -a "$SUM"
import json, sys, torch, transformers
p = torch.cuda.get_device_properties(0)
d = {'device': torch.cuda.get_device_name(0), 'capability': list(torch.cuda.get_device_capability(0)),
     'total_mib': p.total_memory / 2**20, 'torch': torch.__version__, 'cuda': torch.version.cuda,
     'nccl': '.'.join(map(str, torch.cuda.nccl.version())), 'transformers': transformers.__version__}
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
# arm BLK MODE L X CAP N REP [extra flags...]   -> sets ARM_OK
# fp16 + memory-efficient attention throughout. Blocks R, RS, RO use the
# Llama-3.2-3B geometry (tag _llama3b); every other block is tier-A.
arm() {
    local BLK=$1 MODE=$2 L=$3 X=$4 CAP=$5 N=$6 REP=$7; shift 7
    ARM_OK=0
    if [ "$SECONDS" -gt "$DEADLINE" ]; then log "SKIP $BLK $MODE x=$X: past the 22.5 h guard"; return; fi
    local SHP="" SHAPE="--hidden_size 768 --intermediate_size 2048 --num_heads 12 --num_kv_heads 4 --vocab_size 32000"
    case $BLK in R|RS|RO)
        SHP="_llama3b"
        SHAPE="--hidden_size 3072 --intermediate_size 8192 --num_heads 24 --num_kv_heads 8 --vocab_size 128256" ;;
    esac
    local TAG="${BLK}_${MODE}_fp16_mem${SHP}_L${L}_x${X}$( [ "$CAP" != 0 ] && echo _cap${CAP} )_n${N}$( [ "$REP" != 0 ] && echo _r${REP} )"
    local D="$OUT/$TAG"; mkdir -p "$D"
    local F="--layer_checkpoint"
    case $MODE in
        offload)    F="$F --activation_offload" ;;
        staticcpu)  F="$F --cpu_offload" ;;
        offloadcpu) F="$F --activation_offload --cpu_offload" ;;
        evict)      F="$F --placement_engine --engine_live --hbm_budget_gb 14.9769 --prefetch_window_layers 0 --evict_layout_census --force_action evict" ;;
    esac
    [ "$CAP" != 0 ] && F="$F --hbm_cap_gb $CAP"
    PORT=$((PORT + 1))
    local T0=$SECONDS
    if [ -z "$SMOKE" ]; then
        LAUNCH=(torchrun --standalone --nproc_per_node=4 --master_port="$PORT")
    else
        LAUNCH=(srun torchrun --nnodes=4 --nproc_per_node=1 --rdzv_id="${SLURM_JOB_ID}_$PORT"
                --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$PORT")
    fi
    "${LAUNCH[@]}" code/train_fsdp_profile.py $SHAPE --num_layers "$L" \
        --dtype fp16 --attn_impl sdpa_mem_efficient --seq_len $((X * 1024)) --batch_size 1 \
        --steps "$N" --chunked_ce 256 $F --fused_rmsnorm input --use_cache off --seed 1234 \
        --disable_hbm_drift --output_dir "$D" "$@" < /dev/null > "$D/stdout.txt" 2> "$D/stderr.txt"
    if [ -f "$D/steps_rank0.json" ]; then ARM_OK=1; fi
    log "ARM $TAG: $( [ $ARM_OK = 1 ] && echo done || echo 'DID NOT COMPLETE (read stderr.txt)' ) in $((SECONDS - T0)) s"
}
# ladder BLK MODE CAP L X... : climbs, stops at the first failure; if the FIRST
# rung fails (prediction high) it descends one rung at a time, at most four.
ladder() {
    local BLK=$1 MODE=$2 CAP=$3 L=$4; shift 4
    local FIRST=$1 n=0
    for X in "$@"; do
        arm "$BLK" "$MODE" "$L" "$X" "$CAP" 2 0
        n=$((n + 1))
        if [ "$ARM_OK" != 1 ]; then
            log "LADDER $BLK CLOSED at x=$X"
            if [ "$n" = 1 ]; then
                log "LADDER $BLK: the first rung failed -- the prediction was high; descending"
                local STEP=$(( ${2:-$((FIRST + 1))} - FIRST ))
                for k in 1 2 3 4; do
                    arm "$BLK" "$MODE" "$L" $((FIRST - k * STEP)) "$CAP" 2 0
                    [ "$ARM_OK" = 1 ] && break
                done
            fi
            break
        fi
    done
}
reader() { python3 code/tools/remote_frontier.py "$@" --platform l40s "$OUT" 2>&1 | tee -a "$SUM"; }

if [ -z "$SMOKE" ]; then
    # ── B. R-L1 bytes ──
    arm B static  32 24 0 3 0;  arm B static  32 48 0 3 0
    arm B static  64 24 0 3 0;  arm B static  64 48 0 3 0
    arm B offload 64 24 0 3 0;  arm B offload 64 48 0 3 0
    arm B staticcpu  64 48 0 3 0
    arm B offloadcpu 64 48 0 3 0
    arm B evict   64 48 0 3 0
    arm B offload 64 100 0 3 0
    reader bytes
    # ── T. R-L1/R-L2 time: calibrate on 16/32/64, predict, then 100/150 ──
    for XN in 16:20 32:10 64:6; do
        X=${XN%:*}; N=${XN#*:}
        arm T static  64 $X 0 $N 1; arm T offload 64 $X 0 $N 1
        arm T offload 64 $X 0 $N 2; arm T static  64 $X 0 $N 2
    done
    reader timecal --pred "$OUT/l64s_pred_time.txt"
    for XN in 100:5 150:4; do
        X=${XN%:*}; N=${XN#*:}
        arm T static  64 $X 0 $N 1; arm T offload 64 $X 0 $N 1
        arm T offload 64 $X 0 $N 2; arm T static  64 $X 0 $N 2
    done
    # ── P. profiled arms ──
    arm P static  64 64 0 3 0 --torch_profile
    arm P offload 64 64 0 3 0 --torch_profile
    # ── R. R-L3 Llama-3.2-3B slope points, then its native brackets ──
    for X in 16 32 48 96; do arm R static 28 $X 0 3 0; done
    for X in 16 48 96; do arm R offload 28 $X 0 3 0; done
    reader bytes
    ladder RS static  0 28 100 101 102 103 104 105 106
    ladder RO offload 0 28 189 190 191 192 193 194 195 196 197 198
    # ── C. capped ladders, then N. the native tier-A static ladder ──
    if [ "$CAP_OK" = 1 ]; then
        ladder CA static  6 64 31 32 33 34 35 36 37
        ladder CB static  9 64 54 55 56 57 58 59 60
        ladder CC offload 6 64 104 106 108 110 112 114 116
        ladder CD offload 9 64 184 186 188 190 192 194 196
        reader capcal --pred "$OUT/l64s_pred_native.txt"
        RUNGS=$(grep '^RUNGS:' "$OUT/l64s_pred_native.txt" 2>/dev/null | cut -d: -f2)
        if [ -n "$RUNGS" ]; then ladder N static 0 64 $RUNGS
        else log "N skipped: capcal wrote no rungs"; fi
    fi
else
    # ── V100 SMOKE (l64s_v100_smoke.sbatch): every code path, small and fast ──
    arm B static  32 24 0 3 0
    arm B static  64 24 0 3 0;  arm B offload 64 24 0 3 0
    arm B staticcpu 64 24 0 3 0; arm B evict 64 24 0 3 0
    reader bytes
    for XN in 16:6 32:4 64:4; do
        X=${XN%:*}; N=${XN#*:}
        arm T static 64 $X 0 $N 1; arm T offload 64 $X 0 $N 1
    done
    reader timecal --pred "$OUT/l64s_pred_time.txt"
    arm P offload 64 16 0 3 0 --torch_profile
    # the 3B geometry at L=4 (fits a V100): slope arms, then a short native ladder
    for X in 8 16; do arm R static 4 $X 0 3 0; arm R offload 4 $X 0 3 0; done
    reader bytes
    ladder RS static 0 4 53 54 55 56 57 58
    if [ "$CAP_OK" = 1 ]; then
        ladder CA static  6 64 33 34 35 36
        reader capcal --pred "$OUT/l64s_pred_native.txt"
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
log "=== DONE in $((SECONDS / 60)) min: ./l64s_collect.sh $SLURM_JOB_ID ==="
