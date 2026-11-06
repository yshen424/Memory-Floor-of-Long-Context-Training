#!/bin/bash -l
#SBATCH --job-name=l40s_trace
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=l40s_trace_%j.log
#SBATCH --error=l40s_trace_%j.err
#
# Submit from this directory (the directives above carry the same flags):
#   sbatch -A <account> -p <partition> --qos=<qos> --nodes=1 \
#          --exclusive --gres=gpu:l40s:4 --time=24:00:00 l40s_trace.sbatch
# GPU request is typed --gres only; never add --gpus-per-node (they combine).
# Venv: ~/l40s_venv only (../l40s_venv_setup/l40s_venv_setup.sh), launcher-checked.

# ── JOB: THE L40S TRACE. ONE EXCLUSIVE L40S NODE, WORLD 4, ~3 h ──────────────
# Question: why is stock offload's extra time at x=64 a third below kappa*x,
# on two nodes (630188, 634078), when x=32 and x=100 sit on it? Tier-A L=64
# throughout (the A30 sees a related failure of the linear model; this job is
# the L40S half).
#
#  U  UNTRACED static/offload pairs, four repeats each in ABBA order, at
#     x = 48, 56, 64, 72, 80: maps the dip's shape between the measured
#     32 and 100, and measures x=64 on a third node.
#  P  TRACED pairs (--torch_profile, one profiled step) at x = 32, 48, 56,
#     64, 72, 80, 100. tools/copy_exposure.py splits each traced step into
#     compute C, copy-only X, collective/other N and idle I (span = C+X+
#     N+I exactly), so offload's extra time decomposes term by term.
#
#  All: h 768, I 2048, 12/4 heads, v 32000, L=64, fp16 sdpa_mem_efficient,
#  whole-layer, chunked CE 256, fused-input, use_cache off, b=1, seed 1234,
#  --disable_hbm_drift. Blocks named T (untraced) and P (traced) so the
#  shared reader pools the timing exactly as in 630188 / 634078.

# ── PRE-REGISTERED 2026-10-13. TIME NUMBERS ARE L40S-SPECIFIC (ClusterA L40S, ──
# ── one GPU per NUMA node, PCIe Gen4 x16, NCCL host-staged SHM; 621880).     ──
# Measured so far (extra time of offload over static, mean of repeats):
#   x=32 0.204 s, x=64 0.232 / 0.256 s, x=100 0.652 / 0.574 s (634078 / 630188);
#   kappa through the origin 0.0057 s/x.
# TR-1 THE SHAPE OF THE DIP (untraced). Linear model kappa*x, kappa = 0.0057:
#      x=48 0.274 s, x=56 0.319, x=64 0.365, x=72 0.410, x=80 0.456. Lean: x=64
#      again ~0.23-0.26 s (third node); 48-80 below the line, so the deviation
#      is a band, not a point. Recorded per x with its standard error; a mean
#      within +-30% of kappa*x at every x would say the dip was local to 64.
# TR-2 COPIES DO NOT OVERLAP COMPUTE (traced): copy time hidden behind compute
#      <= 5% of copy time at every x, as on V100 (7758: 0.0 ms of 2161 ms; stock
#      offload copies run in order on the compute stream). FALSIFIER: > 20%
#      hidden at any x -- then the copies run on their own stream on this build
#      and the premise of TR-3 changes.
# TR-3 THE MECHANISM: offload's copies fill GPU time the static step leaves idle.
#      Span difference offload - static = dC + dX + dN + dI (identity). Lean:
#      dX ~ copy busy ~ 7.7 ms/x, and the dip is dI: at x=64, dI ~ -(copy busy -
#      0.23 s) ~ -0.26 s; at x=32 and 100 dI small against dX. FALSIFIER: dI
#      within +-0.05 s of 0 at x=64 -- the dip then sits in dC or dN, and those
#      terms name it.
# TR-4 COPY BUSY per traced step = 63 boundaries x 1.5 MiB x x x 2 directions at
#      ~26.3 / 25.1 GB/s = 7.7 ms/x, +-15%.
# TR-5 PROFILING OVERHEAD: the traced span difference within +-50% of the
#      untraced mean at the same x (a profiler perturbs timing; the
#      decomposition is read for its terms, the untraced repeats for the size).
# FAILURE CLASSES: a GPU that will not open stops the job at the PREFLIGHT (exit 3,
# host named). Otherwise none expected (every arm is far below capacity: x <= 100 at
# L=64 peaks at 14.7 GiB static on a 44.4 GiB card). Any OOM is a defect.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
# THE L40S VENV: ~/l40s_venv, launcher-checked before any arm.
VENV=${L40S_VENV:-$HOME/l40s_venv}
venv_ok() (
    [ -f "$1/bin/activate" ] || exit 1
    if [ -f "$1/.module" ]; then module load "$(cat "$1/.module")" 2>/dev/null || true; fi
    source "$1/bin/activate" &&
    python3 -c "import torch.distributed.run, transformers" >/dev/null 2>&1 &&
    torchrun --help >/dev/null 2>&1
)
venv_ok "$VENV" || { echo "VENV $VENV MISSING OR FAILS the launcher check (torchrun / torch.distributed.run import): build it with ../l40s_venv_setup/l40s_venv_setup.sh"; exit 1; }
if [ -f "$VENV/.module" ]; then module load "$(cat "$VENV/.module")" 2>/dev/null || true; fi
source "$VENV/bin/activate"
echo "venv: $(cd "$VENV" && pwd)"
OUT=l40s_trace_out/${SLURM_JOB_ID}
NGPU=$(nvidia-smi -L | wc -l)
if [ "$NGPU" != "4" ]; then echo "THIS NODE HAS $NGPU GPUs, NOT 4: everything here is world 4."; exit 2; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TF_CPP_MIN_LOG_LEVEL=3
export OMP_NUM_THREADS=8
mkdir -p "$OUT"
SUM="$OUT/l40s_summary.txt"
DEADLINE=$((22 * 3600 + 1800))
PORT=30050
log() { echo "$@" | tee -a "$SUM"; }

# ── 0. NODE ─────────────────────────────────────────────────────────────────
{
echo "=== 0. NODE ==="
echo "host: $(hostname)   job: $SLURM_JOB_ID   date: $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version,pci.bus_id --format=csv
echo "--- topology ---"; nvidia-smi topo -m
echo "--- NUMA / RAM ---"; (numactl -H 2>/dev/null || lscpu | grep -i numa) | head -12; free -g | head -2
} >> "$SUM" 2>&1
python3 - "$OUT/l40s_node.json" <<'PY' 2>&1 | tee -a "$SUM"
import json, sys, torch, transformers
p = torch.cuda.get_device_properties(0)
d = {'device': torch.cuda.get_device_name(0), 'capability': list(torch.cuda.get_device_capability(0)),
     'total_mib': p.total_memory / 2**20, 'torch': torch.__version__, 'cuda': torch.version.cuda,
     'nccl': '.'.join(map(str, torch.cuda.nccl.version())), 'transformers': transformers.__version__}
json.dump(d, open(sys.argv[1], 'w'), indent=1)
print(json.dumps(d))
PY

# ── PREFLIGHT: every GPU must open and allocate before any arm runs ────────────
# A GPU that refuses set_device ("CUDA-capable device(s) is/are busy or
# unavailable") is a node fault, not a result.
python3 - <<'PY2' 2>&1 | tee -a "$SUM"
import sys, torch
bad = []
for i in range(torch.cuda.device_count()):
    try:
        torch.cuda.set_device(i); t = torch.empty(256 * 2**20, dtype=torch.uint8, device=f'cuda:{i}'); del t
    except Exception as e:
        bad.append((i, repr(e)[:160]))
for i, e in bad:
    print(f"PREFLIGHT GPU {i} FAILED: {e}")
print("PREFLIGHT ok: all GPUs open and allocate" if not bad else "PREFLIGHT FAILED")
sys.exit(1 if bad else 0)
PY2
if [ "${PIPESTATUS[0]}" != "0" ]; then
    { nvidia-smi --query-gpu=index,pci.bus_id,compute_mode,memory.used --format=csv
      nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv; } 2>&1 | tee -a "$SUM"
    log "NODE FAULT on $(hostname): a GPU is unusable. Not a result. Resubmit with --exclude=$(hostname -s)"
    exit 3
fi

# arm BLK MODE X N REP [extra flags...]   (tier-A, L=64)   -> sets ARM_OK
arm() {
    local BLK=$1 MODE=$2 X=$3 N=$4 REP=$5; shift 5
    ARM_OK=0
    if [ "$SECONDS" -gt "$DEADLINE" ]; then log "SKIP $BLK $MODE x=$X: past the 22.5 h guard"; return; fi
    local TAG="${BLK}_${MODE}_fp16_mem_L64_x${X}_n${N}$( [ "$REP" != 0 ] && echo _r${REP} )"
    local D="$OUT/$TAG"; mkdir -p "$D"
    local F="--layer_checkpoint"; [ "$MODE" = offload ] && F="$F --activation_offload"
    PORT=$((PORT + 1))
    local T0=$SECONDS
    torchrun --standalone --nproc_per_node=4 --master_port="$PORT" code/train_fsdp_profile.py \
        --hidden_size 768 --intermediate_size 2048 --num_heads 12 --num_kv_heads 4 --vocab_size 32000 \
        --num_layers 64 --dtype fp16 --attn_impl sdpa_mem_efficient --seq_len $((X * 1024)) --batch_size 1 \
        --steps "$N" --chunked_ce 256 $F --fused_rmsnorm input --use_cache off --seed 1234 \
        --disable_hbm_drift --output_dir "$D" "$@" < /dev/null > "$D/stdout.txt" 2> "$D/stderr.txt"
    if [ -f "$D/steps_rank0.json" ]; then ARM_OK=1; fi
    log "ARM $TAG: $( [ $ARM_OK = 1 ] && echo done || echo 'DID NOT COMPLETE (read stderr.txt)' ) in $((SECONDS - T0)) s"
}

# ── U. untraced pairs, four repeats, ABBA (block T for the shared reader) ──
for XN in 48:8 56:7 64:6 72:6 80:5; do
    X=${XN%:*}; N=${XN#*:}
    arm T static  $X $N 1; arm T offload $X $N 1
    arm T offload $X $N 2; arm T static  $X $N 2
    arm T static  $X $N 3; arm T offload $X $N 3
    arm T offload $X $N 4; arm T static  $X $N 4
done
# ── P. traced pairs, then the decomposition per x ──
for X in 32 48 56 64 72 80 100; do
    arm P static  $X 3 0 --torch_profile
    arm P offload $X 3 0 --torch_profile
    S=$(ls "$OUT"/P_static_fp16_mem_L64_x${X}_n3/torch_trace_rank*/*.pt.trace.json* 2>/dev/null)
    O=$(ls "$OUT"/P_offload_fp16_mem_L64_x${X}_n3/torch_trace_rank*/*.pt.trace.json* 2>/dev/null)
    if [ -n "$S" ] && [ -n "$O" ]; then
        log "=== P x=$X: decomposition (median over ranks) ==="
        python3 code/tools/copy_exposure.py --offload $O --static $S --json "$OUT/decomp_x${X}.json" 2>&1 | tee -a "$SUM"
    else log "P x=$X: traces missing, no decomposition"; fi
done
python3 code/tools/remote_frontier.py read --platform l40s "$OUT" 2>&1 | sed -n '/=== TIME/,/^===/p' | tee -a "$SUM"
log "=== DONE in $((SECONDS / 60)) min: ./l40s_trace_collect.sh $SLURM_JOB_ID ==="
