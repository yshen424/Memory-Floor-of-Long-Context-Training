#!/bin/bash -l
#SBATCH --job-name=l40s_followup
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=l40s_followup_%j.log
#SBATCH --error=l40s_followup_%j.err
#
# Submit from this directory (the directives above carry the same flags):
#   sbatch -A <account> -p <partition> --qos=<qos> --nodes=1 \
#          --exclusive --gres=gpu:l40s:4 --time=24:00:00 l40s_followup.sbatch
# GPU request is typed --gres only; never add --gpus-per-node (they combine).
# Venv: ~/l40s_venv only (../l40s_venv_setup/l40s_venv_setup.sh builds it once);
# override with L40S_VENV=<path>. The job stops before any arm if it fails the launcher check.

# ── JOB: THE L40S FOLLOW-UP to 630188. ONE EXCLUSIVE L40S NODE, WORLD 4. ─────
# Two things 630188 left open:
#  - the L40S time numbers (x*): its two-point calibration took a noisy x=64
#    pair and the registered tax missed by +40%;
#  - a PRE-REGISTERED capacity prediction at the 3B shape: its registered
#    brackets used nvidia-smi's card total (input error); the corrected model
#    was fitted after the fact.
#
#  T    TIME: static/offload pairs at x=32/64/100, FOUR repeats each, ABBA
#       order (S O O S S O O S), tier-A L=64. Registered from 630188's pooled
#       kappa; no in-job calibration.
#  R    3B static x=16 again, with an allocator snapshot: names the +0.148 MiB
#       that 630188 left unexplained.
#  RS, RO   3B native ladders again (static, stock offload): an IN-SAMPLE
#       REPLICATION of the model fitted on them (different node possible).
#  RSC, ROC 3B native ladders WITH FSDP CPU OFFLOAD (static+CPU, offload+CPU):
#       OUT OF SAMPLE, predicted from that model before the run.
#
#  All: fp16 sdpa_mem_efficient, whole-layer, chunked CE 256, fused-input,
#  use_cache off, b=1, seed 1234, --disable_hbm_drift. Tier-A: h 768, I 2048,
#  12/4 heads, v 32000. 3B: h 3072, I 8192, 24/8 heads, L=28, v 128256,
#  untied. WALL CLOCK ~7 h (T 1.3, R 0.05, RS 0.4, RO 1.1, RSC 0.8, ROC 2.5-3 h,
#  ~10 min NCCL watchdog per closing rung). No arm starts after 22.5 h.

# ── PRE-REGISTERED 2026-10-12. TIME NUMBERS ARE L40S-SPECIFIC (ClusterA L40S, ──
# ── one GPU per NUMA node, PCIe Gen4 x16, NCCL host-staged SHM; 621880).     ──
# FU-T1 OFFLOAD'S EXTRA TIME IS kappa*x, kappa = 0.0055 s/x (630188, pooled over
#       ten repeats at x=16..150): delta(32) 0.176 s, delta(64) 0.353 s,
#       delta(100) 0.552 s; tax with 630188's static steps (7.55 / 26.27 /
#       59.06 s) 2.34% / 1.34% / 0.93%. FALSIFIER: the mean of the four repeats
#       at any x outside +-30% of its delta; or kappa refit through the origin
#       on all twelve repeats outside [0.0044, 0.0066] (+-20%).
# FU-T2 RUN-TO-RUN SPREAD: the standard deviation of the four repeats at each x
#       <= 0.10 s (630188's pairs differed by up to 0.19 s at x >= 64). A larger
#       spread is recorded; it widens the crossings' interval, not the claim.
# FU-T3 CROSSINGS from the refit kappa and 630188's T(x) = 0.005547 x^2 +
#       0.05242 x + 0.1922 s: x*(2%) ~ 39, x*(1%) ~ 90 (the derived numbers;
#       a kappa inside FU-T1 keeps them within ~+-25%).
# FU-T4 static step times within 5% of 630188's at each x (same node class).
#
# Capacity model (harness PLATFORMS['l40s'].capacity, fitted POST HOC on
# 630188's six native brackets, both shapes): fits iff peak <= C0 - alpha * 16x,
# C0 45066.50 MiB, alpha 0.237 (feasible 0.122-0.353). 3B lines (MiB, x=tokens/1024):
#   static           13021.7862 + 312.617188x     offload       15805.9561 + 150.617188x
#   static + CPU      2702.7852 + 312.617188x     offload + CPU  5486.9551 + 150.617188x
#   (CPU offload lowers each by exactly 3P = 12 B x 3.607e9 / 4 = 10319.00 MiB.)
# FU-C1 IN-SAMPLE REPLICATION: RS [101, 102), RO [189, 190), as 630188. A different
#       bracket is run-to-run or node-to-node capacity variation: recorded with
#       the host name, not a test of the model.
# FU-C2 OUT OF SAMPLE, static + CPU: largest fit 133 (fails 134; 134 also inside
#       the model's feasible region, margin at 133 +281 MiB, at 134 -35 MiB).
#       Registered: largest fit in {133, 134}. FALSIFIER: outside 132-135 (+-1 rung).
# FU-C3 OUT OF SAMPLE, offload + CPU: largest fit 256 (feasible 255-257; margin
#       at 256 +51 MiB). FALSIFIER: outside 255-257 (+-1 rung).
# FU-C4 every closure a step-1 backward OOM on exactly 16x MiB (PyTorch prints
#       GiB to 2 decimals: allow +-0.005 GiB). Under CPU offload too.
# FU-C5 SPANS: offload/static 189/101 = 1.871 (as 630188); with CPU offload on
#       both, 256/133 = 1.925 (CPU offload is intercept-only, so the like-for-like
#       span moves toward the slope ratio 2.076).
# FU-R1 the snapshot arm reproduces 630188's 18311.8335 MiB (deterministic
#       allocator); the +0.148 MiB over the second-replay line is named offline
#       from cuda_snapshot_rank*.pkl (replayed on free_requested).
#
# HOST MEMORY: ROC pins the boundaries (~43 GiB/rank at x=256) and holds the
# CPU-offloaded state (~13 GiB/rank): ~230 GiB of the node's ~487 GiB available.
# FAILURE CLASSES (stderr.txt per arm): (0) node/socket failure -> re-run;
# (1) step-1 backward OOM on 16x MiB in a ladder -> the designed closure; (2) any
# OOM in a T or R arm -> a defect; (3) cudaHostAlloc / host OOM -> a host limit.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
# THE L40S VENV: ~/l40s_venv, built once by ../l40s_venv_setup/l40s_venv_setup.sh.
# No search and no fallback: the stale l64s_linktest/l64s_venv is never used. It
# must pass the launcher check before any arm runs (a venv whose torchrun cannot
# import kills every arm in 1 s).
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
OUT=l40s_followup_out/${SLURM_JOB_ID}
NGPU=$(nvidia-smi -L | wc -l)
if [ "$NGPU" != "4" ]; then echo "THIS NODE HAS $NGPU GPUs, NOT 4: everything here is world 4."; exit 2; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TF_CPP_MIN_LOG_LEVEL=3
export OMP_NUM_THREADS=8
mkdir -p "$OUT"
SUM="$OUT/l40s_summary.txt"
DEADLINE=$((22 * 3600 + 1800))
PORT=29950
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

# ── arm runner (as l40s_frontier.sbatch) ────────────────────────────────────
# arm BLK MODE L X CAP N REP [extra flags...]   -> sets ARM_OK
# Blocks R, RS, RO, RSC, ROC use the Llama-3.2-3B geometry; T is tier-A.
arm() {
    local BLK=$1 MODE=$2 L=$3 X=$4 CAP=$5 N=$6 REP=$7; shift 7
    ARM_OK=0
    if [ "$SECONDS" -gt "$DEADLINE" ]; then log "SKIP $BLK $MODE x=$X: past the 22.5 h guard"; return; fi
    local SHP="" SHAPE="--hidden_size 768 --intermediate_size 2048 --num_heads 12 --num_kv_heads 4 --vocab_size 32000"
    case $BLK in R|RS|RO|RSC|ROC)
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
    esac
    PORT=$((PORT + 1))
    local T0=$SECONDS
    torchrun --standalone --nproc_per_node=4 --master_port="$PORT" code/train_fsdp_profile.py $SHAPE --num_layers "$L" \
        --dtype fp16 --attn_impl sdpa_mem_efficient --seq_len $((X * 1024)) --batch_size 1 \
        --steps "$N" --chunked_ce 256 $F --fused_rmsnorm input --use_cache off --seed 1234 \
        --disable_hbm_drift --output_dir "$D" "$@" < /dev/null > "$D/stdout.txt" 2> "$D/stderr.txt"
    if [ -f "$D/steps_rank0.json" ]; then ARM_OK=1; fi
    log "ARM $TAG: $( [ $ARM_OK = 1 ] && echo done || echo 'DID NOT COMPLETE (read stderr.txt)' ) in $((SECONDS - T0)) s"
}
# ladder BLK MODE L X... : climbs, stops at the first failure; if the FIRST rung
# fails (prediction high) it descends one rung at a time, at most four.
ladder() {
    local BLK=$1 MODE=$2 L=$3; shift 3
    local FIRST=$1 n=0
    for X in "$@"; do
        arm "$BLK" "$MODE" "$L" "$X" 0 2 0
        n=$((n + 1))
        if [ "$ARM_OK" != 1 ]; then
            log "LADDER $BLK CLOSED at x=$X"
            if [ "$n" = 1 ]; then
                log "LADDER $BLK: the first rung failed -- the prediction was high; descending"
                for k in 1 2 3 4; do
                    arm "$BLK" "$MODE" "$L" $((FIRST - k)) 0 2 0
                    [ "$ARM_OK" = 1 ] && break
                done
            fi
            break
        fi
    done
}
reader() { python3 code/tools/remote_frontier.py "$@" --platform l40s "$OUT" 2>&1 | tee -a "$SUM"; }

# ── T. time: four repeats per x, ABBA ──
for XN in 32:10 64:6 100:5; do
    X=${XN%:*}; N=${XN#*:}
    arm T static  64 $X 0 $N 1; arm T offload 64 $X 0 $N 1
    arm T offload 64 $X 0 $N 2; arm T static  64 $X 0 $N 2
    arm T static  64 $X 0 $N 3; arm T offload 64 $X 0 $N 3
    arm T offload 64 $X 0 $N 4; arm T static  64 $X 0 $N 4
done
# ── R. the unexplained 3B point, with a snapshot ──
arm R static 28 16 0 3 0 --cuda_snapshot
reader bytes
# ── RS, RO replication; RSC, ROC out of sample ──
ladder RS  static     28 99 100 101 102 103
ladder RO  offload    28 187 188 189 190 191
ladder RSC staticcpu  28 131 132 133 134 135 136
ladder ROC offloadcpu 28 253 254 255 256 257 258 259
reader read
log "=== DONE in $((SECONDS / 60)) min: ./l40s_followup_collect.sh $SLURM_JOB_ID ==="
