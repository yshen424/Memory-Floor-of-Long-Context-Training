#!/bin/bash -l
#SBATCH --job-name=a100_regime
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --constraint=dgx-A100
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=a100_regime_%j.log
#SBATCH --error=a100_regime_%j.err
#
# Submit from this directory. The GPU request is ON THE COMMAND LINE ONLY (typed GRES:
# gpu:a100:8 per node); never add --gpus-per-node:
#   sbatch -A <account> -p <partition> --qos=<qos> --nodes=1 \
#          --constraint=dgx-A100 --exclusive --gres=gpu:a100:4 --time=24:00:00 a100_regime.sbatch
# --constraint=dgx-A100 is REQUIRED: <partition> also holds 4 hpe-A100 nodes (nodeB-gpu001-004)
# whose GPU fabric is not known to be NVSwitch; the job checks the topology itself (exit 4).
# Venv: ~/a100_venv only (../a100_venv_setup/a100_venv_setup.sh; A100_VENV overrides),
# launcher-checked before any arm.

# ── JOB: OFFLOAD'S TIME TAX ON A100 (R-H1 time, R-H2 bytes). ONE EXCLUSIVE DGX A100
# NODE, 4 OF ITS 8 GPUs, WORLD 4. On an NVSwitch node collectives ride NVLink and only
# offload's copies cross the host link (the A30 nodes are PCIe with NVLink pairs, so
# collectives between pairs share the host link).
#
#  0  NODE facts, GPU PREFLIGHT (exit 3), NVSWITCH CHECK (every visible pair NV#, else
#     exit 4: not the R-H1 point), host-copy bandwidth solo and all-4-concurrent (2 GPUs
#     share each PCIe switch on a DGX).
#  B  R-H2 BYTES (3 steps): L=32 static x=24; L=64 static, offload x=24/48
#  T  R-H1 TIME: static/offload pairs, ABBA, 4 repeats at x=16/32/48/64 -> IN-JOB
#     CALIBRATION writes a100_pred_time.txt -> 2 repeats at x=100/150. Every arm logs
#     GPU clocks/power at 1 Hz.
#  H  THE NVSWITCH CONTRAST: the same pairs with NCCL's GPU-to-GPU paths disabled
#     (collectives through host memory, as on PCIe-only nodes), 2 repeats at x=32/100,
#     same node, same GPUs.
#  P  profiled pairs at x=64 and 150: attention-replay share, NCCL duty, and
#     tools/copy_exposure.py (compute / copy-only / other / idle).
#  G  matched footprint: offload + an idle ballast equal to static's extra bytes at
#     x=100 and 150 (tests the A30's footprint effect).
#
#  All: h 768, I 2048, heads 12/4, v 32000, L=64 unless named, fp16 sdpa_mem_efficient
#  (regime B), whole-layer, chunked CE 256, fused-input, use_cache off, b=1, seed 1234,
#  --disable_hbm_drift, CPU offload OFF. No capacity arms.
#  WALL CLOCK: ~8.6 h if the A100 were only as fast as the A30 (B 0.2, T 5.5, H 1.3,
#  P 0.8, G 0.7 h); expected 4-6 h. No arm after 22 h.

# ── PRE-REGISTERED 2026-10-14 (harness/ platform a100 = V100 workspace carried; ──
# ── MiB, x = tokens/1024; delta = offload - static, mean of steps 2+).          ──
# TIME NUMBERS ARE FOR THIS SETUP ONLY: DGX A100, 4 GPUs of 8, NVSwitch, world 4.
# Lines (derived; the A30 carried them to 0.002 MiB):
#   L=32 static          900.6560 + 84.320312x      x=24: 2924.34
#   L=64 static         1476.7966 + 132.320312x     x=24: 4652.48  x=48: 7828.17  x=150: 21324.84
#        (x=16 binds the second replay: 3593.93)
#   L=64 stock offload  1866.8919 + 37.820312x      x=24: 2774.58  x=48: 3682.27  x=150: 7539.94
# H-B1 BYTES CARRY: every B, T and H arm on its line plus ONE common offset, all
#      within 0.05 MiB of each other; lean offset 0.00 (A30 -0.001). FALSIFIER: two
#      arms whose offsets differ by more -- a byte term moved; name it from the arm.
# H-B2 NCCL's transport is not a byte term: H arms sit on the same lines as T arms.
#
# H-N1 THE FABRIC: block T's NCCL channels run via P2P (NVLink), block H's via SHM.
#      FALSIFIER: T channels via SHM or NET -> collectives do not use NVLink here and
#      this job is not the NVSwitch point (R-H1 void; record the node).
# H-T0 SPEED (context; sets the tax's denominator): static T_A100(x) = T_A30(x) /
#      [1.5, 2.5]: x=16 2.3-3.8 s, 32 7.0-11.7, 48 15-25, 64 26-44, 100 64-107,
#      150 144-239 s (A30 fp16 tensor 165 vs A100 312 TFLOPS; HBM 0.93 vs ~2 TB/s).
# H-T1 COPIES: 63 boundaries x 1.5 MiB x 2 directions = 198.2 MB per x per rank.
#      Per profiled offload step, copy busy = 198.2 MB/x / (block 0's concurrent
#      per-GPU bandwidth) within +-20%; at 25 GB/s that is 7.9 ms/x, at 12.5 GB/s
#      (two GPUs per switch) 15.9 ms/x.
# H-T2 TWO NAMED OUTCOMES for offload's tax:
#      (a) the A30 pattern: tax <= 1.5% from x=16, ~0 by x=80-100, negative beyond.
#      (b) FASTER COMPUTE, SAME LINK (LEAN): the copies cost about what they cost on
#          the A30 in seconds while the step is 1.5-2.5x shorter, so the tax is
#          1.5-2.5x the A30's: tax(16) in [2%, 6%], tax(32) in [1%, 4%], tax(64)
#          in [0.2%, 1.5%]; x*(2%) in [16, 48] (A30 < 16, V100 ~79, L40S 32-64).
#      Neither -> a third pattern, read from P's decomposition.
# H-T3 THE IN-JOB PREDICTION: a100_pred_time.txt (T(x) and kappa fitted on x=16/32/64)
#      is written BEFORE x=100/150 run. FALSIFIER: tax at 100 or 150 outside +-30%
#      of the file (both platforms before this one failed this: recorded either way).
# H-T4 FOOTPRINT (LEAN: ABSENT): static at x=150 fills ~21 of ~80 GB here (A30: 21
#      of 23.5). If the A30's slowdown is an occupancy effect it does not appear:
#      G (offload + ballast) within +-0.5% of plain offload at x=100 and 150, and the
#      tax stays >= 0 at x=100/150. FALSIFIER: G slower than plain offload by >= 1%
#      -> the footprint effect does not depend on occupancy.
# H-H1 THE CONTRAST: with collectives through host memory, static is slower
#      (static_H >= static_T) and offload's extra time grows (delta_H > delta_T) at
#      x=32 and x=100, because copies and collectives then share the host link
#      (L40S 621880: D2H copies slowed a 128 MiB all-gather 2.56x). FALSIFIER:
#      delta_H <= delta_T at both x beyond the repeat spread -> moving collectives off
#      the host link does NOT lower offload's cost on this node.
# P    attention-replay share at x=64 ~15-17% (A30 15.3%, V100 15.6-17.1%): context.
#
# ── CONFIRM PLAN (A100_PLAN=confirm; prepared 2026-10-18) ──
# WHY: 8039176 ran on a SHARED DGX node (another job on the other 4 GPUs, each of ours
# sharing a PCIe switch with one of its GPUs). Copies ran at 15-18 GB/s against 26 solo,
# so its tax curve (x*(2%) ~ 78) may carry the neighbour. This plan reruns, EXCLUSIVE,
# the bandwidth probe, block T exactly as 8039176 ran it (x=16/32/48/64 x4, in-job
# prediction, then x=100 x4) and one profiled pair at x=64. ~2.3 h. No B/H/G arms:
# bytes, the host-path contrast and the matched footprint stand from 8039176.
#   sbatch -A <account> -p <partition> --qos=<qos> --nodes=1 \
#          --constraint=dgx-A100 --exclusive --gres=gpu:a100:8 --time=24:00:00 \
#          --export=ALL,A100_PLAN=confirm a100_regime.sbatch
# ALL 8 GPUs are requested (the node is exclusive anyway) so the job can choose ONE GPU
# PER PCIe SWITCH, as 8039176 happened to get (physical 1,2,5,7): a default 4-GPU grant
# (0-3) would put two of our GPUs on each switch, and our own copies would contend.
# The choice groups GPUs by the NICs they reach through PCIe bridges (nvidia-smi topo -m).
# Shared-run reference (8039176): concurrent copy 15.1/15.9 GB/s per GPU; copy busy
# 11.3 ms/x at x=64; static 3.020/8.962/18.395/31.485/75.428 s and offload-static
# +0.187/+0.381/+0.716/+0.814/+0.852 s at x=16/32/48/64/100.
# C-B  Bytes: every T/P arm on its line at -0.001 MiB (as 8039176).
# C-S  Compute is not the neighbour's: static within +-3% of 8039176 at every x.
#      FALSIFIER: >3% -> the neighbour slowed compute too; 8039176's times are void.
# C-W  TWO NAMED OUTCOMES for the concurrent per-GPU copy rate (block 0):
#      (i)  SWITCH: >= 20 GB/s -> the shared run's drop was the neighbour's traffic on
#           our PCIe switches; the shared tax curve is replaced by this one.
#      (ii) HOST (LEAN): within +-10% of 15.1/15.9 -> the drop is host-side (all four
#           copying at once); 8039176's curve stands and this run confirms it.
# C-X  Copy busy at P x=64 = 198.2 MB/x / C-W's rate, +-20% (as H-T1).
# C-T  The curve: delta(x) = 8039176's delta x (C-X ms/x / 11.3), +-30%, at x=32/48/64.
#      Under (ii) x*(2%) in [70, 86]; under (i) x*(2%) in [40, 70].
# C-P  In-job prediction at x=100 (+-30% rule as H-T3). LEAN: misses LOW again (the
#      static-footprint slowdown), by about 8039176's -34%.
#
# FAILURE CLASSES: (0) a GPU that will not open -> exit 3 at the PREFLIGHT, host named,
# resubmit with --exclude; (1) the node is not NVSwitch -> exit 4, resubmit (the
# constraint should prevent it); (2) ANY OOM is a defect: every arm here is far below
# an 80 GB card (largest: static x=150 at 21.3 GiB + the G ballast arm at the same).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
SMOKE=${A100_SMOKE:-}
if [ -z "$SMOKE" ]; then
    venv_ok() (
        [ -f "$1/bin/activate" ] || exit 1
        if [ -f "$1/.module" ]; then module load "$(cat "$1/.module")" 2>/dev/null || true; fi
        source "$1/bin/activate" &&
        python3 -c "import torch.distributed.run, transformers" >/dev/null 2>&1 &&
        torchrun --help >/dev/null 2>&1
    )
    VENV=${A100_VENV:-$HOME/a100_venv}
    venv_ok "$VENV" || { echo "venv $VENV MISSING OR FAILS the launcher check: build it with ../a100_venv_setup/a100_venv_setup.sh; stopping before any arm"; exit 1; }
    if [ -f "$VENV/.module" ]; then module load "$(cat "$VENV/.module")" 2>/dev/null || true; fi
    source "$VENV/bin/activate"
    echo "venv: $(cd "$VENV" && pwd)"
    OUT=a100_out/${SLURM_JOB_ID}
    NGPU=$(nvidia-smi -L | wc -l)
    if [ "$NGPU" -lt 4 ]; then echo "THIS NODE SHOWS $NGPU GPUs, FEWER THAN 4: everything here is world 4."; exit 2; fi
    # Slurm normally exposes exactly the 4 requested GPUs; if all 8 are visible, use the first 4.
    if [ "$NGPU" -gt 4 ] && [ "${A100_PLAN:-full}" = confirm ]; then
        # one GPU per PCIe switch: GPUs behind one switch reach the same NICs via PXB/PIX
        PICK=$(nvidia-smi topo -m | sed 's/\x1b\[[0-9;]*m//g' | python3 -c '
import sys
rows = [l.split() for l in sys.stdin if l.strip()]
head = rows[0]; groups = {}
for r in rows[1:]:
    if not r[0].startswith("GPU"): continue
    nics = tuple(head[i - 1] for i in range(1, min(len(r), len(head) + 1)) if head[i - 1].startswith("NIC") and r[i] in ("PXB", "PIX"))
    if nics: groups.setdefault(nics, []).append(int(r[0][3:]))
picks = sorted(g[0] for g in groups.values())
print(",".join(map(str, picks[:4])) if len(picks) >= 4 else "")')
        if [ -z "$PICK" ]; then PICK=0,2,4,6; echo "GPU PICK: switch groups not readable; DGX A100 convention 0,2,4,6"; fi
        export CUDA_VISIBLE_DEVICES=$PICK; echo "GPU PICK (one per PCIe switch): $PICK"
    fi
    if [ "$NGPU" -gt 4 ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then export CUDA_VISIBLE_DEVICES=0,1,2,3; fi
else
    OUT=a100_out/v100smoke_${SLURM_JOB_ID}
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    if [ "${SLURM_NTASKS:-0}" != "4" ]; then echo "SMOKE NEEDS 4 NODES x 1 GPU (world 4)"; exit 2; fi
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     # as every earlier job
export TF_CPP_MIN_LOG_LEVEL=3
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO          # channel lines name each collective's transport (H-N1)
mkdir -p "$OUT"
SUM="$OUT/a100_summary.txt"
DEADLINE=$((22 * 3600))
PORT=30200
log() { echo "$@" | tee -a "$SUM"; }

# ── 0. NODE ─────────────────────────────────────────────────────────────────
{
echo "=== 0. NODE ==="
echo "host: $(hostname)   job: $SLURM_JOB_ID   date: $(date -Is)   smoke: ${SMOKE:-no}   plan: ${A100_PLAN:-full}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}   SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version,power.limit,clocks.max.sm,pcie.link.gen.max,pcie.link.width.max,pci.bus_id --format=csv
echo "--- topology ---";   nvidia-smi topo -m
echo "--- NVLink ---";     nvidia-smi nvlink -s 2>&1 | head -40
echo "--- fabric manager ---"; systemctl is-active nvidia-fabricmanager 2>&1 | head -2
echo "--- NUMA / RAM ---"; (numactl -H 2>/dev/null || lscpu | grep -i numa) | head -12; free -g | head -2
} >> "$SUM" 2>&1
python3 - "$OUT/a100_node.json" <<'PY' 2>&1 | tee -a "$SUM"
import json, sys, torch, transformers
p = torch.cuda.get_device_properties(0)
d = {'device': torch.cuda.get_device_name(0), 'capability': list(torch.cuda.get_device_capability(0)),
     'total_mib': p.total_memory / 2**20, 'torch': torch.__version__, 'cuda': torch.version.cuda,
     'nccl': '.'.join(map(str, torch.cuda.nccl.version())), 'transformers': transformers.__version__,
     'visible': torch.cuda.device_count()}
json.dump(d, open(sys.argv[1], 'w'), indent=1); print(json.dumps(d))
PY

# ── PREFLIGHT: every GPU must open and allocate before any arm ──
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
    nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv 2>&1 | tee -a "$SUM"
    log "NODE FAULT on $(hostname): a GPU is unusable. Not a result. Resubmit with --exclude=$(hostname -s)"
    exit 3
fi
# ── NVSWITCH CHECK: every pair among the GPUs this job uses must be NVLink (NV#) ──
# Row GPUi, column GPUj of `nvidia-smi topo -m`; every one of the job's GPUs must have a row.
TOPO_AWK='
    BEGIN { n = split(use, u, ","); for (i = 1; i <= n; i++) want["GPU" u[i]] = 1 }
    NR == 1 { for (i = 1; i <= NF; i++) col[i] = $i; next }
    ($1 in want) && !($1 in seen) { seen[$1] = 1; rows++
        for (i = 2; i <= NF; i++) if ((col[i-1] in want) && col[i-1] != $1 && $i !~ /^NV[0-9]/) bad = bad " " $1 "-" col[i-1] ":" $i }
    END { if (rows != n) print "NOT CHECKED: " rows " of " n " GPU rows found"; else print (bad == "" ? "ok" : "NOT NVLINK:" bad) }'
if [ -z "$SMOKE" ]; then
    # nvidia-smi shows exactly our 4 GPUs (cgroup) -> check rows 0-3; it shows all 8 -> ours are CUDA_VISIBLE_DEVICES
    if [ "$NGPU" = 4 ]; then USE=0,1,2,3; else USE=${CUDA_VISIBLE_DEVICES:-0,1,2,3}; fi
    FAB=$(nvidia-smi topo -m | sed 's/\x1b\[[0-9;]*m//g' | awk -v use="$USE" "$TOPO_AWK")
    log "NVSWITCH CHECK (GPUs $USE): $FAB"
    # Stop only on positive evidence; an unreadable table is left to H-N1's NCCL channel lines.
    if [ "${FAB#NOT NVLINK}" != "$FAB" ]; then log "This node does not connect the job's GPUs by NVLink: not the R-H1 point. Resubmit with --constraint=dgx-A100 --exclude=$(hostname -s)"; exit 4; fi
fi
# ── host-copy bandwidth (solo and all GPUs at once) ──
python3 code/tools/host_copy_bw.py --mib 1024 --reps 5 --json "$OUT/a100_bw.json" 2>&1 | tee -a "$SUM"

REASON=clocks_event_reasons.active
nvidia-smi --query-gpu=$REASON --format=csv,noheader >/dev/null 2>&1 || REASON=clocks_throttle_reasons.active
GPUQ="index,timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,$REASON"

# arm BLK MODE L X N REP [extra driver flags...]   (fp16, sdpa_mem_efficient)  -> ARM_OK
# Env per arm: ARM_ENV (block H: NCCL's GPU-to-GPU paths off).
arm() {
    local BLK=$1 MODE=$2 L=$3 X=$4 N=$5 REP=$6; shift 6
    ARM_OK=0
    if [ "$SECONDS" -gt "$DEADLINE" ]; then log "SKIP $BLK $MODE x=$X: past the 22 h guard"; return; fi
    local TAG="${BLK}_${MODE}_fp16_mem_L${L}_x${X}_n${N}$( [ "$REP" != 0 ] && echo _r${REP} )"
    local D="$OUT/$TAG"; mkdir -p "$D"
    local F="--layer_checkpoint"; [ "$MODE" = offload ] && F="$F --activation_offload"
    PORT=$((PORT + 1))
    local T0=$SECONDS
    if [ -z "$SMOKE" ]; then
        LAUNCH=(torchrun --standalone --nproc_per_node=4 --master_port="$PORT")
    else
        LAUNCH=(srun torchrun --nnodes=4 --nproc_per_node=1 --rdzv_id="${SLURM_JOB_ID}_$PORT"
                --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$PORT")
    fi
    nvidia-smi --query-gpu=$GPUQ --format=csv,noheader,nounits -lms 1000 > "$D/gpu.csv" 2>/dev/null &
    local SAMPLER=$!
    env ${ARM_ENV:-} "${LAUNCH[@]}" code/train_fsdp_profile.py \
        --hidden_size 768 --intermediate_size 2048 --num_layers "$L" --num_heads 12 --num_kv_heads 4 \
        --vocab_size 32000 --dtype fp16 --attn_impl sdpa_mem_efficient --seq_len $((X * 1024)) --batch_size 1 \
        --steps "$N" --chunked_ce 256 $F --fused_rmsnorm input --use_cache off --seed 1234 \
        --disable_hbm_drift --output_dir "$D" "$@" < /dev/null > "$D/stdout.txt" 2> "$D/stderr.txt"
    kill $SAMPLER 2>/dev/null; wait $SAMPLER 2>/dev/null
    [ -n "${ARM_ENV:-}" ] && echo "$ARM_ENV" > "$D/arm_env.txt"
    [ -f "$D/steps_rank0.json" ] && ARM_OK=1
    log "ARM $TAG${ARM_ENV:+ [$ARM_ENV]}: $( [ $ARM_OK = 1 ] && echo done || echo 'DID NOT COMPLETE (read stderr.txt)' ) in $((SECONDS - T0)) s"
}
# pairs BLK REPS X:N...   ABBA order within each repeat pair
pairs() {
    local BLK=$1 REPS=$2; shift 2
    for XN in "$@"; do
        local X=${XN%:*} N=${XN#*:} R
        for R in $(seq 1 "$REPS"); do
            if [ $((R % 2)) = 1 ]; then arm "$BLK" static 64 "$X" "$N" "$R"; arm "$BLK" offload 64 "$X" "$N" "$R"
            else                         arm "$BLK" offload 64 "$X" "$N" "$R"; arm "$BLK" static 64 "$X" "$N" "$R"; fi
        done
    done
}
reader() { python3 code/tools/remote_frontier.py "$@" --platform a100 "$OUT" 2>&1 | tee -a "$SUM"; }
HOSTPATH="NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0"

if [ -z "$SMOKE" ] && [ "${A100_PLAN:-full}" = confirm ]; then
    # ── CONFIRM (exclusive rerun of block T + one profiled pair; header C-*) ──
    pairs T 4 16:20 32:10 48:8 64:6
    reader timecal --pred "$OUT/a100_pred_time.txt"
    pairs T 4 100:5
    arm P static 64 64 3 0 --torch_profile; arm P offload 64 64 3 0 --torch_profile
    PX="64"
elif [ -z "$SMOKE" ]; then
    # ── B. R-H2 bytes ──
    arm B static 32 24 3 0
    arm B static 64 24 3 0;  arm B static  64 48 3 0
    arm B offload 64 24 3 0; arm B offload 64 48 3 0
    reader bytes
    # ── T. R-H1: calibrate on 16/32/(48)/64, write the prediction, then 100/150 ──
    pairs T 4 16:20 32:10 48:8 64:6
    reader timecal --pred "$OUT/a100_pred_time.txt"
    pairs T 2 100:5 150:4
    # ── H. collectives through host memory, same node ──
    ARM_ENV=$HOSTPATH; pairs H 2 32:10 100:5; ARM_ENV=
    # ── P. profiled pairs ──
    for X in 64 150; do
        arm P static 64 $X 3 0 --torch_profile; arm P offload 64 $X 3 0 --torch_profile
    done
    # ── G. matched footprint (ballast = static - offload line: -390.0953 + 94.5x MiB) ──
    arm G offload 64 100 5 0 --hbm_ballast_mb 9059.905  --hbm_ballast_chunks 63
    arm G offload 64 150 4 0 --hbm_ballast_mb 13784.905 --hbm_ballast_chunks 63
    PX="64 150"
else
    # ── V100 SMOKE (a100_v100_smoke.sbatch): every code path at small x ──
    arm B static 32 24 3 0; arm B static 64 24 3 0; arm B offload 64 24 3 0
    reader bytes
    pairs T 1 16:6 32:4 64:4
    reader timecal --pred "$OUT/a100_pred_time.txt"
    ARM_ENV=$HOSTPATH; pairs H 1 16:4; ARM_ENV=
    arm P static 64 16 3 0 --torch_profile; arm P offload 64 16 3 0 --torch_profile
    arm G offload 64 24 3 0 --hbm_ballast_mb 1877.905 --hbm_ballast_chunks 63
    PX="16"
fi

# ── profiled-pair analysis ──
for X in $PX; do
    for M in static offload; do
        d="$OUT/P_${M}_fp16_mem_L64_x${X}_n3/"
        TR=$(ls -t "$d"torch_trace_rank0/*.pt.trace.json* 2>/dev/null | head -n 1)
        [ -n "$TR" ] || { log "P x=$X $M: no trace"; continue; }
        log "=== P: $(basename "$d") ==="
        python3 code/tools/attn_recompute_share.py "$TR" --layers 64 --steps 1 --json "${d}attn_share.json" 2>&1 | tail -15 | tee -a "$SUM"
        python3 code/tools/analyze_nccl_duty.py "$d"torch_trace_rank*/*.pt.trace.json* --json_out "${d}duty.json" 2>&1 | tail -25 | tee -a "$SUM"
    done
    S=$(ls "$OUT"/P_static_fp16_mem_L64_x${X}_n3/torch_trace_rank*/*.pt.trace.json* 2>/dev/null)
    O=$(ls "$OUT"/P_offload_fp16_mem_L64_x${X}_n3/torch_trace_rank*/*.pt.trace.json* 2>/dev/null)
    if [ -n "$S" ] && [ -n "$O" ]; then
        log "=== P x=$X: decomposition (median over ranks) ==="
        python3 code/tools/copy_exposure.py --offload $O --static $S --json "$OUT/decomp_x${X}.json" 2>&1 | tee -a "$SUM"
    fi
done
reader read
log "=== DONE in $((SECONDS / 60)) min: ./a100_collect.sh $SLURM_JOB_ID ==="
