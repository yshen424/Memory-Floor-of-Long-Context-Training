#!/bin/bash
# Shared by a30f_tax.sbatch and a30f_native.sbatch (sourced, never run alone).
# Sets up the venv (ClusterA) or the multi-node launch (V100 smoke), writes node
# facts, and defines the arm runner. Every arm logs GPU clocks/power once a second
# to gpu.csv (the tax job's clock hypothesis) and, when HOSTMEM_FLOOR_PCT is set,
# runs a host-memory watchdog that kills the arm before the node can drain.
#   needs: JOBNAME (tax|native), SMOKE (empty on ClusterA), DEADLINE (s)
cd "$SLURM_SUBMIT_DIR"
if [ -z "$SMOKE" ]; then
    # Launcher check, not mere existence (a venv whose torchrun cannot import kills every arm).
    venv_ok() (
        [ -f "$1/bin/activate" ] || exit 1
        if [ -f "$1/.module" ]; then module load "$(cat "$1/.module")" 2>/dev/null || true; fi
        source "$1/bin/activate" &&
        python3 -c "import torch.distributed.run, transformers" >/dev/null 2>&1 &&
        torchrun --help >/dev/null 2>&1
    )
    # ~/a30_venv only (built once by ../a30_venv_setup/a30_venv_setup.sh, same pins as
    # ~/l40s_venv); A30_VENV overrides. No search list.
    VENV=${A30_VENV:-$HOME/a30_venv}
    venv_ok "$VENV" || { echo "venv $VENV MISSING OR FAILS the launcher check: build it with ../a30_venv_setup/a30_venv_setup.sh; stopping before any arm"; exit 1; }
    if [ -f "$VENV/.module" ]; then module load "$(cat "$VENV/.module")" 2>/dev/null || true; fi
    source "$VENV/bin/activate"
    echo "venv: $(cd "$VENV" && pwd)"
    OUT=a30f_out/${JOBNAME}_${SLURM_JOB_ID}
    NGPU=$(nvidia-smi -L | wc -l)
    if [ "$NGPU" != "4" ]; then echo "THIS NODE HAS $NGPU GPUs, NOT 4: everything here is world 4."; exit 2; fi
else
    OUT=a30f_out/v100smoke_${JOBNAME}_${SLURM_JOB_ID}
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    if [ "${SLURM_NTASKS:-0}" != "4" ]; then echo "SMOKE NEEDS 4 NODES x 1 GPU (world 4)"; exit 2; fi
fi
# Every earlier job (V100 and remote, 633912 included) ran with expandable segments
# on; it stays the default here, and the tax job's E block turns it OFF per arm.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TF_CPP_MIN_LOG_LEVEL=3
export OMP_NUM_THREADS=8
mkdir -p "$OUT"
SUM="$OUT/a30f_summary.txt"
PORT=29900
log() { echo "$@" | tee -a "$SUM"; }

# ── 0. NODE ──
{
echo "=== 0. NODE ($JOBNAME) ==="
echo "host: $(hostname)   job: $SLURM_JOB_ID   date: $(date -Is)   smoke: ${SMOKE:-no}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version,power.limit,clocks.max.sm,clocks.max.mem,pci.bus_id --format=csv
echo "--- clocks / power policy ---"; nvidia-smi -q -d CLOCK,POWER,PERFORMANCE 2>&1 | head -80
echo "--- topology ---"; nvidia-smi topo -m
echo "--- RAM ---"; free -g | head -2; grep -E 'MemTotal|MemAvailable' /proc/meminfo
} >> "$SUM" 2>&1
python3 - "$OUT/a30f_node.json" <<'PY' 2>&1 | tee -a "$SUM"
import json, sys, torch
p = torch.cuda.get_device_properties(0)
d = {'device': torch.cuda.get_device_name(0), 'capability': list(torch.cuda.get_device_capability(0)),
     'total_mib': p.total_memory / 2**20, 'torch': torch.__version__}
json.dump(d, open(sys.argv[1], 'w'), indent=1); print(json.dumps(d))
PY
# The throttle field was renamed in newer drivers; use whichever this one accepts.
REASON=clocks_event_reasons.active
nvidia-smi --query-gpu=$REASON --format=csv,noheader >/dev/null 2>&1 || REASON=clocks_throttle_reasons.active
GPUQ="index,timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,$REASON"

# arm BLK MODE DT ATTN L X N REP [extra driver flags...]   -> sets ARM_OK, ARM_HOSTKILL
# Env per arm: ARM_ENV (e.g. "PYTORCH_CUDA_ALLOC_CONF=" for allocator default).
arm() {
    local BLK=$1 MODE=$2 DT=$3 ATTN=$4 L=$5 X=$6 N=$7 REP=$8; shift 8
    ARM_OK=0; ARM_HOSTKILL=0
    if [ "$SECONDS" -gt "$DEADLINE" ]; then log "SKIP $BLK $MODE x=$X: past the deadline guard"; return; fi
    local TAG="${BLK}_${MODE}_${DT}_${ATTN}_L${L}_x${X}_n${N}$( [ "$REP" != 0 ] && echo _r${REP} )"
    local D="$OUT/$TAG"; mkdir -p "$D"
    local F="--layer_checkpoint"; [ "$MODE" = offload ] && F="$F --activation_offload"
    local A=sdpa_mem_efficient; [ "$ATTN" = flash ] && A=sdpa_flash
    PORT=$((PORT + 1))
    local T0=$SECONDS
    if [ -z "$SMOKE" ]; then
        LAUNCH=(torchrun --standalone --nproc_per_node=4 --master_port="$PORT")
    else
        LAUNCH=(srun torchrun --nnodes=4 --nproc_per_node=1 --rdzv_id="${SLURM_JOB_ID}_$PORT"
                --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$PORT")
    fi
    # 1 Hz clocks/power on this node's GPUs (the smoke: the batch node's one V100)
    nvidia-smi --query-gpu=$GPUQ --format=csv,noheader,nounits -lms 1000 > "$D/gpu.csv" 2>/dev/null &
    local SAMPLER=$!
    env ${ARM_ENV:-} "${LAUNCH[@]}" code/train_fsdp_profile.py \
        --hidden_size 768 --intermediate_size 2048 --num_layers "$L" --num_heads 12 --num_kv_heads 4 \
        --vocab_size 32000 --dtype "$DT" --attn_impl "$A" --seq_len $((X * 1024)) --batch_size 1 \
        --steps "$N" --chunked_ce 256 $F --fused_rmsnorm input --use_cache off --seed 1234 \
        --disable_hbm_drift --output_dir "$D" "$@" < /dev/null > "$D/stdout.txt" 2> "$D/stderr.txt" &
    local RUN=$!
    if [ -n "${HOSTMEM_FLOOR_PCT:-}" ]; then
        # Host-memory watchdog: the node must never drain. Kill the arm if available
        # memory falls below HOSTMEM_FLOOR_PCT of total; record the lowest reading.
        ( lo=100; while kill -0 $RUN 2>/dev/null; do
              av=$(awk '/MemAvailable/{a=$2}/MemTotal/{t=$2}END{printf "%d", 100*a/t}' /proc/meminfo)
              [ "$av" -lt "$lo" ] && { lo=$av; echo "$lo" > "$D/hostmem_min_avail_pct"; }
              if [ "$av" -lt "$HOSTMEM_FLOOR_PCT" ]; then
                  echo "HOSTMEM KILL: MemAvailable ${av}% < ${HOSTMEM_FLOOR_PCT}%" > "$D/hostmem_kill"
                  pkill -TERM -P $RUN; kill -TERM $RUN 2>/dev/null; sleep 20
                  pkill -KILL -P $RUN; kill -KILL $RUN 2>/dev/null; break
              fi; sleep 5; done ) &
        local WATCH=$!
    fi
    wait $RUN
    kill $SAMPLER 2>/dev/null; wait $SAMPLER 2>/dev/null
    [ -n "${WATCH:-}" ] && { kill $WATCH 2>/dev/null; wait $WATCH 2>/dev/null; }
    [ -f "$D/steps_rank0.json" ] && ARM_OK=1
    [ -f "$D/hostmem_kill" ] && { ARM_HOSTKILL=1; ARM_OK=0; }
    log "ARM $TAG: $( [ $ARM_OK = 1 ] && echo done || { [ $ARM_HOSTKILL = 1 ] && echo 'KILLED BY THE HOST-MEMORY WATCHDOG' || echo 'DID NOT COMPLETE (read stderr.txt)'; } ) in $((SECONDS - T0)) s"
    ARM_SECONDS=$((SECONDS - T0))
}
reader() { python3 code/tools/a30_followup.py "$@" "$OUT" 2>&1 | tee -a "$SUM"; }
