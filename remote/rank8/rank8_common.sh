#!/bin/bash
# Shared by a30_rank8.sbatch, a100_rank8.sbatch and rank8_v100_smoke.sbatch (sourced, not run).
# The caller sets PLATFORM (a30|a100), PREFIX (file prefix), VENV_DEFAULT, VENV_VAR, HOURS.
# Arms launch with one torchrun per node: on one node (torchrun --standalone) when the arm's
# world fits the batch node's GPUs, else through srun on world/GPN nodes (c10d rendezvous).

r8_setup() {
    set -uo pipefail
    cd "$SLURM_SUBMIT_DIR"
    if [ -z "${R8_V100:-}" ]; then
        venv_ok() (
            [ -f "$1/bin/activate" ] || exit 1
            if [ -f "$1/.module" ]; then module load "$(cat "$1/.module")" 2>/dev/null || true; fi
            source "$1/bin/activate" &&
            python3 -c "import torch.distributed.run, transformers" >/dev/null 2>&1 &&
            torchrun --help >/dev/null 2>&1
        )
        VENV=${!VENV_VAR:-$VENV_DEFAULT}
        venv_ok "$VENV" || { echo "venv $VENV MISSING OR FAILS the launcher check; stopping before any arm"; exit 1; }
        if [ -f "$VENV/.module" ]; then module load "$(cat "$VENV/.module")" 2>/dev/null || true; fi
        source "$VENV/bin/activate"
        OUT=rank8_out/${SLURM_JOB_ID}
    else
        OUT=rank8_out/v100smoke_${PLATFORM}_${SLURM_JOB_ID}
    fi
    mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
    NN=${#NODES[@]}
    MASTER_ADDR=${NODES[0]}
    GPN=$(nvidia-smi -L | wc -l)                 # GPUs on the batch node (every node is alike)
    CPN=${SLURM_CPUS_ON_NODE:-1}
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     # as every earlier job
    export TF_CPP_MIN_LOG_LEVEL=3
    export OMP_NUM_THREADS=8
    export NCCL_DEBUG=INFO                       # channel lines name each collective's transport
    mkdir -p "$OUT"
    SUM="$OUT/${PREFIX}_summary.txt"
    DEADLINE=$(( (HOURS - 1) * 3600 ))
    PORT=30400
    ARM_TMO_DEFAULT=10800                        # 3 h: no single arm here should come near it
    REASON=clocks_event_reasons.active
    nvidia-smi --query-gpu=$REASON --format=csv,noheader >/dev/null 2>&1 || REASON=clocks_throttle_reasons.active
    GPUQ="index,timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,$REASON"
}

log() { echo "$@" | tee -a "$SUM"; }

# one task per node on the first N nodes, every CPU of the node, no binding
on_nodes() {
    local N=$1; shift
    srun --nodes="$N" --ntasks="$N" --ntasks-per-node=1 --cpus-per-task="$CPN" --cpu-bind=none \
         --kill-on-bad-exit=1 "$@"
}

# LAUNCH array for an arm of WORLD ranks
launcher() {
    local WORLD=$1
    PORT=$((PORT + 1))
    if [ "$WORLD" -le "$GPN" ]; then
        LAUNCH=(torchrun --standalone --nproc_per_node="$WORLD" --master_port="$PORT")
    else
        local NNA=$(( WORLD / GPN ))
        if [ $(( NNA * GPN )) != "$WORLD" ] || [ "$NNA" -gt "$NN" ]; then
            log "LAUNCH: world $WORLD does not fit $NN node(s) x $GPN GPUs"; LAUNCH=(false); return
        fi
        # a command array, not the on_nodes function: it runs under timeout(1)
        LAUNCH=(srun --nodes="$NNA" --ntasks="$NNA" --ntasks-per-node=1 --cpus-per-task="$CPN" --cpu-bind=none
                --kill-on-bad-exit=1 torchrun --nnodes="$NNA" --nproc_per_node="$GPN"
                --rdzv_id="${SLURM_JOB_ID}_$PORT" --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$PORT")
    fi
}

# ── 0. nodes: facts, preflight on every node, NCCL across the job's ranks ──────────────
r8_nodes() {
    {
    echo "=== 0. NODES ==="
    echo "job: $SLURM_JOB_ID   nodes: ${NODES[*]}   GPUs per node: $GPN   CPUs per node: $CPN   date: $(date -Is)   plan: ${R8_PLAN:-full}"
    } >> "$SUM"
    on_nodes "$NN" bash -c '
        echo "--- $(hostname -s)"
        nvidia-smi --query-gpu=index,name,memory.total,driver_version,power.limit,clocks.max.sm,pcie.link.gen.max,pcie.link.width.max,pci.bus_id --format=csv,noheader
        echo "topology:"; nvidia-smi topo -m | sed "s/\x1b\[[0-9;]*m//g"
        echo "infiniband:"; for d in /sys/class/infiniband/*; do [ -e "$d" ] || continue
            for p in "$d"/ports/*; do echo "  $(basename "$d") port $(basename "$p"): $(cat "$p/rate" 2>/dev/null) $(cat "$p/state" 2>/dev/null) $(cat "$p/link_layer" 2>/dev/null)"; done; done
        free -g | head -2' >> "$SUM" 2>&1
    python3 - "$OUT/${PREFIX}_node.json" <<'PY' 2>&1 | tee -a "$SUM"
import json, sys, torch, transformers
p = torch.cuda.get_device_properties(0)
d = {'device': torch.cuda.get_device_name(0), 'capability': list(torch.cuda.get_device_capability(0)),
     'total_mib': p.total_memory / 2**20, 'torch': torch.__version__, 'cuda': torch.version.cuda,
     'nccl': '.'.join(map(str, torch.cuda.nccl.version())), 'transformers': transformers.__version__,
     'visible': torch.cuda.device_count()}
json.dump(d, open(sys.argv[1], 'w'), indent=1); print(json.dumps(d))
PY
    # PREFLIGHT: every GPU on every node must open and allocate before any arm
    on_nodes "$NN" python3 -c '
import socket, sys, torch
bad = []
for i in range(torch.cuda.device_count()):
    try:
        torch.cuda.set_device(i); t = torch.empty(256 * 2**20, dtype=torch.uint8, device=f"cuda:{i}"); del t
    except Exception as e:
        bad.append((i, repr(e)[:160]))
h = socket.gethostname().split(".")[0]
for i, e in bad: print(f"PREFLIGHT {h} GPU {i} FAILED: {e}")
print(f"PREFLIGHT {h}: " + ("ok, %d GPUs open and allocate" % torch.cuda.device_count() if not bad else "FAILED"))
sys.exit(1 if bad else 0)' 2>&1 | tee -a "$SUM"
    if [ "${PIPESTATUS[0]}" != "0" ]; then
        log "NODE FAULT: a GPU is unusable (host named above). Not a result. Resubmit with --exclude=<host>"
        exit 3
    fi
    # NCCL all-reduce across the job's ranks: the transport (channel lines) and the bus bandwidth
    cat > "$OUT/nccl_bw.py" <<'PY'
import os, time, torch, torch.distributed as dist
dist.init_process_group('nccl'); torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
r, w = dist.get_rank(), dist.get_world_size()
for mib in (64, 256):
    t = torch.ones(mib * 2**20 // 2, dtype=torch.float16, device='cuda')
    for _ in range(3): dist.all_reduce(t)
    torch.cuda.synchronize(); t0 = time.time(); n = 10
    for _ in range(n): dist.all_reduce(t)
    torch.cuda.synchronize(); dt = (time.time() - t0) / n
    if r == 0:
        print(f'NCCL all_reduce {mib} MiB fp16 over {w} ranks: {dt * 1e3:.2f} ms, bus bandwidth '
              f'{mib * 2**20 * 2 * (w - 1) / w / dt / 1e9:.2f} GB/s', flush=True)
dist.destroy_process_group()
PY
    for W in $R8_NCCL_WORLDS; do
        mkdir -p "$OUT/nccl_w$W"; launcher "$W"
        timeout 600 "${LAUNCH[@]}" "$OUT/nccl_bw.py" > "$OUT/nccl_w$W/stdout.txt" 2> "$OUT/nccl_w$W/stderr.txt"
        local RC=$?
        grep -h '^NCCL all_reduce' "$OUT/nccl_w$W/stdout.txt" | tee -a "$SUM"
        log "NCCL world $W transports: $(grep -hoE ' via (P2P/[A-Za-z]+|SHM|NET/[A-Za-z]+|NVLS|CollNet)' "$OUT/nccl_w$W"/std*.txt | sort | uniq -c | tr '\n' ';')"
        if [ "$RC" != 0 ]; then log "NCCL TEST world $W FAILED (rc $RC; $OUT/nccl_w$W/stderr.txt): the launcher or fabric is broken; stopping"; exit 5; fi
    done
    python3 code/tools/host_copy_bw.py --mib 1024 --reps 5 --json "$OUT/${PREFIX}_bw.json" 2>&1 | tee -a "$SUM"
}

# arm BLK MODE SHAPE L X N REP WORLD [extra driver flags...]   (fp16, sdpa_mem_efficient) -> ARM_OK
# SHAPE: tier_a | llama3b | mistral7b.  ARM_TMO (s) overrides the per-arm timeout.
arm() {
    local BLK=$1 MODE=$2 SHAPE=$3 L=$4 X=$5 N=$6 REP=$7 WORLD=$8; shift 8
    ARM_OK=0
    if [ "$SECONDS" -gt "$DEADLINE" ]; then log "SKIP $BLK $MODE $SHAPE x=$X: past the $((HOURS - 1)) h guard"; return; fi
    local SHP="" FLAGS
    case $SHAPE in
        tier_a)    FLAGS="--hidden_size 768 --intermediate_size 2048 --num_heads 12 --num_kv_heads 4 --vocab_size 32000" ;;
        llama3b)   SHP=_llama3b;   FLAGS="--hidden_size 3072 --intermediate_size 8192 --num_heads 24 --num_kv_heads 8 --vocab_size 128256" ;;
        mistral7b) SHP=_mistral7b; FLAGS="--hidden_size 4096 --intermediate_size 14336 --num_heads 32 --num_kv_heads 8 --vocab_size 32000" ;;
        *) log "unknown shape $SHAPE"; return ;;
    esac
    local TAG="${BLK}_${MODE}_fp16_mem${SHP}_w${WORLD}_L${L}_x${X}_n${N}$( [ "$REP" != 0 ] && echo _r${REP} )"
    local D="$OUT/$TAG"; mkdir -p "$D"
    local F="--layer_checkpoint"; [ "$MODE" = offload ] && F="$F --activation_offload"
    launcher "$WORLD"
    local T0=$SECONDS
    nvidia-smi --query-gpu=$GPUQ --format=csv,noheader,nounits -lms 1000 > "$D/gpu.csv" 2>/dev/null &
    local SAMPLER=$!
    timeout --kill-after=120 "${ARM_TMO:-$ARM_TMO_DEFAULT}" "${LAUNCH[@]}" code/train_fsdp_profile.py $FLAGS --num_layers "$L" \
        --dtype fp16 --attn_impl sdpa_mem_efficient --seq_len $((X * 1024)) --batch_size 1 \
        --steps "$N" --chunked_ce 256 $F --fused_rmsnorm input --use_cache off --seed 1234 \
        --disable_hbm_drift --output_dir "$D" "$@" < /dev/null > "$D/stdout.txt" 2> "$D/stderr.txt"
    local RC=$?
    kill $SAMPLER 2>/dev/null; wait $SAMPLER 2>/dev/null
    [ -f "$D/steps_rank0.json" ] && ARM_OK=1
    [ "$RC" = 124 ] && log "ARM $TAG: TIMED OUT after ${ARM_TMO:-$ARM_TMO_DEFAULT} s"
    log "ARM $TAG: $( [ $ARM_OK = 1 ] && echo done || echo "DID NOT COMPLETE (rc $RC; read stderr.txt)" ) in $((SECONDS - T0)) s"
}

# pairs BLK REPS SHAPE L WORLD X:N ...   static/offload, ABBA order across repeats
pairs() {
    local BLK=$1 REPS=$2 SHAPE=$3 L=$4 WORLD=$5; shift 5
    for XN in "$@"; do
        local X=${XN%:*} N=${XN#*:} R
        for R in $(seq 1 "$REPS"); do
            if [ $((R % 2)) = 1 ]; then arm "$BLK" static "$SHAPE" "$L" "$X" "$N" "$R" "$WORLD"; arm "$BLK" offload "$SHAPE" "$L" "$X" "$N" "$R" "$WORLD"
            else                         arm "$BLK" offload "$SHAPE" "$L" "$X" "$N" "$R" "$WORLD"; arm "$BLK" static "$SHAPE" "$L" "$X" "$N" "$R" "$WORLD"; fi
        done
    done
}

reader() { python3 code/tools/remote_frontier.py "$@" --platform "$PLATFORM" "$OUT" 2>&1 | tee -a "$SUM"; }

# native BLK MODE SHAPE L WORLD STEP "HEADER RUNGS"
# Writes the in-job prediction (reader natpred) BEFORE the ladder, then climbs its rungs
# (2 steps each: a failure can land in step 2) to the first failure; if the first rung
# fails, descends one rung at a time, at most four.
native() {
    local BLK=$1 MODE=$2 SHAPE=$3 L=$4 WORLD=$5 STEP=$6 HDR=$7
    local PF="$OUT/${PREFIX}_pred_native_${BLK}.txt"
    reader natpred --pred "$PF" --ladder "$BLK" --shape "$SHAPE" --L "$L" --world "$WORLD" --mode "$MODE" --step "$STEP" > /dev/null
    local RUNGS
    RUNGS=$(grep -s '^RUNGS:' "$PF" | cut -d' ' -f2-)
    if [ -z "$RUNGS" ]; then RUNGS=$HDR; log "LADDER $BLK: no in-job prediction; header rungs $HDR"; fi
    log "LADDER $BLK ($MODE $SHAPE w$WORLD): rungs $RUNGS"
    local FIRST n=0 X
    FIRST=${RUNGS%% *}
    for X in $RUNGS; do
        arm "$BLK" "$MODE" "$SHAPE" "$L" "$X" 2 0 "$WORLD"
        n=$((n + 1))
        if [ "$ARM_OK" != 1 ]; then
            log "LADDER $BLK CLOSED at x=$X"
            if [ "$n" = 1 ]; then
                log "LADDER $BLK: the first rung failed -- the prediction was high; descending"
                local k
                for k in 1 2 3 4; do
                    arm "$BLK" "$MODE" "$SHAPE" "$L" $((FIRST - k * STEP)) 2 0 "$WORLD"
                    [ "$ARM_OK" = 1 ] && break
                done
            fi
            return
        fi
    done
    log "LADDER $BLK: every rung fit -- the bracket stays open (prediction low)"
}
