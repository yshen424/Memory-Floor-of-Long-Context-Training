#!/bin/bash -l
#SBATCH --job-name=l64s_linktest
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=l64s_linktest_%j.log
#SBATCH --error=l64s_linktest_%j.err
#
# Submit from this directory (directives above carry the same flags):
#   sbatch -A <account> -p <partition> --qos=<qos> --nodes=1 \
#          --exclusive --gres=gpu:l40s:4 --time=01:00:00 l64s_linktest.sbatch
# GPU request is typed --gres only; never add --gpus-per-node (they combine).
# Venv: reuses ../modern_gpu_probe/venv if built; else ./l64s_setup_env.sh once.

# ── JOB: THE PCIe LINK TEST. ONE L40S NODE (PCIe-only, no NVLink), WORLD 4 ───
# Question: on a 2023 PCIe-only node, do host<->device copies and NCCL
# collectives EXCLUDE each other, SHARE the link, or run INDEPENDENTLY? And
# does real offload traffic slow the collectives?
#
#  0  NODE: GPUs, PCIe gen/width, topology, P2P matrix, NUMA, host RAM.
#  1  LINK MICROBENCHMARK, no model, twice: NCCL's default transport, then
#     NCCL_P2P_DISABLE=1 (forces the host-staged SHM path, V100-like).
#  2  TRAINING ARMS, 6 steps, traces on steps 2-4:
#     T1 static whole-layer (control)
#     T2 T1 + forced-overlap probe (copies issued during collectives)
#     T3 stock activation offload (real traffic)
#     T4/T5 = T1/T2 under NCCL_P2P_DISABLE=1 -- ONLY if part 1 shows NCCL
#     chose P2P by default (otherwise they would repeat T1/T2).
#  3  ANALYSIS in-job: analyze_nccl_duty.py per arm -> l64s_summary.txt.
#
#  Training config: L=32 anchor, hidden 768, inter 2048, 12/4 heads,
#  v=32000, regime B (fp16 / sdpa_mem_efficient), whole-layer, chunked CE
#  256, fused-input, use_cache off, b=1, seq 8192 (x=8), seed 1234. Time
#  test, so no HBM cap. Expected wall clock ~20-30 min.

# ── PRE-REGISTERED PREDICTIONS. TIME/BANDWIDTH NUMBERS ARE L40S-SPECIFIC ──
# Every number here is a time ratio or a bandwidth: stated for THIS node's
# topology, never carried to another GPU.
#
# L-P1 PER-GPU COPY SPEED (copy_solo). L40S is PCIe Gen4 x16: 31.5 GB/s raw
#      per direction; pinned cudaMemcpy reaches 0.70-0.85 of raw -> each GPU
#      22-27 GB/s each way. A GPU well below the others is the V100's
#      bimodal-rank pattern (NUMA locality of the pinned buffer); record which.
# L-P2 HOST-SIDE SHARING (copy_all / copy_solo), conditioned on `topo -m`:
#      GPUs on their own root ports (NODE/SYS between them) -> >= 0.90;
#      GPU pairs behind one switch (PIX/PXB) -> ~0.5 if the uplink is x16.
# L-P3 TRANSPORT: NCCL picks P2P iff `topo -p2p r` reports OK for the pair
#      and the path is within NCCL_P2P_LEVEL; otherwise SHM. No lean: recorded.
# L-P4 THE VERDICT (rule in code/l64s_linkbench.py; 128 MiB all_gather):
#      r_copy = in-window copy throughput / copy_all throughput,
#      r_coll = all_gather duration concurrent / solo.
#      EXCLUSION r_copy < 0.10 | INDEPENDENT r_copy >= 0.90 and r_coll <= 1.10
#      | SHARING otherwise.
#      LEAN: SHARING under both transports -- on one node, PCIe arbitrates per
#      packet and nothing in the path serializes whole transfers. EXCLUSION
#      under NCCL_P2P_DISABLE=1 would be an exclusion
#      reading on a 2023 card.
#      EXCLUSION under the default transport would be the strongest result.
# L-P5 TRAINING CROSS-CHECK: T2's trace agrees with part 1's verdict for the
#      same transport -- EXCLUSION => ~0 ms of probe copies inside NCCL
#      kernels; SHARING => probe copies inside NCCL kernels at reduced
#      bandwidth and the ">=50% under migration" NCCL cohort longer than the
#      clean cohort.
# L-P6 REAL TRAFFIC IS SMALL HERE. T3 moves the 32 boundaries:
#      2 B x 768 x 8192 = 12 MiB/layer, 384 MiB each way per rank-step
#      -> 384 MiB / B_copy ~ 15-18 ms each way.
#      Predicted: T3 steady step within +-3% of T1 and NCCL median kernel
#      within +-10% of T1. A larger collective slowdown would be volume-driven
#      collective inflation even at this small volume.
# L-P7 ALL ARMS: step-0 losses agree to < 1e-3 across T1-T5 (the probe and
#      the transport must not touch numerics). Peaks: T4 = T1 (T3 recorded:
#      at x=8 it sits near the static/floor crossover); T2/T5 = T1/T4 + ~52.2 MiB, the probe's
#      54.7 MB of device buffers, which stay allocated across steps.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
# L64S_SMOKE=1 selects the smoke mode: one GPU per node across
# several V100 nodes, the V100 cluster's own python, shorter bench phases.
SMOKE=${L64S_SMOKE:-}
if [ -z "$SMOKE" ]; then
    # One venv serves every cluster: reuse the probe package's venv; build
    # l64s_venv only if it is absent.
    VENV=${L64S_VENV:-}
    [ -z "$VENV" ] && for v in l64s_venv ../modern_gpu_probe/venv; do [ -f "$v/bin/activate" ] && { VENV=$v; break; }; done
    [ -z "$VENV" ] && { echo "no venv: run ../modern_gpu_probe/setup_env.sh or ./l64s_setup_env.sh"; exit 1; }
    if [ -f "$VENV/.module" ]; then module load "$(cat "$VENV/.module")" 2>/dev/null || true; fi
    source "$VENV/bin/activate"
    echo "venv: $(cd "$VENV" && pwd)"
    OUT=l64s_out/${SLURM_JOB_ID}
else
    OUT=l64s_out/v100smoke_${SLURM_JOB_ID}
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
fi
# Same allocator config as every V100 run in the project.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set_launch() {  # $1 = port -> LAUNCH array (env/timeout cannot call a function)
    if [ -z "$SMOKE" ]; then
        LAUNCH=(torchrun --standalone --nproc_per_node="$NGPU" --master_port="$1")
    else
        LAUNCH=(srun torchrun --nnodes="$SLURM_NNODES" --nproc_per_node=1
                --rdzv_id="${SLURM_JOB_ID}_$1" --rdzv_backend=c10d
                --rdzv_endpoint="$MASTER_ADDR:$1")
    fi
}
mkdir -p "$OUT"
SUM="$OUT/l64s_summary.txt"
NGPU=$(nvidia-smi -L | wc -l)
export OMP_NUM_THREADS=8

# ── 0. NODE ─────────────────────────────────────────────────────────────────
{
echo "=== 0. NODE ==="
echo "host: $(hostname)   job: $SLURM_JOB_ID   date: $(date -Is)   GPUs: $NGPU"
nvidia-smi --query-gpu=index,name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max,pcie.link.gen.current,pcie.link.width.current,pci.bus_id --format=csv
echo "--- topology ---";            nvidia-smi topo -m
echo "--- P2P read support ---";    nvidia-smi topo -p2p r
echo "--- P2P write support ---";   nvidia-smi topo -p2p w
echo "--- NUMA ---";                (numactl -H 2>/dev/null || lscpu | grep -i numa) | head -20
echo "--- CPU / RAM ---";           lscpu | grep -E "Model name|Socket|NUMA node\(s\)"; free -g | head -2
python3 - <<'PY'
import torch, transformers, sys
print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.version.cuda,
      "| nccl", ".".join(map(str, torch.cuda.nccl.version())), "| transformers", transformers.__version__)
print("device", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0),
      "| HBM %.2f GiB" % (torch.cuda.get_device_properties(0).total_memory / 2**30))
PY
echo
} > "$SUM" 2>&1

# ── 1. LINK MICROBENCHMARK, default transport then P2P disabled ─────────────
PORT=29700
bench() {   # $1 = label, rest = env assignments
    local label=$1; shift
    PORT=$((PORT + 1))
    echo "=== 1. LINK MICROBENCHMARK: $label ===" >> "$SUM"
    set_launch $PORT
    env NCCL_DEBUG=INFO "$@" timeout 900 "${LAUNCH[@]}" code/l64s_linkbench.py \
        --out "$OUT/bench_$label.json" --phase_s "${L64S_PHASE_S:-2.0}" \
        > "$OUT/bench_$label.log" 2>&1 < /dev/null
    local rc=$?
    echo "--- transport NCCL chose (channel count, all ranks) ---" >> "$SUM"
    grep -oE "via [A-Za-z0-9/_]+" "$OUT/bench_$label.log" | sort | uniq -c >> "$SUM"
    if [ $rc -ne 0 ]; then echo "  FAILED (exit $rc) -- see bench_$label.log" >> "$SUM"; return; fi
    grep -v -E "NCCL INFO|^\[rank|Warning|warn" "$OUT/bench_$label.log" | sed -n '/^NCCL /,$p' >> "$SUM"
    echo >> "$SUM"
}
bench default
bench p2poff NCCL_P2P_DISABLE=1
P2P_USED=$(grep -cE "via P2P" "$OUT/bench_default.log" 2>/dev/null); P2P_USED=${P2P_USED:-0}
echo "P2P channels in default transport: $P2P_USED  -> T4/T5 $([ "$P2P_USED" -gt 0 ] && echo WILL || echo will NOT) run" >> "$SUM"
echo >> "$SUM"

# Probe sizing. overlap_probe.py converts its target duration to an op count
# at a V100 nominal 11 GB/s; scale the target by this node's measured copy
# speed so the train covers the whole step.
LINK_GBPS=$(python3 - "$OUT/bench_default.json" <<'PY'
import json, sys, statistics
try:
    r = json.load(open(sys.argv[1]))["ranks"]
    print("%.2f" % statistics.median(x["copy_all"][d]["throughput_gbps"] for x in r for d in ("d2h", "h2d")))
except Exception:
    print("22.0")
PY
)
echo "probe sizing: measured copy_all median = $LINK_GBPS GB/s" >> "$SUM"

# ── 2. TRAINING ARMS ────────────────────────────────────────────────────────
COMMON="--hidden_size 768 --intermediate_size 2048 --num_layers 32 --num_heads 12
        --num_kv_heads 4 --vocab_size 32000 --dtype fp16 --attn_impl sdpa_mem_efficient
        --batch_size 1 --seq_len 8192 --steps 6 --chunked_ce 256 --fused_rmsnorm input
        --use_cache off --seed 1234 --disable_hbm_drift --layer_checkpoint --torch_profile"

steady_ms() {  # mean step_ms over steps >= 2, all ranks
    python3 - "$1" <<'PY'
import json, glob, sys
v = [s["step_ms"] for p in glob.glob(sys.argv[1] + "/steps_rank*.json")
     for i, s in enumerate(json.load(open(p))["steps"]) if i >= 2]
print("%.0f" % (sum(v) / len(v)) if v else "2000")
PY
}

arm() {  # $1 = name, $2 = extra flags, rest = env assignments
    local name=$1 extra=$2; shift 2
    PORT=$((PORT + 1))
    echo "=== 2. ARM $name: $extra $* ===" | tee -a "$SUM"
    set_launch $PORT
    env NCCL_DEBUG=INFO "$@" timeout 1500 "${LAUNCH[@]}" code/train_fsdp_profile.py $COMMON $extra \
        --output_dir "$OUT/$name" > "$OUT/$name.log" 2>&1 < /dev/null
    local rc=$?
    grep -oE "via [A-Za-z0-9/_]+" "$OUT/$name.log" | sort | uniq -c | sed 's/^/  transport: /' >> "$SUM"
    python3 - "$OUT/$name" "$rc" >> "$SUM" 2>&1 <<'PY'
import json, os, sys, glob
d, rc = sys.argv[1], sys.argv[2]
ps = sorted(glob.glob(d + "/steps_rank*.json"))
if not ps:
    print(f"  FAILED (exit {rc}) -- see {d}.log"); sys.exit()
for p in ps:
    st = json.load(open(p))["steps"]
    print(f"  {os.path.basename(p):<16} loss0={st[0]['loss']:.6f} "
          f"peak_MiB={max(s['peak_mb'] for s in st):.1f} "
          f"step_ms={[round(s['step_ms']) for s in st]}")
pp = os.path.join(d, "probe_rank0.json")
if os.path.exists(pp):
    s = json.load(open(pp))["summary"]
    print(f"  probe rank0: ops={s['n_ops']} " + " ".join(
        f"{k}: med {s[k]['median']:.2f} GB/s (p10 {s[k]['p10']:.2f})" for k in ("d2h", "h2d") if k in s))
PY
    return $rc
}

arm T1_static ""
T1_MS=$(steady_ms "$OUT/T1_static")
PROBE_MS=$(python3 -c "print(round($T1_MS * $LINK_GBPS / 11.0))")
echo "probe target: T1 steady step $T1_MS ms x $LINK_GBPS/11 -> --overlap_probe_ms $PROBE_MS" | tee -a "$SUM"
PROBE="--overlap_probe --overlap_probe_ms $PROBE_MS"
arm T2_probe "$PROBE"
arm T3_offload "--activation_offload"
# The V100 smoke runs T4/T5 regardless, to exercise the code path.
if [ "$P2P_USED" -gt 0 ] || [ -n "$SMOKE" ]; then
    arm T4_static_p2poff "" NCCL_P2P_DISABLE=1
    arm T5_probe_p2poff "$PROBE" NCCL_P2P_DISABLE=1
fi

# ── 3. TRACE ANALYSIS: NCCL duty, copy bandwidth vs overlap, collective side ─
for d in "$OUT"/T*/; do
    name=$(basename "$d")
    echo "=== 3. TRACES: $name (all ranks, steps 2-4) ===" >> "$SUM"
    python3 code/analyze_nccl_duty.py "$d"torch_trace_rank*/*.pt.trace.json* \
        --json_out "$OUT/duty_$name.json" >> "$SUM" 2>&1 || echo "  analysis failed" >> "$SUM"
    echo >> "$SUM"
done
echo "=== DONE: ./l64s_collect.sh $SLURM_JOB_ID ===" >> "$SUM"
cat "$SUM"
