#!/bin/bash -l
#SBATCH --job-name=a30f_tax
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:a30:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=a30f_tax_%j.log
#SBATCH --error=a30f_tax_%j.err
# GPU request is typed --gres only; never add --gpus-per-node (they combine).
#
# ── A30 FOLLOW-UP, JOB A — WHY IS STOCK OFFLOAD FASTER THAN STATIC AT LONG ──
# ── SEQUENCE ON AMPERE?                                                    ──
# 633912 measured offload - static = +0.08 / +0.15 / +0.12 s at x=16/32/64
# and -1.07 / -3.59 s at x=100/150, both repeats, ABBA order: static is the
# slow one (it sits above the quadratic T(x) the job fitted; offload sits
# near it). Only an Ampere node can show it (V100's tax stays positive to
# x=200, 7758). Four candidate causes, one block each:
#   S  sweep x=80/100/120/150, every arm with per-step allocator counters
#      and 1 Hz GPU clock/power/throttle sampling (all hypotheses read it)
#   E  expandable segments OFF (every earlier job ran with it ON)
#   G  offload + a resident ballast equal to static's extra footprint
#   Q  a profiled static/offload pair at x=150 (busy vs idle, by kernel)
# plus M: allocator memory-history snapshots, flash vs mem-efficient at
# x=24, to name the +1.5 MiB/x flash plane's storage.
# All tier-A L=64, lever 3 c=256, fp16 mem-efficient (M: also flash),
# fused-input, cache off, b=1, seed 1234, world 4 on one node.
# WALL CLOCK ~6 h (A30 step times from 633912). Guard: no arm after 22 h.
# SUBMIT: sbatch a30f_tax.sbatch   (README: ClusterA steps)
#
# ── PRE-REGISTERED 2026-10-13 (delta = offload - static, mean of steps 1+) ──
# R1 REPRODUCTION: delta(150) in [-4.6, -2.6] s on both S repeats; delta(100) in
#    [-1.6, -0.6] s. FALSIFIER: either outside -> 633912's sign was a node or
#    session effect; every block below is then read as a null.
# R2 THE FLIP: delta(80) in [-0.8, +0.3] s (between 633912's +0.12 at 64 and
#    -1.07 at 100); delta(120) between delta(100) and delta(150).
# Each hypothesis predicts a signature; the cause is the one whose block removes
# or reproduces >= 70% of delta(150). None doing so leaves the cause OPEN.
# H1 ALLOCATOR (expandable segments map/unmap physical pages as static's working
#    set grows): with expandable segments OFF (E), |delta(150)| <= 1.1 s (70%
#    removed). E is THE test: the driver's counters are partly blind under
#    expandable segments (segment count and cudaFree read 0 on every arm:
#    page mapping is not cudaMalloc), so in S they are
#    context (static 162 vs offload 112 cudaMalloc by step 2 at x=16), and they
#    become a full signature only inside E.
#    (Risk, stated: without expandable segments static x=150 at 21.3 of ~23.5 GB
#    may OOM from fragmentation; an OOM there is recorded, not a defect.)
# H2 FOOTPRINT (kernels run slower with a large resident set): allocator counters
#    alike; offload + ballast (G, 13,784.9 MiB = 633912's static - offload peak
#    at x=150) is slower than plain S offload by >= 2.5 s; at x=100 (9,059.9 MiB)
#    by >= 0.75 s.
# H3 CLOCKS/POWER: mean SM clock during static's x=150 steps >= 0.7% below
#    offload's (the gap is ~1.0% of the step), or a power/thermal throttle reason
#    active in static and not in offload.
# H4 WHERE THE TIME GOES (Q, profiled): under H2/H3 the gap is in compute-busy
#    time (kernels slower); under H1 it is in idle time (span - busy) or in
#    'other' kernels. This block separates H1 from H2/H3 independently of E/G.
# M  FLASH PLANE: the peak live set of the flash arm holds exactly one more
#    1,536 B/token block than its mem-efficient twin (36.0 MiB at x=24); the
#    reader names it from its allocation stack (tools/peak_liveset.py).
#
# FAILURE CLASSES: (0) node/socket -> re-run; (1) an OOM in E static x=150 ->
#    fragmentation without expandable segments, recorded; (2) any other OOM ->
#    a defect; (3) a ballast arm OOM -> the ballast was over-sized: record.
set -uo pipefail
JOBNAME=tax
SMOKE=${A30F_SMOKE:-}
DEADLINE=$((22 * 3600))
source ./a30f_common.sh

if [ -z "$SMOKE" ]; then
    # ── S. sweep (ABBA within each x; two repeats at x=150) ──
    arm S static  fp16 mem 64 80  5 1;  arm S offload fp16 mem 64 80  5 1
    arm S offload fp16 mem 64 100 5 1;  arm S static  fp16 mem 64 100 5 1
    arm S static  fp16 mem 64 120 4 1;  arm S offload fp16 mem 64 120 4 1
    arm S offload fp16 mem 64 150 4 1;  arm S static  fp16 mem 64 150 4 1
    arm S static  fp16 mem 64 150 4 2;  arm S offload fp16 mem 64 150 4 2
    reader tax
    # ── G. ballast (static's extra resident bytes, 633912) ──
    arm G offload fp16 mem 64 150 4 0 --hbm_ballast_mb 13784.904 --hbm_ballast_chunks 63
    arm G offload fp16 mem 64 100 5 0 --hbm_ballast_mb 9059.905  --hbm_ballast_chunks 63
    # ── E. expandable segments OFF (allocator default) ──
    ARM_ENV="PYTORCH_CUDA_ALLOC_CONF="
    arm E static  fp16 mem 64 100 5 0;  arm E offload fp16 mem 64 100 5 0
    arm E offload fp16 mem 64 150 4 0;  arm E static  fp16 mem 64 150 4 0
    ARM_ENV=
    # ── Q. profiled pair at x=150 (2 active profiled steps) ──
    arm Q static  fp16 mem 64 150 4 0 --torch_profile
    arm Q offload fp16 mem 64 150 4 0 --torch_profile
    # ── M. flash-plane snapshots (last step recorded) ──
    arm M static fp16 flash 64 24 3 0 --cuda_snapshot
    arm M static fp16 mem   64 24 3 0 --cuda_snapshot
    QSTEPS=2
else
    # ── V100 SMOKE: every code path at small x (flash cannot run on sm_70) ──
    arm S static  fp16 mem 64 16 3 1;  arm S offload fp16 mem 64 16 3 1
    arm S offload fp16 mem 64 24 3 1;  arm S static  fp16 mem 64 24 3 1
    reader tax
    arm G offload fp16 mem 64 24 3 0 --hbm_ballast_mb 1877.905 --hbm_ballast_chunks 63
    ARM_ENV="PYTORCH_CUDA_ALLOC_CONF="
    arm E static  fp16 mem 64 24 3 0;  arm E offload fp16 mem 64 24 3 0
    ARM_ENV=
    arm Q static  fp16 mem 64 16 4 0 --torch_profile
    arm Q offload fp16 mem 64 16 4 0 --torch_profile
    arm M static fp16 mem 64 16 3 0 --cuda_snapshot
    QSTEPS=2
fi

# ── profiled-arm analysis ──
for d in "$OUT"/Q_*/; do
    [ -d "$d" ] || continue
    TR=$(ls -t "$d"torch_trace_rank0/*.pt.trace.json* 2>/dev/null | head -n 1)
    [ -n "$TR" ] || continue
    log "=== Q: $(basename "$d") ==="
    python3 code/tools/attn_recompute_share.py "$TR" --layers 64 --steps $QSTEPS --json "${d}attn_share.json" 2>&1 | tail -15 | tee -a "$SUM"
done
# ── flash-plane snapshots ──
for d in "$OUT"/M_*/; do
    [ -f "${d}cuda_snapshot_rank0.pkl" ] || continue
    X=$(basename "$d" | sed -E 's/.*_x([0-9]+)_.*/\1/')
    log "=== M: $(basename "$d") ==="
    python3 code/tools/peak_liveset.py "${d}cuda_snapshot_rank0.pkl" $((X * 1024)) > "${d}liveset.txt" 2>&1
    head -40 "${d}liveset.txt" | tee -a "$SUM"
done
reader tax
log "=== DONE in $((SECONDS / 60)) min: ./a30f_collect.sh tax $SLURM_JOB_ID ==="
