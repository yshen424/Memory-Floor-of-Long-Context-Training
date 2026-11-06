#!/bin/bash
#SBATCH --job-name=fsdp_v11_own
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1        # 1 process per node (1 GPU per node)
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/v11own_%j.log
#SBATCH --error=logs/v11own_%j.err

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  V11 — WHAT A PER-MODULE LAYER OWNS AND WHAT IT CAN MOVE, AND THE LOSS'S ║
# ║  BYTES AGAINST VOCABULARY. V100, world 4.                                ║
# ║                                                                          ║
# ║  H  The per-module unit's two bundles per layer OWN 22.05 MiB per 1024   ║
# ║     tokens over the step, of which only the inbound residual (1.50 per   ║
# ║     bundle, 3.00 per layer) is reachable for offloading. Ownership is    ║
# ║     readable only on a bundle that keeps its saves, so block H pairs a   ║
# ║     RETAINING per-module arm (the instrument live, no forced action,     ║
# ║     below its budget) with its recompute twin (--force_action recompute, ║
# ║     everything else equal).                                              ║
# ║  V  With the whole loss, the head and loss hold 14 B per token per       ║
# ║     vocabulary entry at the peak (437.50 MiB/x at v = 32000). Block V    ║
# ║     doubles the vocabulary and reads the slope.                          ║
# ║                                                                          ║
# ║  Tier-A L=64 (h 768, I 2048, 12/4 heads), WHOLE loss (no --chunked_ce),  ║
# ║  regime B fp16/sdpa_mem_efficient, fused-input, cache off, b=1, seed     ║
# ║  1234, 3 steps per arm (failures can land in step 2).                    ║
# ║    C  whole-layer static, v=32000: x = 4         identity control        ║
# ║    H  per-module, v=32000: retain and recompute at x = 4, then x = 2     ║
# ║    V  whole-layer static, v=64000: x = 2, 4, 8                           ║
# ║  Retain = --placement_engine --engine_live --hbm_budget_gb 11.0          ║
# ║  --prefetch_window_layers 2 --evict_layout_census, no --layer_checkpoint ║
# ║  and no --force_action. Recompute twin = the same plus --force_action    ║
# ║  recompute.                                                              ║
# ║  WALL CLOCK ~1 h (8 arms).                                               ║
# ║  SUBMIT WITH:  sbatch run_v11_pm_owned_vocab_4gpu.sh                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── THE PREDICTIONS. PRE-REGISTERED 2026-11-05 ───────────────────────────────
# Peak = max over steps 1+ (MiB); x = tokens/1024. Per-rank activations are
# the same at every world size at b=1, so every byte count below is exact.
#   C  7744 block U measured 3561.99 at x=4 (step 1+), 2691.93 (step 0).
#      C-P1: within ±0.05 MiB, or the build moved and the job is void.
#   H  (decisions) H-P0: on both retain arms every decision is 'retain' with
#      note 'no_pressure', on every bundle, every step. A decision under
#      pressure voids that arm's ownership reading (an evicted bundle's
#      owned bytes are untrusted by construction).
#      H-P1 (THE DECIDER for 22.05): on both retain arms all 128 bundles
#        report owned_bytes_trusted, and per layer
#          attention owns 6.046875·x MiB  (12,681,216 B at x=2; 25,362,432 at x=4)
#          mlp       owns 16.000000·x MiB (33,554,432 B at x=2; 67,108,864 at x=4)
#        = 22.046875 MiB per layer per 1024 tokens, to within 64 B per
#        bundle (a few bytes of non-activation residue are allowed).
#      H-P2 (THE DECIDER for 3.00): every bundle carries exactly one foreign
#        storage of 1.5·x MiB, the inbound residual (3,145,728 B at x=2;
#        6,291,456 B at x=4): 3.00 MiB per layer per 1024 tokens movable.
#      H-P3 (the allocator cross-check): retain − recompute peak =
#        22.046875 × 64 × x = 1411.0000·x MiB (2822.00 at x=2, 5644.00 at
#        x=4), within ±0.5 MiB. Both arms peak in the loss's backward with
#        the same fixed bytes, so the intercepts are equal.
#      H-P4: slopes over x = 2, 4: recompute twin 440.77 + 64 × 6.0078125 =
#        825.27; retain 825.27 + 1411.00 = 2236.27; each within ±0.05
#        (440.77 = 7744 U's 536.77 less 64 boundaries; 6.0078125 = 7938).
#      H-P5: both intercepts 1414.92 ± 1, 7744 U's whole-loss intercept: the
#        fixed bytes at the loss's backward do not depend on the unit. So
#        recompute x=2 3065.46, x=4 4716.00; retain x=2 5887.46, x=4 10360.00.
#   V  v = 64000, whole-layer static, whole loss.
#      V-P1 (THE DECIDER for 437.50 and the vocabulary proportionality):
#        slope over x = 2, 4, 8 = 536.77 + 14 B × 32000 × 1024 / 2^20 =
#        536.77 + 437.50 = 974.27, within ±0.05, the three points on one
#        line to 0.05 MiB.
#      V-P2: intercept 1414.92 + 281.25 = 1696.17 ± 0.5. The 32000 added
#        vocabulary entries add 2 × 32000 × 768 parameters at 6 B each
#        (sharded fp32 master and AdamW 3 B, the root unit's unsharded fp16
#        copy 2 B, the LM head's fp16 gradient 1 B on average). SECOND
#        OUTCOME, registered with it: 1649.30 (= 1696.17 − 46.875) if the
#        LM head's gradient is not yet live at the loss's backward. Any
#        other intercept leaves V-P1 standing.
# Failure classes: (0) node failure / socket timeout / rendezvous — re-run,
#   closes nothing; (1) any OOM — no arm is near capacity (largest predicted
#   peak about 10.4 GB at H retain x=4), so an OOM is a defect, not a
#   bracket; (2) a '[PinnedPool] refusing to pin' error is a pool-sizing
#   defect, never a result; (3) a NaN loss, or step-0 losses differing
#   between arms of the same x and vocabulary, voids that arm.

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

OUTROOT=profile_results/pm_owned4
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

# block:mode:x:vocab
ARMS="C:static:4:32000 H:retain:4:32000 H:recompute:4:32000 H:retain:2:32000 H:recompute:2:32000 V:static:2:64000 V:static:4:64000 V:static:8:64000"

WS=4; NL=64; HID=768; INTER=2048; NH=12; NKV=4
DT=fp16; ATTN=sdpa_mem_efficient; NORM=input; SEED=1234; NSTEPS=3; BS=1
ENGINE="--placement_engine --engine_live --hbm_budget_gb 11.0 --prefetch_window_layers 2 --evict_layout_census"
IDX=0
NARM=$(echo $ARMS | wc -w)

for ARM in $ARMS; do
    OLDIFS=$IFS; IFS=:; set -- $ARM; IFS=$OLDIFS
    BLOCK=$1; MODE=$2; X=$3; VOC=$4; SQ=$((X * 1024))
    case "$MODE" in
        static) MODE_FLAGS="--layer_checkpoint" ;;
        retain) MODE_FLAGS="${ENGINE}" ;;
        recompute) MODE_FLAGS="${ENGINE} --force_action recompute" ;;
    esac
    TAG=V11${BLOCK}_L${NL}_h${HID}_${MODE}_x${X}_v${VOC}_n${NSTEPS}_c0
    OUTDIR=${OUTROOT}/${SLURM_JOB_ID}${ASUF}-${TAG}_b${BS}_${DT}_mem_efficient_world${WS}
    mkdir -p "$OUTDIR"
    IDX=$((IDX + 1))
    echo "ARM ${IDX}/${NARM} block ${BLOCK} ${MODE} x=${X} v=${VOC} -> ${OUTDIR}"
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
            ${MODE_FLAGS} \
            --fused_rmsnorm ${NORM} --use_cache off --seed ${SEED} \
            --disable_hbm_drift --output_dir "$OUTDIR" \
            < /dev/null 2> >(tee "$OUTDIR/stderr.txt" >&2) \
        || echo "ARM ${IDX} (block ${BLOCK}, ${MODE}, x=${X}, v=${VOC}) DID NOT COMPLETE — read the failure class in the header."
    sleep 20
done

echo "RUN COMPLETE: ran ${IDX} of ${NARM} arms."
# ── Reader: every registered check ───────────────────────────────────────────
python3 - "$OUTROOT" "${SLURM_JOB_ID}${ASUF}" <<'PY'
import glob, json, os, re, sys
from collections import Counter
root, job = sys.argv[1], sys.argv[2]
MIB = 1048576
arms = {}
for d in sorted(glob.glob(os.path.join(root, job + '-V11*'))):
    m = re.search(r'-V11([CHV])_L64_h768_(static|retain|recompute)_x(\d+)_v(\d+)_', d)
    blk, mode, x, v = m[1], m[2], int(m[3]), int(m[4])
    f = os.path.join(d, 'steps_rank0.json')
    if not os.path.exists(f):
        print(f"{blk} {mode:9} x={x} v={v}: NO STEPS FILE — read stderr.txt")
        continue
    st = json.load(open(f))['steps']
    p1 = max(s['peak_mb'] for s in st[1:]) if len(st) > 1 else float('nan')
    arms[(blk, mode, x, v)] = dict(dir=d, p1=p1, p0=st[0]['peak_mb'], loss0=st[0]['loss'])
    print(f"{blk} {mode:9} x={x} v={v}: step1+ {p1:10.4f}  step0 {st[0]['peak_mb']:10.4f}  loss0 {st[0]['loss']:.6f}")

def ok(flag): return 'ok' if flag else 'MISS'

a = arms.get(('C', 'static', 4, 32000))
if a:
    print(f"\nC-P1: {a['p1']:.4f} vs 3561.99 ({a['p1'] - 3561.99:+.4f}) -> {ok(abs(a['p1'] - 3561.99) <= 0.05)}")

for x in (4, 2):
    a = arms.get(('H', 'retain', x, 32000))
    if not a:
        continue
    pj = os.path.join(a['dir'], 'profile_rank0.json')
    if not os.path.exists(pj):
        print(f"H x={x}: no profile_rank0.json"); continue
    ps = json.load(open(pj))
    dec = ps.get('placement_decisions') or []
    notes = Counter((e.get('action'), (e.get('reason') or {}).get('note')) for e in dec)
    print(f"\nH x={x} H-P0 decisions: {dict(notes)} -> {ok(set(notes) == {('retain', 'no_pressure')})}")
    ob = ps['placement_summary']['observed_bundle_bytes']
    want = {'attention': round(6.046875 * x * MIB), 'mlp': round(16.0 * x * MIB)}
    trusted = sum(1 for b in ob.values() if b.get('owned_bytes_trusted'))
    bad = [k for k, b in ob.items() if abs(b['owned_bytes'] - want[k.split('_', 1)[1]]) > 64]
    per_layer = sum(b['owned_bytes'] for b in ob.values()) / (len(ob) / 2) / MIB / x
    print(f"H x={x} H-P1: {len(ob)} bundles, {trusted} trusted, {len(bad)} off the registered bytes; "
          f"owned {per_layer:.6f} MiB/layer/x -> {ok(len(ob) == 128 and trusted == 128 and not bad)}")
    inb = round(1.5 * x * MIB)
    have = sum(1 for b in ob.values() if sum(n for k, n in b.get('foreign_storages', {}).items()
                                            if int(k.split('|')[0]) == inb) == 1)
    print(f"H x={x} H-P2: {have} of {len(ob)} bundles carry exactly one {inb} B inbound storage -> {ok(have == len(ob))}")
    r = arms.get(('H', 'recompute', x, 32000))
    if r:
        gap = a['p1'] - r['p1']
        print(f"H x={x} H-P3: retain - recompute = {gap:.4f} vs {1411.0 * x:.2f} ({gap - 1411.0 * x:+.4f}) "
              f"-> {ok(abs(gap - 1411.0 * x) <= 0.5)}")

for mode, want in (('recompute', 825.27), ('retain', 2236.27)):
    p2, p4 = arms.get(('H', mode, 2, 32000)), arms.get(('H', mode, 4, 32000))
    if p2 and p4:
        s = (p4['p1'] - p2['p1']) / 2
        c = p2['p1'] - 2 * s
        print(f"H-P4 {mode}: slope {s:.4f} vs {want} ({s - want:+.4f}) -> {ok(abs(s - want) <= 0.05)}")
        print(f"H-P5 {mode}: intercept {c:.4f} vs 1414.92 ({c - 1414.92:+.4f}) -> {ok(abs(c - 1414.92) <= 1.0)}")

pts = [(x, arms[('V', 'static', x, 64000)]['p1']) for x in (2, 4, 8) if ('V', 'static', x, 64000) in arms]
if len(pts) >= 2:
    n = len(pts); mx = sum(x for x, _ in pts) / n; my = sum(y for _, y in pts) / n
    s = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)
    c = my - s * mx
    res = max(abs(y - c - s * x) for x, y in pts)
    print(f"\nV-P1: slope {s:.4f} vs 974.27 ({s - 974.27:+.4f}), max residual {res:.4f} over {n} points "
          f"-> {ok(abs(s - 974.27) <= 0.05 and res <= 0.05)}")
    print(f"      head + loss per vocabulary entry: {(s - 536.77) * MIB / (32000 * 1024):.4f} B/token "
          f"(registered 14.0000)")
    tag = 'ok (LM-head gradient live)' if abs(c - 1696.17) <= 0.5 else (
          'ok, second outcome (LM-head gradient not live)' if abs(c - 1649.30) <= 0.5 else 'MISS')
    print(f"V-P2: intercept {c:.4f} vs 1696.17 / 1649.30 -> {tag}")
PY
echo "AFTER THE JOB: append one row per arm to profile_results/runs_4gpu.csv"
echo "(back it up as runs_4gpu.csv.bak_pre<job> first) and move this"
echo "script to ${OUTROOT}/."
