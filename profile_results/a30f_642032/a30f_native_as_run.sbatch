#!/bin/bash -l
#SBATCH --job-name=a30f_native
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --qos=<qos>
#SBATCH --nodes=1
#SBATCH --gres=gpu:a30:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=a30f_native_%j.log
#SBATCH --error=a30f_native_%j.err
# GPU request is typed --gres only; never add --gpus-per-node (they combine).
#
# ── A30 FOLLOW-UP, JOB B — THE NATIVE 24 GB STOCK-OFFLOAD BRACKET ────────────
# 633912 measured the native STATIC bracket [164, 166); this measures stock
# offload's, so the A30 span is native, not carried from capped ladders. One
# ladder, 2 steps per rung (a one-step rung cannot bracket: failures can land
# in step 2).
# Tier-A L=64, lever 3 c=256, fp16 mem-efficient, fused-input, cache off,
# b=1, seed 1234, world 4 on one node.
# STEP TIME ~84 min at x~566 (633912's offload x=150 scaled by x^2):
# ~2.9 h per fitting rung, ~2.1 h per failing one. Expected 2 rungs
# (~5 h); at most 6. A rung starts only if it is predicted to end by
# 23.2 h, using the previous rung's measured step time.
# HOST MEMORY: stock offload pins every boundary but the first on the
# host: 63 x 1.5 MiB/x per rank, ~56 GB per rank at x=566, ~224 GB for the
# node = 44% of its 503 GB. A rung starts only if its predicted pinned
# total is <= 55% of MemTotal, and a watchdog kills any arm whose node
# MemAvailable falls below 30%: the node never drains.
# SUBMIT: sbatch a30f_native.sbatch   (README: ClusterA steps)
#
# ── PRE-REGISTERED 2026-10-13 from 633912 alone (harness/) ──
# Line (derived, A30 workspace = V100's, 633912's offset -0.001 MiB):
#   L=64 stock offload  1866.8919 + 37.820312x   x=564: 23197.5  x=568: 23348.8
# Capacity (calibrated on 633912's four capped ladders AND its native static
#   bracket [164, 166); card total 24164.25 MiB, outside-allocator 618.88 MiB):
#   fits iff peak <= 24164.25 - 618.88 - delta - alpha * 4x, with (alpha, delta)
#   anywhere in the region those five brackets admit (alpha 0-0.186, delta
#   21.5-116.9 MiB); centre alpha 0.093, delta 59.5.
# NO-P1 largest fit x = 566 at the centre; in [561, 570] over the whole region.
#      Ladder step 4 from 564: predicted bracket [564, 568). FALSIFIER: the
#      measured largest fit outside [560, 572] (2+ rungs off the range).
# NO-P2 the failure is a step-1 backward OOM on exactly 4x MiB (2,272 at x=568).
# NO-P3 SPAN at native capacity, stock offload over static (both A30, no cap):
#      564/164 = 3.44 (V100 world 4: 356/104 = 3.42; the capped A30 ladders
#      reproduced V100's to the rung). Slope ratio 132.32/37.82 = 3.499 is the
#      structural ceiling; the native span sits below it by the intercepts.
# NO-P4 step time at the fitting rung within +-15% of 84 min.
# NO-P5 host: the node's lowest MemAvailable stays >= 45% (pinned ~44% + model).
#
# FAILURE CLASSES: (0) node/socket -> re-run; (1) step-1 backward OOM on 4x MiB
#    -> the designed closure; (2) WATCHDOG KILL -> host memory, the ladder stops
#    (not a capacity result); (3) cudaHostAlloc failure -> host limit, recorded.
set -uo pipefail
JOBNAME=native
SMOKE=${A30F_SMOKE:-}
DEADLINE=$((23 * 3600))
HOSTMEM_FLOOR_PCT=30
source ./a30f_common.sh

MEMTOTAL_MIB=$(awk '/MemTotal/{printf "%d", $2/1024}' /proc/meminfo)
if [ -z "$SMOKE" ]; then
    START=564; STEP=4; CAPF=""; RANKS_PER_NODE=4; EST_STEP=5040      # s per step at START
else
    # V100 smoke: the same ladder logic on 7760's capped ladder C (6 GiB, offload,
    # [110, 112)), one rank per node.
    START=110; STEP=2; CAPF="--hbm_cap_gb 6"; RANKS_PER_NODE=1; EST_STEP=330
fi
LIMIT=$((23 * 3600 + 720))                                           # a rung must END by 23.2 h
X=$START; DIR=0; LAST_FIT=""; FIRST_FAIL=""; PREV_X=""; RUNGS=0
while [ "$RUNGS" -lt 6 ]; do
    # time guard: ~2.2 steps + startup, scaled from the last measured rung by x^2
    if [ -n "$PREV_X" ]; then EST_STEP=$(( EST_STEP * X * X / (PREV_X * PREV_X) )); fi
    EST=$(( EST_STEP * 22 / 10 + 300 ))
    if [ $((SECONDS + EST)) -gt "$LIMIT" ]; then log "STOP before x=$X: predicted to end after 23.2 h (est ${EST} s)"; break; fi
    # host guard: 63 boundaries x 1.5 MiB/x pinned per rank
    PIN=$(( 63 * 3 * X / 2 * RANKS_PER_NODE ))
    if [ $((PIN * 100)) -gt $((MEMTOTAL_MIB * 55)) ]; then log "STOP before x=$X: predicted pinned ${PIN} MiB > 55% of host ${MEMTOTAL_MIB} MiB"; break; fi
    log "RUNG x=$X: predicted pinned ${PIN} MiB per node ($((PIN * 100 / MEMTOTAL_MIB))% of host), est ${EST} s"
    arm NO offload fp16 mem 64 "$X" 2 0 $CAPF
    RUNGS=$((RUNGS + 1))
    if [ "$ARM_HOSTKILL" = 1 ]; then log "LADDER STOPPED: host-memory watchdog at x=$X (not a capacity result)"; break; fi
    STEP0=$(python3 -c "import json,sys; s=json.load(open(sys.argv[1]))['steps']; print(int(s[0]['step_ms']/1000))" \
            "$OUT/NO_offload_fp16_mem_L64_x${X}_n2/steps_rank0.json" 2>/dev/null)
    [ -n "$STEP0" ] && EST_STEP=$STEP0
    PREV_X=$X
    if [ "$ARM_OK" = 1 ]; then
        LAST_FIT=$X
        [ -n "$FIRST_FAIL" ] && break                                # descended to a fit: bracketed
        DIR=1; X=$((X + STEP))
    else
        FIRST_FAIL=$X
        [ -n "$LAST_FIT" ] && break                                  # climbed to a fail: bracketed
        DIR=-1; X=$((X - STEP))
    fi
done
log "NATIVE OFFLOAD LADDER: largest fit ${LAST_FIT:-none}, smallest fail ${FIRST_FAIL:-none}"
reader native
log "=== DONE in $((SECONDS / 60)) min: ./a30f_collect.sh native $SLURM_JOB_ID ==="
