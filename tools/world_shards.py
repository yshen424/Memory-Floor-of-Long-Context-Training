"""Read the V7 world-size job (run_v7_world_shards.sh) against its pre-registration.

Per arm: the step-0, step-1 and step-2 peaks. Step 1+ against the harness line
(W-P2), steps 1 and 2 against each other (W-P4), and step 1 minus step 0, read
as AdamW (8 B·P/W) plus k fp32 gradient shards (4 B·p_layer/W each): k = W − 1
on stock offload (W-P1, H-W1) and k = 0 on static (W-P3).

Usage:
    PYTHONPATH=. python3 tools/world_shards.py profile_results/world_shards --job <JOBID>
"""
import argparse
import glob
import json
import os
import re

from harness import PLATFORMS, Run, TIER_A, floor_line, param_counts, static_line

V = PLATFORMS['v100']
MiB = 2 ** 20
PAT = re.compile(r'V7_L64_h768_(static|offload)_x(\d+)_n(\d+)_c256_b1_fp16_mem_efficient_world(\d+)$')
TOL = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--job', required=True)
    a = ap.parse_args()
    pl, _, pt = param_counts(TIER_A)
    arms = []
    for d in sorted(glob.glob(os.path.join(a.root, f'{a.job}-*'))):
        m = PAT.search(os.path.basename(d))
        if not m:
            continue
        mode, x, n, W = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        p = os.path.join(d, 'steps_rank0.json')
        st = json.load(open(p))['steps'] if os.path.exists(p) else None
        arms.append((W, mode, x, n, st, d))
    if not arms:
        raise SystemExit('no arms found')
    print(f"{'W':>2} {'mode':8} {'x':>4} {'n':>2} {'step0':>11} {'step1':>11} {'step2':>11} "
          f"{'s1 - line':>10} {'s2 - s1':>8} {'s1 - s0':>10} {'k shards':>9} {'expect k':>8}")
    bad, ks = [], {}
    for W, mode, x, n, st, d in sorted(arms):
        if st is None:
            print(f"{W:>2} {mode:8} {x:>4} {n:>2}  DID NOT COMPLETE — read {d}/stderr.txt")
            bad.append((W, mode, x, 'incomplete'))
            continue
        run = Run(shape=TIER_A, world=W)
        ln = static_line(run, V) if mode == 'static' else floor_line(run, V, 0)
        pk = [s['peak_mb'] for s in st]
        if n == 1:          # the step-0 snapshot arm: step 0 only
            print(f"{W:>2} {mode:8} {x:>4} {n:>2} {pk[0]:11.3f}   (one-step snapshot arm: step 0 only)")
            continue
        adam, shard = 8 * pt / W / MiB, 4 * pl / W / MiB
        d1 = pk[1] - ln.at(x)
        d21 = pk[2] - pk[1] if len(pk) > 2 else float('nan')
        diff = pk[1] - pk[0]
        k = (diff - adam) / shard
        exp = W - 1 if mode == 'offload' else 0
        flags = []
        if abs(d1) > TOL:
            flags.append('W-P2')
        if len(pk) > 2 and abs(d21) > 0.01:
            flags.append('W-P4')
        if abs(k - exp) * shard > TOL:
            flags.append('W-P1' if mode == 'offload' else 'W-P3')
        bad += [(W, mode, x, f) for f in flags]
        ks.setdefault((mode, W), []).append(k)
        print(f"{W:>2} {mode:8} {x:>4} {n:>2} {pk[0]:11.3f} {pk[1]:11.3f} {pk[2] if len(pk) > 2 else float('nan'):11.3f} "
              f"{d1:+10.3f} {d21:+8.3f} {diff:10.3f} {k:9.3f} {exp:>8}" + (f"   <-- {', '.join(flags)} MISS" if flags else ''))
    print()
    for (mode, W), v in sorted(ks.items()):
        print(f"  {mode:8} world {W}: k = {', '.join(f'{x:.3f}' for x in v)}  (H-W1: {W - 1 if mode == 'offload' else 0})")
    print(f"\n{'ALL HELD' if not bad else 'MISSES: ' + str(bad)}")


if __name__ == '__main__':
    main()
