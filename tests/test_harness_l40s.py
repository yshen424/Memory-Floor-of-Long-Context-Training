"""The harness against every measured L40S Llama-3.2-3B peak (630188, 634078).

Covers static, stock offload, static + FSDP CPU offload and offload + CPU offload
at the 3B shape (h 3072, I 8192, 24/8 heads, L=28, v 128256), x = 16-258, on
PLATFORMS['l40s'] (V100 workspace carried). Pins the deep-replay instant: static
x=16 sits on it, not on the second replay. The tier-A L40S arms are
the job reader's (tools/remote_frontier.py) and are not repeated here.

Run:  PYTHONPATH=. python3 tests/test_harness_l40s.py
"""
import glob
import json
import os
import re
import sys

from harness import PLATFORMS, LLAMA32_3B, Run, floor_line, static_line

TOL = 0.05
P = PLATFORMS['l40s']
rows, fails = [], []
for d in sorted(glob.glob('profile_results/l40s4/*/R*_llama3b_*')):
    p = os.path.join(d, 'steps_rank0.json')
    if not os.path.exists(p):
        continue                                    # a ladder's closing rung
    m = re.search(r'_(static|offload|staticcpu|offloadcpu)_fp16_mem_llama3b_L(\d+)_x(\d+)_', os.path.basename(d))
    mode, L, x = m[1], int(m[2]), int(m[3])
    run = Run(shape=LLAMA32_3B.with_layers(L), world=4, cpu_offload=mode.endswith('cpu'))
    ln = static_line(run, P) if mode.startswith('static') else floor_line(run, P, 0)
    st = json.load(open(p))['steps']
    meas = max(s['peak_mb'] for s in st[1:])
    rows.append((d.split('/')[2], mode, x, ln.binding(x), meas, ln.at(x), meas - ln.at(x)))
    if abs(meas - ln.at(x)) > TOL:
        fails.append(rows[-1])
for job, mode, x, how, me, pr, df in rows:
    print(f"{job:>7} {mode:<11} x={x:>3} {how:<14} measured {me:10.4f} predicted {pr:10.4f} {df:+.4f}"
          + ('   <-- MISS' if abs(df) > TOL else ''))
deep = [r for r in rows if r[1] == 'static' and r[2] == 16]
assert deep and all(r[3] == 'deep replay' for r in deep), 'static x=16 must bind on the deep-replay instant'
print(f"\n{len(rows)} L40S 3B peaks, {len(fails)} outside ±{TOL} MiB; max |diff| {max(abs(r[6]) for r in rows):.4f}")
sys.exit(1 if fails else 0)
