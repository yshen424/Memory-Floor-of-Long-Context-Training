"""The capped capacity model against job 7760.

A REGRESSION test of the V100 capacity fit (alpha 0.163, delta 33.1 MiB), which was
fitted on these ladders together with the uncapped brackets; the prediction test
of capping is on the other platforms. Checks: every completing rung on its harness
line (0.05 MiB), every failure exactly 4x MiB, and every ladder's bracket exact
under cap - delta - alpha*request.

Run:  PYTHONPATH=. python3 tests/test_harness_capped.py
"""
import glob
import json
import os
import re
import sys

from harness import (PLATFORMS, Run, TIER_A, V100_CAP_DELTA_MIB, bracket, capped_model,
                     floor_line, static_line)

V = PLATFORMS['v100']
RUN = Run(shape=TIER_A, world=4)
LINES = {'static': static_line(RUN, V), 'offload': floor_line(RUN, V, 0)}


def main():
    lad = {}
    for d in glob.glob('profile_results/capped4/7760-*'):
        m = re.search(r'_([ABCD])_cap(\d+)_(static|offload)_x(\d+)_', d)
        if not m:
            continue
        p = os.path.join(d, 'steps_rank0.json')
        rec = {'x': int(m.group(4)), 'fit': os.path.exists(p)}
        if rec['fit']:
            st = json.load(open(p))['steps']
            rec['peak'] = max(s['peak_mb'] for s in st[1:])
        else:
            e = open(os.path.join(d, 'stderr.txt'), errors='replace').read()
            r = re.findall(r'Tried to allocate ([\d.]+) MiB', e)
            rec['req'] = float(r[-1]) if r else None
        lad.setdefault((m.group(1), int(m.group(2)), m.group(3)), []).append(rec)
    if not lad:
        sys.exit('no 7760 arms found under profile_results/capped4')
    ok = True
    for (blk, cap, mode), rr in sorted(lad.items()):
        ln, model = LINES[mode], capped_model(cap, V100_CAP_DELTA_MIB, PLATFORMS['v100'].capacity.alpha)
        rr.sort(key=lambda r: r['x'])
        for r in rr:
            if r['fit'] and abs(r['peak'] - ln.at(r['x'])) > 0.05:
                print(f"  FAIL  {blk} x={r['x']} peak {r['peak']:.3f} off its line"); ok = False
            if not r['fit'] and (r['req'] is None or abs(r['req'] - 4.0 * r['x']) > 0.01):
                print(f"  FAIL  {blk} x={r['x']} failing request {r['req']} != {4.0 * r['x']}"); ok = False
        _, pf, _ = bracket(ln, RUN, model, [r['x'] for r in rr])
        mf = max(r['x'] for r in rr if r['fit'])
        good = pf == mf
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  ladder {blk} (cap {cap} GiB, {mode}): measured largest fit {mf}, predicted {pf}")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
