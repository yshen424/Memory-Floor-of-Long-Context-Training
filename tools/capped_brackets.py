"""Read the capped-bracket job (run_l64_capped_4gpu.sh) against its
pre-registration.

Per ladder: each rung's measured peak against the harness line (C-P1), the
failing rung's request against 4x MiB (C-P2), the bracket against the predicted
range (C-P3). Then the reserve: each ladder bounds
    delta <= cap - peak(largest fit) - alpha*req(largest fit)
    delta >  cap - peak(smallest fail) - alpha*req(smallest fail)
at alpha = 0.198 (C-P4: one common delta), and the joint (alpha, delta) region
over all four ladders.

Usage:
    python3 tools/capped_brackets.py profile_results/capped4 --job <JOBID>
"""
import argparse
import glob
import json
import os
import re
import sys

from harness import PLATFORMS, Run, TIER_A, floor_line, static_line

ALPHA = 0.198
RUN = Run(shape=TIER_A, world=4)
V = PLATFORMS['v100']
LINES = {'static': static_line(RUN, V), 'offload': floor_line(RUN, V, 0)}
PRED = {'A': (31.5, 34.6), 'B': (54.5, 57.7), 'C': (98.4, 109.2), 'D': (178.0, 188.8)}


def read(root, job):
    pat = re.compile(r'_([ABCD])_cap(\d+)_(static|offload)_x(\d+)_')
    out = {}
    for d in glob.glob(os.path.join(root, f'{job}-*')):
        m = pat.search(os.path.basename(d))
        if not m:
            continue
        blk, cap, mode, x = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
        p = os.path.join(d, 'steps_rank0.json')
        rec = {'cap': cap, 'mode': mode, 'x': x, 'fit': os.path.exists(p)}
        if rec['fit']:
            st = json.load(open(p))['steps']
            rec['peak'] = max(s['peak_mb'] for s in st[1:]) if len(st) > 1 else st[0]['peak_mb']
            rec['steps'] = len(st)
        err = os.path.join(d, 'stderr.txt')
        if not rec['fit'] and os.path.exists(err):
            m2 = re.findall(r'Tried to allocate ([\d.]+) (MiB|GiB)', open(err, errors='replace').read())
            if m2:
                v, u = m2[-1]
                rec['request'] = float(v) * (1024 if u == 'GiB' else 1)
        out.setdefault(blk, []).append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--job', required=True)
    a = ap.parse_args()
    lad = read(a.root, a.job)
    if not lad:
        sys.exit('no arms found')
    bounds = []
    for blk in sorted(lad):
        rr = sorted(lad[blk], key=lambda r: r['x'])
        cap, mode = rr[0]['cap'], rr[0]['mode']
        ln = LINES[mode]
        print(f"--- ladder {blk}: cap {cap} GiB, {mode}")
        for r in rr:
            pred = ln.at(r['x'])
            if r['fit']:
                ok = abs(r['peak'] - pred) <= 0.05
                print(f"  x={r['x']:>4} fit   peak {r['peak']:10.3f} (line {pred:10.3f}, {r['peak'] - pred:+.3f}) {'' if ok else '<-- C-P1 MISS'}")
            else:
                rq = r.get('request')
                ok = rq is not None and abs(rq - 4.0 * r['x']) <= 0.01
                print(f"  x={r['x']:>4} FAIL  request {rq} MiB (4x = {4.0 * r['x']:.2f}) {'' if ok else '<-- C-P2: different mechanism, read stderr.txt'}")
        fits = [r for r in rr if r['fit']]
        fails = [r for r in rr if not r['fit']]
        if fits and fails:
            lf, sf = max(fits, key=lambda r: r['x']), min(fails, key=lambda r: r['x'])
            lo_p, hi_p = PRED[blk]
            print(f"  bracket [{lf['x']}, {sf['x']})  predicted largest fit in [{lo_p}, {hi_p}] -> "
                  f"{'C-P3 ok' if lo_p - 1 <= lf['x'] <= hi_p + 1 else 'C-P3 MISS'}")
            capm = cap * 1024.0
            d_hi = capm - ln.at(lf['x']) - ALPHA * 4.0 * lf['x']
            d_lo = capm - ln.at(sf['x']) - ALPHA * 4.0 * sf['x']
            print(f"  delta in ({d_lo:.1f}, {d_hi:.1f}] MiB at alpha = {ALPHA}")
            bounds.append((blk, capm, ln, lf['x'], sf['x'], d_lo, d_hi))
        else:
            print('  no bracket (ladder did not close, or closed on its first rung)')
    if not bounds:
        return
    lo, hi = max(b[5] for b in bounds), min(b[6] for b in bounds)
    print(f"\nC-P4 common delta at alpha={ALPHA}: "
          + (f"({lo:.1f}, {hi:.1f}] MiB -> HOLDS" if lo < hi else f"none (max lower {lo:.1f} >= min upper {hi:.1f}) -> FALSIFIED"))
    region = []
    for a1000 in range(0, 601):
        al = a1000 / 1000
        l = max(c - ln.at(sx) - al * 4.0 * sx for _, c, ln, fx, sx, _, _ in bounds)
        h = min(c - ln.at(fx) - al * 4.0 * fx for _, c, ln, fx, sx, _, _ in bounds)
        if l < h:
            region.append((al, l, h))
    if region:
        print(f"joint region: alpha in [{region[0][0]:.3f}, {region[-1][0]:.3f}] "
              f"(uncapped V100 feasible 0.000-0.396, chosen 0.198)")
    else:
        print("joint region: EMPTY — no single (alpha, delta) fits all four ladders")


if __name__ == '__main__':
    main()
