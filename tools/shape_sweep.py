"""Read a shape sweep (run_v4_shapes_4gpu.sh, run_v4b_oos_shapes_4gpu.sh) against the harness.

Each arm's prediction is rebuilt from the configuration the arm itself recorded
(steps_rank0.json 'config'), so the job and the reader cannot disagree on a
shape. Per (block, mechanism): each point against the harness peak (the largest
of three instants; the instant that binds is printed), then a
least-squares line through the points where the main instant binds (slope against
K-P1's 0.01 MiB/x, intercept against K-P2's 1 MiB). Then the R1 brackets (K-P3) and every failure's request (K-P4).

Ladder blocks are named <family>S (static) / <family>O (stock offload): RS/RO
(7761), QS/QO (7763). The harness hash is printed so a reader can tell
whether harness/ changed after a job's pre-registration (7763 registered
35cea314...).

Usage:
    python3 tools/shape_sweep.py profile_results/shapes4 --job <JOBID>
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import re
import sys

from harness import PLATFORMS, ModelShape, Run, bracket, floor_line, static_line

V = PLATFORMS['v100']
LADDERS = ('RS', 'RO', 'QS', 'QO')
PAT = re.compile(r'-([A-Z]\w*)_L(\d+)_h(\d+)_i(\d+)_q(\d+)_kv(\d+)_v(\d+)_(static|offload)_x(\d+)_')


def line_for(sh, mode):
    run = Run(shape=sh, world=4)
    return run, (static_line(run, V) if mode == 'static' else floor_line(run, V, 0))


def fit(points):
    n = len(points)
    if n < 2:
        return None
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    sxx = sum((x - mx) ** 2 for x, _ in points)
    m = sum((x - mx) * (y - my) for x, y in points) / sxx
    return my - m * mx, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--job', required=True)
    a = ap.parse_args()
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'harness')
    inner = ''.join(hashlib.sha1(open(f, 'rb').read()).hexdigest() + '  harness/' + os.path.basename(f) + '\n'
                    for f in sorted(glob.glob(os.path.join(here, '*.py'))))
    print(f"harness sha1-of-sha1s {hashlib.sha1(inner.encode()).hexdigest()}")
    groups, ladders = collections.defaultdict(list), collections.defaultdict(list)
    for d in glob.glob(os.path.join(a.root, f'{a.job}-*')):
        m = PAT.search(os.path.basename(d))
        if not m:
            continue
        blk, L, h, I, nq, nkv, V_, mode, x = m.groups()
        sh = ModelShape(int(h), int(I), int(nq), int(nkv), int(L), int(V_))
        p = os.path.join(d, 'steps_rank0.json')
        rec = {'x': int(x), 'fit': os.path.exists(p)}
        if rec['fit']:
            j = json.load(open(p))
            c = j['config']
            assert (c['hidden_size'], c['num_layers'], c['vocab_size']) == (sh.hidden, sh.layers, sh.vocab), d
            st = j['steps']
            rec['peak'] = max(s['peak_mb'] for s in st[1:]) if len(st) > 1 else st[0]['peak_mb']
        else:
            e = open(os.path.join(d, 'stderr.txt'), errors='replace').read() if os.path.exists(os.path.join(d, 'stderr.txt')) else ''
            r = re.findall(r'Tried to allocate ([\d.]+) (MiB|GiB)', e)
            rec['req'] = (float(r[-1][0]) * (1024 if r[-1][1] == 'GiB' else 1)) if r else None
            # PyTorch prints 2 decimals: a GiB request is known only to ±0.005 GiB (7763)
            rec['req_tol'] = (0.005 * 1024 if r and r[-1][1] == 'GiB' else 0.01) if r else None
        key = (blk, sh, mode)
        (ladders if blk in LADDERS else groups)[key].append(rec)

    print(f"{'block':5} {'mech':8} {'pred line':>26} {'fitted line':>26} {'d slope':>9} {'d icpt':>8}  points (measured - predicted)")
    bad = []
    for (blk, sh, mode), rr in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][2])):
        run, ln = line_for(sh, mode)
        pts = [(r['x'], r['peak']) for r in sorted(rr, key=lambda r: r['x']) if r['fit']]
        fails = [r for r in rr if not r['fit']]
        f = fit([(x, y) for x, y in pts if ln.binding(x) == 'main'])
        tag = {'main': '', 'second replay': '(2nd)', 'optimizer step': '(opt)', 'end of backward': '(end)'}
        res = ' '.join(f"x{x}{tag[ln.binding(x)]}:{y - ln.at(x):+.3f}" for x, y in pts)
        if any(abs(y - ln.at(x)) > 0.05 for x, y in pts):
            bad.append(blk)
        if f:
            c, m = f
            ds, di = m - ln.m, c - ln.c
            flag = '' if abs(ds) <= 0.01 and abs(di) <= 1.0 and blk not in bad else '  <-- MISS'
            if flag:
                bad.append(blk)
            print(f"{blk:5} {mode:8} {ln.c:10.3f}+{ln.m:10.6f}x {c:10.3f}+{m:10.6f}x {ds:+9.5f} {di:+8.3f}  {res}{flag}")
        else:
            print(f"{blk:5} {mode:8} {ln.c:10.3f}+{ln.m:10.6f}x {'(one main point)':>26} {'':>9} {'':>8}  {res}")
        for r in fails:
            print(f"      {mode} x={r['x']} FAILED (request {r['req']} MiB) — not predicted for a slope block")
            bad.append(blk)

    print()
    for (blk, sh, mode), rr in sorted(ladders.items()):
        run, ln = line_for(sh, mode)
        rr.sort(key=lambda r: r['x'])
        _, pf, pfail = bracket(ln, run, V.capacity, [r['x'] for r in rr])
        mf = max((r['x'] for r in rr if r['fit']), default=None)
        mfail = min((r['x'] for r in rr if not r['fit']), default=None)
        for r in rr:
            if r['fit']:
                print(f"  {blk} {mode} x={r['x']} fit   peak {r['peak']:.3f} ({r['peak'] - ln.at(r['x']):+.3f} vs line)")
            else:
                exp = 2 * sh.intermediate / 1024 * r['x']
                ok = r['req'] is not None and abs(r['req'] - exp) <= r['req_tol']
                print(f"  {blk} {mode} x={r['x']} FAIL  request {r['req']} MiB (expected {exp:.2f}) {'' if ok else '<-- K-P4 MISS'}")
        within = mf is not None and pf is not None and abs(mf - pf) <= 1
        print(f"  {blk}: measured [{mf}, {mfail})  predicted largest fit {pf}  -> {'K-P3 ok' if within else 'K-P3 MISS'}")
    for fam in sorted({b[0] for (b, _, _) in ladders}):
        st = max((r['x'] for (b, _, _), rr in ladders.items() if b == fam + 'S' for r in rr if r['fit']), default=None)
        of = max((r['x'] for (b, _, _), rr in ladders.items() if b == fam + 'O' for r in rr if r['fit']), default=None)
        if st and of:
            print(f"  {fam} ladder span {of / st:.3f} (pre-registered: R 1.306, Q 1.500)")
    print(f"\nblocks outside tolerance: {sorted(set(bad)) if bad else 'none'}")


if __name__ == '__main__':
    main()
