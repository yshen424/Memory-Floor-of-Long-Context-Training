"""Read the regime-curve job (run_l64_regime_4gpu.sh) against its
pre-registered predictions.

For each x: the paired tax of stock offload and of the gated engine over
static (means of steps 1+), the pre-registered band, the engine's evictions,
and the peak checks. Then the 1% and 2% crossings, by linear interpolation
between adjacent measured x.

Usage:
    python3 tools/regime_curve.py profile_results/regime4 --job <JOBID>
"""
import argparse
import glob
import gzip
import json
import os
import re
import statistics
import sys

# Pre-registered model (job header).
A, B, C = 0.02123, 0.0397, 2.066
K = 0.02646
H_LO, H_HI = 0.805, 1.0
from harness import PLATFORMS, Run, TIER_A, static_line, floor_line
_RUN = Run(shape=TIER_A, world=4)
STATIC_LINE = (static_line(_RUN, PLATFORMS['v100']).c, static_line(_RUN, PLATFORMS['v100']).m)
OFFLOAD_LINE = (floor_line(_RUN, PLATFORMS['v100']).c, floor_line(_RUN, PLATFORMS['v100']).m)


def model_tax(x, h):
    return 100.0 * h * K * x / (A * x * x + B * x + C)


def read_arm(d):
    p = os.path.join(d, 'steps_rank0.json')
    if not os.path.exists(p):
        return None
    st = json.load(open(p))['steps']
    ms = [s['step_ms'] / 1000.0 for s in st]
    steady = ms[1:] if len(ms) > 1 else ms
    out = {
        'mean': statistics.mean(steady),
        'sd': statistics.stdev(steady) if len(steady) > 1 else 0.0,
        'peak_max': max(s['peak_mb'] for s in st),
        'peak_steady': max(s['peak_mb'] for s in st[1:]) if len(st) > 1 else st[0]['peak_mb'],
        'loss0': st[0]['loss'],
    }
        # Fresh runs write profile_rank0.json; the artifact ships it gzipped.
    pf = os.path.join(d, 'profile_rank0.json')
    if not os.path.exists(pf) and os.path.exists(pf + '.gz'):
        pf += '.gz'
    if os.path.exists(pf):
        try:
            with (gzip.open(pf, 'rt') if pf.endswith('.gz') else open(pf)) as f:
                ba = json.load(f)['placement_summary']['by_action']
            out['evict'] = ba.get('evict_to_cpu', 0)
        except (KeyError, ValueError):
            pass
    return out


def collect(root, job):
    arms = {}
    pat = re.compile(r'_(R|P|D)_(static|offload|offload_profile|engine)_x(\d+)_n(\d+)_')
    for d in glob.glob(os.path.join(root, f'{job}-*')):
        m = pat.search(os.path.basename(d))
        if not m:
            continue
        blk, mode, x = m.group(1), m.group(2), int(m.group(3))
        arms[(blk, mode, x)] = read_arm(d)
    return arms


def crossing(points, eps):
    """First x (scanning upward from the peak) where tax falls below eps."""
    pts = sorted(points)
    if not pts:
        return None
    ipk = max(range(len(pts)), key=lambda i: pts[i][1])
    for (x0, t0), (x1, t1) in zip(pts[ipk:], pts[ipk + 1:]):
        if t0 >= eps > t1:
            return x0 + (t0 - eps) * (x1 - x0) / (t0 - t1)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--job', required=True)
    a = ap.parse_args()
    arms = collect(a.root, a.job)
    if not arms:
        sys.exit(f'no arms for job {a.job} under {a.root}')

    xs = sorted({x for (b, _, x) in arms if b == 'R'})
    print(f"{'x':>4} {'static s':>10} {'offload tax %':>14} {'band %':>14} "
          f"{'engine tax %':>13} {'evict':>5} {'peak chk':>9}")
    off_pts = []
    verdicts = []
    for x in xs:
        s, o, e = (arms.get(('R', m, x)) for m in ('static', 'offload', 'engine'))
        if not s:
            print(f'{x:>4}  static missing')
            continue
        lo, hi = model_tax(x, H_LO), model_tax(x, H_HI)
        line = f"{x:>4} {s['mean']:>8.2f}±{s['sd']:.2f}"
        if o:
            tax = 100.0 * (o['mean'] / s['mean'] - 1.0)
            noise = 100.0 * (s['sd'] + o['sd']) / s['mean']
            off_pts.append((x, tax))
            line += f" {tax:>8.2f}±{noise:<5.2f}"
            # R-P2 / R-P3 bands
            if x >= 48:
                ok = 0.7 * lo <= tax <= 1.3 * hi
            elif x <= 24:
                ok = lo - noise <= tax <= 2.5 * hi
            else:
                ok = 0.7 * lo <= tax <= 2.5 * hi
            verdicts.append((x, ok))
        else:
            line += f" {'missing':>14}"
        line += f" {lo:>6.2f}-{hi:<6.2f}"
        if e:
            line += f" {100.0 * (e['mean'] / s['mean'] - 1.0):>13.2f} {e.get('evict', '?'):>5}"
        else:
            line += f" {'missing':>13} {'':>5}"
        # R-P6 peaks (x >= 16), steady steps
        chk = []
        if x >= 16:
            chk.append(abs(s['peak_steady'] - (STATIC_LINE[0] + STATIC_LINE[1] * x)) <= 0.01)
            if o:
                chk.append(abs(o['peak_steady'] - (OFFLOAD_LINE[0] + OFFLOAD_LINE[1] * x)) <= 0.01)
        line += f" {('ok' if all(chk) else 'MISS') if chk else '-':>9}"
        if o and e and not (s['loss0'] == o['loss0'] == e['loss0']):
            line += '  LOSS0 DIFFERS'
        print(line)

    print()
    for eps, lo, hi in ((2.0, 40, 65), (1.0, 90, 130)):
        xc = crossing(off_pts, eps)
        if xc is None and off_pts and min(t for _, t in off_pts if _ == max(x for x, _ in off_pts)) >= eps:
            tag = f'beyond the measured range (x > {max(x for x, _ in off_pts)})'
        else:
            tag = 'in band' if xc is not None and lo <= xc <= hi else 'OUT OF BAND'
        print(f"x*({eps:.0f}%) = {xc if xc is None else round(xc, 1)}   pre-registered [{lo}, {hi}] -> {tag}")
    if off_pts:
        xpk = max(off_pts, key=lambda p: p[1])[0]
        print(f"tax peaks at x={xpk}   pre-registered between 4 and 24 -> "
              f"{'ok' if 4 <= xpk <= 24 else 'FALSIFIES R-P1'}")
    bad = [x for x, ok in verdicts if not ok]
    print(f"band checks (R-P2/R-P3): {'all inside' if not bad else 'outside at x=' + str(bad)}")
    for x in (16, 64):
        d, s = arms.get(('D', 'static', x)), arms.get(('R', 'static', x))
        if d and s:
            print(f"drift x={x}: static {s['mean']:.2f} -> {d['mean']:.2f} s "
                  f"({100.0 * (d['mean'] / s['mean'] - 1.0):+.2f}%)")


if __name__ == '__main__':
    main()
