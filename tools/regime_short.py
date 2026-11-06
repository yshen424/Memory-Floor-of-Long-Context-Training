"""Read the short-end job (run_l64_regime_short_4gpu.sh) against its
pre-registered hypotheses.

Per x: steps 2+ of both repeats pooled per mode; mean, median and 10%-trimmed
mean of the step time; delta = offload - static with its standard error;
delta/x; the tax. Then the decider S-P2 (mean delta/x over x = 2..12 against
0.032 ± 25%), the shape check S-P3 (tax at x=2 below x=8) and the peaks S-P5.

Usage:
    python3 tools/regime_short.py profile_results/regime4_short --job <JOBID>
"""
import argparse
import glob
import json
import math
import os
import re
import statistics
import sys

from harness import PLATFORMS, Run, TIER_A, floor_line, static_line

KAPPA, A, B, C = 0.032, 0.02123, 0.0397, 2.066
SKIP = 2
RUN = Run(shape=TIER_A, world=4)
V = PLATFORMS['v100']


def trimmed(v, f=0.1):
    v = sorted(v)
    k = int(len(v) * f)
    return statistics.mean(v[k:len(v) - k] if len(v) > 2 * k else v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--job', required=True)
    a = ap.parse_args()
    pat = re.compile(r'_S_(static|offload)_x(\d+)_rep([ab])_')
    data, peaks, loss0 = {}, {}, {}
    for d in glob.glob(os.path.join(a.root, f'{a.job}-*')):
        m = pat.search(os.path.basename(d))
        p = os.path.join(d, 'steps_rank0.json')
        if not m or not os.path.exists(p):
            continue
        mode, x = m.group(1), int(m.group(2))
        st = json.load(open(p))['steps']
        data.setdefault((mode, x), []).extend(s['step_ms'] / 1000.0 for s in st[SKIP:])
        peaks.setdefault((mode, x), []).append(max(s['peak_mb'] for s in st[1:]))
        loss0.setdefault(x, set()).add(st[0]['loss'])
    xs = sorted({x for (_, x) in data})
    if not xs:
        sys.exit('no arms found')

    print(f"{'x':>3} {'static mean/med':>17} {'offload mean/med':>18} {'delta s':>9} {'se':>6} "
          f"{'delta/x':>8} {'tax %':>6} {'H-LIN %':>8} {'peaks':>6}")
    taxes, per_x = {}, {}
    for x in xs:
        s, o = data.get(('static', x), []), data.get(('offload', x), [])
        if len(s) < 3 or len(o) < 3:
            print(f'{x:>3} incomplete')
            continue
        d = statistics.mean(o) - statistics.mean(s)
        se = math.sqrt(statistics.variance(s) / len(s) + statistics.variance(o) / len(o))
        tax = 100.0 * d / statistics.mean(s)
        taxes[x], per_x[x] = tax, d / x
        hl = 100.0 * KAPPA * x / (A * x * x + B * x + C)
        sl, fl = static_line(RUN, V).at(x), floor_line(RUN, V).at(x)
        pk_ok = all(abs(p - sl) <= 0.05 for p in peaks.get(('static', x), [])) and \
            all(abs(p - fl) <= 0.05 for p in peaks.get(('offload', x), []))
        print(f"{x:>3} {statistics.mean(s):>8.3f}/{statistics.median(s):<8.3f} "
              f"{statistics.mean(o):>8.3f}/{statistics.median(o):<9.3f} {d:>+9.3f} {se:>6.3f} "
              f"{d / x:>8.4f} {tax:>6.2f} {hl:>8.2f} {'ok' if pk_ok else 'check':>6}"
              f"   (trimmed delta {trimmed(o) - trimmed(s):+.3f})")

    print()
    short = [per_x[x] for x in per_x if x <= 12]
    if short:
        m = statistics.mean(short)
        verdict = 'H-LIN' if 0.024 <= m <= 0.040 else ('H-SHORT' if m < 0.024 else 'ABOVE BOTH')
        print(f"S-P2 decider: mean delta/x over x<=12 = {m:.4f} s/x -> {verdict}")
    if 2 in taxes and 8 in taxes:
        print(f"S-P3 shape: tax(2) = {taxes[2]:.2f}% vs tax(8) = {taxes[8]:.2f}% -> "
              f"{'rise exists' if taxes[2] < taxes[8] else 'FALSIFIED: monotone'}")
    if taxes:
        xpk = max(taxes, key=taxes.get)
        print(f"tax peaks at x={xpk} ({taxes[xpk]:.2f}%)")
    d = [x for x, v in loss0.items() if len(v) > 1]
    print(f"S-P5 step-0 losses: {'identical within each x' if not d else 'DIFFER at x=' + str(d)}")


if __name__ == '__main__':
    main()
