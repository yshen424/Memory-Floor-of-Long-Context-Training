"""Offloading's time cost in an eight-rank job, as the paper reports it.

The paper's time protocol drops the first step of every arm as warm-up and pools
the rest over the repeats (as figures/f12 does for the four-rank platforms).
The in-job reader (tools/remote_frontier.py) drops the first two steps, so its
taxes differ slightly at the lengths that ran few steps.

For each timed length: pooled static step, offload - static, and the tax. With
--fit X1,X2,X3 it also fits T(x) on the static steps at those lengths and kappa
on every one of them but the shortest, as the in-job prediction does, and
prints the predicted tax at the longest timed length and the 2% crossing.

    python3 tools/rank8_time.py profile_results/rank8/rank8_out/8548619 --fit 16,32,64
    python3 tools/rank8_time.py profile_results/rank8/rank8_out/821554 --fit 8,16,32
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.timemodel import TimeCalibration, fit_step_time, fit_kappa   # noqa: E402

ARM = re.compile(r'^T_(?P<mode>static|offload)_.*_x(?P<x>\d+)_n\d+_r(?P<rep>\d+)$')


def load(out):
    steps = {}
    for d in glob.glob(os.path.join(out, 'T_*')):
        m = ARM.match(os.path.basename(d))
        p = os.path.join(d, 'steps_rank0.json')
        if not m or not os.path.exists(p):
            continue
        s = [st['step_ms'] / 1e3 for st in json.load(open(p))['steps'][1:]]
        steps.setdefault(int(m['x']), {}).setdefault(m['mode'], []).extend(s)
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--fit', default='', help='calibration lengths for T(x) and kappa')
    a = ap.parse_args()
    steps = load(a.out)
    rows = []
    for x in sorted(steps):
        s, o = steps[x].get('static'), steps[x].get('offload')
        if not s or not o:
            continue
        ts, to = statistics.mean(s), statistics.mean(o)
        rows.append((x, ts, to - ts))
        print(f'x={x:>4}  static {ts:8.3f} s  offload - static {to - ts:+7.3f} s  tax {100 * (to - ts) / ts:+6.2f}%')
    if not a.fit:
        return
    xs = sorted(int(v) for v in a.fit.split(','))
    cal_rows = [r for r in rows if r[0] in xs]
    aa, bb, cc = fit_step_time([(x, ts) for x, ts, _ in cal_rows])
    kappa = fit_kappa([(x, d) for x, _, d in cal_rows if x >= xs[1]])
    cal = TimeCalibration(aa, bb, cc, kappa)
    last = rows[-1]
    print(f'kappa {kappa:.4f} s per x (fitted on x = {", ".join(str(x) for x in xs[1:])})')
    print(f'predicted tax at x={last[0]}: {100 * cal.tax(last[0]):.2f}%, measured {100 * last[2] / last[1]:.2f}%')
    x2 = cal.crossing(0.02)
    print(f'x*(2%) = {x2:.0f}' if x2 and x2 > 0 else 'x*(2%): the fitted tax stays under 2% at every length')


if __name__ == '__main__':
    main()
