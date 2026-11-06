"""The time model against job 7758 (V100).

A REGRESSION test, not a prediction test: the tolerances below were set after
7758 was read. It checks that (1) kappa refitted from 7758's paired arms at
x >= 24 reproduces the 0.032 s/x the harness carries, (2) the pre-registered T(x)
matches 7758's measured static times inside the fit's data range (x >= 8), (3) the model's tax
is within ±25% of every measured x >= 24 point, and (4) the predicted 2%
crossing falls between the measured x=64 (above 2%) and x=100 (below 2%) arms.
The prediction tests are on the other platforms.

Run:  PYTHONPATH=. python3 tests/test_harness_time.py
"""
import glob
import json
import re
import statistics
import sys

from harness import TimeCalibration, fit_kappa, fit_step_time

JOB = '7758'
CAL = TimeCalibration(0.02123, 0.0397, 2.066, 0.032)


def load():
    out = {}
    for d in glob.glob(f'profile_results/regime4/{JOB}-*'):
        m = re.search(r'_R_(static|offload)_x(\d+)_', d)
        if not m:
            continue
        st = json.load(open(d + '/steps_rank0.json'))['steps']
        out[(m.group(1), int(m.group(2)))] = statistics.mean(s['step_ms'] / 1000 for s in st[1:])
    return out


def main():
    t = load()
    if not t:
        sys.exit(f'no {JOB} arms found under profile_results/regime4')
    ok = True

    def check(name, cond, detail):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name}   {detail}")
        ok &= cond

    xs = sorted(x for (m, x) in t if m == 'static' and ('offload', x) in t)
    long = [x for x in xs if x >= 24]
    kappa = fit_kappa([(x, t[('offload', x)] - t[('static', x)]) for x in long])
    check('kappa refit from 7758 x>=24', abs(kappa - CAL.kappa) <= 0.003, f'{kappa:.4f} vs {CAL.kappa}')
    a, b, c = fit_step_time([(x, t[('static', x)]) for x in xs])
    # The pre-registered fit was made from x = 8..104 only; compare inside that range,
    # and report the extrapolated short end separately rather than hiding it.
    inr = [x for x in xs if x >= 8]
    worst, xw = max((abs(CAL.step_time(x) - t[('static', x)]) / t[('static', x)], x) for x in inr)
    check('pre-registered T(x) matches 7758\'s measured static times inside its data range (x >= 8)',
          worst <= 0.03, f'worst {worst:.1%} at x={xw}; refit on 7758: {a:.5f} x^2 + {b:.4f} x + {c:.3f}')
    for x in [x for x in xs if x < 8]:
        print(f"  NOTE  x={x} is outside the pre-registered fit's range: fit {CAL.step_time(x):.3f} s, "
              f"measured {t[('static', x)]:.3f} s ({CAL.step_time(x) / t[('static', x)] - 1:+.1%}); "
              f"short-end readouts should use the refit")
    for x in long:
        meas = t[('offload', x)] / t[('static', x)] - 1
        pred = CAL.tax(x)
        check(f'tax at x={x} within ±25%', abs(pred - meas) <= 0.25 * meas,
              f'model {pred:.2%}, measured {meas:.2%}')
    xc = CAL.crossing(0.02)
    above = t[('offload', 64)] / t[('static', 64)] - 1 > 0.02
    below = t[('offload', 100)] / t[('static', 100)] - 1 < 0.02
    check('predicted x*(2%) between the measured x=64 and x=100 arms', above and below and 64 < xc < 100,
          f'x*(2%) = {xc:.1f}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
