"""Capacity brackets: calibrate on world-4 brackets, predict the rest.

Calibration set (declared here, before looking at the test verdicts): every
x >= 100 rung of jobs 7743, 7744 and 7747 (world 4: static, stock offload,
forced eviction, the engine at the cap at restore depths 0/1/2).
Test set: every other rung in the run table (V100 jobs 7756, 7758, 7760 and
7765, and the L40S and A100 jobs 630188, 634078, 644212, 8039176, 8051891).

Engine rungs below the cap are excluded: their peak depends on the gate's picks
(tested as dose-response in test_harness_v100.py). Each rung's peak is the
harness line; its request is the replayed MLP activation (4 MiB per x here).
Acceptance: every TEST bracket within ±1 rung; every misclassified rung listed.

Run:  PYTHONPATH=. python3 tests/test_harness_brackets.py
"""
import collections
import csv
import re
import sys

from harness import PLATFORMS, Run, TIER_A, calibrate, floor_line, static_line

V100 = PLATFORMS['v100']
CAL_JOBS = {'7743', '7744', '7747'}


def rungs():
    seen = set()
    for f in ('profile_results/runs_4gpu.csv',):
        for r in csv.DictReader(open(f)):
            k = (r['job_id'], r['run_id'])
            if k in seen:
                continue
            seen.add(k)
            rid = r['run_id']
            if (r['dtype'] != 'fp16' or 'c256' not in rid or r['job_id'] == '7752'
                    or r['num_layers'] not in ('32', '64')
                    or any(t in rid for t in ('permodule', 'vanilla', 'linktest', 'profile', 'snapshot'))):
                continue
            x = int(r['seq_len']) // 1024
            fit = r['ok'] in ('yes', 'ok', 'success')
            m = re.search(r'req_([\d.]+)MiB', r['ok'])
            if x < 100 or (not fit and not m):
                continue
            L, W, cpu = int(r['num_layers']), int(r['world_size']), 'cpu' in rid
            run = Run(shape=TIER_A.with_layers(L), world=W, cpu_offload=cpu)
            mw = re.search(r'_w(\d+)_', rid)
            wr = int(mw.group(1)) if mw else 0
            if 'offload' in rid or 'force_evict' in rid:
                mech, line = 'floor W=0', floor_line(run, V100, 0)
            elif 'engine' in rid:
                if x < 300:
                    continue
                mech, line = f'engine floor W={wr}', floor_line(run, V100, wr)
            else:
                mech, line = 'static', static_line(run, V100)
            mech += ' +cpu' if cpu else ''
            yield dict(job=r['job_id'], W=W, L=L, mech=mech, x=x, peak=line.at(x),
                       req=4.0 * x, fit=fit, measured_req=float(m.group(1)) if m else None)


def main():
    rs = list(rungs())
    cal = [r for r in rs if r['job'] in CAL_JOBS]
    test = [r for r in rs if r['job'] not in CAL_JOBS]
    model = calibrate([(r['peak'], r['req'], r['fit']) for r in cal])
    if model is None:
        sys.exit('calibration set is not separable')
    print(f"calibrated on {len(cal)} world-4 rungs: C0 = {model.c0_mib:.2f} MiB, "
          f"alpha = {model.alpha:.3f} (feasible alpha {model.alpha_range[0]:.3f}-{model.alpha_range[1]:.3f})\n")

    ok = True
    for name, group in (('CALIBRATION', cal), ('TEST', test)):
        print(f"--- {name}")
        print(f"{'job':>5} {'W':>2} {'L':>3} {'mechanism':<20} {'x':>4} {'peak':>10} {'cap(req)':>10} {'margin':>8}  measured  predicted")
        for r in sorted(group, key=lambda r: (r['W'], r['L'], r['mech'], r['x'], r['job'])):
            cap = model.capacity(r['req'])
            pred = r['peak'] <= cap
            flag = '' if pred == r['fit'] else '   <-- MISCLASSIFIED'
            if r['measured_req'] is not None and abs(r['measured_req'] - r['req']) > 0.01:
                flag += f"  (request {r['measured_req']} != {r['req']})"
            print(f"{r['job']:>5} {r['W']:>2} {r['L']:>3} {r['mech']:<20} {r['x']:>4} {r['peak']:>10.2f} "
                  f"{cap:>10.2f} {cap - r['peak']:>+8.2f}  {'fit' if r['fit'] else 'FAIL':>8}  {'fit' if pred else 'FAIL':>9}{flag}")
        # brackets per ladder: predicted largest fit vs measured, in rungs
        lad = collections.defaultdict(list)
        for r in group:
            lad[(r['job'], r['W'], r['L'], r['mech'])].append(r)
        for key, rr in sorted(lad.items()):
            mf = [r['x'] for r in rr if r['fit']]
            mfail = [r['x'] for r in rr if not r['fit']]
            pf = [r['x'] for r in rr if r['peak'] <= model.capacity(r['req'])]
            if mf and mfail and pf:
                steps = sorted({r['x'] for r in rr})
                dm = steps.index(max(mf)) - steps.index(max(pf))
                print(f"    bracket {key}: measured [{max(mf)}, {min(mfail)})  predicted largest fit {max(pf)}  "
                      f"({'exact' if dm == 0 else f'{dm:+d} rung'})")
                if name == 'TEST' and abs(dm) > 1:
                    ok = False
        print()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
