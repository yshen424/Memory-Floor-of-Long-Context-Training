"""The harness's step-0 lines against every measured first-step peak.

Step 0 has no AdamW state (8 B/param/rank), and at the offload floor it also
lacks the (W - 1) shard units that step 1 adds (the count was predicted and
held at worlds 2 and 3, job 7765). Rows: static and stock
offload (and forced eviction, which sits on the same floor), every shape and
world in the run tables, chunked loss c=256, fp16, peak_hbm_mb_first = the
step-0 peak. Engine rows are excluded: their step-0 pick count is not in the
tables (tests/test_harness_v100.py handles them).

Run:  PYTHONPATH=. python3 tests/test_harness_step0.py
"""
import csv
import re
import sys

from harness import PLATFORMS, ModelShape, Run, floor_line, static_line

TOL = 0.05
V100 = PLATFORMS['v100']


def main():
    seen, tested, skipped, fails = set(), [], {}, []
    for f in ('profile_results/runs_4gpu.csv',):
        for r in csv.DictReader(open(f)):
            key = (r['job_id'], r['run_id'])
            if key in seen:
                continue
            seen.add(key)
            rid, fa = r['run_id'], r['force_action']
            why = None
            if r['dtype'] != 'fp16' or 'c256' not in rid:
                why = 'not chunked c=256 / not fp16'
            elif r['ok'] not in ('yes', 'ok', 'success') or not r['peak_hbm_mb_first']:
                why = 'no step-0 peak'
            elif r['job_id'] == '7752' or any(k in rid for k in ('permodule', 'vanilla', 'linktest')):
                why = 'contrast / micro-batching'
            elif r['engine_mode'] in ('live', 'engine_loose', 'engine_strict') or 'engine' in rid:
                why = 'engine (step-0 picks not tabled)'
            elif rid.startswith('a100_G_'):
                why = 'idle ballast held on purpose (8039176 G)'
            elif 'cpu' in rid or 'cpu_of' in fa:
                why = 'CPU offload (step-0 form not derived)'
            if why:
                skipped[why] = skipped.get(why, 0) + 1
                continue
            floor = 'offload' in rid or 'offload_wrapper' in fa or 'force_evict' in rid or fa.startswith('evict')
            L, W = int(r['num_layers']), int(r['world_size'])
            mv = re.search(r'_v(\d+)_', rid)
            shape = ModelShape(int(r['hidden_size']), int(r['intermediate_size']), int(r['num_heads']),
                               int(r['num_kv_heads']), L, int(mv.group(1)) if mv else 32000)
            run = Run(shape=shape, world=W)
            x = int(r['seq_len']) // 1024
            ln = floor_line(run, V100, 0, step0=True) if floor else static_line(run, V100, step0=True)
            meas = float(r['peak_hbm_mb_first'])
            d = meas - ln.at(x)
            rec = (r['job_id'], W, L, shape.hidden, x, 'floor' if floor else 'static', ln.binding(x), meas, ln.at(x), d)
            tested.append(rec)
            if abs(d) > TOL:
                fails.append(rec)
    tested.sort(key=lambda t: (t[5], t[1], t[2], t[3], t[4]))
    for j, W, L, h, x, mech, inst, me, pr, d in tested:
        if abs(d) > TOL:
            print(f"  MISS {j} W{W} L{L} h{h} x{x} {mech:6} [{inst}] measured {me:.3f} predicted {pr:.3f} diff {d:+.3f}")
    by = {}
    for t in tested:
        by[(t[5], t[6])] = by.get((t[5], t[6]), 0) + 1
    print('binding instants:', ', '.join(f'{m} {i}: {n}' for (m, i), n in sorted(by.items())))
    print(f"{len(tested)} step-0 peaks tested, {len(fails)} outside ±{TOL} MiB; "
          f"max |diff| {max(abs(t[9]) for t in tested):.4f} MiB")
    print('not tested:', ', '.join(f'{k}: {v}' for k, v in sorted(skipped.items())))
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
