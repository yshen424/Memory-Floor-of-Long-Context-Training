"""The harness against every measured V100 peak it claims to cover.

One number is calibrated rather than derived: the platform's library
workspace (config.PLATFORMS['v100'].ws_mib, 16.2559 MiB).
Every slope is arithmetic. Every row below is TEST data; fitted on all of them,
the workspace comes out at 16.2561 MiB.

Covered: whole-layer unit, chunked loss c=256, regime B fp16, fused-input,
cache off, b=1, every x (the peak is the largest of three instants: first
replay, second replay, optimizer step), every shape of job 7761 (two harness
terms post hoc from it), mechanisms static / stock offload / forced
eviction / the engine (below the cap and at it), each with and without FSDP CPU
offload. Not derived, and listed rather than tested: per-module and
no-checkpointing contrast lines, unchunked loss, micro-batching (7752), failed
rungs (no peak to compare).

Engine rows below the cap: how many boundaries were absent at the peak step, and
which step set the reported maximum, are not in the run tables, so the test checks
the dose-response claim directly: the gap to the step-1+ or step-0 static line
(step 0 has no AdamW state yet) must be a whole number of boundaries (residual
after rounding <= TOL) within the cap. A boundary is 168-210 MiB at these x, so
an accidental fit inside ±0.05 MiB is unlikely.

Run:  PYTHONPATH=. python3 tests/test_harness_v100.py
"""
import csv
import re
import sys

from harness import PLATFORMS, ModelShape, Run, floor_line, static_line

TOL = 0.05          # MiB
V100 = PLATFORMS['v100']


def rows():
    seen = set()
    for f in ('profile_results/runs_4gpu.csv',):
        for r in csv.DictReader(open(f)):
            key = (r['job_id'], r['run_id'])
            if key in seen:
                continue
            seen.add(key)
            yield r


def classify(r):
    rid, fa, em = r['run_id'], r['force_action'], r['engine_mode']
    cpu = 'cpu' in rid or 'cpu_of' in fa
    if 'force_evict' in rid or fa.startswith('evict'):
        return 'force_evict', cpu
    if 'offload' in rid or 'offload_wrapper' in fa:
        return 'stock_offload', cpu
    if em in ('live', 'engine_loose', 'engine_strict') or 'engine' in rid:
        return 'engine', cpu
    return 'static', cpu


def main():
    tested, skipped, fails = [], [], []
    for r in rows():
        why = None
        if r['dtype'] != 'fp16' or 'c256' not in r['run_id'] and '_c256_' not in r['run_id']:
            why = 'not chunked c=256 / not fp16'
        elif r['job_id'] == '7752':
            why = 'micro-batching (7752)'
        elif r['ok'] not in ('yes', 'ok', 'success'):
            why = 'failed rung (no peak)'
        elif r['platform'] != 'V100':
            why = 'remote platform (judged by its job reader, tools/remote_frontier.py)'
        elif 'permodule' in r['run_id'] or 'vanilla' in r['run_id'] or 'linktest' in r['run_id']:
            why = 'contrast line / link test, not derived'
        elif r['steps'] == '1':
            why = 'one-step run: its peak is step 0 (7765 snapshot arm; tools/world_shards.py reads it)'
        elif not r['peak_hbm_mb_max']:
            why = 'no peak'
        x = int(r['seq_len']) // 1024 if r['seq_len'] else 0
        if why:
            skipped.append((why, r['job_id'], r['run_id'][:60]))
            continue

        L, W = int(r['num_layers']), int(r['world_size'])
        mech, cpu = classify(r)
        m = re.search(r'_w(\d+)_', r['run_id'])
        wr = int(m.group(1)) if (m and mech == 'engine') else 0
        mv = re.search(r'_v(\d+)_', r['run_id'])
        shape = ModelShape(int(r['hidden_size']), int(r['intermediate_size']), int(r['num_heads']),
                           int(r['num_kv_heads']), L, int(mv.group(1)) if mv else 32000)
        run = Run(shape=shape, world=W, cpu_offload=cpu)
        meas = float(r['peak_hbm_mb_max'])
        st = static_line(run, V100).at(x)
        fl = floor_line(run, V100, wr if mech == 'engine' else 0).at(x)
        b = 2 * shape.hidden / 1024.0 * x
        label = f"{mech}{'+cpu' if cpu else ''}"
        sl, fll = static_line(run, V100), floor_line(run, V100, wr if mech == 'engine' else 0)
        if mech == 'static':
            pred, how = st, f'static {sl.binding(x)}'
        elif mech in ('stock_offload', 'force_evict'):
            pred, how = fl, f'floor {fll.binding(x)}'
        else:
            # The reported peak is the max over steps; which step set it, and how
            # many boundaries were absent then, is not in the run table. Candidates:
            # step 1+ at the cap floor; step 1+ or step 0 with a whole number k absent.
            st0 = static_line(run, V100, step0=True).at(x)
            cands = [(fl, f'floor W={wr}')]
            for base, tag in ((st, 'step1+'), (st0, 'step0')):
                k = round((base - meas) / b)
                if 0 <= k <= L - wr - 1:
                    cands.append((base - k * b, f'{tag} {k} absent'))
            pred, how = min(cands, key=lambda c: abs(meas - c[0]))
        d = meas - pred
        rec = (r['job_id'], W, L, x, label, how, meas, pred, d)
        tested.append(rec)
        if abs(d) > TOL or 'OUT OF RANGE' in how:
            fails.append(rec)

    tested.sort(key=lambda t: (t[1], t[2], t[4], t[3], t[0]))
    print(f"{'job':>5} {'W':>2} {'L':>3} {'x':>4} {'mechanism':<20} {'predicted as':<22} "
          f"{'measured':>11} {'predicted':>11} {'diff':>8}")
    for j, W, L, x, lab, how, me, pr, d in tested:
        flag = '' if abs(d) <= TOL else '   <-- MISS'
        print(f"{j:>5} {W:>2} {L:>3} {x:>4} {lab:<20} {how:<22} {me:>11.4f} {pr:>11.4f} {d:>+8.4f}{flag}")
    print(f"\n{len(tested)} measured peaks tested, {len(fails)} outside ±{TOL} MiB; "
          f"max |diff| {max(abs(t[8]) for t in tested):.4f} MiB")
    by = {}
    for why, *_ in skipped:
        by[why] = by.get(why, 0) + 1
    print('not tested (by reason):', ', '.join(f'{k}: {v}' for k, v in sorted(by.items())))
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
