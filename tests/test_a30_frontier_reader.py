"""tools/a30_frontier.py on a synthetic A30 job (arm layout and labels as in job 633912).

Plants: common byte offset +1.25 MiB; the flash plane +1.5 MiB/x; native GQA -2.0 MiB/x; T(x) = 0.01x^2 + 0.03x
+ 1.5 s and kappa 0.020 s/x; capped reserve delta 50 MiB at alpha 0.15; context
600 MiB on a 24,062 MiB card. Checks that bytes/timecal/capcal recover them and
that capcal's native rungs bracket the planted native frontier.

    PYTHONPATH=. python3 tests/test_a30_frontier_reader.py
"""
import json
import os
import re
import subprocess
import sys
import tempfile

from harness import PLATFORMS, Run, TIER_A, static_line, floor_line

P = PLATFORMS['a30']
OFF, GQA, KAPPA, DELTA, ALPHA, CTX, TOTAL = 1.25, -2.0, 0.020, 50.0, 0.15, 600.0, 24062.0
FLASH = 1.5                     # the flash plane, MiB/x at tier-A (633912)
T = lambda x: 0.01 * x * x + 0.03 * x + 1.5


def line(mode, L, gqa):
    r = Run(shape=TIER_A.with_layers(L), world=4, cpu_offload=mode.endswith('cpu'), repeat_kv=not gqa)
    return static_line(r, P) if mode.startswith('static') else floor_line(r, P, 0)


def make(out, blk, mode, dt, attn, gqa, L, x, cap, n, rep, fits=True):
    tag = (f"{blk}_{mode}_{dt}_{attn}{'_gqa' if gqa else ''}_L{L}_x{x}"
           f"{f'_cap{cap}' if cap else ''}_n{n}{f'_r{rep}' if rep else ''}")
    d = os.path.join(out, tag); os.makedirs(d)
    if not fits:
        open(os.path.join(d, 'stderr.txt'), 'w').write(
            f"torch.OutOfMemoryError: CUDA out of memory. Tried to allocate {4.0 * x:.2f} MiB ... backward")
        return
    pk = line(mode, L, False).at(x) + OFF + (GQA * x if gqa else 0) + (FLASH * x if attn == 'flash' else 0)
    t = T(x) + (KAPPA * x if mode == 'offload' else 0) + (0.01 if rep == 2 else 0)
    steps = [{'step': i, 'loss': 10.4, 'peak_mb': pk - (5 if i == 0 else 0), 'step_ms': t * 1e3,
              'outside_alloc_mb': CTX - 3 * (i == 0)} for i in range(n)]
    json.dump({'config': {'attention': {'requested': 'sdpa_flash' if attn == 'flash' else 'sdpa_mem_efficient'}},
               'steps': steps}, open(os.path.join(d, 'steps_rank0.json'), 'w'))


def fits_capped(mode, x, cap):
    return line(mode, 64, False).at(x) + OFF <= cap * 1024 - DELTA - ALPHA * 4 * x


def main():
    out = tempfile.mkdtemp()
    json.dump({'total_mib': TOTAL}, open(os.path.join(out, 'a30_node.json'), 'w'))
    for L, x in ((32, 24), (32, 48), (64, 24), (64, 48)):
        make(out, 'B', 'static', 'fp16', 'mem', 0, L, x, 0, 3, 0)
    for args in (('offload', 24), ('offload', 48), ('staticcpu', 48), ('offloadcpu', 48), ('evict', 48), ('offload', 100)):
        make(out, 'B', args[0], 'fp16', 'mem', 0, 64, args[1], 0, 3, 0)
    for m, x in (('static', 24), ('static', 48), ('offload', 48)):
        make(out, 'D', m, 'bf16', 'mem', 0, 64, x, 0, 3, 0)
        make(out, 'F', m, 'fp16', 'flash', 0, 64, x, 0, 3, 0)
        make(out, 'F', m, 'fp16', 'flash', 1, 64, x, 0, 3, 0)
    for x, n in ((16, 20), (32, 10), (64, 6)):
        for rep in (1, 2):
            for m in ('static', 'offload'):
                make(out, 'T', m, 'fp16', 'mem', 0, 64, x, 0, n, rep)
    for blk, mode, cap, xs in (('CA', 'static', 6, range(31, 38)), ('CB', 'static', 9, range(54, 61)),
                               ('CC', 'offload', 6, range(104, 118, 2)), ('CD', 'offload', 9, range(184, 198, 2))):
        for x in xs:
            ok = fits_capped(mode, x, cap)
            make(out, blk, mode, 'fp16', 'mem', 0, 64, x, cap, 2, 0, fits=ok)
            if not ok:
                break
    tool = [sys.executable, 'tools/a30_frontier.py']
    run = lambda *a: subprocess.run(tool + list(a), capture_output=True, text=True, check=True).stdout
    b = run('bytes', out)
    assert re.search(r'A-P1 common offset over R-A1 arms: \+1\.250 MiB \(spread 0\.000\); HELD', b), b
    assert 'A-P3 HELD' in b and 'A-P4 HELD' in b, b
    assert b.count('-2.0000 MiB/x (prediction') == 3, b
    print('  ok  bytes: offset +1.250 recovered, bf16/flash twins held, native GQA -2.0000 MiB/x on all three')
    tp = os.path.join(out, 'a30_pred_time.txt')
    t = run('timecal', '--pred', tp, out)
    m = re.search(r'kappa = ([\d.]+) s per x', t)
    assert m and abs(float(m.group(1)) - (KAPPA + 0.0)) < 0.001, t
    assert '0.010000 x^2 + 0.03000 x + 1.5' in t.replace('1.5050', '1.5'), t
    print(f"  ok  timecal: kappa {m.group(1)} (planted {KAPPA}), T(x) recovered; prediction file written")
    npf = os.path.join(out, 'a30_pred_native.txt')
    c = run('capcal', '--pred', npf, out)
    m = re.search(r'alpha ([\d.]+) \(feasible ([\d.]+)-([\d.]+)\), delta ([\d.]+)', c)
    assert m and float(m.group(2)) <= ALPHA <= float(m.group(3)), c
    rungs = list(map(int, open(npf).read().split('RUNGS:')[1].split()))
    true_fit = max(x for x in range(20, 400, 2)
                   if line('static', 64, False).at(x) + OFF <= TOTAL - CTX - DELTA - ALPHA * 4 * x)
    assert min(rungs) <= true_fit < max(rungs), (rungs, true_fit)
    print(f"  ok  capcal: alpha {m.group(1)} in [{m.group(2)}, {m.group(3)}] (planted {ALPHA}), delta {m.group(4)} "
          f"(planted {DELTA}); native rungs {rungs} contain the planted frontier {true_fit}")
    r = run('read', out)
    assert 'CAPPED LADDERS' in r and 'TIME (R-A5)' in r, r
    print('  ok  read: full report renders')
    print('ALL PASS')


if __name__ == '__main__':
    main()
