"""tools/a30_followup.py on synthetic A30 follow-up outputs.

Arm prefixes (S, E, G, Q, NO) and prediction labels (H1, R1, R2, NO-P*) follow
the scripts of jobs 642029 (tax) and 642032 (native).
Plants H1 (the allocator) as the cause: static is slower by 0.042 s per x above
x=64, carries 40 extra cudaMalloc/cudaFree per step, and the gap vanishes with
expandable segments off; the ballast and the clocks are neutral. Checks R1/R2,
that the reader attributes the gap to E and not to G or the clocks, the profiled
idle split, and the native ladder verdicts (bracket [564, 568), request 4x).

    PYTHONPATH=. python3 tests/test_a30_followup_reader.py
"""
import json
import os
import subprocess
import sys
import tempfile

T = lambda x: 0.0152 * x * x + 0.05 * x + 0.8
GAP = lambda x: 0.042 * (x - 64) if x > 64 else -0.002 * x


def make(out, blk, mode, x, n, rep=0, fits=True, gap=True, ballast=0.0, conf='expandable_segments:True',
         extra_alloc=0, request=None, sm=1440):
    tag = f"{blk}_{mode}_fp16_mem_L64_x{x}_n{n}{f'_r{rep}' if rep else ''}"
    d = os.path.join(out, tag); os.makedirs(d)
    if not fits:
        open(os.path.join(d, 'stderr.txt'), 'w').write(
            f"torch.OutOfMemoryError: CUDA out of memory. Tried to allocate {request:.2f} MiB ... _engine_run_backward")
        return
    t = T(x) + (GAP(x) if (mode == 'static' and gap) else 0.0)
    steps = []
    for i in range(n):
        k = 100 + i * (50 + (extra_alloc if mode == 'static' else 0))
        steps.append({'step': i, 'loss': 10.5, 'peak_mb': 5000.0 + ballast, 'step_ms': (t + (2 if i == 0 else 0) + 0.01 * (i % 2)) * 1e3,
                      'alloc': {'num_device_alloc': k, 'num_device_free': k - 10, 'num_alloc_retries': 0,
                                'num_sync_all_streams': 0, 'segment.all.current': 200, 'reserved_bytes.all.current': 1}})
    json.dump({'config': {'hbm_ballast_mb': ballast, 'alloc_conf': conf}, 'steps': steps},
              open(os.path.join(d, 'steps_rank0.json'), 'w'))
    with open(os.path.join(d, 'gpu.csv'), 'w') as f:
        for i in range(60):
            for g in range(4):
                f.write(f"{g}, 2026/10/13 10:00:{i:02d}.000, {sm}, 1215, 150.0, 60, 100, 0x0000000000000000\n")
    return d


def main():
    out = tempfile.mkdtemp()
    for x, n, reps in ((80, 5, (1,)), (100, 5, (1,)), (120, 4, (1,)), (150, 4, (1, 2))):
        for r in reps:
            make(out, 'S', 'static', x, n, r, extra_alloc=40)
            make(out, 'S', 'offload', x, n, r)
    make(out, 'G', 'offload', 150, 4, ballast=13784.9)
    make(out, 'G', 'offload', 100, 5, ballast=9059.9)
    for x, n in ((100, 5), (150, 4)):
        make(out, 'E', 'static', x, n, gap=False, conf='')
        make(out, 'E', 'offload', x, n, conf='')
    for mode, busy, span in (('static', 700000.0, 720000.0), ('offload', 700000.0, 712800.0)):
        d = make(out, 'Q', mode, 150, 4)
        json.dump({'gpu_span_ms': span, 'compute_busy_ms': busy, 'kernel_ms': {'attn_fwd': 1.0, 'other': 2.0}},
                  open(os.path.join(d, 'attn_share.json'), 'w'))
    run = lambda *a: subprocess.run([sys.executable, 'tools/a30_followup.py', *a], capture_output=True, text=True, check=True).stdout
    r = run('tax', out)
    assert 'R1 HELD' in r and 'R2 HELD' in r, r
    assert 'E (expandable OFF): delta +0.000 s -> removes >=70% of the gap' in r, r
    assert 'footprint does not explain it' in r, r
    assert 'SM clock static/offload - 1 = +0.00%' in r, r
    assert "num_device_alloc': 90.0" in r and "num_device_alloc': 50.0" in r, r
    assert 'idle' in r and '+7200.0' in r, r
    print('  ok  tax: R1/R2 held, gap 3.612 s attributed to E (allocator), not ballast or clocks; '
          'static +40 cudaMalloc/step; profiled idle +7.2 s')
    nout = tempfile.mkdtemp()
    make(nout, 'NO', 'offload', 564, 2)
    make(nout, 'NO', 'offload', 568, 2, fits=False, request=4 * 568.0)
    open(os.path.join(nout, 'NO_offload_fp16_mem_L64_x564_n2', 'hostmem_min_avail_pct'), 'w').write('52')
    r = run('native', nout)
    assert 'bracket [564, 568)' in r and 'NO-P1 HELD' in r and 'NO-P2 HELD' in r and 'NO-P5 HELD' in r, r
    print('  ok  native: bracket [564, 568), NO-P1/NO-P2/NO-P5 held, span printed')
    print('ALL PASS')


if __name__ == '__main__':
    main()
