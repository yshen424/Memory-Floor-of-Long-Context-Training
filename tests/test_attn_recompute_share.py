"""Synthetic-trace test for tools/attn_recompute_share.py. Standalone:
    PYTHONPATH=. python3 tests/test_attn_recompute_share.py
"""
import gzip
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))
from attn_recompute_share import analyze, classify  # noqa: E402

FWD = 'fmha_cutlassF_f16_aligned_64x64_rf_sm50(PyTorchMemEffAttention::AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 64, true, true>::Params)'
BWD = 'fmha_cutlassB_f16_aligned_64x64_k64_sm50(PyTorchMemEffAttention::AttentionBackwardKernel<cutlass::arch::Sm50, cutlass::half_t, true, false, false, 64, 64, 64, false>::Params)'


def k(name, ts, dur, cat='kernel'):
    return {'ph': 'X', 'cat': cat, 'name': name, 'ts': ts, 'dur': dur}


def whole_layer_step(L, fwd=10.0, bwd=25.0, gemm=5.0):
    """Forward (L attn fwd), then backward with a replay per layer (attn fwd +
    attn bwd), plus one GEMM per pass. 1 us gaps."""
    ev, t = [], 0.0
    for _ in range(L):
        ev.append(k(FWD, t, fwd)); t += fwd + 1
        ev.append(k('volta_h884gemm_128x128', t, gemm)); t += gemm + 1
    for _ in range(L):
        ev.append(k(FWD, t, fwd)); t += fwd + 1
        ev.append(k(BWD, t, bwd)); t += bwd + 1
        ev.append(k('ncclDevKernel_ReduceScatter', t, 3.0)); t += 4
    # A ProfilerStep mirror spanning everything: must be ignored.
    ev.append(k('ProfilerStep#2', -50.0, t + 100, cat='gpu_user_annotation'))
    return ev, t


def main():
    fails = 0

    def check(cond, msg):
        nonlocal fails
        print(('PASS ' if cond else 'FAIL ') + msg)
        fails += 0 if cond else 1

    check(classify(BWD) == 'attn_bwd', 'backward name is not misread as forward')
    check(classify(FWD) == 'attn_fwd', 'forward name classified')

    L = 4
    ev, end = whole_layer_step(L)
    r = analyze(ev, layers=L, steps=1)
    check(r['ok'], 'counts self-check passes on a whole-layer step (2L fwd, L bwd)')
    check(abs(r['attn_replay_ms'] - L * 10.0 / 1e3) < 1e-9, 'replay = half the forward kernel time')
    check(abs(r['gpu_span_ms'] - (end - 1) / 1e3) < 1e-9, 'annotation span excluded from the step span')
    check(abs(r['attn_bwd_over_one_fwd'] - 2.5) < 1e-9, 'bwd/fwd ratio recovered')
    check(abs(r['attn_replay_share_of_span'] - 40.0 / (end - 1)) < 1e-9, 'share of span')

    # Negative control 1: per-module-like trace with only L forward launches.
    ev2 = [e for i, e in enumerate(ev) if not (e['name'] == FWD and i >= 2 * L)]
    r2 = analyze(ev2, layers=L, steps=1)
    check(not r2['ok'] and 'top_kernels_ms' in r2, 'missing replay launches fail the self-check with a diagnosis')

    # Negative control 2: unknown kernel names (a different build) must not pass.
    ev3 = [dict(e, name=e['name'].replace('fmha_cutlass', 'xx').replace('Attention', 'Zz')) for e in ev]
    r3 = analyze(ev3, layers=L, steps=1)
    check(not r3['ok'] and r3['counts']['attn_fwd'] == 0, 'renamed kernels are not silently counted')

    # CLI end to end on a .gz trace; exit code mirrors the self-check.
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 't.pt.trace.json.gz')
        with gzip.open(p, 'wt') as fh:
            json.dump({'traceEvents': ev}, fh)
        tool = os.path.join(os.path.dirname(__file__), '..', 'tools', 'attn_recompute_share.py')
        out = subprocess.run([sys.executable, tool, p, '--layers', str(L), '--json', os.path.join(d, 'o.json')],
                             capture_output=True, text=True)
        check(out.returncode == 0 and os.path.exists(os.path.join(d, 'o.json')), 'CLI runs on .gz and writes JSON')

    print(f'{fails} failure(s)')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
