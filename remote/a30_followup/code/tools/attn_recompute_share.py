import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_nccl_duty import load_trace, gpu_kernel_events, merge_intervals

ATTN_BWD = re.compile(r'fmha_cutlassB|AttentionBackwardKernel|efficient_attention_backward', re.I)
ATTN_FWD = re.compile(r'fmha_cutlassF|AttentionKernel|efficient_attention_forward', re.I)

def classify(name):
    if ATTN_BWD.search(name):
        return 'attn_bwd'
    if ATTN_FWD.search(name):
        return 'attn_fwd'
    if 'nccl' in name.lower():
        return 'nccl'
    if name.startswith('Memcpy') or name.startswith('Memset'):
        return 'memcpy'
    return 'other'

def analyze(events, layers, steps):
    kernels = [e for e in gpu_kernel_events(events)
               if e.get('dur', 0) > 0
               and (e.get('cat') or '').lower() != 'gpu_user_annotation']
    if not kernels:
        return {'ok': False, 'reason': 'no GPU kernel events in trace'}
    t = {'attn_fwd': 0.0, 'attn_bwd': 0.0, 'nccl': 0.0, 'memcpy': 0.0, 'other': 0.0}
    n = dict.fromkeys(t, 0)
    by_name = {}
    compute_spans = []
    for e in kernels:
        c = classify(e.get('name', ''))
        t[c] += e['dur']
        n[c] += 1
        by_name[e['name']] = by_name.get(e['name'], 0.0) + e['dur']
        if c not in ('nccl', 'memcpy'):
            compute_spans.append((e['ts'], e['ts'] + e['dur']))
    start = min(e['ts'] for e in kernels)
    end = max(e['ts'] + e['dur'] for e in kernels)
    span_us = end - start
    compute_busy_us = sum(b - a for a, b in merge_intervals(compute_spans))

    exp_fwd, exp_bwd = 2 * layers * steps, layers * steps
    counts_ok = (n['attn_fwd'] == exp_fwd and n['attn_bwd'] == exp_bwd)

    replay_us = t['attn_fwd'] / 2.0
    res = {
        'ok': counts_ok,
        'counts': n, 'expected_counts': {'attn_fwd': exp_fwd, 'attn_bwd': exp_bwd},
        'kernel_ms': {k: v / 1e3 for k, v in t.items()},
        'gpu_span_ms': span_us / 1e3,
        'compute_busy_ms': compute_busy_us / 1e3,
        'attn_replay_ms': replay_us / 1e3,
        'attn_replay_share_of_span': replay_us / span_us,
        'attn_replay_share_of_compute': replay_us / compute_busy_us if compute_busy_us else None,
        'attn_bwd_over_one_fwd': (t['attn_bwd'] / replay_us) if replay_us else None,
        'top_kernels_ms': sorted(((k, v / 1e3) for k, v in by_name.items()),
                                 key=lambda kv: -kv[1])[:20],
    }
    if not counts_ok:
        res['reason'] = (f"attention kernel counts {n['attn_fwd']}/{n['attn_bwd']} != "
                         f"expected {exp_fwd}/{exp_bwd}: read top_kernels_ms for the "
                         f"real kernel names before trusting any share")
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trace')
    ap.add_argument('--layers', type=int, default=64)
    ap.add_argument('--steps', type=int, default=1)
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    tr = load_trace(a.trace)
    events = tr['traceEvents'] if isinstance(tr, dict) else tr
    res = analyze(events, a.layers, a.steps)
    print(json.dumps({k: v for k, v in res.items() if k != 'top_kernels_ms'}, indent=2))
    if not res.get('ok'):
        print('SELF-CHECK FAILED:', res.get('reason'))
        for k, v in res.get('top_kernels_ms', []):
            print(f'  {v:12.1f} ms  {k[:150]}')
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2))
    sys.exit(0 if res.get('ok') else 1)

if __name__ == '__main__':
    main()
