import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

NCCL_TOKEN = 'nccl'
MEMCPY_TOKENS = ('Memcpy DtoH', 'Memcpy HtoD')

MIN_MEMCPY_US = 20.0

def load_trace(path):
    p = Path(path)
    opener = gzip.open if p.suffix == '.gz' else open
    with opener(p, 'rt') as fh:
        return json.load(fh)

def gpu_kernel_events(events):
    out = []
    for e in events:
        if e.get('ph') != 'X':
            continue
        cat = (e.get('cat') or '').lower()
        if cat not in ('kernel', 'gpu_memcpy', 'gpu_memset', 'gpu_user_annotation'):
            continue
        dur = e.get('dur')
        if dur is None:
            continue
        out.append(e)
    return out

def merge_intervals(spans):
    if not spans:
        return []
    spans = sorted(spans)
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]

def overlap_with_union(span, union):
    s, e = span
    total = 0.0
    for us, ue in union:
        if ue <= s:
            continue
        if us >= e:
            break
        total += min(e, ue) - max(s, us)
    return total

def memcpy_bytes(ev):
    args = ev.get('args') or {}
    for key in ('bytes', 'Bytes', 'size', 'nbytes'):
        if key in args:
            try:
                return int(args[key])
            except (TypeError, ValueError):
                pass
    return None

def analyze(path):
    trace = load_trace(path)
    events = trace.get('traceEvents', trace if isinstance(trace, list) else [])
    kernels = gpu_kernel_events(events)

    nccl_spans = []
    memcpys = []
    all_gpu_spans = []

    for e in kernels:
        name = e.get('name', '')
        s = float(e['ts'])
        d = float(e['dur'])
        span = (s, s + d)
        all_gpu_spans.append(span)
        if NCCL_TOKEN in name.lower():
            nccl_spans.append(span)
        elif any(tok in name for tok in MEMCPY_TOKENS):
            direction = 'D2H' if 'DtoH' in name else 'H2D'
            memcpys.append({
                'span': span,
                'dur_us': d,
                'direction': direction,
                'bytes': memcpy_bytes(e),
            })

    nccl_union = merge_intervals(nccl_spans)
    gpu_union = merge_intervals(all_gpu_spans)

    if not gpu_union:
        return None

    wall_us = gpu_union[-1][1] - gpu_union[0][0]
    nccl_busy_us = sum(e - s for s, e in nccl_union)

    rows = []
    for m in memcpys:
        if m['dur_us'] < MIN_MEMCPY_US:
            continue
        ov = overlap_with_union(m['span'], nccl_union)
        alpha = ov / m['dur_us'] if m['dur_us'] > 0 else 0.0
        gbps = None
        if m['bytes']:
            gbps = m['bytes'] / (m['dur_us'] * 1e-6) / 1e9
        rows.append({
            'direction': m['direction'],
            'dur_us': m['dur_us'],
            'bytes': m['bytes'],
            'alpha_obs': alpha,
            'gbps': gbps,
        })

    mig_union = merge_intervals([m['span'] for m in memcpys])
    nccl_rows = []
    for span in nccl_spans:
        dur = span[1] - span[0]
        ov = overlap_with_union(span, mig_union)
        nccl_rows.append({
            'dur_us': dur,
            'mig_overlap_frac': ov / dur if dur > 0 else 0.0,
        })

    per_stream = defaultdict(int)
    for e in kernels:
        if any(tok in e.get('name', '') for tok in MEMCPY_TOKENS):
            per_stream[(e.get('args') or {}).get('stream')] += 1

    return {
        'path': str(path),
        'wall_us': wall_us,
        'nccl_busy_us': nccl_busy_us,
        'duty': nccl_busy_us / wall_us if wall_us else 0.0,
        'n_nccl_kernels': len(nccl_spans),
        'n_memcpy': len(memcpys),
        'rows': rows,
        'nccl_rows': nccl_rows,
        'per_stream': dict(per_stream),
    }

def pct(vals, q):
    if not vals:
        return float('nan')
    vals = sorted(vals)
    i = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
    return vals[i]

def report(results):
    print()
    print('=' * 74)
    print('NCCL duty cycle')
    print('=' * 74)
    for r in results:
        print(f"  {Path(r['path']).parent.name:<28} "
              f"duty={r['duty']*100:6.2f}%  "
              f"nccl_busy={r['nccl_busy_us']/1000:9.1f} ms  "
              f"window={r['wall_us']/1000:9.1f} ms  "
              f"kernels={r['n_nccl_kernels']:5d}  memcpy={r['n_memcpy']:5d}")

    rows = [row for r in results for row in r['rows']]
    if not rows:
        print('\n  No memcpy spans above threshold -- nothing to say about B(alpha).')
        return

    print()
    print('=' * 74)
    print('Observed B(alpha):  achieved bandwidth bucketed by measured overlap')
    print('=' * 74)
    buckets = [(0.0, 0.05), (0.05, 0.25), (0.25, 0.5),
               (0.5, 0.75), (0.75, 0.95), (0.95, 1.01)]
    for direction in ('D2H', 'H2D'):
        sub = [r for r in rows if r['direction'] == direction and r['gbps']]
        if not sub:
            continue
        print(f"\n  {direction}   (n={len(sub)})")
        print(f"    {'alpha range':<14} {'n':>6} {'p10':>8} {'median':>8} "
              f"{'p90':>8}   {'median MB':>10}")
        for lo, hi in buckets:
            b = [r for r in sub if lo <= r['alpha_obs'] < hi]
            if not b:
                continue
            g = [r['gbps'] for r in b]
            mb = [r['bytes'] / 1e6 for r in b]
            print(f"    [{lo:.2f},{hi:.2f})   {len(b):>6} "
                  f"{pct(g,0.1):>8.2f} {pct(g,0.5):>8.2f} {pct(g,0.9):>8.2f}"
                  f"   {pct(mb,0.5):>10.1f}")

    nrows = [r for res in results for r in res['nccl_rows']]
    if nrows:
        print()
        print('=' * 74)
        print('Collective side: NCCL kernel duration vs concurrent migration')
        print('=' * 74)
        clean = [r['dur_us'] for r in nrows if r['mig_overlap_frac'] < 0.05]
        dirty = [r['dur_us'] for r in nrows if r['mig_overlap_frac'] >= 0.5]
        print(f"    {'cohort':<28} {'n':>6} {'p10':>9} {'median':>9} {'p90':>9}")
        for lab, v in (('no concurrent migration', clean),
                       ('>=50% under migration', dirty)):
            if v:
                print(f"    {lab:<28} {len(v):>6} {pct(v,0.1)/1000:>9.3f} "
                      f"{pct(v,0.5)/1000:>9.3f} {pct(v,0.9)/1000:>9.3f}   ms")
        if clean and dirty:
            infl = pct(dirty, 0.5) / pct(clean, 0.5) if pct(clean, 0.5) else 0
            print(f"\n    NCCL duration inflation under migration: {infl:.2f}x")
        elif not dirty:
            print('\n    No NCCL kernel ran under concurrent migration '
                  '(probe off, or probe failed to collide).')

    agg = defaultdict(int)
    for res in results:
        for s, n in res['per_stream'].items():
            agg[s] += n
    if len(agg) > 1:
        print()
        print('  memcpy ops by CUDA stream: '
              + '  '.join(f'stream {s}: {n}' for s, n in sorted(agg.items(),
                                                                key=lambda x: -x[1])))

    print()
    lo_a = [r['gbps'] for r in rows if r['gbps'] and r['alpha_obs'] < 0.05]
    hi_a = [r['gbps'] for r in rows if r['gbps'] and r['alpha_obs'] >= 0.75]
    if lo_a and hi_a:
        print(f"  idle-window median  ({len(lo_a):5d} ops): {pct(lo_a,0.5):.2f} GB/s")
        print(f"  busy-window median  ({len(hi_a):5d} ops): {pct(hi_a,0.5):.2f} GB/s")
        ratio = pct(lo_a, 0.5) / pct(hi_a, 0.5) if pct(hi_a, 0.5) else float('inf')
        print(f"  degradation factor: {ratio:.2f}x")
    else:
        print('  Not enough spread in observed alpha to compare endpoints '
              f'(n_idle={len(lo_a)}, n_busy={len(hi_a)}).')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('traces', nargs='+')
    ap.add_argument('--json_out', default=None,
                    help='write per-memcpy rows here for plotting')
    args = ap.parse_args()

    results = []
    for t in args.traces:
        r = analyze(t)
        if r is None:
            print(f'  [skip] no GPU events in {t}', file=sys.stderr)
            continue
        results.append(r)

    if not results:
        sys.exit('no usable traces')

    report(results)

    if args.json_out:
        payload = {
            'duty': [{k: v for k, v in r.items() if k != 'rows'} for r in results],
            'rows': [row for r in results for row in r['rows']],
        }
        Path(args.json_out).write_text(json.dumps(payload))
        print(f'\n  wrote {args.json_out}')

if __name__ == '__main__':
    main()
