"""Where stock offload's extra time goes, per traced step: an exact decomposition.

Each chrome trace (one rank, one profiled step) splits its GPU span into four
disjoint parts:
    C  compute   any non-NCCL, non-copy kernel running
    X  copy-only a host<->device copy running and no compute kernel
    N  other     only NCCL / memset activity
    I  idle      nothing on the GPU
so span = C + X + N + I exactly. For a static/offload pair at the same x,
    span_off - span_static = dC + X_off + dN + dI   (X_static ~ 0),
which says whether offload's copies add their full time (dI ~ 0) or fill GPU time
that the static step leaves idle (dI < 0). V100 traces (7758) show stock offload's
copies never overlap compute (hidden 0.0 ms): they run in order on the compute
stream, so "hiding behind compute" is not available to them. Question (L40S):
why is offload's extra time at x=64 a third below kappa*x?

Usage:
    python3 tools/copy_exposure.py --offload <traces...> [--static <traces...>] [--json out.json]
"""
import argparse
import gzip
import json
import statistics
import sys
from pathlib import Path

MIN_US = 20.0


def load(path):
    op = gzip.open if str(path).endswith('.gz') else open
    with op(path, 'rt') as fh:
        t = json.load(fh)
    return t.get('traceEvents', t if isinstance(t, list) else [])


def merge(spans):
    out = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def overlap(a, b):
    """Total overlap (us) of two sorted disjoint unions."""
    i = j = 0
    tot = 0.0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if s < e:
            tot += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def busy(u):
    return sum(e - s for s, e in u)


def analyze(path):
    ev = [e for e in load(path) if e.get('ph') == 'X' and e.get('dur') is not None
          and (e.get('cat') or '').lower() in ('kernel', 'gpu_memcpy', 'gpu_memset')]
    if not ev:
        return None
    copies = {'D2H': [], 'H2D': []}
    compute, nbytes = [], {'D2H': 0, 'H2D': 0}
    for e in ev:
        n, s, d = e.get('name', ''), float(e['ts']), float(e['dur'])
        if 'Memcpy DtoH' in n or 'Memcpy HtoD' in n:
            if d < MIN_US:
                continue
            k = 'D2H' if 'DtoH' in n else 'H2D'
            copies[k].append((s, s + d))
            nbytes[k] += int((e.get('args') or {}).get('bytes', 0) or 0)
        elif 'nccl' in n.lower() or 'Memset' in n or 'Memcpy' in n:
            continue
        else:
            compute.append((s, s + d))
    cu = merge(compute)
    allc = merge(copies['D2H'] + copies['H2D'])
    act = merge([(float(e['ts']), float(e['ts']) + float(e['dur'])) for e in ev])
    span = act[-1][1] - act[0][0]
    r = {'trace': str(path), 'span_ms': span / 1e3, 'compute_busy_ms': busy(cu) / 1e3,
         'copy_busy_ms': busy(allc) / 1e3, 'copy_hidden_ms': overlap(allc, cu) / 1e3}
    r['copy_exposed_ms'] = r['copy_busy_ms'] - r['copy_hidden_ms']
    r['C_ms'], r['X_ms'] = r['compute_busy_ms'], r['copy_exposed_ms']
    r['I_ms'] = (span - busy(act)) / 1e3
    r['N_ms'] = r['span_ms'] - r['C_ms'] - r['X_ms'] - r['I_ms']
    r['exposed_fraction'] = r['copy_exposed_ms'] / r['copy_busy_ms'] if r['copy_busy_ms'] else float('nan')
    for k in ('D2H', 'H2D'):
        u = merge(copies[k])
        r[f'{k}_busy_ms'] = busy(u) / 1e3
        r[f'{k}_exposed_ms'] = (busy(u) - overlap(u, cu)) / 1e3
        r[f'{k}_GBps'] = nbytes[k] / (busy(u) * 1e-6) / 1e9 if busy(u) else float('nan')
    return r


def summarize(rows, label):
    keys = ('span_ms', 'compute_busy_ms', 'copy_busy_ms', 'copy_hidden_ms', 'copy_exposed_ms',
            'exposed_fraction', 'D2H_exposed_ms', 'H2D_exposed_ms', 'D2H_GBps', 'H2D_GBps',
            'C_ms', 'X_ms', 'N_ms', 'I_ms')
    med = {k: statistics.median(r[k] for r in rows) for k in keys}
    print(f"  {label:<8} ranks={len(rows)}  span {med['span_ms']:9.1f} ms  compute {med['compute_busy_ms']:9.1f}  "
          f"copy {med['copy_busy_ms']:7.1f} (hidden {med['copy_hidden_ms']:7.1f}, exposed {med['copy_exposed_ms']:7.1f}, "
          f"{med['exposed_fraction']:.0%})  D2H/H2D exposed {med['D2H_exposed_ms']:.1f}/{med['H2D_exposed_ms']:.1f} ms  "
          f"at {med['D2H_GBps']:.1f}/{med['H2D_GBps']:.1f} GB/s")
    print(f"           span = C {med['C_ms']:.1f} + X {med['X_ms']:.1f} + N {med['N_ms']:.1f} + I {med['I_ms']:.1f} ms")
    return med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--offload', nargs='+', required=True)
    ap.add_argument('--static', nargs='*', default=[])
    ap.add_argument('--json')
    a = ap.parse_args()
    out = {}
    for label, paths in (('offload', a.offload), ('static', a.static)):
        rows = [r for r in (analyze(p) for p in paths) if r]
        if rows:
            out[label] = {'per_rank': rows, 'median': summarize(rows, label)}
    if 'offload' not in out:
        sys.exit('no usable offload traces')
    if 'static' in out:
        o, st = out['offload']['median'], out['static']['median']
        d = {k: o[k] - st[k] for k in ('span_ms', 'C_ms', 'X_ms', 'N_ms', 'I_ms')}
        out['pair'] = d
        print(f"  pair: span {d['span_ms']:+.1f} ms = dC {d['C_ms']:+.1f} + dX {d['X_ms']:+.1f} + dN {d['N_ms']:+.1f} "
              f"+ dI {d['I_ms']:+.1f}   (copy busy {o['copy_busy_ms']:.1f} ms)")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
