"""Reader for the A30 follow-up jobs (remote/a30_followup/).

    python3 tools/a30_followup.py tax    <out dir>   # job A: why offload beats static at long x
    python3 tools/a30_followup.py native <out dir>   # job B: the native stock-offload bracket

Arm directories: <BLK>_<mode>_<dt>_<attn>_L<L>_x<x>_n<n>[_r<k>]. Step times are the
mean of steps 1+ (step 0 carries warm-up); delta = offload - static. Allocator
counters are diffed per step from the cumulative values the driver records;
gpu.csv (1 Hz nvidia-smi) is read over samples with utilization >= 90%.
"""
import glob
import json
import os
import re
import statistics
import sys

TAG = re.compile(r'(?P<blk>[A-Z]+)_(?P<mode>static|offload)_(?P<dt>fp16|bf16)_(?P<attn>mem|flash)'
                 r'_L(?P<L>\d+)_x(?P<x>\d+)_n(?P<n>\d+)(?:_r(?P<rep>\d+))?$')
COUNTERS = ('num_device_alloc', 'num_device_free', 'num_alloc_retries', 'num_sync_all_streams')
IDLE, APP_CLOCKS = 0x1, 0x4          # throttle-reason bits that are not throttling


def load(out):
    arms = []
    for d in sorted(glob.glob(os.path.join(out, '*'))):
        m = TAG.match(os.path.basename(d))
        if not m:
            continue
        a = {k: m[k] for k in ('blk', 'mode', 'dt', 'attn')}
        a.update(L=int(m['L']), x=int(m['x']), n=int(m['n']), rep=int(m['rep'] or 0), dir=d)
        p = os.path.join(d, 'steps_rank0.json')
        a['fit'] = os.path.exists(p)
        if a['fit']:
            j = json.load(open(p))
            st, cfg = j['steps'], j['config']
            later = st[1:] or st
            a['t'] = statistics.mean(s['step_ms'] for s in later) / 1e3
            a['t_sd'] = statistics.stdev(s['step_ms'] for s in later) / 1e3 if len(later) > 1 else 0.0
            a['t0'] = st[0]['step_ms'] / 1e3
            a['peak'] = max(s['peak_mb'] for s in later)
            a['ballast'] = cfg.get('hbm_ballast_mb', 0.0) or 0.0
            a['alloc_conf'] = cfg.get('alloc_conf')
            al = [s.get('alloc') or {} for s in st]
            a['per_step'] = {}
            for k in COUNTERS:
                v = [x.get(k) for x in al]
                if len(v) > 2 and None not in v:
                    a['per_step'][k] = (v[-1] - v[1]) / (len(v) - 2)      # steps 2.. (step 1 allocates AdamW)
            a['segments'] = al[-1].get('segment.all.current') if al else None
        else:
            e = open(os.path.join(d, 'stderr.txt'), errors='replace').read() if os.path.exists(os.path.join(d, 'stderr.txt')) else ''
            r = re.findall(r'Tried to allocate ([\d.]+) (MiB|GiB)', e)
            a['request'] = (float(r[-1][0]) * (1024 if r[-1][1] == 'GiB' else 1)) if r else None
            # torch prints 2 decimals: a GiB request is known only to +-0.005 GiB (642032: 2272 MiB -> "2.22 GiB")
            a['request_tol'] = (5.12 if r[-1][1] == 'GiB' else 0.01) if r else None
            a['backward'] = 'backward' in e
            a['hostkill'] = os.path.exists(os.path.join(d, 'hostmem_kill'))
        a['gpu'] = gpu_stats(os.path.join(d, 'gpu.csv'))
        hm = os.path.join(d, 'hostmem_min_avail_pct')
        a['host_min_avail'] = int(open(hm).read()) if os.path.exists(hm) else None
        arms.append(a)
    return arms


def gpu_stats(p):
    if not os.path.exists(p):
        return None
    sm, pw, tmp, thr, n = [], [], [], 0, 0
    for line in open(p):
        f = [x.strip() for x in line.split(',')]
        if len(f) < 8:
            continue
        try:
            util = float(f[6])
            if util < 90:
                continue
            sm.append(float(f[2])); pw.append(float(f[4])); tmp.append(float(f[5]))
            mask = int(f[7], 16) if f[7].startswith('0x') else 0
            thr += bool(mask & ~(IDLE | APP_CLOCKS)); n += 1
        except ValueError:
            continue
    if not n:
        return None
    return {'sm_mhz': statistics.mean(sm), 'power_w': statistics.mean(pw), 'temp_c': statistics.mean(tmp),
            'throttled_frac': thr / n, 'samples': n}


def find(arms, **kw):
    return [a for a in arms if all(a.get(k) == v for k, v in kw.items())]


def fmt_gpu(g):
    return (f"SM {g['sm_mhz']:.0f} MHz, {g['power_w']:.0f} W, {g['temp_c']:.0f} C, "
            f"throttled {100 * g['throttled_frac']:.0f}% of {g['samples']} samples") if g else 'no gpu.csv'


def fmt_alloc(a):
    ps = a.get('per_step') or {}
    return ', '.join(f"{k.replace('num_', '')} {v:.1f}/step" for k, v in ps.items()) or 'no counters'


def pair_delta(arms, blk, x, rep=None):
    kw = dict(blk=blk, x=x, fit=True)
    if rep is not None:
        kw['rep'] = rep
    s, o = find(arms, mode='static', **kw), find(arms, mode='offload', **kw)
    if not s or not o:
        return None, s, o
    return o[0]['t'] - s[0]['t'], s, o


def within(v, lo, hi):
    return v is not None and lo <= v <= hi


def cmd_tax(arms):
    print('=== A30 FOLLOW-UP A: STEP TIMES (mean of steps 1+; delta = offload - static) ===')
    for a in sorted(arms, key=lambda a: (a['blk'], a['x'], a['rep'], a['mode'])):
        if a['fit']:
            extra = (f"  ballast {a['ballast']:.0f} MiB" if a['ballast'] else '') + \
                    (f"  alloc_conf '{a['alloc_conf']}'" if a['alloc_conf'] is not None else '')
            print(f"  {os.path.basename(a['dir']):<38} {a['t']:9.3f} s (sd {a['t_sd']:.3f}, step0 {a['t0']:.1f})  "
                  f"peak {a['peak']:9.1f}{extra}")
            print(f"      alloc: {fmt_alloc(a)}; segments {a['segments']}   gpu: {fmt_gpu(a['gpu'])}")
        else:
            why = 'HOST WATCHDOG' if a.get('hostkill') else f"request {a.get('request')} MiB, backward {a.get('backward')}"
            print(f"  {os.path.basename(a['dir']):<38} DID NOT COMPLETE ({why})")
    xs = sorted({a['x'] for a in arms if a['blk'] == 'S'})
    print('--- S deltas')
    D = {}
    for x in xs:
        reps = sorted({a['rep'] for a in arms if a['blk'] == 'S' and a['x'] == x})
        ds = [pair_delta(arms, 'S', x, r)[0] for r in reps]
        ds = [d for d in ds if d is not None]
        if ds:
            D[x] = statistics.mean(ds)
            s0 = find(arms, blk='S', x=x, mode='static', fit=True)
            print(f"  x={x:>3}: delta {D[x]:+.3f} s  repeats {[round(d, 3) for d in ds]}  "
                  f"tax {100 * D[x] / s0[0]['t']:+.2f}%")
    if 150 in D or 100 in D:
        r150 = [pair_delta(arms, 'S', 150, r)[0] for r in (1, 2)]
        ok = all(within(d, -4.6, -2.6) for d in r150) and within(D.get(100), -1.6, -0.6)
        rd = lambda v: None if v is None else round(v, 3)
        print(f"R1 {'HELD' if ok else 'FAILED'}: delta(150) repeats {[rd(d) for d in r150]} in [-4.6, -2.6]; "
              f"delta(100) {rd(D.get(100))} in [-1.6, -0.6]")
        if 80 in D:
            ok2 = within(D[80], -0.8, 0.3) and (120 not in D or min(D[100], D[150]) <= D[120] <= max(D[100], D[150]))
            print(f"R2 {'HELD' if ok2 else 'FAILED'}: delta(80) {D[80]:+.3f} in [-0.8, +0.3]; delta(120) "
                  f"{rd(D.get(120))} between delta(100) and delta(150)")
    gap = -D[max(D)] if D and D[max(D)] < 0 else None
    xg = max(D) if D else None
    if gap:
        print(f"--- the gap to explain: static slower by {gap:.3f} s at x={xg}")
        s = find(arms, blk='S', x=xg, mode='static', fit=True); o = find(arms, blk='S', x=xg, mode='offload', fit=True)
        # H1 counters + E
        cs = {k: statistics.mean(a['per_step'].get(k, 0) for a in s) for k in COUNTERS}
        co = {k: statistics.mean(a['per_step'].get(k, 0) for a in o) for k in COUNTERS}
        print(f"H1 counters static {cs}\n   counters offload {co}\n"
              f"   (context only under expandable segments: page mapping is not counted; E is the test)")
        es = find(arms, blk='E', x=xg, fit=True)
        if es:
            print('   counters with expandable segments OFF: ' +
                  '; '.join(f"{a['mode']} {fmt_alloc(a)}" for a in es))
        de, se, oe = pair_delta(arms, 'E', xg)
        if de is not None:
            print(f"H1 E (expandable OFF): delta {de:+.3f} s -> {'removes >=70% of the gap' if abs(de) <= 0.3 * gap else 'gap persists'}")
        elif se or oe:
            print(f"H1 E incomplete at x={xg}: {[os.path.basename(a['dir']) for a in find(arms, blk='E', x=xg)]}")
        # H2 ballast
        g = find(arms, blk='G', x=xg, mode='offload', fit=True)
        if g and o:
            slow = g[0]['t'] - statistics.mean(a['t'] for a in o)
            print(f"H2 G (offload + {g[0]['ballast']:.0f} MiB ballast): {slow:+.3f} s vs plain offload -> "
                  f"{'reproduces >=70% of the gap' if slow >= 0.7 * gap else 'footprint does not explain it'}")
        # H3 clocks
        gs = [a['gpu'] for a in s if a['gpu']]; go = [a['gpu'] for a in o if a['gpu']]
        if gs and go:
            ds = statistics.mean(x['sm_mhz'] for x in gs) / statistics.mean(x['sm_mhz'] for x in go) - 1
            print(f"H3 SM clock static/offload - 1 = {100 * ds:+.2f}% (gap {100 * gap / s[0]['t']:.2f}% of the step); "
                  f"throttled static {statistics.mean(x['throttled_frac'] for x in gs):.2f} vs offload "
                  f"{statistics.mean(x['throttled_frac'] for x in go):.2f}; power {statistics.mean(x['power_w'] for x in gs):.0f} "
                  f"vs {statistics.mean(x['power_w'] for x in go):.0f} W")
    # H4 profiled pair
    q = {a['mode']: a for a in arms if a['blk'] == 'Q' and a['fit']}
    shares = {m: json.load(open(os.path.join(a['dir'], 'attn_share.json'))) for m, a in q.items()
              if os.path.exists(os.path.join(a['dir'], 'attn_share.json'))}
    if len(shares) == 2:
        s, o = shares['static'], shares['offload']
        print('H4 Q profiled (static - offload, ms over the active steps):')
        for k in ('gpu_span_ms', 'compute_busy_ms'):
            print(f"    {k:<16} {s[k] - o[k]:+10.1f}   (static {s[k]:.0f}, offload {o[k]:.0f})")
        print(f"    {'idle':<16} {(s['gpu_span_ms'] - s['compute_busy_ms']) - (o['gpu_span_ms'] - o['compute_busy_ms']):+10.1f}")
        for k in s.get('kernel_ms', {}):
            print(f"    kernel {k:<9} {s['kernel_ms'][k] - o['kernel_ms'].get(k, 0):+10.1f}")
    for a in arms:
        if a['blk'] == 'M':
            p = os.path.join(a['dir'], 'liveset.txt')
            print(f"M {os.path.basename(a['dir'])}: {'liveset.txt written' if os.path.exists(p) else 'no live set (snapshot missing?)'}")


def cmd_native(arms):
    rr = sorted((a for a in arms if a['blk'] == 'NO'), key=lambda a: a['x'])
    print('=== A30 FOLLOW-UP B: NATIVE STOCK-OFFLOAD LADDER ===')
    for a in rr:
        hm = f"host min avail {a['host_min_avail']}%" if a['host_min_avail'] is not None else 'host n/a'
        if a['fit']:
            print(f"  x={a['x']:>3} fit   peak {a['peak']:9.1f}  step0 {a['t0'] / 60:.1f} min  step1 {a['t'] / 60:.1f} min  {hm}")
        else:
            why = 'HOST WATCHDOG KILL' if a.get('hostkill') else f"request {a.get('request')} MiB (4x = {4.0 * a['x']:.1f}), backward {a.get('backward')}"
            print(f"  x={a['x']:>3} FAIL  {why}  {hm}")
    fits = [a['x'] for a in rr if a['fit']]; fails = [a['x'] for a in rr if not a['fit'] and not a.get('hostkill')]
    lf, sf = (max(fits) if fits else None), (min(x for x in fails if not fits or x > max(fits)) if fails else None)
    print(f"  bracket [{lf}, {sf})")
    if lf is not None:
        print(f"NO-P1 {'HELD' if 560 <= lf <= 572 else 'MISS'}: largest fit {lf} (predicted 566, range 561-570; falsifier outside [560, 572])")
        print(f"NO-P3 span {lf}/164 = {lf / 164:.3f} (predicted 3.44; V100 356/104 = 3.42)")
        f = [a for a in rr if a['fit'] and a['x'] == lf][0]
        print(f"NO-P4 {'HELD' if 0.85 * 84 <= f['t'] / 60 <= 1.15 * 84 else 'MISS'}: step time {f['t'] / 60:.1f} min (predicted 84 +- 15%)")
    fl = [a for a in rr if not a['fit'] and a['x'] == sf]
    if fl:
        ok = fl[0].get('request') is not None and abs(fl[0]['request'] - 4.0 * sf) <= fl[0]['request_tol'] and fl[0].get('backward')
        print(f"NO-P2 {'HELD' if ok else 'MISS'}: failure request {fl[0].get('request')} MiB (4x = {4.0 * sf:.1f}), in backward {fl[0].get('backward')}")
    mins = [a['host_min_avail'] for a in rr if a['host_min_avail'] is not None]
    if mins:
        print(f"NO-P5 {'HELD' if min(mins) >= 45 else 'MISS'}: lowest host MemAvailable {min(mins)}% (floor 45%; watchdog 30%)")


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ('tax', 'native'):
        sys.exit(__doc__)
    arms = load(sys.argv[2])
    (cmd_tax if sys.argv[1] == 'tax' else cmd_native)(arms)


if __name__ == '__main__':
    main()
