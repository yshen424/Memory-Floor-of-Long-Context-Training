import argparse
import glob
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness import PLATFORMS, Run, TIER_A, LLAMA32_3B, static_line, floor_line, calibrate
from harness import CapacityModel, bracket
from harness.timemodel import TimeCalibration, fit_step_time, fit_kappa

PLAT = PLATFORMS['a30']
NAME, ARCH, PREFIX = 'A30', 'Ampere', 'a30'
SITES = {'a30': ('A30', 'Ampere', 'a30'), 'l40s': ('L40S', 'Ada', 'l40s')}
LEGACY = {'l40s': 'l64s'}

def pfile(out, name):
    p = os.path.join(out, f'{PREFIX}_{name}')
    old = LEGACY.get(PLAT.name)
    if not os.path.exists(p) and old and os.path.exists(os.path.join(out, f'{old}_{name}')):
        return os.path.join(out, f'{old}_{name}')
    return p
SHAPES = {'': TIER_A, 'llama3b': LLAMA32_3B}
NATIVE_3B = ('RS', 'RO', 'RSC', 'ROC')
TAG = re.compile(r'^(?P<blk>[A-Z]+\d*)_(?P<mode>static|offload|staticcpu|offloadcpu|evict)_'
                 r'(?P<dt>fp16|bf16)_(?P<attn>mem|flash)(?P<gqa>_gqa)?(?:_(?P<shape>llama3b))?_L(?P<L>\d+)_x(?P<x>\d+)'
                 r'(?:_cap(?P<cap>\d+))?_n(?P<n>\d+)(?:_r(?P<rep>\d+))?$')
V100_DELTA, V100_ALPHA = 33.1, 0.163
V100_ALPHA_RANGE = (0.139, 0.186)
TOL = 0.05

def line_for(arm):
    run = Run(shape=SHAPES[arm.get('shape', '')].with_layers(arm['L']), world=4,
              cpu_offload=arm['mode'].endswith('cpu'), repeat_kv=not arm['gqa'])
    if arm['mode'].startswith('static'):
        return static_line(run, PLAT)
    return floor_line(run, PLAT, 0)

def load(out):
    arms = []
    for d in sorted(glob.glob(os.path.join(out, '*'))):
        m = TAG.match(os.path.basename(d))
        if not m or not os.path.isdir(d):
            continue
        a = {'dir': d, 'blk': m['blk'], 'mode': m['mode'], 'dt': m['dt'], 'attn': m['attn'],
             'gqa': bool(m['gqa']), 'shape': m['shape'] or '', 'L': int(m['L']), 'x': int(m['x']),
             'cap': int(m['cap']) if m['cap'] else 0, 'n': int(m['n']),
             'rep': int(m['rep']) if m['rep'] else 0}
        p = os.path.join(d, 'steps_rank0.json')
        a['fit'] = os.path.exists(p)
        if a['fit']:
            j = json.load(open(p))
            st = j['steps']
            a['peak'] = max(s['peak_mb'] for s in st[1:]) if len(st) > 1 else st[0]['peak_mb']
            a['loss0'] = st[0]['loss']
            a['times'] = [s['step_ms'] / 1e3 for s in st[2:]]
            a['outside'] = [s['outside_alloc_mb'] for s in st[1:] if 'outside_alloc_mb' in s]
            a['attention'] = j['config'].get('attention', {})
        else:
            err = os.path.join(d, 'stderr.txt')
            txt = open(err, errors='replace').read() if os.path.exists(err) else ''
            r = re.findall(r'Tried to allocate ([\d.]+) (MiB|GiB)', txt)
            a['request'] = float(r[-1][0]) * (1024 if r[-1][1] == 'GiB' else 1) if r else None
            a['backward'] = 'backward' in txt.lower()
        arms.append(a)
    return arms

def say(*a):
    print(*a, flush=True)

def cmd_bytes(arms):
    byte_arms = [a for a in arms if a['blk'][0] in 'BDFT' and a['fit'] and not a['cap']]
    if not byte_arms:
        say('bytes: no completed byte arms'); return None
    say(f'=== BYTES: every arm against its harness line ({NAME} platform = V100 workspace carried) ===')
    offs = {}
    for a in sorted(byte_arms, key=lambda a: (a['blk'], a['L'], a['mode'], a['x'])):
        pred = line_for(a).at(a['x'])
        off = a['peak'] - pred
        offs.setdefault(a['blk'][0], []).append(off)
        kern = a['attention'].get('requested', '?') + (' +gqa' if a['gqa'] else '')
        say(f"  {os.path.basename(a['dir']):<58} peak {a['peak']:10.3f}  line {pred:10.3f}  "
            f"offset {off:+8.3f}   [{kern}, {a['dt']}]")
    b = offs.get('B', [])
    if b:
        c = statistics.median(b)
        spread = max(b) - min(b)
        say(f"A-P1 common offset over R-A1 arms: {c:+.3f} MiB (spread {spread:.3f}); "
            f"{'HELD' if spread <= TOL else f'MISS: offsets differ, a byte term moved on {ARCH}'}")
        say(f"A-P2 {NAME} workspace = {PLAT.ws_mib + c:.4f} MiB (lean: V100's {PLAT.ws_mib:.4f}, offset 0); "
            f"{'HELD' if abs(c) <= TOL else 'workspace differs: a platform constant, recorded'}")
    _twins(arms)
    _realistic(arms, statistics.median(b) if b else 0.0)
    return statistics.median(b) if b else None

def _realistic(arms, common):
    r = sorted((a for a in arms if a['blk'] == 'R' and a['fit'] and not a['cap']),
               key=lambda a: (a['mode'], a['x']))
    if not r:
        return
    say(f'--- R-L3 Llama-3.2-3B slope arms (harness line + the tier-A common offset {common:+.3f})')
    worst = 0.0
    for a in r:
        ln = line_for(a)
        off = a['peak'] - ln.at(a['x']) - common
        worst = max(worst, abs(off))
        say(f"    {a['mode']:<8} x={a['x']:>3}: peak {a['peak']:10.3f}  line {ln.at(a['x']):10.3f}  "
            f"residual {off:+8.3f}  [instant: {ln.binding(a['x'])}]")
    say(f"R-P1 {'HELD' if worst <= TOL else 'MISS'}: largest residual {worst:.3f} MiB (tolerance {TOL})")

def _twins(arms):
    done = {(a['mode'], a['dt'], a['attn'], a['gqa'], a['L'], a['x']): a for a in arms
            if a['fit'] and not a['cap'] and a['blk'][0] in 'BDF'}
    def twin(key, **chg):
        k = dict(zip(('mode', 'dt', 'attn', 'gqa', 'L', 'x'), key)); k.update(chg)
        return done.get(tuple(k[f] for f in ('mode', 'dt', 'attn', 'gqa', 'L', 'x')))
    for label, sel, chg, pid in (('bf16 vs fp16 (mem-efficient)', lambda k: k[1] == 'bf16' and k[2] == 'mem', {'dt': 'fp16'}, 'A-P3'),
                                 ('flash vs mem-efficient', lambda k: k[2] == 'flash' and not k[3], {'attn': 'mem'}, 'A-P4')):
        rows = [(k, done[k], twin(k, **chg)) for k in done if sel(k)]
        rows = [(k, a, t) for k, a, t in rows if t]
        if rows:
            say(f"--- {label}")
            worst = 0.0
            for k, a, t in rows:
                d = a['peak'] - t['peak']; worst = max(worst, abs(d))
                say(f"    {k[0]:<8} {k[1]} {k[2]} x={k[5]:>3}: {a['peak']:10.3f} vs {t['peak']:10.3f} ({d:+.3f}); "
                    f"step-0 loss {a['loss0']:.4f} vs {t['loss0']:.4f}")
            say(f"{pid} {'HELD' if worst <= TOL else 'MISS'}: largest twin difference {worst:.3f} MiB")
    g = sorted((k, a) for k, a in done.items() if k[3])
    if g:
        say('--- native GQA (flash, k/v at kv width): offset from the full-width twin, per x')
        for k, a in g:
            t = twin(k, gqa=False)
            if t:
                d = a['peak'] - t['peak']
                say(f"    {k[0]:<8} {k[1]} x={k[5]:>3}: {d:+.3f} MiB = {d / k[5]:+.4f} MiB/x "
                    f"(prediction -2.0000 native; 0.0000 if the kernel expands internally)")

def _pairs(arms, xs):
    T = [a for a in arms if a['blk'] == 'T' and a['fit'] and a['x'] in xs]
    by = {}
    for a in T:
        by.setdefault(a['x'], {}).setdefault(a['mode'], []).append(a)
    rows = []
    for x in sorted(by):
        s, o = by[x].get('static', []), by[x].get('offload', [])
        if not s:
            continue
        ts = statistics.mean(t for a in s for t in a['times'])
        to = statistics.mean(t for a in o for t in a['times']) if o else None
        reps = []
        for r in sorted({a['rep'] for a in s}):
            sr = [a for a in s if a['rep'] == r]; orr = [a for a in o if a['rep'] == r]
            if sr and orr:
                reps.append(statistics.mean(orr[0]['times']) - statistics.mean(sr[0]['times']))
        rows.append((x, ts, to, reps))
    return rows

def cmd_timecal(arms, pred):
    rows = _pairs(arms, (16, 32, 64))
    say('=== TIME CALIBRATION (x = 16, 32, 64; static and offload, two repeats each) ===')
    for x, ts, to, reps in rows:
        say(f"  x={x:>3}  static {ts:8.3f} s  offload {to if to is None else round(to, 3)} s  "
            f"delta per repeat {[round(r, 3) for r in reps]}")
    if len({r[0] for r in rows}) < 3:
        say('timecal: fewer than three x values completed; no prediction written'); return
    a, b, c = fit_step_time([(x, ts) for x, ts, _, _ in rows])
    pairs = [(x, statistics.mean(reps)) for x, _, _, reps in rows if reps and x >= 32]
    kappa = fit_kappa(pairs) if pairs else None
    if kappa is None:
        say('timecal: no paired differences at x >= 32; no prediction written'); return
    cal = TimeCalibration(a, b, c, kappa, f'{NAME} in-job: T on x=16/32/64 static, kappa on x=32/64 pairs')
    lines = [f"# {NAME} TIME PREDICTIONS, written {time.strftime('%Y-%m-%dT%H:%M:%S')} BEFORE the x=100/150 arms ran",
             f"# T(x) = {a:.6f} x^2 + {b:.5f} x + {c:.4f} s   (static, L=64, world 4, fp16 mem-efficient)",
             f"# kappa = {kappa:.5f} s per x   (V100: 0.032)",
             f"# T-P3 predicted at x=100: static {cal.step_time(100):.2f} s, tax {cal.tax(100):.2%}; "
             f"x=150: static {cal.step_time(150):.2f} s, tax {cal.tax(150):.2%}",
             f"# T-P4 crossings: x*(2%) = {cal.crossing(0.02) or float('nan'):.1f}, "
             f"x*(1%) = {cal.crossing(0.01) or float('nan'):.1f}   (V100: ~79 and ~150)",
             "# Falsifier: a measured tax at x=100 or 150 outside +-30% of these, or kappa at",
             "# x=100/150 differing from the calibrated kappa by more than 30% (the cost is not linear in x here)."]
    open(pred, 'w').write('\n'.join(lines) + '\n')
    say('\n'.join(lines))

def _ladders(arms):
    lad = {}
    for a in arms:
        if a['blk'].startswith('C') and a['cap']:
            lad.setdefault(a['blk'], []).append(a)
    return {k: sorted(v, key=lambda a: a['x']) for k, v in lad.items()}

def cmd_capcal(arms, pred, node):
    lad = _ladders(arms)
    pts = []
    say('=== CAPPED LADDERS (R-A4) ===')
    for blk, rr in sorted(lad.items()):
        cap = rr[0]['cap'] * 1024.0
        for a in rr:
            pk = a['peak'] if a['fit'] else line_for(a).at(a['x'])
            pts.append((pk - cap, 4.0 * a['x'], a['fit']))
        fits = [a['x'] for a in rr if a['fit']]; fails = [a for a in rr if not a['fit']]
        req = fails[0]['request'] if fails else None
        say(f"  {blk}: cap {rr[0]['cap']} GiB {rr[0]['mode']:<8} bracket [{max(fits) if fits else '<'+str(rr[0]['x'])}, "
            f"{fails[0]['x'] if fails else 'open'})  failing request {req} MiB (4x = {4.0 * fails[0]['x'] if fails else '-'})")
    m = calibrate(pts)
    if m is None:
        say(f'C-P2 MISS: no (alpha, delta) separates the {NAME} capped rungs; falling back to V100 constants')
        delta, alpha, rng = V100_DELTA, V100_ALPHA, None
    else:
        delta, alpha, rng = -m.c0_mib, m.alpha, m.alpha_range
        say(f"C-P2 {NAME} capped reserve: alpha {alpha:.3f} (feasible {rng[0]:.3f}-{rng[1]:.3f}), delta {delta:.1f} MiB "
            f"(V100: alpha 0.139-0.186, delta 26.8-39.4)")
    outside = [v for a in arms if a['fit'] and not a['cap'] for v in a.get('outside', [])]
    ctx = max(outside) if outside else None
    total = node.get('total_mib')
    if ctx is None or total is None:
        say('capcal: no outside-allocator reading or card total; native bracket not predicted'); return
    c0 = total - ctx - delta
    b_offs = [a['peak'] - line_for(a).at(a['x']) for a in arms
              if a['blk'] == 'B' and a['fit'] and not a['cap']]
    off = statistics.median(b_offs) if b_offs else 0.0
    ln = static_line(Run(shape=TIER_A, world=4), PLAT)
    native_fit = lambda al, de: max(x for x in range(20, 400, 2)
                                    if ln.at(x) + off <= total - ctx - de - al * 4.0 * x)
    fit = native_fit(alpha, delta)
    span = [fit]
    for al in [a / 1000 for a in range(0, 1001)]:
        lo = max((p + al * r for p, r, f in pts if f), default=float('-inf'))
        hi = min((p + al * r for p, r, f in pts if not f), default=float('inf'))
        if lo < hi:
            for de in (-hi, -lo):
                span.append(native_fit(al, de))
    f_lo, f_hi = min(span), max(span)
    rungs = list(range(f_lo - 4, f_hi + 8, 2))
    lines = [f"# {NAME} NATIVE STATIC BRACKET, written {time.strftime('%Y-%m-%dT%H:%M:%S')} BEFORE the native ladder ran",
             f"# card total {total:.2f} MiB; outside the allocator {ctx:.2f} MiB (max over uncapped arms; V100 inferred ~488)",
             f"# C0 = total - outside - delta = {c0:.2f} MiB; alpha {alpha:.3f}; line = harness static {off:+.3f} MiB (R-A1 offset)",
             f"# N-P1 largest fit x = {fit} at the centre; in [{f_lo}, {f_hi}] over every (alpha, delta) the four",
             f"#      capped ladders admit (rung step 2). FALSIFIER: the measured largest fit outside that range.",
             f"# N-P2 the failure is a step-1 backward OOM on exactly 4x MiB",
             f"RUNGS: {' '.join(map(str, rungs))}"]
    open(pred, 'w').write('\n'.join(lines) + '\n')
    say('\n'.join(lines))

def cmd_read(arms, out, node):
    say(f"=== {NAME} JOB REPORT: {out} ({sum(a['fit'] for a in arms)} of {len(arms)} arms completed) ===")
    for a in arms:
        if not a['fit'] and not a['cap'] and a['blk'] not in ('N',) + NATIVE_3B:
            say(f"  FAILED (not a ladder rung): {os.path.basename(a['dir'])}, request {a.get('request')} -- read its stderr.txt")
    cmd_bytes(arms)
    rows = _pairs(arms, (16, 32, 64, 100, 150))
    if rows:
        say('=== TIME (R-A5): pooled static, offload, per-repeat delta, tax ===')
        for x, ts, to, reps in rows:
            d = statistics.mean(reps) if reps else float('nan')
            say(f"  x={x:>3}  static {ts:8.3f} s  delta {d:+.3f} s  per x {d / x:+.4f}  tax {d / ts:6.2%}  "
                f"repeats {[round(r, 3) for r in reps]}")
        pf = pfile(out, 'pred_time.txt')
        if os.path.exists(pf):
            say('  pre-registered in-job:'); say(open(pf).read().rstrip())
    lad = _ladders(arms)
    if lad:
        say('=== CAPPED LADDERS vs the V100-carried prediction (C-P1) ===')
        pred = {'CA': 34, 'CB': 57, 'CC': 110, 'CD': 190}
        for blk, rr in sorted(lad.items()):
            fits = [a['x'] for a in rr if a['fit']]
            say(f"  {blk}: largest fit {max(fits) if fits else None}  (V100 measured {pred.get(blk)}; "
                f"{'HELD' if fits and max(fits) == pred.get(blk) else 'differs -- see capcal'})")
            for a in rr:
                if a['fit']:
                    off = a['peak'] - line_for(a).at(a['x'])
                    if abs(off) > TOL + 1:
                        say(f"    x={a['x']} peak off its line by {off:+.3f} MiB")
    nat = sorted((a for a in arms if a['blk'] == 'N'), key=lambda a: a['x'])
    if nat:
        fits = [a['x'] for a in nat if a['fit']]; fails = [a for a in nat if not a['fit']]
        say(f"=== NATIVE STATIC LADDER: bracket [{max(fits) if fits else None}, {fails[0]['x'] if fails else 'open'}), "
            f"failing request {fails[0]['request'] if fails else None} MiB")
        pf = pfile(out, 'pred_native.txt')
        if os.path.exists(pf):
            say(open(pf).read().rstrip())
    _realistic_brackets(arms, node)

def _realistic_brackets(arms, node):
    lad = {}
    for a in arms:
        if a['blk'] in NATIVE_3B:
            lad.setdefault(a['blk'], []).append(a)
    if not lad:
        return
    outside = [v for a in arms if a['fit'] and not a['cap'] for v in a.get('outside', [])]
    ctx, total = (max(outside) if outside else None), node.get('total_mib')
    say('=== R-L3 NATIVE BRACKETS (Llama-3.2-3B, 48 GB, no cap) ===')
    for blk, rr in sorted(lad.items()):
        rr = sorted(rr, key=lambda a: a['x'])
        fits = [a['x'] for a in rr if a['fit']]; fails = [a for a in rr if not a['fit']]
        ln = line_for(rr[0])
        say(f"  {blk} {rr[0]['mode']:<8} bracket [{max(fits) if fits else '<' + str(rr[0]['x'])}, "
            f"{fails[0]['x'] if fails else 'open'})  failing request {fails[0]['request'] if fails else None} MiB "
            f"(16x = {16.0 * fails[0]['x'] if fails else '-'}; backward: {fails[0]['backward'] if fails else '-'})")
        if ctx is not None and total is not None:
            preds = []
            for al in (V100_ALPHA_RANGE[0], V100_ALPHA, V100_ALPHA_RANGE[1]):
                m = CapacityModel(total - ctx - V100_DELTA, al)
                run = Run(shape=SHAPES[rr[0]['shape']].with_layers(rr[0]['L']), world=4)
                preds.append(bracket(ln, run, m, range(1, 2 * rr[-1]['x'] + 50))[1])
            say(f"    harness (V100 reserve carried, measured context {ctx:.1f} MiB): largest fit {preds[1]} "
                f"(alpha range: {min(preds)}-{max(preds)})  measured {max(fits) if fits else None}")
        if PLAT.capacity is not None:
            run = Run(shape=SHAPES[rr[0]['shape']].with_layers(rr[0]['L']), world=4,
                      cpu_offload=rr[0]['mode'].endswith('cpu'))
            p = bracket(ln, run, PLAT.capacity, range(1, 2 * rr[-1]['x'] + 50))[1]
            say(f"    harness native model ({PLAT.capacity.provenance}): largest fit {p}  "
                f"measured {max(fits) if fits else None}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['bytes', 'timecal', 'capcal', 'read'])
    ap.add_argument('out')
    ap.add_argument('--pred')
    ap.add_argument('--platform', choices=sorted(SITES), default='a30')
    a = ap.parse_args()
    global PLAT, NAME, ARCH, PREFIX
    PLAT = PLATFORMS[a.platform]
    NAME, ARCH, PREFIX = SITES[a.platform]
    arms = load(a.out)
    nf = pfile(a.out, 'node.json')
    node = json.load(open(nf)) if os.path.exists(nf) else {}
    if a.cmd == 'bytes':
        cmd_bytes(arms)
    elif a.cmd == 'timecal':
        cmd_timecal(arms, a.pred)
    elif a.cmd == 'capcal':
        cmd_capcal(arms, a.pred, node)
    else:
        cmd_read(arms, a.out, node)

if __name__ == '__main__':
    main()
