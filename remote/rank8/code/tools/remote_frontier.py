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
from harness import PLATFORMS, Run, TIER_A, LLAMA32_3B, MISTRAL7B, static_line, floor_line, calibrate
from harness import CapacityModel, bracket, request_mib_per_x
from harness.timemodel import TimeCalibration, fit_step_time, fit_kappa
sys.path.insert(0, str(Path(__file__).resolve().parent))
from a30_followup import gpu_stats, fmt_gpu

PLAT = PLATFORMS['a30']
NAME, ARCH, PREFIX = 'A30', 'Ampere', 'a30'
SITES = {'a30': ('A30', 'Ampere', 'a30'), 'l40s': ('L40S', 'Ada', 'l40s'), 'a100': ('A100', 'Ampere', 'a100')}
VIA = re.compile(r' via (P2P/[A-Za-z]+|SHM|NET/[A-Za-z]+|NVLS|CollNet)')
LEGACY = {'l40s': 'l64s'}

def pfile(out, name):
    p = os.path.join(out, f'{PREFIX}_{name}')
    old = LEGACY.get(PLAT.name)
    if not os.path.exists(p) and old and os.path.exists(os.path.join(out, f'{old}_{name}')):
        return os.path.join(out, f'{old}_{name}')
    return p
SHAPES = {'': TIER_A, 'llama3b': LLAMA32_3B, 'mistral7b': MISTRAL7B}
NATIVE_3B = ('RS', 'RO', 'RSC', 'ROC')
TAG = re.compile(r'^(?P<blk>[A-Z]+\d*)_(?P<mode>static|offload|staticcpu|offloadcpu|evict)_'
                 r'(?P<dt>fp16|bf16)_(?P<attn>mem|flash)(?P<gqa>_gqa)?(?:_(?P<shape>llama3b|mistral7b))?(?:_w(?P<world>\d+))?_L(?P<L>\d+)_x(?P<x>\d+)'
                 r'(?:_cap(?P<cap>\d+))?_n(?P<n>\d+)(?:_r(?P<rep>\d+))?$')
V100_DELTA, V100_ALPHA = 33.1, 0.163
V100_ALPHA_RANGE = (0.139, 0.186)
TOL = 0.05
AMPERE_DELTA, AMPERE_ALPHA = 59.5, 0.093
AMPERE_ALPHA_RANGE = (0.0, 0.186)

def line_for(arm):
    run = Run(shape=SHAPES[arm.get('shape', '')].with_layers(arm['L']), world=arm.get('world', 4),
              cpu_offload=arm['mode'].endswith('cpu'), repeat_kv=not arm['gqa'],
              attn='flash' if arm['attn'] == 'flash' else 'mem_efficient')
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
             'gqa': bool(m['gqa']), 'shape': m['shape'] or '', 'world': int(m['world']) if m['world'] else 4,
             'L': int(m['L']), 'x': int(m['x']),
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
            a['ballast'] = j['config'].get('hbm_ballast_mb', 0.0) or 0.0
        else:
            err = os.path.join(d, 'stderr.txt')
            txt = open(err, errors='replace').read() if os.path.exists(err) else ''
            r = re.findall(r'Tried to allocate ([\d.]+) (MiB|GiB)', txt)
            a['request'] = float(r[-1][0]) * (1024 if r[-1][1] == 'GiB' else 1) if r else None
            a['backward'] = 'backward' in txt.lower()
        a['gpu'] = gpu_stats(os.path.join(d, 'gpu.csv'))
        vias = {}
        for f in ('stdout.txt', 'stderr.txt'):
            if os.path.exists(os.path.join(d, f)):
                for v in VIA.findall(open(os.path.join(d, f), errors='replace').read()):
                    vias[v] = vias.get(v, 0) + 1
        a['nccl_via'] = vias
        arms.append(a)
    return arms

def say(*a):
    print(*a, flush=True)

def cmd_bytes(arms):
    byte_arms = [a for a in arms if a['blk'][0] in 'BDFTH' and a['fit'] and not a['cap'] and not a.get('ballast')]
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
    def plane(k):
        sh = SHAPES[''].with_layers(k[4])
        return 2 * sh.n_q * sh.head_dim * k[5] / 1024.0
    for label, sel, chg, pid, expect in (('bf16 vs fp16 (mem-efficient)', lambda k: k[1] == 'bf16' and k[2] == 'mem', {'dt': 'fp16'}, 'A-P3', lambda k: 0.0),
                                         ('flash vs mem-efficient', lambda k: k[2] == 'flash' and not k[3], {'attn': 'mem'}, 'A-P4', plane)):
        rows = [(k, done[k], twin(k, **chg)) for k in done if sel(k)]
        rows = [(k, a, t) for k, a, t in rows if t]
        if rows:
            say(f"--- {label}")
            worst = 0.0
            for k, a, t in rows:
                d = a['peak'] - t['peak']; e = expect(k); worst = max(worst, abs(d - e))
                say(f"    {k[0]:<8} {k[1]} {k[2]} x={k[5]:>3}: {a['peak']:10.3f} vs {t['peak']:10.3f} ({d:+.3f}, expected {e:+.3f}); "
                    f"step-0 loss {a['loss0']:.4f} vs {t['loss0']:.4f}")
            say(f"{pid} {'HELD' if worst <= TOL else 'MISS'}: largest twin difference from expected {worst:.3f} MiB")
    g = sorted((k, a) for k, a in done.items() if k[3])
    if g:
        say('--- native GQA (flash, k/v at kv width): offset from the full-width twin, per x')
        for k, a in g:
            t = twin(k, gqa=False)
            if t:
                d = a['peak'] - t['peak']
                say(f"    {k[0]:<8} {k[1]} x={k[5]:>3}: {d:+.3f} MiB = {d / k[5]:+.4f} MiB/x "
                    f"(prediction -2.0000 native; 0.0000 if the kernel expands internally)")

def _pairs(arms, xs, blk='T'):
    T = [a for a in arms if a['blk'] == blk and a['fit'] and a['x'] in xs]
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

def cmd_timecal(arms, pred, xs=(16, 32, 64), at=(100, 150)):
    rows = _pairs(arms, xs)
    T = [a for a in arms if a['blk'] == 'T' and a['fit']]
    s0 = T[0] if T else {'shape': '', 'L': 64, 'world': 4}
    label = (f"{s0['shape'] or 'tier-A'}, L={s0['L']}, world {s0['world']}") if s0['shape'] or s0['world'] != 4 \
        else 'L=64, world 4'
    say(f"=== TIME CALIBRATION (x = {', '.join(map(str, xs))}; static and offload, two repeats each) ===")
    for x, ts, to, reps in rows:
        say(f"  x={x:>3}  static {ts:8.3f} s  offload {to if to is None else round(to, 3)} s  "
            f"delta per repeat {[round(r, 3) for r in reps]}")
    if len({r[0] for r in rows}) < 3:
        say('timecal: fewer than three x values completed; no prediction written'); return
    a, b, c = fit_step_time([(x, ts) for x, ts, _, _ in rows])
    kx = sorted(xs)[1] if len(xs) > 1 else xs[0]
    pairs = [(x, statistics.mean(reps)) for x, _, _, reps in rows if reps and x >= kx]
    kappa = fit_kappa(pairs) if pairs else None
    if kappa is None:
        say('timecal: no paired differences at the calibration lengths; no prediction written'); return
    cal = TimeCalibration(a, b, c, kappa, f'{NAME} in-job: T on x=16/32/64 static, kappa on x=32/64 pairs')
    lines = [f"# {NAME} TIME PREDICTIONS, written {time.strftime('%Y-%m-%dT%H:%M:%S')} BEFORE the x={'/'.join(map(str, at))} arms ran",
             f"# T(x) = {a:.6f} x^2 + {b:.5f} x + {c:.4f} s   (static, {label}, fp16 mem-efficient)",
             f"# kappa = {kappa:.5f} s per x   (V100: 0.032)",
             "# T-P3 predicted at " + '; '.join(f"x={x}: static {cal.step_time(x):.2f} s, tax {cal.tax(x):.2%}" for x in at),
             f"# T-P4 crossings: x*(2%) = {cal.crossing(0.02) or float('nan'):.1f}, "
             f"x*(1%) = {cal.crossing(0.01) or float('nan'):.1f}   (V100: ~79 and ~150)",
             f"# Falsifier: a measured tax at x={' or '.join(map(str, at))} outside +-30% of these, or kappa at",
             f"# x={'/'.join(map(str, at))} differing from the calibrated kappa by more than 30% (the cost is not linear in x here)."]
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

def _context(arms, world):
    same = [v for a in arms if a['fit'] and not a['cap'] and a['world'] == world for v in a.get('outside', [])]
    anyw = [v for a in arms if a['fit'] and not a['cap'] for v in a.get('outside', [])]
    return max(same) if same else (max(anyw) if anyw else None)

def _native_fits(arm, total, ctx, off, step, alphas):
    ln = line_for(arm)
    run = Run(shape=SHAPES[arm['shape']].with_layers(arm['L']), world=arm['world'])
    rq = request_mib_per_x(run)
    out = []
    for al in alphas:
        xs = [x for x in range(step, 4096, step) if ln.at(x) + off <= total - ctx - AMPERE_DELTA - al * rq * x]
        out.append(max(xs) if xs else 0)
    return out

def cmd_natpred(arms, pred, node, blk, shape, L, world, mode, step):
    total = node.get('total_mib')
    ctx = _context(arms, world)
    if total is None or ctx is None:
        say(f'natpred {blk}: no card total or context reading; using the header rungs'); return
    b_offs = [a['peak'] - line_for(a).at(a['x']) for a in arms if a['blk'].startswith('B') and a['fit'] and not a['cap']]
    off = statistics.median(b_offs) if b_offs else 0.0
    arm = {'shape': '' if shape == 'tier_a' else shape, 'L': L, 'world': world, 'mode': mode,
           'gqa': False, 'attn': 'mem'}
    lo, mid, hi = sorted(_native_fits(arm, total, ctx, off, step,
                                      (AMPERE_ALPHA_RANGE[1], AMPERE_ALPHA, AMPERE_ALPHA_RANGE[0])))
    ln = line_for(arm)
    rungs = list(range(max(step, lo - step), hi + 4 * step, step))
    lines = [f"# {NAME} NATIVE {mode.upper()} BRACKET {blk} ({shape}, L={L}, world {world}), written "
             f"{time.strftime('%Y-%m-%dT%H:%M:%S')} BEFORE its ladder ran",
             f"# line {ln.c:.4f} + {ln.m:.6f}x {off:+.3f} MiB (this job's common offset); card total {total:.2f} MiB; "
             f"outside the allocator {ctx:.2f} MiB (max at world {world})",
             f"# Ampere reserve carried from the A30 (633912): delta {AMPERE_DELTA} MiB, alpha {AMPERE_ALPHA} "
             f"(feasible {AMPERE_ALPHA_RANGE[0]}-{AMPERE_ALPHA_RANGE[1]})",
             f"# NP-{blk} largest fit x = {mid} at the centre; {lo}-{hi} over the alpha range; registered "
             f"tolerance +-2 rungs (step {step}) across shapes. FALSIFIER: the measured largest fit outside "
             f"[{lo - 2 * step}, {hi + 2 * step}]",
             f"# the failure is a backward OOM on the replayed MLP activation, {request_mib_per_x(Run(shape=SHAPES[arm['shape']].with_layers(L), world=world)):g}x MiB",
             f"CENTRE: {mid} {lo} {hi} {step}",
             f"RUNGS: {' '.join(map(str, rungs))}"]
    open(pred, 'w').write('\n'.join(lines) + '\n')
    say('\n'.join(lines))

def _native_ladders(arms, out):
    res = {}
    for f in sorted(glob.glob(os.path.join(out, f'{PREFIX}_pred_native_*.txt'))):
        blk = os.path.basename(f)[len(f'{PREFIX}_pred_native_'):-4]
        rr = sorted((a for a in arms if a['blk'] == blk), key=lambda a: a['x'])
        cen = [l for l in open(f) if l.startswith('CENTRE:')]
        if not rr or not cen:
            continue
        mid, lo, hi, step = map(int, cen[0].split()[1:])
        fits = [a['x'] for a in rr if a['fit']]; fails = [a for a in rr if not a['fit']]
        fit = max(fits) if fits else None
        held = fit is not None and lo - 2 * step <= fit <= hi + 2 * step and bool(fails)
        res[blk] = (rr[0], fit, mid)
        say(f"  {blk:<4} {rr[0]['mode']:<8} {rr[0]['shape'] or 'tier-A'} w{rr[0]['world']}: measured "
            f"[{fit if fit is not None else '<' + str(rr[0]['x'])}, {fails[0]['x'] if fails else 'open'})  predicted "
            f"{mid} ({lo}-{hi}, +-2 rungs)  failing request {fails[0]['request'] if fails else None} MiB "
            f"(backward: {fails[0]['backward'] if fails else '-'})  -> {'HELD' if held else 'MISS or OPEN'}")
    for b1, (a1, f1, m1) in res.items():
        for b2, (a2, f2, m2) in res.items():
            if a1['mode'] == 'static' and a2['mode'] == 'offload' and a1['shape'] == a2['shape'] \
                    and a1['world'] == a2['world'] and f1 and f2:
                say(f"  span {b2}/{b1} ({a1['shape'] or 'tier-A'}, world {a1['world']}): measured {f2 / f1:.3f}, "
                    f"predicted {m2 / m1:.3f}")

def cmd_read(arms, out, node):
    say(f"=== {NAME} JOB REPORT: {out} ({sum(a['fit'] for a in arms)} of {len(arms)} arms completed) ===")
    for a in arms:
        if not a['fit'] and not a['cap'] and a['blk'] not in ('N',) + NATIVE_3B + _ladder_blocks(out):
            say(f"  FAILED (not a ladder rung): {os.path.basename(a['dir'])}, request {a.get('request')} -- read its stderr.txt")
    cmd_bytes(arms)
    rows = _pairs(arms, sorted({a['x'] for a in arms if a['blk'] == 'T'}))
    if rows:
        say('=== TIME (R-A5): pooled static, offload, per-repeat delta, tax ===')
        for x, ts, to, reps in rows:
            d = statistics.mean(reps) if reps else float('nan')
            say(f"  x={x:>3}  static {ts:8.3f} s  delta {d:+.3f} s  per x {d / x:+.4f}  tax {d / ts:6.2%}  "
                f"repeats {[round(r, 3) for r in reps]}")
        pf = pfile(out, 'pred_time.txt')
        if os.path.exists(pf):
            say('  pre-registered in-job:'); say(open(pf).read().rstrip())
    _host_path(arms, rows)
    _ballast(arms)
    _nccl_and_clocks(arms, out)
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
    if _ladder_blocks(out):
        say('=== NATIVE LADDERS (in-job predictions, Ampere reserve carried) ===')
        _native_ladders(arms, out)

def _ladder_blocks(out):
    return tuple(os.path.basename(f)[len(f'{PREFIX}_pred_native_'):-4]
                 for f in glob.glob(os.path.join(out, f'{PREFIX}_pred_native_*.txt')))

def _host_path(arms, trows):
    hrows = _pairs(arms, sorted({a['x'] for a in arms if a['blk'] == 'H'}), 'H')
    if not hrows:
        return
    tby = {x: (ts, reps) for x, ts, _, reps in trows}
    say('=== HOST-PATH COLLECTIVES (H: NCCL_P2P_DISABLE=1) vs NVLink (T), same node ===')
    for x, hs, _, hreps in hrows:
        hd = statistics.mean(hreps) if hreps else float('nan')
        ts, treps = tby.get(x, (None, []))
        td = statistics.mean(treps) if treps else float('nan')
        say(f"  x={x:>3}  static {hs:8.3f} s (NVLink {ts if ts is None else round(ts, 3)} s, "
            f"{'' if ts is None else f'{(hs - ts) / ts:+.2%}'})  delta host {hd:+.3f} s  delta NVLink {td:+.3f} s  "
            f"tax host {hd / hs:6.2%}  tax NVLink {td / ts if ts else float('nan'):6.2%}  repeats {[round(r, 3) for r in hreps]}")

def _ballast(arms):
    g = sorted((a for a in arms if a['blk'] == 'G' and a['fit']), key=lambda a: a['x'])
    if not g:
        return
    say('=== MATCHED FOOTPRINT (G: offload + ballast = static\'s extra bytes) ===')
    for a in g:
        tg = statistics.mean(a['times'])
        s = [t for b in arms if b['blk'] == 'T' and b['fit'] and b['x'] == a['x'] and b['mode'] == 'static' for t in b['times']]
        o = [t for b in arms if b['blk'] == 'T' and b['fit'] and b['x'] == a['x'] and b['mode'] == 'offload' for t in b['times']]
        ts = statistics.mean(s) if s else float('nan'); to = statistics.mean(o) if o else float('nan')
        say(f"  x={a['x']:>3}  ballast {a['ballast']:.1f} MiB  offload+ballast {tg:8.3f} s  "
            f"vs static {tg - ts:+.3f} s ({(tg - ts) / ts:+.2%})  vs plain offload {tg - to:+.3f} s ({(tg - to) / to:+.2%})")

def _nccl_and_clocks(arms, out):
    say('=== NCCL TRANSPORTS (channel lines, all ranks) and GPU clocks (1 Hz, util >= 90%) ===')
    for blk in sorted({a['blk'] for a in arms}):
        v = {}
        for a in arms:
            if a['blk'] == blk:
                for k, n in a.get('nccl_via', {}).items():
                    v[k] = v.get(k, 0) + n
        if v:
            say(f"  block {blk}: " + ', '.join(f'{k} x{n}' for k, n in sorted(v.items())))
    for a in arms:
        if a['blk'] in ('T', 'H', 'G') and a['fit'] and a.get('gpu'):
            say(f"  {os.path.basename(a['dir']):<44} {fmt_gpu(a.get('gpu'))}")
    bw = os.path.join(out, f'{PREFIX}_bw.json')
    if os.path.exists(bw):
        b = json.load(open(bw))
        say(f"  host copy bandwidth, all GPUs concurrently, per GPU: D2H {b['concurrent']['d2h']:.2f} GB/s, "
            f"H2D {b['concurrent']['h2d']:.2f} GB/s; solo " +
            '; '.join(f"GPU {d} {v['d2h']:.1f}/{v['h2d']:.1f}" for d, v in b['solo'].items()))

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
    ap.add_argument('cmd', choices=['bytes', 'timecal', 'capcal', 'natpred', 'read'])
    ap.add_argument('out')
    ap.add_argument('--pred')
    ap.add_argument('--xs', default='16,32,64', help='timecal: calibration lengths')
    ap.add_argument('--at', default='100,150', help='timecal: lengths predicted')
    ap.add_argument('--ladder', help='natpred: block name')
    ap.add_argument('--shape', default='tier_a', choices=['tier_a', 'llama3b', 'mistral7b'])
    ap.add_argument('--L', type=int, default=64)
    ap.add_argument('--world', type=int, default=4)
    ap.add_argument('--mode', default='static', choices=['static', 'offload'])
    ap.add_argument('--step', type=int, default=1)
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
        cmd_timecal(arms, a.pred, tuple(map(int, a.xs.split(','))), tuple(map(int, a.at.split(','))))
    elif a.cmd == 'natpred':
        cmd_natpred(arms, a.pred, node, a.ladder, a.shape, a.L, a.world, a.mode, a.step)
    elif a.cmd == 'capcal':
        cmd_capcal(arms, a.pred, node)
    else:
        cmd_read(arms, a.out, node)

if __name__ == '__main__':
    main()
