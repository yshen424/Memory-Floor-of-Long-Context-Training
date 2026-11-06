import json

from .capacity import bracket, request_mib_per_x
from .mechanisms import floor_line, static_line
from .terms import MiB
from .timemodel import host_limit_x, offload_bytes_per_step

MECHS = {
    'static': lambda run, plat: static_line(run, plat),
    'stock_offload': lambda run, plat: floor_line(run, plat, 0),
    'force_evict': lambda run, plat: floor_line(run, plat, 0),
    'engine_cap': lambda run, plat: floor_line(run, plat, 0),
}

def predict(run, platform, mechanisms, ladder, timecal=None):
    out = {'config': {'layers': run.shape.layers, 'hidden': run.shape.hidden,
                      'intermediate': run.shape.intermediate, 'n_q': run.shape.n_q,
                      'n_kv': run.shape.n_kv, 'vocab': run.shape.vocab, 'world': run.world,
                      'chunk': run.chunk, 'act_bytes': run.act_bytes, 'repeat_kv': run.repeat_kv, 'attn': run.attn,
                      'cpu_offload': run.cpu_offload, 'platform': platform.name},
           'calibrated': {'workspace_mib': platform.ws_mib,
                          'capacity': None if platform.capacity is None else
                          {'c0_mib': platform.capacity.c0_mib, 'alpha': platform.capacity.alpha},
                          'provenance': platform.provenance},
           'lines': {}, 'brackets': {}, 'rows': []}
    for m in mechanisms:
        ln = MECHS[m](run, platform)
        out['lines'][m] = {'c_mib': ln.c, 'm_mib_per_x': ln.m, 'kind': 'derived',
                           'other_instants': {n: {'c_mib': o.c, 'm_mib_per_x': o.m,
                                                  'binds_below_x': (o.c - ln.c) / (ln.m - o.m)
                                                  if ln.m > o.m else None}
                                              for n, o in ln.alts}}
        if platform.capacity is not None and ladder:
            rows, fit, fail = bracket(ln, run, platform.capacity, ladder)
            out['brackets'][m] = {'largest_fit': fit, 'smallest_fail': fail, 'kind': 'capacity, ±1 rung',
                                  'failing_request_mib': None if fail is None else request_mib_per_x(run) * fail}
            out['rows'] += [{'mechanism': m, 'x': x, 'peak_mib': pk, 'request_mib': rq,
                             'capacity_mib': cap, 'fits': ok} for x, pk, rq, cap, _, ok in rows]
    st, fl = static_line(run, platform), floor_line(run, platform, 0)
    out['spans'] = {'perfect_relief': st.m / (st.m - run.shape.layers * run.act_bytes * run.shape.hidden / 1024.0),
                    'structural_cap': st.m / fl.m, 'kind': 'derived (ratio of slopes)'}
    if 'static' in out['brackets'] and 'stock_offload' in out['brackets']:
        a, b = out['brackets']['static']['largest_fit'], out['brackets']['stock_offload']['largest_fit']
        if a and b:
            out['spans']['ladder'] = {'value': b / a, 'kind': 'capacity (largest fit over largest fit)'}
    out['host'] = {'offload_bytes_per_x_each_way_mib': offload_bytes_per_step(run, 1) / MiB,
                   'kind': 'derived'}
    if platform.host_ram_gib:
        per_rank = platform.host_ram_gib * 2 ** 30 / max(platform.ranks_per_node, 1)
        out['host']['host_limit_x'] = host_limit_x(run, per_rank)
    if timecal is not None:
        out['time'] = {'kind': 'time statement (platform-measured constants)',
                       'kappa_s_per_x': timecal.kappa, 'step_time_abc': [timecal.a, timecal.b, timecal.c],
                       'tax_peak_x': timecal.tax_peak_x(),
                       'x_star': {f'{e:.0%}': timecal.crossing(e) for e in (0.05, 0.02, 0.01)},
                       'provenance': timecal.provenance}
    return out

def header_block(p):
    c = p['config']
    L = [f"# PREDICTIONS FROM harness/ (H6), platform {c['platform']}, world {c['world']}, "
         f"L={c['layers']} h={c['hidden']} I={c['intermediate']} heads {c['n_q']}/{c['n_kv']} "
         f"v={c['vocab']} chunk={c['chunk']}{' +cpu_offload' if c['cpu_offload'] else ''}",
         f"# Calibrated platform constants: workspace {p['calibrated']['workspace_mib']:.4f} MiB"
         + ('' if not p['calibrated']['capacity'] else
            f"; capacity C0 {p['calibrated']['capacity']['c0_mib']:.2f} MiB, alpha {p['calibrated']['capacity']['alpha']:.3f}"),
         "# Lines (derived; x = tokens/1024, MiB):"]
    for m, ln in p['lines'].items():
        L.append(f"#   {m:<14} {ln['c_mib']:10.4f} + {ln['m_mib_per_x']:.6f}x")
        for n, o in ln.get('other_instants', {}).items():
            if o['binds_below_x'] and o['binds_below_x'] > 0:
                L.append(f"#     {'peak is ' + n:<26} {o['c_mib']:10.4f} + {o['m_mib_per_x']:.6f}x below x = {o['binds_below_x']:.1f}")
    for m, br in p['brackets'].items():
        L.append(f"# Bracket {m:<14} largest fit x={br['largest_fit']}, smallest fail x={br['smallest_fail']}"
                 + (f", failing request {br['failing_request_mib']:.2f} MiB" if br['failing_request_mib'] else '')
                 + "  (capacity statement, ±1 rung)")
    sp = p['spans']
    L.append(f"# Spans: perfect relief {sp['perfect_relief']:.4f} (allowance), structural cap "
             f"{sp['structural_cap']:.4f}" + (f", ladder {sp['ladder']['value']:.3f}" if 'ladder' in sp else ''))
    h = p['host']
    L.append(f"# Host: stock offload pins {h['offload_bytes_per_x_each_way_mib']:.2f} MiB per x per rank"
             + (f"; host memory binds at x ~ {h['host_limit_x']:.0f}" if 'host_limit_x' in h else ''))
    if 'time' in p:
        t = p['time']
        L.append(f"# Time (platform-measured): kappa {t['kappa_s_per_x']:.4f} s/x; tax peaks at x~{t['tax_peak_x']:.1f}; "
                 + ', '.join(f"x*({k}) ~ {v:.0f}" for k, v in t['x_star'].items() if v))
    return '\n'.join(L)

def write_json(p, path):
    with open(path, 'w') as f:
        json.dump(p, f, indent=2, default=str)
