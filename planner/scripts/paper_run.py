"""POET's paper run on the base configuration (tier-A, chunked loss c=256, fp16, repeat_kv;
the census graph's tensor sizes are harness/terms.py's slope terms; MiB per x, x = tokens/1024).

    cd planner/poet_check/scripts && ../.venv/bin/python paper_run.py            # all runs (~1 h on 4 threads)
    ../.venv/bin/python paper_run.py --summary                                    # table from the log only

Every number is a slope. POET's graph holds only the activation tensors, whose sizes all
scale with x, so a schedule's peak is P* x and its optimum does not depend on x. The bytes no
boundary placement can change (parameters, optimizer state, gathered weights and gradients,
the library workspace: F in Proposition 3.1, and the optimizer-step and end-of-backward
instants) are not in the graph and add to every mode alike.

Runs (each in its own process, appended to ../logs/paper_run.jsonl):
  static   minimum peak with no paging: 36.3203 + 1.5 L (Table 3's static slope at L=64: 132.3203)
  inclass  boundary placement only (the paper's class): minimum peak at L = 8, 16, 32, 64; at L=64
           also feasibility at the minimum and at the minimum - 0.001 MiB, and the least-traffic
           schedule at the minimum (POET's own objective)
  wide_replay  also the tensors the replay produces, paged between whole replays (one-stage) and
           between single ops (spread); minimum peak and least-traffic schedule at L=64
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, '..', 'logs', 'paper_run.jsonl')
PY = sys.executable
M_STAR = 37.8203125          # Eq. 4's slope at the base shape: m_nonmig 36.3203125 + m_b 1.5


def solve(tag, L, mode, method, spread=0, budget=None, tlimit=3000, timeout=5400):
    args = [PY, os.path.join(HERE, 'sparse_poet.py'), '--L', str(L), '--gran', 'census', '--spread', str(spread),
            '--mode', mode, '--method', method, '--ucompress', '1', '--threads', '4', '--tlimit', str(tlimit)]
    if budget is not None:
        args += ['--budget', repr(budget)]
    env = dict(os.environ, TMPDIR=os.path.join(HERE, '..', 'tmp'))
    os.makedirs(env['TMPDIR'], exist_ok=True)
    t0 = time.time()
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout, cwd=HERE, env=env)
        line = [l for l in p.stdout.splitlines() if l.startswith('{')]
        rec = json.loads(line[-1]) if line else {'error': (p.stderr or '')[-400:]}
    except subprocess.TimeoutExpired:
        rec = {'error': f'timeout after {timeout} s'}
    rec.update(tag=tag, wall_s=round(time.time() - t0, 1), date=time.strftime('%Y-%m-%dT%H:%M:%S'))
    with open(LOG, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print(f"{tag:<34} {rec.get('status', '?'):<12} min {rec.get('min_budget', '-')!s:<10} "
          f"in {rec.get('pagein_bytes', '-')!s:<10} out {rec.get('pageout_bytes', '-')!s:<10} "
          f"{rec['wall_s']:>7} s {rec.get('error', '')}", flush=True)
    return rec


def summary():
    recs = {}
    for l in open(LOG):
        r = json.loads(l)
        recs[r['tag']] = r                      # last run of each tag wins
    print(f"{'run':<34} {'status':<12} {'min peak':>10} {'page in':>9} {'page out':>9} {'T':>5} {'wall s':>7}")
    for tag, r in recs.items():
        print(f"{tag:<34} {r.get('status', r.get('error', '?'))[:12]:<12} {r.get('min_budget', ''):>10} "
              f"{r.get('pagein_bytes', ''):>9} {r.get('pageout_bytes', ''):>9} {r.get('T', ''):>5} {r.get('wall_s', ''):>7}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', action='store_true')
    a = ap.parse_args()
    if a.summary:
        return summary()
    solve('static L=64', 64, 'static', 'minpeak')
    for L in (8, 16, 32, 64):
        solve(f'inclass L={L}', L, 'inclass', 'minpeak')
    solve('inclass L=64 check at M*', 64, 'inclass', 'check', budget=M_STAR)
    solve('inclass L=64 check at M*-0.001', 64, 'inclass', 'check', budget=M_STAR - 0.001)
    w1 = solve('wide_replay one-stage L=64', 64, 'wide_replay', 'minpeak')
    w2 = solve('wide_replay op-level L=64', 64, 'wide_replay', 'minpeak', spread=1)
    if 'min_budget' in w1:
        solve('wide_replay one-stage L=64 least traffic', 64, 'wide_replay', 'check', budget=w1['min_budget'] + 1e-6)
    if 'min_budget' in w2:
        solve('wide_replay op-level L=64 least traffic', 64, 'wide_replay', 'check', spread=1, budget=w2['min_budget'] + 1e-6)
    summary()


if __name__ == '__main__':
    main()
