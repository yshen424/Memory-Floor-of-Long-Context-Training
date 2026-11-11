"""For each config: minpeak -> P*; then POET's own objective at P* (expect
feasible; objective = min paging volume) and at P* - 1e-3 (expect infeasible).
One JSON line per solve to stdout."""
import json, sys, time
import toygraph as tg
from run_poet import run

TOL = 1e-3


def pred(L, gran, spread, mode):
    t = tg.build(L, gran, spread); R = tg.whole_layer_R(t)
    if mode == "static":
        return max(tg.simulate(t, R, tg.canonical_SRam(t, R, False)))
    if mode == "inclass":
        return max(tg.simulate(t, R, tg.canonical_SRam(t, R, True)))
    return None


def one(L, gran, spread, mode, solver="cbc", threads=8, tlimit=1200, checks=True):
    p = pred(L, gran, spread, mode)
    o, _, _ = run(L, gran, spread, mode, "minpeak", None, solver, threads, tlimit)
    o["predicted"] = None if p is None else round(p, 6)
    print(json.dumps(o), flush=True)
    if not checks or "min_budget" not in o:
        return
    P = o["min_budget"]
    for b in (P + 1e-6, P - TOL):
        c, _, _ = run(L, gran, spread, mode, "check", b, solver, threads, tlimit)
        print(json.dumps(c), flush=True)


if __name__ == "__main__":
    # usage: sweep.py gran spread mode solver checks L1 L2 ...
    gran, spread, mode, solver, checks = sys.argv[1], bool(int(sys.argv[2])), sys.argv[3], sys.argv[4], bool(int(sys.argv[5]))
    for L in map(int, sys.argv[6:]):
        one(L, gran, spread, mode, solver=solver, checks=checks)
