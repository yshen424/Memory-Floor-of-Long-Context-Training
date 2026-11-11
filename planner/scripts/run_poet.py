"""Run the STOCK POETSolver (PuLP) on the toy graph, with constraints added
after construction (no POET source edits):

  mode static  : R fixed to whole-layer, POET's own paging=False
  mode inclass : R fixed, only boundaries b_0..b_{L-1} may page; once layer l's
                 replay starts, b_l stays resident through gnorm1_l
  mode wide    : R fixed, every tensor may page (replay-internal tensors too)
  mode free    : POET's native problem (R free, all paging) -- small L only

  --method minpeak : add Peak >= U[t][k], minimize Peak (min feasible budget)
  --method check   : POET's own energy objective at --budget; report feasibility
"""
import argparse, json, sys, time

import numpy as np
import pulp as pl

from poet.poet_solver import POETSolver
import toygraph as tg


def make_solver(t, R, mode, budget, solver, threads, tlimit):
    g = tg.to_dfgraph(t)
    T = t.T
    cpu = np.asarray(t.cpu, dtype=float).reshape(-1, 1)
    pg = np.asarray(t.ram, dtype=float).reshape(-1, 1)  # paging cost ~ bytes moved
    s = POETSolver(g, cpu, pg, pg, ram_budget_bytes=budget, runtime_budget_ms=None,
                   paging=(mode != "static"), remat=True, solver="pulp-cbc",
                   time_limit_s=tlimit, solve_threads=threads)
    if solver == "highs":
        s.solver = pl.HiGHS(msg=False, timeLimit=tlimit, threads=threads)
    m = s.m
    if mode != "free":
        for st in range(T):
            for i in range(st + 1):
                m += s.R[st][i] == R[st][i]
    if mode == "inclass":
        bset = set(t.boundary_ids())
        need = tg.need_stages(t, R)
        nfwd = sum(1 for x in t.bwd if not x)
        for i in range(T):
            if i in bset:
                bw = [st for st in need[i] if st >= nfwd]
                for st in range(min(bw), max(bw) + 1):
                    m += s.SRam[st][i] == 1
            else:
                for st in range(T):
                    m += s.MIn[st][i] == 0
                    m += s.MOut[st][i] == 0
                    m += s.SSd[st][i] == 0
    if mode == "wide_replay":  # non-boundaries may page out only after their last (re)computation
        bset = set(t.boundary_ids())
        for i in range(T):
            if i not in bset:
                last_c = max(st for st in range(T) if R[st][i])
                for st in range(last_c + 1):
                    m += s.MOut[st][i] == 0
    return s


def run(L, gran, spread, mode, method, budget, solver, threads, tlimit):
    t = tg.build(L, gran, spread)
    R = tg.whole_layer_R(t)
    t0 = time.time()
    BIG = 1e5
    s = make_solver(t, R, mode, BIG if method == "minpeak" else budget, solver, threads, tlimit)
    if method == "minpeak":
        peak = pl.LpVariable("Peak", 0, BIG)
        for st in range(t.T):
            for k in range(t.T):
                s.m += s.U[st][k] <= peak
        s.m.setObjective(peak)
    build_s = time.time() - t0
    nv, nc = s.m.numVariables(), s.m.numConstraints()
    sol = s.solve()
    out = dict(L=L, gran=gran, spread=spread, mode=mode, method=method, solver=solver, T=t.T,
               n_vars=nv, n_cons=nc, build_s=round(build_s, 2), solve_s=round(sol.solve_time_s, 2),
               status=pl.LpStatus[s.m.status], feasible=sol.feasible, finished=sol.finished)
    if method == "minpeak" and sol.feasible:
        out["min_budget"] = round(pl.value(peak), 6)
    if method == "check":
        out["budget"] = budget
    if sol.feasible:
        out["pagein_bytes"] = round(float(np.sum(np.asarray(sol.Min) @ np.asarray(t.ram))), 4)
        out["pageout_bytes"] = round(float(np.sum(np.asarray(sol.Mout) @ np.asarray(t.ram))), 4)
        out["n_pageins"] = int(np.sum(sol.Min)); out["n_pageouts"] = int(np.sum(sol.Mout))
        # independent re-evaluation of the returned schedule
        Rs = [[int(round(v)) for v in row] for row in sol.R]
        Ss = [[int(round(v)) for v in row] for row in sol.SRam]
        try:
            out["sim_peak_of_solution"] = round(max(tg.simulate(t, Rs, Ss)), 6)
        except AssertionError as e:
            out["sim_error"] = str(e)
    return out, t, sol


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--gran", default="fine", choices=["fine", "coarse", "census"])
    ap.add_argument("--spread", type=int, default=0)
    ap.add_argument("--mode", default="inclass", choices=["static", "inclass", "wide", "wide_replay", "free"])
    ap.add_argument("--method", default="minpeak", choices=["minpeak", "check"])
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--solver", default="cbc", choices=["cbc", "highs"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tlimit", type=float, default=1200)
    ap.add_argument("--dump", default=None)
    a = ap.parse_args()
    out, t, sol = run(a.L, a.gran, bool(a.spread), a.mode, a.method, a.budget, a.solver, a.threads, a.tlimit)
    print(json.dumps(out))
    if a.dump and sol.feasible:
        with open(a.dump, "w") as f:
            json.dump(dict(names=t.names, Min=sol.Min, Mout=sol.Mout, SRam=sol.SRam, R=sol.R), f)
