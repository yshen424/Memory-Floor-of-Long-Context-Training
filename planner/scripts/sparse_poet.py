"""Sparse POET: POET's own constraint generators (_create_objective,
_create_correctness_constraints, _create_ram_memory_constraints, inherited
UNMODIFIED from poet.poet_solver.POETSolver), run over decision matrices in
which every entry that the fixed schedule determines is a Python constant
instead of a binary variable.

Substitutions (each preserves the optimum of the in-class / static / wide problem):
  R        = the fixed whole-layer schedule (constants)
  SRam     = 0 before a tensor exists or after its last need (holding a tensor
             longer only adds to U: U is monotone in SRam), and for tensors the
             mode forbids paging, the minimal contiguous retention (canonical)
  SSd/MIn/MOut = 0 for tensors the mode forbids paging, and outside the window
             where a page could matter
  Free_E   = constant wherever POET's hazard count is constant
A FilteringProblem drops constraints that are redundant given variable bounds
and raises on any that is violated by constants (an infeasible fixed schedule).
"""
import math
import pulp as pl

from poet.poet_solver import POETSolver
import toygraph as tg

LE, EQ, GE = pl.LpConstraintLE, pl.LpConstraintEQ, pl.LpConstraintGE


class ConstInfeasible(Exception):
    pass


class FilteringProblem(pl.LpProblem):
    n_dropped = 0

    def __iadd__(self, other):
        o = other[0] if isinstance(other, tuple) else other
        if o is True:
            self.n_dropped += 1
            return self
        if o is False:
            raise ConstInfeasible("constant constraint is False")
        if isinstance(o, pl.LpConstraint):
            lo = hi = o.constant
            for v, c in o.expr.items():
                lb = -math.inf if v.lowBound is None else v.lowBound
                ub = math.inf if v.upBound is None else v.upBound
                if c >= 0:
                    lo += c * lb; hi += c * ub
                else:
                    lo += c * ub; hi += c * lb
            tol = 1e-9
            if o.sense == LE:
                if hi <= tol: self.n_dropped += 1; return self
                if lo > tol: raise ConstInfeasible(str(o))
            elif o.sense == GE:
                if lo >= -tol: self.n_dropped += 1; return self
                if hi < -tol: raise ConstInfeasible(str(o))
            else:
                if abs(lo) <= tol and abs(hi) <= tol: self.n_dropped += 1; return self
                if lo > tol or hi < -tol: raise ConstInfeasible(str(o))
        return super().__iadd__(other)


class SparsePOET(POETSolver):
    def __init__(self, t, R, mode, ram_budget, solver="cbc", time_limit_s=None, solve_threads=None, ucompress=False):
        g = tg.to_dfgraph(t)
        self.g, self.T = g, g.size
        T = self.T
        self.m = FilteringProblem("POETsparse", pl.LpMinimize)
        if solver == "highs":
            self.solver = pl.HiGHS(msg=False, timeLimit=time_limit_s, threads=solve_threads)
        else:
            self.solver = pl.PULP_CBC_CMD(msg=False, timeLimit=time_limit_s, threads=solve_threads)
        self.runtime_budget_ms, self.ram_budget_bytes = None, ram_budget
        self.remat, self.paging = True, mode != "static"
        ub = ram_budget if ram_budget is not None else float("inf")

        need = tg.need_stages(t, R)
        comp = {i: [s for s in range(T) if R[s][i]] for i in range(T)}
        bset = set(t.boundary_ids())
        nfwd = sum(1 for x in t.bwd if not x)
        canon = tg.canonical_SRam(t, R, page_boundaries=False)
        pageable = (lambda i: i in bset) if mode == "inclass" else (lambda i: mode in ("wide", "wide_replay"))

        self.R = R
        z = lambda: [[0] * T for _ in range(T)]
        self.SRam, self.SSd, self.MIn, self.MOut = z(), z(), z(), z()
        nb = 0
        for i in range(T):
            if not need[i]:
                continue
            last = max(need[i])
            if not pageable(i):
                for s in range(T):
                    self.SRam[s][i] = canon[s][i]
                continue
            first_c = min(comp[i])
            # wide_replay: a non-boundary may page out only after its last (re)computation
            p0 = first_c + 1 if (mode != "wide_replay" or i in bset) else max(comp[i]) + 1
            forced = set()
            if mode == "inclass":  # b_l resident from its first replay through its last use
                bw = [s for s in need[i] if s >= nfwd]
                forced = set(range(min(bw), max(bw) + 1))
            for s in range(first_c + 1, last + 1):
                if s in forced:
                    self.SRam[s][i] = 1
                else:
                    self.SRam[s][i] = pl.LpVariable(f"SRam_{s},{i}", cat=pl.LpBinary); nb += 1
            for s in range(p0 + 1, last + 1):
                self.SSd[s][i] = pl.LpVariable(f"SSd_{s},{i}", cat=pl.LpBinary); nb += 1
            for s in range(p0 + 1, last):
                self.MIn[s][i] = pl.LpVariable(f"MIn_{s},{i}", cat=pl.LpBinary); nb += 1
            for s in range(p0, last):
                self.MOut[s][i] = pl.LpVariable(f"MOut_{s},{i}", cat=pl.LpBinary); nb += 1
        if ucompress:
            # POET's recurrence gives U[t][k] == U[t][k-1] when neither k nor k-1 runs in stage t
            # (nothing added, nothing freed): alias them to one variable.
            self.U = []
            for s in range(T):
                row = []
                for k in range(T):
                    if k == 0 or R[s][k] or R[s][k - 1]:
                        row.append(pl.LpVariable(f"U_{s},{k}", 0, ub))
                    else:
                        row.append(row[-1])
                self.U.append(row)
        else:
            self.U = [[pl.LpVariable(f"U_{s},{k}", 0, ub) for k in range(T)] for s in range(T)]
        E = g.edge_list
        self.Free_E = [[0] * len(E) for _ in range(T)]
        for s in range(T):
            for e, (i, k) in enumerate(E):
                if not R[s][k]:
                    continue
                later = sum(R[s][j] for j in g.successors(i) if j > k)
                nxt = self.SRam[s + 1][i] if s + 1 < T else 0
                if later >= 1:
                    self.Free_E[s][e] = 0
                elif isinstance(nxt, int):
                    self.Free_E[s][e] = 1 - nxt
                else:
                    self.Free_E[s][e] = pl.LpVariable(f"Free_E_{s},{e}", cat=pl.LpBinary); nb += 1
        self.n_binaries = nb

        import numpy as np
        cpu = np.asarray(t.cpu, dtype=float).reshape(-1, 1)
        pg = np.asarray(t.ram, dtype=float).reshape(-1, 1)
        self._create_objective(cpu, pg, pg)
        # _initialize_variables: every entry it constrains is a constant 0/1 here; checked instead
        assert all(R[s][s] == 1 for s in range(T))
        assert all(R[s][i] == 0 for s in range(T) for i in range(s + 1, T))
        self._create_correctness_constraints()
        if ram_budget is not None:
            self._create_ram_memory_constraints([g.cost_ram[i] for i in range(T)])

    def get_result(self, var_matrix, dtype=int):
        if not self.is_feasible():
            return None
        val = lambda x: x if isinstance(x, (int, float)) else pl.value(x)
        return [[dtype(round(val(x)) if dtype is int else val(x)) for x in row] for row in var_matrix]


def run_sparse(L, gran, spread, mode, method, budget, solver="cbc", threads=8, tlimit=1200, bw=None, ucompress=False):
    import time, numpy as np
    t = tg.build(L, gran, spread)
    R = tg.whole_layer_R(t)
    t0 = time.time()
    BIG = 1e5
    s = SparsePOET(t, R, mode, BIG if method == "minpeak" else budget, solver, tlimit, threads, ucompress)
    if bw is not None:  # overlap model: bytes paged during stage t must fit in beta * compute time of stage t
        for st in range(t.T):
            moved = pl.lpSum(t.ram[i] * (s.MIn[st][i] + s.MOut[st][i]) for i in range(t.T)
                             if not (isinstance(s.MIn[st][i], int) and isinstance(s.MOut[st][i], int)))
            s.m += moved <= bw * sum(R[st][k] * t.cpu[k] for k in range(t.T))
    if method == "minpeak":
        peak = pl.LpVariable("Peak", 0, BIG)
        for st in range(t.T):
            for k in range(st + 1):  # U[t][k>t] only decreases
                if R[st][k]:
                    s.m += s.U[st][k] <= peak
        s.m.setObjective(peak)
    build_s = time.time() - t0
    nv, nc = s.m.numVariables(), s.m.numConstraints()
    sol = s.solve()
    out = dict(impl="sparse_u" if ucompress else "sparse", L=L, gran=gran, spread=spread, mode=mode, method=method, solver=solver, T=t.T,
               n_vars=nv, n_bin=s.n_binaries, n_cons=nc, n_dropped=s.m.n_dropped, build_s=round(build_s, 2),
               solve_s=round(sol.solve_time_s, 2), status=pl.LpStatus[s.m.status], feasible=sol.feasible)
    if method == "minpeak" and sol.feasible:
        out["min_budget"] = round(pl.value(peak), 6)
    if method == "check":
        out["budget"] = budget
    if sol.feasible:
        out["pagein_bytes"] = round(float(np.sum(np.asarray(sol.Min) @ np.asarray(t.ram))), 4)
        out["pageout_bytes"] = round(float(np.sum(np.asarray(sol.Mout) @ np.asarray(t.ram))), 4)
        try:
            out["sim_peak_of_solution"] = round(max(tg.simulate(t, sol.R, sol.SRam)), 6)
        except AssertionError as e:
            out["sim_error"] = str(e)
    return out


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--gran", default="fine")
    ap.add_argument("--spread", type=int, default=0)
    ap.add_argument("--mode", default="inclass")
    ap.add_argument("--method", default="minpeak")
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--solver", default="cbc")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tlimit", type=float, default=1200)
    ap.add_argument("--bw", type=float, default=None)
    ap.add_argument("--ucompress", type=int, default=0)
    a = ap.parse_args()
    o = run_sparse(a.L, a.gran, bool(a.spread), a.mode, a.method, a.budget, a.solver, a.threads, a.tlimit, a.bw, bool(a.ucompress))
    o["bw"] = a.bw
    print(json.dumps(o), flush=True)
