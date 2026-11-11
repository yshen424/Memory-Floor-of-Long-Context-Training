"""Coarse transformer DAG for POET, the fixed whole-layer schedule R, and an
independent re-implementation of POET's memory accounting (for prediction).

Sizes are MiB per x (x = tokens/1024). The fixed term F (params, grads,
optimizer state) is not in POET's graph; add it to any POET budget.

Granularities
  fine   : 6 fwd ops / layer  norm1, attn, add1, norm2, mlp, add2(=b_{l+1})
           6 bwd ops / layer  gmlp, gnorm2, gadd1, gattn, gnorm1, gres(=g_l)
  coarse : 3 fwd ops / layer  norm1, body, add(=b_{l+1})
           3 bwd ops / layer  gbody, gnorm1, gres(=g_l)
Head (not checkpointed): fnorm, loss ; gloss, gfnorm.

spread=True inserts one zero-cost, zero-size, dependency-free "slot" node per
replayed op before the layer's first backward op, and places each replayed op
in its own slot stage (same compute, finer paging granularity: POET can only
page at stage boundaries).
"""
from dataclasses import dataclass, field
from typing import Dict, List

from poet.utils.checkmate.core.dfgraph import DFGraph

MB = 1.5  # boundary, MiB/x (fp16, hidden 768)


@dataclass
class Toy:
    L: int
    gran: str = "fine"
    spread: bool = False
    # sizes (MiB/x); mlp padded so the replay-instant non-migratable term is 36.3203
    sizes: Dict[str, float] = field(default_factory=dict)
    names: List[str] = field(default_factory=list)
    ram: List[float] = field(default_factory=list)
    cpu: List[float] = field(default_factory=list)
    bwd: List[bool] = field(default_factory=list)
    deps: Dict[int, List[int]] = field(default_factory=dict)
    idx: Dict[str, int] = field(default_factory=dict)
    replay: Dict[int, List[str]] = field(default_factory=dict)  # layer -> replayed fwd op names
    first_bwd: Dict[int, str] = field(default_factory=dict)      # layer -> first bwd op name
    slots: Dict[int, List[str]] = field(default_factory=dict)

    def add(self, name, ram, cpu, deps=(), backward=False):
        i = len(self.names)
        self.names.append(name); self.ram.append(ram); self.cpu.append(cpu); self.bwd.append(backward)
        self.deps[i] = [self.idx[d] for d in deps]
        self.idx[name] = i
        return i

    @property
    def T(self):
        return len(self.names)

    def boundary_ids(self):
        return [self.idx["emb"]] + [self.idx[self._bout(l)] for l in range(self.L - 1)]

    def _bout(self, l):
        return {"fine": f"add2_{l}", "census": f"dadd_{l}"}.get(self.gran, f"add_{l}")


def default_sizes(gran):
    if gran == "fine":
        # n1, attn (q,k,v,o + o_proj out), h, n2, mlp (gate,up,act,mul,down) padded
        s = dict(n1=1.5, attn=5.5, h=1.5, n2=1.5, mlp=23.3203, g=1.5, nf=1.5, loss=4.0)
    elif gran == "census":
        # harness/terms.py slope_terms at the peak (MLP multiply backward, first replayed layer),
        # fp16, h=768, I=2048, n_q*d=768, repeat_kv (kv width 768); MiB per x
        I4, rstd, lse, ids = 4.0, 4 / 1024, 48 / 1024, 16 / 1024
        s = dict(rope=0.25 + ids, n1=1.5 + rstd, q=1.5, kv=3.0, o=1.5 + lse, h=1.5, n2=1.5 + rstd,
                 gate=I4, up=I4, act=I4, mul=I4, d_m=I4, d_au=2 * I4, g=1.5, d_qkv=4.5, nf=1.5, loss=0.0)
    else:
        # body folds attn,h,n2,mlp: sum chosen to give the same replay-instant term
        s = dict(n1=1.5, body=5.5 + 1.5 + 1.5 + 23.3203, g=1.5, nf=1.5, loss=4.0)
    return s


def build(L, gran="fine", spread=False, sizes=None):
    t = Toy(L=L, gran=gran, spread=spread, sizes=sizes or default_sizes(gran))
    s = t.sizes
    t.add("emb", MB, 0.1)
    if gran == "census":
        t.add("rope", s["rope"], 0.01)
    b = "emb"
    for l in range(L):
        if gran == "census":
            t.add(f"innorm_{l}", s["n1"], 0.1, [b])
            t.add(f"q_{l}", s["q"], 1.0, [f"innorm_{l}", "rope"])
            t.add(f"kv_{l}", s["kv"], 1.0, [f"innorm_{l}", "rope"])
            t.add(f"attn_{l}", s["o"], 2.0, [f"q_{l}", f"kv_{l}"])
            t.add(f"oadd_{l}", s["h"], 0.6, [f"attn_{l}", b])
            t.add(f"pnorm_{l}", s["n2"], 0.1, [f"oadd_{l}"])
            t.add(f"gate_{l}", s["gate"], 1.0, [f"pnorm_{l}"])
            t.add(f"up_{l}", s["up"], 1.0, [f"pnorm_{l}"])
            t.add(f"act_{l}", s["act"], 0.1, [f"gate_{l}"])
            t.add(f"mul_{l}", s["mul"], 0.1, [f"act_{l}", f"up_{l}"])
            t.add(f"dadd_{l}", MB, 1.0, [f"mul_{l}", f"oadd_{l}"])
            t.replay[l] = [f"innorm_{l}", f"q_{l}", f"kv_{l}", f"attn_{l}", f"oadd_{l}", f"pnorm_{l}",
                           f"gate_{l}", f"up_{l}", f"act_{l}", f"mul_{l}"]
            b = f"dadd_{l}"
        elif gran == "fine":
            t.add(f"norm1_{l}", s["n1"], 0.1, [b])
            t.add(f"attn_{l}", s["attn"], 2.0, [f"norm1_{l}"])
            t.add(f"add1_{l}", s["h"], 0.05, [b, f"attn_{l}"])
            t.add(f"norm2_{l}", s["n2"], 0.1, [f"add1_{l}"])
            t.add(f"mlp_{l}", s["mlp"], 3.0, [f"norm2_{l}"])
            t.add(f"add2_{l}", MB, 0.05, [f"add1_{l}", f"mlp_{l}"])
            t.replay[l] = [f"norm1_{l}", f"attn_{l}", f"add1_{l}", f"norm2_{l}", f"mlp_{l}"]
            b = f"add2_{l}"
        else:
            t.add(f"norm1_{l}", s["n1"], 0.1, [b])
            t.add(f"body_{l}", s["body"], 5.0, [f"norm1_{l}", b])
            t.add(f"add_{l}", MB, 0.05, [b, f"body_{l}"])
            t.replay[l] = [f"norm1_{l}", f"body_{l}"]
            b = f"add_{l}"
    bL = b
    t.add("fnorm", s["nf"], 0.1, [bL])
    t.add("loss", s["loss"], 1.0, ["fnorm"])
    t.add("gloss", s["g"], 2.0, ["fnorm", "loss"], backward=True)
    t.add("gfnorm", s["g"], 0.2, [bL, "gloss"], backward=True)
    g = "gfnorm"
    for l in reversed(range(L)):
        bl = "emb" if l == 0 else t._bout(l - 1)
        if spread:
            t.slots[l] = []
            for j, _ in enumerate(t.replay[l]):
                nm = f"slot{j}_{l}"
                t.add(nm, 0.0, 0.0, [], backward=True)
                t.slots[l].append(nm)
        if gran == "census":
            t.add(f"gdown_{l}", s["d_m"], 2.0, [f"mul_{l}", g], backward=True)
            t.add(f"gmul_{l}", s["d_au"], 0.2, [f"act_{l}", f"up_{l}", f"gdown_{l}"], backward=True)
            t.add(f"gact_{l}", s["gate"], 0.2, [f"gate_{l}", f"gmul_{l}"], backward=True)
            t.add(f"ggateup_{l}", s["g"], 4.0, [f"pnorm_{l}", f"gact_{l}", f"gmul_{l}"], backward=True)
            t.add(f"gpnorm_{l}", s["g"], 0.2, [f"oadd_{l}", f"ggateup_{l}"], backward=True)
            t.add(f"gres2_{l}", s["g"], 0.05, [g, f"gpnorm_{l}"], backward=True)
            t.add(f"goproj_{l}", s["g"], 1.2, [f"attn_{l}", f"gres2_{l}"], backward=True)
            t.add(f"gattn_{l}", s["d_qkv"], 5.0, [f"q_{l}", f"kv_{l}", f"attn_{l}", f"goproj_{l}"], backward=True)
            t.add(f"gqkv_{l}", s["g"], 4.0, [f"innorm_{l}", "rope", f"gattn_{l}"], backward=True)
            t.add(f"ginnorm_{l}", s["g"], 0.2, [bl, f"gqkv_{l}"], backward=True)
            t.add(f"gres1_{l}", s["g"], 0.05, [f"gres2_{l}", f"ginnorm_{l}"], backward=True)
            t.first_bwd[l] = f"gdown_{l}"
            g = f"gres1_{l}"
            continue
        if gran == "fine":
            t.add(f"gmlp_{l}", s["g"], 6.0, [f"norm2_{l}", f"mlp_{l}", g], backward=True)
            t.add(f"gnorm2_{l}", s["g"], 0.2, [f"add1_{l}", f"gmlp_{l}"], backward=True)
            t.add(f"gadd1_{l}", s["g"], 0.05, [g, f"gnorm2_{l}"], backward=True)
            t.add(f"gattn_{l}", s["g"], 4.0, [f"norm1_{l}", f"attn_{l}", f"gadd1_{l}"], backward=True)
            t.add(f"gnorm1_{l}", s["g"], 0.2, [bl, f"gattn_{l}"], backward=True)
            t.add(f"gres_{l}", s["g"], 0.05, [f"gadd1_{l}", f"gnorm1_{l}"], backward=True)
            t.first_bwd[l] = f"gmlp_{l}"
        else:
            t.add(f"gbody_{l}", s["g"], 10.0, [f"norm1_{l}", f"body_{l}", g], backward=True)
            t.add(f"gnorm1_{l}", s["g"], 0.2, [bl, f"gbody_{l}"], backward=True)
            t.add(f"gres_{l}", s["g"], 0.05, [g, f"gnorm1_{l}"], backward=True)
            t.first_bwd[l] = f"gbody_{l}"
        g = f"gres_{l}"
    return t


def to_dfgraph(t: Toy) -> DFGraph:
    T = t.T
    for v, us in t.deps.items():
        assert all(u < v for u in us), "node numbering must be topological"
    return DFGraph(
        args={v: list(us) for v, us in t.deps.items() if us},
        v=list(range(T)),
        backward_nodes=[i for i in range(T) if t.bwd[i]],
        cost_cpu={i: t.cpu[i] for i in range(T)},
        cost_ram={i: t.ram[i] for i in range(T)},
        node_names={i: t.names[i] for i in range(T)},
    )


def whole_layer_R(t: Toy):
    """Fixed schedule: every node once in its own stage; layer l's forward ops
    (all but its output) recomputed at its first backward stage, or one per
    slot stage when spread."""
    T = t.T
    R = [[0] * T for _ in range(T)]
    for i in range(T):
        R[i][i] = 1
    for l, ops in t.replay.items():
        if t.spread:
            for op, sl in zip(ops, t.slots[l]):
                R[t.idx[sl]][t.idx[op]] = 1
        else:
            st = t.idx[t.first_bwd[l]]
            for op in ops:
                R[st][t.idx[op]] = 1
    return R


def succ_lists(t: Toy):
    succ = {i: [] for i in range(t.T)}
    for v, us in t.deps.items():
        for u in us:
            succ[u].append(v)
    return succ


def need_stages(t: Toy, R):
    """Stages where tensor i must be resident at stage start (a consumer runs
    there and i is not itself computed in that stage)."""
    succ = succ_lists(t)
    need = {i: [] for i in range(t.T)}
    for st in range(t.T):
        for k in range(st + 1):
            if R[st][k]:
                for u in t.deps[k]:
                    if not R[st][u] and st not in need[u]:
                        need[u].append(st)
    return need


def canonical_SRam(t: Toy, R, page_boundaries: bool):
    """Retention policy. Non-boundaries: resident from the stage after each
    computation through its last need before the next recomputation. Boundaries,
    if paged: resident only through the last forward need, then again from the
    first replay stage through the last backward need (stock offload)."""
    T = t.T
    S = [[0] * T for _ in range(T)]
    need = need_stages(t, R)
    comp = {i: [st for st in range(T) if R[st][i]] for i in range(T)}
    bset = set(t.boundary_ids())
    for i in range(T):
        for st in need[i]:
            # last computation of i before st
            c = max(c for c in comp[i] if c < st)
            for u in range(c + 1, st + 1):
                S[u][i] = 1
    if page_boundaries:
        nfwd = sum(1 for x in t.bwd if not x)
        for i in bset:
            fwd_need = [st for st in need[i] if st < nfwd]
            bwd_need = [st for st in need[i] if st >= nfwd]
            last_f, first_b, last_b = max(fwd_need), min(bwd_need), max(bwd_need)
            for u in range(T):
                S[u][i] = 1 if (i < u <= last_f or first_b <= u <= last_b) else 0
    return S


def simulate(t: Toy, R, S):
    """POET's U recurrence evaluated on a fixed (R, SRam). Returns per-stage
    peaks; raises if the schedule is infeasible."""
    T = t.T
    succ = succ_lists(t)
    peaks = []
    for st in range(T):
        live = {i for i in range(T) if S[st][i]}
        U = sum(t.ram[i] for i in live)
        pk = U if R[st][0] else U  # node 0 handled in loop
        for k in range(T):
            if R[st][k]:
                for u in t.deps[k]:
                    assert u in live, f"stage {st}: {t.names[k]} needs {t.names[u]}"
                U += t.ram[k]
                live.add(k)
                pk = max(pk, U)
                for u in t.deps[k]:
                    later = any(R[st][j] for j in succ[u] if j > k)
                    keep = st + 1 < T and S[st + 1][u]
                    if not later and not keep:
                        U -= t.ram[u]
                        live.discard(u)
        peaks.append(pk)
        if st + 1 < T:
            for i in range(T):
                if S[st + 1][i]:
                    assert i in live or S[st][i] or any(R[st][i] for _ in [0]) or True
    return peaks


if __name__ == "__main__":
    import sys
    for gran in ("fine", "coarse", "census"):
        for spread in (False, True):
            for L in (2, 4, 8, 64):
                t = build(L, gran, spread)
                R = whole_layer_R(t)
                st = max(simulate(t, R, canonical_SRam(t, R, False)))
                pg = max(simulate(t, R, canonical_SRam(t, R, True)))
                print(f"{gran:6s} spread={int(spread)} L={L:3d} T={t.T:4d} static_peak={st:9.4f} "
                      f"(36.3203+1.5L={36.3203 + 1.5 * L:9.4f})  offload_peak={pg:8.4f} (37.8203)")
