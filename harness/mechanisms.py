"""Every mechanism as a line peak(x) = c + m·x in MiB, x = tokens/1024."""
from dataclasses import dataclass, field

from .terms import (MiB, slope_terms, intercept_terms, floor_extra_terms,
                    second_replay_terms, optimizer_terms, backward_end_terms,
                    deep_replay_terms)


@dataclass
class Line:
    c: float                                   # MiB
    m: float                                   # MiB per x
    parts: dict = field(default_factory=dict)  # name -> (MiB intercept, MiB/x slope)
    alts: tuple = ()                           # (name, Line): other peak instants

    def at(self, x):
        """The peak is the largest of the instants; c, m are the main one's."""
        return max([self.c + self.m * x] + [ln.at(x) for _, ln in self.alts])

    def binding(self, x):
        best = ('main', self.c + self.m * x)
        for name, ln in self.alts:
            if ln.at(x) > best[1]:
                best = (name, ln.at(x))
        return best[0]


def _optimizer_alt(run, platform, step0=False):
    """The parameter-dominated instants every mechanism shares: the optimizer
    step (the same in step 0, which allocates AdamW's state) and the end of
    backward (no AdamW state yet in step 0)."""
    out = []
    for name, t in (('optimizer step', optimizer_terms(run, platform)),
                    ('end of backward', backward_end_terms(run, platform, step0))):
        if t is not None:
            it, per_tok = t
            out.append((name, Line(sum(it.values()) / MiB, sum(per_tok.values()) / 1024.0)))
    return tuple(out)


def _slopes(run):
    mig, non = slope_terms(run)
    return sum(mig.values()) / 1024.0, sum(non.values()) / 1024.0, mig, non


def static_line(run, platform, step0=False):
    m_mig, m_non, mig, non = _slopes(run)
    it = intercept_terms(run, platform, step0)
    parts = {k: (v / MiB, 0.0) for k, v in it.items()}
    parts.update({k: (0.0, v / 1024.0) for k, v in {**mig, **non}.items()})
    c = sum(it.values()) / MiB
    ex_c, ex_m = second_replay_terms(run)
    second = Line(c + sum(ex_c.values()) / MiB, m_mig + m_non + sum(ex_m.values()) / 1024.0)
    # No second-replay instant in step 0 (POST HOC, 22 static step-0 peaks over 7
    # shapes all on the first-replay line, tests/test_harness_step0.py): the
    # previous layer's fp16 weight gradients it holds are live only from step 1
    # on, as the 7765 world-2 snapshots show.
    alts = () if step0 else (('second replay', second),)
    if not step0 and not run.cpu_offload and run.shape.layers >= 3:
        dc, dm = deep_replay_terms(run)
        alts += (('deep replay', Line(second.c + sum(dc.values()) / MiB, second.m + sum(dm.values()) / 1024.0)),)
    return Line(c, m_mig + m_non, parts, alts + _optimizer_alt(run, platform, step0))


def partial_line(run, platform, absent, step0=False):
    """`absent` boundaries off the device at the peak instant, below the cap."""
    cap = run.shape.layers - 1
    if not 0 <= absent <= cap:
        raise ValueError(f"absent={absent} outside [0, {cap}]")
    ln = static_line(run, platform, step0)
    b = run.act_bytes * run.shape.hidden / 1024.0
    return Line(ln.c, ln.m - absent * b, {**ln.parts, f'{absent} boundaries absent': (0.0, -absent * b)},
                _optimizer_alt(run, platform, step0))


def floor_line(run, platform, restore_depth=0, step0=False):
    """At the pick cap: every boundary but the restore_depth+1 nearest the peak
    is absent, and the peak relocates to backward layer max(W_r, 1). step0: the
    first step (no AdamW state, W - 1 fewer shard units)."""
    m_mig, m_non, _, non = _slopes(run)
    b = run.act_bytes * run.shape.hidden / 1024.0
    resident = restore_depth + 1
    it = intercept_terms(run, platform, step0)
    ex = floor_extra_terms(run, restore_depth, step0)
    parts = {k: (v / MiB, 0.0) for k, v in {**it, **ex}.items()}
    parts.update({k: (0.0, v / 1024.0) for k, v in non.items()})
    parts[f'{resident} resident boundaries'] = (0.0, resident * b)
    return Line((sum(it.values()) + sum(ex.values())) / MiB, m_non + resident * b, parts,
                _optimizer_alt(run, platform, step0))


def line_for(mechanism, run, platform, restore_depth=0, absent=None, kw_step0=False):
    """Mechanisms the closed form covers. Stock offload and forced eviction move
    every boundary, so they sit on the restore-depth-0 floor."""
    if mechanism == 'static':
        return static_line(run, platform)
    if mechanism in ('stock_offload', 'force_evict'):
        return floor_line(run, platform, 0)
    if mechanism == 'engine_cap':
        return floor_line(run, platform, restore_depth)
    if mechanism == 'engine_partial':
        return partial_line(run, platform, absent, kw_step0)
    raise ValueError(f"mechanism {mechanism!r} is not derived")


def peak(mechanism, run, platform, x, **kw):
    return line_for(mechanism, run, platform, **kw).at(x)


if __name__ == '__main__':
    from .config import Run, PLATFORMS, TIER_A
    v = PLATFORMS['v100']
    for L in (32, 64):
        for W in (2, 3, 4):
            r = Run(shape=TIER_A.with_layers(L), world=W)
            s, f = static_line(r, v), floor_line(r, v)
            print(f"L={L} W={W}: static {s.c:9.4f} + {s.m:.6f}x   floor {f.c:9.4f} + {f.m:.6f}x   "
                  f"span cap {s.m / f.m:.4f}")
