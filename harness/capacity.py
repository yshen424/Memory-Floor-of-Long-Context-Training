"""Capacity and brackets.

Measured on V100: no single capacity separates the rungs that fit
from the rungs that failed. What does is a capacity that shrinks with the size of
the request that fails, the replayed MLP activation (2·I bytes per token):

    fits  iff  peak(x) <= C0 - alpha * request(x)

i.e. a large contiguous request needs headroom in proportion to its size
(fragmentation). C0 and alpha are PLATFORM CONSTANTS, calibrated from a declared
set of brackets and recorded, not claimed. Capacity also depends on the
allocation pattern (a fit bounds it from below; an out-of-memory failure never
bounds it from above), so every bracket is predicted to ±1 rung.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class CapacityModel:
    c0_mib: float
    alpha: float
    alpha_range: tuple = (None, None)     # feasible alphas on the calibration set
    provenance: str = ''

    def capacity(self, request_mib):
        return self.c0_mib - self.alpha * request_mib


def calibrate(points, alpha_grid=None):
    """points: (peak_mib, request_mib, fit: bool). Returns the CapacityModel at the
    centre of the feasible region (alpha midpoint, C0 midpoint at that alpha),
    or None if the points cannot be separated."""
    alpha_grid = alpha_grid or [a / 1000 for a in range(0, 1001)]
    region = []
    for a in alpha_grid:
        lo = max((p + a * r for p, r, f in points if f), default=float('-inf'))
        hi = min((p + a * r for p, r, f in points if not f), default=float('inf'))
        if lo < hi:
            region.append((a, lo, hi))
    if not region:
        return None
    a_mid = (region[0][0] + region[-1][0]) / 2
    a, lo, hi = min(region, key=lambda t: abs(t[0] - a_mid))
    return CapacityModel((lo + hi) / 2, a, (region[0][0], region[-1][0]))


def request_mib_per_x(run):
    """The request that fails at the frontier: the replayed MLP activation."""
    return run.act_bytes * run.shape.intermediate / 1024.0


def bracket(line, run, model, xs):
    """Per rung: (x, peak, request, capacity, margin, fits). Plus the predicted
    largest fit and smallest fail on this ladder."""
    rows, rq = [], request_mib_per_x(run)
    for x in sorted(xs):
        pk, req = line.at(x), rq * x
        cap = model.capacity(req)
        rows.append((x, pk, req, cap, cap - pk, pk <= cap))
    fits = [r[0] for r in rows if r[5]]
    fails = [r[0] for r in rows if not r[5]]
    return rows, (max(fits) if fits else None), (min(fails) if fails else None)


def capped_model(cap_gib, delta_mib, alpha, provenance=''):
    """Capacity under --hbm_cap_gb (set_per_process_memory_fraction): the cap bounds
    the caching allocator, the CUDA context lies outside it, so
        fits iff peak <= cap - delta - alpha * request.
    V100 (7760): delta 27-39 MiB at alpha 0.163."""
    return CapacityModel(cap_gib * 1024.0 - delta_mib, alpha, provenance=provenance)
