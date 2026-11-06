from dataclasses import dataclass

@dataclass(frozen=True)
class CapacityModel:
    c0_mib: float
    alpha: float
    alpha_range: tuple = (None, None)
    provenance: str = ''

    def capacity(self, request_mib):
        return self.c0_mib - self.alpha * request_mib

def calibrate(points, alpha_grid=None):
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
    return run.act_bytes * run.shape.intermediate / 1024.0

def bracket(line, run, model, xs):
    rows, rq = [], request_mib_per_x(run)
    for x in sorted(xs):
        pk, req = line.at(x), rq * x
        cap = model.capacity(req)
        rows.append((x, pk, req, cap, cap - pk, pk <= cap))
    fits = [r[0] for r in rows if r[5]]
    fails = [r[0] for r in rows if not r[5]]
    return rows, (max(fits) if fits else None), (min(fails) if fails else None)

def capped_model(cap_gib, delta_mib, alpha, provenance=''):
    return CapacityModel(cap_gib * 1024.0 - delta_mib, alpha, provenance=provenance)
