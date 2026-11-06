import math
from dataclasses import dataclass

@dataclass(frozen=True)
class TimeCalibration:
    a: float
    b: float
    c: float
    kappa: float
    provenance: str = ''

    def step_time(self, x):
        return self.a * x * x + self.b * x + self.c

    def tax(self, x):
        return self.kappa * x / self.step_time(x)

    def tax_peak_x(self):
        return math.sqrt(self.c / self.a)

    def crossing(self, eps):
        A, B, C = eps * self.a, eps * self.b - self.kappa, eps * self.c
        disc = B * B - 4 * A * C
        if disc < 0:
            return None
        return (-B + math.sqrt(disc)) / (2 * A)

def fit_step_time(points):
    import numpy as np
    xs = np.array([p[0] for p in points], dtype=float)
    ts = np.array([p[1] for p in points], dtype=float)
    A = np.vstack([xs ** 2, xs, np.ones_like(xs)]).T
    (a, b, c), *_ = np.linalg.lstsq(A, ts, rcond=None)
    return float(a), float(b), float(c)

def fit_kappa(pairs):
    num = sum(x * d for x, d in pairs)
    den = sum(x * x for x, _ in pairs)
    return num / den

def offload_bytes_per_step(run, x):
    return run.shape.layers * run.act_bytes * run.shape.hidden * x * 1024

def pinned_host_bytes(run, x):
    from .terms import param_counts
    b = offload_bytes_per_step(run, x)
    if run.cpu_offload:
        b += 16 * param_counts(run.shape)[2] / run.world
    return b

def host_limit_x(run, host_bytes_per_rank):
    from .terms import param_counts
    fixed = 16 * param_counts(run.shape)[2] / run.world if run.cpu_offload else 0
    per_x = run.shape.layers * run.act_bytes * run.shape.hidden * 1024
    return (host_bytes_per_rank - fixed) / per_x
