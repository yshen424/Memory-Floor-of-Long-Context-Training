"""The time model.

Static step time is quadratic in x (attention) with a constant floor:
    T(x) = a·x² + b·x + c                                  seconds
Stock offload adds a cost linear in x, one per-platform constant kappa
(seconds per x): copies plus the offload path's own per-byte overhead.
    tax(x) = kappa·x / T(x)
It peaks at x = sqrt(c/a) and falls as ~1/x beyond; it crosses eps at the
larger root of  eps·a·x² + (eps·b − kappa)·x + eps·c = 0.

T and kappa are MEASURED per platform, shape and world (they are time
statements, never carried across hardware). Remote jobs follow
calibrate-then-predict: fit T on short arms, measure kappa at one x, then
write the predicted crossing to the log before the long arms run.
"""
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
        """Largest x at which the tax falls to eps (None if it never exceeds eps)."""
        A, B, C = eps * self.a, eps * self.b - self.kappa, eps * self.c
        disc = B * B - 4 * A * C
        if disc < 0:
            return None
        return (-B + math.sqrt(disc)) / (2 * A)


def fit_step_time(points):
    """Least-squares a, b, c from (x, seconds) pairs (needs >= 3 distinct x)."""
    import numpy as np
    xs = np.array([p[0] for p in points], dtype=float)
    ts = np.array([p[1] for p in points], dtype=float)
    A = np.vstack([xs ** 2, xs, np.ones_like(xs)]).T
    (a, b, c), *_ = np.linalg.lstsq(A, ts, rcond=None)
    return float(a), float(b), float(c)


def fit_kappa(pairs):
    """kappa from (x, offload_s - static_s) pairs, least squares through the origin."""
    num = sum(x * d for x, d in pairs)
    den = sum(x * x for x, _ in pairs)
    return num / den


def offload_bytes_per_step(run, x):
    """Bytes stock offload moves per step, each direction: every boundary."""
    return run.shape.layers * run.act_bytes * run.shape.hidden * x * 1024


def pinned_host_bytes(run, x):
    """Pinned host bytes per rank under stock offload (all boundaries), plus the
    FSDP CPU-offload state (fp32 master, gradient, AdamW: 16 B/param/rank)."""
    from .terms import param_counts
    b = offload_bytes_per_step(run, x)
    if run.cpu_offload:
        b += 16 * param_counts(run.shape)[2] / run.world
    return b


def host_limit_x(run, host_bytes_per_rank):
    """Largest x whose pinned bytes fit in the given host memory per rank."""
    from .terms import param_counts
    fixed = 16 * param_counts(run.shape)[2] / run.world if run.cpu_offload else 0
    per_x = run.shape.layers * run.act_bytes * run.shape.hidden * 1024
    return (host_bytes_per_rank - fixed) / per_x
