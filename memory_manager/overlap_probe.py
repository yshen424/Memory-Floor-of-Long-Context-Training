"""Forced PCIe/NCCL overlap probe.

Activation migrations can land entirely in the gaps between FSDP collectives,
so the PCIe/NCCL coupling is not guaranteed to occur in-system (the collective
duty cycle is independent of model width; only tokens-per-step moves it).

This probe forces the overlap instead of waiting for it. It issues synthetic
transfers of realistic size on a private stream while real FSDP collectives
run, so we can measure both sides of the trade:

    serialize:  t_collective + t_migration
    overlap:    max(t_collective', t_migration')

and decide whether avoiding collective windows is worth anything.

Two design points make this safe and precise:

  * The source buffers are **synthetic** -- never read by autograd. So the
    probe stream needs no dependency on the compute stream, which is what
    keeps real migrations out of collective windows (the wait_stream in
    MemoryStreamManager's evict path), and a copy cannot race a tensor
    autograd still reads.

  * The probe **saturates** a stretch of time rather than trying to aim at a
    collective. Hitting a few-millisecond window from Python is hopeless --
    launch jitter is comparable to the window. Saturating and classifying each
    copy post-hoc by its measured overlap needs no timing precision at all.

The probe's own CUDA-event timings are authoritative for probe bandwidth. The
chrome trace supplies the other half: whether NCCL kernels got slower while the
probe was running.
"""

from __future__ import annotations

import json
from typing import Optional

import torch

# Bundle sizes (MB) of the kind the engine moves. Using a realistic
# distribution rather than a round number keeps the measurement comparable to
# what the engine moves.
DEFAULT_SIZES_MB = (1.2, 9.4, 18.9, 25.2)

# Used only to convert a target duration into an op count at arm() time. Being
# wrong here just makes the train longer or shorter than intended; it does not
# affect any measurement, since every op is individually event-timed.
NOMINAL_BW_GBPS = 11.0


class OverlapProbe:
    """Saturates PCIe with synthetic transfers while real collectives run."""

    def __init__(
        self,
        device: torch.device,
        sizes_mb: tuple = DEFAULT_SIZES_MB,
        directions: tuple = ('d2h', 'h2d'),
        duty: float = 1.0,
        enabled: bool = True,
    ) -> None:
        self._device = device
        self._sizes_mb = tuple(sizes_mb)
        self._directions = tuple(directions)
        self._duty = max(0.01, min(1.0, float(duty)))
        self._enabled = enabled
        self._stream = torch.cuda.Stream(device=device)
        self._records: list[dict] = []
        self._pending: list[dict] = []
        self._armed_step: Optional[int] = None

        # One device buffer + one pinned host buffer per size. uint8 because a
        # memcpy is byte-blind; dtype would only matter if anyone read these.
        self._dev: dict[float, torch.Tensor] = {}
        self._host: dict[float, torch.Tensor] = {}
        for mb in self._sizes_mb:
            n = int(mb * 1e6)
            self._dev[mb] = torch.empty(n, dtype=torch.uint8, device=device)
            self._host[mb] = torch.empty(n, dtype=torch.uint8, device='cpu',
                                         pin_memory=True)

        props = torch.cuda.get_device_properties(device)
        # clock_rate is kHz; cycles-per-ms = kHz. Only used for duty < 1.0.
        self._cycles_per_ms = int(getattr(props, 'clock_rate', 1_000_000))

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def buffer_bytes(self) -> int:
        return sum(int(mb * 1e6) for mb in self._sizes_mb) * 2

    def _op_plan(self, n_ops: int) -> list[tuple]:
        """Round-robin (size, direction) so no single size dominates a phase."""
        plan = []
        combos = [(mb, d) for mb in self._sizes_mb for d in self._directions]
        for i in range(n_ops):
            plan.append(combos[i % len(combos)])
        return plan

    def arm(self, step: int, target_ms: float) -> int:
        """Enqueue a back-to-back train of copies covering ~target_ms.

        The whole train is enqueued up front so the GPU executes it without
        further CPU involvement -- a Python thread paced by time.sleep would
        add GIL jitter comparable to the effect we are trying to measure.
        """
        if not self._enabled:
            return 0

        mean_mb = sum(self._sizes_mb) / len(self._sizes_mb)
        ms_per_op = (mean_mb * 1e6) / (NOMINAL_BW_GBPS * 1e9) * 1000.0
        ms_per_op /= self._duty
        n_ops = max(1, int(target_ms / max(ms_per_op, 1e-6)))

        sleep_ms = 0.0
        if self._duty < 1.0:
            base = (mean_mb * 1e6) / (NOMINAL_BW_GBPS * 1e9) * 1000.0
            sleep_ms = base * (1.0 / self._duty - 1.0)

        plan = self._op_plan(n_ops)
        with torch.cuda.stream(self._stream):
            for mb, direction in plan:
                ev_s = torch.cuda.Event(enable_timing=True)
                ev_e = torch.cuda.Event(enable_timing=True)
                ev_s.record(self._stream)
                if direction == 'd2h':
                    self._host[mb].copy_(self._dev[mb], non_blocking=True)
                else:
                    self._dev[mb].copy_(self._host[mb], non_blocking=True)
                ev_e.record(self._stream)
                self._pending.append({
                    'step': step, 'size_mb': mb, 'direction': direction,
                    'ev_s': ev_s, 'ev_e': ev_e,
                })
                if sleep_ms > 0:
                    # NOTE: _sleep is a spin kernel and occupies an SM. Only
                    # used for duty < 1.0; the duty-1.0 path is unaffected.
                    torch.cuda._sleep(int(sleep_ms * self._cycles_per_ms))

        self._armed_step = step
        return n_ops

    def reap(self) -> float:
        """Drain the train, convert events to per-op timings.

        Returns milliseconds spent waiting for the train to finish -- i.e. how
        far the probe overran the step it was armed for. Call this *after* the
        step timer has stopped so the wait does not contaminate step_ms.
        """
        if not self._pending:
            return 0.0
        import time as _time
        t0 = _time.perf_counter()
        self._stream.synchronize()
        overrun_ms = (_time.perf_counter() - t0) * 1000.0

        for p in self._pending:
            ms = p['ev_s'].elapsed_time(p['ev_e'])
            nbytes = int(p['size_mb'] * 1e6)
            self._records.append({
                'step': p['step'],
                'size_mb': p['size_mb'],
                'direction': p['direction'],
                'ms': round(ms, 4),
                'gbps': round(nbytes / (ms * 1e-3) / 1e9, 3) if ms > 0 else 0.0,
            })
        self._pending.clear()
        return overrun_ms

    def summary(self) -> dict:
        out = {'n_ops': len(self._records), 'duty': self._duty,
               'sizes_mb': list(self._sizes_mb)}
        for d in self._directions:
            g = sorted(r['gbps'] for r in self._records if r['direction'] == d)
            if not g:
                continue
            out[d] = {
                'n': len(g),
                'p10': g[len(g) // 10],
                'median': g[len(g) // 2],
                'p90': g[9 * len(g) // 10],
            }
        return out

    def save(self, path: str) -> None:
        with open(path, 'w') as fh:
            json.dump({'summary': self.summary(), 'ops': self._records}, fh)
