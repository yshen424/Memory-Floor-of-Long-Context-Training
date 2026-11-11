import time
from dataclasses import dataclass
from typing import Optional

import torch

from .calibration import CalibrationResult
from .pinned_pool import PinnedBuffer

@dataclass
class InflightOp:
    tensor_id: str
    direction: str
    event: 'torch.cuda.Event'
    submitted_step: int
    nbytes: int
    buffer: Optional['PinnedBuffer'] = None
    event_start: Optional['torch.cuda.Event'] = None
    predicted_ms_idle: Optional[float] = None
    predicted_ms_contention: Optional[float] = None

class MemoryStreamManager:

    def __init__(
        self,
        device: torch.device,
        dry_run: bool = True,
        calibration: Optional[CalibrationResult] = None,
    ) -> None:
        self._device = device
        self._stream = torch.cuda.Stream(device=device)
        self._inflight: list[InflightOp] = []
        self._dry_run: bool = dry_run
        self._calibration: Optional[CalibrationResult] = calibration

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    def _record_event(self) -> 'torch.cuda.Event':
        ev = torch.cuda.Event()
        with torch.cuda.stream(self._stream):
            ev.record(self._stream)
        return ev

    def _predicted_ms_pair(
        self, nbytes: int,
    ) -> tuple[Optional[float], Optional[float]]:
        if self._calibration is None:
            return (None, None)
        idle_bw = self._calibration.pcie_bw_idle_one_way_bytes_per_s
        nccl_bw = self._calibration.pcie_bw_under_nccl_one_way_bytes_per_s
        idle_ms = (nbytes / idle_bw) * 1000.0 if idle_bw > 0 else None
        if nccl_bw > 0:
            contention_ms = (nbytes / nccl_bw) * 1000.0
        else:
            contention_ms = idle_ms
        return (idle_ms, contention_ms)

    def submit_evict(
        self,
        tensor_id: str,
        nbytes: int,
        step: int,
        src_tensor: Optional['torch.Tensor'] = None,
        host_buf: Optional['PinnedBuffer'] = None,
    ) -> InflightOp:
        live = (not self._dry_run) and (src_tensor is not None) and (host_buf is not None)
        compute_stream = torch.cuda.current_stream() if live else None
        ev_start: Optional[torch.cuda.Event] = None
        with torch.cuda.stream(self._stream):
            if live:
                self._stream.wait_stream(compute_stream)
                ev_start = torch.cuda.Event(enable_timing=True)
                ev_start.record(self._stream)
                n = src_tensor.numel() * src_tensor.element_size()
                src_dense = src_tensor.contiguous()
                src_view = src_dense.view(-1).view(torch.uint8)
                host_buf.tensor[:n].copy_(src_view, non_blocking=True)
                src_tensor.record_stream(self._stream)
                src_dense.record_stream(self._stream)
            ev = torch.cuda.Event(enable_timing=live)
            ev.record(self._stream)
        pred_idle, pred_cont = self._predicted_ms_pair(nbytes)
        op = InflightOp(tensor_id=tensor_id, direction='evict',
                        event=ev, submitted_step=step, nbytes=nbytes,
                        buffer=host_buf if live else None,
                        event_start=ev_start,
                        predicted_ms_idle=pred_idle,
                        predicted_ms_contention=pred_cont)
        self._inflight.append(op)
        return op

    def submit_prefetch(
        self,
        tensor_id: str,
        nbytes: int,
        step: int,
        host_buf: Optional['PinnedBuffer'] = None,
        dst_tensor: Optional['torch.Tensor'] = None,
    ) -> InflightOp:
        live = (not self._dry_run) and (host_buf is not None) and (dst_tensor is not None)
        compute_stream = torch.cuda.current_stream() if live else None
        ev_start: Optional[torch.cuda.Event] = None
        with torch.cuda.stream(self._stream):
            if live:
                self._stream.wait_stream(compute_stream)
                ev_start = torch.cuda.Event(enable_timing=True)
                ev_start.record(self._stream)
                n = dst_tensor.numel() * dst_tensor.element_size()
                dst_view = dst_tensor.view(-1).view(torch.uint8)
                dst_view.copy_(host_buf.tensor[:n], non_blocking=True)
                dst_tensor.record_stream(self._stream)
            ev = torch.cuda.Event(enable_timing=live)
            ev.record(self._stream)
        pred_idle, pred_cont = self._predicted_ms_pair(nbytes)
        op = InflightOp(tensor_id=tensor_id, direction='prefetch',
                        event=ev, submitted_step=step, nbytes=nbytes,
                        buffer=host_buf if live else None,
                        event_start=ev_start,
                        predicted_ms_idle=pred_idle,
                        predicted_ms_contention=pred_cont)
        self._inflight.append(op)
        return op

    def wait_compute_on(self, op: InflightOp) -> None:
        if op.event is None:
            return
        torch.cuda.current_stream().wait_event(op.event)

    def wait(self, op: InflightOp) -> float:
        if op.event.query():
            return 0.0
        t0 = time.perf_counter()
        op.event.synchronize()
        return (time.perf_counter() - t0) * 1000.0

    def reap_completed(self) -> list[InflightOp]:
        done: list[InflightOp] = []
        pending: list[InflightOp] = []
        for op in self._inflight:
            if op.event.query():
                done.append(op)
            else:
                pending.append(op)
        self._inflight = pending
        return done

    def op_pcie_ms(self, op: InflightOp) -> Optional[float]:
        if op.event_start is None:
            return None
        return op.event_start.elapsed_time(op.event)
