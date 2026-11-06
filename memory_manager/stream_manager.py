"""InflightOp + MemoryStreamManager — async transfer abstraction.

Owns the dedicated CUDA mem stream used for D2H / H2D activation
transfers, with bracketing CUDA events for observed-vs-predicted timing.
Live and dry-run modes share the submit/wait/reap lifecycle so smoke
tests can exercise the engine without real bytes moving.
"""

import time
from dataclasses import dataclass
from typing import Optional

import torch

from .calibration import CalibrationResult
from .pinned_pool import PinnedBuffer


# ---------------------------------------------------------------------------
# InflightOp + MemoryStreamManager
# ---------------------------------------------------------------------------

@dataclass
class InflightOp:
    """An async memcpy submission record.

    Live mode (``MemoryStreamManager.dry_run=False`` with a tensor handle
    present) issues a real ``cudaMemcpyAsync`` and records an event after
    it. Dry-run mode records only the event so the rest of the engine
    (event-query, wait, reap) can be exercised end-to-end without bytes
    moving — used by smoke tests and ablation runs.

    ``buffer``: the pinned host
    buffer involved in the transfer. For an evict, this is where bytes
    are being copied *to*; for a prefetch, where bytes are coming
    *from*. The buffer is held in the InflightOp so the caller (the
    engine) can release it back to the pool at the right point in the
    evict→prefetch round-trip.

    ``event_start``: a second CUDA event recorded on the mem stream
    *immediately before* the ``.copy_()`` enqueue. Together with
    ``event`` (the completion event, semantically ``event_end``), the
    pair brackets the actual memcpy. ``event_start.elapsed_time(event)``
    returns real GPU-side transfer ms. None in dry-run / no-op submissions (no
    copy enqueued, no start event to record).

    ``predicted_ms_idle`` / ``predicted_ms_contention``: the cost-model
    prediction at submit time, in ms, computed from calibration.
    ``predicted_ms_idle`` uses the no-contention PCIe bandwidth
    (``pcie_bw_idle_one_way``) — answers "is the physical link healthy?".
    ``predicted_ms_contention`` uses the under-NCCL bandwidth — answers "did
    the engine's cost math match reality?". Both are None when
    MemoryStreamManager was constructed without a CalibrationResult (e.g., unit
    tests). At reap time, the engine compares these against the observed
    GPU-side ms (op_pcie_ms) to flag calibration drift.
    """
    tensor_id: str
    direction: str            # 'evict' | 'prefetch'
    event: 'torch.cuda.Event'
    submitted_step: int
    nbytes: int
    buffer: Optional['PinnedBuffer'] = None
    event_start: Optional['torch.cuda.Event'] = None
    predicted_ms_idle: Optional[float] = None
    predicted_ms_contention: Optional[float] = None


class MemoryStreamManager:
    """Owns the single mem stream and the in-flight op table.

    Live mode: submit_evict/submit_prefetch enqueue a real cudaMemcpyAsync
    plus bracketing CUDA events for observed-vs-predicted timing. Dry-run
    mode records only the events (event.query/synchronize still behave
    normally on an empty stream), so smoke tests can exercise the
    submit/wait/reap lifecycle without real bytes moving.

    The ``dry_run`` switch lives here. When ``dry_run=False`` AND the
    caller supplies the relevant tensor / buffer handles, submit_* issue
    real D2H / H2D copies before recording the completion event. When
    ``dry_run=False`` but handles are absent (the record carries no
    ``record.tensor``), the call falls back to the no-op event path.
    """

    def __init__(
        self,
        device: torch.device,
        dry_run: bool = True,
        calibration: Optional[CalibrationResult] = None,
    ) -> None:
        self._device = device
        self._stream = torch.cuda.Stream(device=device)
        self._inflight: list[InflightOp] = []
        # dry_run lives here, not on PlacementEngine. The engine's dry_run
        # kwarg is a thin forwarder that the integration may use to
        # configure the stream manager — but the stream manager is the
        # authoritative owner of the switch.
        self._dry_run: bool = dry_run
        # Calibration is used at submit time to stamp predicted-ms fields
        # onto each InflightOp. Optional — unit tests instantiate without
        # it and the predicted fields stay None (op_pcie_ms still works;
        # drift stats see an empty sample set and report zeros).
        self._calibration: Optional[CalibrationResult] = calibration

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    def _record_event(self) -> 'torch.cuda.Event':
        ev = torch.cuda.Event()
        with torch.cuda.stream(self._stream):
            ev.record(self._stream)
        return ev

    # Pair of predicted-ms numbers for one transfer of ``nbytes``. The two
    # numbers diagnose different failures:
    #   - idle: ratio of observed to idle PCIe → physical link health
    #   - contention: ratio of observed to under-NCCL PCIe → cost-model
    #     accuracy (this is the bandwidth the engine's
    #     ``_available_overlap_ms`` actually used when it decided to evict)
    # Calibration measures D2H only — the same number is used for both
    # directions here. The asymmetric H2D number is a known calibration
    # gap.
    # Returns (None, None) if calibration is absent or both bw fields are 0.
    # Contention falls back to idle when under-NCCL bw is 0 (single-rank
    # smoke tests skip the NCCL benchmark) so the cost-model column is
    # never silently absent in multi-step runs.
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
            contention_ms = idle_ms  # smoke-test fallback
        return (idle_ms, contention_ms)

    def submit_evict(
        self,
        tensor_id: str,
        nbytes: int,
        step: int,
        src_tensor: Optional['torch.Tensor'] = None,
        host_buf: Optional['PinnedBuffer'] = None,
    ) -> InflightOp:
        """Submit a D2H copy of ``src_tensor`` into ``host_buf``.

        In dry-run mode (or when src_tensor / host_buf are absent), no
        cudaMemcpyAsync is issued — just a no-op event for the engine's
        bookkeeping. In live mode with both handles, the bytes are
        actually copied on the mem stream and the completion event
        fires when the copy finishes.
        """
        live = (not self._dry_run) and (src_tensor is not None) and (host_buf is not None)
        # TRAP: capture the compute stream BEFORE entering the mem-stream
        # context. Inside ``with torch.cuda.stream(self._stream)``,
        # ``current_stream()`` returns ``self._stream`` itself, so an
        # in-block ``wait_stream(current_stream())`` is a self-wait no-op:
        # the copy races the producing kernel and backward gradients come
        # out NaN. A small model can hide this, because its producing
        # kernel has already finished by the time pack_fn runs.
        compute_stream = torch.cuda.current_stream() if live else None
        # Per-op timing bracket. ``ev_start`` is recorded on the mem stream
        # right before the .copy_() enqueue; ``ev`` (completion) right after.
        # Both events need ``enable_timing=True`` for elapsed_time() to work.
        # The bracket is only meaningful in live mode — dry-run path leaves
        # event_start=None.
        ev_start: Optional[torch.cuda.Event] = None
        with torch.cuda.stream(self._stream):
            if live:
                # Make the mem stream wait for the compute stream's current
                # point before issuing the D2H. src_tensor was just produced
                # on the compute stream — without this sync, the memcpy
                # could race the producing kernel and copy garbage.
                self._stream.wait_stream(compute_stream)
                ev_start = torch.cuda.Event(enable_timing=True)
                ev_start.record(self._stream)
                # Flat uint8 view: a memcpy is byte-blind, so we don't
                # need to preserve dtype/shape at this layer — the engine
                # restores shape/dtype on the prefetch side.
                n = src_tensor.numel() * src_tensor.element_size()
                # TRAP: flatten BEFORE the dtype view.
                # .view(torch.uint8) on a 0-dim tensor of a wider dtype
                # raises "self.dim() cannot be 0 to view Long as Byte",
                # and the memory-efficient SDPA kernel saves a 0-dim int64
                # philox seed.
                src_dense = src_tensor.contiguous()
                src_view = src_dense.view(-1).view(torch.uint8)
                host_buf.tensor[:n].copy_(src_view, non_blocking=True)
                # Tell the allocator both storages may be touched by the mem
                # stream — prevents reuse mid-copy. src_tensor is recorded
                # as well as src_dense: if the save is NON-CONTIGUOUS,
                # src_dense is a fresh buffer and the .contiguous() gather is
                # enqueued on the mem stream reading src_tensor, whose block
                # the caching allocator may free against the COMPUTE stream
                # once pack_fn returns a descriptor and autograd drops its
                # last reference. For a contiguous save the second call is
                # a no-op.
                src_tensor.record_stream(self._stream)
                src_dense.record_stream(self._stream)
            # Completion event. Enable timing in live mode so the engine
            # can compute ev_start.elapsed_time(ev) at reap time. Dry-run
            # keeps the cheaper default (no-timing) event.
            ev = torch.cuda.Event(enable_timing=live)
            ev.record(self._stream)
        # Stamp predicted ms at submit time. Computed for every submission
        # (even dry-run) so the field is consistently populated when
        # calibration is present; reap-time logging keys on whether observed_ms
        # is also present.
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
        """Submit an H2D copy of ``host_buf`` into ``dst_tensor``.

        Same dry-run / live semantics as submit_evict. ``dst_tensor``
        must be a pre-allocated HBM tensor of at least nbytes (the
        engine / autograd unpack-hook owns the allocation).
        """
        live = (not self._dry_run) and (host_buf is not None) and (dst_tensor is not None)
        # Same self-wait trap as submit_evict — capture the compute stream
        # BEFORE the mem-stream context so wait_stream actually crosses
        # streams.
        compute_stream = torch.cuda.current_stream() if live else None
        # Per-op timing bracket, mirrors submit_evict. See
        # InflightOp.event_start docstring.
        ev_start: Optional[torch.cuda.Event] = None
        with torch.cuda.stream(self._stream):
            if live:
                # Mirror the evict path — wait on the compute stream
                # before writing dst. dst was just allocated via
                # torch.empty on the compute-stream allocator; the storage
                # is technically immediately usable, but if a caller ever
                # initializes dst before submit_prefetch (e.g., torch.zeros)
                # this guards against racing the init kernel.
                self._stream.wait_stream(compute_stream)
                ev_start = torch.cuda.Event(enable_timing=True)
                ev_start.record(self._stream)
                n = dst_tensor.numel() * dst_tensor.element_size()
                # Same 0-dim trap as submit_evict: flatten before the dtype
                # view, or the restore of a 0-dim tensor raises.
                dst_view = dst_tensor.view(-1).view(torch.uint8)
                dst_view.copy_(host_buf.tensor[:n], non_blocking=True)
                dst_tensor.record_stream(self._stream)
            ev = torch.cuda.Event(enable_timing=live)
            ev.record(self._stream)
        # See submit_evict — same predicted-ms stamping. Direction-asymmetry
        # in PCIe bw is a known calibration gap; the D2H number is reused
        # for H2D.
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
        """Order the compute stream behind ``op`` without blocking the CPU.

        ``wait()`` stops the Python thread until the
        copy lands, which drains the kernel-enqueue pipeline and turns copy
        time into step time roughly 1:1. This instead records the
        dependency on the GPU: the compute stream will not start the
        kernels enqueued after this point until ``op`` completes, while the
        CPU keeps running ahead. Use when the caller only needs the bytes
        to be there before subsequent *kernels* read them, which is the
        case for an unpacked activation.
        """
        if op.event is None:
            return
        torch.cuda.current_stream().wait_event(op.event)

    def wait(self, op: InflightOp) -> float:
        """Block until ``op`` completes; return CPU wall time waited (ms).

        Returns 0.0 if the event was already done at call time (no stall).
        Dry-run mode: almost always 0.0 because the stream had no actual
        work. Live mode: real cudaMemcpyAsync latencies show up here.
        """
        if op.event.query():
            return 0.0
        t0 = time.perf_counter()
        op.event.synchronize()
        return (time.perf_counter() - t0) * 1000.0

    def reap_completed(self) -> list[InflightOp]:
        """Move finished ops out of the in-flight list, return them."""
        done: list[InflightOp] = []
        pending: list[InflightOp] = []
        for op in self._inflight:
            if op.event.query():
                done.append(op)
            else:
                pending.append(op)
        self._inflight = pending
        return done

    # Real GPU-side memcpy duration in ms.
    # Returns None when no start event was recorded (dry-run / no-op
    # submission). Caller must ensure both events have fired; reap_completed
    # already gates on the completion event, so calling op_pcie_ms on a
    # reaped op is safe.
    def op_pcie_ms(self, op: InflightOp) -> Optional[float]:
        if op.event_start is None:
            return None
        # elapsed_time blocks the host until both events have fired and
        # raises if either was recorded without enable_timing. submit_*
        # only sets event_start with enable_timing=True, so this is safe.
        return op.event_start.elapsed_time(op.event)
