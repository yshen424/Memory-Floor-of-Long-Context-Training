"""
Calibration helpers — measured at engine init each launch.

The placement engine's evict-vs-recompute decision needs realistic
per-byte / per-flop cost numbers. Hard-coding V100 spec values is
brittle (varies by NCCL config, allocator state, contention with
other jobs), so we measure fresh at every launch and feed the
results into the engine's cost model.

Benchmarks:

  pcie_bandwidth
      One-way HBM -> pinned-CPU memcpy with pre-allocated buffers on
      both ends. The "best-case" PCIe number: no allocator overhead,
      no fragmentation.

  realistic_round_trip
      Evict + free + alloc + prefetch loop with a fresh HBM tensor
      every iteration. Captures the cudaMalloc / segment-allocation
      cost we'll actually pay in the live evict/prefetch path.
      Always slower than pcie_bandwidth — the delta is the allocator
      overhead.

  gpu_flops
      Square fp16 matmul. fp16 matches LLaMA's training compute dtype,
      so this is the right baseline for recompute-cost estimation.

  pcie_bw_under_nccl_one_way
      Same one-way HBM -> pinned-CPU memcpy as ``pcie_bandwidth``, but
      timed *while a background thread issues back-to-back NCCL
      all-gathers on every rank*. On a cluster without GPUDirect RDMA,
      NCCL stages through host DRAM and competes with activation
      D2H/H2D on the same PCIe link. The engine's
      ``_available_overlap_ms`` uses this number for backward-phase
      windows; ``pcie_bw_idle_one_way`` for the rest.

  nccl_allgather_bytes_per_s
      Effective bytes/sec of a single-rank dist.all_gather across the
      world. Used by the engine's ``_exposed_reallgather_ms`` to charge
      recompute the cost of re-gathering a layer's params when the FSDP
      coordinator has freed them.

Naming: ``pcie_bw_idle_one_way_bytes_per_s`` and
``pcie_bw_idle_round_trip_bytes_per_s`` make the "idle" vs "under-NCCL"
distinction explicit in every downstream consumer.

All values land in CalibrationResult and get serialized into
profile_rank<N>.json (schema_version 2) for offline analysis.
"""

import threading
import time
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.distributed as dist


# Sanity bounds. The under-NCCL number can fall far below the idle number;
# we don't error out, just flag it in JSON. Set for V100, PCIe 3.0,
# Ethernet-class NCCL, no GDR.
_SANITY_BOUNDS = {
    'pcie_bw_idle_one_way_bytes_per_s': 8 * 1e9,
    'pcie_bw_idle_round_trip_bytes_per_s': 4 * 1e9,
    'pcie_bw_under_nccl_one_way_bytes_per_s': 1 * 1e9,
    'gpu_flops_per_s': 5 * 1e12,
    'nccl_allgather_bytes_per_s': 0.2 * 1e9,
}


@dataclass
class CalibrationResult:
    pcie_bw_idle_one_way_bytes_per_s: float
    pcie_bw_idle_round_trip_bytes_per_s: float
    gpu_flops_per_s: float
    # Zero is a sentinel "not measured" —
    # the engine treats this as "fall back to idle bandwidth" so the
    # cost model still runs in single-rank smoke tests.
    pcie_bw_under_nccl_one_way_bytes_per_s: float = 0.0
    nccl_allgather_bytes_per_s: float = 0.0
    # Sanity flags — True iff the measurement landed below the bound.
    # Surfaces in JSON so a degenerate calibration can be spotted
    # at a glance without re-running the numbers by hand.
    sanity_flags: dict = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    # ------------------------------------------------------------------
    # Alias properties: ``pcie_bw_bytes_per_s`` /
    # ``realistic_round_trip_bytes_per_s`` are read-only views of the
    # idle fields.
    # ------------------------------------------------------------------
    @property
    def pcie_bw_bytes_per_s(self) -> float:
        return self.pcie_bw_idle_one_way_bytes_per_s

    @property
    def realistic_round_trip_bytes_per_s(self) -> float:
        return self.pcie_bw_idle_round_trip_bytes_per_s


def measure_pcie_bandwidth(size_mb: int = 1024, iterations: int = 5) -> float:
    """One-way HBM -> pinned-CPU memcpy throughput, in bytes/sec.

    Source (HBM) and destination (pinned CPU) tensors are allocated
    once before timing starts. This isolates raw PCIe transfer cost
    from allocator overhead.
    """
    size_bytes = size_mb * 1024 * 1024
    n = size_bytes // 2  # fp16 = 2 bytes per element
    src = torch.empty(n, dtype=torch.float16, device='cuda')
    dst = torch.empty(n, dtype=torch.float16, pin_memory=True)
    # Warmup so the first iteration doesn't include any lazy init cost.
    dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iterations):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    total_bytes = size_bytes * iterations
    return total_bytes / elapsed


def measure_realistic_round_trip(size_mb: int = 64, iterations: int = 5) -> float:
    """Effective throughput for a full evict+free+alloc+prefetch cycle.

    Each iteration:
      1. alloc fresh HBM tensor (paid: cudaMalloc / segment alloc)
      2. copy HBM -> pinned CPU                  (evict leg)
      3. free the HBM tensor                     (paid: caching alloc bookkeeping)
      4. alloc fresh HBM tensor                  (paid again)
      5. copy pinned CPU -> HBM                  (prefetch leg)
      6. free

    The reported bytes/sec is total_bytes_moved / total_time, where
    total_bytes_moved counts both legs (so the metric is directly
    comparable to a one-direction bandwidth). The delta vs
    measure_pcie_bandwidth is the allocator/segment overhead we
    actually pay in the live path.
    """
    size_bytes = size_mb * 1024 * 1024
    n = size_bytes // 2
    host = torch.empty(n, dtype=torch.float16, pin_memory=True)
    # Warmup: allocator path needs at least one fresh alloc before timing.
    warm = torch.empty(n, dtype=torch.float16, device='cuda')
    host.copy_(warm, non_blocking=True)
    del warm
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iterations):
        gpu = torch.empty(n, dtype=torch.float16, device='cuda')  # alloc
        host.copy_(gpu, non_blocking=True)                        # HBM -> CPU
        del gpu                                                   # free
        gpu2 = torch.empty(n, dtype=torch.float16, device='cuda') # alloc fresh
        gpu2.copy_(host, non_blocking=True)                       # CPU -> HBM
        del gpu2
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    total_bytes = size_bytes * 2 * iterations  # two legs per iteration
    return total_bytes / elapsed


def measure_gpu_flops(matrix_dim: int = 4096, iterations: int = 5) -> float:
    """Square fp16 matmul throughput, in FLOPs/sec.

    Each matmul is 2 * matrix_dim^3 FLOPs (multiply-add pairs counted
    as two ops, matching the standard convention). fp16 is the right
    compute dtype because LLaMA training is mixed-precision fp16.
    """
    a = torch.randn(matrix_dim, matrix_dim, dtype=torch.float16, device='cuda')
    b = torch.randn(matrix_dim, matrix_dim, dtype=torch.float16, device='cuda')
    # Warmup: first matmul often triggers cuBLAS algo selection.
    _ = a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = a @ b
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    flops_per_iter = 2 * matrix_dim ** 3
    return (flops_per_iter * iterations) / elapsed


# ----------------------------------------------------------------------
# NCCL-aware PCIe benchmark.
#
# All ranks spawn a background thread issuing back-to-back all_gather calls;
# meanwhile rank 0 (and only rank 0) times its own D2H+H2D on a private stream.
# Other ranks block on the barrier so they don't race ahead.
#
# Skipped (returns 0.0) when:
#   - dist is not initialized   (single-rank smoke test)
#   - world_size < 2            (need at least 2 ranks for all_gather)
# In those cases the engine falls back to idle bandwidth in the
# overlap-window math, which over-estimates available headroom but
# keeps the code path runnable.
# ----------------------------------------------------------------------

def _nccl_saturator_loop(nccl_bytes: int, n_iters: int, device: torch.device,
                         world_size: int) -> None:
    """Background-thread NCCL saturator. Local to this module so the
    threading lifecycle stays inside calibration.py."""
    torch.cuda.set_device(device)
    src = torch.zeros(nccl_bytes // 4, dtype=torch.float32, device=device)
    outs = [torch.zeros_like(src) for _ in range(world_size)]
    for _ in range(n_iters):
        dist.all_gather(outs, src)


def measure_pcie_bw_under_nccl(
    size_mb: int = 64,
    iterations: int = 10,
    nccl_tensor_mb: int = 256,
    nccl_n_iters: int = 100,
    settle_seconds: float = 0.2,
) -> float:
    """One-way HBM -> pinned-CPU bandwidth while NCCL all-gathers run.

    Returns bytes/sec measured on the same private PCIe stream as
    measure_pcie_bandwidth(). Returns 0.0 if dist is not initialized
    or world_size < 2.
    """
    if not dist.is_available() or not dist.is_initialized():
        return 0.0
    world_size = dist.get_world_size()
    if world_size < 2:
        return 0.0

    rank = dist.get_rank()
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    # Spawn saturator on every rank — the collective is matched call-for-
    # call across ranks. Matching n_iters lets all ranks exit together.
    sat = threading.Thread(
        target=_nccl_saturator_loop,
        args=(nccl_tensor_mb * 1024 * 1024, nccl_n_iters, device, world_size),
        daemon=False,
    )
    sat.start()
    # Let the saturator queue some work before we begin timing PCIe.
    time.sleep(settle_seconds)

    measured_bw = 0.0
    if rank == 0:
        size_bytes = size_mb * 1024 * 1024
        n = size_bytes // 2
        src = torch.empty(n, dtype=torch.float16, device=device)
        dst = torch.empty(n, dtype=torch.float16, pin_memory=True)
        stream = torch.cuda.Stream(device=device)
        # CUDA-event timing so we measure only the PCIe stream's work.
        # (Wall-clock would include time spent waiting on the NCCL stream.)
        per_iter_ms: list[float] = []
        with torch.cuda.stream(stream):
            # Warmup
            dst.copy_(src, non_blocking=True)
        stream.synchronize()
        for _ in range(iterations):
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream):
                start_ev.record(stream)
                dst.copy_(src, non_blocking=True)
                end_ev.record(stream)
            end_ev.synchronize()
            per_iter_ms.append(start_ev.elapsed_time(end_ev))
        avg_ms = sum(per_iter_ms) / len(per_iter_ms)
        measured_bw = size_bytes / (avg_ms / 1e3) if avg_ms > 0 else 0.0

    # Drain the background saturator on every rank.
    sat.join()
    if dist.is_initialized():
        dist.barrier()
    torch.cuda.synchronize()

    # Broadcast rank-0's measurement to all ranks so every rank's
    # CalibrationResult carries the same number (the engine uses it
    # in identical decisions across ranks).
    bw_tensor = torch.tensor([measured_bw], dtype=torch.float64, device=device)
    dist.broadcast(bw_tensor, src=0)
    return float(bw_tensor.item())


def measure_nccl_allgather_bandwidth(
    size_mb: int = 64,
    iterations: int = 10,
    warmup: int = 3,
) -> float:
    """Effective bytes/sec of dist.all_gather, measured per-rank.

    The "bytes/sec" reported here is the per-rank input bytes divided
    by the average elapsed time. This is the number the engine's
    ``_exposed_reallgather_ms`` consumes: bytes_to_gather / bw * 1e3
    gives the wall-clock cost of re-materializing one layer's params.

    Returns 0.0 if dist is not initialized or world_size < 2.
    """
    if not dist.is_available() or not dist.is_initialized():
        return 0.0
    world_size = dist.get_world_size()
    if world_size < 2:
        return 0.0
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    size_bytes = size_mb * 1024 * 1024
    n = size_bytes // 4  # float32
    src = torch.zeros(n, dtype=torch.float32, device=device)
    outs = [torch.zeros_like(src) for _ in range(world_size)]

    for _ in range(warmup):
        dist.all_gather(outs, src)
    torch.cuda.synchronize()
    dist.barrier()

    t0 = time.perf_counter()
    for _ in range(iterations):
        dist.all_gather(outs, src)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # Per-rank "bytes gathered" = input size (the engine multiplies by
    # whatever the W-aware coordinator hands it).
    bw = (size_bytes * iterations) / elapsed if elapsed > 0 else 0.0

    # Broadcast rank-0's number so every rank holds the same value.
    bw_tensor = torch.tensor([bw], dtype=torch.float64, device=device)
    if dist.get_rank() != 0:
        bw_tensor.fill_(0.0)
    dist.broadcast(bw_tensor, src=0)
    return float(bw_tensor.item())


def _check_sanity(result: CalibrationResult) -> dict:
    """Compare each field against _SANITY_BOUNDS; return {field: bool below}."""
    flags: dict = {}
    for field, lower in _SANITY_BOUNDS.items():
        v = getattr(result, field, 0.0)
        # 0.0 is the "not measured" sentinel; skip flagging it.
        flags[field] = bool(0.0 < v < lower)
    return flags


def run_calibration(verbose: bool = True) -> CalibrationResult:
    """Run all benchmarks; print a human-readable summary.

    Called once at engine init. Verbose prints land in the SLURM log
    so the numbers can be sanity-checked before training proceeds.

    Sanity bounds (V100 / PCIe-3 / Ethernet NCCL, no GDR):
      - PCIe idle one-way        >= 8 GB/s
      - PCIe idle round-trip     >= 4 GB/s
      - PCIe under-NCCL one-way  >= 1 GB/s
      - GPU fp16 FLOPs           >= 5 TFLOPs
      - NCCL allgather           >= 0.2 GB/s
    """
    if verbose:
        print("[Calibration] START")
    pcie_idle = measure_pcie_bandwidth()
    if verbose:
        print(f"[Calibration]   PCIe idle one-way:     {pcie_idle / 1e9:6.2f} GB/s")
    rrt = measure_realistic_round_trip()
    if verbose:
        print(f"[Calibration]   PCIe idle round-trip:  {rrt / 1e9:6.2f} GB/s")
    flops = measure_gpu_flops()
    if verbose:
        print(f"[Calibration]   GPU fp16 matmul:       {flops / 1e12:6.2f} TFLOPs")

    # Under-NCCL PCIe + NCCL allgather. Both gracefully return 0.0
    # outside a distributed run.
    pcie_nccl = measure_pcie_bw_under_nccl()
    if verbose:
        if pcie_nccl > 0:
            ratio = pcie_nccl / pcie_idle if pcie_idle > 0 else 0.0
            print(f"[Calibration]   PCIe under NCCL:       "
                  f"{pcie_nccl / 1e9:6.2f} GB/s  "
                  f"({(1 - ratio) * 100:.0f}% degradation)")
        else:
            print(f"[Calibration]   PCIe under NCCL:       skipped (no dist)")
    nccl_ag = measure_nccl_allgather_bandwidth()
    if verbose:
        if nccl_ag > 0:
            print(f"[Calibration]   NCCL all_gather:       {nccl_ag / 1e9:6.2f} GB/s")
        else:
            print(f"[Calibration]   NCCL all_gather:       skipped (no dist)")

    result = CalibrationResult(
        pcie_bw_idle_one_way_bytes_per_s=pcie_idle,
        pcie_bw_idle_round_trip_bytes_per_s=rrt,
        gpu_flops_per_s=flops,
        pcie_bw_under_nccl_one_way_bytes_per_s=pcie_nccl,
        nccl_allgather_bytes_per_s=nccl_ag,
    )
    result.sanity_flags = _check_sanity(result)
    if verbose:
        flagged = [k for k, v in result.sanity_flags.items() if v]
        if flagged:
            print(f"[Calibration] WARNING: below sanity bound: {flagged}")
        print(f"[Calibration] DONE")
    return result
