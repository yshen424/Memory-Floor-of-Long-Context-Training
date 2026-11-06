import threading
import time
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.distributed as dist

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
    pcie_bw_under_nccl_one_way_bytes_per_s: float = 0.0
    nccl_allgather_bytes_per_s: float = 0.0
    sanity_flags: dict = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @property
    def pcie_bw_bytes_per_s(self) -> float:
        return self.pcie_bw_idle_one_way_bytes_per_s

    @property
    def realistic_round_trip_bytes_per_s(self) -> float:
        return self.pcie_bw_idle_round_trip_bytes_per_s

def measure_pcie_bandwidth(size_mb: int = 1024, iterations: int = 5) -> float:
    size_bytes = size_mb * 1024 * 1024
    n = size_bytes // 2
    src = torch.empty(n, dtype=torch.float16, device='cuda')
    dst = torch.empty(n, dtype=torch.float16, pin_memory=True)
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
    size_bytes = size_mb * 1024 * 1024
    n = size_bytes // 2
    host = torch.empty(n, dtype=torch.float16, pin_memory=True)
    warm = torch.empty(n, dtype=torch.float16, device='cuda')
    host.copy_(warm, non_blocking=True)
    del warm
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iterations):
        gpu = torch.empty(n, dtype=torch.float16, device='cuda')
        host.copy_(gpu, non_blocking=True)
        del gpu
        gpu2 = torch.empty(n, dtype=torch.float16, device='cuda')
        gpu2.copy_(host, non_blocking=True)
        del gpu2
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    total_bytes = size_bytes * 2 * iterations
    return total_bytes / elapsed

def measure_gpu_flops(matrix_dim: int = 4096, iterations: int = 5) -> float:
    a = torch.randn(matrix_dim, matrix_dim, dtype=torch.float16, device='cuda')
    b = torch.randn(matrix_dim, matrix_dim, dtype=torch.float16, device='cuda')
    _ = a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = a @ b
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    flops_per_iter = 2 * matrix_dim ** 3
    return (flops_per_iter * iterations) / elapsed

def _nccl_saturator_loop(nccl_bytes: int, n_iters: int, device: torch.device,
                         world_size: int) -> None:
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
    if not dist.is_available() or not dist.is_initialized():
        return 0.0
    world_size = dist.get_world_size()
    if world_size < 2:
        return 0.0

    rank = dist.get_rank()
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    sat = threading.Thread(
        target=_nccl_saturator_loop,
        args=(nccl_tensor_mb * 1024 * 1024, nccl_n_iters, device, world_size),
        daemon=False,
    )
    sat.start()
    time.sleep(settle_seconds)

    measured_bw = 0.0
    if rank == 0:
        size_bytes = size_mb * 1024 * 1024
        n = size_bytes // 2
        src = torch.empty(n, dtype=torch.float16, device=device)
        dst = torch.empty(n, dtype=torch.float16, pin_memory=True)
        stream = torch.cuda.Stream(device=device)
        per_iter_ms: list[float] = []
        with torch.cuda.stream(stream):
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

    sat.join()
    if dist.is_initialized():
        dist.barrier()
    torch.cuda.synchronize()

    bw_tensor = torch.tensor([measured_bw], dtype=torch.float64, device=device)
    dist.broadcast(bw_tensor, src=0)
    return float(bw_tensor.item())

def measure_nccl_allgather_bandwidth(
    size_mb: int = 64,
    iterations: int = 10,
    warmup: int = 3,
) -> float:
    if not dist.is_available() or not dist.is_initialized():
        return 0.0
    world_size = dist.get_world_size()
    if world_size < 2:
        return 0.0
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    size_bytes = size_mb * 1024 * 1024
    n = size_bytes // 4
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

    bw = (size_bytes * iterations) / elapsed if elapsed > 0 else 0.0

    bw_tensor = torch.tensor([bw], dtype=torch.float64, device=device)
    if dist.get_rank() != 0:
        bw_tensor.fill_(0.0)
    dist.broadcast(bw_tensor, src=0)
    return float(bw_tensor.item())

def _check_sanity(result: CalibrationResult) -> dict:
    flags: dict = {}
    for field, lower in _SANITY_BOUNDS.items():
        v = getattr(result, field, 0.0)
        flags[field] = bool(0.0 < v < lower)
    return flags

def run_calibration(verbose: bool = True) -> CalibrationResult:
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
