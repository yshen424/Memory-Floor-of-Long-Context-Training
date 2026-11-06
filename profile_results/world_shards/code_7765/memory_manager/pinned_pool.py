import os
from dataclasses import dataclass

import torch

@dataclass
class PinnedBuffer:
    tensor: torch.Tensor
    capacity_bytes: int
    in_use: bool = False

HOST_PIN_MAX_SHARE = 0.60

def _check_host_headroom(total_bytes: int) -> None:
    try:
        with open('/proc/meminfo') as f:
            meminfo = {ln.split(':')[0]: int(ln.split()[1]) * 1024
                       for ln in f if ln.startswith('MemAvailable')}
        available = meminfo['MemAvailable']
    except (OSError, KeyError, ValueError, IndexError):
        return
    ranks_here = max(1, int(os.environ.get('LOCAL_WORLD_SIZE', '1')))
    limit = HOST_PIN_MAX_SHARE * available / ranks_here
    if total_bytes > limit:
        gib = 1024 ** 3
        raise RuntimeError(
            f"[PinnedPool] refusing to pin {total_bytes / gib:.2f} GiB: the "
            f"limit is {HOST_PIN_MAX_SHARE:.0%} of MemAvailable "
            f"({available / gib:.1f} GiB) over {ranks_here} rank(s) on this "
            f"node = {limit / gib:.1f} GiB. Pinned memory is unreclaimable, "
            f"and a pool this size hangs the node instead of failing. Size "
            f"the pool from what the arm can stage (_derive_pool_buckets).")

class PinnedBufferPool:

    def __init__(self,
                 bucket_sizes_bytes,
                 buffers_per_bucket: int = 4,
                 device: str = 'cpu') -> None:
        if device != 'cpu':
            raise ValueError(
                "PinnedBufferPool only supports device='cpu' "
                "(pinned memory is host memory).")

        if isinstance(bucket_sizes_bytes, dict):
            counts = {int(s): int(n) for s, n in bucket_sizes_bytes.items()}
        else:
            counts = {int(s): buffers_per_bucket
                      for s in set(bucket_sizes_bytes)}

        self._bucket_sizes: list[int] = sorted(counts)
        self._buffers_by_bucket: dict[int, int] = counts
        _uniq = set(counts.values())
        self._buffers_per_bucket = (
            _uniq.pop() if len(_uniq) == 1 else None)
        self._buckets: dict[int, list[PinnedBuffer]] = {}

        total_bytes = sum(s * counts[s] for s in self._bucket_sizes)
        _check_host_headroom(total_bytes)
        print(f"[PinnedPool] init START: {len(self._bucket_sizes)} buckets, "
              f"{sum(counts.values())} buffers, "
              f"{total_bytes / (1024 ** 3):.2f} GiB, pre-allocating...")
        for size in self._bucket_sizes:
            buffers: list[PinnedBuffer] = []
            for _ in range(counts[size]):
                t = torch.empty(size, dtype=torch.uint8, pin_memory=True)
                buffers.append(PinnedBuffer(tensor=t, capacity_bytes=size, in_use=False))
            self._buckets[size] = buffers
        print(f"[PinnedPool] init DONE: "
              f"{total_bytes / (1024 * 1024):.1f} MB pre-allocated")

        self._acquire_served: dict[int, dict[int, int]] = {}
        self._acquire_count: int = 0
        self._payload_bytes_total: int = 0
        self._charged_bytes_total: int = 0
        self._denied_by_size: dict[int, int] = {}
        self._in_use_count: dict[int, int] = {s: 0 for s in self._bucket_sizes}
        self._high_water: dict[int, int] = {s: 0 for s in self._bucket_sizes}
        self._denied_by_class: dict[tuple[int, str], int] = {}

    def acquire(self, nbytes: int) -> PinnedBuffer:
        found_any_fitting = False
        for size in self._bucket_sizes:
            if size < nbytes:
                continue
            found_any_fitting = True
            for buf in self._buckets[size]:
                if not buf.in_use:
                    buf.in_use = True
                    self._record_acquire(nbytes, size)
                    return buf
        if not found_any_fitting:
            raise RuntimeError(
                f"PinnedBufferPool.acquire({nbytes}): no bucket large enough. "
                f"Max bucket size: {self._bucket_sizes[-1]}")
        raise RuntimeError(
            f"PinnedBufferPool.acquire({nbytes}): every bucket >= {nbytes} "
            f"is fully in use. Buckets tried: "
            f"{[s for s in self._bucket_sizes if s >= nbytes]}.")

    def _record_acquire(self, nbytes: int, served_size: int) -> None:
        self._acquire_count += 1
        self._payload_bytes_total += nbytes
        self._charged_bytes_total += served_size
        by_served = self._acquire_served.setdefault(nbytes, {})
        by_served[served_size] = by_served.get(served_size, 0) + 1
        n = self._in_use_count[served_size] + 1
        self._in_use_count[served_size] = n
        if n > self._high_water[served_size]:
            self._high_water[served_size] = n

    def release(self, buf: PinnedBuffer) -> None:
        if buf.in_use:
            self._in_use_count[buf.capacity_bytes] -= 1
        buf.in_use = False

    def has_capacity(self, nbytes: int, kind: str = None) -> bool:
        for size in self._bucket_sizes:
            if size < nbytes:
                continue
            for buf in self._buckets[size]:
                if not buf.in_use:
                    return True
        self._denied_by_size[nbytes] = self._denied_by_size.get(nbytes, 0) + 1
        key = (nbytes, kind or 'unknown')
        self._denied_by_class[key] = self._denied_by_class.get(key, 0) + 1
        return False

    def has_capacity_for_sizes(self, sizes: list[int], kind: str = None) -> bool:
        if not sizes:
            return True
        free = {s: sum(1 for b in self._buckets[s] if not b.in_use)
                for s in self._bucket_sizes}
        for nbytes in sorted(sizes, reverse=True):
            for size in self._bucket_sizes:
                if size >= nbytes and free[size] > 0:
                    free[size] -= 1
                    break
            else:
                self._denied_by_size[nbytes] = (
                    self._denied_by_size.get(nbytes, 0) + 1)
                key = (nbytes, kind or 'unknown')
                self._denied_by_class[key] = self._denied_by_class.get(key, 0) + 1
                return False
        return True

    def stats(self) -> dict:
        out: dict = {}
        for size, buffers in self._buckets.items():
            in_use = sum(1 for b in buffers if b.in_use)
            out[size] = {
                'in_use': in_use,
                'total': len(buffers),
                'capacity_bytes': size,
            }
        return out

    def telemetry(self) -> dict:
        served = {
            str(req): {str(sz): n for sz, n in sorted(by.items())}
            for req, by in sorted(self._acquire_served.items())
        }
        provisioned = sum(s * len(b) for s, b in self._buckets.items())
        return {
            'bucket_sizes_bytes': list(self._bucket_sizes),
            'buffers_per_bucket': self._buffers_per_bucket,
            'buffers_by_bucket': {
                str(s): n for s, n in sorted(self._buffers_by_bucket.items())},
            'provisioned_bytes': provisioned,
            'acquires': self._acquire_count,
            'payload_bytes_total': self._payload_bytes_total,
            'charged_bytes_total': self._charged_bytes_total,
            'rounding_overhead': (
                round(self._charged_bytes_total / self._payload_bytes_total, 4)
                if self._payload_bytes_total else None),
            'acquire_served': served,
            'denials_by_requested_size': {
                str(k): v for k, v in sorted(self._denied_by_size.items())},
            'denials_by_class': {
                f'{n}|{k}': v
                for (n, k), v in sorted(self._denied_by_class.items())},
            'denials_total': sum(self._denied_by_size.values()),
            'high_water_in_use': {
                str(k): v for k, v in sorted(self._high_water.items())},
            'buffers_total': sum(len(b) for b in self._buckets.values()),
        }
