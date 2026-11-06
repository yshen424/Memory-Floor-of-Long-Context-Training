"""
Pinned Buffer Pool — pre-allocated host (pinned) memory for offload.

Bucket model
------------
A small set of power-of-2 size buckets, each holding a fixed number
of pinned tensors. ``acquire(nbytes)`` rounds the request up to the
smallest bucket that fits and returns the first free buffer in that
bucket; ``release(buf)`` flips ``in_use`` back to False.

Why fixed buckets rather than ad-hoc allocation:
  * Pinned memory allocation is slow (host-side pinning is a kernel
    op), so we want to pay it once at init.
  * Free-list reuse avoids fragmentation in host memory across long
    training runs.
  * O(1) acquire/release keeps the engine's per-decision overhead low.
"""

import os
from dataclasses import dataclass

import torch


@dataclass
class PinnedBuffer:
    """A single pre-allocated pinned host buffer.

    ``tensor`` is a 1-D uint8 view of the underlying pinned allocation,
    sized to the bucket's ``capacity_bytes`` (not the caller's
    requested nbytes — the bucket is the granularity).
    """
    tensor: torch.Tensor
    capacity_bytes: int
    in_use: bool = False


# TRAP: pinned pages cannot be swapped or reclaimed, so a pool near node DRAM
# does not fail cleanly -- it starves the kernel, the process goes unkillable,
# and Slurm drains the node ("Kill task failed") or loses it outright. Refuse
# before pinning anything. The share is of MemAvailable divided among this
# node's ranks.
HOST_PIN_MAX_SHARE = 0.60


def _check_host_headroom(total_bytes: int) -> None:
    try:
        with open('/proc/meminfo') as f:
            meminfo = {ln.split(':')[0]: int(ln.split()[1]) * 1024
                       for ln in f if ln.startswith('MemAvailable')}
        available = meminfo['MemAvailable']
    except (OSError, KeyError, ValueError, IndexError):
        return  # no /proc/meminfo: nothing to check against
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
    """Bucketed free-list of pre-allocated pinned host buffers."""

    def __init__(self,
                 bucket_sizes_bytes,
                 buffers_per_bucket: int = 4,
                 device: str = 'cpu') -> None:
        if device != 'cpu':
            # Pinned memory is host memory by definition. The 'device'
            # parameter exists in the signature for API symmetry but
            # any non-'cpu' value is a programmer error.
            raise ValueError(
                "PinnedBufferPool only supports device='cpu' "
                "(pinned memory is host memory).")

        # Accept either a plain list of sizes (uniform ``buffers_per_bucket``,
        # the form the CPU tests use) or a {size: count} mapping. Uniform
        # counts over-provision: giving a 256 MiB bucket the same 32 buffers
        # as a 1 MiB bucket spends 8 GiB on a class with no demand, while the
        # classes that do have demand fall through into it and get charged
        # 256 MiB for a 9 MiB payload. Counts come from per-layer save
        # multiplicity.
        if isinstance(bucket_sizes_bytes, dict):
            counts = {int(s): int(n) for s, n in bucket_sizes_bytes.items()}
        else:
            counts = {int(s): buffers_per_bucket
                      for s in set(bucket_sizes_bytes)}

        # Sorted ascending so acquire() does a forward linear scan and
        # returns the smallest-fitting bucket first.
        self._bucket_sizes: list[int] = sorted(counts)
        self._buffers_by_bucket: dict[int, int] = counts
        # None when counts are non-uniform; kept for telemetry back-compat.
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

        # Run-accumulated pool behavior. ``stats()`` reports only the
        # instantaneous snapshot, which cannot answer which size classes are
        # actually requested and which bucket ends up serving them; these
        # counters let provisioning be checked against measured demand.
        self._acquire_served: dict[int, dict[int, int]] = {}
        self._acquire_count: int = 0
        self._payload_bytes_total: int = 0
        self._charged_bytes_total: int = 0
        self._denied_by_size: dict[int, int] = {}
        self._in_use_count: dict[int, int] = {s: 0 for s in self._bucket_sizes}
        self._high_water: dict[int, int] = {s: 0 for s in self._bucket_sizes}
        # Denials keyed by size alone can leave a starving class
        # unidentifiable (a byte count need not factor into the config's
        # dimensions), so key by (nbytes, kind) too.
        self._denied_by_class: dict[tuple[int, str], int] = {}

    def acquire(self, nbytes: int) -> PinnedBuffer:
        """Return a free PinnedBuffer with ``capacity_bytes >= nbytes``.

        Walks buckets in ascending order starting from the smallest that
        fits, falling through to larger buckets when the smallest-fitting
        one is fully in use.

        Raises:
            RuntimeError if no bucket large enough exists, or if every
            eligible bucket is fully in use (matches ``has_capacity``
            returning False).
        """
        # Fall through to larger buckets when the smallest-fitting bucket is
        # full. has_capacity walks all eligible buckets, so acquire must too,
        # or a caller can pass the gate and then fail here.
        # The copy in submit_evict only writes ``nbytes`` into the buffer,
        # so a slightly-oversized bucket is safe — only marginal waste in
        # pinned-memory footprint per outstanding op.
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
        """Book one successful acquire. See __init__ for why."""
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
        """Mark ``buf`` as free for reuse. No-op if already free."""
        if buf.in_use:
            self._in_use_count[buf.capacity_bytes] -= 1
        buf.in_use = False

    # ------------------------------------------------------------------
    # Capacity gate for the engine. The placement engine's decision
    # branches on has_capacity *before* the cost compare so a full pool
    # deterministically forces recompute. Lookup-only: must not mutate
    # ``in_use`` flags.
    # ------------------------------------------------------------------
    def has_capacity(self, nbytes: int, kind: str = None) -> bool:
        """True iff some bucket ``>= nbytes`` has at least one free buffer.

        Returns False both when no bucket is large enough *and* when
        the smallest-fitting bucket is fully in use. The engine treats
        either case the same way: CPU DRAM target unavailable for this
        tensor, fall back to recompute.

        ``kind`` is telemetry only and never affects the answer.
        """
        for size in self._bucket_sizes:
            if size < nbytes:
                continue
            for buf in self._buckets[size]:
                if not buf.in_use:
                    return True
        # A denial here is what surfaces as 'cpu_dram_full' and
        # forces recompute. Counted by requested size so we can see *which*
        # class starves. Both gates call this (decision-time in
        # _decide_evict_vs_recompute, then pack-time in pack_fn), so one
        # denied tensor can book two counts — read this as denial events,
        # not distinct tensors.
        self._denied_by_size[nbytes] = self._denied_by_size.get(nbytes, 0) + 1
        key = (nbytes, kind or 'unknown')
        self._denied_by_class[key] = self._denied_by_class.get(key, 0) + 1
        return False

    # ------------------------------------------------------------------
    # pack_fn acquires one buffer PER TENSOR, so the decision-time gate for
    # a bundle must ask for one buffer per member, not ONE free buffer that
    # fits the whole bundle — a size no bucket serves, which would force
    # the bundle to recompute. This asks the question the physical path
    # actually asks.
    # ------------------------------------------------------------------
    def has_capacity_for_sizes(self, sizes: list[int], kind: str = None) -> bool:
        """True iff the pool can supply one buffer for *each* of ``sizes``.

        Lookup-only: must not mutate ``in_use``. Assigns largest request
        first to the smallest bucket that fits, which is the assignment
        that succeeds whenever any assignment does.

        ``kind`` is telemetry only and never affects the answer. A denial
        is booked against the member that could not be placed, so
        ``denials_by_class`` names a size that really exists on the wire.
        """
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
        """Per-bucket usage snapshot for logging / JSON output."""
        out: dict = {}
        for size, buffers in self._buckets.items():
            in_use = sum(1 for b in buffers if b.in_use)
            out[size] = {
                'in_use': in_use,
                'total': len(buffers),
                'capacity_bytes': size,
            }
        return out

    # ------------------------------------------------------------------
    # JSON-bound, so every key is a string: json.dump coerces int keys
    # silently.
    # ------------------------------------------------------------------
    def telemetry(self) -> dict:
        """Run-accumulated demand histogram and per-bucket pressure."""
        served = {
            str(req): {str(sz): n for sz, n in sorted(by.items())}
            for req, by in sorted(self._acquire_served.items())
        }
        provisioned = sum(s * len(b) for s, b in self._buckets.items())
        return {
            'bucket_sizes_bytes': list(self._bucket_sizes),
            # None when counts are non-uniform (the mapping form); read
            # 'buffers_by_bucket' instead.
            'buffers_per_bucket': self._buffers_per_bucket,
            'buffers_by_bucket': {
                str(s): n for s, n in sorted(self._buffers_by_bucket.items())},
            'provisioned_bytes': provisioned,
            'acquires': self._acquire_count,
            'payload_bytes_total': self._payload_bytes_total,
            'charged_bytes_total': self._charged_bytes_total,
            # >1.0 is the power-of-2 rounding tax: bytes locked up per byte
            # actually staged. Multiplies with the misallocation loss that
            # 'high_water' exposes.
            'rounding_overhead': (
                round(self._charged_bytes_total / self._payload_bytes_total, 4)
                if self._payload_bytes_total else None),
            'acquire_served': served,
            'denials_by_requested_size': {
                str(k): v for k, v in sorted(self._denied_by_size.items())},
            # '<nbytes>|<kind>' so a starving class can
            # be named rather than inferred from its byte count.
            'denials_by_class': {
                f'{n}|{k}': v
                for (n, k), v in sorted(self._denied_by_class.items())},
            'denials_total': sum(self._denied_by_size.values()),
            # Peak simultaneous occupancy per bucket, counting fallthrough.
            # A bucket at 0 never served anything; a bucket at its full
            # count was saturated. Cross-reference 'acquire_served' to tell
            # a natural fit from an oversized fallthrough.
            'high_water_in_use': {
                str(k): v for k, v in sorted(self._high_water.items())},
            'buffers_total': sum(len(b) for b in self._buckets.values()),
        }
