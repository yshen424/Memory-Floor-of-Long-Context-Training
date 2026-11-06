"""
Tensor Registry — Component 1 of the tiered memory manager.

This module owns ``TensorRecord`` (the per-tensor metadata dataclass) and
``TensorRegistry`` (a thin wrapper around a {key: TensorRecord} dict).

Decision payloads (`PlacementDecisionReason`) live on `PlacementDecision`
in the engine, not on per-tensor records.
"""

from dataclasses import dataclass
from typing import Any, Iterator, Optional


@dataclass
class TensorRecord:
    """Lifecycle record for a single managed tensor."""
    name: str
    size_bytes: int
    shape: tuple
    dtype: str
    component: str          # 'attention_qkv', 'attention_scores', 'swiglu_gate', etc.
    layer_idx: int
    category: str           # 'parameter', 'activation', 'gradient', 'optimizer_state'
    group_id: Optional[str] = None   # allocation group (e.g., 'layer_5_swiglu')
    birth_time_ms: float = 0.0
    recompute_cost_flops: int = 0
    # One-shot snapshot of the schedule step at which this tensor will next
    # be read, populated once at init by AccessPredictor.populate_registry().
    # IMPORTANT: this field is for telemetry / debugging only. Runtime
    # decisions (eviction scoring, prefetch lookahead) must query the
    # AccessSchedule directly via next_read_after(...) — that's the
    # authoritative source as current_step advances.
    predicted_next_access_step: Optional[int] = None
    # The placement engine's per-tensor working state, written by
    # PlacementEngine.
    #
    # tier — where the tensor currently lives. Engine flips this each
    #   time it decides to evict / recompute / re-materialize. Values:
    #     'hbm'              resident on GPU (default at registration)
    #     'cpu_dram'         evicted to pinned host memory
    #     'recompute_marker' dropped; will be recomputed before next read
    #     'freed'            permanently released (post-backward end of life)
    #
    # in_flight — True iff an async memcpy for this tensor is pending on
    #   the memory stream. Set True at submit time, False when the event
    #   is reaped. The flag is maintained even when no real memcpy runs,
    #   so on_pre_backward's stall-detection logic is exercisable.
    #
    # checkpoint_enabled — True iff the engine chose recompute over
    #   offload for this activation. Distinct from tier='recompute_marker':
    #   this field records the *decision*, the tier records the *state*.
    tier: str = 'hbm'
    in_flight: bool = False
    checkpoint_enabled: bool = False
    # The live torch.Tensor handle the engine copies bytes out of (D2H evict)
    # and into (H2D prefetch). Populated by:
    #   * Sub-module forward hooks — for activation records.
    #   * Not used for parameter / optimizer-state records; FSDP owns
    #     those tensors and the FSDP coordinator reaches them via
    #     FSDP's API rather than the registry.
    # Typed as ``Any`` to avoid importing torch at this leaf module (the
    # registry stays import-light so it can be exercised in CPU-only unit
    # tests without dragging the full torch dependency).
    tensor: Optional[Any] = None


class TensorRegistry:
    """Source-of-truth map of key -> TensorRecord for the tiered memory manager.

    A thin wrapper around a dict with the operations the profiler, access
    predictor, placement engine and FSDP coordinator use.

    Semantics note: ``register`` overwrites on duplicate key.
    """

    def __init__(self) -> None:
        self._records: dict[str, TensorRecord] = {}

    def register(self, key: str, record: TensorRecord) -> None:
        """Insert or overwrite the record at ``key``."""
        self._records[key] = record

    def get(self, key: str) -> Optional[TensorRecord]:
        """Return the record at ``key``, or None if absent."""
        return self._records.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self._records

    def __len__(self) -> int:
        return len(self._records)

    def items(self) -> Iterator[tuple[str, TensorRecord]]:
        """Iterate (key, record) pairs — used for serialization and ad-hoc filtering."""
        return iter(self._records.items())

    def groups(self) -> dict[str, dict]:
        """Aggregate records by ``group_id`` into the shape the profiler's
        ``get_allocation_groups()`` serializer expects:

            {gid: {'tensors': [key, ...], 'total_bytes': int}}

        Records with ``group_id is None`` are omitted.
        """
        out: dict[str, dict] = {}
        for name, rec in self._records.items():
            if rec.group_id is None:
                continue
            bucket = out.setdefault(rec.group_id, {'tensors': [], 'total_bytes': 0})
            bucket['tensors'].append(name)
            bucket['total_bytes'] += rec.size_bytes
        return out

    # ------------------------------------------------------------------
    # These methods support the engine's tier-management workflow:
    # query records by current tier, sum bytes for pressure decisions,
    # and flip per-tensor flags as evict/prefetch ops are submitted.
    # unregister exists to release records of full-param tensors once
    # FSDP frees them. Freed activations are NOT unregistered — those
    # records stay so the JSON output can show post-run tier='freed'
    # state for inspection.
    #
    # Mutating methods (set_tier/set_in_flight/unregister) raise
    # KeyError on missing keys. Mutation on a missing key is a
    # programmer error — distinct from get(), which returns None to
    # let callers probe.
    # ------------------------------------------------------------------

    def set_tier(self, key: str, tier: str) -> None:
        """Update ``record.tier``. Raises KeyError if ``key`` is unknown."""
        self._records[key].tier = tier

    def set_in_flight(self, key: str, flag: bool) -> None:
        """Update ``record.in_flight``. Raises KeyError if ``key`` is unknown."""
        self._records[key].in_flight = flag

    def by_tier(self, tier: str) -> Iterator[tuple[str, TensorRecord]]:
        """Iterate (key, record) pairs whose ``record.tier == tier``."""
        return ((k, r) for k, r in self._records.items() if r.tier == tier)

    def total_bytes_in_tier(self, tier: str) -> int:
        """Sum ``size_bytes`` across all records currently in ``tier``."""
        return sum(r.size_bytes for r in self._records.values() if r.tier == tier)

    # Both the placement engine and the FSDPCoordinator walk param records (the
    # coordinator for shard-byte sums in estimate_allgather_bytes; the
    # engine for various param-side lookups). Using an explicit
    # by_category iterator keeps the filter in one place instead of
    # repeating ``if r.category == X`` at each call site.
    def by_category(self, category: str) -> Iterator[tuple[str, TensorRecord]]:
        """Iterate (key, record) pairs whose ``record.category == category``."""
        return ((k, r) for k, r in self._records.items() if r.category == category)

    def unregister(self, key: str) -> None:
        """Remove the record at ``key``. Raises KeyError if absent."""
        del self._records[key]
