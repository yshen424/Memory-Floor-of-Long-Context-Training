"""Placement decision data classes.

Pure data containers shared by the placement engine, the profiler's
JSON serializer, and tests. No torch dependency; safe to import from
CPU-only contexts.
"""

from dataclasses import asdict, dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# PlacementDecision
# ---------------------------------------------------------------------------

# Structured reason payload. Every decision carries enough numeric detail
# for the diagnostic figure (action vs layer_idx, colored by exposed cost
# difference). Non-cost decisions (retain / no_op / stall / prefetch) only
# fill the ``note`` field; the four cost numbers stay at their 0.0 defaults
# and the JSON serializer drops zero rows by convention.
@dataclass
class PlacementDecisionReason:
    """Structured payload for PlacementDecision.reason.

    Field semantics:
      transfer_ms             — raw PCIe one-way cost at idle bw.
      overlap_ms              — ms-equivalent of available window budget.
      exposed_evict_ms        — max(0, transfer_ms - overlap_ms).
      exposed_recompute_ms    — recompute_ms + reallgather_ms.
      recompute_ms            — raw SM-bound recompute cost.
      reallgather_ms          — NCCL all-gather cost charged to recompute
                                iff coordinator.params_resident(L) is False.
      queued_bytes_in_window  — bytes of prior mem-stream ops that share
                                this window's PCIe budget, including bytes
                                DEBITED by earlier decisions in the same
                                step, not just ops instantaneously in flight.
      window_capacity_bytes   — gross PCIe bytes the window can carry
                                before any debit.
      window_debited_bytes    — of that capacity, bytes already claimed by
                                earlier committed evictions whose windows
                                overlap this one. The gap between these two
                                fields is what concurrent decisions share.
      cpu_dram_full           — pinned pool had no free buffer for this
                                tensor (deterministic recompute trigger).
      projected_peak_bytes    — planner's projected HBM peak over
                                [current_step, L_bwd_event]. 0 on paths
                                that never projected.
      projected_peak_event_idx— step_idx where the projected peak occurs.
      anchor_live_bytes       — the allocator's live footprint the
                                projection was anchored on. A projection is
                                this plus the schedule's modelled delta, and
                                the delta is reproducible offline, so this is
                                the only half of the gate's arithmetic that a
                                run has to report. 0 on paths that never
                                projected.
      bundle_contribution_bytes
                              — bytes this bundle adds to the
                                projected peak. (peak_with − peak_without).
      note                    — freeform label for the diagnostic figure
                                ('late_prefetch_no_overlap',
                                 'pcie_saturated_in_window',
                                 'pcie_under_nccl_window',
                                 'no_pressure', 'free_after_bwd',
                                 'prefetch_incomplete', ...).
    """
    transfer_ms: float = 0.0
    overlap_ms: float = 0.0
    exposed_evict_ms: float = 0.0
    exposed_recompute_ms: float = 0.0
    recompute_ms: float = 0.0
    reallgather_ms: float = 0.0
    queued_bytes_in_window: int = 0
    window_capacity_bytes: int = 0
    window_debited_bytes: int = 0
    cpu_dram_full: bool = False
    projected_peak_bytes: int = 0
    projected_peak_event_idx: int = 0
    anchor_live_bytes: int = 0
    bundle_contribution_bytes: int = 0
    note: str = ''

    def __str__(self) -> str:
        # Human-readable fallback used by print(decision.reason);
        # JSON serialization in fsdp_profiler.save() goes through to_dict().
        return self.note or '<unset>'

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PlacementDecision:
    """One record per choice the engine makes at a decision point.

    ``action`` is one of:
        'evict_to_cpu'      — chose to offload this tensor
        'recompute'         — chose to recompute instead of offload
        'prefetch_to_hbm'   — chose to bring back from CPU
        'retain'            — explicit "keep in HBM" (no pressure)
        'stall'             — backward arrived before prefetch finished
        'no_op'             — nothing required at this point

    ``stall_breakdown`` is populated only for 'stall' actions:
        {'pcie_ms': float, 'allocator_ms': float}

    ``reason`` is a PlacementDecisionReason; a decision with only a label
    wraps it as PlacementDecisionReason(note=...).
    """
    step_idx: int
    layer_idx: int
    tensor_id: str
    action: str
    estimated_cost_ms: float
    reason: PlacementDecisionReason
    stall_breakdown: Optional[dict] = None
