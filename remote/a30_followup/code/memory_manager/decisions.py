from dataclasses import asdict, dataclass
from typing import Optional

@dataclass
class PlacementDecisionReason:
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
        return self.note or '<unset>'

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class PlacementDecision:
    step_idx: int
    layer_idx: int
    tensor_id: str
    action: str
    estimated_cost_ms: float
    reason: PlacementDecisionReason
    stall_breakdown: Optional[dict] = None
