"""memory_manager — tiered memory management for FSDP LLaMA training.

Package contents:

    TensorRecord, TensorRegistry              (tensor_registry.py)
    AccessEvent, AccessSchedule, AccessPredictor (access_predictor.py)
    HBMBudget                                 (hbm_budget.py)
    PlacementDecision, PlacementDecisionReason (decisions.py)
    InflightOp, MemoryStreamManager           (stream_manager.py)
    PlacementEngine                           (placement_engine.py)
    PinnedBuffer, PinnedBufferPool            (pinned_pool.py)
    CalibrationResult, run_calibration        (calibration.py)
    NoOpCoordinator, FSDPCoordinator          (fsdp_coordinator.py)

Integration wiring lives in fsdp_profiler.py / train_fsdp_profile.py.
"""

from .tensor_registry import TensorRecord, TensorRegistry
from .access_predictor import AccessEvent, AccessSchedule, AccessPredictor
from .placement_engine import (
    HBMBudget,
    PlacementDecision,
    PlacementDecisionReason,
    InflightOp,
    MemoryStreamManager,
    PlacementEngine,
)
from .pinned_pool import PinnedBuffer, PinnedBufferPool
from .calibration import CalibrationResult, run_calibration
from .fsdp_coordinator import NoOpCoordinator, FSDPCoordinator

__all__ = [
    "TensorRecord",
    "TensorRegistry",
    "AccessEvent",
    "AccessSchedule",
    "AccessPredictor",
    "HBMBudget",
    "PlacementDecision",
    "PlacementDecisionReason",
    "InflightOp",
    "MemoryStreamManager",
    "PlacementEngine",
    "PinnedBuffer",
    "PinnedBufferPool",
    "CalibrationResult",
    "run_calibration",
    "NoOpCoordinator",
    "FSDPCoordinator",
]
