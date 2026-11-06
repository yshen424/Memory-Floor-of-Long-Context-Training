"""HBMBudget — logical + physical HBM accounting."""

import torch

from .tensor_registry import TensorRegistry


# ---------------------------------------------------------------------------
# HBMBudget
# ---------------------------------------------------------------------------

class HBMBudget:
    """Tracks logical (registry-side) and physical (torch.cuda) HBM use.

    Logical and physical diverge because PyTorch's caching allocator
    holds segments past the lifetime of individual tensors, so even
    after the engine "frees" a tensor (tier flip + unregister), the
    GPU may still be holding the underlying segment. Both signals
    matter:

      * logical_used  — what the engine's bookkeeping believes. Drives
                        scoring decisions because it's the quantity
                        the engine itself controls.
      * physical_*    — what torch.cuda actually sees. Catches real OOM
                        risk and fragmentation pressure that logical
                        accounting can miss.

    Pressure decisions in the engine compare both: if either signal
    exceeds the budget for the next op, we evict.
    """

    def __init__(self, total_bytes: int, registry: TensorRegistry) -> None:
        self._total_bytes = total_bytes
        self._registry = registry

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def logical_used(self) -> int:
        """Bytes the registry currently attributes to tier 'hbm'."""
        return self._registry.total_bytes_in_tier('hbm')

    def physical_allocated(self) -> int:
        """torch.cuda.memory_allocated() — sum of live tensor bytes.

        Returns 0 when CUDA is unavailable so CPU-only smoke tests can
        exercise the engine without crashing. Production rank-0 will
        always have CUDA.
        """
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.memory_allocated()

    def physical_reserved(self) -> int:
        """torch.cuda.memory_reserved() — bytes held in cached segments.

        Same CUDA-unavailable guard as physical_allocated().
        """
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.memory_reserved()

    def headroom(self) -> int:
        """Logical headroom = budget - logical_used (can go negative)."""
        return self._total_bytes - self.logical_used()

    def physical_headroom(self) -> int:
        """Physical headroom = budget - physical_reserved (can go negative)."""
        return self._total_bytes - self.physical_reserved()

    def pressure_signal(self, next_op_need_bytes: int) -> tuple[bool, bool]:
        """Return (logical_pressure, physical_pressure) for the next op.

        Each is True iff (current_use + next_op_need) > total_bytes for
        that lens. The engine evicts when EITHER returns True.
        """
        logical_pressure = (self.logical_used() + next_op_need_bytes) > self._total_bytes
        physical_pressure = (self.physical_reserved() + next_op_need_bytes) > self._total_bytes
        return logical_pressure, physical_pressure
