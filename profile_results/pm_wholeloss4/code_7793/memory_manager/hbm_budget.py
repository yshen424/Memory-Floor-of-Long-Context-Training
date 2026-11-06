import torch

from .tensor_registry import TensorRegistry

class HBMBudget:

    def __init__(self, total_bytes: int, registry: TensorRegistry) -> None:
        self._total_bytes = total_bytes
        self._registry = registry

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def logical_used(self) -> int:
        return self._registry.total_bytes_in_tier('hbm')

    def physical_allocated(self) -> int:
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.memory_allocated()

    def physical_reserved(self) -> int:
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.memory_reserved()

    def headroom(self) -> int:
        return self._total_bytes - self.logical_used()

    def physical_headroom(self) -> int:
        return self._total_bytes - self.physical_reserved()

    def pressure_signal(self, next_op_need_bytes: int) -> tuple[bool, bool]:
        logical_pressure = (self.logical_used() + next_op_need_bytes) > self._total_bytes
        physical_pressure = (self.physical_reserved() + next_op_need_bytes) > self._total_bytes
        return logical_pressure, physical_pressure
