from dataclasses import dataclass
from typing import Any, Iterator, Optional

@dataclass
class TensorRecord:
    name: str
    size_bytes: int
    shape: tuple
    dtype: str
    component: str
    layer_idx: int
    category: str
    group_id: Optional[str] = None
    birth_time_ms: float = 0.0
    recompute_cost_flops: int = 0
    predicted_next_access_step: Optional[int] = None
    tier: str = 'hbm'
    in_flight: bool = False
    checkpoint_enabled: bool = False
    tensor: Optional[Any] = None

class TensorRegistry:

    def __init__(self) -> None:
        self._records: dict[str, TensorRecord] = {}

    def register(self, key: str, record: TensorRecord) -> None:
        self._records[key] = record

    def get(self, key: str) -> Optional[TensorRecord]:
        return self._records.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self._records

    def __len__(self) -> int:
        return len(self._records)

    def items(self) -> Iterator[tuple[str, TensorRecord]]:
        return iter(self._records.items())

    def groups(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for name, rec in self._records.items():
            if rec.group_id is None:
                continue
            bucket = out.setdefault(rec.group_id, {'tensors': [], 'total_bytes': 0})
            bucket['tensors'].append(name)
            bucket['total_bytes'] += rec.size_bytes
        return out

    def set_tier(self, key: str, tier: str) -> None:
        self._records[key].tier = tier

    def set_in_flight(self, key: str, flag: bool) -> None:
        self._records[key].in_flight = flag

    def by_tier(self, tier: str) -> Iterator[tuple[str, TensorRecord]]:
        return ((k, r) for k, r in self._records.items() if r.tier == tier)

    def total_bytes_in_tier(self, tier: str) -> int:
        return sum(r.size_bytes for r in self._records.values() if r.tier == tier)

    def by_category(self, category: str) -> Iterator[tuple[str, TensorRecord]]:
        return ((k, r) for k, r in self._records.items() if r.category == category)

    def unregister(self, key: str) -> None:
        del self._records[key]
