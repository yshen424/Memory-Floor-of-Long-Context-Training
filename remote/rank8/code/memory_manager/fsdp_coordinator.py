from typing import Any

from .tensor_registry import TensorRegistry

def _extract_layer_idx(name: str) -> int:
    parts = name.split('.')
    for i, p in enumerate(parts):
        if p == 'layers' and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return -1
    return -1

class NoOpCoordinator:

    def __init__(self, registry: TensorRegistry, world_size: int = 1,
                 verbose: bool = True) -> None:
        self._registry = registry
        self._world_size = max(int(world_size), 1)
        if verbose:
            print(f"[FSDPCoordinator-NoOp] init DONE: world_size={self._world_size}")

    def estimate_allgather_bytes(self, layer_idx: int) -> int:
        shard_total = sum(
            r.size_bytes for _, r in self._registry.by_category('parameter')
            if r.layer_idx == layer_idx
        )
        if self._world_size <= 1:
            return 0
        return shard_total * (self._world_size - 1)

    def params_resident(self, layer_idx: int) -> bool:
        return False

    def on_pre_forward(self, layer_idx: int) -> None:
        pass

    def on_post_forward(self, layer_idx: int) -> None:
        pass

    def on_pre_backward(self, layer_idx: int) -> None:
        pass

    def on_post_backward(self, layer_idx: int) -> None:
        pass

class FSDPCoordinator:

    _VALID_STATES = ('sharded', 'materializing', 'full')
    _DIAG_TRANSITIONS_CAP = 256

    def __init__(
        self,
        fsdp_model: Any,
        registry: TensorRegistry,
        world_size: int = 1,
        verbose: bool = True,
    ) -> None:
        self._registry = registry
        self._world_size = max(int(world_size), 1)
        self._verbose = verbose
        self._layer_to_fsdp_module: dict[int, Any] = {}
        self._layer_state: dict[int, str] = {}
        self._diag_transitions_logged: int = 0
        self._build_layer_map(fsdp_model)
        if verbose:
            print(
                f"[FSDPCoordinator] init DONE: world_size={self._world_size}, "
                f"layers_tracked={len(self._layer_state)}"
            )

    def _build_layer_map(self, fsdp_model: Any) -> None:
        layer_count = 0
        for name, module in fsdp_model.named_modules():
            cls_name = module.__class__.__name__
            if cls_name != 'LlamaDecoderLayer':
                continue
            layer_idx = _extract_layer_idx(name)
            if layer_idx < 0:
                layer_idx = layer_count
            layer_count += 1
            self._layer_to_fsdp_module[layer_idx] = module
            self._layer_state[layer_idx] = 'sharded'

    def estimate_allgather_bytes(self, layer_idx: int) -> int:
        shard_total = sum(
            r.size_bytes for _, r in self._registry.by_category('parameter')
            if r.layer_idx == layer_idx
        )
        if self._world_size <= 1:
            return 0
        return shard_total * (self._world_size - 1)

    def params_resident(self, layer_idx: int) -> bool:
        return self._layer_state.get(layer_idx) == 'full'

    def _log_transition(self, layer_idx: int, old: str, new: str,
                        hook: str, note: str = '') -> None:
        if not self._verbose:
            return
        if self._diag_transitions_logged >= self._DIAG_TRANSITIONS_CAP:
            return
        suffix = f' ({note})' if note else ''
        print(
            f"[FSDPCoordinator] L={layer_idx} {hook}: "
            f"{old} -> {new}{suffix}"
        )
        self._diag_transitions_logged += 1
        if self._diag_transitions_logged == self._DIAG_TRANSITIONS_CAP:
            print(
                f"[FSDPCoordinator] transition log cap reached "
                f"({self._DIAG_TRANSITIONS_CAP}); silencing further "
                "transition prints."
            )

    def on_pre_forward(self, layer_idx: int) -> None:
        if layer_idx not in self._layer_state:
            return
        old = self._layer_state[layer_idx]
        self._layer_state[layer_idx] = 'full'
        self._log_transition(layer_idx, old, 'full', 'on_pre_forward')

    def on_post_forward(self, layer_idx: int) -> None:
        if layer_idx not in self._layer_state:
            return
        old = self._layer_state[layer_idx]
        self._layer_state[layer_idx] = 'sharded'
        self._log_transition(layer_idx, old, 'sharded', 'on_post_forward')

    def on_pre_backward(self, layer_idx: int) -> None:
        if layer_idx not in self._layer_state:
            return
        old = self._layer_state[layer_idx]
        self._layer_state[layer_idx] = 'full'
        self._log_transition(layer_idx, old, 'full', 'on_pre_backward')

    def on_post_backward(self, layer_idx: int) -> None:
        if layer_idx not in self._layer_state:
            return
        old = self._layer_state[layer_idx]
        self._layer_state[layer_idx] = 'sharded'
        self._log_transition(layer_idx, old, 'sharded', 'on_post_backward')

    def layer_state(self, layer_idx: int) -> str:
        return self._layer_state.get(layer_idx, 'sharded')
