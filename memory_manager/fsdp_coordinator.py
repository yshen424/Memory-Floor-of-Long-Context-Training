"""
FSDP Coordinator — Component 4 of the tiered memory manager.

Two classes live here:

  * ``NoOpCoordinator`` — CPU-friendly stub. Honors the engine's
    coordinator call surface but does nothing to FSDP itself. CPU
    smoke tests and unit tests instantiate the engine without a real
    FSDP-wrapped model, and it documents the minimum API surface a
    coordinator must expose.

  * ``FSDPCoordinator`` — real coordinator for cluster runs, and a pure
    observer: FSDP internals are untouched. Tracks a per-layer FSDP
    state machine driven by engine hook fire points; exposes
    ``params_resident(L)`` and ``estimate_allgather_bytes(L)`` so the
    engine's recompute cost model can reason about FSDP without
    intercepting it. ``params_resident(L)`` matters for the recompute
    side: when FSDP itself has L full at backward read time, recompute
    pays no re-gather.

The placement engine calls into the coordinator for:

  * ``estimate_allgather_bytes(layer_idx)`` — used in the engine's
    next-op HBM-need estimate and the recompute cost model's
    re-gather penalty. Implemented as the per-rank on-network volume
    of a ring all_gather: ``shard_total × (W - 1)``. Reduces to 0 at
    W=1; approaches ``(W-1)/W × P_L`` as W grows.

  * ``params_resident(layer_idx) -> bool`` — the engine queries this
    from ``_exposed_reallgather_ms``; recompute pays no re-gather
    penalty when the coordinator reports params resident. NoOp returns
    False unconditionally. FSDPCoordinator returns based on its
    per-layer state machine.

  * ``on_pre_forward(L)`` / ``on_post_forward(L)`` /
    ``on_pre_backward(L)`` / ``on_post_backward(L)`` — the engine
    forwards these callbacks; the coordinator advances its per-layer
    state machine accordingly. NoOp's versions are pass-through so the
    engine can call unconditionally.
"""

from typing import Any

from .tensor_registry import TensorRegistry


def _extract_layer_idx(name: str) -> int:
    """Pull layer index out of a dotted module name.

    Mirrors the parsing in fsdp_profiler.FSDPMemoryProfiler — looks
    for a '.layers.N.' segment and returns N as int. Returns -1 if
    the name doesn't carry one (e.g., embedding / lm_head modules).
    """
    parts = name.split('.')
    for i, p in enumerate(parts):
        if p == 'layers' and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return -1
    return -1


class NoOpCoordinator:
    """Placeholder coordinator for CPU-only tests.

    Honors the call surface the placement engine expects, but does
    nothing to FSDP itself.
    """

    def __init__(self, registry: TensorRegistry, world_size: int = 1,
                 verbose: bool = True) -> None:
        self._registry = registry
        self._world_size = max(int(world_size), 1)
        if verbose:
            print(f"[FSDPCoordinator-NoOp] init DONE: world_size={self._world_size}")

    def estimate_allgather_bytes(self, layer_idx: int) -> int:
        """Per-rank bytes-on-network for layer L's all-gather.

        The per-rank on-network volume of a ring all_gather (not the
        post-gather HBM residency ``shard_total × W``):

            shard_total × (W - 1)

        Returns 0 if no parameter records exist for the layer (normal
        state on rank 0 before FSDP has wrapped).
        """
        shard_total = sum(
            r.size_bytes for _, r in self._registry.by_category('parameter')
            if r.layer_idx == layer_idx
        )
        if self._world_size <= 1:
            return 0
        return shard_total * (self._world_size - 1)

    def params_resident(self, layer_idx: int) -> bool:
        """Always False — NoOp has no FSDP state to track.

        Conservative answer: biases the cost model toward evict-to-CPU
        during dry-run / CPU tests.
        The real FSDPCoordinator returns based on actual per-layer
        state.
        """
        return False

    # The engine forwards hook callbacks to the coordinator
    # unconditionally. NoOp's versions are
    # pass-throughs so callers don't need to special-case which
    # coordinator they hold.

    def on_pre_forward(self, layer_idx: int) -> None:
        pass

    def on_post_forward(self, layer_idx: int) -> None:
        pass

    def on_pre_backward(self, layer_idx: int) -> None:
        pass

    def on_post_backward(self, layer_idx: int) -> None:
        pass


class FSDPCoordinator:
    """Pure observer of FSDP's per-layer parameter state.

    Tracks an FSDP per-layer state machine driven by engine hook fire
    points; exposes ``params_resident(L)`` accurately so the engine's
    cost model knows when re-gather is free. The observation surface —
    state machine, params_resident, and estimate_allgather_bytes — lets
    the cost model reason about whatever FSDP itself decides.

    State machine (per layer in {'sharded', 'materializing', 'full'},
    held in ``self._layer_state``):

      Initial:                    'sharded'
      on_pre_forward(L):          sharded -> full
        (materializing is collapsed into full: by the time the
        engine's pre-hook returns control to the layer's forward,
        FSDP has gathered.)
      on_post_forward(L):         full -> sharded
      on_pre_backward(L):         (if not 'full') sharded -> full
        (FSDP's BACKWARD_PRE prefetch arrives by here.)
      on_post_backward(L):        full -> sharded

    State is per-layer here, not per-record on TensorRecord. State is
    layer-uniform — every shard of layer L transitions together — so per-layer
    is the honest granularity.

    No phantom full-param TensorRecord is registered on
    materialization. FSDP's full-param HBM is already
    visible to HBMBudget.pressure_signal via torch.cuda.memory_*().
    """

    _VALID_STATES = ('sharded', 'materializing', 'full')
    # Cap on the one-shot transition log printed during init. 16 layers × 4
    # hooks per step = 64 transitions per step. 256 covers ~4 steps so we can
    # see steady-state behavior across step boundaries. Silenced entirely when
    # verbose=False; auto-silences when the cap is reached.
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
        # Per-layer state. Initialized below by _build_layer_map.
        self._layer_to_fsdp_module: dict[int, Any] = {}
        self._layer_state: dict[int, str] = {}
        # Diagnostic counter for one-shot transition logging.
        self._diag_transitions_logged: int = 0
        self._build_layer_map(fsdp_model)
        if verbose:
            print(
                f"[FSDPCoordinator] init DONE: world_size={self._world_size}, "
                f"layers_tracked={len(self._layer_state)}"
            )

    # ── Init helpers ────────────────────────────────────────────────────

    def _build_layer_map(self, fsdp_model: Any) -> None:
        """Walk the FSDP-wrapped model, identify decoder layers, init state.

        Reuses the traversal pattern from fsdp_profiler.py — match
        modules by class name 'LlamaDecoderLayer', extract layer index
        from the dotted module name when possible, fall back to
        discovery order otherwise.
        """
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

    # ── Query surface (consumed by PlacementEngine) ─────────────────────

    def estimate_allgather_bytes(self, layer_idx: int) -> int:
        """Per-rank bytes-on-network for layer L's all-gather.

        Same formula as NoOp: ``shard_total × (W - 1)``. Feeds the
        criterion's re-all-gather term on the recompute side.
        """
        shard_total = sum(
            r.size_bytes for _, r in self._registry.by_category('parameter')
            if r.layer_idx == layer_idx
        )
        if self._world_size <= 1:
            return 0
        return shard_total * (self._world_size - 1)

    def params_resident(self, layer_idx: int) -> bool:
        """True iff layer L's params are currently a full (gathered) tensor.

        The engine's ``_exposed_reallgather_ms(L)`` returns 0 when the
        coordinator reports L resident. Returns False for layer indices
        we don't track (embedding / lm_head wraps); the engine treats
        that the same as 'sharded'.
        """
        return self._layer_state.get(layer_idx) == 'full'

    # ── State-machine callbacks ─────────────────────────────────────────
    # These are forwarded by PlacementEngine from its own hook
    # callbacks at exactly the same fire points. Untracked layer
    # indices are silently ignored — embedding / lm_head wraps fire
    # the same callbacks but aren't in _layer_state.

    def _log_transition(self, layer_idx: int, old: str, new: str,
                        hook: str, note: str = '') -> None:
        """Print first N transitions (diagnostic)."""
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
        """sharded -> full.

        By the time the engine's pre-hook returns control to the
        layer's forward, FSDP has gathered. The intermediate
        'materializing' state is collapsed.
        """
        if layer_idx not in self._layer_state:
            return
        old = self._layer_state[layer_idx]
        self._layer_state[layer_idx] = 'full'
        self._log_transition(layer_idx, old, 'full', 'on_pre_forward')

    def on_post_forward(self, layer_idx: int) -> None:
        """full -> sharded, unconditionally."""
        if layer_idx not in self._layer_state:
            return
        old = self._layer_state[layer_idx]
        self._layer_state[layer_idx] = 'sharded'
        self._log_transition(layer_idx, old, 'sharded', 'on_post_forward')

    def on_pre_backward(self, layer_idx: int) -> None:
        """(if not already full) sharded -> full.

        FSDP's BACKWARD_PRE prefetch has arrived by here.
        """
        if layer_idx not in self._layer_state:
            return
        old = self._layer_state[layer_idx]
        self._layer_state[layer_idx] = 'full'
        self._log_transition(layer_idx, old, 'full', 'on_pre_backward')

    def on_post_backward(self, layer_idx: int) -> None:
        """full -> sharded."""
        if layer_idx not in self._layer_state:
            return
        old = self._layer_state[layer_idx]
        self._layer_state[layer_idx] = 'sharded'
        self._log_transition(layer_idx, old, 'sharded', 'on_post_backward')

    # ── Diagnostics ─────────────────────────────────────────────────────

    def layer_state(self, layer_idx: int) -> str:
        """Read-only state lookup. Returns 'sharded' for untracked layers."""
        return self._layer_state.get(layer_idx, 'sharded')
