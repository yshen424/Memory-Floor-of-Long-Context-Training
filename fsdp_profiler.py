"""
FSDP-aware Memory Profiler for HuggingFace LLaMA.

Adapted for distributed training:
- Each rank captures its own profile independently
- FSDP-sharded parameter sizes are tracked (local shard vs full)
- NCCL communication events are timed via CUDA events
- All output is tagged with rank to avoid file collisions
- Hooks attach to HuggingFace LlamaDecoderLayer (not our custom model)

Key measurements:
- Per-layer activation breakdown (attention scores vs SwiGLU vs QKV)
- FSDP all-gather timing per layer
- Peak HBM timeline across forward/backward/optimizer
- Sharded vs unsharded parameter memory
- Allocation groups (SwiGLU triples) with sizes
"""

import torch
import torch.distributed as dist
import time
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
from contextlib import contextmanager

# TensorRecord lives in memory_manager.tensor_registry; imported at module
# scope so call sites that instantiate TensorRecord(...) resolve here.
from memory_manager import AccessPredictor, TensorRecord, TensorRegistry


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class MemorySnapshot:
    """Point-in-time capture of memory state."""
    timestamp_ms: float
    phase: str
    allocated_bytes: int
    reserved_bytes: int
    peak_allocated_bytes: int
    # High-water mark since the *previous* snapshot only, so consecutive
    # snapshots form an envelope of the step rather than a monotone staircase.
    interval_peak_bytes: int = 0


@dataclass
class CommEvent:
    """Record of a single NCCL communication event."""
    phase: str              # 'fwd_allgather_layer_3', 'bwd_reducescatter_layer_3'
    layer_idx: int
    start_ms: float
    end_ms: float
    duration_ms: float
    estimated_bytes: int    # bytes transferred (estimated from tensor sizes)


@dataclass
class LayerProfile:
    """Memory and timing profile for a single transformer layer."""
    layer_idx: int
    # Forward
    fwd_start_mem_bytes: int = 0
    fwd_end_mem_bytes: int = 0
    fwd_delta_bytes: int = 0         # memory growth during forward
    fwd_duration_ms: float = 0.0
    # Backward
    bwd_start_mem_bytes: int = 0
    bwd_end_mem_bytes: int = 0
    bwd_delta_bytes: int = 0
    bwd_duration_ms: float = 0.0
    # Activation breakdown (estimated from model config)
    attn_scores_bytes: int = 0
    swiglu_group_bytes: int = 0
    qkv_bytes: int = 0
    layer_input_bytes: int = 0


# ─── Profiler ─────────────────────────────────────────────────────────────────

class FSDPMemoryProfiler:
    """
    Memory profiler for FSDP-wrapped HuggingFace LLaMA.

    Usage:
        profiler = FSDPMemoryProfiler(model, rank, world_size, model_config)
        profiler.attach_hooks()

        for step in range(num_steps):
            with profiler.profile_step(step):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                loss.backward()
                profiler.mark_phase("backward_end")
                optimizer.step()
                profiler.mark_phase("optimizer_end")

        profiler.report()
        profiler.save("profile_results")
    """

    def __init__(self, model, rank: int, world_size: int, model_config,
                 enabled: bool = True, materializes_attn_scores: bool = True):
        self.model = model
        self.rank = rank
        self.world_size = world_size
        self.model_config = model_config   # HuggingFace LlamaConfig
        self.enabled = enabled
        # The memory-efficient kernel never forms the s x s tensor, so the
        # telemetry must not report one under it. Default True is the
        # score-materializing math kernel.
        self.materializes_attn_scores = materializes_attn_scores

        # Data stores
        self.snapshots: list[MemorySnapshot] = []
        self.layer_profiles: dict[int, LayerProfile] = {}
        self.registry: TensorRegistry = TensorRegistry()
        self.comm_events: list[CommEvent] = []
        self.timeline: list[dict] = []

        # State
        self._step_num = 0
        self._phase = "idle"
        self._start_time = 0.0
        self._hooks = []
        self._layer_timers = {}

        # Optional PlacementEngine attachment. When None, the profiler is an
        # observer only (schema_version 1). When attached via attach_engine(...)
        # the per-layer hooks forward to the engine and save() emits schema 2.
        self.engine: Optional[object] = None
        # Recompute unit the attached engine selected; set by attach_engine.
        # Decides whether the planner runs on the inner decoder layer or on
        # the checkpoint wrapper that precedes it.
        self._bundle_unit: str = 'per_module'

        # Per-interval peak attribution. CUDA exposes only a high-water mark since the last reset, so an
        # interval's own peak requires resetting at every snapshot. That would
        # destroy the step-level peak the engine reads at step end
        # (observed_peak_mb — the number behind every peak figure in the
        # ledger), so the running max below shadows it: snapshots and the
        # engine both read this instead of the raw CUDA counter, so the
        # step-level peak keeps its semantics while the per-interval field
        # carries the envelope.
        self._step_peak_running_max: int = 0
        self._interval_peak_enabled: bool = True
        # Index into self.snapshots of this step's first snapshot. Snapshot
        # indices are run-global and monotone, which is what lets the engine
        # use them as absence-interval ordinals without a second counter that could drift out of agreement.
        self._step_snapshot_start: int = 0

        # Precompute per-layer activation sizes from config
        self._precompute_activation_sizes()

    def _precompute_activation_sizes(self):
        """
        Estimate per-layer activation sizes from the model config.
        These don't depend on runtime — they're determined by architecture
        and the batch_size/seq_len we'll set at profiling time.
        """
        cfg = self.model_config
        self._cfg_hidden = cfg.hidden_size
        self._cfg_intermediate = cfg.intermediate_size
        self._cfg_num_heads = cfg.num_attention_heads
        self._cfg_num_kv_heads = cfg.num_key_value_heads
        self._cfg_head_dim = cfg.hidden_size // cfg.num_attention_heads
        self._cfg_num_layers = cfg.num_hidden_layers

    def _activation_sizes_for_batch(self, batch_size: int, seq_len: int,
                                     dtype_size: int = 2) -> dict:
        """
        Compute activation sizes for a given batch/seq_len.
        Returns per-layer breakdown in bytes.
        """
        # Attention scores: (batch, num_heads, seq_len, seq_len).
        #
        # HF upcasts the softmax to fp32 and autograd saves the fp32 output,
        # so the saved tensor is 4 bytes per element whatever the training
        # dtype. The constant is shared with the decision path's copy of this
        # formula so the two cannot drift apart.
        #
        # Telemetry only — no decision reads these bytes. The live planner
        # sizes bundles from AccessPredictor and measures occupancy with
        # torch.cuda.memory_allocated().
        scores_elem_bytes = max(dtype_size, AccessPredictor._SCORES_ELEM_BYTES)
        attn_scores = (batch_size * self._cfg_num_heads * seq_len * seq_len
                       * scores_elem_bytes)
        if not self.materializes_attn_scores:
            # No s x s tensor exists under the cutlass kernel. The key keeps
            # its name so the JSON schema and the per-layer table are shared
            # across kernels, but here it carries what that kernel retains
            # beyond q/k/v — rotated q and k, the attention output, and the
            # fp32 logsumexp. See AccessPredictor._efficient_attn_slot_bytes
            # for the derivation.
            attn_scores = (
                2 * batch_size * seq_len * self._cfg_hidden * dtype_size
                + batch_size * seq_len * self._cfg_num_kv_heads
                * self._cfg_head_dim * dtype_size
                + batch_size * self._cfg_num_heads * seq_len * 4
            )

        # SwiGLU intermediates: (batch, seq_len, intermediate_dim), and autograd
        # saves FOUR distinct storages of that shape per MLP, not three.
        # Derivation in AccessPredictor._SWIGLU_SAVED_PLANES; the constant is
        # shared with the decision path's copy so the two cannot drift apart,
        # the same way _SCORES_ELEM_BYTES is shared above. Telemetry only.
        swiglu_single = batch_size * seq_len * self._cfg_intermediate * dtype_size
        swiglu_group = swiglu_single * AccessPredictor._SWIGLU_SAVED_PLANES

        # QKV: Q full heads, K and V reduced (GQA)
        q_size = batch_size * seq_len * self._cfg_num_heads * self._cfg_head_dim * dtype_size
        kv_size = batch_size * seq_len * self._cfg_num_kv_heads * self._cfg_head_dim * dtype_size
        qkv_total = q_size + 2 * kv_size

        # Layer input: (batch, seq_len, hidden)
        layer_input = batch_size * seq_len * self._cfg_hidden * dtype_size

        return {
            'attn_scores': attn_scores,
            'swiglu_group': swiglu_group,
            'swiglu_single': swiglu_single,
            'qkv': qkv_total,
            'q': q_size,
            'kv_each': kv_size,
            'layer_input': layer_input,
        }

    # ─── Timing and Memory Helpers ────────────────────────────────────────────

    # The is_available guards match step_peak_bytes below: on the cluster they
    # are always true, so timings and byte counts are unchanged there. Off-GPU
    # they let the standalone hook-wiring tests drive a real model on CPU.
    def _now_ms(self) -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.perf_counter() - self._start_time) * 1000

    def _get_mem(self) -> dict:
        if not torch.cuda.is_available():
            return {'allocated': 0, 'reserved': 0, 'peak': 0}
        return {
            'allocated': torch.cuda.memory_allocated(),
            'reserved': torch.cuda.memory_reserved(),
            'peak': torch.cuda.max_memory_allocated(),
        }

    def step_peak_bytes(self) -> int:
        """Step-level peak HBM, shadowing CUDA's counter.

        The engine reads this at step end instead of
        ``torch.cuda.max_memory_allocated()``: once per-interval attribution
        starts resetting the CUDA counter, only this running max still holds
        the whole step's high-water mark. Returns the larger of the two so the
        value is correct whether or not this step was snapshot-profiled.
        """
        raw = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        return max(raw, self._step_peak_running_max)

    def current_interval_ordinal(self) -> tuple:
        """``(index, phase)`` of the most recent snapshot.

        The engine stamps this on a bundle when its D2H completes and again
        when its restore is issued, so an absence interval is expressed in the
        same units as the per-interval peaks — no second counter, no
        agreement-by-construction between two clocks. Snapshots are taken
        before the engine hook at every layer boundary, so the index a hook
        reads is the interval that just ended. ``(-1, '')`` before any
        snapshot exists.
        """
        if not self.snapshots:
            return (-1, '')
        return (len(self.snapshots) - 1, self.snapshots[-1].phase)

    def peak_interval_ordinal(self) -> tuple:
        """``(index, phase, bytes)`` of this step's highest-peak interval.

        The peak *instant* is only ever localized to an interval, so this is
        the finest honest answer to "when did the peak happen". Read by the
        engine at step end to decide whether a bundle's absence covered it.
        """
        window = self.snapshots[self._step_snapshot_start:]
        if not window:
            return (-1, '', 0)
        best = max(range(len(window)),
                   key=lambda i: window[i].interval_peak_bytes)
        return (self._step_snapshot_start + best,
                window[best].phase,
                window[best].interval_peak_bytes)

    def _read_interval_peak_and_reset(self) -> int:
        """Peak since the previous snapshot, then re-arm for the next one.

        Folds the reading into the running max before resetting, so the step
        peak survives the reset that makes the interval measurement possible.
        """
        if not torch.cuda.is_available():
            return 0
        interval_peak = int(torch.cuda.max_memory_allocated())
        if interval_peak > self._step_peak_running_max:
            self._step_peak_running_max = interval_peak
        if self._interval_peak_enabled:
            torch.cuda.reset_peak_memory_stats()
        return interval_peak

    def _take_snapshot(self, phase: str):
        if not self.enabled:
            return
        stats = self._get_mem()
        # Ordering matters: _now_ms() synchronizes, so the reset below lands
        # after every kernel this interval issued has retired. No extra sync.
        timestamp_ms = self._now_ms()
        interval_peak = self._read_interval_peak_and_reset()
        snap = MemorySnapshot(
            timestamp_ms=timestamp_ms,
            phase=phase,
            allocated_bytes=stats['allocated'],
            reserved_bytes=stats['reserved'],
            # Unchanged semantics: cumulative high-water since step start.
            peak_allocated_bytes=self._step_peak_running_max or stats['peak'],
            interval_peak_bytes=interval_peak,
        )
        self.snapshots.append(snap)
        self.timeline.append({
            'time_ms': snap.timestamp_ms,
            'event': 'snapshot',
            'phase': phase,
            'allocated_mb': snap.allocated_bytes / (1024**2),
            'peak_mb': snap.peak_allocated_bytes / (1024**2),
            'interval_peak_mb': snap.interval_peak_bytes / (1024**2),
        })

    # ─── Hook Management ──────────────────────────────────────────────────────

    def attach_hooks(self):
        """
        Attach forward/backward hooks to each LlamaDecoderLayer inside
        the FSDP-wrapped model.

        FSDP wraps each LlamaDecoderLayer, so we need to walk the module
        tree and find them inside the FSDP wrappers.
        """
        # Record sharded parameter sizes
        total_local = 0
        for name, param in self.model.named_parameters():
            size = param.numel() * param.element_size()
            total_local += size
            self.registry.register(
                f"param_shard_{name}",
                TensorRecord(
                    name=name, size_bytes=size,
                    shape=tuple(param.shape), dtype=str(param.dtype),
                    component='parameter_shard', layer_idx=self._extract_layer_idx(name),
                    category='parameter',
                ),
            )

        self.timeline.append({
            'time_ms': 0, 'event': 'param_inventory',
            'local_shard_mb': total_local / (1024**2),
            'estimated_full_mb': total_local * self.world_size / (1024**2),
        })

        # Walk module tree to find decoder layers
        layer_count = 0
        for name, module in self.model.named_modules():
            # HuggingFace LlamaDecoderLayer class name
            cls_name = module.__class__.__name__
            if cls_name == 'LlamaDecoderLayer':
                layer_idx = self._extract_layer_idx(name)
                if layer_idx < 0:
                    layer_idx = layer_count
                layer_count += 1

                self.layer_profiles[layer_idx] = LayerProfile(layer_idx=layer_idx)

                h_pre = module.register_forward_pre_hook(
                    self._make_fwd_pre_hook(layer_idx))
                h_post = module.register_forward_hook(
                    self._make_fwd_post_hook(layer_idx))
                h_bwd_pre = module.register_full_backward_pre_hook(
                    self._make_bwd_pre_hook(layer_idx))
                h_bwd_post = module.register_full_backward_hook(
                    self._make_bwd_post_hook(layer_idx))
                self._hooks.extend([h_pre, h_post, h_bwd_pre, h_bwd_post])

        if self.rank == 0:
            print(f"  [Profiler] Attached hooks to {layer_count} decoder layers")
            print(f"  [Profiler] Local param shards: {total_local / (1024**2):.1f} MB")

    def detach_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # Placement-engine attachment. Single setter; the engine itself is constructed in train_fsdp_profile.py
    # so the profiler doesn't grow a hard dependency on memory_manager
    # internals beyond the registry it already owns.
    def attach_engine(self, engine) -> None:
        """Wire a PlacementEngine into the profiler hooks.

        Once attached, every fwd-post / bwd-pre / bwd-post layer hook
        forwards to the engine, and save() emits the engine's JSON
        additions (schema_version=2, calibration, hbm_budget_bytes,
        placement_decisions). Call before the training loop starts.
        """
        self.engine = engine
        # The engine's step-end peak read must see the intervals this profiler
        # has already reset away.
        if hasattr(engine, '_step_peak_source'):
            engine._step_peak_source = lambda: self._step_peak_running_max
        # Absence intervals are measured in snapshot indices.
        if hasattr(engine, '_interval_ordinal_source'):
            engine._interval_ordinal_source = self.current_interval_ordinal
            engine._peak_interval_source = self.peak_interval_ordinal
        if self.rank == 0:
            print("  [Profiler] PlacementEngine attached — hooks will dispatch "
                  "to engine; JSON output will use schema_version=2.")

        # Submodule bundle hooks. Walk the module tree once more to find
        # LlamaAttention and LlamaMLP submodules. Pre-hook calls
        # engine.enter_bundle so pack_fn (which fires inline during the
        # submodule's forward, before the LlamaDecoderLayer post-hook) can
        # attribute each autograd-saved tensor to the correct bundle.
        # Post-hook clears the context.
        #
        # These hooks live in self._hooks so detach_hooks() removes them
        # in lockstep with the layer hooks.
        #
        # Under the whole-layer unit the submodule hooks are not just
        # redundant, they are too late. checkpoint_wrapper saves the
        # layer's residual-stream boundary at the WRAPPER's entry, before any
        # submodule inside the layer runs, so a bundle opened by an attention
        # or mlp pre-hook is opened after the only save the outer seam will
        # ever see. pack_fn's first guard would then decline every boundary
        # and the engine would report a clean step having decided nothing.
        # Validated statically in tests/test_checkpoint_bundle_ordering.py
        # (H-A and H-C). The wrapper's own pre-hook does fire first, so that
        # is where both the decision and the bundle go.
        self._bundle_unit = getattr(engine, '_bundle_unit', 'per_module')
        if self._bundle_unit == 'whole_layer':
            self._attach_whole_layer_bundle_hooks()
            return

        attn_count = 0
        mlp_count = 0
        for name, module in self.model.named_modules():
            cls_name = module.__class__.__name__
            if cls_name == 'LlamaAttention':
                layer_idx = self._extract_layer_idx(name)
                # with_kwargs=True so the engine can
                # capture (args, kwargs) to replay the submodule forward at
                # backward time when the bundle is marked recompute.
                h_pre = module.register_forward_pre_hook(
                    self._make_submodule_pre_hook(layer_idx, 'attention'),
                    with_kwargs=True)
                h_post = module.register_forward_hook(
                    self._make_submodule_post_hook())
                self._hooks.extend([h_pre, h_post])
                attn_count += 1
            elif cls_name == 'LlamaMLP':
                layer_idx = self._extract_layer_idx(name)
                h_pre = module.register_forward_pre_hook(
                    self._make_submodule_pre_hook(layer_idx, 'mlp'),
                    with_kwargs=True)
                h_post = module.register_forward_hook(
                    self._make_submodule_post_hook())
                self._hooks.extend([h_pre, h_post])
                mlp_count += 1
        if self.rank == 0:
            print(f"  [Profiler] submodule bundle hooks attached: "
                  f"{attn_count} LlamaAttention, {mlp_count} LlamaMLP")

    # ── Whole-layer unit: wrapper-level bundle + decision ──────────────────

    def _attach_whole_layer_bundle_hooks(self) -> None:
        """Open one 'layer' bundle per CheckpointWrapper, before it saves.

        Two hooks on the wrapper, registered in this order so the decision is
        in hand before the bundle opens and therefore before the boundary is
        packed:

          1. ``on_pre_forward(L)`` — the planner. It cannot run on the inner
             decoder layer's pre-hook, which under checkpointing fires after
             the boundary save: at layer L's save only layers 0..L-1 would
             have been decided, so the decision map would be empty for L.
          2. ``enter_bundle(L, 'layer')`` — the attribution context pack_fn
             reads.

        The inner layer keeps its timing/registry/snapshot pre-hook; only the
        planner call moves. ``_make_fwd_pre_hook`` skips it in this mode.
        """
        wrapped = 0
        for name, module in self.model.named_modules():
            if module.__class__.__name__ != 'CheckpointWrapper':
                continue
            layer_idx = self._extract_layer_idx(name)
            if layer_idx < 0:
                continue
            inner = getattr(module, '_checkpoint_wrapped_module', module)
            h_dec = module.register_forward_pre_hook(
                self._make_layer_decision_hook(layer_idx))
            h_pre = module.register_forward_pre_hook(
                self._make_layer_bundle_pre_hook(layer_idx, inner),
                with_kwargs=True)
            h_post = module.register_forward_hook(
                self._make_submodule_post_hook())
            self._hooks.extend([h_dec, h_pre, h_post])
            wrapped += 1
        if self.rank == 0:
            print(f"  [Profiler] whole-layer bundle hooks attached to "
                  f"{wrapped} CheckpointWrapper modules; submodule "
                  f"attention/mlp bundles suppressed")
        if wrapped == 0:
            raise RuntimeError(
                "whole-layer recompute unit selected but no CheckpointWrapper "
                "was found in the model. The engine would open no bundles and "
                "decline every save, reporting a clean step having decided "
                "nothing. Pass --layer_checkpoint, or select the per-module "
                "unit.")

    def _make_layer_decision_hook(self, layer_idx: int):
        def hook(module, args):
            if self.engine is None:
                return
            decisions = self.engine.on_pre_forward(layer_idx)
            self.timeline.append({
                'time_ms': self._now_ms(),
                'event': 'placement_decision',
                'hook': 'pre_fwd_wrapper',
                'layer_idx': layer_idx,
                'count': len(decisions),
            })
        return hook

    def _make_layer_bundle_pre_hook(self, layer_idx: int, inner):
        def hook(module, args, kwargs):
            if self.engine is not None:
                self.engine.enter_bundle(
                    layer_idx, 'layer',
                    inputs=args, kwargs=kwargs, module=inner,
                )
        return hook

    # ── Submodule bundle hook factories ────────────────────────────────────

    def _make_submodule_pre_hook(self, layer_idx: int, kind: str):
        """Enter bundle context before a LlamaAttention / LlamaMLP forward.

        The hook is registered with_kwargs=True so it can hand the engine ``(args, kwargs, module)`` and let it cache
        them for bundles that will be recomputed during backward. Engine
        ignores the args/kwargs for non-recompute bundles, so the only cost
        on the hot path is a couple of extra dict refs.
        """
        def hook(module, args, kwargs):
            if self.engine is not None:
                self.engine.enter_bundle(
                    layer_idx, kind,
                    inputs=args, kwargs=kwargs, module=module,
                )
        return hook

    def _make_submodule_post_hook(self):
        """Exit bundle context after a LlamaAttention / LlamaMLP forward."""
        def hook(module, inputs, output):
            if self.engine is not None:
                self.engine.exit_bundle()
        return hook

    def _extract_layer_idx(self, name: str) -> int:
        parts = name.split('.')
        for i, part in enumerate(parts):
            if part == 'layers' and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    pass
        return -1

    # ─── Hook Factories ───────────────────────────────────────────────────────

    def _make_fwd_pre_hook(self, layer_idx):
        def hook(module, inputs):
            if not self.enabled:
                return
            self._phase = f"fwd_layer_{layer_idx}"
            mem = self._get_mem()
            self._layer_timers[f"fwd_{layer_idx}"] = self._now_ms()
            self.layer_profiles[layer_idx].fwd_start_mem_bytes = mem['allocated']
            self._take_snapshot(f"pre_fwd_layer_{layer_idx}")

            # Capture batch dimensions from the first input
            if isinstance(inputs, tuple) and len(inputs) > 0:
                inp = inputs[0]
                if isinstance(inp, torch.Tensor) and inp.dim() == 3:
                    batch, seq_len, _ = inp.shape
                    # Compute activation sizes for this batch
                    dtype_size = inp.element_size()
                    sizes = self._activation_sizes_for_batch(batch, seq_len, dtype_size)
                    prof = self.layer_profiles[layer_idx]
                    prof.attn_scores_bytes = sizes['attn_scores']
                    prof.swiglu_group_bytes = sizes['swiglu_group']
                    prof.qkv_bytes = sizes['qkv']
                    prof.layer_input_bytes = sizes['layer_input']

                    # Record SwiGLU allocation group.
                    # The group holds four planes across three records — the
                    # gate record carries the gate projection's output AND the
                    # activation's output, which autograd saves as two
                    # separate storages. Three records keep the group's
                    # membership, and every ordinal keyed off it, fixed.
                    group_id = f"layer_{layer_idx}_swiglu"
                    for suffix in ['gate', 'up', 'product']:
                        _planes = (AccessPredictor._SWIGLU_GATE_PLANES
                                   if suffix == 'gate' else 1)
                        self.registry.register(
                            f"act_{group_id}_{suffix}",
                            TensorRecord(
                                name=f"{group_id}_{suffix}",
                                size_bytes=sizes['swiglu_single'] * _planes,
                                shape=(batch, seq_len, self._cfg_intermediate),
                                dtype=str(inp.dtype),
                                component=f'swiglu_{suffix}',
                                layer_idx=layer_idx,
                                category='activation',
                                group_id=group_id,
                                birth_time_ms=self._now_ms(),
                                recompute_cost_flops=(
                                    2 * batch * seq_len * self._cfg_hidden * self._cfg_intermediate
                                ),
                            ),
                        )

                    # Record attention scores tensor.
                    # The registry key stays `..._attn_scores` in
                    # both regimes — it is a schedule ordinal anchor, not just a
                    # label — but the shape and component name follow the kernel
                    # so a reader of the JSON is not told an s x s tensor exists
                    # when none does.
                    scores_live = self.materializes_attn_scores
                    self.registry.register(
                        f"act_layer_{layer_idx}_attn_scores",
                        TensorRecord(
                            name=f"layer_{layer_idx}_attn_scores",
                            size_bytes=sizes['attn_scores'],
                            shape=((batch, self._cfg_num_heads, seq_len, seq_len)
                                   if scores_live else (batch, seq_len, -1)),
                            dtype=str(inp.dtype),
                            component=('attention_scores' if scores_live
                                       else 'attention_efficient_saved'),
                            layer_idx=layer_idx,
                            category='activation',
                            birth_time_ms=self._now_ms(),
                        ),
                    )

                    # Record QKV (noting GQA asymmetry)
                    self.registry.register(
                        f"act_layer_{layer_idx}_q",
                        TensorRecord(
                            name=f"layer_{layer_idx}_q",
                            size_bytes=sizes['q'],
                            shape=(batch, seq_len, self._cfg_num_heads, self._cfg_head_dim),
                            dtype=str(inp.dtype),
                            component='attention_q',
                            layer_idx=layer_idx,
                            category='activation',
                            birth_time_ms=self._now_ms(),
                        ),
                    )
                    for kv_name in ['k', 'v']:
                        self.registry.register(
                            f"act_layer_{layer_idx}_{kv_name}",
                            TensorRecord(
                                name=f"layer_{layer_idx}_{kv_name}",
                                size_bytes=sizes['kv_each'],
                                shape=(batch, seq_len, self._cfg_num_kv_heads, self._cfg_head_dim),
                                dtype=str(inp.dtype),
                                component=f'attention_{kv_name}',
                                layer_idx=layer_idx,
                                category='activation',
                                birth_time_ms=self._now_ms(),
                            ),
                        )

            # Forward to the placement engine BEFORE the layer's submodules
            # run. on_pre_forward(L) makes the per-layer eviction decision so
            # pack_fn (which fires inline during LlamaAttention/LlamaMLP
            # forwards) has a decision in hand by the time autograd asks to
            # save a tensor. Placed after the registry.register() calls above
            # so the engine sees layer L's activation records when it iterates.
            # Under the whole-layer unit this call has already
            # happened on the checkpoint wrapper's pre-hook. It must not run
            # again here — the boundary is packed by now, and a second call
            # would advance the step cursor a second time per layer.
            if self.engine is not None and self._bundle_unit != 'whole_layer':
                decisions = self.engine.on_pre_forward(layer_idx)
                self.timeline.append({
                    'time_ms': self._now_ms(),
                    'event': 'placement_decision',
                    'hook': 'pre_fwd',
                    'layer_idx': layer_idx,
                    'count': len(decisions),
                })
        return hook

    def _make_fwd_post_hook(self, layer_idx):
        def hook(module, inputs, output):
            if not self.enabled:
                return
            mem = self._get_mem()
            start = self._layer_timers.get(f"fwd_{layer_idx}", self._now_ms())
            duration = self._now_ms() - start

            prof = self.layer_profiles[layer_idx]
            prof.fwd_end_mem_bytes = mem['allocated']
            prof.fwd_delta_bytes = mem['allocated'] - prof.fwd_start_mem_bytes
            prof.fwd_duration_ms = duration

            self._take_snapshot(f"post_fwd_layer_{layer_idx}")
            self.timeline.append({
                'time_ms': self._now_ms(),
                'event': 'fwd_layer_complete',
                'layer': layer_idx,
                'duration_ms': round(duration, 3),
                'mem_delta_mb': prof.fwd_delta_bytes / (1024**2),
                'allocated_mb': mem['allocated'] / (1024**2),
            })

            # Forward to the placement engine if attached. The engine returns its decision list; we only stash a
            # one-line timeline marker (count) so the JSON doesn't double-store
            # them (the full list lives in placement_decisions at save() time).
            if self.engine is not None:
                decisions = self.engine.on_post_forward(layer_idx)
                self.timeline.append({
                    'time_ms': self._now_ms(),
                    'event': 'placement_decision',
                    'hook': 'post_fwd',
                    'layer_idx': layer_idx,
                    'count': len(decisions),
                })
        return hook

    def _make_bwd_pre_hook(self, layer_idx):
        def hook(module, grad_output):
            if not self.enabled:
                return
            self._phase = f"bwd_layer_{layer_idx}"
            mem = self._get_mem()
            self._layer_timers[f"bwd_{layer_idx}"] = self._now_ms()
            self.layer_profiles[layer_idx].bwd_start_mem_bytes = mem['allocated']
            self._take_snapshot(f"pre_bwd_layer_{layer_idx}")

            # Engine pre-backward callback.
            # Classifies each of L's activations (retain / stall / recompute).
            if self.engine is not None:
                decisions = self.engine.on_pre_backward(layer_idx)
                self.timeline.append({
                    'time_ms': self._now_ms(),
                    'event': 'placement_decision',
                    'hook': 'pre_bwd',
                    'layer_idx': layer_idx,
                    'count': len(decisions),
                })
        return hook

    def _make_bwd_post_hook(self, layer_idx):
        def hook(module, grad_input, grad_output):
            if not self.enabled:
                return
            mem = self._get_mem()
            start = self._layer_timers.get(f"bwd_{layer_idx}", self._now_ms())
            duration = self._now_ms() - start

            prof = self.layer_profiles[layer_idx]
            prof.bwd_end_mem_bytes = mem['allocated']
            prof.bwd_delta_bytes = mem['allocated'] - prof.bwd_start_mem_bytes
            prof.bwd_duration_ms = duration

            self._take_snapshot(f"post_bwd_layer_{layer_idx}")
            self.timeline.append({
                'time_ms': self._now_ms(),
                'event': 'bwd_layer_complete',
                'layer': layer_idx,
                'duration_ms': round(duration, 3),
                'mem_delta_mb': prof.bwd_delta_bytes / (1024**2),
                'allocated_mb': mem['allocated'] / (1024**2),
            })

            # Engine post-backward callback.
            # Frees L's activations (tier='freed') and prefetches L-1's.
            if self.engine is not None:
                decisions = self.engine.on_post_backward(layer_idx)
                self.timeline.append({
                    'time_ms': self._now_ms(),
                    'event': 'placement_decision',
                    'hook': 'post_bwd',
                    'layer_idx': layer_idx,
                    'count': len(decisions),
                })
        return hook

    # ─── Step-level Profiling ─────────────────────────────────────────────────

    @contextmanager
    def profile_step(self, step_num: int):
        self._step_num = step_num
        self._start_time = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        # Re-arm the shadow max together with the CUDA counter it shadows.
        self._step_peak_running_max = 0
        self._step_snapshot_start = len(self.snapshots)
        torch.cuda.synchronize()

        self._take_snapshot("step_start")
        try:
            yield self
        finally:
            torch.cuda.synchronize()
            self._take_snapshot("step_end")
            self._phase = "idle"

    def mark_phase(self, phase: str):
        self._phase = phase
        self._take_snapshot(phase)

    # ─── Analysis ─────────────────────────────────────────────────────────────

    def get_memory_breakdown(self) -> dict:
        """Breakdown by category and component."""
        bd = {
            'parameters': {'local_shard_bytes': 0, 'estimated_full_bytes': 0, 'count': 0},
            'activations': {
                'total_bytes': 0, 'count': 0,
                'attention_scores_bytes': 0,
                'swiglu_groups_bytes': 0,
                'qkv_bytes': 0,
            },
        }
        for name, rec in self.registry.items():
            if rec.category == 'parameter':
                bd['parameters']['local_shard_bytes'] += rec.size_bytes
                bd['parameters']['count'] += 1
            elif rec.category == 'activation':
                bd['activations']['total_bytes'] += rec.size_bytes
                bd['activations']['count'] += 1
                if 'attention_scores' in rec.component:
                    bd['activations']['attention_scores_bytes'] += rec.size_bytes
                elif 'swiglu' in rec.component:
                    bd['activations']['swiglu_groups_bytes'] += rec.size_bytes
                elif 'attention' in rec.component:
                    bd['activations']['qkv_bytes'] += rec.size_bytes

        bd['parameters']['estimated_full_bytes'] = (
            bd['parameters']['local_shard_bytes'] * self.world_size
        )
        return bd

    def get_per_layer_summary(self) -> list[dict]:
        summaries = []
        for idx in sorted(self.layer_profiles.keys()):
            p = self.layer_profiles[idx]
            summaries.append({
                'layer': idx,
                'fwd_ms': round(p.fwd_duration_ms, 3),
                'bwd_ms': round(p.bwd_duration_ms, 3),
                'fwd_delta_mb': round(p.fwd_delta_bytes / (1024**2), 2),
                'bwd_delta_mb': round(p.bwd_delta_bytes / (1024**2), 2),
                'attn_scores_mb': round(p.attn_scores_bytes / (1024**2), 2),
                'swiglu_group_mb': round(p.swiglu_group_bytes / (1024**2), 2),
                'qkv_mb': round(p.qkv_bytes / (1024**2), 2),
                'layer_input_mb': round(p.layer_input_bytes / (1024**2), 2),
            })
        return summaries

    def get_allocation_groups(self) -> dict:
        return self.registry.groups()

    def get_fsdp_summary(self) -> dict:
        """FSDP-specific summary: sharding ratio, estimated comm volume."""
        bd = self.get_memory_breakdown()
        local_mb = bd['parameters']['local_shard_bytes'] / (1024**2)
        full_mb = bd['parameters']['estimated_full_bytes'] / (1024**2)

        # Estimated all-gather volume per layer (unsharded params for one decoder layer)
        per_layer_param_bytes = bd['parameters']['local_shard_bytes'] / max(self._cfg_num_layers, 1)
        per_layer_full_bytes = per_layer_param_bytes * self.world_size

        return {
            'world_size': self.world_size,
            'local_shard_mb': round(local_mb, 2),
            'full_model_mb': round(full_mb, 2),
            'sharding_ratio': round(1.0 / self.world_size, 3),
            'per_layer_allgather_mb': round(per_layer_full_bytes / (1024**2), 2),
            # In forward+backward, each layer is all-gathered twice
            # (once fwd, once bwd) — this is the redundancy FSDP introduces
            'total_allgather_per_step_mb': round(
                2 * self._cfg_num_layers * per_layer_full_bytes / (1024**2), 2
            ),
        }

    # ─── Reporting ────────────────────────────────────────────────────────────

    def report(self) -> str:
        lines = []
        lines.append("=" * 72)
        lines.append(f"FSDP Memory Profile — Rank {self.rank}/{self.world_size}")
        lines.append("=" * 72)

        # Peak memory
        if self.snapshots:
            peak = max(self.snapshots, key=lambda s: s.allocated_bytes)
            lines.append(f"\nPeak HBM: {peak.allocated_bytes / (1024**2):.1f} MB "
                        f"at phase '{peak.phase}' ({peak.timestamp_ms:.1f} ms)")

        # FSDP summary
        fsdp = self.get_fsdp_summary()
        lines.append(f"\n--- FSDP Summary ---")
        lines.append(f"  World size:                {fsdp['world_size']}")
        lines.append(f"  Local param shards:        {fsdp['local_shard_mb']:>8.1f} MB")
        lines.append(f"  Full model (estimated):    {fsdp['full_model_mb']:>8.1f} MB")
        lines.append(f"  Per-layer all-gather:      {fsdp['per_layer_allgather_mb']:>8.2f} MB")
        lines.append(f"  Total all-gather/step:     {fsdp['total_allgather_per_step_mb']:>8.1f} MB"
                     f"  (2× for fwd+bwd)")

        # Activation breakdown
        bd = self.get_memory_breakdown()
        lines.append(f"\n--- Activation Breakdown (per-step totals) ---")
        lines.append(f"  Attention scores (quadratic): "
                     f"{bd['activations']['attention_scores_bytes'] / (1024**2):>8.1f} MB")
        lines.append(f"  SwiGLU groups (3× intermediate): "
                     f"{bd['activations']['swiglu_groups_bytes'] / (1024**2):>8.1f} MB")
        lines.append(f"  QKV projections (GQA):        "
                     f"{bd['activations']['qkv_bytes'] / (1024**2):>8.1f} MB")

        # Per-layer
        per_layer = self.get_per_layer_summary()
        if per_layer:
            lines.append(f"\n--- Per-Layer Profile ---")
            lines.append(
                f"{'Lyr':>3} | {'Fwd ms':>7} | {'Bwd ms':>7} | "
                f"{'Fwd Δ MB':>8} | {'Bwd Δ MB':>8} | "
                f"{'Attn MB':>7} | {'SwiGLU MB':>9} | {'QKV MB':>6}"
            )
            lines.append("-" * 76)
            for s in per_layer:
                lines.append(
                    f"{s['layer']:>3} | {s['fwd_ms']:>7.2f} | {s['bwd_ms']:>7.2f} | "
                    f"{s['fwd_delta_mb']:>8.2f} | {s['bwd_delta_mb']:>8.2f} | "
                    f"{s['attn_scores_mb']:>7.1f} | {s['swiglu_group_mb']:>9.1f} | "
                    f"{s['qkv_mb']:>6.1f}"
                )

        # Allocation groups
        groups = self.get_allocation_groups()
        if groups:
            lines.append(f"\n--- Allocation Groups ---")
            for gid, info in sorted(groups.items()):
                lines.append(f"  {gid}: {info['total_bytes']/(1024**2):.2f} MB "
                            f"({len(info['tensors'])} tensors)")

        # Eviction candidates
        lines.append(f"\n--- Top Eviction Candidates (by size) ---")
        activations = [(n, r) for n, r in self.registry.items()
                      if r.category == 'activation']
        activations.sort(key=lambda x: x[1].size_bytes, reverse=True)
        for name, rec in activations[:10]:
            lines.append(f"  {rec.name:>35}: {rec.size_bytes/(1024**2):>7.2f} MB "
                        f"[{rec.component}] L{rec.layer_idx}")

        # Memory timeline summary
        fwd_snaps = [s for s in self.snapshots if s.phase.startswith('post_fwd')]
        bwd_snaps = [s for s in self.snapshots if s.phase.startswith('post_bwd')]
        if fwd_snaps:
            fwd_peak = max(s.allocated_bytes for s in fwd_snaps)
            lines.append(f"\n--- Timeline Summary ---")
            lines.append(f"  Forward peak:  {fwd_peak / (1024**2):.1f} MB")
        if bwd_snaps:
            bwd_peak = max(s.allocated_bytes for s in bwd_snaps)
            lines.append(f"  Backward peak: {bwd_peak / (1024**2):.1f} MB")

        report = "\n".join(lines)
        print(report)
        return report

    def save(self, output_dir: str):
        """Save profile data to JSON. Files are per-rank.

        schema_version is 1 when no PlacementEngine is attached (the
        observer-only format). When attached,
        schema_version becomes 2 and three top-level keys are added:
        ``calibration``, ``hbm_budget_bytes``, ``placement_decisions``.
        Per-record dicts also gain ``tier``, ``in_flight``,
        ``checkpoint_enabled``, ``predicted_next_access_step`` (all
        present with safe defaults from the registry).
        """
        os.makedirs(output_dir, exist_ok=True)

        engine_attached = self.engine is not None

        # Build the per-record dict. The observer-only path writes 8 keys;
        # the engine path adds 4 more. Both paths iterate the registry
        # identically.
        if engine_attached:
            tensor_records_dict = {
                name: {
                    'name': r.name,
                    'size_mb': round(r.size_bytes / (1024**2), 3),
                    'shape': r.shape,
                    'component': r.component,
                    'layer_idx': r.layer_idx,
                    'category': r.category,
                    'group_id': r.group_id,
                    'recompute_cost_flops': r.recompute_cost_flops,
                    # schema_version 2 only ---------
                    'tier': r.tier,
                    'in_flight': r.in_flight,
                    'checkpoint_enabled': r.checkpoint_enabled,
                    'predicted_next_access_step': r.predicted_next_access_step,
                }
                for name, r in self.registry.items()
            }
        else:
            tensor_records_dict = {
                name: {
                    'name': r.name,
                    'size_mb': round(r.size_bytes / (1024**2), 3),
                    'shape': r.shape,
                    'component': r.component,
                    'layer_idx': r.layer_idx,
                    'category': r.category,
                    'group_id': r.group_id,
                    'recompute_cost_flops': r.recompute_cost_flops,
                }
                for name, r in self.registry.items()
            }

        data = {
            # schema_version makes the format gate explicit for
            # downstream consumers.
            'schema_version': 2 if engine_attached else 1,
            'rank': self.rank,
            'world_size': self.world_size,
            'step': self._step_num,
            'fsdp_summary': self.get_fsdp_summary(),
            'memory_breakdown': self.get_memory_breakdown(),
            'per_layer_summary': self.get_per_layer_summary(),
            'allocation_groups': self.get_allocation_groups(),
            'timeline': sorted(self.timeline, key=lambda x: x['time_ms']),
            'snapshots': [
                {
                    'timestamp_ms': round(s.timestamp_ms, 3),
                    'phase': s.phase,
                    'allocated_mb': round(s.allocated_bytes / (1024**2), 2),
                    'reserved_mb': round(s.reserved_bytes / (1024**2), 2),
                    'peak_mb': round(s.peak_allocated_bytes / (1024**2), 2),
                    'interval_peak_mb': round(s.interval_peak_bytes / (1024**2), 2),
                }
                for s in self.snapshots
            ],
            'tensor_records': tensor_records_dict,
        }

        # Engine-only additions. Kept after the base fields so a dict-merge
        # in a downstream tool preserves order and lets these keys be
        # ignored cleanly when missing.
        if engine_attached:
            engine = self.engine
            cal = engine._calibration  # noqa: SLF001 — engine-internal handle
            # Calibration fields. pcie_bw_gbps and round_trip_gbps are the
            # idle one-way and round-trip host-link bandwidths; the other
            # keys carry the under-NCCL and NCCL-allgather measurements.
            data['calibration'] = {
                'pcie_bw_gbps': round(
                    cal.pcie_bw_idle_one_way_bytes_per_s / (1024**3), 3),
                'round_trip_gbps': round(
                    cal.pcie_bw_idle_round_trip_bytes_per_s / (1024**3), 3),
                'gpu_tflops': round(
                    cal.gpu_flops_per_s / 1e12, 3),
                'pcie_bw_under_nccl_gbps': round(
                    getattr(cal, 'pcie_bw_under_nccl_one_way_bytes_per_s', 0.0)
                    / (1024**3), 3),
                'nccl_allgather_gbps': round(
                    getattr(cal, 'nccl_allgather_bytes_per_s', 0.0)
                    / (1024**3), 3),
                'sanity_flags': getattr(cal, 'sanity_flags', None) or {},
            }
            data['hbm_budget_bytes'] = engine._budget.total_bytes  # noqa: SLF001
            # The structured reason payload is serialized via
            # PlacementDecisionReason.to_dict(); the ``hasattr`` guard
            # tolerates a ``reason`` that is a bare string.
            data['placement_decisions'] = [
                {
                    'step_idx': d.step_idx,
                    'layer_idx': d.layer_idx,
                    'tensor_id': d.tensor_id,
                    'action': d.action,
                    'estimated_cost_ms': round(d.estimated_cost_ms, 4),
                    'reason': (d.reason.to_dict()
                               if hasattr(d.reason, 'to_dict')
                               else {'note': str(d.reason)}),
                    'stall_breakdown': d.stall_breakdown,
                }
                for d in engine.decisions
            ]
            data['placement_summary'] = engine.summary()

        filepath = os.path.join(output_dir, f"profile_rank{self.rank}.json")
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        if self.rank == 0:
            sv = data['schema_version']
            print(f"\n  [Profiler] Saved to {output_dir}/profile_rank*.json "
                  f"(schema_version={sv})")
