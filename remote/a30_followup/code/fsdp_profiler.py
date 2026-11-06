import torch
import torch.distributed as dist
import time
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
from contextlib import contextmanager

from memory_manager import AccessPredictor, TensorRecord, TensorRegistry

@dataclass
class MemorySnapshot:
    timestamp_ms: float
    phase: str
    allocated_bytes: int
    reserved_bytes: int
    peak_allocated_bytes: int
    interval_peak_bytes: int = 0

@dataclass
class CommEvent:
    phase: str
    layer_idx: int
    start_ms: float
    end_ms: float
    duration_ms: float
    estimated_bytes: int

@dataclass
class LayerProfile:
    layer_idx: int
    fwd_start_mem_bytes: int = 0
    fwd_end_mem_bytes: int = 0
    fwd_delta_bytes: int = 0
    fwd_duration_ms: float = 0.0
    bwd_start_mem_bytes: int = 0
    bwd_end_mem_bytes: int = 0
    bwd_delta_bytes: int = 0
    bwd_duration_ms: float = 0.0
    attn_scores_bytes: int = 0
    swiglu_group_bytes: int = 0
    qkv_bytes: int = 0
    layer_input_bytes: int = 0

class FSDPMemoryProfiler:

    def __init__(self, model, rank: int, world_size: int, model_config,
                 enabled: bool = True, materializes_attn_scores: bool = True):
        self.model = model
        self.rank = rank
        self.world_size = world_size
        self.model_config = model_config
        self.enabled = enabled
        self.materializes_attn_scores = materializes_attn_scores

        self.snapshots: list[MemorySnapshot] = []
        self.layer_profiles: dict[int, LayerProfile] = {}
        self.registry: TensorRegistry = TensorRegistry()
        self.comm_events: list[CommEvent] = []
        self.timeline: list[dict] = []

        self._step_num = 0
        self._phase = "idle"
        self._start_time = 0.0
        self._hooks = []
        self._layer_timers = {}

        self.engine: Optional[object] = None
        self._bundle_unit: str = 'per_module'

        self._step_peak_running_max: int = 0
        self._interval_peak_enabled: bool = True
        self._step_snapshot_start: int = 0

        self._precompute_activation_sizes()

    def _precompute_activation_sizes(self):
        cfg = self.model_config
        self._cfg_hidden = cfg.hidden_size
        self._cfg_intermediate = cfg.intermediate_size
        self._cfg_num_heads = cfg.num_attention_heads
        self._cfg_num_kv_heads = cfg.num_key_value_heads
        self._cfg_head_dim = cfg.hidden_size // cfg.num_attention_heads
        self._cfg_num_layers = cfg.num_hidden_layers

    def _activation_sizes_for_batch(self, batch_size: int, seq_len: int,
                                     dtype_size: int = 2) -> dict:
        scores_elem_bytes = max(dtype_size, AccessPredictor._SCORES_ELEM_BYTES)
        attn_scores = (batch_size * self._cfg_num_heads * seq_len * seq_len
                       * scores_elem_bytes)
        if not self.materializes_attn_scores:
            attn_scores = (
                2 * batch_size * seq_len * self._cfg_hidden * dtype_size
                + batch_size * seq_len * self._cfg_num_kv_heads
                * self._cfg_head_dim * dtype_size
                + batch_size * self._cfg_num_heads * seq_len * 4
            )

        swiglu_single = batch_size * seq_len * self._cfg_intermediate * dtype_size
        swiglu_group = swiglu_single * AccessPredictor._SWIGLU_SAVED_PLANES

        q_size = batch_size * seq_len * self._cfg_num_heads * self._cfg_head_dim * dtype_size
        kv_size = batch_size * seq_len * self._cfg_num_kv_heads * self._cfg_head_dim * dtype_size
        qkv_total = q_size + 2 * kv_size

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
        raw = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        return max(raw, self._step_peak_running_max)

    def current_interval_ordinal(self) -> tuple:
        if not self.snapshots:
            return (-1, '')
        return (len(self.snapshots) - 1, self.snapshots[-1].phase)

    def peak_interval_ordinal(self) -> tuple:
        window = self.snapshots[self._step_snapshot_start:]
        if not window:
            return (-1, '', 0)
        best = max(range(len(window)),
                   key=lambda i: window[i].interval_peak_bytes)
        return (self._step_snapshot_start + best,
                window[best].phase,
                window[best].interval_peak_bytes)

    def _read_interval_peak_and_reset(self) -> int:
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
        timestamp_ms = self._now_ms()
        interval_peak = self._read_interval_peak_and_reset()
        snap = MemorySnapshot(
            timestamp_ms=timestamp_ms,
            phase=phase,
            allocated_bytes=stats['allocated'],
            reserved_bytes=stats['reserved'],
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

    def attach_hooks(self):
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

        layer_count = 0
        for name, module in self.model.named_modules():
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

    def attach_engine(self, engine) -> None:
        self.engine = engine
        if hasattr(engine, '_step_peak_source'):
            engine._step_peak_source = lambda: self._step_peak_running_max
        if hasattr(engine, '_interval_ordinal_source'):
            engine._interval_ordinal_source = self.current_interval_ordinal
            engine._peak_interval_source = self.peak_interval_ordinal
        if self.rank == 0:
            print("  [Profiler] PlacementEngine attached — hooks will dispatch "
                  "to engine; JSON output will use schema_version=2.")

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
            print(f"  [Profiler] Subpart-B bundle hooks attached: "
                  f"{attn_count} LlamaAttention, {mlp_count} LlamaMLP")

    def _attach_whole_layer_bundle_hooks(self) -> None:
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

    def _make_submodule_pre_hook(self, layer_idx: int, kind: str):
        def hook(module, args, kwargs):
            if self.engine is not None:
                self.engine.enter_bundle(
                    layer_idx, kind,
                    inputs=args, kwargs=kwargs, module=module,
                )
        return hook

    def _make_submodule_post_hook(self):
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

    def _make_fwd_pre_hook(self, layer_idx):
        def hook(module, inputs):
            if not self.enabled:
                return
            self._phase = f"fwd_layer_{layer_idx}"
            mem = self._get_mem()
            self._layer_timers[f"fwd_{layer_idx}"] = self._now_ms()
            self.layer_profiles[layer_idx].fwd_start_mem_bytes = mem['allocated']
            self._take_snapshot(f"pre_fwd_layer_{layer_idx}")

            if isinstance(inputs, tuple) and len(inputs) > 0:
                inp = inputs[0]
                if isinstance(inp, torch.Tensor) and inp.dim() == 3:
                    batch, seq_len, _ = inp.shape
                    dtype_size = inp.element_size()
                    sizes = self._activation_sizes_for_batch(batch, seq_len, dtype_size)
                    prof = self.layer_profiles[layer_idx]
                    prof.attn_scores_bytes = sizes['attn_scores']
                    prof.swiglu_group_bytes = sizes['swiglu_group']
                    prof.qkv_bytes = sizes['qkv']
                    prof.layer_input_bytes = sizes['layer_input']

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

    @contextmanager
    def profile_step(self, step_num: int):
        self._step_num = step_num
        self._start_time = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
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

    def get_memory_breakdown(self) -> dict:
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
        bd = self.get_memory_breakdown()
        local_mb = bd['parameters']['local_shard_bytes'] / (1024**2)
        full_mb = bd['parameters']['estimated_full_bytes'] / (1024**2)

        per_layer_param_bytes = bd['parameters']['local_shard_bytes'] / max(self._cfg_num_layers, 1)
        per_layer_full_bytes = per_layer_param_bytes * self.world_size

        return {
            'world_size': self.world_size,
            'local_shard_mb': round(local_mb, 2),
            'full_model_mb': round(full_mb, 2),
            'sharding_ratio': round(1.0 / self.world_size, 3),
            'per_layer_allgather_mb': round(per_layer_full_bytes / (1024**2), 2),
            'total_allgather_per_step_mb': round(
                2 * self._cfg_num_layers * per_layer_full_bytes / (1024**2), 2
            ),
        }

    def report(self) -> str:
        lines = []
        lines.append("=" * 72)
        lines.append(f"FSDP Memory Profile — Rank {self.rank}/{self.world_size}")
        lines.append("=" * 72)

        if self.snapshots:
            peak = max(self.snapshots, key=lambda s: s.allocated_bytes)
            lines.append(f"\nPeak HBM: {peak.allocated_bytes / (1024**2):.1f} MB "
                        f"at phase '{peak.phase}' ({peak.timestamp_ms:.1f} ms)")

        fsdp = self.get_fsdp_summary()
        lines.append(f"\n--- FSDP Summary ---")
        lines.append(f"  World size:                {fsdp['world_size']}")
        lines.append(f"  Local param shards:        {fsdp['local_shard_mb']:>8.1f} MB")
        lines.append(f"  Full model (estimated):    {fsdp['full_model_mb']:>8.1f} MB")
        lines.append(f"  Per-layer all-gather:      {fsdp['per_layer_allgather_mb']:>8.2f} MB")
        lines.append(f"  Total all-gather/step:     {fsdp['total_allgather_per_step_mb']:>8.1f} MB"
                     f"  (2× for fwd+bwd)")

        bd = self.get_memory_breakdown()
        lines.append(f"\n--- Activation Breakdown (per-step totals) ---")
        lines.append(f"  Attention scores (quadratic): "
                     f"{bd['activations']['attention_scores_bytes'] / (1024**2):>8.1f} MB")
        lines.append(f"  SwiGLU groups (3× intermediate): "
                     f"{bd['activations']['swiglu_groups_bytes'] / (1024**2):>8.1f} MB")
        lines.append(f"  QKV projections (GQA):        "
                     f"{bd['activations']['qkv_bytes'] / (1024**2):>8.1f} MB")

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

        groups = self.get_allocation_groups()
        if groups:
            lines.append(f"\n--- Allocation Groups ---")
            for gid, info in sorted(groups.items()):
                lines.append(f"  {gid}: {info['total_bytes']/(1024**2):.2f} MB "
                            f"({len(info['tensors'])} tensors)")

        lines.append(f"\n--- Top Eviction Candidates (by size) ---")
        activations = [(n, r) for n, r in self.registry.items()
                      if r.category == 'activation']
        activations.sort(key=lambda x: x[1].size_bytes, reverse=True)
        for name, rec in activations[:10]:
            lines.append(f"  {rec.name:>35}: {rec.size_bytes/(1024**2):>7.2f} MB "
                        f"[{rec.component}] L{rec.layer_idx}")

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
        os.makedirs(output_dir, exist_ok=True)

        engine_attached = self.engine is not None

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

        if engine_attached:
            engine = self.engine
            cal = engine._calibration
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
            data['hbm_budget_bytes'] = engine._budget.total_bytes
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
