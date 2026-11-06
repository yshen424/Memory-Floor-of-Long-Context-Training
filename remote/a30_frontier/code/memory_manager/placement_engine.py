import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import torch

from .access_predictor import AccessPredictor, AccessSchedule
from .calibration import CalibrationResult
from .fsdp_coordinator import NoOpCoordinator
from .pinned_pool import PinnedBuffer, PinnedBufferPool
from .tensor_registry import TensorRecord, TensorRegistry

from .decisions import PlacementDecision, PlacementDecisionReason
from .hbm_budget import HBMBudget
from .stream_manager import InflightOp, MemoryStreamManager

@dataclass
class PackedEvict:
    host_buf: PinnedBuffer
    shape: tuple
    dtype: Any
    device: Any
    evict_op: InflightOp
    nbytes: int
    pack_id: str
    layer_idx: int
    kind: str
    prefetch_op: Optional[InflightOp] = None
    dst: Optional[torch.Tensor] = None
    evict_ordinal: Optional[int] = None
    d2h_done_ordinal: Optional[int] = None
    restore_ordinal: Optional[int] = None
    src_stride: Optional[tuple] = None
    src_storage_offset: Optional[int] = None
    src_contiguous: Optional[bool] = None
    ref: Optional[torch.Tensor] = None

class PlacementEngine:

    EVENTS_PER_FWD_LAYER = 8
    EVENTS_PER_BWD_LAYER = 16

    _OVERBOOK_REPORT_RATIO = 0.5

    def __init__(
        self,
        registry: TensorRegistry,
        predictor: AccessPredictor,
        hbm_budget: HBMBudget,
        buffer_pool: PinnedBufferPool,
        mem_stream,
        calibration: CalibrationResult,
        fsdp_coord: NoOpCoordinator,
        prefetch_window_layers: int = 2,
        dry_run: bool = True,
        verbose: bool = True,
        overlap_safety_factor: float = 0.8,
        default_layer_fwd_ms: float = 30.0,
        default_layer_bwd_ms: float = 60.0,
        per_op_log_max: int = 2000,
        window_debit: bool = True,
        observed_relief: bool = True,
        bundle_unit: str = 'per_module',
    ) -> None:
        if bundle_unit not in ('per_module', 'whole_layer'):
            raise ValueError(
                f"bundle_unit must be 'per_module' or 'whole_layer', "
                f"got {bundle_unit!r}")
        self._bundle_unit = bundle_unit
        self._foreign_provenance_excluded: frozenset[str] = (
            frozenset({'module', 'kwarg'}) if bundle_unit == 'whole_layer'
            else frozenset({'module', 'input', 'kwarg'}))
        if bundle_unit == 'whole_layer':
            self._BUNDLE_COMPONENTS = {'layer': self._WHOLE_LAYER_COMPONENTS}
            self.EVENTS_PER_FWD_LAYER = 15
            self.EVENTS_PER_BWD_LAYER = 24
        self._registry = registry
        self._predictor = predictor
        self._budget = hbm_budget
        self._pool = buffer_pool
        self._mem_stream = mem_stream
        self._calibration = calibration
        self._coord = fsdp_coord
        self._prefetch_window_layers = prefetch_window_layers
        self._dry_run = dry_run
        self._verbose = verbose
        self._step_peak_source = None
        self._overlap_safety_factor: float = float(overlap_safety_factor)
        self._default_layer_fwd_ms: float = float(default_layer_fwd_ms)
        self._default_layer_bwd_ms: float = float(default_layer_bwd_ms)

        self._schedule: AccessSchedule = predictor.build_schedule()
        self._registry_populated: bool = False
        self._num_layers: int = predictor.num_layers

        self._current_step: int = 0
        self._training_step_count: int = 0

        self._decisions: list[PlacementDecision] = []
        self._decisions_at_step_start: int = 0
        self._plan_ns: int = 0
        self._plan_calls: int = 0
        self._inflight_by_tensor: dict[str, InflightOp] = {}

        self._per_layer_fwd_ms: dict[int, float] = {}
        self._per_layer_bwd_ms: dict[int, float] = {}

        self._total_recomputed_flops: int = 0

        self._total_pcie_ms_evict: float = 0.0
        self._total_pcie_ms_prefetch: float = 0.0
        self._total_bytes_evict: int = 0
        self._total_bytes_prefetch: int = 0
        self._total_timed_ops_evict: int = 0
        self._total_timed_ops_prefetch: int = 0

        self._per_op_log_max: int = int(per_op_log_max)
        self._per_op_latency_log: list[dict] = []
        self._nbytes_hist: dict[str, dict[int, int]] = {
            'evict': {}, 'prefetch': {}}
        self._ratio_obs_over_idle_evict: list[float] = []
        self._ratio_obs_over_idle_prefetch: list[float] = []
        self._ratio_obs_over_cont_evict: list[float] = []
        self._ratio_obs_over_cont_prefetch: list[float] = []

        self._current_bundle: Optional[tuple[int, str]] = None

        self._pre_decisions_by_layer: dict[int, list[PlacementDecision]] = {}

        self._engine_live: bool = False
        self._packed_evict_bundles: dict[tuple[int, str], bool] = {}
        self._packed_events: list[tuple] = []
        self._packed_zero_elem_count: int = 0
        self._evict_realized_bytes: int = 0
        self._evict_realized_count: int = 0
        self._evict_denied_bytes: int = 0
        self._evict_denied_count: int = 0
        self._evict_denial_log: list[dict] = []
        self._evict_denial_log_max: int = 512
        self._evict_skipped_nonhbm_bytes: int = 0
        self._evict_skipped_nonhbm_count: int = 0
        self._evict_skipped_nonhbm_kinds: set[str] = set()
        self._no_relief_bytes: int = 0
        self._no_relief_count: int = 0
        self._no_relief_notes: set[str] = set()
        self._infeasible_steps: list[dict] = []
        self._training_step: int = 0
        self._evict_filter: str = 'all'
        self._evict_filter_kind: Optional[str] = None
        self._evict_filter_bytes: Optional[set] = None
        self._evict_filter_bytes_exclude: bool = False
        self._evict_filtered_bytes: int = 0
        self._evict_filtered_count: int = 0
        self._evict_filtered_classes: dict[str, int] = {}
        self._layout_census_enabled: bool = False
        self._layout_census: dict[str, dict] = {}
        self._verify_restore: bool = False
        self._verify_log: dict[str, dict] = {}
        self._verify_checked: int = 0
        self._verify_failed: int = 0
        _msdev = getattr(mem_stream, '_device', None)
        self._evict_device_type: Optional[str] = (
            torch.device(_msdev).type if _msdev is not None else None)
        self._evict_partial_bundles: set[tuple[int, str]] = set()
        self._evict_staged_bundles: set[tuple[int, str]] = set()
        self._evict_denied_bundles: set[tuple[int, str]] = set()
        self._strict_evict: bool = False
        self._evict_denial_warned: set[tuple] = set()

        self._interval_ordinal_source = None
        self._peak_interval_source = None
        self._record_by_pack_id: dict[str, PackedEvict] = {}
        self._absence_log: list[dict] = []
        self._absence_log_max: int = 512
        self._absence_bytes_total: int = 0
        self._absence_bytes_covering_peak: int = 0
        self._absence_empty_window_count: int = 0
        self._absence_unrestored_count: int = 0
        self._packed_by_bundle: dict[tuple[int, str], list[PackedEvict]] = {}
        self._prefetch_issued_bundles: set[tuple[int, str]] = set()
        self._prefetch_ahead_issued: int = 0
        self._prefetch_ahead_hits: int = 0
        self._prefetch_on_demand: int = 0

        self._recompute_observed_ms: dict[tuple, list[float]] = {}

        self._recompute_class_ms: dict[str, list[float]] = {}
        self._class_probe_issued: set = set()

        self._hbm_drift_log: list[dict] = []

        self._force_action: str = 'none'
        self._force_stride: int = 0

        self._planner_excluded_tids: set[str] = set()
        self._planner_restore_ordinals: dict[str, int] = {}
        self._planner_picks_this_step: int = 0
        self._planner_max_picks_per_step: int = (
            self._num_layers * len(self._BUNDLE_COMPONENTS))

        self._packed_recompute_bundles: dict[tuple[int, str], bool] = {}
        self._recompute_input_cache: dict[tuple[int, str], tuple[tuple, dict]] = {}
        self._recompute_module_cache: dict[tuple[int, str], Any] = {}
        self._recompute_save_counters: dict[tuple[int, str], int] = {}
        self._recompute_tensor_cache: dict[tuple[int, str], list] = {}
        self._in_recompute: bool = False

        self._window_debit_bytes: dict[int, float] = {}
        self._window_debit_enabled: bool = window_debit
        self._window_debit_committed_bytes: int = 0
        self._window_debit_refusals: int = 0

        self._obs_bundle_bytes_cur: dict[tuple[int, str], int] = {}
        self._obs_bundle_members_cur: dict[tuple[int, str], list] = {}
        self._obs_bundle_bytes: dict[tuple[int, str], int] = {}
        self._obs_bundle_members: dict[tuple[int, str], list] = {}

        self._overbooked_bundles: dict[tuple[int, str], dict] = {}
        self._overbook_warned: bool = False

        self._obs_bundle_owned_cur: dict[tuple[int, str], int] = {}
        self._obs_bundle_owned: dict[tuple[int, str], int] = {}
        self._obs_bundle_dropped_cur: set[tuple[int, str]] = set()
        self._obs_declined_bundles: set[tuple[int, str]] = set()
        self._bundle_seen_ptrs: set[int] = set()
        self._bundle_foreign_ptrs: set[int] = set()
        self._bundle_foreign_src: dict[int, str] = {}
        self._observed_relief: bool = observed_relief

        self._obs_bundle_foreign_cur: dict[tuple[int, str], dict[str, int]] = {}
        self._obs_bundle_foreign: dict[tuple[int, str], dict[str, int]] = {}
        self._obs_bundle_distinct_cur: dict[tuple[int, str], int] = {}
        self._obs_bundle_distinct: dict[tuple[int, str], int] = {}

        if verbose:
            print(f"[PlacementEngine] init DONE: dry_run={dry_run}, "
                  f"budget={hbm_budget.total_bytes / (1024 ** 3):.1f} GB, "
                  f"prefetch_window={prefetch_window_layers} layers, "
                  f"num_layers={self._num_layers}, "
                  f"schedule_events={len(self._schedule)}, "
                  f"overlap_safety_factor={self._overlap_safety_factor}")

    def on_training_step_begin(self) -> None:
        self._current_step = 0
        self._training_step += 1
        self._decisions_at_step_start = len(self._decisions)
        self._reap_inflight()
        self._pre_decisions_by_layer.clear()
        self._packed_evict_bundles.clear()
        self._packed_events.clear()
        for records in self._packed_by_bundle.values():
            for r in records:
                if r.prefetch_op is None:
                    self._pool.release(r.host_buf)
        self._packed_by_bundle.clear()
        self._record_by_pack_id.clear()
        self._prefetch_issued_bundles.clear()
        self._packed_recompute_bundles.clear()
        self._recompute_input_cache.clear()
        self._recompute_module_cache.clear()
        self._recompute_save_counters.clear()
        self._recompute_tensor_cache.clear()
        self._planner_excluded_tids.clear()
        self._planner_restore_ordinals.clear()
        self._planner_picks_this_step = 0
        self._window_debit_bytes.clear()
        self._obs_bundle_bytes_cur.clear()
        self._obs_bundle_members_cur.clear()
        self._obs_bundle_owned_cur.clear()
        self._obs_bundle_foreign_cur.clear()
        self._obs_bundle_distinct_cur.clear()
        self._obs_bundle_dropped_cur.clear()

    def _promote_observed_relief(self) -> None:
        if not self._observed_relief:
            return
        bm = self._schedule.bytes_by_tensor
        whole_layer = self._bundle_unit == 'whole_layer'
        for key, owned in self._obs_bundle_owned_cur.items():
            layer_idx, kind = key
            analytic = (
                bm.get(self._predictor.residual_id(layer_idx, kind), 0)
                if whole_layer else
                sum(bm.get(tid, 0) for tid in self._named_tids(
                    self._bundle_tensor_ids(layer_idx, kind))))
            if analytic > 0 and owned < analytic * self._OVERBOOK_REPORT_RATIO:
                self._overbooked_bundles[key] = {
                    'step': self._training_step_count,
                    'observed_owned_bytes': int(owned),
                    'analytic_bytes': int(analytic),
                    'ratio': round(analytic / max(1, int(owned)), 2),
                    'trusted': key not in self._obs_bundle_dropped_cur,
                }
            if key in self._obs_bundle_dropped_cur:
                self._obs_declined_bundles.add(key)
                continue
            self._obs_bundle_owned[key] = owned
            bm[self._predictor.residual_id(layer_idx, kind)] = (
                int(owned) if whole_layer else max(0, int(owned) - analytic))
        if self._overbooked_bundles and not self._overbook_warned:
            self._overbook_warned = True
            worst = max(self._overbooked_bundles.items(),
                        key=lambda kv: kv[1]['ratio'])
            print(f"[PlacementEngine] WARNING: analytic byte model over-books "
                  f"{len(self._overbooked_bundles)} bundle(s). Worst: "
                  f"L{worst[0][0]}_{worst[0][1]} booked "
                  f"{worst[1]['analytic_bytes'] / (1024 ** 2):.1f} MiB against "
                  f"{worst[1]['observed_owned_bytes'] / (1024 ** 2):.1f} MiB "
                  f"observed ({worst[1]['ratio']}x). The residual correction "
                  f"is clamped at zero and CANNOT walk this back, so every "
                  f"peak projection this run is fiction. Check that the byte "
                  f"model matches --attn_impl.")

    def on_training_step_end(self) -> None:
        self._reap_inflight()

        for key, nbytes in self._obs_bundle_bytes_cur.items():
            self._obs_bundle_bytes[key] = nbytes
            self._obs_bundle_members[key] = list(
                self._obs_bundle_members_cur.get(key, ()))
            self._obs_bundle_foreign[key] = dict(
                self._obs_bundle_foreign_cur.get(key, {}))
            self._obs_bundle_distinct[key] = (
                self._obs_bundle_distinct_cur.get(key, 0))
        self._promote_observed_relief()

        if not self._registry_populated:
            self._predictor.populate_registry(self._registry)
            self._registry_populated = True

        self._reconcile_absence()

        self._update_hbm_drift()

        if self._verbose:
            recent = self._decisions[self._decisions_at_step_start:]
            counts: dict[str, int] = defaultdict(int)
            for d in recent:
                counts[d.action] += 1
            print(f"[PlacementEngine] step {self._training_step_count}: "
                  f"decisions={dict(counts)}")
        self._training_step_count += 1

    def _reconcile_absence(self) -> None:
        if self._peak_interval_source is None:
            return
        peak_ordinal, peak_phase, peak_bytes = self._peak_interval_source()
        if peak_ordinal < 0:
            return
        for records in self._packed_by_bundle.values():
            for r in records:
                if r.d2h_done_ordinal is None:
                    self._absence_unrestored_count += 1
                    continue
                first_absent = r.d2h_done_ordinal + 1
                last_absent = ((r.restore_ordinal - 1)
                               if r.restore_ordinal is not None
                               else peak_ordinal)
                span = max(0, last_absent - first_absent + 1)
                covered = span > 0 and first_absent <= peak_ordinal <= last_absent
                self._absence_bytes_total += r.nbytes
                if covered:
                    self._absence_bytes_covering_peak += r.nbytes
                if span == 0:
                    self._absence_empty_window_count += 1
                if len(self._absence_log) < self._absence_log_max:
                    self._absence_log.append({
                        'step': self._training_step_count,
                        'layer': r.layer_idx,
                        'kind': r.kind,
                        'nbytes': r.nbytes,
                        'evict_ordinal': r.evict_ordinal,
                        'd2h_done_ordinal': r.d2h_done_ordinal,
                        'restore_ordinal': r.restore_ordinal,
                        'absent_intervals': span,
                        'peak_ordinal': peak_ordinal,
                        'peak_phase': peak_phase,
                        'peak_bytes': peak_bytes,
                        'covered_peak': covered,
                    })

    def _update_hbm_drift(self) -> None:
        observed_peak = self._read_observed_peak_and_reset()

        recent = self._decisions[self._decisions_at_step_start:]
        predicted_peak = 0
        predicted_anchor = 0
        for d in recent:
            r = d.reason
            if r is None:
                continue
            p = getattr(r, 'projected_peak_bytes', 0)
            if p > predicted_peak:
                predicted_peak = p
                predicted_anchor = getattr(r, 'anchor_live_bytes', 0)

        if predicted_peak <= 0:
            reason = 'no_planner_projection'
        elif observed_peak <= 0:
            reason = 'observed_zero'
        else:
            reason = 'observed'

        self._hbm_drift_log.append({
            'step': self._training_step_count,
            'observed_peak_mb': observed_peak / (1024 * 1024),
            'predicted_peak_mb': predicted_peak / (1024 * 1024),
            'predicted_anchor_mb': predicted_anchor / (1024 * 1024),
            'step_ratio': 1.0,
            'cumulative_correction': 1.0,
            'updated': False,
            'reason': reason,
        })

        if self._verbose:
            e = self._hbm_drift_log[-1]
            print(f"[PlacementEngine] peak step {self._training_step_count}: "
                  f"observed={e['observed_peak_mb']:.1f} MB, "
                  f"predicted={e['predicted_peak_mb']:.1f} MB, "
                  f"reason={e['reason']}")

    def _read_observed_peak_and_reset(self) -> int:
        if not torch.cuda.is_available():
            return 0
        peak = int(torch.cuda.max_memory_allocated())
        if self._step_peak_source is not None:
            peak = max(peak, int(self._step_peak_source()))
        torch.cuda.reset_peak_memory_stats()
        return peak

    _FOREIGN_CONTAINER_DEPTH: int = 2

    @staticmethod
    def _collect_storage_ptrs(obj, out: set, depth: int = None) -> None:
        if depth is None:
            depth = PlacementEngine._FOREIGN_CONTAINER_DEPTH
        if torch.is_tensor(obj):
            out.add(obj.untyped_storage().data_ptr())
        elif depth > 0:
            if isinstance(obj, (tuple, list)):
                for v in obj:
                    PlacementEngine._collect_storage_ptrs(v, out, depth - 1)
            elif isinstance(obj, dict):
                for v in obj.values():
                    PlacementEngine._collect_storage_ptrs(v, out, depth - 1)

    @staticmethod
    def _module_storage_ptrs(module) -> set:
        ptrs: set = set()
        for sub in module.modules():
            for p in sub._parameters.values():
                if p is not None:
                    ptrs.add(p.untyped_storage().data_ptr())
            for b in sub._buffers.values():
                if b is not None:
                    ptrs.add(b.untyped_storage().data_ptr())
            for v in sub.__dict__.values():
                if torch.is_tensor(v):
                    ptrs.add(v.untyped_storage().data_ptr())
        return ptrs

    def enter_bundle(
        self, layer_idx: int, kind: str,
        inputs: Optional[tuple] = None,
        kwargs: Optional[dict] = None,
        module: Optional[Any] = None,
    ) -> None:
        if self._in_recompute:
            return
        self._current_bundle = (layer_idx, kind)
        bundle = (layer_idx, kind)
        self._bundle_seen_ptrs = set()
        foreign: set[int] = set()
        src: dict[int, str] = {}
        if module is not None:
            for _p in self._module_storage_ptrs(module):
                src.setdefault(_p, 'module')
        _from_inputs: set[int] = set()
        self._collect_storage_ptrs(inputs, _from_inputs)
        for _p in _from_inputs:
            src.setdefault(_p, 'input')
        _from_kwargs: set[int] = set()
        self._collect_storage_ptrs(kwargs, _from_kwargs)
        for _p in _from_kwargs:
            src.setdefault(_p, 'kwarg')
        src.pop(0, None)
        foreign = {p for p, prov in src.items()
                   if prov in self._foreign_provenance_excluded}
        self._bundle_foreign_ptrs = foreign
        self._bundle_foreign_src = src
        if (self._engine_live
                and self._packed_recompute_bundles.get(bundle, False)
                and module is not None):
            sanitized_kwargs: dict = (
                dict(kwargs) if kwargs is not None else {})
            if 'past_key_value' in sanitized_kwargs:
                sanitized_kwargs['past_key_value'] = None
            if 'use_cache' in sanitized_kwargs:
                sanitized_kwargs['use_cache'] = False
            self._recompute_input_cache[bundle] = (
                tuple(inputs) if inputs is not None else (),
                sanitized_kwargs,
            )
            self._recompute_module_cache[bundle] = module

    def exit_bundle(self) -> None:
        if self._in_recompute:
            return
        self._current_bundle = None

    @property
    def current_bundle(self) -> Optional[tuple[int, str]]:
        return self._current_bundle

    @property
    def engine_live(self) -> bool:
        return self._engine_live

    @engine_live.setter
    def engine_live(self, value: bool) -> None:
        self._engine_live = bool(value)

    @property
    def force_action(self) -> str:
        return self._force_action

    @force_action.setter
    def force_action(self, value: str) -> None:
        v = str(value)
        if v not in ('none', 'evict', 'recompute'):
            raise ValueError(
                f"force_action must be 'none' | 'evict' | 'recompute', "
                f"got {value!r}")
        self._force_action = v

    @property
    def force_stride(self) -> int:
        return self._force_stride

    @force_stride.setter
    def force_stride(self, value: int) -> None:
        v = int(value)
        if v < 0:
            raise ValueError(f"force_stride must be >= 0, got {value!r}")
        self._force_stride = v

    @property
    def evict_filter(self) -> str:
        return self._evict_filter

    @evict_filter.setter
    def evict_filter(self, value: str) -> None:
        v = str(value).strip()
        kind, byteset, exclude = None, None, False
        aliases = {'attn': 'attention'}
        kinds = (*PlacementEngine._BUNDLE_COMPONENTS, 'layer')
        if v in kinds or v in aliases:
            kind = aliases.get(v, v)
        elif v.startswith('bytes=') or v.startswith('notbytes='):
            exclude = v.startswith('notbytes=')
            raw = v.split('=', 1)[1]
            try:
                byteset = {int(t) for t in raw.split(',') if t.strip()}
            except ValueError:
                raise ValueError(
                    f"evict_filter byte list must be integers, got {raw!r}")
            if not byteset:
                raise ValueError(f"evict_filter byte list is empty: {v!r}")
        elif v != 'all':
            raise ValueError(
                f"evict_filter must be 'all' | "
                f"{' | '.join(repr(k) for k in kinds)} | "
                f"'bytes=N,...' | 'notbytes=N,...', got {value!r}")
        self._evict_filter = v
        self._evict_filter_kind = kind
        self._evict_filter_bytes = byteset
        self._evict_filter_bytes_exclude = exclude

    def _evict_filter_allows(self, kind: str, nbytes: int) -> bool:
        if self._evict_filter_kind is not None:
            return kind == self._evict_filter_kind
        if self._evict_filter_bytes is not None:
            hit = nbytes in self._evict_filter_bytes
            return (not hit) if self._evict_filter_bytes_exclude else hit
        return True

    @property
    def layout_census(self) -> bool:
        return self._layout_census_enabled

    @layout_census.setter
    def layout_census(self, value: bool) -> None:
        self._layout_census_enabled = bool(value)

    @property
    def verify_restore(self) -> bool:
        return self._verify_restore

    @verify_restore.setter
    def verify_restore(self, value: bool) -> None:
        self._verify_restore = bool(value)

    def _diag_class(self, kind: str, tensor) -> str:
        return (f'{kind}/{tensor.dtype}/'
                f'{tensor.numel() * tensor.element_size()}')

    def _record_layout(self, kind: str, tensor) -> None:
        key = self._diag_class(kind, tensor)
        row = self._layout_census.get(key)
        if row is None:
            row = {'n': 0, 'noncontiguous': 0, 'offset_nonzero': 0,
                   'storage_larger': 0, 'shape': tuple(tensor.shape),
                   'strides': []}
            self._layout_census[key] = row
        row['n'] += 1
        if not tensor.is_contiguous():
            row['noncontiguous'] += 1
        if tensor.storage_offset() != 0:
            row['offset_nonzero'] += 1
        if (tensor.untyped_storage().nbytes()
                > tensor.numel() * tensor.element_size()):
            row['storage_larger'] += 1
        st = str(tuple(tensor.stride()))
        if st not in row['strides'] and len(row['strides']) < 8:
            row['strides'].append(st)

    def _verify_one(self, record: PackedEvict, dst) -> None:
        ref = record.ref
        record.ref = None
        key = self._diag_class(record.kind, ref)
        row = self._verify_log.get(key)
        if row is None:
            row = {'n': 0, 'value_mismatch': 0, 'layout_mismatch': 0,
                   'max_abs_diff': 0.0, 'nan_restored': 0, 'nan_original': 0,
                   'shape': tuple(ref.shape),
                   'src_stride': str(record.src_stride),
                   'restored_stride': str(tuple(dst.stride()))}
            self._verify_log[key] = row
        row['n'] += 1
        self._verify_checked += 1
        failed = False
        if tuple(dst.stride()) != record.src_stride or record.src_storage_offset:
            row['layout_mismatch'] += 1
            failed = True
        if dst.dtype.is_floating_point:
            nan_dst = int(torch.isnan(dst).sum().item())
            nan_ref = int(torch.isnan(ref).sum().item())
            row['nan_restored'] += nan_dst
            row['nan_original'] += nan_ref
            diff = (dst.float() - ref.float()).abs()
            mx = float(torch.nan_to_num(
                diff, nan=float('inf')).max().item()) if diff.numel() else 0.0
        else:
            mx = float((dst != ref).sum().item())
        if mx > 0.0:
            row['value_mismatch'] += 1
            failed = True
        row['max_abs_diff'] = max(row['max_abs_diff'], mx)
        if failed:
            self._verify_failed += 1

    _PACKED_SENTINEL: str = '_step4.5_packed_v1'
    _PACKED_RECOMPUTE_SENTINEL: str = '_step6a_recompute_v1'

    def _current_ordinal(self) -> Optional[int]:
        if self._interval_ordinal_source is None:
            return None
        ordinal, _phase = self._interval_ordinal_source()
        return ordinal if ordinal >= 0 else None

    def _record_evict_denial(self, layer_idx: int, kind: str,
                             nbytes: int, reason: str) -> None:
        bundle = (layer_idx, kind)
        self._evict_denied_bytes += nbytes
        self._evict_denied_count += 1
        self._evict_denied_bundles.add(bundle)
        if bundle in self._evict_staged_bundles:
            self._evict_partial_bundles.add(bundle)
        if len(self._evict_denial_log) < self._evict_denial_log_max:
            self._evict_denial_log.append({
                'step': self._current_step,
                'layer': layer_idx,
                'kind': kind,
                'nbytes': nbytes,
                'reason': reason,
            })
        msg = (f"planned eviction denied at pack time: L{layer_idx} {kind}, "
               f"{nbytes / (1024**2):.1f} MiB, reason={reason}. The tensor "
               f"stays resident and its relief was already booked.")
        if self._strict_evict:
            raise RuntimeError(f"[PlacementEngine] strict_evict: {msg}")
        warn_key = (kind, reason.split(':')[0], self._current_step)
        if warn_key not in self._evict_denial_warned:
            self._evict_denial_warned.add(warn_key)
            print(f"[PlacementEngine] WARNING: {msg}")

    def _note_no_relief_under_pressure(
            self, layer_idx: int, kind: str, nbytes: int, note: str) -> None:
        self._no_relief_bytes += int(nbytes)
        self._no_relief_count += 1
        self._no_relief_notes.add(note)
        msg = (f"bundle under pressure got no relief: L{layer_idx} {kind}, "
               f"{nbytes / (1024**2):.1f} MiB, note={note}. The projection is "
               f"over budget and no action removes these bytes.")
        if self._strict_evict:
            raise RuntimeError(f"[PlacementEngine] strict_evict: {msg}")
        warn_key = ('no_relief', kind, note, self._current_step)
        if warn_key not in self._evict_denial_warned:
            self._evict_denial_warned.add(warn_key)
            print(f"[PlacementEngine] WARNING: {msg}")

    def _note_infeasible(self, peak_bytes: int, budget_bytes: int,
                         peak_event_idx: int, anchor_bytes: int = 0) -> None:
        shortfall = int(peak_bytes - budget_bytes)
        self._infeasible_steps.append({
            'training_step': self._training_step,
            'projected_peak_bytes': int(peak_bytes),
            'budget_bytes': int(budget_bytes),
            'shortfall_bytes': shortfall,
            'peak_event_idx': int(peak_event_idx),
            'anchor_live_bytes': int(anchor_bytes),
            'picks': self._planner_picks_this_step,
        })
        msg = (f"projection still over capacity after every available pick: "
               f"{peak_bytes / (1024**2):.1f} MiB against a "
               f"{budget_bytes / (1024**2):.1f} MiB budget, short by "
               f"{shortfall / (1024**2):.1f} MiB with "
               f"{self._planner_picks_this_step} bundles picked "
               f"(anchor {anchor_bytes / (1024**2):.1f} MiB).")
        if self._strict_evict:
            raise RuntimeError(f"[PlacementEngine] strict_evict: {msg}")
        warn_key = ('infeasible', self._training_step)
        if warn_key not in self._evict_denial_warned:
            self._evict_denial_warned.add(warn_key)
            print(f"[PlacementEngine] WARNING: {msg}")

    def pack_fn(self, tensor):
        if not self._engine_live:
            return tensor
        bundle = self._current_bundle
        if bundle is None:
            return tensor
        if tensor.numel() == 0:
            self._packed_zero_elem_count += 1
            return tensor
        obs_nbytes = tensor.numel() * tensor.element_size()
        self._obs_bundle_bytes_cur[bundle] = (
            self._obs_bundle_bytes_cur.get(bundle, 0) + obs_nbytes)
        self._obs_bundle_members_cur.setdefault(bundle, []).append(obs_nbytes)
        _storage = tensor.untyped_storage()
        _ptr = _storage.data_ptr()
        if _ptr not in self._bundle_seen_ptrs:
            self._bundle_seen_ptrs.add(_ptr)
            self._obs_bundle_distinct_cur[bundle] = (
                self._obs_bundle_distinct_cur.get(bundle, 0) + 1)
            if _ptr not in self._bundle_foreign_ptrs:
                self._obs_bundle_owned_cur[bundle] = (
                    self._obs_bundle_owned_cur.get(bundle, 0)
                    + _storage.nbytes())
            else:
                _hist = self._obs_bundle_foreign_cur.setdefault(bundle, {})
                _k = (f"{_storage.nbytes()}|"
                      f"{self._bundle_foreign_src.get(_ptr, 'unknown')}")
                _hist[_k] = _hist.get(_k, 0) + 1
        if (self._packed_recompute_bundles.get(bundle, False)
                and bundle in self._recompute_module_cache):
            idx = self._recompute_save_counters.get(bundle, 0)
            self._recompute_save_counters[bundle] = idx + 1
            self._obs_bundle_dropped_cur.add(bundle)
            return (
                self._PACKED_RECOMPUTE_SENTINEL,
                bundle,
                idx,
                self._current_step,
                tuple(tensor.shape),
                tensor.dtype,
            )
        if not self._packed_evict_bundles.get(bundle, False):
            return tensor
        nbytes = tensor.numel() * tensor.element_size()
        layer_idx, kind = bundle
        if (self._evict_device_type is not None
                and tensor.device.type != self._evict_device_type):
            self._evict_skipped_nonhbm_bytes += nbytes
            self._evict_skipped_nonhbm_count += 1
            self._evict_skipped_nonhbm_kinds.add(
                f'{tensor.device.type}/{tensor.dtype}/dim{tensor.dim()}')
            return tensor
        if not self._evict_filter_allows(kind, nbytes):
            self._evict_filtered_bytes += nbytes
            self._evict_filtered_count += 1
            key = self._diag_class(kind, tensor)
            self._evict_filtered_classes[key] = (
                self._evict_filtered_classes.get(key, 0) + 1)
            return tensor
        if not self._pool.has_capacity(nbytes, kind=kind):
            self._record_evict_denial(layer_idx, kind, nbytes, 'pool_no_capacity')
            return tensor

        try:
            host_buf = self._pool.acquire(nbytes)
        except RuntimeError as exc:
            self._record_evict_denial(layer_idx, kind, nbytes,
                                      f'acquire_failed: {exc}')
            return tensor
        pack_id = (
            f"pack_step{self._current_step}_L{layer_idx}_{kind}_"
            f"n{len(self._packed_events)}"
        )
        if self._layout_census_enabled:
            self._record_layout(kind, tensor)
        ref = (tensor.detach().clone() if self._verify_restore else None)
        op = self._mem_stream.submit_evict(
            tensor_id=pack_id, nbytes=nbytes, step=self._current_step,
            src_tensor=tensor, host_buf=host_buf,
        )
        self._packed_events.append((layer_idx, kind, nbytes, pack_id))
        self._evict_realized_bytes += nbytes
        self._evict_realized_count += 1
        self._evict_staged_bundles.add(bundle)
        if bundle in self._evict_denied_bundles:
            self._evict_partial_bundles.add(bundle)
        record = PackedEvict(
            host_buf=host_buf,
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            device=tensor.device,
            evict_op=op,
            nbytes=nbytes,
            pack_id=pack_id,
            layer_idx=layer_idx,
            kind=kind,
            evict_ordinal=self._current_ordinal(),
            src_stride=tuple(tensor.stride()),
            src_storage_offset=tensor.storage_offset(),
            src_contiguous=tensor.is_contiguous(),
            ref=ref,
        )
        self._packed_by_bundle.setdefault(bundle, []).append(record)
        self._record_by_pack_id[pack_id] = record
        return (self._PACKED_SENTINEL, record)

    def unpack_fn(self, packed):
        if (isinstance(packed, tuple)
                and len(packed) > 0
                and packed[0] == self._PACKED_RECOMPUTE_SENTINEL):
            _, bundle, idx, _step, exp_shape, exp_dtype = packed
            cache = self._recompute_tensor_cache.get(bundle)
            if cache is None:
                cache = self._rerun_bundle_forward(bundle)
                self._recompute_tensor_cache[bundle] = cache
            if idx >= len(cache):
                raise RuntimeError(
                    f"[PlacementEngine] recompute replay for bundle "
                    f"{bundle} captured only {len(cache)} tensors but "
                    f"unpack asked for index {idx}. The replay's save "
                    f"sequence is shorter than the original forward's — "
                    f"check that the submodule's forward path doesn't "
                    f"branch on grad mode or stateful kwargs."
                )
            t = cache[idx]
            if tuple(t.shape) != exp_shape or t.dtype != exp_dtype:
                raise RuntimeError(
                    f"[PlacementEngine] recompute replay tensor mismatch "
                    f"for bundle {bundle} idx {idx}: expected "
                    f"shape={exp_shape} dtype={exp_dtype}, got "
                    f"shape={tuple(t.shape)} dtype={t.dtype}. The "
                    f"submodule's save sequence diverged between the "
                    f"original forward and the replay."
                )
            return t
        if not (isinstance(packed, tuple)
                and len(packed) > 0
                and packed[0] == self._PACKED_SENTINEL):
            return packed
        _, record = packed
        if record.dst is None:
            self._issue_prefetch_for_record(record)
            self._prefetch_on_demand += 1
        else:
            self._prefetch_ahead_hits += 1
        self._mem_stream.wait_compute_on(record.prefetch_op)
        dst = record.dst
        if record.ref is not None:
            self._verify_one(record, dst)
        record.dst = None
        return dst

    def _issue_prefetch_for_record(self, record: PackedEvict) -> None:
        if record.prefetch_op is not None:
            return
        dst = torch.empty(record.shape, dtype=record.dtype,
                          device=record.device)
        record.restore_ordinal = self._current_ordinal()
        record.dst = dst
        record.prefetch_op = self._mem_stream.submit_prefetch(
            tensor_id=record.pack_id + '_h2d', nbytes=record.nbytes,
            step=self._current_step, host_buf=record.host_buf,
            dst_tensor=dst,
        )

    def _issue_physical_prefetch(self, layer_idx: int) -> int:
        issued = 0
        for depth in range(1, self._prefetch_window_layers + 1):
            target = layer_idx - depth
            if target < 0:
                break
            for kind in self._BUNDLE_COMPONENTS:
                bundle = (target, kind)
                if bundle in self._prefetch_issued_bundles:
                    continue
                records = self._packed_by_bundle.get(bundle)
                if not records:
                    continue
                for record in records:
                    self._issue_prefetch_for_record(record)
                    issued += 1
                self._prefetch_issued_bundles.add(bundle)
        self._prefetch_ahead_issued += issued
        return issued

    def _rerun_bundle_forward(self, bundle: tuple[int, str]) -> list:
        inputs, kwargs = self._recompute_input_cache[bundle]
        module = self._recompute_module_cache[bundle]
        layer_idx, bundle_kind = bundle
        captured: list = []

        def record_pack(t):
            d = t.detach()
            captured.append(d)
            return d

        def identity_unpack(t):
            return t

        use_cuda_events = torch.cuda.is_available()
        if use_cuda_events:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            import time as _time
            t0 = _time.perf_counter()

        self._in_recompute = True
        try:
            with torch.enable_grad(), \
                    torch.autograd.graph.saved_tensors_hooks(
                        record_pack, identity_unpack):
                _ = module(*inputs, **kwargs)
        finally:
            self._in_recompute = False

        if use_cuda_events:
            end.record()
            torch.cuda.synchronize()
            wall_ms = start.elapsed_time(end)
        else:
            wall_ms = (_time.perf_counter() - t0) * 1000.0

        self.record_recompute_timing(layer_idx, wall_ms, kind=bundle_kind)
        return captured

    def on_pre_forward(self, layer_idx: int) -> list[PlacementDecision]:
        self._coord.on_pre_forward(layer_idx)
        self._current_step += self.EVENTS_PER_FWD_LAYER
        self._reap_inflight()
        _plan_t0 = time.perf_counter_ns()
        decisions = self._plan_for_layer_forward(layer_idx)
        self._plan_ns += time.perf_counter_ns() - _plan_t0
        self._plan_calls += 1
        self._pre_decisions_by_layer[layer_idx] = decisions
        self._decisions.extend(decisions)
        return decisions

    _BUNDLE_COMPONENTS: dict[str, tuple[str, ...]] = {
        'attention': ('q', 'k', 'v', 'attn_scores'),
        'mlp': ('swiglu_gate', 'swiglu_up', 'swiglu_product'),
    }

    _WHOLE_LAYER_COMPONENTS: tuple[str, ...] = (
        'q', 'k', 'v', 'attn_scores',
        'swiglu_gate', 'swiglu_up', 'swiglu_product',
    )

    def _kind_for_tensor_id(self, key: str) -> Optional[str]:
        for kind, comps in self._BUNDLE_COMPONENTS.items():
            for c in comps:
                if key.endswith(f'_{c}'):
                    return kind
        return None

    def _bundle_tensor_ids(self, layer_idx: int, kind: str) -> set[str]:
        comps = self._BUNDLE_COMPONENTS.get(kind, ())
        tids = {f'act_layer_{layer_idx}_{c}' for c in comps}
        if tids:
            tids.add(self._predictor.residual_id(layer_idx, kind))
        return tids

    def _bundle_byte_estimate(self, tensor_ids: set[str], key=None) -> int:
        if key is not None:
            obs = self._obs_bundle_bytes.get(key)
            if obs:
                return obs
        if self._bundle_unit == 'whole_layer':
            return self._predictor.boundary_bytes()
        bm = self._schedule.bytes_by_tensor
        return sum(bm.get(tid, 0) for tid in self._named_tids(tensor_ids))

    def _bundle_member_sizes(self, tensor_ids: set[str], key=None) -> list:
        if key is not None:
            obs = self._obs_bundle_members.get(key)
            if obs:
                return list(obs)
        if self._bundle_unit == 'whole_layer':
            return [self._predictor.boundary_bytes()]
        bm = self._schedule.bytes_by_tensor
        return [n for n in (bm.get(tid, 0)
                            for tid in self._named_tids(tensor_ids)) if n > 0]

    def _named_tids(self, tensor_ids) -> set[str]:
        return {t for t in tensor_ids if not t.endswith('_residual')}

    def _bundle_flop_estimate(self, bundle_bytes: int) -> int:
        return bundle_bytes * 2

    def _bundle_retain_decision(
        self,
        layer_idx: int,
        kind: str,
        note: str,
        projected_peak_bytes: int = 0,
        projected_peak_event_idx: int = 0,
        bundle_contribution_bytes: int = 0,
        anchor_live_bytes: int = 0,
    ) -> PlacementDecision:
        return PlacementDecision(
            step_idx=self._current_step,
            layer_idx=layer_idx,
            tensor_id=f'bundle_L{layer_idx}_{kind}',
            action='retain',
            estimated_cost_ms=0.0,
            reason=PlacementDecisionReason(
                note=note,
                projected_peak_bytes=int(projected_peak_bytes),
                projected_peak_event_idx=int(projected_peak_event_idx),
                bundle_contribution_bytes=int(bundle_contribution_bytes),
                anchor_live_bytes=int(anchor_live_bytes),
            ),
        )

    def _restore_ordinal_for_layer(self, layer_idx: int) -> Optional[int]:
        restore_layer = min(layer_idx + self._prefetch_window_layers,
                            self._num_layers - 1)
        if restore_layer < layer_idx:
            restore_layer = layer_idx
        return self._schedule.bwd_event_for_layer(restore_layer)

    def _plan_for_layer_forward(
        self, layer_idx: int,
    ) -> list[PlacementDecision]:
        new_decisions: list[PlacementDecision] = []

        if self._force_action != 'none':
            return self._plan_forced(layer_idx)

        current_live = self._budget.physical_allocated()
        budget = self.effective_budget()
        L_bwd_event = self._schedule.bwd_event_for_layer(layer_idx)

        if L_bwd_event is None:
            for kind in self._BUNDLE_COMPONENTS:
                new_decisions.append(self._bundle_retain_decision(
                    layer_idx, kind, note='no_bwd_event'))
                if self._engine_live:
                    self._packed_evict_bundles[(layer_idx, kind)] = False
                    self._packed_recompute_bundles[(layer_idx, kind)] = False
            return new_decisions

        fwd_lo = self._schedule.fwd_create_event_for_layer(layer_idx)
        if fwd_lo is None:
            fwd_lo = self._current_step

        peak_bytes, peak_step = self._schedule.peak_in_range(
            fwd_lo, L_bwd_event, current_live,
            exclude_tensors=self._planner_excluded_tids,
            restore_at=self._planner_restore_ordinals)

        if peak_bytes <= budget:
            for kind in self._BUNDLE_COMPONENTS:
                new_decisions.append(self._bundle_retain_decision(
                    layer_idx, kind, note='no_pressure',
                    projected_peak_bytes=peak_bytes,
                    projected_peak_event_idx=peak_step,
                    anchor_live_bytes=current_live))
                if self._engine_live:
                    self._packed_evict_bundles[(layer_idx, kind)] = False
                    self._packed_recompute_bundles[(layer_idx, kind)] = False
            return new_decisions

        excluded = set(self._planner_excluded_tids)
        restores = dict(self._planner_restore_ordinals)
        layer_restore = self._restore_ordinal_for_layer(layer_idx)
        running_peak = peak_bytes
        running_peak_step = peak_step
        for kind in self._BUNDLE_COMPONENTS:
            tids = self._bundle_tensor_ids(layer_idx, kind)
            bundle_bytes = self._bundle_byte_estimate(tids, key=(layer_idx, kind))
            if bundle_bytes == 0:
                new_decisions.append(self._bundle_retain_decision(
                    layer_idx, kind, note='no_byte_model',
                    projected_peak_bytes=running_peak,
                    projected_peak_event_idx=running_peak_step,
                    anchor_live_bytes=current_live))
                if self._engine_live:
                    self._packed_evict_bundles[(layer_idx, kind)] = False
                    self._packed_recompute_bundles[(layer_idx, kind)] = False
                continue

            if running_peak <= budget:
                new_decisions.append(self._bundle_retain_decision(
                    layer_idx, kind, note='budget_satisfied',
                    projected_peak_bytes=running_peak,
                    projected_peak_event_idx=running_peak_step,
                    anchor_live_bytes=current_live))
                if self._engine_live:
                    self._packed_evict_bundles[(layer_idx, kind)] = False
                    self._packed_recompute_bundles[(layer_idx, kind)] = False
                continue

            if self._planner_picks_this_step >= self._planner_max_picks_per_step:
                new_decisions.append(self._bundle_retain_decision(
                    layer_idx, kind, note='pick_cap_reached',
                    projected_peak_bytes=running_peak,
                    projected_peak_event_idx=running_peak_step,
                    anchor_live_bytes=current_live))
                self._note_no_relief_under_pressure(
                    layer_idx, kind, bundle_bytes, 'pick_cap_reached')
                if self._engine_live:
                    self._packed_evict_bundles[(layer_idx, kind)] = False
                    self._packed_recompute_bundles[(layer_idx, kind)] = False
                continue

            candidate_excluded = excluded | tids
            candidate_restores = dict(restores)
            if layer_restore is not None:
                for tid in tids:
                    candidate_restores[tid] = layer_restore
            cand_peak, cand_peak_step = self._schedule.peak_in_range(
                fwd_lo, L_bwd_event, current_live,
                exclude_tensors=candidate_excluded,
                restore_at=candidate_restores)
            contribution = running_peak - cand_peak

            if contribution <= 0:
                early = (layer_restore is not None
                         and layer_restore <= running_peak_step)
                new_decisions.append(self._bundle_retain_decision(
                    layer_idx, kind,
                    note=('no_contribution_restored_before_peak' if early
                          else 'no_contribution'),
                    projected_peak_bytes=running_peak,
                    projected_peak_event_idx=running_peak_step,
                    bundle_contribution_bytes=contribution,
                    anchor_live_bytes=current_live))
                if self._engine_live:
                    self._packed_evict_bundles[(layer_idx, kind)] = False
                    self._packed_recompute_bundles[(layer_idx, kind)] = False
                continue

            flops = self._bundle_flop_estimate(bundle_bytes)
            rep_tid = next(iter(self._named_tids(tids)))
            action, reason = self._decide_evict_vs_recompute(
                size_bytes=bundle_bytes,
                flops=flops,
                layer_idx=layer_idx,
                representative_tensor_id=rep_tid,
                decision_step=self._current_step,
                kind=kind,
                member_sizes=self._bundle_member_sizes(tids, key=(layer_idx, kind)),
            )
            reason.projected_peak_bytes = int(cand_peak)
            reason.projected_peak_event_idx = int(cand_peak_step)
            reason.bundle_contribution_bytes = int(contribution)
            reason.anchor_live_bytes = int(current_live)

            if action == 'recompute':
                cost = reason.exposed_recompute_ms
                self._total_recomputed_flops += int(flops)
            elif action == 'retain':
                cost = 0.0
            else:
                cost = reason.exposed_evict_ms

            if self._engine_live:
                self._packed_evict_bundles[(layer_idx, kind)] = (
                    action == 'evict_to_cpu')
                self._packed_recompute_bundles[(layer_idx, kind)] = (
                    action == 'recompute')

            new_decisions.append(PlacementDecision(
                step_idx=self._current_step,
                layer_idx=layer_idx,
                tensor_id=f'bundle_L{layer_idx}_{kind}',
                action=action,
                estimated_cost_ms=cost,
                reason=reason,
            ))

            if action == 'retain':
                self._note_no_relief_under_pressure(
                    layer_idx, kind, bundle_bytes, reason.note)
                continue

            excluded = candidate_excluded
            restores = candidate_restores
            running_peak = cand_peak
            running_peak_step = cand_peak_step
            self._planner_picks_this_step += 1

        self._planner_excluded_tids = excluded
        self._planner_restore_ordinals = restores

        if layer_idx == self._num_layers - 1 and running_peak > budget:
            self._note_infeasible(running_peak, budget, running_peak_step,
                                  current_live)

        return new_decisions

    def _plan_forced(self, layer_idx: int) -> list[PlacementDecision]:
        action_out = ('evict_to_cpu' if self._force_action == 'evict'
                      else 'recompute')
        if self._force_stride > 0 and self._force_action == 'recompute':
            action_out = ('evict_to_cpu'
                          if layer_idx % self._force_stride == 0
                          else 'recompute')
        new_decisions: list[PlacementDecision] = []
        for kind in self._BUNDLE_COMPONENTS:
            if action_out == 'recompute' and kind == 'layer':
                new_decisions.append(self._bundle_retain_decision(
                    layer_idx, kind,
                    note='recompute_unavailable_whole_layer'))
                if self._engine_live:
                    self._packed_evict_bundles[(layer_idx, kind)] = False
                    self._packed_recompute_bundles[(layer_idx, kind)] = False
                continue

            tids = self._bundle_tensor_ids(layer_idx, kind)
            bundle_bytes = self._bundle_byte_estimate(tids, key=(layer_idx, kind))

            if bundle_bytes == 0:
                new_decisions.append(self._bundle_retain_decision(
                    layer_idx, kind, note='no_byte_model_force_skip'))
                if self._engine_live:
                    self._packed_evict_bundles[(layer_idx, kind)] = False
                    self._packed_recompute_bundles[(layer_idx, kind)] = False
                continue

            flops = self._bundle_flop_estimate(bundle_bytes)
            rep_tid = next(iter(self._named_tids(tids)))
            _decided_action, reason = self._decide_evict_vs_recompute(
                size_bytes=bundle_bytes,
                flops=flops,
                layer_idx=layer_idx,
                representative_tensor_id=rep_tid,
                decision_step=self._current_step,
                kind=kind,
                member_sizes=self._bundle_member_sizes(tids, key=(layer_idx, kind)),
                commit=False,
            )
            reason.note = (
                f'forced_stride{self._force_stride}_'
                f'{"evict" if action_out == "evict_to_cpu" else "recompute"}'
                if self._force_stride > 0 and self._force_action == 'recompute'
                else f'forced_{self._force_action}')
            if action_out == 'recompute':
                cost = reason.exposed_recompute_ms
                self._total_recomputed_flops += int(flops)
            else:
                cost = reason.exposed_evict_ms

            if self._engine_live:
                self._packed_evict_bundles[(layer_idx, kind)] = (
                    action_out == 'evict_to_cpu')
                self._packed_recompute_bundles[(layer_idx, kind)] = (
                    action_out == 'recompute')

            new_decisions.append(PlacementDecision(
                step_idx=self._current_step,
                layer_idx=layer_idx,
                tensor_id=f'bundle_L{layer_idx}_{kind}',
                action=action_out,
                estimated_cost_ms=cost,
                reason=reason,
            ))

        return new_decisions

    def on_post_forward(self, layer_idx: int) -> list[PlacementDecision]:
        self._reap_inflight()
        new_decisions: list[PlacementDecision] = []

        if self._engine_live:
            for key, rec in self._activations_for_layer(layer_idx):
                comp = rec.component or ''
                if comp.startswith('attention'):
                    bundle = (layer_idx, 'attention')
                elif comp.startswith('swiglu'):
                    bundle = (layer_idx, 'mlp')
                else:
                    continue
                if self._packed_evict_bundles.get(bundle, False):
                    if rec.tier == 'hbm':
                        self._registry.set_tier(key, 'cpu_dram')
                elif self._packed_recompute_bundles.get(bundle, False):
                    if rec.tier == 'hbm':
                        self._registry.set_tier(key, 'recompute_marker')
                        rec.checkpoint_enabled = True

        window = self._prefetch_window_layers * self.EVENTS_PER_FWD_LAYER
        new_decisions.extend(self._prefetch_lookahead(window))

        self._coord.on_post_forward(layer_idx)

        self._decisions.extend(new_decisions)
        return new_decisions

    def on_pre_backward(self, layer_idx: int) -> list[PlacementDecision]:
        self._coord.on_pre_backward(layer_idx)
        self._reap_inflight()
        self._issue_physical_prefetch(layer_idx)
        new_decisions: list[PlacementDecision] = []

        for key, rec in self._activations_for_layer(layer_idx):
            new_decisions.append(self._handle_pre_backward_tensor(key, rec, layer_idx))

        self._decisions.extend(new_decisions)
        return new_decisions

    def on_post_backward(self, layer_idx: int) -> list[PlacementDecision]:
        self._current_step += self.EVENTS_PER_BWD_LAYER
        self._reap_inflight()
        new_decisions: list[PlacementDecision] = []

        for key, rec in self._activations_for_layer(layer_idx):
            if rec.tier != 'freed':
                self._registry.set_tier(key, 'freed')
            new_decisions.append(PlacementDecision(
                step_idx=self._current_step, layer_idx=layer_idx,
                tensor_id=key, action='no_op',
                estimated_cost_ms=0.0,
                reason=PlacementDecisionReason(note='free_after_bwd')))

        for kind in self._BUNDLE_COMPONENTS:
            bundle = (layer_idx, kind)
            self._recompute_tensor_cache.pop(bundle, None)
            self._recompute_input_cache.pop(bundle, None)
            self._recompute_module_cache.pop(bundle, None)
            self._recompute_save_counters.pop(bundle, None)

        window = self._prefetch_window_layers * self.EVENTS_PER_BWD_LAYER
        new_decisions.extend(self._prefetch_lookahead(window))

        self._coord.on_post_backward(layer_idx)

        self._decisions.extend(new_decisions)
        return new_decisions

    @property
    def decisions(self) -> list[PlacementDecision]:
        return list(self._decisions)

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def total_recomputed_flops(self) -> int:
        return self._total_recomputed_flops

    def effective_budget(self) -> int:
        return self._budget.total_bytes

    @property
    def hbm_drift_enabled(self) -> bool:
        return False

    @property
    def hbm_drift_log(self) -> list[dict]:
        return list(self._hbm_drift_log)

    def summary(self) -> dict:
        counts: dict[str, int] = defaultdict(int)
        for d in self._decisions:
            counts[d.action] += 1
        exposed_stall_pcie_ms = 0.0
        exposed_stall_alloc_ms = 0.0
        for d in self._decisions:
            if d.action == 'stall' and d.stall_breakdown:
                exposed_stall_pcie_ms += d.stall_breakdown.get('pcie_ms', 0.0)
                exposed_stall_alloc_ms += d.stall_breakdown.get('allocator_ms', 0.0)
        def _bw(total_bytes: int, total_ms: float) -> float:
            return (total_bytes / (total_ms / 1000.0)) if total_ms > 0 else 0.0
        def _stats(samples: list[float]) -> dict:
            if not samples:
                return {'n': 0, 'mean': 0.0, 'p95': 0.0}
            xs = sorted(samples)
            p95 = xs[min(len(xs) - 1, int(len(xs) * 0.95))]
            return {
                'n': len(samples),
                'mean': round(sum(samples) / len(samples), 4),
                'p95': round(p95, 4),
            }
        idle_evict = _stats(self._ratio_obs_over_idle_evict)
        idle_prefetch = _stats(self._ratio_obs_over_idle_prefetch)
        cont_evict = _stats(self._ratio_obs_over_cont_evict)
        cont_prefetch = _stats(self._ratio_obs_over_cont_prefetch)
        drift_flag = (cont_evict['p95'] > 2.0
                      or cont_prefetch['p95'] > 2.0)
        if len(self._per_op_latency_log) > self._per_op_log_max:
            per_op_log = self._per_op_latency_log[-self._per_op_log_max:]
            per_op_truncated = True
        else:
            per_op_log = list(self._per_op_latency_log)
            per_op_truncated = False
        return {
            'total_decisions': len(self._decisions),
            'by_action': dict(counts),
            'training_steps': self._training_step_count,
            'planner_decision': {
                'calls': self._plan_calls,
                'total_ms': round(self._plan_ns / 1e6, 4),
                'per_step_ms': round(
                    self._plan_ns / 1e6 / max(1, self._training_step_count), 4),
            },
            'total_recomputed_flops': self._total_recomputed_flops,
            'exposed_stall_pcie_ms': round(exposed_stall_pcie_ms, 4),
            'exposed_stall_allocator_ms': round(exposed_stall_alloc_ms, 4),
            'observed_pcie_evict': {
                'ops': self._total_timed_ops_evict,
                'bytes': self._total_bytes_evict,
                'pcie_ms': round(self._total_pcie_ms_evict, 4),
                'observed_bw_bytes_per_s': round(
                    _bw(self._total_bytes_evict, self._total_pcie_ms_evict), 1),
            },
            'observed_pcie_prefetch': {
                'ops': self._total_timed_ops_prefetch,
                'bytes': self._total_bytes_prefetch,
                'pcie_ms': round(self._total_pcie_ms_prefetch, 4),
                'observed_bw_bytes_per_s': round(
                    _bw(self._total_bytes_prefetch, self._total_pcie_ms_prefetch), 1),
            },
            'latency_drift': {
                'obs_over_idle_evict': idle_evict,
                'obs_over_idle_prefetch': idle_prefetch,
                'obs_over_contention_evict': cont_evict,
                'obs_over_contention_prefetch': cont_prefetch,
                'calibration_drift_flag': bool(drift_flag),
                'drift_threshold_p95': 2.0,
            },
            'hbm_drift': {
                'enabled': False,
                'per_step_ratio_clamp': [1.0, 1.0],
                'cumulative_correction_final': 1.0,
                'effective_budget_bytes_final': self.effective_budget(),
                'log': list(self._hbm_drift_log),
            },
            'window_debit': {
                'enabled': self._window_debit_enabled,
                'committed_bytes': self._window_debit_committed_bytes,
                'refusals': self._window_debit_refusals,
            },
            'observed_bundle_bytes': {
                f'L{L}_{kind}': {
                    'bytes': nbytes,
                    'members': len(self._obs_bundle_members.get((L, kind), ())),
                    'owned_bytes': self._obs_bundle_owned.get((L, kind), 0),
                    'owned_bytes_trusted':
                        (L, kind) not in self._obs_declined_bundles,
                    'residual_bytes': self._schedule.bytes_by_tensor.get(
                        self._predictor.residual_id(L, kind), 0),
                    'analytic_bytes': self._bundle_byte_estimate(
                        self._bundle_tensor_ids(L, kind)),
                    'foreign_storages': dict(sorted(
                        self._obs_bundle_foreign.get((L, kind), {}).items())),
                    'distinct_storages': self._obs_bundle_distinct.get(
                        (L, kind), 0),
                }
                for (L, kind), nbytes in sorted(self._obs_bundle_bytes.items())
            },
            'overbooked_bundles': {
                f'L{L}_{kind}': info
                for (L, kind), info in sorted(self._overbooked_bundles.items())
            },
            'observed_relief': self._observed_relief,
            'force_action': self._force_action,
            'force_stride': self._force_stride,
            'evict_filter': self._evict_filter,
            'layout_census': self._layout_census_enabled,
            'verify_restore': self._verify_restore,
            'recompute_ms_seed': {
                'ratio': 1.0,
                'prefill_n': 0,
                'per_layer': {
                    str(L): {
                        'seed_samples': [],
                        'observed_samples': [
                            s
                            for (kL, _k), ss
                            in sorted(self._recompute_observed_ms.items(),
                                      key=lambda kv: str(kv[0]))
                            if kL == L
                            for s in ss
                        ],
                    }
                    for L in sorted({kL for kL, _k in self._recompute_observed_ms})
                },
                'per_layer_kind': {
                    f'{kL}:{k}': list(ss)
                    for (kL, k), ss in sorted(
                        self._recompute_observed_ms.items(),
                        key=lambda kv: str(kv[0]))
                },
            },
            'recompute_class_seed': {
                'enabled': True,
                'probes_issued': sorted(
                    f'{k}@step{s}' for k, s in self._class_probe_issued),
                'samples': {
                    k: {
                        'n': len(v),
                        'mean_ms': (sum(v) / len(v)) if v else None,
                        'min_ms': min(v) if v else None,
                        'max_ms': max(v) if v else None,
                    }
                    for k, v in sorted(self._recompute_class_ms.items())
                },
            },
            'per_op_latency': per_op_log,
            'per_op_latency_truncated': per_op_truncated,
            'nbytes_histogram': {
                d: {str(k): v for k, v in sorted(h.items())}
                for d, h in sorted(self._nbytes_hist.items())
            },
            'pool': self._pool.telemetry(),
            'evict_realization': {
                'strict': self._strict_evict,
                'realized_ops': self._evict_realized_count,
                'realized_bytes': self._evict_realized_bytes,
                'denied_ops': self._evict_denied_count,
                'denied_bytes': self._evict_denied_bytes,
                'realized_fraction': (
                    self._evict_realized_bytes
                    / (self._evict_realized_bytes + self._evict_denied_bytes)
                    if (self._evict_realized_bytes + self._evict_denied_bytes) > 0
                    else None
                ),
                'partial_bundles': sorted(
                    f'L{L}_{k}' for L, k in self._evict_partial_bundles),
                'no_relief_ops': self._no_relief_count,
                'no_relief_bytes': self._no_relief_bytes,
                'no_relief_notes': sorted(self._no_relief_notes),
                'infeasible_steps': list(self._infeasible_steps),
                'denials': list(self._evict_denial_log),
                'denials_truncated':
                    self._evict_denied_count > len(self._evict_denial_log),
                'skipped_nonhbm_ops': self._evict_skipped_nonhbm_count,
                'skipped_nonhbm_bytes': self._evict_skipped_nonhbm_bytes,
                'skipped_nonhbm_kinds':
                    sorted(self._evict_skipped_nonhbm_kinds),
                'zero_elem_passthrough': self._packed_zero_elem_count,
                'filter': self._evict_filter,
                'filtered_ops': self._evict_filtered_count,
                'filtered_bytes': self._evict_filtered_bytes,
                'filtered_classes': dict(
                    sorted(self._evict_filtered_classes.items())),
            },
            'evict_layout_census': {
                k: dict(v) for k, v in sorted(self._layout_census.items())},
            'restore_verification': {
                'enabled': self._verify_restore,
                'checked': self._verify_checked,
                'failed': self._verify_failed,
                'by_class': {
                    k: dict(v) for k, v in sorted(self._verify_log.items())},
            },
            'absence': {
                'measured': self._peak_interval_source is not None,
                'staged_bytes': self._absence_bytes_total,
                'bytes_covering_peak': self._absence_bytes_covering_peak,
                'peak_coverage_fraction': (
                    self._absence_bytes_covering_peak / self._absence_bytes_total
                    if self._absence_bytes_total > 0 else None
                ),
                'empty_windows': self._absence_empty_window_count,
                'unrestored_at_step_end': self._absence_unrestored_count,
                'prefetch_window_layers': self._prefetch_window_layers,
                'log': list(self._absence_log),
                'log_truncated': len(self._absence_log) >= self._absence_log_max,
            },
            'prefetch_ahead': {
                'issued': self._prefetch_ahead_issued,
                'hits': self._prefetch_ahead_hits,
                'on_demand': self._prefetch_on_demand,
                'window_layers': self._prefetch_window_layers,
            },
        }

    def print_summary(self) -> None:
        s = self.summary()
        print(f"[PlacementEngine] training end: "
              f"total_decisions={s['total_decisions']}, "
              f"by_action={s['by_action']}, "
              f"steps={s['training_steps']}, "
              f"recomputed_tflops={s['total_recomputed_flops'] / 1e12:.3f}")
        ev = s['observed_pcie_evict']
        pf = s['observed_pcie_prefetch']
        if ev['ops'] > 0 or pf['ops'] > 0:
            print(f"[PlacementEngine] observed PCIe: "
                  f"evict ops={ev['ops']} bw={ev['observed_bw_bytes_per_s']/1e9:.2f} GB/s, "
                  f"prefetch ops={pf['ops']} bw={pf['observed_bw_bytes_per_s']/1e9:.2f} GB/s")
        er = s['evict_realization']
        if er['realized_fraction'] is not None:
            print(f"[PlacementEngine] evict realization: "
                  f"{er['realized_fraction'] * 100:.1f}% of planned mass "
                  f"({er['realized_bytes'] / (1024**3):.3f} GiB staged, "
                  f"{er['denied_bytes'] / (1024**3):.3f} GiB denied in "
                  f"{er['denied_ops']} denials), "
                  f"partial bundles={len(er['partial_bundles'])}")
        if er['skipped_nonhbm_ops']:
            print(f"[PlacementEngine] evict skipped {er['skipped_nonhbm_ops']} "
                  f"non-HBM saves ({er['skipped_nonhbm_bytes']} B, "
                  f"{', '.join(er['skipped_nonhbm_kinds'])}) — these hold no "
                  f"device memory, so skipping them costs no relief.")
        ab = s['absence']
        if ab['peak_coverage_fraction'] is not None:
            print(f"[PlacementEngine] absence at peak: "
                  f"{ab['peak_coverage_fraction'] * 100:.1f}% of staged bytes "
                  f"gone during the peak interval "
                  f"({ab['bytes_covering_peak'] / (1024**3):.3f} of "
                  f"{ab['staged_bytes'] / (1024**3):.3f} GiB), "
                  f"empty windows={ab['empty_windows']}, "
                  f"W={ab['prefetch_window_layers']}")
        ld = s['latency_drift']
        ce = ld['obs_over_contention_evict']
        cp = ld['obs_over_contention_prefetch']
        if ce['n'] > 0 or cp['n'] > 0:
            flag = ' DRIFT' if ld['calibration_drift_flag'] else ''
            print(f"[PlacementEngine] cost-model accuracy: "
                  f"evict obs/contention p95={ce['p95']:.2f} (n={ce['n']}), "
                  f"prefetch obs/contention p95={cp['p95']:.2f} (n={cp['n']})"
                  f"{flag}")

    def _activations_for_layer(self, layer_idx: int):
        for key, rec in self._registry.items():
            if rec.category == 'activation' and rec.layer_idx == layer_idx:
                yield key, rec

    def _estimate_recompute_flops(self, rec: TensorRecord) -> int:
        if rec.recompute_cost_flops > 0:
            return rec.recompute_cost_flops
        return rec.size_bytes * 2

    def _evict_cost_ms(self, size_bytes: int) -> float:
        bw = self._calibration.realistic_round_trip_bytes_per_s
        if bw <= 0:
            return float('inf')
        return (size_bytes / bw) * 1000.0

    def _recompute_cost_ms(
        self, flops: int, layer_idx: Optional[int] = None,
        kind: Optional[str] = None,
    ) -> float:
        fps = self._calibration.gpu_flops_per_s
        if layer_idx is not None:
            real = self._recompute_observed_ms.get((layer_idx, kind), ())
            if real:
                return sum(real) / len(real)
        if kind is not None:
            cls = self._recompute_class_ms.get(kind, ())
            if cls:
                return sum(cls) / len(cls)
        if fps <= 0:
            return float('inf')
        return (flops / fps) * 1000.0

    def _apply_decision(
        self, decision: PlacementDecision, rec: TensorRecord,
    ) -> None:
        key = decision.tensor_id
        if decision.action == 'evict_to_cpu':
            op = self._mem_stream.submit_evict(
                key, rec.size_bytes, self._current_step)
            self._inflight_by_tensor[key] = op
            self._registry.set_tier(key, 'cpu_dram')
            self._registry.set_in_flight(key, True)
        elif decision.action == 'recompute':
            self._registry.set_tier(key, 'recompute_marker')
            rec.checkpoint_enabled = True
        elif decision.action == 'prefetch_to_hbm':
            op = self._mem_stream.submit_prefetch(
                key, rec.size_bytes, self._current_step)
            self._inflight_by_tensor[key] = op
            self._registry.set_in_flight(key, True)

    def _prefetch_lookahead(self, window_size_events: int) -> list[PlacementDecision]:
        decisions: list[PlacementDecision] = []
        lo = self._current_step
        hi = self._current_step + window_size_events
        seen: set[str] = set()

        for ev in self._schedule.events_in_range(lo, hi):
            if ev.access_type != 'read':
                continue
            if not ev.tensor_id.startswith('act_'):
                continue
            if ev.tensor_id in seen:
                continue
            rec = self._registry.get(ev.tensor_id)
            if rec is None:
                continue
            if rec.tier == 'cpu_dram' and not rec.in_flight:
                cost_ms = self._evict_cost_ms(rec.size_bytes)
                d = PlacementDecision(
                    step_idx=self._current_step, layer_idx=ev.layer_idx,
                    tensor_id=ev.tensor_id, action='prefetch_to_hbm',
                    estimated_cost_ms=cost_ms,
                    reason=PlacementDecisionReason(
                        note=f'lookahead_for_layer_{ev.layer_idx}'))
                self._apply_decision(d, rec)
                decisions.append(d)
                seen.add(ev.tensor_id)
        return decisions

    def _handle_pre_backward_tensor(
        self, key: str, rec: TensorRecord, layer_idx: int,
    ) -> PlacementDecision:
        if rec.tier == 'hbm' and not rec.in_flight:
            return PlacementDecision(
                step_idx=self._current_step, layer_idx=layer_idx,
                tensor_id=key, action='retain',
                estimated_cost_ms=0.0,
                reason=PlacementDecisionReason(note='hbm_resident'))

        if rec.tier == 'cpu_dram' and rec.in_flight:
            op = self._inflight_by_tensor.get(key)
            wait_ms = self._mem_stream.wait(op) if op is not None else 0.0
            self._registry.set_tier(key, 'hbm')
            self._registry.set_in_flight(key, False)
            self._inflight_by_tensor.pop(key, None)
            if wait_ms > 0:
                pcie_ms_real = (self._mem_stream.op_pcie_ms(op)
                                if op is not None else None)
                if pcie_ms_real is not None:
                    pcie_part = min(pcie_ms_real, wait_ms)
                    alloc_part = max(0.0, wait_ms - pcie_ms_real)
                    breakdown_source = 'measured'
                else:
                    pcie_part = wait_ms * 0.8
                    alloc_part = wait_ms * 0.2
                    breakdown_source = 'heuristic_80_20_dryrun'
                return PlacementDecision(
                    step_idx=self._current_step, layer_idx=layer_idx,
                    tensor_id=key, action='stall',
                    estimated_cost_ms=wait_ms,
                    reason=PlacementDecisionReason(note='prefetch_incomplete'),
                    stall_breakdown={
                        'pcie_ms': pcie_part,
                        'allocator_ms': alloc_part,
                        'source': breakdown_source,
                    })
            return PlacementDecision(
                step_idx=self._current_step, layer_idx=layer_idx,
                tensor_id=key, action='retain',
                estimated_cost_ms=0.0,
                reason=PlacementDecisionReason(note='prefetch_ready'))

        if rec.tier == 'cpu_dram' and not rec.in_flight:
            evict_ms = self._evict_cost_ms(rec.size_bytes)
            self._registry.set_tier(key, 'hbm')
            return PlacementDecision(
                step_idx=self._current_step, layer_idx=layer_idx,
                tensor_id=key, action='stall',
                estimated_cost_ms=evict_ms,
                reason=PlacementDecisionReason(note='policy_miss'),
                stall_breakdown={'pcie_ms': evict_ms, 'allocator_ms': 0.0,
                                 'source': 'estimated_no_op'})

        if rec.tier == 'recompute_marker':
            flops = self._estimate_recompute_flops(rec)
            ms = self._recompute_cost_ms(
                flops, layer_idx=layer_idx,
                kind=self._kind_for_tensor_id(key))
            self._registry.set_tier(key, 'hbm')
            self._registry.set_in_flight(key, False)
            return PlacementDecision(
                step_idx=self._current_step, layer_idx=layer_idx,
                tensor_id=key, action='recompute',
                estimated_cost_ms=ms,
                reason=PlacementDecisionReason(
                    recompute_ms=ms, note='checkpointed_layer'))

        return PlacementDecision(
            step_idx=self._current_step, layer_idx=layer_idx,
            tensor_id=key, action='no_op',
            estimated_cost_ms=0.0,
            reason=PlacementDecisionReason(note=f'unexpected_tier_{rec.tier}'))

    def _reap_inflight(self) -> None:
        done = self._mem_stream.reap_completed()
        for op in done:
            pcie_ms = self._mem_stream.op_pcie_ms(op)
            if pcie_ms is not None:
                if op.direction == 'evict':
                    self._total_pcie_ms_evict += pcie_ms
                    self._total_bytes_evict += op.nbytes
                    self._total_timed_ops_evict += 1
                else:
                    self._total_pcie_ms_prefetch += pcie_ms
                    self._total_bytes_prefetch += op.nbytes
                    self._total_timed_ops_prefetch += 1

                hist = self._nbytes_hist.setdefault(op.direction, {})
                hist[op.nbytes] = hist.get(op.nbytes, 0) + 1
                self._per_op_latency_log.append({
                    'tensor_id': op.tensor_id,
                    'direction': op.direction,
                    'nbytes': op.nbytes,
                    'observed_ms': round(pcie_ms, 4),
                    'predicted_ms_idle': (
                        round(op.predicted_ms_idle, 4)
                        if op.predicted_ms_idle is not None else None),
                    'predicted_ms_contention': (
                        round(op.predicted_ms_contention, 4)
                        if op.predicted_ms_contention is not None else None),
                })
                if op.predicted_ms_idle is not None and op.predicted_ms_idle > 0:
                    r = pcie_ms / op.predicted_ms_idle
                    if op.direction == 'evict':
                        self._ratio_obs_over_idle_evict.append(r)
                    else:
                        self._ratio_obs_over_idle_prefetch.append(r)
                if (op.predicted_ms_contention is not None
                        and op.predicted_ms_contention > 0):
                    r = pcie_ms / op.predicted_ms_contention
                    if op.direction == 'evict':
                        self._ratio_obs_over_cont_evict.append(r)
                    else:
                        self._ratio_obs_over_cont_prefetch.append(r)
            if op.direction == 'prefetch' and op.buffer is not None:
                self._pool.release(op.buffer)
            if op.direction == 'evict':
                rec = self._record_by_pack_id.get(op.tensor_id)
                if rec is not None and rec.d2h_done_ordinal is None:
                    rec.d2h_done_ordinal = self._current_ordinal()
            try:
                if op.direction == 'prefetch':
                    rec = self._registry.get(op.tensor_id)
                    if rec is not None and rec.tier != 'freed':
                        self._registry.set_tier(op.tensor_id, 'hbm')
                self._registry.set_in_flight(op.tensor_id, False)
            except KeyError:
                pass
            self._inflight_by_tensor.pop(op.tensor_id, None)

    def record_layer_timing(
        self, layer_idx: int, phase: str, ms: float,
    ) -> None:
        if ms is None or ms <= 0:
            return
        if phase == 'fwd' or phase == 'recompute':
            self._per_layer_fwd_ms[layer_idx] = float(ms)
        elif phase == 'bwd':
            self._per_layer_bwd_ms[layer_idx] = float(ms)

    def record_recompute_timing(
        self, layer_idx: int, wall_ms: float, kind: str = None,
    ) -> None:
        if wall_ms is None or wall_ms <= 0:
            return
        self._recompute_observed_ms.setdefault(
            (layer_idx, kind), []).append(float(wall_ms))
        if kind is not None:
            self._recompute_class_ms.setdefault(kind, []).append(
                float(wall_ms))

    def _avg_known_layer_ms(self, phase: str) -> float:
        if phase == 'fwd':
            table = self._per_layer_fwd_ms
            default = self._default_layer_fwd_ms
        else:
            table = self._per_layer_bwd_ms
            default = self._default_layer_bwd_ms
        if not table:
            return default
        return sum(table.values()) / len(table)

    def _event_to_ms(self, ev) -> float:
        if ev.phase == 'bwd':
            layer_ms = self._per_layer_bwd_ms.get(
                ev.layer_idx, self._default_layer_bwd_ms)
            events_per_layer = self.EVENTS_PER_BWD_LAYER
        else:
            layer_ms = self._per_layer_fwd_ms.get(
                ev.layer_idx, self._default_layer_fwd_ms)
            events_per_layer = self.EVENTS_PER_FWD_LAYER
        return layer_ms / max(events_per_layer, 1)

    def _step_to_ms(self, step_distance: int, current_phase: str) -> float:
        if step_distance <= 0:
            return 0.0
        if current_phase == 'bwd':
            per_event = self._avg_known_layer_ms('bwd') / max(
                self.EVENTS_PER_BWD_LAYER, 1)
        else:
            per_event = self._avg_known_layer_ms('fwd') / max(
                self.EVENTS_PER_FWD_LAYER, 1)
        return step_distance * per_event

    def _classify_window_nccl_phase(
        self, lo_step: int, hi_step: int,
    ) -> tuple[float, float]:
        active_ms = 0.0
        idle_ms = 0.0
        for ev in self._time_slots_in_range(lo_step, hi_step):
            ev_ms = self._event_to_ms(ev)
            if ev.phase == 'bwd':
                active_ms += ev_ms
            else:
                idle_ms += ev_ms
        return active_ms, idle_ms

    def _time_slots_in_range(self, lo_step: int, hi_step: int):
        seen: set[int] = set()
        for ev in self._schedule.events_in_range(lo_step, hi_step):
            if ev.step_idx in seen:
                continue
            seen.add(ev.step_idx)
            yield ev

    def _window_slot_capacity_bytes(
        self, lo_step: int, hi_step: int,
    ) -> tuple[float, float, dict]:
        bw_idle = self._calibration.pcie_bw_idle_one_way_bytes_per_s
        bw_active = self._calibration.pcie_bw_under_nccl_one_way_bytes_per_s
        if bw_active <= 0:
            bw_active = bw_idle
        active_ms = 0.0
        idle_ms = 0.0
        slots: dict = {}
        for ev in self._time_slots_in_range(lo_step, hi_step):
            ev_ms = self._event_to_ms(ev)
            if ev.phase == 'bwd':
                active_ms += ev_ms
                cap = (ev_ms / 1000.0) * bw_active
            else:
                idle_ms += ev_ms
                cap = (ev_ms / 1000.0) * bw_idle
            slots[ev.step_idx] = slots.get(ev.step_idx, 0.0) + cap
        return active_ms, idle_ms, slots

    def _debit_window(self, slots: dict, nbytes: int) -> None:
        if not slots or nbytes <= 0:
            return
        total = sum(slots.values())
        if total <= 0:
            first = min(slots)
            self._window_debit_bytes[first] = (
                self._window_debit_bytes.get(first, 0.0) + float(nbytes))
        else:
            for slot, cap in slots.items():
                self._window_debit_bytes[slot] = (
                    self._window_debit_bytes.get(slot, 0.0)
                    + nbytes * (cap / total))
        self._window_debit_committed_bytes += int(nbytes)

    def _queued_bytes_in_window(self, lo_step: int, hi_step: int) -> int:
        total = 0
        for op in getattr(self._mem_stream, '_inflight', []):
            total += int(op.nbytes)
        return total

    def _available_overlap_ms(
        self,
        size_bytes: int,
        representative_tensor_id: str,
        decision_step: int,
    ) -> tuple[float, dict]:
        nxt = self._schedule.next_read_after(
            representative_tensor_id, decision_step)
        if nxt is None:
            return 1e4, {'active_ms': 0.0, 'idle_ms': 1e4, 'queued_bytes': 0,
                         'capacity_bytes': 0.0, 'debited_bytes': 0.0,
                         'slots': {}}

        active_ms, idle_ms, slots = self._window_slot_capacity_bytes(
            decision_step, nxt)
        bw_idle = self._calibration.pcie_bw_idle_one_way_bytes_per_s

        capacity_bytes = sum(slots.values())

        queued = self._queued_bytes_in_window(decision_step, nxt)
        debited = 0.0
        if self._window_debit_enabled:
            for slot in slots:
                debited += self._window_debit_bytes.get(slot, 0.0)
        volume_bytes = max(0.0, capacity_bytes - queued - debited)

        volume_bytes *= self._overlap_safety_factor

        telemetry = {
            'active_ms': active_ms,
            'idle_ms': idle_ms,
            'queued_bytes': int(queued + debited),
            'capacity_bytes': capacity_bytes,
            'debited_bytes': debited,
            'slots': slots,
        }

        if bw_idle <= 0:
            return 0.0, telemetry
        overlap_ms = volume_bytes / bw_idle * 1000.0
        return overlap_ms, telemetry

    def _estimate_reallgather_overlap_ms(self, layer_idx: int) -> float:
        if layer_idx >= self._num_layers - 1:
            return 0.0
        return self._per_layer_bwd_ms.get(
            layer_idx + 1, self._default_layer_bwd_ms)

    def _exposed_reallgather_ms(self, layer_idx: int) -> float:
        if self._coord.params_resident(layer_idx):
            return 0.0
        nbytes = self._coord.estimate_allgather_bytes(layer_idx)
        bw = getattr(
            self._calibration, 'nccl_allgather_bytes_per_s', 0.0)
        if bw <= 0:
            return 0.0
        raw_ms = nbytes / bw * 1000.0
        overlap_ms = self._estimate_reallgather_overlap_ms(layer_idx)
        return max(0.0, raw_ms - overlap_ms)

    def _decide_evict_vs_recompute(
        self,
        size_bytes: int,
        flops: int,
        layer_idx: int,
        representative_tensor_id: str,
        decision_step: int,
        kind: str = None,
        member_sizes: list = None,
        commit: bool = True,
    ) -> tuple[str, PlacementDecisionReason]:
        recompute_available = (kind != 'layer')

        if member_sizes:
            _has_capacity = self._pool.has_capacity_for_sizes(
                member_sizes, kind=kind)
        else:
            _has_capacity = self._pool.has_capacity(size_bytes, kind=kind)
        if not _has_capacity:
            if not recompute_available:
                return ('retain', PlacementDecisionReason(
                    cpu_dram_full=True,
                    note='no_relief_cpu_dram_full',
                ))
            recompute_ms = self._recompute_cost_ms(
                flops, layer_idx=layer_idx, kind=kind)
            reallgather_ms = self._exposed_reallgather_ms(layer_idx)
            return ('recompute', PlacementDecisionReason(
                recompute_ms=recompute_ms,
                reallgather_ms=reallgather_ms,
                exposed_recompute_ms=recompute_ms + reallgather_ms,
                cpu_dram_full=True,
                note='cpu_dram_full',
            ))

        _probe_key = (kind, self._training_step_count)
        if (commit and kind is not None and recompute_available
                and _probe_key not in self._class_probe_issued
                and not self._recompute_class_ms.get(kind)):
            self._class_probe_issued.add(_probe_key)
            recompute_ms = self._recompute_cost_ms(
                flops, layer_idx=layer_idx, kind=kind)
            reallgather_ms = self._exposed_reallgather_ms(layer_idx)
            return ('recompute', PlacementDecisionReason(
                recompute_ms=recompute_ms,
                reallgather_ms=reallgather_ms,
                exposed_recompute_ms=recompute_ms + reallgather_ms,
                note='class_probe',
            ))

        bw_idle = self._calibration.pcie_bw_idle_one_way_bytes_per_s
        transfer_ms = (size_bytes / bw_idle * 1000.0) if bw_idle > 0 else float('inf')

        overlap_ms, window_telemetry = self._available_overlap_ms(
            size_bytes, representative_tensor_id, decision_step)
        exposed_evict_ms = max(0.0, transfer_ms - overlap_ms)

        recompute_ms = self._recompute_cost_ms(
            flops, layer_idx=layer_idx, kind=kind)
        reallgather_ms = self._exposed_reallgather_ms(layer_idx)
        exposed_recompute_ms = recompute_ms + reallgather_ms

        if exposed_evict_ms <= exposed_recompute_ms:
            action = 'evict_to_cpu'
            if overlap_ms <= 0.0 and exposed_evict_ms > 0:
                note = 'pcie_saturated_in_window'
            elif window_telemetry['active_ms'] > 0 and window_telemetry['idle_ms'] == 0:
                note = 'pcie_under_nccl_window'
            else:
                note = 'evict_under_exposed_cost'
        else:
            action = 'recompute'
            flipped = False
            debited = window_telemetry.get('debited_bytes', 0.0)
            if debited > 0 and bw_idle > 0:
                undebited_ms = (
                    max(0.0, window_telemetry.get('capacity_bytes', 0.0)
                        - window_telemetry['queued_bytes'] + debited)
                    * self._overlap_safety_factor / bw_idle * 1000.0)
                flipped = (max(0.0, transfer_ms - undebited_ms)
                           <= exposed_recompute_ms)
            if flipped:
                self._window_debit_refusals += 1
                note = 'window_budget_exhausted'
            elif overlap_ms <= 0.0:
                note = 'late_prefetch_no_overlap'
            elif window_telemetry['queued_bytes'] > 0:
                note = 'pcie_saturated_in_window'
            else:
                note = 'recompute_under_exposed_cost'
            if not recompute_available:
                action = 'evict_to_cpu'
                note = f'evict_fallback_{note}'

        if (action == 'evict_to_cpu' and commit
                and self._window_debit_enabled):
            self._debit_window(window_telemetry.get('slots', {}), size_bytes)

        return (action, PlacementDecisionReason(
            transfer_ms=transfer_ms,
            overlap_ms=overlap_ms,
            exposed_evict_ms=exposed_evict_ms,
            exposed_recompute_ms=exposed_recompute_ms,
            recompute_ms=recompute_ms,
            reallgather_ms=reallgather_ms,
            queued_bytes_in_window=window_telemetry['queued_bytes'],
            window_capacity_bytes=int(window_telemetry.get('capacity_bytes', 0)),
            window_debited_bytes=int(window_telemetry.get('debited_bytes', 0)),
            note=note,
        ))
