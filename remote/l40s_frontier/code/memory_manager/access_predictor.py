import bisect
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator, Optional

from .tensor_registry import TensorRegistry

@dataclass(frozen=True)
class AccessEvent:
    step_idx: int
    phase: str
    layer_idx: int
    tensor_id: str
    access_type: str

class AccessSchedule:

    def __init__(
        self,
        events: list[AccessEvent],
        bytes_by_tensor: Optional[dict[str, int]] = None,
    ) -> None:
        self.events: list[AccessEvent] = list(events)
        self.bytes_by_tensor: dict[str, int] = dict(bytes_by_tensor or {})
        self._read_steps_by_tensor: dict[str, list[int]] = defaultdict(list)
        self._first_bwd_read_by_layer: dict[int, int] = {}
        self._first_fwd_create_by_layer: dict[int, int] = {}
        for ev in self.events:
            if ev.access_type == 'read':
                self._read_steps_by_tensor[ev.tensor_id].append(ev.step_idx)
            if ev.phase == 'bwd' and ev.access_type == 'read':
                if ev.layer_idx not in self._first_bwd_read_by_layer:
                    self._first_bwd_read_by_layer[ev.layer_idx] = ev.step_idx
            if ev.phase == 'fwd' and ev.access_type == 'create':
                if ev.layer_idx not in self._first_fwd_create_by_layer:
                    self._first_fwd_create_by_layer[ev.layer_idx] = ev.step_idx

    def next_read_after(self, tensor_id: str, step_idx: int) -> Optional[int]:
        reads = self._read_steps_by_tensor.get(tensor_id)
        if not reads:
            return None
        i = bisect.bisect_right(reads, step_idx)
        return reads[i] if i < len(reads) else None

    def events_in_range(self, lo: int, hi: int) -> Iterator[AccessEvent]:
        for ev in self.events:
            if ev.step_idx >= hi:
                break
            if ev.step_idx >= lo:
                yield ev

    def bwd_event_for_layer(self, layer_idx: int) -> Optional[int]:
        return self._first_bwd_read_by_layer.get(layer_idx)

    def fwd_create_event_for_layer(self, layer_idx: int) -> Optional[int]:
        return self._first_fwd_create_by_layer.get(layer_idx)

    def occupancy_curve(
        self,
        start_step: int,
        end_step: int,
        current_live_bytes: int = 0,
        exclude_tensors: Optional[set[str]] = None,
    ) -> list[tuple[int, int]]:
        excluded = exclude_tensors or set()
        live = current_live_bytes
        curve: list[tuple[int, int]] = []
        for ev in self.events:
            if ev.step_idx < start_step:
                continue
            if ev.step_idx > end_step:
                break
            if ev.tensor_id in excluded:
                continue
            if ev.access_type == 'create':
                live += self.bytes_by_tensor.get(ev.tensor_id, 0)
            elif ev.access_type == 'free':
                live -= self.bytes_by_tensor.get(ev.tensor_id, 0)
            curve.append((ev.step_idx, live))
        return curve

    def peak_in_range(
        self,
        start_step: int,
        end_step: int,
        current_live_bytes: int = 0,
        exclude_tensors: Optional[set[str]] = None,
        restore_at: Optional[dict[str, int]] = None,
    ) -> tuple[int, int]:
        excluded = exclude_tensors or set()
        pending = sorted(
            (ordinal, tid) for tid, ordinal in (restore_at or {}).items()
            if tid in excluded)
        pi = 0
        released: set[str] = set()
        held: dict[str, int] = {}

        live = current_live_bytes
        peak_bytes = live
        peak_step = start_step
        for ev in self.events:
            if ev.step_idx > end_step:
                break
            in_window = ev.step_idx >= start_step
            if in_window:
                while pi < len(pending) and pending[pi][0] <= ev.step_idx:
                    ordinal, tid = pending[pi]
                    pi += 1
                    if tid in released:
                        continue
                    released.add(tid)
                    back = held.pop(tid, 0)
                    if back:
                        live += back
                        if live > peak_bytes:
                            peak_bytes = live
                            peak_step = ordinal
            if ev.tensor_id in excluded and ev.tensor_id not in released:
                nbytes = self.bytes_by_tensor.get(ev.tensor_id, 0)
                if ev.access_type == 'create':
                    held[ev.tensor_id] = held.get(ev.tensor_id, 0) + nbytes
                elif ev.access_type == 'free':
                    held[ev.tensor_id] = held.get(ev.tensor_id, 0) - nbytes
                continue
            if not in_window:
                continue
            if ev.access_type == 'create':
                live += self.bytes_by_tensor.get(ev.tensor_id, 0)
            elif ev.access_type == 'free':
                live -= self.bytes_by_tensor.get(ev.tensor_id, 0)
            if live > peak_bytes:
                peak_bytes = live
                peak_step = ev.step_idx
        return peak_bytes, peak_step

    def contribution_window_for(
        self,
        tensor_ids: set[str],
        current_step: int,
        end_step: int,
        current_live_bytes: int = 0,
    ) -> tuple[int, int, int]:
        peak_with, peak_step_with = self.peak_in_range(
            current_step, end_step, current_live_bytes, exclude_tensors=None)
        peak_without, _peak_step_without = self.peak_in_range(
            current_step, end_step, current_live_bytes,
            exclude_tensors=tensor_ids)
        return peak_step_with, peak_with, peak_without

    def __len__(self) -> int:
        return len(self.events)

    def summary(self) -> dict:
        by_phase: dict[str, int] = defaultdict(int)
        by_type: dict[str, int] = defaultdict(int)
        for ev in self.events:
            by_phase[ev.phase] += 1
            by_type[ev.access_type] += 1
        return {
            'total_events': len(self.events),
            'unique_tensors': len({ev.tensor_id for ev in self.events}),
            'by_phase': dict(by_phase),
            'by_access_type': dict(by_type),
        }

class AccessPredictor:

    _ACTIVATION_COMPONENTS: tuple[str, ...] = (
        'q', 'k', 'v',
        'attn_scores',
        'swiglu_gate',
        'swiglu_up',
        'swiglu_product',
    )

    def __init__(
        self,
        num_layers: int,
        seq_len: int,
        checkpoint_policy: Optional[dict[int, bool]] = None,
        verbose: bool = True,
        *,
        hidden_size: int = 0,
        intermediate_size: int = 0,
        num_heads: int = 0,
        num_kv_heads: int = 0,
        batch_size: int = 0,
        dtype_bytes: int = 2,
        materializes_attn_scores: bool = True,
        bundle_kinds: tuple[str, ...] = ('attention', 'mlp'),
        vocab_size: int = 0,
        chunked_ce_tokens: int = 0,
    ) -> None:
        self.num_layers = num_layers
        self.seq_len = seq_len
        self._RESIDUAL_KINDS = tuple(bundle_kinds)
        self.checkpoint_policy: dict[int, bool] = dict(checkpoint_policy or {})
        self.verbose = verbose
        self.materializes_attn_scores = materializes_attn_scores
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads > 0 else num_heads
        self.batch_size = batch_size
        self.dtype_bytes = dtype_bytes
        self.vocab_size = vocab_size
        self.chunked_ce_tokens = chunked_ce_tokens
        self._schedule: Optional[AccessSchedule] = None

    def build_schedule(self) -> AccessSchedule:
        if self.verbose:
            print(f"[AccessPredictor] build_schedule START "
                  f"(num_layers={self.num_layers}, seq_len={self.seq_len}, "
                  f"checkpointed_layers="
                  f"{sorted(L for L, on in self.checkpoint_policy.items() if on)})")

        events: list[AccessEvent] = []
        step = 0

        def _emit_residual(phase: str, L: int, comp: str, access: str,
                           skip_surviving: bool = False) -> None:
            for kind in self._RESIDUAL_KINDS:
                if skip_surviving and (
                        kind in self._CHECKPOINT_SURVIVING_RESIDUAL_KINDS):
                    continue
                if self._RESIDUAL_ANCHOR.get(kind) == comp:
                    events.append(AccessEvent(
                        step, phase, L, self.residual_id(L, kind), access))

        for L in range(self.num_layers):
            events.append(AccessEvent(step, 'fwd', L, f'param_layer_{L}', 'read'))
            step += 1

            for comp in self._ACTIVATION_COMPONENTS:
                events.append(AccessEvent(
                    step, 'fwd', L, self._activation_id(L, comp), 'create'))
                _emit_residual('fwd', L, comp, 'create')
                step += 1

            if self.checkpoint_policy.get(L, False):
                for comp in self._ACTIVATION_COMPONENTS:
                    events.append(AccessEvent(
                        step, 'fwd', L, self._activation_id(L, comp), 'free'))
                    _emit_residual('fwd', L, comp, 'free', skip_surviving=True)
                    step += 1

        head_layer = self.num_layers
        emit_head_ce = self.vocab_size > 0 and bool(events)
        if emit_head_ce:
            for tid in (self._FINAL_NORM_ID, self._LOGITS_ID):
                events.append(AccessEvent(
                    step - 1, 'fwd', head_layer, tid, 'create'))

        if emit_head_ce:
            events.append(AccessEvent(
                step, 'bwd', head_layer, self._LOGIT_GRAD_ID, 'create'))
            for tid in (self._LOGIT_GRAD_ID, self._LOGITS_ID,
                        self._FINAL_NORM_ID):
                events.append(AccessEvent(
                    step, 'bwd', head_layer, tid, 'free'))

        for L in reversed(range(self.num_layers)):
            if self.checkpoint_policy.get(L, False):
                events.append(AccessEvent(
                    step, 'recompute', L, f'param_layer_{L}', 'read'))
                step += 1
                for comp in self._ACTIVATION_COMPONENTS:
                    events.append(AccessEvent(
                        step, 'recompute', L,
                        self._activation_id(L, comp), 'create'))
                    _emit_residual('recompute', L, comp, 'create',
                                   skip_surviving=True)
                    step += 1

            events.append(AccessEvent(step, 'bwd', L, f'param_layer_{L}', 'read'))
            if self._layer_bwd_transient_bytes():
                events.append(AccessEvent(
                    step, 'bwd', L, self.layer_bwd_id(L), 'create'))
            step += 1
            for comp in self._ACTIVATION_COMPONENTS:
                events.append(AccessEvent(
                    step, 'bwd', L, self._activation_id(L, comp), 'read'))
                _emit_residual('bwd', L, comp, 'read')
                step += 1

            for comp in self._ACTIVATION_COMPONENTS:
                events.append(AccessEvent(
                    step, 'bwd', L, self._activation_id(L, comp), 'free'))
                _emit_residual('bwd', L, comp, 'free')
                step += 1

            if self._layer_bwd_transient_bytes():
                events.append(AccessEvent(
                    step, 'bwd', L, self.layer_bwd_id(L), 'free'))
            events.append(AccessEvent(step, 'bwd', L, f'grad_layer_{L}', 'free'))
            step += 1

        bytes_map = self._build_bytes_map()
        schedule = AccessSchedule(events, bytes_by_tensor=bytes_map)
        self._schedule = schedule

        if self.verbose:
            s = schedule.summary()
            total_activation_bytes = sum(bytes_map.values())
            print(f"[AccessPredictor] build_schedule DONE "
                  f"(events={s['total_events']}, "
                  f"unique_tensors={s['unique_tensors']}, "
                  f"by_phase={s['by_phase']}, "
                  f"by_access_type={s['by_access_type']}, "
                  f"total_activation_bytes={total_activation_bytes})")
        return schedule

    _SCORES_ELEM_BYTES = 4

    _SWIGLU_SAVED_PLANES = 4
    _SWIGLU_GATE_PLANES = _SWIGLU_SAVED_PLANES - 2

    def _activation_bytes(self, component: str) -> int:
        if self.batch_size <= 0 or self.seq_len <= 0:
            return 0
        B, S = self.batch_size, self.seq_len
        H = self.hidden_size
        head_dim = (H // self.num_heads) if self.num_heads > 0 else 0
        kv_dim = self.num_kv_heads * head_dim
        elem_bytes = self.dtype_bytes
        if component == 'q':
            elems = B * H * S
        elif component in ('k', 'v'):
            elems = B * kv_dim * S
        elif component == 'attn_scores':
            if not self.materializes_attn_scores:
                return self._efficient_attn_slot_bytes()
            elems = B * self.num_heads * S * S
            elem_bytes = self._SCORES_ELEM_BYTES
        elif component == 'swiglu_gate':
            elems = B * S * self.intermediate_size * self._SWIGLU_GATE_PLANES
        elif component in ('swiglu_up', 'swiglu_product'):
            elems = B * S * self.intermediate_size
        else:
            elems = 0
        return elems * elem_bytes

    def _efficient_attn_slot_bytes(self) -> int:
        B, S = self.batch_size, self.seq_len
        H = self.hidden_size
        head_dim = (H // self.num_heads) if self.num_heads > 0 else 0
        kv_dim = self.num_kv_heads * head_dim
        q_rotated = B * S * H * self.dtype_bytes
        k_rotated = B * S * kv_dim * self.dtype_bytes
        attn_out = B * S * H * self.dtype_bytes
        logsumexp = B * self.num_heads * S * self._SCORES_ELEM_BYTES
        return q_rotated + k_rotated + attn_out + logsumexp

    def _build_bytes_map(self) -> dict[str, int]:
        bytes_map: dict[str, int] = {}
        for L in range(self.num_layers):
            for comp in self._ACTIVATION_COMPONENTS:
                bytes_map[self._activation_id(L, comp)] = (
                    self._activation_bytes(comp))
            for kind in self._RESIDUAL_KINDS:
                bytes_map[self.residual_id(L, kind)] = (
                    self.boundary_bytes()
                    if kind in self._CHECKPOINT_SURVIVING_RESIDUAL_KINDS
                    else 0)
        transient = self._layer_bwd_transient_bytes()
        if transient:
            for L in range(self.num_layers):
                bytes_map[self.layer_bwd_id(L)] = transient
        bytes_map.update(self._head_ce_bytes())
        return bytes_map

    _FINAL_NORM_ID = 'nonmig_final_norm'
    _LOGITS_ID = 'nonmig_logits'
    _LOGIT_GRAD_ID = 'nonmig_logit_grad'

    _LOGIT_FWD_BYTES_PER_TOKEN_VOCAB = 12
    _LOGIT_BWD_BYTES_PER_TOKEN_VOCAB = 2

    _LOGIT_CHUNKED_BYTES_PER_TOKEN_VOCAB = 12

    _FINAL_NORM_PLANES = 2

    _BWD_GRAD_INTERMEDIATE_PLANES = 3
    _BWD_GRAD_HIDDEN_PLANES = 2

    def _layer_bwd_transient_bytes(self) -> int:
        if self.batch_size <= 0 or self.seq_len <= 0:
            return 0
        tokens = self.batch_size * self.seq_len
        width = (self._BWD_GRAD_INTERMEDIATE_PLANES * self.intermediate_size
                 + self._BWD_GRAD_HIDDEN_PLANES * self.hidden_size)
        return tokens * width * self.dtype_bytes

    @staticmethod
    def layer_bwd_id(layer_idx: int) -> str:
        return f'nonmig_layer_bwd_{layer_idx}'

    def _head_ce_bytes(self) -> dict[str, int]:
        if self.vocab_size <= 0 or self.batch_size <= 0 or self.seq_len <= 0:
            return {}
        tokens = self.batch_size * self.seq_len
        final_norm = (tokens * self.hidden_size
                      * self.dtype_bytes * self._FINAL_NORM_PLANES)
        if self.chunked_ce_tokens > 0:
            chunk = min(self.chunked_ce_tokens, tokens)
            return {
                self._FINAL_NORM_ID: final_norm,
                self._LOGITS_ID: (chunk * self.vocab_size
                                  * self._LOGIT_CHUNKED_BYTES_PER_TOKEN_VOCAB),
                self._LOGIT_GRAD_ID: 0,
            }
        return {
            self._FINAL_NORM_ID: final_norm,
            self._LOGITS_ID: (tokens * self.vocab_size
                              * self._LOGIT_FWD_BYTES_PER_TOKEN_VOCAB),
            self._LOGIT_GRAD_ID: (tokens * self.vocab_size
                                  * self._LOGIT_BWD_BYTES_PER_TOKEN_VOCAB),
        }

    @staticmethod
    def _activation_id(layer_idx: int, component: str) -> str:
        return f"act_layer_{layer_idx}_{component}"

    _RESIDUAL_KINDS: tuple[str, ...] = ('attention', 'mlp')

    _RESIDUAL_ANCHOR: dict[str, str] = {
        'attention': 'attn_scores',
        'mlp': 'swiglu_product',
        'layer': 'swiglu_product',
    }

    _CHECKPOINT_SURVIVING_RESIDUAL_KINDS: frozenset = frozenset({'layer'})

    def boundary_bytes(self) -> int:
        return (self.batch_size * self.seq_len
                * self.hidden_size * self.dtype_bytes)

    @staticmethod
    def residual_id(layer_idx: int, kind: str) -> str:
        return f"act_layer_{layer_idx}_{kind}_residual"

    def rebuild_on_seq_len_change(self, new_seq_len: int) -> AccessSchedule:
        self.seq_len = new_seq_len
        return self.build_schedule()

    def rebuild_on_policy_change(
        self,
        checkpoint_policy: dict[int, bool],
    ) -> AccessSchedule:
        self.checkpoint_policy = dict(checkpoint_policy)
        return self.build_schedule()

    def populate_registry(self, registry: TensorRegistry) -> int:
        if self._schedule is None:
            raise RuntimeError(
                "AccessPredictor.populate_registry() called before "
                "build_schedule(); build the schedule first.")

        birth_step: dict[str, int] = {}
        for ev in self._schedule.events:
            if ev.access_type == 'create' and ev.tensor_id not in birth_step:
                birth_step[ev.tensor_id] = ev.step_idx

        updated = 0
        for key, record in registry.items():
            if key not in birth_step:
                continue
            nxt = self._schedule.next_read_after(key, birth_step[key])
            if nxt is not None:
                record.predicted_next_access_step = nxt
                updated += 1

        if self.verbose:
            print(f"[AccessPredictor] populate_registry stamped "
                  f"{updated} record(s) with predicted_next_access_step")
        return updated

    @property
    def schedule(self) -> Optional[AccessSchedule]:
        return self._schedule
