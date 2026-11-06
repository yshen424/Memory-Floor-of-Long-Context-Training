"""
Access Pattern Predictor — Component 2 of the tiered memory manager.

This module owns three pieces:

    * ``AccessEvent``      — the per-access record (one row of the schedule).
    * ``AccessSchedule``   — the immutable, ordered list of events plus an
                             inverted read index for O(log k) lookups.
    * ``AccessPredictor``  — builds the schedule from architectural inputs.

Design decisions:

    1. The schedule's time axis is **integer step ordinals**, not wall-clock
       milliseconds. Every event gets a dense monotonic ``step_idx``; the
       placement engine's eviction scoring formula
       ``(next_read_step - current_step) / size_bytes`` cares only about
       relative ordering, not physical time. Wall-clock cost (PCIe transfer
       vs recompute, plus the overlap window) lives in the placement engine,
       not here.

    2. The schedule is **static** — built once at init from the model
       architecture and (batch_size, seq_len, checkpoint_policy). The
       schedule re-builds only when seq_len changes or the placement engine
       flips a layer's checkpoint policy. There is no online learning.

    3. ``TensorRecord.predicted_next_access_step`` is a **one-shot
       snapshot** populated by ``populate_registry()`` for telemetry only.
       The engine's runtime decisions must go through
       ``AccessSchedule.next_read_after(...)`` so the answer reflects the
       current step, not a stale init value.

Tensor-id naming follows the registry keys produced by fsdp_profiler.py:
    activations:   act_layer_{L}_{q|k|v|attn_scores|swiglu_gate|swiglu_up|swiglu_product}
    params:        param_layer_{L}   ← logical id (the registry uses
                                       param_shard_{full_fsdp_name}, which
                                       the predictor cannot know at init;
                                       the engine resolves the
                                       logical id to per-shard records by
                                       (category == 'parameter' and
                                       layer_idx == L)).
    gradients:     grad_layer_{L}    ← logical id, freed after reduce-scatter

Scope:
    - A schedule that covers the fwd/bwd pass of N decoder layers.
    - An optional ``checkpoint_policy`` (default: all layers retained).
    - Query primitives the placement engine needs: ``next_read_after``
      (eviction scoring), ``events_in_range`` (prefetch lookahead).

The overlap window needs wall-clock estimates, but the step→ms mapping
lives on ``PlacementEngine._step_to_ms`` so ``AccessSchedule`` stays
immutable and rank-invariant.

The schedule also carries a per-tensor byte model and an
occupancy-projection API (``occupancy_curve`` / ``peak_in_range`` /
``contribution_window_for``). The byte model is derived from architecture
parameters (``hidden_size`` / ``intermediate_size`` / ``num_heads`` /
``num_kv_heads`` / ``batch_size`` / ``dtype_bytes``) so the planner can
project HBM occupancy *before* L's bundles are ever created. Params and
gradients are intentionally not byte-modeled here (FSDP owns those
lifecycles); the planner targets activations.

The predictor stays *static* (no profile-pass warmup). Projections are
anchored on the allocator's measured live bytes, so whatever the static
byte model does not carry (NCCL buffers, allocator fragmentation,
workspace) is already in the baseline.
"""

import bisect
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator, Optional

from .tensor_registry import TensorRegistry


# ────────────────────────────────────────────────────────────────────────────
# AccessEvent
# ────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AccessEvent:
    """One row of the access schedule.

    ``step_idx`` is the monotonic time axis (event ordinal), not ms.
    Events with the same ``step_idx`` would be ambiguous, so the predictor
    assigns a unique step to every event.

    ``phase`` distinguishes:
        'fwd'        — forward pass on the original execution path.
        'bwd'        — backward pass.
        'recompute'  — shadow forward that runs during backward when a
                       layer was checkpointed (its activations were freed
                       at the end of forward and must be regenerated before
                       backward can use them).

    ``access_type`` is one of:
        'create' — tensor is allocated and written for the first time.
        'read'   — tensor is read (used as an input to a kernel).
        'free'   — tensor is released; HBM/CPU memory reclaimable after.
    """
    step_idx: int
    phase: str          # 'fwd' | 'bwd' | 'recompute'
    layer_idx: int
    tensor_id: str      # matches TensorRegistry key for activations
    access_type: str    # 'create' | 'read' | 'free'


# ────────────────────────────────────────────────────────────────────────────
# AccessSchedule
# ────────────────────────────────────────────────────────────────────────────

class AccessSchedule:
    """Immutable, ordered access schedule with an inverted read-index.

    Built by ``AccessPredictor.build_schedule()``. Treat ``events`` as
    read-only after construction — mutating it invalidates the index.

    The inverted index maps ``tensor_id -> sorted list of step_idx`` at
    which the tensor is read. Lookups via ``next_read_after`` use
    ``bisect`` for O(log k) where k is the number of reads of that tensor
    (typically small: each activation is read once during backward, plus
    maybe once during recompute).
    """

    def __init__(
        self,
        events: list[AccessEvent],
        bytes_by_tensor: Optional[dict[str, int]] = None,
    ) -> None:
        self.events: list[AccessEvent] = list(events)  # defensive copy
        # Per-tensor byte model. Tensors absent from the map count as 0 bytes
        # so synthetic schedules (unit tests, params/grads) walk cleanly.
        self.bytes_by_tensor: dict[str, int] = dict(bytes_by_tensor or {})
        # Inverted index: tensor_id -> [step_idx, ...] of read events.
        # Events are emitted in step_idx order, so each list is already sorted.
        self._read_steps_by_tensor: dict[str, list[int]] = defaultdict(list)
        # Inverted index: layer_idx -> first bwd read step_idx (for the planner's
        # "until L's backward consumes its activations" range).
        self._first_bwd_read_by_layer: dict[int, int] = {}
        # layer_idx -> first fwd create step_idx (for the planner's "start of
        # L's forward window" projection lower bound). The planner advances
        # _current_step in on_pre_forward(L) before projecting, so a window
        # starting at the cursor would walk PAST L's own create events and
        # contribution_window_for would return 0 for L's bundles.
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
        """Return the smallest step_idx > ``step_idx`` where ``tensor_id`` is
        read, or None if no future read exists.

        Used by the placement engine's eviction scoring at every decision
        point — must be cheap. Binary search via bisect.
        """
        reads = self._read_steps_by_tensor.get(tensor_id)
        if not reads:
            return None
        i = bisect.bisect_right(reads, step_idx)
        return reads[i] if i < len(reads) else None

    def events_in_range(self, lo: int, hi: int) -> Iterator[AccessEvent]:
        """Yield events with ``lo <= step_idx < hi``.

        Used by the placement engine for prefetch lookahead ("what tensors
        will be read in the next 2 layers?"). Linear scan with early-exit;
        could swap to binary-search-bounded if profiling shows it matters.
        """
        for ev in self.events:
            if ev.step_idx >= hi:
                break
            if ev.step_idx >= lo:
                yield ev

    # ── Occupancy projection API ──────────────────────────────────────────
    #
    # These three methods underpin the per-layer planner. The
    # planner asks: "if I keep L's bundles in HBM through L's backward,
    # will any future event in that window push live bytes over budget?"
    # The walk starts from a measured ``current_live_bytes`` (anchored to
    # the runtime allocator, not the schedule) so under-modeled bytes are
    # already accounted for in the baseline.

    def bwd_event_for_layer(self, layer_idx: int) -> Optional[int]:
        """Return the step_idx of layer ``layer_idx``'s first backward read,
        or None if the layer has no backward reads (synthetic schedules).

        This is the natural end-of-window for "keep L's bundles live through
        their backward consumption" — the planner uses it as the upper
        bound of the projection range at ``on_pre_forward(L)``.
        """
        return self._first_bwd_read_by_layer.get(layer_idx)

    def fwd_create_event_for_layer(self, layer_idx: int) -> Optional[int]:
        """Return the step_idx of layer ``layer_idx``'s first fwd create event,
        or None if the layer has no creates (synthetic schedules).

        This is the natural start-of-window for "L's bundles are about to be
        created" — the planner uses it as the lower bound of the projection
        range at ``on_pre_forward(L)``. Using ``self._current_step`` would
        be wrong because the engine advances the cursor BY
        ``EVENTS_PER_FWD_LAYER`` at the top of ``on_pre_forward``, so the
        cursor sits at end-of-L's-forward — past L's own creates.
        """
        return self._first_fwd_create_by_layer.get(layer_idx)

    def occupancy_curve(
        self,
        start_step: int,
        end_step: int,
        current_live_bytes: int = 0,
        exclude_tensors: Optional[set[str]] = None,
    ) -> list[tuple[int, int]]:
        """Return the projected live-bytes curve over ``[start_step, end_step]``.

        Walks events with ``start_step <= step_idx <= end_step``, applying
        ``+size`` on 'create' and ``-size`` on 'free' (reads don't move
        bytes). Returns a list of ``(step_idx, live_bytes_after_event)``
        sampled at every event in the range.

        ``current_live_bytes`` anchors the walk to the measured HBM
        footprint at ``start_step`` — the schedule alone doesn't know about
        NCCL buffers, allocator overhead, or tensors that were live before
        ``start_step``. The caller passes the allocator's view.

        ``exclude_tensors`` skips create/free events for any tensor_id in
        the set. Used by ``contribution_window_for`` to model "what if this
        bundle were freed/never-created."
        """
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
            # reads don't change live bytes
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
        """Return ``(peak_bytes, peak_step_idx)`` over ``[start_step, end_step]``.

        Same walk as ``occupancy_curve`` but only tracks the running max.
        If the range is empty, returns ``(current_live_bytes, start_step)``
        — the anchor itself is the peak.

        Used by the per-layer planner: at ``on_pre_forward(L)``, ask whether
        the budget will be exceeded between now and L's backward read.

        ``exclude_tensors`` alone models *lifetime* absence: the tensor is
        treated as if it never landed. That is what a recompute buys and
        what ``contribution_window_for`` asks about, but it is not what a
        migration buys. ``restore_at`` maps an excluded tensor_id to the
        ordinal at which its bytes come back, so the tensor is absent only
        over ``[create, restore)``. A bundle whose restore lands at or
        before the peak instant then shows zero contribution --- which is
        the physical truth, and the reason the planner needs this rather
        than a scalar derate: it separates "restored too early" from "the
        copy did not finish", which one fudge factor cannot.

        Excluded tensors with no entry in ``restore_at`` keep the lifetime
        semantics. ``occupancy_curve`` does not model restores; it is a
        diagnostic and the engine does not call it.

        The restore hands back only what the exclusion is actually holding
        out at that ordinal, which is what makes this safe to apply to a
        whole bundle. Under the whole-layer unit a bundle's tensor_ids
        include the layer's replay transients, and those are created and
        freed again inside the absence interval — they were never in HBM to
        migrate. Re-admitting the bundle's nominal size would invent them.
        """
        excluded = exclude_tensors or set()
        # Ascending re-admission queue. Only excluded tensors matter: a
        # tensor still in the walk was never absent to begin with.
        pending = sorted(
            (ordinal, tid) for tid, ordinal in (restore_at or {}).items()
            if tid in excluded)
        pi = 0
        released: set[str] = set()       # excluded tids past their restore
        held: dict[str, int] = {}        # excluded tid -> bytes suppressed now

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
                # Bookkeeping runs before ``start_step`` too: a bundle picked
                # by an earlier layer was created outside this window, and
                # its restore still has to give those bytes back.
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
        """Return ``(peak_step_with, peak_bytes_with, peak_bytes_without)``.

        For a bundle of tensors (e.g., ``{q, k, v, attn_scores}`` for one
        layer's attention bundle), compute the projected peak with the
        bundle live vs with the bundle excluded entirely. The planner uses
        the diff to decide whether dropping this bundle would resolve a
        projected breach.

        - ``peak_with`` is the natural peak (bundle stays in HBM through
          its backward read).
        - ``peak_without`` is the peak when the bundle is treated as if it
          never landed in HBM (create/free events for these tensor_ids are
          skipped).

        ``current_step`` and ``end_step`` define the projection window.
        Typical usage: ``current_step = self._current_step``,
        ``end_step = self.bwd_event_for_layer(L)`` for layer L's own
        bundles.
        """
        peak_with, peak_step_with = self.peak_in_range(
            current_step, end_step, current_live_bytes, exclude_tensors=None)
        peak_without, _peak_step_without = self.peak_in_range(
            current_step, end_step, current_live_bytes,
            exclude_tensors=tensor_ids)
        return peak_step_with, peak_with, peak_without

    def __len__(self) -> int:
        return len(self.events)

    def summary(self) -> dict:
        """Human-readable counts — useful for logging and debugging.

        Not on the hot path; the engine doesn't call this.
        """
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


# ────────────────────────────────────────────────────────────────────────────
# AccessPredictor
# ────────────────────────────────────────────────────────────────────────────

class AccessPredictor:
    """Builds a static access schedule for one fwd+bwd pass of an N-layer
    LLaMA model.

    The predictor is a template-driven event generator: for each layer it
    emits a fixed set of (create/read/free) events in the order LLaMA's
    decoder layer actually accesses tensors. Sizes and per-tensor bytes
    are intentionally NOT computed here — the registry already records
    actual byte counts at first creation, so the predictor stays purely
    structural.

    Args:
        num_layers: number of LlamaDecoderLayer blocks in the model.
        seq_len: current sequence length (kept for the rebuild API;
            doesn't change event ordering).
        checkpoint_policy: optional mapping ``{layer_idx: bool}`` — True
            means "discard this layer's activations after forward and
            recompute them before backward." Default: all layers retained.
        verbose: if True, print a short summary line on build (used for
            human checkpointing in SLURM logs).
    """

    # Per-layer activation tensor-id suffixes, in roughly the order LLaMA
    # creates them during a forward pass through a decoder layer. The
    # ordering here is approximate — what matters for the predictor is
    # only that all of layer L's creates happen before any of L+1's, and
    # before any of L's reads in backward.
    _ACTIVATION_COMPONENTS: tuple[str, ...] = (
        'q', 'k', 'v',              # QKV projections
        'attn_scores',              # softmax(QK^T) post-mask
        'swiglu_gate',              # SwiGLU components (kept as a group)
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
        # The recompute unit's bundle kinds. ('attention','mlp') is the
        # per-module unit; ('layer',) is the whole-layer unit, where
        # checkpoint_wrapper saves one residual-stream boundary per layer and
        # nothing else. The class default is shadowed per instance: a kind
        # that is not in this tuple emits no residual event and occupies no
        # byte slot.
        self._RESIDUAL_KINDS = tuple(bundle_kinds)
        self.checkpoint_policy: dict[int, bool] = dict(checkpoint_policy or {})
        self.verbose = verbose
        # Which attention kernel the run will dispatch to. True (default) is
        # the score-materializing math kernel. False is the cutlass
        # memory-efficient kernel, which never forms the s x s tensor.
        self.materializes_attn_scores = materializes_attn_scores
        # Architecture knobs for the byte model. Default 0 means "don't
        # compute bytes" — synthetic unit-test schedules that only care
        # about ordering work without supplying these. The placement
        # engine must pass real values for the occupancy projection to
        # be meaningful.
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads > 0 else num_heads
        self.batch_size = batch_size
        self.dtype_bytes = dtype_bytes
        # 0 means the head/CE term is absent from the schedule. The head and
        # the loss carry non-migratable mass that the layer-stack model
        # cannot see; see "The head and the loss" below.
        self.vocab_size = vocab_size
        # The chunked loss's chunk size in tokens, mirroring the driver's
        # --chunked_ce. 0 is HuggingFace's unchunked loss. Above 0 the logit
        # chain stops scaling with sequence length --- that is the whole
        # lever.
        self.chunked_ce_tokens = chunked_ce_tokens
        self._schedule: Optional[AccessSchedule] = None

    # ── Schedule construction ──────────────────────────────────────────────

    def build_schedule(self) -> AccessSchedule:
        """Generate the full fwd+bwd event list and return an AccessSchedule.

        The schedule is also cached on ``self._schedule`` for later use by
        ``populate_registry()``.
        """
        if self.verbose:
            # Human-checkpoint marker — visible in SLURM log so the user can
            # confirm the predictor ran.
            print(f"[AccessPredictor] build_schedule START "
                  f"(num_layers={self.num_layers}, seq_len={self.seq_len}, "
                  f"checkpointed_layers="
                  f"{sorted(L for L, on in self.checkpoint_policy.items() if on)})")

        events: list[AccessEvent] = []
        step = 0  # monotonic event counter — the time axis

        # Emit a residual event alongside its anchor component, sharing the
        # anchor's step_idx rather than consuming one. Sharing is what keeps
        # PlacementEngine.EVENTS_PER_{FWD,BWD}_LAYER and every schedule
        # ordinal — including the window ledger's slot keys — independent of
        # the residual slots.
        def _emit_residual(phase: str, L: int, comp: str, access: str,
                           skip_surviving: bool = False) -> None:
            for kind in self._RESIDUAL_KINDS:
                if skip_surviving and (
                        kind in self._CHECKPOINT_SURVIVING_RESIDUAL_KINDS):
                    continue
                if self._RESIDUAL_ANCHOR.get(kind) == comp:
                    events.append(AccessEvent(
                        step, phase, L, self.residual_id(L, kind), access))

        # ── Forward: layer 0 → N-1 ────────────────────────────────────────
        for L in range(self.num_layers):
            # Param read (FSDP all-gather precedes the layer's compute).
            events.append(AccessEvent(step, 'fwd', L, f'param_layer_{L}', 'read'))
            step += 1

            # Activations created during the layer's forward pass.
            for comp in self._ACTIVATION_COMPONENTS:
                events.append(AccessEvent(
                    step, 'fwd', L, self._activation_id(L, comp), 'create'))
                _emit_residual('fwd', L, comp, 'create')
                step += 1

            # If checkpointed, the layer interior is immediately discarded;
            # it will be regenerated in the recompute block during backward.
            # The surviving residual kinds are exempt --- see
            # _CHECKPOINT_SURVIVING_RESIDUAL_KINDS.
            if self.checkpoint_policy.get(L, False):
                for comp in self._ACTIVATION_COMPONENTS:
                    events.append(AccessEvent(
                        step, 'fwd', L, self._activation_id(L, comp), 'free'))
                    _emit_residual('fwd', L, comp, 'free', skip_surviving=True)
                    step += 1

        # ── Head + loss, forward side ─────────────────────────────────────
        # Shares the last forward event's ordinal rather than consuming one:
        # EVENTS_PER_FWD_LAYER, the window ledger's slot keys and every
        # ordinal helper would otherwise shift. Same tactic as _emit_residual.
        # layer_idx is num_layers --- a sentinel past the last real layer, so
        # no per-layer lookup can collide with it.
        head_layer = self.num_layers
        emit_head_ce = self.vocab_size > 0 and bool(events)
        if emit_head_ce:
            for tid in (self._FINAL_NORM_ID, self._LOGITS_ID):
                events.append(AccessEvent(
                    step - 1, 'fwd', head_layer, tid, 'create'))

        # ── Backward: layer N-1 → 0 ───────────────────────────────────────
        # The cross-entropy backward runs before any layer's backward, and in
        # regime B it is the peak instant. Emitted at the same
        # ordinal as layer N-1's first backward event and ahead of it in list
        # order, which is the order occupancy_curve applies them in. Creating
        # and freeing within one ordinal is what the peak walk needs: the
        # running max records the create, and the layer stack below never sees
        # bytes the real step had already released.
        if emit_head_ce:
            events.append(AccessEvent(
                step, 'bwd', head_layer, self._LOGIT_GRAD_ID, 'create'))
            for tid in (self._LOGIT_GRAD_ID, self._LOGITS_ID,
                        self._FINAL_NORM_ID):
                events.append(AccessEvent(
                    step, 'bwd', head_layer, tid, 'free'))

        for L in reversed(range(self.num_layers)):
            # If this layer was checkpointed, a shadow forward runs first
            # so the backward kernels have activations to consume. FSDP
            # re-gathers the params before this block executes.
            if self.checkpoint_policy.get(L, False):
                events.append(AccessEvent(
                    step, 'recompute', L, f'param_layer_{L}', 'read'))
                step += 1
                for comp in self._ACTIVATION_COMPONENTS:
                    events.append(AccessEvent(
                        step, 'recompute', L,
                        self._activation_id(L, comp), 'create'))
                    # A surviving residual was never freed, so re-creating it
                    # here would book the boundary twice.
                    _emit_residual('recompute', L, comp, 'create',
                                   skip_surviving=True)
                    step += 1

            # Backward kernels read both the activations and the params
            # (gradient computation needs W^T).
            events.append(AccessEvent(step, 'bwd', L, f'param_layer_{L}', 'read'))
            # The gradients this layer's backward is computing are co-resident
            # with the saves that produce them; see "The layer backward's
            # gradient transient" below. Shares the param read's ordinal
            # and is appended after it, so the replayed interior above is
            # already counted when the transient lands.
            if self._layer_bwd_transient_bytes():
                events.append(AccessEvent(
                    step, 'bwd', L, self.layer_bwd_id(L), 'create'))
            step += 1
            for comp in self._ACTIVATION_COMPONENTS:
                events.append(AccessEvent(
                    step, 'bwd', L, self._activation_id(L, comp), 'read'))
                _emit_residual('bwd', L, comp, 'read')
                step += 1

            # Activations are released once gradients are computed.
            for comp in self._ACTIVATION_COMPONENTS:
                events.append(AccessEvent(
                    step, 'bwd', L, self._activation_id(L, comp), 'free'))
                _emit_residual('bwd', L, comp, 'free')
                step += 1

            # Gradients exist briefly until reduce-scatter; predictor emits
            # a logical free event so the placement engine can model the
            # HBM these grads occupy. FSDP performs the actual
            # reduce-scatter.
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

    # ── Byte model ─────────────────────────────────────────────────────────
    #
    # All sizes in *bytes*. Activations dominate planning; params/grads are
    # FSDP-managed and stay at 0 in the byte map (the planner doesn't pick
    # them for evict/recompute).

    # HuggingFace LlamaAttention computes
    # `softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)`, and
    # softmax backward saves its *own* fp32 output. So the scores tensor
    # autograd hands to pack_fn is fp32 regardless of the training dtype.
    # Modelling scores at the 2-byte training width would book half the
    # bytes the planner is deciding about, and make Gate 0 ask for a buffer
    # size the physical path never requests.
    _SCORES_ELEM_BYTES = 4

    # HuggingFace LlamaMLP is
    # `down_proj(act_fn(gate_proj(x)) * up_proj(x))`, and autograd saves FOUR
    # distinct intermediate-width storages from it, not three:
    #   gate      gate_proj's output — SiLU backward saves its INPUT
    #   silu_out  the activation's output  ─┐ mul backward saves BOTH operands
    #   up        up_proj's output         ─┘
    #   product   the mul's output — down_proj's mm saves its input
    # Verified by running the installed LlamaMLP under saved_tensors_hooks and
    # deduplicating by storage data_ptr: 9 saves, 4 owned storages
    # (tests/test_swiglu_plane_count.py). A three-plane model under-books one
    # full plane of relief per layer.
    #
    # Priced into an existing slot, not added as an eighth component:
    # EVENTS_PER_FWD_LAYER is 1 param read + 7 activation creates, and the
    # window ledger's slot keys, _RESIDUAL_ANCHOR and every ordinal helper key
    # off that count. The extra storage belongs to the gate path, so
    # `swiglu_gate` carries both halves of it while `swiglu_up` and
    # `swiglu_product` stay single tensors. Same tactic, and same reason, as
    # _efficient_attn_slot_bytes below.
    _SWIGLU_SAVED_PLANES = 4
    _SWIGLU_GATE_PLANES = _SWIGLU_SAVED_PLANES - 2

    def _activation_bytes(self, component: str) -> int:
        """Bytes for one activation tensor of the given component.

        Shapes follow HuggingFace LlamaAttention / LlamaMLP at the points
        ``fsdp_profiler.py`` registers them:
          q:               (B, num_heads, S, head_dim)         → B·hidden·S
          k, v:            (B, num_kv_heads, S, head_dim)      → B·kv_dim·S
          attn_scores:     (B, num_heads, S, S)                → B·num_heads·S²
          swiglu_up,
          swiglu_product:  (B, S, intermediate_size)           → B·S·intermediate
          swiglu_gate:     two of those — see _SWIGLU_SAVED_PLANES

        Element width is ``dtype_bytes`` for every component except
        ``attn_scores`` — see ``_SCORES_ELEM_BYTES``.

        When ``materializes_attn_scores`` is False the ``attn_scores`` slot
        holds what the memory-efficient kernel saves instead — see
        ``_efficient_attn_slot_bytes``.
        """
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
        # Gate carries TWO planes, not one — SiLU backward saves its input
        # and mul backward saves both operands.
        elif component == 'swiglu_gate':
            elems = B * S * self.intermediate_size * self._SWIGLU_GATE_PLANES
        elif component in ('swiglu_up', 'swiglu_product'):
            elems = B * S * self.intermediate_size
        else:
            elems = 0
        return elems * elem_bytes

    # Under the cutlass memory-efficient kernel there is no s x s tensor at
    # all, so the `attn_scores` slot is repriced rather than removed.
    # Repriced, not deleted, because the slot owns a schedule ordinal that
    # EVENTS_PER_{FWD,BWD}_LAYER, the window ledger's slot keys and
    # _RESIDUAL_ANCHOR['attention'] all key off; dropping the component would
    # shift every ordinal after it.
    #
    # What the slot holds instead is everything the efficient attention path
    # saves beyond the three projections q/k/v that already have their own
    # slots: the rotary-rotated q and k (apply_rotary_pos_emb returns new
    # storages, so pre- and post-rotary are both live), the attention output,
    # and the fp32 logsumexp the kernel keeps for its own backward.
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
        """Produce the tensor_id → bytes lookup table for the byte model.

        Activations only. Param/grad ids land with 0 bytes — they are in
        the events list for ordering purposes but FSDP, not the planner,
        owns their physical lifecycle.
        """
        bytes_map: dict[str, int] = {}
        for L in range(self.num_layers):
            for comp in self._ACTIVATION_COMPONENTS:
                bytes_map[self._activation_id(L, comp)] = (
                    self._activation_bytes(comp))
            # Residual slots start at 0 bytes; PlacementEngine writes them at
            # step end from what pack_fn observed.
            #
            # The surviving kinds are the exception. Under the whole-layer unit
            # the slot is not an anonymous remainder but the layer's input
            # boundary, and all L of them accumulated ARE the migratable supply.
            # Left at zero the schedule projects only the one live interior and
            # under-states the peak L-fold --- the direction that under-relieves
            # a feasibility gate into an OOM. It is also the one residual whose
            # size is known analytically rather than only by observation.
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

    # ── The head and the loss ──────────────────────────────────────────────
    #
    # Everything above models the layer stack, which is what a placement
    # decision can move. Unchunked, none of it is where the regime B peak
    # lives: the peak instant is the cross-entropy backward, and the mass that
    # puts it there is the logit chain, which no bundle owns. Left out, the
    # schedule reports no pressure on a card about to OOM.
    #
    # The chunked loss inverts that. With a fixed chunk SIZE the chain is a
    # constant rather than a multiple of sequence length, so the peak instant
    # leaves the CE backward for a layer replay and the layer stack becomes
    # the peak again. The schedule reproduces that move rather than being
    # told it --- see tests/test_head_ce_byte_model.py section 7.
    #
    # The three slots below are deliberately NOT registry keys and carry no
    # 'act_' prefix, so the gate can see their bytes and no path can ever
    # select them for eviction.
    _FINAL_NORM_ID = 'nonmig_final_norm'
    _LOGITS_ID = 'nonmig_logits'
    _LOGIT_GRAD_ID = 'nonmig_logit_grad'

    # Bytes per (token x vocab entry) in the logit chain. 12 B are live from
    # the head's forward through the CE backward; the backward adds its fp16
    # gradient w.r.t. logits for a total of 14 B at the peak instant. The
    # per-vocab-entry root-FSDP parameter and optimizer state stays at 0 here
    # for the same reason every other param does --- the planner anchors
    # params through the allocator's current_live, not the byte map. Held to
    # the measured four-rank lines by tests/test_head_ce_byte_model.py.
    _LOGIT_FWD_BYTES_PER_TOKEN_VOCAB = 12
    _LOGIT_BWD_BYTES_PER_TOKEN_VOCAB = 2

    # Under the chunked loss the peak instant prices at 12 B, not 14, which is
    # not derivable from the unchunked constants: the replayed chain and the
    # gradient w.r.t. that chunk's logits are NOT both resident at the peak.
    # The unchunked path keeps 14: it never replays, so the
    # chain it saved and the gradient are co-resident. One chunk covering the
    # whole sequence is therefore NOT the unchunked case --- chunked_ce
    # checkpoints every chunk and HuggingFace's loss checkpoints none.
    _LOGIT_CHUNKED_BYTES_PER_TOKEN_VOCAB = 12

    # The final RMSNorm sits outside every checkpoint wrapper, so two hidden
    # planes survive forward: the last layer's output boundary (no layer's
    # input, therefore claimed by no residual slot) and the norm's own output.
    # The term is depth-independent: 2 x hidden x 2 B per token.
    _FINAL_NORM_PLANES = 2

    # ── The layer backward's gradient transient ────────────────────────────
    #
    # Everything above models SAVES. A backward also allocates the gradients
    # it is computing, and they are co-resident with the saves that produce
    # them. Unchunked this does not matter: the peak instant is the CE
    # backward, before any layer backward runs. Under the chunked loss the
    # peak instant IS a layer backward, and omitting the term under-projects
    # the peak by a depth-independent 14.27 MiB/x (tools/layer_bwd_livebytes.py)
    # --- the direction that under-relieves a feasibility gate into an OOM.
    #
    # Counted, not fitted. tools/layer_bwd_livebytes.py profiles the live set
    # through a real backward and buckets it by width in elements per token:
    # the MLP holds SEVEN intermediate-width planes co-resident against the
    # four this model carries as saves. Three grad planes is also what the
    # graph gives --- `down(silu(gate(x)) * up(x))` produces grad-product,
    # grad-silu and grad-up, and mul backward needs all three at once.
    #
    # The two hidden-width planes are the residual branch: `h = x + mlp(...)`
    # makes the gradient arriving at the layer and the gradient entering the
    # MLP two distinct tensors.
    #
    # The MLP stage is the binding one in regime B --- intermediate is 2.67x
    # hidden here, so the attention stage's grads are never the widest set, and
    # they are deliberately NOT modelled. In regime A they would be (the scores
    # grad is quadratic); that case is not measured, so the gate does not
    # apply there.
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
        """Not a registry key and no 'act_' prefix — visible to the peak walk,
        reachable by no eviction path. Same contract as the head/CE slots."""
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

    # ── Activation id helper ───────────────────────────────────────────────

    @staticmethod
    def _activation_id(layer_idx: int, component: str) -> str:
        """Build the tensor-id that matches fsdp_profiler.py's registry key
        for this activation. Components: q, k, v, attn_scores,
        swiglu_gate, swiglu_up, swiglu_product.

        See the activation registration sites in fsdp_profiler.py, whose
        keys these strings must match exactly.
        """
        return f"act_layer_{layer_idx}_{component}"

    # The seven components above are the tensors we can name. They are not
    # all the tensors autograd saves — an attention bundle saves more objects
    # than its named four. The remainder (rotary cos/sin, the bf16
    # probability tensor, the pre-softmax logits, q/k/v reshapes that alias
    # their projections) has no stable name across Transformers versions, so
    # it gets one anonymous slot per bundle whose byte count pack_fn measures.
    # Zero until observed; see PlacementEngine._promote_observed_relief.
    _RESIDUAL_KINDS: tuple[str, ...] = ('attention', 'mlp')

    # The named component whose create/free/read events each residual shadows.
    # Chosen as the LAST component of its bundle so the residual is never the
    # first create or first bwd read of a layer — those two positions define
    # _first_fwd_create_by_layer / _first_bwd_read_by_layer, and the planner's
    # projection window is anchored on them.
    _RESIDUAL_ANCHOR: dict[str, str] = {
        'attention': 'attn_scores',
        'mlp': 'swiglu_product',
        # The whole-layer unit's slot shadows the layer's last component for
        # the same reason: it must not define the layer's first fwd create or
        # first bwd read. Consulted only when 'layer' is in _RESIDUAL_KINDS.
        'layer': 'swiglu_product',
    }

    # Residual kinds that SURVIVE the checkpoint discard at end-of-forward.
    # Under the whole-layer unit the residual slot is the layer's input
    # boundary --- the one tensor checkpoint_wrapper keeps --- and all L of
    # them accumulated are the entire migratable supply, so freeing it with
    # the interior models that supply out of existence and under-projects the
    # peak. Under per-module the slot is the anonymous remainder of the
    # bundle's saves, which the discard really does take.
    _CHECKPOINT_SURVIVING_RESIDUAL_KINDS: frozenset = frozenset({'layer'})

    def boundary_bytes(self) -> int:
        """Bytes in one residual-stream boundary.

        Under the whole-layer unit this is the entire mass a layer's
        checkpoint saves, and the only thing a placement decision can move.
        Measured as exactly ``2 B x d_model`` per token.
        """
        return (self.batch_size * self.seq_len
                * self.hidden_size * self.dtype_bytes)

    @staticmethod
    def residual_id(layer_idx: int, kind: str) -> str:
        """Tensor-id of the anonymous remainder slot for one bundle.

        Not a registry key — no profiler hook registers it. It exists only
        in the schedule's byte model so ``peak_in_range`` can see bytes the
        named components do not cover.
        """
        return f"act_layer_{layer_idx}_{kind}_residual"

    # ── Rebuild API (seq_len / policy change) ──────────────────────────────

    def rebuild_on_seq_len_change(self, new_seq_len: int) -> AccessSchedule:
        """Re-build the schedule for a new sequence length.

        Event ordering is independent of seq_len, but downstream consumers
        (the placement engine's byte-budget calculations) depend on it via
        per-tensor sizes the registry tracks. Exposed for API symmetry with
        ``rebuild_on_policy_change``.
        """
        self.seq_len = new_seq_len
        return self.build_schedule()

    def rebuild_on_policy_change(
        self,
        checkpoint_policy: dict[int, bool],
    ) -> AccessSchedule:
        """Re-build the schedule with an updated checkpoint policy.

        For a caller that flips a layer between retained and checkpointed.
        """
        self.checkpoint_policy = dict(checkpoint_policy)
        return self.build_schedule()

    # ── Telemetry: write predicted next-access into the registry ───────────

    def populate_registry(self, registry: TensorRegistry) -> int:
        """Walk the schedule once and stamp ``predicted_next_access_step``
        onto each activation record that already exists in ``registry``.

        Returns the number of records updated. Skips records whose tensor_id
        does not appear in the schedule (e.g., parameter shards — those use
        a different naming convention and are handled by the FSDP
        coordinator).

        Reminder: this is a one-shot snapshot at init. Do NOT consult this
        field at runtime — query ``schedule.next_read_after(...)`` instead,
        because the snapshot ages instantly as ``current_step`` advances.
        """
        if self._schedule is None:
            raise RuntimeError(
                "AccessPredictor.populate_registry() called before "
                "build_schedule(); build the schedule first.")

        # First scan: for each tensor_id with a create event, record its
        # birth step. We use the first create (recomputes re-create the
        # same tensor_id; the original fwd create is the natural anchor).
        birth_step: dict[str, int] = {}
        for ev in self._schedule.events:
            if ev.access_type == 'create' and ev.tensor_id not in birth_step:
                birth_step[ev.tensor_id] = ev.step_idx

        # Second pass: for each registry record whose key is in birth_step,
        # look up the next read after the birth step.
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

    # ── Accessors ──────────────────────────────────────────────────────────

    @property
    def schedule(self) -> Optional[AccessSchedule]:
        """The most recently built schedule, or None if build_schedule()
        has not yet been called."""
        return self._schedule
