"""
Placement Engine — Component 3 of the tiered memory manager.

Hosts ``PlacementEngine``, the decision core. Supporting classes live in
their own modules:
    - HBMBudget                          → hbm_budget.py
    - PlacementDecision, PlacementDecisionReason → decisions.py
    - InflightOp, MemoryStreamManager    → stream_manager.py

All four names are re-exported from this module below so
``from memory_manager.placement_engine import ...`` paths work.

Integration wiring lives in fsdp_profiler.py (per-layer + submodule hooks)
and train_fsdp_profile.py (driver).
"""

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

# Re-exports — keep `from .placement_engine import ...` import paths stable.
# See decisions.py / hbm_budget.py / stream_manager.py for the classes.
from .decisions import PlacementDecision, PlacementDecisionReason  # noqa: F401
from .hbm_budget import HBMBudget  # noqa: F401
from .stream_manager import InflightOp, MemoryStreamManager  # noqa: F401


# ---------------------------------------------------------------------------
# Packed-eviction descriptor.
# ---------------------------------------------------------------------------
@dataclass
class PackedEvict:
    """State of one activation that pack_fn staged out to host memory.

    It is a mutable object on purpose: pack_fn creates it during
    *forward*, the backward prefetcher fills in ``prefetch_op`` / ``dst``
    some layers ahead of the read, and unpack_fn consumes it. An immutable
    tuple could not carry that hand-off.

    ``dst is None`` means no prefetch has been issued; unpack_fn falls
    back to the on-demand path (expected for the first layers of backward,
    which backward reaches before any lookahead window covers them).
    """
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
    # Realized-absence accounting, in layer-interval ordinals (profiler
    # snapshot indices). The bytes are only provably gone between the D2H
    # completing and the restore allocating its destination — eviction buys a
    # duty-cycled window, not lifetime absence, so relief can only be claimed
    # for the intervals inside that window.
    evict_ordinal: Optional[int] = None
    d2h_done_ordinal: Optional[int] = None
    restore_ordinal: Optional[int] = None
    # Layout of the ORIGINAL save, and an optional in-HBM reference copy of
    # it. Both are None unless a
    # diagnostic flag is on. The layout fields exist because the restore path
    # rebuilds the tensor with ``torch.empty(shape, dtype, device)`` — always
    # contiguous, always offset 0 — so a save that was a strided view comes
    # back with the right VALUES and the wrong LAYOUT, which a value
    # comparison cannot see. ``ref`` is the value-side check and costs the
    # eviction's own relief, so it is a short-sequence diagnostic only.
    src_stride: Optional[tuple] = None
    src_storage_offset: Optional[int] = None
    src_contiguous: Optional[bool] = None
    ref: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# PlacementEngine
# ---------------------------------------------------------------------------

class PlacementEngine:
    """Decision-making core of the tiered memory manager.

    Lifecycle:

        construct
            └─ build schedule + populate registry's predicted_next_access_step
        per training step:
            on_training_step_begin()                   reset _current_step
            for L=0..N-1:
                on_pre_forward(L)                      decide L's bundles
                (training fwd L runs; pack_fn intercepts saves per flags)
                on_post_forward(L)                     prefetch lookahead
            for L=N-1..0:
                on_pre_backward(L)                     classify L's activations
                (training bwd L runs; unpack_fn restores evict/recompute)
                on_post_backward(L)                    free + prefetch L-1
            on_training_step_end()

    Eviction scoring:
        score = (next_read_step - current_step) / size_bytes
        higher = better candidate (won't be needed soon relative to size)
        SwiGLU triples score as a group (min next_read / sum bytes)
        tie-break: lower layer_idx evicts first

    Evict-vs-recompute: compares exposed_evict_ms (transfer cost minus
    available overlap on the mem stream) against exposed_recompute_ms
    (recompute_ms plus exposed_reallgather_ms).
    """

    # Per-layer event counts in the schedule. These constants assume
    # checkpoint_policy is all-False at registration time. The engine
    # advances _current_step by these constants at each hook fire point.
    EVENTS_PER_FWD_LAYER = 8   # 1 param read + 7 activation creates
    EVENTS_PER_BWD_LAYER = 16  # 1 param read + 7 reads + 7 frees + 1 grad free

    # Report a bundle whose observed owned bytes land below this fraction of
    # its analytic estimate. The healthy direction is observation ABOVE the
    # model, so anything this far below means the model is pricing tensors the
    # kernel never creates. 0.5 rather than something tighter: this is a "the
    # model is wrong about the architecture" alarm, not a calibration check.
    _OVERBOOK_REPORT_RATIO = 0.5

    def __init__(
        self,
        registry: TensorRegistry,
        predictor: AccessPredictor,
        hbm_budget: HBMBudget,
        buffer_pool: PinnedBufferPool,
        mem_stream,                   # MemoryStreamManager or fake (CPU tests)
        calibration: CalibrationResult,
        fsdp_coord: NoOpCoordinator,
        prefetch_window_layers: int = 2,
        dry_run: bool = True,
        verbose: bool = True,
        # Exposed-cost knobs.
        # ``overlap_safety_factor`` deflates the available overlap window
        # so borderline decisions tilt toward recompute when our wall-clock
        # estimate is uncertain. Default 0.8.
        # ``default_layer_*_ms`` is the fallback used in step 0 and for any
        # layer whose timing the integration has not yet pushed via
        # record_layer_timing(). Tune to the cluster's per-layer cost or
        # leave at defaults — the engine averages known values when sized,
        # so a single step of timing data already replaces the defaults.
        overlap_safety_factor: float = 0.8,
        default_layer_fwd_ms: float = 30.0,
        default_layer_bwd_ms: float = 60.0,
        # Cap on per-op latency records written to JSON. Stats (mean / p95 /
        # drift flag) are computed over ALL ops; only the per-op log itself is
        # truncated to the last N for JSON size. 2000 covers ~6 steps of normal
        # traffic (~320 ops/step) without bloating the JSON.
        per_op_log_max: int = 2000,
        # Treat the overlap window O(w) as a budget consumed by each committed
        # eviction rather than a constant every decision may read in full.
        # Off lets every decision read the full window (ablation).
        window_debit: bool = True,
        # Feed the peak projection from the bytes pack_fn observes a bundle
        # actually own, not from the seven-component analytic model. Off
        # uses the analytic model alone (ablation).
        observed_relief: bool = True,
        # The recompute UNIT. 'per_module' is the attention/mlp bundle pair.
        # 'whole_layer' collapses them into one 'layer' bundle, which is the
        # unit checkpoint_wrapper imposes: the outer saved-tensor seam sees
        # one residual-stream boundary per layer and nothing else
        # (tests/test_checkpoint_bundle_ordering.py). Selecting it here is what
        # makes every _BUNDLE_COMPONENTS iteration in the planner enumerate one
        # bundle per layer instead of two.
        bundle_unit: str = 'per_module',
    ) -> None:
        if bundle_unit not in ('per_module', 'whole_layer'):
            raise ValueError(
                f"bundle_unit must be 'per_module' or 'whole_layer', "
                f"got {bundle_unit!r}")
        self._bundle_unit = bundle_unit
        # Which enter_bundle provenances are subtracted from a bundle's OWNED
        # (relief) bytes. 'module' is FSDP's unsharded weights, never the
        # bundle's to free. 'input' is the inbound residual, which only belongs
        # to another bundle under the per-module unit — under whole-layer it is
        # the sole thing the bundle owns and the only tensor migration can take.
        self._foreign_provenance_excluded: frozenset[str] = (
            frozenset({'module', 'kwarg'}) if bundle_unit == 'whole_layer'
            else frozenset({'module', 'input', 'kwarg'}))
        if bundle_unit == 'whole_layer':
            self._BUNDLE_COMPONENTS = {'layer': self._WHOLE_LAYER_COMPONENTS}
            # The whole-layer schedule is built with checkpoint_policy all-True,
            # which adds 7 interior frees to each forward layer and an 8-ordinal
            # recompute block to each backward layer. The cursor advances by
            # these, so they must match the schedule the predictor actually
            # built or every ordinal-keyed lookup — the window ledger's slot
            # keys included — reads the wrong layer.
            self.EVENTS_PER_FWD_LAYER = 15   # 8 + 7 interior frees
            self.EVENTS_PER_BWD_LAYER = 24   # 16 + 1 param read + 7 re-creates
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
        # The profiler resets CUDA's high-water mark at every snapshot to
        # attribute peaks to intervals, which would truncate the step-level
        # read below. attach_engine() installs the profiler's shadow max here
        # so observed_peak_mb stays a step-level peak; None on the
        # no-profiler path.
        self._step_peak_source = None
        # Exposed-cost knobs as instance state.
        self._overlap_safety_factor: float = float(overlap_safety_factor)
        self._default_layer_fwd_ms: float = float(default_layer_fwd_ms)
        self._default_layer_bwd_ms: float = float(default_layer_bwd_ms)

        # Schedule construction is the engine's responsibility: the
        # predictor is freestanding until build_schedule() is called here.
        self._schedule: AccessSchedule = predictor.build_schedule()
        # populate_registry() must run AFTER the first forward pass —
        # activations are only registered when the profiler's
        # forward-pre-hook fires for each layer; at engine init the registry
        # holds only param shards. Deferred to the end of the first training
        # step via _registry_populated flag.
        self._registry_populated: bool = False
        self._num_layers: int = predictor.num_layers

        # Current step cursor — maps the live training pass to a position
        # in the static schedule. Reset to 0 per training step. Advanced
        # in EVENTS_PER_FWD_LAYER / EVENTS_PER_BWD_LAYER chunks at each
        # hook.
        self._current_step: int = 0
        self._training_step_count: int = 0

        # Telemetry — accumulates across every training step.
        self._decisions: list[PlacementDecision] = []
        self._decisions_at_step_start: int = 0
        # What the policy itself costs to run, as distinct from
        # what it decides to do. Wall clock inside the per-layer planner only
        # — projection, criterion and gate — with the coordinator hook and the
        # inflight reap deliberately outside, since those are mechanism.
        self._plan_ns: int = 0
        self._plan_calls: int = 0
        # tensor_id -> last InflightOp submitted (for stall-recovery wait).
        self._inflight_by_tensor: dict[str, InflightOp] = {}

        # Per-layer wall-clock tables. Populated by
        # record_layer_timing() from the profiler's LayerProfile after
        # each step. Empty → engine uses the defaults above for the
        # _step_to_ms / _event_to_ms helpers.
        self._per_layer_fwd_ms: dict[int, float] = {}
        self._per_layer_bwd_ms: dict[int, float] = {}

        # Running accumulator of FLOPs the engine has charged to recompute.
        # Surfaced in summary() and JSON.
        self._total_recomputed_flops: int = 0

        # Per-op timing accumulators. _reap_inflight calls
        # mem_stream.op_pcie_ms on each completed op and rolls the numbers
        # up here. Surfaced in summary() / JSON to validate against the
        # calibration's predicted ms.
        # Split by direction so D2H (evict) and H2D (prefetch) can be
        # diffed against the asymmetric PCIe lanes individually.
        self._total_pcie_ms_evict: float = 0.0
        self._total_pcie_ms_prefetch: float = 0.0
        self._total_bytes_evict: int = 0
        self._total_bytes_prefetch: int = 0
        self._total_timed_ops_evict: int = 0
        self._total_timed_ops_prefetch: int = 0

        # Per-op observed-vs-predicted latency log. Each entry: {tensor_id,
        # direction, nbytes, observed_ms, predicted_ms_idle,
        # predicted_ms_contention}. Two accumulators of ratios
        # (observed/predicted) feed the summary stats and the calibration drift
        # flag. Ratio lists are kept per-direction because D2H and H2D map to
        # different physical PCIe lanes; aggregating would hide
        # direction-specific drift.
        self._per_op_log_max: int = int(per_op_log_max)
        self._per_op_latency_log: list[dict] = []
        # Untruncated size histogram.
        # _per_op_latency_log is capped to the last _per_op_log_max records
        # at summary() time, so on a real run it survives only as a tail
        # sample. Pool provisioning needs the whole run's demand, and a
        # counter is cheap enough to keep unbounded.
        self._nbytes_hist: dict[str, dict[int, int]] = {
            'evict': {}, 'prefetch': {}}
        # Ratio samples: observed_ms / predicted_ms_*  per direction.
        # Captured at reap time iff both observed and predicted are
        # available (live mode + calibration present).
        self._ratio_obs_over_idle_evict: list[float] = []
        self._ratio_obs_over_idle_prefetch: list[float] = []
        self._ratio_obs_over_cont_evict: list[float] = []
        self._ratio_obs_over_cont_prefetch: list[float] = []

        # Current submodule bundle.
        # Set by enter_bundle() from the LlamaAttention/LlamaMLP forward
        # pre-hooks; cleared by exit_bundle() from the matching post-hooks.
        # Read by pack_fn to attribute each autograd-saved
        # tensor to the right bundle (attention or mlp of layer L). Forward
        # is single-threaded so a plain instance attribute is sufficient; if
        # a future change introduces parallel forwards in the same process,
        # switch to threading.local.
        self._current_bundle: Optional[tuple[int, str]] = None

        # Pre-decision cache keyed by layer_idx. Populated by
        # on_pre_forward(L); read by on_post_forward to know whether
        # eviction was already decided + executed. Cleared at training-step
        # boundary so old decisions don't bleed across steps. pack_fn reads
        # this same structure to know whether each autograd-saved tensor in the
        # current bundle has been marked for offload.
        self._pre_decisions_by_layer: dict[int, list[PlacementDecision]] = {}

        # Live offload control.
        # When ``engine_live`` is True (set by the integration when
        # --engine_live CLI flag is on), pack_fn / unpack_fn become the
        # authoritative offload path. _apply_decision still flips registry
        # tier flags and records telemetry — the actual bytes flow through
        # pack_fn / unpack_fn.
        self._engine_live: bool = False
        # Bundle-level evict map: (layer_idx, kind) -> should evict?
        # Computed in on_pre_forward by collapsing per-component decisions
        # via bundle-as-one ("any component marked evict → bundle evicts").
        # Read by pack_fn on every save_for_backward.
        self._packed_evict_bundles: dict[tuple[int, str], bool] = {}
        # Per-step accumulator of pack_fn events for telemetry. Cleared at
        # step start. Each entry: (layer_idx, kind, nbytes, op_id).
        self._packed_events: list[tuple] = []
        # Zero-element saves passed through un-booked. Under the whole-layer
        # unit this should be exactly num_layers per step — one checkpoint
        # placeholder per wrapper — so it doubles as the check that the seam
        # sees one boundary per layer and nothing else.
        self._packed_zero_elem_count: int = 0
        # Planned-vs-realized eviction. A denied pool acquire returns the
        # tensor unchanged: the bundle is neither evicted nor recomputed, so
        # its bytes stay resident while the planner still books their
        # relief. Below the memory ceiling that distorts measurements; above
        # the ceiling it is the OOM. These counters make the gap a first-class
        # number, and ``_strict_evict`` turns it into an exception for frontier
        # runs.
        self._evict_realized_bytes: int = 0
        self._evict_realized_count: int = 0
        self._evict_denied_bytes: int = 0
        self._evict_denied_count: int = 0
        self._evict_denial_log: list[dict] = []
        self._evict_denial_log_max: int = 512
        # Saves that are not on the GPU at all. The
        # memory-efficient SDPA kernel saves its philox RNG seed/offset as
        # 0-dim int64 CPU tensors. Evicting one is meaningless — it holds no
        # HBM, so it frees none — and counting its bytes as realized relief
        # would over-book by exactly the foreign-set error mode. It is NOT a
        # denial: nothing planned was lost, so it must not raise under
        # strict_evict. Separate counter, reported separately.
        self._evict_skipped_nonhbm_bytes: int = 0
        self._evict_skipped_nonhbm_count: int = 0
        self._evict_skipped_nonhbm_kinds: set[str] = set()
        # A bundle the planner needed relief from and the criterion
        # could give none. Only reachable under pressure, so every hit is a
        # projected peak nothing acted on — a measurement error below the
        # ceiling and an OOM above it, same class as a denied eviction.
        self._no_relief_bytes: int = 0
        self._no_relief_count: int = 0
        self._no_relief_notes: set[str] = set()
        # Feasibility gate: steps whose projection stayed over
        # capacity after every bundle in the step had been offered. Kept apart
        # from the no_relief counters on purpose — those sum over bundles, and
        # a step-level residual folded in would corrupt a quoted total. The
        # distinction is the verdict: bundles the criterion would not release,
        # versus a breach that survives releasing everything.
        self._infeasible_steps: list[dict] = []
        # Training-step index, distinct from _current_step, which is an event
        # ordinal within the step. Starts at 0 and on_training_step_begin makes
        # the first step 1, so a record written before any step is visibly 0.
        self._training_step: int = 0
        # TRAP: the restore rebuilds with torch.empty (dense, offset 0), so a
        # non-contiguous save — the q saves the fused cutlass backward reads —
        # comes back with the right values and the wrong LAYOUT, and the
        # backward produces NaN. Correct restoration needs the stride and
        # storage offset, rebuilt with as_strided. Restore-path instruments,
        # all inert by default: _evict_filter bisects by bundle kind or byte
        # class, _layout_census records the ORIGINAL layout of every evicted
        # save (a value comparison cannot see this), _verify_restore keeps an
        # HBM reference copy to catch a torn or racing copy — that one costs
        # the eviction's entire relief, so it is never a frontier arm.
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
        # The device whose bytes eviction is trying to relieve. Compared by
        # type, not index: 'is this save on the host or on the accelerator' is
        # the only distinction that matters, and cuda vs cuda:0 must not read
        # as a mismatch. None disables the check, which is what CPU-only test
        # harnesses want.
        _msdev = getattr(mem_stream, '_device', None)
        self._evict_device_type: Optional[str] = (
            torch.device(_msdev).type if _msdev is not None else None)
        # Bundles with at least one staged save and at least one denial: the
        # worst outcome, since the copies were paid but the storages are still
        # resident, so no relief is realized.
        self._evict_partial_bundles: set[tuple[int, str]] = set()
        self._evict_staged_bundles: set[tuple[int, str]] = set()
        self._evict_denied_bundles: set[tuple[int, str]] = set()
        self._strict_evict: bool = False
        # One warning per (kind, reason) per step keeps a denial storm from
        # burying the log while still making the first one impossible to miss.
        self._evict_denial_warned: set[tuple] = set()

        # Realized absence.
        # Ordinals come from the profiler's snapshot index — see attach_engine.
        # Left None when no profiler is wired, in which case absence is simply
        # not measured rather than guessed.
        self._interval_ordinal_source = None
        self._peak_interval_source = None
        # pack_id -> live record, so _reap_inflight can stamp a D2H completion
        # without walking every bundle's list.
        self._record_by_pack_id: dict[str, PackedEvict] = {}
        # One entry per evicted bundle-member per step, with the interval span
        # over which its bytes were provably gone and whether that span covered
        # the step's peak interval. This is what makes "the relief was
        # delivered at the instant it was needed" a measurement.
        self._absence_log: list[dict] = []
        self._absence_log_max: int = 512
        self._absence_bytes_total: int = 0
        self._absence_bytes_covering_peak: int = 0
        self._absence_empty_window_count: int = 0
        self._absence_unrestored_count: int = 0
        # Bundle -> live PackedEvict records for the current step, so the
        # backward-side prefetcher can find a layer's staged tensors and start
        # their H2D *before* unpack_fn asks for them. Mirrors
        # _packed_evict_bundles: the planner and the physical path meet at
        # bundle granularity, never at tensor identity.
        self._packed_by_bundle: dict[tuple[int, str], list[PackedEvict]] = {}
        # Bundles whose H2D has already been issued this step. Guards
        # against a second issue from an overlapping lookahead window.
        self._prefetch_issued_bundles: set[tuple[int, str]] = set()
        # Prefetch-ahead effectiveness counters, run-cumulative. 'ahead_hits'
        # is a restore that found its bytes already in flight or landed;
        # 'on_demand' is one that had to issue the H2D at the read. A run
        # dominated by on_demand means the lookahead never reached the
        # bundles, and prefetch-ahead bought nothing.
        self._prefetch_ahead_issued: int = 0
        self._prefetch_ahead_hits: int = 0
        self._prefetch_on_demand: int = 0

        # Measured recompute wall-clock, fed by unpack_fn via
        # record_recompute_timing. TRAP: keyed (layer_idx, KIND) — a layer's
        # attention sample must never be served to its mlp bundle.
        self._recompute_observed_ms: dict[tuple, list[float]] = {}

        # Per-CLASS fallback pool. A FLOPs-per-byte proxy prices an attention
        # replay far too cheap, which makes `exposed_evict_ms == 0` an
        # algebraic identity. Layers are identical here, so one class sample
        # generalizes.
        self._recompute_class_ms: dict[str, list[float]] = {}
        # A class only earns a sample if something in it is physically
        # recomputed, and mlp never is, so one bundle per class is forced to
        # recompute on first sight. Tracked at DECISION time, not sample time,
        # or every layer would probe while waiting for the first sample.
        self._class_probe_issued: set = set()

        # Per-step observed-vs-projected peak log (log only; the budget is
        # static). Its `observed_peak_mb` is the canonical peak for every run
        # in the run table.
        self._hbm_drift_log: list[dict] = []

        # Force-action knob for baselines.
        # 'none' (default) → planner runs its full pressure/coverage logic
        # and routes through the exposed-cost criterion. 'evict' or
        # 'recompute' → planner bypasses all of that and marks every
        # reducible bundle for the forced action every step (overriding the
        # breach check and the per-step pick cap). Cost-model reason is
        # still filled, for telemetry parity.
        self._force_action: str = 'none'
        # Fixed migration stride on a recompute base. 0 = off.
        self._force_stride: int = 0

        # Cumulative-coverage planner state.
        # Across all layers in one step the planner accumulates which
        # activation tensor_ids it has already chosen to evict / recompute,
        # so the next layer's projection walks a schedule that has those
        # tids excluded. Without this, every layer sees the SAME projected
        # peak (full fwd peak) and picks its bundles independently, which
        # over-picks. Cleared in on_training_step_begin so decisions don't
        # leak between steps.
        self._planner_excluded_tids: set[str] = set()
        # When each excluded tid's bytes come back. A migration buys absence
        # over an INTERVAL, not a lifetime — booking the lifetime over-states
        # the relief. Keyed by the same tids as the set above and cleared
        # with it.
        self._planner_restore_ordinals: dict[str, int] = {}
        # Per-step pick cap: a safety rail against degenerate planning, not a
        # throttle. Read off the bundles that actually exist (layers x bundles
        # per layer), so under the whole-layer unit it never sits below the
        # layer count, where a cap-induced retain under pressure means
        # accepting the OOM.
        self._planner_picks_this_step: int = 0
        self._planner_max_picks_per_step: int = (
            self._num_layers * len(self._BUNDLE_COMPONENTS))

        # Per-bundle recompute state. Mirrors
        # _packed_evict_bundles: pack_fn checks the map; if True for the
        # current bundle, drop the tensor + return a recompute descriptor.
        # unpack_fn then re-runs the bundle's submodule forward on demand,
        # caching the captured tensors so multiple unpacks within one bundle
        # share a single re-execution.
        self._packed_recompute_bundles: dict[tuple[int, str], bool] = {}
        # (L, kind) -> (args, kwargs) captured at bundle entry. Used by
        # _rerun_bundle_forward to call module(*args, **kwargs). Only
        # populated for bundles where the recompute decision applies.
        self._recompute_input_cache: dict[tuple[int, str], tuple[tuple, dict]] = {}
        # (L, kind) -> nn.Module reference for the bundle's submodule
        # (LlamaAttention or LlamaMLP). Populated alongside input cache.
        self._recompute_module_cache: dict[tuple[int, str], Any] = {}
        # (L, kind) -> int. Incremented every time pack_fn returns a
        # recompute descriptor for the bundle. Order matters: unpack_fn
        # reads the cache list at the same index.
        self._recompute_save_counters: dict[tuple[int, str], int] = {}
        # (L, kind) -> list[Tensor] captured during the re-execution.
        # Built lazily by the first unpack_fn call for the bundle; reused
        # by subsequent unpacks within the same bundle.
        self._recompute_tensor_cache: dict[tuple[int, str], list] = {}
        # Re-entrancy guard. True while _rerun_bundle_forward is mid-run;
        # enter_bundle / exit_bundle / engine-side bookkeeping short-circuit
        # so the re-run doesn't corrupt the outer step's state.
        self._in_recompute: bool = False

        # ── The overlap window O(w) as a DEBITED ledger ───────────────────
        # Capuchin's max(0, |b|/B(w) - O(w)) is a single-decision formula:
        # with many bundles deciding per step, each would subtract the same
        # window in full and the criterion would price the whole set as free.
        # O(w) is also measured from the step the criterion is lengthening,
        # so the estimate inflates as the policy degrades.
        # Keyed by schedule event ordinal, so two decisions compete only over
        # the slots their windows share. Reset per step.
        self._window_debit_bytes: dict[int, float] = {}
        self._window_debit_enabled: bool = window_debit
        # Running totals for the run summary.
        self._window_debit_committed_bytes: int = 0
        self._window_debit_refusals: int = 0

        # ── Observed bundle byte model ────────────────────────────────────
        # AccessPredictor's analytic model undercounts what autograd actually
        # saves, and hand-enumerating HF's saves re-breaks on every
        # Transformers bump, so pack_fn reports the truth instead. Step 0 uses
        # the analytic estimate; steps 1+ use the observation.
        self._obs_bundle_bytes_cur: dict[tuple[int, str], int] = {}
        self._obs_bundle_members_cur: dict[tuple[int, str], list] = {}
        self._obs_bundle_bytes: dict[tuple[int, str], int] = {}
        self._obs_bundle_members: dict[tuple[int, str], list] = {}

        # Bundles whose observed owned bytes came in far below the
        # analytic model. See _promote_observed_relief for why this direction
        # cannot be self-correcting and therefore has to be reported.
        self._overbooked_bundles: dict[tuple[int, str], dict] = {}
        self._overbook_warned: bool = False

        # ── The RELIEF half ───────────────────────────────────────────────
        # TRAP: the sum above is the right COST number — every save costs a
        # pinned buffer and a D2H copy, aliases included — and the WRONG relief
        # number, because dropping a bundle frees storages, not saves. Three
        # classes of byte in that sum free nothing:
        #
        #   aliases   a 9 MiB projection saved three times sums to 27 MiB and
        #             frees 9
        #   params    FSDP's unsharded weights are saved by the matmuls; they
        #             belong to FSDP's lifecycle, not the bundle's
        #   the input the residual-stream tensor entering the bundle is the
        #             PREVIOUS bundle's output — freeing it is that bundle's
        #             relief to claim, and double-counting it here would let
        #             two bundles each book the same 9 MiB
        #
        # The input clause is PER-MODULE ONLY. Under the whole-layer unit the
        # inbound residual is the bundle's entire owned mass and no other
        # bundle can claim it: layer L-1's own owned set is ITS input, so a
        # boundary excluded here is credited to nobody and falls through the
        # gap between bundles. Excluding it would leave this dict empty for
        # every layer bundle, skip the promotion loop entirely and take the
        # over-book alarm offline — a silence, not a zero. See
        # _foreign_provenance_excluded.
        #
        # Taking the summed figure as relief can project a NEGATIVE runtime
        # anchor, so it is not merely inaccurate.
        self._obs_bundle_owned_cur: dict[tuple[int, str], int] = {}
        self._obs_bundle_owned: dict[tuple[int, str], int] = {}
        # TRAP: bundles that dropped at least one save for RECOMPUTE this step.
        # Their owned-byte observation is untrustworthy and must not be
        # promoted — the recompute branch drops a storage with no stream
        # dependency, so the allocator can hand the same address to a later
        # save in the SAME bundle, where the ptr-keyed dedup mistakes it for an
        # alias and swallows a real storage (signature: SwiGLU mlp books three
        # intermediate-width planes instead of four). Eviction is immune,
        # because submit_evict's record_stream holds the block until the D2H
        # retires.
        self._obs_bundle_dropped_cur: set[tuple[int, str]] = set()
        # Cumulative across the run: bundles whose observation was refused for
        # the reason above. Non-empty is EXPECTED under recompute, not a fault;
        # it is the visible gap that replaces a silently wrong number.
        self._obs_declined_bundles: set[tuple[int, str]] = set()
        # Per-bundle scratch, valid only between enter_bundle and exit_bundle.
        # Sets of data_ptr ints, never tensor refs — holding refs here would
        # pin the very HBM the engine exists to release.
        self._bundle_seen_ptrs: set[int] = set()
        self._bundle_foreign_ptrs: set[int] = set()
        # Provenance label per foreign ptr — 'module' (weight
        # state), 'input' (positional args) or 'kwarg'. Membership is the set
        # above and is unchanged; this only records which rule matched first,
        # so the histogram below can say WHICH branch subtracted a storage
        # rather than only how big it was. First match wins; a ptr reachable
        # both ways is labeled 'module'.
        self._bundle_foreign_src: dict[int, str] = {}
        self._observed_relief: bool = observed_relief

        # Sizes of the storages the foreign filter SUBTRACTS, as a
        # size -> count histogram per bundle. Diagnostic only. A misclassified
        # storage moves the projection while changing no other telemetry,
        # which is why relief errors hide here; a bucket at
        # B*S*intermediate*dtype would name a subtracted intermediate-width
        # plane (an mlp bundle owns 4 — see tests/test_swiglu_plane_count.py).
        # Keys are "<nbytes>|<source>" so the bucket also says which branch
        # subtracted.
        self._obs_bundle_foreign_cur: dict[tuple[int, str], dict[str, int]] = {}
        self._obs_bundle_foreign: dict[tuple[int, str], dict[str, int]] = {}
        # How many DISTINCT storages the bundle sighted at all.
        # Separates the two ways a relief storage can go missing. The filter
        # subtracting it shows up in the histogram above; a storage never
        # sighted because a dropped tensor's address was recycled by a later
        # save in the same bundle shows up ONLY here, as a short count, since
        # the seen-ptr dedup swallows it in silence. The CPU model of an mlp
        # bundle sights 6 (flat param, inbound residual, four planes).
        self._obs_bundle_distinct_cur: dict[tuple[int, str], int] = {}
        self._obs_bundle_distinct: dict[tuple[int, str], int] = {}

        if verbose:
            print(f"[PlacementEngine] init DONE: dry_run={dry_run}, "
                  f"budget={hbm_budget.total_bytes / (1024 ** 3):.1f} GB, "
                  f"prefetch_window={prefetch_window_layers} layers, "
                  f"num_layers={self._num_layers}, "
                  f"schedule_events={len(self._schedule)}, "
                  f"overlap_safety_factor={self._overlap_safety_factor}")

    # ── Public hooks ────────────────────────────────────────────────────────

    def on_training_step_begin(self) -> None:
        """Reset the schedule cursor at the start of every training step."""
        self._current_step = 0
        # _current_step is an EVENT ORDINAL and resets to 0 here,
        # so it cannot identify a training step. The gate's verdict is one
        # record per step and needs to say which.
        self._training_step += 1
        self._decisions_at_step_start = len(self._decisions)
        self._reap_inflight()
        # Clear pre-decisions from the previous step so a stale entry can't
        # make on_post_forward think the new step's layer was already decided.
        self._pre_decisions_by_layer.clear()
        # Clear bundle evict map + per-step pack events so pack_fn sees fresh
        # state.
        self._packed_evict_bundles.clear()
        self._packed_events.clear()
        # A record whose prefetch never issued still
        # holds its pinned buffer — backward may not reach every packed
        # bundle (e.g. an early OOM, or a bundle whose grad path was cut).
        # Reclaim those here so a missed unpack cannot leak the pool across
        # steps. Records that did prefetch are released by _reap_inflight.
        for records in self._packed_by_bundle.values():
            for r in records:
                if r.prefetch_op is None:
                    self._pool.release(r.host_buf)
        self._packed_by_bundle.clear()
        self._record_by_pack_id.clear()
        self._prefetch_issued_bundles.clear()
        # Clear the recompute side of the bundle state. Save counters and the
        # tensor cache must be empty going into the new step so pack_fn issues
        # fresh indices and the next backward starts with no stale captures.
        self._packed_recompute_bundles.clear()
        self._recompute_input_cache.clear()
        self._recompute_module_cache.clear()
        self._recompute_save_counters.clear()
        self._recompute_tensor_cache.clear()
        # Reset cumulative-coverage state so the
        # new step starts with no carry-over exclusions.
        self._planner_excluded_tids.clear()
        self._planner_restore_ordinals.clear()
        # Reset the per-step pick counter alongside the exclusion set.
        self._planner_picks_this_step = 0
        # The window budget is per-step. Carrying a debit across
        # steps would make step N's evictions charge step N+1's window.
        self._window_debit_bytes.clear()
        # Bundle byte observations are re-gathered every step; the promoted
        # table from the previous step stays live until this one completes.
        self._obs_bundle_bytes_cur.clear()
        self._obs_bundle_members_cur.clear()
        self._obs_bundle_owned_cur.clear()
        self._obs_bundle_foreign_cur.clear()
        self._obs_bundle_distinct_cur.clear()
        self._obs_bundle_dropped_cur.clear()

    def _promote_observed_relief(self) -> None:
        """Write each bundle's measured owned bytes into the schedule's byte
        model as an anonymous residual.

        The residual, not a rescale, is what gets written. ``q`` and
        ``attn_scores`` really are their analytic sizes — the four named
        numbers are independently correct and are read per-tensor elsewhere
        (the registry, the prefetch scorer, Gate 0's analytic fallback).
        Multiplying them by a bundle-level ratio would corrupt four right
        answers to fix one wrong total. So the named components keep their
        sizes and the unmodelled remainder goes into one slot the schedule
        already has events for.

        Clamped at zero: a bundle whose observation lands BELOW the analytic
        sum gets residual 0 rather than a negative one. A negative residual
        would silently cancel a sibling's contribution inside ``peak_in_range``,
        which walks a single running total.

        The clamp means this correction only ever moves the model UP, so an
        over-estimating byte model is permanent — observation cannot walk it
        back (an analytic attention bundle pricing a scores tensor the
        memory-efficient kernel never forms stays over-priced for the whole
        run). The clamp stays (a negative residual is still wrong), but the
        condition is reported instead of absorbed — it is the signature of a
        byte model that does not match the kernel, and it must never be
        silent.
        """
        if not self._observed_relief:
            return
        bm = self._schedule.bytes_by_tensor
        whole_layer = self._bundle_unit == 'whole_layer'
        for key, owned in self._obs_bundle_owned_cur.items():
            layer_idx, kind = key
            # Under whole-layer the named components are the layer's INTERIOR,
            # which the wrapper frees inside its own forward; the bundle's mass
            # is the boundary alone. Comparing the observation against the
            # interior would make the alarm fire on every layer, and make the
            # residual write below clamp the boundary to zero --- deleting the
            # entire migratable supply from the byte model.
            analytic = (
                bm.get(self._predictor.residual_id(layer_idx, kind), 0)
                if whole_layer else
                sum(bm.get(tid, 0) for tid in self._named_tids(
                    self._bundle_tensor_ids(layer_idx, kind))))
            # The over-book alarm reads the untrusted observation on purpose.
            # It is a coarse alarm — it fires only when the analytic model is
            # more than 2x the observation — while the address-reuse bias is
            # bounded at one intermediate-width plane (4/3 on mlp, 0 on
            # attention), far under the threshold. A kernel mismatch in a
            # pure-recompute run must stay catchable. Raising an alarm and
            # booking a quantity are different levels of trust in the same
            # number.
            if analytic > 0 and owned < analytic * self._OVERBOOK_REPORT_RATIO:
                self._overbooked_bundles[key] = {
                    'step': self._training_step_count,
                    'observed_owned_bytes': int(owned),
                    'analytic_bytes': int(analytic),
                    'ratio': round(analytic / max(1, int(owned)), 2),
                    'trusted': key not in self._obs_bundle_dropped_cur,
                }
            # A bundle that dropped saves for recompute is not booked at all —
            # its owned-byte reading is untrustworthy (address reuse makes the
            # ptr dedup swallow a real storage), so the analytic model stays
            # the sole authority. Declining changes no projection (the clamp
            # below swallowed every low reading anyway); it makes the gap
            # VISIBLE instead of silently absorbed. Restoring the booking needs
            # a storage identity that survives address reuse — deliberately not
            # built, because its failure direction is OVER-counting, which
            # would inflate planned relief and under-provision the gate.
            if key in self._obs_bundle_dropped_cur:
                self._obs_declined_bundles.add(key)
                continue
            self._obs_bundle_owned[key] = owned
            # Per-module: the slot is what the observation has that the named
            # components do not. Whole-layer: there is no remainder --- the slot
            # IS the bundle, so the observation replaces the analytic seed
            # outright rather than being differenced against it.
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
        """Drain any still-pending ops, log the step's observed peak, and
        emit a per-step log."""
        self._reap_inflight()

        # Promote this step's pack_fn observations into the live
        # byte table. Only bundles actually observed are overwritten, so a
        # bundle that got no saves this step keeps its previous truth rather
        # than silently reverting to the analytic estimate.
        for key, nbytes in self._obs_bundle_bytes_cur.items():
            self._obs_bundle_bytes[key] = nbytes
            self._obs_bundle_members[key] = list(
                self._obs_bundle_members_cur.get(key, ()))
            self._obs_bundle_foreign[key] = dict(
                self._obs_bundle_foreign_cur.get(key, {}))
            self._obs_bundle_distinct[key] = (
                self._obs_bundle_distinct_cur.get(key, 0))
        self._promote_observed_relief()

        # Stamp predicted_next_access_step onto registry records
        # after the first step completes — by now every layer's forward
        # pre-hook has fired and all activation records exist.
        if not self._registry_populated:
            self._predictor.populate_registry(self._registry)
            self._registry_populated = True

        # Reconcile every staged bundle's
        # absence window against the interval that actually held the peak.
        # Must run before the step counter increments so entries are keyed by
        # the step that just ended, and while _packed_by_bundle is still live
        # (on_training_step_begin clears it).
        self._reconcile_absence()

        # Log this step's observed-vs-predicted peak. Must run BEFORE the step
        # counter increments so log entries are keyed by the step that
        # just ended.
        self._update_hbm_drift()

        if self._verbose:
            recent = self._decisions[self._decisions_at_step_start:]
            counts: dict[str, int] = defaultdict(int)
            for d in recent:
                counts[d.action] += 1
            print(f"[PlacementEngine] step {self._training_step_count}: "
                  f"decisions={dict(counts)}")
        self._training_step_count += 1

    # ── Realized-absence reconciliation ────────────────────────────────────

    def _reconcile_absence(self) -> None:
        """Book, per staged bundle-member, whether its bytes were gone when
        the peak happened.

        A bundle's bytes are provably absent only over the intervals strictly
        between the D2H completing and the restore taking its destination
        storage. Booking relief for anything outside that span is the error
        this method exists to make visible: a restore that lands before the
        peak interval un-does the relief while still paying for both copies,
        and the planner has no way to notice on its own.

        No-op when no profiler supplies interval ordinals — better to report
        nothing than to report an unmeasured guarantee.
        """
        if self._peak_interval_source is None:
            return
        peak_ordinal, peak_phase, peak_bytes = self._peak_interval_source()
        if peak_ordinal < 0:
            return
        for records in self._packed_by_bundle.values():
            for r in records:
                if r.d2h_done_ordinal is None:
                    # Still in flight at step end: never provably absent.
                    self._absence_unrestored_count += 1
                    continue
                # Open interval — the boundary intervals contain both states.
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

    # ── Per-step peak observation (log only) ───────────────────────────────

    def _update_hbm_drift(self) -> None:
        """Record this step's observed peak beside the planner's projection.

        Log only. ``observed_peak_mb`` is the canonical peak for the run and
        is populated whatever the planner did. 'step_ratio',
        'cumulative_correction' and 'updated' are fixed values kept for
        schema compatibility.
        """
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
            # predicted = anchor + the schedule's modelled delta, and only the
            # anchor needs measuring — the delta replays offline from the
            # config. Splitting the projection is what tells a byte-model error
            # from an allocator-state error.
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
        """Return the step's peak allocated bytes (or 0 on CPU-only), then
        reset the high-water mark for the next step.

        The raw CUDA counter only covers the span since the profiler's last
        interval reset, so the profiler's shadow max carries the rest of the
        step. Taking the larger of the two is exact whether or not the
        profiler was enabled for this step.
        """
        if not torch.cuda.is_available():
            return 0
        peak = int(torch.cuda.max_memory_allocated())
        if self._step_peak_source is not None:
            peak = max(peak, int(self._step_peak_source()))
        torch.cuda.reset_peak_memory_stats()
        return peak

    # ── Submodule bundle context ───────────────────────────────────────────

    # Depth 2 reaches position_embeddings=(cos, sin) and past_key_value-style
    # nestings without walking an unbounded object graph on the hot path.
    _FOREIGN_CONTAINER_DEPTH: int = 2

    @staticmethod
    def _collect_storage_ptrs(obj, out: set, depth: int = None) -> None:
        """Add the storage data_ptr of every tensor reachable in ``obj``."""
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
        """Storage ptrs of everything the module owns as weight state.

        ``module.parameters()`` is not sufficient. FSDP with
        ``use_orig_params=False`` calls ``delattr`` on each original parameter
        and ``setattr``s an unsharded VIEW of the flat param in its place, so
        for the whole duration of forward the parameters live in the module's
        ``__dict__`` as plain tensors and are invisible to ``_parameters``.
        A saved weight then looks like a first-sighted storage the bundle owns,
        and since a view reports its whole storage the bundle is credited with
        the entire flat param.
        """
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
        """Mark the start of a LlamaAttention or LlamaMLP forward.

        Called by the submodule forward-pre-hook (wired in fsdp_profiler's
        ``attach_engine``). pack_fn reads ``self._current_bundle``
        to attribute each autograd-saved tensor to the right bundle.

        Defensive: overwrites any stale bundle context. If a prior submodule
        forward raised before its post-hook could clear the context, the
        next pre-hook still produces correct state.

        The optional ``inputs`` / ``kwargs`` / ``module`` arguments let the
        profiler hand over what's needed to physically re-run this bundle's
        forward at backward time. We only capture them for bundles already
        marked recompute by ``on_pre_forward`` — for evict / retain bundles the
        inputs are not needed and holding them would just pin extra HBM. The
        short-circuit on ``_in_recompute`` is what lets
        ``_rerun_bundle_forward`` call ``module(...)`` without the submodule's
        own forward-pre-hook clobbering our outer-step bundle context.
        """
        if self._in_recompute:
            return
        self._current_bundle = (layer_idx, kind)
        bundle = (layer_idx, kind)
        # Snapshot the storages pack_fn must not
        # count as this bundle's own — FSDP's unsharded params and the
        # residual-stream tensor arriving from the previous bundle. Captured
        # for every bundle, not just recompute-marked ones, because the
        # relief model has to be right for bundles the planner refuses too.
        # data_ptr ints only; no tensor is retained.
        self._bundle_seen_ptrs = set()
        foreign: set[int] = set()
        # TRAP: weights must be collected with
        # _module_storage_ptrs, NOT module.parameters(), and args/kwargs with
        # _collect_storage_ptrs, NOT torch.is_tensor. FSDP(use_orig_params=
        # False) deregisters the real Parameters for the duration of forward,
        # so parameters() is EMPTY; and HF passes position_embeddings as a
        # (cos, sin) TUPLE, which is_tensor rejects. A missed weight is not a
        # small error — a view reports its whole storage, so the bundle gets
        # credited with the entire flat param.
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
            # TRAP: sanitize stateful kwargs. HuggingFace LlamaAttention takes
            # ``past_key_value`` (a DynamicCache) and ``use_cache``. The
            # original forward calls ``past_key_value.update(K, V, layer_idx)``
            # which APPENDS K,V into the cache slot and returns the
            # concatenation. If the replay sees the same (now-populated) cache,
            # it appends a SECOND time and produces a 2× seq-len K/V tensor,
            # which surfaces as a replay shape mismatch. The replay should not
            # interact with the cache at all; per- forward DynamicCache slots
            # start empty, so ``past_key_value =None`` reproduces the original
            # K/V exactly. ``use_cache= False`` further short-circuits any
            # cache-update code paths inside the attention forward.
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
        """Mark the end of a LlamaAttention or LlamaMLP forward."""
        if self._in_recompute:
            return
        self._current_bundle = None

    @property
    def current_bundle(self) -> Optional[tuple[int, str]]:
        """Read-only view of the live bundle context (None outside a submodule)."""
        return self._current_bundle

    @property
    def engine_live(self) -> bool:
        return self._engine_live

    @engine_live.setter
    def engine_live(self, value: bool) -> None:
        self._engine_live = bool(value)

    # force_action property. Validates against the
    # three legal values so a typo at the CLI fails loudly at engine setup
    # rather than silently degrading to 'none'.
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

    # ── Restore-path diagnostics (bisection + verification) ────────────────

    @property
    def evict_filter(self) -> str:
        return self._evict_filter

    @evict_filter.setter
    def evict_filter(self, value: str) -> None:
        """Restrict which saves eviction is allowed to touch.

        Grammar, one term only:
            'all'                     no restriction (default)
            'attention' (or 'attn')   only that bundle kind
            'mlp'                     only that bundle kind
            'bytes=N[,M...]'          only saves of exactly those byte sizes
            'notbytes=N[,M..]'        every save EXCEPT those byte sizes

        TRAP: the bundle kind is the string 'attention'. 'attn' is an alias,
        and the kind is checked against the kinds that actually exist rather
        than against a literal a reader might guess wrong: an unmapped kind
        matches NOTHING and evicts zero saves, which reads as a clean result
        and means nothing. filtered_classes reports class names, not just a
        count, so such a mismatch is visible in the telemetry.

        A filtered-out save passes through resident and is counted apart from
        denials: nothing planned was lost, the user excluded it on
        purpose, so it must not raise under strict_evict. The consequence is
        that a filtered arm's realized-relief numbers are meaningless by
        construction — the planner still books the whole bundle. That is
        acceptable because the filter exists to localize a correctness bug,
        never to measure a frontier.
        """
        v = str(value).strip()
        kind, byteset, exclude = None, None, False
        aliases = {'attn': 'attention'}
        # class, not self: the driver validates a mistyped flag against a bare
        # probe object before any engine exists, so the accepted set is the
        # union over both units rather than the one this instance selected.
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

    def _diag_class(self, kind: str, tensor) -> str:  # noqa: ANN001
        """Census/verification bucket: bundle kind, dtype, exact byte size.

        Deliberately the same coordinates the evicted size classes are already
        quoted in, so a census row maps onto a known tensor without a lookup —
        e.g. attn/torch.float32/196608 is the efficient kernel's fp32 logsumexp
        at s=4096.
        """
        return (f'{kind}/{tensor.dtype}/'
                f'{tensor.numel() * tensor.element_size()}')

    def _record_layout(self, kind: str, tensor) -> None:  # noqa: ANN001
        """One free census row per evicted save. No clone, no sync.

        Three independent ways a save can be something other than a compact
        tensor that owns its storage, each of which the restore path erases:
        a non-contiguous stride pattern, a nonzero storage offset, and a
        storage larger than the tensor (i.e. a view into someone else's
        buffer). The D2H stages ``.contiguous().view(-1)``, so all three come
        back as a dense tensor at offset 0.
        """
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

    def _verify_one(self, record: PackedEvict, dst) -> None:  # noqa: ANN001
        """Compare one restored tensor against the copy kept at pack time.

        Two failure families, reported apart because they mean different
        things. A VALUE mismatch means the bytes themselves are wrong — a torn
        copy, a stream-ordering hole, a buffer reused under the D2H. A LAYOUT
        mismatch means the bytes are right and the tensor handed to backward
        is not the tensor forward saved; the restore rebuilds dense at offset
        0, so any strided or offset save reports here with max_abs_diff of
        exactly zero. A backward kernel that reads layout metadata rather than
        just shape can break on the second with nothing wrong in the first,
        which is why max-abs-diff alone is the wrong instrument.

        The ``.item()`` below synchronizes the compute stream, which is
        already ordered behind the H2D — so the read is correctly ordered and
        no explicit device sync is needed. It does serialize the step, which
        is the stated cost of this mode.
        """
        ref = record.ref
        record.ref = None  # release the HBM as early as possible
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
            # nan_to_num on the DIFFERENCE, so a NaN that appears on exactly
            # one side scores as infinite rather than silently comparing equal.
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

    # ── saved_tensors_hooks pack/unpack ────────────────────────────────────

    # Sentinel constant for descriptor tuples returned by pack_fn. Allows
    # unpack_fn to distinguish "I packed this" from "the user returned a
    # plain tensor" without subclassing torch.Tensor (which is fragile).
    _PACKED_SENTINEL: str = '_step4.5_packed_v1'
    # Separate sentinel for recompute descriptors.
    # The two paths share pack_fn / unpack_fn but their payloads differ
    # (evict carries host_buf + op; recompute carries (L, kind, idx)).
    _PACKED_RECOMPUTE_SENTINEL: str = '_step6a_recompute_v1'

    # ── Absence ordinals ───────────────────────────────────────────────────
    def _current_ordinal(self) -> Optional[int]:
        """Index of the layer interval that just ended, or None if unmeasured."""
        if self._interval_ordinal_source is None:
            return None
        ordinal, _phase = self._interval_ordinal_source()
        return ordinal if ordinal >= 0 else None

    # ── Denials must be loud ───────────────────────────────────────────────
    def _record_evict_denial(self, layer_idx: int, kind: str,
                             nbytes: int, reason: str) -> None:
        """Book a planned eviction the physical path could not perform.

        The planner has already counted this bundle's bytes as relief. If the
        save silently stays resident, the projection and the allocator disagree
        by exactly these bytes — which is a measurement error below the memory
        ceiling and an OOM above it. Under ``--strict_evict`` the disagreement
        is raised instead of logged, so a frontier run cannot quietly report a
        relief it never delivered.
        """
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
        """Book a bundle the planner needed relief from and could not get.

        Only reachable when the projected peak is over budget and this bundle
        contributes to it, so the projection keeps bytes that no action will
        remove. Below the ceiling that is a measurement error; above it the
        step OOMs, which is why ``--strict_evict`` raises here rather than
        letting a frontier run report a ceiling it never reached.
        """
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
        """Book a step still over capacity after every available pick.

        Above the memory ceiling the planner's question is feasible-versus-
        infeasible, and this is the answer in the negative: the breach survived
        releasing every bundle the step has, so what remains is mass migration
        cannot touch and no placement of this configuration fits. That is a
        statement about the SHAPE, not about the policy, which is why it is
        recorded separately from a bundle the criterion declined to release.
        """
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

    def pack_fn(self, tensor):  # noqa: ANN001 — autograd-typed callback
        """Autograd saved-tensor pack hook.

        Fires for every ``save_for_backward`` call inside the context of
        ``torch.autograd.graph.saved_tensors_hooks(pack, unpack)``. Gates:

        * Engine not live → return tensor unchanged.
        * Outside a managed submodule (current_bundle is None) → unchanged.
          This is the right behavior for e.g. lm_head saves, the embedding
          layer, anything outside LlamaAttention / LlamaMLP forwards.
        * Bundle not marked for eviction in the pre-decision → unchanged.
        * PinnedBufferPool can't accommodate `nbytes` → unchanged (we never
          block in pack_fn).

        Otherwise: acquire a host buffer, submit a D2H copy on the mem
        stream, return a descriptor tuple that unpack_fn can decode.
        Autograd will drop its GPU reference to the tensor once we return,
        and the HBM storage becomes reclaimable (subject to record_stream
        held by the mem stream).

        Why bundle-as-one rather than per-tensor: pack_fn doesn't know
        which registry record a given saved tensor maps to (there's no
        stable identity from the autograd Function side), and trying to
        match by shape is fragile. The bundle-as-one decision honors the
        physical-intuition grouping (Q/K/V/attn_scores move together,
        SwiGLU triple moves together) without needing per-tensor mapping.
        """
        # Cheap guards first — pack_fn is on a hot path.
        if not self._engine_live:
            return tensor
        bundle = self._current_bundle
        if bundle is None:
            return tensor
        # Zero-element saves are pass-through and must not be
        # observed or booked. checkpoint_wrapper emits one per layer (the
        # _NoopSaveInputs placeholder). Booking one costs a real pinned buffer
        # — the pool picks the smallest bucket whose size is not < nbytes, and
        # every bucket satisfies that for nbytes 0 — and an empty tensor's
        # data_ptr is 0, which would enter the per-bundle storage census as a
        # spurious distinct storage. Placed before the observation block so
        # the byte model never sees it either.
        if tensor.numel() == 0:
            self._packed_zero_elem_count += 1
            return tensor
        # Observe what autograd ACTUALLY saves for this bundle.
        # Placed before the evict/recompute branches so every bundle is
        # measured regardless of the action taken — the byte model has to be
        # right for bundles the planner refuses, too. Replay-safe: the
        # recompute path re-enters forward with _in_recompute set, and
        # enter_bundle short-circuits on that flag, so replayed saves see
        # _current_bundle is None and never reach here.
        obs_nbytes = tensor.numel() * tensor.element_size()
        self._obs_bundle_bytes_cur[bundle] = (
            self._obs_bundle_bytes_cur.get(bundle, 0) + obs_nbytes)
        self._obs_bundle_members_cur.setdefault(bundle, []).append(obs_nbytes)
        # And the relief figure alongside it. A view
        # pins its whole storage, so storage.nbytes() — not numel x itemsize —
        # is what dropping this save would actually release.
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
                # Size histogram of what the filter subtracted.
                # See _obs_bundle_foreign in __init__ for why. Bounded by the
                # distinct-storage count of one bundle, so the hot-path
                # cost is one dict increment per storage, not per save.
                _hist = self._obs_bundle_foreign_cur.setdefault(bundle, {})
                _k = (f"{_storage.nbytes()}|"
                      f"{self._bundle_foreign_src.get(_ptr, 'unknown')}")
                _hist[_k] = _hist.get(_k, 0) + 1
        # Recompute branch. If the bundle was marked
        # recompute by on_pre_forward AND the profiler captured a module
        # ref + inputs at enter_bundle, drop the tensor and return a
        # descriptor that unpack_fn will resolve by re-running the bundle's
        # forward. The module-ref check is the safety net for the smoke
        # test path where inputs may not have been wired yet — without it
        # we'd issue a recompute descriptor we can never honor at unpack.
        if (self._packed_recompute_bundles.get(bundle, False)
                and bundle in self._recompute_module_cache):
            idx = self._recompute_save_counters.get(bundle, 0)
            self._recompute_save_counters[bundle] = idx + 1
            # This is the drop with no stream dependency, so from
            # here on this bundle's owned-byte observation can be corrupted by
            # address reuse. Mark it; _promote_observed_relief refuses to book
            # it. See _obs_bundle_dropped_cur in __init__.
            self._obs_bundle_dropped_cur.add(bundle)
            # Store shape + dtype on the descriptor so unpack_fn can
            # sanity-check the replayed cache entry. A mismatch between
            # the descriptor's expected shape and cache[idx].shape means
            # the replay produced a different save sequence than the
            # original forward — fail loudly here, not downstream in a
            # cryptic BMM backward shape error.
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
        # The pool's has_capacity gate is the runtime fallback. The
        # decision-time gate already checks
        # _pool.has_capacity inside _decide_evict_vs_recompute, so this
        # second check should rarely fire — but pool state can drift
        # between decision and pack time as the same step submits multiple
        # evicts. Belt-and-suspenders.
        nbytes = tensor.numel() * tensor.element_size()
        layer_idx, kind = bundle
        # TRAP: a save that is not in HBM cannot be evicted out of HBM. The
        # memory-efficient SDPA kernel saves its philox seed/offset on the
        # CPU, and staging one raises on record_stream (CUDA-only). Skipping is
        # correct on both axes: the copy would free nothing, and booking its
        # bytes as realized relief would make the projection disagree with the
        # allocator. Pass the tensor through unchanged and count it apart from
        # the denials, which mean something entirely different.
        if (self._evict_device_type is not None
                and tensor.device.type != self._evict_device_type):
            self._evict_skipped_nonhbm_bytes += nbytes
            self._evict_skipped_nonhbm_count += 1
            self._evict_skipped_nonhbm_kinds.add(
                f'{tensor.device.type}/{tensor.dtype}/dim{tensor.dim()}')
            return tensor
        # Bisection filter. Deliberate exclusion, not a denial —
        # see the evict_filter setter for why it must not raise under
        # strict_evict and why a filtered arm's relief numbers are void.
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
            # has_capacity and acquire walk the same buckets, so this is a
            # can't-happen — but a silent retention here would be invisible,
            # which is what the denial accounting exists to prevent.
            self._record_evict_denial(layer_idx, kind, nbytes,
                                      f'acquire_failed: {exc}')
            return tensor
        # Unique tensor_id for this packed event. Not a registry key —
        # the autograd-saved tensor doesn't have a stable identity tied
        # to any single registry record under bundle-as-one. The id() is
        # only useful as a debug breadcrumb; uniqueness is guaranteed by
        # the (step, count) prefix.
        pack_id = (
            f"pack_step{self._current_step}_L{layer_idx}_{kind}_"
            f"n{len(self._packed_events)}"
        )
        if self._layout_census_enabled:
            self._record_layout(kind, tensor)
        # The reference copy must be taken BEFORE the D2H is submitted: the
        # point of the check is to compare what backward gets against what
        # forward produced, and a clone taken after the copy is enqueued would
        # still be correct here (same stream) but would read as ambiguous.
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
        # The descriptor is (SENTINEL, PackedEvict) so the backward
        # prefetcher can write prefetch_op / dst into it between forward and
        # the read. The sentinel string lets unpack_fn tell "I packed this"
        # from a plain tensor that passed through unchanged.
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

    def unpack_fn(self, packed):  # noqa: ANN001 — autograd-typed callback
        """Autograd saved-tensor unpack hook.

        Fires when backward needs a saved tensor that pack_fn earlier
        returned a descriptor for. Two cases:

        * Packed is a plain ``torch.Tensor`` → pack_fn returned unchanged;
          backward gets the original tensor (still HBM-resident).
        * Packed is our descriptor → return the prefetched HBM tensor,
          ordering the compute stream behind the H2D.

        The H2D is issued by ``_issue_physical_prefetch`` some layers
        earlier, and nothing here blocks the CPU. Issuing the H2D here and
        CPU-blocking on it would put every restored byte on the critical
        path by construction:

        * no D2H wait is needed — both copies are enqueued on the mem
          stream, so same-stream ordering already guarantees the H2D
          cannot start before the D2H finishes;
        * the H2D wait is a stream-stream ``wait_event``, which orders
          the compute stream behind the copy without stopping the CPU, so
          kernel enqueue keeps running ahead.

        The on-demand path below is the fallback for descriptors no
        lookahead window reached — the first layers of backward, and any
        bundle whose prefetch was skipped.
        """
        # Recompute descriptor handling. First call
        # for a bundle re-executes the bundle's submodule forward and
        # caches every saved tensor in order. Subsequent calls (same
        # bundle, different idx) pull from the cache without re-running.
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
            # Fallback: no lookahead window reached this bundle. Issue the
            # H2D now; no D2H wait is needed (same-stream ordering covers
            # it).
            self._issue_prefetch_for_record(record)
            self._prefetch_on_demand += 1
        else:
            self._prefetch_ahead_hits += 1
        # Order the compute stream behind the H2D instead of blocking the
        # CPU on it. The bytes are not needed until the backward kernels
        # that read `dst` run, and those are enqueued after this point on
        # the same compute stream.
        self._mem_stream.wait_compute_on(record.prefetch_op)
        dst = record.dst
        if record.ref is not None:
            self._verify_one(record, dst)
        # Drop the engine's reference so the tensor's lifetime is autograd's
        # alone; the record stays in _packed_by_bundle only for telemetry
        # until the step-start clear. The host buffer is released when the
        # prefetch op is reaped (_reap_inflight), which is the single owner.
        record.dst = None
        return dst

    # ── Physical prefetch-ahead ────────────────────────────────────────────

    def _issue_prefetch_for_record(self, record: PackedEvict) -> None:
        """Allocate the destination and start the H2D for one record.

        Guarded on ``prefetch_op`` rather than ``dst`` because unpack_fn
        clears ``dst`` when it hands the tensor to autograd; keying on the
        op makes a second issue impossible for the life of the record.
        """
        if record.prefetch_op is not None:
            return
        # torch.empty doesn't kernel-launch, so the storage is usable
        # immediately; submit_prefetch does the wait-for-compute-stream.
        dst = torch.empty(record.shape, dtype=record.dtype,
                          device=record.device)
        # Absence ends HERE, not when the copy
        # lands. torch.empty takes the HBM immediately, so the bytes are back
        # on the card from this interval onward.
        record.restore_ordinal = self._current_ordinal()
        record.dst = dst
        record.prefetch_op = self._mem_stream.submit_prefetch(
            tensor_id=record.pack_id + '_h2d', nbytes=record.nbytes,
            step=self._current_step, host_buf=record.host_buf,
            dst_tensor=dst,
        )

    def _issue_physical_prefetch(self, layer_idx: int) -> int:
        """Start H2D for bundles backward will read in the next k layers.

        Called from on_pre_backward(L). Backward descends, so the layers
        about to be read are L-1 .. L-k; issuing here buys each of them up
        to k layers of lead time over the read in unpack_fn.

        ``_prefetch_window_layers`` is the only bound on how much staged
        activation becomes HBM-resident early, so it doubles as the cap on
        prefetch-ahead's added peak.
        """
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

    # ── Physical recompute helper ──────────────────────────────────────────

    def _rerun_bundle_forward(self, bundle: tuple[int, str]) -> list:
        """Re-execute a bundle's submodule forward and capture saved tensors.

        Invoked lazily by ``unpack_fn`` the first time a recompute
        descriptor for ``bundle`` is requested. Calls ``module(*args,
        **kwargs)`` under a nested ``saved_tensors_hooks`` whose pack
        appends every save_for_backward target to ``captured``. Returns
        that list — its order matches the order ``pack_fn`` originally
        handed out indices, so callers can index by the descriptor's
        ``idx`` field.

        Re-entrancy: sets ``_in_recompute`` so the submodule's own
        forward-pre-hook (which calls ``enter_bundle``) short-circuits.
        Without that guard the re-run would overwrite the outer step's
        ``_current_bundle`` and shred input/module caches for later
        bundles in the same step.

        Timing: CUDA events when on GPU (no host-side fudge), fallback
        to ``time.perf_counter`` on CPU (smoke tests). The measured
        wall-clock flows into ``record_recompute_timing(L, ms)`` — the
        cost model reads the mean of these samples.
        """
        inputs, kwargs = self._recompute_input_cache[bundle]
        module = self._recompute_module_cache[bundle]
        layer_idx, bundle_kind = bundle
        captured: list = []

        def record_pack(t):
            # TRAP: return a detached view to BOTH our cache AND the inner
            # autograd's saved-tensor slot. Returning `t` (grad_fn intact) to
            # autograd keeps the inner Function chain alive through the
            # autograd save slots, which pins SDPA-internal saved tensors
            # past on_post_backward. We never run the inner backward
            # (the discarded `_` from module(...) is the only owner of the
            # inner graph), so dropping grad_fn from the saved slots is safe.
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
                # Discard the output — only the captured saved tensors
                # matter. The new autograd graph spun up by this forward
                # is garbage-collected when this scope exits because no
                # external reference survives.
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
        """Pre-decide layer L's bundle fates before any of L's submodules run.

        Per-layer planning at the layer's own creation moment. The planner
        consults the predictor's projected HBM occupancy from
        now through L's backward read; if the projected peak exceeds budget,
        each of L's bundles is examined for its byte contribution and routed
        through the exposed-cost criterion. Decisions land in
        ``_packed_*_bundles`` so pack_fn drops/dumps L's tensors at save
        time.

        Step-cursor semantics: advance ``_current_step`` by
        ``EVENTS_PER_FWD_LAYER`` so the cursor reflects "start of layer L's
        forward" at decision time.

        Forward the hook to the coordinator BEFORE the planner runs so
        ``params_resident(L)`` reflects the post-allgather state when the
        cost model queries it through ``_exposed_reallgather_ms``.
        """
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

    # ── Per-layer planner ──────────────────────────────────────────────────

    # Component → bundle classification (bundle-as-one). Keeping it as a
    # class constant makes it the
    # single source of truth that ``_bundle_tensor_ids`` and the
    # post-forward tier-flip both consult.
    _BUNDLE_COMPONENTS: dict[str, tuple[str, ...]] = {
        'attention': ('q', 'k', 'v', 'attn_scores'),
        'mlp': ('swiglu_gate', 'swiglu_up', 'swiglu_product'),
    }

    # The whole-layer unit owns everything the layer produces. The analytic
    # sum over these is wrong for the same reason it is wrong per-module (the
    # seam sees one boundary, not seven components), and pack_fn's observation
    # shadows it from step 1 — but the named ids are still what the projection
    # excludes and what _decide_evict_vs_recompute keys its window lookup on,
    # so the bundle must own them rather than the residual slot alone.
    _WHOLE_LAYER_COMPONENTS: tuple[str, ...] = (
        'q', 'k', 'v', 'attn_scores',
        'swiglu_gate', 'swiglu_up', 'swiglu_product',
    )

    def _kind_for_tensor_id(self, key: str) -> Optional[str]:
        """Bundle class owning activation ``key`` (``act_layer_<L>_<comp>``),
        or None if the id is not a bundle member."""
        for kind, comps in self._BUNDLE_COMPONENTS.items():
            for c in comps:
                if key.endswith(f'_{c}'):
                    return kind
        return None

    def _bundle_tensor_ids(self, layer_idx: int, kind: str) -> set[str]:
        """Predictor activation IDs for one bundle of layer ``layer_idx``.

        These IDs are the keys ``AccessPredictor._activation_id`` produces
        and that appear in the schedule's events. The planner uses them as
        ``exclude_tensors`` for ``contribution_window_for`` and as
        representative tensor IDs for ``_decide_evict_vs_recompute``.
        """
        comps = self._BUNDLE_COMPONENTS.get(kind, ())
        tids = {f'act_layer_{layer_idx}_{c}' for c in comps}
        # The residual belongs to the bundle for
        # every purpose these ids serve — it must leave the projection when
        # the bundle is excluded, or the planner books relief it won't get.
        if tids:
            tids.add(self._predictor.residual_id(layer_idx, kind))
        return tids

    # The analytic model undercounts what autograd actually saves, so from
    # step 1 on, pack_fn's observation shadows it via the ``key`` argument.
    #
    # TRAP: both helpers below serve the COST side, so both must DROP the
    # residual. The residual is a RELIEF-side quantity — storages the bundle
    # pins, not a tensor anyone copies — so summing it into transfer_ms prices
    # a transfer that never happens, and handing it to _bundle_member_sizes
    # asks the pool for a pinned buffer no save will ever fill. It is filtered
    # explicitly rather than left to the coincidence that the residual is 0 at
    # step 0 and shadowed thereafter.
    def _bundle_byte_estimate(self, tensor_ids: set[str], key=None) -> int:
        """Sum bytes the predictor attributes to a bundle's tensors.

        Returns 0 if the predictor's byte model was not populated (synthetic
        unit tests, smoke tests). The planner falls back to retain in that
        case because no peak projection is meaningful without bytes.

        When ``key`` is a ``(layer_idx, kind)`` pair that pack_fn has already
        observed in a completed step, the observation wins.
        """
        if key is not None:
            obs = self._obs_bundle_bytes.get(key)
            if obs:
                return obs
        if self._bundle_unit == 'whole_layer':
            # The named components are the layer's interior, which the wrapper
            # creates and frees inside its own forward. The seam never sees
            # them and no decision can move them; the boundary is the bundle.
            return self._predictor.boundary_bytes()
        bm = self._schedule.bytes_by_tensor
        return sum(bm.get(tid, 0) for tid in self._named_tids(tensor_ids))

    # Gate 0 needs the bundle's *per-tensor* sizes, not their sum. pack_fn
    # acquires one pinned buffer per tensor, so the feasible question is
    # "N buffers of these sizes?" — asking for one buffer of the sum invents
    # demand for a buffer that no bucket serves and no tensor needs.
    def _bundle_member_sizes(self, tensor_ids: set[str], key=None) -> list:
        """Per-tensor byte sizes for a bundle, zero-size members dropped.

        ``key`` selects pack_fn's observed member list when available — the
        analytic model both undercounts the total and gets the *arity* wrong
        (4 modelled attention members against more actually saved), and
        Gate 0 asks the pool for one buffer per member.
        """
        if key is not None:
            obs = self._obs_bundle_members.get(key)
            if obs:
                return list(obs)
        if self._bundle_unit == 'whole_layer':
            # Same reason as _bundle_byte_estimate above, but this one gates
            # Gate 0: the interior never crosses the seam, so asking the pool
            # for a buffer per named component invents a demand no save will
            # ever fill. Only bites at step 0, before the observed member list
            # exists.
            return [self._predictor.boundary_bytes()]
        bm = self._schedule.bytes_by_tensor
        return [n for n in (bm.get(tid, 0)
                            for tid in self._named_tids(tensor_ids)) if n > 0]

    def _named_tids(self, tensor_ids) -> set[str]:
        """``tensor_ids`` with the residual slots removed."""
        return {t for t in tensor_ids if not t.endswith('_residual')}

    def _bundle_flop_estimate(self, bundle_bytes: int) -> int:
        """Recompute FLOPs for a bundle.

        At ``on_pre_forward(L)`` L's registry records don't exist yet, so
        this uses a fallback heuristic: 2 FLOPs per output byte.
        """
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
        """Build a synthetic retain decision for one bundle.

        Synthetic ``tensor_id = bundle_L{L}_{kind}`` because L's activation
        records don't exist yet at decision time. JSON consumers can still
        group by ``layer_idx`` and ``tensor_id`` to see what the planner
        chose per bundle.
        """
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
        """Schedule ordinal at which layer ``layer_idx``'s migrated bytes
        return to HBM, or None if the schedule has no backward for it.

        ``_issue_physical_prefetch(K)`` runs at ``on_pre_backward(K)`` and
        stages layers ``K-1 .. K-W``, and backward descends, so layer T is
        restored by the HIGHEST backward layer that reaches it — ``min(T+W,
        num_layers-1)``. The last layer is never prefetched (nothing above
        it runs first); it restores lazily inside ``unpack_fn`` at its own
        backward read, which is what the ``min`` degenerates to.

        The ordinal is that layer's first backward read. The physical issue
        point is ``on_pre_backward``, a hair earlier, so this is optimistic
        by less than one layer of backward events — and the destination
        buffer is allocated at issue, not at completion, so the bytes are
        back at the issue instant, not the arrival one.

        Contribution is measured before the criterion runs, so every
        candidate is priced as a migration. A recompute would buy lifetime
        absence instead, so under the per-module unit this understates a
        recompute's relief — the direction that under-picks, not the one
        that books relief the step cannot deliver. Under the whole-layer
        unit the question does not arise: recompute is refused there.
        """
        restore_layer = min(layer_idx + self._prefetch_window_layers,
                            self._num_layers - 1)
        if restore_layer < layer_idx:
            restore_layer = layer_idx
        return self._schedule.bwd_event_for_layer(restore_layer)

    def _plan_for_layer_forward(
        self, layer_idx: int,
    ) -> list[PlacementDecision]:
        """Per-layer planner. Picks each of L's bundle fates.

        Algorithm:

          1. ``current_live = budget.physical_allocated()`` (runtime anchor).
          2. ``L_bwd_event = schedule.bwd_event_for_layer(L)``.
          3. ``peak, peak_step = schedule.peak_in_range(now, L_bwd, anchor)``.
          4. If ``peak <= budget``: all bundles retain.
          5. Else for each bundle, compute ``contribution_window_for`` and:
             - If the bundle doesn't contribute to the breach, retain.
             - Else run ``_decide_evict_vs_recompute`` and write the result
               into ``_packed_evict_bundles`` / ``_packed_recompute_bundles``.

        Each branch produces exactly one synthetic PlacementDecision per
        bundle (attention, mlp); pack_fn telemetry is the ground truth for
        what physically happened.
        """
        new_decisions: list[PlacementDecision] = []

        # Force-action short-circuit. When set, the planner bypasses pressure
        # projection, contribution analysis, the per-step pick cap, and the
        # exposed-cost criterion. Every reducible bundle is marked for the
        # forced action every step. Cost-model reason is still filled by
        # _decide_evict_vs_recompute, for telemetry parity.
        if self._force_action != 'none':
            return self._plan_forced(layer_idx)

        current_live = self._budget.physical_allocated()
        # The static byte budget; see effective_budget.
        budget = self.effective_budget()
        L_bwd_event = self._schedule.bwd_event_for_layer(layer_idx)

        # Final layer in a synthetic schedule may have no bwd read event
        # (e.g., zero-byte test predictor). Default to retain; nothing to
        # project, no pressure expected.
        if L_bwd_event is None:
            for kind in self._BUNDLE_COMPONENTS:
                new_decisions.append(self._bundle_retain_decision(
                    layer_idx, kind, note='no_bwd_event'))
                if self._engine_live:
                    self._packed_evict_bundles[(layer_idx, kind)] = False
                    self._packed_recompute_bundles[(layer_idx, kind)] = False
            return new_decisions

        # Projection lower bound: start-of-L's-forward, NOT _current_step.
        # _current_step was advanced by EVENTS_PER_FWD_LAYER at the top of
        # ``on_pre_forward`` so the cursor reflects end-of-L's-forward; using
        # it as the lo would walk PAST L's own creates and ``contribution_
        # window_for`` would return 0 for L's bundles. ``fwd_create_event_
        # for_layer`` is the right anchor — first ``phase=='fwd'``,
        # ``access_type=='create'``, ``layer_idx==L`` event.
        fwd_lo = self._schedule.fwd_create_event_for_layer(layer_idx)
        if fwd_lo is None:
            fwd_lo = self._current_step

        # Cumulative-coverage projection: walk with all tensor IDs that
        # earlier layers in this step have already chosen to evict /
        # recompute. Without this exclusion every layer sees the same
        # full-fwd peak and picks bundles independently, which over-picks.
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

        # Pressure: walk each bundle in turn and pick only as many as
        # needed to bring the running projected peak under budget. The
        # running ``excluded`` set is local to this layer's pass; if any
        # picks land, copy them into ``self._planner_excluded_tids`` so
        # downstream layers see the smaller peak.
        excluded = set(self._planner_excluded_tids)
        restores = dict(self._planner_restore_ordinals)
        layer_restore = self._restore_ordinal_for_layer(layer_idx)
        running_peak = peak_bytes
        running_peak_step = peak_step
        for kind in self._BUNDLE_COMPONENTS:
            tids = self._bundle_tensor_ids(layer_idx, kind)
            bundle_bytes = self._bundle_byte_estimate(tids, key=(layer_idx, kind))
            if bundle_bytes == 0:
                # No byte model (synthetic schedule) — can't reason about
                # contribution. Default to retain.
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
                # Earlier bundles in this same layer already pushed the
                # running peak under budget; no further picks needed.
                new_decisions.append(self._bundle_retain_decision(
                    layer_idx, kind, note='budget_satisfied',
                    projected_peak_bytes=running_peak,
                    projected_peak_event_idx=running_peak_step,
                    anchor_live_bytes=current_live))
                if self._engine_live:
                    self._packed_evict_bundles[(layer_idx, kind)] = False
                    self._packed_recompute_bundles[(layer_idx, kind)] = False
                continue

            # If we've already hit the per-step pick cap, stop.
            if self._planner_picks_this_step >= self._planner_max_picks_per_step:
                new_decisions.append(self._bundle_retain_decision(
                    layer_idx, kind, note='pick_cap_reached',
                    projected_peak_bytes=running_peak,
                    projected_peak_event_idx=running_peak_step,
                    anchor_live_bytes=current_live))
                # The cap is the bundle count, so reaching it
                # under pressure means every bundle in the step was already
                # picked and the breach survived them all — an infeasibility,
                # not a throttle. Book it like any other unrelieved bundle so
                # --strict_evict stops the run instead of letting it report a
                # ceiling it never reached.
                self._note_no_relief_under_pressure(
                    layer_idx, kind, bundle_bytes, 'pick_cap_reached')
                if self._engine_live:
                    self._packed_evict_bundles[(layer_idx, kind)] = False
                    self._packed_recompute_bundles[(layer_idx, kind)] = False
                continue

            # What if we add this bundle to the exclusion set? The bundle is
            # absent from its own creation until backward restores it, so
            # the candidate walk carries that ordinal rather than deleting
            # the bytes outright.
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
                # Either the bundle's bytes never enter the window, or its
                # absence interval closes at or before the peak instant, in
                # which case the copy buys nothing and the honest answer is
                # to keep the bytes. The second case is a prefetch-depth
                # signal, so it gets its own note rather than being folded in.
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

            # Bundle contributes. Route through the exposed-cost criterion
            # to pick evict vs recompute, then commit.
            flops = self._bundle_flop_estimate(bundle_bytes)
            # The residual is a byte-model slot, not a real activation —
            # the window lookup must key off a named component.
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
                # Nothing was freed, so the bundle's bytes must stay in the
                # projection. Excluding them here would hand every later
                # layer a peak that no action produced.
                self._note_no_relief_under_pressure(
                    layer_idx, kind, bundle_bytes, reason.note)
                continue

            # Commit this pick to the running state.
            excluded = candidate_excluded
            restores = candidate_restores
            running_peak = cand_peak
            running_peak_step = cand_peak_step
            self._planner_picks_this_step += 1

        # Persist the picks for downstream layers' projections in the same step.
        self._planner_excluded_tids = excluded
        self._planner_restore_ordinals = restores

        # Feasibility gate: only the last layer has been offered every bundle
        # in the step, so only its window can distinguish "the planner has
        # not got there yet" from "there is nothing left to pick". Limitation:
        # at the frontier this window does not end on the peak instant — once
        # the pick cap binds, the peak relocates to post_bwd_layer_W, outside
        # it. Widening the window does not help: the schedule's own maximum
        # over the whole remaining step is this same ordinal, because the
        # byte model carries no gradient shards and the relocation is
        # gradient shards piling up.
        if layer_idx == self._num_layers - 1 and running_peak > budget:
            self._note_infeasible(running_peak, budget, running_peak_step,
                                  current_live)

        return new_decisions

    # ── Forced-baseline planner short-circuit ─────────────────────────────

    def _plan_forced(self, layer_idx: int) -> list[PlacementDecision]:
        """Force every reducible bundle to the configured force_action.

        Used for the pure-evict and pure-recompute baselines. Bypasses the
        pressure projection, contribution analysis, per-step pick cap, and
        the exposed-cost criterion. Cost-model reason is still filled via
        _decide_evict_vs_recompute, so the forced decisions carry the same
        telemetry as criterion runs.

        Bundles with zero bytes (synthetic test schedules) retain.
        """
        action_out = ('evict_to_cpu' if self._force_action == 'evict'
                      else 'recompute')
        # Fixed migration stride on a recompute base. Every
        # k-th layer migrates, the rest recompute. Criterion-free by
        # construction, so it measures the PHYSICS (does moving fewer bytes
        # make each byte cheaper?) rather than the policy. k=1 degenerates to
        # pure evict and k>num_layers to pure recompute, so the sweep is
        # anchored at both ends by the forced baselines.
        if self._force_stride > 0 and self._force_action == 'recompute':
            action_out = ('evict_to_cpu'
                          if layer_idx % self._force_stride == 0
                          else 'recompute')
        new_decisions: list[PlacementDecision] = []
        for kind in self._BUNDLE_COMPONENTS:
            if action_out == 'recompute' and kind == 'layer':
                # Whole-layer unit: there is no replay that reproduces this
                # bundle's save (see _decide_evict_vs_recompute). Forcing it
                # would book relief the step cannot deliver, so retain and
                # say so rather than report a number the run did not earn.
                # Deliberately NOT the criterion's evict fallback: this path
                # is only reachable under --force_stride, where retaining is
                # what leaves the wrapper's own recompute in place and keeps
                # the stride a density sweep instead of pure evict.
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
                # No byte model → forcing is a no-op. Retain to keep
                # behavior parity with the unforced path's no_byte_model
                # branch.
                new_decisions.append(self._bundle_retain_decision(
                    layer_idx, kind, note='no_byte_model_force_skip'))
                if self._engine_live:
                    self._packed_evict_bundles[(layer_idx, kind)] = False
                    self._packed_recompute_bundles[(layer_idx, kind)] = False
                continue

            # Still run the cost model to fill telemetry. The action it
            # returns is ignored — we use the forced one. This keeps
            # exposed_evict_ms / exposed_recompute_ms in the JSON.
            flops = self._bundle_flop_estimate(bundle_bytes)
            # The residual is a byte-model slot, not a real activation —
            # the window lookup must key off a named component.
            rep_tid = next(iter(self._named_tids(tids)))
            _decided_action, reason = self._decide_evict_vs_recompute(
                size_bytes=bundle_bytes,
                flops=flops,
                layer_idx=layer_idx,
                representative_tensor_id=rep_tid,
                decision_step=self._current_step,
                kind=kind,
                member_sizes=self._bundle_member_sizes(tids, key=(layer_idx, kind)),
                # Telemetry-only call: the action below is forced, not taken
                # from the criterion, so it must not charge the window ledger.
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
        """Post-forward hook — tier reconciliation + prefetch lookahead.

        Decisions for L are made in ``on_pre_forward(L)`` so pack_fn has
        them in hand before any of L's submodules save. This hook runs
        after L's activations have been registered by the profiler and:

          1. Reconciles registry tiers for L's activations against the
             bundle-level planner decisions stashed in
             ``_packed_*_bundles``. The planner doesn't touch the registry
             (the records don't exist yet at decision time), so the tier
             flip happens here for telemetry consistency: evicted bundles
             flip to ``cpu_dram``, recompute bundles flip to
             ``recompute_marker``.
          2. Prefetch lookahead.
          3. Forwards to the coordinator.
        """
        self._reap_inflight()
        new_decisions: list[PlacementDecision] = []

        # ── Tier reconciliation ────────────────────────────────────────
        # Walk L's now-registered activations and flip tier based on the
        # planner's bundle-level decision. Engine-live mode only — dry-run
        # and synthetic tests don't have pack_fn moving bytes, so leaving
        # tier='hbm' matches reality there.
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

        # Prefetch lookahead. Forward-phase window — small (lookahead in
        # forward usually finds nothing in cpu_dram).
        window = self._prefetch_window_layers * self.EVENTS_PER_FWD_LAYER
        new_decisions.extend(self._prefetch_lookahead(window))

        # Forward the post-forward hook to the coordinator so its state
        # machine knows L's params have been re-sharded; the coordinator
        # tracks whatever FSDP itself does.
        self._coord.on_post_forward(layer_idx)

        self._decisions.extend(new_decisions)
        return new_decisions

    def on_pre_backward(self, layer_idx: int) -> list[PlacementDecision]:
        """Verify L's activations are HBM-resident before backward starts.

        Possible per-activation states:
          tier == 'hbm', not in_flight        retain
          tier == 'cpu_dram', in_flight       wait on event (stall if wait > 0)
          tier == 'cpu_dram', not in_flight   policy miss (stall)
          tier == 'recompute_marker'          recompute decision (logical
                                              re-classification; the actual
                                              re-execution happens inside
                                              unpack_fn at backward read)
          tier == 'freed'                     no_op + warning

        Forward the hook to the coordinator FIRST so its state machine
        knows L's params are
        gathered for the backward (FSDP's BACKWARD_PRE prefetch
        completes before this fires). _exposed_reallgather_ms queries
        params_resident below the recompute branch in
        _handle_pre_backward_tensor.
        """
        self._coord.on_pre_backward(layer_idx)
        self._reap_inflight()
        # Start the H2D for the layers backward reads
        # next, before their unpack_fn asks. Placed after _reap_inflight so
        # buffers freed by just-completed prefetches are available again.
        self._issue_physical_prefetch(layer_idx)
        new_decisions: list[PlacementDecision] = []

        for key, rec in self._activations_for_layer(layer_idx):
            new_decisions.append(self._handle_pre_backward_tensor(key, rec, layer_idx))

        self._decisions.extend(new_decisions)
        return new_decisions

    def on_post_backward(self, layer_idx: int) -> list[PlacementDecision]:
        """Free L's activations and prefetch L-1's from cpu_dram (lookahead).

        Forward the hook to the coordinator AFTER the engine's own
        bookkeeping; the coordinator transitions L to 'sharded'.

        Drop the per-bundle recompute caches for L.
        ``_recompute_tensor_cache[bundle]`` holds every tensor the bundle's
        submodule forward saved during the replay unpack_fn triggered. Without
        this clear, all recomputed layers' replay caches stack in HBM through
        the remainder of backward and OOM it mid-backward. The replay tensors
        are no longer needed once L's backward kernels have consumed them,
        which is exactly here. Same logic for the input / module / save-counter
        caches.
        """
        self._current_step += self.EVENTS_PER_BWD_LAYER
        self._reap_inflight()
        new_decisions: list[PlacementDecision] = []

        for key, rec in self._activations_for_layer(layer_idx):
            # Do NOT unregister — keep records so the JSON output
            # preserves a post-run tier='freed' snapshot.
            if rec.tier != 'freed':
                self._registry.set_tier(key, 'freed')
            new_decisions.append(PlacementDecision(
                step_idx=self._current_step, layer_idx=layer_idx,
                tensor_id=key, action='no_op',
                estimated_cost_ms=0.0,
                reason=PlacementDecisionReason(note='free_after_bwd')))

        # Drop L's per-bundle recompute caches so the
        # replay tensors get GC'd before the next layer's backward stacks
        # its own.
        for kind in self._BUNDLE_COMPONENTS:
            bundle = (layer_idx, kind)
            self._recompute_tensor_cache.pop(bundle, None)
            self._recompute_input_cache.pop(bundle, None)
            self._recompute_module_cache.pop(bundle, None)
            self._recompute_save_counters.pop(bundle, None)

        # Lookahead for L-1's bwd reads — backward-phase window is larger.
        window = self._prefetch_window_layers * self.EVENTS_PER_BWD_LAYER
        new_decisions.extend(self._prefetch_lookahead(window))

        self._coord.on_post_backward(layer_idx)

        self._decisions.extend(new_decisions)
        return new_decisions

    # ── Accessors / telemetry ───────────────────────────────────────────────

    @property
    def decisions(self) -> list[PlacementDecision]:
        """Read-only copy of the full decision log."""
        return list(self._decisions)

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def total_recomputed_flops(self) -> int:
        """Running FLOPs charged to recompute decisions.

        Feeds the recompute-volume metric in the JSON output."""
        return self._total_recomputed_flops

    def effective_budget(self) -> int:
        """The byte budget the planner runs against: the static total."""
        return self._budget.total_bytes

    @property
    def hbm_drift_enabled(self) -> bool:
        """Always False; kept for compatibility (the budget is static)."""
        return False

    @property
    def hbm_drift_log(self) -> list[dict]:
        """Per-step drift entries — feeds the placement_summary JSON block."""
        return list(self._hbm_drift_log)

    def summary(self) -> dict:
        """Aggregate counts across all decisions, all training steps."""
        counts: dict[str, int] = defaultdict(int)
        for d in self._decisions:
            counts[d.action] += 1
        # Tally exposed stall (PCIe + allocator) across every 'stall'
        # decision.
        exposed_stall_pcie_ms = 0.0
        exposed_stall_alloc_ms = 0.0
        for d in self._decisions:
            if d.action == 'stall' and d.stall_breakdown:
                exposed_stall_pcie_ms += d.stall_breakdown.get('pcie_ms', 0.0)
                exposed_stall_alloc_ms += d.stall_breakdown.get('allocator_ms', 0.0)
        # Observed per-op timing rolled up across the run.
        # ``observed_bw_*_bytes_per_s`` is derived from totals (not per-op
        # averages) so it matches how the calibrator reports its number. These
        # are compared against calibration to flag drift. A zero op-count means
        # engine_live=False or no bundles were marked evict — emit zeros, not
        # divide-by-zero.
        def _bw(total_bytes: int, total_ms: float) -> float:
            return (total_bytes / (total_ms / 1000.0)) if total_ms > 0 else 0.0
        # Drift stats. mean + p95 of
        # observed/predicted ratios per direction × per predicted basis.
        # ``calibration_drift_flag`` fires when the contention-aware p95
        # exceeds 2× in either direction — i.e., the engine's cost math
        # was at least half off on the slow tail. Idle ratio is logged
        # for cluster sanity but does not feed the drift flag (every op
        # under NCCL looks slower than idle, which is expected).
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
        # Cap per-op log at write time. Stats above were computed over
        # the unbounded ratio lists, so truncation here doesn't bias them.
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
            # The policy's own cost. Read this against step_ms
            # before crediting or blaming the criterion for step time — it is
            # CPU work on the critical path, so it is a real cost, but it is
            # the only cost a force-action arm does NOT pay.
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
            # Per-step observed-vs-projected peak log. Each entry carries one
            # step's observed peak (torch.cuda.max_memory_allocated) and the
            # planner's projection for that step. `observed_peak_mb` is the
            # canonical peak for every run-table row. The ratio/correction/
            # enabled fields are fixed values kept for schema compatibility.
            'hbm_drift': {
                'enabled': False,
                'per_step_ratio_clamp': [1.0, 1.0],
                'cumulative_correction_final': 1.0,
                'effective_budget_bytes_final': self.effective_budget(),
                'log': list(self._hbm_drift_log),
            },
            # Overlap window as a consumed budget. ``refusals`` is the
            # counterfactual count — recompute decisions an undebited window
            # would have authorized as evictions. Zero means the ledger never
            # bound and the run is not testing it.
            'window_debit': {
                'enabled': self._window_debit_enabled,
                'committed_bytes': self._window_debit_committed_bytes,
                'refusals': self._window_debit_refusals,
            },
            # pack_fn's observed bundle byte model vs the analytic one.
            # Post-processing divides these to get the analytic model's
            # undercount.
            'observed_bundle_bytes': {
                f'L{L}_{kind}': {
                    'bytes': nbytes,
                    'members': len(self._obs_bundle_members.get((L, kind), ())),
                    # 'bytes' is the COST figure —
                    # every save, aliases included, because every save costs a
                    # pinned buffer and a copy. 'owned_bytes' is the RELIEF
                    # figure — distinct storages, params and the inbound
                    # residual stream removed. 'residual_bytes' is what the
                    # latter adds on top of the analytic model. The expected
                    # ordering is bytes >= owned >= analytic; an inversion
                    # means the foreign-storage filter over-subtracted.
                    'owned_bytes': self._obs_bundle_owned.get((L, kind), 0),
                    # False means this bundle dropped saves for
                    # recompute, so 'owned_bytes' was NOT measured and is not
                    # a small number — it is no number. Read it as absent.
                    # Only 'analytic_bytes' is authoritative for that bundle.
                    'owned_bytes_trusted':
                        (L, kind) not in self._obs_declined_bundles,
                    'residual_bytes': self._schedule.bytes_by_tensor.get(
                        self._predictor.residual_id(L, kind), 0),
                    'analytic_bytes': self._bundle_byte_estimate(
                        self._bundle_tensor_ids(L, kind)),
                    # "<nbytes>|<source>" → count for the storages
                    # the foreign filter subtracted. Expect FSDP's flat param
                    # and the inbound residual stream and nothing else; a
                    # bucket at B*S*intermediate*dtype means a relief storage
                    # is being thrown away, and its source names the branch of
                    # the rule that did it. See _obs_bundle_foreign in __init__.
                    'foreign_storages': dict(sorted(
                        self._obs_bundle_foreign.get((L, kind), {}).items())),
                    # Distinct storages sighted. Short against the
                    # save list means the seen-ptr dedup swallowed one, i.e.
                    # an address was recycled after a drop — a different bug
                    # from the filter subtracting it. See __init__.
                    'distinct_storages': self._obs_bundle_distinct.get(
                        (L, kind), 0),
                }
                for (L, kind), nbytes in sorted(self._obs_bundle_bytes.items())
            },
            # Bundles the analytic model over-booked. Non-empty
            # means the byte model does not match the attention kernel this run
            # dispatched to, and every projection is fiction — the residual
            # correction is clamped at zero and cannot walk an over-estimate
            # back. Check this before reading any peak projection.
            'overbooked_bundles': {
                f'L{L}_{kind}': info
                for (L, kind), info in sorted(self._overbooked_bundles.items())
            },
            'observed_relief': self._observed_relief,
            # Emitted so CSV backfill can read the
            # forced-baseline configuration without parsing argv from the log.
            'force_action': self._force_action,
            'force_stride': self._force_stride,
            'evict_filter': self._evict_filter,
            'layout_census': self._layout_census_enabled,
            'verify_restore': self._verify_restore,
            # Measured per-layer recompute wall-clock. Key and shape are fixed
            # for the convergence plot; 'ratio'/'prefill_n'/'seed_samples' are
            # fixed values kept for schema compatibility.
            # 'per_layer' flattens across kinds; 'per_layer_kind' carries the
            # (layer, attention|mlp) split the cost model actually divides by.
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
            # Per-class recompute cost pool, in place of the 2 FLOPs/byte
            # fallback, which prices an attention replay far too cheap.
            # 'probes' counts the bundles forced to recompute to buy a class
            # its first sample; 'mean_ms' is what the criterion divides
            # against.
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
            # Full-run demand histogram
            # and pool pressure. Together these say whether the pinned pool
            # is provisioned against the sizes the run actually asks for.
            'nbytes_histogram': {
                d: {str(k): v for k, v in sorted(h.items())}
                for d, h in sorted(self._nbytes_hist.items())
            },
            'pool': self._pool.telemetry(),
            # What the planner asked the physical path to evict vs what it
            # actually staged. Any claim about the criterion must cite
            # 'realized_fraction' directly. 'partial_bundles' is the worst
            # outcome — the copies
            # were paid but some of the bundle's storages are still resident,
            # so the bundle delivers no relief at all.
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
                # Bundles the planner needed relief from and the
                # criterion could not supply. Distinct from a denial — nothing
                # was planned, so nothing was lost at pack time — but the
                # consequence is the same shortfall in the projection.
                'no_relief_ops': self._no_relief_count,
                'no_relief_bytes': self._no_relief_bytes,
                'no_relief_notes': sorted(self._no_relief_notes),
                # Feasibility gate: the gate's verdict in the
                # negative. Non-empty means the breach survived every pick the
                # step had, so the SHAPE is past the frontier — a different
                # finding from the criterion declining to release bundles.
                'infeasible_steps': list(self._infeasible_steps),
                'denials': list(self._evict_denial_log),
                'denials_truncated':
                    self._evict_denied_count > len(self._evict_denial_log),
                # Saves that never occupied HBM, so
                # skipping them costs no relief. Kept out of denied_bytes on
                # purpose — mixing them in would make realized_fraction look
                # like a starved pool when nothing was lost.
                'skipped_nonhbm_ops': self._evict_skipped_nonhbm_count,
                'skipped_nonhbm_bytes': self._evict_skipped_nonhbm_bytes,
                'skipped_nonhbm_kinds':
                    sorted(self._evict_skipped_nonhbm_kinds),
                # Zero-element saves passed through before any
                # booking. Under the whole-layer unit, expect exactly
                # num_layers x steps — one checkpoint placeholder per wrapper.
                'zero_elem_passthrough': self._packed_zero_elem_count,
                # Saves the bisection filter excluded on purpose.
                # Kept out of BOTH denied_bytes and skipped_nonhbm_bytes: a
                # denial is a pool failure, a nonhbm skip is a free pass, and
                # this is the user restricting the arm. When this is
                # nonzero the arm is a diagnostic and its realized_fraction
                # and absence numbers must not be read as a measurement.
                'filter': self._evict_filter,
                'filtered_ops': self._evict_filtered_count,
                'filtered_bytes': self._evict_filtered_bytes,
                'filtered_classes': dict(
                    sorted(self._evict_filtered_classes.items())),
            },
            # Restore-path diagnostics. Empty unless the matching flag was
            # passed. 'layout_census' is the cheap one: any nonzero
            # 'noncontiguous', 'offset_nonzero' or 'storage_larger' row names a
            # save the restore path cannot reproduce faithfully, because it
            # rebuilds with torch.empty(shape) — dense, offset 0.
            # 'restore_verification' is the expensive one and separates wrong
            # BYTES from a wrong LAYOUT carrying right bytes.
            'evict_layout_census': {
                k: dict(v) for k, v in sorted(self._layout_census.items())},
            'restore_verification': {
                'enabled': self._verify_restore,
                'checked': self._verify_checked,
                'failed': self._verify_failed,
                'by_class': {
                    k: dict(v) for k, v in sorted(self._verify_log.items())},
            },
            # Realized absence, in
            # layer-interval ordinals. 'peak_coverage_fraction' is the number
            # a frontier claim rests on — the share of staged bytes that were
            # provably gone during the interval that held the peak. Bytes
            # outside that span were paid for and delivered nothing.
            # 'empty_windows' counts restores that landed at or before the
            # D2H completion's own interval: the prefetch window un-did the
            # relief. Null everywhere when no profiler supplied ordinals.
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
            # Did prefetch-ahead actually reach the
            # reads? 'on_demand' counts restores that still had to issue
            # their H2D inside unpack_fn.
            'prefetch_ahead': {
                'issued': self._prefetch_ahead_issued,
                'hits': self._prefetch_ahead_hits,
                'on_demand': self._prefetch_on_demand,
                'window_layers': self._prefetch_window_layers,
            },
        }

    def print_summary(self) -> None:
        """End-of-training human-checkpoint log line."""
        s = self.summary()
        print(f"[PlacementEngine] training end: "
              f"total_decisions={s['total_decisions']}, "
              f"by_action={s['by_action']}, "
              f"steps={s['training_steps']}, "
              f"recomputed_tflops={s['total_recomputed_flops'] / 1e12:.3f}")
        # Observed PCIe bandwidth line. Surfaces the measured D2H / H2D
        # throughput so the live ops can be checked against the calibrated
        # number (the JSON carries the automated comparison).
        ev = s['observed_pcie_evict']
        pf = s['observed_pcie_prefetch']
        if ev['ops'] > 0 or pf['ops'] > 0:
            print(f"[PlacementEngine] observed PCIe: "
                  f"evict ops={ev['ops']} bw={ev['observed_bw_bytes_per_s']/1e9:.2f} GB/s, "
                  f"prefetch ops={pf['ops']} bw={pf['observed_bw_bytes_per_s']/1e9:.2f} GB/s")
        # Planned vs realized eviction mass.
        # Printed whenever anything was planned, including the all-realized
        # case — "100%" is the statement a frontier run needs on the record.
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
        # Did the relief exist when it mattered?
        ab = s['absence']
        if ab['peak_coverage_fraction'] is not None:
            print(f"[PlacementEngine] absence at peak: "
                  f"{ab['peak_coverage_fraction'] * 100:.1f}% of staged bytes "
                  f"gone during the peak interval "
                  f"({ab['bytes_covering_peak'] / (1024**3):.3f} of "
                  f"{ab['staged_bytes'] / (1024**3):.3f} GiB), "
                  f"empty windows={ab['empty_windows']}, "
                  f"W={ab['prefetch_window_layers']}")
        # Drift summary. Always print
        # the contention p95s (the cost-model accuracy signal) when any
        # samples exist; the idle p95 is JSON-only — it's a physical
        # link check, not a research-relevant number.
        ld = s['latency_drift']
        ce = ld['obs_over_contention_evict']
        cp = ld['obs_over_contention_prefetch']
        if ce['n'] > 0 or cp['n'] > 0:
            flag = ' DRIFT' if ld['calibration_drift_flag'] else ''
            print(f"[PlacementEngine] cost-model accuracy: "
                  f"evict obs/contention p95={ce['p95']:.2f} (n={ce['n']}), "
                  f"prefetch obs/contention p95={cp['p95']:.2f} (n={cp['n']})"
                  f"{flag}")

    # ── Internal helpers ────────────────────────────────────────────────────

    def _activations_for_layer(self, layer_idx: int):
        """Yield (key, record) for activation records belonging to ``layer_idx``."""
        for key, rec in self._registry.items():
            if rec.category == 'activation' and rec.layer_idx == layer_idx:
                yield key, rec

    def _estimate_recompute_flops(self, rec: TensorRecord) -> int:
        """Use the record's recompute_cost_flops if set (SwiGLU has it).
        Otherwise fall back to a 2-FLOPs-per-output-byte heuristic.
        """
        if rec.recompute_cost_flops > 0:
            return rec.recompute_cost_flops
        return rec.size_bytes * 2

    def _evict_cost_ms(self, size_bytes: int) -> float:
        """Round-trip cost in ms via the calibration's realistic round-trip."""
        bw = self._calibration.realistic_round_trip_bytes_per_s
        if bw <= 0:
            return float('inf')
        return (size_bytes / bw) * 1000.0

    def _recompute_cost_ms(
        self, flops: int, layer_idx: Optional[int] = None,
        kind: Optional[str] = None,
    ) -> float:
        """Recompute cost in ms.

        Three-tier lookup: this layer's own measured samples for this
        bundle class (fed by unpack_fn via ``record_recompute_timing``),
        then the CLASS pool, then the ``flops / gpu_flops_per_s`` proxy.
        The proxy prices an attention replay far too cheap, which would
        make ``exposed_evict_ms == 0`` an algebraic identity for any
        authorized eviction. It is reached only before any sample of any
        layer in the class exists, which the step-0 class probe closes.
        """
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
        """Mutate registry / mem_stream to reflect the decision.

        Live mode (engine_live=True): submit_* issues real cudaMemcpyAsync
        in addition to the tier / in_flight flips. Dry-run mode: tier and
        in_flight flips only; submit_* records a no-op event for the
        engine's bookkeeping.
        """
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
            # tier stays cpu_dram until the prefetch's event fires; in_flight=True
            self._registry.set_in_flight(key, True)
        # retain / stall / no_op carry no side effects beyond telemetry.

    def _prefetch_lookahead(self, window_size_events: int) -> list[PlacementDecision]:
        """Find activations whose next read falls in [current_step, current_step+window).

        For each such read whose tensor is in cpu_dram and not already
        in flight, submit a prefetch and record a decision. Params and
        grads are skipped — FSDP owns their lifecycle.
        """
        decisions: list[PlacementDecision] = []
        lo = self._current_step
        hi = self._current_step + window_size_events
        seen: set[str] = set()  # avoid duplicate prefetch in the same window

        for ev in self._schedule.events_in_range(lo, hi):
            if ev.access_type != 'read':
                continue
            if not ev.tensor_id.startswith('act_'):
                # Only engine-managed activations. param_layer_/grad_layer_
                # ids are not registry keys and FSDP handles params.
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
        """Classify a single activation's state at bwd-entry time."""
        if rec.tier == 'hbm' and not rec.in_flight:
            return PlacementDecision(
                step_idx=self._current_step, layer_idx=layer_idx,
                tensor_id=key, action='retain',
                estimated_cost_ms=0.0,
                reason=PlacementDecisionReason(note='hbm_resident'))

        if rec.tier == 'cpu_dram' and rec.in_flight:
            op = self._inflight_by_tensor.get(key)
            wait_ms = self._mem_stream.wait(op) if op is not None else 0.0
            # Either way the prefetch is now complete — flip registry state.
            self._registry.set_tier(key, 'hbm')
            self._registry.set_in_flight(key, False)
            self._inflight_by_tensor.pop(key, None)
            if wait_ms > 0:
                # Real decomposition when the op has timing events. pcie_ms is
                # the actual GPU-side memcpy duration; allocator_ms is whatever
                # else contributed to the CPU wall stall (stream-sync overhead,
                # prior queued ops on the mem stream, allocator pauses). Falls
                # back to an 80/20 heuristic only when the op has no
                # event_start (dry-run path). The ``source`` field lets JSON
                # consumers distinguish measured vs heuristic.
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
            # Policy miss — backward reached this tensor before any
            # prefetch was scheduled. We record the miss and flip the
            # tier so subsequent logic sees a consistent state. Under
            # live mode the actual synchronous fetch happens via the
            # unpack_fn path; the cost recorded here is for telemetry.
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
            # In live mode the actual re-execution happens later
            # inside unpack_fn when autograd asks for the saved tensor; this
            # branch is logical bookkeeping for the registry + decision log.
            # The flip back to 'hbm' is a logical statement ("the activation
            # will be alive again by the time this layer's grad runs"), not
            # a physical placement. Pass layer_idx so the cost model uses
            # measured wall-clocks from prior steps when available.
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

        # tier == 'freed' or any unexpected value — shouldn't happen in
        # the normal flow; record a no_op so we can grep for it.
        return PlacementDecision(
            step_idx=self._current_step, layer_idx=layer_idx,
            tensor_id=key, action='no_op',
            estimated_cost_ms=0.0,
            reason=PlacementDecisionReason(note=f'unexpected_tier_{rec.tier}'))

    def _reap_inflight(self) -> None:
        """Reap completed mem-stream ops; flip registry tier/in_flight."""
        done = self._mem_stream.reap_completed()
        for op in done:
            # Harvest GPU-side timing
            # before doing anything else. op_pcie_ms returns None for
            # dry-run/no-op submissions (event_start absent); in live
            # mode both events have already fired (reap_completed gated
            # on op.event.query()), so elapsed_time is non-blocking.
            pcie_ms = self._mem_stream.op_pcie_ms(op)
            if pcie_ms is not None:
                if op.direction == 'evict':
                    self._total_pcie_ms_evict += pcie_ms
                    self._total_bytes_evict += op.nbytes
                    self._total_timed_ops_evict += 1
                else:  # 'prefetch'
                    self._total_pcie_ms_prefetch += pcie_ms
                    self._total_bytes_prefetch += op.nbytes
                    self._total_timed_ops_prefetch += 1

                # Per-op log + ratio samples. Predicted-ms fields may be
                # None when calibration was not threaded into
                # MemoryStreamManager (unit tests);
                # in that case we still log observed but skip the ratios.
                # Ratio samples are uncapped (cheap, used for summary
                # stats); the JSON-bound per-op log is capped to the last
                # _per_op_log_max records at summary() time.
                # Gated on pcie_ms so the planner's byte-free bookkeeping
                # submissions (_apply_decision, no src/dst handles) stay out
                # of the demand histogram — only ops that moved bytes count.
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
            # Single owner for pool release. The bytes
            # are in HBM once the H2D completes, so the pinned buffer is
            # free from here. Must be the prefetch op, never the evict op —
            # the evict's buffer is exactly what the H2D still has to read.
            if op.direction == 'prefetch' and op.buffer is not None:
                self._pool.release(op.buffer)
            # Absence begins when the D2H
            # completes — until then record_stream keeps the source storage
            # alive, so the bytes are still on the card. Stamped at the first
            # reap that observes the op, which is an upper bound on the true
            # completion interval and therefore books LESS absence than
            # occurred. Under-claiming is the right direction for a relief
            # guarantee.
            if op.direction == 'evict':
                rec = self._record_by_pack_id.get(op.tensor_id)
                if rec is not None and rec.d2h_done_ordinal is None:
                    rec.d2h_done_ordinal = self._current_ordinal()
            try:
                # Prefetches: bring tier back to 'hbm' on completion.
                # Evicts: tier already flipped at submit; just clear in_flight.
                if op.direction == 'prefetch':
                    rec = self._registry.get(op.tensor_id)
                    # Only flip back to 'hbm' if the record isn't 'freed' —
                    # an in-flight prefetch for an already-freed tensor would
                    # be a bug, but we defensively skip.
                    if rec is not None and rec.tier != 'freed':
                        self._registry.set_tier(op.tensor_id, 'hbm')
                self._registry.set_in_flight(op.tensor_id, False)
            except KeyError:
                pass  # record was unregistered between submit and reap
            self._inflight_by_tensor.pop(op.tensor_id, None)

    # ── Cost-model inputs for the exposed-cost criterion ──────────────────
    # Pure functions of calibration, schedule and engine wall-clock state.

    def record_layer_timing(
        self, layer_idx: int, phase: str, ms: float,
    ) -> None:
        """Push a per-layer fwd/bwd ms into the engine's wall-clock table.

        Called by the integration (train_fsdp_profile.py) after each
        step, fed from FSDPProfiler.layer_profiles[L].fwd_duration_ms /
        bwd_duration_ms. The engine averages these into _step_to_ms /
        _event_to_ms; until the first push, the defaults stand.

        ``phase`` is 'fwd' or 'bwd'. 'recompute' is bucketed with 'fwd'
        because a recompute is structurally a re-run of the forward pass.
        """
        if ms is None or ms <= 0:
            return
        if phase == 'fwd' or phase == 'recompute':
            self._per_layer_fwd_ms[layer_idx] = float(ms)
        elif phase == 'bwd':
            self._per_layer_bwd_ms[layer_idx] = float(ms)
        # Unknown phases are silently ignored — see _event_to_ms fallback.

    def record_recompute_timing(
        self, layer_idx: int, wall_ms: float, kind: str = None,
    ) -> None:
        """Feed measured recompute wall-clock back into the cost model.

        Called from the ``unpack_fn`` replay path after each physical
        re-execution of layer ``layer_idx``'s forward. Subsequent calls
        to ``_recompute_cost_ms(flops, layer_idx=L)`` will use the
        running mean of these samples instead of the
        flops / gpu_flops_per_s estimate. Mirrors the
        observed-vs-predicted pattern for PCIe ops.

        Silently no-ops on non-positive ms (defensive — CUDA events can
        report zero / negative under degenerate conditions).

        The sample also lands in a per-CLASS pool. A layer that has never
        been recomputed has no sample of its own — the class pool is what
        those fall back on instead of the FLOP proxy.
        """
        if wall_ms is None or wall_ms <= 0:
            return
        self._recompute_observed_ms.setdefault(
            (layer_idx, kind), []).append(float(wall_ms))
        if kind is not None:
            self._recompute_class_ms.setdefault(kind, []).append(
                float(wall_ms))

    def _avg_known_layer_ms(self, phase: str) -> float:
        """Mean of recorded per-layer timings for ``phase``; falls back
        to the engine's default if the table is empty."""
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
        """Wall-clock ms attributable to one AccessEvent.

        Per-layer ms / events_per_layer. 'recompute' events share the
        fwd table because they re-run forward kernels.
        """
        if ev.phase == 'bwd':
            layer_ms = self._per_layer_bwd_ms.get(
                ev.layer_idx, self._default_layer_bwd_ms)
            events_per_layer = self.EVENTS_PER_BWD_LAYER
        else:  # 'fwd' or 'recompute'
            layer_ms = self._per_layer_fwd_ms.get(
                ev.layer_idx, self._default_layer_fwd_ms)
            events_per_layer = self.EVENTS_PER_FWD_LAYER
        return layer_ms / max(events_per_layer, 1)

    def _step_to_ms(self, step_distance: int, current_phase: str) -> float:
        """Convert a schedule step-distance to ms.

        Used for high-level summaries / debugging. The fine-grained
        window math goes through _classify_window_nccl_phase, which
        consults per-event ms via _event_to_ms.

        For windows that straddle phase boundaries the caller should
        use _classify_window_nccl_phase instead; this helper assumes a
        single dominant phase.
        """
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
        """Split [lo_step, hi_step) into (NCCL-active ms, NCCL-idle ms).

        Conservative rule:
        backward events = NCCL-active (FSDP issues all_gather /
        reduce_scatter on the backward critical path). Forward and
        recompute events = NCCL-idle (forward prefetches are typically
        already in flight from the previous layer).
        """
        active_ms = 0.0
        idle_ms = 0.0
        for ev in self._time_slots_in_range(lo_step, hi_step):
            ev_ms = self._event_to_ms(ev)
            if ev.phase == 'bwd':
                active_ms += ev_ms
            else:
                idle_ms += ev_ms
        return active_ms, idle_ms

    # TRAP: the two window helpers price wall-clock per schedule ORDINAL --
    # _event_to_ms divides a layer's measured ms by
    # EVENTS_PER_{FWD,BWD}_LAYER, which counts ordinals. Residual events
    # deliberately share their anchor's ordinal rather than consuming one, so
    # walking raw events here would bill the same slot twice and inflate
    # O(w) — and a bigger window prices transfers cheaper and authorizes more
    # migration. Where every event owns a distinct ordinal the dedup is a
    # no-op.
    def _time_slots_in_range(self, lo_step: int, hi_step: int):
        seen: set[int] = set()
        for ev in self._schedule.events_in_range(lo_step, hi_step):
            if ev.step_idx in seen:
                continue
            seen.add(ev.step_idx)
            yield ev

    # ── O(w) as a consumed budget ──────────────────────────────────────────
    # See the ledger comment in __init__ for why. These two helpers are the
    # whole mechanism: one prices a window per event slot, the other charges
    # a committed eviction against the slots it will actually occupy.

    def _window_slot_capacity_bytes(
        self, lo_step: int, hi_step: int,
    ) -> tuple[float, float, dict]:
        """(active_ms, idle_ms, {event step_idx: PCIe bytes that slot carries}).

        Same phase rule and same totals as _classify_window_nccl_phase — this
        version additionally keeps the per-slot breakdown, which is what makes
        the window divisible between competing decisions. Two decisions whose
        windows overlap partially compete only over the slots they share.
        """
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
        """Charge ``nbytes`` against a window's slots, proportional to capacity.

        Proportional rather than greedy fill-earliest: the mem stream does not
        promise where inside the window a copy lands, and proportional spread
        is the only allocation that leaves the *relative* pricing of two
        overlapping windows independent of decision order.
        """
        if not slots or nbytes <= 0:
            return
        total = sum(slots.values())
        if total <= 0:
            # Zero-capacity window (calibration not populated yet). Charge the
            # first slot so a later decision over the same span still sees it.
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
        """Bytes-in-flight on the mem stream that share this PCIe budget.

        Heuristic: assume every currently in-flight op completes
        somewhere inside the window. The mem stream serializes ops, so
        their PCIe bytes deduct from the same window's budget regardless
        of which exact event slot they land in.
        """
        # NOTE: lo/hi are unused in the heuristic; they're kept on the
        # signature so a later refinement can swap in event-time-based
        # filtering without touching the call sites.
        total = 0
        # _inflight is the manager's private list; touching it here is
        # justified because the engine owns the manager's lifecycle.
        for op in getattr(self._mem_stream, '_inflight', []):
            total += int(op.nbytes)
        return total

    def _available_overlap_ms(
        self,
        size_bytes: int,
        representative_tensor_id: str,
        decision_step: int,
    ) -> tuple[float, dict]:
        """ms-equivalent (at idle PCIe bandwidth) of transfer budget the
        window provides for this tensor.

        Returns (overlap_ms, telemetry). ``telemetry`` carries the raw
        sub-window numbers so the caller can record them in
        PlacementDecisionReason for the diagnostic figure.

        Three terms:
          1. Time term: window in step-ordinals, mapped to ms via
             _classify_window_nccl_phase (per-event sum, phase-correct).
          2. NCCL-active fraction: bwd sub-window charged at
             pcie_bw_under_nccl, fwd/recompute sub-window at pcie_bw_idle.
          3. Queue-contention: subtract bytes of in-flight mem-stream ops
             AND bytes already debited by earlier committed evictions whose
             windows overlap this one.

        Result is multiplied by _overlap_safety_factor.

        Capuchin's max(0, |b|/B(w) − O(w)) is written for one decision; with
        many bundles deciding per step, each reading the same O(w) in full,
        the criterion would price the whole set as free. Term 3's second
        half — debiting — makes the window a resource that runs out.
        """
        nxt = self._schedule.next_read_after(
            representative_tensor_id, decision_step)
        if nxt is None:
            # No future read in this step's schedule. Treat as infinite
            # overlap (transfer always hides) but cap to a sane number
            # so downstream max(0, transfer_ms - overlap_ms) stays well-
            # defined. 10 s is far larger than any single step.
            return 1e4, {'active_ms': 0.0, 'idle_ms': 1e4, 'queued_bytes': 0,
                         'capacity_bytes': 0.0, 'debited_bytes': 0.0,
                         'slots': {}}

        active_ms, idle_ms, slots = self._window_slot_capacity_bytes(
            decision_step, nxt)
        bw_idle = self._calibration.pcie_bw_idle_one_way_bytes_per_s

        # Available transfer volume in the window, byte-budgeted by phase.
        capacity_bytes = sum(slots.values())

        # Queue contention: bytes already scheduled on the mem stream.
        queued = self._queued_bytes_in_window(decision_step, nxt)
        # Ledger: bytes earlier decisions in this step already claimed from
        # the slots this window spans.
        debited = 0.0
        if self._window_debit_enabled:
            for slot in slots:
                debited += self._window_debit_bytes.get(slot, 0.0)
        volume_bytes = max(0.0, capacity_bytes - queued - debited)

        # Safety factor — deflates the window so borderline decisions
        # tilt toward recompute when our wall-clock estimate is uncertain.
        volume_bytes *= self._overlap_safety_factor

        telemetry = {
            'active_ms': active_ms,
            'idle_ms': idle_ms,
            # Reported as one number because both are bytes this window can
            # no longer offer: in-flight ops plus same-step commitments.
            'queued_bytes': int(queued + debited),
            'capacity_bytes': capacity_bytes,
            'debited_bytes': debited,
            'slots': slots,
        }

        # Convert to ms-equivalent at idle bw so transfer_ms - overlap_ms
        # is meaningful (transfer_ms was computed at idle bw above).
        if bw_idle <= 0:
            return 0.0, telemetry
        overlap_ms = volume_bytes / bw_idle * 1000.0
        return overlap_ms, telemetry

    def _estimate_reallgather_overlap_ms(self, layer_idx: int) -> float:
        """Backward-compute window available to hide L's re-gather behind
        async NCCL.

        Under FSDP's ``BACKWARD_PRE`` policy, the all-gather of L's
        params is launched during L+1's backward compute, runs async on
        the NCCL stream, and completes (in steady state) before L's
        backward starts. The size of this overlap budget is L+1's
        backward wall-clock — measured per layer via
        ``record_layer_timing(L+1, 'bwd', ms)``.

        For ``layer_idx == num_layers - 1`` (the first layer to enter
        backward), no preceding backward compute exists; overlap = 0.

        The ``_per_layer_bwd_ms`` table is empty for the first cluster
        step; falls back to ``_default_layer_bwd_ms`` until populated.
        """
        if layer_idx >= self._num_layers - 1:
            return 0.0
        return self._per_layer_bwd_ms.get(
            layer_idx + 1, self._default_layer_bwd_ms)

    def _exposed_reallgather_ms(self, layer_idx: int) -> float:
        """NCCL re-gather cost charged to a recompute decision, exposed.

        Overlap-aware. Raw cost is ``nbytes / nccl_bw``; exposed cost
        subtracts the available overlap with the next-up backward layer's
        compute (under ``BACKWARD_PRE`` this is when the all-gather runs
        async).

        Zero if the coordinator says L's params are already resident,
        or if NCCL bw is 0 (single-rank smoke tests).
        """
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
        """Exposed-cost evict-vs-recompute for one record or one group.

        Gate 0: CPU DRAM capacity. If the pinned pool
        can't supply a buffer for ``size_bytes``, recompute is forced
        regardless of cost numbers.

        ``member_sizes`` is the per-tensor byte
        list when the caller is deciding for a bundle. ``pack_fn``
        acquires one buffer per tensor, so with it Gate 0 asks for N
        buffers of those sizes rather than one buffer of their sum.

        ``commit`` controls whether an ``evict_to_cpu`` outcome
        charges its bytes against the overlap-window ledger. Callers that
        invoke this for telemetry only and discard the action — the forced
        baselines in ``_plan_forced`` — must pass False, or they debit a
        window for traffic their own forced action may never generate.

        Gate 1: exposed-cost compare.
            transfer_ms          = size_bytes / pcie_idle_bw × 1e3
            overlap_ms           = _available_overlap_ms(...)
            exposed_evict_ms     = max(0, transfer_ms - overlap_ms)

            recompute_ms         = flops / gpu_flops × 1e3
            reallgather_ms       = _exposed_reallgather_ms(layer_idx)
            exposed_recompute_ms = recompute_ms + reallgather_ms

            evict ≤ recompute   → evict_to_cpu
            else                 → recompute

        Whole-layer unit: recompute is unavailable. The bundle's only save
        is the layer's INPUT; replaying this module reproduces its OUTPUT,
        so the input can only come from layer L-1 — a chain back to the
        embedding, which is a different mechanism.

        The fallback is EVICT, not retain. This criterion runs
        only when the projected peak already exceeds the budget, so there is
        no feasible retain to fall back to — refusing to act means accepting
        the OOM. Gate 1 therefore pays the transfer it just priced as the
        more expensive option, keeping the original note so the reason
        recompute won is still readable. Gate 0 is the exception: with no
        pool buffer there is nothing to evict INTO, so it retains and the
        caller reports a bundle that got no relief under pressure.
        """
        recompute_available = (kind != 'layer')

        # ── Gate 0: CPU DRAM full ──────────────────────────────────────
        # Bundle callers pass member_sizes so this matches pack_fn's
        # one-buffer-per-tensor acquisition. Passing size_bytes alone would
        # ask for a single buffer as large as the whole bundle, which no
        # bucket serves, and force the bundle to recompute.
        if member_sizes:
            _has_capacity = self._pool.has_capacity_for_sizes(
                member_sizes, kind=kind)
        else:
            _has_capacity = self._pool.has_capacity(size_bytes, kind=kind)
        if not _has_capacity:
            if not recompute_available:
                # No buffer to evict into and no replay to fall back on, so
                # unlike Gate 1 there is no evict fallback here — this bundle
                # has no relief available at all. The caller raises under
                # strict_evict: a pool too small to serve the frontier is an
                # implementation bug, never a finding.
                return ('retain', PlacementDecisionReason(
                    cpu_dram_full=True,
                    note='no_relief_cpu_dram_full',
                ))
            # Pass layer_idx so _recompute_cost_ms uses measured wall-clock
            # once replay samples exist.
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

        # ── Gate 0b: class cost probe ──────────────────────────────────
        # `_recompute_cost_ms` can only learn a class's true cost from a
        # physical replay, and under the FLOP proxy a class that is never
        # replayed would stay priced at the proxy forever. Force one bundle
        # per class to recompute until the class owns a sample.
        # Keyed by (kind, training step) rather than kind alone so a probe
        # whose replay never physically fires is retried next step instead
        # of stranding the class on the proxy; the sample guard stops it
        # after the first success, so the cost is one recompute per class,
        # once.
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

        # ── Gate 1: exposed-cost compare ───────────────────────────────
        bw_idle = self._calibration.pcie_bw_idle_one_way_bytes_per_s
        transfer_ms = (size_bytes / bw_idle * 1000.0) if bw_idle > 0 else float('inf')

        overlap_ms, window_telemetry = self._available_overlap_ms(
            size_bytes, representative_tensor_id, decision_step)
        exposed_evict_ms = max(0.0, transfer_ms - overlap_ms)

        # Pass layer_idx so _recompute_cost_ms uses measured wall-clock once
        # replay samples exist.
        recompute_ms = self._recompute_cost_ms(
            flops, layer_idx=layer_idx, kind=kind)
        reallgather_ms = self._exposed_reallgather_ms(layer_idx)
        exposed_recompute_ms = recompute_ms + reallgather_ms

        # Diagnostic note — categorizes the corner that drove this decision.
        if exposed_evict_ms <= exposed_recompute_ms:
            action = 'evict_to_cpu'
            # Note absent overlap_ms == 0 even for evicts kept default ''.
            if overlap_ms <= 0.0 and exposed_evict_ms > 0:
                note = 'pcie_saturated_in_window'
            elif window_telemetry['active_ms'] > 0 and window_telemetry['idle_ms'] == 0:
                note = 'pcie_under_nccl_window'
            else:
                note = 'evict_under_exposed_cost'
        else:
            action = 'recompute'
            # Did the ledger flip this one? Recompute the undebited window and
            # ask whether it would have authorized the eviction. This counts
            # the migrations the shared-budget accounting refuses that a
            # per-decision formula could not see.
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
                # No replay can free this bundle and the projection is
                # already over budget, so evict-and-pay is the only feasible
                # outcome. Keep the original note: it records which corner
                # made recompute look cheaper, which is the number that says
                # how much time the infeasibility cost.
                action = 'evict_to_cpu'
                note = f'evict_fallback_{note}'

        # An eviction we are actually going to make consumes the window it
        # just priced itself against. Charged before returning so the very
        # next bundle in this layer's loop sees the smaller window.
        # D2H only, by design: the H2D restore is real traffic on the
        # same link but lands in a later window than the one being priced, so
        # charging it here would debit the wrong slots. The ledger therefore
        # understates link pressure by roughly 2x — a known, deliberate bound.
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
