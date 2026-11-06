"""Does the engine, on a real Llama model, actually SEE the whole-layer unit?

The static probe (tests/test_checkpoint_bundle_ordering.py) settled the hook
ordering on a stand-in module: the checkpoint wrapper saves the layer's
residual boundary at the WRAPPER's entry, so the observer's per-submodule
bundle hooks and the inner layer's decision hook both fire too late. This test
asks the same question of the shipped wiring on HuggingFace's LlamaForCausalLM,
and it is the CPU smoke test that has to pass before any cluster job.

WHAT WOULD MAKE THIS FAIL QUIETLY. Nothing here crashes when it is wrong. If
the bundle never opens, pack_fn's first guard returns every boundary unchanged
and the engine reports a clean step with zero decisions -- the silent no-op.
So every assertion below is a COUNT, and the control arm re-runs the same model
with the per-module unit to show the count it produces is zero.

CHECKS, in the order a failure should be read:
  1. CONTROL: per-module bundles under checkpointing observe nothing. This is
     the silent no-op the whole-layer wiring avoids; if this arm observes
     anything, the premise is wrong and the rest of the test means nothing.
  2. One bundle per layer, every one of kind 'layer', and exactly num_layers
     of them. Not 2L (submodule bundles leaked in) and not 0 (no-op).
  3. Layer L's decision is in hand BEFORE layer L's boundary is packed. The
     bundle can be open and the decision map still empty; those are separate
     failures with the same symptom.
  4. Exactly num_layers zero-element placeholders, passed through un-booked.
     A booked one takes a real pinned buffer from the smallest bucket.
  5. Gradients match a no-engine baseline. Attaching the engine must not
     change the math.
  6. Forced layer-level recompute still gives correct gradients.
  7. Layer-level recompute is REFUSED at this unit, and the fallback is evict.
     Under this unit the only save is the layer's INPUT, and enter_bundle
     would cache that same storage for the replay, so the action would book
     relief it cannot deliver while paying for a full layer replay.

Device-agnostic; runs on CPU in a few seconds.

Run:  PYTHONPATH=. python3 tests/test_whole_layer_unit.py
"""

import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing, checkpoint_wrapper, CheckpointImpl)

from fsdp_profiler import FSDPMemoryProfiler
from memory_manager.access_predictor import AccessPredictor
from memory_manager.calibration import CalibrationResult
from memory_manager.fsdp_coordinator import NoOpCoordinator
from memory_manager.placement_engine import HBMBudget, PlacementEngine
from memory_manager.tensor_registry import TensorRegistry

L = 4
HIDDEN = 64
SEQ = 16
VOCAB = 128


def build_model():
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=2 * HIDDEN,
        num_hidden_layers=L, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=SEQ, use_cache=False,
        attn_implementation='eager')
    model = LlamaForCausalLM(cfg).to(torch.float32)
    model.config.use_cache = False
    return model, cfg


def wrap(model):
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(
            m, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=lambda m: isinstance(m, LlamaDecoderLayer))


class StubPool:
    """Stands in for PinnedBufferPool, which needs CUDA's pin_memory allocator.

    This test is about hook ordering and attribution, not transfers, so the
    pool is never asked for a buffer -- but it IS asked whether it has one,
    and a stub that always says yes is the conservative answer here: it cannot
    mask a missing bundle or a late decision, only a pool-capacity problem
    that this test does not claim to cover.
    """

    def __init__(self, has_capacity=True):
        self.acquires = 0
        self._has_capacity = has_capacity
        # Gate 0 asks "N buffers of these sizes?", and a stub that answers
        # without reading `sizes` cannot see a wrong question (a whole-layer
        # bundle asking for one buffer per interior component).
        self.asked_sizes = []

    def has_capacity(self, nbytes, *a, **k):
        return self._has_capacity

    def has_capacity_for_sizes(self, sizes, *a, **k):
        self.asked_sizes.append(list(sizes))
        return self._has_capacity

    def acquire(self, nbytes):
        self.acquires += 1
        raise RuntimeError(
            f"StubPool.acquire({nbytes}) — this test should never move bytes. "
            f"A 0-byte acquire here means a placeholder was booked.")

    def release(self, buf):
        pass

    def telemetry(self):
        return {}


class StubStream:
    """Stands in for MemoryStreamManager, which constructs a CUDA stream.

    Same reasoning as StubPool: no bytes move in this test. reap_completed
    returning nothing is what the engine sees on a step with no transfers.
    """

    def reap_completed(self):
        return []

    def wait(self):
        pass

    def wait_compute_on(self, *a, **k):
        pass

    def op_pcie_ms(self, *a, **k):
        return None

    def submit_evict(self, *a, **k):
        raise RuntimeError("StubStream.submit_evict — no bytes should move.")

    def submit_prefetch(self, *a, **k):
        raise RuntimeError("StubStream.submit_prefetch — no bytes should move.")


def build_engine(model, cfg, unit, budget_bytes=int(14 * 1024 ** 3),
                 pcie_bw=11.4e9, pool_has_capacity=True):
    registry = TensorRegistry()
    predictor = AccessPredictor(
        num_layers=L, seq_len=SEQ, verbose=False,
        hidden_size=HIDDEN, intermediate_size=2 * HIDDEN,
        num_heads=4, num_kv_heads=2, batch_size=1, dtype_bytes=4,
        bundle_kinds=(('layer',) if unit == 'whole_layer'
                      else ('attention', 'mlp')),
        checkpoint_policy=({i: True for i in range(L)}
                           if unit == 'whole_layer' else None))
    pool = StubPool(has_capacity=pool_has_capacity)
    calib = CalibrationResult(
        pcie_bw_idle_one_way_bytes_per_s=pcie_bw,
        pcie_bw_idle_round_trip_bytes_per_s=6.4e9,
        pcie_bw_under_nccl_one_way_bytes_per_s=2.9e9,
        nccl_allgather_bytes_per_s=1.0e9,
        gpu_flops_per_s=10e12)
    engine = PlacementEngine(
        registry=registry, predictor=predictor,
        hbm_budget=HBMBudget(total_bytes=budget_bytes, registry=registry),
        buffer_pool=pool,
        mem_stream=StubStream(),
        calibration=calib,
        fsdp_coord=NoOpCoordinator(registry=registry, world_size=1,
                                   verbose=False),
        prefetch_window_layers=2, dry_run=True, verbose=False,
        bundle_unit=unit)
    engine.engine_live = True

    profiler = FSDPMemoryProfiler(model, rank=0, world_size=1, model_config=cfg,
                            enabled=True)
    profiler.attach_hooks()
    profiler.attach_engine(engine)
    return engine, profiler


def run_step(model, engine=None):
    """One forward + backward. Returns (loss, decision-order log)."""
    torch.manual_seed(1)
    ids = torch.randint(0, VOCAB, (1, SEQ))
    order = []
    if engine is None:
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        return out.loss, order

    inner_pack = engine.pack_fn

    def pack(t):
        # Snapshot what the engine knows AT the save, not after the step.
        order.append((engine.current_bundle, t.numel(),
                      sorted(engine._pre_decisions_by_layer)))
        return inner_pack(t)

    with torch.autograd.graph.saved_tensors_hooks(pack, engine.unpack_fn):
        out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    return out.loss, order


def grads(model):
    return {n: p.grad.clone() for n, p in model.named_parameters()
            if p.grad is not None}


def main():
    failures = []

    def check(ok, label, detail):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
        if not ok:
            failures.append(label)

    print(f"model: L={L} hidden={HIDDEN} seq={SEQ}, checkpoint-wrapped, CPU\n")

    # ── CONTROL: per-submodule hooks under checkpointing ──────────────────
    print("CONTROL -- per-module unit under checkpointing (per-submodule hooks)")
    m_ctl, cfg_ctl = build_model()
    wrap(m_ctl)
    eng_ctl, _ = build_engine(m_ctl, cfg_ctl, 'per_module')
    _, order_ctl = run_step(m_ctl, eng_ctl)
    observed_ctl = dict(eng_ctl._obs_bundle_bytes_cur)
    saw_bundle_ctl = [b for b, n, d in order_ctl if b is not None]
    check(not observed_ctl and not saw_bundle_ctl,
          "per-module observes nothing under checkpointing",
          f"{len(observed_ctl)} bundles booked, {len(saw_bundle_ctl)} saves "
          f"landed in a bundle, out of {len(order_ctl)} saves seen -- this is "
          f"the silent no-op the whole-layer wiring avoids")

    # ── TREATMENT: the whole-layer unit ──────────────────────────────────
    print("\nTREATMENT -- whole-layer unit on the CheckpointWrapper")
    m, cfg = build_model()
    base = build_model()[0]
    base.load_state_dict(m.state_dict())
    wrap(m)
    wrap(base)
    eng, _ = build_engine(m, cfg, 'whole_layer')
    _, order = run_step(m, eng)

    # The per-step pick cap is read off the bundles the unit actually offers:
    # one per layer under whole-layer, num_layers x 2 under per-module. A
    # fixed cap of 64 would sit AT the layer count at L=64 and BELOW it at
    # L=128, where a cap-induced retain under pressure is an accepted OOM.
    check(eng._planner_max_picks_per_step == L
          and eng_ctl._planner_max_picks_per_step == 2 * L,
          "the pick cap counts the bundles the unit offers",
          f"whole-layer {eng._planner_max_picks_per_step} (expected {L}), "
          f"per-module {eng_ctl._planner_max_picks_per_step} (expected "
          f"{2 * L}); at the L=32 anchor per-module evaluates to "
          f"{32 * 2}")

    observed = dict(eng._obs_bundle_bytes_cur)
    kinds = {k for _, k in observed}
    check(len(observed) == L and kinds == {'layer'},
          "one 'layer' bundle per layer",
          f"{len(observed)} bundles, kinds={sorted(kinds)}, "
          f"expected {L} and {{'layer'}}")

    boundaries = [(b, d) for b, n, d in order if b is not None and n > 0]
    check(len(boundaries) == L,
          "exactly num_layers non-empty saves reach the seam",
          f"{len(boundaries)} of {len(order)} saves, expected {L}")

    late = [b for b, d in boundaries if b[0] not in d]
    check(not late,
          "layer L's decision precedes layer L's boundary save",
          f"{len(late)} boundaries packed before their own layer was "
          f"decided, expected 0" if late else
          "every boundary found its own layer already in the decision map")

    check(eng._packed_zero_elem_count == L,
          "zero-element placeholders passed through un-booked",
          f"{eng._packed_zero_elem_count} passed through, expected {L}; "
          f"pool acquires attempted: {eng._pool.acquires}")

    sizes = sorted({n for b, n, d in order if b is not None and n > 0})
    check(len(sizes) == 1 and sizes[0] == SEQ * HIDDEN,
          "every boundary is one residual-stream tensor",
          f"distinct non-empty save sizes {sizes}, expected [{SEQ * HIDDEN}] "
          f"(= seq x hidden)")

    # ── Correctness: the engine must not change the math ─────────────────
    run_step(base, None)
    g_a, g_b = grads(m), grads(base)
    worst, worst_name = 0.0, ''
    for n in g_b:
        d = (g_a[n] - g_b[n]).abs().max().item()
        if d > worst:
            worst, worst_name = d, n
    check(worst < 1e-5 and len(g_b) > 0 and len(g_a) == len(g_b),
          "gradients match the no-engine baseline",
          f"max abs diff {worst:.3e} at {worst_name or 'none — all equal'} "
          f"over {len(g_b)} parameter tensors (engine arm has {len(g_a)})")

    # ── Is recompute REFUSED for a layer bundle? ─────────────────────────
    # The action cannot free a byte under this unit: the saved tensor IS the
    # layer's input, and the replay cache would hold a live reference to the
    # very storage pack_fn was asked to drop. The engine therefore refuses
    # recompute for 'layer' bundles and retains instead, rather than booking
    # relief it cannot deliver. These checks are the regression guard on that
    # refusal, so they assert BOTH that nothing was recomputed and that the
    # refusal left no pin behind.
    print("\nLAYER-LEVEL RECOMPUTE -- is the action refused?")
    m_rc, cfg_rc = build_model()
    wrap(m_rc)
    eng_rc, _ = build_engine(m_rc, cfg_rc, 'whole_layer')
    eng_rc.force_action = 'recompute'

    # Sample the replay cache AT THE WRITE, not after the step: backward pops
    # every entry per layer in on_post_backward, so a post-step read of
    # _recompute_input_cache is always empty and always passes.
    cached_ptrs: dict = {}
    real_enter = eng_rc.enter_bundle

    def enter(layer_idx, kind, *a, **k):
        real_enter(layer_idx, kind, *a, **k)
        entry = eng_rc._recompute_input_cache.get((layer_idx, kind))
        if entry is not None:
            cached_ptrs[(layer_idx, kind)] = {
                t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
                for t in entry[0] if torch.is_tensor(t)}

    eng_rc.enter_bundle = enter
    packed_ptrs: dict = {}
    real_pack = eng_rc.pack_fn

    def pack_rc(t):
        b = eng_rc.current_bundle
        if b is not None and t.numel() > 0:
            packed_ptrs.setdefault(b, set()).add(
                t.untyped_storage().data_ptr())
        return real_pack(t)

    base_rc = build_model()[0]
    base_rc.load_state_dict(m_rc.state_dict())
    wrap(base_rc)
    run_step(base_rc)
    g_base_rc = grads(base_rc)

    replayed = False
    try:
        torch.manual_seed(1)
        ids = torch.randint(0, VOCAB, (1, SEQ))
        with torch.autograd.graph.saved_tensors_hooks(pack_rc, eng_rc.unpack_fn):
            out = m_rc(input_ids=ids, labels=ids)
        out.loss.backward()
        replayed = True
    except Exception as exc:
        print(f"  (replay raised: {type(exc).__name__}: {exc})")

    # This arm is only meaningful if the boundaries actually reached the
    # seam; without it every count below is zero for the wrong reason.
    check(len(packed_ptrs) == L and replayed,
          "the forced arm reached the seam at all (guards a vacuous pass)",
          f"{len(packed_ptrs)} of {L} layer bundles packed a boundary; "
          f"step {'completed' if replayed else 'DIED'}")

    notes = [d.reason.note for d in eng_rc.decisions
             if d.tensor_id.startswith('bundle_L')]
    actions = sorted({d.action for d in eng_rc.decisions
                      if d.tensor_id.startswith('bundle_L')})
    check(actions == ['retain']
          and all(n == 'recompute_unavailable_whole_layer' for n in notes)
          and len(notes) == L,
          "forced recompute is refused and retained with a named reason",
          f"{len(notes)} layer decisions, actions={actions}, "
          f"notes={sorted(set(notes))}")

    check(len(eng_rc._obs_bundle_dropped_cur) == 0,
          "no recompute descriptor is issued for a layer bundle",
          f"{len(eng_rc._obs_bundle_dropped_cur)} bundles booked as dropped "
          f"(each would be relief the step cannot deliver)")

    pinned = [(b, cached_ptrs[b][p])
              for b, ptrs in packed_ptrs.items()
              for p in ptrs if p in cached_ptrs.get(b, {})]
    check(not pinned and not cached_ptrs,
          "the refusal leaves no replay cache pinning a boundary",
          f"{len(cached_ptrs)} replay caches, {len(pinned)} of which alias "
          f"their own boundary ({[f'L{b[0]}:{n}B' for b, n in pinned[:4]]})"
          if cached_ptrs else "no replay input cached at all")

    if replayed:
        g_rc = grads(m_rc)
        worst = max((float((g_rc[n] - g_base_rc[n]).abs().max())
                     for n in g_base_rc if n in g_rc), default=float('inf'))
        check(worst < 1e-5 and len(g_rc) == len(g_base_rc) and g_base_rc,
              "the refusal leaves gradients correct",
              f"max abs diff {worst:.3e} over {len(g_base_rc)} tensors")

    # ── The CRITERION must refuse it too, not just the forced path ───────
    # _plan_forced and _decide_evict_vs_recompute are separate code paths and
    # only the second one runs on a real frontier job. Squeeze the budget so
    # the planner projects a breach and actually reaches the criterion.
    print("\nTHE CRITERION UNDER PRESSURE -- does it ever pick recompute?")
    m_cr, cfg_cr = build_model()
    wrap(m_cr)
    eng_cr, _ = build_engine(m_cr, cfg_cr, 'whole_layer', budget_bytes=1024)
    run_step(m_cr, eng_cr)

    cr = [d for d in eng_cr.decisions if d.tensor_id.startswith('bundle_L')]
    notes_cr = sorted({d.reason.note for d in cr})
    reached = [d for d in cr if d.reason.note not in
               ('no_pressure', 'no_bwd_event', 'no_byte_model',
                'budget_satisfied', 'pick_cap_reached', 'no_contribution')]
    check(bool(reached),
          "the criterion was actually reached (guards a vacuous pass)",
          f"{len(reached)} of {len(cr)} layer decisions got past the "
          f"pre-criterion branches; notes={notes_cr}")
    check(all(d.action != 'recompute' for d in cr),
          "the criterion never picks recompute for a layer bundle",
          f"actions={sorted({d.action for d in cr})}, notes={notes_cr}")

    # ── What Gate 0 ASKS the pool for, at step 0 ─────────────────────────
    # The byte estimate and the member-size list are separate paths. A
    # member-size list without the whole-layer branch would make Gate 0 ask
    # for one buffer per named interior component -- a 128 MiB SwiGLU demand
    # at the frontier shape, against a 48 MiB top bucket. It only bites at
    # step 0, because from step 1 pack_fn's observed member list shadows the
    # analytic one, so a frontier arm would die before step 1 while feasible
    # arms run clean. Assert the QUESTION, not the
    # answer: the stub always says yes, so a wrong question is invisible in
    # the decisions and shows up only as a pool denial on real hardware.
    boundary_b = 1 * SEQ * HIDDEN * 4          # b x seq x hidden x fp32
    asked = eng_cr._pool.asked_sizes
    check(bool(asked) and all(a == [boundary_b] for a in asked),
          "Gate 0 asks for exactly one boundary buffer per layer bundle",
          f"{len(asked)} requests, distinct={sorted({tuple(a) for a in asked})}"
          f", expected every one to be [{boundary_b}]")

    # CONTROL: the branch must be unit-specific, or it is just a constant.
    # Per-module genuinely saves many tensors per bundle and Gate 0 must
    # keep asking for one buffer each. Unwrapped -- under a checkpoint
    # wrapper per-module observes nothing (the CONTROL at the top).
    m_pmg, cfg_pmg = build_model()
    eng_pmg, _ = build_engine(m_pmg, cfg_pmg, 'per_module', budget_bytes=1024)
    run_step(m_pmg, eng_pmg)
    asked_pm = eng_pmg._pool.asked_sizes
    check(bool(asked_pm) and any(len(a) > 1 for a in asked_pm),
          "CONTROL: per-module still asks for one buffer per member",
          f"{len(asked_pm)} requests, member counts="
          f"{sorted({len(a) for a in asked_pm})}, expected some > 1")

    # ── Exercise the refusal itself, not just its absence ────────────────
    # Above, evict won on cost every time, so the guard never had to fire --
    # a pass for the wrong reason. Price the link into the ground so
    # recompute WOULD win, and separately starve the pool so Gate 0 would
    # have forced recompute. The two branches diverge: Gate 1 falls back to
    # EVICT (the criterion only runs over budget, so retaining is the OOM),
    # while Gate 0 has no buffer to evict into and can only retain.
    print("\nTHE REFUSAL ITSELF -- both paths that would return recompute")
    # Note on the lever: crushing pcie_bw does NOT make evict lose. Both
    # transfer_ms and the overlap window scale as 1/bw, so the comparison is
    # bandwidth-invariant by construction. Zeroing the overlap window is what
    # leaves the transfer fully exposed and lets recompute win.
    for label, kw, no_overlap, want_note, want_action in (
            ("recompute wins on cost -> evict fallback",
             dict(budget_bytes=1024), True, 'evict_fallback_', 'evict_to_cpu'),
            ("pool has no buffer (Gate 0) -> retain",
             dict(budget_bytes=1024, pool_has_capacity=False), False,
             'no_relief_cpu_dram_full', 'retain')):
        m_x, cfg_x = build_model()
        wrap(m_x)
        eng_x, _ = build_engine(m_x, cfg_x, 'whole_layer', **kw)
        if no_overlap:
            eng_x._overlap_safety_factor = 0.0
        run_step(m_x, eng_x)
        dx = [d for d in eng_x.decisions if d.tensor_id.startswith('bundle_L')]
        fired = [d for d in dx if d.reason.note.startswith(want_note)]
        check(bool(fired) and all(d.action == want_action for d in fired),
              f"{label}",
              f"{len(fired)} of {len(dx)} decisions hit '{want_note}*', "
              f"actions={sorted({d.action for d in fired})}, "
              f"all notes={sorted({d.reason.note for d in dx})}")

    # ── The no-relief bookkeeping, and that strict_evict refuses it ───────
    # Gate 0 leaves bytes in the projection that no action removes. Below the
    # ceiling that is a measurement error; above it the step OOMs, so a
    # frontier run must not be able to report a ceiling it never reached.
    print("\nNO-RELIEF BOOKKEEPING -- counted, and fatal under strict_evict")
    m_nr, cfg_nr = build_model()
    wrap(m_nr)
    eng_nr, _ = build_engine(m_nr, cfg_nr, 'whole_layer',
                             budget_bytes=1024, pool_has_capacity=False)
    run_step(m_nr, eng_nr)
    tel = eng_nr.summary()['evict_realization']
    check(tel['no_relief_ops'] > 0 and tel['no_relief_bytes'] > 0
          and 'no_relief_cpu_dram_full' in tel['no_relief_notes'],
          "no-relief bundles are counted in telemetry",
          f"ops={tel['no_relief_ops']}, bytes={tel['no_relief_bytes']}, "
          f"notes={tel['no_relief_notes']}")

    m_se, cfg_se = build_model()
    wrap(m_se)
    eng_se, _ = build_engine(m_se, cfg_se, 'whole_layer',
                             budget_bytes=1024, pool_has_capacity=False)
    eng_se._strict_evict = True
    try:
        run_step(m_se, eng_se)
        raised = ''
    except RuntimeError as exc:
        raised = str(exc)
    check('strict_evict' in raised and 'no relief' in raised,
          "strict_evict raises rather than reporting an unearned ceiling",
          f"raised={raised[:120] or '<nothing>'}")

    # ── Relief attribution: the boundary is the bundle's own mass ────────
    # The foreign rule was written for the per-module unit, where the inbound
    # residual belongs to the producing bundle. Under whole-layer it is the
    # ONLY thing a bundle owns, and excluding it would leave the owned dict
    # empty -- which skips the promotion loop and takes the over-book alarm
    # offline rather than merely reading zero.
    print("\nRELIEF ATTRIBUTION -- the residual boundary is owned, not foreign")
    m_rel, cfg_rel = build_model()
    wrap(m_rel)
    eng_rel, _ = build_engine(m_rel, cfg_rel, 'whole_layer')
    run_step(m_rel, eng_rel)
    owned = dict(eng_rel._obs_bundle_owned_cur)
    boundary = 1 * SEQ * HIDDEN * 4           # b x seq x hidden x fp32
    check(len(owned) == L and all(v == boundary for v in owned.values()),
          "every layer bundle owns exactly one residual boundary",
          f"{len(owned)}/{L} bundles booked, sizes={sorted(set(owned.values()))}"
          f", expected {boundary} B each")
    # Pre-registered: total owned is the migratable supply the span arithmetic
    # quotes -- 2 B x d_model per layer per token, here fp32 on CPU.
    check(sum(owned.values()) == L * boundary,
          "total owned equals L x boundary (the migratable supply)",
          f"{sum(owned.values())} B against L*boundary = {L}*{boundary} = "
          f"{L * boundary} B")
    check(not any(k.endswith('|input')
                  for hist in eng_rel._obs_bundle_foreign_cur.values()
                  for k in hist),
          "no boundary is subtracted as foreign under whole-layer",
          f"foreign histogram={dict(eng_rel._obs_bundle_foreign_cur) or '{}'}")

    # The alarm must be SILENT, and silence has two very different causes:
    # the analytic model agrees with the observation, or the promotion loop
    # never ran (an empty owned dict skips the loop and takes the alarm
    # offline under the headline unit). So assert both the
    # silence and the loop's execution, then prove the alarm still fires by
    # perturbing the seed. Without the third arm this check passes vacuously.
    eng_rel.on_training_step_end()
    alarm = eng_rel._overbooked_bundles
    check(len(alarm) == 0,
          "the over-book alarm is silent: analytic equals the boundary",
          f"{len(alarm)}/{L} bundles flagged")
    check(len(eng_rel._obs_bundle_owned) == L,
          "...and silent because the loop RAN, not because it was skipped",
          f"{len(eng_rel._obs_bundle_owned)}/{L} bundles promoted")
    bm_rel = eng_rel._schedule.bytes_by_tensor
    check(all(bm_rel[eng_rel._predictor.residual_id(i, 'layer')] == boundary
              for i in range(L)),
          "the promotion leaves every boundary at its measured size",
          f"slots={[bm_rel[eng_rel._predictor.residual_id(i, 'layer')] for i in range(L)]}"
          f", expected {boundary} B each (differencing against the interior "
          f"would clamp these to 0 and delete the migratable supply)")
    for i in range(L):
        bm_rel[eng_rel._predictor.residual_id(i, 'layer')] = boundary * 8
    eng_rel._overbook_warned = True
    eng_rel._obs_bundle_owned_cur = {(i, 'layer'): boundary for i in range(L)}
    eng_rel._promote_observed_relief()
    check(len(eng_rel._overbooked_bundles) == L,
          "the alarm is still REACHABLE (seed inflated 8x -> it fires)",
          f"{len(eng_rel._overbooked_bundles)}/{L} flagged at ratio "
          f"{max((v['ratio'] for v in eng_rel._overbooked_bundles.values()), default=0)}x")

    # ── The projection itself: the quantity the feasibility gate reads ─────
    #
    # The gate treats projection > capacity as a hard constraint, so the
    # projection's ERROR SIGN decides whether the gate over-relieves (wasted
    # traffic) or under-relieves (an OOM it was built to prevent). Both signs
    # are possible here: a schedule that models all L layer interiors as
    # simultaneously resident over-projects, and boundary slots that carry zero
    # bytes drop the migratable supply entirely (under). Pre-registered
    # truth, derived independently of the walk: ONE layer's interior live at a
    # time, plus ALL L boundaries, which the wrapper is the only thing keeping.
    pred_p = eng_rel._predictor
    sched_p = eng_rel._schedule
    last_ev = max(e.step_idx for e in sched_p.events)
    interior = sum(pred_p._activation_bytes(c)
                   for c in pred_p._ACTIVATION_COMPONENTS)
    # The peak instant is a layer BACKWARD, and a
    # backward is co-resident with the gradients it is computing. Written as
    # its own term rather than folded into `interior`, because it is a
    # different physical quantity -- the interior is saves, this is transients
    # -- and only the first is what the recompute unit governs.
    transient = pred_p._layer_bwd_transient_bytes()
    truth = interior + L * boundary + transient
    projected, peak_step = sched_p.peak_in_range(0, last_ev)
    check(projected == truth,
          "the projected peak equals one interior + L boundaries + one "
          "backward transient, exactly",
          f"projected {projected} B against truth {interior} + {L}x{boundary} "
          f"+ {transient} = {truth} B (peak at event {peak_step})")

    # Control: leaving checkpoint_policy unset stacks every interior and
    # over-projects by ~L.
    pred_old = AccessPredictor(
        num_layers=L, seq_len=SEQ, verbose=False, hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN, num_heads=4, num_kv_heads=2,
        batch_size=1, dtype_bytes=4, bundle_kinds=('layer',))
    sched_old = pred_old.build_schedule()
    for i in range(L):
        sched_old.bytes_by_tensor[pred_old.residual_id(i, 'layer')] = 0
    old_peak = sched_old.peak_in_range(
        0, max(e.step_idx for e in sched_old.events))[0]
    check(old_peak == L * interior + transient,
          "CONTROL: without checkpoint_policy the schedule stacks all L "
          "interiors and no boundary",
          f"{old_peak} B = L x interior + transient exactly, against truth {truth} B "
          f"({old_peak / truth:.1f}x at L={L}; 12.0x at the L=64 headline "
          f"shape in regime B, 44.0x in regime A -- the ratio is "
          f"shape-dependent, the identity is not)")

    # The boundary must survive the checkpoint discard, and survive it ONCE:
    # freeing it models the supply out of existence, re-creating it in the
    # recompute block books it twice.
    bal = {}
    for e in sched_p.events:
        if e.tensor_id == pred_p.residual_id(0, 'layer'):
            bal[e.access_type] = bal.get(e.access_type, 0) + 1
    check(bal == {'create': 1, 'read': 1, 'free': 1},
          "each boundary is created once, read once, freed once",
          f"L0 boundary events={bal} (a 2nd create is the recompute block "
          f"double-booking; a 2nd free is the forward discard taking it)")

    # The per-module rule must not move: the per-module contrast arm still
    # measures that unit's migratable fraction, and there the inbound residual is
    # the producing bundle's to claim. Unwrapped, because the per-module
    # hooks observe nothing under a checkpoint wrapper (the CONTROL above).
    m_pm, cfg_pm = build_model()
    eng_pm, _ = build_engine(m_pm, cfg_pm, 'per_module')
    run_step(m_pm, eng_pm)
    pm_inputs = [k for hist in eng_pm._obs_bundle_foreign_cur.values()
                 for k in hist if k.endswith('|input')]
    check(len(pm_inputs) > 0,
          "per-module still subtracts the inbound residual as foreign",
          f"{len(pm_inputs)} input-provenance storages subtracted across "
          f"{len(eng_pm._obs_bundle_foreign_cur)} bundles")

    print("\n" + "=" * 68)
    if failures:
        print(f"FAILED ({len(failures)}): {', '.join(failures)}")
        return 1
    print("All checks passed. The engine sees the whole-layer unit: L "
          "boundaries, each in its own layer bundle, each decided before it "
          "is packed, with L placeholders passed through un-booked.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
