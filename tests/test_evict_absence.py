"""
Evict must mean absence at the peak instant.

Two things would make "we evicted N bytes" an unfalsifiable statement:

  (denial) A denied pool acquire returns the tensor unchanged. The planner
       has already booked that bundle's relief, so unless the denial is
       recorded the projection and the allocator disagree by exactly those
       bytes and nothing says so.
  (absence) Eviction buys a duty-cycled window, not lifetime absence: the
       bytes are provably gone only between the D2H completing and the restore
       taking its destination storage. A restore that lands before the peak
       interval pays for both copies and delivers no relief at all — and the
       planner cannot see that from the inside.

This test pins the telemetry that makes both statements measurable, and the
strict mode that makes a denial fatal for runs above the memory ceiling.

Usage:
    PYTHONPATH=. python3 tests/test_evict_absence.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from memory_manager.placement_engine import PackedEvict, PlacementEngine

MB = 1024 * 1024


class FakePool:
    """Pool whose capacity is a simple budget of buffers."""

    def __init__(self, free_buffers: int):
        self.free = free_buffers
        self.released = 0

    def has_capacity(self, nbytes, kind=None):
        return self.free > 0

    def acquire(self, nbytes):
        if self.free <= 0:
            raise RuntimeError("FakePool exhausted")
        self.free -= 1
        return SimpleNamespace(capacity_bytes=nbytes, in_use=True)

    def release(self, buf):
        self.released += 1
        self.free += 1


class FakeStream:
    def __init__(self):
        self.submitted = []

    def submit_evict(self, tensor_id, nbytes, step, src_tensor, host_buf):
        op = SimpleNamespace(tensor_id=tensor_id, direction='evict',
                             nbytes=nbytes, buffer=host_buf)
        self.submitted.append(op)
        return op


def build_engine(pool, ordinal_holder=None, strict=False):
    """Harness carrying only the state pack_fn's evict path reads."""
    eng = SimpleNamespace()
    eng._engine_live = True
    eng._current_bundle = None
    eng._current_step = 0
    eng._training_step_count = 0
    eng._pool = pool
    eng._mem_stream = FakeStream()

    eng._obs_bundle_bytes_cur = {}
    eng._obs_bundle_members_cur = {}
    eng._obs_bundle_owned_cur = {}
    eng._bundle_seen_ptrs = set()
    eng._bundle_foreign_ptrs = set()
    # Foreign-provenance + distinct-storage diagnostics.
    eng._bundle_foreign_src = {}
    eng._obs_bundle_foreign_cur = {}
    eng._obs_bundle_distinct_cur = {}
    eng._packed_recompute_bundles = {}
    eng._recompute_module_cache = {}
    eng._packed_evict_bundles = {}
    eng._packed_events = []
    eng._packed_by_bundle = {}
    eng._record_by_pack_id = {}

    eng._evict_realized_bytes = 0
    eng._evict_realized_count = 0
    eng._evict_denied_bytes = 0
    eng._evict_denied_count = 0
    eng._evict_denial_log = []
    eng._evict_denial_log_max = 512
    eng._evict_partial_bundles = set()
    eng._evict_staged_bundles = set()
    eng._evict_denied_bundles = set()
    eng._strict_evict = strict
    eng._evict_denial_warned = set()
    # The non-HBM skip counters. _evict_device_type is None here so this
    # harness's CPU tensors stay evictable — on a GPU it is 'cuda' and the
    # memory-efficient kernel's CPU philox seed is skipped instead of
    # crashing the stager.
    eng._evict_skipped_nonhbm_bytes = 0
    eng._evict_skipped_nonhbm_count = 0
    eng._evict_skipped_nonhbm_kinds = set()
    eng._evict_device_type = None
    # The regime B restore diagnostics. All inert here — this test is
    # about denials and absence, not about the restore path's fidelity — but
    # pack_fn reads them unconditionally, so the harness must carry them.
    eng._evict_filter = 'all'
    eng._evict_filter_kind = None
    eng._evict_filter_bytes = None
    eng._evict_filter_bytes_exclude = False
    eng._evict_filtered_bytes = 0
    eng._evict_filtered_count = 0
    eng._evict_filtered_classes = {}
    eng._layout_census_enabled = False
    eng._layout_census = {}
    eng._verify_restore = False
    eng._verify_log = {}
    eng._verify_checked = 0
    eng._verify_failed = 0

    eng._absence_log = []
    eng._absence_log_max = 512
    eng._absence_bytes_total = 0
    eng._absence_bytes_covering_peak = 0
    eng._absence_empty_window_count = 0
    eng._absence_unrestored_count = 0
    eng._prefetch_window_layers = 2

    # ordinal_holder is a one-element list the test advances to simulate
    # layer-interval boundaries.
    if ordinal_holder is None:
        eng._interval_ordinal_source = None
    else:
        eng._interval_ordinal_source = lambda: (ordinal_holder[0], 'phase')
    eng._peak_interval_source = None

    for name in ('pack_fn', '_record_evict_denial', '_current_ordinal',
                 '_reconcile_absence', '_evict_filter_allows', '_diag_class',
                 '_record_layout', '_verify_one'):
        setattr(eng, name, getattr(PlacementEngine, name).__get__(eng))
    eng._PACKED_SENTINEL = PlacementEngine._PACKED_SENTINEL
    eng._PACKED_RECOMPUTE_SENTINEL = PlacementEngine._PACKED_RECOMPUTE_SENTINEL
    return eng


def save(eng, bundle, mbytes):
    """Drive one save_for_backward of ``mbytes`` MiB through pack_fn."""
    eng._current_bundle = bundle
    eng._packed_evict_bundles[bundle] = True
    t = torch.empty(mbytes * MB, dtype=torch.uint8)
    return eng.pack_fn(t)


def test_denial_is_recorded_not_silent():
    """Denial: a denied acquire books the bytes it failed to relieve."""
    eng = build_engine(FakePool(free_buffers=0))
    out = save(eng, (7, 'attention'), 4)

    assert torch.is_tensor(out), "denied save must pass the tensor through"
    assert eng._evict_denied_count == 1
    assert eng._evict_denied_bytes == 4 * MB
    entry = eng._evict_denial_log[0]
    assert (entry['layer'], entry['kind']) == (7, 'attention'), entry
    assert entry['reason'] == 'pool_no_capacity', entry
    print(f"  [denial] booked: L{entry['layer']} {entry['kind']} "
          f"{entry['nbytes'] // MB} MiB, reason={entry['reason']}")


def test_realized_fraction_matches_the_mix():
    """Denial: planned-vs-realized is computed live, not reconstructed later."""
    eng = build_engine(FakePool(free_buffers=2))
    save(eng, (0, 'attention'), 10)     # staged
    save(eng, (1, 'attention'), 10)     # staged
    save(eng, (2, 'attention'), 20)     # denied — pool exhausted

    realized = eng._evict_realized_bytes
    denied = eng._evict_denied_bytes
    frac = realized / (realized + denied)
    assert (realized, denied) == (20 * MB, 20 * MB), (realized, denied)
    assert abs(frac - 0.5) < 1e-9, frac
    print(f"  [denial] realized fraction {frac * 100:.1f}% "
          f"({realized // MB} MiB staged, {denied // MB} MiB denied)")


def test_partial_bundle_is_flagged():
    """Denial: a half-staged bundle pays the copies and relieves nothing."""
    eng = build_engine(FakePool(free_buffers=1))
    bundle = (3, 'mlp')
    save(eng, bundle, 8)     # staged
    save(eng, bundle, 8)     # denied

    assert eng._evict_partial_bundles == {bundle}, eng._evict_partial_bundles
    print(f"  [denial] partial bundle flagged: L{bundle[0]}_{bundle[1]} — copies "
          f"paid, storages still resident")


def test_strict_mode_raises():
    """Denial: above the ceiling a denial is an OOM, so make it stop the run."""
    eng = build_engine(FakePool(free_buffers=0), strict=True)
    try:
        save(eng, (5, 'attention'), 4)
    except RuntimeError as exc:
        assert 'strict_evict' in str(exc), exc
        print(f"  [denial] strict_evict raised instead of retaining silently")
        return
    raise AssertionError("strict_evict did not raise on a denied eviction")


def test_non_hbm_save_is_skipped_not_denied():
    """A save that never occupied device memory.

    The memory-efficient SDPA kernel saves its philox RNG seed/offset as a
    0-dim int64 CPU tensor. Staging it would fail twice over — viewing it as
    flat uint8, then record_stream, which is CUDA-only — and both failures
    are the same category error: asking the stager to relieve bytes that were
    never in HBM.

    Three properties, and the third is the one that matters: it must NOT count
    as a denial, because strict_evict aborts on denials and a run that skips a
    weightless CPU scalar has lost nothing it planned.
    """
    eng = build_engine(FakePool(free_buffers=8), strict=True)
    eng._evict_device_type = 'cuda'          # as on a GPU
    bundle = (3, 'attention')
    eng._current_bundle = bundle
    eng._packed_evict_bundles[bundle] = True

    seed = torch.tensor(90210, dtype=torch.int64)   # 0-dim, CPU — the philox seed
    out = eng.pack_fn(seed)

    assert out is seed, "a non-HBM save must pass through unchanged"
    assert eng._evict_skipped_nonhbm_count == 1
    assert eng._evict_skipped_nonhbm_bytes == 8
    assert eng._evict_denied_count == 0, \
        "skipping a weightless CPU save is not a denial and must not raise " \
        "under strict_evict"
    assert eng._evict_realized_bytes == 0, \
        "its bytes must not be booked as relief — they free no device memory"

    # A real device-side save in the same bundle still stages normally, so the
    # guard does not disarm eviction for everything that follows it.
    dev = torch.empty(4 * MB, dtype=torch.uint8)
    eng._evict_device_type = dev.device.type
    staged = eng.pack_fn(dev)
    assert not torch.is_tensor(staged), "device-side save must still stage"
    print(f"  [non-HBM] non-HBM save skipped: {eng._evict_skipped_nonhbm_bytes} B "
          f"({', '.join(eng._evict_skipped_nonhbm_kinds)}), 0 denials under "
          f"strict_evict, device-side save in the same bundle still staged")


def test_evict_ordinal_is_stamped():
    """Absence: the staging interval is recorded at pack time."""
    holder = [4]
    eng = build_engine(FakePool(free_buffers=4), ordinal_holder=holder)
    save(eng, (2, 'attention'), 4)
    holder[0] = 9
    save(eng, (6, 'attention'), 4)

    ordinals = [r.evict_ordinal
                for recs in eng._packed_by_bundle.values() for r in recs]
    assert sorted(ordinals) == [4, 9], ordinals
    # The pack_id index is what lets a D2H completion find its record.
    assert len(eng._record_by_pack_id) == 2, eng._record_by_pack_id
    print(f"  [absence] evict ordinals stamped at pack time: {sorted(ordinals)}")


def make_record(layer, kind, mbytes, d2h_done, restore):
    return PackedEvict(
        host_buf=None, shape=(mbytes * MB,), dtype=torch.uint8, device='cpu',
        evict_op=None, nbytes=mbytes * MB,
        pack_id=f'L{layer}_{kind}', layer_idx=layer, kind=kind,
        evict_ordinal=(d2h_done - 1) if d2h_done is not None else None,
        d2h_done_ordinal=d2h_done, restore_ordinal=restore,
    )


def reconcile(records, peak_ordinal):
    eng = build_engine(FakePool(free_buffers=0))
    eng._peak_interval_source = lambda: (peak_ordinal, 'post_bwd_layer_9', 999)
    eng._packed_by_bundle = {('x', 'y'): records}
    eng._reconcile_absence()
    return eng


def test_absence_covers_or_misses_the_peak():
    """Absence: the headline number — did the bytes exist elsewhere when it mattered."""
    covering = make_record(10, 'attention', 100, d2h_done=3, restore=12)
    missing = make_record(11, 'attention', 100, d2h_done=3, restore=6)
    eng = reconcile([covering, missing], peak_ordinal=8)

    assert eng._absence_bytes_total == 200 * MB
    assert eng._absence_bytes_covering_peak == 100 * MB
    by_layer = {e['layer']: e for e in eng._absence_log}
    assert by_layer[10]['covered_peak'] is True, by_layer[10]
    assert by_layer[11]['covered_peak'] is False, by_layer[11]
    # The missing one still spent both copies: absent over intervals 4..5,
    # while the peak sat at 8.
    assert by_layer[11]['absent_intervals'] == 2, by_layer[11]
    print("  [absence] peak at ordinal 8: bundle absent 4..11 covers it; "
          "bundle absent 4..5 (restored early) does not — 50% coverage, "
          "both copies paid")


def test_early_restore_leaves_an_empty_window():
    """Absence: a restore inside the D2H's own interval relieves nothing at all."""
    eng = reconcile([make_record(4, 'mlp', 50, d2h_done=7, restore=8)],
                    peak_ordinal=9)
    assert eng._absence_empty_window_count == 1
    assert eng._absence_bytes_covering_peak == 0
    assert eng._absence_log[0]['absent_intervals'] == 0
    print("  [absence] restore at ordinal 8 after a D2H completing in 7: zero "
          "absent intervals — the prefetch window un-did the relief")


def test_never_completed_d2h_claims_nothing():
    """Absence: a copy still in flight at step end was never provably absent."""
    eng = reconcile([make_record(1, 'mlp', 50, d2h_done=None, restore=None)],
                    peak_ordinal=5)
    assert eng._absence_unrestored_count == 1
    assert eng._absence_bytes_total == 0
    assert eng._absence_log == []
    print("  [absence] in-flight-at-step-end books no absence rather than "
          "assuming it")


def test_unmeasured_without_a_profiler():
    """Absence: with no interval source, report nothing instead of guessing."""
    eng = build_engine(FakePool(free_buffers=0))
    eng._packed_by_bundle = {
        ('x', 'y'): [make_record(1, 'mlp', 50, d2h_done=2, restore=9)]}
    eng._reconcile_absence()
    assert eng._absence_log == []
    assert eng._absence_bytes_total == 0
    assert eng._current_ordinal() is None
    print("  [absence] no profiler wired: absence unmeasured, not fabricated")


def test_profiler_supplies_the_ordinals():
    """The two clocks are one clock: ordinals ARE snapshot indices."""
    from test_interval_peak import FakeCuda, patch_cuda, build_fake_profiler

    fake = FakeCuda()
    prof = build_fake_profiler()

    with patch_cuda(fake):
        prof._take_snapshot("step_start")           # ordinal 0
        fake.allocate(500 * MB)
        prof._take_snapshot("post_fwd_layer_0")     # ordinal 1, peak 500
        fake.allocate(4400 * MB)
        prof._take_snapshot("post_bwd_layer_9")     # ordinal 2, peak 4400
        fake.allocate(100 * MB)
        prof._take_snapshot("step_end")             # ordinal 3, peak 100

        cur = prof.current_interval_ordinal()
        peak = prof.peak_interval_ordinal()

    assert cur == (3, 'step_end'), cur
    assert peak[0] == 2 and peak[1] == 'post_bwd_layer_9', peak
    assert peak[2] // MB == 4400, peak
    print(f"  [wiring] peak interval is ordinal {peak[0]} ({peak[1]}), "
          f"{peak[2] // MB} MiB — the ordinal an absence window is checked "
          f"against")


def main():
    print("Evict means absence at the peak instant")
    test_denial_is_recorded_not_silent()
    test_realized_fraction_matches_the_mix()
    test_partial_bundle_is_flagged()
    test_strict_mode_raises()
    test_non_hbm_save_is_skipped_not_denied()
    test_evict_ordinal_is_stamped()
    test_absence_covers_or_misses_the_peak()
    test_early_restore_leaves_an_empty_window()
    test_never_completed_d2h_claims_nothing()
    test_unmeasured_without_a_profiler()
    test_profiler_supplies_the_ordinals()
    print("ALL PASS")


if __name__ == '__main__':
    main()
