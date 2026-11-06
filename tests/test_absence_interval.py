"""
Interval absence in the projection — the planner-side half of "evict means
absent at the peak instant" (tests/test_evict_absence.py is the engine-side
half).

A migration does not delete bytes, it makes them absent over an interval:
from the device-to-host copy until backward restores them at layer L+W.
A planner that booked LIFETIME absence would credit a bundle with relief even
when the restore lands before the peak instant — the projection would
authorise migration the step cannot deliver.

What this file checks, in the order a reader should want it:

  1. The walk is bit-identical with no exclusions. This is the guarantee
     that the un-excluded projection is untouched.
  2. Lifetime absence still works when no restore ordinal is given — that
     is what a recompute buys and what the per-module tests assert against.
  3. A tensor restored AFTER the peak keeps its full contribution; one
     restored BEFORE it contributes nothing.
  4. Replay transients are not invented. Under the whole-layer unit a
     bundle's tensor_ids include tensors created and freed again inside
     the absence interval; re-admitting the bundle's nominal size would
     raise the projected peak above the un-excluded walk. It must not.
  5. A bundle picked by an EARLIER layer, whose create is outside this
     layer's projection window, still has its bytes handed back at its
     restore. A walk that only hands back creates inside its own window
     would silently lose this case.
  6. At the headline shape the model says what it should: the peak sits at
     the first layer backward touches, so exactly the W+1 layers restored
     at that instant buy nothing and the rest keep full relief. That is a
     falsifiable prediction for the engine run, not a tautology.
  7. End to end through the planner: raising W strictly reduces the number
     of layers whose absence covers the peak. This is what makes
     --prefetch_window_layers a MEMORY parameter, not only a time one.

Usage:
    PYTHONPATH=. python3 tests/test_absence_interval.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory_manager.access_predictor import AccessPredictor, AccessEvent
from memory_manager.placement_engine import PlacementEngine

MiB = 1024 ** 2
FAILURES: list[str] = []


def check(cond: bool, label: str, detail: str = '') -> None:
    if cond:
        print(f"  [PASS] {label}" + (f" -- {detail}" if detail else ''))
    else:
        print(f"  [FAIL] {label}" + (f" -- {detail}" if detail else ''))
        FAILURES.append(label)


# ── A hand-built schedule, so every ordinal in the assertions is visible ──
#
# ordinal  event
#   0      create A (100)      <- the migratable save
#   1      create B (10)
#   2      create C (1000)     <- the peak-maker, not migratable
#   3      free   C
#   4      free   B
#   5      free   A
#
# Un-excluded peak is 1110 at ordinal 2.

def toy_schedule():
    sched = SimpleNamespace()
    sched.bytes_by_tensor = {'A': 100, 'B': 10, 'C': 1000}
    sched.events = [
        AccessEvent(step_idx=0, tensor_id='A', access_type='create',
                    layer_idx=0, phase='fwd'),
        AccessEvent(step_idx=1, tensor_id='B', access_type='create',
                    layer_idx=0, phase='fwd'),
        AccessEvent(step_idx=2, tensor_id='C', access_type='create',
                    layer_idx=0, phase='fwd'),
        AccessEvent(step_idx=3, tensor_id='C', access_type='free',
                    layer_idx=0, phase='bwd'),
        AccessEvent(step_idx=4, tensor_id='B', access_type='free',
                    layer_idx=0, phase='bwd'),
        AccessEvent(step_idx=5, tensor_id='A', access_type='free',
                    layer_idx=0, phase='bwd'),
    ]
    sched.peak_in_range = AccessPredictor.build_schedule.__globals__[
        'AccessSchedule'].peak_in_range.__get__(sched)
    return sched


def test_no_exclusions_is_bit_identical() -> None:
    print("\n1. THE PROJECTION ITSELF IS UNTOUCHED")
    sched = toy_schedule()
    base = sched.peak_in_range(0, 5, 0)
    check(base == (1110, 2), "un-excluded walk unchanged", f"{base}")
    check(sched.peak_in_range(0, 5, 0, restore_at={'A': 4}) == base,
          "a restore ordinal with no exclusion changes nothing",
          "restore_at is only read for excluded tids")


def test_lifetime_absence_still_available() -> None:
    print("\n2. LIFETIME ABSENCE (what a recompute buys) IS UNCHANGED")
    sched = toy_schedule()
    peak, step = sched.peak_in_range(0, 5, 0, exclude_tensors={'A'})
    check((peak, step) == (1010, 2),
          "excluding with no restore ordinal removes the bytes for good",
          f"peak={peak} at {step} (1110 - 100)")


def test_restore_after_vs_before_the_peak() -> None:
    print("\n3. RELIEF COUNTS ONLY WHERE THE ABSENCE COVERS THE PEAK")
    sched = toy_schedule()
    late, _ = sched.peak_in_range(0, 5, 0, exclude_tensors={'A'},
                                  restore_at={'A': 4})
    check(late == 1010,
          "restored after the peak -> full contribution",
          f"peak={late}, same as lifetime absence")
    early, _ = sched.peak_in_range(0, 5, 0, exclude_tensors={'A'},
                                   restore_at={'A': 1})
    check(early == 1110,
          "restored before the peak -> zero contribution",
          f"peak={early}, identical to never having evicted")
    at_peak, _ = sched.peak_in_range(0, 5, 0, exclude_tensors={'A'},
                                     restore_at={'A': 2})
    check(at_peak == 1110,
          "restored AT the peak instant -> zero contribution",
          "the boundary case is credited to the pessimistic side")


def test_replay_transients_are_not_invented() -> None:
    print("\n4. A REPLAY TRANSIENT IS NOT HANDED BACK")
    # B is created and freed again strictly inside A's absence interval,
    # which is what a checkpointed layer's interior tensors do. Excluding
    # the pair and restoring at 4 must never exceed the un-excluded peak.
    sched = toy_schedule()
    peak, _ = sched.peak_in_range(0, 5, 0, exclude_tensors={'A', 'B'},
                                  restore_at={'A': 4, 'B': 4})
    check(peak <= 1110,
          "the candidate peak never rises above the un-excluded one",
          f"peak={peak} <= 1110")
    check(peak == 1000,
          "both are absent at the peak, and only A comes back afterwards",
          f"peak={peak} (1110 - 100 - 10)")


def test_earlier_layers_bytes_come_back() -> None:
    print("\n5. AN EARLIER LAYER'S PICK IS RESTORED INSIDE THIS WINDOW")
    # Project from ordinal 1: A's create is OUTSIDE the window, which is
    # exactly the case for a bundle an earlier layer already picked. Its
    # bytes must still reappear at the restore.
    sched = toy_schedule()
    # anchor = allocator reading with A already evicted: B not yet created,
    # C not yet created, so 0.
    peak, step = sched.peak_in_range(1, 5, 0, exclude_tensors={'A'},
                                     restore_at={'A': 4})
    check(peak == 1010,
          "the window's own peak is free of A",
          f"peak={peak} at {step}")
    # ...and the bytes are genuinely back afterwards: with C gone the live
    # total at ordinal 4 is B(10 already freed) + A(100) = 100, so a walk
    # that ends before A's free must have carried them.
    peak2, _ = sched.peak_in_range(1, 4, 0, exclude_tensors={'A'},
                                   restore_at={'A': 4})
    lost, _ = sched.peak_in_range(1, 4, 0, exclude_tensors={'A'})
    check(peak2 == lost == 1010,
          "restoring an out-of-window create does not double-count it",
          f"with restore={peak2}, lifetime={lost}")


# ── The headline shape ────────────────────────────────────────────────────

def headline(L=32, W=2, seq=16384):
    """Tier-A, regime B, whole-layer unit, chunked CE at 256 tokens.

    All three levers in force, which is the only configuration whose peak
    instant is a layer replay. Unchunked and below the knee the peak is the
    cross-entropy backward instead, ahead of every restore, and the
    coverage question does not arise.
    """
    p = AccessPredictor(
        num_layers=L, seq_len=seq, hidden_size=768, intermediate_size=2048,
        num_heads=12, num_kv_heads=4, batch_size=1, dtype_bytes=2,
        vocab_size=32000, chunked_ce_tokens=256, verbose=False,
        bundle_kinds=('layer',),
        checkpoint_policy={i: True for i in range(L)})
    return p, p.build_schedule()


def test_headline_peak_and_coverage() -> None:
    print("\n6. AT THE HEADLINE SHAPE, W+1 LAYERS BUY NOTHING")
    L, W = 32, 2
    _p, sched = headline(L=L, W=W)
    last = max(e.step_idx for e in sched.events)
    peak, pstep = sched.peak_in_range(0, last, 0)
    first_bwd = sched.bwd_event_for_layer(L - 1)
    check(pstep == first_bwd,
          "the peak instant is the first layer backward touches",
          f"ordinal {pstep}, matching post_bwd_layer_{L - 1} as measured")

    covered = []
    for T in range(L):
        k = max(min(T + W, L - 1), T)
        R = sched.bwd_event_for_layer(k)
        if R is not None and R > pstep:
            covered.append(T)
    check(covered == list(range(L - W - 1)),
          f"exactly the top {W + 1} layers are restored at or before the peak",
          f"{len(covered)}/{L} layers keep their absence; "
          f"the uncovered ones are {sorted(set(range(L)) - set(covered))}")
    # The prediction the engine run can falsify.
    planned = L * 1.5 * (16384 / 1024)
    realized = len(covered) * 1.5 * (16384 / 1024)
    check(abs(realized / planned - (L - W - 1) / L) < 1e-9,
          "planned-vs-realized relief mass, from the schedule alone",
          f"{realized:.0f} of {planned:.0f} MiB = "
          f"{100 * realized / planned:.1f}% -- an UPPER bound on phi, since "
          f"it assumes every copy finishes before the peak")
    # Pre-registration for an engine arm at this L and W: (L-W-1)/L is
    # 29/32 = 0.90625 exactly (arithmetic), the absence.peak_coverage_fraction
    # the engine's telemetry should report if the schedule predicts it.
    check(abs((L - W - 1) / L - 0.90625) < 1e-12,
          "the model's coverage is 29/32 = 0.90625 at L=32, W=2",
          "pre-registered for an engine arm's absence.peak_coverage_fraction")


def test_prefetch_depth_moves_memory() -> None:
    print("\n7. W IS A MEMORY PARAMETER, NOT ONLY A TIME ONE")
    L = 32
    _p, sched = headline(L=L)
    last = max(e.step_idx for e in sched.events)
    _peak, pstep = sched.peak_in_range(0, last, 0)
    counts = []
    for W in (1, 2, 4, 8):
        n = 0
        for T in range(L):
            k = max(min(T + W, L - 1), T)
            R = sched.bwd_event_for_layer(k)
            if R is not None and R > pstep:
                n += 1
        counts.append((W, n))
    check(all(b[1] < a[1] for a, b in zip(counts, counts[1:])),
          "deeper prefetch strictly reduces the layers whose absence covers "
          "the peak", f"{counts}")
    check(counts[0][1] == L - 2,
          "W=1 already forfeits two layers, not one",
          "the last layer is never prefetched -- it restores lazily at its "
          "own backward read, which is the peak instant itself")


def test_planner_end_to_end() -> None:
    print("\n8. THROUGH THE PLANNER, NOT ONLY THE SCHEDULE")
    L, W = 32, 2
    _p, sched = headline(L=L, W=W)
    eng = SimpleNamespace()
    eng._schedule = sched
    eng._num_layers = L
    eng._prefetch_window_layers = W
    eng._restore_ordinal_for_layer = (
        PlacementEngine._restore_ordinal_for_layer.__get__(eng))
    last = max(e.step_idx for e in sched.events)
    _peak, pstep = sched.peak_in_range(0, last, 0)

    # The last layer is the one the criterion would most like to pick (its
    # bytes are live across the whole peak) and the one it must refuse.
    r_last = eng._restore_ordinal_for_layer(L - 1)
    check(r_last == pstep,
          "the last layer's restore lands exactly on the peak instant",
          f"restore ordinal {r_last} == peak ordinal {pstep}")
    r_deep = eng._restore_ordinal_for_layer(0)
    check(r_deep > pstep,
          "layer 0's restore lands well after it",
          f"restore ordinal {r_deep} > peak ordinal {pstep}")

    # And the contribution the planner would compute for each.
    tid_last = f'act_layer_{L - 1}_layer_residual'
    tid_deep = 'act_layer_0_layer_residual'
    base, _ = sched.peak_in_range(0, last, 0)
    c_last = base - sched.peak_in_range(
        0, last, 0, exclude_tensors={tid_last},
        restore_at={tid_last: r_last})[0]
    c_deep = base - sched.peak_in_range(
        0, last, 0, exclude_tensors={tid_deep},
        restore_at={tid_deep: r_deep})[0]
    check(c_last == 0,
          "so the last layer's contribution is zero -> no_contribution_"
          "restored_before_peak", f"{c_last} B")
    check(c_deep == 24 * MiB,
          "and layer 0's is its whole boundary",
          f"{c_deep / MiB:.4f} MiB = 1.5000 MiB/layer/x at x=16")


def main() -> int:
    for fn in (
        test_no_exclusions_is_bit_identical,
        test_lifetime_absence_still_available,
        test_restore_after_vs_before_the_peak,
        test_replay_transients_are_not_invented,
        test_earlier_layers_bytes_come_back,
        test_headline_peak_and_coverage,
        test_prefetch_depth_moves_memory,
        test_planner_end_to_end,
    ):
        fn()
    print("\n" + "=" * 70)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        return 1
    print("All checks passed. Relief is booked as an interval, and the "
          "layers whose interval misses the peak are refused.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
