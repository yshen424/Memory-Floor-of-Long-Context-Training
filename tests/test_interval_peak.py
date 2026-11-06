"""
Per-interval peak attribution smoke test.

CUDA reports only a high-water mark since the last reset, so attributing a peak
to the interval that caused it requires resetting at every snapshot. That reset
would destroy the step-level peak the engine reports as ``observed_peak_mb``.
This test pins the contract that makes both readings coexist.

What this verifies:

  1. Each snapshot's ``interval_peak_bytes`` is the peak of ITS OWN interval,
     so a peak that falls back down is visible instead of being masked by a
     monotone staircase.
  2. ``peak_allocated_bytes`` on every snapshot stays the cumulative step
     high-water mark — bit-identical to a profiler that never resets.
  3. ``step_peak_bytes()`` returns the whole step's peak even though the raw
     CUDA counter has been reset several times inside the step.
  4. On an unprofiled step (``enabled=False``) nothing is reset, the shadow max
     stays 0, and ``step_peak_bytes()`` falls through to the raw counter.
  5. ``_interval_peak_enabled=False`` (the ``--no_interval_peak`` control)
     performs no mid-step reset, and snapshot peaks are the cumulative
     staircase.
  6. The engine's ``_read_observed_peak_and_reset`` returns the same step peak
     with the profiler's shadow max wired in as it returns with no profiler
     resetting the counter.

Usage:
    PYTHONPATH=. python3 tests/test_interval_peak.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import fsdp_profiler
from fsdp_profiler import FSDPMemoryProfiler
from memory_manager.placement_engine import PlacementEngine


class FakeCuda:
    """Minimal stand-in for the CUDA allocator's peak bookkeeping.

    ``allocate(n)`` sets current usage; the high-water mark rises with it and
    only falls when ``reset_peak_memory_stats`` is called, exactly as CUDA
    behaves.
    """

    def __init__(self):
        self.current = 0
        self.peak = 0
        self.resets = 0

    def allocate(self, nbytes: int) -> None:
        self.current = nbytes
        self.peak = max(self.peak, nbytes)

    # ── torch.cuda surface ────────────────────────────────────────────────
    def is_available(self):
        return True

    def synchronize(self):
        pass

    def memory_allocated(self):
        return self.current

    def memory_reserved(self):
        return self.current

    def max_memory_allocated(self):
        return self.peak

    def reset_peak_memory_stats(self):
        self.resets += 1
        self.peak = self.current


_CUDA_ATTRS = (
    'is_available', 'synchronize', 'memory_allocated',
    'memory_reserved', 'max_memory_allocated', 'reset_peak_memory_stats',
)


class patch_cuda:
    """Temporarily redirect the torch.cuda calls the profiler makes."""

    def __init__(self, fake):
        self.fake = fake
        self.saved = {}

    def __enter__(self):
        for attr in _CUDA_ATTRS:
            self.saved[attr] = getattr(torch.cuda, attr)
            setattr(torch.cuda, attr, getattr(self.fake, attr))
        return self.fake

    def __exit__(self, *exc):
        for attr, fn in self.saved.items():
            setattr(torch.cuda, attr, fn)
        return False


def build_fake_profiler(enabled: bool = True, interval_peak: bool = True):
    """A profiler harness carrying only the snapshot path's state."""
    prof = SimpleNamespace()
    prof.enabled = enabled
    prof.snapshots = []
    prof.timeline = []
    prof._start_time = 0.0
    prof._step_peak_running_max = 0
    prof._interval_peak_enabled = interval_peak
    prof._step_snapshot_start = 0
    for name in ('_get_mem', '_now_ms', '_take_snapshot',
                 '_read_interval_peak_and_reset', 'step_peak_bytes',
                 'current_interval_ordinal', 'peak_interval_ordinal'):
        setattr(prof, name, getattr(FSDPMemoryProfiler, name).__get__(prof))
    return prof


MB = 1024 * 1024


def test_interval_peaks_are_per_interval():
    """Case 1 + 2: intervals isolate their own peak; step peak stays cumulative."""
    fake = FakeCuda()
    prof = build_fake_profiler()
    with patch_cuda(fake):
        prof._take_snapshot("step_start")          # interval 0: peak 0
        fake.allocate(900 * MB)                    # interval 1 peak: 900
        fake.allocate(100 * MB)
        prof._take_snapshot("post_fwd_layer_0")
        fake.allocate(300 * MB)                    # interval 2 peak: 300
        fake.allocate(50 * MB)
        prof._take_snapshot("post_fwd_layer_1")
        fake.allocate(1500 * MB)                   # interval 3 peak: 1500
        prof._take_snapshot("post_bwd_layer_1")

    intervals = [s.interval_peak_bytes // MB for s in prof.snapshots]
    assert intervals == [0, 900, 300, 1500], intervals

    # A cumulative counter would have reported 0/900/900/1500 and hidden the
    # 300 MB interval entirely — that masking is what per-interval resets remove.
    cumulative = [s.peak_allocated_bytes // MB for s in prof.snapshots]
    assert cumulative == [0, 900, 900, 1500], cumulative

    # Every snapshot reset once, so the interval measurement is not free of
    # bookkeeping but is free of kernels.
    assert fake.resets == 4, fake.resets
    print("  [1+2] per-interval peaks isolate 300 MB that the staircase hid: "
          f"{intervals} vs cumulative {cumulative}")


def test_step_peak_survives_the_resets():
    """Case 3: the shadow max holds the step peak the raw counter no longer has."""
    fake = FakeCuda()
    prof = build_fake_profiler()
    with patch_cuda(fake):
        prof._take_snapshot("step_start")
        fake.allocate(4400 * MB)                   # the step's true peak
        prof._take_snapshot("post_fwd_layer_0")
        fake.allocate(200 * MB)                    # falls back down
        prof._take_snapshot("step_end")

        raw = torch.cuda.max_memory_allocated()
        step_peak = prof.step_peak_bytes()

    assert raw // MB == 200, raw // MB            # counter forgot the peak
    assert step_peak // MB == 4400, step_peak // MB
    print(f"  [3] raw counter reads {raw // MB} MB after resets; "
          f"step_peak_bytes() still reads {step_peak // MB} MB")


def test_unprofiled_step_falls_through():
    """Case 4: with the profiler disabled nothing resets and the raw counter rules."""
    fake = FakeCuda()
    prof = build_fake_profiler(enabled=False)
    with patch_cuda(fake):
        prof._take_snapshot("step_start")
        fake.allocate(4400 * MB)
        prof._take_snapshot("step_end")
        step_peak = prof.step_peak_bytes()

    assert prof.snapshots == []
    assert fake.resets == 0, fake.resets
    assert prof._step_peak_running_max == 0
    assert step_peak // MB == 4400, step_peak // MB
    print(f"  [4] unprofiled step: 0 resets, shadow max unused, "
          f"step_peak_bytes() = {step_peak // MB} MB from the raw counter")


def test_control_flag_restores_the_staircase():
    """Case 5: --no_interval_peak does no mid-step reset."""
    fake = FakeCuda()
    prof = build_fake_profiler(interval_peak=False)
    with patch_cuda(fake):
        prof._take_snapshot("step_start")
        fake.allocate(900 * MB)
        prof._take_snapshot("post_fwd_layer_0")
        fake.allocate(300 * MB)
        prof._take_snapshot("post_fwd_layer_1")
        step_peak = prof.step_peak_bytes()

    assert fake.resets == 0, fake.resets
    # Without resets an "interval" peak degenerates to the cumulative one.
    intervals = [s.interval_peak_bytes // MB for s in prof.snapshots]
    assert intervals == [0, 900, 900], intervals
    assert step_peak // MB == 900, step_peak // MB
    print(f"  [5] control path: 0 resets, peaks degenerate to {intervals}")


def test_engine_observed_peak_is_unchanged():
    """Case 6: observed_peak_mb reads the same with the profiler resetting."""
    fake = FakeCuda()
    prof = build_fake_profiler()

    eng = SimpleNamespace()
    eng._step_peak_source = None
    eng._read_observed_peak_and_reset = (
        PlacementEngine._read_observed_peak_and_reset.__get__(eng))

    # Baseline: no profiler wired in, no mid-step resets.
    with patch_cuda(fake):
        fake.allocate(4400 * MB)
        fake.allocate(200 * MB)
        before = eng._read_observed_peak_and_reset()

    # Same allocation trace, but the profiler now resets at every snapshot.
    fake2 = FakeCuda()
    eng._step_peak_source = lambda: prof._step_peak_running_max
    with patch_cuda(fake2):
        prof._take_snapshot("step_start")
        fake2.allocate(4400 * MB)
        prof._take_snapshot("post_fwd_layer_0")
        fake2.allocate(200 * MB)
        prof._take_snapshot("step_end")
        after = eng._read_observed_peak_and_reset()

    assert before == after, (before, after)
    assert after // MB == 4400, after // MB
    print(f"  [6] observed peak identical with and without interval resets: "
          f"{before // MB} MB")


def test_snapshot_json_carries_the_field():
    """The envelope has to reach the saved JSON to be usable."""
    fake = FakeCuda()
    prof = build_fake_profiler()
    with patch_cuda(fake):
        fake.allocate(700 * MB)
        prof._take_snapshot("post_fwd_layer_0")

    entry = prof.timeline[-1]
    assert 'interval_peak_mb' in entry, entry
    assert abs(entry['interval_peak_mb'] - 700.0) < 0.01, entry
    assert hasattr(fsdp_profiler.MemorySnapshot, '__dataclass_fields__')
    assert 'interval_peak_bytes' in fsdp_profiler.MemorySnapshot.__dataclass_fields__
    print("  [json] timeline carries interval_peak_mb; snapshot dataclass "
          "carries interval_peak_bytes")


def main():
    print("Per-interval peak attribution")
    test_interval_peaks_are_per_interval()
    test_step_peak_survives_the_resets()
    test_unprofiled_step_falls_through()
    test_control_flag_restores_the_staircase()
    test_engine_observed_peak_is_unchanged()
    test_snapshot_json_carries_the_field()
    print("ALL PASS")


if __name__ == '__main__':
    main()
