"""The instrument's byte schedule against the language-model head and the loss.

At the regime B peak with the whole loss, the binding mass is the logit chain
(language-model head and cross-entropy), which no placement can move. The
instrument's schedule must therefore carry it as non-migratable mass: visible
to the feasibility gate, reachable by no eviction path. With the chunked loss
the chain stops scaling with sequence length and the peak returns to the last
layer's backward.

The targets are over-determined: one byte model, with no per-depth and no
per-vocabulary tuning, is held to the measured four-rank lines.

CHECKS, in the order a failure should be read:
  1. REGRESSION: vocab_size=0 leaves the schedule bit-identical (same event
     count, ordinals and byte map).
  2. ORDINAL INVARIANCE: the term consumes no step. The window helpers walk
     ordinals, not events, so a new ordinal would silently widen the window.
  3. THE SLOPE with the whole loss, against the measured L=64 line (7744
     block U).
  4. THE ABSOLUTE PEAK at L=64, x=4.
  5. VOCABULARY LINEARITY: the peak rises exactly 14 B per token per
     vocabulary entry, the term the chunked loss removes.
  6. SAFETY: the non-migratable slots are visible to the peak walk and
     reachable by no eviction path (no registry key, no 'act_' prefix, no
     read event).
  7. THE CHUNKED LOSS: a fixed chunk SIZE moves the logit chain from the slope
     to the intercept (12 B per chunk token per vocabulary entry, 7938), the peak
     instant leaves the loss's backward for the last layer's backward, and the
     projected slope stays above the measured four-rank lines at L=32, 64 and
     128 (7744, 7761) by one depth-independent constant.

Pure arithmetic; no torch, no CUDA, runs in under a second.

Run:  PYTHONPATH=. python3 tests/test_head_ce_byte_model.py
"""

import sys

from memory_manager.access_predictor import AccessPredictor

MIB = 1024.0 * 1024.0

# Measured four-rank lines, V100, fused-input, --use_cache off, regime B
# (sdpa_mem_efficient, fp16), v=32000, b=1, whole-layer checkpointing, MiB per
# 1024 tokens. Whole loss: 7744 block U (1414.92 + 536.77x, x = 2, 4, 8).
MEASURED_CKPT_SLOPE = {64: 536.77}
# Chunked loss c=256: 7744 blocks T and S (L=32, 64) and 7761 block D1 (L=128).
MEASURED_CHUNKED_SLOPE = {32: 84.3203, 64: 132.3203, 128: 228.3203}

TIER_A = dict(hidden_size=768, intermediate_size=2048,
              num_heads=12, num_kv_heads=4, batch_size=1, dtype_bytes=2,
              materializes_attn_scores=False)

# The schedule carries two families of non-migratable slot: the head/loss
# chain exists only when there is a vocabulary; the per-layer backward
# transient exists at every vocabulary including zero.
HEAD_CE_IDS = ('nonmig_logits', 'nonmig_logit_grad', 'nonmig_final_norm')
BWD_TRANSIENT_PREFIX = 'nonmig_layer_bwd_'


def is_head_ce(tensor_id: str) -> bool:
    return tensor_id in HEAD_CE_IDS


failures: list[str] = []


def check(name: str, ok: bool, detail: str = '') -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ''))
    if not ok:
        failures.append(name)


def build(num_layers: int, seq_len: int, vocab_size: int, chunk: int = 0):
    p = AccessPredictor(num_layers=num_layers, seq_len=seq_len, verbose=False,
                        vocab_size=vocab_size, chunked_ce_tokens=chunk,
                        bundle_kinds=('layer',),
                        checkpoint_policy={L: True for L in range(num_layers)},
                        **TIER_A)
    return p.build_schedule()


def peak_at(num_layers: int, seq_len: int, vocab_size: int,
            chunk: int = 0) -> tuple[float, int]:
    sched = build(num_layers, seq_len, vocab_size, chunk)
    last = max(ev.step_idx for ev in sched.events)
    peak_bytes, peak_step = sched.peak_in_range(0, last)
    return peak_bytes / MIB, peak_step


def peak_mib(num_layers: int, seq_len: int, vocab_size: int,
             chunk: int = 0) -> float:
    return peak_at(num_layers, seq_len, vocab_size, chunk)[0]


def slope_mib_per_x(num_layers: int, vocab_size: int, chunk: int = 0,
                    lo_x: int = 16, hi_x: int = 24) -> float:
    lo = peak_mib(num_layers, lo_x * 1024, vocab_size, chunk)
    hi = peak_mib(num_layers, hi_x * 1024, vocab_size, chunk)
    return (hi - lo) / (hi_x - lo_x)


# ── 1. Regression: vocab_size=0 changes nothing ───────────────────────────
print("\n1. REGRESSION -- vocab_size=0 leaves the schedule bit-identical")
off = build(32, 4096, 0)
sig_off = [(e.step_idx, e.phase, e.layer_idx, e.tensor_id, e.access_type)
           for e in off.events]
on = build(32, 4096, 32000)
sig_on = [(e.step_idx, e.phase, e.layer_idx, e.tensor_id, e.access_type)
          for e in on.events]
check("vocab_size=0 emits no head/CE event",
      not any(is_head_ce(e.tensor_id) for e in off.events))
check("vocab_size=0 adds no byte-map entry",
      not any(is_head_ce(k) for k in off.bytes_by_tensor))
check("every pre-existing event survives unchanged, in order",
      [s for s in sig_on if not is_head_ce(s[3])] == sig_off)
check("every pre-existing byte-map entry survives unchanged",
      {k: v for k, v in on.bytes_by_tensor.items()
       if not is_head_ce(k)} == dict(off.bytes_by_tensor))

# ── 2. Ordinal invariance ─────────────────────────────────────────────────
print("\n2. ORDINAL INVARIANCE -- the term consumes no step")
max_off = max(e.step_idx for e in off.events)
max_on = max(e.step_idx for e in on.events)
check("max ordinal unchanged", max_off == max_on, f"{max_off} vs {max_on}")
check("ordinal count unchanged",
      len({e.step_idx for e in off.events}) == len({e.step_idx for e in on.events}))
nonmig_steps = sorted({e.step_idx for e in on.events
                       if e.tensor_id.startswith('nonmig_')})
# Compared against the ordinals REAL events occupy, not against the vocab=0
# schedule: the backward transient is present at vocab=0 too, so the latter
# would pass by construction.
real_steps = {e.step_idx for e in on.events
              if not e.tensor_id.startswith('nonmig_')}
check("every non-migratable event shares an existing anchor's ordinal",
      all(s in real_steps for s in nonmig_steps),
      f"{len(nonmig_steps)} ordinals, {min(nonmig_steps)}..{max(nonmig_steps)}")
# The planner's own window anchors must not move.
check("first fwd create per layer unchanged",
      all(on.fwd_create_event_for_layer(L) == off.fwd_create_event_for_layer(L)
          for L in range(32)))
check("first bwd read per layer unchanged",
      all(on.bwd_event_for_layer(L) == off.bwd_event_for_layer(L)
          for L in range(32)))

# ── 3. The slope with the whole loss, against 7744 block U ────────────────
print("\n3. SLOPE -- whole loss, against the measured four-rank line (7744 U)")
residuals = {}
for L, measured in sorted(MEASURED_CKPT_SLOPE.items()):
    pts = [(s / 1024.0, peak_mib(L, s, 32000)) for s in (1024, 2048, 4096)]
    # Three collinear points: slope from the ends, then confirm the middle.
    slope = (pts[-1][1] - pts[0][1]) / (pts[-1][0] - pts[0][0])
    intercept = pts[0][1] - slope * pts[0][0]
    lin_err = max(abs(y - (slope * x + intercept)) for x, y in pts)
    residuals[L] = measured - slope
    check(f"L={L}: schedule slope is linear in tokens",
          lin_err < 1e-9, f"max deviation {lin_err:.2e} MiB")
    check(f"L={L}: slope {slope:.2f} vs measured {measured:.2f}",
          abs(residuals[L]) < 0.30, f"under-projects by {residuals[L]:.2f} MiB/x")
check("the residual is positive and under 0.5 MiB/x",
      0 < residuals[64] < 0.5,
      "one unmodelled non-layer term (rotary tables / embedding output)")

# ── 4. The absolute peak at L=64, x=4 ─────────────────────────────────────
print("\n4. ABSOLUTE PEAK -- L=64, s=4096, v=32000")
got = peak_mib(64, 4096, 32000)
boundaries = 64 * 1.50 * 4          # 1.50 MiB/layer/x: one fp16 boundary
head_ce = 4096 * 32000 * 14 / MIB   # 14 B/token/vocab at the peak instant
final_norm = 3.00 * 4               # one fp32 hidden-width plane, 4 B x 768 per token
want = boundaries + head_ce + final_norm
check(f"peak == {want:.2f} MiB", abs(got - want) < 1e-6, f"got {got:.5f}")
check("boundaries + head/CE + final norm is the whole projection",
      abs(want - (384.00 + 1750.00 + 12.00)) < 1e-6)
fitted = MEASURED_CKPT_SLOPE[64] * 4
check(f"within 1.5 MiB of 7744 block U's slope term at x=4, {fitted:.2f} MiB",
      abs(got - fitted) < 1.5, f"short by {fitted - got:.2f} MiB")
check("the head/CE term dominates -- this is the span's denominator",
      head_ce / want > 0.80, f"{100 * head_ce / want:.1f}% of the peak")

# ── 5. Vocabulary linearity ───────────────────────────────────────────────
print("\n5. VOCABULARY -- 14 B per token per vocab entry at the peak instant")
base = peak_mib(64, 4096, 8000)
for v in (16000, 32000, 64000, 128000):
    got_v = peak_mib(64, 4096, v)
    want_v = base + (v - 8000) * 4096 * 14 / MIB
    check(f"v={v}: peak scales exactly", abs(got_v - want_v) < 1e-6,
          f"{got_v:.2f} vs {want_v:.2f} MiB")
check("depth does not move the head/CE term",
      abs((peak_mib(128, 4096, 32000) - peak_mib(64, 4096, 32000)) - 384.00)
      < 1e-6, "64 extra layers buy exactly 64 x 1.50 x 4 MiB of boundary")

# ── 6. Safety: visible to the gate, invisible to every evict path ─────────
print("\n6. SAFETY -- the gate can see this mass and can never evict it")
head_ids = {e.tensor_id for e in on.events if is_head_ce(e.tensor_id)}
bwd_ids = {e.tensor_id for e in on.events
           if e.tensor_id.startswith(BWD_TRANSIENT_PREFIX)}
nonmig_ids = head_ids | bwd_ids
check("three head/CE slots and one backward transient per layer",
      len(head_ids) == 3 and len(bwd_ids) == 32,
      f"{len(head_ids)} head/CE, {len(bwd_ids)} transient")
check("none carries the 'act_' prefix the prefetch path filters on",
      not any(t.startswith('act_') for t in nonmig_ids))
check("none is a bundle id",
      not any(t.startswith('bundle_') for t in nonmig_ids))
check("no read event -- the prefetch lookahead only walks reads",
      not any(e.access_type == 'read' for e in on.events
              if e.tensor_id.startswith('nonmig_')))
ce_free = max(e.step_idx for e in on.events
              if is_head_ce(e.tensor_id) and e.access_type == 'free')
check("the head/CE bytes are released before backward reaches any layer",
      ce_free < on.bwd_event_for_layer(31),
      f"last head/CE free at {ce_free}, layer 31's backward at "
      f"{on.bwd_event_for_layer(31)}")
# The transient's free shares the grad_layer free ordinal, so at most ONE
# layer's transient is ever live: the term is a constant, not L x constant.
# Getting this wrong grows the projection a second per-layer slope.
live = max_live = 0
for e in sorted(on.events, key=lambda ev: ev.step_idx):
    if not e.tensor_id.startswith(BWD_TRANSIENT_PREFIX):
        continue
    live += 1 if e.access_type == 'create' else -1
    max_live = max(max_live, live)
check("at most one layer's backward transient is ever live",
      max_live == 1 and live == 0,
      f"max co-resident {max_live}, balance {live}")

# ── 7. The chunked loss ───────────────────────────────────────────────────
print("\n7. CHUNKED LOSS -- fixed chunk size (L=32, v=32000, regime B)")

# 7a. chunk=0 is the whole loss; structure must be untouched either way.
ch = build(32, 4096, 32000, 256)
sig_ch = [(e.step_idx, e.phase, e.layer_idx, e.tensor_id, e.access_type)
          for e in ch.events]
check("7a chunk=0 reproduces the whole-loss projection",
      peak_mib(64, 4096, 32000, 0) == 2146.00,
      f"{peak_mib(64, 4096, 32000, 0):.5f} MiB at L=64, x=4")
check("7a chunking moves no event and consumes no ordinal",
      sig_ch == sig_on and max(e.step_idx for e in ch.events) == max_on)
check("7a the three slots survive chunking",
      {e.tensor_id for e in ch.events if is_head_ce(e.tensor_id)} == head_ids)

# 7b. The per-chunk constant: 375.00 MiB per 1024 chunk tokens at v=32000,
# which is 12 B per token per vocabulary entry -- the forward-resident chain,
# not the whole loss's 14 B.
step = peak_mib(32, 8192, 32000, 2048) - peak_mib(32, 8192, 32000, 1024)
check("7b chunk-size step is 375.00 MiB per 1024 chunk tokens",
      abs(step - 375.00) < 1e-9, f"{step:.4f} MiB = "
      f"{step * MIB / (1024 * 32000):.4f} B/token/vocab")
# Measured at four ranks (7938 block K, L=32, x=8): each chunk-size step is
# exactly 12 B per chunk token per vocabulary entry while the chunk is shorter
# than the sequence.
MEASURED_CHUNK_STEP = {(1024, 2048): 375.0000, (2048, 4096): 749.9995}
for (lo_c, hi_c), meas in MEASURED_CHUNK_STEP.items():
    mdl = peak_mib(32, 8192, 32000, hi_c) - peak_mib(32, 8192, 32000, lo_c)
    check(f"7b c={lo_c}->{hi_c}: schedule step {mdl:.4f} vs measured {meas:.4f} MiB",
          abs(mdl - meas) < 0.05, f"{mdl - meas:+.4f} MiB")
check("7b one chunk covering the sequence is NOT the whole loss",
      peak_mib(32, 8192, 32000, 8192) < peak_mib(32, 8192, 32000, 0),
      "the chunked loss checkpoints every chunk; the whole loss checkpoints "
      "none, so the chain is replayed rather than held beside its gradient")

# 7c. THE FORM CHANGE. Taking the term out of the peak's slope leaves the
# loss's instant BELOW another one, so the peak relocates and its slope is set
# by a different term. The peak is the max of two lines, which is why the chunk
# size is a system knob rather than a constant.
sl_unchunked = slope_mib_per_x(32, 32000, 0)
sl_256 = slope_mib_per_x(32, 32000, 256)
sl_128 = slope_mib_per_x(32, 32000, 128)
# c=4096 keeps the 16..24 window on the chunk line alone; c=2048 crosses the
# floor near x=22 and would measure a blend of the two lines.
sl_chunkline = slope_mib_per_x(32, 32000, 4096)
check("7c the chunk-resident line keeps only boundaries and the final norm",
      abs(sl_chunkline - (32 * 1.50 + 3.00)) < 1e-9,
      f"{sl_chunkline:.2f} MiB/x, and 437.50 of head/CE slope is gone")
check("7c removing it drops the loss's instant BELOW the layer-interior floor",
      sl_chunkline < sl_256,
      f"chunk line {sl_chunkline:.2f} < floor {sl_256:.4f} MiB/x -- the peak "
      "has to move (7d)")
check("7c below the knee the slope no longer depends on chunk size",
      abs(sl_256 - sl_128) < 1e-9, f"c=128 and c=256 both {sl_256:.4f} MiB/x")
check("7c the floor is the steeper line, so chunk size stops mattering at "
      "large x",
      peak_at(32, 24 * 1024, 32000, 1024)[1]
      == peak_at(32, 24 * 1024, 32000, 256)[1],
      "c=1024 rides the chunk line at x=16 and the floor at x=24 -- the two "
      "lines cross, and the crossover is the knee")

# 7d. THE PEAK INSTANT MOVES. The byte model is told nothing about where the
# peak was measured; it only knows the chain got smaller.
ce_step = peak_at(32, 16 * 1024, 32000, 0)[1]
ch_step = peak_at(32, 16 * 1024, 32000, 256)[1]
ce_ids = {e.tensor_id for e in build(32, 16 * 1024, 32000, 0).events
          if e.step_idx == ce_step}
ch_evs = [e for e in ch.events if e.step_idx == ch_step]
ch_ids = {e.tensor_id for e in ch_evs}
check("7d whole loss: the peak instant is the loss's backward",
      any(t.startswith('nonmig_logit') for t in ce_ids), f"ordinal {ce_step}")
check("7d chunked: the peak instant moves off the loss's backward",
      ch_step != ce_step and not any(is_head_ce(t) for t in ch_ids),
      f"ordinal {ce_step} -> {ch_step}")
check("7d and it lands on the LAST layer's backward, where 7744 block T "
      "measured it",
      all(e.phase == 'bwd' and e.layer_idx == 31 for e in ch_evs)
      and f'{BWD_TRANSIENT_PREFIX}31' in ch_ids,
      f"ordinal {ch_step}: {sorted(ch_ids)} -- measured post_bwd_layer_31")
l64_step = peak_at(64, 16 * 1024, 32000, 256)[1]
l64_evs = [e for e in build(64, 4096, 32000, 256).events
           if e.step_idx == l64_step]
check("7d the same instant at L=64 -- layer 63's backward (7744 block S)",
      bool(l64_evs) and all(e.phase == 'bwd' and e.layer_idx == 63
                            for e in l64_evs),
      f"ordinal {l64_step}: {sorted(e.tensor_id for e in l64_evs)} -- "
      "measured post_bwd_layer_63")

# 7e. THE LAYER BACKWARD'S TRANSIENT, COUNTED. A backward also allocates the
# gradients it is computing, co-resident with the saves that produce them.
# tools/layer_bwd_livebytes.py profiles a real backward and counts seven
# co-resident intermediate-width planes against the four the model carries as
# saves, plus two hidden-width planes for the residual branch -- 3 + 2, which
# is 15.00 MiB/x at tier-A width and dtype. Not a fit: the alternatives
# nearest in magnitude (3+1 = 13.50, 2+2 = 11.00) both leave the projection
# SHORT, which is the out-of-memory direction a gate must never take.
over = {L: slope_mib_per_x(L, 32000, 256) - m
        for L, m in MEASURED_CHUNKED_SLOPE.items()}
check("7e the chunked projection OVER-shoots at every depth -- the safe "
      "direction",
      all(v > 0 for v in over.values()),
      " and ".join(f"L={L}: {slope_mib_per_x(L, 32000, 256):.4f} vs measured "
                   f"{MEASURED_CHUNKED_SLOPE[L]:.4f}, +{over[L]:.4f} MiB/x "
                   f"({100 * over[L] / MEASURED_CHUNKED_SLOPE[L]:.2f}%)"
                   for L in sorted(over)))
# The error has to be DEPTH-INDEPENDENT or the term was mis-attributed: a
# per-layer error would grow with L and turn the span back into a constant.
check("7e the remaining error is depth-independent -- a constant, not a "
      "per-layer term",
      max(over.values()) - min(over.values()) < 1e-3,
      ", ".join(f"L={L} {over[L]:.6f}" for L in sorted(over)) + " MiB/x")
check("7e the whole-loss projection is untouched -- the transient only "
      "matters at the instant the chunked loss exposes",
      all(abs(slope_mib_per_x(L, 32000, 0) - s) < 1e-9
          for L, s in ((32, 488.50), (64, 536.50), (128, 632.50))),
      "488.50 / 536.50 / 632.50 MiB/x")
# The knee is where the chunk line meets the floor, so the floor sets it.
check("7e the knee at x=8 lies between 512 and 1024 chunk tokens",
      peak_mib(32, 8192, 32000, 512) == peak_mib(32, 8192, 32000, 256)
      and peak_mib(32, 8192, 32000, 1024) > peak_mib(32, 8192, 32000, 512),
      "below it the peak is the floor and independent of c")

print("\n" + "=" * 72)
if failures:
    print(f"FAILED ({len(failures)}): " + "; ".join(failures))
    sys.exit(1)
print("All checks passed. The projection includes the head and the loss.")
