"""CPU test for the MLP's saved-plane count.

What recompute physically frees per layer is the retaining per-module arm's
peak minus its recompute twin's, per layer: 22.0468 MiB per 1024 tokens at
the tier-A width (7941, x = 2 and 4). A three-plane MLP model books 18.047.
The 4.000 MiB gap is one full intermediate-width fp16 plane per layer, a 22%
under-booking of relief.

Direction matters. Under-booking is the safe direction for a feasibility gate,
but it makes the planner buy more migration than it needs — and migration is
the expensive thing, so the error costs step time.

The reason: HuggingFace LlamaMLP is `down_proj(act_fn(gate_proj(x)) *
up_proj(x))` and autograd saves FOUR distinct intermediate-width storages from
it, not three. Section 1 below proves that against the installed Transformers
rather than asserting it.

What this verifies:

  1. The real LlamaMLP, under saved_tensors_hooks, owns four distinct
     intermediate-width storages across nine saves.
  2. The predictor books four planes per mlp bundle, in both regimes.
  3. The per-layer total matches the measured physical relief; the
     counterfactual 3-plane model misses it by exactly one plane.
  4. Schedule ordinals and event lists do not depend on the gate slot's plane
     count — the reason the fourth plane sits on the gate slot instead of in
     an added component.
  5. The profiler's independent copy of the formula agrees with the
     predictor's, and the registry records sum to the group total.
  6. An mlp bundle whose observed owned bytes come in at three planes leaves
     the four-plane model standing and does not trip the over-book alarm.

Usage:
    PYTHONPATH=. python3 tests/test_swiglu_plane_count.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from memory_manager.access_predictor import AccessPredictor
from memory_manager.placement_engine import PlacementEngine

MB = 1024 * 1024
_failures = []


def check(cond, label):
    if cond:
        print(f"[PASS] {label}")
    else:
        print(f"[FAIL] {label}")
        _failures.append(label)


AD768 = dict(
    num_layers=32,
    hidden_size=768,
    intermediate_size=2048,
    num_heads=12,
    num_kv_heads=4,
    batch_size=1,
    dtype_bytes=2,
)
MLP_COMPONENTS = ('swiglu_gate', 'swiglu_up', 'swiglu_product')
ATTN_COMPONENTS = ('q', 'k', 'v', 'attn_scores')


# ── 1. what autograd actually saves, measured ──────────────────────────────
#
# Run the installed LlamaMLP and classify every save the way pack_fn does:
# deduplicate by storage data_ptr, then subtract the two foreign classes the
# engine subtracts (the module's weight state and the bundle's input). Tiny
# shapes — this is a structural count, not a size measurement.
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP

_H, _I, _S = 8, 16, 4
_cfg = LlamaConfig(hidden_size=_H, intermediate_size=_I, num_hidden_layers=1,
                   num_attention_heads=2, num_key_value_heads=1, vocab_size=32,
                   mlp_bias=False)
_mlp = LlamaMLP(_cfg)
_x = torch.randn(1, _S, _H, requires_grad=True)

_saves = []


def _pack(t):
    st = t.untyped_storage()
    _saves.append((st.data_ptr(), st.nbytes()))
    return t


with torch.autograd.graph.saved_tensors_hooks(_pack, lambda t: t):
    _mlp(_x)

_foreign = PlacementEngine._module_storage_ptrs(_mlp)
PlacementEngine._collect_storage_ptrs((_x,), _foreign)

_seen, _owned = set(), []
for _ptr, _nbytes in _saves:
    if _ptr in _seen:
        continue
    _seen.add(_ptr)
    if _ptr not in _foreign:
        _owned.append(_nbytes)

_plane = _S * _I * _x.element_size()
check(len(_saves) == 9,
      f"LlamaMLP produces 9 saves ({len(_saves)}) — the members=9 the engine's "
      f"run JSONs report.")
check(len(_owned) == AccessPredictor._SWIGLU_SAVED_PLANES,
      f"...backed by {len(_owned)} owned storages, and the model books "
      f"{AccessPredictor._SWIGLU_SAVED_PLANES}.")
check(all(n == _plane for n in _owned),
      f"every owned storage is one intermediate-width plane ({_plane} B): "
      f"{_owned}.")


# ── 2. the predictor books four planes, in both regimes ────────────────────
for materializes in (True, False):
    pred = AccessPredictor(verbose=False, seq_len=6144,
                           materializes_attn_scores=materializes, **AD768)
    mlp = sum(pred._activation_bytes(c) for c in MLP_COMPONENTS)
    check(mlp == 4 * (1 * 6144 * 2048 * 2),
          f"mlp bundle is 96 MiB at s=6144 (materializes={materializes}).")

# The extra plane is on the gate slot, and only there.
pred = AccessPredictor(verbose=False, seq_len=6144,
                       materializes_attn_scores=False, **AD768)
_one = 1 * 6144 * 2048 * 2
check(pred._activation_bytes('swiglu_gate') == 2 * _one,
      "the gate slot carries two planes (gate_proj output + activation "
      "output, two distinct storages).")
check(pred._activation_bytes('swiglu_up') == _one
      and pred._activation_bytes('swiglu_product') == _one,
      "up and product stay single tensors.")


# ── 3. against the measured physical relief ────────────────────────────────
#
# The load-bearing check. 22.0468 MiB per 1024 tokens per layer is what the
# retaining per-module arm minus its recompute twin actually freed (7941:
# 2821.9844 MiB at x=2 and 5643.9688 at x=4 over 64 layers), and the
# instrument's registry reads 22.0469 owned on all 128 bundles of the same arms.
MEASURED_MIB_PER_1K_PER_LAYER = 22.0468

pred = AccessPredictor(verbose=False, seq_len=1024,
                       materializes_attn_scores=False, **AD768)
modelled = sum(pred._activation_bytes(c)
               for c in ATTN_COMPONENTS + MLP_COMPONENTS) / MB
old_modelled = modelled - (1024 * 2048 * 2) / MB

check(abs(modelled - MEASURED_MIB_PER_1K_PER_LAYER) < 0.01,
      f"regime B per-layer relief: model {modelled:.3f} vs measured "
      f"{MEASURED_MIB_PER_1K_PER_LAYER} MiB per 1024 tokens.")
check(abs(old_modelled - MEASURED_MIB_PER_1K_PER_LAYER) > 3.9,
      f"the counterfactual 3-plane model books {old_modelled:.3f} — short by "
      f"{MEASURED_MIB_PER_1K_PER_LAYER - old_modelled:.3f} MiB, one plane.")
check(round((modelled - old_modelled) * 1024) == 4096,
      "the 3-vs-4-plane difference is exactly one plane and nothing else moved.")


# ── 4. ordinals are untouched ──────────────────────────────────────────────
# Carrying the fourth plane on an existing slot rather than adding an eighth
# component is the only reason this holds: EVENTS_PER_FWD_LAYER is 1 param
# read + 7 activation creates, and the window ledger, the residual anchor and
# every ordinal helper key off it.
check(len(AccessPredictor._ACTIVATION_COMPONENTS) == 7,
      "still 7 activation components per layer.")
check(PlacementEngine.EVENTS_PER_FWD_LAYER == 8
      and PlacementEngine.EVENTS_PER_BWD_LAYER == 16,
      "EVENTS_PER_{FWD,BWD}_LAYER are unchanged.")

def build(gate_planes):
    """Build a schedule with the gate slot forced to a given plane count.

    gate_planes=1 gives the three-plane model, so the two schedules below are
    a genuine comparison rather than a restatement of the same code.
    """
    saved = AccessPredictor._SWIGLU_GATE_PLANES
    AccessPredictor._SWIGLU_GATE_PLANES = gate_planes
    try:
        return AccessPredictor(verbose=False, seq_len=2048,
                               materializes_attn_scores=False,
                               **AD768).build_schedule()
    finally:
        AccessPredictor._SWIGLU_GATE_PLANES = saved


sched, sched_old = build(2), build(1)
ev = lambda s: [(e.step_idx, e.phase, e.layer_idx, e.tensor_id, e.access_type)
                for e in s.events]
check(ev(sched) == ev(sched_old),
      f"the event list is identical to the three-plane model's "
      f"({len(sched.events)} events) — ordinals did not move.")
check(sched._first_fwd_create_by_layer == sched_old._first_fwd_create_by_layer
      and sched._first_bwd_read_by_layer == sched_old._first_bwd_read_by_layer,
      "the projection window's anchors are unchanged.")
check(all(sched.bytes_by_tensor[k] == sched_old.bytes_by_tensor[k]
          for k in sched.bytes_by_tensor if not k.endswith('swiglu_gate')),
      "every slot except swiglu_gate is byte-identical to the three-plane model.")
check(sched.bytes_by_tensor['act_layer_0_swiglu_gate']
      == 2 * sched.bytes_by_tensor['act_layer_0_swiglu_up'],
      "the byte map carries the fourth plane; the keys are unchanged.")


# ── 5. the profiler's independent copy agrees ──────────────────────────────
# Two copies of one formula can drift apart silently. The constant is
# shared; this check keeps it that way.
from fsdp_profiler import FSDPMemoryProfiler


def profiler_sizes(seq_len=6144, dtype_size=2):
    prof = SimpleNamespace()
    prof.materializes_attn_scores = False
    prof._cfg_hidden = AD768['hidden_size']
    prof._cfg_intermediate = AD768['intermediate_size']
    prof._cfg_num_heads = AD768['num_heads']
    prof._cfg_num_kv_heads = AD768['num_kv_heads']
    prof._cfg_head_dim = AD768['hidden_size'] // AD768['num_heads']
    return FSDPMemoryProfiler._activation_sizes_for_batch(
        prof, 1, seq_len, dtype_size)


sizes = profiler_sizes()
pred = AccessPredictor(verbose=False, seq_len=6144,
                       materializes_attn_scores=False, **AD768)
check(sizes['swiglu_group']
      == sum(pred._activation_bytes(c) for c in MLP_COMPONENTS),
      f"profiler and predictor agree on the mlp group "
      f"({sizes['swiglu_group'] / MB:.0f} MiB).")
check(sizes['swiglu_group']
      == sizes['swiglu_single'] * (AccessPredictor._SWIGLU_GATE_PLANES + 2),
      "the three registry records sum to the group total (gate carries two).")
check(sizes['swiglu_group'] == 4 * (1 * 6144 * 2048 * 2),
      "the mlp group is 96 MiB at s=6144 (a three-plane count gives 72 MiB).")


# ── 6. the observation promotion leaves the four-plane model standing ──────
#
# The engine's observed-relief path reports THREE planes owned for an mlp
# bundle on GPU, so the promotion sees an observation below the analytic
# model. That must neither walk the model back nor trip the over-book alarm,
# whose threshold is 0.5x.
def promote(observed_owned_planes):
    pred = AccessPredictor(verbose=False, seq_len=6144,
                           materializes_attn_scores=False, **AD768)
    eng = PlacementEngine.__new__(PlacementEngine)
    eng._observed_relief = True
    eng._bundle_unit = 'per_module'
    eng._predictor = pred
    eng._schedule = pred.build_schedule()
    eng._training_step_count = 1
    eng._obs_bundle_owned = {}
    eng._obs_bundle_owned_cur = {
        (0, 'mlp'): observed_owned_planes * (1 * 6144 * 2048 * 2)}
    eng._overbooked_bundles = {}
    eng._overbook_warned = False
    # No drop in this fixture, so the observation stays trusted and the
    # recompute-drop gate does not engage.
    eng._obs_bundle_dropped_cur = set()
    eng._obs_declined_bundles = set()
    eng._promote_observed_relief()
    return eng


eng = promote(3)
check(eng._schedule.bytes_by_tensor['act_layer_0_mlp_residual'] == 0,
      "a three-plane observation writes a zero residual, not a negative one.")
check(sum(eng._schedule.bytes_by_tensor[f'act_layer_0_{c}']
          for c in MLP_COMPONENTS) == 4 * (1 * 6144 * 2048 * 2),
      "...and the four-plane model still stands after the promotion.")
check(not eng._overbooked_bundles,
      "the over-book alarm stays silent at 4-vs-3 planes (1.33x, "
      "threshold 2x).")

eng = promote(5)
check(eng._schedule.bytes_by_tensor['act_layer_0_mlp_residual']
      == 1 * 6144 * 2048 * 2,
      "an observation ABOVE the model still lands in the residual slot, so "
      "the four-plane count does not disable the correction path.")


print()
if _failures:
    print(f"{len(_failures)} FAILURE(S):")
    for f in _failures:
        print(f"  - {f}")
    sys.exit(1)
print("all checks passed.")
