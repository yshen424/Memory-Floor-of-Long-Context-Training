"""What is live at the peak instant of a checkpointed layer's backward?

WHY THIS EXISTS. The schedule's projection of the layer stack is short:
`1.50*L + 22.047` MiB/x against a measured `1.50*L + 36.320`. The four-rank static
lines at L=32, 64 and 128 (7744, 7761) show the 14.27 MiB/x gap is
DEPTH-INDEPENDENT, so it is one layer's worth of mass the byte model does not
carry --- but depths say nothing about WHICH mass. Fitting a constant to close it is the vacuous fit
`tests/test_head_ce_byte_model.py` exists to refuse.

WHAT IT MEASURES. The byte model counts saved activations only. A backward
pass also allocates the gradients it is computing, and those are co-resident
with the saves that produced them. This profiles `backward()` alone, so
everything the forward left behind arrives as PREEXISTING and every gradient
appears as a CREATE, and reports the live set at the peak bucketed by width
in elements per token --- which is what names a tensor as hidden-width,
intermediate-width or kv-width regardless of dtype or device.

WHY CPU IS ENOUGH. Which tensors are simultaneously live is a property of the
autograd graph, not of the allocator or the kernel. Widths are read per token
and priced at fp16 afterwards, so running fp32 on CPU costs nothing. The one
thing it cannot see is the attention kernel's own saves, which differ between
regimes --- those are already modelled (`_efficient_attn_slot_bytes`).

Run:  PYTHONPATH=. python3 tools/layer_bwd_livebytes.py
"""

import collections
import sys

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl, checkpoint_wrapper)
from torch.profiler import ProfilerActivity, profile
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel

HIDDEN = 768
INTERMEDIATE = 2048
NUM_HEADS = 12
NUM_KV_HEADS = 4
HEAD_DIM = HIDDEN // NUM_HEADS
KV_DIM = NUM_KV_HEADS * HEAD_DIM
NUM_LAYERS = 4
ELEM = 4  # fp32 on CPU; widths are read per token, so dtype cancels


def build(seq_len: int):
    cfg = LlamaConfig(
        vocab_size=32, hidden_size=HIDDEN, intermediate_size=INTERMEDIATE,
        num_hidden_layers=NUM_LAYERS, num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS, max_position_embeddings=seq_len + 8,
        use_cache=False, attn_implementation='sdpa')
    model = LlamaModel(cfg)
    for i, layer in enumerate(model.layers):
        model.layers[i] = checkpoint_wrapper(
            layer, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
    return model


# Parameters, their .grad and optimizer state are anchored through the
# allocator's live bytes, not the byte map, so they are base and not slope.
# Everything else in a backward is either a save or a gradient transient,
# which is exactly the quantity under test.
_BASE_CATEGORIES = ('PARAMETER', 'GRADIENT', 'OPTIMIZER_STATE')


def live_curve(seq_len: int):
    """Return the activation-side live set at the peak of `backward()`."""
    model = build(seq_len)
    emb = torch.randn(1, seq_len, HIDDEN, requires_grad=True)
    out = model(inputs_embeds=emb).last_hidden_state
    loss = out.pow(2).sum()

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True,
                 record_shapes=True, with_stack=True) as prof:
        loss.backward()

    mem = prof._memory_profile()
    live: dict = {}
    total = 0
    peak = 0
    peak_live: dict = {}
    for _ts, action, (tkey, version), nbytes in mem.timeline:
        cat = mem._categories.get(tkey, version)
        if cat is not None and cat.name in _BASE_CATEGORIES:
            continue
        name = action.name
        if name in ('PREEXISTING', 'CREATE'):
            live[tkey] = (nbytes, cat.name if cat else 'UNCATEGORIZED')
            total += nbytes
        elif name == 'DESTROY':
            total -= live.pop(tkey, (0, ''))[0]
        else:  # INCREMENT_VERSION -- same storage, no allocation change
            continue
        if total > peak:
            peak = total
            peak_live = dict(live)
    return peak, peak_live


def report(seq_len: int):
    peak, peak_live = live_curve(seq_len)
    print(f"\nseq_len={seq_len}  activation-side peak live = "
          f"{peak / 1048576:.3f} MiB ({len(peak_live)} tensors)")
    # Bucket by width in elements per token. A tensor whose element count is
    # not a multiple of the token count is not a per-token plane at all.
    buckets: collections.Counter = collections.Counter()
    other = 0
    for nb, cat in peak_live.values():
        elems = nb // ELEM
        if elems and elems % seq_len == 0:
            buckets[(elems // seq_len, cat)] += 1
        else:
            other += nb
    names = {HIDDEN: 'hidden', INTERMEDIATE: 'intermediate', KV_DIM: 'kv',
             NUM_HEADS: 'per-head'}
    print(f"  {'width':>8} {'planes':>7}  {'MiB@fp16/1024tok':>17}  "
          f"{'category':<17} name")
    fp16_total = 0.0
    for (width, cat), count in sorted(buckets.items(), reverse=True):
        per_x = width * 2 * count * 1024 / 1048576
        fp16_total += per_x
        print(f"  {width:>8} {count:>7}  {per_x:>17.4f}  {cat:<17} "
              f"{names.get(width, '')}")
    print(f"  non-plane remainder: {other / 1048576:.3f} MiB")
    print(f"  TOTAL if priced at fp16: {fp16_total:.4f} MiB per 1024 tokens")
    return peak, collections.Counter({k[0]: v for k, v in buckets.items()})


def isolate_mlp(seq_len: int):
    """Max co-resident intermediate-width planes across an MLP backward.

    Isolated because this is the one part of the layer whose plane count is
    kernel- and dtype-independent: `down(silu(gate(x)) * up(x))` is the same
    graph everywhere. The count includes the free schedule, since a save is
    dropped as soon as its op's backward consumes it.
    """
    from transformers.models.llama.modeling_llama import LlamaMLP
    cfg = LlamaConfig(hidden_size=HIDDEN, intermediate_size=INTERMEDIATE,
                      mlp_bias=False)
    mlp = LlamaMLP(cfg)
    x = torch.randn(1, seq_len, HIDDEN, requires_grad=True)
    y = mlp(x)
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True,
                 record_shapes=True, with_stack=True) as prof:
        y.backward(torch.ones_like(y))

    mem = prof._memory_profile()
    plane = INTERMEDIATE * seq_len * ELEM
    live: set = set()
    best = 0
    for _ts, action, (tkey, version), nbytes in mem.timeline:
        cat = mem._categories.get(tkey, version)
        if cat is not None and cat.name in _BASE_CATEGORIES:
            continue
        if nbytes != plane:
            continue
        if action.name in ('PREEXISTING', 'CREATE'):
            live.add(tkey)
            best = max(best, len(live))
        elif action.name == 'DESTROY':
            live.discard(tkey)
    return best


if __name__ == '__main__':
    torch.manual_seed(0)
    results = {s: report(s) for s in (512, 1024)}
    (p1, b1), (p2, b2) = results[512], results[1024]
    print(f"\nslope of peak live: "
          f"{(p2 - p1) / (1024 - 512) :.1f} B/token at fp32")
    print(f"bucket agreement across the two lengths: "
          f"{'SAME' if b1 == b2 else 'DIFFERENT -- ' + str((b1, b2))}")

    print("\nISOLATED MLP -- max co-resident intermediate-width planes")
    for s in (512, 1024, 2048):
        n = isolate_mlp(s)
        print(f"  seq_len={s:>5}: {n} planes "
              f"= {n * INTERMEDIATE * 2 * 1024 / 1048576:.2f} MiB/1024tok at fp16")
    print("  byte model carries 4 (_SWIGLU_SAVED_PLANES), saves only")
    sys.exit(0)
