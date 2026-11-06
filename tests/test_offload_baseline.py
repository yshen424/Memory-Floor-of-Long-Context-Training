"""The offload-everything baseline sees exactly the engine's save set.

THE QUESTION. The evaluation needs the obvious baseline: unconditional
activation offloading, stock PyTorch, no criterion and no gate.
For that arm to be a control rather than a different experiment, it must move
the SAME BYTES the engine moves -- one fp16 residual boundary per layer -- and
not the layer interior, which the wrapper already recomputes.

WHY THE ORDER OF WRAPPING DECIDES THAT. Non-reentrant checkpointing saves the
region's inputs through ``_NoopSaveInputs.apply`` BEFORE it enters its own
``_checkpoint_hook`` context, and every save made inside the region is caught
by that inner pair and discarded for replay. So an outer saved-tensor context
sees the boundary and nothing else -- the same mechanism that makes the
engine's seam see exactly L bundles (tests/test_checkpoint_seam_visibility.py).
``offload_wrapper``
installs ``save_on_cpu`` as such an outer context, so

    offload_wrapper(checkpoint_wrapper(layer))     offloads the boundary
    checkpoint_wrapper(offload_wrapper(layer))     offloads the interior

and only the first is comparable to the engine. This test pins that order.

WHAT IT DOES NOT COVER. The device transfer itself: on a CPU-only host
``save_on_cpu`` degrades its own pin_memory to False and the copy is a no-op.
What is verified here is the SAVE SET -- how many tensors are routed to the
offload context and what shape they are -- plus losslessness against an
un-offloaded twin. Transfer behaviour is a hardware question and belongs to
the cluster arm.

Run:  PYTHONPATH=. python3 tests/test_offload_baseline.py
"""

import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import torch.distributed.algorithms._checkpoint.checkpoint_wrapper as cw
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing, checkpoint_wrapper, offload_wrapper,
    CheckpointImpl)

L = 4
HIDDEN = 64
SEQ = 16
VOCAB = 128
BATCH = 2

PASS = 0
FAIL = 0


def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f'  PASS  {name}')
    else:
        FAIL += 1
        print(f'  FAIL  {name}   {detail}')


def build_model():
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=2 * HIDDEN,
        num_hidden_layers=L, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=SEQ, use_cache=False,
        attn_implementation='eager')
    model = LlamaForCausalLM(cfg).to(torch.float32)
    model.config.use_cache = False
    return model


def wrap(model, offload):
    """The driver's own composition, verbatim in structure."""
    fn = (lambda m: offload_wrapper(
              checkpoint_wrapper(m, checkpoint_impl=CheckpointImpl.NO_REENTRANT))
          ) if offload else (
          lambda m: checkpoint_wrapper(
              m, checkpoint_impl=CheckpointImpl.NO_REENTRANT))
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=fn,
        check_fn=lambda m: isinstance(m, LlamaDecoderLayer))


def run(model):
    torch.manual_seed(1234)
    ids = torch.randint(0, VOCAB, (BATCH, SEQ))
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    # Each wrapper adds a ``_checkpoint_wrapped_module`` level, and the
    # offloaded arm carries one more of them than its twin, so parameter names
    # must be normalized before the two can be compared at all.
    grads = {n.replace('_checkpoint_wrapped_module.', ''): p.grad.detach().clone()
             for n, p in model.named_parameters() if p.grad is not None}
    return out.loss.item(), grads


class RecordingSaveOnCPU(torch.autograd.graph.saved_tensors_hooks):
    """Stands in for ``save_on_cpu`` and records what reaches its pack hook.

    Signature-compatible with the real thing because ``OffloadWrapper.forward``
    calls it as ``save_on_cpu(pin_memory=True)``.
    """

    record: list = []

    def __init__(self, pin_memory=False, device_type='cuda'):
        def pack(t):
            RecordingSaveOnCPU.record.append(tuple(t.shape))
            return t.to('cpu', copy=True)

        def unpack(t):
            return t

        super().__init__(pack, unpack)


def main():
    print(__doc__.split('Run:')[0].strip().splitlines()[0])
    print()

    # ── 1. The save set, counted ───────────────────────────────────────────
    print('1. What reaches the offload context')
    model = build_model()
    wrap(model, offload=True)
    RecordingSaveOnCPU.record = []
    real_save_on_cpu = cw.save_on_cpu
    cw.save_on_cpu = RecordingSaveOnCPU
    try:
        loss_off, grads_off = run(model)
    finally:
        cw.save_on_cpu = real_save_on_cpu
    rec = RecordingSaveOnCPU.record
    # ``_NoopSaveInputs.apply(dummy, kwargs, *args)`` saves a zero-element
    # sentinel alongside the real inputs, so every layer routes two entries
    # and one of them is empty. It carries no bytes and therefore no traffic.
    payload = [s for s in rec if all(d > 0 for d in s)]
    empties = [s for s in rec if any(d == 0 for d in s)]

    check('one byte-carrying tensor offloaded per layer',
          len(payload) == L, f'expected {L}, got {len(payload)} -> {payload}')
    check('every offloaded tensor is a residual boundary (batch, seq, hidden)',
          all(s == (BATCH, SEQ, HIDDEN) for s in payload),
          f'shapes were {sorted(set(payload))}')
    check('the only other entries are zero-element sentinels',
          len(empties) == L and all(s == (0,) for s in empties),
          f'unexpected non-payload shapes {sorted(set(empties))}')
    check('no interior activation reached the offload context',
          not any(s[-1] == 2 * HIDDEN for s in rec),
          'an intermediate-width tensor was offloaded: the wrapping order is '
          'inverted and this arm is not comparable to the engine')

    # ── 2. Losslessness against the un-offloaded twin ──────────────────────
    print('\n2. Losslessness against static whole-layer checkpointing')
    twin = build_model()
    wrap(twin, offload=False)
    loss_ref, grads_ref = run(twin)

    check('loss is bit-identical to the un-offloaded twin',
          loss_off == loss_ref, f'{loss_off!r} vs {loss_ref!r}')
    same_keys = set(grads_off) == set(grads_ref)
    check('same parameter set produced gradients', same_keys)
    if same_keys:
        worst_name, worst = '', 0.0
        for n in grads_ref:
            d = (grads_off[n] - grads_ref[n]).abs().max().item()
            if d > worst:
                worst, worst_name = d, n
        check('gradients are bit-identical to the un-offloaded twin',
              worst == 0.0, f'worst |delta| {worst:.3e} at {worst_name}')

    # ── 3. The real save_on_cpu runs, unpatched ────────────────────────────
    # The recording stand-in proves the ROUTING; this proves the arm the
    # driver actually runs does not raise and stays lossless.
    print('\n3. Stock save_on_cpu, unpatched')
    stock = build_model()
    wrap(stock, offload=True)
    loss_stock, grads_stock = run(stock)
    check('stock offload_wrapper completes forward and backward',
          torch.isfinite(torch.tensor(loss_stock)).item(),
          f'loss was {loss_stock!r}')
    check('stock offload_wrapper is lossless',
          loss_stock == loss_ref, f'{loss_stock!r} vs {loss_ref!r}')

    # ── 4. The inverted order is detectably different ──────────────────────
    # A negative control: if this ever stops differing, the test above has
    # stopped discriminating and the claim it protects is unguarded.
    print('\n4. Negative control: the inverted wrapping order')
    inv = build_model()
    apply_activation_checkpointing(
        inv,
        checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(
            offload_wrapper(m), checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=lambda m: isinstance(m, LlamaDecoderLayer))
    RecordingSaveOnCPU.record = []
    cw.save_on_cpu = RecordingSaveOnCPU
    try:
        run(inv)
    finally:
        cw.save_on_cpu = real_save_on_cpu
    inv_rec = RecordingSaveOnCPU.record
    check('inverted order routes a different save set',
          len(inv_rec) != L or any(s != (BATCH, SEQ, HIDDEN) for s in inv_rec),
          f'inverted order produced the same {len(inv_rec)} boundaries, so '
          f'this test cannot tell the two compositions apart')
    print(f'       (inverted order routed {len(inv_rec)} tensors, '
          f'{len(set(inv_rec))} distinct shapes)')

    print(f'\n{PASS} passed, {FAIL} failed')
    return 1 if FAIL else 0


if __name__ == '__main__':
    raise SystemExit(main())
