"""Can STOCK eager PyTorch save attention to host instead of replaying it?

THE QUESTION (can attention be saved instead of recomputed?). Selective
activation checkpointing (SAC) can mark the SDPA op MUST_SAVE, so the replay
skips the attention kernel. If the stock offload wrapper around that also
moves the SAC-saved outputs to host, PyTorch already realizes the idea with no
policy, and its time win is not ours to claim.

THE COMPOSITION TESTED is the driver's offload baseline with one change:

    offload_wrapper(checkpoint_wrapper(layer, context_fn=SAC(save SDPA)))

WHAT IS MEASURED, per arm, on a tiny Llama with attn_implementation='sdpa':
  - attention-forward launches in forward vs during backward (SAC realized?)
  - what reaches the offload context's pack hook (SAC outputs offloaded?)
  - loss and gradients against the static twin (lossless?)

SCOPE. Eager mode, the installed PyTorch. It does NOT cover torch.compile
paths: TorchTitan's graph trainer tags SAC MUST_SAVE
tensors and offloads them under a budget on the traced graph — a different,
compiled mechanism this test cannot speak for.

Run:  PYTHONPATH=. python3 tests/test_sac_offload_composition.py
"""

import functools

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import torch.distributed.algorithms._checkpoint.checkpoint_wrapper as cw
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing, checkpoint_wrapper, offload_wrapper,
    CheckpointImpl)

from tests.test_offload_baseline import RecordingSaveOnCPU

L, HIDDEN, SEQ, VOCAB, BATCH, HEADS = 4, 64, 16, 128, 2, 4

aten = torch.ops.aten
SDPA_FWD = {
    getattr(aten, n).default for n in (
        '_scaled_dot_product_flash_attention_for_cpu',
        '_scaled_dot_product_flash_attention',
        '_scaled_dot_product_efficient_attention',
        '_scaled_dot_product_cudnn_attention',
    ) if hasattr(aten, n)
}

PASS = FAIL = 0


def check(name, cond, detail=''):
    global PASS, FAIL
    PASS, FAIL = (PASS + 1, FAIL) if cond else (PASS, FAIL + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + ('' if cond else f'   {detail}'))


def sac_policy(ctx, op, *args, **kwargs):
    return CheckpointPolicy.MUST_SAVE if op in SDPA_FWD else CheckpointPolicy.PREFER_RECOMPUTE


def build_model():
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=2 * HIDDEN,
        num_hidden_layers=L, num_attention_heads=HEADS, num_key_value_heads=2,
        max_position_embeddings=SEQ, use_cache=False, attn_implementation='sdpa')
    model = LlamaForCausalLM(cfg).to(torch.float32)
    model.config.use_cache = False
    return model


def wrap(model, offload, sac):
    kw = {'context_fn': functools.partial(create_selective_checkpoint_contexts, sac_policy)} if sac else {}
    inner = lambda m: checkpoint_wrapper(m, checkpoint_impl=CheckpointImpl.NO_REENTRANT, **kw)
    fn = (lambda m: offload_wrapper(inner(m))) if offload else inner
    apply_activation_checkpointing(model, checkpoint_wrapper_fn=fn,
                                   check_fn=lambda m: isinstance(m, LlamaDecoderLayer))
    return model


class CountSDPA(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.n = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in SDPA_FWD:
            self.n += 1
        return func(*args, **(kwargs or {}))


def run(model, record=False):
    torch.manual_seed(1234)
    ids = torch.randint(0, VOCAB, (BATCH, SEQ))
    real = cw.save_on_cpu
    RecordingSaveOnCPU.record = []
    if record:
        cw.save_on_cpu = RecordingSaveOnCPU
    try:
        with CountSDPA() as fwd_count:
            out = model(input_ids=ids, labels=ids)
        with CountSDPA() as bwd_count:
            out.loss.backward()
    finally:
        cw.save_on_cpu = real
    grads = {n.replace('_checkpoint_wrapped_module.', ''): p.grad.detach().clone()
             for n, p in model.named_parameters() if p.grad is not None}
    payload = [s for s in RecordingSaveOnCPU.record if all(d > 0 for d in s)]
    return out.loss.item(), grads, fwd_count.n, bwd_count.n, payload


def max_grad_delta(a, b):
    return max((a[n] - b[n]).abs().max().item() for n in b)


def main():
    print(__doc__.split('WHAT IS MEASURED')[0].strip().splitlines()[0], '\n')
    check('an SDPA forward op exists in this build', len(SDPA_FWD) > 0)
    boundary = (BATCH, SEQ, HIDDEN)

    print('1. Static whole-layer twin (the reference)')
    loss_ref, g_ref, f, b, _ = run(wrap(build_model(), offload=False, sac=False))
    check('attention runs in forward AND in the backward replay', f == L and b == L, f'fwd {f}, bwd {b}')

    print('\n2. SAC alone: the replay skips attention (the time win exists)')
    loss_s, g_s, f, b, _ = run(wrap(build_model(), offload=False, sac=True))
    check('no attention forward during backward', f == L and b == 0, f'fwd {f}, bwd {b}')
    check('lossless vs twin', loss_s == loss_ref and max_grad_delta(g_s, g_ref) < 1e-6,
          f'{loss_s} vs {loss_ref}, grad {max_grad_delta(g_s, g_ref):.2e}')

    print('\n3. Negative control: offload + checkpoint, no SAC')
    _, _, f, b, pay_plain = run(wrap(build_model(), offload=True, sac=False), record=True)
    check('offload context sees exactly L boundaries', pay_plain == [boundary] * L, f'{pay_plain}')

    print('\n4. THE COMPOSITION: offload + checkpoint + SAC(save SDPA)')
    loss_c, g_c, f, b, pay = run(wrap(build_model(), offload=True, sac=True), record=True)
    check('composition runs and skips the attention replay', f == L and b == 0, f'fwd {f}, bwd {b}')
    check('lossless vs twin', loss_c == loss_ref and max_grad_delta(g_c, g_ref) < 1e-6,
          f'{loss_c} vs {loss_ref}')
    sac_reached_host = len(pay) > L or any(s != boundary for s in pay)
    print(f'        offload context saw: {pay}')
    print(f'        VERDICT: SAC-saved attention outputs '
          f"{'DO' if sac_reached_host else 'do NOT'} reach the offload context")

    print('\n5. Positive control: the recorder WOULD see attention outputs if routed to it')
    # Inverted order: the offload context is inside the checkpoint region, so
    # every save in the region — including attention's — passes its pack hook.
    m = build_model()
    apply_activation_checkpointing(
        m, checkpoint_wrapper_fn=lambda mod: checkpoint_wrapper(
            offload_wrapper(mod), checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=lambda mod: isinstance(mod, LlamaDecoderLayer))
    _, _, _, _, pay_inv = run(m, record=True)
    attn_shaped = [s for s in pay_inv if len(s) == 4 and s[1] == HEADS]
    check('recorder sees (batch, heads, seq, head_dim) tensors when they are routed',
          len(attn_shaped) > 0, f'{sorted(set(pay_inv))}')

    print(f'\n{PASS} passed, {FAIL} failed')
    print('VERDICT (eager):', 'FAILS — stock eager composition offloads attention outputs'
          if sac_reached_host else
          'PASSES in eager mode — SAC outputs stay where SAC keeps them (on the device on GPU)')
    raise SystemExit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
