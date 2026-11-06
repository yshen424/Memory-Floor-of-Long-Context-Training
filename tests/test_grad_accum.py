"""Two accumulated b=1 micro-batches train exactly what one b=2 step trains.

THE QUESTION. The micro-batching baseline compares the engine running a b=2
step against gradient accumulation running the SAME two samples as two b=1
micro-batches. For that to be a comparison of execution strategies and not of
workloads, the accumulated arm must see the same tokens and produce the same
loss and the same gradients as the b=2 arm. This test pins all three, on the
driver's own loss path (whole-layer checkpointing + chunked cross-entropy),
by reproducing the driver's micro-batch split and its loss/N scaling.

WHAT IT DOES NOT COVER. FSDP's reduce-scatter per micro-batch and the step
time: those are hardware questions and belong to the cluster job. The token
split is checked against the driver's get_batch directly.

Run:  PYTHONPATH=. python3 tests/test_grad_accum.py
"""

import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing, checkpoint_wrapper, CheckpointImpl)

import chunked_ce
from train_fsdp_profile import get_batch

L, HIDDEN, SEQ, VOCAB, CHUNK = 4, 64, 32, 128, 16
PASS = FAIL = 0


def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f'  PASS  {name}')
    else:
        FAIL += 1
        print(f'  FAIL  {name}   {detail}')


def build():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=HIDDEN,
                      intermediate_size=2 * HIDDEN, num_hidden_layers=L,
                      num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=SEQ, use_cache=False,
                      attn_implementation='eager')
    m = LlamaForCausalLM(cfg).to(torch.float64)
    m.config.use_cache = False
    apply_activation_checkpointing(
        m, checkpoint_wrapper_fn=lambda mod: checkpoint_wrapper(
            mod, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=lambda mod: isinstance(mod, LlamaDecoderLayer))
    chunked_ce.install(m, CHUNK)
    return m


def grads(m):
    return {n: p.grad.detach().clone() for n, p in m.named_parameters()
            if p.grad is not None}


def draw(batch, accum, seed=1234):
    # The driver's draw: ONE call for batch x accum rows, then split.
    g = torch.Generator().manual_seed(seed)
    full = get_batch(batch * accum, SEQ, VOCAB, device='cpu', generator=g)
    return full, (full.split(batch, dim=0) if accum > 1 else (full,))


def main():
    print(__doc__.split('Run:')[0].strip().splitlines()[0])
    print()

    print('1. Same tokens')
    full2, _ = draw(2, 1)
    full1, micro = draw(1, 2)
    check('accumulated arm draws the b=2 arm\'s tokens exactly',
          torch.equal(full2, full1))
    check('two micro-batches of one row each',
          len(micro) == 2 and all(t.shape == (1, SEQ) for t in micro))
    check('micro-batches are the b=2 rows, in order',
          torch.equal(torch.cat(micro, 0), full2))

    print('\n2. Same loss and gradients, on the driver\'s loss path')
    a = build()
    out = a(input_ids=full2, labels=full2)
    out.loss.backward()
    loss_b2, g_b2 = out.loss.item(), grads(a)

    b = build()
    loss_sum = None
    for ids in micro:                       # the driver's accumulation loop
        o = b(input_ids=ids, labels=ids)
        (o.loss / 2).backward()
        d = o.loss.detach()
        loss_sum = d if loss_sum is None else loss_sum + d
    loss_acc, g_acc = (loss_sum / 2).item(), grads(b)

    check('reported loss equals the b=2 loss',
          abs(loss_acc - loss_b2) < 1e-12, f'{loss_acc!r} vs {loss_b2!r}')
    same = set(g_b2) == set(g_acc)
    check('same parameter set produced gradients', same)
    if same:
        worst = max((g_b2[n] - g_acc[n]).abs().max().item() for n in g_b2)
        scale = max(g_b2[n].abs().max().item() for n in g_b2)
        check('gradients equal the b=2 gradients to round-off',
              worst <= 1e-10 * max(scale, 1.0),
              f'worst |delta| {worst:.3e} (grad scale {scale:.3e})')

    print('\n3. The negative control: no 1/N scaling doubles the gradient')
    c = build()
    for ids in micro:
        c(input_ids=ids, labels=ids).loss.backward()
    g_bad = grads(c)
    n0 = next(iter(g_b2))
    ratio = (g_bad[n0].norm() / g_b2[n0].norm()).item()
    check('unscaled accumulation is detectably wrong (ratio ~2)',
          abs(ratio - 2.0) < 1e-6, f'ratio {ratio:.6f}')

    print(f'\n{PASS} passed, {FAIL} failed')
    return 1 if FAIL else 0


if __name__ == '__main__':
    raise SystemExit(main())
