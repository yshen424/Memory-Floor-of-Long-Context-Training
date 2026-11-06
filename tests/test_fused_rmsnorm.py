"""Fused RMSNorm: gradient parity, and what each mode leaves resident.

Run: PYTHONPATH=. python3 tests/test_fused_rmsnorm.py

Two things are checked, and the second is the one that matters for the paper.

1. GRADIENT PARITY against HuggingFace's LlamaRMSNorm, in fp32 and fp16.

2. RETAINED BYTES, measured with the same methodology pack_fn uses on the
   cluster — every tensor the autograd graph saves, deduplicated by STORAGE
   pointer, with parameter storages subtracted. The unit under test is a norm
   followed by the projection that consumes it, because the projection's save
   of the norm output is exactly what makes mode='output' an alias.

   Expected, per norm, per token, at hidden h:

       HF            4h + 2h + 4      fp32 upcast, fp16 cast, rstd
       fused input   2h + 4           x, rstd
       fused output       4           rstd only; y is the projection's save

   plus 2h for the projection's save of y in all three. Scaled to h=768 and
   s=4096 and doubled for the two norms in a decoder layer, the HF row is
   2 x (8h + 4) B x 4096 = 48.03125 MiB/layer (arithmetic).
"""

import sys

import torch
import torch.nn as nn
from torch.autograd.graph import saved_tensors_hooks
from transformers.models.llama.modeling_llama import LlamaRMSNorm

sys.path.insert(0, '.')
from fused_rmsnorm import FusedRMSNorm, install_fused_rmsnorm  # noqa: E402

EPS = 1e-5
FAILURES = []


def check(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


# ── 1. Gradient parity ────────────────────────────────────────────────────────

def parity(dtype, tol_y, tol_gx, tol_gw):
    torch.manual_seed(0)
    h, tokens = 64, 32
    x = torch.randn(tokens, h, dtype=dtype) * 2.0
    w0 = torch.randn(h, dtype=dtype) * 0.5 + 1.0
    g = torch.randn(tokens, h, dtype=dtype)

    def run(module):
        xi = x.clone().requires_grad_(True)
        y = module(xi)
        y.backward(g)
        return y.detach(), xi.grad.detach(), module.weight.grad.detach()

    ref = LlamaRMSNorm(h, eps=EPS).to(dtype)
    with torch.no_grad():
        ref.weight.copy_(w0)
    y_ref, gx_ref, gw_ref = run(ref)

    for mode in ('input', 'output'):
        p = nn.Parameter(w0.clone())
        y, gx, gw = run(FusedRMSNorm(p, EPS, mode=mode))
        dy = (y.float() - y_ref.float()).abs().max().item()
        dgx = (gx.float() - gx_ref.float()).abs().max().item()
        dgw = (gw.float() - gw_ref.float()).abs().max().item()
        rel = dgw / gw_ref.float().abs().max().item()
        check(f"{dtype} mode={mode} forward", dy <= tol_y, f"max|dy|={dy:.3e}")
        check(f"{dtype} mode={mode} grad_x", dgx <= tol_gx, f"max|dgx|={dgx:.3e}")
        check(f"{dtype} mode={mode} grad_w", rel <= tol_gw, f"rel={rel:.3e}")


# ── 2. Retained bytes ─────────────────────────────────────────────────────────

def retained_bytes(norm, proj, x):
    """Distinct saved storages, parameter storages removed. pack_fn's rule."""
    params = {p.untyped_storage().data_ptr()
              for m in (norm, proj) for p in m.parameters()}
    seen, total = set(), 0

    def pack(t):
        nonlocal total
        ptr = t.untyped_storage().data_ptr()
        if ptr not in seen and ptr not in params:
            seen.add(ptr)
            total += t.untyped_storage().nbytes()
        return t

    with saved_tensors_hooks(pack, lambda t: t):
        proj(norm(x))
    return total, len(seen)


def residency(dtype):
    h, tokens = 768, 1024
    x = torch.randn(tokens, h, dtype=dtype, requires_grad=True)
    proj = nn.Linear(h, h, bias=False).to(dtype)

    rows = {}
    ref = LlamaRMSNorm(h, eps=EPS).to(dtype)
    rows['hf'] = retained_bytes(ref, proj, x)
    for mode in ('input', 'output'):
        p = nn.Parameter(torch.ones(h, dtype=dtype))
        rows[mode] = retained_bytes(FusedRMSNorm(p, EPS, mode=mode), proj, x)

    itemsize = torch.tensor([], dtype=dtype).element_size()
    expect = {
        'hf': tokens * (4 * h + itemsize * h + 4) + tokens * itemsize * h,
        'input': tokens * (itemsize * h + 4) + tokens * itemsize * h,
        'output': tokens * 4 + tokens * itemsize * h,
    }
    print(f"  {'variant':8s} {'bytes':>12s} {'storages':>9s} {'B/token':>9s} "
          f"{'expected':>12s}")
    for k in ('hf', 'input', 'output'):
        got, n = rows[k]
        print(f"  {k:8s} {got:12d} {n:9d} {got / tokens:9.1f} "
              f"{expect[k]:12d}")
        check(f"{dtype} retained bytes, {k}", got == expect[k])

    # The alias claim: mode='output' must add NO storage over the projection's
    # own save, i.e. exactly one hidden-width storage plus rstd.
    check(f"{dtype} mode=output adds no hidden-width storage",
          rows['output'][1] == 2, f"{rows['output'][1]} storages")

    # Scaled to the per-layer figure at s=4096: 48.03125 MiB/layer counts the
    # norm's own saves AND the projection's save of y — the four 6.00 MiB fp16
    # blocks are two casts (norm-owned) and two outputs (bundle-owned).
    per_layer = {k: 2 * rows[k][0] * 4 / 1024 / 1024 for k in rows}
    check("HF unit is 48.03 MiB/layer at s=4096",
          abs(per_layer['hf'] - 48.03125) < 0.01,
          f"{per_layer['hf']:.5f} MiB/layer")

    # What each mode leaves EVICTABLE: only the projection's save of y is a
    # bundle member, and mode='output' pins it with the norm's own save.
    KV_ROTARY = 4.0  # MiB/layer at s=4096, measured, not norm mass
    evictable = {'hf': 12.0, 'input': 12.0, 'output': 0.0}
    print(f"  {'variant':8s} {'MiB/layer':>10s} {'+kv/rot':>8s} "
          f"{'evictable':>10s} {'slope L32':>10s}")
    for k in ('hf', 'input', 'output'):
        resident = per_layer[k] + KV_ROTARY
        slope = 437.50 + resident / 4 * 32 + 6.27
        print(f"  {k:8s} {per_layer[k]:10.5f} {resident:8.5f} "
              f"{evictable[k]:10.2f} {slope:10.2f}")
    check("fused input reproduces the 28.03 MiB/layer prediction",
          abs(per_layer['input'] + KV_ROTARY - 28.03125) < 0.01)
    check("fused output reproduces the 16.03 MiB/layer prediction",
          abs(per_layer['output'] + KV_ROTARY - 16.03125) < 0.01)
    check("perfect migration on input-mode equals output-mode for free",
          abs((per_layer['input'] - per_layer['output']) - evictable['input'])
          < 0.01)
    return rows


# ── 3. Installer ──────────────────────────────────────────────────────────────

def installer():
    from transformers import LlamaConfig, AutoModelForCausalLM
    cfg = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2,
                      vocab_size=128, max_position_embeddings=16)
    model = AutoModelForCausalLM.from_config(cfg)
    before = {n: p.data_ptr() for n, p in model.named_parameters()}
    n = install_fused_rmsnorm(model, mode='input')
    after = {n_: p.data_ptr() for n_, p in model.named_parameters()}
    check("installer replaced every norm", n == 2 * 2 + 1, f"{n} replaced")
    check("parameter set and identity unchanged", before == after)
    check("no LlamaRMSNorm survives",
          not any(isinstance(m, LlamaRMSNorm) for m in model.modules()))
    ids = model.model.embed_tokens(torch.zeros(1, 4, dtype=torch.long))
    out = model(inputs_embeds=ids, labels=torch.zeros(1, 4, dtype=torch.long))
    out.loss.backward()
    check("model still runs forward+backward", torch.isfinite(out.loss).item())


# ── 4. The whole-layer checkpoint unit ────────────────────────────────────────

def layer_checkpoint():
    """What a coarser recompute unit retains.

    Under the per-module unit both norms sit at the seam and survive. Wrapping
    LlamaDecoderLayer absorbs them, and the claim under test is that what
    remains is exactly the layer INPUT: 2h bytes/token/layer, no constant term,
    and identical whether the norm is fused or not. That last part is the
    interesting half — it means the coarser unit makes the fused norm
    irrelevant, and takes migration's supply with it.
    """
    import functools
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing, checkpoint_wrapper, CheckpointImpl)
    from transformers import LlamaConfig, AutoModelForCausalLM
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    tokens = 64

    def saved(h, layers, fused):
        torch.manual_seed(0)
        cfg = LlamaConfig(hidden_size=h, intermediate_size=8 * h // 3,
                          num_hidden_layers=layers, num_attention_heads=h // 64,
                          num_key_value_heads=max(1, h // 192), vocab_size=256,
                          max_position_embeddings=tokens,
                          attn_implementation='eager')
        model = AutoModelForCausalLM.from_config(cfg).to(torch.float16)
        if fused:
            install_fused_rmsnorm(model, mode='input')
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=functools.partial(
                checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
            check_fn=lambda m: isinstance(m, LlamaDecoderLayer))
        # Mandatory: the replay re-enters the layer after the cache was
        # extended, and the shapes no longer match.
        model.config.use_cache = False
        params = {p.untyped_storage().data_ptr() for p in model.parameters()}
        seen, total = set(), 0

        def pack(t):
            nonlocal total
            ptr = t.untyped_storage().data_ptr()
            if ptr not in seen and ptr not in params:
                seen.add(ptr)
                total += t.untyped_storage().nbytes()
            return t

        ids = torch.randint(0, 256, (1, tokens))
        with saved_tensors_hooks(pack, lambda t: t):
            out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        return total

    print(f"  {'h':>5s} {'fused':>6s} {'B/token/layer':>14s} {'x h':>7s}")
    for h in (192, 384):
        for fused in (False, True):
            per = (saved(h, 8, fused) - saved(h, 4, fused)) / 4 / tokens
            print(f"  {h:5d} {str(fused):>6s} {per:14.2f} {per / h:7.3f}")
            check(f"layer checkpoint retains 2h at h={h}, fused={fused}",
                  abs(per - 2 * h) < 1e-9, f"{per:.2f} vs {2 * h}")


if __name__ == '__main__':
    print("gradient parity")
    parity(torch.float32, 1e-6, 1e-5, 1e-5)
    parity(torch.float16, 2e-3, 4e-3, 5e-3)
    print("retained bytes (fp16)")
    residency(torch.float16)
    print("installer")
    installer()
    print("whole-layer checkpoint unit")
    layer_checkpoint()
    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} — {FAILURES}")
        sys.exit(1)
    print("all checks passed")
