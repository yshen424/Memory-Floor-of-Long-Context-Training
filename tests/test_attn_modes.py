"""Static checks for the attention modes that need sm80+ (bf16 mem-efficient, flash).

CPU only. (1) The refusal rules: sdpa_flash and bf16 sdpa_mem_efficient refuse below
sm80 and are accepted at sm80+. (2) --native_gqa gives the same outputs and
gradients as HF's repeat_kv path. (3) The saved-bytes change is recorded, not asserted:
the CPU math kernel expands k/v internally, so only flash on sm80+ can show the
-2.0 MiB/x the harness predicts (--no_repeat_kv).

    PYTHONPATH=. python3 tests/test_attn_modes.py
"""
import torch

import train_fsdp_profile as drv
from transformers.integrations.sdpa_attention import sdpa_attention_forward as hf_sdpa
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def expect_raise(fn, what):
    try:
        fn()
    except ValueError as e:
        print(f"  ok  refuses {what}: {str(e)[:70]}...")
        return
    raise AssertionError(f"{what} was accepted")


def test_refusals():
    real = drv._capability
    try:
        drv._capability = lambda: (7, 0)
        expect_raise(lambda: drv.pin_attention_backend('sdpa_flash'), 'flash on sm_70')
        expect_raise(lambda: drv.pin_attention_backend('sdpa_mem_efficient', dtype=torch.bfloat16),
                     'bf16 mem_efficient on sm_70')
        drv.pin_attention_backend('sdpa_mem_efficient', dtype=torch.float16)   # fp16 still fine
        drv._capability = lambda: (8, 0)
        drv.pin_attention_backend('sdpa_flash', dtype=torch.bfloat16)
        drv.pin_attention_backend('sdpa_mem_efficient', dtype=torch.bfloat16)
        print("  ok  sm80: flash and bf16 mem_efficient accepted")
        expect_raise(lambda: drv.pin_attention_backend('eager', native_gqa=True), 'native_gqa under eager')
    finally:
        drv._capability = real
        ALL_ATTENTION_FUNCTIONS['sdpa'] = hf_sdpa
        torch.backends.cuda.enable_math_sdp(True)
    assert drv.materializes_attn_scores('sdpa_flash') is False
    assert drv.materializes_attn_scores('sdpa_mem_efficient') is False
    assert drv.materializes_attn_scores('sdpa_math') is True
    print("  ok  flash counted as non-materializing")


class _Mod:
    def __init__(self, groups):
        self.num_key_value_groups = groups


def _run(fn, nq, nkv, s=64, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, nq, s, d, generator=g, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, nkv, s, d, generator=g, dtype=torch.float64, requires_grad=True)
    v = torch.randn(1, nkv, s, d, generator=g, dtype=torch.float64, requires_grad=True)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: (saved.append(t.numel() * t.element_size()), t)[1],
                                                  lambda t: t):
        out, _ = fn(_Mod(nq // nkv), q, k, v, None)
    out.sum().backward()
    return out.detach(), q.grad, k.grad, v.grad, saved


def test_native_gqa():
    drv.install_native_gqa_sdpa()
    native = ALL_ATTENTION_FUNCTIONS['sdpa']
    ALL_ATTENTION_FUNCTIONS['sdpa'] = hf_sdpa
    nq, nkv = 12, 4
    a = _run(hf_sdpa, nq, nkv)
    b = _run(native, nq, nkv)
    for name, x, y in zip(('out', 'dq', 'dk', 'dv'), a[:4], b[:4]):
        err = (x - y).abs().max().item()
        assert err < 1e-10, f"{name} differs by {err}"
    print("  ok  native GQA matches repeat_kv: outputs and all three gradients (fp64, < 1e-10)")
    s, d, el = 64, 16, 8
    expansion = 2 * (nq - nkv) * s * d * el
    saving = sum(a[4]) - sum(b[4])
    # The CPU math kernel expands k/v INSIDE the op when enable_gqa=True, so it saves
    # the same bytes either way: the saving is a property of the KERNEL (flash keeps
    # k/v at kv width), not of the call. Only the A30 flash arm can measure it.
    assert saving in (0, expansion), (saving, expansion)
    print(f"  ok  CPU math kernel saving = {saving} B (expected 0: it expands internally); "
          f"a native kernel would save {expansion} B here -> Tier-A 2048 B/token = 2.0 MiB/x")

if __name__ == '__main__':
    test_refusals()
    test_native_gqa()
    print("ALL PASS")
