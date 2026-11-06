"""
FSDP LLaMA Training with Integrated Memory Profiling.

Training setup:
- HuggingFace LlamaConfig + AutoModelForCausalLM
- FSDP wrapping per LlamaDecoderLayer
- MixedPrecision (bf16 or fp16)
- AdamW optimizer
- NCCL backend, torchrun launch

Profiling (FSDPMemoryProfiler):
- Per-layer memory snapshots (forward + backward)
- Activation breakdown (attention scores vs SwiGLU vs QKV)
- FSDP sharding analysis (local shard vs full model)
- Allocation group tracking (SwiGLU triples)
- Memory timeline export (JSON per rank)
- Optional PyTorch profiler trace export
- Configurable via command line args

Usage:
    srun torchrun ... train_fsdp_profile.py --batch_size 8 --seq_len 512 --steps 5
"""

import contextlib
import os
import sys
import argparse
import functools
import time
import json

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, BackwardPrefetch
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from transformers import LlamaConfig, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from fsdp_profiler import FSDPMemoryProfiler

# Placement-engine imports live inside the gating block below so a run without
# --placement_engine doesn't pay any import overhead or fail in
# environments where the memory_manager package isn't on the path.


# ── Distributed Init ──────────────────────────────────────────────────────────

def init_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.barrier(device_ids=[local_rank])
    return local_rank


# ── Model ─────────────────────────────────────────────────────────────────────

def _sdpa_preflight(dtype, head_dim: int, n_q: int = 2, n_kv: int = 2,
                    native_gqa: bool = False) -> dict:
    """Run one tiny SDPA forward+backward under the pinned backends.

    With math disabled there is no fallback, so a shape the
    efficient kernel cannot take raises inside the first real forward — 32
    layers and one FSDP all-gather deep, where the message is buried. This
    surfaces the same failure on line one of the log, with its reason. It uses
    the run's own head counts and, under --native_gqa, the un-expanded k/v.
    """
    if not torch.cuda.is_available():
        return {'preflight': 'skipped — no cuda'}
    try:
        kv = n_kv if native_gqa else n_q
        q = torch.randn(1, n_q, 128, head_dim, device='cuda',
                        dtype=dtype, requires_grad=True)
        k = torch.randn(1, kv, 128, head_dim, device='cuda',
                        dtype=dtype, requires_grad=True)
        extra = {'enable_gqa': True} if native_gqa and kv != n_q else {}
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, k, is_causal=True, **extra)
        out.sum().backward()
        del q, k, out
        torch.cuda.empty_cache()
        return {'preflight': 'ok'}
    except Exception as exc:
        return {'preflight': 'FAILED', 'preflight_error': str(exc)[:400]}


def _outside_allocator_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info()
    return (total - free - torch.cuda.memory_reserved()) / 2**20


_ALLOC_KEYS = ('num_alloc_retries', 'num_ooms', 'num_device_alloc', 'num_device_free',
               'num_sync_all_streams', 'segment.all.current', 'reserved_bytes.all.current')


def _alloc_counters() -> dict:
    if not torch.cuda.is_available():
        return {}
    st = torch.cuda.memory_stats()
    return {k: st.get(k) for k in _ALLOC_KEYS}


def _capability() -> tuple:
    if not torch.cuda.is_available():
        return (0, 0)
    return torch.cuda.get_device_capability()


def install_native_gqa_sdpa():
    """--native_gqa: HF's SDPA path expands k
    and v to every query head (repeat_kv) before the kernel, so the kernel saves
    them at full width. This replaces it with the same call minus repeat_kv,
    passing enable_gqa=True, so k and v are saved at n_kv width (harness
    --no_repeat_kv: -2.0 MiB/x at the Tier-A shape). The kernel must accept
    GQA natively; the preflight proves it before the model is built."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    def sdpa_native_gqa(module, query, key, value, attention_mask,
                        dropout=0.0, scaling=None, is_causal=None, **kwargs):
        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key.shape[-2]]
        query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
        if is_causal is None:
            is_causal = query.shape[2] > 1 and causal_mask is None
        out = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=causal_mask, dropout_p=dropout,
            scale=scaling, is_causal=is_causal,
            enable_gqa=key.shape[1] != query.shape[1])
        return out.transpose(1, 2).contiguous(), None

    ALL_ATTENTION_FUNCTIONS['sdpa'] = sdpa_native_gqa


def pin_attention_backend(mode: str, rank: int = 0, dtype=None,
                          head_dim: int = 64, n_q: int = 2, n_kv: int = 2,
                          native_gqa: bool = False) -> dict:
    """Fix which attention kernel runs, and report what was fixed.

    The s x s attention tensor is the only activation class quadratic in
    sequence length, and whether it is materialized is decided, if left alone,
    by PyTorch's SDPA dispatcher — a heuristic over dtype, head_dim, mask shape
    and device capability. If it selects the memory-efficient kernel at some
    sequence length it stops materializing the tensor entirely, and the ceiling
    would move for reasons having nothing to do with the policy under test.
    On a sweep whose whole point is where the ceiling lies, that is a silent
    confound, so the kernel is pinned and recorded rather than inferred.

    'sdpa_math' pins the score-materializing math kernel.

    'sdpa_mem_efficient' pins the memory-efficient (cutlass) kernel.
    PyTorch 2.6 compiles the cutlass efficient kernels for f16/f32 on
    sm50/70/75/80 but for bf16 on sm80+ ONLY, so on V100 (sm_70) the bf16 arm
    has no kernel to select and is rejected below rather than silently falling
    back; on sm80+ (A30, A100, L40S) bf16 is accepted.

    'sdpa_flash' pins FlashAttention-2 with math and efficient disabled; it
    needs sm80+.

    'sdpa_auto' leaves the dispatcher free. Kept for compatibility.
    """
    info = {'requested': mode}
    if mode == 'sdpa_auto':
        info['note'] = 'dispatcher left free — kernel choice not pinned'
    elif mode == 'sdpa_math':
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    elif mode == 'sdpa_mem_efficient':
        if dtype is torch.bfloat16 and _capability() < (8, 0):
            raise ValueError(
                "--attn_impl sdpa_mem_efficient needs --dtype fp16 on this card: "
                "the cutlass efficient kernels are built for bf16 on sm80+ only, "
                f"and this is sm_{_capability()[0]}{_capability()[1]}, so the bf16 "
                "arm would have no kernel to run.")
        # Math off is the point, not an oversight. With a fallback available a
        # kernel that cannot dispatch would silently reproduce the math numbers,
        # and the arm would measure nothing while looking like it worked.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        info['guarantee'] = ('math and flash are disabled, so a step that '
                             'completes proves the efficient kernel served '
                             'every attention call')
    elif mode == 'sdpa_flash':
        if _capability() < (8, 0):
            raise ValueError(
                "--attn_impl sdpa_flash needs sm80+ (FlashAttention-2); this is "
                f"sm_{_capability()[0]}{_capability()[1]}.")
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        info['guarantee'] = ('math and efficient are disabled, so a step that '
                             'completes proves the flash kernel served every '
                             'attention call')
    elif mode != 'eager':
        raise ValueError(f"unknown --attn_impl {mode!r}")
    if torch.cuda.is_available():
        info['flash_enabled'] = torch.backends.cuda.flash_sdp_enabled()
        info['mem_efficient_enabled'] = torch.backends.cuda.mem_efficient_sdp_enabled()
        info['math_enabled'] = torch.backends.cuda.math_sdp_enabled()
    if native_gqa:
        if not mode.startswith('sdpa'):
            raise ValueError("--native_gqa needs an sdpa --attn_impl")
        install_native_gqa_sdpa()
        info['native_gqa'] = True
    if mode.startswith('sdpa') and dtype is not None:
        info.update(_sdpa_preflight(dtype, head_dim, n_q, n_kv, native_gqa))
    if rank == 0:
        print(f"  [Attention] --attn_impl {mode} -> {info}")
    return info


def materializes_attn_scores(attn_impl: str) -> bool:
    """Whether this run's attention kernel forms an (B, heads, S, S) tensor.

    Drives the activation byte model in both AccessPredictor and the profiler.
    Only 'sdpa_mem_efficient' and 'sdpa_flash' are treated as non-materializing,
    and deliberately so: each disables every other backend, so a step that
    completes proves the pinned kernel ran. 'sdpa_auto' leaves the dispatcher
    free and its choice is not knowable before the run, so it keeps the
    score-materializing model — the conservative direction.
    """
    return attn_impl not in ('sdpa_mem_efficient', 'sdpa_flash')


def build_model(args) -> tuple:
    """Build model and return (model, config)."""
    config = LlamaConfig(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        max_position_embeddings=args.seq_len,
        vocab_size=args.vocab_size,
        rms_norm_eps=1e-5,
        # 'eager' selects HF's own attention (fp32 softmax upcast); the two
        # sdpa modes differ only in whether the kernel choice is pinned below.
        attn_implementation=('eager' if args.attn_impl == 'eager' else 'sdpa'),
    )
    # FSDP is built without sync_module_states, so each rank draws its own
    # initial weights. Seeding here makes every rank agree AND makes two arms
    # comparable — which is what a loss-curve control between the chunked and
    # unchunked loss requires. Off by default (--seed -1).
    if getattr(args, 'seed', -1) >= 0:
        torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_config(config)
    return model, config


# ── FSDP Wrapping ─────────────────────────────────────────────────────────────

def wrap_with_fsdp(model, local_rank, args):
    """Apply FSDP wrapping: FULL_SHARD per decoder layer, mixed precision."""

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer},
    )

    # V100 has no native bf16 support; --dtype selects bf16 or fp16.
    if args.dtype == 'bf16':
        param_dtype = torch.bfloat16
    else:
        param_dtype = torch.float16

    mixed_precision = MixedPrecision(
        param_dtype=param_dtype,
        reduce_dtype=param_dtype,
        buffer_dtype=param_dtype,
    )

    # backward_prefetch toggle. Default BACKWARD_PRE overlaps each layer's
    # all-gather with the prior layer's backward compute;
    # --no_backward_prefetch sets it to None, disabling the overlap.
    # Kept for compatibility; off by default.
    bwd_prefetch = (None if getattr(args, 'no_backward_prefetch', False)
                    else BackwardPrefetch.BACKWARD_PRE)

    # FSDP's own parameter+gradient offload, the ZeRO-Offload analogue.
    # Offloading params also forces the optimizer step onto the CPU.
    cpu_offload = (CPUOffload(offload_params=True)
                   if getattr(args, 'cpu_offload', False) else None)

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        device_id=torch.cuda.current_device(),
        backward_prefetch=bwd_prefetch,
        forward_prefetch=True,
        cpu_offload=cpu_offload,
    )

    return model


# ── Utilities ─────────────────────────────────────────────────────────────────

def print_sharding_proof(model, rank, world_size):
    local_param_count = sum(p.numel() for p in model.parameters())
    local_param_mb = sum(p.numel() * p.element_size()
                         for p in model.parameters()) / 1024**2

    count_tensor = torch.tensor(local_param_count, dtype=torch.long, device="cuda")
    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
    total_params = count_tensor.item()

    print(
        f"[Rank {rank}] Sharded params: {local_param_count:>12,} "
        f"({local_param_mb:.1f} MB)  |  "
        f"Summed: {total_params:>12,} (≈ full model)"
    )


def print_gpu_memory(rank, label=""):
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    print(
        f"[Rank {rank}] GPU mem {label:20s} | "
        f"alloc={allocated:7.1f} MB | "
        f"rsrvd={reserved:7.1f} MB | "
        f"total={total:7.1f} MB"
    )


def get_batch(batch_size, seq_len, vocab_size, device, generator=None):
    return torch.randint(0, vocab_size, (batch_size, seq_len), device=device,
                         generator=generator)


# Pinned pool bucket sizing. The pool's bucket list must be chosen *before*
# the first forward pass (the engine constructs it). The registry has no
# activation records yet at that point, so expected per-component sizes are
# computed from (batch, seq_len, model_config) — the same formulas the
# profiler uses in _activation_sizes_for_batch.
#
# Two sizing traps:
#
#   1. HF LLaMA's eager attention runs softmax in fp32
#      (``softmax(..., dtype=torch.float32)``) and that fp32 tensor is what
#      autograd saves for softmax backward, so a bucket sized at
#      ``dtype_size`` (2 for bf16) can never stage it at any buffer count.
#   2. A power-of-2 ladder with fall-through serves small payloads from
#      buffers many times their size.
#
# So: exact sizes for the classes that can be named, counts proportional to
# per-layer save multiplicity, and a small geometric ladder underneath to
# absorb unnamed classes (see the denials_by_class telemetry, which exists to
# identify them). Returns {size_bytes: buffer_count}.
def _derive_pool_buckets(model_config, batch_size: int, seq_len: int,
                         dtype_size: int = 2, num_layers: int = None,
                         cover_fp32_upcast: bool = False,
                         count_multiplier: float = 1.0,
                         scores_materialized: bool = True,
                         whole_layer_unit: bool = False,
                         staged_layers: int = None,
                         verbose: bool = True) -> dict[int, int]:
    hidden = model_config.hidden_size
    intermediate = model_config.intermediate_size
    num_heads = model_config.num_attention_heads
    num_kv_heads = model_config.num_key_value_heads
    head_dim = hidden // num_heads
    L = num_layers or model_config.num_hidden_layers
    # Fund only the layers that can actually reach the pool. --force_action
    # recompute stages nothing; funding the per-module seam's worst case
    # anyway pins 101.95 GiB per rank at L=64, x=33, and a pinned pool near
    # host-memory size hangs the node. The small ladder and a single
    # catch-all stay as insurance; named classes scale with this.
    staged = L if staged_layers is None else max(0, min(L, staged_layers))

    # (elements, saves per layer). Multiplicity is the number of tensors of
    # that shape one decoder layer can hand to pack_fn in a single forward,
    # so L x multiplicity buffers covers the worst case where every layer's
    # saves of that class are staged simultaneously — which is the actual
    # peak, since forward creates all of them before backward reads any.
    #
    # The seam's offer depends on the RECOMPUTE UNIT. Under --layer_checkpoint
    # the wrapper hides its own interior saves, so pack_fn sees exactly one
    # tensor per layer — the fp16 residual-stream boundary — and not one of
    # the classes below. Funding them anyway asks cudaHostAlloc for 340 GiB
    # at L=64, s=114688 and dies before step 1. The insurance sizing changes
    # with it: at that shape the per-module ladder alone is tens of GiB,
    # because its rungs and its count are set for a per-module seam that
    # offers 17 tensors a bundle.
    if whole_layer_unit:
        classes = [(batch_size * seq_len * hidden, 1)]         # residual boundary
        # Insurance, not funding. One tensor per layer can reach this seam, so
        # an unnamed class would REPLACE the boundary rather than add to it;
        # a handful of buffers keeps it served-and-named instead of
        # structurally impossible, which is what --evict_layout_census is for.
        ladder_count, ladder_ceiling = 8, 8 * 1024 * 1024
        fp32_variants = False
        catchall = (batch_size * seq_len * hidden * 4, 2)
    else:
        classes = [
            (batch_size * seq_len * num_kv_heads * head_dim, 2),   # K, V
            (batch_size * seq_len * hidden, 3),                    # input, Q, attn out
            (batch_size * seq_len * intermediate, 3),              # SwiGLU gate/up/down
        ]
        ladder_count, ladder_ceiling = 128, None
        fp32_variants = True
        catchall = (max(elems * 4 for elems, _ in classes) * 4,
                    max(1, staged // 4))
    scores = (batch_size * num_heads * seq_len * seq_len, 1)   # attention scores

    counts: dict[int, int] = {}

    def add(size_bytes: int, n: int) -> None:
        n = max(1, int(round(n * count_multiplier)))
        counts[size_bytes] = max(counts.get(size_bytes, 0), n)

    # Small ladder: cheap insurance for unnamed classes. 128 KiB x 128 is
    # 16 MiB — the whole ladder costs less than one 288 MiB buffer.
    #
    # 128 buffers per rung: Gate 0 asks with pack_fn's OBSERVED member list
    # (17 tensors per attention bundle, not the 4 the analytic model names),
    # so the ladder classes are requested per bundle, and a bundle whose
    # 128 KiB members can't be served fails Gate 0 entirely.
    small = 128 * 1024
    ladder_top = min(elems for elems, _ in classes) * dtype_size
    if ladder_ceiling is not None:
        ladder_top = min(ladder_top, ladder_ceiling)
    while small < ladder_top:
        add(small, ladder_count)
        small *= 2

    # Both dtype widths per class: bf16 as produced, fp32 for the ops that
    # upcast (RMSNorm and softmax both do). The hidden plane's fp32 variant
    # (RMSNorm's upcast) is the one fp32 width that reaches the pool, and the
    # member list asks for 3 per attention bundle, so it is provisioned at
    # FULL rate. The other two fp32 widths are provisioned at half rate, so
    # host memory is not pinned for widths with no demand.
    # Skipped entirely under the whole-layer unit: the RMSNorm upcast happens
    # INSIDE the checkpoint region and never reaches the seam.
    hidden_plane_elems = batch_size * seq_len * hidden
    for elems, mult in classes:
        if staged == 0:
            break
        add(elems * dtype_size, staged * mult)
        if not fp32_variants:
            continue
        if elems == hidden_plane_elems:
            add(elems * 4, staged * mult)
        else:
            add(elems * 4, -(-staged * mult // 2))

    # Catch-all for request classes that match no formula here. Without a
    # bucket above the named classes they would go from "requested, always
    # denied" to "structurally impossible". Sized at 4x the largest named
    # class, still below the fp32 scores class so that class stays behind
    # --pool_cover_attn_scores.
    # Under the whole-layer unit this is the boundary's fp32 width
    # at 2 buffers — the one plausible unnamed class, provisioned as a token
    # so a surprise arrives as a served-and-named request rather than as a
    # --strict_evict abort mid-run.
    add(*catchall)

    # --pool_cover_attn_scores. Attention scores are the largest class and
    # the worst migration candidate — a host round trip costs far more than
    # recomputing QK^T — so making them stageable is kept behind a flag.
    # No scores class exists under the memory-efficient kernel, so
    # honouring the flag there would pin GiB of host memory for a tensor
    # that is never created — and the small ladder below already serves the
    # logsumexp width that kernel does save. Ignored loudly rather than
    # silently, since the flag is normally policy-bearing.
    # The whole-layer unit is a second reason to ignore it — the
    # scores live inside the checkpoint region, so they are never offered to
    # pack_fn even in regime A, where they do exist.
    if cover_fp32_upcast and (not scores_materialized or whole_layer_unit):
        if verbose:
            reason = ("the whole-layer checkpoint region never offers its "
                      "interior attention scores to the seam"
                      if whole_layer_unit else
                      "the memory-efficient kernel materializes no "
                      "attention-score tensor to cover")
            print(f"  [Pool] --pool_cover_attn_scores ignored: {reason}.")
        cover_fp32_upcast = False

    if cover_fp32_upcast:
        # fp32 width only: HF's softmax upcast is what autograd saves, so no
        # bf16 scores-width request reaches the pool.
        #
        # Full coverage (L x 1), deliberately: a short count would make the
        # pool the gate again partway through the step.
        if staged:
            add(scores[0] * 4, staged * scores[1])

    return counts


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FSDP LLaMA + Memory Profiling")

    # Model architecture
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--intermediate_size', type=int, default=2816)
    parser.add_argument('--num_layers', type=int, default=16)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--num_kv_heads', type=int, default=4)
    parser.add_argument('--vocab_size', type=int, default=32000)
    # The fused-norm baseline. HF's LlamaRMSNorm retains 6h+4 bytes per token
    # per norm, so every span is quoted against a norm that does not. The two
    # modes are NOT interchangeable:
    # 'input' saves x and leaves the norm's OUTPUT evictable by the bundle
    # that consumes it; 'output' saves y, costs zero extra bytes because that
    # storage is the bundle's save anyway, and PINS it — which takes regime B's
    # migratable supply to zero. See fused_rmsnorm.py.
    parser.add_argument('--fused_rmsnorm', choices=['off', 'input', 'output'],
                        default='off',
                        help="Replace LlamaRMSNorm with a memory-lean autograd "
                             "Function. 'input' is the honest fused baseline "
                             "(what production fused kernels save); 'output' is "
                             "the leaner-peak variant that forecloses migration.")
    # Under the per-module recompute unit (attention, mlp) both norms live at
    # the SEAM between units and their saves survive. A whole-layer unit
    # absorbs them into the replay — removing the residue AND migration's
    # supply in one move, since a checkpointed layer retains only its input.
    # Combined with --placement_engine this selects the engine's
    # WHOLE-LAYER recompute unit — one 'layer' bundle per wrapper, owning the
    # single residual-stream boundary the outer seam sees. Alone it is the
    # static checkpointing baseline, observer-only.
    parser.add_argument('--layer_checkpoint', action='store_true',
                        help="Wrap each LlamaDecoderLayer in non-reentrant "
                             "activation checkpointing (classic per-layer "
                             "gradient checkpointing). With --placement_engine "
                             "this selects the whole-layer recompute unit.")
    # The offload-everything baseline, and the control that separates a
    # placement POLICY from its IMPLEMENTATION. Stock PyTorch, no criterion,
    # no gate, no pool. It sees byte-for-byte the same save set as the
    # engine's seam because the checkpoint region's inputs are saved by
    # _NoopSaveInputs BEFORE the inner hook context is entered, so an outer
    # saved-tensor context catches exactly one residual boundary per layer.
    parser.add_argument('--activation_offload', action='store_true',
                        help="Offload every checkpointed layer's saved "
                             "boundary to pinned host memory via PyTorch's "
                             "offload_wrapper. Requires --layer_checkpoint. "
                             "Unconditional: no criterion, no feasibility "
                             "gate, no pinned pool.")
    # The head/CE term is ~99% of the span's denominator (derivation in
    # chunked_ce.py). Chunking is by token COUNT, not chunk count, so one
    # chunk's logit chain is a constant and the term leaves the peak's slope
    # entirely rather than merely shrinking within it.
    parser.add_argument('--chunked_ce', type=int, default=0, metavar='TOKENS',
                        help="Compute the LM head and cross-entropy this many "
                             "tokens at a time, each chunk checkpointed. 0 "
                             "(default) keeps HuggingFace's unchunked loss.")
    # Off by default. Its job is the chunked-CE loss-curve control, which
    # compares per-step losses between two arms and is worthless unless both see the same weights and the same
    # tokens. FSDP is built without sync_module_states, so this also makes the
    # ranks agree with each other rather than merely with the other arm.
    parser.add_argument('--seed', type=int, default=-1,
                        help="Seed model init and the batch stream. -1 "
                             "(default) leaves both unseeded.")
    # LlamaConfig defaults use_cache to True, a KV cache no trainer keeps —
    # except under --layer_checkpoint, which forces it off because the replay
    # re-enters the layer after the cache was extended. With the cache on, a
    # checkpointed arm and an un-checkpointed one are two different models.
    parser.add_argument('--use_cache', choices=['on', 'off'], default='on',
                        help="Whether the model retains its KV cache. 'on' is "
                             "LlamaConfig's default; 'off' is what a real "
                             "trainer does. "
                             "--layer_checkpoint forces 'off' regardless.")
    # Pin the attention kernel. See
    # pin_attention_backend() for why this must not be left to the dispatcher.
    parser.add_argument('--attn_impl',
                        choices=['sdpa_math', 'sdpa_mem_efficient',
                                 'sdpa_flash', 'sdpa_auto', 'eager'],
                        default='sdpa_math',
                        help="Which attention kernel materializes the s x s "
                             "tensor. 'sdpa_math' (default) pins PyTorch to the "
                             "math kernel rather than leaving it to the "
                             "dispatcher's discretion. "
                             "'sdpa_mem_efficient' pins the cutlass efficient "
                             "kernel, which does NOT materialize the s x s "
                             "tensor; on sm_70 it requires --dtype fp16 (bf16 "
                             "kernels exist for sm80+ only) and it disables math "
                             "so a failed dispatch raises instead of falling back. "
                             "'sdpa_flash' pins FlashAttention-2 (sm80+), same "
                             "no-fallback rule. "
                             "'sdpa_auto' leaves the dispatcher free (kept for "
                             "compatibility). 'eager' uses HF's own attention, whose "
                             "fp32 softmax upcast makes the class LARGER — a "
                             "different constant, so do not mix it into a fit.")

    parser.add_argument('--native_gqa', action='store_true',
                        help="Pass k/v to SDPA at num_kv_heads width (enable_gqa) "
                             "instead of HF's repeat_kv expansion, so they are "
                             "saved at kv width. Needs a kernel with native GQA "
                             "(flash on sm80+); the preflight proves it.")

    # Training
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--grad_accum', type=int, default=1, metavar='N',
                        help="Micro-batches per optimizer step (gradient "
                             "accumulation). --batch_size is the MICRO-batch; "
                             "the step trains batch_size x N samples. The N "
                             "micro-batches are rows of ONE batch drawn as for "
                             "--batch_size batch_size*N, so an accumulated arm "
                             "sees the same tokens as its unaccumulated twin. "
                             "FSDP's default path: each micro-batch reduce-"
                             "scatters into the sharded grads (no no_sync, which "
                             "would hold full unsharded grads). Engine arms "
                             "refuse N>1: untested combination.")
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dtype', type=str, default='bf16', choices=['bf16', 'fp16'])

    # Profiling
    parser.add_argument('--profile_steps', type=str, default='all',
                        help="Which steps to profile: 'all', 'last', or '2,3,4'")
    parser.add_argument('--output_dir', type=str, default='profile_results')
    parser.add_argument('--torch_profile', action='store_true',
                        help="Also run PyTorch profiler (generates Chrome trace)")
    parser.add_argument('--cuda_snapshot', action='store_true',
                        help="Capture CUDA memory snapshot on last step")

    # Placement-engine CLI flags. Off by default.
    parser.add_argument('--placement_engine', action='store_true',
                        help="Attach the dry-run PlacementEngine. Adds "
                             "schema_version=2 sections to the JSON output.")
    parser.add_argument('--hbm_ballast_mb', type=float, default=0.0, metavar='MIB',
                        help="Hold MIB MiB of idle device memory for the whole run, in "
                             "--hbm_ballast_chunks equal tensors allocated before step 0. "
                             "Gives a mechanism another one's resident footprint without "
                             "its data flow (the A30 negative-tax test). "
                             "The ballast is counted in peak_mb; config records it.")
    parser.add_argument('--hbm_ballast_chunks', type=int, default=63,
                        help="Number of ballast tensors (default 63 = the boundaries "
                             "static keeps resident at L=64).")
    parser.add_argument('--hbm_cap_gb', type=float, default=0.0, metavar='GIB',
                        help="Cap the caching allocator at GIB GiB per GPU "
                             "(torch.cuda.set_per_process_memory_fraction), so a "
                             "large-HBM card behaves like a small one and the "
                             "frontier ladders stay short. 0 = no cap. The cap "
                             "bounds allocator memory only; the CUDA context "
                             "lives outside it. Spans are ratios of slopes and "
                             "cancel capacity, so a capped card reproduces them; "
                             "brackets belong to the capped configuration.")
    parser.add_argument('--hbm_budget_gb', type=float, default=14.0,
                        help="Engine HBM budget in GB. Default 14.0 "
                             "(V100 has 16 GB; 2 GB headroom for "
                             "allocator/NCCL overhead).")
    parser.add_argument('--prefetch_window_layers', type=int, default=2,
                        help="How many layers ahead the engine looks "
                             "for prefetch candidates. Default 2.")
    # Pinned-pool provisioning: a multiplier on the per-bucket counts that
    # _derive_pool_buckets sizes to each bucket's own per-layer save
    # multiplicity. Pool exhaustion (cpu_dram_full) forces recompute on the
    # decisions that reach the evict-vs-recompute gate, which makes the
    # placement criterion unobservable.
    parser.add_argument('--pool_buffers_per_bucket', type=float, default=1.0,
                        help="Multiplier on the per-bucket buffer counts "
                             "derived from per-layer save multiplicity. 1.0 "
                             "(default) covers the analytic worst case. Use "
                             ">1 to add headroom, <1 to starve the pool "
                             "deliberately for an ablation.")
    # Attention scores are the largest activation class and the worst
    # migration candidate — a host round trip costs far more than recomputing
    # QK^T. By default the pool provisions no bucket for them; this flag
    # makes them evictable and hands the choice to the criterion.
    parser.add_argument('--pool_cover_attn_scores', action='store_true',
                        help="Provision pinned buckets for the attention-score "
                             "class, making it evictable. Adds one fp32 "
                             "scores-sized pinned buffer per staged layer.")
    # Exposed-cost knob. Deflates the available overlap window so borderline
    # decisions tilt toward recompute when the wall-clock estimate is
    # uncertain.
    parser.add_argument('--overlap_safety_factor', type=float, default=0.8,
                        help="Deflation factor on _available_overlap_ms "
                             "(0..1). Lower = more recompute, higher = "
                             "more migration. Default 0.8.")
    # Switch MemoryStreamManager from no-op event recording to real
    # cudaMemcpyAsync. Off by default (dry run).
    parser.add_argument('--engine_live', action='store_true',
                        help="Run MemoryStreamManager in live mode "
                             "(real cudaMemcpyAsync instead of no-op "
                             "event recording). Default off.")

    # FSDP coordinator. 'fsdp' tracks actual FSDP parameter state
    # (params_resident). 'noop' returns params_resident=False
    # unconditionally, for ablations and CPU smoke tests where no FSDP
    # model exists. Kept for compatibility; default 'fsdp'.
    parser.add_argument('--coordinator', type=str, default='fsdp',
                        choices=['fsdp', 'noop'],
                        help="Which FSDP coordinator to attach. "
                             "Default 'fsdp' = real FSDPCoordinator; "
                             "'noop' = NoOpCoordinator for "
                             "ablation / smoke runs.")

    # FSDP CPU offload, and a switch that disables FSDP's BACKWARD_PRE
    # prefetch (kept for compatibility; off by default).
    parser.add_argument('--cpu_offload', action='store_true',
                        help="FSDP CPUOffload(offload_params=True): params and "
                             "grads live on the host, the optimizer steps on "
                             "CPU. The ZeRO-Offload analogue. To see its "
                             "traffic against the collectives, pair with "
                             "--torch_profile and read the trace with "
                             "tools/analyze_nccl_duty.py.")
    parser.add_argument('--no_backward_prefetch', action='store_true',
                        default=False,
                        help="Disable FSDP's BACKWARD_PRE prefetch.")

    # Accepted no-op, kept for compatibility: job scripts pass it, and the
    # planner always sees the raw HBM budget.
    parser.add_argument('--disable_hbm_drift', action='store_true',
                        default=False,
                        help="Accepted no-op, kept for compatibility. The "
                             "planner always sees the raw HBM budget. The "
                             "per-step observed-vs-projected peak LOG is "
                             "unaffected.")

    # The overlap window O(w) is a shared resource, debited per committed
    # eviction. This flag gives every decision the full window instead.
    parser.add_argument('--no_window_debit', action='store_true',
                        default=False,
                        help="Let every decision read the full overlap "
                             "window O(w) instead of a window debited per "
                             "committed eviction. Ablation control for the "
                             "window-debit mechanism; off by default.")

    # Snapshots report the peak of the interval since the previous snapshot,
    # which requires resetting CUDA's high-water mark at each one. Step-level
    # peaks are preserved by a shadow max in the profiler, so this flag is
    # the control for the resets, not a safety valve.
    parser.add_argument('--no_interval_peak', action='store_true',
                        default=False,
                        help="Disable per-interval peak attribution; "
                             "snapshot peak_mb reverts to the cumulative "
                             "step high-water mark and no mid-step reset of "
                             "the CUDA counter occurs. Ablation control.")

    # The peak projection's seven-component analytic byte model books less
    # relief for an attention bundle than its storages hold, so pack_fn
    # measures the owned bytes and the difference lands in a per-bundle
    # residual slot. This flag holds the residual at zero, which moves
    # projected_peak_bytes.
    parser.add_argument('--no_observed_relief', action='store_true',
                        default=False,
                        help="Book a bundle's relief in the peak projection "
                             "as the analytic per-component sum rather than "
                             "the storage bytes pack_fn observes it owns. "
                             "Ablation control for the observed-relief "
                             "mechanism.")

    # A denied pool acquire leaves the tensor resident after the planner has
    # booked its relief, so the projection and the allocator disagree by
    # those bytes. Below the memory ceiling that distorts measurements; above
    # the ceiling it IS the OOM. Denials are always logged and counted; this
    # flag makes them fatal, which is what a frontier run wants — a run that
    # cannot deliver the relief it planned should stop, not report a number
    # it did not earn.
    parser.add_argument('--strict_evict', action='store_true',
                        default=False,
                        help="Raise on any planned eviction the pinned pool "
                             "denies at pack time, instead of logging it and "
                             "leaving the tensor resident. Recommended for "
                             "runs above the memory ceiling.")

    # Force-action knob. Bypasses the exposed-cost criterion and the
    # planner's pressure/coverage logic; marks every reducible bundle for
    # the forced action every step. 'none' (default) = let the cost
    # criterion decide. 'evict' = pure-evict baseline. 'recompute'
    # = pure-recompute baseline.
    parser.add_argument('--force_action', type=str, default='none',
                        choices=['none', 'evict', 'recompute'],
                        help="Force every bundle's placement decision. "
                             "'none' (default) runs the exposed-cost "
                             "criterion. 'evict' forces pure-evict "
                             "baseline (always migrate). 'recompute' "
                             "forces pure-recompute baseline (always "
                             "checkpoint).")
    # Fixed migration stride on a recompute base.
    parser.add_argument('--force_stride', type=int, default=0,
                        help="With --force_action recompute, migrate every "
                             "k-th layer and recompute the rest. 0 (default) "
                             "= off. Criterion-free sparse-migration policy; "
                             "k=1 degenerates to pure evict and k>num_layers "
                             "to pure recompute.")

    # Restore-path diagnostics. A restore that hands backward a wrongly
    # laid-out tensor yields non-finite gradients while the forward loss
    # stays finite; these three flags locate such a fault by measurement.
    # The filter voids the arm's relief accounting and the verifier defeats
    # the relief outright, so neither belongs on a frontier run.
    parser.add_argument('--evict_filter', type=str, default='all',
                        help="Restrict which saves eviction may touch, to "
                             "bisect a restore fault. 'all' (default), "
                             "'attention' or 'mlp' for one bundle kind, "
                             "'bytes=N,M' for exact byte sizes only, or "
                             "'notbytes=N,M' for everything else. Excluded "
                             "saves stay resident and are counted apart from "
                             "denials, so they do NOT raise under "
                             "--strict_evict; the arm's realized-relief "
                             "numbers are void by construction.")
    parser.add_argument('--evict_layout_census', action='store_true',
                        default=False,
                        help="Record the original layout (contiguity, storage "
                             "offset, storage size) of every evicted save, by "
                             "size class. Free — no clone, no sync. A restore "
                             "that rebuilds with torch.empty(shape) brings a "
                             "strided or offset save back re-laid-out with "
                             "correct values, which a value comparison "
                             "cannot detect; the census records the layouts "
                             "that would expose it.")
    parser.add_argument('--verify_restore', action='store_true',
                        default=False,
                        help="Keep an HBM reference copy of every evicted "
                             "save and compare the restored tensor against it "
                             "at unpack, reporting value and layout "
                             "mismatches separately per size class. Defeats "
                             "the memory relief and serializes the step: run "
                             "at short sequence length as a diagnostic, never "
                             "as a frontier arm.")

    # Per-step memory-growth diagnostic. When on, writes one JSONL line per
    # step to mem_diag_rank<N>.jsonl in --output_dir with allocator stats,
    # optimizer-state bytes, and engine-side cache sizes.
    # --diag_snapshot_steps selects the steps that dump a CUDA memory
    # snapshot.
    parser.add_argument('--diagnose_memory_growth', action='store_true',
                        default=False,
                        help="Per-step memory-growth diagnostic JSONL "
                             "(allocator stats, optimizer-state bytes, "
                             "engine cache sizes).")
    parser.add_argument('--diag_snapshot_steps', type=str, default='',
                        help="Comma-separated step indices to dump CUDA "
                             "memory snapshots on (rank 0 only). Only "
                             "active when --diagnose_memory_growth is "
                             "set. Empty disables. Example: '0,9'.")

    # Forced PCIe/NCCL overlap probe. Saturates PCIe with synthetic transfers
    # while real collectives run, to measure whether overlapping host
    # transfers with the collectives is cheaper or dearer than serializing
    # them.
    parser.add_argument('--overlap_probe', action='store_true', default=False,
                        help="Saturate PCIe with synthetic transfers during "
                             "profiled steps to force overlap with NCCL "
                             "collectives. Measurement only; the buffers are "
                             "synthetic and never read by autograd.")
    parser.add_argument('--overlap_probe_duty', type=float, default=1.0,
                        help="Fraction of time the probe keeps the link busy. "
                             "1.0 = back-to-back (worst case, run this first). "
                             "Below 1.0 inserts spin-kernel gaps that occupy "
                             "an SM, so treat sub-1.0 results with care.")
    parser.add_argument('--overlap_probe_ms', type=float, default=2000.0,
                        help="Target duration of the probe train per armed "
                             "step. Should be near the step time so the probe "
                             "spans both forward and backward.")
    parser.add_argument('--overlap_probe_steps', type=str, default='profiled',
                        help="'profiled' (default) arms the probe only on "
                             "torch-profiled steps so the chrome trace "
                             "captures it; 'all'; or a comma-separated list.")

    args = parser.parse_args()

    # Fail a mistyped --evict_filter here, not after FSDP has
    # wrapped the model. The setter is the single source of the grammar, so
    # this validates against it rather than restating it.
    if args.evict_filter != 'all':
        from memory_manager.placement_engine import PlacementEngine
        PlacementEngine.evict_filter.fset(
            type('_Probe', (), {})(), args.evict_filter)

    diag_snapshot_step_set: set[int] = set()
    if args.diagnose_memory_growth and args.diag_snapshot_steps:
        diag_snapshot_step_set = {int(s) for s in args.diag_snapshot_steps.split(',') if s.strip()}

    # ── Init distributed ──────────────────────────────────────────────────────
    local_rank = init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Capacity cap: must precede every allocation, so it sits right after the
    # device is bound. Refuses a cap above the card rather than silently
    # running uncapped.
    if args.hbm_cap_gb > 0:
        total = torch.cuda.get_device_properties(local_rank).total_memory
        cap = int(args.hbm_cap_gb * 2**30)
        if cap > total:
            raise SystemExit(f"--hbm_cap_gb {args.hbm_cap_gb} exceeds this "
                             f"card's {total / 2**30:.2f} GiB")
        torch.cuda.set_per_process_memory_fraction(cap / total, local_rank)
        if rank == 0:
            print(f"[rank 0] HBM cap: allocator limited to {args.hbm_cap_gb} GiB "
                  f"of {total / 2**30:.2f} GiB ({cap / total:.4f} of the card)")

    # Parse which steps to profile
    if args.profile_steps == 'all':
        profile_step_set = set(range(args.steps))
    elif args.profile_steps == 'last':
        profile_step_set = {args.steps - 1}
    else:
        profile_step_set = set(int(x) for x in args.profile_steps.split(','))

    # Which steps the overlap probe arms on. 'profiled' tracks the
    # torch.profiler schedule (wait=1, warmup=1, active=N) rather than
    # profile_step_set, because the probe's value is being visible in the
    # chrome trace alongside the NCCL kernels it collides with.
    if not args.overlap_probe:
        overlap_probe_step_set: set[int] = set()
    elif args.overlap_probe_steps == 'all':
        overlap_probe_step_set = set(range(args.steps))
    elif args.overlap_probe_steps == 'profiled':
        overlap_probe_step_set = set(range(2, 2 + max(0, min(3, args.steps - 2))))
    else:
        overlap_probe_step_set = {
            int(x) for x in args.overlap_probe_steps.split(',') if x.strip()
        }

    if rank == 0:
        print(f"\n{'='*72}")
        print(f"  FSDP LLaMA + Memory Profiling")
        print(f"  {world_size} GPUs | NCCL | {args.dtype}")
        print(f"  batch={args.batch_size}  seq_len={args.seq_len}  steps={args.steps}")
        print(f"  Profile steps: {sorted(profile_step_set)}")
        if args.overlap_probe:
            print(f"  Overlap probe: steps {sorted(overlap_probe_step_set)} "
                  f"duty={args.overlap_probe_duty} "
                  f"target={args.overlap_probe_ms:.0f} ms")
        print(f"{'='*72}\n")

    # ── Build model on CPU ────────────────────────────────────────────────────
    attn_backend_info = pin_attention_backend(
        args.attn_impl, rank,
        dtype=(torch.bfloat16 if args.dtype == 'bf16' else torch.float16),
        head_dim=args.hidden_size // args.num_heads,
        n_q=args.num_heads, n_kv=args.num_kv_heads,
        native_gqa=args.native_gqa)
    model, model_config = build_model(args)
    attn_backend_info['resolved'] = getattr(
        model.config, '_attn_implementation', 'unknown')

    # The fused-norm baseline. Must run before FSDP wrapping — the
    # parameters are reused so the flat param is unchanged, but the module tree
    # is read at wrap time. The replacement count is printed on every rank
    # because an arm that silently kept HF's norm would look like a null result.
    fused_norm_count = 0
    if args.fused_rmsnorm != 'off':
        from fused_rmsnorm import install_fused_rmsnorm
        fused_norm_count = install_fused_rmsnorm(model, mode=args.fused_rmsnorm)
        print(f"[rank {rank}] fused RMSNorm mode={args.fused_rmsnorm}: "
              f"{fused_norm_count} norms replaced", flush=True)

    # Set before the checkpoint block so the forced-off override
    # below is visibly an override and not a silent disagreement.
    model.config.use_cache = (args.use_cache == 'on')
    effective_use_cache = model.config.use_cache

    # Classic per-layer checkpointing, the static baseline. Applied before
    # FSDP so the wrapper sits inside the FSDP unit.
    layer_ckpt_count = 0
    # With --placement_engine the two combine, and combining them IS the
    # whole-layer recompute unit. The hooks nest rather than conflict — the
    # engine's context wraps the whole forward, the wrapper installs its own
    # inside, and only the innermost pair sees a save, so the engine's seam
    # sees exactly the per-layer residual boundary
    # (tests/test_checkpoint_seam_visibility.py).
    if args.activation_offload and not args.layer_checkpoint:
        raise ValueError(
            "--activation_offload requires --layer_checkpoint. Without the "
            "wrapper there is no per-layer boundary to offload: the outer "
            "saved-tensor context would catch every interior save instead, "
            "which is a different mechanism and not comparable to the engine.")

    if args.layer_checkpoint:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            apply_activation_checkpointing, checkpoint_wrapper, CheckpointImpl,
            offload_wrapper)
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        layer_ckpt_count = sum(
            isinstance(m, LlamaDecoderLayer) for m in model.modules())
        # Mandatory, not a tuning choice: the replay re-enters the layer after
        # the cache has already been extended, and the shapes no longer match.
        # It also removes the KV cache from this arm's per-layer mass (~4
        # MiB/layer at s=4096), so arms compared against it must pass
        # --use_cache off.
        if args.use_cache == 'on':
            print(f"[rank {rank}] OVERRIDE: --layer_checkpoint forces "
                  f"use_cache=off. Any arm compared against this one must pass "
                  f"--use_cache off or the comparison is confounded.", flush=True)
        model.config.use_cache = False
        effective_use_cache = False
        ckpt_fn = functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
        if args.activation_offload:
            # OUTSIDE the checkpoint wrapper, which is the only order that
            # offloads the boundary rather than the interior: the inner hook
            # pair shadows every save made within the region, so the outer
            # context sees only what _NoopSaveInputs saved before it.
            _inner_ckpt_fn = ckpt_fn
            ckpt_fn = lambda m: offload_wrapper(_inner_ckpt_fn(m))
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=ckpt_fn,
            check_fn=lambda m: isinstance(m, LlamaDecoderLayer))
        print(f"[rank {rank}] layer checkpointing: {layer_ckpt_count} "
              f"LlamaDecoderLayer wrapped"
              f"{', each offloaded to pinned host memory' if args.activation_offload else ''}",
              flush=True)

    # Also before FSDP, and for the same reason — the
    # replacement forward must run INSIDE the root FSDP unit so lm_head's
    # parameters are gathered when the chunks ask for them.
    if args.chunked_ce > 0:
        import chunked_ce
        chunked_ce.install(model, args.chunked_ce)
        print(f"[rank {rank}] chunked CE: {args.chunked_ce} tokens per chunk, "
              f"{-(-args.seq_len * args.batch_size // args.chunked_ce)} chunks, "
              f"peak logit chain "
              f"{chunked_ce.peak_logit_bytes(args.chunked_ce, args.vocab_size)/1024**2:.2f}"
              f" MiB (sequence-length independent)", flush=True)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        total_mb = sum(p.numel() * p.element_size()
                       for p in model.parameters()) / 1024**2
        print(f"  Full model: {total_params:,} params ({total_mb:.1f} MB in fp32)")

    # ── Pre-flight memory estimate ────────────────────────────────────────────
    if rank == 0:
        dtype_size = 2  # bf16/fp16
        param_count = sum(p.numel() for p in model.parameters())
        # Per-GPU after FSDP sharding
        sharded_params_mb = param_count * dtype_size / world_size / (1024**2)
        # Optimizer: Adam stores fp32 master + 2 moments, on sharded params
        sharded_optim_mb = param_count * (4 + 4 + 4) / world_size / (1024**2)
        # Gradients (sharded)
        sharded_grad_mb = param_count * dtype_size / world_size / (1024**2)
        # Activations (NOT sharded — each GPU computes full activations for its batch)
        bs = args.batch_size
        sl = args.seq_len
        per_layer_act = (
            bs * args.num_heads * sl * sl * dtype_size         # attn scores
            + 3 * bs * sl * args.intermediate_size * dtype_size  # SwiGLU
            + bs * sl * (args.num_heads + 2 * args.num_kv_heads)
              * (args.hidden_size // args.num_heads) * dtype_size  # QKV
            + bs * sl * args.hidden_size * dtype_size              # layer input
        )
        total_act_mb = per_layer_act * args.num_layers / (1024**2)
        # FSDP temp buffer: one full layer unsharded during compute
        fsdp_buffer_mb = param_count * dtype_size / args.num_layers / (1024**2)

        est_total = sharded_params_mb + sharded_optim_mb + sharded_grad_mb + total_act_mb + fsdp_buffer_mb
        print(f"\n  --- Pre-flight Estimate (per GPU) ---")
        print(f"  Sharded params:    {sharded_params_mb:>8.1f} MB")
        print(f"  Sharded optimizer: {sharded_optim_mb:>8.1f} MB")
        print(f"  Sharded gradients: {sharded_grad_mb:>8.1f} MB")
        print(f"  Activations:       {total_act_mb:>8.1f} MB (NOT sharded)")
        print(f"  FSDP buffer:       {fsdp_buffer_mb:>8.1f} MB (1 layer unsharded)")
        print(f"  TOTAL:             {est_total:>8.1f} MB")
        print(f"  {'OK' if est_total < 15000 else 'LIKELY OOM'} for 16GB V100\n")

    # ── FSDP wrap ─────────────────────────────────────────────────────────────
    model = wrap_with_fsdp(model, local_rank, args)

    if rank == 0:
        print("  ✓ FSDP wrapping complete")

    # ── Sharding proof ────────────────────────────────────────────────────────
    print_sharding_proof(model, rank, world_size)
    dist.barrier(device_ids=[local_rank])
    print_gpu_memory(rank, label="after FSDP wrap")
    dist.barrier(device_ids=[local_rank])

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    # ── Profiler ──────────────────────────────────────────────────────────────
    scores_materialized = materializes_attn_scores(args.attn_impl)
    if rank == 0 and not scores_materialized:
        print(f"  [Attention] byte model: no s x s scores tensor "
              f"(--attn_impl {args.attn_impl})")
    profiler = FSDPMemoryProfiler(
        model, rank, world_size, model_config, enabled=True,
        materializes_attn_scores=scores_materialized,
    )
    # --no_interval_peak control: with per-interval peaks off the snapshots
    # report a cumulative staircase and no CUDA counter is reset mid-step.
    profiler._interval_peak_enabled = not args.no_interval_peak
    profiler.attach_hooks()

    # Construct and attach PlacementEngine when --placement_engine is passed. Default flow (flag off) bypasses
    # this entire block, leaving the profiler in observer-only mode.
    engine = None
    if args.placement_engine:
        from memory_manager import (
            AccessPredictor, HBMBudget, MemoryStreamManager,
            PinnedBufferPool, PlacementEngine,
            FSDPCoordinator, NoOpCoordinator,
        )
        from memory_manager.calibration import run_calibration

        if rank == 0:
            print(f"\n  [Engine] Constructing dry-run PlacementEngine...")

        # Calibration: pcie / round-trip / fp16 flops. Sized for V100.
        calibration = run_calibration(verbose=(rank == 0))

        # Predictor: same number of layers and seq_len that the training
        # loop will use. Without --layer_checkpoint the policy starts empty
        # and the engine flips layers during pressure handling. WITH it, every
        # layer is wrapped before the first step, so the policy is all-True
        # from the start: leaving it empty would model all L layer interiors
        # as simultaneously resident and over-project the peak ~L-fold.
        # The architecture knobs populate the predictor's per-tensor byte
        # model; the planner in on_pre_forward(L) needs them to project HBM
        # peaks.
        predictor = AccessPredictor(
            num_layers=model_config.num_hidden_layers,
            seq_len=args.seq_len,
            verbose=(rank == 0),
            hidden_size=model_config.hidden_size,
            intermediate_size=model_config.intermediate_size,
            num_heads=model_config.num_attention_heads,
            num_kv_heads=model_config.num_key_value_heads,
            batch_size=args.batch_size,
            dtype_bytes=(2 if args.dtype in ('bf16', 'fp16') else 4),
            materializes_attn_scores=scores_materialized,
            vocab_size=model_config.vocab_size,
            chunked_ce_tokens=args.chunked_ce,
            bundle_kinds=(('layer',) if args.layer_checkpoint
                          else ('attention', 'mlp')),
            checkpoint_policy=(
                {L: True for L in range(model_config.num_hidden_layers)}
                if args.layer_checkpoint else None),
        )

        # Pinned pool: sized from expected activation sizes (registry is
        # still empty of activations at this point — they're created on
        # the first forward pass).
        bucket_counts = _derive_pool_buckets(
            model_config, args.batch_size, args.seq_len,
            dtype_size=(2 if args.dtype in ('bf16', 'fp16') else 4),
            num_layers=args.num_layers,
            cover_fp32_upcast=args.pool_cover_attn_scores,
            count_multiplier=args.pool_buffers_per_bucket,
            scores_materialized=scores_materialized,
            whole_layer_unit=args.layer_checkpoint,
            # Forced recompute stages nothing; a stride stages every k-th layer.
            staged_layers=(
                0 if args.force_action == 'recompute' and args.force_stride == 0
                else len(range(0, args.num_layers, args.force_stride))
                if args.force_action == 'recompute'
                else None),
            verbose=(rank == 0),
        )
        # bucket_counts is a {size: count} mapping.
        pool = PinnedBufferPool(bucket_sizes_bytes=bucket_counts)

        mem_stream = MemoryStreamManager(
            device=torch.device(f"cuda:{local_rank}"),
            # dry_run is the inverse of --engine_live ("--engine_live ON
            # => dry_run OFF").
            dry_run=(not args.engine_live),
            # Calibration is threaded into the stream manager so submit_* can stamp predicted-ms
            # values onto every InflightOp. The engine compares these
            # against observed GPU-side ms at reap time to flag drift.
            calibration=calibration,
        )
        if rank == 0:
            print(f"  [Engine] MemoryStreamManager dry_run="
                  f"{mem_stream.dry_run} (engine_live={args.engine_live})")

        budget = HBMBudget(
            total_bytes=int(args.hbm_budget_gb * (1024 ** 3)),
            registry=profiler.registry,
        )

        # Default coordinator is the real FSDPCoordinator; --coordinator
        # noop reports params_resident=False unconditionally, for ablation
        # runs. `model` is the FSDP-wrapped module from wrap_with_fsdp().
        if args.coordinator == 'fsdp':
            coord = FSDPCoordinator(
                fsdp_model=model, registry=profiler.registry,
                world_size=world_size, verbose=(rank == 0))
        elif args.coordinator == 'noop':
            coord = NoOpCoordinator(
                registry=profiler.registry,
                world_size=world_size, verbose=(rank == 0))
        else:
            raise ValueError(f"unknown --coordinator: {args.coordinator!r}")

        engine = PlacementEngine(
            registry=profiler.registry,
            predictor=predictor,
            hbm_budget=budget,
            buffer_pool=pool,
            mem_stream=mem_stream,
            calibration=calibration,
            fsdp_coord=coord,
            prefetch_window_layers=args.prefetch_window_layers,
            dry_run=True,
            verbose=(rank == 0),
            # Exposed-cost knob.
            overlap_safety_factor=args.overlap_safety_factor,
            # O(w) as a consumed budget.
            window_debit=not args.no_window_debit,
            # Peak relief from observed owned storage bytes.
            observed_relief=not args.no_observed_relief,
            # --layer_checkpoint selects the whole-layer unit.
            bundle_unit=('whole_layer' if args.layer_checkpoint
                         else 'per_module'),
        )
        if rank == 0:
            if args.layer_checkpoint:
                print("  [Engine] recompute unit = WHOLE LAYER — one 'layer' "
                      "bundle per CheckpointWrapper; attention/mlp bundles "
                      "suppressed. Expect exactly num_layers boundaries and "
                      "num_layers zero-element placeholders per step.")
            print(f"  [Engine] window_debit={not args.no_window_debit} "
                  f"({'O(w) debited per committed eviction' if not args.no_window_debit else 'full O(w) per decision (control)'})")
            print(f"  [Engine] observed_relief="
                  f"{not args.no_observed_relief} "
                  f"({'peak relief from observed owned storage bytes' if not args.no_observed_relief else 'analytic per-component sum (control)'})")

        # Wire the live-offload switch onto the engine itself. The MemoryStreamManager already knows
        # via its dry_run kwarg; the engine reads engine_live separately
        # so pack_fn / unpack_fn can gate on it.
        engine.engine_live = args.engine_live
        if rank == 0:
            print(f"  [Engine] engine_live={engine.engine_live} "
                  f"(pack_fn/unpack_fn {'active' if engine.engine_live else 'inert'})")

        # --strict_evict: make a denied eviction fatal.
        engine._strict_evict = args.strict_evict
        if rank == 0:
            print(f"  [Engine] strict_evict={args.strict_evict} "
                  f"({'a denied eviction raises' if args.strict_evict else 'a denied eviction is logged and counted'})")

        # Forced policy. Under the whole-layer unit the engine's recompute action
        # cannot reproduce a layer's input, so a pure-recompute arm would be a
        # no-op wearing a baseline's label. Die here rather than spend the job.
        if args.grad_accum > 1:
            raise SystemExit(
                "--grad_accum > 1 with --placement_engine is untested: the "
                "engine's per-step lifecycle and pick accounting assume one "
                "forward/backward per step. Run accumulation arms without the "
                "engine.")
        if (args.layer_checkpoint and args.force_action == 'recompute'
                and args.force_stride == 0):
            raise SystemExit(
                "--force_action recompute is a no-op under --layer_checkpoint: "
                "the checkpoint wrapper already owns every recompute, and the "
                "one save left at the engine's seam is the layer's INPUT, "
                "which replaying that same layer cannot reproduce. The "
                "whole-layer recompute baseline is the wrapper on its own — "
                "run with --layer_checkpoint and WITHOUT --placement_engine "
                "--engine_live. (--force_stride > 0 is still allowed: it "
                "becomes an eviction-density sweep with the non-evicted "
                "layers retained.)")
        engine.force_action = args.force_action
        if rank == 0:
            mode_desc = {
                'none': 'exposed-cost criterion (ours)',
                'evict': 'pure-evict baseline (always migrate)',
                'recompute': 'pure-recompute baseline (always checkpoint)',
            }[args.force_action]
            print(f"  [Engine] force_action={args.force_action} ({mode_desc})")
        engine.force_stride = args.force_stride
        if rank == 0 and args.force_stride > 0:
            n_mig = len(range(0, args.num_layers, args.force_stride))
            print(f"  [Engine] force_stride={args.force_stride} "
                  f"(migrate {n_mig}/{args.num_layers} layers, "
                  f"recompute the rest)")

        # Restore-path diagnostics. Loud on rank 0 when armed —
        # an arm that silently carries a filter or a verifier would report
        # relief numbers nobody could interpret later.
        engine.evict_filter = args.evict_filter
        engine.layout_census = args.evict_layout_census
        engine.verify_restore = args.verify_restore
        if rank == 0 and args.evict_filter != 'all':
            print(f"  [Engine] evict_filter={args.evict_filter} "
                  f"(DIAGNOSTIC: excluded saves stay resident; this arm's "
                  f"realized-relief and absence numbers are void)")
        if rank == 0 and args.evict_layout_census:
            print("  [Engine] evict_layout_census=True "
                  "(original layout of every evicted save, free)")
        if rank == 0 and args.verify_restore:
            print("  [Engine] verify_restore=True "
                  "(DIAGNOSTIC: reference copy per evicted save — defeats "
                  "the relief and serializes the step)")

        profiler.attach_engine(engine)

    # ── Optional: PyTorch profiler for Chrome trace ───────────────────────────
    torch_profiler = None
    if args.torch_profile:
        torch_profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1, warmup=1, active=min(3, args.steps - 2), repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.path.join(args.output_dir, f"torch_trace_rank{rank}")
            ),
            # record_shapes/profile_memory/with_stack are off for the
            # duty-cycle measurement. Their CPU-side bookkeeping inflates
            # inter-kernel gaps, which biases NCCL-busy-fraction downward and
            # stretches memcpy durations (i.e. understates achieved bandwidth).
            # Re-enable only if a run needs shape/stack attribution.
        )

    # ── Optional: forced PCIe/NCCL overlap probe ──────────────────────────────
    overlap_probe = None
    if args.overlap_probe:
        from memory_manager.overlap_probe import OverlapProbe
        overlap_probe = OverlapProbe(
            device=torch.device(f"cuda:{local_rank}"),
            duty=args.overlap_probe_duty,
        )
        if rank == 0:
            print(f"  [Probe] buffers={overlap_probe.buffer_bytes/1e6:.1f} MB "
                  f"(device + pinned host)")

    # ── Optional: CUDA memory snapshot ────────────────────────────────────────
    cuda_snapshot_enabled = args.cuda_snapshot

    # ── Training loop ─────────────────────────────────────────────────────────
    if rank == 0:
        print(f"\n{'─'*72}")
        print(f"  Step | Rank | Loss     | Avg Loss | Peak MB | Step ms")
        print(f"{'─'*72}")

    os.makedirs(args.output_dir, exist_ok=True)
    step_results = []

    # Per-step memory-growth diagnostic JSONL.
    diag_jsonl_path = None
    diag_jsonl_fh = None
    if args.diagnose_memory_growth:
        diag_jsonl_path = os.path.join(args.output_dir, f"mem_diag_rank{rank}.jsonl")
        diag_jsonl_fh = open(diag_jsonl_path, 'w')
        if rank == 0:
            print(f"  [Diag] memory-growth JSONL → {diag_jsonl_path}")
            if diag_snapshot_step_set:
                print(f"  [Diag] CUDA snapshot steps: {sorted(diag_snapshot_step_set)}")

    ballast = []
    if args.hbm_ballast_mb > 0:
        per = int(args.hbm_ballast_mb * 2**20) // max(args.hbm_ballast_chunks, 1)
        ballast = [torch.zeros(per, dtype=torch.uint8, device=f"cuda:{local_rank}")
                   for _ in range(max(args.hbm_ballast_chunks, 1))]
        torch.cuda.synchronize()
        if rank == 0:
            print(f"  [Ballast] holding {len(ballast)} x {per / 2**20:.2f} MiB "
                  f"= {len(ballast) * per / 2**20:.2f} MiB idle on the device")

    if torch_profiler:
        torch_profiler.start()

    # Offset by rank so the arm keeps data-parallel semantics (each rank a
    # different batch) while still replaying token-for-token across arms.
    batch_gen = None
    if args.seed >= 0:
        batch_gen = torch.Generator(device=f"cuda:{local_rank}")
        batch_gen.manual_seed(args.seed + 1000 * rank)

    if args.grad_accum < 1:
        raise SystemExit("--grad_accum must be >= 1")
    for step in range(args.steps):
        # One draw of batch_size x grad_accum rows, split into micro-batches,
        # so an accumulated arm sees exactly its unaccumulated twin's tokens.
        # At grad_accum=1 this is a single draw of batch_size rows.
        full_ids = get_batch(args.batch_size * args.grad_accum, args.seq_len,
                             args.vocab_size, device=f"cuda:{local_rank}",
                             generator=batch_gen)
        micro_batches = (full_ids.split(args.batch_size, dim=0)
                         if args.grad_accum > 1 else (full_ids,))

        do_profile = step in profile_step_set
        profiler.enabled = do_profile

        # Start CUDA snapshot on last step if requested
        if cuda_snapshot_enabled and step == args.steps - 1:
            try:
                torch.cuda.memory._record_memory_history(
                    enabled='all', context='all', stacks='all')
            except AttributeError:
                cuda_snapshot_enabled = False

        # Per-step diag CUDA snapshot (rank 0 only).
        diag_snapshot_active = (
            args.diagnose_memory_growth and rank == 0 and step in diag_snapshot_step_set
        )
        if diag_snapshot_active:
            try:
                torch.cuda.memory._record_memory_history(
                    enabled='all', context='all', stacks='all')
            except AttributeError:
                diag_snapshot_active = False

        # Arm the probe BEFORE the step timer so the enqueue cost
        # (one launch per op) stays out of step_ms. The transfers themselves
        # do land inside the step, which is the point -- their effect on
        # step_ms is one of the things being measured.
        probe_armed = 0
        if overlap_probe is not None and step in overlap_probe_step_set:
            probe_armed = overlap_probe.arm(step, args.overlap_probe_ms)

        torch.cuda.reset_peak_memory_stats()
        step_start = time.perf_counter()

        # Engine step lifecycle.
        # Called outside profile_step() so it runs regardless of
        # whether THIS step is being snapshot-profiled. The engine's
        # own per-hook callbacks remain gated on profiler.enabled.
        if engine is not None:
            engine.on_training_step_begin()

        with profiler.profile_step(step):
            optimizer.zero_grad(set_to_none=True)

            # Install pack/unpack
            # saved-tensor hooks for the duration of forward when
            # --engine_live is on. The hooks only affect tensors saved
            # *inside* the context (PyTorch stores the unpack_fn alongside
            # each packed tensor, so it remains callable during backward
            # even after the context exits).
            if engine is not None and engine.engine_live:
                pack_ctx = torch.autograd.graph.saved_tensors_hooks(
                    engine.pack_fn, engine.unpack_fn)
            else:
                pack_ctx = contextlib.nullcontext()

            if args.grad_accum == 1:
                input_ids = full_ids
                # Forward
                with pack_ctx:
                    outputs = model(input_ids=input_ids, labels=input_ids)
                    loss = outputs.loss

                profiler.mark_phase("forward_end")

                # Backward
                loss.backward()

                profiler.mark_phase("backward_end")
            else:
                # Equal-length micro-batches, each loss a token mean, so the
                # mean of the micro losses is the full batch's token mean and
                # dividing each backward by N reproduces its gradient.
                loss_sum = None
                for input_ids in micro_batches:
                    outputs = model(input_ids=input_ids, labels=input_ids)
                    micro_loss = outputs.loss
                    profiler.mark_phase("forward_end")
                    (micro_loss / args.grad_accum).backward()
                    profiler.mark_phase("backward_end")
                    d = micro_loss.detach()
                    loss_sum = d if loss_sum is None else loss_sum + d
                    del outputs, micro_loss
                loss = loss_sum / args.grad_accum

            # Grad norm
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.detach().float().norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5

            # Optimizer
            optimizer.step()

            profiler.mark_phase("optimizer_end")

        # Push the per-layer wall-clock timings the
        # profiler just collected into the engine so the next step's
        # overlap-window math uses real numbers instead of defaults.
        # Done BEFORE on_training_step_end so the populate_registry log
        # line at end-of-step reflects the warmed table.
        if engine is not None:
            for L, prof in profiler.layer_profiles.items():
                if prof.fwd_duration_ms > 0:
                    engine.record_layer_timing(L, 'fwd', prof.fwd_duration_ms)
                if prof.bwd_duration_ms > 0:
                    engine.record_layer_timing(L, 'bwd', prof.bwd_duration_ms)

        # Read the step peak BEFORE the engine's end-of-step drain, which
        # resets CUDA's high-water mark: read after it, steps_rank<N>.json's
        # peak_mb would hold only the bytes allocated during the drain itself
        # and disagree with the engine's observed_peak_mb. Goes through the
        # profiler so the per-interval resets are folded back in.
        peak_mb = profiler.step_peak_bytes() / (1024**2)

        # End-of-step engine drain + log.
        if engine is not None:
            engine.on_training_step_end()

        step_time = (time.perf_counter() - step_start) * 1000

        # Reap AFTER step_time is taken. If the probe train
        # outlives the step, that wait is overrun, not step cost.
        probe_overrun_ms = 0.0
        if probe_armed:
            probe_overrun_ms = overlap_probe.reap()

        # Avg loss across ranks
        avg_loss = loss.detach().clone()
        dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)

        print(
            f"  {step+1:>4} | {rank:>4} | {loss.item():.4f}   | "
            f"{avg_loss.item():.4f}   | {peak_mb:>7.1f} | {step_time:>7.1f}"
        )

        step_results.append({
            'step': step,
            'loss': loss.item(),
            'avg_loss': avg_loss.item(),
            'grad_norm': grad_norm,
            'peak_mb': peak_mb,
            'step_ms': round(step_time, 1),
            'profiled': do_profile,
            'probe_ops': probe_armed,
            'probe_overrun_ms': round(probe_overrun_ms, 1),
            # Device bytes the caching allocator does not own (CUDA context,
            # NCCL, cuBLAS handles) at step end. The uncapped capacity is
            # C0 = card - this - delta, so a card's native bracket can be
            # predicted from its own capped ladders plus this.
            'outside_alloc_mb': round(_outside_allocator_mb(), 2),
            # Cumulative caching-allocator counters at step end (the A30
            # negative-tax test). The reader diffs them per step:
            # retries mean a synchronizing free-and-retry, device alloc/free are
            # cudaMalloc/cudaFree calls.
            'alloc': _alloc_counters(),
        })

        # Per-step memory-growth diagnostic record.
        if diag_jsonl_fh is not None:
            stats = torch.cuda.memory_stats()
            optim_state_bytes = 0
            for state in optimizer.state.values():
                for v in state.values():
                    if torch.is_tensor(v):
                        optim_state_bytes += v.element_size() * v.numel()

            engine_fields = {
                'registry_records': None,
                'recompute_cache_entries': None,
                'recompute_cache_mb': None,
                'packed_evict_marks': None,
                'packed_recompute_marks': None,
            }
            if engine is not None:
                cache_bytes = 0
                for entries in engine._recompute_tensor_cache.values():
                    for t in entries:
                        if torch.is_tensor(t):
                            cache_bytes += t.element_size() * t.numel()
                engine_fields = {
                    'registry_records': len(engine._registry._records),
                    'recompute_cache_entries': len(engine._recompute_tensor_cache),
                    'recompute_cache_mb': cache_bytes / (1024 ** 2),
                    'packed_evict_marks': sum(1 for v in engine._packed_evict_bundles.values() if v),
                    'packed_recompute_marks': sum(1 for v in engine._packed_recompute_bundles.values() if v),
                }

            diag_record = {
                'step': step,
                'peak_mb': peak_mb,
                'allocated_current_mb': stats.get('allocated_bytes.all.current', 0) / (1024 ** 2),
                'allocated_peak_mb': stats.get('allocated_bytes.all.peak', 0) / (1024 ** 2),
                'reserved_current_mb': stats.get('reserved_bytes.all.current', 0) / (1024 ** 2),
                'reserved_peak_mb': stats.get('reserved_bytes.all.peak', 0) / (1024 ** 2),
                'active_current_mb': stats.get('active_bytes.all.current', 0) / (1024 ** 2),
                'allocation_count': stats.get('allocation.all.current', 0),
                'optim_state_mb': optim_state_bytes / (1024 ** 2),
                **engine_fields,
            }
            diag_jsonl_fh.write(json.dumps(diag_record) + '\n')
            diag_jsonl_fh.flush()

        if torch_profiler:
            torch_profiler.step()

        # Save CUDA snapshot
        if cuda_snapshot_enabled and step == args.steps - 1:
            try:
                snapshot = torch.cuda.memory._snapshot()
                torch.cuda.memory._record_memory_history(enabled=None)
                import pickle
                snap_path = os.path.join(args.output_dir,
                                         f"cuda_snapshot_rank{rank}.pkl")
                with open(snap_path, 'wb') as f:
                    pickle.dump(snapshot, f)
                if rank == 0:
                    print(f"\n  [CUDA snapshot] Saved to {snap_path}")
            except Exception as e:
                if rank == 0:
                    print(f"\n  [CUDA snapshot] Failed: {e}")

        # Per-step diag CUDA snapshot save (rank 0 only).
        if diag_snapshot_active:
            try:
                snapshot = torch.cuda.memory._snapshot()
                torch.cuda.memory._record_memory_history(enabled=None)
                import pickle
                snap_path = os.path.join(args.output_dir,
                                         f"cuda_snapshot_rank{rank}_step{step}.pkl")
                with open(snap_path, 'wb') as f:
                    pickle.dump(snapshot, f)
                print(f"\n  [Diag snapshot] Saved step {step} → {snap_path}")
            except Exception as e:
                print(f"\n  [Diag snapshot] Failed at step {step}: {e}")

        dist.barrier(device_ids=[local_rank])

    if torch_profiler:
        torch_profiler.stop()

    # Overlap-probe results. Its own CUDA-event timings are the
    # authoritative record of probe bandwidth; the chrome trace supplies the
    # other half of the ledger (whether NCCL kernels slowed down alongside).
    if overlap_probe is not None:
        probe_path = os.path.join(args.output_dir, f"probe_rank{rank}.json")
        overlap_probe.save(probe_path)
        if rank == 0:
            s = overlap_probe.summary()
            print(f"\n  [Probe] {s['n_ops']} ops @ duty={s['duty']}")
            for d in ('d2h', 'h2d'):
                if d in s:
                    v = s[d]
                    print(f"  [Probe] {d.upper()}: n={v['n']} GB/s "
                          f"p10={v['p10']:.2f} med={v['median']:.2f} "
                          f"p90={v['p90']:.2f}")
            print(f"  [Probe] → {probe_path}")

    # Close per-step diag JSONL.
    if diag_jsonl_fh is not None:
        diag_jsonl_fh.close()
        if rank == 0:
            print(f"  [Diag] JSONL closed: {diag_jsonl_path}")

    # ── Final memory state ────────────────────────────────────────────────────
    dist.barrier(device_ids=[local_rank])
    print_gpu_memory(rank, label="after training")

    # ── Generate and save profiler report ─────────────────────────────────────
    dist.barrier(device_ids=[local_rank])
    if rank == 0:
        print()
    profiler.report()
    profiler.save(args.output_dir)

    # Engine end-of-run summary.
    if engine is not None and rank == 0:
        engine.print_summary()

    # ── Save step results ─────────────────────────────────────────────────────
    results_path = os.path.join(args.output_dir, f"steps_rank{rank}.json")
    with open(results_path, 'w') as f:
        json.dump({
            'config': {
                'hidden_size': args.hidden_size,
                'intermediate_size': args.intermediate_size,
                'num_layers': args.num_layers,
                'num_heads': args.num_heads,
                'num_kv_heads': args.num_kv_heads,
                'batch_size': args.batch_size,
                'grad_accum': args.grad_accum,
                'hbm_cap_gb': args.hbm_cap_gb,
                'hbm_ballast_mb': args.hbm_ballast_mb,
                'hbm_ballast_chunks': args.hbm_ballast_chunks if args.hbm_ballast_mb > 0 else 0,
                'alloc_conf': os.environ.get('PYTORCH_CUDA_ALLOC_CONF', ''),
                'seq_len': args.seq_len,
                'vocab_size': args.vocab_size,
                'dtype': args.dtype,
                'world_size': world_size,
                # Which kernel produced the s x s class this run's peak depends
                # on. Recorded so a fit is never made across two of them.
                'attention': attn_backend_info,
                # The norm implementation moves the peak slope AND
                # decides whether the bundle's input is evictable, so a peak is
                # not comparable across modes. Count is proof the swap happened.
                'fused_rmsnorm': args.fused_rmsnorm,
                'native_gqa': args.native_gqa,
                'fused_rmsnorm_replaced': fused_norm_count,
                'layer_checkpoint': args.layer_checkpoint,
                'layer_checkpoint_wrapped': layer_ckpt_count,
                # The offload baseline. An arm carrying this is
                # migrating every boundary unconditionally with no criterion,
                # so its peak is comparable to the engine's but its traffic
                # and its step time are not the engine's to claim.
                'activation_offload': args.activation_offload,
                # The EFFECTIVE value, after --layer_checkpoint's
                # override — not the flag. Two arms are only comparable when
                # this field matches.
                'use_cache': effective_use_cache,
                'cpu_offload': args.cpu_offload,
                # A span is not comparable across two values of this. 0 is
                # HuggingFace's unchunked loss.
                'chunked_ce_tokens': args.chunked_ce,
                # What the feasibility gate ran against. No code can check
                # that this is THIS config's own measured bracket rather than
                # a borrowed one, so it is recorded beside the shape it ran at.
                'hbm_budget_gb': args.hbm_budget_gb,
                'engine_live': args.engine_live,
                'strict_evict': args.strict_evict,
                # A MEMORY parameter: the planner prices relief as an
                # interval, so this decides how many layers' absence still
                # covers the peak instant, not only prefetch timing.
                # Schedule-side coverage is (L-W-1)/L.
                'prefetch_window_layers': args.prefetch_window_layers,
            },
            'steps': step_results,
        }, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    dist.barrier(device_ids=[local_rank])
    if rank == 0:
        avg_peak = sum(r['peak_mb'] for r in step_results) / len(step_results)
        avg_time = sum(r['step_ms'] for r in step_results) / len(step_results)
        print(f"\n{'='*72}")
        print(f"  ✓ Completed {args.steps} steps on {world_size} GPUs")
        print(f"  ✓ Avg peak memory: {avg_peak:.1f} MB")
        print(f"  ✓ Avg step time:   {avg_time:.1f} ms")
        print(f"  ✓ Profiles saved to {args.output_dir}/")
        print(f"{'='*72}\n")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    profiler.detach_hooks()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
