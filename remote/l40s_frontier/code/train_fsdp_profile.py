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

def init_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.barrier(device_ids=[local_rank])
    return local_rank

def _sdpa_preflight(dtype, head_dim: int, n_q: int = 2, n_kv: int = 2,
                    native_gqa: bool = False) -> dict:
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

def _capability() -> tuple:
    if not torch.cuda.is_available():
        return (0, 0)
    return torch.cuda.get_device_capability()

def install_native_gqa_sdpa():
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
    info = {'requested': mode}
    if mode == 'sdpa_auto':
        info['note'] = 'dispatcher left free'
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
    return attn_impl not in ('sdpa_mem_efficient', 'sdpa_flash')

def build_model(args) -> tuple:
    config = LlamaConfig(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        max_position_embeddings=args.seq_len,
        vocab_size=args.vocab_size,
        rms_norm_eps=1e-5,
        attn_implementation=('eager' if args.attn_impl == 'eager' else 'sdpa'),
    )
    if getattr(args, 'seed', -1) >= 0:
        torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_config(config)
    return model, config

def wrap_with_fsdp(model, local_rank, args):

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer},
    )

    if args.dtype == 'bf16':
        param_dtype = torch.bfloat16
    else:
        param_dtype = torch.float16

    mixed_precision = MixedPrecision(
        param_dtype=param_dtype,
        reduce_dtype=param_dtype,
        buffer_dtype=param_dtype,
    )

    bwd_prefetch = (None if getattr(args, 'no_backward_prefetch', False)
                    else BackwardPrefetch.BACKWARD_PRE)

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
    staged = L if staged_layers is None else max(0, min(L, staged_layers))

    if whole_layer_unit:
        classes = [(batch_size * seq_len * hidden, 1)]
        ladder_count, ladder_ceiling = 8, 8 * 1024 * 1024
        fp32_variants = False
        catchall = (batch_size * seq_len * hidden * 4, 2)
    else:
        classes = [
            (batch_size * seq_len * num_kv_heads * head_dim, 2),
            (batch_size * seq_len * hidden, 3),
            (batch_size * seq_len * intermediate, 3),
        ]
        ladder_count, ladder_ceiling = 128, None
        fp32_variants = True
        catchall = (max(elems * 4 for elems, _ in classes) * 4,
                    max(1, staged // 4))
    scores = (batch_size * num_heads * seq_len * seq_len, 1)

    counts: dict[int, int] = {}

    def add(size_bytes: int, n: int) -> None:
        n = max(1, int(round(n * count_multiplier)))
        counts[size_bytes] = max(counts.get(size_bytes, 0), n)

    small = 128 * 1024
    ladder_top = min(elems for elems, _ in classes) * dtype_size
    if ladder_ceiling is not None:
        ladder_top = min(ladder_top, ladder_ceiling)
    while small < ladder_top:
        add(small, ladder_count)
        small *= 2

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

    add(*catchall)

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
        if staged:
            add(scores[0] * 4, staged * scores[1])

    return counts

def main():
    parser = argparse.ArgumentParser(description="FSDP LLaMA + Memory Profiling")

    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--intermediate_size', type=int, default=2816)
    parser.add_argument('--num_layers', type=int, default=16)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--num_kv_heads', type=int, default=4)
    parser.add_argument('--vocab_size', type=int, default=32000)
    parser.add_argument('--fused_rmsnorm', choices=['off', 'input', 'output'],
                        default='off',
                        help="Replace LlamaRMSNorm with a memory-lean autograd "
                             "Function. 'input' is the honest fused baseline "
                             "(what production fused kernels save); 'output' is "
                             "the leaner-peak variant that forecloses migration.")
    parser.add_argument('--layer_checkpoint', action='store_true',
                        help="Wrap each LlamaDecoderLayer in non-reentrant "
                             "activation checkpointing (classic per-layer "
                             "gradient checkpointing). With --placement_engine "
                             "this selects the whole-layer recompute unit.")
    parser.add_argument('--activation_offload', action='store_true',
                        help="Offload every checkpointed layer's saved "
                             "boundary to pinned host memory via PyTorch's "
                             "offload_wrapper. Requires --layer_checkpoint. "
                             "Unconditional: no criterion, no feasibility "
                             "gate, no pinned pool.")
    parser.add_argument('--chunked_ce', type=int, default=0, metavar='TOKENS',
                        help="Compute the LM head and cross-entropy this many "
                             "tokens at a time, each chunk checkpointed. 0 "
                             "(default) keeps HuggingFace's unchunked loss.")
    parser.add_argument('--seed', type=int, default=-1,
                        help="Seed model init and the batch stream. -1 "
                             "(default) leaves both unseeded.")
    parser.add_argument('--use_cache', choices=['on', 'off'], default='on',
                        help="Whether the model retains its KV cache. 'on' is "
                             "LlamaConfig's default; 'off' is what a real trainer does. "
                             "--layer_checkpoint forces 'off' regardless.")
    parser.add_argument('--attn_impl',
                        choices=['sdpa_math', 'sdpa_mem_efficient',
                                 'sdpa_flash', 'sdpa_auto', 'eager'],
                        default='sdpa_math',
                        help="Which attention kernel materializes the s x s "
                             "tensor. 'sdpa_math' (default) pins PyTorch to the "
                             "math kernel instead of leaving the choice to the dispatcher. "
                             "'sdpa_mem_efficient' pins the cutlass efficient "
                             "kernel, which does NOT materialize the s x s "
                             "tensor; on sm_70 it requires --dtype fp16 (bf16 "
                             "kernels exist for sm80+ only) and it disables math "
                             "so a failed dispatch raises instead of falling back. "
                             "'sdpa_flash' pins FlashAttention-2 (sm80+), same "
                             "no-fallback rule. "
                             "'sdpa_auto' leaves the dispatcher unpinned. "
                             "'eager' uses HF's own attention, whose "
                             "fp32 softmax upcast makes the class LARGER — a "
                             "different constant, so do not mix it into a fit.")

    parser.add_argument('--native_gqa', action='store_true',
                        help="Pass k/v to SDPA at num_kv_heads width (enable_gqa) "
                             "instead of HF's repeat_kv expansion, so they are "
                             "saved at kv width. Needs a kernel with native GQA "
                             "(flash on sm80+); the preflight proves it.")

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

    parser.add_argument('--profile_steps', type=str, default='all',
                        help="Which steps to profile: 'all', 'last', or '2,3,4'")
    parser.add_argument('--output_dir', type=str, default='profile_results')
    parser.add_argument('--torch_profile', action='store_true',
                        help="Also run PyTorch profiler (generates Chrome trace)")
    parser.add_argument('--cuda_snapshot', action='store_true',
                        help="Capture CUDA memory snapshot on last step")

    parser.add_argument('--placement_engine', action='store_true',
                        help="Attach the dry-run PlacementEngine "
                             "(Step 3a). Adds schema_version=2 sections "
                             "to the JSON output.")
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
    parser.add_argument('--pool_buffers_per_bucket', type=float, default=1.0,
                        help="Multiplier on the per-bucket buffer counts "
                             "derived from per-layer save multiplicity. 1.0 "
                             "(default) covers the analytic worst case: at "
                             "AD-768/b=6 that is ~712 buffers / ~8.3 GiB. Use "
                             ">1 to add headroom, <1 to starve the pool "
                             "deliberately for an ablation.")
    parser.add_argument('--pool_cover_attn_scores', action='store_true',
                        help="Provision pinned buckets for the attention-score "
                             "class, making it evictable. Adds ~11 GiB of "
                             "pinned host memory at AD-768/b=6.")
    parser.add_argument('--overlap_safety_factor', type=float, default=0.8,
                        help="Deflation factor on _available_overlap_ms "
                             "(0..1). Lower = more recompute, higher = "
                             "more migration. Default 0.8.")
    parser.add_argument('--engine_live', action='store_true',
                        help="Run MemoryStreamManager in live mode "
                             "(real cudaMemcpyAsync instead of no-op "
                             "event recording). Step 4.5; default OFF.")

    parser.add_argument('--coordinator', type=str, default='fsdp',
                        choices=['fsdp', 'noop'],
                        help="Which FSDP coordinator to attach. "
                             "Default 'fsdp' = real FSDPCoordinator "
                             "(Step 5a); 'noop' = NoOpCoordinator for "
                             "ablation / smoke runs.")

    parser.add_argument('--cpu_offload', action='store_true',
                        help="FSDP CPUOffload(offload_params=True): params and "
                             "grads live on the host, the optimizer steps on "
                             "CPU. The ZeRO-Offload analogue, and the SUBJECT "
                             "of the D3/D4 overlap probe. Pair with "
                             "--torch_profile and read the trace with "
                             "tools/analyze_nccl_duty.py.")
    parser.add_argument('--no_backward_prefetch', action='store_true',
                        default=False,
                        help="Disable FSDP's BACKWARD_PRE prefetch.")

    parser.add_argument('--disable_hbm_drift', action='store_true',
                        default=False,
                        help="Accepted no-op. The HBM budget-correction loop "
                             "is disabled; the planner always sees "
                             "the raw budget. The per-step observed-vs-"
                             "projected peak LOG is unaffected.")

    parser.add_argument('--no_window_debit', action='store_true',
                        default=False,
                        help="Use the overlap formula in "
                             "which every decision reads the full window "
                             "O(w). Ablation control for the window-debit "
                             "mechanism.")

    parser.add_argument('--no_interval_peak', action='store_true',
                        default=False,
                        help="Disable per-interval peak attribution; "
                             "snapshot peak_mb reverts to the cumulative "
                             "step high-water mark and no mid-step reset of "
                             "the CUDA counter occurs. Ablation control.")

    parser.add_argument('--no_observed_relief', action='store_true',
                        default=False,
                        help="Use the analytic peak projection, "
                             "in which a bundle's relief is the analytic "
                             "per-component sum rather than the storage "
                             "bytes pack_fn observes it own. Ablation "
                             "control for the observed-relief mechanism.")

    parser.add_argument('--strict_evict', action='store_true',
                        default=False,
                        help="Raise on any planned eviction the pinned pool "
                             "denies at pack time, instead of logging it and "
                             "leaving the tensor resident. Recommended for "
                             "runs above the memory ceiling.")

    parser.add_argument('--force_action', type=str, default='none',
                        choices=['none', 'evict', 'recompute'],
                        help="Force every bundle's placement decision. "
                             "'none' (default) runs the exposed-cost "
                             "criterion. 'evict' forces pure-evict "
                             "baseline (always migrate). 'recompute' "
                             "forces pure-recompute baseline (always "
                             "checkpoint). Used for S2 headline "
                             "experiment.")
    parser.add_argument('--force_stride', type=int, default=0,
                        help="With --force_action recompute, migrate every "
                             "k-th layer and recompute the rest. 0 (default) "
                             "= off. Criterion-free sparse-migration policy; "
                             "k=1 degenerates to pure evict and k>num_layers "
                             "to pure recompute.")

    parser.add_argument('--evict_filter', type=str, default='all',
                        help="Restrict which saves eviction may touch, to "
                             "bisect the regime B NaN. 'all' (default), "
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
                             "size class. Free — no clone, no sync. This is "
                             "the instrument for the leading hypothesis: the "
                             "restore rebuilds with torch.empty(shape), so a "
                             "strided or offset save comes back re-laid-out "
                             "with correct values, which a value comparison "
                             "cannot detect.")
    parser.add_argument('--verify_restore', action='store_true',
                        default=False,
                        help="Keep an HBM reference copy of every evicted "
                             "save and compare the restored tensor against it "
                             "at unpack, reporting value and layout "
                             "mismatches separately per size class. Defeats "
                             "the memory relief and serializes the step: run "
                             "at short sequence length as a diagnostic, never "
                             "as a frontier arm.")

    parser.add_argument('--diagnose_memory_growth', action='store_true',
                        default=False,
                        help="Per-step memory-growth diagnostic JSONL. "
                             "Discardable.")
    parser.add_argument('--diag_snapshot_steps', type=str, default='',
                        help="Comma-separated step indices to dump CUDA "
                             "memory snapshots on (rank 0 only). Only "
                             "active when --diagnose_memory_growth is "
                             "set. Empty disables. Example: '0,9'.")

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

    if args.evict_filter != 'all':
        from memory_manager.placement_engine import PlacementEngine
        PlacementEngine.evict_filter.fset(
            type('_Probe', (), {})(), args.evict_filter)

    diag_snapshot_step_set: set[int] = set()
    if args.diagnose_memory_growth and args.diag_snapshot_steps:
        diag_snapshot_step_set = {int(s) for s in args.diag_snapshot_steps.split(',') if s.strip()}

    local_rank = init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

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

    if args.profile_steps == 'all':
        profile_step_set = set(range(args.steps))
    elif args.profile_steps == 'last':
        profile_step_set = {args.steps - 1}
    else:
        profile_step_set = set(int(x) for x in args.profile_steps.split(','))

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

    attn_backend_info = pin_attention_backend(
        args.attn_impl, rank,
        dtype=(torch.bfloat16 if args.dtype == 'bf16' else torch.float16),
        head_dim=args.hidden_size // args.num_heads,
        n_q=args.num_heads, n_kv=args.num_kv_heads,
        native_gqa=args.native_gqa)
    model, model_config = build_model(args)
    attn_backend_info['resolved'] = getattr(
        model.config, '_attn_implementation', 'unknown')

    fused_norm_count = 0
    if args.fused_rmsnorm != 'off':
        from fused_rmsnorm import install_fused_rmsnorm
        fused_norm_count = install_fused_rmsnorm(model, mode=args.fused_rmsnorm)
        print(f"[rank {rank}] fused RMSNorm mode={args.fused_rmsnorm}: "
              f"{fused_norm_count} norms replaced", flush=True)

    model.config.use_cache = (args.use_cache == 'on')
    effective_use_cache = model.config.use_cache

    layer_ckpt_count = 0
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
        if args.use_cache == 'on':
            print(f"[rank {rank}] OVERRIDE: --layer_checkpoint forces "
                  f"use_cache=off. Any arm compared against this one must pass "
                  f"--use_cache off or the comparison is confounded.", flush=True)
        model.config.use_cache = False
        effective_use_cache = False
        ckpt_fn = functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
        if args.activation_offload:
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

    if rank == 0:
        dtype_size = 2
        param_count = sum(p.numel() for p in model.parameters())
        sharded_params_mb = param_count * dtype_size / world_size / (1024**2)
        sharded_optim_mb = param_count * (4 + 4 + 4) / world_size / (1024**2)
        sharded_grad_mb = param_count * dtype_size / world_size / (1024**2)
        bs = args.batch_size
        sl = args.seq_len
        per_layer_act = (
            bs * args.num_heads * sl * sl * dtype_size
            + 3 * bs * sl * args.intermediate_size * dtype_size
            + bs * sl * (args.num_heads + 2 * args.num_kv_heads)
              * (args.hidden_size // args.num_heads) * dtype_size
            + bs * sl * args.hidden_size * dtype_size
        )
        total_act_mb = per_layer_act * args.num_layers / (1024**2)
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

    model = wrap_with_fsdp(model, local_rank, args)

    if rank == 0:
        print("  ✓ FSDP wrapping complete")

    print_sharding_proof(model, rank, world_size)
    dist.barrier(device_ids=[local_rank])
    print_gpu_memory(rank, label="after FSDP wrap")
    dist.barrier(device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    scores_materialized = materializes_attn_scores(args.attn_impl)
    if rank == 0 and not scores_materialized:
        print(f"  [Attention] byte model: no s x s scores tensor "
              f"(--attn_impl {args.attn_impl})")
    profiler = FSDPMemoryProfiler(
        model, rank, world_size, model_config, enabled=True,
        materializes_attn_scores=scores_materialized,
    )
    profiler._interval_peak_enabled = not args.no_interval_peak
    profiler.attach_hooks()

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

        calibration = run_calibration(verbose=(rank == 0))

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

        bucket_counts = _derive_pool_buckets(
            model_config, args.batch_size, args.seq_len,
            dtype_size=(2 if args.dtype in ('bf16', 'fp16') else 4),
            num_layers=args.num_layers,
            cover_fp32_upcast=args.pool_cover_attn_scores,
            count_multiplier=args.pool_buffers_per_bucket,
            scores_materialized=scores_materialized,
            whole_layer_unit=args.layer_checkpoint,
            staged_layers=(
                0 if args.force_action == 'recompute' and args.force_stride == 0
                else len(range(0, args.num_layers, args.force_stride))
                if args.force_action == 'recompute'
                else None),
            verbose=(rank == 0),
        )
        pool = PinnedBufferPool(bucket_sizes_bytes=bucket_counts)

        mem_stream = MemoryStreamManager(
            device=torch.device(f"cuda:{local_rank}"),
            dry_run=(not args.engine_live),
            calibration=calibration,
        )
        if rank == 0:
            print(f"  [Engine] MemoryStreamManager dry_run="
                  f"{mem_stream.dry_run} (engine_live={args.engine_live})")

        budget = HBMBudget(
            total_bytes=int(args.hbm_budget_gb * (1024 ** 3)),
            registry=profiler.registry,
        )

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
            overlap_safety_factor=args.overlap_safety_factor,
            window_debit=not args.no_window_debit,
            observed_relief=not args.no_observed_relief,
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
                  f"({'peak relief from observed owned storage bytes' if not args.no_observed_relief else 'analytic per-component sum (pre-candidate-5 control)'})")

        engine.engine_live = args.engine_live
        if rank == 0:
            print(f"  [Engine] engine_live={engine.engine_live} "
                  f"(pack_fn/unpack_fn {'active' if engine.engine_live else 'inert'})")

        engine._strict_evict = args.strict_evict
        if rank == 0:
            print(f"  [Engine] strict_evict={args.strict_evict} "
                  f"({'a denied eviction raises' if args.strict_evict else 'a denied eviction is logged and counted'})")

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
        )

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

    cuda_snapshot_enabled = args.cuda_snapshot

    if rank == 0:
        print(f"\n{'─'*72}")
        print(f"  Step | Rank | Loss     | Avg Loss | Peak MB | Step ms")
        print(f"{'─'*72}")

    os.makedirs(args.output_dir, exist_ok=True)
    step_results = []

    diag_jsonl_path = None
    diag_jsonl_fh = None
    if args.diagnose_memory_growth:
        diag_jsonl_path = os.path.join(args.output_dir, f"mem_diag_rank{rank}.jsonl")
        diag_jsonl_fh = open(diag_jsonl_path, 'w')
        if rank == 0:
            print(f"  [Diag] memory-growth JSONL → {diag_jsonl_path}")
            if diag_snapshot_step_set:
                print(f"  [Diag] CUDA snapshot steps: {sorted(diag_snapshot_step_set)}")

    if torch_profiler:
        torch_profiler.start()

    batch_gen = None
    if args.seed >= 0:
        batch_gen = torch.Generator(device=f"cuda:{local_rank}")
        batch_gen.manual_seed(args.seed + 1000 * rank)

    if args.grad_accum < 1:
        raise SystemExit("--grad_accum must be >= 1")
    for step in range(args.steps):
        full_ids = get_batch(args.batch_size * args.grad_accum, args.seq_len,
                             args.vocab_size, device=f"cuda:{local_rank}",
                             generator=batch_gen)
        micro_batches = (full_ids.split(args.batch_size, dim=0)
                         if args.grad_accum > 1 else (full_ids,))

        do_profile = step in profile_step_set
        profiler.enabled = do_profile

        if cuda_snapshot_enabled and step == args.steps - 1:
            try:
                torch.cuda.memory._record_memory_history(
                    enabled='all', context='all', stacks='all')
            except AttributeError:
                cuda_snapshot_enabled = False

        diag_snapshot_active = (
            args.diagnose_memory_growth and rank == 0 and step in diag_snapshot_step_set
        )
        if diag_snapshot_active:
            try:
                torch.cuda.memory._record_memory_history(
                    enabled='all', context='all', stacks='all')
            except AttributeError:
                diag_snapshot_active = False

        probe_armed = 0
        if overlap_probe is not None and step in overlap_probe_step_set:
            probe_armed = overlap_probe.arm(step, args.overlap_probe_ms)

        torch.cuda.reset_peak_memory_stats()
        step_start = time.perf_counter()

        if engine is not None:
            engine.on_training_step_begin()

        with profiler.profile_step(step):
            optimizer.zero_grad(set_to_none=True)

            if engine is not None and engine.engine_live:
                pack_ctx = torch.autograd.graph.saved_tensors_hooks(
                    engine.pack_fn, engine.unpack_fn)
            else:
                pack_ctx = contextlib.nullcontext()

            if args.grad_accum == 1:
                input_ids = full_ids
                with pack_ctx:
                    outputs = model(input_ids=input_ids, labels=input_ids)
                    loss = outputs.loss

                profiler.mark_phase("forward_end")

                loss.backward()

                profiler.mark_phase("backward_end")
            else:
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

            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.detach().float().norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5

            optimizer.step()

            profiler.mark_phase("optimizer_end")

        if engine is not None:
            for L, prof in profiler.layer_profiles.items():
                if prof.fwd_duration_ms > 0:
                    engine.record_layer_timing(L, 'fwd', prof.fwd_duration_ms)
                if prof.bwd_duration_ms > 0:
                    engine.record_layer_timing(L, 'bwd', prof.bwd_duration_ms)

        peak_mb = profiler.step_peak_bytes() / (1024**2)

        if engine is not None:
            engine.on_training_step_end()

        step_time = (time.perf_counter() - step_start) * 1000

        probe_overrun_ms = 0.0
        if probe_armed:
            probe_overrun_ms = overlap_probe.reap()

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
            'outside_alloc_mb': round(_outside_allocator_mb(), 2),
        })

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

    if diag_jsonl_fh is not None:
        diag_jsonl_fh.close()
        if rank == 0:
            print(f"  [Diag] JSONL closed: {diag_jsonl_path}")

    dist.barrier(device_ids=[local_rank])
    print_gpu_memory(rank, label="after training")

    dist.barrier(device_ids=[local_rank])
    if rank == 0:
        print()
    profiler.report()
    profiler.save(args.output_dir)

    if engine is not None and rank == 0:
        engine.print_summary()

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
                'seq_len': args.seq_len,
                'vocab_size': args.vocab_size,
                'dtype': args.dtype,
                'world_size': world_size,
                'attention': attn_backend_info,
                'fused_rmsnorm': args.fused_rmsnorm,
                'native_gqa': args.native_gqa,
                'fused_rmsnorm_replaced': fused_norm_count,
                'layer_checkpoint': args.layer_checkpoint,
                'layer_checkpoint_wrapped': layer_ckpt_count,
                'activation_offload': args.activation_offload,
                'use_cache': effective_use_cache,
                'cpu_offload': args.cpu_offload,
                'chunked_ce_tokens': args.chunked_ce,
                'hbm_budget_gb': args.hbm_budget_gb,
                'engine_live': args.engine_live,
                'strict_evict': args.strict_evict,
                'prefetch_window_layers': args.prefetch_window_layers,
            },
            'steps': step_results,
        }, f, indent=2)

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

    profiler.detach_hooks()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
