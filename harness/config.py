"""Configuration and platform profiles."""
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class ModelShape:
    hidden: int = 768
    intermediate: int = 2048
    n_q: int = 12
    n_kv: int = 4
    layers: int = 64
    vocab: int = 32000

    @property
    def head_dim(self):
        return self.hidden // self.n_q

    def with_layers(self, layers):
        return replace(self, layers=layers)


TIER_A = ModelShape()


@dataclass(frozen=True)
class Run:
    """One arm: shape, sharding and the levers. Only the configuration the
    closed form covers is accepted: whole-layer unit, chunked loss, regime B."""
    shape: ModelShape = TIER_A
    world: int = 4
    chunk: int = 256                 # chunked-CE tokens; 0 (unchunked) is not derived
    act_bytes: int = 2               # fp16 / bf16 activations
    repeat_kv: bool = True           # HF SDPA path expands k,v to n_q heads before saving
    attn: str = 'mem_efficient'      # regime-B kernel: 'mem_efficient' or 'flash' (sm80+)
    cpu_offload: bool = False        # FSDP CPUOffload(offload_params=True)

    def __post_init__(self):
        if self.attn not in ('mem_efficient', 'flash'):
            raise ValueError(f"attn={self.attn!r}: only the regime-B kernels are derived")
        if self.chunk <= 0:
            raise ValueError("unchunked loss is not derived (peak sits in the loss backward)")


@dataclass(frozen=True)
class Platform:
    name: str
    ws_mib: float                    # library workspace at the peak (recorded per platform)
    usable_mib: float                # device capacity the allocator can reach
    ranks_per_node: int = 1
    host_ram_gib: float = 0.0
    d2h_gibps: float = 0.0           # measured bulk link rates (time model)
    h2d_gibps: float = 0.0
    capacity: object = field(default=None, compare=False)   # CapacityModel
    provenance: dict = field(default_factory=dict, compare=False)


# V100 / PyTorch 2.6. ws_mib is the one CALIBRATED byte constant: 16.2559 MiB,
# the two 8.125 MiB blocks every snapshot carries. It is the calibrated 31.2588 MiB
# less the shape term carried by terms.py (15.0029 = 2 B·p_layer + 2 B·h·I at
# tier-A: a prefetched layer plus one MLP weight gradient, named from 7763's
# snapshots). Four-rank fit: on the 251 V100 peaks of
# tests/test_harness_v100.py (worlds 2-4) it comes out at 16.2561, no peak more
# than 0.023 MiB off; the A30 measured 16.2546.
from .capacity import CapacityModel

# V100 capacity, calibrated on the uncapped brackets plus the four capped ladders
# of 7760: one alpha fits everything only in [0.139, 0.186]; centre 0.163, uncapped
# C0 15627.64 MiB, capped reserve delta 33.1 MiB. The V100 brackets in
# profile_results/ (7743/7744/7747/7756/7758, 7765 at two, three and four ranks,
# 7760 capped) give the same alpha range, delta in (26.8, 39.4], and C0 in
# [15617.37, 15629.70) at alpha 0.163. This is a fit, not a prediction: the remote
# cards are where it is tested.
_V100_CAPACITY = CapacityModel(c0_mib=15627.64, alpha=0.163, alpha_range=(0.139, 0.186),
                               provenance='uncapped brackets + 7760 capped ladders')
V100_CAP_DELTA_MIB = 33.1

PLATFORMS = {
    'v100': Platform(
        name='v100', ws_mib=16.2559, usable_mib=15.77 * 1024, ranks_per_node=1,
        host_ram_gib=112000 / 1024, d2h_gibps=6.54, h2d_gibps=7.73,
        capacity=_V100_CAPACITY,
        provenance={'capacity': 'calibrated on the x>=100 rungs of 7743/7744/7747 (tests/test_harness_brackets.py)',
                    'ws_mib': 'calibrated 31.2588, less the shape term of terms.py; four-rank fit 16.2561 (tests/test_harness_v100.py)',
                    'links': '7744 x=356 copy telemetry'}),
    # A30-24G (ClusterA, sm80, 4 per node, 503.4 GiB host). ws_mib is CARRIED from V100,
    # net of the shape term (the job package's harness copy carries 31.2588 with no
    # shape term: same totals at tier-A) (the L40S 621880 static arm landed on the
    # V100 line to 0.00 MiB, so the lean is that the workspace is a library constant);
    # capacity is left None because the A30 job calibrates it in-job (its capped
    # ladders give delta and alpha, the per-step outside_alloc_mb gives the context)
    # before predicting its native bracket. usable_mib is nominal; the job reads the card.
    'a30': Platform(
        name='a30', ws_mib=16.2559, usable_mib=24 * 1024.0, ranks_per_node=4,
        host_ram_gib=503.4, capacity=None,
        provenance={'ws_mib': 'carried from V100 (lean); measured by the A30 job',
                    'capacity': 'calibrated in-job (tools/a30_frontier.py)'}),
    # L40S-48G (ClusterA, Ada sm89, 4 per node, PCIe-only, 503.4 GiB host). ws_mib CARRIED
    # from V100: 621880's static and offload arms (L=32, x=8) sit on these lines to
    # 0.001 MiB. Capacity calibrated in-job, as on the A30 (tools/remote_frontier.py).
    'l40s': Platform(
        name='l40s', ws_mib=16.2559, usable_mib=45459.5, ranks_per_node=4,
        host_ram_gib=503.4, d2h_gibps=25.9 / 1.0737, h2d_gibps=24.8 / 1.0737,
        # NATIVE capacity, fitted POST HOC on 630188's six native brackets (3B static and
        # offload, tier-A static); one (C0, alpha) fits both shapes. The follow-up job
        # registers its new brackets from it.
        capacity=CapacityModel(c0_mib=45066.4971, alpha=0.237, alpha_range=(0.122, 0.353),
                               provenance='630188 post hoc: native brackets of tier-A and Llama-3.2-3B'),
        provenance={'ws_mib': 'carried from V100; 621880 reproduced to 0.001 MiB',
                    # TRAP (630188): nvidia-smi reports 46068 MiB, the allocator sees
                    # 45459.5 (torch total_memory); the 608.5 MiB gap cost the registered 3B brackets 2-4 rungs.
                    'usable_mib': 'torch total_memory read by 630188 (NOT nvidia-smi 46068)',
                    'native_capacity': '630188 post hoc: one (C0, alpha) fits all six native brackets of both shapes; '
                                       'at V100 alpha 0.163, C0 in [44859.5, 44918.7) MiB',
                    'links': '621880 solo copy, one GPU at a time (GB/s -> GiB/s)',
                    'capacity': 'calibrated in-job (tools/remote_frontier.py)'}),
    # A100 (ClusterB DGX A100, sm80, NVSwitch, 8 per node of which the job uses 4,
    # ~2 TB host). ws_mib CARRIED from V100 (the A30, same sm80 build, measured 16.2546).
    # No capacity: the A100 jobs (remote/a100_regime/) measure time, not brackets.
    'a100': Platform(
        name='a100', ws_mib=16.2559, usable_mib=80 * 1024.0, ranks_per_node=4,
        host_ram_gib=2015, capacity=None,
        provenance={'ws_mib': 'carried from V100; A30 (same arch) measured 16.2546',
                    'usable_mib': 'nominal; the job reads the card',
                    'capacity': 'not measured on A100'}),
}

# Llama-3.2-3B geometry (untied embeddings, as the driver builds it): the L40S
# realistic shape. Its predictions are an out-of-sample test of the post hoc shape
# terms in terms.py.
LLAMA32_3B = ModelShape(hidden=3072, intermediate=8192, n_q=24, n_kv=8, layers=28, vocab=128256)

# A 7B-like shape with grouped-query attention (hidden 4096, intermediate 14336,
# 32 query and 8 key-value heads, Llama architecture, untied embeddings, v=32000):
# the eight-rank A100 job's model, where a boundary is small against the replayed layer.
MISTRAL7B = ModelShape(hidden=4096, intermediate=14336, n_q=32, n_kv=8, layers=32, vocab=32000)

SHAPES = {'tier_a': TIER_A, 'llama3b': LLAMA32_3B, 'mistral7b': MISTRAL7B}
