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
    shape: ModelShape = TIER_A
    world: int = 4
    chunk: int = 256
    act_bytes: int = 2
    repeat_kv: bool = True
    attn: str = 'mem_efficient'
    cpu_offload: bool = False

    def __post_init__(self):
        if self.attn not in ('mem_efficient', 'flash'):
            raise ValueError(f"attn={self.attn!r}: only the regime-B kernels are derived")
        if self.chunk <= 0:
            raise ValueError("unchunked loss is not derived (peak sits in the loss backward)")

@dataclass(frozen=True)
class Platform:
    name: str
    ws_mib: float
    usable_mib: float
    ranks_per_node: int = 1
    host_ram_gib: float = 0.0
    d2h_gibps: float = 0.0
    h2d_gibps: float = 0.0
    capacity: object = field(default=None, compare=False)
    provenance: dict = field(default_factory=dict, compare=False)

from .capacity import CapacityModel

_V100_CAPACITY = CapacityModel(c0_mib=15627.64, alpha=0.163, alpha_range=(0.139, 0.186),
                               provenance='refit: uncapped brackets + 7760 capped ladders')
V100_CAP_DELTA_MIB = 33.1

PLATFORMS = {
    'v100': Platform(
        name='v100', ws_mib=16.2559, usable_mib=15.77 * 1024, ranks_per_node=1,
        host_ram_gib=112, d2h_gibps=6.54, h2d_gibps=7.73,
        capacity=_V100_CAPACITY,
        provenance={'capacity': 'calibrated on the x>=100 rungs of 7743/7744/7747 (tests/test_harness_brackets.py)',
                    'ws_mib': 'calibrated 31.2588, less the V4 shape term',
                    'links': '7744 x=356 copy telemetry'}),
    'a30': Platform(
        name='a30', ws_mib=16.2559, usable_mib=24 * 1024.0, ranks_per_node=4,
        host_ram_gib=515, capacity=None,
        provenance={'ws_mib': 'carried from V100 (lean); measured by the A30 job',
                    'capacity': 'calibrated in-job (tools/a30_frontier.py)'}),
    'l40s': Platform(
        name='l40s', ws_mib=16.2559, usable_mib=45459.5, ranks_per_node=4,
        host_ram_gib=515, d2h_gibps=25.9 / 1.0737, h2d_gibps=24.8 / 1.0737,
        capacity=CapacityModel(c0_mib=45066.4971, alpha=0.237, alpha_range=(0.122, 0.353),
                               provenance='630188 post hoc: native brackets of tier-A and Llama-3.2-3B'),
        provenance={'ws_mib': 'carried from V100; 621880 reproduced to 0.001 MiB',
                    'usable_mib': 'torch total_memory read by 630188 (NOT nvidia-smi 46068)',
                    'native_capacity': '630188 post hoc: one (C0, alpha) fits all six native brackets of both shapes; '
                                       'at V100 alpha 0.163, C0 in [44859.5, 44918.7) MiB',
                    'links': '621880 solo copy, one GPU at a time (GB/s -> GiB/s)',
                    'capacity': 'calibrated in-job (tools/remote_frontier.py)'}),
    'a100': Platform(
        name='a100', ws_mib=16.2559, usable_mib=80 * 1024.0, ranks_per_node=4,
        host_ram_gib=2015, capacity=None,
        provenance={'ws_mib': 'carried from V100; A30 (same arch) measured 16.2546',
                    'usable_mib': 'nominal; the job reads the card',
                    'capacity': 'not measured on A100'}),
}

LLAMA32_3B = ModelShape(hidden=3072, intermediate=8192, n_q=24, n_kv=8, layers=28, vocab=128256)
