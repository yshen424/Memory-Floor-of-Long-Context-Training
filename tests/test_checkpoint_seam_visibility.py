"""Does the engine's seam survive whole-layer checkpointing?

WHY. Under the whole-layer recompute unit the per-layer non-migratable mass is
zero and the entire per-layer increment is one fp16 residual-stream boundary,
2 B x hidden per layer per token. Every one of those bytes is migratable -- IF
a placement engine can reach it. The engine has exactly one seam,
saved_tensors_hooks, and PyTorch's checkpoint wrapper installs
saved_tensors_hooks of its own.

THE QUESTION. When a layer is wrapped in checkpoint_wrapper(NO_REENTRANT),
which tensors still reach an OUTER saved_tensors_hooks pack_fn?

THE TWO OUTCOMES.
  H1 (seam survives, engine is buildable on whole-layer): the outer pack_fn
     sees ~one tensor per wrapped layer, shaped like the residual stream. The
     checkpoint region hides its interior saves but the layer-boundary input is
     still routed through the outer saved-tensor machinery, so
     --layer_checkpoint and --placement_engine can be combined.
  H2 (seam is blind, the mechanism must be rebuilt): the outer pack_fn sees
     ZERO tensors per wrapped layer. The CheckpointFrame holds the boundary
     inputs as ordinary Python references, never offering them to the outer
     hooks. Then whole-layer + our engine needs our own checkpoint
     implementation, not PyTorch's wrapper.

Either outcome is a diagnosis. Device-agnostic: saved-tensor hook dispatch does
not depend on device, so CPU is sufficient and CUDA is used when present.

Run:  PYTHONPATH=. python3 tests/test_checkpoint_seam_visibility.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing, checkpoint_wrapper, CheckpointImpl)

B, S, D = 2, 16, 32
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


class Layer(nn.Module):
    """Stands in for LlamaDecoderLayer: several interior saves, one residual
    boundary in and one out."""

    def __init__(self, d):
        super().__init__()
        self.up = nn.Linear(d, 2 * d, bias=False)
        self.down = nn.Linear(2 * d, d, bias=False)

    def forward(self, x):
        return x + self.down(F.silu(self.up(x)))


class Stack(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.layers = nn.ModuleList([Layer(d) for _ in range(n)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def run(checkpointed):
    torch.manual_seed(0)
    model = Stack(D, 3).to(DEV)
    if checkpointed:
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(
                m, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
            check_fn=lambda m: isinstance(m, Layer))

    seen = []
    in_forward = {'flag': True}

    def pack(t):
        if in_forward['flag']:
            seen.append(tuple(t.shape))
        return t

    def unpack(t):
        return t

    x = torch.randn(B, S, D, device=DEV, requires_grad=True)
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        out = model(x)
    in_forward['flag'] = False
    out.sum().backward()

    resid = (B, S, D)
    return seen, sum(1 for s in seen if s == resid), x.grad


def main():
    print(f"device: {DEV}   residual-stream shape: {(B, S, D)}   layers: 3\n")

    base_seen, base_resid, base_grad = run(checkpointed=False)
    ckpt_seen, ckpt_resid, ckpt_grad = run(checkpointed=True)

    print(f"BASELINE (no checkpointing)")
    print(f"  saves reaching the outer pack_fn : {len(base_seen)}")
    print(f"  of which residual-shaped         : {base_resid}")
    print(f"  shapes                           : {base_seen}\n")

    print(f"WHOLE-LAYER CHECKPOINTED (NO_REENTRANT)")
    print(f"  saves reaching the outer pack_fn : {len(ckpt_seen)}")
    print(f"  of which residual-shaped         : {ckpt_resid}")
    print(f"  shapes                           : {ckpt_seen}\n")

    # Positive control: if the outer hooks never fired even WITHOUT
    # checkpointing, the probe is measuring nothing and every count above is
    # uninterpretable.
    if len(base_seen) == 0:
        print("POSITIVE CONTROL FAILED -- the outer hooks never fired at all.")
        return
    print("POSITIVE CONTROL OK -- the outer hooks fire on the unwrapped model.")

    if not torch.allclose(base_grad, ckpt_grad, atol=1e-5):
        print("WARNING: checkpointed gradients differ from baseline.")

    print()
    if ckpt_resid >= 3:
        print(f"VERDICT: H1 -- the seam SURVIVES. {ckpt_resid} residual-shaped "
              f"boundaries reach the outer pack_fn, so a placement engine can "
              f"still see and migrate the whole-layer boundary mass: "
              f"--layer_checkpoint and --placement_engine can be combined.")
    elif ckpt_resid == 0:
        print(f"VERDICT: H2 -- the seam is BLIND. The checkpoint region hides "
              f"every save including the layer boundaries, so PyTorch's wrapper "
              f"cannot be combined with the engine. Whole-layer + engine "
              f"requires our own checkpoint implementation.")
    else:
        print(f"VERDICT: PARTIAL -- {ckpt_resid} of 3 boundaries visible. "
              f"Neither hypothesis as written; inspect the shape list.")


if __name__ == '__main__':
    main()
