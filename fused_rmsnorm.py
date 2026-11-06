"""Memory-lean RMSNorm, as the fused-kernel baseline every span is quoted against.

HuggingFace's ``LlamaRMSNorm`` retains three tensors per norm per forward: the
fp32 upcast (4h bytes/token), the fp16 cast of the normalized value (2h) and
rstd (4). This module removes that residue, so the frontier is measured
without it.

THE DESIGN FORK THAT MATTERS. A fused norm must retain ONE hidden-width tensor
per norm to reconstruct the normalized value in backward, and the choice of
WHICH one decides whether migration has anything left to evict:

  mode='input'   saves x (the pre-norm residual stream) and rstd. Retains
                 2h + 4 bytes/token. The norm's OUTPUT stays retained by the
                 downstream projection, i.e. by the bundle, so eviction can
                 still free it. Numerically exact.

  mode='output'  saves y (the norm's output) and rstd, reconstructing
                 x_hat = y / w. y is the same storage the downstream
                 projection already saves, so the norm's save is an ALIAS and
                 costs zero extra bytes — the leanest possible peak. But that
                 save PINS the bundle's input through backward, so eviction
                 frees nothing and migration's supply in regime B goes to zero.

Both remove the same 6h bytes/token that HF retains; they differ only in what
they leave evictable. 'input' is the default because it leaves the norm's
output evictable; 'output' exists so the two can be run as a paired arm.
"""

import torch
import torch.nn as nn


class _FusedRMSNormSaveInput(torch.autograd.Function):
    """Saves (x, weight, rstd). Nothing else survives the forward."""

    @staticmethod
    def forward(ctx, x, weight, eps):
        x32 = x.float()
        rstd = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
        y = (x32 * rstd).to(x.dtype) * weight
        ctx.save_for_backward(x, weight, rstd)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, rstd = ctx.saved_tensors
        x_hat = x.float() * rstd
        g = grad_out.float()
        grad_weight = (g * x_hat).reshape(-1, x_hat.shape[-1]).sum(0)
        gw = g * weight.float()
        # In-place from here: gw is our own temporary, never a saved tensor.
        gw -= x_hat * (gw * x_hat).mean(-1, keepdim=True)
        gw *= rstd
        return gw.to(x.dtype), grad_weight.to(weight.dtype), None


class _FusedRMSNormSaveOutput(torch.autograd.Function):
    """Saves (y, weight, rstd), where y is an alias of the downstream save.

    Zero extra bytes, at the cost of pinning the bundle's input. The division
    by ``weight`` is exact at initialization (weight is all ones) and degrades
    only if a weight approaches zero, which is why this mode is not the
    default for anything but an A/B against ``mode='input'``.
    """

    @staticmethod
    def forward(ctx, x, weight, eps):
        x32 = x.float()
        rstd = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
        y = (x32 * rstd).to(x.dtype) * weight
        ctx.save_for_backward(y, weight, rstd)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        y, weight, rstd = ctx.saved_tensors
        w32 = weight.float()
        x_hat = y.float() / w32
        g = grad_out.float()
        grad_weight = (g * x_hat).reshape(-1, x_hat.shape[-1]).sum(0)
        gw = g * w32
        gw -= x_hat * (gw * x_hat).mean(-1, keepdim=True)
        gw *= rstd
        return gw.to(y.dtype), grad_weight.to(weight.dtype), None


class FusedRMSNorm(nn.Module):
    """Drop-in replacement for ``LlamaRMSNorm`` that reuses its parameter."""

    def __init__(self, weight: nn.Parameter, eps: float, mode: str = 'input'):
        super().__init__()
        if mode not in ('input', 'output'):
            raise ValueError(f"fused rmsnorm mode must be input|output, got {mode}")
        self.weight = weight
        self.variance_epsilon = eps
        self.mode = mode
        self._fn = (_FusedRMSNormSaveInput if mode == 'input'
                    else _FusedRMSNormSaveOutput)

    def forward(self, hidden_states):
        return self._fn.apply(hidden_states, self.weight, self.variance_epsilon)

    def extra_repr(self):
        return (f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}, "
                f"mode={self.mode}")


def install_fused_rmsnorm(model, mode: str = 'input') -> int:
    """Swap every LlamaRMSNorm in ``model`` for the fused version, in place.

    Must run BEFORE FSDP wrapping: the parameter objects are reused, so the
    flat param and every state-dict key are unchanged, but the module tree is
    read at wrap time. Returns how many norms were replaced, so a run can
    prove in its log that the arm is the arm it claims to be.
    """
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    replaced = 0
    for parent in model.modules():
        for child_name, child in list(parent.named_children()):
            if isinstance(child, LlamaRMSNorm):
                setattr(parent, child_name, FusedRMSNorm(
                    child.weight, child.variance_epsilon, mode=mode))
                replaced += 1
    return replaced
