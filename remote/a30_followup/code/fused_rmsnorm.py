import torch
import torch.nn as nn

class _FusedRMSNormSaveInput(torch.autograd.Function):

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
        gw -= x_hat * (gw * x_hat).mean(-1, keepdim=True)
        gw *= rstd
        return gw.to(x.dtype), grad_weight.to(weight.dtype), None

class _FusedRMSNormSaveOutput(torch.autograd.Function):

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
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    replaced = 0
    for parent in model.modules():
        for child_name, child in list(parent.named_children()):
            if isinstance(child, LlamaRMSNorm):
                setattr(parent, child_name, FusedRMSNorm(
                    child.weight, child.variance_epsilon, mode=mode))
                replaced += 1
    return replaced
