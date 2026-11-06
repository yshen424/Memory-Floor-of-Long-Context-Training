"""WHEN does the whole-layer boundary save reach the engine's seam, and is a
bundle open at that moment?

WHY SEEING IT IS NOT ENOUGH. tests/test_checkpoint_seam_visibility.py shows
that the outer pack_fn SEES the per-layer residual boundary under
checkpoint_wrapper(NO_REENTRANT). This test asks when. That matters because
pack_fn's very first guard is:

    if self._current_bundle is None:   ->   return tensor unchanged

and `_current_bundle` is set by enter_bundle(layer_idx, kind), which the
observer calls from per-SUBMODULE forward hooks (kind is 'attention' or 'mlp').
The boundary tensor is saved by the checkpoint frame's input saver, which runs
at the WRAPPER's entry -- before any submodule inside the layer has been
touched. If no bundle is open at that instant, the engine sees the boundary,
declines it at the first guard, and reports a clean run with zero decisions.

That is the dangerous outcome: not a crash, a SILENT NO-OP. A whole-layer engine
arm would come back with a plausible peak, zero evictions, and nothing in the
logs saying why.

THE OUTCOMES.
  H-A  BOUNDARY ARRIVES WITH NO BUNDLE OPEN (submodule hooks only). The remedy
       is a layer-level bundle entered on the WRAPPER's forward-pre-hook, not
       on the inner layer's, since the inner module's hook fires too late by
       construction.
  H-B  BOUNDARY ARRIVES INSIDE A SUBMODULE BUNDLE. Worse than H-A, because the
       engine would book layer L's boundary against some (L', kind) bundle and
       the mis-attribution would be invisible -- relief would be credited to a
       bundle that does not own the bytes. Would need an explicit reject.
  H-C  A WRAPPER-LEVEL PRE-HOOK FIRES BEFORE THE BOUNDARY SAVE. This is the
       wiring fsdp_profiler.py uses for the whole-layer unit; the probe tests
       it directly rather than assuming it.

Also counts the zero-element placeholders and reports whether they arrive in
the same window as the boundary, since they must be passed through un-booked
and the two are only separable if we know where they land.

Device-agnostic; hook dispatch does not depend on device.

Run:  PYTHONPATH=. python3 tests/test_checkpoint_bundle_ordering.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing, checkpoint_wrapper, CheckpointImpl)

B, S, D = 2, 16, 32
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
RESID = (B, S, D)


class Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.q(x) * torch.sigmoid(self.v(x))


class Mlp(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.up = nn.Linear(d, 2 * d, bias=False)
        self.down = nn.Linear(2 * d, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.up(x)))


class Layer(nn.Module):
    """Stands in for LlamaDecoderLayer: two bundle-bearing submodules and a
    residual boundary in and out."""

    def __init__(self, d):
        super().__init__()
        self.attn = Attn(d)
        self.mlp = Mlp(d)

    def forward(self, x):
        x = x + self.attn(x)
        return x + self.mlp(x)


class Stack(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.layers = nn.ModuleList([Layer(d) for _ in range(n)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def run(wrapper_level_hook):
    """Returns the log of (shape, numel, bundle-open-at-that-instant)."""
    torch.manual_seed(0)
    model = Stack(D, 3).to(DEV)
    inner_layers = list(model.layers)

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(
            m, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=lambda m: isinstance(m, Layer))

    state = {'bundle': None, 'in_forward': True, 'decided': []}
    log = []

    # --- per-SUBMODULE bundle hooks (the observer's per-module wiring) ------
    for idx, layer in enumerate(inner_layers):
        for kind, sub in (('attention', layer.attn), ('mlp', layer.mlp)):
            sub.register_forward_pre_hook(
                lambda m, a, i=idx, k=kind: state.__setitem__('bundle', (i, k)))
            sub.register_forward_hook(
                lambda m, a, o: state.__setitem__('bundle', None))

    # --- a DECISION hook on the inner layer: on_pre_forward(L), attached to
    # the inner LlamaDecoderLayer by class name. If this fires after the
    # boundary save, the decision map is empty when pack_fn reads it, which is
    # a second and independent reason the engine would no-op.
    for idx, layer in enumerate(inner_layers):
        layer.register_forward_pre_hook(
            lambda m, a, i=idx: state['decided'].append(i))

    # --- the layer-level bundle, entered on the WRAPPER ---------------------
    if wrapper_level_hook:
        for idx, wrapped in enumerate(model.layers):
            wrapped.register_forward_pre_hook(
                lambda m, a, i=idx: state.__setitem__('bundle', (i, 'layer')))
            wrapped.register_forward_hook(
                lambda m, a, o: state.__setitem__('bundle', None))

    def pack(t):
        if state['in_forward']:
            log.append((tuple(t.shape), t.numel(), state['bundle'],
                        tuple(state['decided'])))
        return t

    x = torch.randn(B, S, D, device=DEV, requires_grad=True)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = model(x)
    state['in_forward'] = False
    out.sum().backward()
    return log, x.grad


def summarize(label, log):
    print(f"{label}")
    print(f"  {'shape':<16} {'numel':>7}  {'bundle open':<18} decided-so-far")
    for shape, numel, bundle, decided in log:
        mark = ''
        if shape == RESID:
            mark = '   <-- BOUNDARY'
        elif numel == 0:
            mark = '   <-- placeholder'
        print(f"  {str(shape):<16} {numel:>7}  {str(bundle):<18}"
              f"{str(list(decided)):<12}{mark}")
    bounds = [b for s, n, b, d in log if s == RESID]
    holes = [b for s, n, b, d in log if n == 0]
    decisions = [d for s, n, b, d in log if s == RESID]
    print(f"  boundaries: {len(bounds)}   placeholders: {len(holes)}")
    return bounds, holes, decisions


def main():
    print(f"device: {DEV}   residual shape: {RESID}   layers: 3\n")

    log_a, grad_a = run(wrapper_level_hook=False)
    bounds_a, holes_a, dec_a = summarize(
        "SUBMODULE: per-submodule bundle hooks only",
        log_a)

    print()
    log_b, grad_b = run(wrapper_level_hook=True)
    bounds_b, holes_b, dec_b = summarize(
        "WRAPPER: a layer-level bundle entered on the wrapper's pre-hook",
        log_b)

    if not torch.allclose(grad_a, grad_b, atol=1e-5):
        print("\nWARNING: adding the wrapper-level hook changed gradients.")

    print("\n" + "=" * 68)
    if not bounds_a:
        print("PROBE INVALID -- no residual-shaped boundary reached pack_fn at "
              "all, which contradicts tests/test_checkpoint_seam_visibility.py. "
              "Stop and re-check.")
        return

    if all(b is None for b in bounds_a):
        print("VERDICT (submodule): H-A -- the boundary arrives with NO BUNDLE OPEN.")
        print("  pack_fn's first guard returns it unchanged, so a whole-layer")
        print("  engine built on submodule hooks is a SILENT NO-OP: it would run,")
        print("  decide nothing, and report a clean step. Per-submodule hooks")
        print("  fire strictly after the checkpoint frame saves the boundary.")
    elif any(b is not None for b in bounds_a):
        print("VERDICT (submodule): H-B -- the boundary arrives INSIDE a submodule")
        print(f"  bundle {[b for b in bounds_a if b is not None]}, so the engine")
        print("  would MIS-ATTRIBUTE layer-boundary bytes to a bundle that does")
        print("  not own them. Needs an explicit reject, not just a new hook.")

    print()
    if bounds_b and all(b is not None and b[1] == 'layer' for b in bounds_b):
        print("VERDICT (wrapper): H-C CONFIRMED -- a wrapper-level pre-hook")
        print("  fires BEFORE the boundary save, so every boundary lands in its")
        print(f"  own layer bundle: {bounds_b}")
        print("  The whole-layer unit is implementable by adding a layer-level")
        print("  bundle; the inner module's hook cannot work by construction.")
    else:
        print("VERDICT (wrapper): H-C REJECTED -- the wrapper-level hook does")
        print(f"  NOT cover the boundary save. Observed: {bounds_b}")
        print("  The boundary is saved outside ANY module hook's scope, so the")
        print("  bundle must be opened somewhere other than a forward-pre-hook.")

    print()
    own = [i in d for i, d in enumerate(dec_a)]
    if not any(own):
        print("DECISION ORDER: the inner layer's pre-hook -- a decision hook")
        print("  placed there -- has NOT fired when layer L's")
        print(f"  boundary is packed (decided-so-far at each boundary: {dec_a}).")
        print("  So even with a bundle open, the decision map for L would be empty")
        print("  at pack time. The decision point belongs on the wrapper too;")
        print("  a layer-level bundle alone is necessary but not sufficient.")
    else:
        print(f"DECISION ORDER: layer L's decision IS available at its own")
        print(f"  boundary save ({dec_a}); only the bundle must sit on the wrapper.")

    print()
    print(f"placeholders (numel 0) land in: {holes_b}")
    print("  These must be passed through un-booked: a 0-byte save that reaches")
    print("  the pool gate acquires a real pinned buffer (every bucket satisfies")
    print("  `size < 0` as false, so the smallest bucket matches) and charges it.")


if __name__ == '__main__':
    main()
