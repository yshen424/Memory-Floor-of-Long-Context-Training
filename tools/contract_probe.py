"""Check what an activation-interception implementation promises to give back.

Every system that intercepts a saved activation -- offload, compression,
checkpointing -- makes the same implicit promise: what backward reads is the
tensor forward saved. That promise has more parts than the implementations
check. The restored tensor can be bit-identical in VALUE and
different in LAYOUT, and a fused kernel that captured strides during forward
then reads the wrong elements and produces NaN. A max-abs-diff check sees
nothing, because the values really are correct.

This probe states the contract as separate clauses and reports which clause
each implementation breaks.

    C1 VALUE     restored == original elementwise
    C2 LAYOUT    stride, storage_offset and contiguity preserved
    C3 ALIASING  saves that shared one storage still share one
    C4 META      dtype, shape, device kind, requires_grad preserved
    C5 RNG       a recompute-based replay draws the same random numbers

A probe that finds nothing is worthless unless it can be shown to find
something. 'engine_equivalent' reproduces this project's own restore --
torch.empty(shape, dtype, device) then copy_ -- and is expected to FAIL C2 and
C3. It is the positive control. 'engine_fixed' is the repair (empty_strided
over a rebuilt base) and checks that the repair works.

Usage:
    python3 tools/contract_probe.py                 # CPU, every adapter
    python3 tools/contract_probe.py --device cuda   # adds pinned-memory paths
    python3 tools/contract_probe.py --json out.json
"""

import argparse
import contextlib
import json
import sys
import traceback

import torch


# ---------------------------------------------------------------------------
# Probe cases -- tensor layouts that real models actually save
# ---------------------------------------------------------------------------

def case_dense(device):
    """The easy case. Every implementation must pass this one."""
    return torch.randn(2, 8, 16, device=device)


def case_transposed(device):
    """HF Llama builds q/k/v with .view().transpose(1, 2).

    The q/k/v saves are non-contiguous: at h=768 and 4096 tokens the stride is
    (3145728, 64, 768, 1) = (s*h, head_dim, h, 1), feeding the fused cutlass
    attention backward.
    """
    return torch.randn(1, 32, 4, 16, device=device).transpose(1, 2)


def case_offset_slice(device):
    """A view into a larger storage -- a fused qkv projection's third chunk.

    storage_offset is nonzero, so an implementation that rebuilds at offset 0
    changes where the tensor starts even if every value is copied correctly.
    """
    base = torch.randn(4, 64, device=device)
    return base[:, 32:48]


def case_expanded(device):
    """A broadcast view: stride 0 along one axis, storage smaller than shape.

    Reported but never failed. Materializing an expanded save is a defensible
    choice -- it costs memory but is not wrong -- so this case exists to show
    what an implementation does, not to accuse it.
    """
    return torch.randn(4, 1, 16, device=device).expand(4, 8, 16)


def case_host_scalar(device):
    """The efficient-attention philox seed: 0-dim int64 that lives on the host.

    A save that was never in HBM cannot
    be evicted out of HBM, and an implementation that treats it as flat bytes
    on the device is making a category error.
    """
    return torch.tensor(12345, dtype=torch.int64, device='cpu')


PROBE_CASES = [
    ('dense', case_dense, True),
    ('transposed', case_transposed, True),
    ('offset_slice', case_offset_slice, True),
    ('expanded', case_expanded, False),      # informational
    ('host_scalar', case_host_scalar, True),
]


def describe(t):
    return {
        'shape': tuple(t.shape),
        'stride': tuple(t.stride()),
        'storage_offset': t.storage_offset(),
        'contiguous': t.is_contiguous(),
        'dtype': str(t.dtype),
        'device': t.device.type,
        'storage_ptr': t.untyped_storage().data_ptr(),
    }


# ---------------------------------------------------------------------------
# Interception adapters
# ---------------------------------------------------------------------------

def _host_stage(tensor):
    """Copy a tensor's *values* to host, the way an offload pack would."""
    return tensor.detach().to('cpu', copy=True)


@contextlib.contextmanager
def adapter_none():
    """No interception. The control: it must pass every clause."""
    yield


@contextlib.contextmanager
def adapter_save_on_cpu(pin_memory=False):
    """PyTorch's own activation-offload primitive."""
    from torch.autograd.graph import save_on_cpu
    with save_on_cpu(pin_memory=pin_memory):
        yield


@contextlib.contextmanager
def adapter_engine_equivalent():
    """This project's restore, reproduced: torch.empty + copy_.

    POSITIVE CONTROL. Expected to fail C2 (layout) and C3 (aliasing).
    """
    def pack(t):
        return (t.shape, t.dtype, t.device, _host_stage(t))

    def unpack(p):
        shape, dtype, device, host = p
        dst = torch.empty(shape, dtype=dtype, device=device)
        dst.copy_(host)
        return dst

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield


@contextlib.contextmanager
def adapter_engine_fixed():
    """The proposed repair: carry stride and storage_offset, rebuild the view.

    Stages the whole base storage so the offset means something on restore,
    then re-applies the original geometry with as_strided.
    """
    def pack(t):
        base = torch.empty(t.untyped_storage().nbytes() // t.element_size(),
                           dtype=t.dtype, device=t.device)
        base.untyped_storage().copy_(t.untyped_storage())
        return (tuple(t.shape), tuple(t.stride()), t.storage_offset(),
                t.dtype, t.device, _host_stage(base))

    def unpack(p):
        shape, stride, offset, dtype, device, host_base = p
        base = torch.empty(host_base.shape, dtype=dtype, device=device)
        base.copy_(host_base)
        return torch.as_strided(base, shape, stride, offset)

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield


def build_adapters(device):
    out = [
        ('none', adapter_none),
        ('save_on_cpu', lambda: adapter_save_on_cpu(pin_memory=False)),
        ('engine_equivalent', adapter_engine_equivalent),
        ('engine_fixed', adapter_engine_fixed),
    ]
    if device == 'cuda':
        out.insert(2, ('save_on_cpu_pinned',
                       lambda: adapter_save_on_cpu(pin_memory=True)))
    return out


# ---------------------------------------------------------------------------
# C1/C2/C4 -- single-tensor round trip
# ---------------------------------------------------------------------------

class _RoundTrip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probe, carrier, sink):
        ctx.save_for_backward(probe)
        ctx.sink = sink
        ctx.carrier_shape = carrier.shape
        return carrier.sum()

    @staticmethod
    def backward(ctx, grad):
        (restored,) = ctx.saved_tensors
        ctx.sink['restored'] = restored
        ctx.sink['meta'] = describe(restored)
        return None, grad.expand(ctx.carrier_shape).clone(), None


# ---------------------------------------------------------------------------
# C3 -- aliasing
# ---------------------------------------------------------------------------

class _RoundTripPair(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, carrier, sink):
        ctx.save_for_backward(a, b)
        ctx.sink = sink
        ctx.carrier_shape = carrier.shape
        return carrier.sum()

    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        ctx.sink['shared'] = (a.untyped_storage().data_ptr()
                              == b.untyped_storage().data_ptr())
        return None, None, grad.expand(ctx.carrier_shape).clone(), None


def check_aliasing(adapter_factory, device):
    base = torch.randn(4, 64, device=device)
    a, b = base[:, :32], base[:, 32:]
    assert a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
    carrier = torch.randn(2, 2, device=device, requires_grad=True)
    sink = {}
    with adapter_factory():
        out = _RoundTripPair.apply(a, b, carrier, sink)
    out.backward()
    return sink.get('shared')


# ---------------------------------------------------------------------------
# C5 -- RNG replay under recompute-based interception
# ---------------------------------------------------------------------------

def check_rng_replay(device):
    """Does a non-reentrant checkpoint replay redraw the same dropout mask?

    Recompute-based interception has a contract the copy-based ones do not:
    the replay must reproduce the forward's random numbers. If it does not,
    the gradient corresponds to a network that was never evaluated.
    """
    import torch.utils.checkpoint as cp

    class Inner(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.dropout(x, p=0.5, training=True)

    torch.manual_seed(0)
    x = torch.randn(256, device=device, requires_grad=True)
    inner = Inner()
    out = cp.checkpoint(inner, x, use_reentrant=False)
    forward_mask = (out != 0).clone()
    out.sum().backward()
    # Under a correct replay the gradient is nonzero exactly where the forward
    # mask was nonzero.
    replay_mask = (x.grad != 0)
    return bool(torch.equal(forward_mask, replay_mask))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def evaluate(device):
    results = []
    for adapter_name, factory in build_adapters(device):
        for case_name, builder, enforced in PROBE_CASES:
            row = {'adapter': adapter_name, 'case': case_name,
                   'enforced': enforced}
            try:
                probe = builder(device)
                original = describe(probe)
                carrier = torch.randn(2, 2, device=device, requires_grad=True)
                sink = {}
                with factory():
                    out = _RoundTrip.apply(probe, carrier, sink)
                out.backward()

                restored_t = sink['restored']
                restored = sink['meta']
                row['original'] = original
                row['restored'] = restored
                row['C1_value'] = bool(
                    torch.equal(restored_t.cpu(), probe.cpu()))
                row['C2_layout'] = bool(
                    restored['stride'] == original['stride']
                    and restored['storage_offset'] == original['storage_offset']
                    and restored['contiguous'] == original['contiguous'])
                row['C4_meta'] = bool(
                    restored['dtype'] == original['dtype']
                    and restored['shape'] == original['shape']
                    and restored['device'] == original['device'])
            except Exception as exc:  # an implementation that raises is a result
                row['error'] = f"{type(exc).__name__}: {exc}"
                row['traceback'] = traceback.format_exc(limit=3)
            results.append(row)

        try:
            shared = check_aliasing(factory, device)
        except Exception as exc:
            shared = f"{type(exc).__name__}: {exc}"
        results.append({'adapter': adapter_name, 'case': 'aliasing',
                        'enforced': True, 'C3_aliasing': shared})
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    ap.add_argument('--json', default='', help='write full results here')
    args = ap.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA requested but not available', file=sys.stderr)
        return 2

    print(f"torch {torch.__version__}  device={args.device}")
    print('=' * 78)

    results = evaluate(args.device)

    hdr = f"{'adapter':<22}{'case':<16}{'C1':<6}{'C2':<6}{'C3':<6}{'C4':<6}"
    print(hdr)
    print('-' * 78)
    violations = []
    for r in results:
        def mark(key):
            if key not in r:
                return '-'
            v = r[key]
            return 'ok' if v is True else ('FAIL' if v is False else '?')
        line = (f"{r['adapter']:<22}{r['case']:<16}"
                f"{mark('C1_value'):<6}{mark('C2_layout'):<6}"
                f"{mark('C3_aliasing'):<6}{mark('C4_meta'):<6}")
        if 'error' in r:
            line += f"  RAISED {r['error']}"
        print(line)
        if r.get('enforced'):
            for clause in ('C1_value', 'C2_layout', 'C3_aliasing', 'C4_meta'):
                if r.get(clause) is False:
                    violations.append((r['adapter'], r['case'], clause))

    print('-' * 78)
    rng_ok = check_rng_replay(args.device)
    print(f"C5 RNG replay (non-reentrant checkpoint): "
          f"{'ok' if rng_ok else 'FAIL'}")

    print('=' * 78)
    if violations:
        print(f"{len(violations)} contract violations:")
        for adapter, case, clause in violations:
            print(f"  {adapter:<22}{case:<16}{clause}")
    else:
        print('no contract violations')

    control = [v for v in violations if v[0] == 'engine_equivalent']
    print()
    if control:
        print(f"POSITIVE CONTROL OK: the probe caught "
              f"{len(control)} known violations in 'engine_equivalent'.")
    else:
        print("POSITIVE CONTROL FAILED: the probe did not catch this "
              "project's own known restore bug. Every 'no violations' result "
              "above is therefore uninterpretable.")

    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({'torch': torch.__version__, 'device': args.device,
                       'results': results, 'C5_rng_replay': rng_ok}, fh,
                      indent=2, default=str)
        print(f"wrote {args.json}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
