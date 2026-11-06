"""The harness and the engine's runtime byte model must not drift apart silently.

The harness is a standalone closed form, and this test keeps it tied to the
engine's byte model (memory_manager/access_predictor.py) so the repository has
one byte model in substance.

What must hold:
  1. The migratable unit is identical: one boundary = 2 B x hidden per token.
  2. The engine's non-migratable count stays on the SAFE side of the harness
     (a gate that under-projects OOMs), by exactly the known gap: 744 B/token
     = 0.7266 MiB/x. The engine carries a coarser count (the gate.up product
     plane as live, k/v at kv width, its own attention slot); the harness
     carries the plane-by-plane count. If this gap changes, one of the two
     models moved: re-derive, do not loosen the number.

Run:  PYTHONPATH=. python3 tests/test_harness_engine_consistency.py
"""
import sys

from harness import Run, TIER_A, slope_terms
from memory_manager.access_predictor import AccessPredictor

KNOWN_GAP = 744          # bytes per token, engine minus harness (difference of two exact counts)


def main():
    ok = True
    run = Run(shape=TIER_A, world=4)
    mig, non = slope_terms(run)
    h_boundary = mig['boundaries'] // run.shape.layers
    h_non = sum(non.values())

    ap = AccessPredictor(num_layers=64, seq_len=1024, verbose=False, hidden_size=768,
                         intermediate_size=2048, num_heads=12, num_kv_heads=4,
                         batch_size=1, dtype_bytes=2, materializes_attn_scores=False,
                         bundle_kinds=('layer',), vocab_size=32000, chunked_ce_tokens=256)
    bm = ap._build_bytes_map()
    tokens = 1024
    e_boundary = ap.boundary_bytes() // tokens
    interior = sum(v for k, v in bm.items()
                   if k.startswith('act_layer_0_') and not k.endswith('layer_residual'))
    e_non = (interior + ap._layer_bwd_transient_bytes()) // tokens

    def check(name, cond, detail):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name}   {detail}")
        ok &= cond

    check('boundary bytes per token identical', e_boundary == h_boundary,
          f'engine {e_boundary}, harness {h_boundary}')
    check('engine non-migratable on the safe side', e_non >= h_non,
          f'engine {e_non}, harness {h_non} B/token')
    check('gap equals the known 744 B/token', e_non - h_non == KNOWN_GAP,
          f'gap {e_non - h_non} B/token = {(e_non - h_non) / 1024:.4f} MiB/x')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
