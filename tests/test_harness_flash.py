"""The flash plane (post hoc on 633912, A30): every flash arm sits one
query-width fp16 plane above its mem-efficient line. Checks the harness against the
seven measured flash peaks and that the term is zero for the mem-efficient kernel.

    PYTHONPATH=. python3 tests/test_harness_flash.py
"""
from harness import PLATFORMS, Run, TIER_A, static_line, floor_line

A30 = PLATFORMS['a30']
# (mechanism, native GQA, x, measured step-1+ peak MiB), 633912 F block, L=64, world 4
MEASURED = [('static', False, 24, 4688.483), ('static', False, 48, 7900.170),
            ('offload', False, 48, 3754.266), ('static', True, 24, 4640.483),
            ('static', True, 48, 7804.170), ('offload', True, 48, 3658.266)]   # bf16 x=48 = fp16 x=48


def line(mode, gqa, attn):
    r = Run(shape=TIER_A.with_layers(64), world=4, repeat_kv=not gqa, attn=attn)
    return static_line(r, A30) if mode == 'static' else floor_line(r, A30, 0)


def main():
    worst = 0.0
    for mode, gqa, x, pk in MEASURED:
        d = pk - line(mode, gqa, 'flash').at(x)
        worst = max(worst, abs(d))
        assert abs(d) < 0.01, (mode, gqa, x, d)
    print(f"  ok  six distinct flash peaks on their lines, max |diff| {worst:.4f} MiB")
    for mode in ('static', 'offload'):
        dm = line(mode, False, 'flash').m - line(mode, False, 'mem_efficient').m
        assert abs(dm - 1.5) < 1e-9, dm
    print("  ok  flash slope = mem-efficient + 1.5000 MiB/x (2 B x n_q*d) on both mechanisms")
    try:
        Run(attn='math')
        raise AssertionError('regime A accepted')
    except ValueError:
        print("  ok  regime A (math) refused")
    print("ALL PASS")


if __name__ == '__main__':
    main()
