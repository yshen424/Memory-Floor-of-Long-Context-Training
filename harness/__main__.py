"""python3 -m harness --platform v100 --layers 64 --world 4 --ladder 96:400:4 [--shape llama3b] [--json out.json]"""
import argparse
from dataclasses import replace

from .config import PLATFORMS, Run, SHAPES
from .report import header_block, predict, write_json
from .timemodel import TimeCalibration

# V100, L=64, world 4: T(x) from 7744 S + 7747 M1; kappa from 7758 x>=24.
TIME_CAL = {('v100', 64, 4): TimeCalibration(0.02123, 0.0397, 2.066, 0.032,
                                             'T: 7744 S + 7747 M1; kappa: 7758 x>=24')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--platform', default='v100')
    ap.add_argument('--shape', default='tier_a', choices=sorted(SHAPES))
    ap.add_argument('--layers', type=int, default=None, help='default: the shape\'s own depth (tier_a: 64)')
    ap.add_argument('--world', type=int, default=4)
    ap.add_argument('--ranks_per_node', type=int, default=None, help='host memory is shared by this many ranks (x_host)')
    ap.add_argument('--cpu_offload', action='store_true')
    ap.add_argument('--no_repeat_kv', action='store_true', help='native-GQA kernel (k,v saved at kv width)')
    ap.add_argument('--attn', default='mem_efficient', choices=('mem_efficient', 'flash'))
    ap.add_argument('--mechanisms', default='static,stock_offload')
    ap.add_argument('--ladder', default='', help='start:stop:step in x')
    ap.add_argument('--json')
    a = ap.parse_args()
    shape = SHAPES[a.shape]
    layers = a.layers or (64 if a.shape == 'tier_a' else shape.layers)
    run = Run(shape=shape.with_layers(layers), world=a.world, cpu_offload=a.cpu_offload,
              repeat_kv=not a.no_repeat_kv, attn=a.attn)
    plat = PLATFORMS[a.platform]
    if a.ranks_per_node:
        plat = replace(plat, ranks_per_node=a.ranks_per_node)
    ladder = list(range(*map(int, a.ladder.split(':')))) if a.ladder else []
    p = predict(run, plat, a.mechanisms.split(','), ladder, TIME_CAL.get((a.platform, layers, a.world)) if a.shape == 'tier_a' else None)
    print(header_block(p))
    if a.json:
        write_json(p, a.json)


if __name__ == '__main__':
    main()
