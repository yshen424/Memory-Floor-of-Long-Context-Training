import argparse

from .config import PLATFORMS, Run, TIER_A
from .report import header_block, predict, write_json
from .timemodel import TimeCalibration

TIME_CAL = {('v100', 64, 4): TimeCalibration(0.02123, 0.0397, 2.066, 0.032,
                                             'T: 7744 S + 7747 M1; kappa: 7758 x>=24')}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--platform', default='v100')
    ap.add_argument('--layers', type=int, default=64)
    ap.add_argument('--world', type=int, default=4)
    ap.add_argument('--cpu_offload', action='store_true')
    ap.add_argument('--no_repeat_kv', action='store_true', help='native-GQA kernel (k,v saved at kv width)')
    ap.add_argument('--mechanisms', default='static,stock_offload')
    ap.add_argument('--ladder', default='', help='start:stop:step in x')
    ap.add_argument('--json')
    a = ap.parse_args()
    run = Run(shape=TIER_A.with_layers(a.layers), world=a.world, cpu_offload=a.cpu_offload,
              repeat_kv=not a.no_repeat_kv)
    plat = PLATFORMS[a.platform]
    ladder = list(range(*map(int, a.ladder.split(':')))) if a.ladder else []
    p = predict(run, plat, a.mechanisms.split(','), ladder, TIME_CAL.get((a.platform, a.layers, a.world)))
    print(header_block(p))
    if a.json:
        write_json(p, a.json)

if __name__ == '__main__':
    main()
