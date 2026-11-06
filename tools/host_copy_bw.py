"""Pinned host<->device copy bandwidth, each visible GPU alone and then all at once.

Stock offload's copies are pinned-memory transfers of whole boundaries, so this is the
bandwidth that sets their busy time. On an HGX/DGX A100 board two GPUs share a PCIe switch
(and its x16 uplink), so the concurrent figure can be about half the solo one; the job
quotes whichever matches how its four ranks copy (all at once).

    python3 tools/host_copy_bw.py [--mib 1024] [--reps 5] [--json out.json]

GB/s = 1e9 bytes per second (harness/ stores GiB/s).
"""
import argparse
import json
import time

import torch


def bench(devs, direction, nbytes, reps):
    """Seconds per copy, every device copying concurrently (one stream each)."""
    bufs = []
    for d in devs:
        h = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        g = torch.empty(nbytes, dtype=torch.uint8, device=f'cuda:{d}')
        bufs.append((d, h, g, torch.cuda.Stream(device=d)))
    for _ in range(2):                                    # warm-up
        for d, h, g, s in bufs:
            with torch.cuda.stream(s):
                (g.copy_(h, non_blocking=True) if direction == 'h2d' else h.copy_(g, non_blocking=True))
        for d, *_ in bufs:
            torch.cuda.synchronize(d)
    t0 = time.perf_counter()
    for _ in range(reps):
        for d, h, g, s in bufs:
            with torch.cuda.stream(s):
                (g.copy_(h, non_blocking=True) if direction == 'h2d' else h.copy_(g, non_blocking=True))
    for d, *_ in bufs:
        torch.cuda.synchronize(d)
    return (time.perf_counter() - t0) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mib', type=int, default=1024)
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--json')
    a = ap.parse_args()
    n = a.mib * 2**20
    devs = list(range(torch.cuda.device_count()))
    if not devs:
        raise SystemExit('host_copy_bw: no CUDA device visible; nothing measured')
    out = {'mib': a.mib, 'devices': devs, 'solo': {}, 'concurrent': {}}
    for d in devs:
        for dr in ('d2h', 'h2d'):
            out['solo'].setdefault(str(d), {})[dr] = n / bench([d], dr, n, a.reps) / 1e9
    for dr in ('d2h', 'h2d'):
        out['concurrent'][dr] = n / bench(devs, dr, n, a.reps) / 1e9      # per GPU, all copying
    for d in devs:
        s = out['solo'][str(d)]
        print(f"BW GPU {d} solo: D2H {s['d2h']:6.2f} GB/s  H2D {s['h2d']:6.2f} GB/s")
    c = out['concurrent']
    print(f"BW all {len(devs)} concurrent, per GPU: D2H {c['d2h']:6.2f} GB/s  H2D {c['h2d']:6.2f} GB/s")
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
