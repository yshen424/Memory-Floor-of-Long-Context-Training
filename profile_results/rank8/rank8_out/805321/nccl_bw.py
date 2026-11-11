import os, time, torch, torch.distributed as dist
dist.init_process_group('nccl'); torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
r, w = dist.get_rank(), dist.get_world_size()
for mib in (64, 256):
    t = torch.ones(mib * 2**20 // 2, dtype=torch.float16, device='cuda')
    for _ in range(3): dist.all_reduce(t)
    torch.cuda.synchronize(); t0 = time.time(); n = 10
    for _ in range(n): dist.all_reduce(t)
    torch.cuda.synchronize(); dt = (time.time() - t0) / n
    if r == 0:
        print(f'NCCL all_reduce {mib} MiB fp16 over {w} ranks: {dt * 1e3:.2f} ms, bus bandwidth '
              f'{mib * 2**20 * 2 * (w - 1) / w / dt / 1e9:.2f} GB/s', flush=True)
dist.destroy_process_group()
