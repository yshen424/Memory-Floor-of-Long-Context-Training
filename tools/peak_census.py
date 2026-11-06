"""Size census of the live set at the reconstructed peak of a CUDA allocator
snapshot: every live block grouped by size, with its allocation site. Names the
CONSTANT storage at the peak; tools/peak_liveset.py names the
token-proportional planes.

Usage: python3 tools/peak_census.py <cuda_snapshot_rankN.pkl> [bytes to flag ...]
"""
import pickle, collections, sys
s = pickle.load(open(sys.argv[1], 'rb'))
want = [int(v) for v in sys.argv[2:]]
live = {b['address']: b['size'] for sg in s['segments'] for b in sg['blocks'] if b['state'] == 'active_allocated'}
tr = s['device_traces'][0]
tot = sum(live.values()); best = (tot, len(tr), dict(live))
for i in range(len(tr) - 1, -1, -1):
    e = tr[i]; a = e['action']
    if a == 'alloc' and e['addr'] in live: tot -= live.pop(e['addr'])
    # TRAP (7763): replay free_REQUESTED, not free_completed. A block freed while
    # another stream still uses it completes its free late, but the allocated
    # counter (and so the measured peak) drops at the request. Replaying
    # completions overstated 7763's O1 x=32 peak by 6.4 GiB (63 chunk LM-head
    # gradients); requests reconstruct all three 7763 snapshots to 0.001 MiB.
    elif a == 'free_requested': live[e['addr']] = e['size']; tot += e['size']
    if tot > best[0]: best = (tot, i, dict(live))
tot, ipk, ls = best
last = {}
for i in range(ipk):
    e = tr[i]
    if e['action'] == 'alloc': last[e['addr']] = e
def lab(e):
    if e is None: return '(before trace window)'
    fr = [f for f in e.get('frames', []) if f.get('filename', '').endswith('.py')]
    keep = [f"{f['name']}@{f['filename'].split('/')[-1]}:{f['line']}" for f in fr
            if any(k in f['filename'] for k in ('fsdp', 'transformers', 'torch/nn', 'checkpoint', 'autograd', 'chunked_ce', 'fused_rmsnorm', 'optim', 'train_fsdp'))]
    return ' <- '.join(keep[:3]) if keep else (fr[0]['name'] + '@' + fr[0]['filename'].split('/')[-1] if fr else 'C++ only')
print(f"peak {tot/2**20:.4f} MiB at trace index {ipk}/{len(tr)}; {len(ls)} blocks")
# the event at the peak
for j in range(max(0, ipk - 2), min(len(tr), ipk + 2)):
    e = tr[j]; print('  event', j, e['action'], f"{e['size']/2**20:.4f} MiB", lab(e))
by = collections.defaultdict(list)
for a, z in ls.items(): by[z].append(lab(last.get(a)))
for z in sorted(by, reverse=True):
    c = collections.Counter(by[z])
    flag = '  <== WANTED' if any(abs(z - w) < 4096 for w in want) else ''
    print(f"{len(by[z]):4d} x {z/2**20:12.5f} MiB = {len(by[z])*z/2**20:10.4f}{flag}   " + '; '.join(f"{n}x {k}" for k, n in c.most_common(2)))
