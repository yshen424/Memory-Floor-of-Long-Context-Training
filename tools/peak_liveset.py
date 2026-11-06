"""Reconstruct the live set at the peak from a CUDA allocator snapshot and name
every token-proportional block from its allocation stack.

Usage: python3 tools/peak_liveset.py <cuda_snapshot_rankN.pkl> <tokens per rank>
"""
import pickle, collections, sys
if len(sys.argv) != 3:
    sys.exit(__doc__)
p=sys.argv[1]; TOK=int(sys.argv[2])
s=pickle.load(open(p,'rb'))
live={b['address']:b['size'] for sg in s['segments'] for b in sg['blocks'] if b['state']=='active_allocated'}
tr=s['device_traces'][0]
tot=sum(live.values()); best=(tot,len(tr),dict(live))
for i in range(len(tr)-1,-1,-1):
    e=tr[i]; a=e['action']
    if a=='alloc' and e['addr'] in live: tot-=live.pop(e['addr'])
    # free_requested, not free_completed: the trap is explained in tools/peak_census.py
    elif a=='free_requested': live[e['addr']]=e['size']; tot+=e['size']
    if tot>best[0]: best=(tot,i,dict(live))
tot,ipk,ls=best
# last alloc event (before peak) per live address
lastalloc={}
for i in range(ipk):
    e=tr[i]
    if e['action']=='alloc': lastalloc[e['addr']]=e
def label(e):
    if e is None: return '(allocated before trace window)'
    fr=[f for f in e.get('frames',[]) if f.get('filename','').endswith('.py')]
    keep=[f"{f['name']}@{f['filename'].split('/')[-1]}:{f['line']}" for f in fr if any(k in f['filename'] for k in ('transformers','torch/nn','torch/utils/checkpoint','torch/autograd','chunked_ce','fused_rmsnorm','memory_manager','train_fsdp'))]
    return ' <- '.join(keep[:3]) if keep else (fr[0]['name']+'@'+fr[0]['filename'].split('/')[-1] if fr else 'C++ only')
rows=collections.defaultdict(list)
for a,sz in ls.items():
    bpt=sz/TOK
    if bpt>=1 and abs(bpt-round(bpt*2)/2)<0.02 and sz>TOK:  # token-proportional
        rows[round(bpt,1)].append(label(lastalloc.get(a)))
tot_tp=0
for bpt in sorted(rows, reverse=True):
    print(f"--- {bpt} B/token x{len(rows[bpt])}  (= {bpt*len(rows[bpt])/1024:.4f} MiB/x)")
    tot_tp+=bpt*len(rows[bpt])
    for l,n in collections.Counter(rows[bpt]).most_common(): print(f"    {n} x {l}")
print('token-proportional total', tot_tp, 'B/token =', round(tot_tp/1024,4), 'MiB/x')
