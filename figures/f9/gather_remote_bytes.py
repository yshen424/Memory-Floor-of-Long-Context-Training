"""Re-read every A30 / L40S / A100 byte arm against its harness line -> remote_bytes.json (fig9 (a)).
res = measured - current harness line; res_reg = measured - the line registered before the run.
Run from the repo root:  PYTHONPATH=. python3 figures/f9/gather_remote_bytes.py
"""
import glob, json, os, re
from harness import PLATFORMS, TIER_A, LLAMA32_3B, Run, static_line, floor_line
TAG = re.compile(r'^(?P<blk>[A-Z]+)_(?P<mode>static|offload|evict|staticcpu|offloadcpu)_(?P<dt>fp16|bf16)_(?P<attn>mem|flash)'
                 r'(?P<gqa>_gqa)?(?P<l3b>_llama3b)?_L(?P<L>\d+)_x(?P<x>\d+)(?:_cap(?P<cap>[\d.]+))?_n\d+(?:_r\d+)?$')
JOBS = [('A30','a30','633912','profile_results/a30_633912/a30_out/633912'),
        ('A30','a30','642029','profile_results/a30f_642029/a30f_out'),
        ('A30','a30','642032','profile_results/a30f_642032/a30f_out'),
        ('L40S','l40s','630188','profile_results/l40s4/630188'),
        ('L40S','l40s','634078','profile_results/l40s4/634078'),
        ('L40S','l40s','644212','profile_results/l40s4/644212'),
        ('A100','a100','8039176','profile_results/a100_8039176/a100_out/8039176'),
        ('A100','a100','8051891','profile_results/a100_8051891/a100_out/8051891')]
out=[]
for gpu, pk, job, root in JOBS:
    P = PLATFORMS[pk]
    for d in sorted(glob.glob(root+'/*')):
        m = TAG.match(os.path.basename(d)); f = os.path.join(d,'steps_rank0.json')
        if not m or not os.path.exists(f): 
            if os.path.isdir(d) and not os.path.exists(f): pass
            continue
        if m['blk']=='G': continue          # ballast arms: peak includes the idle ballast
        shape = (LLAMA32_3B if m['l3b'] else TIER_A).with_layers(int(m['L']))
        run = Run(shape=shape, world=4, repeat_kv=not m['gqa'], attn='flash' if m['attn']=='flash' else 'mem_efficient',
                  act_bytes=2, cpu_offload=m['mode'].endswith('cpu'))
        ln = static_line(run,P) if m['mode'].startswith('static') else floor_line(run,P,0)
        st = json.load(open(f))['steps']
        if len(st)<2: continue
        x=int(m['x']); meas=max(s['peak_mb'] for s in st[1:])
        # the flash plane was found in 633912; the follow-up jobs registered it
        flash_miss = m['attn'] == 'flash' and job == '633912'
        posthoc = flash_miss or ln.binding(x)=='deep replay'
        # the line as REGISTERED before the run: no flash plane (registered 0),
        # no deep-replay instant (added post hoc)
        reg = max([ln.c + ln.m * x] + [a.at(x) for n, a in ln.alts if n != 'deep replay'])
        if flash_miss:
            reg -= 1.5 * x
        out.append(dict(gpu=gpu,job=job,arm=os.path.basename(d),mode=m['mode'],shape='3B' if m['l3b'] else 'tierA',
                        x=x,cap=m['cap'],pred=ln.at(x),meas=meas,res=meas-ln.at(x),res_reg=meas-reg,instant=ln.binding(x),posthoc=posthoc))
json.dump(out, open('figures/f9/remote_bytes.json','w'), indent=1)
for g in ('A30','L40S','A100'):
    r=[o for o in out if o['gpu']==g]
    print(g, len(r), 'max|res| %.4f'%max(abs(o['res']) for o in r), 'posthoc', sum(o['posthoc'] for o in r),
          'max|res| pre-reg %.4f'%max(abs(o['res']) for o in r if not o['posthoc']))
for o in out:
    if abs(o['res'])>0.003 or o['posthoc']: print(o['gpu'],o['job'],o['arm'],round(o['res'],4),o['instant'])
