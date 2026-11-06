# l40s_frontier — bytes, time and capacity on L40S (R-L1 … R-L3; job 630188)

**What it measures:** do the byte statements carry to Ada (R-L1)? What are stock
offload's time tax and its crossing `x*` on a PCIe-only node whose collectives are
host-staged (R-L1, R-L2; 621880 showed NCCL uses host memory on this topology)? And
for a realistic shape, Llama-3.2-3B, do the harness's lines and native 48 GB
brackets land where predicted (R-L3)? That is the first out-of-sample test of the
terms the shape sweep (7761) added post hoc. One exclusive 4-GPU L40S node: ~5–9 h expected, with a
guard that starts no arm after 22.5 h. Pre-registered predictions are in the header
of `l40s_frontier.sbatch`; the time and tier-A native predictions are **written by
the job itself** (`l40s_pred_time.txt`, `l40s_pred_native.txt`) before the arms
they predict. R-L4 (host memory) is not run; the link
test R-L5 is 621880.

Every file and output here starts with `l40s_`, because the clusters share one
file system.

## Step 0: on the local V100 cluster first (~45–60 min; job 7764)

```bash
cd remote/l40s_frontier
./l40s_build.sh                      # refresh code/ from the tree, pack ../l40s_frontier.tar.gz
sbatch l40s_v100_smoke.sbatch        # 4 V100 nodes; tests the SNAPSHOT that ships
```
A code check: the tier-A paths, plus the 3B geometry at 4 layers (the full 28 does
not fit a V100) and the native-ladder path. Predictions are in its header; check
`l40s_out/v100smoke_<jobid>/l40s_summary.txt` against them.

## Steps on ClusterA (login node, inside this directory)

```bash
# 1. environment: ~/l40s_venv, built once with ../l40s_venv_setup/l40s_venv_setup.sh

# 2. submit
sbatch -A <account> -p <partition> --qos=<qos> --nodes=1 \
       --exclusive --gres=gpu:l40s:4 --time=24:00:00 l40s_frontier.sbatch

# 3. when it finishes, pack the results
./l40s_collect.sh <jobid>            # -> l40s_frontier_results_<jobid>.tar.gz (a few MB)
```

The job reads `code/` when it **starts**, not when it is submitted: a rebuilt
`code/` copied over this one while the job waits reaches it. The header
predictions freeze at submission.

## What runs

| block | what | why |
|---|---|---|
| 0, K | node facts, card total; allocator cap check at 6 and 9 GiB | topology names every time number |
| B | tier-A byte arms (as on the A30) | R-L1: one common offset, lean 0.00 |
| T, P | static/offload pairs at x = 16/32/64 → in-job calibration → x = 100/150; two profiled arms | R-L1/R-L2: κ, the tax, `x*` with host-staged collectives; NCCL duty |
| R | Llama-3.2-3B slope points: static x = 16/32/48/96, offload 16/48/96 | R-L3: out-of-sample test of the shape term and the second-replay class |
| RS, RO | 3B native 48 GB brackets: static ~103, stock offload ~194 | R-L3: span 1.88 predicted at a realistic shape |
| C, N | tier-A capped ladders and native ladder | the Ada allocator reserve; native capacity predicted in-job |

`code/` is a snapshot of the V100 tree. The reader
is `tools/remote_frontier.py --platform l40s`, the same tool the A30 job runs.
