# a30_frontier — bytes, kernels, time and capacity on A30 (R-A1 … R-A5; job 633912)

**What it measures:** do the byte statements carry to Ampere unchanged (R-A1)?
Is bf16 with the memory-efficient kernel byte-identical to fp16 (R-A2)? What does
FlashAttention save, with and without native GQA (R-A3)? Where are stock
offload's time tax and its crossing `x*` on Ampere (R-A5)? Do the capped
brackets, and a native 24 GB bracket predicted in-job, land where predicted
(R-A4)? One exclusive 4-GPU A30 node: ~5–8 h expected, ~12 h if the A30 were no
faster than the V100, with a guard that starts no arm after 22.5 h.
Pre-registered predictions are in the header of `a30_frontier.sbatch`. The
capacity and time predictions are **written by the job itself**
(`a30_pred_time.txt`, `a30_pred_native.txt`), each before the arms it predicts.

Every file and output here starts with `a30_`, because the clusters share one
file system. Node facts are block 0.

## Step 0: on the local V100 cluster first (~1.5 h; job 7762)

```bash
cd remote/a30_frontier
./a30_build.sh                      # refresh code/ from the tree, pack ../a30_frontier.tar.gz
sbatch a30_v100_smoke.sbatch        # 4 V100 nodes; tests the SNAPSHOT that ships
```
Runs every code path at small x, checks that the bf16 and flash modes refuse on
sm_70, and tests the two in-job predictors against answers already measured on V100.
The capped ladders must reproduce 7760, and the native-bracket method must predict
the known [104, 108). Predictions are in its header. Check
`a30_out/v100smoke_<jobid>/a30_summary.txt` against them before copying the package
to ClusterA.

## Steps on ClusterA (login node, inside this directory)

```bash
# 1. environment: ~/a30_venv, built once with ../a30_venv_setup/a30_venv_setup.sh

# 2. submit
sbatch -A <account> -p <partition> --qos=<qos> --nodes=1 \
       --exclusive --gres=gpu:a30:4 --time=24:00:00 a30_frontier.sbatch

# 3. when it finishes, pack the results
./a30_collect.sh <jobid>            # -> a30_results_<jobid>.tar.gz (a few MB)
```

The results are `a30_results_<jobid>.tar.gz` (summary in
`a30_out/<jobid>/a30_summary.txt`). Traces stay on the cluster unless asked for
(`./a30_collect.sh <jobid> traces`).

## What runs

| block | what | why |
|---|---|---|
| 0, K | node facts (NVLink? P2P?), card total; allocator cap check at 6 and 9 GiB | topology names every time number; a broken cap voids C and N only |
| B | byte arms: static/offload at L=32/64, CPU offload, force-evict, offload x=100 | R-A1: one common offset across all arms, lean 0.00 |
| D | bf16 + memory-efficient | R-A2: byte-identical to fp16 |
| F | FlashAttention fp16/bf16; native GQA | R-A3: = mem-efficient; native GQA −2.0 MiB/x (or 0 if the kernel expands k/v) |
| T, P | static/offload pairs at x = 16/32/64 → in-job calibration → x = 100/150; two profiled arms | R-A5: κ, the tax, `x*`, attention-replay share, copy bandwidth |
| C | V100's four capped ladders (6/9 GiB, static/offload) | R-A4: same brackets as 7760 if the reserve is an allocator property |
| N | native static ladder, rungs written in-job | R-A4: native capacity predicted from C + the measured context |

`code/` is a snapshot of the V100 tree, refreshed by `./a30_build.sh`. It contains the driver, engine, harness and readers.
