# a100_regime — stock offload's time tax on a DGX A100 node (R-H1, R-H2)

**What it measures:** where stock offload's time tax and its crossing `x*` sit on an
NVSwitch node, where collectives ride NVLink and only offload's copies cross the host
link (R-H1); and whether the byte lines carry to the A100 as they did to the A30 (R-H2).
Block H runs the same pairs with collectives forced through host memory on the same
GPUs, so the effect of moving collectives off the host link is measured directly.

One exclusive DGX A100 node, 4 of its 8 GPUs, world 4: ~4–6 h expected, ~8.6 h if the
A100 were no faster than the A30; no arm starts after 22 h. Predictions are in the
header of `a100_regime.sbatch`; the time prediction for x=100/150 is written by the job
itself (`a100_pred_time.txt`) before those arms run. No capacity arms.

## Step 0: on the local V100 cluster (~60–80 min)

```bash
cd remote/a100_regime
./a100_build.sh                      # refresh code/ from the tree, pack ../a100_regime.tar.gz
sbatch a100_v100_smoke.sbatch        # 4 V100 nodes; tests the snapshot that ships
```
Check `a100_out/v100smoke_<jobid>/a100_summary.txt` against the smoke's S-P1..S-P6
before copying the package to ClusterB.

## On ClusterB (login node, inside this directory)

The venv is `~/a100_venv` (`A100_VENV` overrides); the job checks its launcher before any arm.

```bash
sbatch -A <account> -p <partition> --qos=<qos> --nodes=1 \
       --constraint=dgx-A100 --exclusive --gres=gpu:a100:4 --time=24:00:00 a100_regime.sbatch
./a100_collect.sh <jobid>            # when it finishes -> a100_results_<jobid>.tar.gz (a few MB)
```
**`--constraint=dgx-A100` is required.** The partition also has four `hpe-A100` nodes
(nodeB-gpu001–004) whose GPU fabric is not known to be NVSwitch. The job checks the topology
itself and stops with exit 4 if the GPUs are not NVLink-connected. It stops with exit 3
if a GPU will not open. Either way, resubmit with `--exclude=<host>`.

Traces stay on the cluster unless asked for (`./a100_collect.sh <jobid> traces`).

## What runs

| block | what | answers |
|---|---|---|
| 0 | node facts, GPU preflight, NVSwitch check, host-copy bandwidth (solo and all 4 at once) | names the topology every time number needs |
| B | static/offload byte arms at x = 24/48 (L=32 static x=24) | R-H2: the V100 lines carry |
| T | static/offload pairs, 4 repeats at x = 16/32/48/64, then the in-job prediction, then 2 repeats at x = 100/150 | R-H1: the tax curve and `x*` |
| H | the same pairs with `NCCL_P2P_DISABLE=1` at x = 32/100 | does NVSwitch lower offload's cost? |
| P | profiled pairs at x = 64/150, split into compute / copy-only / other / idle time | transfer share, attention-replay share |
| G | offload + an idle ballast the size of static's extra memory, x = 100/150 | offload's cost at equal footprint (the A30's footprint effect) |

`code/` is a snapshot of the V100 tree.

## Shared-node copy (job 8039176)

`a100_regime_shared.sbatch` is the same job on half a DGX node (4 GPUs, 64 CPUs,
900 GB, 12 h):
```bash
sbatch -A <account> -p <partition> --qos=<qos> --nodes=1 \
       --constraint=dgx-A100 --gres=gpu:a100:4 --export=ALL,A100_SIBLING=<exclusive job id> \
       a100_regime_shared.sbatch
```
If it starts first, it cancels the still-pending exclusive job itself. If the exclusive job
starts first, `scancel` this one. Time numbers from a shared run are read against
`node_load.txt` (load and allocation logged before every arm); bytes are unaffected.

## Exclusive confirmation rerun (job 8051891, `A100_PLAN=confirm`)

8039176 ran shared, so its time curve may carry the neighbour's PCIe traffic. The confirm
plan reruns the bandwidth probe, block T exactly as 8039176 ran it, and one profiled pair
at x=64, on an EXCLUSIVE node (~2.3 h). It asks for all 8 GPUs and uses one per PCIe
switch, the layout 8039176 had:
```bash
sbatch -A <account> -p <partition> --qos=<qos> --nodes=1 \
       --constraint=dgx-A100 --exclusive --gres=gpu:a100:8 --time=24:00:00 \
       --export=ALL,A100_PLAN=confirm a100_regime.sbatch
./a100_collect.sh <jobid>
```
Predictions (C-B … C-P) are in the script header, above FAILURE CLASSES.
