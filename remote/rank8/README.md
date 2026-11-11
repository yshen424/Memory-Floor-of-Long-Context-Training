# rank8 — eight ranks across two A30 nodes, and on one DGX A100

**What it measures.** Whether the byte lines hold when the ranks span nodes (slopes unchanged, intercepts moved by the sharded state), how far a second node moves a model's frontier, and how the analysis changes for a 7B-like model with grouped-query attention, where a boundary is small against the replayed layer.

| job | where | blocks | measures |
|---|---|---|---|
| `a30_rank8.sbatch` | 2 exclusive A30 nodes on ClusterA, 8 GPUs, InfiniBand | B, BR, KS8/KO8, BR4/KS4/KO4, T | base-shape lines at world 8 across two nodes; a Llama-3.2-3B-like model on one node against two (frontiers predicted in-job); its offloading cost on two nodes |
| `a100_rank8.sbatch` | 1 exclusive DGX A100 on ClusterB, all 8 GPUs, NVSwitch | B, BZ, T, ZS/ZO | base-shape lines at world 8 in one node; a 7B-like model's lines, offloading cost and frontiers |

Every prediction is in the job script's header. The frontier ladders write their in-job prediction (`<prefix>_pred_native_<BLK>.txt`) before they run, and the time block writes `<prefix>_pred_time.txt` before its last length runs. Both jobs run their blocks most-decisive first.

Results: `profile_results/rank8/rank8_out/` holds the full jobs 821554 (A30) and 8548619 (A100), the A30 two-node smoke 805321, and the V100 smokes of these scripts (the A30 half 7943, the A100 half 7944).

## Step 0: on the local V100 cluster (~45–60 min)

```bash
cd remote/rank8
./rank8_build.sh                      # refresh code/ from the tree, pack ../rank8.tar.gz
sbatch rank8_v100_smoke.sbatch        # 4 V100 nodes x 1 GPU: both job scripts in smoke form
```
Check `rank8_out/v100smoke_a30_<jobid>/a30_summary.txt` and `rank8_out/v100smoke_a100_<jobid>/a100_summary.txt` against the smoke's S-P1..S-P4 before copying the package to the clusters.

## On the clusters (inside this directory)

**ClusterA, the two-node smoke first (~20 min).** It checks the two-node launch, InfiniBand and the 3B arms.
```bash
sbatch --gres=gpu:a30:4 --time=01:00:00 --export=ALL,R8_PLAN=smoke a30_rank8.sbatch
sbatch --gres=gpu:a30:4 a30_rank8.sbatch          # the full job: 2 nodes (in the script), ~2-6 h
```

**ClusterB:**
```bash
sbatch --gres=gpu:a100:8 a100_rank8.sbatch        # ~5-10 h, 24 h requested
```

**When a job finishes:** `./rank8_collect.sh <jobid>` writes `rank8_results_<jobid>.tar.gz`.

**Reading the shipped results** (from the repository root):
```bash
python3 tools/remote_frontier.py read profile_results/rank8/rank8_out/821554 --platform a30    # or 8548619 --platform a100
python3 tools/rank8_time.py profile_results/rank8/rank8_out/8548619 --fit 16,32,64              # the paper's time numbers
```
The reader prints every registered verdict; its time pooling drops the first two steps of each arm, as the in-job prediction did. The paper drops only the first, as for the four-rank platforms, and `tools/rank8_time.py` reproduces its numbers.

**Exit codes.**
- 3: a GPU would not open (the host is named in the summary). Resubmit with `--exclude=<host>`.
- 4: the A100 node is not NVSwitch.
- 5: the NCCL test across the job's ranks failed, a launcher or fabric problem.

## Tooling

- **`tools/remote_frontier.py`:**
  - Arm tags carry `_w<world>`; the 7B-like geometry is `_mistral7b`.
  - `natpred` writes a native ladder's prediction from the Ampere capacity reserve (the A30's capped ladders, 633912: delta 59.5 MiB, alpha 0.093) plus the job's measured card total and the memory outside the allocator at that rank count.
  - `timecal` takes `--xs` (calibration lengths) and `--at` (lengths predicted).
  - `read` judges every ladder that has a prediction file and prints the spans.
- **`harness/`:** `python3 -m harness --shape llama3b|mistral7b --world W --ranks_per_node N` prints these predictions; x_host depends on ranks per node.
- **The launcher (`rank8_common.sh`):** an arm whose world fits one node runs `torchrun --standalone` on the batch node; otherwise it runs `srun` with one `torchrun` per node (c10d rendezvous, every CPU, no binding).
