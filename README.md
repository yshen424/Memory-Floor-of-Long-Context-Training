# Artifact: What Bounds Long-Context Training on GPU Clusters: A Memory Floor No Offloading Placement Can Pass

This repository holds what the paper's measurements and predictions rest on:
- the job scripts, whose headers carry each prediction registered before the run;
- every run's measured peaks and the allocator snapshots taken at the peak;
- the prediction model and the tests that hold it to the measurements;
- the training driver and instrument that produced the runs;
- the scripts that draw the paper's figures;
- the planner run (POET's exact program on the base configuration).

The paper's results are four-rank jobs; two- and three-rank arms test world size, and two eight-rank jobs (across two A30 nodes, and on one DGX A100) test the rank count and multiple nodes. The jobs ran on V100, A30, L40S and A100 nodes with PyTorch 2.6.

## Layout

| Path | What it holds |
|---|---|
| `harness/` | The prediction model: per-mechanism peak lines, binding instants, capacity brackets, time model. `python3 -m harness --help`. |
| `tests/` | Standalone test scripts: the model against every measured peak, plus checks of the levers, the checkpoint unit, the observer and the job readers. |
| `tools/` | Readers that judge each job against its registration, and the snapshot analysers (`peak_liveset.py`, `peak_census.py`). |
| `train_fsdp_profile.py`, `fsdp_profiler.py`, `fused_rmsnorm.py`, `chunked_ce.py`, `memory_manager/` | The driver (FSDP training loop), the per-event memory observer, the two levers and the instrument used to place boundaries and record absence at the peak. |
| `profile_results/` | One folder per job, holding the job script beside its outputs. `runs_4gpu.csv` has one row per arm: platform, job, configuration, peaks, step times, losses and the instrument's pick counts. |
| `remote/` | The remote job packages (build, submit and collect scripts, READMEs), each with the `code/` it ran, plus `host_facts/` for the platform table. |
| `figures/` | One folder per figure script, the shared style (`paperstyle.py`), and the measured constants used by several figures (`world4.py`). |
| `planner/` | The planner run: POET's integer program on the base configuration's graph, its logs and how to rerun it (`planner/README.md`). |

## Requirements

- **Prediction model, tools and figures:** Python ≥ 3.10 with numpy, matplotlib and Pillow. They run on a CPU.
- **Tests:** most also need PyTorch 2.6 and transformers 4.50. They run on a CPU.
- **Training driver:** 4 GPUs, PyTorch 2.6.0+cu124, transformers 4.50.0 and SLURM. See a job script for the full flag set.

The pinned versions are in `requirements.txt` (`pip install -r requirements.txt`; PyTorch from its CUDA 12.4 index).

## Reproducing the paper's numbers

All commands run from the repository root.

```bash
# Registered predictions for a configuration (here: V100, L=64, four ranks, ladder x=100..108):
PYTHONPATH=. python3 -m harness --platform v100 --layers 64 --world 4 --ladder 100:112:4

# The model against the measurements (each script prints its table and verdict):
for t in tests/test_*.py; do PYTHONPATH=. python3 "$t" || echo "FAILED $t"; done

# A figure (writes figures/fN/figN.png and figures/rendered/figN.png):
cd figures/f15 && python3 plot_fig15.py
```

| Paper | Script | | Paper | Script |
|---|---|---|---|---|
| Fig. 1 | `figures/f1/plot_fig1.py` | | Fig. 6 | `figures/f14/plot_fig14.py` |
| Fig. 2 | (a drawing) | | Fig. 7 | `figures/f15/plot_fig15.py` |
| Fig. 3 | `figures/f5/plot_fig5.py` | | Fig. 8 | `figures/f10/plot_fig10.py` |
| Fig. 4 | `figures/f6/plot_fig6.py` | | Fig. 9 | `figures/f12/plot_fig12.py` |
| Fig. 5 | `figures/f13/plot_fig13.py` | | Fig. 10 | `figures/f9/plot_fig9.py` |

**The eight-rank jobs and the sizing table:** `remote/rank8/README.md` (the reader's report, and `tools/rank8_time.py` for the time numbers as the paper pools them). **The planner run:** `planner/README.md`.

Fig. 7 (registered against measured peaks) reads `figures/f15/parity_points.csv` for the V100 arms. It reads `figures/f9/remote_bytes.json` for the A30, L40S and A100 arms, which `PYTHONPATH=. python3 figures/f9/gather_remote_bytes.py` rebuilds from `profile_results/`.

## Inside a job folder

- **The job script** (`run_*.sh`, `*_as_run.sbatch`): its header holds the configuration, the arms and the predictions registered before submission; its body runs the arms.
- **One folder per arm**, holding:
  - `steps_rank<r>.json`: the configuration the arm ran (`config`) and, per step, the allocator's peak (`peak_mb`), the step time and the loss.
  - `profile_rank0.json.gz`: the observer's per-event record for rank 0, including the peak within every interval between schedule events.
  - `cuda_snapshot_rank<r>.pkl` (snapshot arms only): a PyTorch allocator snapshot (`torch.cuda.memory._dump_snapshot`). `tools/peak_liveset.py` reconstructs the live set at the peak from it.
  - on most V100 arms, `stderr.txt`; on remote arms, `stdout.txt` and `stderr.txt`, and on some `gpu.csv`, `attn_share.json` and `duty.json`.
- **Remote jobs' predictions written in-job** (`*_pred_*.txt`), before the arms they predict, and the job readers' summaries (`*_summary.txt`, `decomp_x*.json`, `liveset.txt`).

Not included:
- **Profiler traces** (`*.pt.trace.json`, about 2 GB). The attention-replay share derived from them is kept in `attn_share.json`.
- **Observer records of ranks other than 0.** Every rank's step peaks are in `steps_rank<r>.json`.
- **Job-level scheduler logs.** Each arm's own outputs are kept.

## Notes on the text in this repository

- **Job scripts.** Headers were condensed to the configuration, the arms, the registered predictions and the failure classes. Every command and every registered value is unchanged.
- **Code as run.** The code under `remote/*/code/` and `profile_results/*/code_<job>/` is the code each job ran, with comments, docstrings and build stamps removed and a few help and provenance strings shortened; its executable statements are unchanged. The top-level code is the current version, and its tests, tools and figure scripts read this repository's files.
- **Anonymization.** The following were replaced throughout, including inside the snapshots' stack frames:
  - home paths, by `/path/to/repo` and `/home/user`;
  - the clusters' names, by `ClusterA` (A30, L40S) and `ClusterB` (A100);
  - node host names, by `nodeA…` and `nodeB-…`;
  - the scheduler account, partitions and QOS, by `<account>`, `<partition>` and `<qos>`;
  - private IP addresses, by `10.0.0.N`.

## License

MIT (`LICENSE`).
