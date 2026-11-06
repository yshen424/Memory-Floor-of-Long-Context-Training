# l40s_linktest — the PCIe link test on the L40S cluster (job 621880)

**What it measures:** on a PCIe-only node (L40S, no NVLink), do host↔device
copies and NCCL collectives **exclude** each other, **share** the link, or run **independently**? And does real offload traffic
slow the collectives? One exclusive 4-GPU node, about 20–30 min.
Pre-registered predictions are in the header of `l40s_linktest.sbatch`.

Every file and output here starts with `l40s_`, because the clusters share one
file system.

## Steps on the L40S cluster (login node, inside this directory)

```bash
# 1. environment: ~/l40s_venv, built once with ../l40s_venv_setup/l40s_venv_setup.sh

# 2. (optional) when would it start? Nothing is queued.
sbatch --test-only l40s_linktest.sbatch

# 3. submit
sbatch -A <account> -p <partition> --qos=<qos> --nodes=1 \
       --exclusive --gres=gpu:l40s:4 --time=01:00:00 l40s_linktest.sbatch

# 4. when it finishes, pack the results
./l40s_collect.sh <jobid>          # -> l40s_results_<jobid>.tar.gz (a few MB)
```

The results are `l40s_results_<jobid>.tar.gz` (summary in
`l40s_out/<jobid>/l40s_summary.txt`). Traces stay on the
cluster unless asked for (`./l40s_collect.sh <jobid> traces`, ~1–1.5 GB).

## What runs

| part | what | why |
|---|---|---|
| 0 | GPUs, PCIe gen/width, `topo -m`, P2P matrix, NUMA, RAM | the topology decides how to read everything else |
| 1 | `code/l40s_linkbench.py`, no model: copies alone, all GPUs at once, both directions at once, all-gather alone, all-gather + copies concurrently. Run with NCCL's default transport, then with `NCCL_P2P_DISABLE=1` (host-staged, V100-like) | the clean link answer, plus a verdict line |
| 2 | training arms at the L=32 anchor (regime B, whole-layer, chunked CE 256, b=1, seq 8192): T1 static, T2 + forced-overlap probe, T3 stock offload; T4/T5 = T1/T2 with P2P off, only if NCCL chose P2P by default | the same question inside real FSDP training |
| 3 | `code/analyze_nccl_duty.py` on every arm's traces | NCCL duty, copy bandwidth vs overlap, collective slowdown |

`code/` holds the driver and engine this job ran, plus the microbenchmark and
the trace analyzer.
