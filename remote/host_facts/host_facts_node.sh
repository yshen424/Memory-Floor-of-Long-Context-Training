#!/bin/bash
# ONE NODE'S HOST AND SOFTWARE FACTS for Table 1. Read-only:
# no GPU work beyond nvidia-smi and torch's version queries. host_facts.sbatch
# runs it once per allocated node through srun; every probe is guarded, so a
# missing tool prints "n/a" instead of stopping the job.
set -u
sec() { echo; echo "--- $1 ---"; }
try() { "$@" 2>/dev/null || echo "n/a ($1)"; }

echo "=== NODE $(hostname)   job ${SLURM_JOB_ID:-none}   $(date -Is) ==="

sec "CPU (model, sockets, cores, NUMA)"
try lscpu | grep -E '^(Model name|Socket\(s\)|Core\(s\) per socket|Thread\(s\) per core|CPU\(s\)|NUMA node\(s\)|CPU max MHz|L3 cache)'
try lscpu | grep -E '^NUMA node[0-9]+ CPU'

sec "host memory"
grep -E '^(MemTotal|MemAvailable|HugePages_Total|Hugepagesize)' /proc/meminfo
try numactl -H | grep -E 'size|available'
echo "ulimit -l (pinned/locked memory, KiB): $(ulimit -l)"
CG=$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup 2>/dev/null)
for f in /sys/fs/cgroup${CG}/memory.max /sys/fs/cgroup/memory${CG}/memory.limit_in_bytes; do
    [ -r "$f" ] && echo "job cgroup memory limit ($f): $(cat "$f")"
done

sec "SLURM view of this node"
try scontrol show node "$(hostname -s)" | grep -E 'CoresPerSocket|Sockets|RealMemory|MemSpecLimit|OS=|Gres=|AvailableFeatures'
if [ -n "${SLURM_JOB_ID:-}" ]; then
    try scontrol show job "$SLURM_JOB_ID" | grep -E 'OverSubscribe|Shared|NumNodes|TRES|MinMemory'
    echo "other jobs on this node:"; try squeue -h -w "$(hostname -s)" -o '%i %u %T %b %C'
fi

sec "OS and kernel"
grep -E '^PRETTY_NAME' /etc/os-release 2>/dev/null; uname -r

sec "GPU, driver, PCIe link (max and host max; current may downshift at idle)"
try nvidia-smi --query-gpu=index,name,memory.total,driver_version,pci.bus_id --format=csv
try nvidia-smi --query-gpu=index,pcie.link.gen.max,pcie.link.width.max,pcie.link.gen.current,pcie.link.width.current --format=csv
try nvidia-smi --query-gpu=index,pcie.link.gen.hostmax,pcie.link.gen.gpumax --format=csv
nvidia-smi -q 2>/dev/null | grep -E 'CUDA Version|Product Name|Max *: *[0-9]+x?$|Current *: *[0-9]+x?$|Host Max|Device Max|Device Current' | head -40
sec "GPU topology"
try nvidia-smi topo -m

sec "network adapters (link layer, rate)"
if [ -d /sys/class/infiniband ]; then
    for p in /sys/class/infiniband/*/ports/*; do
        echo "$(basename "$(dirname "$(dirname "$p")")") port $(basename "$p"): $(cat "$p/link_layer" 2>/dev/null), $(cat "$p/rate" 2>/dev/null), $(cat "$p/state" 2>/dev/null)"
    done
else echo "no /sys/class/infiniband"; fi
try lspci | grep -iE 'infiniband|ethernet|mellanox|network' | head -8
try ip -br link | grep -v '^lo' | head -8

sec "software (python and torch from the job's venv)"
python3 - <<'PY' 2>&1
import sys
print("python", sys.version.split()[0])
try:
    import torch
    print("torch", torch.__version__, "| CUDA runtime", torch.version.cuda,
          "| cuDNN", torch.backends.cudnn.version(),
          "| NCCL", ".".join(map(str, torch.cuda.nccl.version())))
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0))
except Exception as e:
    print("torch n/a:", repr(e)[:200])
try:
    import transformers; print("transformers", transformers.__version__)
except Exception as e:
    print("transformers n/a:", repr(e)[:200])
PY
echo "=== END $(hostname) ==="
