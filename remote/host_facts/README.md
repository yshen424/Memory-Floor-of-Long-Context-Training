# host_facts — Table 1's host and software columns

Read-only: no training and no GPU kernels, about a minute of wall clock per job. It collects what
the logs of the paper's jobs do not already hold: the CPU model and sockets, host memory and the
job's cgroup limit, the pinned-memory `ulimit -l`, OS and kernel, each GPU's PCIe generation and
width, the network link layer and rate, and torch/CUDA/cuDNN/NCCL from the venv each platform's
runs used.

**Collected:** V100 7865, A30 703038, L40S 703039, A100 8213074 (`host_facts_*.txt` in this
directory). 7865's file interleaves its four nodes; the script now writes one file per node and
concatenates them.

**Already known from the logs (no need to re-collect):** driver 595.71.05 on ClusterA and the DGX
A100 nodes; torch 2.6.0+cu124, CUDA runtime 12.4, NCCL 2.21.5, transformers 4.50.0 on A30/L40S
(and NCCL 2.21.5 on A100); A30 and A100 PCIe gen4 x16; DGX A100 2 sockets, 128 CPUs,
RealMemory 2,064,040 MB, kernel 5.14 (el9); L40S MemTotal 503.4 GiB, 4 NUMA nodes;
V100 nodes RealMemory 112,000 MB, 2 sockets x 8 cores, NCCL over mlx5 (RoCE/IB).

## Submit (one job per platform, from this directory)

```bash
# V100 (local): the same four-node shape as the paper's jobs
sbatch --nodes=4 --ntasks-per-node=1 --gres=gpu:1 host_facts.sbatch v100

# ClusterA A30 and L40S (copy this directory over first; ~/a30_venv and ~/l40s_venv must exist)
sbatch -A <account> -p <partition>  --qos=<qos>  --nodes=1 --exclusive --gres=gpu:a30:4  --time=00:10:00 host_facts.sbatch a30
sbatch -A <account> -p <partition> --qos=<qos> --nodes=1 --exclusive --gres=gpu:l40s:4 --time=00:10:00 host_facts.sbatch l40s

# ClusterB DGX A100 (~/a100_venv). A light request (1 GPU, 1 CPU, 8 GB, 5 minutes) lets the
# backfill scheduler start it in a gap on a busy node. Keep --constraint=dgx-A100: the facts
# must come from the DGX nodes the paper used, not the partition's four hpe-A100 nodes.
A100_OPTS="-A <account> -p <partition> --qos=<qos> --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G --constraint=dgx-A100 --gres=gpu:a100:1 --time=00:05:00"
sbatch --test-only $A100_OPTS host_facts.sbatch a100      # prints the expected start; submits nothing
sbatch $A100_OPTS host_facts.sbatch a100
# Fallback if the estimate is late: the preemptable partition holds the same DGX nodes; a
# one-minute job is unlikely to be preempted, and a preempted one can just be resubmitted.
#   swap in: -p <partition> --qos=<preemptable qos>
```

Nothing collected needs the whole node: CPU model, host memory, OS and kernel, PCIe link and NIC
are node properties, and the software comes from the venv. With one GPU the topology block shows
only that GPU; the 8-GPU NVSwitch topology is in `profile_results/a100_8051891/`.
`squeue -u $USER --start` shows the scheduler's start estimate.

To probe a node the paper actually used, add `-w <node>` (A30: nodeA019, nodeA022; L40S: nodeA034,
nodeA041; A100: nodeB-dgx007); any node of the partition is fine otherwise, but a busy named node can
queue longer.

**Exclusivity of the paper's remote jobs** (login node, no allocation):

```bash
./job_records.sh 633912 642029 642032 630188 634078 644212   # on ClusterA
./job_records.sh 8039176 8051891                              # on ClusterB
```

## Output

`host_facts_<platform>_<jobid>.txt` from each job (and the `.err` if one is non-empty), plus the
two `job_records_*.txt` files. The V100 cluster has accounting disabled, so its jobs' node lists
come from the job logs instead.
