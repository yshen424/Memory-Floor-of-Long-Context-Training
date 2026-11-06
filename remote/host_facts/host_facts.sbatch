#!/bin/bash -l
#SBATCH --job-name=host_facts
#SBATCH --time=00:10:00
#SBATCH --output=host_facts_%j.log
#SBATCH --error=host_facts_%j.err
#
# HOST AND SOFTWARE FACTS FOR TABLE 1, one short job per platform. Read-only: no
# training, no GPU kernels. The partition, QoS and GPU request go ON THE COMMAND
# LINE ONLY; the one argument names the platform, which picks the venv whose
# torch/CUDA/cuDNN/NCCL the paper's runs used. Exact commands: README.md here.
#
# Writes host_facts_<platform>_<jobid>.txt next to this script: one block per
# allocated node (CPU model and sockets, host memory and the job's cgroup
# limit, pinned-memory ulimit, OS and kernel, GPU PCIe generation and width,
# topology, network link layer and rate, software versions). Send that file back.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
PLAT=${1:?usage: sbatch <cluster options> host_facts.sbatch v100|a30|l40s|a100}
case "$PLAT" in
    v100) VENV="" ;;                          # local nodes: the system python3 the V100 jobs use
    a30)  VENV=${A30_VENV:-$HOME/a30_venv} ;;
    l40s) VENV=${L40S_VENV:-$HOME/l40s_venv} ;;
    a100) VENV=${A100_VENV:-$HOME/a100_venv} ;;   # never `module purge` on ClusterB
    *) echo "unknown platform $PLAT"; exit 2 ;;
esac
if [ -n "$VENV" ]; then
    [ -f "$VENV/bin/activate" ] || { echo "VENV $VENV MISSING"; exit 1; }
    if [ -f "$VENV/.module" ]; then module load "$(cat "$VENV/.module")" 2>/dev/null || true; fi
    source "$VENV/bin/activate"
fi

OUT=host_facts_${PLAT}_${SLURM_JOB_ID}.txt
{
    echo "### host facts: platform $PLAT, job $SLURM_JOB_ID, nodes $SLURM_JOB_NODELIST ($SLURM_NNODES), $(date -Is)"
    echo "### venv: ${VENV:-system python3}   python3: $(command -v python3)"
    # One file per node, concatenated after: four nodes writing one stream interleave.
    PARTS=host_facts_parts_${SLURM_JOB_ID}; mkdir -p "$PARTS"
    srun --ntasks-per-node=1 --ntasks="$SLURM_NNODES" \
        bash -c "./host_facts_node.sh > $PARTS/\$(hostname -s).txt 2>&1" < /dev/null
    cat "$PARTS"/*.txt && rm -rf "$PARTS"
} > "$OUT" 2>&1
echo "wrote $OUT ($(wc -l < "$OUT") lines)"
