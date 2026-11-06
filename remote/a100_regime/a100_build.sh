#!/bin/bash
# LOCAL only: refresh code/ from the V100 tree and pack ../a100_regime.tar.gz.
# Run it after any change to the driver, harness or readers, and before the V100
# smoke: the smoke tests the SNAPSHOT, which is what ships.
set -euo pipefail
cd "$(dirname "$0")"
SRC=../..
TOOLS="remote_frontier.py a30_followup.py copy_exposure.py analyze_nccl_duty.py attn_recompute_share.py host_copy_bw.py"
rm -rf code; mkdir -p code/tools
cp "$SRC"/{train_fsdp_profile.py,fsdp_profiler.py,chunked_ce.py,fused_rmsnorm.py} code/
cp -r "$SRC"/memory_manager "$SRC"/harness code/
for t in $TOOLS; do cp "$SRC/tools/$t" code/tools/; done
find code -name __pycache__ -prune -exec rm -rf {} +
( cd "$SRC" && sha1sum train_fsdp_profile.py fsdp_profiler.py chunked_ce.py fused_rmsnorm.py \
      memory_manager/*.py harness/*.py $(for t in $TOOLS; do echo tools/$t; done) ) > code/SNAPSHOT.sha1
date -Is > code/SNAPSHOT.date
cd .. && tar czf a100_regime.tar.gz --exclude=a100_out --exclude='*.log' --exclude='*.err' a100_regime
echo "packed $(pwd)/a100_regime.tar.gz ($(du -h a100_regime.tar.gz | cut -f1)); snapshot $(cat a100_regime/code/SNAPSHOT.date)"
