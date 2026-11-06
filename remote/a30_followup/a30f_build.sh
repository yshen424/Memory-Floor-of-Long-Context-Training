#!/bin/bash
# LOCAL only: refresh code/ from the V100 tree and pack ../a30_followup.tar.gz.
# Run it after any change to the driver, harness or readers, and before the V100
# smoke: the smoke tests the SNAPSHOT, which is what ships.
set -euo pipefail
cd "$(dirname "$0")"
SRC=../..
rm -rf code; mkdir -p code/tools
cp "$SRC"/{train_fsdp_profile.py,fsdp_profiler.py,chunked_ce.py,fused_rmsnorm.py} code/
cp -r "$SRC"/memory_manager "$SRC"/harness code/
cp "$SRC"/tools/{a30_followup.py,attn_recompute_share.py,analyze_nccl_duty.py,peak_liveset.py} code/tools/
find code -name __pycache__ -prune -exec rm -rf {} +
( cd "$SRC" && sha1sum train_fsdp_profile.py fsdp_profiler.py chunked_ce.py fused_rmsnorm.py \
      memory_manager/*.py harness/*.py tools/a30_followup.py tools/attn_recompute_share.py tools/analyze_nccl_duty.py tools/peak_liveset.py ) > code/SNAPSHOT.sha1
date -Is > code/SNAPSHOT.date
cd .. && tar czf a30_followup.tar.gz --exclude=a30f_out --exclude='*.log' --exclude='*.err' a30_followup
echo "packed $(pwd)/a30_followup.tar.gz ($(du -h a30_followup.tar.gz | cut -f1)); snapshot $(cat a30_followup/code/SNAPSHOT.date)"
