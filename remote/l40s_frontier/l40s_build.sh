#!/bin/bash
# LOCAL only: refresh code/ from the V100 tree and pack the package.
#   ./l40s_build.sh            -> code/ snapshot + ../l40s_frontier.tar.gz
# Run it again after any change to the driver, engine, harness or readers, and
# before the V100 smoke: the smoke tests the SNAPSHOT, which is what ships.
set -euo pipefail
cd "$(dirname "$0")"
SRC=../..
rm -rf code; mkdir -p code/tools
cp "$SRC"/{train_fsdp_profile.py,fsdp_profiler.py,chunked_ce.py,fused_rmsnorm.py} code/
cp -r "$SRC"/memory_manager "$SRC"/harness code/
cp "$SRC"/tools/{remote_frontier.py,attn_recompute_share.py,analyze_nccl_duty.py} code/tools/
find code -name __pycache__ -prune -exec rm -rf {} +
( cd "$SRC" && sha1sum train_fsdp_profile.py fsdp_profiler.py chunked_ce.py fused_rmsnorm.py \
      memory_manager/*.py harness/*.py tools/remote_frontier.py tools/attn_recompute_share.py \
      tools/analyze_nccl_duty.py ) > code/SNAPSHOT.sha1
date -Is > code/SNAPSHOT.date
cd .. && tar czf l40s_frontier.tar.gz --exclude=l40s_out --exclude=l40s_venv --exclude='*.log' --exclude='*.err' l40s_frontier
echo "packed $(pwd)/l40s_frontier.tar.gz ($(du -h l40s_frontier.tar.gz | cut -f1)); snapshot $(cat l40s_frontier/code/SNAPSHOT.date)"
