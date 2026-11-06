#!/bin/bash -l
# Build ~/a30_venv ONCE on the ClusterA LOGIN node:   bash a30_venv_setup.sh
# Every A30 job (a30_frontier, a30_followup and later ones) uses this venv and no
# other; A30_VENV=<path> overrides. Same pins as ~/l40s_venv and the versions
# 633912 ran (python312 module, torch 2.6.0+cu124, transformers 4.50.0), so the
# A30 and L40S results share one software stack while the two venvs stay
# independent (rebuilding one never touches the other's jobs).
# Takes ~5-10 min (the torch wheel is ~800 MB). Needs no GPU.
set -euo pipefail
DEST=${1:-$HOME/a30_venv}
if [ -e "$DEST" ]; then
    echo "$DEST already exists. Jobs rely on it; to rebuild, move it aside first:"
    echo "    mv $DEST ${DEST}_old_\$(date +%Y%m%d)   # a moved venv is broken: delete it after the rebuild"
    exit 1
fi
MOD=""
if module load python312 2>/dev/null; then MOD=python312; fi
PY=$(command -v python3.12 || true)
[ -n "$PY" ] || { echo "no python3.12 (module python312 did not load); ask before using another python: torch 2.6 + this code were tested on 3.12 (L40S) and 3.10 (V100)"; exit 1; }
echo "using $PY ($($PY --version 2>&1))${MOD:+ via module $MOD}"
"$PY" -m venv "$DEST"
[ -n "$MOD" ] && echo "$MOD" > "$DEST/.module"
source "$DEST/bin/activate"
pip install --upgrade pip
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.50.0 numpy
# ── checks: the versions, sm_80 in the build, and the launcher every job checks ──
python3 - <<'PY'
import torch, transformers
assert torch.__version__ == '2.6.0+cu124', torch.__version__
assert transformers.__version__ == '4.50.0', transformers.__version__
arch = torch._C._cuda_getArchFlags() if hasattr(torch._C, '_cuda_getArchFlags') else ''
assert 'sm_80' in arch, f"this torch build has no sm_80 kernels: {arch!r}"
import torch.distributed.run  # noqa: F401  (torchrun needs it)
print(f"torch {torch.__version__} (cuda {torch.version.cuda}), transformers {transformers.__version__}, arch {arch}")
PY
torchrun --help >/dev/null && echo "launcher check: torchrun ok"
{ echo "built $(date -Is) on $(hostname) by a30_venv_setup.sh"; "$PY" --version; pip freeze | grep -iE '^(torch|transformers|numpy|nvidia-nccl)'; } > "$DEST/VERSIONS"
cat "$DEST/VERSIONS"
echo "A30 environment ready: $DEST"
