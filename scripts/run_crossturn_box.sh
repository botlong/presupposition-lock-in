#!/usr/bin/env bash
# Cross-turn + downstream-harm batch for a multi-GPU box (32B/72B).
# Nothing large loads until its gates pass:
#   gate 1  CUDA matmul smoke — import success does not prove kernel support on new GPUs;
#   gate 2  render gate per model (tokenizer-only, seconds) — the closed-turn episode must
#           render for THIS tokenizer, else the run silently degrades to the single-turn design;
#   gate 3  3-item pilot through the full pipeline before each 120-item run.
# Snapshots are recorded BEFORE running, and
# every parquet's row count is asserted before the batch may declare success.
# Usage (on the box, from repo root):
#   bash scripts/run_crossturn_box.sh <model-id-or-local-path> [more models...] \
#     2>&1 | tee results/crossturn_box.log
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
[ "$#" -ge 1 ] || { echo "usage: $0 <model-id-or-path> [...]"; exit 2; }

python -c "import torch" 2>/dev/null || python -m pip install -q torch
python -c "import transformers, accelerate, numpy, pandas, pyarrow, scipy, yaml" 2>/dev/null \
  || python -m pip install -q -U transformers accelerate numpy pandas pyarrow scipy pyyaml || true
if ! python -c "import torch; (torch.randn(8,8,device='cuda')@torch.randn(8,8,device='cuda')).sum().item()" 2>/dev/null; then
  echo ">> GPU matmul failed (likely sm/CUDA mismatch) -> installing torch cu128"
  python -m pip install -q --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
fi
python -c "import torch; (torch.randn(8,8,device='cuda')@torch.randn(8,8,device='cuda')).sum().item(); print('CUDA smoke ok')"

for M in "$@"; do
  TAG=$(basename "$M" | sed 's/___/./g; s/-Instruct$//')
  echo "════════════════ $M  (tag: $TAG) ════════════════"
  python scripts/record_snapshot.py --model "$M"
  python scripts/check_render.py --model "$M"
  python scripts/crossturn_gate.py --model "$M" --n 3 --out "results/_pilot_crossturn_$TAG.parquet"
  python scripts/crossturn_gate.py --model "$M" --out "results/crossturn_gate_$TAG.parquet" \
    2>&1 | tee "results/crossturn_${TAG}_box.log"
  python scripts/downstream_harm.py --model "$M" --out "results/downstream_harm_$TAG.parquet" \
    2>&1 | tee "results/downstream_harm_${TAG}_box.log"
  python - "$TAG" <<'EOF'
import sys
import pandas as pd
tag = sys.argv[1]
ct = pd.read_parquet(f"results/crossturn_gate_{tag}.parquet")
dh = pd.read_parquet(f"results/downstream_harm_{tag}.parquet")
assert len(ct) in (1680, 1920), f"crossturn rows {len(ct)} not in (1680, 1920)"
assert ct.groupby("protocol").size().nunique() == 1, "protocol row imbalance"
assert len(dh) == 1440, f"downstream rows {len(dh)} != 1440"
print(f"row counts ok: crossturn={len(ct)} downstream={len(dh)}")
EOF
  rm -f "results/_pilot_crossturn_$TAG.parquet"
done

tar czf "results/crossturn_box_$(date +%F).tar.gz" \
  results/crossturn_gate_*.parquet results/downstream_harm_*.parquet \
  results/model_snapshots.jsonl results/*_box.log
echo "ALL BOX RUNS DONE -> results/crossturn_box_$(date +%F).tar.gz"
