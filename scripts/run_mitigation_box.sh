#!/usr/bin/env bash
# Run mitigation conditions for models that require the multi-GPU host. The explicit CUDA check
# prevents a costly model load from starting while the host is in CPU-only mode.
# Usage (on the box, from repo root):  bash scripts/run_mitigation_box.sh  2>&1 | tee results/mitigation_box.log
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -c "import torch; assert torch.cuda.is_available(), 'no GPU — switch the box to GPU mode first'"
python -c "import torch; print('GPU OK |', torch.cuda.device_count(), 'x', torch.cuda.get_device_name(0))"
if [ "$#" -gt 0 ]; then MODELS=("$@"); else MODELS=("Qwen/Qwen2.5-32B-Instruct" "Qwen/Qwen2.5-72B-Instruct"); fi
SETS=("data/propositions.jsonl" "data/arith_hard.jsonl")   # phenomenon + worst-harm regime
mkdir -p results
for M in "${MODELS[@]}"; do
  for P in "${SETS[@]}"; do
    echo "######## MIT $M / $P ########"
    HF_HUB_OFFLINE=1 python scripts/mitigation_gate.py --model "$M" --props "$P" \
      2>&1 | grep -vE "Loading|it/s\]|torch_dtype|Setting|generation flags|attention mask|^\s*$" || true
  done
done
echo "MIT_BOX_ALLDONE — copy results/mitigation_box.log back to the workstation"
