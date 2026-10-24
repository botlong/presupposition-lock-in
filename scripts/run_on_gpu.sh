#!/usr/bin/env bash
# Run large Qwen models on a multi-GPU host and retain raw-logit measurements.
# Using the same readout as smaller models is required for a same-family scale comparison.
# Usage (on the GPU box, inside the repo):  bash scripts/run_on_gpu.sh  2>&1 | tee results/frontier_gpu.log
set -eo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"   # make `import actionlockin` work without pip install -e

# Preserve a working torch install because replacing it can break the host's CUDA compatibility.
python -c "import torch" 2>/dev/null || python -m pip install -q torch
# Only install if something's actually missing — the box already has the stack; a -U upgrade
# would need network (PyPI), waste time, and risk breaking a known-good transformers/torch.
python -c "import transformers, accelerate, numpy, pandas, scipy, yaml" 2>/dev/null \
  || python -m pip install -q -U transformers accelerate numpy pandas pyarrow scipy pyyaml || true
# Import success does not prove that installed kernels support the GPU architecture, so execute a
# real CUDA matmul before starting an expensive model load.
if ! python -c "import torch; (torch.randn(8,8,device='cuda')@torch.randn(8,8,device='cuda')).sum().item()" 2>/dev/null; then
  echo ">> GPU matmul failed (likely sm_120/CUDA mismatch) -> installing torch cu128"
  python -m pip install -q --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
fi
python -c "import torch; x=(torch.randn(8,8,device='cuda')@torch.randn(8,8,device='cuda')).sum().item(); print('GPU OK | torch',torch.__version__,'| gpus',torch.cuda.device_count(),'|',torch.cuda.get_device_name(0))"
export HF_HOME="${HF_HOME:-$PWD/models/huggingface}"
export TOKENIZERS_PARALLELISM=false

# Pass model ids as args to override; default = both 32B (needs ~70GB disk, fits 1x96G/80G)
#    and 72B (needs ~145GB disk + ~150GB VRAM => 2 cards). DISK-LIMITED? run just 32B:
#       bash scripts/run_on_gpu.sh Qwen/Qwen2.5-32B-Instruct
if [ "$#" -gt 0 ]; then MODELS=("$@"); else MODELS=("Qwen/Qwen2.5-32B-Instruct" "Qwen/Qwen2.5-72B-Instruct"); fi
echo "models to run: ${MODELS[*]}"; df -h . | tail -1
# proposition sets: factual (phenomenon) + arithmetic (uncertainty/harm) + hard arithmetic (difficulty)
SETS=("data/propositions.jsonl" "data/arith_props.jsonl" "data/arith_hard.jsonl")

mkdir -p results
for M in "${MODELS[@]}"; do
  for P in "${SETS[@]}"; do
    echo "######## $M  /  $P ########"
    # pipefail propagates a real run failure (set -e aborts); the inner `|| true` only absorbs
    # grep's exit-1 when every line is filtered out, so a genuine python crash is NOT swallowed.
    HF_HUB_OFFLINE=1 python scripts/reasoning_gate.py --model "$M" --props "$P" 2>&1 \
      | { grep -vE "Loading|torch_dtype|it/s\]|Setting|generation flags|attention mask|^\s*$" || true; }
  done
done
echo "ALLDONE — copy results/frontier_gpu.log back to the workstation"
