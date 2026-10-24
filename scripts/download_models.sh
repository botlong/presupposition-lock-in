#!/usr/bin/env bash
# Download Qwen2.5 weights with NO GPU needed -> run this in AutoDL 无卡模式 (cheap CPU mode).
# Cost-optimal frontier workflow:
#   1. boot the box in 无卡模式 (no GPU billing) and run THIS to pull weights to the data disk;
#   2. switch to GPU mode and run scripts/run_on_gpu.sh (weights cached -> minimal GPU time).
# Resumes interrupted downloads automatically (huggingface_hub keeps .incomplete blobs).
# Cache dir MUST equal reasoning_gate.py's cache_dir="models/huggingface" so the gate finds them.
# Usage (on the box, from repo root):  bash scripts/download_models.sh [MODEL_ID ...]
set -u
cd "$(dirname "$0")/.."
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"   # mainland mirror; direct HF is blocked here
export TOKENIZERS_PARALLELISM=false
CACHE="$PWD/models/huggingface"                              # == gate's cache_dir (relative to repo root)
HFCLI="$(command -v hf || command -v huggingface-cli || echo /root/miniconda3/bin/hf)"
if [ "$#" -gt 0 ]; then MODELS=("$@"); else MODELS=("Qwen/Qwen2.5-32B-Instruct" "Qwen/Qwen2.5-72B-Instruct"); fi
mkdir -p results "$CACHE"
echo "mirror=$HF_ENDPOINT  cache=$CACHE  cli=$HFCLI  models=${MODELS[*]}"
df -h "$CACHE" | tail -1
for M in "${MODELS[@]}"; do
  echo "######## DL START $M $(date -u +%H:%M:%S) ########"
  n=0
  # hf-mirror occasionally drops a connection mid-shard; retry resumes from .incomplete.
  until "$HFCLI" download "$M" --cache-dir "$CACHE"; do
    n=$((n+1)); echo ">> retry $n for $M $(date -u +%H:%M:%S)"; [ "$n" -ge 30 ] && { echo ">> GAVE UP on $M"; break; }; sleep 10
  done
  echo "######## DL DONE $M $(date -u +%H:%M:%S)  cache=$(du -sh "$CACHE" 2>/dev/null|cut -f1) ########"
done
echo "DLDONE-ALL $(date -u +%H:%M:%S)"
