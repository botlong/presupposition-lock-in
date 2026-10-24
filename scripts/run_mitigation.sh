#!/usr/bin/env bash
# Mitigation experiment across local models x {factual, hard-arith}. Local GPU, cache-only.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
for M in Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-7B-Instruct; do
  for P in data/propositions.jsonl data/arith_hard.jsonl; do
    echo "######## MIT $M / $P ########"
    .venv/bin/python scripts/mitigation_gate.py --model "$M" --props "$P" \
      2>&1 | grep -vE "Loading|it/s\]|torch_dtype|Setting|generation flags|attention mask|^\s*$" || true
  done
done
echo "MIT_ALLDONE"
