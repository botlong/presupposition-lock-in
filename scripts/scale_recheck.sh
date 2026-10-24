#!/usr/bin/env bash
# Route smaller Qwen models through the same gate entry point as the large-model batch so scale
# comparisons do not inherit script-level differences.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
for M in Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-7B-Instruct; do
  echo "######## $M / data/propositions.jsonl ########"
  .venv/bin/python scripts/reasoning_gate.py --model "$M" --props data/propositions.jsonl \
    2>&1 | grep -vE "Loading|it/s\]|torch_dtype|Setting|generation flags|attention mask|^\s*$" || true
done
echo "RECHECK_DONE"
