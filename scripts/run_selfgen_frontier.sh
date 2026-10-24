#!/usr/bin/env bash
# Run the instructed self-generation protocol on 32B/72B. A fixed item count and the shared
# implementation keep estimates comparable across model sizes.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
OUT=results/selfgen_frontier.log
: > "$OUT"

# Resolve local model paths: prefer HF cache, fall back to ModelScope (dots -> ___).
resolve() {  # $1 = short name e.g. Qwen2.5-32B-Instruct
  local hf="models/huggingface/models--Qwen--$1"
  local ms_name=${1//./___}
  local ms="models/modelscope/Qwen/$ms_name"
  if [ -d "$hf" ]; then echo "Qwen/$1"            # HF cache: pass repo id, HF_HOME finds it
  elif [ -d "$ms" ]; then echo "$ms"              # ModelScope: pass the dir path
  else echo "MISSING:$1" ; fi
}

export HF_HOME=${HF_HOME:-$PWD/models/huggingface}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

for SHORT in Qwen2.5-32B-Instruct Qwen2.5-72B-Instruct; do
  M=$(resolve "$SHORT")
  echo "===== $SHORT  ($M) =====" | tee -a "$OUT"
  if [[ "$M" == MISSING:* ]]; then
    echo "  !! model not found on box; download first" | tee -a "$OUT"; continue
  fi
  $PY scripts/selfgen_variant.py --model "$M" --n 120 2>&1 | tee -a "$OUT"
  echo | tee -a "$OUT"
done
echo "DONE -> $OUT"
