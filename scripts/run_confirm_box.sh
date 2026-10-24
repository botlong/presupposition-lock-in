#!/usr/bin/env bash
# Confirmation batch for a multi-GPU box: sampled-decoding readout validation,
# cross-turn mitigation grid, and the corrective-evidence arm on hard arithmetic.
# Same gate discipline as run_crossturn_box.sh — nothing large loads until its gates pass:
#   gate 1  CUDA matmul smoke;
#   gate 2  per model: sha256 snapshot (incl. weight shards) BEFORE running;
#   gate 3  closed-turn render gate, only for models running chat-episode experiments;
#   gate 4  per model: 3-item sampled pilot, whose forced log-odds must reproduce the
#           committed gate parquet rows — same weights on the same hardware reproduce
#           bf16-exactly, so a large deviation means wrong weights, not noise;
#   then per experiment: full run -> row-count assert scoped to THIS run's outputs -> tar.
# Per-experiment model lists are overridable via env:
#   SAMPLED_MODELS  sampled_readout on arith_hard        (default 72B 7B)
#   CTMIT_MODELS    crossturn_mitigation on both sets    (default 32B 72B)
#   RECOV_MODELS    downstream_harm on arith_hard        (default 72B)
# Usage (on the box, from repo root):
#   bash scripts/run_confirm_box.sh 2>&1 | tee results/confirm_box.log
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
D=/root/autodl-tmp
SAMPLED_MODELS=${SAMPLED_MODELS:-"$D/Qwen2.5-72B-Instruct $D/Qwen2.5-7B-Instruct"}
CTMIT_MODELS=${CTMIT_MODELS:-"$D/Qwen2.5-32B-Instruct $D/Qwen2.5-72B-Instruct"}
RECOV_MODELS=${RECOV_MODELS:-"$D/Qwen2.5-72B-Instruct"}

python -c "import torch" 2>/dev/null || python -m pip install -q torch
python -c "import transformers, accelerate, numpy, pandas, pyarrow, scipy, yaml" 2>/dev/null \
  || python -m pip install -q -U transformers accelerate numpy pandas pyarrow scipy pyyaml || true
if ! python -c "import torch; (torch.randn(8,8,device='cuda')@torch.randn(8,8,device='cuda')).sum().item()" 2>/dev/null; then
  echo ">> GPU matmul failed (likely sm/CUDA mismatch) -> installing torch cu128"
  python -m pip install -q --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
fi
python -c "import torch; (torch.randn(8,8,device='cuda')@torch.randn(8,8,device='cuda')).sum().item(); print('CUDA smoke ok')"

tag() { basename "$1" | sed 's/___/./g; s/-Instruct$//'; }
ALL=$(printf '%s\n' $SAMPLED_MODELS $CTMIT_MODELS $RECOV_MODELS | awk '!seen[$0]++')
CHAT=$(printf '%s\n' $CTMIT_MODELS $RECOV_MODELS | awk '!seen[$0]++')

for M in $ALL; do
  echo "──────────── snapshot: $M ────────────"
  python scripts/record_snapshot.py --model "$M"
done
for M in $CHAT; do
  echo "──────────── render gate: $M ────────────"
  python scripts/check_render.py --model "$M"
done
for M in $ALL; do
  T=$(tag "$M")
  echo "──────────── weight spot-check: $M (tag: $T) ────────────"
  python scripts/sampled_readout.py --model "$M" --n 3 --out "results/_pilot_sampledread_$T.parquet"
  python - "$T" <<'EOF'
import sys
import pandas as pd
tag = sys.argv[1]
ref_path = f"results/gate_{tag}_arith_hard.parquet"
pilot = pd.read_parquet(f"results/_pilot_sampledread_{tag}.parquet")
try:
    ref = pd.read_parquet(ref_path)
except FileNotFoundError:
    print(f"spot-check skipped: no committed reference {ref_path}")
    sys.exit(0)
got = pilot.groupby(["prop_id", "cell"]).logodds_forced.first()
key = {"none": ("none", "na", 0), "commit3": ("action", "commitment", 3)}
diffs = {}
for (pid, cell), lo in got.items():
    cond, fr, dose = key[cell]
    want = ref[(ref.prop_id == pid) & (ref.condition == cond)
               & (ref.framing == fr) & (ref.dose == dose)].logodds_P
    if len(want) == 1:
        diffs[(pid, cell)] = abs(float(lo) - float(want.iloc[0]))
assert diffs, "spot-check matched no reference rows"
worst = max(diffs.values())
print(f"spot-check vs {ref_path}: {len(diffs)} cells, max |dlogodds| = {worst:.3f}")
assert worst <= 1.0, (
    f"forced log-odds deviate from the committed gate rows by {worst:.3f} — "
    "wrong or corrupted weights (identical weights on identical hardware reproduce bf16-exactly)")
EOF
  rm -f "results/_pilot_sampledread_$T.parquet"
done

PRODUCED=()   # "parquet expected_rows" pairs for THIS invocation only
LOGS=()

for M in $SAMPLED_MODELS; do
  T=$(tag "$M")
  echo "════════════════ sampled_readout  $M  (tag: $T) ════════════════"
  python scripts/sampled_readout.py --model "$M" \
    --out "results/sampledread_${T}_arith_hard.parquet" \
    2>&1 | tee "results/sampledread_${T}_box.log"
  PRODUCED+=("results/sampledread_${T}_arith_hard.parquet 4200")
  LOGS+=("results/sampledread_${T}_box.log")
done

for M in $CTMIT_MODELS; do
  T=$(tag "$M")
  echo "════════════════ crossturn_mitigation  $M  (tag: $T) ════════════════"
  python scripts/crossturn_mitigation.py --model "$M" --n 3 --out "results/_pilot_ctmit_$T.parquet"
  rm -f "results/_pilot_ctmit_$T.parquet"
  for P in data/propositions.jsonl data/arith_hard.jsonl; do
    S=$(basename "$P" .jsonl)
    [ "$S" = propositions ] && WANT=1200 || WANT=1000
    python scripts/crossturn_mitigation.py --model "$M" --props "$P" \
      --out "results/crossturn_mit_${T}_${S}.parquet" \
      2>&1 | tee "results/crossturn_mit_${T}_${S}_box.log"
    PRODUCED+=("results/crossturn_mit_${T}_${S}.parquet $WANT")
    LOGS+=("results/crossturn_mit_${T}_${S}_box.log")
  done
done

for M in $RECOV_MODELS; do
  T=$(tag "$M")
  echo "════════════════ downstream_harm arith_hard  $M  (tag: $T) ════════════════"
  python scripts/downstream_harm.py --model "$M" --props data/arith_hard.jsonl --n 3 \
    --out "results/_pilot_recov_$T.parquet"
  rm -f "results/_pilot_recov_$T.parquet"
  python scripts/downstream_harm.py --model "$M" --props data/arith_hard.jsonl \
    --out "results/downstream_harm_${T}_arith_hard.parquet" \
    2>&1 | tee "results/downstream_harm_${T}_arith_hard_box.log"
  PRODUCED+=("results/downstream_harm_${T}_arith_hard.parquet 1200")
  LOGS+=("results/downstream_harm_${T}_arith_hard_box.log")
done

printf '%s\n' "${PRODUCED[@]}" | python - <<'EOF'
import sys
import pandas as pd
for line in sys.stdin:
    f, want = line.split()
    n = len(pd.read_parquet(f))
    assert n == int(want), f"{f}: rows {n} != {want}"
    print(f"row count ok: {f} = {n}")
EOF

tar czf "results/confirm_box_$(date +%F).tar.gz" \
  results/model_snapshots.jsonl \
  $(printf '%s\n' "${PRODUCED[@]}" | cut -d' ' -f1) "${LOGS[@]}"
echo "ALL CONFIRM RUNS DONE -> results/confirm_box_$(date +%F).tar.gz"
