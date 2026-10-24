"""Llama-3.1-8B runs on the Llama-matched arithmetic set (data/arith_hard_llama.jsonl).

The main Llama runs use the canonical batch unchanged:
    python scripts/run_gpu_all.py --model <Llama-3.1-8B-Instruct> --tag Llama-3.1-8B
This wrapper re-runs the channel, gate and mitigation grids of that batch with the hard-arithmetic set
swapped for the Llama-matched set (made by scripts/gen_arith_llama.py), then renames the channel and
mitigation outputs to the committed *_arith_hard_llama names:

    python scripts/run_llama_matched.py <Llama-3.1-8B-Instruct dir>
"""
import importlib.util
import pathlib
import sys

spec = importlib.util.spec_from_file_location("rga", "scripts/run_gpu_all.py")
rga = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rga)
from actionlockin.scorer import HFScorer

SET = "data/arith_hard_llama.jsonl"
rga.GATE_SETS = (SET,)
src_load = rga.load_propositions
rga.load_propositions = lambda p: src_load(SET if p == "data/arith_hard.jsonl" else p)  # channel/mitigation -> new set
s = HFScorer(sys.argv[1], device_map="auto", dtype="bfloat16", cache_dir=None)
R = "results_llama_hard"; tag = "Llama-3.1-8B"
rga.run_channel(s, tag, R); rga.run_gate(s, tag, R); rga.run_mitigation(s, tag, R)
for kind in ("channel", "mitigation"):
    src = pathlib.Path(R) / f"{kind}_{tag}_arith_hard.parquet"
    src.rename(src.with_name(f"{kind}_{tag}_arith_hard_llama.parquet"))
print("DONE")
