"""Run the full-set log-odds gate across locally available models.

Primary readout = log-odds(True vs False). Unlike the probability readout, log-odds has no 0/1
ceiling, so confident models (7B+) are no longer starved of usable items by mid-band calibration:
the FULL proposition set contributes on every model. Runs every locally-cached model (no downloads),
saves per-model rows, and emits per-model log-odds contrasts + verdict + answer-flip accuracy harm
(probability-based) + a cluster-bootstrap pooled summary.
"""
from __future__ import annotations
import sys, gc, traceback
from pathlib import Path
import pandas as pd

from actionlockin.propositions import load_propositions
from actionlockin.run import run_for_scorer
from actionlockin.scorer import HFScorer
from actionlockin.analyze import contrasts, evaluate_phenomenon, accuracy_harm, pooled_summary

MODELS = [
    # Restrict this entry point to cache-resident models so it never downloads weights implicitly;
    # the GPU batch script handles models that require a multi-GPU host.
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]
CACHE = "models/huggingface"
PROPS = "data/propositions.jsonl"
VALUE = "logodds_P"
OUT_PARQUET = "results/gate_multimodel.parquet"
OUT_MD = "results/gate_multimodel.md"
KEYS = ("h1_masked_commit_delta", "h2_action_specific", "h3_commit_vs_query",
        "h3_commit_vs_query_adj", "h4_dose_slope", "h4_dose_slope_adj",
        "h5_placebo", "h6_accuracy_shift")


def short(m: str) -> str:
    return m.split("/")[-1]


def main() -> None:
    Path("results").mkdir(parents=True, exist_ok=True)
    props = load_propositions(PROPS)
    frames = []
    for name in MODELS:
        print(f"\n=== {name} ===", flush=True)
        try:
            s = HFScorer(name, device_map="auto", dtype="bfloat16", cache_dir=CACHE)
            df = run_for_scorer(s, props)          # ALL props; log-odds needs no mid-band filter
            df.to_parquet(f"results/gate_{short(name)}.parquet")
            frames.append(df)
            print(f"  {df['prop_id'].nunique()} props, {len(df)} rows", flush=True)
            del s; gc.collect()
            try:
                import torch; torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception:
            print(f"  FAILED on {name}:\n{traceback.format_exc()}", flush=True)

    if not frames:
        print("no model produced rows; aborting", flush=True); sys.exit(1)

    allrows = pd.concat(frames, ignore_index=True)
    allrows.to_parquet(OUT_PARQUET)

    lines = ["# Multi-model gate (v2, LOG-ODDS readout, full proposition set)\n"]
    for m, g in allrows.groupby("model"):
        c = contrasts(g, value=VALUE); v = evaluate_phenomenon(c); h = accuracy_harm(g)
        lines.append(f"## {m}  (n={g['prop_id'].nunique()})  verdict={v['verdict']}")
        for k in KEYS:
            x = c[k]; lines.append(f"- {k}: {x['estimate']:+.3f} [{x['ci_low']:+.3f}, {x['ci_high']:+.3f}]")
        lines.append(f"- accuracy(false): none={h['acc_none']:.2f} assertion={h['acc_assertion']:.2f} "
                     f"commit3={h['acc_commit3']:.2f}  flip_rate={h['flip_rate']:.2f} (n_false={h['n_false']})")
        lines.append("")
    pool = pooled_summary(allrows, keys=KEYS, value=VALUE)
    lines.append("## POOLED (cluster bootstrap over models, log-odds)")
    for k, x in pool.items():
        pm = {short(mm): round(vv, 3) for mm, vv in x["per_model"].items()}
        lines.append(f"- {k}: {x['estimate']:+.3f} [{x['ci_low']:+.3f}, {x['ci_high']:+.3f}]  per_model={pm}")
    Path(OUT_MD).write_text("\n".join(lines))
    print("\n" + "\n".join(lines), flush=True)
    print(f"\nwrote {OUT_PARQUET} and {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
