"""Measure lock-in on reasoning claims that provide more uncertainty headroom than factual recall.

Reports log-odds contrasts, gold-split H1, and answer-flip accuracy. False items are separated
because accepting their committed premise moves the decision in the harmful direction.
"""
from __future__ import annotations
import argparse, pathlib, numpy as np
from actionlockin.scorer import HFScorer
from actionlockin.run import run_for_scorer
from actionlockin.propositions import load_propositions
from actionlockin.analyze import contrasts, belief_shift_by_gold, accuracy_harm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--props", default="data/reasoning_props.jsonl")
    ap.add_argument("--out", default=None,
                    help="parquet path for per-prop rows (default results/gate_<model>_<props>.parquet); "
                         "retains frontier 32B/72B rows so the headline numbers are reproducible")
    args = ap.parse_args()
    out = args.out or (f"results/gate_{args.model.split('/')[-1]}_"
                       f"{pathlib.Path(args.props).stem}.parquet")
    s = HFScorer(args.model, device_map="auto", dtype="bfloat16", cache_dir="models/huggingface")
    props = load_propositions(args.props)
    df = run_for_scorer(s, props)
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)  # Paired intervals and flip audits require item-level rows.
    print(f"wrote {len(df)} per-proposition rows to {out}")
    c = contrasts(df, value="logodds_P"); g = belief_shift_by_gold(df, value="logodds_P"); h = accuracy_harm(df)
    print(f"\n========== {args.model}  reasoning claims n={df['prop_id'].nunique()} ==========")
    for k in ("h1_masked_commit_delta", "h3_commit_vs_query", "h4_dose_slope", "h2_action_specific", "h5_placebo"):
        x = c[k]; print(f"  {k:24s} {x['estimate']:+.3f} [{x['ci_low']:+.3f},{x['ci_high']:+.3f}]")
    print("  gold-split H1 (commit3-none, log-odds):")
    for lbl in ("all", "true", "false"):
        x = g[lbl]; print(f"    {lbl:6s} n={x['n']:2d}  H1={x['h1']['estimate']:+.3f} [{x['h1']['ci_low']:+.3f},{x['h1']['ci_high']:+.3f}]")
    print(f"  accuracy(false claims): none={h['acc_none']:.2f} -> commit3={h['acc_commit3']:.2f} "
          f"(assertion={h['acc_assertion']:.2f})  flip_rate={h['flip_rate']:.2f}  n_false={h['n_false']}")


if __name__ == "__main__":
    main()
