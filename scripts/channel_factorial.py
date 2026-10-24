#!/usr/bin/env python
"""Measure whether named source attribution changes the factive-content effect.

Is the pre-result lock-in specific to the agent's OWN presupposition, or does the SAME factive claim
move belief just as much when attributed to the user or system? Every source arm uses one
length-matched assistant readback at the same position and varies only the source label on an
identical factive predicate. Per-proposition rows are retained because paired bootstrap intervals
cannot be reconstructed from aggregate summaries.

Usage:
  python scripts/channel_factorial.py --model Qwen/Qwen2.5-7B-Instruct \
      --props data/arith_hard.jsonl --out results/channel_Qwen2.5-7B_arith_hard.parquet
"""
import argparse
import pathlib
from actionlockin.scorer import HFScorer
from actionlockin.propositions import load_propositions
from actionlockin.run import run_for_scorer
from actionlockin.conditions import build_channel_conditions
from actionlockin.analyze import channel_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--props", default="data/arith_hard.jsonl")
    ap.add_argument("--out", default=None,
                    help="parquet path for per-prop rows (default results/channel_<model>_<props>.parquet)")
    args = ap.parse_args()

    out = args.out or (f"results/channel_{args.model.split('/')[-1]}_"
                       f"{pathlib.Path(args.props).stem}.parquet")

    s = HFScorer(args.model, device_map="auto", dtype="bfloat16", cache_dir="models/huggingface")
    props = load_propositions(args.props)
    df = run_for_scorer(s, props, builder=build_channel_conditions, padding_side="left")

    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)  # Paired source contrasts require item-level rows.
    print(f"wrote {len(df)} per-proposition rows to {out}")

    r = channel_report(df, value="logodds_P")
    print(f"\n===== SOURCE-ATTRIBUTION FACTORIAL  {args.model}  n={df['prop_id'].nunique()}  ({args.props}) =====")
    print(f"  baseline none acc(false claims) = {r['_none_acc_false']:.2f}   (n_false={r['_n_false']})")
    print(f"  {'arm':12s} {'source':8s} {'belief Δ vs none (log-odds)':32s} acc(false)  flip")
    for lbl, src in [("commit", "self"), ("source_self", "self"), ("query", "self"),
                     ("isolate", "self"), ("source_user", "user"), ("source_system", "system")]:
        d = r[lbl]; e = d["delta_vs_none"]
        print(f"  {lbl:12s} {src:8s} {e['estimate']:+.3f} [{e['ci_low']:+.3f},{e['ci_high']:+.3f}]"
              f"{'':8s} {d['acc_false']:.2f}       {d['flip_rate']:.2f}")
    print("\n  CHANNEL TEST (per-prop paired, >0 => self-source privilege):")
    for k, v in r["_contrasts"].items():
        print(f"    {k:30s} {v['estimate']:+.3f} [{v['ci_low']:+.3f},{v['ci_high']:+.3f}]  (n={v['n']})")


if __name__ == "__main__":
    main()
