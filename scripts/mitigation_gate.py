#!/usr/bin/env python
"""Mitigation experiment: can we keep the self-issued tool call on record but stop it from
locking in belief? Runs none + dose-3 commitment (baseline) + three deployable mitigations
(disclaim / reframe / isolate) and reports the remaining belief lock-in (log-odds delta vs none)
and false-claim accuracy for each.

Usage:  python scripts/mitigation_gate.py --model Qwen/Qwen2.5-7B-Instruct --props data/propositions.jsonl
"""
import argparse
import pathlib
from actionlockin.scorer import HFScorer
from actionlockin.propositions import load_propositions
from actionlockin.run import run_for_scorer
from actionlockin.conditions import build_mitigation_conditions
from actionlockin.analyze import mitigation_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--props", default="data/propositions.jsonl")
    ap.add_argument("--out", default=None,
                    help="parquet path for per-prop rows (default results/mitigation_<model>_<props>.parquet)")
    args = ap.parse_args()
    out = args.out or (f"results/mitigation_{args.model.split('/')[-1]}_"
                       f"{pathlib.Path(args.props).stem}.parquet")
    s = HFScorer(args.model, device_map="auto", dtype="bfloat16", cache_dir="models/huggingface")
    props = load_propositions(args.props)
    df = run_for_scorer(s, props, builder=build_mitigation_conditions)
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)   # always saved so per-proposition rows are reproducible
    print(f"wrote {len(df)} per-proposition rows to {out}")
    r = mitigation_report(df, value="logodds_P")
    print(f"\n===== MITIGATION  {args.model}  n={df['prop_id'].nunique()}  ({args.props}) =====")
    print(f"  baseline none acc(false claims) = {r['_none_acc_false']:.2f}   (n_false={r['_n_false']})")
    print(f"  {'condition':14s} {'belief Δ vs none (log-odds)':30s} acc(false)  flip")
    for lbl in ("commit3", "mit_disclaim", "mit_reframe", "mit_isolate"):
        d = r[lbl]; e = d["delta_vs_none"]
        tag = "  <-- baseline (harmful)" if lbl == "commit3" else ""
        print(f"  {lbl:14s} {e['estimate']:+.3f} [{e['ci_low']:+.3f},{e['ci_high']:+.3f}]"
              f"{'':8s} {d['acc_false']:.2f}       {d['flip_rate']:.2f}{tag}")


if __name__ == "__main__":
    main()
