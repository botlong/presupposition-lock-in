#!/usr/bin/env python
"""One-load-per-model GPU batch.

Loads a single HFScorer ONCE and runs every experiment for that model
(channel source-attribution + gate on 3 proposition sets + mitigation), saving a
per-proposition parquet for each. Item-level rows are required for paired bootstrap analyses,
and loading a large model once avoids repeating the dominant memory and startup cost.

Usage (on the GPU box, from repo root):
  python scripts/run_gpu_all.py --model Qwen/Qwen2.5-7B-Instruct --tag Qwen2.5-7B
  python scripts/run_gpu_all.py --model <local 72B path> --tag Qwen2.5-72B
  python scripts/run_gpu_all.py --stub --tag smoke   # no GPU; validates all report code paths
"""
from __future__ import annotations
import argparse, pathlib
from actionlockin.propositions import load_propositions
from actionlockin.run import run_for_scorer
from actionlockin.conditions import (build_conditions, build_channel_conditions,
                                     build_mitigation_conditions)
from actionlockin.analyze import (channel_report, mitigation_report,
                                  contrasts, belief_shift_by_gold, accuracy_harm)

GATE_SETS = ("data/propositions.jsonl", "data/arith_props.jsonl", "data/arith_hard.jsonl")


def _save(df, path):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    print(f"  wrote {len(df)} rows -> {path}", flush=True)


def _ci(x):
    return f"{x['estimate']:+.3f} [{x['ci_low']:+.3f},{x['ci_high']:+.3f}]"


def run_channel(s, tag, R):
    props = load_propositions("data/arith_hard.jsonl")
    df = run_for_scorer(s, props, builder=build_channel_conditions, padding_side="left")
    _save(df, f"{R}/channel_{tag}_arith_hard.parquet")
    r = channel_report(df, value="logodds_P")
    print(f"[CHANNEL {tag}] none_acc_false={r['_none_acc_false']:.2f} n_false={r['_n_false']}", flush=True)
    for k in ("commit", "source_self", "query", "isolate", "source_user", "source_system"):
        d = r[k]; print(f"   {k:14s} d={_ci(d['delta_vs_none'])} acc={d['acc_false']:.2f} flip={d['flip_rate']:.2f}")
    for key, v in r["_contrasts"].items():
        print(f"   contrast {key:34s} {_ci(v)} n={v['n']}")


def run_gate(s, tag, R):
    for ps in GATE_SETS:
        props = load_propositions(ps)
        df = run_for_scorer(s, props)  # default builder=build_conditions, right padding
        stem = pathlib.Path(ps).stem
        _save(df, f"{R}/gate_{tag}_{stem}.parquet")
        c = contrasts(df, value="logodds_P")
        g = belief_shift_by_gold(df, value="logodds_P")
        h = accuracy_harm(df)
        print(f"[GATE {tag} {stem}] n={df['prop_id'].nunique()}", flush=True)
        for k in ("h1_masked_commit_delta", "h3_commit_vs_query", "h4_dose_slope",
                  "h2_action_specific", "h5_placebo"):
            print(f"   {k:24s} {_ci(c[k])}")
        for lbl in ("all", "true", "false"):
            x = g[lbl]; print(f"   gold {lbl:6s} n={x['n']:2d} H1={_ci(x['h1'])}")
        print(f"   acc(false): none={h['acc_none']:.2f}->commit3={h['acc_commit3']:.2f} "
              f"(assert={h['acc_assertion']:.2f}) flip={h['flip_rate']:.2f} n_false={h['n_false']}")


def run_mitigation(s, tag, R):
    props = load_propositions("data/arith_hard.jsonl")
    df = run_for_scorer(s, props, builder=build_mitigation_conditions)
    _save(df, f"{R}/mitigation_{tag}_arith_hard.parquet")
    r = mitigation_report(df, value="logodds_P")
    print(f"[MITIGATION {tag}] none_acc_false={r['_none_acc_false']:.2f} n_false={r['_n_false']}", flush=True)
    for lbl in ("commit3", "mit_disclaim", "mit_reframe", "mit_isolate"):
        d = r[lbl]; print(f"   {lbl:14s} d={_ci(d['delta_vs_none'])} acc={d['acc_false']:.2f} flip={d['flip_rate']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--tag", required=True, help="short label for output filenames, e.g. Qwen2.5-72B")
    ap.add_argument("--resultsdir", default="results")
    ap.add_argument("--stub", action="store_true", help="use StubScorer (no GPU) to validate code paths")
    args = ap.parse_args()

    if args.stub:
        from actionlockin.scorer import StubScorer
        s = StubScorer("stub", lambda flat: 0.85 if "confirmed and verified" in flat
                       or "commit" in flat.lower() else 0.2)
    else:
        from actionlockin.scorer import HFScorer
        s = HFScorer(args.model, device_map="auto", dtype="bfloat16", cache_dir="models/huggingface")
    print(f"\n##### MODEL LOADED: {args.tag} ({'STUB' if args.stub else args.model}) #####", flush=True)

    run_channel(s, args.tag, args.resultsdir)
    run_gate(s, args.tag, args.resultsdir)
    run_mitigation(s, args.tag, args.resultsdir)
    print(f"##### DONE {args.tag} #####", flush=True)


if __name__ == "__main__":
    main()
