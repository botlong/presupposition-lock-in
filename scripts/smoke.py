"""Run the complete scoring pipeline on one small local model.

Loads ONE small model from the local HF cache (cache_dir), runs the full pipeline
on a slice of propositions, and writes results/smoke.parquet. A real model is necessary here
because stub scoring cannot validate chat-template rendering, token IDs, or GPU execution.
"""
from __future__ import annotations
import argparse
from pathlib import Path

from actionlockin.propositions import load_propositions
from actionlockin.calibration import filter_midband
from actionlockin.run import run_for_scorer
from actionlockin.scorer import HFScorer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--cache-dir", default="models/huggingface")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--low", type=float, default=0.2)
    ap.add_argument("--high", type=float, default=0.8)
    ap.add_argument("--props", default="data/propositions.jsonl")
    ap.add_argument("--out", default="results/smoke.parquet")
    args = ap.parse_args()

    Path("results").mkdir(parents=True, exist_ok=True)
    print(f"loading {args.model} (cache_dir={args.cache_dir}, device_map={args.device_map}) ...")
    s = HFScorer(args.model, device_map=args.device_map, cache_dir=args.cache_dir)

    props = load_propositions(args.props)[: args.n]
    kept = filter_midband(s, props, args.low, args.high)
    print(f"calibration: {len(kept)}/{len(props)} props kept in baseline P(True) in [{args.low},{args.high}]")
    if not kept:
        raise SystemExit("no propositions survived mid-band calibration; widen band or check probe")

    df = run_for_scorer(s, kept)
    df.to_parquet(args.out)
    print(f"wrote {len(df)} rows to {args.out}")
    print("\nper-cell mean p_of_P:")
    print(df.groupby(["condition", "framing", "dose"])["p_of_P"].mean().to_string())


if __name__ == "__main__":
    main()
