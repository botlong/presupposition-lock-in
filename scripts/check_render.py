"""On-box pre-flight gate: verify the cross-turn episode renders correctly for a model's
tokenizer BEFORE any weights load or GPU time is spent on it. Tokenizer-only, so it costs
seconds; a failure here means the template would silently produce a different experiment
than the one the paper describes, which must abort the batch."""
from __future__ import annotations
import argparse
from actionlockin.crossturn import verify_rendering


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model id or local path")
    ap.add_argument("--cache-dir", default="models/huggingface")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    print(f"{args.model}: {verify_rendering(tok)}")


if __name__ == "__main__":
    main()
