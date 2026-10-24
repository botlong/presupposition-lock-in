"""Test whether H3 survives a richer non-presuppositional query control.

The terse query can differ from the commitment in in-context fluency. A richer query narrows that
gap while retaining ``whether`` semantics, making fluency a less plausible explanation for H3.
"""
from __future__ import annotations
import argparse, numpy as np
from actionlockin.scorer import HFScorer
from actionlockin.conditions import SYSTEM, belief_probe
from actionlockin.propositions import load_propositions

# non-presuppositional ("whether ... actually true") but fluent and comparable in length to dose-3 commit
def variants(c):
    return {
        "none":      "Let me consider this claim carefully before answering.",
        "commit3":   f'log_fact(fact="{c}", status="confirmed and verified")',
        "query":     f'check(query="whether {c} is true")',
        "query_rich":f'investigate(question="whether it is actually the case that {c}", thoroughness="high")',
    }


def ci(x):
    x = np.asarray(x, float); r = np.random.default_rng(0)
    b = [r.choice(x, len(x), replace=True).mean() for _ in range(1500)]
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--out", default=None,
                    help="per-item parquet (default results/fluency_matched_<model>.parquet)")
    args = ap.parse_args()
    out = args.out or f"results/fluency_matched_{args.model.split('/')[-1]}.parquet"
    s = HFScorer(args.model, device_map="auto", dtype="bfloat16", cache_dir="models/huggingface")
    props = load_propositions("data/propositions.jsonl")[: args.n]

    keys = ["none", "commit3", "query", "query_rich"]
    lo = {k: [] for k in keys}
    flu = {k: [] for k in keys}     # in-context fluency = mean middle log-prob
    rows = []                       # per-item audit trail; printed aggregates are means over these
    for p in props:
        u = belief_probe(p); v = variants(p.claim)
        for k in keys:
            ld = s.logit_diff(SYSTEM, u, v[k])
            lps = s.middle_logprobs(SYSTEM, u, v[k])
            fl = sum(lps) / len(lps) if lps else 0.0
            lo[k].append(ld); flu[k].append(fl)
            rows.append({"model": args.model, "prop_id": p.id, "gold": p.gold,
                         "condition": k, "logodds_P": float(ld), "mid_mean_logprob": float(fl)})

    import pandas as pd
    import pathlib
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out)
    print(f"wrote {len(rows)} per-item rows -> {out}")

    L = {k: np.array(lo[k]) for k in keys}
    print(f"\n========== {args.model}  n={len(props)} ==========")
    print("  fluency (mean in-context middle log-prob; closer to 0 = more fluent):")
    for k in keys:
        print(f"    {k:11s} {np.mean(flu[k]):+.3f}")
    print("  belief shift vs none (log-odds) and H3 contrasts:")
    for k in ("commit3", "query", "query_rich"):
        e = ci(L[k] - L["none"]); print(f"    {k:11s} - none = {e[0]:+.3f} [{e[1]:+.3f},{e[2]:+.3f}]")
    e_orig = ci(L["commit3"] - L["query"]); e_rich = ci(L["commit3"] - L["query_rich"])
    print(f"  H3 commit3 - query(terse) = {e_orig[0]:+.3f} [{e_orig[1]:+.3f},{e_orig[2]:+.3f}]")
    print(f"  H3 commit3 - query_RICH   = {e_rich[0]:+.3f} [{e_rich[1]:+.3f},{e_rich[2]:+.3f}]  "
          f"(fluency gap commit3-query_rich = {np.mean(flu['commit3'])-np.mean(flu['query_rich']):+.3f})")


if __name__ == "__main__":
    main()
