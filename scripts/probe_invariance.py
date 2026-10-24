"""Probe-wording invariance check: do H1 (commit3-none) and H3 (commit3-query) stay positive
when the belief-probe wording changes? The paper claims invariance across three wordings on the
two small Qwen models; this script is the committed, reproducible form of that check.

The answer cue ("Answer:") and the " True"/" False" readout tokens are fixed by the scorer;
only the user-turn probe wording varies. Middles (none / query / commit3) are the exact
condition texts of the main experiment (actionlockin.conditions).

Usage:  python scripts/probe_invariance.py --model Qwen/Qwen2.5-1.5B-Instruct \
            --out results/probe_invariance_Qwen2.5-1.5B.parquet
"""
from __future__ import annotations
import argparse
import pathlib

import numpy as np

from actionlockin.conditions import SYSTEM, build_conditions, pad_to_match
from actionlockin.propositions import load_propositions
from actionlockin.scorer import HFScorer

WORDINGS = {
    "w1_standard": ('Question: Is the following claim true? Claim: "{c}"\n'
                    "Answer with a single word, True or False."),
    "w2_statement": ('Consider this statement: "{c}"\n'
                     "Is it true or false? Reply with exactly one word: True or False."),
    "w3_determine": ('Claim: "{c}"\n'
                     "Determine whether the claim is true. Respond with one word (True or False)."),
}
CELLS = {"none": ("none", "na", 0), "query": ("action", "query", 0),
         "commit3": ("action", "commitment", 3)}


def ci(x, n_boot: int = 2000, seed: int = 0):
    x = np.asarray(x, float)
    r = np.random.default_rng(seed)
    b = [r.choice(x, len(x), replace=True).mean() for _ in range(n_boot)]
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--props", default="data/propositions.jsonl")
    ap.add_argument("--out", default=None,
                    help="parquet path for per-proposition rows (default results/probe_invariance_<model>.parquet)")
    args = ap.parse_args()
    out = args.out or f"results/probe_invariance_{args.model.split('/')[-1]}.parquet"

    s = HFScorer(args.model, device_map="auto", dtype="bfloat16", cache_dir="models/huggingface")
    props = load_propositions(args.props)

    rows = []
    for p in props:
        mids = {name: ct.middle for name, key in CELLS.items()
                for ct in build_conditions(p)
                if (ct.condition, ct.framing, ct.dose) == key}
        # same length control as the main experiment (run_for_scorer): pad every cell
        # to the per-proposition max token count with the single-token filler
        target = max(len(s.tokenize(m)) for m in mids.values())
        mids = {name: pad_to_match(s.tokenize, m, target, filler=" note")
                for name, m in mids.items()}
        for wname, tmpl in WORDINGS.items():
            user = tmpl.format(c=p.claim)
            for cell, mid in mids.items():
                rows.append({"model": args.model, "prop_id": p.id, "gold": p.gold,
                             "wording": wname, "cell": cell,
                             "logodds_P": s.logit_diff(SYSTEM, user, mid)})

    import pandas as pd
    df = pd.DataFrame(rows)
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"wrote {len(df)} rows -> {out}")

    print(f"\n===== PROBE-WORDING INVARIANCE  {args.model}  n={df['prop_id'].nunique()} =====")
    piv = df.pivot_table(index=["wording", "prop_id"], columns="cell", values="logodds_P")
    for wname in WORDINGS:
        w = piv.loc[wname]
        h1 = ci((w["commit3"] - w["none"]).to_numpy())
        h3 = ci((w["commit3"] - w["query"]).to_numpy())
        print(f"  {wname:13s} H1={h1[0]:+.3f} [{h1[1]:+.3f},{h1[2]:+.3f}]   "
              f"H3={h3[0]:+.3f} [{h3[1]:+.3f},{h3[2]:+.3f}]")


if __name__ == "__main__":
    main()
