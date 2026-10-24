"""Cross-turn mitigation grid: the mitigated representations as COMPLETED assistant messages.

The mitigation contrast conditions on an open single-turn prefix; deployed agents re-read
their actions across a turn boundary, where the lock-in is strongest. This runs the
mitigation cells (none, dose-3 commitment, disclaim/reframe/isolate) with the middle as a
closed assistant turn under BOTH pre-registered probe orders — order sensitivity is a
finding to report, not a knob to tune. Length matching follows the mitigation grid's rule
(per-proposition target over the mitigation builder's own cells).
"""
from __future__ import annotations
import argparse
import pathlib
import numpy as np
from actionlockin.scorer import HFScorer
from actionlockin.conditions import SYSTEM, belief_probe, build_mitigation_conditions, pad_to_match
from actionlockin.crossturn import PROTOCOLS, build_messages
from actionlockin.propositions import load_propositions

LABELS = [("commit3", ("action", "commitment", 3)),
          ("mit_disclaim", ("action", "mit_disclaim", 3)),
          ("mit_reframe", ("action", "mit_reframe", 3)),
          ("mit_isolate", ("action", "mit_isolate", 3))]


def ci(x):
    x = np.asarray(x, float); r = np.random.default_rng(0)
    b = [r.choice(x, len(x), replace=True).mean() for _ in range(1500)]
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--props", default="data/propositions.jsonl")
    ap.add_argument("--out", default=None,
                    help="per-item parquet (default results/crossturn_mit_<model>_<props>.parquet)")
    args = ap.parse_args()
    stem = pathlib.Path(args.props).stem
    out = args.out or f"results/crossturn_mit_{args.model.split('/')[-1]}_{stem}.parquet"
    s = HFScorer(args.model, device_map="auto", dtype="bfloat16", cache_dir="models/huggingface")
    props = load_propositions(args.props)[: args.n]

    rows = []
    for p in props:
        probe = belief_probe(p)
        cts = build_mitigation_conditions(p)
        target = max(len(s.tokenize(ct.middle)) for ct in cts)
        for proto in PROTOCOLS:
            for ct in cts:
                mid = pad_to_match(s.tokenize, ct.middle, target, filler=" note")
                lo = s.logit_diff_chat(build_messages(proto, SYSTEM, probe, mid))
                rows.append({"model": args.model, "protocol": proto, "prop_id": p.id,
                             "gold": p.gold, "condition": ct.condition, "framing": ct.framing,
                             "dose": ct.dose, "logodds_P": float(lo)})

    import pandas as pd
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out)
    print(f"wrote {len(df)} per-item rows -> {out}")

    r = summarize(df)
    for proto in PROTOCOLS:
        s_ = r[proto]
        print(f"\n========== {args.model}  CROSS-TURN MITIGATION protocol {proto}  "
              f"{stem}  n={s_['n']} ==========")
        print(f"  false claims n={s_['n_false']}; base-correct under none n={s_['n_base']}")
        print(f"  {'cell':13s} {'belief Δ vs none (log-odds)':30s} acc(false)  flip(base-correct)")
        for name, _ in LABELS:
            c = s_["cells"][name]; e = c["delta_ci"]; fl = c["flip_ci"]
            print(f"  {name:13s} {e[0]:+.3f} [{e[1]:+.3f},{e[2]:+.3f}]{'':8s} "
                  f"{c['acc_false']:.2f}       {fl[0]:.2f} [{fl[1]:.2f},{fl[2]:.2f}]")


def summarize(df) -> dict:
    """Aggregate a cross-turn mitigation frame per protocol: paired belief delta vs none for
    the commitment and each mitigation, correct-rejection accuracy on false claims, and the
    flip rate over the base-correct subset (none log-odds < 0 on a false claim)."""
    out = {}
    for proto in PROTOCOLS:
        d = df[df.protocol == proto]
        w = d.pivot_table(index="prop_id", columns=["condition", "framing", "dose"],
                          values="logodds_P")
        none = w[("none", "na", 0)]
        gold = d.groupby("prop_id").gold.first().reindex(none.index)
        false_idx = none.index[~gold]
        base_correct = false_idx[(none.loc[false_idx] < 0)]
        cells = {}
        for name, key in LABELS:
            col = w[key]
            cells[name] = {
                "delta_ci": ci((col - none).to_numpy()),
                "acc_false": (float((col.loc[false_idx] < 0).mean())
                              if len(false_idx) else float("nan")),
                "flip_ci": (ci((col.loc[base_correct] >= 0).astype(float).to_numpy())
                            if len(base_correct) else (float("nan"),) * 3)}
        out[proto] = {"n": int(len(none)), "n_false": int(len(false_idx)),
                      "n_base": int(len(base_correct)), "cells": cells}
    return out


if __name__ == "__main__":
    main()
