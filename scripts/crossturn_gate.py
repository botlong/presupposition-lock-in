"""Cross-turn belief grid: the commitment as a COMPLETED assistant message.

The single-turn grid conditions on an open assistant prefix; deployed agents re-read their
actions across a turn boundary. This runs the same seven condition cells with the middle as a
closed assistant turn (actionlockin/crossturn.py), under BOTH pre-registered probe orders --
order sensitivity is a finding to report, not a knob to tune. An optional structured tool_calls
arm uses the template's native tool-call rendering where the tokenizer supports it (not
token-length-matched; structured rendering fixes its own length).
"""
from __future__ import annotations
import argparse, pathlib
import numpy as np
from actionlockin.scorer import HFScorer
from actionlockin.conditions import SYSTEM, belief_probe, build_conditions, pad_to_match
from actionlockin.crossturn import PROTOCOLS, build_messages, toolcall_message
from actionlockin.propositions import load_propositions


def ci(x):
    x = np.asarray(x, float); r = np.random.default_rng(0)
    b = [r.choice(x, len(x), replace=True).mean() for _ in range(1500)]
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--props", default="data/propositions.jsonl")
    ap.add_argument("--no-toolcall", action="store_true",
                    help="skip the structured tool_calls arm (templates without tool support)")
    ap.add_argument("--out", default=None,
                    help="per-item parquet (default results/crossturn_gate_<model>.parquet)")
    args = ap.parse_args()
    out = args.out or f"results/crossturn_gate_{args.model.split('/')[-1]}.parquet"
    s = HFScorer(args.model, device_map="auto", dtype="bfloat16", cache_dir="models/huggingface")
    props = load_propositions(args.props)[: args.n]

    toolcall_ok = not args.no_toolcall
    if toolcall_ok:
        try:  # feature-detect native tool-call rendering before spending forward passes on it
            txt = s.tok.apply_chat_template(
                [{"role": "user", "content": "x"}, toolcall_message("test claim")],
                tokenize=False)
            toolcall_ok = "test claim" in txt
        except Exception:
            toolcall_ok = False
    if not toolcall_ok:
        print("NOTE: structured tool_calls arm disabled for this model/template")

    rows = []
    for p in props:
        probe = belief_probe(p)
        cts = build_conditions(p)
        target = max(len(s.tokenize(ct.middle)) for ct in cts)   # same matching rule as the grid
        cells = [(ct.condition, ct.framing, ct.dose,
                  pad_to_match(s.tokenize, ct.middle, target, filler=" note"), True)
                 for ct in cts]
        if toolcall_ok:
            cells.append(("action", "commitment_tc", 3, toolcall_message(p.claim), False))
        for proto in PROTOCOLS:
            for cond, framing, dose, mid, matched in cells:
                lo = s.logit_diff_chat(build_messages(proto, SYSTEM, probe, mid))
                rows.append({"model": args.model, "protocol": proto, "prop_id": p.id,
                             "gold": p.gold, "condition": cond, "framing": framing,
                             "dose": dose, "length_matched": matched,
                             "logodds_P": float(lo)})

    import pandas as pd
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out)
    print(f"wrote {len(df)} per-item rows -> {out}")

    for proto in PROTOCOLS:
        d = df[df.protocol == proto]
        w = d.pivot_table(index="prop_id", columns=["condition", "framing", "dose"],
                          values="logodds_P")
        none = w[("none", "na", 0)]
        print(f"\n========== {args.model}  CROSS-TURN protocol {proto}  n={len(none)} ==========")
        for name, key in [("H1 commit3-none", ("action", "commitment", 3)),
                          ("H3 commit3-query", None),
                          ("H5 placebo-none", ("action", "placebo", 0)),
                          ("H2 commit3-assertion", None)]:
            if name == "H3 commit3-query":
                delta = w[("action", "commitment", 3)] - w[("action", "query", 0)]
            elif name == "H2 commit3-assertion":
                delta = w[("action", "commitment", 3)] - w[("assertion", "na", 0)]
            else:
                delta = w[key] - none
            e = ci(delta.to_numpy())
            print(f"  {name:22s} {e[0]:+.3f} [{e[1]:+.3f},{e[2]:+.3f}]")
        if ("action", "commitment_tc", 3) in w.columns:
            for name, other in [("TC toolcall-none", none),
                                ("TC toolcall-assertion", w[("assertion", "na", 0)])]:
                e = ci((w[("action", "commitment_tc", 3)] - other).to_numpy())
                print(f"  {name:22s} {e[0]:+.3f} [{e[1]:+.3f},{e[2]:+.3f}]")


if __name__ == "__main__":
    main()
