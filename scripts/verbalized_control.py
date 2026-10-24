"""Instrument check for the frontier null: does VERBALIZED confidence (0-100) capture the lock-in on a
local model where the TOKEN-LEVEL log-odds H1 is known to be large (+1.0 on Qwen2.5-1.5B)?
If verbalized H1 here is ~0, verbalized confidence is insensitive and the frontier (GPT-5.x) verbalized
null is uninformative; if positive, the instrument works and the frontier null is meaningful.
"""
from __future__ import annotations
import argparse, re, numpy as np
from actionlockin.scorer import HFScorer
from actionlockin.conditions import SYSTEM, belief_probe, build_conditions
from actionlockin.propositions import load_propositions

CONF_Q = "On a scale of 0 to 100, how confident are you that the claim is TRUE? Reply with only a number."


def gen_number(s, messages, max_new=8):
    torch = s._torch
    ids = s._render_chat(messages).to(s.model.device)
    with torch.no_grad():
        out = s.model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=s.tok.eos_token_id)
    txt = s.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    m = re.search(r"\d{1,3}", txt)
    return max(0, min(100, int(m.group()))) if m else None


def ci(x):
    x = np.asarray([v for v in x if v is not None], float)
    if len(x) < 2: return (float("nan"),) * 3
    r = np.random.default_rng(0); b = [r.choice(x, len(x), replace=True).mean() for _ in range(1500)]
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--out", default=None,
                    help="per-item parquet (default results/verbalized_<model>.parquet)")
    args = ap.parse_args()
    out = args.out or f"results/verbalized_{args.model.split('/')[-1]}.parquet"
    s = HFScorer(args.model, device_map="auto", dtype="bfloat16", cache_dir="models/huggingface")
    props = load_propositions("data/propositions.jsonl")[: args.n]
    conf = {"none": [], "commit": [], "query": []}
    lo = {"none": [], "commit": [], "query": []}      # token log-odds for the same items, for comparison
    keymap = {"none": ("none", "na", 0), "commit": ("action", "commitment", 3), "query": ("action", "query", 0)}
    rows = []  # per-item audit trail; printed aggregates are means over these
    for p in props:
        u = belief_probe(p)
        mids = {k: ct.middle for k, ct in {(x.condition, x.framing, x.dose): x for x in build_conditions(p)}.items()}
        for cond, key in keymap.items():
            c = gen_number(s, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": u},
                               {"role": "assistant", "content": mids[key]},
                               {"role": "user", "content": CONF_Q}])
            ld = s.logit_diff(SYSTEM, u, mids[key])
            conf[cond].append(c); lo[cond].append(ld)
            rows.append({"model": args.model, "prop_id": p.id, "gold": p.gold, "condition": cond,
                         "verbalized_conf": None if c is None else float(c), "logodds_P": float(ld)})

    import pandas as pd
    import pathlib
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out)
    print(f"wrote {len(rows)} per-item rows -> {out}")

    cN, cC, cQ = ci(conf["none"]), ci(conf["commit"]), ci(conf["query"])
    print(f"\n========== {args.model}  n={len(props)} ==========")
    print(f"  verbalized confidence(0-100): none={cN[0]:.1f}  commit={cC[0]:.1f}  query={cQ[0]:.1f}")
    cd = np.array([a - b for a, b in zip(conf["commit"], conf["none"]) if a is not None and b is not None], float)
    e = ci(cd); print(f"  VERBALIZED H1 (commit-none) = {e[0]:+.2f} [{e[1]:+.2f},{e[2]:+.2f}]  (n={len(cd)})")
    ld = np.array(lo["commit"]) - np.array(lo["none"])
    el = ci(ld); print(f"  TOKEN log-odds H1 (same items) = {el[0]:+.2f} [{el[1]:+.2f},{el[2]:+.2f}]")
    print("  -> if verbalized H1 ~ 0 while token H1 >> 0, verbalized confidence is INSENSITIVE")
    print("     and the frontier (GPT-5.x) verbalized null is uninformative.")


if __name__ == "__main__":
    main()
