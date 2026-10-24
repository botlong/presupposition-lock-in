"""Downstream-harm experiment: does a pre-result COMMITMENT action make an agent resist
later CORRECTIVE evidence and answer worse? (the 'agent won't change its mind' failure mode)

Multi-step: [system] [user: decide if claim true] [assistant: STEP-1 manipulation]
[user: Observation = an authoritative fact-check pointing to the TRUTH] -> read True/False.

Conditions for the agent's own step-1 turn (all the same texts as the core experiment):
  none | query (non-presup action) | commit (presup action log_fact P confirmed) | assertion (declarative).
Correct answer = gold. Harm = on items where the committed content conflicts with truth (esp. FALSE
claims), commitment lowers post-evidence accuracy vs none, despite identical corrective evidence.
We also run a NO-evidence arm to expose the interaction (evidence moves 'none' but is blunted by commit).
"""
from __future__ import annotations
import argparse, pathlib, numpy as np
from actionlockin.scorer import HFScorer
from actionlockin.conditions import SYSTEM, build_conditions
from actionlockin.propositions import load_propositions

SYS = "You are a careful tool-using assistant. You record notes with tools and use observations to answer questions."
CONDS = [("none", ("none", "na", 0)), ("query", ("action", "query", 0)),
         ("commit", ("action", "commitment", 3)), ("assertion", ("assertion", "na", 0))]


ANSWER = "Answer with a single word, True or False."

def strong_turn(gold: bool) -> str:
    v = "correct and well-supported" if gold else "a common misconception and is false"
    return f"Observation: An authoritative fact-check finds this claim is {v}.\n{ANSWER}"

def weak_turn(gold: bool) -> str:
    v = "might be accurate" if gold else "might not be accurate"
    return f"Observation: A quick preliminary check suggests this claim {v}.\n{ANSWER}"

def noevid_turn(gold: bool) -> str:
    return f"Based on your assessment, {ANSWER}"

LEVELS = [("none", noevid_turn), ("weak", weak_turn), ("strong", strong_turn)]


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
                    help="per-item parquet (default results/downstream_harm_<model>[_<props>].parquet)")
    args = ap.parse_args()
    # suffix keyed on the exact canonical path, not the basename: another file that merely
    # shares the stem must not overwrite the canonical parquet under the default name
    is_canonical = pathlib.Path(args.props) == pathlib.Path("data/propositions.jsonl")
    suffix = "" if is_canonical else f"_{pathlib.Path(args.props).stem}"
    out = args.out or f"results/downstream_harm_{args.model.split('/')[-1]}{suffix}.parquet"
    s = HFScorer(args.model, device_map="auto", dtype="bfloat16", cache_dir="models/huggingface")
    props = load_propositions(args.props)[: args.n]

    # acc[level][split][cond] = list correct(0/1); lo[...] = post-evidence logodds(True)
    acc = {e: {g: {c: [] for c, _ in CONDS} for g in ("false", "true")} for e, _ in LEVELS}
    lo = {e: {g: {c: [] for c, _ in CONDS} for g in ("false", "true")} for e, _ in LEVELS}
    rows = []  # per-item audit trail (the printed aggregates are means over these rows)
    for p in props:
        split = "true" if p.gold else "false"
        u1 = f'Determine whether the following claim is true. Claim: "{p.claim}"'
        mids = {key: ct.middle for key, ct in
                {(c.condition, c.framing, c.dose): c for c in build_conditions(p)}.items()}
        for cond, key in CONDS:
            base = [{"role": "system", "content": SYS}, {"role": "user", "content": u1},
                    {"role": "assistant", "content": mids[key]}]
            for etag, fn in LEVELS:
                msgs = base + [{"role": "user", "content": fn(p.gold)}]
                ld = s.logit_diff_chat(msgs)          # prob(True)>=0.5 iff logodds>=0
                correct = (ld >= 0) if p.gold else (ld < 0)
                acc[etag][split][cond].append(1.0 if correct else 0.0)
                lo[etag][split][cond].append(ld)
                rows.append({"model": args.model, "prop_id": p.id, "gold": p.gold,
                             "condition": cond, "evidence": etag,
                             "logodds_post": float(ld), "correct": bool(correct)})

    import pandas as pd
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out)
    print(f"wrote {len(rows)} per-item rows -> {out}")

    print(f"\n========== {args.model}  n={len(props)} ==========")
    for split in ("false", "true"):
        nfa = len(acc["none"][split]["none"])
        print(f"\n--- {split.upper()} claims (n={nfa}); correct answer = {split=='true'} ---")
        print(f"  {'cond':9s} | acc: none  weak  strong")
        for cond, _ in CONDS:
            a = [np.mean(acc[e][split][cond]) for e, _ in LEVELS]
            print(f"  {cond:9s} |      {a[0]:.2f}  {a[1]:.2f}   {a[2]:.2f}")
        # headline harms on FALSE claims: no-evidence decision harm + weak-evidence resistance-to-correction
        for cond in ("commit", "assertion"):
            for lvl in ("none", "weak"):
                d = np.mean(acc[lvl][split][cond]) - np.mean(acc[lvl][split]["none"])
                print(f"  HARM {cond} @ {lvl}-evidence: dAcc(vs none-cond) = {d:+.2f}")


if __name__ == "__main__":
    main()
