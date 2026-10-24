"""Sampled-decoding validation of the forced-token answer readout.

The flip/accuracy analyses read the decision as the sign of the True-vs-False answer-token
log-odds at a teacher-forced cue. A deployed agent samples its answer instead, so the
forced-token readout needs a behavioral counterpart: the none and dose-3 commitment cells
are decoded from the same single-turn episode with one greedy continuation plus K
stochastic samples, and each decoded answer is parsed from the retained text. Per-item
concordance between the forced readout and decoded behavior is then computable from the
saved rows alone.

Padding targets come from the full seven-cell grid so the padded middles are byte-identical
to the ones the forced-token gate runs scored. Sampling is seeded per (proposition, cell)
so draws do not depend on iteration order.
"""
from __future__ import annotations
import argparse
import pathlib
import zlib
import numpy as np
from actionlockin.scorer import HFScorer
from actionlockin.conditions import SYSTEM, ANSWER_CUE, belief_probe, build_conditions, pad_to_match
from actionlockin.elicit import parse_tf_answer
from actionlockin.propositions import load_propositions

CELLS = (("none", ("none", "na", 0)), ("commit3", ("action", "commitment", 3)))


def ci(x):
    x = np.asarray(x, float); r = np.random.default_rng(0)
    b = [r.choice(x, len(x), replace=True).mean() for _ in range(1500)]
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--props", default="data/arith_hard.jsonl")
    ap.add_argument("--k", type=int, default=20, help="stochastic samples per cell")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=6)
    ap.add_argument("--out", default=None,
                    help="per-generation parquet (default results/sampledread_<model>_<props>.parquet)")
    args = ap.parse_args()
    stem = pathlib.Path(args.props).stem
    out = args.out or f"results/sampledread_{args.model.split('/')[-1]}_{stem}.parquet"
    s = HFScorer(args.model, device_map="auto", dtype="bfloat16", cache_dir="models/huggingface")
    torch = s._torch
    props = load_propositions(args.props)[: args.n]

    rows = []
    for p in props:
        user = belief_probe(p)
        cts = {(c.condition, c.framing, c.dose): c for c in build_conditions(p)}
        target = max(len(s.tokenize(c.middle)) for c in cts.values())
        pre = s._prefix_ids(SYSTEM, user).to(s.model.device)
        for cell, key in CELLS:
            mid = pad_to_match(s.tokenize, cts[key].middle, target, filler=" note")
            lo = s.logit_diff(SYSTEM, user, mid)
            cont = s.tok(f"{mid}\n{ANSWER_CUE}", add_special_tokens=False,
                         return_tensors="pt")["input_ids"].to(s.model.device)
            ids = torch.cat([pre, cont], dim=1)
            gens = []
            with torch.no_grad():
                g = s.model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                                     num_beams=1, repetition_penalty=1.0,
                                     pad_token_id=s.tok.eos_token_id)
                gens.append(("greedy", -1,
                             s.tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)))
                # explicit top_k/top_p/repetition_penalty: the model's generation_config must
                # not silently shape the temperature-only sampling distribution
                torch.manual_seed(zlib.crc32(f"{p.id}|{cell}".encode()) & 0x7FFFFFFF)
                g = s.model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=True,
                                     temperature=args.temperature, top_k=0, top_p=1.0,
                                     repetition_penalty=1.0, num_return_sequences=args.k,
                                     pad_token_id=s.tok.eos_token_id)
                for j in range(args.k):
                    gens.append(("sample", j,
                                 s.tok.decode(g[j, ids.shape[1]:], skip_special_tokens=True)))
            for decode, idx, text in gens:
                rows.append({"model": args.model, "prop_id": p.id, "gold": p.gold,
                             "cell": cell, "decode": decode, "sample_idx": idx,
                             "gen": text, "answer": parse_tf_answer(text),
                             "logodds_forced": float(lo)})

    import pandas as pd
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out)
    print(f"wrote {len(df)} per-generation rows -> {out}")

    r = summarize(df)
    print(f"\n========== {args.model}  SAMPLED READOUT  {stem}  n={df.prop_id.nunique()} "
          f"(K={args.k} @ T={args.temperature}) ==========")
    print(f"  unparsable answers: {r['other_rate']:.1%} of all generations")
    for cell, _ in CELLS:
        c = r["cells"][cell]; a = c["agree_ci"]
        print(f"  {cell:8s} P(answer True):  forced {c['forced_true']:.2f}   "
              f"greedy {c['greedy_true']:.2f}   sampled {c['sampled_true']:.2f}   "
              f"| forced-greedy agreement {a[0]:.2f} [{a[1]:.2f},{a[2]:.2f}] "
              f"(n={c['agree_n']} parseable)")
    if r["flip"] is not None:
        fl = r["flip"]
        print(f"\n  FLIP on false claims, base-correct under forced none (n={fl['n']}):")
        print(f"    forced  flip = {fl['forced'][0]:.2f} [{fl['forced'][1]:.2f},{fl['forced'][2]:.2f}]")
        print(f"    greedy  flip = {fl['greedy'][0]:.2f} [{fl['greedy'][1]:.2f},{fl['greedy'][2]:.2f}]")
        print(f"    sampled True-fraction = {fl['sampled'][0]:.2f} "
              f"[{fl['sampled'][1]:.2f},{fl['sampled'][2]:.2f}]")


def summarize(df) -> dict:
    """Aggregate a sampled-readout frame: per-cell True-rates under the three readouts, the
    forced-vs-greedy agreement, and — on FALSE claims the none cell correctly rejects under
    the forced readout — whether the commitment flip replicates under real decoding."""
    out = {"other_rate": float((df.answer == "other").mean()), "cells": {}, "flip": None}
    per_item = {}
    for cell, _ in CELLS:
        d = df[df.cell == cell]
        forced = d.groupby("prop_id").logodds_forced.first() >= 0
        gans = d[d.decode == "greedy"].set_index("prop_id").answer
        greedy = gans == "true"
        samp = (d[d.decode == "sample"].assign(t=lambda x: x.answer == "true")
                .groupby("prop_id").t.mean())
        per_item[cell] = (forced, greedy, samp)
        # agreement only over items whose greedy continuation actually answered: an
        # unparsable greedy folded into not-True would count forced-False items as
        # spurious agreements
        valid = gans[gans.isin(("true", "false"))].index
        out["cells"][cell] = {
            "forced_true": float(forced.mean()), "greedy_true": float(greedy.mean()),
            "sampled_true": float(samp.mean()),
            "agree_n": int(len(valid)),
            "agree_ci": (ci((forced.reindex(valid) == greedy.reindex(valid))
                            .astype(float).to_numpy())
                         if len(valid) else (float("nan"),) * 3)}
    gold = df.groupby("prop_id").gold.first()
    f_none = per_item["none"][0]
    f_c3, g_c3, s_c3 = per_item["commit3"]
    base = gold.index[(~gold) & (~f_none.reindex(gold.index).fillna(True))]
    if len(base):
        out["flip"] = {"n": int(len(base)),
                       "forced": ci(f_c3.reindex(base).astype(float).to_numpy()),
                       "greedy": ci(g_c3.reindex(base).astype(float).to_numpy()),
                       "sampled": ci(s_c3.reindex(base).to_numpy())}
    return out


if __name__ == "__main__":
    main()
