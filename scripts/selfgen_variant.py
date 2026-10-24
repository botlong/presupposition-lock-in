"""Ecological (non-teacher-forced) variant — does the lock-in survive when the model
GENERATES the commitment action itself, instead of us injecting it?

For each proposition: (turn 1) ask the model to record the claim as an established fact via a tool
call; the model GENERATES its own assistant turn; we then use that self-generated turn as the middle
and read belief (log-odds) on the fixed probe, vs the none baseline. If self-generated commitments
raise belief comparably to teacher-forced ones, the effect is not an artifact of our wording/injection.
"""
from __future__ import annotations
import argparse, numpy as np
from actionlockin.scorer import HFScorer
from actionlockin.conditions import SYSTEM, belief_probe, build_conditions
from actionlockin.propositions import load_propositions
from actionlockin.selfgen import strict_faithful


def gen_prompt(claim: str) -> str:
    return (f'You have a tool: log_fact(fact=..., status=...). Record the following claim as an '
            f'established fact by issuing a single tool call, then stop.\nClaim: "{claim}"\n'
            f'Respond with ONLY the tool call.')


def ci(x):
    x = np.asarray(x, float); r = np.random.default_rng(0)
    b = [r.choice(x, len(x), replace=True).mean() for _ in range(1500)]
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--out", default=None,
                    help="path to save retained per-proposition rows as parquet (for auditability)")
    args = ap.parse_args()
    s = HFScorer(args.model, device_map="auto", dtype="bfloat16", cache_dir="models/huggingface")
    torch = s._torch
    props = load_propositions("data/propositions.jsonl")[: args.n]

    selfgen_shift, tf_shift, presupposed = [], [], 0
    faith_shift, unfaith_shift = [], []
    examples = []
    rows = []
    for p in props:
        # 1) self-generate the commitment action
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": gen_prompt(p.claim)}]
        ids = s._input_ids(s.tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")).to(s.model.device)
        with torch.no_grad():
            out = s.model.generate(ids, max_new_tokens=48, do_sample=False,
                                   pad_token_id=s.tok.eos_token_id)
        gen = s.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        gen = gen.split("\n")[0].strip() or gen.strip()        # first line = the call

        # 2) belief readouts: none vs self-generated middle (vs teacher-forced commit3)
        u = belief_probe(p)
        cts = {(c.condition, c.framing, c.dose): c for c in build_conditions(p)}
        lo_none = s.logit_diff(SYSTEM, u, cts[("none", "na", 0)].middle)
        lo_self = s.logit_diff(SYSTEM, u, gen)
        lo_tf = s.logit_diff(SYSTEM, u, cts[("action", "commitment", 3)].middle)
        selfgen_shift.append(lo_self - lo_none)
        tf_shift.append(lo_tf - lo_none)
        is_faithful = strict_faithful(gen, p.claim)
        if is_faithful:
            presupposed += 1; faith_shift.append(lo_self - lo_none)
        else:
            unfaith_shift.append(lo_self - lo_none)
        rows.append({"model": args.model, "prop_id": p.id, "gold": p.gold,
                     "faithful": bool(is_faithful), "logodds_none": float(lo_none),
                     "logodds_selfgen": float(lo_self), "logodds_tf": float(lo_tf),
                     "h1_selfgen": float(lo_self - lo_none), "h1_tf": float(lo_tf - lo_none),
                     "gen": gen})
        if len(examples) < 4:
            examples.append((p.claim[:40], gen[:70]))

    e_self, e_tf = ci(selfgen_shift), ci(tf_shift)
    print(f"\n{args.model}  n={len(props)}")
    print(f"  self-generated commitment FAITHFULLY presupposes claim: {presupposed}/{len(props)} "
          f"(strict criterion: generation contains the full claim verbatim; "
          f"rest rewrote, paraphrased, or negated it)")
    print(f"  H1 SELF-GEN (all)        = {e_self[0]:+.3f} [{e_self[1]:+.3f}, {e_self[2]:+.3f}]")
    if faith_shift:
        ef = ci(faith_shift); print(f"  H1 SELF-GEN (faithful)   = {ef[0]:+.3f} [{ef[1]:+.3f}, {ef[2]:+.3f}]  (n={len(faith_shift)})")
    if unfaith_shift:
        eu = ci(unfaith_shift); print(f"  H1 SELF-GEN (unfaithful) = {eu[0]:+.3f} [{eu[1]:+.3f}, {eu[2]:+.3f}]  (n={len(unfaith_shift)})")
    print(f"  H1 TEACHER-FORCED        = {e_tf[0]:+.3f} [{e_tf[1]:+.3f}, {e_tf[2]:+.3f}]")
    print("  example self-generated actions:")
    for cl, g in examples:
        print(f"    [{cl}...] -> {g}")

    if args.out:
        import pandas as pd
        pd.DataFrame(rows).to_parquet(args.out)
        print(f"  wrote per-proposition rows -> {args.out}")


if __name__ == "__main__":
    main()
