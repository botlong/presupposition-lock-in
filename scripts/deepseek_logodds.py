"""Probe DeepSeek-V3 through the API's top-logprob response.

The script computes logprob(True) - logprob(False) at the answer position. API flooring and top-k
censoring make this diagnostic non-equivalent to raw local logits, so callers must not combine its
output with the open-weight measurements as if they shared a scale.
"""
from __future__ import annotations
import argparse, json, os, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from actionlockin.conditions import SYSTEM, belief_probe, build_conditions
from actionlockin.env import load_env, require_env
from actionlockin.propositions import load_propositions

load_env()
BASE = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
KEY = require_env("DEEPSEEK_API_KEY")
TOP = int(os.environ.get("LO_TOP", "20"))
CONDS = [("none", ("none", "na", 0)), ("commit", ("action", "commitment", 3)),
         ("query", ("action", "query", 0)), ("assertion", ("assertion", "na", 0))]


def call(messages, model, retries=4):
    body = {"model": model, "messages": messages, "max_tokens": 4, "temperature": 0,
            "logprobs": True, "top_logprobs": TOP}
    data = json.dumps(body).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request(BASE + "/chat/completions", data=data,
                  headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json", "User-Agent": "py"})
            return json.load(urllib.request.urlopen(req, timeout=120))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and a < retries - 1:
                continue
            return None
        except Exception:
            return None
    return None


def logodds(messages, model):
    r = call(messages, model)
    try:
        toks = r["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    except Exception:
        return None
    real = [t for t in toks if t["logprob"] > -100]   # drop API floor sentinels (-9999)
    if not real:
        return None
    lt = next((t["logprob"] for t in real if t["token"].strip().lower() == "true"), None)
    lf = next((t["logprob"] for t in real if t["token"].strip().lower() == "false"), None)
    if lt is None and lf is None:
        return None                                   # model answered neither token
    # Floor-imputation removes selection bias: a token absent from top-k has logprob <= the
    # smallest reported logprob, so imputing that floor yields a CONSERVATIVE (downward-biased
    # in magnitude) but unbiased-in-inclusion log-odds. logprob(T)-logprob(F)=logit(T)-logit(F).
    floor = min(t["logprob"] for t in real)
    lt = lt if lt is not None else floor
    lf = lf if lf is not None else floor
    return lt - lf


def one(prop, model):
    u = belief_probe(prop)
    mids = {k: c.middle for k, c in {(x.condition, x.framing, x.dose): x for x in build_conditions(prop)}.items()}
    out = {}
    for cond, key in CONDS:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": u},
                {"role": "assistant", "content": mids[key]},
                {"role": "user", "content": "Answer with exactly one word, True or False."}]
        out[cond] = logodds(msgs, model)
    return prop.gold, out


def ci(x):
    x = np.asarray([v for v in x if v is not None], float)
    if len(x) < 2: return (float("nan"), float("nan"), float("nan"), len(x))
    r = np.random.default_rng(0); b = [r.choice(x, len(x), replace=True).mean() for _ in range(1500)]
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5), len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--props", default="data/arith_props.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    props = load_propositions(args.props)[: args.n]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(lambda p: one(p, args.model), props))

    def delta(a, b, split=None):
        out = []
        for gold, o in rows:
            if split is not None and bool(gold) != split:
                continue
            if o[a] is not None and o[b] is not None:
                out.append(o[a] - o[b])
        return out
    print(f"\n========== {args.model}  ({args.props}, n={len(props)})  TOKEN-LEVEL log-odds ==========")
    for nm, a, b in [("H1 commit-none", "commit", "none"), ("H3 commit-query", "commit", "query"),
                     ("H2 commit-assertion", "commit", "assertion")]:
        e = ci(delta(a, b)); print(f"  {nm:22s} {e[0]:+.3f} [{e[1]:+.3f},{e[2]:+.3f}]  (n={e[3]})")
    for split, lbl in [(True, "gold=True"), (False, "gold=False")]:
        e = ci(delta("commit", "none", split)); print(f"  H1 {lbl:10s} {e[0]:+.3f} [{e[1]:+.3f},{e[2]:+.3f}]  (n={e[3]})")
    # data coverage
    both = sum(1 for _, o in rows if o["none"] is not None and o["commit"] is not None)
    print(f"  items with clean True/False readout in none&commit: {both}/{len(rows)}")


if __name__ == "__main__":
    main()
