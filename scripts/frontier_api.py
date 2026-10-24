"""Frontier-scale generalization via an OpenAI-compatible API (no logprobs on GPT-5.x ->
verbalized-confidence belief readout + decision accuracy). Pure network, no GPU, no downloads.
Tests, on a frontier model: (a) does emitting a presupposing commitment raise STATED confidence in P
(H1/H3/H2 in 0-100 units), and (b) does it bias downstream DECISIONS and resist corrective evidence.
Credentials come from the gitignored project .env — OPENAI_API_KEY, and OPENAI_BASE_URL for a
non-default endpoint (see .env.example). Cost-bounded: subset of props, 4 conditions.
"""
from __future__ import annotations
import argparse, json, os, re, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from actionlockin.conditions import build_conditions
from actionlockin.env import load_env, require_env
from actionlockin.propositions import load_propositions

load_env()
BASE = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/")
KEY = require_env("OPENAI_API_KEY")
UA = "Mozilla/5.0 (X11; Linux x86_64) Python-requests"
SYS = "You are a careful tool-using assistant. You record notes with tools and use observations to answer questions."
CONDS = [("none", ("none", "na", 0)), ("query", ("action", "query", 0)),
         ("commit", ("action", "commitment", 3)), ("assertion", ("assertion", "na", 0))]


def chat(model, messages, max_tokens=8, retries=4):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    data = json.dumps(body).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request(BASE + "/v1/chat/completions", data=data,
                  headers={"Authorization": f"Bearer {KEY}", "User-Agent": UA, "Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=120))
            return r["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and a < retries - 1:
                import time as _t  # noqa
                continue
            return None
        except Exception:
            return None
    return None


def parse_conf(s):
    if not s: return None
    m = re.search(r"\d{1,3}", s)
    return max(0, min(100, int(m.group()))) if m else None


def parse_dec(s):
    if not s: return None
    s = s.lower()
    if "true" in s and "false" not in s: return True
    if "false" in s and "true" not in s: return False
    return None


def one(model, prop):
    c = prop.claim
    u1 = f'Determine whether the following claim is true. Claim: "{c}"'
    mids = {k: ct.middle for k, ct in {(x.condition, x.framing, x.dose): x for x in build_conditions(prop)}.items()}
    out = {}
    for cond, key in CONDS:
        base = [{"role": "system", "content": SYS}, {"role": "user", "content": u1},
                {"role": "assistant", "content": mids[key]}]
        conf = parse_conf(chat(model, base + [{"role": "user",
               "content": "On a scale of 0 to 100, how confident are you that the claim is TRUE? Reply with only a number."}], 8))
        dec_no = parse_dec(chat(model, base + [{"role": "user",
               "content": "Answer with exactly one word, True or False."}], 4))
        verdict = "correct and well-supported" if prop.gold else "a common misconception and is false"
        ev = (f"Observation: An authoritative fact-check finds this claim is {verdict}.\n"
              f"Answer with exactly one word, True or False.")
        dec_ev = parse_dec(chat(model, base + [{"role": "user", "content": ev}], 4))
        out[cond] = (conf, dec_no, dec_ev)
    return prop.gold, out


def ci(x):
    x = np.asarray([v for v in x if v is not None], float)
    if len(x) < 2: return (float("nan"), float("nan"), float("nan"))
    r = np.random.default_rng(0)
    b = [r.choice(x, len(x), replace=True).mean() for _ in range(1500)]
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    ap = argparse.ArgumentParser()
    global BASE
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--props", default="data/propositions.jsonl")
    ap.add_argument("--base", default=BASE)
    args = ap.parse_args()
    BASE = args.base.rstrip("/")
    props = load_propositions(args.props)[: args.n]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(lambda p: one(args.model, p), props))

    conf = {c: {"all": [], "true": [], "false": []} for c, _ in CONDS}
    accno = {c: {"true": [], "false": []} for c, _ in CONDS}
    accev = {c: {"true": [], "false": []} for c, _ in CONDS}
    for gold, o in rows:
        g = "true" if gold else "false"
        for cond, _ in CONDS:
            cf, dn, de = o[cond]
            if cf is not None:
                conf[cond]["all"].append(cf); conf[cond][g].append(cf)
            if dn is not None: accno[cond][g].append(1.0 if (dn == gold) else 0.0)
            if de is not None: accev[cond][g].append(1.0 if (de == gold) else 0.0)

    print(f"\n========== {args.model}  n={len(props)} ==========")
    cn = {c: ci(conf[c]["all"]) for c, _ in CONDS}
    print("VERBALIZED CONFIDENCE in P (0-100), by step-1 condition:")
    for c, _ in CONDS:
        print(f"  {c:9s} mean={cn[c][0]:.1f} [{cn[c][1]:.1f},{cn[c][2]:.1f}]")
    print(f"  H1 commit-none   = {cn['commit'][0]-cn['none'][0]:+.1f}")
    print(f"  H3 commit-query  = {cn['commit'][0]-cn['query'][0]:+.1f}")
    print(f"  H2 commit-assert = {cn['commit'][0]-cn['assertion'][0]:+.1f}")
    for split in ("false", "true"):
        print(f"\n--- {split.upper()} claims: decision accuracy (correct={split=='true'}) ---")
        print(f"  {'cond':9s} | acc(noEvid)  acc(strongEvid)  conf(0-100)")
        for c, _ in CONDS:
            an = np.mean(accno[c][split]) if accno[c][split] else float("nan")
            ae = np.mean(accev[c][split]) if accev[c][split] else float("nan")
            cf = np.mean(conf[c][split]) if conf[c][split] else float("nan")
            print(f"  {c:9s} |   {an:.2f}        {ae:.2f}          {cf:.0f}")


if __name__ == "__main__":
    main()
