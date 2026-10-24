"""Compute Bonferroni-corrected CIs for the 32B/72B confirmatory gate.

Corrected bootstrap bounds require retained per-proposition deltas; aggregate intervals do not
contain enough information to derive them.

Family size k=20 (H1,H3,H4,H5 x 5 models) -> per-test alpha=0.05/20=0.0025 -> two-sided 99.75%
bootstrap CI (percentiles 0.125 / 99.875), n_boot=20000, seeds {0,1,2}. value = logodds_P.
"""
import numpy as np, pandas as pd
from actionlockin.analyze import _pivot

LO, HI = 0.125, 99.875   # two-sided 99.75% (Bonferroni k=20)
NB = 20000

def boot(x, lo=LO, hi=HI, nb=NB, seeds=(0, 1, 2)):
    x = np.asarray(x, float); n = len(x)
    los, his = [], []
    for s in seeds:
        r = np.random.default_rng(s)
        means = x[r.integers(0, n, size=(nb, n))].mean(axis=1)
        los.append(np.percentile(means, lo)); his.append(np.percentile(means, hi))
    return x.mean(), min(los), max(his), max(h - l for l, h in zip(los, his)) - min(h - l for l, h in zip(los, his))

def dose_slopes(df, value="logodds_P"):
    ac = df[(df.condition == "action") & (df.framing == "commitment")]
    out = []
    for _, g in ac.groupby("prop_id"):
        if g["dose"].nunique() >= 2:
            out.append(np.polyfit(g["dose"].to_numpy(float), g[value].to_numpy(float), 1)[0])
    return np.array(out)

for path, name in [("results/gate_Qwen2.5-32B_propositions.parquet", "Qwen2.5-32B"),
                   ("results/gate_Qwen2.5-72B_propositions.parquet", "Qwen2.5-72B")]:
    df = pd.read_parquet(path)
    p = _pivot(df, value="logodds_P").dropna(subset=["none", "assertion", "query", "commit3", "placebo"])
    h1 = (p["commit3"] - p["none"]).to_numpy()
    h3 = (p["commit3"] - p["query"]).to_numpy()
    h5 = (p["placebo"] - p["none"]).to_numpy()
    h4 = dose_slopes(df)
    print(f"\n=== {name}  (n={len(p)} paired props) ===")
    for lab, x, side in [("H1 commit-none", h1, "lo"), ("H3 commit-query", h3, "lo"),
                         ("H4 dose slope", h4, "lo"), ("H5 placebo-none", h5, "hi")]:
        est, clo, chi, seedspread = boot(x)
        bound = clo if side == "lo" else chi
        flag = ("> 0 OK" if (side == "lo" and clo > 0) else
                "<= 0 OK (specificity)" if (side == "hi" and chi <= 0) else "FAIL")
        print(f"  {lab:16s} est={est:+.3f}  99.75%CI=[{clo:+.3f},{chi:+.3f}]  "
              f"corr-{side}={bound:+.3f}  {flag}  (seed-spread {seedspread:.4f})")
