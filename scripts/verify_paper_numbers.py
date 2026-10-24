"""Independent recomputation of every number in the paper (journal version) that can be
derived from the retained parquets in results/, including the Llama-3.1-8B runs, the six-model
Bonferroni family and the two-sided mitigation rates (also guards the figures' transcribed
constants against drift). Prints PASS/FAIL per claim; exits non-zero on any FAIL.

Usage (from repo root):  python scripts/verify_paper_numbers.py
"""
from __future__ import annotations
import pathlib
import sys
import numpy as np
import pandas as pd

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from actionlockin.analyze import (contrasts, _pivot, _boot_ci, belief_shift_by_gold,
                                  accuracy_harm, mitigation_report, channel_report)
from actionlockin.selfgen import strict_faithful

R = "results/"
GATE_PROPS = {
    "1.5B":    R + "gate_Qwen2.5-1.5B_propositions.parquet",
    "7B":      R + "gate_Qwen2.5-7B_propositions.parquet",
    "32B":     R + "gate_Qwen2.5-32B_propositions.parquet",
    "72B":     R + "gate_Qwen2.5-72B_propositions.parquet",
    "Mistral": R + "gate_Mistral-7B-Instruct-v0.3.parquet",
}
GATE_HARD = {m: R + f"gate_Qwen2.5-{m}_arith_hard.parquet" for m in ("1.5B", "7B", "32B", "72B")}
GATE_EASY = {m: R + f"gate_Qwen2.5-{m}_arith_props.parquet" for m in ("1.5B", "7B", "32B", "72B")}
MIT = {m: R + f"mitigation_Qwen2.5-{m}_arith_hard.parquet" for m in ("1.5B", "7B", "32B", "72B")}
CHAN_HARD = {m: R + f"channel_Qwen2.5-{m}_arith_hard.parquet" for m in ("1.5B", "7B", "32B", "72B")}
CHAN_FACT = {m: R + f"channel_Qwen2.5-{m}_propositions.parquet" for m in ("1.5B", "7B", "32B", "72B")}
SELFGEN = {m: R + f"selfgen_Qwen2.5-{m}.parquet" for m in ("1.5B", "7B", "32B", "72B")}
OLD_SMALL = {  # Independent run artifacts are needed to quantify cross-run stability.
    "1.5B": R + "gate_Qwen2.5-1.5B-Instruct.parquet",
    "7B":   R + "gate_Qwen2.5-7B-Instruct.parquet",
}

fails = []
def chk(label, got, want, tol):
    """got/want: float or tuple of floats (est, lo, hi). tol: abs tolerance (scalar)."""
    g = np.atleast_1d(np.asarray(got, float))
    w = np.atleast_1d(np.asarray(want, float))
    ok = g.shape == w.shape and np.all(np.abs(g - w) <= tol)
    tag = "PASS" if ok else "FAIL"
    if not ok:
        fails.append(label)
    print(f"[{tag}] {label}\n       got  {np.round(g, 4).tolist()}\n       want {np.round(w, 4).tolist()}  (tol {tol})")
    return ok

def chk_bool(label, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    if not cond:
        fails.append(label)
    print(f"[{tag}] {label}  {detail}")
    return cond

def tup(d):
    return (d["estimate"], d["ci_low"], d["ci_high"])

def paired(df, a, b, value="logodds_P", gold=None):
    """per-prop paired delta a-b with _boot_ci (2000 boots, seed 0)."""
    sub = df if gold is None else df[df.gold == gold]
    w = _pivot(sub, value=value)
    return _boot_ci((w[a] - w[b]).to_numpy())

print("=" * 100)
print("SECTION 1: Table 1 + Fig 1 + Fig 3 (gate, 120 propositions, log-odds)")
print("=" * 100)
T1 = {  # Table 1 rounds these contrasts to two decimal places.
    "1.5B": (0.89, 0.79, 0.34, -1.02, -1.21),
    "7B": (1.01, 0.90, 0.89, -1.37, -2.08),
    "32B": (3.19, 2.24, 1.07, -1.09, -2.71),
    "72B": (5.08, 3.43, 2.02, 0.02, -4.19),
    "Mistral": (3.05, 2.06, 1.16, -0.71, -5.77),
}
FIG3_H1 = {"1.5B": (0.892, 0.663, 1.106), "7B": (1.015, 0.614, 1.452),
           "32B": (3.193, 2.616, 3.768), "72B": (5.084, 4.297, 5.851)}
FIG3_H3 = {"1.5B": (0.794, 0.567, 1.007), "7B": (0.903, 0.593, 1.233),
           "32B": (2.240, 1.650, 2.860), "72B": (3.425, 2.682, 4.169)}
FIG1_H1 = {"1.5B": (0.89, 0.66, 1.11), "7B": (1.01, 0.61, 1.45), "Mistral": (3.05, 2.36, 3.80)}
FIG1_H3 = {"1.5B": (0.79, 0.57, 1.01), "7B": (0.90, 0.59, 1.23), "Mistral": (2.06, 1.54, 2.59)}
FIG1_H4 = {"1.5B": (0.34, 0.28, 0.40), "7B": (0.89, 0.68, 1.10), "Mistral": (1.16, 0.94, 1.38)}

gate_c = {}
gate_df = {}
for m, path in GATE_PROPS.items():
    df = pd.read_parquet(path)
    gate_df[m] = df
    c = contrasts(df, value="logodds_P")
    gate_c[m] = c
    got = (c["h1_masked_commit_delta"]["estimate"], c["h3_commit_vs_query"]["estimate"],
           c["h4_dose_slope"]["estimate"], c["h5_placebo"]["estimate"],
           c["h2_action_specific"]["estimate"])
    chk(f"Table1 {m} (H1,H3,H4,H5,H2)", got, T1[m], 0.0051)
    # per-model CI excludes 0 for H1/H3/H4 (paper: 'every per-model CI excluding 0')
    chk_bool(f"H1/H3/H4 CI_low>0 at {m}",
             c["h1_masked_commit_delta"]["ci_low"] > 0 and c["h3_commit_vs_query"]["ci_low"] > 0
             and c["h4_dose_slope"]["ci_low"] > 0)

for m in FIG3_H1:
    chk(f"Fig3 H1 {m}", tup(gate_c[m]["h1_masked_commit_delta"]), FIG3_H1[m], 0.0006)
    chk(f"Fig3 H3 {m}", tup(gate_c[m]["h3_commit_vs_query"]), FIG3_H3[m], 0.0006)
chk("Fig3 Mistral H1 point (x marker 3.046)", gate_c["Mistral"]["h1_masked_commit_delta"]["estimate"], 3.046, 0.0006)
for m in FIG1_H1:
    chk(f"Fig1 H1 {m}", tup(gate_c[m]["h1_masked_commit_delta"]), FIG1_H1[m], 0.006)
    chk(f"Fig1 H3 {m}", tup(gate_c[m]["h3_commit_vs_query"]), FIG1_H3[m], 0.006)
    chk(f"Fig1 H4 {m}", tup(gate_c[m]["h4_dose_slope"]), FIG1_H4[m], 0.006)

print()
print("=" * 100)
print("SECTION 2: \\S4.2 scale (SD, Cohen's d, probability units, placebo-by-gold, H2@hard)")
print("=" * 100)
# none-baseline SD grows 3.3 -> 17.4 (1.5B -> 72B)
for m, want in [("1.5B", 3.3), ("72B", 17.4)]:
    w = _pivot(gate_df[m], value="logodds_P")
    sd0 = np.std(w["none"].to_numpy())          # population SD
    sd1 = np.std(w["none"].to_numpy(), ddof=1)  # sample SD
    ok = min(abs(sd0 - want), abs(sd1 - want)) <= 0.05
    chk_bool(f"none-baseline log-odds SD {m} ~ {want}", ok, f"ddof0={sd0:.3f} ddof1={sd1:.3f}")

# Cohen's d (paired, commit3-none): 0.70/0.43/0.97/1.13
for m, want in [("1.5B", 0.70), ("7B", 0.43), ("32B", 0.97), ("72B", 1.13)]:
    w = _pivot(gate_df[m], value="logodds_P")
    d = (w["commit3"] - w["none"]).to_numpy()
    d0 = d.mean() / np.std(d)
    d1 = d.mean() / np.std(d, ddof=1)
    ok = min(abs(d0 - want), abs(d1 - want)) <= 0.005
    chk_bool(f"Cohen's d {m} = {want}", ok, f"ddof0={d0:.4f} ddof1={d1:.4f}")

# H1 in probability units small: <=0.10 on the Qwen ladder; 0.10 (2dp) on Mistral
for m in GATE_PROPS:
    cp = contrasts(gate_df[m], value="p_of_P")
    est = cp["h1_masked_commit_delta"]["estimate"]
    cap = 0.105 if m == "Mistral" else 0.10
    chk_bool(f"H1 in probability units small at {m} (cap {cap})", est <= cap + 1e-9, f"est={est:.4f}")

# placebo split by gold (placebo - none), log-odds
PLACEBO_FALSE = {"7B": 0.96, "72B": 2.03, "Mistral": 0.60, "32B": 0.02}
for m, want in PLACEBO_FALSE.items():
    est, lo, hi = paired(gate_df[m], "placebo", "none", gold=False)
    chk(f"placebo-none on FALSE claims {m}", est, want, 0.005)
    if m == "32B":
        chk_bool("placebo FALSE 32B CI includes 0", lo <= 0 <= hi, f"[{lo:.3f},{hi:.3f}]")
    else:
        chk_bool(f"placebo FALSE {m} CI excludes 0", lo > 0, f"[{lo:.3f},{hi:.3f}]")
est, lo, hi = paired(gate_df["1.5B"], "placebo", "none", gold=False)
chk_bool("placebo FALSE 1.5B <= 0 (clean specificity)", hi <= 0, f"{est:.3f} [{lo:.3f},{hi:.3f}]")
est, lo, hi = paired(gate_df["1.5B"], "placebo", "none", gold=True)
chk_bool("placebo TRUE 1.5B <= 0 (clean specificity)", hi <= 0, f"{est:.3f} [{lo:.3f},{hi:.3f}]")
PLACEBO_TRUE = {"7B": -3.28, "32B": -2.00, "72B": -1.62}
for m, want in PLACEBO_TRUE.items():
    est, lo, hi = paired(gate_df[m], "placebo", "none", gold=True)
    chk(f"placebo-none on TRUE claims {m}", est, want, 0.005)

# H2 at 72B: -4.2 (props) and -11.0 (hard arithmetic)
chk("H2 72B props = -4.2", gate_c["72B"]["h2_action_specific"]["estimate"], -4.2, 0.05)
hard72 = pd.read_parquet(GATE_HARD["72B"])
c_h72 = contrasts(hard72, value="logodds_P")
chk("H2 72B arith_hard = -11.0", c_h72["h2_action_specific"]["estimate"], -11.0, 0.05)

print()
print("=" * 100)
print("SECTION 3: Bonferroni family (k=20, 99.75% CIs, n_boot=20000, seeds 0/1/2)")
print("=" * 100)
LOq, HIq, NB = 0.125, 99.875, 20000

def boot_bounds(x, seeds=(0, 1, 2)):
    x = np.asarray(x, float); n = len(x)
    los, his = [], []
    for s in seeds:
        r = np.random.default_rng(s)
        means = x[r.integers(0, n, size=(NB, n))].mean(axis=1)
        los.append(np.percentile(means, LOq)); his.append(np.percentile(means, HIq))
    return x.mean(), los, his

def dose_slopes(df, value="logodds_P"):
    ac = df[(df.condition == "action") & (df.framing == "commitment")]
    out = []
    for _, g in ac.groupby("prop_id"):
        if g["dose"].nunique() >= 2:
            out.append(np.polyfit(g["dose"].to_numpy(float), g[value].to_numpy(float), 1)[0])
    return np.array(out)

corr_lo_all, corr_lo_big, h5_hi = {}, {}, {}
seed_spread = 0.0        # max across-seed variation of the DECISION bound (paper wording)
width_spread = 0.0       # bonferroni_frontier.py convention: across-seed CI-width spread
for m in GATE_PROPS:
    w = _pivot(gate_df[m], value="logodds_P")
    h1 = (w["commit3"] - w["none"]).to_numpy()
    h3 = (w["commit3"] - w["query"]).to_numpy()
    h5 = (w["placebo"] - w["none"]).to_numpy()
    h4 = dose_slopes(gate_df[m])
    for lab, x, side in [("H1", h1, "lo"), ("H3", h3, "lo"), ("H4", h4, "lo"), ("H5", h5, "hi")]:
        est, los, his = boot_bounds(x)
        bound = min(los) if side == "lo" else max(his)
        seed_spread = max(seed_spread,
                          (max(los) - min(los)) if side == "lo" else (max(his) - min(his)))
        widths = [h - l for l, h in zip(los, his)]
        width_spread = max(width_spread, max(widths) - min(widths))
        if side == "lo":
            corr_lo_all[(m, lab)] = bound
            if m in ("32B", "72B"):
                corr_lo_big[(m, lab)] = bound
        else:
            h5_hi[m] = bound
        print(f"    {m:8s} {lab}: est={est:+.3f} corrected-{side} bound={bound:+.3f}")

chk_bool("all 15 H1/H3/H4 corrected lower bounds > 0", all(v > 0 for v in corr_lo_all.values()),
         f"min={min(corr_lo_all.values()):+.3f} at {min(corr_lo_all, key=corr_lo_all.get)}")
chk("smallest bound overall = +0.23 (1.5B H4)", min(corr_lo_all.values()), 0.235, 0.005)
chk_bool("argmin is 1.5B H4", min(corr_lo_all, key=corr_lo_all.get) == ("1.5B", "H4"))
h1_bounds = {k: v for k, v in corr_lo_all.items() if k[1] == "H1"}
chk("smallest H1 bound = +0.37 (7B)", min(h1_bounds.values()), 0.37, 0.005)
chk_bool("H1 argmin is 7B", min(h1_bounds, key=h1_bounds.get) == ("7B", "H1"))
chk("32B/72B corrected lower bounds min = +0.67", min(corr_lo_big.values()), 0.67, 0.005)
chk("32B/72B corrected lower bounds max = +3.82", max(corr_lo_big.values()), 3.82, 0.005)
chk_bool("H5 corrected upper <= -0.33 through 32B",
         all(h5_hi[m] <= -0.33 for m in ("1.5B", "7B", "32B")),
         f"1.5B={h5_hi['1.5B']:+.3f} 7B={h5_hi['7B']:+.3f} 32B={h5_hi['32B']:+.3f} (Mistral={h5_hi['Mistral']:+.3f})")
chk_bool("H5 corrected upper NOT <= 0 at 72B", h5_hi["72B"] > 0, f"72B={h5_hi['72B']:+.3f}")
chk_bool("corrected bounds vary < 0.07 across seeds", seed_spread < 0.07,
         f"max bound spread={seed_spread:.4f} (repo width-spread convention: {width_spread:.4f})")

print()
print("=" * 100)
print("SECTION 4: \\S4.3 content vs form controls")
print("=" * 100)
# (i) dose-1 - query
D1Q = {"1.5B": 0.11, "7B": -0.88, "32B": 0.11, "72B": -0.61}
for m, want in D1Q.items():
    df = gate_df[m]
    d1 = df[(df.condition == "action") & (df.framing == "commitment") & (df.dose == 1)] \
        .groupby("prop_id")["logodds_P"].mean()
    q = df[(df.condition == "action") & (df.framing == "query") & (df.dose == 0)] \
        .groupby("prop_id")["logodds_P"].mean()
    delta = (d1 - q).dropna().to_numpy()
    est, lo, hi = _boot_ci(delta)
    chk(f"dose1-query {m}", est, want, 0.005)
    if m in ("7B", "72B"):
        chk_bool(f"dose1-query {m} significantly negative", hi < 0, f"[{lo:.3f},{hi:.3f}]")
    # dose3-query positive at every scale
    chk_bool(f"dose3-query > 0 at {m}", gate_c[m]["h3_commit_vs_query"]["ci_low"] > 0)

# (ii) claim-presence share = (query-none)/H1
for m in GATE_PROPS:
    w = _pivot(gate_df[m], value="logodds_P")
    share = (w["query"] - w["none"]).mean() / (w["commit3"] - w["none"]).mean()
    print(f"    claim-presence share {m}: {share:.3f}")
    if m in ("1.5B", "7B"):
        chk_bool(f"claim share ~11% at {m}", abs(share - 0.11) <= 0.015, f"{share:.3f}")
    if m in ("32B", "72B"):
        chk_bool(f"claim share ~30-33% at {m}", 0.295 <= share <= 0.335, f"{share:.3f}")

# (iii) fluency: commit3 - query mean in-context logprob (+0.25/+1.46/+1.70 at 1.5B/7B/72B)
FLU = {"1.5B": 0.25, "7B": 1.45, "72B": 1.70}
for m, want in FLU.items():
    df = gate_df[m]
    c3 = df[(df.condition == "action") & (df.framing == "commitment") & (df.dose == 3)] \
        .groupby("prop_id")["mid_mean_logprob"].mean()
    q = df[(df.condition == "action") & (df.framing == "query") & (df.dose == 0)] \
        .groupby("prop_id")["mid_mean_logprob"].mean()
    chk(f"fluency commit3-query {m}", (c3 - q).mean(), want, 0.005)

# adjusted H3 (regression): ~0 / -6.5[-11.8,-1.3] / -7.1 / +2.5 / -20
ADJ = {"1.5B": (None, None), "7B": (-6.6, (-11.8, -1.5)), "Mistral": (-7.1, None),
       "32B": (2.5, None), "72B": (-20.0, None)}
for m, (want, ci) in ADJ.items():
    adj = gate_c[m]["h3_commit_vs_query_adj"]
    print(f"    adjusted H3 {m}: {adj['estimate']:+.3f} [{adj['ci_low']:+.3f},{adj['ci_high']:+.3f}]")
    if m == "1.5B":
        chk_bool("adjusted H3 1.5B ~ 0", abs(adj["estimate"]) < 0.5, f"{adj['estimate']:+.3f}")
    elif m == "72B":
        chk_bool("adjusted H3 72B ~ -20 (artifactual)", abs(adj["estimate"] - (-20)) <= 1.0,
                 f"{adj['estimate']:+.3f}")
    else:
        chk(f"adjusted H3 {m}", adj["estimate"], want, 0.05)
    if ci:
        chk(f"adjusted H3 {m} CI", (adj["ci_low"], adj["ci_high"]), ci, 0.05)

print()
print("=" * 100)
print("SECTION 5: \\S4.4 source attribution (channel) + Appendix Table 2")
print("=" * 100)
APP = {  # (scale, set): (self-user (est,lo,hi), self-system (est,lo,hi))
    ("1.5B", "arith"): ((1.14, 1.07, 1.20), (-0.04, -0.11, 0.03)),
    ("1.5B", "fact"):  ((0.25, -0.02, 0.49), (-0.24, -0.43, -0.06)),
    ("7B", "arith"):   ((2.94, 2.72, 3.16), (2.23, 1.92, 2.57)),
    ("7B", "fact"):    ((0.86, 0.52, 1.20), (1.17, 0.81, 1.58)),
    ("32B", "arith"):  ((4.52, 4.22, 4.79), (1.29, 1.03, 1.56)),
    ("32B", "fact"):   ((1.35, 0.76, 1.90), (0.99, 0.49, 1.45)),
    ("72B", "arith"):  ((2.20, 1.92, 2.47), (0.13, -0.15, 0.40)),
    ("72B", "fact"):   ((-1.88, -2.28, -1.51), (-2.53, -2.90, -2.21)),
}
chan = {}
for m in ("1.5B", "7B", "32B", "72B"):
    for tag, path in [("arith", CHAN_HARD[m]), ("fact", CHAN_FACT[m])]:
        df = pd.read_parquet(path)
        chan[(m, tag)] = df
        rep = channel_report(df, value="logodds_P")
        su = tup(rep["_contrasts"]["source_self_minus_source_user"])
        ss = tup(rep["_contrasts"]["source_self_minus_source_system"])
        wu, ws = APP[(m, tag)]
        chk(f"App self-user {m} {tag}", su, wu, 0.005)
        chk(f"App self-system {m} {tag}", ss, ws, 0.005)
        # length matching: all 7 arms share identical mid_n_tokens per prop
        neq = int((df.groupby("prop_id")["mid_n_tokens"].nunique() > 1).sum())
        chk_bool(f"0 unequal-length items {m} {tag}", neq == 0, f"unequal={neq}")

# 72B false-claim deltas vs none: factual self +8.99 / user +12.01; arith self +15.84 / user +14.28
def arm_delta_false(df, cond, fr, dose):
    sub = df[df.gold == False]
    arm = sub[(sub.condition == cond) & (sub.framing == fr) & (sub.dose == dose)] \
        .groupby("prop_id")["logodds_P"].mean()
    none = sub[(sub.condition == "none") & (sub.framing == "na") & (sub.dose == 0)] \
        .groupby("prop_id")["logodds_P"].mean()
    return _boot_ci((arm - none).dropna().to_numpy())

est, lo, hi = arm_delta_false(chan[("72B", "fact")], "assertion", "self", 0)
chk("72B factual FALSE: self vs none = +8.99", est, 8.99, 0.005)
est, lo, hi = arm_delta_false(chan[("72B", "fact")], "assertion", "user", 0)
chk("72B factual FALSE: user vs none = +12.01", est, 12.01, 0.005)
est, lo, hi = arm_delta_false(chan[("72B", "arith")], "assertion", "self", 0)
chk("72B arith FALSE: self vs none = +15.84", est, 15.84, 0.005)
est, lo, hi = arm_delta_false(chan[("72B", "arith")], "assertion", "user", 0)
chk("72B arith FALSE: user vs none = +14.28", est, 14.28, 0.005)

# clipped +-10 de-saturation: 72B factual self-user = -0.85 [-1.16,-0.57]
df = chan[("72B", "fact")].copy()
df["clip"] = df["logodds_P"].clip(-10, 10)
s = df[(df.condition == "assertion") & (df.framing == "self") & (df.dose == 0)].groupby("prop_id")["clip"].mean()
u = df[(df.condition == "assertion") & (df.framing == "user") & (df.dose == 0)].groupby("prop_id")["clip"].mean()
chk("72B factual self-user clipped +-10", _boot_ci((s - u).dropna().to_numpy()),
    (-0.85, -1.16, -0.57), 0.005)

# certainty-matched window on 72B arith (declared def: baseline |logodds| <= 8): +2.52 [+2.30,+2.74]
df = chan[("72B", "arith")]
none = df[(df.condition == "none") & (df.framing == "na") & (df.dose == 0)].groupby("prop_id")["logodds_P"].mean()
s = df[(df.condition == "assertion") & (df.framing == "self") & (df.dose == 0)].groupby("prop_id")["logodds_P"].mean()
u = df[(df.condition == "assertion") & (df.framing == "user") & (df.dose == 0)].groupby("prop_id")["logodds_P"].mean()
keep = none[none.abs() <= 8].index
chk("72B arith self-user, window |baseline logodds|<=8", _boot_ci((s[keep] - u[keep]).dropna().to_numpy()),
    (2.52, 2.30, 2.73), 0.0051)

print()
print("=" * 100)
print("SECTION 6: \\S4.5 self-generation")
print("=" * 100)
def ci1500(x):
    x = np.asarray(x, float); r = np.random.default_rng(0)
    b = [r.choice(x, len(x), replace=True).mean() for _ in range(1500)]
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

# Faithfulness is re-derived from (gen, claim) with the strict verbatim-containment rule;
# the parquets' stored `faithful` column is a keyword heuristic and is not read.
_props = pd.read_json("data/propositions.jsonl", lines=True)
_CLAIM = dict(zip(_props["id"], _props["claim"]))
SG_FAITH = {"1.5B": (1.15, 0.86, 1.45), "32B": (2.14, 1.49, 2.77), "72B": (4.58, 3.91, 5.26)}
SG_TF = {"32B": 3.03, "72B": 5.47, "7B": 1.54}
SG_N = {"1.5B": 59, "7B": 120, "32B": 120, "72B": 118}
_sgdf = {}
for m in ("1.5B", "7B", "32B", "72B"):
    df = pd.read_parquet(SELFGEN[m])
    df["claim"] = df["prop_id"].map(_CLAIM)
    df["strict"] = df.apply(lambda r: strict_faithful(r["gen"], r["claim"]), axis=1)
    _sgdf[m] = df
    nf = int(df["strict"].sum())
    print(f"    {m}: strict-faithful {nf}/{len(df)}")
    chk_bool(f"{m} strict-faithful = {SG_N[m]}/120", nf == SG_N[m], f"{nf}")
    fa = df[df["strict"]]["h1_selfgen"].to_numpy()
    for style, fn in [("boot2000", lambda x: _boot_ci(x)), ("boot1500", ci1500)]:
        est, lo, hi = fn(fa)
        if m in SG_FAITH:
            print(f"      faithful [{style}] {est:+.3f} [{lo:+.3f},{hi:+.3f}]  target {SG_FAITH[m]}")
        else:
            print(f"      faithful [{style}] {est:+.3f} [{lo:+.3f},{hi:+.3f}]  (7B: paper -0.08 [-0.62,+0.44])")
    if m in SG_FAITH:
        chk(f"selfgen faithful {m} estimate", fa.mean(), SG_FAITH[m][0], 0.005)
    if m == "7B":
        chk("selfgen faithful 7B estimate (null)", fa.mean(), -0.08, 0.005)
    tf = df["h1_tf"].to_numpy()
    if m in SG_TF:
        chk(f"selfgen run-matched TF {m}", tf.mean(), SG_TF[m], 0.005)
    if m == "1.5B":
        # negated subset: the generation introduces a negation the claim itself does not contain
        neg = df[~df["strict"] & df.apply(
            lambda r: (" not " in " " + r["gen"].lower() + " " or "cannot" in r["gen"].lower())
                      and " not " not in " " + r["claim"].lower() + " ", axis=1)]
        un = neg["h1_selfgen"].to_numpy()
        chk_bool("1.5B negated subset n=27", len(un) == 27, f"n={len(un)}")
        for style, fn in [("boot2000", lambda x: _boot_ci(x)), ("boot1500", ci1500)]:
            est, lo, hi = fn(un)
            print(f"      1.5B negated [{style}] {est:+.3f} [{lo:+.3f},{hi:+.3f}]  paper -0.93 [-1.63,-0.31]")
        chk("1.5B negated-subset estimate", un.mean(), -0.93, 0.005)
    if m == "72B":
        chk_bool("72B non-verbatim n=2 (too few to read)", int((~df["strict"]).sum()) == 2,
                 f"n={int((~df['strict']).sum())}")
    # status strings in faithful self-generated calls
    gens = df[df["strict"]]["gen"].str.lower()
    n_estab = int(gens.str.contains('status="established"').sum())
    n_conf = int(gens.str.contains("confirmed and verified").sum())
    print(f"      status counts (faithful): established={n_estab}, confirmed-and-verified={n_conf}, other={nf - n_estab - n_conf}")

q32 = _sgdf["32B"][_sgdf["32B"]["strict"]]["h1_selfgen"].mean() / _sgdf["32B"]["h1_tf"].mean()
q72 = _sgdf["72B"][_sgdf["72B"]["strict"]]["h1_selfgen"].mean() / _sgdf["72B"]["h1_tf"].mean()
chk_bool("faithful/TF ratio 71-84% at 32B/72B", 0.705 <= q32 <= 0.845 and 0.705 <= q72 <= 0.845,
         f"32B={q32:.3f} 72B={q72:.3f}")

print()
print("=" * 100)
print("SECTION 7: \\S4.6 mechanism stratification")
print("=" * 100)
# uncertain = baseline (none) p_of_P in [0.2, 0.8]?  n should be 34 at 1.5B; 3 at 7B
for m in ("1.5B", "7B", "Mistral"):
    df = gate_df[m]
    wp = _pivot(df, value="p_of_P")
    wl = _pivot(df, value="logodds_P")
    false_ids = df[df.gold == False]["prop_id"].unique()
    fp = wp[wp.index.isin(false_ids)]
    fl = wl[wl.index.isin(false_ids)]
    unc = fp[(fp["none"] >= 0.2) & (fp["none"] <= 0.8)].index
    sat = fp[(fp["none"] < 0.2)].index
    d_unc = (fl.loc[unc, "commit3"] - fl.loc[unc, "none"]).to_numpy()
    d_sat = (fl.loc[sat, "commit3"] - fl.loc[sat, "none"]).to_numpy()
    est_u = ci1500(d_unc) if len(d_unc) > 1 else (np.nan,) * 3
    est_u2 = _boot_ci(d_unc) if len(d_unc) > 1 else (np.nan,) * 3
    est_s = _boot_ci(d_sat) if len(d_sat) > 1 else (np.nan,) * 3
    print(f"    {m}: uncertain-false n={len(unc)}  H1={est_u2[0]:+.3f} [{est_u2[1]:+.3f},{est_u2[2]:+.3f}] "
          f"(1500-boot [{est_u[1]:+.3f},{est_u[2]:+.3f}]) | saturated-false n={len(sat)} H1={est_s[0]:+.3f}")
    if m == "1.5B":
        chk_bool("1.5B uncertain-false n=13 (mid-band def)", len(unc) == 13, f"n={len(unc)}")
        chk("1.5B uncertain-false H1 = +1.05 [+0.65,+1.42]", est_u2, (1.05, 0.65, 1.42), 0.0051)
    if m == "7B":
        chk_bool("7B uncertain-false = 3/54", len(unc) == 3, f"n={len(unc)}/{len(fp)}")
    if m == "Mistral":
        chk("Mistral saturated-false H1 = +2.24", d_sat.mean() if len(d_sat) else np.nan, 2.24, 0.005)

print()
print("=" * 100)
print("SECTION 8: \\S4.7 downstream harm (gate accuracy/flip) + Fig 2 + Fig 4")
print("=" * 100)
HARM_HARD = {  # model: (acc_none, acc_commit3, flip)
    "1.5B": (0.52, None, 0.73), "7B": (0.95, None, 0.05),
    "32B": (0.98, None, 0.11), "72B": (0.59, 0.14, 0.76),
}
for m, (a0, a1, fl) in HARM_HARD.items():
    h = accuracy_harm(pd.read_parquet(GATE_HARD[m]))
    print(f"    {m} arith_hard: none={h['acc_none']:.3f} commit3={h['acc_commit3']:.3f} flip={h['flip_rate']:.3f} n_false={h['n_false']}")
    chk(f"arith_hard baseline acc {m}", h["acc_none"], a0, 0.005)
    chk(f"arith_hard flip {m}", h["flip_rate"], fl, 0.005)
    if a1 is not None:
        chk(f"arith_hard commit acc {m}", h["acc_commit3"], a1, 0.005)

# §4.7 text quotes the 72B flip with a per-item bootstrap 95% CI and denominator: 76% [62,88] (26/34)
h72h = accuracy_harm(pd.read_parquet(GATE_HARD["72B"]))
chk("72B hard flip CI [0.62,0.88]",
    (h72h["flip_rate_ci"]["ci_low"], h72h["flip_rate_ci"]["ci_high"]), (0.618, 0.882), 0.005)
chk("72B hard flip counts 26/34",
    (round(h72h["flip_rate"] * h72h["flip_denom"]), h72h["flip_denom"]), (26, 34), 0)

# 7B/32B factual decisions
h7 = accuracy_harm(gate_df["7B"])
chk("7B factual acc none=0.93", h7["acc_none"], 0.93, 0.005)
chk("7B factual acc commit3=0.93", h7["acc_commit3"], 0.93, 0.005)
h32 = accuracy_harm(gate_df["32B"])
chk("32B factual flip=0.02", h32["flip_rate"], 0.02, 0.005)

# Fig 2 bars: 7B gold-false H1 on factual / arith_props / arith_hard
g7f = belief_shift_by_gold(gate_df["7B"], value="logodds_P")["false"]["h1"]
chk("Fig2 factual bar (7B false H1)", tup(g7f), (0.242, -0.192, 0.738), 0.005)
g7e = belief_shift_by_gold(pd.read_parquet(GATE_EASY["7B"]), value="logodds_P")["false"]["h1"]
chk("Fig2 easy-arith bar / text +0.48 [+0.14,+0.82]", tup(g7e), (0.479, 0.140, 0.824), 0.005)
g7h = belief_shift_by_gold(pd.read_parquet(GATE_HARD["7B"]), value="logodds_P")["false"]["h1"]
chk("Fig2 hard bar / text +2.19 [+1.67,+2.72] n=58", tup(g7h), (2.193, 1.667, 2.722), 0.005)

# Fig 4 lines: flips for 32B/72B on factual / easy / hard
for m, want in [("32B", (0.02, 0.02, 0.11)), ("72B", (0.04, 0.30, 0.76))]:
    f_fact = accuracy_harm(gate_df[m])["flip_rate"]
    f_easy = accuracy_harm(pd.read_parquet(GATE_EASY[m]))["flip_rate"]
    f_hard = accuracy_harm(pd.read_parquet(GATE_HARD[m]))["flip_rate"]
    chk(f"Fig4 {m} flips (factual, easy, hard)", (f_fact, f_easy, f_hard), want, 0.005)

# Fig 4 (paper Fig 5) Wilson error bars: flip counts k/n quoted in the figure and its caption
for m, want in [("32B", (1, 53, 2, 93, 6, 57)), ("72B", (2, 53, 27, 91, 26, 34))]:
    got = []
    for d in (gate_df[m], pd.read_parquet(GATE_EASY[m]), pd.read_parquet(GATE_HARD[m])):
        h = accuracy_harm(d)
        got += [round(h["flip_rate"] * h["flip_denom"]), h["flip_denom"]]
    chk(f"Fig4 {m} flip counts k/n (factual, easy, hard)", tuple(got), want, 0)

print()
print("=" * 100)
print("SECTION 9: \\S4.8 mitigation + Fig 5")
print("=" * 100)
FIG5_BELIEF = {"1.5B": [0.86, -4.65, -3.63, -1.01], "7B": [3.23, 0.48, -0.23, 0.51],
               "32B": [2.10, -1.19, -1.61, -1.50], "72B": [6.02, -0.90, -0.37, -0.25]}
FIG5_FLIP = {"1.5B": [0.53, 0.0, 0.0, 0.0], "7B": [0.07, 0.0, 0.0, 0.0],
             "32B": [0.14, 0.0, 0.0, 0.0], "72B": [0.68, 0.03, 0.08, 0.05]}
ORDER = ("commit3", "mit_disclaim", "mit_reframe", "mit_isolate")
for m in ("1.5B", "7B", "32B", "72B"):
    rep = mitigation_report(pd.read_parquet(MIT[m]), value="logodds_P")
    beliefs = [rep[k]["delta_vs_none"]["estimate"] for k in ORDER]
    flips = [rep[k]["flip_rate"] for k in ORDER]
    chk(f"Fig5 belief row {m}", beliefs, FIG5_BELIEF[m], 0.005)
    chk(f"Fig5 flip row {m}", flips, FIG5_FLIP[m], 0.005)
    accs = {k: rep[k]["acc_false"] for k in ORDER}
    print(f"    {m}: none_acc={rep['_none_acc_false']:.3f}  accs={ {k: round(v,3) for k,v in accs.items()} }")
    if m == "1.5B":
        chk("1.5B mitigation baseline acc 0.52->0.24", (rep["_none_acc_false"], accs["commit3"]), (0.52, 0.24), 0.005)
        c = rep["commit3"]["flip_rate_ci"]
        chk("1.5B mitigation flip CI [0.37,0.70] (text: 53% [37,70])", (c["ci_low"], c["ci_high"]), (0.367, 0.700), 0.005)
    if m == "7B":
        d = rep["mit_disclaim"]["delta_vs_none"]; i = rep["mit_isolate"]["delta_vs_none"]
        chk("7B disclaim residual +0.48 [+0.01,+0.94]", tup(d), (0.48, 0.01, 0.94), 0.005)
        chk("7B isolate residual +0.51 [+0.28,+0.73]", tup(i), (0.51, 0.28, 0.73), 0.005)
    if m == "72B":
        chk("72B mitigation baseline acc 0.66->0.21", (rep["_none_acc_false"], accs["commit3"]), (0.66, 0.21), 0.005)
        c = rep["commit3"]["flip_rate_ci"]
        chk("72B mitigation flip CI [0.53,0.82] (text: 68% [53,82])", (c["ci_low"], c["ci_high"]), (0.526, 0.816), 0.005)
        fix_los = [rep[k]["flip_rate_ci"]["ci_low"] for k in ORDER[1:]]
        fix_his = [rep[k]["flip_rate_ci"]["ci_high"] for k in ORDER[1:]]
        chk("72B fixes flip CI lows all 0 (text: CIs within [0,16])", max(fix_los), 0.0, 1e-9)
        chk("72B fixes flip CI max high 0.158 (text: CIs within [0,16])", max(fix_his), 0.158, 0.005)
        iso = rep["mit_isolate"]["delta_vs_none"]
        chk("72B isolate delta -0.25, CI incl 0", iso["estimate"], -0.25, 0.005)
        chk_bool("72B isolate CI includes 0", iso["ci_low"] <= 0 <= iso["ci_high"],
                 f"[{iso['ci_low']:+.3f},{iso['ci_high']:+.3f}]")

print()
print("=" * 100)
print("SECTION 10: cross-run reproducibility (< 0.025 log-odds claim)")
print("=" * 100)
KEYS5 = ("h1_masked_commit_delta", "h3_commit_vs_query", "h4_dose_slope", "h5_placebo", "h2_action_specific")
maxdiff = 0.0
if not all(pathlib.Path(q).exists() for q in OLD_SMALL.values()):
    # The cross-run agreement claim is a paper number; its inputs are committed provenance
    # artifacts, so their absence must fail the verification rather than silently skip.
    chk_bool("cross-run comparison parquets present (committed; see results/README.md)", False,
             "missing " + ", ".join(q for q in OLD_SMALL.values() if not pathlib.Path(q).exists()))
else:
    for m, path in OLD_SMALL.items():
        old = contrasts(pd.read_parquet(path), value="logodds_P")
        for k in KEYS5:
            d = abs(old[k]["estimate"] - gate_c[m][k]["estimate"])
            maxdiff = max(maxdiff, d)
            print(f"    {m} {k}: old={old[k]['estimate']:+.4f} new={gate_c[m][k]['estimate']:+.4f} |diff|={d:.4f}")
    chk_bool("1.5B/7B values reproduce across runs to within 0.03 (max |Δ| 0.025)",
             maxdiff < 0.03 and abs(maxdiff - 0.025) <= 0.0005, f"max diff={maxdiff:.4f}")

print()
print("=" * 100)
print("SECTION 11: fluency-matched richer-query + verbalized-confidence controls (per-item parquets)")
print("=" * 100)
FLU = {"1.5B": R + "fluency_matched_Qwen2.5-1.5B-Instruct.parquet",
       "7B": R + "fluency_matched_Qwen2.5-7B-Instruct.parquet",
       "Mistral": R + "fluency_matched_Mistral-7B-Instruct-v0.3.parquet"}
FLU_RICH_H3 = {"1.5B": 1.52, "Mistral": 3.31}   # \S4.4: commit3 - query_rich; 7B: CI includes 0
for m, path in FLU.items():
    if not pathlib.Path(path).exists():
        chk_bool(f"fluency control parquet present ({m})", False, f"missing {path}")
        continue
    d = pd.read_parquet(path)
    w = d.pivot_table(index="prop_id", columns="condition", values="logodds_P")
    rich = (w["commit3"] - w["query_rich"]).to_numpy()
    est, lo, hi = ci1500(rich)
    print(f"    {m}: H3 commit3-query_rich = {est:+.3f} [{lo:+.3f},{hi:+.3f}]")
    if m in FLU_RICH_H3:
        chk(f"fluency-matched H3 (rich) {m}", est, FLU_RICH_H3[m], 0.005)
        chk_bool(f"fluency-matched H3 (rich) {m} CI > 0", lo > 0, f"lo={lo:+.3f}")
    else:
        chk_bool("fluency-matched H3 (rich) 7B null (CI includes 0)", lo <= 0 <= hi,
                 f"[{lo:+.3f},{hi:+.3f}]")

VERB = R + "verbalized_Qwen2.5-1.5B-Instruct.parquet"
if not pathlib.Path(VERB).exists():
    chk_bool("verbalized control parquet present (1.5B)", False, f"missing {VERB}")
else:
    d = pd.read_parquet(VERB)
    wc = d.pivot_table(index="prop_id", columns="condition", values="verbalized_conf")
    wl = d.pivot_table(index="prop_id", columns="condition", values="logodds_P")
    vd = (wc["commit"] - wc["none"]).dropna().to_numpy()
    ve, vlo, vhi = ci1500(vd)
    te, tlo, thi = ci1500((wl["commit"] - wl["none"]).to_numpy())
    print(f"    1.5B verbalized H1 = {ve:+.2f} [{vlo:+.2f},{vhi:+.2f}] (n={len(vd)}); "
          f"token H1 same items = {te:+.2f} [{tlo:+.2f},{thi:+.2f}]")
    chk("verbalized H1 (1.5B) estimate", ve, -4.79, 0.005)
    chk_bool("verbalized H1 negative while token H1 positive (\\S3 opposite-sign claim)",
             vhi < 0 < tlo, f"verb hi={vhi:+.2f}, token lo={tlo:+.2f}")

print()
print("=" * 100)
print("SECTION 12: cross-turn grid (closed assistant turn, protocols A+B, per-item parquets)")
print("=" * 100)
# Recompute must reproduce the pinned numbers exactly (ci1500 mirrors crossturn_gate.ci:
# rng(0), 1500, percentile), so tolerances only absorb documentation rounding.
CT_FILES = {"1.5B": (R + "crossturn_gate_Qwen2.5-1.5B-Instruct.parquet", 1920),
            "7B": (R + "crossturn_gate_Qwen2.5-7B-Instruct.parquet", 1920),
            "Mistral": (R + "crossturn_gate_Mistral-7B-Instruct-v0.3.parquet", 1680),
            "32B": (R + "crossturn_gate_Qwen2.5-32B.parquet", 1920),
            "72B": (R + "crossturn_gate_Qwen2.5-72B.parquet", 1920)}
CT_H1 = {("1.5B", "A"): (0.775, 0.0005), ("1.5B", "B"): (4.083, 0.0005),
         ("7B", "A"): (-0.125, 0.0005), ("7B", "B"): (3.16, 0.005),
         ("Mistral", "A"): (2.92, 0.005), ("Mistral", "B"): (12.74, 0.005),
         ("32B", "A"): (3.700, 0.0005), ("32B", "B"): (4.227, 0.0005),
         ("72B", "A"): (2.198, 0.0005), ("72B", "B"): (4.132, 0.0005)}
# Placebo (H5) never approaches the commitment shift, but its small residual is
# sign-INCONSISTENT across cells; each cell is pinned with its CI position so the paper
# cannot round this into "placebo is null everywhere".
CT_H5 = {("1.5B", "A"): (0.164, "pos"), ("7B", "A"): (-0.661, "neg"),
         ("Mistral", "A"): (-1.349, "neg"), ("32B", "A"): (-1.419, "neg"),
         ("72B", "A"): (0.369, "pos"),
         ("1.5B", "B"): (-0.258, "neg"), ("7B", "B"): (-0.662, "neg"),
         ("Mistral", "B"): (0.092, "null"), ("32B", "B"): (1.276, "pos"),
         ("72B", "B"): (0.403, "null")}
ct_stats = {}
for m, (path, want_rows) in CT_FILES.items():
    if not pathlib.Path(path).exists():
        chk_bool(f"cross-turn parquet present ({m})", False, f"missing {path}")
        continue
    d = pd.read_parquet(path)
    chk(f"cross-turn rows ({m})", len(d), want_rows, 0)
    chk_bool(f"cross-turn protocol balance ({m})", d.groupby("protocol").size().nunique() == 1)
    chk_bool(f"cross-turn only the TC arm unmatched ({m})",
             bool((d.length_matched == (d.framing != "commitment_tc")).all()))
    for proto in ("A", "B"):
        w = d[d.protocol == proto].pivot_table(
            index="prop_id", columns=["condition", "framing", "dose"], values="logodds_P")
        none, c3 = w[("none", "na", 0)], w[("action", "commitment", 3)]
        st = {"H1": ci1500((c3 - none).to_numpy()),
              "H3": ci1500((c3 - w[("action", "query", 0)]).to_numpy()),
              "H5": ci1500((w[("action", "placebo", 0)] - none).to_numpy()),
              "H2": ci1500((c3 - w[("assertion", "na", 0)]).to_numpy())}
        if ("action", "commitment_tc", 3) in w.columns:
            tc = w[("action", "commitment_tc", 3)]
            st["TCn"] = ci1500((tc - none).to_numpy())
            st["TCa"] = ci1500((tc - w[("assertion", "na", 0)]).to_numpy())
        ct_stats[(m, proto)] = st
        print(f"    {m} {proto}: H1 {st['H1'][0]:+.3f} [{st['H1'][1]:+.3f},{st['H1'][2]:+.3f}] | "
              f"H5 {st['H5'][0]:+.3f} [{st['H5'][1]:+.3f},{st['H5'][2]:+.3f}] | "
              f"H3 {st['H3'][0]:+.3f}")
for (m, proto), (want, tol) in CT_H1.items():
    if (m, proto) not in ct_stats:
        continue
    est, lo, hi = ct_stats[(m, proto)]["H1"]
    chk(f"cross-turn H1 {m} protocol {proto}", est, want, tol)
    if (m, proto) == ("7B", "A"):
        chk_bool("cross-turn H1 7B-A null (CI includes 0)", lo <= 0 <= hi,
                 f"[{lo:+.3f},{hi:+.3f}]")
    else:
        chk_bool(f"cross-turn H1 {m}-{proto} CI > 0", lo > 0, f"lo={lo:+.3f}")
if ("1.5B", "A") in ct_stats:
    chk("cross-turn placebo 1.5B-A (cross-validated vs independent implementation)",
        ct_stats[("1.5B", "A")]["H5"][0], 0.164, 0.0005)
for (m, proto), (want, sign) in CT_H5.items():
    if (m, proto) not in ct_stats:
        continue
    pest, plo, phi = ct_stats[(m, proto)]["H5"]
    chk(f"cross-turn placebo {m} protocol {proto}", pest, want, 0.0005)
    ok = {"neg": phi < 0, "pos": plo > 0, "null": plo <= 0 <= phi}[sign]
    chk_bool(f"cross-turn placebo CI {sign} ({m}-{proto})", ok, f"[{plo:+.3f},{phi:+.3f}]")
    h1est = ct_stats[(m, proto)]["H1"][0]
    if (m, proto) != ("7B", "A"):
        chk_bool(f"cross-turn commit exceeds placebo ({m}-{proto})", h1est > pest,
                 f"H1={h1est:+.3f} vs H5={pest:+.3f}")
for m in CT_FILES:
    if (m, "B") not in ct_stats:
        continue
    chk_bool(f"cross-turn H3 positive in act-first order ({m}-B)",
             ct_stats[(m, "B")]["H3"][1] > 0, f"lo={ct_stats[(m, 'B')]['H3'][1]:+.3f}")
chk_bool("cross-turn TC arm absent for Mistral (template has no native tool rendering)",
         "TCn" not in ct_stats.get(("Mistral", "A"), {}))
# Plain-text H2 stays <=0 probe-first everywhere but flips positive act-first at 32B/72B;
# the structured arm out-conditions the assertion in EVERY act-first cell. Both facts feed
# \S cross-turn, so every cell is pinned with its CI position.
CT_H2 = {("1.5B", "A"): (-0.604, "neg"), ("7B", "A"): (-3.814, "neg"),
         ("Mistral", "A"): (-4.898, "neg"), ("32B", "A"): (-0.299, "null"),
         ("72B", "A"): (-0.244, "neg"),
         ("1.5B", "B"): (0.050, "null"), ("7B", "B"): (-0.812, "neg"),
         ("Mistral", "B"): (0.334, "null"), ("32B", "B"): (3.069, "pos"),
         ("72B", "B"): (1.189, "pos")}
for (m, proto), (want, sign) in CT_H2.items():
    if (m, proto) not in ct_stats:
        continue
    hest, hlo, hhi = ct_stats[(m, proto)]["H2"]
    chk(f"cross-turn text H2 {m} protocol {proto}", hest, want, 0.0005)
    ok = {"neg": hhi < 0, "pos": hlo > 0, "null": hlo <= 0 <= hhi}[sign]
    chk_bool(f"cross-turn text H2 CI {sign} ({m}-{proto})", ok, f"[{hlo:+.3f},{hhi:+.3f}]")
# Structured-vs-text is scale-by-order dependent: positive in every act-first cell, but
# negative-or-null probe-first except 32B. Pinned with CI endpoints (TC arm caveat:
# not token-length-matched).
CT_TCA = {("1.5B", "A"): (-0.407, -0.514, -0.302), ("1.5B", "B"): (0.93, 0.70, 1.17),
          ("7B", "A"): (-3.689, -4.525, -2.888), ("7B", "B"): (0.99, 0.56, 1.47),
          ("32B", "A"): (1.191, 0.523, 1.781), ("32B", "B"): (6.080, 5.263, 6.987),
          ("72B", "A"): (-1.312, -1.620, -1.044), ("72B", "B"): (1.545, 1.152, 1.983)}
for (m, proto), (want, wlo, whi) in CT_TCA.items():
    if (m, proto) not in ct_stats:
        continue
    est, lo, hi = ct_stats[(m, proto)]["TCa"]
    chk(f"cross-turn TC-assertion ({m}-{proto})", est, want, 0.005)
    chk(f"cross-turn TC-assertion ({m}-{proto}) CI lo", lo, wlo, 0.005)
    chk(f"cross-turn TC-assertion ({m}-{proto}) CI hi", hi, whi, 0.005)
CT_TCN = {("1.5B", "A"): 0.972, ("1.5B", "B"): 4.96,
          ("7B", "A"): -0.001, ("32B", "A"): 5.190, ("32B", "B"): 7.239,
          ("72B", "A"): 1.129, ("72B", "B"): 4.489}
for (m, proto), want in CT_TCN.items():
    if (m, proto) not in ct_stats:
        continue
    chk(f"cross-turn TC-none ({m}-{proto})", ct_stats[(m, proto)]["TCn"][0], want, 0.005)

print()
print("=" * 100)
print("SECTION 13: cross-turn downstream harm (three-turn episodes, per-item parquets)")
print("=" * 100)
# Decision-level harm concentrates in small models, any corrective evidence restores
# accuracy, and 32B/72B sit at ceiling on this probe -- the belief shift (SECTION 12)
# grows with scale while THIS harm probe saturates away.
DH_FILES = {"1.5B": R + "downstream_harm_Qwen2.5-1.5B-Instruct.parquet",
            "7B": R + "downstream_harm_Qwen2.5-7B-Instruct.parquet",
            "Mistral": R + "downstream_harm_Mistral-7B-Instruct-v0.3.parquet",
            "32B": R + "downstream_harm_Qwen2.5-32B.parquet",
            "72B": R + "downstream_harm_Qwen2.5-72B.parquet"}
DH_FALSE_DACC = {  # (commit@none, commit@weak, assertion@none, assertion@weak) on FALSE claims
    "1.5B": (-0.30, 0.00, -0.37, 0.00), "7B": (0.00, 0.02, -0.07, 0.02),
    "Mistral": (-0.07, 0.00, -0.28, 0.00), "32B": (0.00, 0.00, 0.00, 0.00),
    "72B": (-0.02, 0.00, -0.02, 0.00)}
DH_FALSE_BASE = {"1.5B": 0.83, "7B": 0.93, "Mistral": 0.87, "32B": 0.98, "72B": 0.98}
DH_TRUE_DACC = {"32B": (0.05, 0.02, 0.08, 0.00), "72B": (0.08, 0.00, 0.09, 0.00)}
for m, path in DH_FILES.items():
    if not pathlib.Path(path).exists():
        chk_bool(f"downstream parquet present ({m})", False, f"missing {path}")
        continue
    d = pd.read_parquet(path)
    chk(f"downstream rows ({m})", len(d), 1440, 0)
    chk_bool(f"downstream correct column consistent ({m})",
             bool((d.correct == ((d.logodds_post >= 0) == d.gold)).all()))
    fa = d[~d.gold]
    chk(f"downstream FALSE items ({m})", fa.prop_id.nunique(), 54, 0)
    acc = fa.groupby(["condition", "evidence"]).correct.mean()
    print(f"    {m} FALSE: none@none={acc[('none', 'none')]:.3f} "
          f"commit@none={acc[('commit', 'none')]:.3f} "
          f"assertion@none={acc[('assertion', 'none')]:.3f}")
    chk(f"downstream FALSE baseline acc ({m})", acc[("none", "none")],
        DH_FALSE_BASE[m], 0.005)
    for label, got, want in [
            ("commit@none", acc[("commit", "none")] - acc[("none", "none")],
             DH_FALSE_DACC[m][0]),
            ("commit@weak", acc[("commit", "weak")] - acc[("none", "weak")],
             DH_FALSE_DACC[m][1]),
            ("assertion@none", acc[("assertion", "none")] - acc[("none", "none")],
             DH_FALSE_DACC[m][2]),
            ("assertion@weak", acc[("assertion", "weak")] - acc[("none", "weak")],
             DH_FALSE_DACC[m][3])]:
        chk(f"downstream FALSE dAcc {label} ({m})", got, want, 0.005)
    chk_bool(f"downstream strong evidence restores FALSE acc to 1.00 ({m})",
             bool((fa[fa.evidence == "strong"].groupby("condition").correct.mean()
                   == 1.0).all()))
    if m in DH_TRUE_DACC:
        tr = d[d.gold]
        tacc = tr.groupby(["condition", "evidence"]).correct.mean()
        for label, got, want in [
                ("commit@none", tacc[("commit", "none")] - tacc[("none", "none")],
                 DH_TRUE_DACC[m][0]),
                ("commit@weak", tacc[("commit", "weak")] - tacc[("none", "weak")],
                 DH_TRUE_DACC[m][1]),
                ("assertion@none", tacc[("assertion", "none")] - tacc[("none", "none")],
                 DH_TRUE_DACC[m][2]),
                ("assertion@weak", tacc[("assertion", "weak")] - tacc[("none", "weak")],
                 DH_TRUE_DACC[m][3])]:
            chk(f"downstream TRUE dAcc {label} ({m})", got, want, 0.005)

print()
print("=" * 100)
print("SECTION 14: rented-box confirmation runs (sampled readout, cross-turn mitigation, arith recovery)")
print("=" * 100)
print("-- 14A: hard-arithmetic recovery arm (72B; App. arithrec) --")
d = pd.read_parquet(R + "downstream_harm_Qwen2.5-72B_arith_hard.parquet")
chk("arithrec rows", len(d), 1200, 0)
chk_bool("arithrec correct column consistent",
         bool((d.correct == ((d.logodds_post >= 0) == d.gold)).all()))
fa, tr = d[~d.gold], d[d.gold]
chk("arithrec n false / true", (fa.prop_id.nunique(), tr.prop_id.nunique()), (58, 42), 0)
accF = fa.groupby(["condition", "evidence"]).correct.mean()
accT = tr.groupby(["condition", "evidence"]).correct.mean()
chk("arithrec FALSE no-evidence acc row (none, query, commit, assertion)",
    (accF[("none", "none")], accF[("query", "none")], accF[("commit", "none")],
     accF[("assertion", "none")]),
    (0.845, 0.672, 0.483, 0.224), 0.005)
chk("arithrec FALSE dAcc assertion@none = -0.62",
    accF[("assertion", "none")] - accF[("none", "none")], -0.621, 0.005)
chk("arithrec FALSE dAcc commit@none = -0.36",
    accF[("commit", "none")] - accF[("none", "none")], -0.362, 0.005)
chk_bool("arithrec FALSE weak+strong restore ALL conditions to 1.00",
         all(float(accF[(c, e)]) == 1.0
             for c in ("none", "query", "commit", "assertion") for e in ("weak", "strong")))
chk("arithrec TRUE none row (none, weak, strong)",
    (accT[("none", "none")], accT[("none", "weak")], accT[("none", "strong")]),
    (0.571, 0.976, 0.976), 0.005)
chk("arithrec TRUE query/commit @none (0.69, 0.90)",
    (accT[("query", "none")], accT[("commit", "none")]), (0.690, 0.905), 0.005)
chk("arithrec TRUE dAcc assertion@none = +0.43",
    accT[("assertion", "none")] - accT[("none", "none")], 0.429, 0.005)
chk("arithrec TRUE dAcc commit@none = +0.33",
    accT[("commit", "none")] - accT[("none", "none")], 0.333, 0.005)
chk_bool("arithrec TRUE assertion row all 1.00",
         all(float(accT[("assertion", e)]) == 1.0 for e in ("none", "weak", "strong")))

print("-- 14B: sampled-decoding validation (App. sampledread) --")
import importlib.util as _ilu
def _load_script(name):  # scripts/ is not a package; load by path like tests/test_confirm_summaries.py
    spec = _ilu.spec_from_file_location(name, pathlib.Path("scripts") / f"{name}.py")
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
_sr = _load_script("sampled_readout")
SR_WANT = {
    "72B": {"none": (0.580, 0.580, 0.575), "commit3": (0.920, 0.920, 0.914),
            "flip_n": 34, "forced": (0.7647, 0.6176, 0.8824), "greedy": (0.7647, 0.6176, 0.8824),
            "sampled": (0.7529, 0.6206, 0.8691)},
    "7B":  {"none": (0.130, 0.130, 0.127), "commit3": (0.190, 0.190, 0.192),
            "flip_n": 55, "forced": (0.0545, 0.0, 0.1273), "greedy": (0.0545, 0.0, 0.1273),
            "sampled": (0.0518, 0.0150, 0.0982)},
}
for m, w in SR_WANT.items():
    df = pd.read_parquet(R + f"sampledread_Qwen2.5-{m}_arith_hard.parquet")
    r = _sr.summarize(df)
    chk(f"sampledread {m} unparsable rate = 0", r["other_rate"], 0.0, 0)
    for cell in ("none", "commit3"):
        c = r["cells"][cell]
        chk(f"sampledread {m} {cell} True-rates (forced, greedy, sampled)",
            (c["forced_true"], c["greedy_true"], c["sampled_true"]), w[cell], 0.005)
        chk(f"sampledread {m} {cell} forced-greedy agreement 100/100",
            (c["agree_ci"][0], c["agree_ci"][1], c["agree_ci"][2], c["agree_n"]),
            (1.0, 1.0, 1.0, 100), 0)
    fl = r["flip"]
    chk(f"sampledread {m} flip n", fl["n"], w["flip_n"], 0)
    chk(f"sampledread {m} flip forced (72B == 26/34 headline)", fl["forced"], w["forced"], 0.005)
    chk(f"sampledread {m} flip greedy", fl["greedy"], w["greedy"], 0.005)
    chk(f"sampledread {m} flip sampled", fl["sampled"], w["sampled"], 0.005)

print("-- 14C: cross-turn mitigation grid (App. ctmit) --")
_cm = _load_script("crossturn_mitigation")
CT_FLIP = {  # (model, proto): (n_base, commit3, disclaim, reframe, isolate) flip (est, lo, hi)
    ("32B", "A"): (58, (0.1379, 0.0599, 0.2241), (0.0172, 0.0, 0.0517),
                   (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ("32B", "B"): (56, (0.7143, 0.5893, 0.8393), (0.4643, 0.3214, 0.5893),
                   (0.1071, 0.0357, 0.1964), (0.0, 0.0, 0.0)),
    ("72B", "A"): (41, (0.5366, 0.3902, 0.6829), (0.0976, 0.0244, 0.1951),
                   (0.1463, 0.0488, 0.2683), (0.0, 0.0, 0.0)),
    ("72B", "B"): (46, (0.5217, 0.3696, 0.6522), (0.1304, 0.0435, 0.2391),
                   (0.2391, 0.1304, 0.3696), (0.1522, 0.0652, 0.2609)),
}
CT_DELTA = {  # (commit3, isolate) belief delta vs none, per arith cell (text quote)
    ("32B", "A"): (5.218, 0.873), ("32B", "B"): (12.330, 0.001),
    ("72B", "A"): (3.511, -0.180), ("72B", "B"): (3.536, 0.389),
}
a_c3, a_dis, a_ref, a_iso = [], [], [], []
for (m, proto), (nb, c3, dis, ref, iso) in CT_FLIP.items():
    df = pd.read_parquet(R + f"crossturn_mit_Qwen2.5-{m}_arith_hard.parquet")
    s = _cm.summarize(df)[proto]
    chk(f"ctmit arith {m}-{proto} n_base", s["n_base"], nb, 0)
    for name, w in [("commit3", c3), ("mit_disclaim", dis), ("mit_reframe", ref),
                    ("mit_isolate", iso)]:
        chk(f"ctmit arith {m}-{proto} {name} flip", s["cells"][name]["flip_ci"], w, 0.005)
    chk(f"ctmit arith {m}-{proto} commit3/isolate belief delta",
        (s["cells"]["commit3"]["delta_ci"][0], s["cells"]["mit_isolate"]["delta_ci"][0]),
        CT_DELTA[(m, proto)], 0.005)
    a_c3.append(s["cells"]["commit3"]["flip_ci"][0])
    a_dis.append(s["cells"]["mit_disclaim"]["flip_ci"][0])
    a_ref.append(s["cells"]["mit_reframe"]["flip_ci"][0])
    a_iso.append(s["cells"]["mit_isolate"]["flip_ci"][0])
chk_bool("ctmit arith commit3: 3 cells 52-71%, fourth ~14%",
         sum(0.52 <= f <= 0.72 for f in a_c3) == 3 and abs(min(a_c3) - 0.138) <= 0.005,
         f"{[round(f, 3) for f in a_c3]}")
chk_bool("ctmit arith isolate: three 0%, fourth ~15%",
         sum(f == 0.0 for f in a_iso) == 3 and abs(max(a_iso) - 0.152) <= 0.005,
         f"{[round(f, 3) for f in a_iso]}")
chk_bool("ctmit arith worst cells: disclaim 46% > reframe 24% > isolate 15%",
         abs(max(a_dis) - 0.464) <= 0.005 and abs(max(a_ref) - 0.239) <= 0.005
         and abs(max(a_iso) - 0.152) <= 0.005,
         f"dis={max(a_dis):.3f} ref={max(a_ref):.3f} iso={max(a_iso):.3f}")
p_c3, p_mit = [], []
for m in ("32B", "72B"):
    df = pd.read_parquet(R + f"crossturn_mit_Qwen2.5-{m}_propositions.parquet")
    r = _cm.summarize(df)
    for proto in ("A", "B"):
        s = r[proto]
        p_c3.append(s["cells"]["commit3"]["flip_ci"][0])
        p_mit += [s["cells"][k]["flip_ci"][0]
                  for k in ("mit_disclaim", "mit_reframe", "mit_isolate")]
chk("ctmit factual commit3 flips (4 cells)", tuple(p_c3),
    (0.0192, 0.0943, 0.0377, 0.0755), 0.005)
chk_bool("ctmit factual commit3 flips within 2-9%",
         all(0.019 <= f <= 0.095 for f in p_c3), f"{[round(f, 4) for f in p_c3]}")
chk_bool("ctmit factual every mitigation flip <= 2%", max(p_mit) <= 0.02,
         f"max={max(p_mit):.4f}")

print()
print("=" * 100)
print("SECTION 15: Llama-3.1-8B (third family), six-model Bonferroni (k=24), two-sided mitigation")
print("=" * 100)
from actionlockin.analyze import benign_true_report
LL = {s: pd.read_parquet(R + f"gate_Llama-3.1-8B_{s}.parquet")
      for s in ("propositions", "arith_props", "arith_hard", "arith_hard_llama")}
c = contrasts(LL["propositions"], value="logodds_P")
chk("Table1 Llama (H1,H3,H4,H5,H2)",
    tuple(c[k]["estimate"] for k in KEYS5[:4]) + (c["h2_action_specific"]["estimate"],),
    (0.96, 0.55, 0.30, -0.01, -3.94), 0.0051)
w = _pivot(LL["propositions"], value="logodds_P")
d = (w["commit3"] - w["none"]).to_numpy()
chk_bool("Llama none-baseline SD ~ 5.4", abs(np.std(w["none"].to_numpy(), ddof=1) - 5.4) <= 0.05)
chk_bool("Cohen's d Llama = 1.05", abs(d.mean() / np.std(d) - 1.05) <= 0.005)
chk("Llama placebo-none on FALSE claims +0.46 [0.33,0.59]",
    paired(LL["propositions"], "placebo", "none", gold=False), (0.46, 0.33, 0.59), 0.006)
bg = belief_shift_by_gold(LL["propositions"], value="logodds_P")
chk("Llama H1 by gold (true, false)", (bg["true"]["h1"]["estimate"], bg["false"]["h1"]["estimate"]),
    (1.33, 0.51), 0.005)
chk("Llama mixed-arith H1", contrasts(LL["arith_props"], value="logodds_P")["h1_masked_commit_delta"]["estimate"],
    0.45, 0.005)
chk("Llama hard-arith H1 (main grid)",
    contrasts(LL["arith_hard"], value="logodds_P")["h1_masked_commit_delta"]["estimate"], -0.38, 0.005)
cm = contrasts(LL["arith_hard_llama"], value="logodds_P")
chk("Llama-matched H1 CI", tup(cm["h1_masked_commit_delta"]), (1.84, 1.75, 1.94), 0.006)
chk("Llama-matched (H3, H4)", (cm["h3_commit_vs_query"]["estimate"], cm["h4_dose_slope"]["estimate"]),
    (1.28, 0.77), 0.005)
chk("Llama-matched commit3-placebo", paired(LL["arith_hard_llama"], "commit3", "placebo"),
    (1.13, 1.04, 1.22), 0.006)
QWEN_ARITH_H1 = {"arith_props": (0.32, 0.83, 1.61, 6.10), "arith_hard": (1.15, 2.42, 1.53, 7.27)}
for s, want in QWEN_ARITH_H1.items():
    got = [contrasts(pd.read_parquet(R + f"gate_Qwen2.5-{m}_{s}.parquet"),
                     value="logodds_P")["h1_masked_commit_delta"]["estimate"]
           for m in ("1.5B", "7B", "32B", "72B")]
    chk(f"Qwen {s} H1 ladder", got, want, 0.005)

# Table 3 bootstrap CIs (lo, hi) per model, Qwen 1.5B/7B/32B/72B then Llama-8B
T3_CI = {"arith_props": [(0.25, 0.40), (0.54, 1.16), (1.19, 2.05), (5.58, 6.62), (0.33, 0.58)],
         "arith_hard": [(1.09, 1.20), (2.05, 2.79), (1.19, 1.87), (6.93, 7.60), (-0.46, -0.31)]}
for s, want in T3_CI.items():
    dfs = [pd.read_parquet(R + f"gate_Qwen2.5-{m}_{s}.parquet") for m in ("1.5B", "7B", "32B", "72B")] + [LL[s]]
    for (lo, hi), d, m in zip(want, dfs, ("1.5B", "7B", "32B", "72B", "Llama")):
        h = contrasts(d, value="logodds_P")["h1_masked_commit_delta"]
        chk(f"Table 3 {s} {m} H1 CI", (h["ci_low"], h["ci_high"]), (lo, hi), 0.006)

# Six-model Bonferroni family: k = 4 contrasts x 6 models = 24, two-sided 99.79% intervals.
LOq, HIq = 100 * 0.05 / 24 / 2, 100 - 100 * 0.05 / 24 / 2
fam = dict(gate_df, Llama=LL["propositions"])
lo24, hi24, spread24 = {}, {}, 0.0
for m, df in fam.items():
    w = _pivot(df, value="logodds_P")
    for lab, x, side in [("H1", w["commit3"] - w["none"], "lo"), ("H3", w["commit3"] - w["query"], "lo"),
                         ("H4", dose_slopes(df), "lo"), ("H5", w["placebo"] - w["none"], "hi")]:
        _, los, his = boot_bounds(np.asarray(x, float))
        b = los if side == "lo" else his
        spread24 = max(spread24, max(b) - min(b))
        lo24[(m, lab)], hi24[(m, lab)] = min(los), max(his)
dec = {k: v for k, v in lo24.items() if k[1] != "H5"}
chk_bool("k=24: all 18 H1/H3/H4 corrected lower bounds > 0", all(v > 0 for v in dec.values()))
chk("k=24 smallest bound overall = +0.23", min(dec.values()), 0.23, 0.0051)
chk_bool("k=24 smallest bounds are the 1.5B and Llama dose slopes",
         sorted(dec, key=dec.get)[:2] == [("Llama", "H4"), ("1.5B", "H4")]
         or sorted(dec, key=dec.get)[:2] == [("1.5B", "H4"), ("Llama", "H4")])
chk("k=24 smallest H1 bound = +0.34 (7B)", min(v for k, v in dec.items() if k[1] == "H1"), 0.34, 0.0051)
chk("k=24 H5 corrected CI 72B", (lo24[("72B", "H5")], hi24[("72B", "H5")]), (-0.89, 0.89), 0.006)
chk("k=24 H5 corrected CI Llama", (lo24[("Llama", "H5")], hi24[("Llama", "H5")]), (-0.20, 0.19), 0.006)
chk_bool("k=24 decision bounds vary < 0.07 across seeds", spread24 < 0.07, f"max spread={spread24:.4f}")

# Decision harm with per-item bootstrap CIs (Table harm).
HARM_CI = {"gate_Qwen2.5-1.5B_arith_hard": (0.73, 0.57, 0.90), "gate_Qwen2.5-7B_arith_hard": (0.05, 0.00, 0.13),
           "gate_Qwen2.5-32B_arith_hard": (0.11, 0.04, 0.19), "gate_Llama-3.1-8B_arith_hard": (0.0, 0.0, 0.0),
           "gate_Llama-3.1-8B_arith_hard_llama": (0.36, 0.22, 0.53)}
for f, want in HARM_CI.items():
    h = accuracy_harm(pd.read_parquet(R + f + ".parquet"))
    chk(f"harm flip CI {f}", (h["flip_rate"],) + tuple(h["flip_rate_ci"][k] for k in ("ci_low", "ci_high")),
        want, 0.006)
h = accuracy_harm(LL["arith_hard_llama"])
chk("Llama-matched false acc none->commit3, flips 13/36",
    (h["acc_none"], h["acc_commit3"], round(h["flip_rate"] * h["flip_denom"]), h["flip_denom"]),
    (0.72, 0.46, 13, 36), 0.005)

# Two-sided mitigation: false-claim flips and true-claim downgrades (base-correct true claims pushed
# below 0.5). (flip, flip_lo, flip_hi, downgrade, down_lo, down_hi, down_denom) per representation.
MIT2 = {
    "Qwen2.5-1.5B_arith_hard": {"commit3": (0.53, 0.37, 0.70, 0.0, 0.0, 0.0, 20),
                                "mit_disclaim": (0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 20),
                                "mit_reframe": (0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 20),
                                "mit_isolate": (0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 20)},
    "Qwen2.5-7B_arith_hard": {"commit3": (0.07, 0.02, 0.14, 0.0, 0.0, 0.0, 7),
                              "mit_disclaim": (0.0, 0.0, 0.0, 0.86, 0.57, 1.0, 7),
                              "mit_reframe": (0.0, 0.0, 0.0, 0.14, 0.0, 0.43, 7),
                              "mit_isolate": (0.0, 0.0, 0.0, 0.57, 0.29, 0.86, 7)},
    "Qwen2.5-32B_arith_hard": {"commit3": (0.14, 0.05, 0.22, 0.0, 0.0, 0.0, 6),
                               "mit_disclaim": (0.0, 0.0, 0.0, 0.67, 0.33, 1.0, 6),
                               "mit_reframe": (0.0, 0.0, 0.0, 0.67, 0.33, 1.0, 6),
                               "mit_isolate": (0.0, 0.0, 0.0, 0.67, 0.33, 1.0, 6)},
    "Qwen2.5-72B_arith_hard": {"commit3": (0.68, 0.53, 0.82, 0.0, 0.0, 0.0, 31),
                               "mit_disclaim": (0.03, 0.0, 0.08, 0.19, 0.06, 0.32, 31),
                               "mit_reframe": (0.08, 0.0, 0.16, 0.10, 0.0, 0.23, 31),
                               "mit_isolate": (0.05, 0.0, 0.13, 0.10, 0.0, 0.23, 31)},
    "Llama-3.1-8B_arith_hard_llama": {"commit3": (0.34, 0.18, 0.50, 0.0, 0.0, 0.0, 16),
                                      "mit_disclaim": (0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 16),
                                      "mit_reframe": (0.13, 0.03, 0.26, 0.0, 0.0, 0.0, 16),
                                      "mit_isolate": (0.21, 0.08, 0.34, 0.0, 0.0, 0.0, 16)},
}
for f, rows in MIT2.items():
    df = pd.read_parquet(R + f"mitigation_{f}.parquet")
    mf, bt = mitigation_report(df), benign_true_report(df)
    for k, want in rows.items():
        a, b = mf[k], bt[k]
        chk(f"two-sided mitigation {f} {k}",
            (a["flip_rate"], a["flip_rate_ci"]["ci_low"], a["flip_rate_ci"]["ci_high"],
             b["downgrade_rate"], b["downgrade_rate_ci"]["ci_low"], b["downgrade_rate_ci"]["ci_high"],
             b["downgrade_denom"]), want, 0.006)
LM = mitigation_report(pd.read_parquet(R + "mitigation_Llama-3.1-8B_arith_hard_llama.parquet"))
chk("Llama-matched belief shift (disclaim, isolate)",
    (LM["mit_disclaim"]["delta_vs_none"]["estimate"], LM["mit_isolate"]["delta_vs_none"]["estimate"]),
    (-1.92, 0.96), 0.005)
LH = mitigation_report(pd.read_parquet(R + "mitigation_Llama-3.1-8B_arith_hard.parquet"))
chk("Llama hard-set mitigation-grid shift (commit3, isolate)",
    (LH["commit3"]["delta_vs_none"]["estimate"], LH["mit_isolate"]["delta_vs_none"]["estimate"]),
    (0.31, 1.58), 0.005)
CH = {"arith_hard": (2.01, 1.93, 2.10, 1.38, 1.32, 1.45), "arith_hard_llama": (1.93, 1.78, 2.10, 0.97, 0.89, 1.05)}
for s, want in CH.items():
    rep = channel_report(pd.read_parquet(R + f"channel_Llama-3.1-8B_{s}.parquet"), value="logodds_P")
    su, ss = rep["_contrasts"]["source_self_minus_source_user"], rep["_contrasts"]["source_self_minus_source_system"]
    chk(f"Llama source factorial {s} (self-user, self-system)", tup(su) + tup(ss), want, 0.006)

# Cross-turn true-claim downgrades of argument isolation stay <= 9% in every arithmetic cell.
iso_down = []
for m in ("32B", "72B"):
    g = pd.read_parquet(R + f"crossturn_mit_Qwen2.5-{m}_arith_hard.parquet")
    for _, gp in g.groupby("protocol"):
        none = gp[gp.condition == "none"].set_index("prop_id").logodds_P
        bc = [i for i in gp[gp.gold == True].prop_id.unique() if none[i] > 0]
        iso = gp[gp.framing == "mit_isolate"].set_index("prop_id").logodds_P
        iso_down.append((int(sum(iso[i] <= 0 for i in bc)), len(bc)))
chk("ctmit isolate true-claim downgrades (32B A, 32B B, 72B A, 72B B) as (k, n)",
    [x for kn in iso_down for x in kn], [0, 2, 1, 11, 1, 26, 0, 31], 0)
g7 = pd.read_parquet(GATE_HARD["7B"])
n7 = g7[g7.condition == "none"]
chk("7B hard-set baseline accuracy (true, false)",
    (((n7.p_of_P >= 0.5) & n7.gold).sum() / n7.gold.sum(),
     ((n7.p_of_P < 0.5) & ~n7.gold.astype(bool)).sum() / (~n7.gold.astype(bool)).sum()),
    (0.24, 0.95), 0.005)

print()
print("SECTION 16: tau2-bench argument census (Limitations; recompute with scripts/tau2_argument_census.py)")
import json
cen = json.load(open(R + "tau2_argument_census.json"))
ft = cen["free_text_arguments_by_tool"]
chk("tau2 census (sims, calls, free-text calls, handoff summaries, calculator, other)",
    (cen["n_simulations"], cen["n_tool_calls"], cen["n_calls_with_free_text_argument"],
     ft["transfer_to_human_agents.summary"], ft["calculate.expression"],
     sum(ft.values()) - ft["transfer_to_human_agents.summary"] - ft["calculate.expression"]),
    (4448, 38235, 1738, 1473, 256, 9), 0)
chk("tau2 census free-text share (%)", 100 * cen["n_calls_with_free_text_argument"] / cen["n_tool_calls"], 4.5, 0.05)
chk_bool("tau2 census: four agent models", len(cen["tool_calls_by_model"]) == 4)

print()
print("SECTION 17: same-turn mitigation table as counts with Wilson 95% intervals (Table tab:mit)")
from statsmodels.stats.proportion import proportion_confint
# (flip k, flip n, downgrade k, downgrade n) per representation; Wilson bounds in percent are pinned
# per cell below, so the table's [lo,hi] cannot drift from the parquets.
MITW = {
    "Qwen2.5-1.5B_arith_hard": {"commit3": (16, 30, 0, 20), "mit_disclaim": (0, 30, 20, 20),
                                "mit_reframe": (0, 30, 20, 20), "mit_isolate": (0, 30, 20, 20)},
    "Qwen2.5-7B_arith_hard": {"commit3": (4, 56, 0, 7), "mit_disclaim": (0, 56, 6, 7),
                              "mit_reframe": (0, 56, 1, 7), "mit_isolate": (0, 56, 4, 7)},
    "Qwen2.5-32B_arith_hard": {"commit3": (8, 58, 0, 6), "mit_disclaim": (0, 58, 4, 6),
                               "mit_reframe": (0, 58, 4, 6), "mit_isolate": (0, 58, 4, 6)},
    "Qwen2.5-72B_arith_hard": {"commit3": (26, 38, 0, 31), "mit_disclaim": (1, 38, 6, 31),
                               "mit_reframe": (3, 38, 3, 31), "mit_isolate": (2, 38, 3, 31)},
    "Llama-3.1-8B_arith_hard_llama": {"commit3": (13, 38, 0, 16), "mit_disclaim": (0, 38, 16, 16),
                                      "mit_reframe": (5, 38, 0, 16), "mit_isolate": (8, 38, 0, 16)},
}
WILSON = {(16, 30): (36, 70), (0, 30): (0, 11), (4, 56): (3, 17), (0, 56): (0, 6), (8, 58): (7, 25),
          (0, 58): (0, 6), (26, 38): (53, 81), (1, 38): (0, 13), (3, 38): (3, 21), (2, 38): (1, 17),
          (13, 38): (21, 50), (0, 38): (0, 9), (5, 38): (6, 27), (8, 38): (11, 36), (0, 20): (0, 16),
          (20, 20): (84, 100), (0, 7): (0, 35), (6, 7): (49, 97), (1, 7): (3, 51), (4, 7): (25, 84),
          (0, 6): (0, 39), (4, 6): (30, 90), (0, 31): (0, 11), (6, 31): (9, 36), (3, 31): (3, 25),
          (0, 16): (0, 19), (16, 16): (81, 100)}
for f, rows in MITW.items():
    df = pd.read_parquet(R + f"mitigation_{f}.parquet")
    mf, bt = mitigation_report(df), benign_true_report(df)
    for k, want in rows.items():
        got = (round(mf[k]["flip_rate"] * mf[k]["flip_denom"]), mf[k]["flip_denom"],
               round(bt[k]["downgrade_rate"] * bt[k]["downgrade_denom"]), bt[k]["downgrade_denom"])
        chk(f"mitigation counts {f} {k}", got, want, 0)
for (k, n), want in WILSON.items():
    lo, hi = proportion_confint(k, n, method="wilson")
    chk(f"Wilson 95% interval {k}/{n} (%)", (round(100 * lo), round(100 * hi)), want, 0)

print()
print("SECTION 18: cross-turn mitigation, two-sided counts with Wilson 95% intervals (Table tab:ctmit)")
# Per cell: (flip k, flip n, flip lo, flip hi, downgrade k, downgrade n, down lo, down hi), with a
# baseline acceptance/rejection read as logodds >= 0 / < 0 under the none middle of that protocol.
CTW = {
    ("32B", "A"): {"commitment": (8, 58, 7, 25, 0, 2, 0, 66), "mit_disclaim": (1, 58, 0, 9, 0, 2, 0, 66),
                   "mit_reframe": (0, 58, 0, 6, 0, 2, 0, 66), "mit_isolate": (0, 58, 0, 6, 0, 2, 0, 66)},
    ("32B", "B"): {"commitment": (40, 56, 59, 82, 0, 11, 0, 26), "mit_disclaim": (26, 56, 34, 59, 1, 11, 2, 38),
                   "mit_reframe": (6, 56, 5, 21, 5, 11, 21, 72), "mit_isolate": (0, 56, 0, 6, 0, 11, 0, 26)},
    ("72B", "A"): {"commitment": (22, 41, 39, 68, 0, 26, 0, 13), "mit_disclaim": (4, 41, 4, 23, 1, 26, 1, 19),
                   "mit_reframe": (6, 41, 7, 28, 0, 26, 0, 13), "mit_isolate": (0, 41, 0, 9, 1, 26, 1, 19)},
    ("72B", "B"): {"commitment": (24, 46, 38, 66, 0, 32, 0, 11), "mit_disclaim": (6, 46, 6, 26, 2, 32, 2, 20),
                   "mit_reframe": (11, 46, 14, 38, 1, 32, 1, 16), "mit_isolate": (7, 46, 8, 28, 1, 32, 1, 16)},
}
for (m, proto), cells in CTW.items():
    x = pd.read_parquet(R + f"crossturn_mit_Qwen2.5-{m}_arith_hard.parquet")
    x = x[x.protocol == proto]
    w = x.pivot_table(index="prop_id", columns="framing", values="logodds_P")
    gold = x.groupby("prop_id").gold.first()
    fb = w[(gold == False) & (w["na"] < 0)]
    tb = w[(gold == True) & (w["na"] >= 0)]
    for k, want in cells.items():
        fk, dk = int((fb[k] >= 0).sum()), int((tb[k] < 0).sum())
        flo, fhi = proportion_confint(fk, len(fb), method="wilson")
        dlo, dhi = proportion_confint(dk, len(tb), method="wilson")
        got = (fk, len(fb), round(100 * flo), round(100 * fhi), dk, len(tb), round(100 * dlo), round(100 * dhi))
        chk(f"ctmit two-sided {m}-{proto} {k}", got, want, 0)

print()
print("=" * 100)
print(f"DONE. {len(fails)} FAILED CHECKS" + (":" if fails else ""))
for f in fails:
    print(f"  - {f}")
sys.exit(1 if fails else 0)
