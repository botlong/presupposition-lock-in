from __future__ import annotations
import numpy as np
import pandas as pd

_COV = ["mid_n_tokens", "mid_mean_logprob", "mid_claim_salience"]


def _pivot(df: pd.DataFrame, value: str = "p_of_P") -> pd.DataFrame:
    """Per-proposition cell means of `value`. df MUST be a single model (prop_id unique per model)."""
    def cell(cond, fr, dose):
        sub = df[(df.condition == cond) & (df.framing == fr) & (df.dose == dose)]
        return sub.groupby("prop_id")[value].mean()
    out = pd.DataFrame({
        "none": cell("none", "na", 0),
        "assertion": cell("assertion", "na", 0),
        "query": cell("action", "query", 0),
        "commit3": cell("action", "commitment", 3),
        "placebo": cell("action", "placebo", 0),
    })
    return out.dropna()


def _boot_ci(x, n_boot: int = 2000, seed: int = 0):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = [rng.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)]
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _pack(est, lo, hi):
    return {"estimate": est, "ci_low": lo, "ci_high": hi}


def _adjusted_coef(sub: pd.DataFrame, predictors, value: str = "p_of_P",
                   n_boot: int = 1000, seed: int = 0):
    """OLS value ~ 1 + predictors; return coef of predictors[0] with a cluster(prop) bootstrap CI."""
    if sub.empty:
        return _pack(0.0, 0.0, 0.0)

    def fit(frame):
        # Standardization and removal of near-constant columns keep the design matrix conditioned;
        # length matching can otherwise make a covariate collinear with the intercept. Rescale the
        # target coefficient so callers still receive it in the predictor's original units.
        y = frame[value].to_numpy(float)
        kept_cols, target_scale = [], None
        for i, c in enumerate(predictors):
            x = frame[c].to_numpy(float)
            s = x.std()
            if s < 1e-8:                       # constant within this (resampled) frame -> skip
                if i == 0:
                    return 0.0                 # variable of interest has no variance here
                continue
            kept_cols.append((i, (x - x.mean()) / s))
            if i == 0:
                target_scale = s
        X = np.column_stack([np.ones(len(frame))] + [col for _, col in kept_cols])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        for pos, (i, _) in enumerate(kept_cols, start=1):
            if i == 0:
                return float(beta[pos] / target_scale)   # standardized -> raw units
        return 0.0

    sub = sub.reset_index(drop=True)
    est = fit(sub)
    props = sub["prop_id"].unique()
    pid = sub["prop_id"].to_numpy()
    groups = {p: np.where(pid == p)[0] for p in props}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(props, size=len(props), replace=True)
        idx = np.concatenate([groups[p] for p in pick])
        boots.append(fit(sub.iloc[idx]))
    return _pack(est, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def contrasts(df: pd.DataFrame, value: str = "p_of_P") -> dict:
    w = _pivot(df, value=value)
    res = {}
    for key, a, b in [
        ("h1_masked_commit_delta", "commit3", "none"),
        ("h2_action_specific", "commit3", "assertion"),
        ("h3_commit_vs_query", "commit3", "query"),      # raw, diagnostic only
        ("h5_placebo", "placebo", "none"),
    ]:
        res[key] = _pack(*_boot_ci((w[a] - w[b]).to_numpy()))

    # Control length, fluency, and salience so the H3 coefficient isolates the commit indicator.
    h3 = df[((df.condition == "action") & (df.framing == "query") & (df.dose == 0)) |
            ((df.condition == "action") & (df.framing == "commitment") & (df.dose == 3))].copy()
    h3["is_commit"] = (h3["framing"] == "commitment").astype(float)
    res["h3_commit_vs_query_adj"] = _adjusted_coef(h3, ["is_commit"] + _COV, value=value)

    # H4 dose slope within commitment: raw + adjusted
    ac = df[(df.condition == "action") & (df.framing == "commitment")].copy()
    slopes = []
    for _, g in ac.groupby("prop_id"):
        if g["dose"].nunique() >= 2:
            slopes.append(np.polyfit(g["dose"].to_numpy(float), g[value].to_numpy(float), 1)[0])
    res["h4_dose_slope"] = _pack(*_boot_ci(np.array(slopes))) if slopes else _pack(0.0, 0.0, 0.0)
    res["h4_dose_slope_adj"] = _adjusted_coef(ac, ["dose"] + _COV, value=value)

    # H6 exploratory: on gold==False, signed move toward wrong (commit3 - none)
    false_ids = df[df.gold == False]["prop_id"].unique()
    wf = w[w.index.isin(false_ids)]
    res["h6_accuracy_shift"] = _pack(*_boot_ci((wf["commit3"] - wf["none"]).to_numpy())) if len(wf) else _pack(0.0, 0.0, 0.0)
    return res


def evaluate_kill_criteria(c: dict) -> dict:
    ok2 = c["h2_action_specific"]["ci_low"] > 0
    ok3 = c["h3_commit_vs_query_adj"]["ci_low"] > 0
    reasons = [
        f"H2 action-specific CI_low={c['h2_action_specific']['ci_low']:.3f} (>0 required): {ok2}",
        f"H3 commit>query (confound-adjusted) CI_low={c['h3_commit_vs_query_adj']['ci_low']:.3f} (>0 required): {ok3}",
    ]
    return {"verdict": "GO" if (ok2 and ok3) else "NULL", "reasons": reasons}


def belief_shift_by_gold(df: pd.DataFrame, value: str = "p_of_P") -> dict:
    """Characterize whether lock-in is a blind yes-bias or amplifies the model's prior.

    Splits H1 (commit3 − none) and H3 (commit3 − query) by gold label. If the shift is large on
    gold==True and ~0 on gold==False, the action AMPLIFIES the model's existing lean rather than
    pushing belief toward 'True' regardless of content (and accuracy harm is correspondingly
    concentrated where the model is uncertain/wrong, not universal).
    """
    out = {}
    for lbl, sub in [("all", df), ("true", df[df.gold == True]), ("false", df[df.gold == False])]:
        w = _pivot(sub, value=value)
        if len(w) == 0:
            out[lbl] = {"n": 0, "h1": _pack(0.0, 0.0, 0.0), "h3": _pack(0.0, 0.0, 0.0)}
            continue
        out[lbl] = {"n": int(len(w)),
                    "h1": _pack(*_boot_ci((w["commit3"] - w["none"]).to_numpy())),
                    "h3": _pack(*_boot_ci((w["commit3"] - w["query"]).to_numpy()))}
    return out


def evaluate_phenomenon(c: dict) -> dict:
    """Evaluate the confirmatory raw contrasts after design-level length matching.

    The phenomenon is ESTABLISHED iff, per model:
      H1 (commit raises pre-result belief) ci_low > 0  AND
      H3 (commit > non-presuppositional query) ci_low > 0  AND
      H4 (dose-response slope) ci_low > 0,
    with H5 (placebo specificity) not positive (ci_high <= 0).
    The confound-adjusted H3/H4 regressions are an unstable sensitivity check (condition is
    confounded with fluency); they do NOT gate. H2 is reported as a signed dissociation, not gated.
    """
    h1 = c["h1_masked_commit_delta"]["ci_low"] > 0
    h3 = c["h3_commit_vs_query"]["ci_low"] > 0
    h4 = c["h4_dose_slope"]["ci_low"] > 0
    h5 = c["h5_placebo"]["ci_high"] <= 0
    ok = h1 and h3 and h4 and h5
    reasons = [
        f"H1 commit>none ci_low={c['h1_masked_commit_delta']['ci_low']:.3f} (>0): {h1}",
        f"H3 commit>query ci_low={c['h3_commit_vs_query']['ci_low']:.3f} (>0): {h3}",
        f"H4 dose slope ci_low={c['h4_dose_slope']['ci_low']:.3f} (>0): {h4}",
        f"H5 placebo ci_high={c['h5_placebo']['ci_high']:.3f} (<=0, specificity): {h5}",
    ]
    return {"verdict": "ESTABLISHED" if ok else "NOT-ESTABLISHED", "reasons": reasons}


def accuracy_harm(df: pd.DataFrame) -> dict:
    """Decision-level harm on gold==False items (df = single model).

    Beyond the probability shift (h6), this measures whether the commitment action flips
    the model's *answer*: on a false claim the correct answer is False (p_of_P < 0.5).
    Reports correct-rejection accuracy under none / assertion / commit3, and the flip rate
    (of items correctly rejected at baseline, the fraction pushed to a wrong 'True' by commit3).
    """
    w = _pivot(df)
    false_ids = df[df.gold == False]["prop_id"].unique()
    wf = w[w.index.isin(false_ids)]
    n = len(wf)
    if n == 0:
        return {"n_false": 0, "acc_none": 0.0, "acc_assertion": 0.0,
                "acc_commit3": 0.0, "flip_rate": 0.0}
    acc = lambda col: float((wf[col] < 0.5).mean())   # gold False -> correct iff p_of_P < 0.5
    base_correct = wf["none"] < 0.5
    denom = int(base_correct.sum())
    flips = int((base_correct & (wf["commit3"] >= 0.5)).sum())
    # Flip rate is a proportion over a small base-correct denominator; report it with the same
    # per-item percentile bootstrap as every belief quantity so the text can carry an interval.
    flip_ind = (wf.loc[base_correct, "commit3"] >= 0.5).astype(float).to_numpy()
    return {"n_false": int(n),
            "acc_none": acc("none"),
            "acc_assertion": acc("assertion"),
            "acc_commit3": acc("commit3"),
            "flip_rate": float(flips / denom) if denom else 0.0,
            "flip_denom": denom,
            "flip_rate_ci": _pack(*_boot_ci(flip_ind))}


def mitigation_report(df: pd.DataFrame, value: str = "logodds_P") -> dict:
    """Efficacy of each mitigation (df = single model, from build_mitigation_conditions).

    For the dose-3 commitment baseline and each mitigated representation, reports the per-prop
    paired belief delta vs `none` (the remaining lock-in) with a bootstrap CI, plus correct-
    rejection accuracy and flip rate on gold==False items. A mitigation works iff its belief
    delta collapses toward 0 (vs the large commit3 delta) and false-claim accuracy is restored.
    """
    labels = {
        "commit3": ("action", "commitment", 3),
        "mit_disclaim": ("action", "mit_disclaim", 3),
        "mit_reframe": ("action", "mit_reframe", 3),
        "mit_isolate": ("action", "mit_isolate", 3),
    }

    def cell(cond, fr, dose, val):
        sub = df[(df.condition == cond) & (df.framing == fr) & (df.dose == dose)]
        return sub.groupby("prop_id")[val].mean()

    none = df[df.condition == "none"].groupby("prop_id")[value].mean()
    none_p = df[df.condition == "none"].groupby("prop_id")["p_of_P"].mean()
    false_ids = set(df[df.gold == False]["prop_id"].unique())
    base_false = none_p[none_p.index.isin(false_ids)]
    base_correct = set(base_false[base_false < 0.5].index)

    out = {"_n_false": int(len(base_false)),
           "_none_acc_false": float((base_false < 0.5).mean()) if len(base_false) else 0.0}
    for lbl, (cond, fr, dose) in labels.items():
        s = cell(cond, fr, dose, value)
        paired = (s - none).dropna()
        sp = cell(cond, fr, dose, "p_of_P")
        spf = sp[sp.index.isin(false_ids)]
        acc = float((spf < 0.5).mean()) if len(spf) else 0.0
        flips = sum(1 for i in base_correct if i in sp.index and sp.loc[i] >= 0.5)
        # Same per-item bootstrap as accuracy_harm; missing items keep their non-flip (0) coding
        # so the estimate matches the point flip_rate exactly.
        flip_ind = np.array([1.0 if (i in sp.index and sp.loc[i] >= 0.5) else 0.0
                             for i in sorted(base_correct)])
        out[lbl] = {"n": int(len(paired)),
                    "delta_vs_none": _pack(*_boot_ci(paired.to_numpy())),
                    "acc_false": acc,
                    "flip_rate": float(flips / len(base_correct)) if base_correct else 0.0,
                    "flip_denom": int(len(base_correct)),
                    "flip_rate_ci": _pack(*_boot_ci(flip_ind)) if base_correct else _pack(0.0, 0.0, 0.0)}
    return out


def benign_true_report(df: pd.DataFrame, value: str = "logodds_P") -> dict:
    """Cost of each mitigation on gold==True items (df = single model, from
    build_mitigation_conditions).

    Removing lock-in on false claims is worthless if it also suppresses belief in TRUE claims
    the agent recorded. For the dose-3 commitment and each mitigated representation, reports
    the per-prop paired belief delta vs `none` on gold==True items, accuracy on true claims
    (p_of_P >= 0.5), and the downgrade rate: of true items correctly affirmed at baseline,
    the fraction pushed below 0.5 by the representation.
    """
    labels = {
        "commit3": ("action", "commitment", 3),
        "mit_disclaim": ("action", "mit_disclaim", 3),
        "mit_reframe": ("action", "mit_reframe", 3),
        "mit_isolate": ("action", "mit_isolate", 3),
    }

    def cell(cond, fr, dose, val):
        sub = df[(df.condition == cond) & (df.framing == fr) & (df.dose == dose)]
        return sub.groupby("prop_id")[val].mean()

    none = df[df.condition == "none"].groupby("prop_id")[value].mean()
    none_p = df[df.condition == "none"].groupby("prop_id")["p_of_P"].mean()
    true_ids = set(df[df.gold == True]["prop_id"].unique())
    base_true = none_p[none_p.index.isin(true_ids)]
    base_correct = set(base_true[base_true >= 0.5].index)

    out = {"_n_true": int(len(base_true)),
           "_none_acc_true": float((base_true >= 0.5).mean()) if len(base_true) else 0.0}
    for lbl, (cond, fr, dose) in labels.items():
        s = cell(cond, fr, dose, value)
        paired = (s - none).dropna()
        paired = paired[paired.index.isin(true_ids)]
        sp = cell(cond, fr, dose, "p_of_P")
        spt = sp[sp.index.isin(true_ids)]
        acc = float((spt >= 0.5).mean()) if len(spt) else 0.0
        downs = sum(1 for i in base_correct if i in sp.index and sp.loc[i] < 0.5)
        # Same per-item bootstrap as the flip rates; missing items keep their non-downgrade (0)
        # coding so the estimate matches the point rate exactly.
        down_ind = np.array([1.0 if (i in sp.index and sp.loc[i] < 0.5) else 0.0
                             for i in sorted(base_correct)])
        out[lbl] = {"n": int(len(paired)),
                    "delta_vs_none": _pack(*_boot_ci(paired.to_numpy())),
                    "acc_true": acc,
                    "downgrade_rate": float(downs / len(base_correct)) if base_correct else 0.0,
                    "downgrade_denom": int(len(base_correct)),
                    "downgrade_rate_ci": (_pack(*_boot_ci(down_ind)) if base_correct
                                          else _pack(0.0, 0.0, 0.0))}
    return out


def channel_report(df: pd.DataFrame, value: str = "logodds_P") -> dict:
    """Source-attribution factorial report (df = single model, from build_channel_conditions).

    Per-arm belief delta vs none + false-claim accuracy + flip (as in mitigation_report), PLUS the
    direct paired self-vs-other contrasts with bootstrap CIs --- the actual channel test. All source
    arms are one impersonal assistant readback sentence at the same position, length-matched, so the
    only difference between source_self / source_user / source_system is the named source attribution
    on the identical factive predicate. A self-source privilege shows `source_self - source_user > 0`
    and `source_self - source_system > 0`.
    """
    labels = {
        "commit":      ("action", "commitment", 3),   # self, tool-call form
        "source_self": ("assertion", "self", 0),       # impersonal record attributed to assistant self
        "query":       ("action", "query", 0),         # self, non-presuppositional
        "isolate":     ("action", "mit_isolate", 3),   # self, argument withheld
        "source_user": ("assertion", "user", 0),       # same claim, attributed to the user
        "source_system": ("assertion", "system", 0),   # same claim, attributed to the system
    }

    def cell(cond, fr, dose, val):
        sub = df[(df.condition == cond) & (df.framing == fr) & (df.dose == dose)]
        return sub.groupby("prop_id")[val].mean()

    none = df[df.condition == "none"].groupby("prop_id")[value].mean()
    none_p = df[df.condition == "none"].groupby("prop_id")["p_of_P"].mean()
    false_ids = set(df[df.gold == False]["prop_id"].unique())
    base_false = none_p[none_p.index.isin(false_ids)]
    base_correct = set(base_false[base_false < 0.5].index)

    cells = {}   # arm -> per-prop value series, retained for the paired contrasts
    out = {"_n_false": int(len(base_false)),
           "_none_acc_false": float((base_false < 0.5).mean()) if len(base_false) else 0.0}
    for lbl, (cond, fr, dose) in labels.items():
        s = cell(cond, fr, dose, value)
        cells[lbl] = s
        paired = (s - none).dropna()
        sp = cell(cond, fr, dose, "p_of_P")
        spf = sp[sp.index.isin(false_ids)]
        acc = float((spf < 0.5).mean()) if len(spf) else 0.0
        flips = sum(1 for i in base_correct if i in sp.index and sp.loc[i] >= 0.5)
        out[lbl] = {"n": int(len(paired)),
                    "delta_vs_none": _pack(*_boot_ci(paired.to_numpy())),
                    "acc_false": acc,
                    "flip_rate": float(flips / len(base_correct)) if base_correct else 0.0}

    # the channel test proper: per-proposition paired self-vs-other contrasts (>0 => self-source privilege)
    out["_contrasts"] = {}
    for a, b in [("source_self", "source_user"), ("source_self", "source_system"),
                 ("commit", "source_user"), ("commit", "source_system")]:
        paired = (cells[a] - cells[b]).dropna()
        out["_contrasts"][f"{a}_minus_{b}"] = {"n": int(len(paired)), **_pack(*_boot_ci(paired.to_numpy()))}
    return out


def pooled_summary(df: pd.DataFrame,
                   keys=("h2_action_specific", "h3_commit_vs_query_adj"),
                   value: str = "p_of_P", n_boot: int = 2000, seed: int = 0) -> dict:
    """Per-model estimates + cluster bootstrap over MODELS (the replication unit).

    Never averages a proposition across models; each model yields its own contrast first.
    """
    models = sorted(df["model"].unique())
    per_model = {m: contrasts(df[df.model == m], value=value) for m in models}
    rng = np.random.default_rng(seed)
    out = {}
    for k in keys:
        ests = np.array([per_model[m][k]["estimate"] for m in models], dtype=float)
        boots = [rng.choice(ests, size=len(ests), replace=True).mean() for _ in range(n_boot)]
        out[k] = {
            "estimate": float(ests.mean()),
            "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)),
            "per_model": {m: per_model[m][k]["estimate"] for m in models},
        }
    return out
