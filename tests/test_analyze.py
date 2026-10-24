import numpy as np, pandas as pd
from actionlockin.analyze import (contrasts, evaluate_kill_criteria, pooled_summary,
                                  accuracy_harm, evaluate_phenomenon, belief_shift_by_gold)


def test_belief_shift_by_gold_separates_true_false():
    rows = []
    # true items: big commit lock-in (none 0.3 -> commit3 0.9); false items: ~none movement
    for i in range(6):
        rows += _cells(f"t{i}", True, 0.3, 0.9, 0.3, 0.3, 0.5, 0.7, 0.9)
    for i in range(6):
        rows += _cells(f"f{i}", False, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3)
    out = belief_shift_by_gold(pd.DataFrame(rows))
    assert out["true"]["n"] == 6 and out["false"]["n"] == 6
    assert out["true"]["h1"]["estimate"] > 0.5      # strong lock-in on true items
    assert abs(out["false"]["h1"]["estimate"]) < 0.05  # ~0 on false items


def test_evaluate_phenomenon_established_and_not():
    est = {"h1_masked_commit_delta": {"ci_low": 0.2}, "h3_commit_vs_query": {"ci_low": 0.1},
           "h4_dose_slope": {"ci_low": 0.05}, "h5_placebo": {"ci_high": -0.1}}
    assert evaluate_phenomenon(est)["verdict"] == "ESTABLISHED"
    # placebo positive -> specificity fails -> NOT-ESTABLISHED
    bad = {**est, "h5_placebo": {"ci_high": 0.3}}
    assert evaluate_phenomenon(bad)["verdict"] == "NOT-ESTABLISHED"
    # H1 CI crosses 0 -> NOT-ESTABLISHED
    bad2 = {**est, "h1_masked_commit_delta": {"ci_low": -0.01}}
    assert evaluate_phenomenon(bad2)["verdict"] == "NOT-ESTABLISHED"


def _cells(prop_id, gold, none, assertion, query, placebo, c1, c2, c3):
    spec = [("none", "na", 0, none), ("assertion", "na", 0, assertion),
            ("action", "query", 0, query), ("action", "placebo", 0, placebo),
            ("action", "commitment", 1, c1), ("action", "commitment", 2, c2),
            ("action", "commitment", 3, c3)]
    return [{"model": "m", "prop_id": prop_id, "gold": gold, "condition": cd, "framing": fr,
             "dose": d, "p_of_P": p, "mid_n_tokens": 12, "mid_mean_logprob": -1.0,
             "mid_claim_salience": 0.5} for cd, fr, d, p in spec]


def test_accuracy_harm_flip_on_false_items():
    rows = []
    # 4 false items: none correctly rejects (0.2), commit3 flips to wrong (0.7), assertion wrong (0.8)
    for i in range(4):
        rows += _cells(f"f{i}", False, 0.2, 0.8, 0.3, 0.2, 0.5, 0.6, 0.7)
    # 2 true items (ignored by accuracy_harm, which scopes to gold==False)
    for i in range(2):
        rows += _cells(f"t{i}", True, 0.7, 0.9, 0.7, 0.7, 0.7, 0.8, 0.9)
    h = accuracy_harm(pd.DataFrame(rows))
    assert h["n_false"] == 4
    assert h["acc_none"] == 1.0        # all 4 false claims correctly rejected at baseline
    assert h["acc_commit3"] == 0.0     # commit3 flips all of them to wrong 'True'
    assert h["acc_assertion"] == 0.0
    assert h["flip_rate"] == 1.0


def test_accuracy_harm_empty_false_is_safe():
    rows = []
    for i in range(2):
        rows += _cells(f"t{i}", True, 0.7, 0.9, 0.7, 0.7, 0.7, 0.8, 0.9)
    h = accuracy_harm(pd.DataFrame(rows))
    assert h["n_false"] == 0 and h["flip_rate"] == 0.0


def _row(i, cond, fr, dose, p, ntok=12, salience=0.5, model="m", seed_noise=0.0):
    return {"model": model, "prop_id": f"p{i}", "gold": bool(i % 2),
            "condition": cond, "framing": fr, "dose": dose,
            "p_of_P": float(min(1, max(0, p))),
            "mid_n_tokens": ntok, "mid_mean_logprob": -1.0 + seed_noise,
            "mid_claim_salience": salience}


def _synth(action_specific=0.15, commit_minus_query=0.10, n=80, seed=0, model="m"):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        base = rng.uniform(0.3, 0.6)
        none = base
        assertion = base + 0.05
        commit3 = base + 0.05 + action_specific
        query = commit3 - commit_minus_query
        # tiny covariate jitter so the adjusted regression is identifiable, no real confound
        j = lambda: rng.normal(0, 0.01)
        rows += [
            _row(i, "none", "na", 0, none, ntok=12, salience=0.0, model=model, seed_noise=j()),
            _row(i, "assertion", "na", 0, assertion, ntok=12, model=model, seed_noise=j()),
            _row(i, "action", "query", 0, query, ntok=12, model=model, seed_noise=j()),
            _row(i, "action", "commitment", 1, base + 0.05 + action_specific * 0.33, model=model, seed_noise=j()),
            _row(i, "action", "commitment", 2, base + 0.05 + action_specific * 0.66, model=model, seed_noise=j()),
            _row(i, "action", "commitment", 3, commit3, model=model, seed_noise=j()),
            _row(i, "action", "placebo", 0, none + rng.normal(0, 0.005), ntok=12, salience=0.0, model=model, seed_noise=j()),
        ]
    return pd.DataFrame(rows)


def _synth_h3_length_artifact(n=120, seed=1):
    """commit3 has higher p ONLY because it is longer; is_commit has no real effect.

    p = base + slope*n_tokens (+ tiny noise); kept well within (0,1) so no clamping.
    commit3 and query token counts overlap widely so length and is_commit are separable.
    """
    rng = np.random.default_rng(seed)
    rows = []
    slope = 0.03
    for i in range(n):
        base = rng.uniform(0.10, 0.25)
        commit3_ntok = int(rng.integers(12, 17))   # 12..16
        query_ntok = int(rng.integers(10, 15))     # 10..14  (wide overlap -> identifiable)
        commit3 = base + slope * commit3_ntok + rng.normal(0, 0.001)
        query = base + slope * query_ntok + rng.normal(0, 0.001)
        rows += [
            _row(i, "none", "na", 0, base, ntok=14, salience=0.0),
            _row(i, "assertion", "na", 0, base, ntok=14),   # H2 = commit3 - base is large -> ok2 passes
            _row(i, "action", "query", 0, query, ntok=query_ntok),
            _row(i, "action", "commitment", 1, commit3 - 0.10, ntok=12),
            _row(i, "action", "commitment", 2, commit3 - 0.05, ntok=13),
            _row(i, "action", "commitment", 3, commit3, ntok=commit3_ntok),
            _row(i, "action", "placebo", 0, base + rng.normal(0, 0.005), ntok=14, salience=0.0),
        ]
    return pd.DataFrame(rows)


def test_contrasts_recover_known_effects():
    c = contrasts(_synth(action_specific=0.15, commit_minus_query=0.10))
    assert abs(c["h2_action_specific"]["estimate"] - 0.15) < 0.03
    assert abs(c["h3_commit_vs_query"]["estimate"] - 0.10) < 0.03
    assert abs(c["h3_commit_vs_query_adj"]["estimate"] - 0.10) < 0.04  # ~raw (no real confound)
    assert c["h4_dose_slope"]["estimate"] > 0


def test_kill_criteria_go_when_effects_present():
    c = contrasts(_synth(action_specific=0.15, commit_minus_query=0.10))
    assert evaluate_kill_criteria(c)["verdict"] == "GO"


def test_kill_criteria_null_when_no_action_specific_effect():
    c = contrasts(_synth(action_specific=0.0, commit_minus_query=0.0))
    assert evaluate_kill_criteria(c)["verdict"] == "NULL"


def test_adjusted_h3_catches_length_artifact_and_verdict_is_null():
    c = contrasts(_synth_h3_length_artifact())
    assert c["h3_commit_vs_query"]["ci_low"] > 0          # RAW delta would (wrongly) pass
    assert abs(c["h3_commit_vs_query_adj"]["estimate"]) < 0.04  # adjusted ~ 0
    assert evaluate_kill_criteria(c)["verdict"] == "NULL"  # gate reads the adjusted H3


def test_pooled_summary_reports_per_model_and_cluster_ci():
    df = pd.concat([_synth(seed=1, model="a"), _synth(seed=2, model="b"),
                    _synth(seed=3, model="c")], ignore_index=True)
    out = pooled_summary(df)
    assert set(out["h2_action_specific"]["per_model"].keys()) == {"a", "b", "c"}
    assert "ci_low" in out["h2_action_specific"] and "ci_high" in out["h2_action_specific"]


def _mit_cells(prop_id, gold, none_lo, c3_lo, mit_lo, none_p, c3_p, mit_p):
    spec = [("none", "na", 0, none_lo, none_p),
            ("action", "commitment", 3, c3_lo, c3_p),
            ("action", "mit_disclaim", 3, mit_lo, mit_p),
            ("action", "mit_reframe", 3, mit_lo, mit_p),
            ("action", "mit_isolate", 3, mit_lo, mit_p)]
    return [{"model": "m", "prop_id": prop_id, "gold": gold, "condition": cd, "framing": fr,
             "dose": d, "logodds_P": lo, "p_of_P": p, "mid_n_tokens": 12,
             "mid_mean_logprob": -1.0, "mid_claim_salience": 0.5} for cd, fr, d, lo, p in spec]


def test_mitigation_report_detects_working_mitigation():
    from actionlockin.analyze import mitigation_report
    rows = []
    # false items: none rejects (logodds -2, p 0.2); commit3 locks in (logodds +3, p 0.8 -> flips);
    # mitigations leave belief at the none level (logodds -2, p 0.2 -> no flip, still rejected)
    for i in range(8):
        rows += _mit_cells(f"f{i}", False, none_lo=-2.0, c3_lo=+3.0, mit_lo=-2.0,
                           none_p=0.2, c3_p=0.8, mit_p=0.2)
    out = mitigation_report(pd.DataFrame(rows), value="logodds_P")
    assert out["_n_false"] == 8
    assert out["commit3"]["delta_vs_none"]["estimate"] > 2.0   # large baseline lock-in
    assert out["commit3"]["flip_rate"] == 1.0                  # commit3 flips every false claim
    assert out["commit3"]["acc_false"] == 0.0
    for m in ("mit_disclaim", "mit_reframe", "mit_isolate"):
        assert abs(out[m]["delta_vs_none"]["estimate"]) < 0.5  # lock-in removed (delta ~ 0)
        assert out[m]["flip_rate"] == 0.0                      # no flips
        assert out[m]["acc_false"] == 1.0                      # false claims still rejected


def test_benign_true_report_detects_costly_mitigation():
    from actionlockin.analyze import benign_true_report
    rows = []
    # true items: none affirms (logodds +2, p 0.8); commit3 keeps them affirmed; the mitigations
    # here suppress belief below threshold (logodds -2, p 0.2 -> downgraded, accuracy lost)
    for i in range(8):
        rows += _mit_cells(f"t{i}", True, none_lo=+2.0, c3_lo=+3.0, mit_lo=-2.0,
                           none_p=0.8, c3_p=0.9, mit_p=0.2)
    out = benign_true_report(pd.DataFrame(rows), value="logodds_P")
    assert out["_n_true"] == 8
    assert out["_none_acc_true"] == 1.0
    assert out["commit3"]["acc_true"] == 1.0
    assert out["commit3"]["downgrade_rate"] == 0.0
    for m in ("mit_disclaim", "mit_reframe", "mit_isolate"):
        assert out[m]["delta_vs_none"]["estimate"] < -3.0      # belief in true claims taxed
        assert out[m]["downgrade_rate"] == 1.0                 # every affirmed item downgraded
        assert out[m]["acc_true"] == 0.0
        assert out[m]["downgrade_denom"] == 8
