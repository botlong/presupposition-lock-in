from actionlockin.propositions import Proposition
from actionlockin.scorer import StubScorer
from actionlockin.run import run_for_scorer

P = Proposition("p1", "the Nile is longer than the Amazon", True, "copper conducts electricity")


def test_run_for_scorer_shape_and_columns():
    # action middles ('log_fact('/'check(') -> 0.8, declarative assertion -> 0.6, else -> 0.4
    def rule(flat):
        if "log_fact(" in flat or "check(" in flat: return 0.8
        if "confirmed and verified" in flat: return 0.6
        return 0.4
    s = StubScorer("stub", rule=rule)
    df = run_for_scorer(s, [P])
    expected_cols = {"model","prop_id","gold","condition","framing","dose",
                     "p_of_P","logodds_P","mid_n_tokens","mid_mean_logprob","mid_claim_salience"}
    assert expected_cols.issubset(set(df.columns))
    # log-odds readout is sign-consistent with the probability readout
    assert ((df["logodds_P"] > 0) == (df["p_of_P"] > 0.5)).all()
    # 7 condition cells from build_conditions (none, assertion, query, placebo, commit d1/d2/d3)
    assert len(df) == 7
    assert (df["model"] == "stub").all()
    none_row = df[df.condition == "none"].iloc[0]
    assertion_row = df[df.condition == "assertion"].iloc[0]
    action_row = df[(df.condition=="action") & (df.framing=="commitment") & (df.dose==3)].iloc[0]
    assert none_row.p_of_P == 0.4
    assert assertion_row.p_of_P == 0.6
    assert action_row.p_of_P == 0.8


def test_run_for_scorer_pads_middles_to_common_length():
    s = StubScorer("stub", rule=lambda flat: 0.5)
    df = run_for_scorer(s, [P])
    # after padding, every cell's middle has the same (max) whitespace-token count
    assert df["mid_n_tokens"].nunique() == 1
