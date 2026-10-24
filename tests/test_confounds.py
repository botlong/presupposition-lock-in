from actionlockin.scorer import StubScorer
from actionlockin.confounds import text_features


def test_text_features_counts_mean_and_salience():
    s = StubScorer("stub", rule=lambda p: 0.5)  # middle_logprobs = -1.0 per token
    f = text_features(s, "sys", "user", "alpha beta gamma", "beta")
    assert f["n_tokens"] == 3
    assert abs(f["mean_logprob"] - (-1.0)) < 1e-9
    # "beta" ends at char 10 of 16 -> salience ~0.625
    assert 0.5 < f["claim_salience"] < 0.7


def test_text_features_absent_claim_zero_salience():
    s = StubScorer("stub", rule=lambda p: 0.5)
    f = text_features(s, "sys", "user", "alpha beta gamma", "zzz")
    assert f["claim_salience"] == 0.0


def test_text_features_empty_is_safe():
    s = StubScorer("stub", rule=lambda p: 0.5)
    f = text_features(s, "sys", "user", "", "x")
    assert f["n_tokens"] == 0
    assert f["mean_logprob"] == 0.0
    assert f["claim_salience"] == 0.0
