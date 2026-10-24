from actionlockin.scorer import StubScorer


def test_stub_prob_true_uses_rule_on_flattened_text():
    s = StubScorer(name="stub", rule=lambda p: 0.7 if "commitment-marker" in p else 0.3)
    assert s.prob_true("sys", "user", "has commitment-marker here") == 0.7
    assert s.prob_true("sys", "user", "plain") == 0.3


def test_stub_middle_logprobs_length_matches_tokens():
    s = StubScorer(name="stub", rule=lambda p: 0.5)
    lp = s.middle_logprobs("sys", "user", "a b c")
    assert len(lp) == 3 and all(x == -1.0 for x in lp)


def test_stub_tokenize_is_whitespace():
    s = StubScorer(name="stub", rule=lambda p: 0.5)
    assert s.tokenize("a b c") == ["a", "b", "c"]


def test_stub_prob_true_clamped_to_unit_interval():
    s = StubScorer(name="stub", rule=lambda p: 1.5)  # rule may overshoot
    assert 0.0 <= s.prob_true("s", "u", "x") <= 1.0


def test_stub_logit_diff_is_logodds_of_prob_true():
    import math
    s = StubScorer(name="stub", rule=lambda p: 0.7 if "x" in p else 0.5)
    assert abs(s.logit_diff("s", "u", "x here") - math.log(0.7 / 0.3)) < 1e-6
    assert s.logit_diff("s", "u", "plain") == 0.0          # p=0.5 -> log-odds 0
    s2 = StubScorer(name="stub", rule=lambda p: 0.2)
    assert s2.logit_diff("s", "u", "z") < 0                # p<0.5 -> negative log-odds


def test_stub_logit_diff_finite_at_extremes():
    s = StubScorer(name="stub", rule=lambda p: 1.0)        # p clamps, log-odds stays finite
    assert s.logit_diff("s", "u", "x") > 0 and s.logit_diff("s", "u", "x") < float("inf")


def test_stub_chat_flattens_all_turns():
    # rule sees every turn's content (so a downstream 'evidence' turn is in scope)
    s = StubScorer(name="stub", rule=lambda flat: 0.9 if "log_fact(" in flat else 0.2)
    committed = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"},
                 {"role": "assistant", "content": 'log_fact(fact="P")'},
                 {"role": "user", "content": "evidence: P is false. Answer:"}]
    plain = [{"role": "user", "content": "q"}, {"role": "user", "content": "evidence"}]
    assert s.prob_true_chat(committed) == 0.9
    assert s.prob_true_chat(plain) == 0.2
    assert s.logit_diff_chat(committed) > 0 > s.logit_diff_chat(plain)
