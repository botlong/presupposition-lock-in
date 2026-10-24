from actionlockin.propositions import Proposition
from actionlockin.conditions import build_conditions
from actionlockin.scorer import StubScorer
from actionlockin.elicit import parse_tf_answer, prob_of_P

P = Proposition("p1", "the Nile is longer than the Amazon", True, "copper conducts electricity")


def test_prob_of_P_routes_middle_to_scorer():
    # rule: higher P(P) when the assistant middle contains a 'log_fact(' commitment action
    s = StubScorer("stub", rule=lambda flat: 0.8 if "log_fact(" in flat else 0.4)
    cts = {(c.condition, c.framing): c for c in build_conditions(P)}
    assert prob_of_P(s, P, cts[("none", "na")]) == 0.4
    assert prob_of_P(s, P, cts[("action", "commitment")]) == 0.8


def test_parse_tf_answer_reads_only_the_first_word():
    assert parse_tf_answer(" True") == "true"
    assert parse_tf_answer("False.") == "false"
    assert parse_tf_answer("TRUE!\nBecause the record says so.") == "true"
    assert parse_tf_answer('"False"') == "false"
    # hedged or delayed answers must not be credited
    assert parse_tf_answer("I think True") == "other"
    assert parse_tf_answer("The claim is false") == "other"
    assert parse_tf_answer("") == "other"
