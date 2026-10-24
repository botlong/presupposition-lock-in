from actionlockin.propositions import Proposition
from actionlockin.scorer import StubScorer
from actionlockin.calibration import baseline_prob_true, filter_midband

P1 = Proposition("p1", "claim one", True, "u1")
P2 = Proposition("p2", "claim two", False, "u2")
P3 = Proposition("p3", "claim three", True, "u3")


def _rule(flat):
    if "claim one" in flat: return 0.05   # too confident-false -> drop
    if "claim two" in flat: return 0.50   # mid -> keep
    if "claim three" in flat: return 0.95 # too confident-true -> drop
    return 0.5


def test_baseline_prob_true_reads_bare_probe():
    s = StubScorer("stub", rule=_rule)
    assert baseline_prob_true(s, P2) == 0.50


def test_filter_midband_keeps_only_uncertain():
    s = StubScorer("stub", rule=_rule)
    kept = filter_midband(s, [P1, P2, P3], 0.2, 0.8)
    assert [p.id for p in kept] == ["p2"]
