from actionlockin.propositions import Proposition
from actionlockin.conditions import (
    ConditionText, SYSTEM, ANSWER_CUE, belief_probe, build_conditions, pad_to_match,
)

P = Proposition("p1", "the Nile is longer than the Amazon", True, "copper conducts electricity")
ws = lambda s: s.split()  # stub whitespace tokenizer


def test_probe_is_fixed_and_contains_claim_no_cue():
    probe = belief_probe(P)
    assert P.claim in probe
    assert "True or False" in probe
    assert not probe.rstrip().endswith(ANSWER_CUE)  # cue is appended by the scorer, not the probe


def test_build_conditions_covers_required_cells():
    cts = build_conditions(P)
    keys = {(c.condition, c.framing, c.dose) for c in cts}
    assert ("none", "na", 0) in keys
    assert ("assertion", "na", 0) in keys
    assert ("action", "query", 0) in keys
    assert ("action", "commitment", 1) in keys
    assert ("action", "commitment", 3) in keys
    assert ("action", "placebo", 0) in keys


def test_build_mitigation_conditions_cells_and_isolation():
    from actionlockin.conditions import build_mitigation_conditions
    cts = {(c.condition, c.framing, c.dose): c for c in build_mitigation_conditions(P)}
    assert ("none", "na", 0) in cts
    assert ("action", "commitment", 3) in cts            # harmful baseline
    for m in ("mit_disclaim", "mit_reframe", "mit_isolate"):
        assert ("action", m, 3) in cts
    # disclaim keeps the call verbatim but tags it unverified; reframe restates P with a non-factive status
    assert P.claim in cts[("action", "mit_disclaim", 3)].middle
    assert "unverified" in cts[("action", "mit_disclaim", 3)].middle.lower()
    assert P.claim in cts[("action", "mit_reframe", 3)].middle
    assert "hypothesis" in cts[("action", "mit_reframe", 3)].middle.lower()
    # isolate withholds the presupposing argument (P absent) yet keeps the action on record
    assert P.claim not in cts[("action", "mit_isolate", 3)].middle
    assert "log_fact(" in cts[("action", "mit_isolate", 3)].middle


def test_assertion_is_content_matched_and_action_presupposes_P():
    cts = {(c.condition, c.framing): c for c in build_conditions(P)}
    assertion = cts[("assertion", "na")].middle
    # assertion carries the SAME factive lexicon as dose-3 ("confirmed and verified"),
    # declarative (non-action) form — NOT a maximal "proven true" that would saturate belief
    assert P.claim in assertion
    assert "confirmed and verified" in assertion and "log_fact(" not in assertion
    commit = cts[("action", "commitment")].middle  # dose 3 is the canonical commitment cell
    # premise-style action: files P as an established fact (no proof-seeking implicature)
    assert "log_fact(" in commit and P.claim in commit


def test_action_cells_have_no_result_line():
    # Any result marker would add evidence or an absence-of-evidence implicature to a readout that
    # is intended to measure the context before a tool result exists.
    for c in build_conditions(P):
        if c.condition == "action":
            assert "result" not in c.middle.lower()
            assert "returned" not in c.middle.lower()


def test_all_commitment_rungs_presuppose_query_is_non_presuppositional():
    cts = {(c.condition, c.framing, c.dose): c for c in build_conditions(P)}
    # query (dose 0) is the only non-presuppositional action wording
    assert "whether" in cts[("action", "query", 0)].middle
    # every commitment rung embeds P factively (no interrogative "whether")
    for d in (1, 2, 3):
        assert "whether" not in cts[("action", "commitment", d)].middle


def test_placebo_uses_unrelated_not_P():
    cts = {(c.condition, c.framing): c for c in build_conditions(P)}
    placebo = cts[("action", "placebo")].middle
    assert P.unrelated_claim in placebo
    assert P.claim not in placebo


def test_pad_to_match_reaches_target_without_truncation():
    out = pad_to_match(ws, "short text", target_n=6)
    assert len(ws(out)) >= 6
    assert out.startswith("short text")
