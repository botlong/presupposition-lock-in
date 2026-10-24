from glob import glob

import pytest
from tokenizers import Tokenizer

from actionlockin.conditions import build_channel_conditions, _FACTIVE, pad_to_match
from actionlockin.run import run_for_scorer
from actionlockin.analyze import channel_report
from actionlockin.propositions import Proposition
from actionlockin.scorer import StubScorer


def _prop(i: int, gold: bool) -> Proposition:
    return Proposition(f"p{i}", f"claim number {i} holds", gold, "the sky is blue")


def _by(cts):
    return {(c.condition, c.framing): c for c in cts}


def _qwen_tokenizer() -> Tokenizer:
    paths = glob("models/huggingface/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*/tokenizer.json")
    if not paths:  # fresh clone without the HF cache: skip, don't fail (repo stays GPU/network-free)
        pytest.skip(
            "cached Qwen tokenizer.json absent; populate models/huggingface for "
            "Qwen/Qwen2.5-1.5B-Instruct to run the real-tokenizer padding tests"
        )
    try:
        return Tokenizer.from_file(paths[0])
    except Exception:
        pytest.skip("cached Qwen tokenizer.json unreadable; repopulate models/huggingface")


def _find_subsequence(haystack: list[int], needle: list[int]) -> int:
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return -1


def test_seven_arms_and_framings():
    cts = build_channel_conditions(_prop(1, False))
    assert len(cts) == 7
    assert set(_by(cts)) == {("none", "na"), ("action", "commitment"), ("assertion", "self"),
                             ("action", "query"), ("action", "mit_isolate"),
                             ("assertion", "user"), ("assertion", "system")}


def test_identical_predicate_only_source_differs():
    p = _prop(2, False)
    pred = _FACTIVE.format(c=p.claim)            # "confirmed and verified that <claim>"
    by = _by(build_channel_conditions(p))
    # the SAME factive predicate appears in all three source arms; only the subject/source differs
    assert pred in by[("assertion", "self")].middle
    assert pred in by[("assertion", "user")].middle
    assert pred in by[("assertion", "system")].middle
    assert by[("assertion", "self")].middle == f"On record from the assistant (self): it is {pred}."
    assert by[("assertion", "user")].middle == f"On record from the user: it is {pred}."
    assert by[("assertion", "system")].middle == f"On record from the system: it is {pred}."


def test_position_clean_content_only_in_assistant_mid():
    # The factive content must occupy the same assistant slot in every arm; changing another turn
    # would make source attribution inseparable from distance to the answer position.
    p = _prop(3, False)
    pred = _FACTIVE.format(c=p.claim)
    for c in build_channel_conditions(p):
        if c.framing in ("self", "user", "system"):
            assert pred in c.middle


def test_length_matched_after_channel_padding():
    # Channel runs opt into left padding; pad_to_match defaults remain right padding for gate/mitigation.
    cts = build_channel_conditions(_prop(4, False))
    tok = lambda s: s.split()
    target = max(len(tok(c.middle)) for c in cts)
    padded = [pad_to_match(tok, c.middle, target, filler=" note", side="left") for c in cts]
    assert len({len(tok(m)) for m in padded}) == 1
    assert pad_to_match(tok, "short text", 4, filler=" note") == "short text note note"


def test_runner_and_report_with_stub():
    s = StubScorer("stub", lambda flat: 0.9 if "confirmed and verified" in flat else 0.1)
    df = run_for_scorer(
        s, [_prop(i, False) for i in range(8)], builder=build_channel_conditions, padding_side="left"
    )
    r = channel_report(df)
    for k in ("commit", "source_self", "query", "isolate", "source_user", "source_system"):
        assert k in r
    # the three source arms (+ commit) all carry the factive phrase -> all flip the false items
    assert r["source_self"]["flip_rate"] == 1.0 and r["source_user"]["flip_rate"] == 1.0
    assert r["isolate"]["flip_rate"] == 0.0
    # the paired channel contrasts are computed; with the stub all sources read identically -> ~0
    assert "_contrasts" in r and "source_self_minus_source_user" in r["_contrasts"]
    assert abs(r["_contrasts"]["source_self_minus_source_user"]["estimate"]) < 1e-6
    assert abs(r["_contrasts"]["source_self_minus_source_system"]["estimate"]) < 1e-6


def test_real_qwen_offsets_position_clean_for_source_arms():
    tok = _qwen_tokenizer()
    tokenize = lambda s: tok.encode(s, add_special_tokens=False).ids
    p = _prop(5, False)
    cts = build_channel_conditions(p)
    target = max(len(tokenize(c.middle)) for c in cts)
    source_arms = [c for c in cts if c.framing in ("self", "user", "system")]
    predicate_ids = tokenize(" " + _FACTIVE.format(c=p.claim))
    claim_ids = tokenize(" " + p.claim)

    predicate_offsets = []
    claim_to_probe_distances = []
    for c in source_arms:
        mid = pad_to_match(tokenize, c.middle, target, filler=" note", side="left")
        mid_ids = tokenize(mid)
        full_ids = tokenize(f"{mid}\nAnswer:")
        predicate_start = _find_subsequence(mid_ids, predicate_ids)
        claim_start = _find_subsequence(full_ids, claim_ids)
        assert predicate_start >= 0, c.middle
        assert claim_start >= 0, c.middle
        predicate_offsets.append(predicate_start)
        claim_to_probe_distances.append(len(full_ids) - claim_start)

    assert len(set(predicate_offsets)) == 1
    assert len(set(claim_to_probe_distances)) == 1
