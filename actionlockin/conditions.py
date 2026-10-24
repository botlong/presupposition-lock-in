from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Literal
from actionlockin.propositions import Proposition

SYSTEM = "You are a careful assistant. You can record information with tools, then answer the question."
ANSWER_CUE = "Answer:"  # Kept outside the probe so every condition shares the same answer position.

# Experimental condition invariants
# Commitment text files P as a premise rather than searching for proof, because proof-seeking
# language adds an uncertainty implicature unrelated to the tested factive strength. Result text
# is absent because any returned observation would contaminate the pre-result readout. The
# assertion reuses dose 3's factive wording so H2 varies form without strengthening the claim
# enough to saturate the True/False readout. Only the query arm is non-presuppositional.
_COMMIT = {
    1: 'log_fact(fact="{c}")',
    2: 'log_fact(fact="{c}", status="established")',
    3: 'log_fact(fact="{c}", status="confirmed and verified")',
}
_QUERY = 'check(query="whether {c} is true")'        # dose 0, non-presuppositional info-seeking action
_ASSERTION = "It is confirmed and verified that {c}."  # declarative match to dose-3's factive claim


@dataclass(frozen=True)
class ConditionText:
    prop_id: str
    condition: str   # none | assertion | action
    framing: str     # na | commitment | query | placebo | user | system | self | mit_*
    dose: int
    middle: str


def belief_probe(prop: Proposition) -> str:
    # user question only; the answer cue is appended by the scorer inside the assistant turn
    return (
        f'Question: Is the following claim true? Claim: "{prop.claim}"\n'
        f"Answer with a single word, True or False."
    )


def build_conditions(prop: Proposition) -> list[ConditionText]:
    c = prop.claim
    out = [
        ConditionText(prop.id, "none", "na", 0,
                      "Let me consider this claim carefully before answering."),
        # assertion content-matched to dose-3's factive force ("confirmed and verified"),
        # declarative (non-action) form, P stated once. NOT a maximal "proven true" (would saturate).
        ConditionText(prop.id, "assertion", "na", 0,
                      _ASSERTION.format(c=c)),
        # query: a non-presuppositional info-seeking action (does not file P as a fact)
        ConditionText(prop.id, "action", "query", 0,
                      _QUERY.format(c=c)),
        # placebo: a dose-3-strength commitment action on an UNRELATED claim
        ConditionText(prop.id, "action", "placebo", 0,
                      _COMMIT[3].format(c=prop.unrelated_claim)),
    ]
    # commitment: premise-style action filing P as an established fact, graded factive; NO result line
    for d, tmpl in _COMMIT.items():
        out.append(ConditionText(prop.id, "action", "commitment", d, tmpl.format(c=c)))
    return out


PaddingSide = Literal["right", "left"]


def pad_to_match(tokenize: Callable[[str], list], text: str, target_n: int,
                 filler: str = " Note this.", side: PaddingSide = "right") -> str:
    """Length-match text by appending or prepending filler.

    The default is append/right padding for the gate and mitigation experiments. The
    channel/source experiment opts into left padding so its substantive sentence stays
    right-anchored against the answer cue.
    """
    if side not in ("right", "left"):
        raise ValueError("side must be 'right' or 'left'")
    out = text
    while len(tokenize(out)) < target_n:
        if side == "right":
            out += filler
        else:
            out = filler + out
    return out


# Mitigation invariants
# Each mitigation must preserve evidence that an action occurred while preventing an unverified
# argument from entering the belief context as fact. The arms vary how strongly they enforce that
# separation without dropping the action record:
#   * disclaim — keep the call verbatim, append a framework guard tagging it unverified/pending;
#   * reframe  — log the same claim but with a NON-factive status (hypothesis pending result);
#   * isolate  — record that an action occurred but withhold its presupposing argument.
_MIT = {
    "disclaim": ('log_fact(fact="{c}", status="confirmed and verified")  '
                 "# UNVERIFIED self-entry: no result has confirmed this; "
                 "treat the claim as a pending hypothesis, not a fact."),
    "reframe": 'log_fact(fact="{c}", status="unverified hypothesis, pending result")',
    "isolate": "log_fact(<argument withheld from the reasoning context until the tool result returns>)",
}


def build_mitigation_conditions(prop: Proposition) -> list[ConditionText]:
    """none + the harmful dose-3 commitment baseline + three mitigated representations of it.
    Each mitigation keeps the action on record but blocks the factive presupposition from being
    re-read as belief. Compare each `mit_*` cell's belief delta vs none against the commit3 delta."""
    c = prop.claim
    out = [
        ConditionText(prop.id, "none", "na", 0,
                      "Let me consider this claim carefully before answering."),
        ConditionText(prop.id, "action", "commitment", 3, _COMMIT[3].format(c=c)),
    ]
    for name, tmpl in _MIT.items():
        out.append(ConditionText(prop.id, "action", f"mit_{name}", 3, tmpl.format(c=c)))
    return out


# Source-attribution invariants
# Source arms must hold illocutionary form fixed so their contrast identifies the named source
# rather than wording. They therefore use the same impersonal record and factive predicate,
# varying only the source label:
#   "On record from {source}: it is confirmed and verified that P."
# Left padding right-anchors the substantive sentence against the answer cue; filler after the
# claim would otherwise make source arms differ in claim-to-probe distance.
_FACTIVE = "confirmed and verified that {c}"   # constant predicate; only the subject/source changes
_CHANNEL_SOURCE_TEMPLATE = "On record from {source}: it is " + _FACTIVE + "."


def build_channel_conditions(prop: Proposition) -> list[ConditionText]:
    c = prop.claim
    return [
        ConditionText(prop.id, "none", "na", 0,
                      "Let me consider this claim carefully before answering."),
        ConditionText(prop.id, "action", "commitment", 3, _COMMIT[3].format(c=c)),     # self, tool-call form
        ConditionText(prop.id, "assertion", "self", 0,
                      _CHANNEL_SOURCE_TEMPLATE.format(source="the assistant (self)", c=c)),
        ConditionText(prop.id, "action", "query", 0, _QUERY.format(c=c)),              # self, non-presup
        ConditionText(prop.id, "action", "mit_isolate", 3, _MIT["isolate"].format(c=c)),  # self, arg withheld
        ConditionText(prop.id, "assertion", "user", 0,
                      _CHANNEL_SOURCE_TEMPLATE.format(source="the user", c=c)),
        ConditionText(prop.id, "assertion", "system", 0,
                      _CHANNEL_SOURCE_TEMPLATE.format(source="the system", c=c)),
    ]
