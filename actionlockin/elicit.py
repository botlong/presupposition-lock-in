from __future__ import annotations
from actionlockin.propositions import Proposition
from actionlockin.conditions import ConditionText, belief_probe, SYSTEM
from actionlockin.scorer import Scorer


def prob_of_P(scorer: Scorer, prop: Proposition, ct: ConditionText) -> float:
    return scorer.prob_true(SYSTEM, belief_probe(prop), ct.middle)


def parse_tf_answer(text: str) -> str:
    """Map a decoded answer continuation to 'true' | 'false' | 'other'.

    Only the first word after the cue counts as the answer; scanning further would credit
    an answer token the continuation merely mentions rather than commits to.
    """
    words = text.strip().split()
    if not words:
        return "other"
    w = words[0].strip(".,!?:;\"'`").lower()
    return w if w in ("true", "false") else "other"
