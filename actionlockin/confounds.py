from __future__ import annotations
from actionlockin.scorer import Scorer


def text_features(scorer: Scorer, system: str, user: str, middle: str, claim: str) -> dict:
    lps = scorer.middle_logprobs(system, user, middle) if middle else []
    n = len(lps)
    mean = (sum(lps) / n) if n else 0.0
    if claim and middle and claim in middle:
        salience = (middle.rfind(claim) + len(claim)) / len(middle)
    else:
        salience = 0.0
    return {"n_tokens": n, "mean_logprob": mean, "claim_salience": salience}
