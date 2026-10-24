from __future__ import annotations
from actionlockin.propositions import Proposition
from actionlockin.conditions import belief_probe, SYSTEM
from actionlockin.scorer import Scorer


def baseline_prob_true(scorer: Scorer, prop: Proposition) -> float:
    return scorer.prob_true(SYSTEM, belief_probe(prop), "")


def filter_midband(scorer: Scorer, props: list[Proposition],
                   low: float, high: float) -> list[Proposition]:
    return [p for p in props if low <= baseline_prob_true(scorer, p) <= high]
