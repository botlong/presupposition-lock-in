from __future__ import annotations
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Proposition:
    id: str
    claim: str
    gold: bool
    unrelated_claim: str


def load_propositions(path: str) -> list[Proposition]:
    props: list[Proposition] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            props.append(Proposition(d["id"], d["claim"], bool(d["gold"]), d["unrelated_claim"]))
    return props
