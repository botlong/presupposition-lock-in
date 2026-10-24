"""Faithfulness classification for self-generated commitment actions.

The strict criterion is verbatim containment of the full original claim (case-insensitive).
It is deterministic and auditable from the retained `gen` column alone: paraphrases and
rewrites count as unfaithful even when semantically equivalent, so the faithful-subset
effect is estimated on generations whose presupposed content is exactly the tested claim.
A lexical keyword/negation heuristic is not acceptable here: it passes negated recordings
whose negation word it does not list (e.g. "cannot") and content rewrites that keep enough
keywords, and it rejects faithful recordings of claims that themselves contain a negation.
"""
from __future__ import annotations


def strict_faithful(gen: str, claim: str) -> bool:
    """True iff the generation contains the complete original claim verbatim (case-insensitive)."""
    return claim.lower() in gen.lower()
