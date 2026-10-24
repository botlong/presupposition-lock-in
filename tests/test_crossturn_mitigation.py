"""The cross-turn mitigation grid reuses the closed-turn episode: every mitigation cell must
keep the claim confined to the probe (plus, for presupposing cells, the middle itself), and
each middle must render as a CLOSED assistant turn, or the mitigation contrast silently
degrades to the single-turn design."""
import pytest
from actionlockin.conditions import SYSTEM, belief_probe, build_mitigation_conditions
from actionlockin.crossturn import build_messages
from actionlockin.propositions import Proposition

P = Proposition(id="t001", claim="octopuses have three hearts", gold=True,
                unrelated_claim="a pentagon has five sides")


def test_mitigation_cells_and_labels():
    cts = build_mitigation_conditions(P)
    assert [(c.condition, c.framing, c.dose) for c in cts] == [
        ("none", "na", 0), ("action", "commitment", 3), ("action", "mit_disclaim", 3),
        ("action", "mit_reframe", 3), ("action", "mit_isolate", 3)]


def test_claim_occurrences_isolate_the_manipulation():
    for proto in ("A", "B"):
        for c in build_mitigation_conditions(P):
            msgs = build_messages(proto, SYSTEM, belief_probe(P), c.middle)
            assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
            outside_middle = " ".join(m["content"] for m in msgs if m["content"] != c.middle)
            assert outside_middle.count(P.claim) == 1  # the probe only
            if c.framing == "mit_isolate":
                assert P.claim not in c.middle          # argument withheld


# ---- rendered-token golden (needs cached tokenizer; offline only) ----
def _tok(name):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(name, cache_dir="models/huggingface",
                                             local_files_only=True)
    except Exception:
        return None


TOK = _tok("Qwen/Qwen2.5-1.5B-Instruct")
needs_tok = pytest.mark.skipif(TOK is None, reason="cached Qwen2.5-1.5B tokenizer unavailable")


@needs_tok
def test_mitigation_middles_render_closed():
    for c in build_mitigation_conditions(P):
        for proto in ("A", "B"):
            msgs = build_messages(proto, SYSTEM, belief_probe(P), c.middle)
            text = TOK.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            assert f"<|im_start|>assistant\n{c.middle}<|im_end|>" in text
            assert text.rstrip().endswith("<|im_start|>assistant")
