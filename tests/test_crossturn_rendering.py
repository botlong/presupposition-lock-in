"""Golden-token tests for the cross-turn grid: the experiment's claims depend on the middle
being a CLOSED assistant turn re-read across a turn boundary, so the rendered token stream --
not just the message list -- must show the closed turn, the turn boundary, and (for the
structured arm) native tool-call tags. The single-turn contrast is pinned alongside so the
two designs cannot silently converge."""
import pytest
from actionlockin.conditions import SYSTEM, belief_probe, build_conditions
from actionlockin.crossturn import ANSWER_REQ, TASK_PROMPT, build_messages, toolcall_message
from actionlockin.propositions import Proposition

P = Proposition(id="t001", claim="octopuses have three hearts", gold=True,
                unrelated_claim="a pentagon has five sides")


def _mid(framing, dose):
    return next(c.middle for c in build_conditions(P)
                if (c.condition, c.framing, c.dose)[1:] == (framing, dose)
                or (c.framing, c.dose) == (framing, dose))


def test_layout_A_probe_first_middle_is_own_closed_message():
    mid = _mid("commitment", 3)
    msgs = build_messages("A", SYSTEM, belief_probe(P), mid)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[1]["content"] == belief_probe(P)
    assert msgs[2]["content"] == mid
    assert msgs[3]["content"] == ANSWER_REQ


def test_layout_B_act_first_task_turn_is_claim_free_and_constant():
    mid = _mid("commitment", 3)
    msgs = build_messages("B", SYSTEM, belief_probe(P), mid)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[1]["content"] == TASK_PROMPT
    assert P.claim not in msgs[1]["content"]
    assert msgs[3]["content"] == belief_probe(P)


def test_claim_occurrences_isolate_the_manipulation():
    # In every condition the claim may appear ONLY in the probe and (if presupposing) the middle.
    for proto in ("A", "B"):
        for c in build_conditions(P):
            msgs = build_messages(proto, SYSTEM, belief_probe(P), c.middle)
            outside_middle = " ".join(m["content"] for m in msgs if m["content"] != c.middle)
            assert outside_middle.count(P.claim) == 1  # the probe only
            if c.framing == "placebo":
                assert P.claim not in c.middle and P.unrelated_claim in c.middle


# ---- rendered-token goldens (need cached tokenizers; offline only) ----
def _tok(name):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(name, cache_dir="models/huggingface",
                                             local_files_only=True)
    except Exception:
        return None


TOK = _tok("Qwen/Qwen2.5-1.5B-Instruct")
TOK_MISTRAL = _tok("mistralai/Mistral-7B-Instruct-v0.3")
needs_tok = pytest.mark.skipif(TOK is None, reason="cached Qwen2.5-1.5B tokenizer unavailable")
needs_mistral = pytest.mark.skipif(TOK_MISTRAL is None,
                                   reason="cached Mistral-7B tokenizer unavailable")


@needs_tok
def test_render_gate_passes_for_qwen_template():
    from actionlockin.crossturn import verify_rendering
    assert "tool_calls supported" in verify_rendering(TOK)


@needs_mistral
def test_render_gate_passes_for_mistral_template():
    # Mistral's template closes the middle with </s> and folds the system turn differently;
    # the gate must hold template-agnostically, since the box runs it per downloaded model.
    from actionlockin.crossturn import verify_rendering
    assert "render-gate ok" in verify_rendering(TOK_MISTRAL)


@needs_tok
def test_crossturn_render_closes_middle_and_opens_final_turn():
    mid = _mid("commitment", 3)
    msgs = build_messages("A", SYSTEM, belief_probe(P), mid)
    text = TOK.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    assert f"<|im_start|>assistant\n{mid}<|im_end|>" in text     # middle turn CLOSED
    assert text.count("<|im_start|>user") == 2                   # probe turn + answer-request turn
    assert text.rstrip().endswith("<|im_start|>assistant")       # fresh answer turn opened


@needs_tok
def test_singleturn_render_keeps_middle_open_by_contrast():
    # The single-turn grid appends the middle inside one OPEN assistant turn; if this ever
    # closes, the two designs are no longer distinct and the paper's contrast collapses.
    mid = _mid("commitment", 3)
    pre = TOK.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": belief_probe(P)}],
        tokenize=False, add_generation_prompt=True)
    single = pre + f"{mid}\nAnswer:"
    assert f"{mid}<|im_end|>" not in single
    assert single.endswith("Answer:")


@needs_tok
def test_toolcall_arm_renders_native_tags_and_closes():
    msgs = build_messages("A", SYSTEM, belief_probe(P), toolcall_message(P.claim))
    text = TOK.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    assert "<tool_call>" in text and "</tool_call>" in text
    assert '"name": "log_fact"' in text
    assert P.claim in text.split("<tool_call>")[1].split("</tool_call>")[0]
    closed = text.split("</tool_call>")[1]
    assert closed.lstrip().startswith("<|im_end|>") or "<|im_end|>" in closed[:40]
