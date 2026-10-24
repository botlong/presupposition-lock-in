from __future__ import annotations
import math
from typing import Protocol, Callable, runtime_checkable
from actionlockin.conditions import ANSWER_CUE


@runtime_checkable
class Scorer(Protocol):
    name: str
    def tokenize(self, text: str) -> list: ...
    def prob_true(self, system: str, user: str, assistant: str) -> float: ...
    def logit_diff(self, system: str, user: str, assistant: str) -> float: ...
    def middle_logprobs(self, system: str, user: str, assistant: str) -> list[float]: ...
    def prob_true_chat(self, messages: list) -> float: ...
    def logit_diff_chat(self, messages: list) -> float: ...


class StubScorer:
    def __init__(self, name: str, rule: Callable[[str], float]):
        self.name = name
        self._rule = rule

    def tokenize(self, text: str) -> list:
        return text.split()

    def prob_true(self, system: str, user: str, assistant: str) -> float:
        flat = f"{system}\n{user}\n{assistant}"
        return min(1.0, max(0.0, float(self._rule(flat))))

    def logit_diff(self, system: str, user: str, assistant: str) -> float:
        # log-odds consistent with prob_true (monotone), clamped to avoid +/-inf at 0/1
        p = self.prob_true(system, user, assistant)
        eps = 1e-6
        p = min(1.0 - eps, max(eps, p))
        return math.log(p / (1.0 - p))

    def middle_logprobs(self, system: str, user: str, assistant: str) -> list[float]:
        return [-1.0 for _ in assistant.split()]

    def prob_true_chat(self, messages: list) -> float:
        flat = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return min(1.0, max(0.0, float(self._rule(flat))))

    def logit_diff_chat(self, messages: list) -> float:
        p = self.prob_true_chat(messages)
        eps = 1e-6
        p = min(1.0 - eps, max(eps, p))
        return math.log(p / (1.0 - p))


class HFScorer:
    """Real model. Not unit-tested (needs weights/GPU); exercised in smoke tasks.

    Renders prompts with the model's native chat template so the injected middle is a
    genuine assistant turn (the 'reads back its own action' mechanism the claim is about).
    """
    def __init__(
        self,
        name: str,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        cache_dir: str | None = None,
        revision: str | None = None,  # Pinning prevents upstream weight changes from altering reruns.
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=getattr(torch, dtype), device_map=device_map,
            cache_dir=cache_dir, revision=revision)
        self.model.eval()
        self._torch = torch
        self._true_id = self._first_token_id(" True")
        self._false_id = self._first_token_id(" False")

    def _input_ids(self, encoded):
        if isinstance(encoded, dict):
            encoded = encoded["input_ids"]
        elif hasattr(encoded, "input_ids"):
            encoded = encoded.input_ids
        if not hasattr(encoded, "to"):
            encoded = self._torch.tensor(encoded, dtype=self._torch.long)
        if encoded.ndim == 1:
            encoded = encoded.unsqueeze(0)
        return encoded

    def _first_token_id(self, s: str) -> int:
        return self.tok(s, add_special_tokens=False)["input_ids"][0]

    def tokenize(self, text: str) -> list:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def _prefix_ids(self, system: str, user: str):
        # render [system, user] and OPEN the assistant turn (native special tokens, single BOS)
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            rendered = self.tok.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            return self._input_ids(rendered)
        except Exception:
            # some templates (e.g. Mistral-v0.3) reject a system role -> fold it into the user turn
            msgs = [{"role": "user", "content": f"{system}\n\n{user}"}]
            rendered = self.tok.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            return self._input_ids(rendered)

    def _tf_logits(self, system: str, user: str, assistant: str):
        """Raw (logit_True, logit_False) at the answer position after the assistant middle."""
        torch = self._torch
        pre = self._prefix_ids(system, user).to(self.model.device)
        # continuation = the assistant's own middle, then the fixed answer cue
        cont = self.tok(f"{assistant}\n{ANSWER_CUE}", add_special_tokens=False,
                        return_tensors="pt")["input_ids"].to(self.model.device)
        ids = torch.cat([pre, cont], dim=1)
        with torch.no_grad():
            logits = self.model(ids).logits[0, -1]
        return logits[self._true_id].item(), logits[self._false_id].item()

    def prob_true(self, system: str, user: str, assistant: str) -> float:
        lt, lf = self._tf_logits(system, user, assistant)
        m = max(lt, lf)
        et, ef = pow(2.718281828, lt - m), pow(2.718281828, lf - m)
        return et / (et + ef)

    def logit_diff(self, system: str, user: str, assistant: str) -> float:
        # primary belief readout: log-odds of True vs False (unbounded -> no 0/1 ceiling)
        lt, lf = self._tf_logits(system, user, assistant)
        return lt - lf

    def middle_logprobs(self, system: str, user: str, assistant: str) -> list[float]:
        torch = self._torch
        pre = self._prefix_ids(system, user).to(self.model.device)
        mid = self.tok(assistant, add_special_tokens=False,
                       return_tensors="pt")["input_ids"].to(self.model.device)
        ids = torch.cat([pre, mid], dim=1)
        with torch.no_grad():
            logits = self.model(ids).logits[0]
        logprobs = torch.log_softmax(logits[:-1], dim=-1)   # predicts ids[1:]
        tgt = ids[0, 1:]
        per_tok = logprobs.gather(1, tgt.unsqueeze(1)).squeeze(1)
        n_pre = pre.shape[1]
        # middle tokens are ids[n_pre:]; predicted by per_tok[n_pre-1:]
        return per_tok[n_pre - 1:].tolist()

    # ---- multi-turn (for the downstream-harm experiment: commit -> evidence -> decide) ----
    def _render_chat(self, messages: list):
        """Render an arbitrary [{'role','content'}...] chat, OPENING a final assistant turn."""
        try:
            return self._input_ids(self.tok.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"))
        except Exception:
            # templates that reject a system role -> fold it into the first user turn
            msgs = [dict(m) for m in messages]
            if msgs and msgs[0]["role"] == "system":
                sys = msgs.pop(0)["content"]
                for i, m in enumerate(msgs):
                    if m["role"] == "user":
                        msgs[i] = {"role": "user", "content": f"{sys}\n\n{m['content']}"}
                        break
            return self._input_ids(self.tok.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"))

    def _tf_logits_chat(self, messages: list):
        torch = self._torch
        pre = self._render_chat(messages).to(self.model.device)
        cont = self.tok(ANSWER_CUE, add_special_tokens=False,
                        return_tensors="pt")["input_ids"].to(self.model.device)
        ids = torch.cat([pre, cont], dim=1)
        with torch.no_grad():
            logits = self.model(ids).logits[0, -1]
        return logits[self._true_id].item(), logits[self._false_id].item()

    def prob_true_chat(self, messages: list) -> float:
        lt, lf = self._tf_logits_chat(messages)
        m = max(lt, lf)
        et, ef = pow(2.718281828, lt - m), pow(2.718281828, lf - m)
        return et / (et + ef)

    def logit_diff_chat(self, messages: list) -> float:
        lt, lf = self._tf_logits_chat(messages)
        return lt - lf
