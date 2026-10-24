# Presupposition Lock-In

Code, data and results for the paper:

> **Presupposition Lock-In: Verified-Sounding Tool Calls Shift LLM Agents Toward Unchecked Claims
> Before Results Return**
> Hua Long, Haolong Shi, Chongyang Dai (Chongqing University of Technology)

## What this studies

An LLM agent acts by emitting tool calls whose arguments often *presuppose* facts — e.g.
`log_fact(fact="the venue is in Berlin", status="confirmed")`. We measure whether emitting such a call
shifts the agent's belief in the presupposed proposition **P before any tool result returns**, read as
the log-odds `logit(True) − logit(False)` of a fixed True/False probe rendered in each model's native
chat template (teacher-forced assistant turn).

Models: Qwen2.5-1.5B/7B/32B/72B-Instruct, Mistral-7B-Instruct-v0.3 and Llama-3.1-8B-Instruct (six open
models, three families). Pure inference, no training.

### Key findings

- **Pre-result shift.** A presupposing commitment raises pre-result belief in P on every model, exceeds
  a non-presuppositional query, and grows with commitment strength; these survive Bonferroni correction.
- **Content, not form (within a turn).** A content-matched declarative assertion shifts belief more
  than the tool call, so the driver is the presupposed content rather than action syntax. Our
  pre-registered headline hypothesis (H2 > 0) was **refuted** and is reported as such. Re-read as a
  completed earlier turn, the call can outweigh the assertion at 32B and 72B.
- **Decision-level harm.** Where the model is uncertain, the shift flips correct rejections of false
  claims (76% at Qwen2.5-72B, 36% at Llama-3.1-8B on uncertainty-matched arithmetic sets). The bias is
  pre-evidential and recovers under corrective evidence.
- **Mitigation, checked on both sides.** Within a turn, withholding the presupposing argument until the
  result returns removes most flips at Qwen2.5-72B while wrongly rejecting 3 of 31 true claims. On the
  smaller models and Llama, no representation tested stops false acceptances without also rejecting
  true claims.

## Repository layout

```
actionlockin/        analysis package: belief readout, contrasts, mitigation/channel reports
scripts/             experiment entry points and paper-number verification
data/                claim sets with gold labels (artifacts of record; see results/README.md)
results/             per-proposition result parquets + manifest results/README.md
tests/               unit tests (pytest; no GPU needed)
preregistration.md   pre-registration v1 → v2 trail + dated completion addendum
```

## Install & run

Python ≥ 3.11. From the repo root:

```
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"                  # core analysis + pytest (CPU-only)
pytest
python scripts/verify_paper_numbers.py   # recompute the paper's numbers from results/*.parquet
python scripts/reconcile_gold.py --check # gold-label consistency gate
```

Model-running scripts additionally need `pip install -e ".[gpu]"`. Exact package versions are pinned in
`requirements-lock.txt`, and model snapshot revisions are recorded in `results/README.md`.
The canonical per-model batch is `scripts/run_gpu_all.py`.

## License

MIT (see `LICENSE`).
