# results/ — canonical artifacts and provenance

Every number in the paper recomputes from the **canonical** per-proposition
parquets below (checked by `python scripts/verify_paper_numbers.py`, one PASS/FAIL line per
transcribed claim — the script exits non-zero on any failure or missing input — plus
`python scripts/reconcile_gold.py --check` for gold-label consistency). Canonical files are
committed; superseded/aux files stay local-only and are listed so no one re-derives paper
numbers from the wrong run.

## Canonical (committed; the artifacts of record)

| Files | Producer | Run |
|---|---|---|
| `gate_Qwen2.5-{1.5B,7B,32B,72B}_{propositions,arith_props,arith_hard}.parquet` | `scripts/run_gpu_all.py` | 2026-06-24 batch, rented 2×RTX PRO 6000; corrected gold labels (p030/a151/a161) |
| `gate_Mistral-7B-Instruct-v0.3.parquet` | `scripts/gate_multimodel.py` | 2026-06-18 local (Mistral's only run; predates the label fix — its 120-prop set is unaffected by the arith fixes; p030 handled by `reconcile_gold`) |
| `channel_Qwen2.5-*_{propositions,arith_hard}.parquet` | `scripts/run_gpu_all.py` | 2026-06-24 batch (source-attribution arms) |
| `mitigation_Qwen2.5-*_arith_hard.parquet` | `scripts/run_gpu_all.py` / `scripts/mitigation_gate.py` | 2026-06-24 batch |
| `selfgen_Qwen2.5-*.parquet` | `scripts/selfgen_variant.py` | self-generated-commitment variant. The stored `faithful` column is the superseded keyword heuristic; the paper and `verify_paper_numbers.py` re-derive faithfulness from `gen` with the strict verbatim criterion (`actionlockin/selfgen.py`) |
| `gate_Qwen2.5-{1.5B,7B}-Instruct.parquet` | `scripts/gate_multimodel.py` | 2026-06-18 small-model gate — committed **only** as provenance for the cross-run agreement claim (max \|Δ\| = 0.025 log-odds); do not quote elsewhere |
| `probe_invariance_Qwen2.5-{1.5B,7B}.parquet` | `scripts/probe_invariance.py` | 2026-07-08 local (3 probe wordings) |
| `downstream_harm_{Qwen2.5-1.5B,Qwen2.5-7B,Mistral-7B}-Instruct*.parquet` | `scripts/downstream_harm.py` | 2026-07-08 local (per-item rows of the three-turn episodes) |
| `downstream_harm_Qwen2.5-{32B,72B}.parquet` | `scripts/downstream_harm.py` | 2026-07-13 rented box (2×RTX PRO 6000 Blackwell, bf16), gated batch `scripts/run_crossturn_box.sh` |
| `fluency_matched_{Qwen2.5-1.5B,Qwen2.5-7B,Mistral-7B}-Instruct*.parquet` | `scripts/fluency_matched_h3.py` | 2026-07-11 local (richer-query control; reproduces the 2026-07-08 logged aggregates bit-identically) |
| `verbalized_Qwen2.5-1.5B-Instruct.parquet` | `scripts/verbalized_control.py` | 2026-07-11 local (verbalized-confidence control) |
| `crossturn_gate_{Qwen2.5-1.5B,Qwen2.5-7B,Mistral-7B}-Instruct*.parquet` | `scripts/crossturn_gate.py` | 2026-07-11 local — belief grid with the commitment as a **closed** assistant message, probe-first (A) and act-first (B) protocols, plus a native structured `tool_calls` arm on the Qwen models (not length-matched; `length_matched` column flags it) |
| `crossturn_gate_Qwen2.5-{32B,72B}.parquet` | `scripts/crossturn_gate.py` | 2026-07-13 rented box — same cross-turn design as the local runs, through the gated batch (`scripts/run_crossturn_box.sh`: render gate, snapshot recording, 3-item pilot, row-count asserts) |
| `sampledread_Qwen2.5-{7B,72B}_arith_hard.parquet` | `scripts/sampled_readout.py` | 2026-07-27 rented box (2×RTX PRO 6000 Blackwell, bf16), gated batch `scripts/run_confirm_box.sh` — greedy + 20 temperature-0.7 samples per cell alongside the forced-token readout; the batch's weight spot-check reproduced the 2026-06-24 gate cells at max \|Δlogodds\| = 0.000 |
| `crossturn_mit_Qwen2.5-{32B,72B}_{propositions,arith_hard}.parquet` | `scripts/crossturn_mitigation.py` | 2026-07-27 rented box, same gated batch — cross-turn (closed assistant message) mitigation grid, protocols A/B, five cells (none / commit3 / three mitigations) |
| `downstream_harm_Qwen2.5-72B_arith_hard.parquet` | `scripts/downstream_harm.py --props data/arith_hard.jsonl` | 2026-07-27 rented box, same gated batch — hard-arithmetic recovery arm of the three-turn harm episodes |
| `{gate,channel,mitigation}_Llama-3.1-8B_*.parquet` (`gate_*` on propositions, arith_props, arith_hard) | `scripts/run_gpu_all.py --tag Llama-3.1-8B` | 2026-10-22 local, single RTX 4090, bf16 — third-family replication, added after the pre-registered runs |
| `{gate,channel,mitigation}_Llama-3.1-8B_arith_hard_llama.parquet` | `scripts/run_llama_matched.py` | 2026-10-22, same machine — the same grids on the Llama-matched set `data/arith_hard_llama.jsonl` (`scripts/gen_arith_llama.py final a33 20261022`, byte-identical) |
| `pilot_Llama-3.1-8B_classes.txt` | `scripts/gen_arith_llama.py pilot` | 2026-10-22 recorded output of the eleven-class baseline pilot that chose the a33 class |
| `model_snapshots.jsonl` | `scripts/record_snapshot.py` | content-addressed manifests (sha256 of config/weight-index/tokenizer files; entries recorded from 2026-07-27 on also hash every weight shard) recorded at run time; the ModelScope local-dir models are identified by manifest because that cache exposes no VCS revision |

## Model provenance (paper runs)

| Model | Weights source | Snapshot |
|---|---|---|
| Qwen/Qwen2.5-1.5B-Instruct | HF cache `models/huggingface` | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Qwen/Qwen2.5-7B-Instruct | HF cache | `a09a35458c702b33eeacc393d103063234e8bc28` |
| mistralai/Mistral-7B-Instruct-v0.3 | HF cache | `c170c708c41dac9275d15a8fff4eca08d52bab71` |
| Qwen/Qwen2.5-32B-Instruct | ModelScope mirror on the rented box (2026-06-24 single-turn batch) | commit not recorded by that cache |
| Qwen/Qwen2.5-72B-Instruct | ModelScope mirror on the rented box (2026-06-24 single-turn batch) | commit not recorded by that cache |
| Qwen/Qwen2.5-32B-Instruct | ModelScope download, rented box (2026-07-13 cross-turn + downstream) | sha256 manifest in `model_snapshots.jsonl` |
| Qwen/Qwen2.5-72B-Instruct | ModelScope download, rented box (2026-07-13 cross-turn + downstream) | sha256 manifest in `model_snapshots.jsonl` |
| meta-llama/Llama-3.1-8B-Instruct | ModelScope mirror, local (2026-10-22) | run with PyTorch 2.14.1 (CUDA 12.6), Transformers 5.12.1; the `model` column holds the local path `D:\\alock\\models\\Llama-3.1-8B-Instruct` |
| Qwen/Qwen2.5-{7B,32B,72B}-Instruct | ModelScope download verified against the manifests above, disk-copied to the rented box (2026-07-27 confirmation batch) | shard-level sha256 manifest in `model_snapshots.jsonl`; in-batch weight spot-check vs the 2026-06-24 gate cells: max \|Δlogodds\| = 0.000 |

Note: the `model` column of the rented-box parquets holds the box-local path
(`/root/autodl-tmp/.../Qwen2___5-72B-Instruct` for 2026-06-24,
`/root/autodl-tmp/Qwen2.5-{32B,72B}-Instruct` for 2026-07-13), which **is**
`Qwen/Qwen2.5-{32B,72B}-Instruct`; the parquets are kept byte-identical to the run rather
than rewritten. `HFScorer` accepts a `revision=` argument to pin snapshots on re-runs.

## Superseded / aux (local-only, kept for provenance — do NOT quote in the paper)

- `gate_multimodel.parquet` — combined 2026-06-18 small-model gate output (pre-label-fix); its
  per-model halves are the committed `gate_*-Instruct.parquet` provenance files above.
- `smoke.parquet`, `runs.parquet` (if present) — stub/pilot outputs.
- `superseded/*.log` — pre-goldfix run logs showing `n_false=53`; the corrected sets have
  `n_false=54` (propositions). Moved out of the top level so they cannot be mistaken for
  current evidence. Current dated rerun logs (`*_rerun-2026-07-08.log`, `probe_invariance_*.log`)
  match the paper.
- The fluency-matched and verbalized controls retain per-item parquets (canonical table above);
  their dated run logs are kept locally as secondary evidence of the printed aggregates.

## Public tau2-bench census

- `tau2_argument_census.json` — argument-type census of the public tau2-bench trajectories
  (sierra-research/tau2-bench@d8e915f, `data/tau2/results/final/`, default runs of airline,
  retail and telecom; base runs for gpt-4.1-mini). Produced by
  `python scripts/tau2_argument_census.py <tau2-bench>/data/tau2/results/final`; the trajectories
  themselves are not redistributed here. Cited in the Limitations section.

## Data files

`data/*.jsonl` are the **artifacts of record** (0 label errors by `tests/test_data_labels.py` and
`reconcile_gold.py --check`). `scripts/gen_arith.py` regenerates a *fresh* correct set, not the committed one — re-running does
not byte-reproduce `arith_props.jsonl` (the committed file predates a percent-branch fix that
shifted the RNG draws, and a 2026-07-07 dedup, 200→196).
