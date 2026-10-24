"""Census of tool-call argument types in the public tau2-bench trajectories.

Counts how often agents in natural (unperturbed) tau2-bench runs put free text, rather than
identifiers, numbers or enums, into tool-call arguments, and which tools those arguments belong to.
It is a structural count with a fixed rule, not a judgment of content.

Input: the trajectory files released with tau2-bench (sierra-research/tau2-bench at commit
d8e915f, data/tau2/results/final/), restricted to the default runs of the airline, retail and
telecom domains (base runs for gpt-4.1-mini, which has no default run).

Usage (from repo root):
    python scripts/tau2_argument_census.py /path/to/tau2-bench/data/tau2/results/final
Writes results/tau2_argument_census.json.
"""
from __future__ import annotations
import collections
import glob
import json
import os
import sys

MIN_WORDS = 4  # a string argument with at least this many words counts as free text


def is_free_text(v) -> bool:
    return isinstance(v, str) and len(v.split()) >= MIN_WORDS


def select_files(root: str) -> list[str]:
    files = []
    for f in sorted(glob.glob(os.path.join(root, "*.json"))):
        name = os.path.basename(f)
        if "workflow" in name:
            continue
        if "_default_" in name or name.startswith("gpt-4.1-mini") and "_base_" in name:
            files.append(f)
    return files


def main(root: str) -> dict:
    files = select_files(root)
    n_sims = n_calls = n_free_calls = 0
    free_by_tool = collections.Counter()
    calls_by_model = collections.Counter()
    for f in files:
        model = os.path.basename(f).split("_")[0]
        for sim in json.load(open(f))["simulations"]:
            n_sims += 1
            for m in sim["messages"]:
                if m.get("role") != "assistant":
                    continue
                for tc in m.get("tool_calls") or []:
                    n_calls += 1
                    calls_by_model[model] += 1
                    args = tc.get("arguments") or {}
                    free = [k for k, v in args.items() if is_free_text(v)]
                    if free:
                        n_free_calls += 1
                        for k in free:
                            free_by_tool[f"{tc['name']}.{k}"] += 1
    out = {
        "source": "sierra-research/tau2-bench@d8e915f data/tau2/results/final",
        "files": [os.path.basename(f) for f in files],
        "min_words_free_text": MIN_WORDS,
        "n_simulations": n_sims,
        "n_tool_calls": n_calls,
        "tool_calls_by_model": dict(calls_by_model),
        "n_calls_with_free_text_argument": n_free_calls,
        "free_text_arguments_by_tool": dict(free_by_tool.most_common()),
    }
    return out


if __name__ == "__main__":
    res = main(sys.argv[1])
    os.makedirs("results", exist_ok=True)
    with open("results/tau2_argument_census.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "files"}, indent=2))
