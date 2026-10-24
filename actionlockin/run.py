from __future__ import annotations
import argparse
import pandas as pd
from actionlockin.propositions import Proposition, load_propositions
from actionlockin.conditions import ConditionText, PaddingSide, build_conditions, belief_probe, SYSTEM, pad_to_match
from actionlockin.confounds import text_features
from actionlockin.elicit import prob_of_P
from actionlockin.scorer import Scorer
from typing import Callable


def run_for_scorer(scorer: Scorer, props: list[Proposition],
                   builder: Callable[[Proposition], list[ConditionText]] = build_conditions,
                   padding_side: PaddingSide = "right") -> pd.DataFrame:
    rows = []
    for prop in props:
        cts = builder(prop)
        target = max(len(scorer.tokenize(ct.middle)) for ct in cts)  # per-model, per-prop
        for ct in cts:
            # single-token filler -> every cell reaches exactly `target` tokens
            mid = pad_to_match(scorer.tokenize, ct.middle, target, filler=" note", side=padding_side)
            ct_p = ConditionText(ct.prop_id, ct.condition, ct.framing, ct.dose, mid)
            user = belief_probe(prop)
            feats = text_features(scorer, SYSTEM, user, mid, prop.claim)
            rows.append({
                "model": scorer.name, "prop_id": prop.id, "gold": prop.gold,
                "condition": ct.condition, "framing": ct.framing, "dose": ct.dose,
                # probability readout (for calibration + answer-flip accuracy);
                # log-odds readout (primary: unbounded, no 0/1 ceiling on confident items)
                "p_of_P": prob_of_P(scorer, prop, ct_p),
                "logodds_P": scorer.logit_diff(SYSTEM, user, mid),
                "mid_n_tokens": feats["n_tokens"],
                "mid_mean_logprob": feats["mean_logprob"],
                "mid_claim_salience": feats["claim_salience"],
            })
    return pd.DataFrame(rows)


def _main() -> None:
    """Run the bounded-probability workflow configured in ``configs/default.yaml``.

    This entry point applies a mid-band filter because probability differences saturate near zero
    and one. Full-set log-odds workflows call ``run_for_scorer`` from the batch scripts instead.
    """
    import yaml
    from pathlib import Path
    from actionlockin.calibration import filter_midband
    from actionlockin.scorer import HFScorer
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    props = load_propositions(cfg["paths"]["propositions"])
    frames = []
    for name in cfg["models"]:
        scorer = HFScorer(
            name,
            device_map=cfg.get("device_map", "auto"),
            cache_dir=cfg.get("cache_dir"),
        )
        kept = filter_midband(scorer, props, cfg["calibration"]["low"], cfg["calibration"]["high"])
        frames.append(run_for_scorer(scorer, kept))
        del scorer
    out = pd.concat(frames, ignore_index=True)
    Path(cfg["paths"]["results"]).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cfg["paths"]["results"])
    print(f"wrote {len(out)} rows to {cfg['paths']['results']}")


if __name__ == "__main__":
    _main()
