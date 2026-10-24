#!/usr/bin/env python
"""Benign-side cost of the mitigations, from the retained mitigation parquets.

The mitigation report shows the fixes remove lock-in on FALSE claims; this reads the same
per-proposition rows on the gold==True side and reports what each representation does to
belief in true claims: paired belief delta vs none, accuracy on true claims, and the
downgrade rate among true items correctly affirmed at baseline. Pure reanalysis — no model
is loaded.

Usage:  python scripts/benign_true.py [--files results/mitigation_*.parquet ...]
"""
from __future__ import annotations
import argparse
import glob
import pandas as pd
from actionlockin.analyze import benign_true_report

DEFAULT_GLOB = "results/mitigation_Qwen2.5-*_arith_hard.parquet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=None,
                    help=f"mitigation parquets (default: {DEFAULT_GLOB})")
    ap.add_argument("--value", default="logodds_P")
    args = ap.parse_args()
    files = args.files or sorted(glob.glob(DEFAULT_GLOB))
    if not files:
        raise SystemExit(f"no mitigation parquets match {DEFAULT_GLOB}")
    for f in files:
        df = pd.read_parquet(f)
        r = benign_true_report(df, value=args.value)
        print(f"\n===== BENIGN TRUE  {f}  n_true={r['_n_true']} "
              f"none_acc_true={r['_none_acc_true']:.2f} =====")
        print(f"  {'cell':13s} {'belief Δ vs none (log-odds)':30s} acc(true)  "
              f"downgrade(base-correct)")
        for lbl in ("commit3", "mit_disclaim", "mit_reframe", "mit_isolate"):
            d = r[lbl]; e = d["delta_vs_none"]; c = d["downgrade_rate_ci"]
            print(f"  {lbl:13s} {e['estimate']:+.3f} [{e['ci_low']:+.3f},{e['ci_high']:+.3f}]"
                  f"{'':8s} {d['acc_true']:.2f}       "
                  f"{d['downgrade_rate']:.2f} [{c['ci_low']:.2f},{c['ci_high']:.2f}] "
                  f"(n={d['downgrade_denom']})")


if __name__ == "__main__":
    main()
