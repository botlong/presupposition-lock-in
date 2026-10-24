"""Reconcile the `gold` column of results parquets against the single source of truth.

The `data/*.jsonl` proposition files are authoritative for each item's gold label
(propositions, arith_hard, arith_props, reasoning_props — disjoint id namespaces p/h/a/r).
Raw model-output columns are functions of the model input and do not depend on the gold label.
Re-deriving only ``gold`` from the authoritative JSONL therefore avoids an unnecessary inference
rerun while preserving every measured value. Atomic replacement and before/after equality guards
are required so reconciliation cannot silently modify any non-gold column.

Usage:
  python scripts/reconcile_gold.py            # fix any stale gold in results/*.parquet
  python scripts/reconcile_gold.py --check    # report only; exit 1 if any gold is stale
"""
from __future__ import annotations

import contextlib
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

SOURCE_GLOB = "data/*.jsonl"
GLOB = "results/*.parquet"


def load_gold(source_glob: str = SOURCE_GLOB) -> dict:
    """Union gold map across every data/*.jsonl that carries a gold field.

    id namespaces are disjoint (p/h/a/r); a same-id/different-gold collision across files
    is a data error and aborts. Records without a gold field are not gold sources and are
    skipped (a parquet using such ids will fail the reconcile `unknown` guard loudly).
    """
    gold: dict = {}
    origin: dict = {}
    for path in sorted(glob.glob(source_glob)):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                pid = o.get("id", o.get("prop_id"))
                if pid is None or "gold" not in o:
                    continue
                g = bool(o["gold"])
                if pid in gold and gold[pid] != g:
                    raise ValueError(
                        f"conflicting gold for {pid}: {origin[pid]}={gold[pid]} vs {path}={g}"
                    )
                gold[pid] = g
                origin[pid] = path
    if not gold:
        raise ValueError(f"no gold-carrying records found under {source_glob}")
    return gold


def _values_equal(a: pd.Series, b: pd.Series) -> bool:
    """NaN-safe elementwise equality on same-length, row-aligned columns."""
    av, bv = a.to_numpy(), b.to_numpy()
    if av.shape != bv.shape:
        return False
    if np.issubdtype(av.dtype, np.floating) or np.issubdtype(bv.dtype, np.floating):
        return np.array_equal(av.astype(float), bv.astype(float), equal_nan=True)
    return np.array_equal(av, bv)


def reconcile(path: str, gold: dict, dry_run: bool = False):
    """Return (path, n_changed_cells, n_false_props) or None if not a gold parquet."""
    df = pd.read_parquet(path)
    if "gold" not in df.columns or "prop_id" not in df.columns:
        return None  # not a gold-carrying gate parquet

    unknown = sorted(set(df["prop_id"]) - set(gold))
    if unknown:
        raise ValueError(f"{path}: prop_ids absent from all {SOURCE_GLOB} sources: {unknown}")

    want = df["prop_id"].map(gold).astype(bool)
    changed = int((want.to_numpy() != df["gold"].to_numpy()).sum())
    n_false = int((~df.groupby("prop_id")["gold"].first().astype(bool)).sum())

    if changed == 0 or dry_run:
        # Dry-run reports the reconciled class balance so check mode matches write mode semantically.
        if changed and dry_run:
            n_false = int((~pd.Series(want.values, index=df["prop_id"]).groupby(level=0)
                           .first().astype(bool)).sum())
        return (path, changed, n_false)

    new = df.copy()
    new["gold"] = want.to_numpy()
    # A label-only reconciliation must never alter measured model outputs.
    for c in df.columns:
        if c == "gold":
            continue
        assert _values_equal(df[c], new[c]), f"{path}: non-gold column {c} changed in memory"

    tmp = path + ".tmp"
    try:
        new.to_parquet(tmp)
        back = pd.read_parquet(tmp)
        assert list(back.columns) == list(df.columns), f"{path}: column order changed on write"
        assert back.shape == df.shape, f"{path}: shape changed on write"
        assert np.array_equal(back["gold"].to_numpy(),
                              back["prop_id"].map(gold).astype(bool).to_numpy()), \
            f"{path}: post-write gold != source"
        for c in df.columns:
            if c == "gold":
                continue
            assert back[c].dtype == df[c].dtype, f"{path}: post-write dtype changed for {c}"
            assert _values_equal(df[c], back[c]), f"{path}: post-write non-gold column {c} changed"
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise
    os.replace(tmp, path)
    n_false = int((~back.groupby("prop_id")["gold"].first().astype(bool)).sum())
    return (path, changed, n_false)


def main(argv) -> None:
    dry = ("--check" in argv) or ("--dry-run" in argv)
    gold = load_gold()
    n_false_src = sum(1 for v in gold.values() if v is False)
    print(f"sources {SOURCE_GLOB}: {len(gold)} items, n_false={n_false_src}")

    any_stale = False
    for path in sorted(glob.glob(GLOB)):
        res = reconcile(path, gold, dry_run=dry)
        if res is None:
            continue
        p, changed, n_false = res
        if changed:
            any_stale = True
            verb = "WOULD FIX" if dry else "FIXED"
            print(f"  {verb} {p}: {changed} gold cells -> n_false={n_false}")
        else:
            print(f"  OK {p}: gold already consistent (n_false={n_false})")

    if dry and any_stale:
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
