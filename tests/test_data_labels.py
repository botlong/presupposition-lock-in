"""Validate every programmatically checkable label against the committed artifact.

Arithmetic labels are recomputed rather than trusted, and the Russia/Pluto comparison has an
independent numerical assertion because it cannot be derived by the arithmetic parser.
"""
import json
import math
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def _rows(name):
    return [json.loads(l) for l in (DATA / name).read_text().splitlines() if l.strip()]


def _arith_true_value(claim):
    for pat, f in [
        (r"(\d+) multiplied by (\d+) equals (\d+)", lambda a, b: a * b),
        (r"the square of (\d+) equals (\d+)", lambda a: a * a),
        (r"(\d+) percent of (\d+) equals (\d+)", lambda p, base: base * p / 100),
        (r"the sum of (\d+) and (\d+) equals (\d+)", lambda a, b: a + b),
    ]:
        m = re.match(pat, claim)
        if m:
            g = list(map(int, m.groups()))
            return f(*g[:-1]), g[-1]
    return None, None


def _check_arith(name):
    bad = []
    for d in _rows(name):
        tv, stated = _arith_true_value(d["claim"])
        assert tv is not None, f"{name}: unparsed claim {d['claim']!r}"
        if (abs(tv - stated) < 1e-9) != d["gold"]:
            bad.append((d["id"], d["claim"], f"true={tv}", f"gold={d['gold']}"))
    assert not bad, f"{name} mislabeled rows: {bad}"


def test_arith_props_labels_exact():
    _check_arith("arith_props.jsonl")


def test_arith_hard_labels_exact():
    _check_arith("arith_hard.jsonl")


def test_p030_russia_vs_pluto_surface_area():
    # Pluto's spherical surface area is about 17.74M km^2, so the comparison must use
    # surface area rather than the much smaller projected disk area.
    p030 = next(d for d in _rows("propositions.jsonl") if d["id"] == "p030")
    pluto = 4 * math.pi * 1188.3 ** 2
    russia = 17_098_242
    assert (russia > pluto) is False
    assert p030["gold"] is False


def test_label_balances_are_pinned():
    # Class counts are part of the artifact contract; drift would change every gold-split analysis.
    props = _rows("propositions.jsonl")
    arith = _rows("arith_props.jsonl")
    hard = _rows("arith_hard.jsonl")
    reasoning = _rows("reasoning_props.jsonl")
    assert (len(props), sum(not d["gold"] for d in props)) == (120, 54)
    assert (len(arith), sum(d["gold"] for d in arith)) == (196, 94)
    assert (len(hard), sum(not d["gold"] for d in hard)) == (100, 58)
    assert (len(reasoning), sum(not d["gold"] for d in reasoning)) == (36, 14)
