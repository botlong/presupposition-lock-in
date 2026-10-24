"""Generate difficulty-graded arithmetic claims with exactly computable gold labels.

False claims perturb the exact result by a small nonzero amount. Multi-digit arithmetic provides
measurement headroom where factual recall may saturate. The committed JSONL remains the artifact of
record; this generator creates candidate sets and does not encode artifact-level deduplication.
"""
import json, random

random.seed(20260619)
UNREL = ["the violin has four strings", "copper conducts electricity", "Paris is the capital of France",
         "a triangle has three sides", "honey is produced by bees", "iron is a metal", "a week has seven days",
         "the Pacific is an ocean", "oak is a type of tree", "salt dissolves in water", "penguins are birds",
         "a square has four sides", "the Sun emits light", "milk contains calcium", "granite is a rock"]


def perturb(v):
    # plausible wrong value: change by a small amount or tweak a digit, never equal to v
    while True:
        d = random.choice([-1, 1]) * random.choice([1, 2, 3, 9, 10, 18, 20, 90, 100])
        if v + d != v and v + d > 0:
            return v + d


def gen_one(i):
    kind = random.choice(["mul", "mul", "mul", "add", "sq", "pct"])  # weight multiplication
    if kind == "mul":
        a = random.randint(12, 99); b = random.randint(12, 99)
        if random.random() < 0.35: b = random.randint(101, 499)      # harder 2x3-digit
        v = a * b; tmpl = f"{a} multiplied by {b} equals {{c}}"
    elif kind == "add":
        a = random.randint(1000, 9999); b = random.randint(1000, 9999)
        v = a + b; tmpl = f"the sum of {a} and {b} equals {{c}}"
    elif kind == "sq":
        a = random.randint(13, 89); v = a * a; tmpl = f"the square of {a} equals {{c}}"
    else:  # pct
        # Integer division is exact only when base*p is divisible by 100. Multiples of 20 guarantee
        # that invariant for the selected odd multiples of five.
        base = random.choice([40, 60, 80, 120, 160, 200, 240, 300]); p = random.choice([5, 15, 25, 35, 45])
        assert base * p % 100 == 0, f"non-integer percent: {p}% of {base}"
        v = base * p // 100; tmpl = f"{p} percent of {base} equals {{c}}"
    gold = random.random() < 0.5
    c = v if gold else perturb(v)
    return {"id": f"a{i:03d}", "claim": tmpl.format(c=c), "gold": bool(gold),
            "unrelated_claim": UNREL[i % len(UNREL)]}


def main():
    n = 200
    rows = [gen_one(i) for i in range(1, n + 1)]
    nt = sum(r["gold"] for r in rows)
    with open("data/arith_props.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} arithmetic claims ({nt} true / {len(rows)-nt} false) to data/arith_props.jsonl")
    for r in rows[:5]:
        print("  ", r["claim"], "->", r["gold"])


if __name__ == "__main__":
    main()
