"""Difficulty-class generator for the Llama-matched arithmetic set (data/arith_hard_llama.jsonl).

Claims use the arith_hard perturbation scheme (false = exact result +/- a small offset). Difficulty is
chosen at the CLASS level from a baseline-only pilot of eleven classes run in two rounds (60 claims
each): round 1 = PILOT1 (multiplication), round 2 = CLASSES. Their recorded output is
results/pilot_Llama-3.1-8B_classes.txt. The chosen class is a33 (3-digit addition); the final set is
drawn fresh from it with a separate seed, so no item is selected on model behaviour:

    python scripts/gen_arith_llama.py final a33 20261022 data/arith_hard_llama.jsonl   # byte-identical
    python scripts/gen_arith_llama.py pilot <model_dir>                                 # needs a GPU
"""
import json, random, sys
UNREL = ["the violin has four strings", "copper conducts electricity", "Paris is the capital of France",
         "a triangle has three sides", "honey is produced by bees", "iron is a metal", "a week has seven days",
         "the Pacific is an ocean", "oak is a type of tree", "salt dissolves in water", "penguins are birds",
         "a square has four sides", "the Sun emits light", "milk contains calcium", "granite is a rock"]
OFFSETS = [1, 2, 3, 7, 11, 20, 100, 111, 200]          # the perturbation set observed in arith_hard
CLASSES = {"m12": ("mul", (3, 9), (12, 99)), "m12b": ("mul", (6, 9), (21, 99)), "m22s": ("mul", (11, 25), (11, 25)),
           "a22": ("add", (10, 99), (10, 99)), "a33": ("add", (100, 999), (100, 999)), "sq2": ("sq", (11, 40), None)}
# Round-1 pilot classes (multiplication). Round 1 drew offsets without the `o < v` filter used below.
PILOT1 = {"m22": ((12, 99), (12, 99)), "m23": ((12, 99), (101, 999)),
          "m22b": ((40, 99), (40, 99)), "m23s": ((12, 49), (101, 499)), "m33": ((101, 999), (101, 999))}
TEMPL = {"mul": "{a} multiplied by {b} equals {c}", "add": "the sum of {a} and {b} equals {c}", "sq": "the square of {a} equals {c}"}
def gen(cls, n, seed, prefix):
    r = random.Random(seed); kind, ar, br = CLASSES[cls]; out = []
    for i in range(1, n + 1):
        a = r.randint(*ar); b = r.randint(*br) if br else a
        v = a * b if kind in ("mul", "sq") else a + b; gold = r.random() < 0.5
        c = v if gold else v + r.choice([-1, 1]) * r.choice([o for o in OFFSETS if o < v])
        out.append({"id": f"{prefix}{i:03d}", "claim": TEMPL[kind].format(a=a, b=b, c=c), "gold": gold,
                    "unrelated_claim": UNREL[i % len(UNREL)]})
    return out
def gen_pilot1(cls, n, seed, prefix):
    r = random.Random(seed); (a0, a1), (b0, b1) = PILOT1[cls]; out = []
    for i in range(1, n + 1):
        a, b = r.randint(a0, a1), r.randint(b0, b1); v = a * b; gold = r.random() < 0.5
        c = v if gold else v + r.choice([-1, 1]) * r.choice(OFFSETS)
        out.append({"id": f"{prefix}{i:03d}", "claim": TEMPL["mul"].format(a=a, b=b, c=c), "gold": gold,
                    "unrelated_claim": UNREL[i % len(UNREL)]})
    return out
if __name__ == "__main__":
    if sys.argv[1] == "pilot":
        from actionlockin.scorer import HFScorer
        from actionlockin.propositions import Proposition
        from actionlockin.conditions import build_conditions, belief_probe, SYSTEM
        s = HFScorer(sys.argv[2], device_map="cuda", dtype="bfloat16")
        rounds = [(c, gen_pilot1(c, 60, 1000 + list(PILOT1).index(c), "pl")) for c in PILOT1] + \
                 [(c, gen(c, 60, 1000 + list(CLASSES).index(c), "pl")) for c in CLASSES]
        for cls, rows in rounds:
            lo = {True: [], False: []}
            for d in rows:
                p = Proposition(d["id"], d["claim"], d["gold"], d["unrelated_claim"])
                none = [c for c in build_conditions(p) if c.condition == "none"][0]
                lo[d["gold"]].append(s.logit_diff(SYSTEM, belief_probe(p), none.middle))
            import numpy as np
            t, f = np.array(lo[True]), np.array(lo[False]); allv = np.concatenate([t, f])
            print(f"{cls:5s} accT={(t>0).mean():.2f} accF={(f<0).mean():.2f} meanT={t.mean():+.2f} meanF={f.mean():+.2f} "
                  f"|lo|<2: {(abs(allv)<2).mean():.2f}  sep(T-F)={t.mean()-f.mean():+.2f}", flush=True)
    else:  # final: python scripts/gen_arith_llama.py final <cls> <seed> <out>
        rows = gen(sys.argv[2], 100, int(sys.argv[3]), "L")
        with open(sys.argv[4], "w") as fh:
            for d in rows: fh.write(json.dumps(d) + "\n")
        print("wrote", len(rows), sum(d["gold"] for d in rows), "true")
