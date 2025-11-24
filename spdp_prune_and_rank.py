
# spdp_prune_and_rank.py
import math
import random
from spdp_rank_core import spdp_rank
from build_circuits import *

def prune_vars(vars, p, force_include=None):
    kept = [v for v in vars if random.random() < p]
    if force_include:
        for v in force_include:
            if v not in kept:
                kept.append(v)
    return kept

def run_spdp_test(name, builder, n=1024, k=3, samples=256):
    print(f"=== Running SPDP Test: {name.upper()} (n = {n}) ===")
    poly, vars = builder(n)

    # Add Tseitin contradiction
    poly[((z := "z", 1),)] = 1
    poly[((z, 0),)] = 1
    vars.append(z)

    # Prune
    p = 1 / math.sqrt(len(vars))
    kept = prune_vars(vars, p, force_include=[z])
    pruned_poly = {
        m: c for m, c in poly.items()
        if all(v in kept for v, _ in m)
    }

    live = len(kept)
    rank = spdp_rank(pruned_poly, kept, k=k, samples=samples)
    thresh = math.ceil(math.sqrt(n))
    passed = rank <= thresh

    print(f"Live Vars: {live}, SPDP Rank: {rank}, Threshold: {thresh}")
    print("✅ PASS" if passed else "❌ FAIL")

    return {
        "Circuit": name,
        "n": n,
        "LiveVars": live,
        "Rank": rank,
        "Threshold": thresh,
        "Pass": passed
    }
