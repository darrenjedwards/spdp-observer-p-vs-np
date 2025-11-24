#!/usr/bin/env python3
"""
spdp_rank_perm3x3_strong.py

Symbolic SPDP rank evaluation for the 3x3 permanent.
Used to benchmark the collapse-resistance of hard polynomials.
"""

import itertools
import numpy as np

# ----- Generate 3x3 permanent polynomial -----
def generate_perm3x3():
    n = 9
    terms = list(itertools.permutations(range(3)))
    poly = {}
    for sigma in terms:
        mon = [0] * n
        for i in range(3):
            mon[3*i + sigma[i]] += 1
        poly[tuple(mon)] = 1  # all coefficients = 1
    return poly, n

# ----- Derivatives -----
def derivative(p, var_idx):
    res = {}
    for m, c in p.items():
        if m[var_idx]:
            new = list(m)
            new[var_idx] -= 1
            res[tuple(new)] = res.get(tuple(new), 0) + c * m[var_idx]
    return res

def derivative_multi(p, multi):
    res = p
    for idx, times in enumerate(multi):
        for _ in range(times):
            res = derivative(res, idx)
            if not res:
                return {}
    return res

# ----- SPDP rank -----
def spdp_rank(poly, n, k=4, r=200):
    rows = []
    keys = list(poly.keys())
    for _ in range(r):
        idxs = np.random.choice(n, k, replace=True)
        multi = [0] * n
        for i in idxs:
            multi[i] += 1
        d = derivative_multi(poly, multi)
        v = [d.get(m, 0.0) for m in keys]
        rows.append(v)
    mat = np.array(rows)
    return np.linalg.matrix_rank(mat)

# ----- Main run -----
def run_perm3x3_spdp():
    poly, n = generate_perm3x3()
    rank = spdp_rank(poly, n, k=4, r=200)
    print(f"SPDP rank of perm3x3 (r=200, k=4): {rank} / {len(poly)}")

if __name__ == "__main__":
    run_perm3x3_spdp()
