#!/usr/bin/env python3
"""
batch_collapse_test.py

Runs batch SPDP rank tests on various circuit families across multiple input sizes.
Used for generating collapse statistics and plots in the P ≠ NP paper.
"""

import random, math
import numpy as np
from tqdm import tqdm
import pandas as pd

# ----- Generate circuit families -----
def randdeg3(n, seed):
    random.seed(seed)
    return [sorted(random.sample(range(n), 3)) for _ in range(n)]

def majority(n, seed):
    random.seed(seed)
    center = n // 2
    return [[i, center, (i+1)%n] for i in range(n)]

def addressing(n, seed):
    random.seed(seed)
    m = int(math.log2(n))
    return [[i] + random.sample(range(n), m) for i in range(n)]

# ----- Polynomial from circuit -----
def poly_from_circuit(circuit, n):
    poly = {}
    for mon in circuit:
        exponents = [0] * n
        for idx in mon:
            exponents[idx] += 1
        key = tuple(exponents)
        poly[key] = poly.get(key, 0) + 1
    return poly

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
def spdp_rank(poly, n, k=4, r=32):
    rows = []
    for _ in range(r):
        idxs = random.sample(range(n), k)
        multi = [0] * n
        for i in idxs:
            multi[i] += 1
        d = derivative_multi(poly, multi)
        if not d:
            rows.append([0.0] * len(poly))
        else:
            v = []
            for m in poly.keys():
                v.append(d.get(m, 0.0))
            rows.append(v)
    mat = np.array(rows)
    return np.linalg.matrix_rank(mat)

# ----- Main batch loop -----
def run_batch():
    families = {
        "randdeg3": randdeg3,
        "majority": majority,
        "addressing": addressing
    }
    ns = [64, 128, 256]
    k, r = 4, 64
    trials = 20
    rows = []

    for name, gen in families.items():
        for n in ns:
            threshold = int(math.sqrt(n))
            collapse = 0
            for seed in tqdm(range(trials), desc=f"{name}, n={n}"):
                circ = gen(n, seed)
                poly = poly_from_circuit(circ, n)
                rank = spdp_rank(poly, n, k=k, r=r)
                if rank <= threshold:
                    collapse += 1
            rows.append(dict(family=name, n=n, collapsed=collapse, total=trials))

    df = pd.DataFrame(rows)
    print(df)
    df.to_csv("collapse_summary.csv", index=False)

if __name__ == "__main__":
    run_batch()
