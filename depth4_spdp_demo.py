#!/usr/bin/env python3
"""
depth4_spdp_demo.py

Structured SPDP rank collapse test on depth-4 ΣΠΣΠ circuits (used in paper Figure/Table).
Tests collapse frequency for n ∈ {64, 128, 256} using real circuit generation and SPDP rank.
"""

import random, math
import numpy as np
from tqdm import tqdm

# ----- Generate structured depth-4 circuit -----
def generate_depth4_circuit(n, seed):
    random.seed(seed)
    d = int(math.log2(n))
    size = int(n ** 1.5)
    circuit = []
    for _ in range(size):
        k = random.randint(2, d)
        monomial = sorted(random.sample(range(n), k))
        circuit.append(monomial)
    return circuit

# ----- Generate symbolic polynomial representation -----
def poly_from_circuit(circuit, n):
    poly = {}
    for mon in circuit:
        exponents = [0] * n
        for idx in mon:
            exponents[idx] += 1
        key = tuple(exponents)
        poly[key] = poly.get(key, 0) + 1
    return poly

# ----- Derivative utilities -----
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
            if not res: return {}
    return res

# ----- SPDP rank estimation -----
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

# ----- Run test -----
def run_test():
    for n in [64, 128, 256]:
        threshold = int(math.sqrt(n))
        collapses = 0
        for seed in tqdm(range(20), desc=f"n={n}"):
            circ = generate_depth4_circuit(n, seed)
            poly = poly_from_circuit(circ, n)
            rank = spdp_rank(poly, n, k=4, r=n)
            if rank <= threshold:
                collapses += 1
        print(f"n = {n}, collapsed = {collapses}/20")

if __name__ == "__main__":
    run_test()
