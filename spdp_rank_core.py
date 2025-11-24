
# spdp_rank_core.py
import numpy as np
from tqdm import tqdm

def directional_derivative(poly, vars, v):
    new = {}
    for mon, coeff in poly.items():
        md = dict(mon)
        for i, var in enumerate(vars):
            pwr = md.get(var, 0)
            if pwr:
                new_mon = tuple(
                    (x, md.get(x, 0) - (1 if x == var else 0))
                    for x in vars if md.get(x, 0) - (1 if x == var else 0) > 0
                )
                new[new_mon] = new.get(new_mon, 0) + coeff * pwr * v[i]
    return new

def spdp_rank(poly, vars, k=3, samples=256):
    rows = []
    for _ in tqdm(range(samples), desc="SPDP Matrix"):
        v = np.random.randn(len(vars))
        p = poly.copy()
        for _ in range(k):
            p = directional_derivative(p, vars, v)
        if p:
            rows.append([p.get(m, 0) for m in sorted(p)])
    if not rows:
        return 0
    M = np.array(rows)
    return np.linalg.matrix_rank(M)
