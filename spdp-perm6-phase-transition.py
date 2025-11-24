import numpy as np
from itertools import permutations, combinations
from scipy.linalg import svd
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# — perm₆ as numeric function —
def perm6_eval(x):
    mat = np.array(x).reshape((6, 6))
    return sum(np.prod([mat[i, p[i]] for i in range(6)]) for p in permutations(range(6)))

# — Directional Derivative (finite difference, binomial scheme) —
def directional_derivative(f, x, direction, order, delta=1e-3):
    coeff = 0
    for i in range(order + 1):
        sign = (-1)**i
        weight = sp.binomial(order, i)
        shifted = x + (i - order / 2) * delta * direction
        coeff += sign * weight * f(shifted)
    return coeff / (delta**order)

# — SPDP Matrix Builder —
def numeric_spdp_matrix(poly_func, var_count, k, r, n_samples=200, delta=1e-3):
    X = np.random.uniform(-1, 1, (n_samples, var_count))
    rows = []
    print(f"🔧 Building SPDP matrix for (k={k}, r={r}), samples={n_samples}")
    for i in tqdm(range(n_samples), desc=f"Samples (k={k}, r={r})"):
        x = X[i]
        row = []
        for proj_vars in combinations(range(var_count), r):
            direction = np.zeros(var_count)
            for v in proj_vars:
                direction[v] = 1
            try:
                val = directional_derivative(poly_func, x, direction, k, delta)
            except Exception:
                val = 0
            row.append(val)
        rows.append(row)
    return np.array(rows, dtype=np.float64)

# — Main Sweep for k=4 and r = 1..4 —
k = 4
r_values = [1, 2, 3, 4]
spdp_ranks = []

for r in r_values:
    mat = numeric_spdp_matrix(perm6_eval, var_count=36, k=k, r=r, n_samples=200)
    _, svals, _ = svd(mat, full_matrices=False)
    rank = np.sum(svals > 1e-10)
    spdp_ranks.append(rank)
    print(f"✅ SPDP rank for perm₆ at (k={k}, r={r}) = {rank}")

# — Plot Rank Curve —
plt.figure(figsize=(7, 4))
plt.plot(r_values, spdp_ranks, marker='o', linewidth=2)
plt.title("SPDP Rank of perm₆ at Derivative Order k=4")
plt.xlabel("Projection Dimension r")
plt.ylabel("SPDP Rank (n=200)")
plt.xticks(r_values)
plt.grid(True)
plt.tight_layout()
plt.show()

# — Save Results —
df = pd.DataFrame({'r': r_values, 'rank': spdp_ranks})
df.to_csv("perm6_k4_spdp_ranks.csv", index=False)
print("💾 Results saved to perm6_k4_spdp_ranks.csv")
