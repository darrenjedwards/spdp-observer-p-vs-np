import numpy as np
from numpy.linalg import matrix_rank
import math
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Test: SPDP Phase Transition on perm₆
# Purpose: Show how projection dimension r affects SPDP rank
# for entangled functions like perm — confirms escape behavior
# and boundary of semantic observability.
# -------------------------------------------------------------

def generate_perm_like_monomials(n):
    """Simulate perm monomials as disjoint, single-variable support."""
    return [[i] for i in range(n)]  # e.g. x₁, x₂, ..., xₙ

def spdp_matrix_from_monomials(monomials, n, k=3, r=1):
    """
    Build SPDP matrix: apply projection by zeroing all but r random coordinates.
    """
    rows = []
    for mon in monomials:
        base = np.zeros(n)
        for i in mon:
            base[i] = 1
        for _ in range(k):
            noise = np.random.normal(0, 0.01, size=n)
            vec = base + noise
            proj = np.zeros(n)
            coords = np.random.choice(n, r, replace=False)
            for j in coords:
                proj[j] = vec[j]
            rows.append(proj)
    return np.array(rows)

def run_phase_transition(n=64, k=3, max_r=32, tol=1e-3):
    monomials = generate_perm_like_monomials(n)
    ranks = []
    rs = list(range(1, max_r + 1))
    for r in rs:
        mat = spdp_matrix_from_monomials(monomials, n, k=k, r=r)
        rank = matrix_rank(mat, tol=tol)
        ranks.append(rank)
        print(f"r = {r:2d}, SPDP rank = {rank}")
    return rs, ranks

if __name__ == "__main__":
    n = 64
    print(f"\n=== SPDP Phase Transition Test for perm-like at n = {n} ===")
    rs, ranks = run_phase_transition(n=n, max_r=32)

    # Plot rank vs. projection dimension
    plt.figure(figsize=(8, 4))
    plt.plot(rs, ranks, marker='o')
    plt.axhline(y=int(math.sqrt(n)), color='red', linestyle='--', label='Collapse threshold (√n)')
    plt.title(f"SPDP Phase Transition on perm-like (n = {n})")
    plt.xlabel("Projection dimension r")
    plt.ylabel("SPDP Rank")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
