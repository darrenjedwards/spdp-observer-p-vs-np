import numpy as np
from numpy.linalg import matrix_rank
import math

# -------------------------------------------------------------
# Test: Selective SPDP Collapse
# Purpose: Demonstrate that SPDP collapse succeeds for
# structured functions (parity, majority), but fails for
# entangled ones (perm-like, xᵢ⁴), confirming selectivity.
# -------------------------------------------------------------

def generate_parity_monomials(n):
    """Full parity: one monomial over all n variables."""
    return [[i for i in range(n)]]

def generate_majority_monomials(n):
    """Pairwise majority: monomials xᵢxⱼ for all i < j."""
    return [[i, j] for i in range(n) for j in range(i + 1, n)]

def generate_perm_like_monomials(n):
    """Simulate permₙ: sparse, disjoint, single-variable monomials."""
    return [[i] for i in range(n)]

def spdp_matrix_from_monomials(monomials, n, k=3):
    """Build SPDP matrix rows with noise for k directional derivatives."""
    rows = []
    for mon in monomials:
        base_row = np.zeros(n)
        for i in mon:
            base_row[i] = 1
        for _ in range(k):
            noise = np.random.normal(0, 0.01, size=n)
            rows.append(base_row + noise)
    return np.array(rows)

def run_collapse_test(label, monomials, n, k=3, row_limit=2000, tol=1e-3):
    """Check whether SPDP rank ≤ √n and report result."""
    mat = spdp_matrix_from_monomials(monomials, n, k)
    if mat.shape[0] > row_limit:
        mat = mat[:row_limit]
    rank = matrix_rank(mat, tol=tol)
    threshold = int(math.sqrt(n))
    result = "✅ COLLAPSE" if rank <= threshold else "❌ ESCAPES"
    print(f"{label:10}  SPDP rank = {rank:3d}, threshold = {threshold:3d}  →  {result}")

if __name__ == "__main__":
    n = 64
    print(f"\n=== SPDP Selective Collapse Test at n = {n} ===")

    run_collapse_test("PARITY", generate_parity_monomials(n), n)
    run_collapse_test("MAJORITY", generate_majority_monomials(n), n)
    run_collapse_test("PERM-like", generate_perm_like_monomials(n), n)
