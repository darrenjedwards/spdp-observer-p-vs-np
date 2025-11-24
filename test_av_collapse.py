import numpy as np
from numpy.linalg import matrix_rank
import math
import random

# -------------------------------------------------------------
# Test: AV(C) SPDP Collapse
# Purpose: Simulate a general (depth > 4) Boolean circuit,
# apply an AV-style reduction to ΣΠΣΠ (depth-4) form,
# and test whether SPDP rank collapses to ≤ √n.
# -------------------------------------------------------------

def generate_general_boolean_circuit(n):
    """
    Simulates a general Boolean circuit as a list of monomials,
    each involving log(n)-sized random subsets of variables.
    """
    terms = []
    max_degree = max(2, int(math.log2(n)))
    for _ in range(n ** 2):  # simulate poly(n)-sized circuit
        degree = random.randint(2, max_degree)
        variables = random.sample(range(n), degree)
        terms.append(variables)
    return terms

def av_reduce_to_depth4(circuit_terms):
    """
    Simulate AV-style ΣΠΣΠ reduction:
    Break high-degree monomials into smaller degree ≤ 4 ones.
    """
    grouped = []
    for monomial in circuit_terms:
        if len(monomial) > 4:
            grouped.append(monomial[:2])
            grouped.append(monomial[2:])
        else:
            grouped.append(monomial)
    return grouped

def spdp_matrix_from_monomials(monomials, n, k=3):
    """
    Generate an approximate SPDP matrix:
    For each monomial, simulate k noisy partial derivative rows.
    """
    rows = []
    for mon in monomials:
        base_row = np.zeros(n)
        for i in mon:
            base_row[i] = 1
        for _ in range(k):
            noise = np.random.normal(0, 0.01, size=n)
            rows.append(base_row + noise)
    return np.array(rows)

def run_av_spdp_collapse_test(n=128, k=3, row_limit=2000, tol=1e-3):
    """
    Run full AV-style SPDP collapse test.
    """
    raw_circuit = generate_general_boolean_circuit(n)
    depth4 = av_reduce_to_depth4(raw_circuit)
    spdp_mat = spdp_matrix_from_monomials(depth4, n, k)

    if spdp_mat.shape[0] > row_limit:
        spdp_mat = spdp_mat[:row_limit]

    rank = matrix_rank(spdp_mat, tol=tol)
    return rank

if __name__ == "__main__":
    n = 128
    print(f"\n=== SPDP Collapse Test on AV(C) for n = {n} ===")
    rank = run_av_spdp_collapse_test(n=n)
    threshold = int(math.sqrt(n))
    print(f"SPDP rank = {rank}, threshold = {threshold}")
    if rank <= threshold:
        print("✅ COLLAPSE CONFIRMED")
    else:
        print("❌ COLLAPSE FAILED")
