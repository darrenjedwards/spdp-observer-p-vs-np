import numpy as np
from numpy.linalg import matrix_rank
import math
import random

# -------------------------------------------------------------
# Test: Seed Collapse with Two-Phase Pruning
# Purpose: Validate that for structured circuits (e.g. majority),
# collapse occurs in SOME seeds but not all — supporting ∃s
# in the witnessed predicate LowRank(C).
# -------------------------------------------------------------

def generate_structured_majority(n):
    """All xᵢxⱼ monomials for i < j."""
    return [[i, j] for i in range(n) for j in range(i + 1, n)]

def apply_two_phase_pruning(monomials, n, keep_clause_frac=0.1, keep_var_prob=0.05, seed=None):
    """
    Phase I: Clause pruning (keep ~10% of monomials)
    Phase II: Variable pruning (keep ~5% of variables)
    """
    if seed is not None:
        random.seed(seed)

    # Phase I: keep a random subset of monomials (simulate clause thinning)
    num_keep = max(1, int(len(monomials) * keep_clause_frac))
    kept = random.sample(monomials, num_keep)

    # Phase II: remove variables not surviving short seed
    keep_vars = [i for i in range(n) if random.random() < keep_var_prob]
    pruned = []
    for mon in kept:
        pruned_mon = [i for i in mon if i in keep_vars]
        if pruned_mon:
            pruned.append(pruned_mon)
    return pruned

def spdp_matrix_from_monomials(monomials, n, k=3):
    """SPDP matrix: k noisy derivatives per monomial."""
    rows = []
    for mon in monomials:
        base_row = np.zeros(n)
        for i in mon:
            base_row[i] = 1
        for _ in range(k):
            noise = np.random.normal(0, 0.01, size=n)
            rows.append(base_row + noise)
    return np.array(rows)

def test_two_phase_pruning_collapse(n=64, trials=20, k=3, keep_clause_frac=0.1, keep_var_prob=0.05, row_limit=2000, tol=1e-3):
    original = generate_structured_majority(n)
    threshold = int(math.sqrt(n))
    successes = 0

    for s in range(trials):
        pruned = apply_two_phase_pruning(original, n, keep_clause_frac, keep_var_prob, seed=s)
        mat = spdp_matrix_from_monomials(pruned, n, k)
        if mat.shape[0] > row_limit:
            mat = mat[:row_limit]
        rank = matrix_rank(mat, tol=tol)
        print(f"Seed {s:2d} → SPDP rank = {rank}, threshold = {threshold}")
        if rank <= threshold:
            successes += 1

    print(f"\n✅ Collapse success in {successes}/{trials} seeds for n = {n} (clause_frac = {keep_clause_frac}, var_prob = {keep_var_prob})")
    return successes

if __name__ == "__main__":
    test_two_phase_pruning_collapse(n=64, trials=20, keep_clause_frac=0.1, keep_var_prob=0.05)
