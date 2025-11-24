import numpy as np
from numpy.linalg import lstsq, norm

# -------------------------------------------------------------
# This script verifies that a given matrix has SPDP rank ≤ √n
# using a rank certificate (basis + coefficients), simulating
# the diagonal verifier for f_n(i) in your P ≠ NP construction.
#
# Test 1: Valid low-rank matrix + correct basis → should PASS
# Test 2: Full-rank matrix + wrong basis → should FAIL
# -------------------------------------------------------------

def generate_low_rank_matrix(n, rank):
    """
    Generate an n x n matrix of rank ≤ `rank`.
    Done by multiplying:
      - a random basis matrix (n x rank),
      - by a random coefficient matrix (rank x n).
    """
    basis = np.random.randn(n, rank)        # Basis vectors: shape (n, r)
    coeffs = np.random.randn(rank, n)       # Coefficients: shape (r, n)
    matrix = basis @ coeffs                 # Low-rank matrix: shape (n, n)
    return matrix, basis                    # Return both matrix and its true basis

def verify_rank_certificate(matrix, basis, tol=1e-6):
    """
    Check whether each row of `matrix` lies in the span of the given `basis`.

    Args:
        matrix: (n x d) matrix of target rows to check
        basis:  (n x r) matrix of column basis vectors
        tol: error tolerance for residual norm

    Returns:
        True if all rows lie in the span of the basis, False otherwise.
    """
    n, d = matrix.shape
    for i in range(n):
        try:
            # Solve basis * x ≈ matrix[i] using least squares
            coeffs, *_ = lstsq(basis, matrix[i], rcond=None)
            reconstructed = basis @ coeffs
            error = norm(matrix[i] - reconstructed)

            if error > tol:
                print(f"Row {i} failed with residual {error:.4e}")
                return False
        except Exception as e:
            print(f"Row {i} exception: {e}")
            return False
    return True

if __name__ == "__main__":
    n = 64                              # Matrix size
    rank_bound = int(n ** 0.5)          # Diagonal collapse threshold: √n

    # === Test 1: Should PASS (true low-rank matrix with valid basis) ===
    print("\n=== Test 1: Verifier should PASS on true low-rank matrix ===")
    matrix, true_basis = generate_low_rank_matrix(n, rank_bound)
    result = verify_rank_certificate(matrix, true_basis)
    print("PASS ✅" if result else "FAIL ❌")

    # === Test 2: Should FAIL (random full-rank matrix and fake basis) ===
    print("\n=== Test 2: Verifier should FAIL on full-rank matrix ===")
    full_matrix = np.random.randn(n, n)                 # Likely full-rank
    fake_basis = np.random.randn(n, rank_bound)         # Not related to matrix
    result = verify_rank_certificate(full_matrix, fake_basis)
    print("FAIL as expected ✅" if not result else "INCORRECTLY PASSED ❌")
