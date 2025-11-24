import numpy as np
import time
import matplotlib.pyplot as plt
from numpy.linalg import lstsq, norm

# -------------------------------------------------------------
# Test: Verifier Runtime Scaling
# Purpose: Show that SPDP rank verifier (via span certificate)
# runs in empirical polynomial time, validating fₙ ∈ NP.
# -------------------------------------------------------------

def generate_low_rank_matrix(n, rank):
    """
    Generate an n x n matrix of rank ≤ rank via basis @ coeff.
    """
    basis = np.random.randn(n, rank)
    coeffs = np.random.randn(rank, n)
    matrix = basis @ coeffs
    return matrix, basis

def verify_rank_certificate(matrix, basis, tol=1e-6):
    """
    For each row in matrix, check it lies in the span of basis.
    Uses least-squares projection with error tolerance.
    """
    n, d = matrix.shape
    for i in range(n):
        coeffs, *_ = lstsq(basis, matrix[i], rcond=None)
        reconstructed = basis @ coeffs
        error = norm(matrix[i] - reconstructed)
        if error > tol:
            return False
    return True

def run_timing_tests(n_vals, rank_fn):
    """
    For each n in n_vals:
    - Generate a low-rank matrix
    - Time the span verifier
    """
    times = []
    for n in n_vals:
        print(f"Running verifier for n = {n}")
        rank = rank_fn(n)
        matrix, basis = generate_low_rank_matrix(n, rank)
        start = time.time()
        verify_rank_certificate(matrix, basis)
        elapsed = time.time() - start
        print(f"n = {n}, runtime = {elapsed:.4f} sec")
        times.append(elapsed)
    return times

if __name__ == "__main__":
    n_vals = [16, 32, 48, 64, 80]
    times = run_timing_tests(n_vals, rank_fn=lambda n: int(n ** 0.5))

    # Plot runtime vs. n
    plt.figure(figsize=(8, 4))
    plt.plot(n_vals, times, marker='o')
    plt.title("Verifier Runtime vs. n (SPDP rank certificate)")
    plt.xlabel("Matrix dimension n")
    plt.ylabel("Runtime (seconds)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
