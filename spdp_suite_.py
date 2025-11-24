import sympy as sp
import numpy as np
import time
import csv
import math

def generate_random_poly(n, degree=3, seed=None):
    if seed is not None:
        np.random.seed(seed)
    x = sp.symbols(f'x0:{n}')
    monomials = []
    for _ in range(n * 5):  # Adjustable density
        vars_in_term = np.random.choice(n, size=degree, replace=False)
        monomial = 1
        for i in vars_in_term:
            monomial *= x[i]
        monomials.append(monomial)
    return sum(monomials), x

def directional_derivative(f, x, v):
    """Take directional derivative of f along vector v."""
    assert len(v) == len(x)
    return sum(vi * sp.diff(f, xi) for vi, xi in zip(v, x))

def build_spdp_matrix(f, x, k, r, samples=100):
    n = len(x)
    rows = []
    for _ in range(samples):
        deriv = f
        for _ in range(k):
            v = np.random.randn(n)
            deriv = directional_derivative(deriv, x, v)
        mask = np.random.choice(n, r, replace=False)
        deriv_proj = deriv.subs({x[i]: 0 for i in mask})
        val = deriv_proj.subs({x[i]: np.random.rand() for i in range(n)})
        rows.append([float(sp.N(val))])
    return np.array(rows)

def spdp_rank(f, x, k, r, samples=100):
    M = build_spdp_matrix(f, x, k, r, samples)
    U, S, Vh = np.linalg.svd(M, full_matrices=False)
    tol = 1e-10
    return np.sum(S > tol), M.shape[0]

def run_scaling_suite(output_csv, sizes=[10, 12, 16, 20, 24, 32, 40, 48, 56, 64]):
    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["n", "rank", "samples", "runtime_seconds"])
        for n in sizes:
            poly, x = generate_random_poly(n, degree=3, seed=n)
            start = time.time()
            rank, samples = spdp_rank(poly, x, k=3, r=3, samples=100)
            end = time.time()
            writer.writerow([n, rank, samples, round(end - start, 2)])

run_scaling_suite("/mnt/data/spdp_scaling_results.csv")
