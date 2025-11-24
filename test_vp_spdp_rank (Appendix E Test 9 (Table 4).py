import sympy as sp
import numpy as np
import csv

def generate_linear_adder(n):
    x = sp.symbols(f'x0:{n}')
    return sum(x), x

def generate_fft_output_poly(n):
    x = sp.symbols(f'x0:{n}')
    # Basic butterfly pattern: sum of x[i] * ω^i
    omega = sp.exp(2 * sp.pi * sp.I / n)
    return sum(x[i] * omega**i for i in range(n)), x

def build_spdp_matrix(f, x, k=1, r=1, samples=10):
    n = len(x)
    rows = []
    for _ in range(samples):
        v = np.random.randn(n)
        deriv = sum(vi * sp.diff(f, xi) for vi, xi in zip(v, x))
        mask = np.random.choice(n, r, replace=False)
        deriv_proj = deriv.subs({x[i]: 0 for i in mask})
        val = deriv_proj.subs({x[i]: np.random.rand() for i in range(n)})
        rows.append([float(sp.re(sp.N(val)))])  # Take real part
    return np.array(rows)

def spdp_rank(f, x, k=1, r=1, samples=10):
    M = build_spdp_matrix(f, x, k, r, samples)
    U, S, Vh = np.linalg.svd(M, full_matrices=False)
    return np.sum(S > 1e-10)

def run_test(output_csv):
    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["CircuitType", "n", "SPDP(1,1,1)_Rank", "sqrt(n)"])
        for n in range(4, 11):  # Adders from n=4 to 10
            f, x = generate_linear_adder(n)
            r = spdp_rank(f, x, k=1, r=1, samples=10)
            writer.writerow(["Adder", n, r, int(np.ceil(np.sqrt(n)))])
        f_fft, x_fft = generate_fft_output_poly(4)
        r_fft = spdp_rank(f_fft, x_fft, k=1, r=1, samples=10)
        writer.writerow(["FFT_Output", 4, r_fft, 2])

run_test("/mnt/data/test9_vp_spdp.csv")
