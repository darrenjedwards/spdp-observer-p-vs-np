#!/usr/bin/env python3
"""
spdp.py  –  minimal Shifted-Projection Partial-Derivative (SPDP) toolkit
-----------------------------------------------------------------------

 * Monomials are n-tuples of non-negative ints.
 * A polynomial is a dict {monomial-tuple → coefficient}.
 * All arithmetic is over ℚ (ints are fine for rank testing).

Quick CLI demo
--------------
$ python spdp.py --n 4 --poly and_all --k 2 --l 2 --r 2
SPDP_rank = 0
"""

import itertools
import math
import numpy as np
from argparse import ArgumentParser


# ----------------------------------------------------------------------
#  Basic polynomial utilities
# ----------------------------------------------------------------------
def zero_poly():
    return {}


def add_poly(p, q):
    r = p.copy()
    for m, c in q.items():
        r[m] = r.get(m, 0) + c
        if r[m] == 0:
            del r[m]
    return r


def mul_monom_poly(mono, p):
    res = {}
    for m, c in p.items():
        res[tuple(a + b for a, b in zip(m, mono))] = res.get(
            tuple(a + b for a, b in zip(m, mono)), 0
        ) + c
    return res


def derivative(p, var_idx):
    """∂/∂x_{var_idx} (p)."""
    res = {}
    for m, c in p.items():
        if m[var_idx]:
            new_m = list(m)
            new_m[var_idx] -= 1
            res[tuple(new_m)] = res.get(tuple(new_m), 0) + c * m[var_idx]
    return res


def derivative_multi(p, multi):
    """Partial derivative by a multi-index."""
    res = p
    for idx, times in enumerate(multi):
        for _ in range(times):
            res = derivative(res, idx)
            if not res:
                return {}
    return res


def poly_degree(p):
    return -1 if not p else max(sum(m) for m in p)


# ----------------------------------------------------------------------
#  Combinatorics helpers
# ----------------------------------------------------------------------
def generate_monomials(n, max_deg):
    """All n-variate monomials of total degree ≤ max_deg (as exponent tuples)."""
    monos = []
    for deg in range(max_deg + 1):
        for comb in itertools.combinations_with_replacement(range(n), deg):
            exps = [0] * n
            for idx in comb:
                exps[idx] += 1
            monos.append(tuple(exps))
    return monos


def gen_multi_indices(k, indices, n):
    """All multi-indices of total order k supported on 'indices'."""
    out = []
    current = [0] * n

    def rec(rem_k, pos):
        if pos == len(indices):
            if rem_k == 0:
                out.append(tuple(current))
            return
        idx = indices[pos]
        for t in range(rem_k + 1):
            current[idx] = t
            rec(rem_k - t, pos + 1)
        current[idx] = 0

    rec(k, 0)
    return out


# ----------------------------------------------------------------------
#  Core SPDP rank
# ----------------------------------------------------------------------
def spdp_rank(poly, n, k, l, r):
    """
    Compute  SPDP_{k,l,r}(poly)  (minimum rank over all |Y|=r).
    WARNING: exponential in n choose r – use only for n ≤ ~20.
    """
    best = math.inf
    deg_p = poly_degree(poly)
    for Y in itertools.combinations(range(n), r):
        Y = list(Y)

        # k-th derivatives supported in Y
        multi_indices = gen_multi_indices(k, Y, n)
        nonzero_derivs = [d for d in (derivative_multi(poly, m) for m in multi_indices) if d]

        if not nonzero_derivs:
            best = 0
            continue

        # shift monomials (degree ≤ l) supported in Y
        shift_full = []
        for mono_small in generate_monomials(len(Y), l):
            full = [0] * n
            for pos, var_idx in enumerate(Y):
                full[var_idx] = mono_small[pos]
            shift_full.append(tuple(full))

        # build span
        polys = []
        for deriv in nonzero_derivs:
            for shift in shift_full:
                polys.append(mul_monom_poly(shift, deriv))

        # project onto Y
        proj_polys = []
        basis = generate_monomials(r, deg_p + l)
        mon_idx = {m: j for j, m in enumerate(basis)}

        for q in polys:
            vec = [0] * len(basis)
            for m, c in q.items():
                if all(m[i] == 0 for i in range(n) if i not in Y):
                    compressed = tuple(m[i] for i in Y)
                    vec[mon_idx[compressed]] += c
            proj_polys.append(vec)

        if not proj_polys:
            best = 0
            continue

        M = np.array(proj_polys, dtype=float)
        rank = np.linalg.matrix_rank(M)
        best = min(best, rank)
    return best


# ----------------------------------------------------------------------
#  Simple test polynomials
# ----------------------------------------------------------------------
def poly_and_all(n):
    """x1⋯xn."""
    mono = [1] * n
    return {tuple(mono): 1}


def poly_sum(n):
    """x1 + … + xn."""
    return {tuple([1 if i == j else 0 for i in range(n)]): 1 for j in range(n)}


def poly_rand(n, max_deg=3, density=0.3, seed=42):
    """Random multilinear polynomial over ±1 coefficients."""
    import random

    random.seed(seed)
    p = {}
    for m in generate_monomials(n, max_deg):
        if random.random() < density and sum(m) > 0:
            p[m] = random.choice([-1, 1])
    return p


# ----------------------------------------------------------------------
#  Command-line driver
# ----------------------------------------------------------------------
def main():
    ap = ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--l", type=int, default=2)
    ap.add_argument("--r", type=int, default=2)
    ap.add_argument("--poly", choices=["and_all", "sum", "rand"], default="and_all")
    args = ap.parse_args()

    if args.poly == "and_all":
        P = poly_and_all(args.n)
    elif args.poly == "sum":
        P = poly_sum(args.n)
    else:
        P = poly_rand(args.n)

    rank = spdp_rank(P, args.n, args.k, args.l, args.r)
    print(f"SPDP_rank = {rank}")


if __name__ == "__main__":
    main()
