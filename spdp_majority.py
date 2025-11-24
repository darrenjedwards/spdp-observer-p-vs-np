"""
spdp_majority.py

Computes the SPDP rank for the 3-variable majority function using
second-order directional derivatives and degree-1 shift monomials.

Function:
    Majority(x0, x1, x2) = x0*x1 + x0*x2 + x1*x2

SPDP parameters:
    k = 2   # order of partial derivatives
    ℓ = 1   # shift monomial degree
    r = full projection (no pruning)

Output:
    - SPDP matrix size
    - Symbolic SPDP rank

Dependencies:
    sympy
"""

import sympy as sp
from itertools import combinations, product

# SPDP parameters
k = 2
l = 1

# Define majority function
x0, x1, x2 = sp.symbols('x0 x1 x2')
maj_expr = sp.expand(x0*x1 + x0*x2 + x1*x2)
vars_list = [x0, x1, x2]

def k_order_derivatives(poly, vars_list, order):
    derivs = set()
    for idxs in product(range(len(vars_list)), repeat=order):
        var_seq = [vars_list[i] for i in idxs]
        d = poly
        for v in var_seq:
            d = sp.diff(d, v)
        if d != 0:
            derivs.add(sp.expand(d))
    return list(derivs)

def shift_monomials(vars_list, max_deg):
    monoms = [1]
    for d in range(1, max_deg + 1):
        for var_combo in combinations(vars_list, d):
            monoms.append(sp.Mul(*var_combo))
    return monoms

def compute_spdp(expr, name=""):
    partials = k_order_derivatives(expr, vars_list, k)
    shifts = shift_monomials(vars_list, l)
    spdp_terms = [sp.expand(shift * deriv) for deriv in partials for shift in shifts]
    spdp_terms_unique = list(set(spdp_terms))
    monomial_basis = list({mon for expr in spdp_terms_unique for mon in expr.as_ordered_terms()})
    monomial_basis = list(set(monomial_basis))
    matrix = sp.Matrix([[term.as_coefficients_dict().get(mon, 0) for mon in monomial_basis]
                        for term in spdp_terms_unique])
    rank = matrix.rank()
    print(f"{name} SPDP matrix size: {len(spdp_terms_unique)} × {len(monomial_basis)}")
    print(f"{name} SPDP rank: {rank}")
    print()

# Run test
compute_spdp(maj_expr, "Majority (3-bit)")
