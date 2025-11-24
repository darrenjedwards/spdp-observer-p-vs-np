"""
spdp_depth5.py

Computes the SPDP rank for a depth-5 style polynomial:
    f(x) = (x0 + x1)(x2 + x3)(x4 + x5)

This simulates a moderately entangled composed polynomial to test how
SPDP behaves with deeper but regular circuit structure.

SPDP parameters:
    k = 2   # order of partial derivatives
    ℓ = 1   # shift monomial degree

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

# Define depth-5 style expression
x0, x1, x2, x3, x4, x5 = sp.symbols('x0 x1 x2 x3 x4 x5')
depth5_expr = sp.expand((x0 + x1)*(x2 + x3)*(x4 + x5))
vars_list = [x0, x1, x2, x3, x4, x5]

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
compute_spdp(depth5_expr, "Depth-5 Polynomial")
