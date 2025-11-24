"""
spdp_determinant.py

Computes the SPDP rank for the 3×3 determinant polynomial under partial
variable pruning. This test demonstrates that SPDP rank remains high for
algebraically structured functions with symmetry (like det), even after
projection.

Function:
    det_3x3 = symbolic determinant of a 3×3 matrix

SPDP parameters:
    k = 2   # order of partial derivatives
    ℓ = 1   # shift monomial degree
    r = full projection (minus pruned variables)

Pruned variables:
    x_{1,1}, x_{2,1}, x_{2,2}

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

# Define 3×3 matrix and compute determinant
X = sp.Matrix(3, 3, lambda i, j: sp.symbols(f'x{i}{j}'))
det_expr = sp.expand(X.det())
det_vars = [X[i, j] for i in range(3) for j in range(3)]

# Prune the same variables as for the permanent
pruned_vars = [X[1, 1], X[2, 1], X[2, 2]]
def prune_expr(expr, vars_to_zero):
    return sp.expand(expr.subs({v: 0 for v in vars_to_zero}))

pruned_det = prune_expr(det_expr, pruned_vars)

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
    partials = k_order_derivatives(expr, det_vars, k)
    shifts = shift_monomials(det_vars, l)
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
compute_spdp(pruned_det, "Determinant (3×3, pruned)")
