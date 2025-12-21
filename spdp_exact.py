#!/usr/bin/env python3
"""
spdp_exact.py  (reference implementation)

Exact SPDP rank/codimension following the paper's core definitions.

Key choices:
- Boolean/multilinear quotient (squarefree monomials, set-derivatives) for the main SPDP pipeline.
- Standard polynomial ring option (sparse exponents, multi-index derivatives) for control families.

All ranks are computed exactly over GF(p) with p = 1,000,003.
"""

from __future__ import annotations
import math, itertools
from dataclasses import dataclass
from typing import Dict, Tuple, Iterable, List

PRIME = 1_000_003

def mod_inv(a: int, p: int = PRIME) -> int:
    a %= p
    if a == 0:
        raise ZeroDivisionError("inverse of 0")
    return pow(a, p - 2, p)

# -----------------------------
# Boolean / multilinear quotient
# -----------------------------

@dataclass
class BoolPoly:
    """Multilinear polynomial over GF(PRIME) in the Boolean quotient ring."""
    n: int
    terms: Dict[int, int]  # bitmask -> coeff

    @staticmethod
    def zero(n: int) -> "BoolPoly":
        return BoolPoly(n, {})

    @staticmethod
    def one(n: int) -> "BoolPoly":
        return BoolPoly(n, {0: 1})

    @staticmethod
    def var(n: int, i: int) -> "BoolPoly":
        return BoolPoly(n, {1 << i: 1})

    def degree(self) -> int:
        return max((m.bit_count() for m in self.terms.keys()), default=0)

    def add(self, other: "BoolPoly") -> "BoolPoly":
        assert self.n == other.n
        out = dict(self.terms)
        for m, c in other.terms.items():
            out[m] = (out.get(m, 0) + c) % PRIME
            if out[m] == 0:
                del out[m]
        return BoolPoly(self.n, out)

    def scale(self, a: int) -> "BoolPoly":
        a %= PRIME
        if a == 0:
            return BoolPoly.zero(self.n)
        out = {m: (c * a) % PRIME for m, c in self.terms.items()}
        out = {m: c for m, c in out.items() if c != 0}
        return BoolPoly(self.n, out)

    def mul(self, other: "BoolPoly") -> "BoolPoly":
        """Product in the Boolean quotient (squarefree union of masks)."""
        assert self.n == other.n
        out: Dict[int, int] = {}
        for m1, c1 in self.terms.items():
            for m2, c2 in other.terms.items():
                m = m1 | m2
                out[m] = (out.get(m, 0) + c1 * c2) % PRIME
                if out[m] == 0:
                    del out[m]
        return BoolPoly(self.n, out)

    def mul_monomial(self, mono_mask: int) -> "BoolPoly":
        out: Dict[int, int] = {}
        for m, c in self.terms.items():
            mm = m | mono_mask
            out[mm] = (out.get(mm, 0) + c) % PRIME
            if out[mm] == 0:
                del out[mm]
        return BoolPoly(self.n, out)

    def derivative_set(self, S_mask: int) -> "BoolPoly":
        """Set-derivative ∂_S in the multilinear setting."""
        out: Dict[int, int] = {}
        for m, c in self.terms.items():
            if (m & S_mask) == S_mask:
                mm = m ^ S_mask
                out[mm] = (out.get(mm, 0) + c) % PRIME
                if out[mm] == 0:
                    del out[mm]
        return BoolPoly(self.n, out)

# -----------------------------
# Standard polynomial ring
# -----------------------------

Mon = Tuple[Tuple[int, int], ...]  # sorted (var, exp)

@dataclass
class StdPoly:
    n: int
    terms: Dict[Mon, int]

    @staticmethod
    def zero(n: int) -> "StdPoly":
        return StdPoly(n, {})

    def degree(self) -> int:
        return max((sum(e for _, e in mon) for mon in self.terms.keys()), default=0)

    def mul_monomial(self, mono: Mon) -> "StdPoly":
        mono_dict = dict(mono)
        out: Dict[Mon, int] = {}
        for mon, c in self.terms.items():
            md = dict(mon)
            for v, e in mono_dict.items():
                md[v] = md.get(v, 0) + e
            new_mon = tuple(sorted((v, e) for v, e in md.items() if e))
            out[new_mon] = (out.get(new_mon, 0) + c) % PRIME
            if out[new_mon] == 0:
                del out[new_mon]
        return StdPoly(self.n, out)

    def derivative_multi(self, alpha: Tuple[int, ...]) -> "StdPoly":
        out: Dict[Mon, int] = {}
        for mon, c0 in self.terms.items():
            md = dict(mon)
            coeff = c0
            ok = True
            for v, a in enumerate(alpha):
                if a == 0:
                    continue
                e = md.get(v, 0)
                if e < a:
                    ok = False
                    break
                ff = 1
                for t in range(a):
                    ff = (ff * (e - t)) % PRIME
                coeff = (coeff * ff) % PRIME
                new_e = e - a
                if new_e == 0:
                    md.pop(v, None)
                else:
                    md[v] = new_e
            if not ok or coeff == 0:
                continue
            new_mon = tuple(sorted((v, e) for v, e in md.items() if e))
            out[new_mon] = (out.get(new_mon, 0) + coeff) % PRIME
            if out[new_mon] == 0:
                del out[new_mon]
        return StdPoly(self.n, out)

# -----------------------------
# Enumerators
# -----------------------------

def enumerate_shifts_boolean(n: int, ell: int) -> List[int]:
    out = [0]
    for d in range(1, ell + 1):
        for comb in itertools.combinations(range(n), d):
            mask = 0
            for i in comb:
                mask |= 1 << i
            out.append(mask)
    return out

def enumerate_derivatives_set(n: int, k: int) -> List[int]:
    out: List[int] = []
    for comb in itertools.combinations(range(n), k):
        mask = 0
        for i in comb:
            mask |= 1 << i
        out.append(mask)
    return out

def enumerate_shifts_standard(n: int, ell: int) -> List[Mon]:
    out: List[Mon] = [tuple()]
    for t in range(1, ell + 1):
        for idxs in itertools.combinations_with_replacement(range(n), t):
            exp: Dict[int, int] = {}
            for i in idxs:
                exp[i] = exp.get(i, 0) + 1
            out.append(tuple(sorted(exp.items())))
    return out

def enumerate_derivatives_multi(n: int, k: int):
    alpha = [0] * n
    def rec(i: int, rem: int):
        if i == n - 1:
            alpha[i] = rem
            yield tuple(alpha)
            return
        for a in range(rem + 1):
            alpha[i] = a
            yield from rec(i + 1, rem - a)
    return rec(0, k)

# -----------------------------
# Streaming rank (sparse Gaussian elim)
# -----------------------------

def rank_stream(vectors: Iterable[Dict[int, int]], ncols: int) -> int:
    basis: Dict[int, Dict[int, int]] = {}
    rank = 0
    for vec0 in vectors:
        v = {j: (val % PRIME) for j, val in vec0.items() if (val % PRIME) != 0}
        while v:
            pivot = max(v.keys())
            if pivot in basis:
                b = basis[pivot]
                factor = v[pivot]
                for j, bj in b.items():
                    v[j] = (v.get(j, 0) - factor * bj) % PRIME
                    if v[j] == 0:
                        v.pop(j, None)
            else:
                inv = mod_inv(v[pivot])
                for j in list(v.keys()):
                    v[j] = (v[j] * inv) % PRIME
                basis[pivot] = v
                rank += 1
                break
    return rank

# -----------------------------
# SPDP rank
# -----------------------------

def spdp_rank_boolean(p: BoolPoly, k: int, ell: int) -> Tuple[int, int, int]:
    deg = p.degree()
    D = max(0, deg - k + ell)

    cols: List[int] = []
    for d in range(0, D + 1):
        for comb in itertools.combinations(range(p.n), d):
            mask = 0
            for i in comb:
                mask |= 1 << i
            cols.append(mask)
    col_index = {m: i for i, m in enumerate(cols)}
    ncols = len(cols)

    shifts = enumerate_shifts_boolean(p.n, ell)
    derivs = enumerate_derivatives_set(p.n, k) if k <= p.n else []

    def vectors():
        for S in derivs:
            dp = p.derivative_set(S)
            if not dp.terms:
                continue
            for m in shifts:
                g = dp.mul_monomial(m)
                vec: Dict[int, int] = {}
                for mon, coeff in g.terms.items():
                    j = col_index.get(mon)
                    if j is not None:
                        vec[j] = (vec.get(j, 0) + coeff) % PRIME
                if vec:
                    yield vec

    r = rank_stream(vectors(), ncols)
    return r, ncols, D

def spdp_rank_standard(p: StdPoly, k: int, ell: int, derivative_mode: str = "multi") -> Tuple[int, int, int]:
    deg = p.degree()
    D = max(0, deg - k + ell)

    cols: List[Mon] = [tuple()]
    for t in range(1, D + 1):
        for idxs in itertools.combinations_with_replacement(range(p.n), t):
            exp: Dict[int, int] = {}
            for i in idxs:
                exp[i] = exp.get(i, 0) + 1
            cols.append(tuple(sorted(exp.items())))
    col_index = {m: i for i, m in enumerate(cols)}
    ncols = len(cols)

    shifts = enumerate_shifts_standard(p.n, ell)

    if derivative_mode == "multi":
        deriv_iter = enumerate_derivatives_multi(p.n, k)
    elif derivative_mode == "set":
        # set-derivatives as special case of multi with 0/1 exponents
        masks = list(itertools.combinations(range(p.n), k))
        alphas = []
        for comb in masks:
            alpha = [0]*p.n
            for i in comb:
                alpha[i]=1
            alphas.append(tuple(alpha))
        deriv_iter = alphas
    else:
        raise ValueError("derivative_mode must be 'multi' or 'set'")

    def vectors():
        for alpha in deriv_iter:
            dp = p.derivative_multi(alpha)
            if not dp.terms:
                continue
            for m in shifts:
                g = dp.mul_monomial(m)
                vec: Dict[int, int] = {}
                for mon, coeff in g.terms.items():
                    j = col_index.get(mon)
                    if j is not None:
                        vec[j] = (vec.get(j, 0) + coeff) % PRIME
                if vec:
                    yield vec

    r = rank_stream(vectors(), ncols)
    return r, ncols, D
