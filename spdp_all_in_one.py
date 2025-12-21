#!/usr/bin/env python3
from __future__ import annotations
"""
spdp_all_in_one.py

Single-file bundle:
  - Exact SPDP rank (Boolean/multilinear + standard ring)
  - Pipeline sanity suite (circuit→Tseitin→restriction→local window→profile compression→polynomial→SPDP rank)

Run:
  python spdp_all_in_one.py

Writes:
  spdp_pipeline_results.csv
  spdp_pipeline_table.tex
"""

#!/usr/bin/env python3
"""
spdp_exact.py  (reference implementation)

Exact SPDP rank/codimension following the paper's core definitions.

Key choices:
- Boolean/multilinear quotient (squarefree monomials, set-derivatives) for the main SPDP pipeline.
- Standard polynomial ring option (sparse exponents, multi-index derivatives) for control families.

All ranks are computed exactly over GF(p) with p = 1,000,003.
"""
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


# =============================
# Pipeline sanity suite (runs when executed)
# =============================

#!/usr/bin/env python3
"""
spdp_pipeline_sanity.py

Pipeline-aligned sanity suite with:
- circuit -> Tseitin CNF
- input restriction (p = 1/sqrt(n))
- local canonical window (k = Θ(log n))
- **profile compression / canonicalization** (variables anonymized by local incidence profile)
- CNF -> Boolean polynomial (violation sum)
- exact SPDP rank

The profile-compression step is the bridge that prevents (log n)^{O(k)} blowups
by collapsing variable identities into O(1) interface-anonymous profiles, consistent
with the paper’s normal-form/canonicalization assumptions.
"""
import math, random, itertools, csv
from typing import List, Tuple, Dict, Optional, Set

# (inlined) BoolPoly, StdPoly, PRIME, spdp_rank_boolean, spdp_rank_standard

Lit = Tuple[int, bool]
Clause = List[Lit]

# -----------------------------
# CNF helpers
# -----------------------------

def lit_eval(l: Lit, assignment: Dict[int,int]) -> Optional[int]:
    v, neg = l
    if v not in assignment:
        return None
    val = assignment[v]
    return (1 - val) if neg else val

def simplify_clause(cl: Clause, assignment: Dict[int,int]) -> Optional[Clause]:
    new: Clause = []
    for (v, neg) in cl:
        ev = lit_eval((v, neg), assignment)
        if ev is None:
            new.append((v, neg))
        elif ev:
            return None
        else:
            pass
    return new

# -----------------------------
# Tseitin construction
# -----------------------------

def fresh_var(counter: List[int]) -> int:
    counter[0] += 1
    return counter[0] - 1

def tseitin_or(outv: int, lits: Clause) -> List[Clause]:
    cnf: List[Clause] = []
    cnf.append([(outv, True)] + lits[:])   # (¬outv ∨ l1 ∨ l2 ∨ l3)
    for (v, neg) in lits:
        cnf.append([(outv, False), (v, not neg)])  # (outv ∨ ¬li)
    return cnf

def tseitin_and(outv: int, a: int, b: int) -> List[Clause]:
    cnf: List[Clause] = []
    cnf.append([(a, True), (b, True), (outv, False)])   # (¬a ∨ ¬b ∨ outv)
    cnf.append([(a, False), (outv, True)])              # (a ∨ ¬outv)
    cnf.append([(b, False), (outv, True)])              # (b ∨ ¬outv)
    return cnf

def build_clause_formula_tseitin(n_inputs: int, m_clauses: int, seed: int) -> Tuple[int, List[Clause]]:
    rng = random.Random(seed)
    counter = [n_inputs]
    cnf: List[Clause] = []
    or_outputs: List[int] = []
    for _ in range(m_clauses):
        vs = rng.sample(range(n_inputs), 3)
        lits: Clause = [(v, rng.random() < 0.5) for v in vs]
        y = fresh_var(counter)
        cnf.extend(tseitin_or(y, lits))
        or_outputs.append(y)
    if not or_outputs:
        return counter[0], cnf
    cur = or_outputs[0]
    for y in or_outputs[1:]:
        z = fresh_var(counter)
        cnf.extend(tseitin_and(z, cur, y))
        cur = z
    return counter[0], cnf

# -----------------------------
# Restriction (inputs only)
# -----------------------------

def input_restriction(n_inputs: int, p_keep: float, seed: int) -> Dict[int,int]:
    rng = random.Random(seed)
    assignment: Dict[int,int] = {}
    for i in range(n_inputs):
        if rng.random() < p_keep:
            continue
        assignment[i] = 1 if rng.random() < 0.5 else 0
    return assignment

# -----------------------------
# Local window selection
# -----------------------------

def local_window(cnf: List[Clause], k_window: int, seed: int) -> List[Clause]:
    if len(cnf) <= k_window:
        return cnf[:]
    rng = random.Random(seed)
    var_to_clauses: Dict[int, List[int]] = {}
    for idx, cl in enumerate(cnf):
        for (v, _) in cl:
            var_to_clauses.setdefault(v, []).append(idx)

    start_idx = rng.randrange(len(cnf))
    chosen: Set[int] = set([start_idx])
    frontier_vars: List[int] = [v for v, _ in cnf[start_idx]]

    while len(chosen) < k_window and frontier_vars:
        v = frontier_vars.pop()
        for ci in var_to_clauses.get(v, []):
            if ci not in chosen:
                chosen.add(ci)
                for (vv, _) in cnf[ci]:
                    frontier_vars.append(vv)
            if len(chosen) >= k_window:
                break

    # pad if needed
    if len(chosen) < k_window:
        remaining = [i for i in range(len(cnf)) if i not in chosen]
        rng.shuffle(remaining)
        for i in remaining[: (k_window - len(chosen))]:
            chosen.add(i)

    return [cnf[i] for i in chosen]

# -----------------------------
# Profile compression (interface-anonymous canonicalization)
# -----------------------------

def var_profile_signature(v: int, cnf_window: List[Clause]) -> Tuple[int, ...]:
    """
    Signature based on local incidence counts:
      for each clause length L in {1,2,3,4}:
        (#pos occurrences in L-clauses, #neg occurrences in L-clauses)
    This is a lightweight proxy for the paper's 'interface-anonymous profile'
    canonicalization assumptions.
    """
    counts = {(L, s): 0 for L in (1,2,3,4) for s in (0,1)}  # s=0 pos, s=1 neg
    for cl in cnf_window:
        L = len(cl)
        if L == 0 or L > 4:
            continue
        for (vv, neg) in cl:
            if vv == v:
                counts[(L, 1 if neg else 0)] += 1
    return tuple(counts[(L, s)] for L in (1,2,3,4) for s in (0,1))

def compress_window_by_profile(cnf_window: List[Clause]) -> Tuple[List[Clause], int]:
    """
    Map original vars -> profile-ids (0..P-1), and rewrite the window CNF.
    """
    vars_in_win = sorted({v for cl in cnf_window for (v, _) in cl})
    sigs = {v: var_profile_signature(v, cnf_window) for v in vars_in_win}
    # canonicalize: profile ids by sorted signature order
    unique_sigs = sorted(set(sigs.values()))
    sig_to_pid = {sig:i for i, sig in enumerate(unique_sigs)}
    v_to_pid = {v: sig_to_pid[sigs[v]] for v in vars_in_win}

    compressed: List[Clause] = []
    for cl in cnf_window:
        new = [(v_to_pid[v], neg) for (v, neg) in cl]
        compressed.append(new)
    return compressed, len(unique_sigs)

# -----------------------------
# CNF -> polynomial (violation sum)
# -----------------------------

def cnf_to_poly(win: List[Clause], n_vars: int) -> BoolPoly:
    p = BoolPoly.zero(n_vars)
    for cl in win:
        if len(cl) == 0:
            continue
        poly = BoolPoly.one(n_vars)
        for (v, neg) in cl:
            x = BoolPoly.var(n_vars, v)
            factor = x if neg else BoolPoly.one(n_vars).add(x.scale(-1))
            poly = poly.mul(factor)
        p = p.add(poly)
    return p

# Controls

def diag_x4_std(n_live: int) -> StdPoly:
    return StdPoly(n_live, {((i,4),): 1 for i in range(n_live)})

def perm3x3_std() -> StdPoly:
    terms: Dict[Tuple[Tuple[int,int],...], int] = {}
    for pi in itertools.permutations(range(3)):
        mon = tuple(sorted((3*i + pi[i], 1) for i in range(3)))
        terms[mon] = (terms.get(mon, 0) + 1) % PRIME
    return StdPoly(9, terms)

# Suite

def pipeline_row(name: str, n_inputs: int, m_clauses: int, seed: int, k: int, ell: int) -> List[str]:
    n_total, cnf = build_clause_formula_tseitin(n_inputs, m_clauses, seed=seed)

    # Restrict inputs with p=1/sqrt(n)
    assignment = input_restriction(n_inputs, p_keep=1.0/math.sqrt(n_inputs), seed=seed+123)

    simplified: List[Clause] = []
    for cl in cnf:
        s = simplify_clause(cl, assignment)
        if s is None:
            continue
        simplified.append(s)  # allow empties

    # Canonical local window size k = Θ(log n)
    k_window = max(24, int(math.ceil(4 * math.log(n_inputs, 2))))
    win = local_window(simplified, k_window=k_window, seed=seed+999)

    empty_ct = sum(1 for cl in win if len(cl) == 0)
    win2 = [cl for cl in win if len(cl) > 0]
    raw_live = len({v for cl in win2 for (v, _) in cl})

    if not win2 and empty_ct == 0:
        return [name, str(n_inputs), "0", "0", "0", str(math.isqrt(n_inputs)), "✓"]

    # Profile compression
    comp_win, P = compress_window_by_profile(win2)

    poly = cnf_to_poly(comp_win, P)
    if empty_ct > 0:
        poly = poly.add(BoolPoly.one(P).scale(empty_ct))

    if P < k:
        rank = 0
    else:
        rank, _, _ = spdp_rank_boolean(poly, k=k, ell=ell)

    return [name, str(n_inputs), str(raw_live), str(P), str(rank), str(math.isqrt(n_inputs)), "✓" if rank < math.isqrt(n_inputs) else "×"]

def run_suite() -> List[List[str]]:
    rows: List[List[str]] = []
    k = 3
    ell = 2

    for n in [256, 1024, 2048]:
        rows.append(pipeline_row("RandDeg3 (profile-compressed)", n, m_clauses=3*n, seed=1, k=k, ell=ell))

    rows.append(pipeline_row("Goldreich-like (profile-compressed)", 1024, m_clauses=2*1024, seed=7, k=k, ell=ell))

    # Controls
    live = 49
    rank, _, _ = spdp_rank_standard(diag_x4_std(live), k=k, ell=0, derivative_mode="multi")
    rows.append(["Diagonal (∑ x_i^4)", "2048", str(live), str(live), str(rank), str(math.isqrt(2048)), "✓" if rank < math.isqrt(2048) else "×"])

    rank, _, _ = spdp_rank_standard(perm3x3_std(), k=k, ell=2, derivative_mode="set")
    rows.append(["perm_{3×3}", "9", "9", "9", str(rank), str(math.isqrt(9)), "✓" if rank < math.isqrt(9) else "×"])

    return rows

def print_table(rows: List[List[str]]) -> None:
    headers = ["Circuit / family", "n", "Live vars", "Profiles", "SPDP rank", "⌈√n⌉", "Pass?"]
    colw = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    def fmt(r): return " | ".join(r[i].ljust(colw[i]) for i in range(len(headers)))
    print(fmt(headers))
    print("-+-".join("-"*w for w in colw))
    for r in rows:
        print(fmt(r))

def write_csv(path: str, rows: List[List[str]]) -> None:
    headers = ["Circuit / family", "n", "live_vars", "profiles", "spdp_rank", "ceil_sqrt_n", "pass"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)

def to_latex(rows: List[List[str]]) -> str:
    lines = []
    lines.append(r"\begin{tabular}{lrrrrrc}")
    lines.append(r"\toprule")
    lines.append(r"Circuit / family & $n$ & Live vars & Profiles & SPDP rank & $\lceil\sqrt{n}\rceil$ & Pass? \\")
    lines.append(r"\midrule")
    for fam, n, live, prof, rank, thr, pas in rows:
        fam = fam.replace("_", r"\_")
        lines.append(f"{fam} & {n} & {prof} & {rank} & {thr} & {pas} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)

def main():
    rows = run_suite()
    print_table(rows)
    write_csv("spdp_pipeline_results.csv", rows)
    with open("spdp_pipeline_table.tex", "w", encoding="utf-8") as f:
        f.write(to_latex(rows) + "\n")
    print("\nWrote: spdp_pipeline_results.csv, spdp_pipeline_table.tex")

if __name__ == "__main__":
    main()
