#!/usr/bin/env python3
"""
spdp_backend.py

Backend adapter for spdp_emergence_test.py

It plugs your existing pipeline code (spdp_pipeline_sanity.py + spdp_exact.py)
into the emergence ablation harness by exposing:

  compute_matrix(instance: dict, regime: str) -> (M, meta)

Regimes:
  - R0_RAW  : no profile compression; variables are kept distinct (but remapped to 0..L-1)
             meta["profiles"] reports the *measured* number of distinct profile signatures in the raw window.
  - R1_WEAK : canonical renaming only (sort vars by signature, then remap uniquely); no quotienting.
             meta["profiles"] is again the measured distinct-signature count.
  - R2_FULL : profile compression / interface-anonymous quotient (your current compress_window_by_profile).

Matrix returned is a *row-sampled SPDP row span* (lower bound):
  rows correspond to randomly sampled (k-derivative set S, shift monomial m of degree <= ell)
  columns correspond to monomials of degree <= D = deg(p)-k+ell in the Boolean quotient.

This avoids constructing the full (potentially enormous) SPDP matrix while remaining
comparable across regimes (same sampling policy keyed by seed+regime).

Usage: put this file next to spdp_pipeline_sanity.py and spdp_exact.py, then run
  python spdp_emergence_test.py --backend spdp_backend ...
"""

from __future__ import annotations

import math
import itertools
from typing import Dict, Any, Tuple, List

import numpy as np

# Import your pipeline building blocks
import spdp_pipeline_sanity as pipe
from spdp_exact import BoolPoly, PRIME


def _canon_remap_unique_by_signature(win2: List[pipe.Clause]) -> Tuple[List[pipe.Clause], int, int]:
    """
    Canonical renaming ONLY (no quotient):
      - compute signature for each var in window
      - sort vars by (signature, original_id)
      - assign new ids 0..L-1 uniquely
    Returns (remapped_window, L, Praw_unique_signatures)
    """
    vars_in_win = sorted({v for cl in win2 for (v, _) in cl})
    sigs = {v: pipe.var_profile_signature(v, win2) for v in vars_in_win}
    unique_sigs = sorted(set(sigs.values()))
    Praw = len(unique_sigs)

    ordered = sorted(vars_in_win, key=lambda v: (sigs[v], v))
    v_to_new = {v: i for i, v in enumerate(ordered)}

    remapped = [[(v_to_new[v], neg) for (v, neg) in cl] for cl in win2]
    return remapped, len(vars_in_win), Praw


def _raw_remap_keep_ids(win2: List[pipe.Clause]) -> Tuple[List[pipe.Clause], int, int]:
    """
    No canonicalization, no quotient:
      - remap just to a dense 0..L-1 index set (preserves distinctness)
      - measure Praw = number of distinct signatures (reported, but not used to quotient)
    """
    vars_in_win = sorted({v for cl in win2 for (v, _) in cl})
    sigs = {v: pipe.var_profile_signature(v, win2) for v in vars_in_win}
    Praw = len(set(sigs.values()))

    v_to_new = {v: i for i, v in enumerate(vars_in_win)}
    remapped = [[(v_to_new[v], neg) for (v, neg) in cl] for cl in win2]
    return remapped, len(vars_in_win), Praw


def _cols_up_to_degree(n_vars: int, D: int) -> List[int]:
    """List of monomial bitmasks of degree <= D (Boolean/multilinear)."""
    cols: List[int] = []
    for d in range(0, D + 1):
        for comb in itertools.combinations(range(n_vars), d):
            mask = 0
            for i in comb:
                mask |= 1 << i
            cols.append(mask)
    return cols


def _shifts_up_to_degree(n_vars: int, ell: int) -> List[int]:
    """Shift monomials (bitmasks) of degree <= ell in Boolean/multilinear ring."""
    shifts: List[int] = [0]
    for d in range(1, ell + 1):
        for comb in itertools.combinations(range(n_vars), d):
            mask = 0
            for i in comb:
                mask |= 1 << i
            shifts.append(mask)
    return shifts


def _sample_row_span_matrix(poly: BoolPoly, k: int, ell: int, seed: int, max_rows: int = 4096) -> np.ndarray:
    """
    Build a sampled SPDP row-span matrix for rank estimation.
    Rows are sampled (S, m) pairs:
      S: k-subset (as bitmask) for derivative_set
      m: shift monomial bitmask (deg<=ell)
    """
    n_vars = poly.n
    if n_vars < k or k <= 0:
        return np.zeros((1, 1), dtype=np.int64)

    deg = poly.degree()
    D = max(0, deg - k + ell)
    cols = _cols_up_to_degree(n_vars, D)
    col_index = {m: i for i, m in enumerate(cols)}
    ncols = len(cols)

    # If there are no columns (shouldn't happen), return trivial matrix
    if ncols == 0:
        return np.zeros((1, 1), dtype=np.int64)

    shifts = _shifts_up_to_degree(n_vars, ell)

    # Choose number of sampled rows; enough to saturate rank up to ncols
    target = max(512, min(max_rows, 4 * ncols))
    rng = np.random.default_rng(seed)

    M = np.zeros((target, ncols), dtype=np.int64)

    for t in range(target):
        # sample k distinct indices for S
        idx = rng.choice(n_vars, size=k, replace=False)
        S_mask = 0
        for i in idx:
            S_mask |= (1 << int(i))

        m_shift = int(shifts[int(rng.integers(0, len(shifts)))])

        dp = poly.derivative_set(S_mask)
        if not dp.terms:
            continue
        g = dp.mul_monomial(m_shift)

        # Fill row t
        for mon, coeff in g.terms.items():
            j = col_index.get(mon)
            if j is not None:
                M[t, j] = (M[t, j] + int(coeff)) % PRIME

    return M


def compute_matrix(instance: Dict[str, Any], regime: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Required by spdp_emergence_test.py

    instance fields used:
      - family: str (label only)
      - n: int (n_inputs)
      - seed: int
      - payload: optional dict with overrides:
          m_clauses: int
          k: int
          ell: int
          k_window: int  (window size override)
    """
    fam = instance.get("family", "RandDeg3")
    n_inputs = int(instance.get("n", 1024))
    seed = int(instance.get("seed", 0))
    payload = instance.get("payload", {}) or {}

    # Match your sanity suite defaults unless overridden
    m_clauses = int(payload.get("m_clauses", 3 * n_inputs))
    k = int(payload.get("k", 3))
    ell = int(payload.get("ell", 2))

    # Build CNF via Tseitin
    _n_total, cnf = pipe.build_clause_formula_tseitin(n_inputs, m_clauses, seed=seed)

    # Restrict inputs with p = 1/sqrt(n)
    assignment = pipe.input_restriction(n_inputs, p_keep=1.0 / math.sqrt(max(n_inputs, 1)), seed=seed + 123)

    simplified: List[pipe.Clause] = []
    for cl in cnf:
        s = pipe.simplify_clause(cl, assignment)
        if s is None:
            continue
        simplified.append(s)  # allow empties

    # Window selection (kept identical across regimes for fair ablation)
    k_window = int(payload.get("k_window", max(24, int(math.ceil(4 * math.log(max(n_inputs, 2), 2))))))
    win = pipe.local_window(simplified, k_window=k_window, seed=seed + 999)

    empty_ct = sum(1 for cl in win if len(cl) == 0)
    win2 = [cl for cl in win if len(cl) > 0]

    # Handle degenerate case
    if not win2 and empty_ct == 0:
        return np.zeros((1, 1), dtype=np.int64), {"live_vars": 0, "profiles": 0}

    # Apply regime transform
    if regime == "R0_RAW":
        reg_win, L, Praw = _raw_remap_keep_ids(win2)
        n_vars = L
        P_report = Praw  # measured, not imposed
    elif regime == "R1_WEAK":
        reg_win, L, Praw = _canon_remap_unique_by_signature(win2)
        n_vars = L
        P_report = Praw  # measured, not imposed
    elif regime == "R2_FULL":
        # Full interface-anonymous quotient (your current proxy)
        reg_win, P = pipe.compress_window_by_profile(win2)
        n_vars = P
        P_report = P
    else:
        raise ValueError(f"Unknown regime: {regime}")

    # CNF -> polynomial (violation sum)
    poly = pipe.cnf_to_poly(reg_win, n_vars)
    if empty_ct > 0:
        poly = poly.add(BoolPoly.one(n_vars).scale(empty_ct))

    # Sampled SPDP row-span matrix
    # Deterministic by seed+regime to keep runs comparable
    regime_salt = (hash(regime) % 100000)
    M = _sample_row_span_matrix(poly, k=k, ell=ell, seed=seed + regime_salt, max_rows=int(payload.get("max_rows_backend", 4096)))

    meta = {
        "live_vars": int(n_vars if regime != "R2_FULL" else P_report),  # for FULL, live_vars==profiles after quotient
        "profiles": int(P_report),
        "k": k,
        "ell": ell,
        "prime": PRIME,
        "window_size": k_window,
        "empty_clauses": int(empty_ct),
        "note": "Matrix is a sampled SPDP row-span (lower bound rank).",
        "family": fam,
    }
    # For RAW/WEAK, also expose raw live var count
    if regime in ("R0_RAW", "R1_WEAK"):
        meta["raw_live_vars"] = int(n_vars)

    return M, meta
