#!/usr/bin/env python3
"""SPDP Emergence Test (ETP-1)

Ablation test to determine whether SPDP rank collapse is present in RAW windows
(i.e., without canonicalization/profile compression), versus how much the proof-aligned
canonicalization/compression sharpens it.

You run the SAME instances/seeds through three regimes:

  R0_RAW   : no canonicalization, no profile compression
  R1_WEAK  : weaker-than-proof canonicalization/compression (backend-defined)
  R2_FULL  : proof regime (canonical windows + interface-anonymous profile compression)

This script is backend-agnostic: you provide a backend module that implements:

  compute_matrix(instance: dict, regime: str) -> (M: np.ndarray, meta: dict)

Meta must contain:
  - live_vars : int (L)
  - profiles  : int (P) observed under that regime (measured, not assumed)

Rank is computed exactly over F_p by Gaussian elimination mod a prime.
For large matrices, optional sketching (row/col subsampling) gives a rank LOWER BOUND.
"""

from __future__ import annotations

import argparse, importlib, json, pathlib, time, csv, math
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def _inv_mod(a: int, p: int) -> int:
    return pow(int(a) % p, p - 2, p)


def rank_mod_prime(M: np.ndarray, p: int) -> int:
    """Exact rank over F_p via row-reduction."""
    if M.ndim != 2:
        raise ValueError("M must be 2D")
    A = (M.astype(np.int64, copy=False) % p).copy()
    m, n = A.shape
    r = 0
    row = 0
    for col in range(n):
        pivot = None
        for i in range(row, m):
            if int(A[i, col]) % p != 0:
                pivot = i
                break
        if pivot is None:
            continue
        if pivot != row:
            A[[row, pivot]] = A[[pivot, row]]
        inv = _inv_mod(int(A[row, col]), p)
        A[row, :] = (A[row, :] * inv) % p
        for i in range(m):
            if i == row:
                continue
            factor = int(A[i, col]) % p
            if factor != 0:
                A[i, :] = (A[i, :] - factor * A[row, :]) % p
        r += 1
        row += 1
        if row == m:
            break
    return r


def sketch_matrix(M: np.ndarray, max_rows: Optional[int], max_cols: Optional[int], seed: int) -> np.ndarray:
    """Randomly subsample rows/cols (rank lower bound)."""
    rng = np.random.default_rng(seed)
    m, n = M.shape
    row_idx = np.arange(m)
    col_idx = np.arange(n)
    if max_rows is not None and m > max_rows:
        row_idx = rng.choice(m, size=max_rows, replace=False)
    if max_cols is not None and n > max_cols:
        col_idx = rng.choice(n, size=max_cols, replace=False)
    return M[np.sort(row_idx)][:, np.sort(col_idx)]


@dataclass
class RegimeResult:
    instance_id: str
    family: str
    n: int
    seed: int
    regime: str
    live_vars: int
    profiles: int
    rows: int
    cols: int
    rank: int
    rank_ratio_L: float
    profiles_ratio_L: float
    rank_ratio_profiles: float
    elapsed_s: float


def load_instances(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def mean(xs: List[float]) -> float:
    return float(sum(xs) / max(len(xs), 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True, help="JSONL instances file")
    ap.add_argument("--backend", required=True, help="Python module path, e.g. spdp_backend")
    ap.add_argument("--outdir", default="spdp_emergence_out", help="Output directory")
    ap.add_argument("--prime", type=int, default=2147483647, help="Prime modulus for rank")
    ap.add_argument("--regimes", nargs="+", default=["R0_RAW", "R1_WEAK", "R2_FULL"])
    ap.add_argument("--max-rows", type=int, default=None, help="Sketch cap rows")
    ap.add_argument("--max-cols", type=int, default=None, help="Sketch cap cols")
    ap.add_argument("--tau", type=float, default=0.20, help="Emergence threshold for rank_ratio_L (RAW)")
    ap.add_argument("--progress", action="store_true", help="Show progress bars")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    backend = importlib.import_module(args.backend)
    if not hasattr(backend, "compute_matrix"):
        raise RuntimeError("Backend must define compute_matrix(instance, regime)")

    instances = load_instances(args.instances)
    it = instances
    if args.progress and tqdm is not None:
        it = tqdm(it, desc="Instances")

    results: List[RegimeResult] = []

    for inst in it:
        iid = inst.get("id", f"{inst.get('family','?')}_n{inst.get('n','?')}_s{inst.get('seed','?')}")
        fam = inst.get("family", "?")
        n = int(inst.get("n", -1))
        seed = int(inst.get("seed", 0))

        for regime in args.regimes:
            t0 = time.time()
            M, meta = backend.compute_matrix(inst, regime)
            if not isinstance(M, np.ndarray):
                M = np.asarray(M)

            M2 = sketch_matrix(M, args.max_rows, args.max_cols, seed=seed)
            r = rank_mod_prime(M2, args.prime)
            elapsed = time.time() - t0

            L = int(meta.get("live_vars", M.shape[1]))
            P = int(meta.get("profiles", -1))

            rrL = float(r) / float(max(L, 1))
            prL = float(P) / float(max(L, 1)) if P >= 0 else float("nan")
            rrP = float(r) / float(max(P, 1)) if P > 0 else float("nan")

            results.append(RegimeResult(
                instance_id=str(iid), family=str(fam), n=n, seed=seed, regime=str(regime),
                live_vars=L, profiles=P, rows=int(M2.shape[0]), cols=int(M2.shape[1]),
                rank=int(r), rank_ratio_L=rrL, profiles_ratio_L=prL, rank_ratio_profiles=rrP,
                elapsed_s=float(elapsed)
            ))

    # CSV output
    csv_path = outdir / "emergence_ablation.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(RegimeResult.__annotations__.keys()))
        for x in results:
            w.writerow([
                x.instance_id, x.family, x.n, x.seed, x.regime, x.live_vars, x.profiles,
                x.rows, x.cols, x.rank, f"{x.rank_ratio_L:.6f}", f"{x.profiles_ratio_L:.6f}",
                f"{x.rank_ratio_profiles:.6f}", f"{x.elapsed_s:.3f}"
            ])

    # Emergence scores by family (RAW only)
    fam_scores: Dict[str, Dict[str, Any]] = {}
    for fam in sorted({r.family for r in results}):
        raw = [r for r in results if r.family == fam and r.regime == "R0_RAW"]
        if raw:
            score = sum(1 for r in raw if (r.rank / max(r.cols,1)) <= args.tau) / len(raw)
            fam_scores[fam] = {"raw_count": len(raw), "E_tau": score, "tau": args.tau}

    score_path = outdir / "emergence_scores.json"
    with open(score_path, "w", encoding="utf-8") as f:
        json.dump(fam_scores, f, indent=2)

    # LaTeX table: mean over seeds per family/regime
    families = sorted({r.family for r in results})
    regimes = list(args.regimes)

    agg_rows = []
    for fam in families:
        for reg in regimes:
            rr = [r for r in results if r.family == fam and r.regime == reg]
            if not rr:
                continue
            Lm = int(round(mean([float(x.live_vars) for x in rr])))
            Pvals = [float(x.profiles) for x in rr if x.profiles >= 0]
            Pm = mean(Pvals) if Pvals else float("nan")
            rm = mean([float(x.rank) for x in rr])
            rrL = mean([float(x.rank_ratio_L) for x in rr])
            prL_vals = [float(x.profiles_ratio_L) for x in rr if not math.isnan(x.profiles_ratio_L)]
            prL = mean(prL_vals) if prL_vals else float("nan")
            agg_rows.append((fam, reg, Lm, Pm, rm, rrL, prL))

    tex_path = outdir / "table_emergence_ablation.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\\centering\n")
        f.write("\\caption{Emergence ablation: raw vs weak vs full canonicalization. Reported values are means over seeds. "
                " $L$ = live variables, $P$ = observed profiles, $r$ = rank over $\\mathbb{F}_p$.}\\n")
        f.write("\\begin{tabular}{llrrrrr}\\toprule\n")
        f.write("Family & Regime & $L$ & $P$ & $r$ & $r/L$ & $P/L$ \\\\ \\midrule\n")
        for fam, reg, Lm, Pm, rm, rrL, prL in agg_rows:
            Pcell = f"{Pm:.1f}" if not np.isnan(Pm) else "--"
            prcell = f"{prL:.3f}" if not np.isnan(prL) else "--"
            f.write(f"{fam} & {reg} & {Lm:d} & {Pcell} & {rm:.1f} & {rrL:.3f} & {prcell} \\\\n")
        f.write("\\bottomrule\\end{tabular}\n")
        f.write("\\end{table}\n")

    print("Wrote:")
    print(f"  {csv_path}")
    print(f"  {score_path}")
    print(f"  {tex_path}")
    if fam_scores:
        print("\nEmergence scores (RAW):")
        for fam, d in fam_scores.items():
            print(f"  {fam:20s} E_tau={d['E_tau']:.3f} (tau={d['tau']}, n={d['raw_count']})")


if __name__ == "__main__":
    main()
