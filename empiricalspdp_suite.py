# SPDX: MIT
"""
spdp_suite_FIXED.py  –  CORRUPTION-FREE full SPDP suite
    • FIXED resume logic (no duplicates)  • row-by-row CSV append  
    • alternating GPUs  • n 46–64 step 2 (continue from your clean data)
    • adaptive sampling + derivative order cap for crash prevention
    • ENHANCED CSV OUTPUT with timing and complexity metrics
empirical_results_full/
├── scaling.csv Appendix A.1–A.6
├── easy_vs_hard.csv Appendix A.5, Figure 4
├── spdp_vs_spd.csv Appendix E.10
├── core_rules.csv Appendix E.6
└── heatmap.csv Appendix E.10–11


"""

import csv, gc, itertools, math, os, time
from itertools import combinations, product
from math import ceil, log2, comb
from pathlib import Path
import numpy as np

# ───── CONFIG ─────
GPU_IDS        = [0, 1]        # set [0] for single GPU
N_MIN, N_MAX   = 46, 64        # CONTINUE from where clean data ends
STEP           = 2
BASE_SAMPLES   = 100
POLY_COUNT     = 5
OUTDIR         = Path("empirical_results_full")
OUTDIR.mkdir(exist_ok=True)

# ───── ADAPTIVE PARAMETERS for n=64 feasibility ─────
def get_adaptive_params(n):
    """Scale down computation for larger n to maintain feasibility"""
    # Adaptive sampling: more samples for small n, fewer for large n
    samples = max(20, min(100, 120 - n))
    
    # Derivative order cap: prevent explosive recursion
    k = min(int(log2(n)), 4)  # Cap at 4th derivatives
    
    print(f"Adaptive params for n={n}: samples={samples}, k={k}")
    return samples, k

# ───── basic helpers ─────
def rand_poly_list(n, count=POLY_COUNT, degree=3, seed=0):
    rng = np.random.default_rng(seed)
    monos = [(tuple(rng.choice(n, degree, replace=False)), rng.integers(1, 5))
             for _ in range(count)]
    def build(m):      # single-monomial poly
        def poly(x): return sum(c * np.prod(x[list(sub)], dtype=float)
                                for sub, c in m)
        return poly
    return [build([m]) for m in monos]

def dderiv(p, x, d, k, delta=1e-3):
    return p(x) if k == 0 else (dderiv(p,x+delta*d,d,k-1,delta) -
                                dderiv(p,x-delta*d,d,k-1,delta))/(2*delta)

def spdp(polys, n, k, r, samples, seed):
    total_projections = comb(n, r)
    print(f"Computing SPDP matrix: {total_projections} × {samples}")
    
    rng = np.random.default_rng(seed)
    rows = []
    
    # Progress reporting for large computations
    report_interval = max(1, total_projections // 10)  # Report every 10%
    
    for i, proj in enumerate(combinations(range(n), r)):
        if i % report_interval == 0:
            progress = (i / total_projections) * 100
            print(f"  Progress: {progress:.1f}% ({i}/{total_projections})")
        
        d = np.zeros(n); d[list(proj)] = 1
        rows.append([sum(dderiv(p,rng.random(n),d,k) for p in polys)
                     for _ in range(samples)])
    
    result = np.asarray(rows, np.float32)
    print(f"Matrix shape: {result.shape}")
    return result

def rank_gpu(a, gpu=0, tol=1e-8):
    print(f"Computing rank on GPU {gpu}, matrix shape: {a.shape}")
    try:
        import cupy as cp
        with cp.cuda.Device(gpu):
            # warm-up
            _ = cp.linalg.svd(cp.random.randn(8,8))
            
            g = cp.asarray(a)
            if g.shape[0] > g.shape[1]:
                g = g.T
                
            s = cp.linalg.svd(g, compute_uv=False, full_matrices=False)
            result = int(cp.count_nonzero(s > tol).get())
            print(f"Rank computed: {result}")
            return result
            
    except Exception as e:
        print(f"ERROR in rank_gpu: {e}")
        raise

# ───── FIXED CSV helpers: read existing rows for resume ─────
def csv_done_values(fname, key_index):
    """Return SET of already completed values to prevent duplicates"""
    done = set()
    if fname.exists():
        try:
            with open(fname, 'r', newline='') as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header
                for row in reader:
                    if len(row) > key_index and row[key_index].strip():
                        # Only count COMPLETE rows (with timing data)
                        if len(row) >= 8 and row[7].strip():  # Has time_sec
                            done.add(int(row[key_index]))
        except Exception as e:
            print(f"Warning: Could not read {fname}: {e}")
    return done

def csv_done_pairs(fname, i, j):
    """Return SET of already completed (k,r) pairs"""
    done = set()
    if fname.exists():
        try:
            with open(fname, 'r', newline='') as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header
                for row in reader:
                    if len(row) > max(i,j) and row[i].strip() and row[j].strip():
                        # Only count COMPLETE rows (with timing data)
                        if len(row) >= 7 and row[6].strip():  # Has time_sec
                            done.add((int(row[i]), int(row[j])))
        except Exception as e:
            print(f"Warning: Could not read {fname}: {e}")
    return done

# ───── Suite runners with adaptive parameters and timing ─────
def run_scaling(n, gpu):  # A
    print(f"\n=== SCALING SUITE n={n} ===")
    start_time = time.time()
    samples, k = get_adaptive_params(n)
    r = 3
    projections = comb(n, r)
    print(f"Parameters: k={k}, r={r}, samples={samples}, projections={projections}")
    
    polys = rand_poly_list(n, seed=n)
    matrix = spdp(polys, n, k, r, samples, n)
    rank = rank_gpu(matrix, gpu)
    
    elapsed = time.time() - start_time
    result = ["scaling", n, rank, ceil(n**0.5), k, samples, projections, round(elapsed, 2)]
    print(f"Completed in {elapsed:.2f}s")
    return result

def run_spdp_vs_spd(n, gpu):  # B
    print(f"\n=== SPDP vs SPD SUITE n={n} ===")
    start_time = time.time()
    samples, k = get_adaptive_params(n)
    r = 3
    
    polys = rand_poly_list(n, seed=10*n)
    spdp_r = rank_gpu(spdp(polys, n, k, r, samples, 10*n), gpu)
    spd_r = rank_gpu(spdp(polys, n, k, 1, samples, 20*n), gpu)
    
    elapsed = time.time() - start_time
    result = ["spdp_vs_spd", n, spdp_r, spd_r, ceil(n**0.5), k, samples, round(elapsed, 2)]
    print(f"Completed in {elapsed:.2f}s")
    return result

def run_easy_vs_hard(n, gpu):  # C
    print(f"\n=== EASY vs HARD SUITE n={n} ===")
    start_time = time.time()
    samples, k = get_adaptive_params(n)
    r = 3
    
    add_rank = rank_gpu(spdp([lambda x: np.sum(x)], n, k, r, samples, 3*n), gpu)
    perm_rank = rank_gpu(spdp(
        [lambda x: sum(np.prod(x[np.random.permutation(n)]) for _ in range(n))],
        n, k, r, samples, 5*n), gpu)
    
    elapsed = time.time() - start_time
    result = ["easy_vs_hard", n, add_rank, perm_rank, ceil(n**0.5), k, samples, round(elapsed, 2)]
    print(f"Completed in {elapsed:.2f}s")
    return result

def run_core_rules(n, gpu):    # D
    print(f"\n=== CORE RULES SUITE n={n} ===")
    start_time = time.time()
    samples, k = get_adaptive_params(n)
    r = 3
    
    r1 = rank_gpu(spdp(rand_poly_list(n, seed=n), n, k, r, samples, n), gpu)
    r2 = rank_gpu(spdp(rand_poly_list(n, seed=n+1), n, k, r, samples, n+1), gpu)
    
    elapsed = time.time() - start_time
    result = ["core_rules", n, r1, r2, ceil(n**0.5), k, samples, round(elapsed, 2)]
    print(f"Completed in {elapsed:.2f}s")
    return result

def run_heatmap_row(n_fixed, k, r, gpu):
    print(f"\n=== HEATMAP k={k}, r={r} ===")
    start_time = time.time()
    samples, _ = get_adaptive_params(n_fixed)  # Use adaptive sampling
    
    rank = rank_gpu(spdp(rand_poly_list(n_fixed, seed=100*k+r),
                         n_fixed, k, r, samples, 100*k+r), gpu)
    
    elapsed = time.time() - start_time
    result = ["heatmap", n_fixed, k, r, rank, samples, round(elapsed, 2)]
    print(f"Completed in {elapsed:.2f}s")
    return result

# ───── Main loop ─────
print("Setting up CSV headers...")
headers = {
    "scaling":      ["suite","n","rank","sqrt_n","k","samples","projections","time_sec"],
    "spdp_vs_spd":  ["suite","n","spdp_rank","spd_rank","sqrt_n","k","samples","time_sec"],
    "easy_vs_hard": ["suite","n","add_rank","perm_rank","sqrt_n","k","samples","time_sec"],
    "core_rules":   ["suite","n","rank_first","rank_last","sqrt_n","k","samples","time_sec"],
    "heatmap":      ["suite","n_fixed","k","r","rank","samples","time_sec"],
}

for tag, h in headers.items():
    p = OUTDIR / f"{tag}.csv"
    if not p.exists():
        with open(p, "w", newline="") as f: 
            csv.writer(f).writerow(h)

gpu_cycle = itertools.cycle(GPU_IDS)
start = time.time()

print(f"\nStarting computation from n={N_MIN} to n={N_MAX}")

# --- suites A–D ---
SUITE_FUNCS = {
    "scaling":      run_scaling,
    "spdp_vs_spd":  run_spdp_vs_spd,
    "easy_vs_hard": run_easy_vs_hard,
    "core_rules":   run_core_rules,
}

for tag, runner in SUITE_FUNCS.items():
    print(f"\n{'='*50}")
    print(f"PROCESSING SUITE: {tag.upper()}")
    print(f"{'='*50}")
    
    done = csv_done_values(OUTDIR / f"{tag}.csv", key_index=1)  # n is 2nd col
    print(f"Already completed n values: {sorted(done)}")
    
    for n in range(N_MIN, N_MAX+1, STEP):
        if n in done:
            print(f"✓ SKIP {tag} n={n} (already completed)")
            continue
            
        gpu = next(gpu_cycle)
        print(f"\n► RUNNING {tag} n={n} on GPU {gpu}")
        
        try:
            row = runner(n, gpu)
            
            # ATOMIC write to prevent corruption
            with open(OUTDIR / f"{tag}.csv", "a", newline="") as f:
                csv.writer(f).writerow(row)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk
            
            print(f"✅ SAVED: {row}")
            
        except Exception as e:
            print(f"❌ ERROR during {tag} n={n}: {e}")
            raise
            
        gc.collect()

# --- suite F heatmap (n_fixed=24) ---
print(f"\n{'='*50}")
print(f"PROCESSING SUITE: HEATMAP")
print(f"{'='*50}")

n_fixed = 24
heat_file = OUTDIR / "heatmap.csv"
done_pairs = csv_done_pairs(heat_file, 2, 3)  # (k,r) columns
print(f"Already completed (k,r) pairs: {sorted(done_pairs)}")

for k, r in product(range(2, 6), range(1, 5)):
    if (k, r) in done_pairs:
        print(f"✓ SKIP heatmap k={k},r={r} (already completed)")
        continue
        
    gpu = next(gpu_cycle)
    print(f"\n► RUNNING heatmap k={k} r={r} on GPU {gpu}")
    
    try:
        row = run_heatmap_row(n_fixed, k, r, gpu)
        
        # ATOMIC write to prevent corruption
        with open(heat_file, "a", newline="") as f:
            csv.writer(f).writerow(row)
            f.flush()
            os.fsync(f.fileno())  # Force write to disk
        
        print(f"✅ SAVED: {row}")
        
    except Exception as e:
        print(f"❌ ERROR during heatmap k={k},r={r}: {e}")
        raise
        
    gc.collect()

print(f"\n🎉 ALL SUITES COMPLETED in {time.time()-start:.1f}s")
print(f"📊 Results saved in {OUTDIR}/")
print(f"🔬 Ready for P≠NP analysis!")