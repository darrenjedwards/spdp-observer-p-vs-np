
# spdp_batch_runner.py
from spdp_prune_and_rank import run_spdp_test
from build_circuits import (
    build_majority_poly, build_addressing_poly, build_parity_poly,
    build_randdeg3_poly, build_crvw_extractor, build_goldreich_prf,
    build_diagonal_poly
)

def run_all_tests(n=1024, k=3, samples=256):
    results = []

    results.append(run_spdp_test("Majority", build_majority_poly, n, k, samples))
    results.append(run_spdp_test("Addressing", build_addressing_poly, n, k, samples))
    results.append(run_spdp_test("Parity", build_parity_poly, n, k, samples))
    results.append(run_spdp_test("RandDeg3", build_randdeg3_poly, n, k, samples))

    if n % 3 == 0:
        results.append(run_spdp_test("CRVW", build_crvw_extractor, n, k, samples))
    else:
        print("Skipping CRVW: n must be divisible by 3")

    results.append(run_spdp_test("Goldreich", build_goldreich_prf, n, k, samples))
    results.append(run_spdp_test("Diagonal-d3", lambda n: build_diagonal_poly(n, d=3), n, k, samples))
    results.append(run_spdp_test("Diagonal-d4", lambda n: build_diagonal_poly(n, d=4), n, k, samples))

    return results

if __name__ == "__main__":
    run_all_tests(n=1024)
