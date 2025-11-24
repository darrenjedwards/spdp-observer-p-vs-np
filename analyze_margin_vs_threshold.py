
# analyze_margin_vs_threshold.py

def analyze_margins(results):
    print("\n=== SPDP Collapse Margin Analysis ===")
    for res in results:
        name = res["Circuit"]
        n = res["n"]
        rank = res["Rank"]
        threshold = res["Threshold"]
        margin = threshold - rank
        print(f"{name:<15} n={n:<7} Rank={rank:<3} Threshold={threshold:<3} Margin={margin:<3} {'PASS' if res['Pass'] else 'FAIL'}")

if __name__ == "__main__":
    from spdp_batch_runner import run_all_tests
    results = run_all_tests(n=1024)
    analyze_margins(results)
