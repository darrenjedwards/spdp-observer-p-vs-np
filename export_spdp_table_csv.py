
# export_spdp_table_csv.py
import csv

def export_results_to_csv(results, filename="spdp_results.csv"):
    if not results:
        print("No results to export.")
        return
    keys = results[0].keys()
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"✅ Results exported to {filename}")

if __name__ == "__main__":
    from spdp_batch_runner import run_all_tests
    results = run_all_tests(n=1024)
    export_results_to_csv(results, "spdp_results_1024.csv")
