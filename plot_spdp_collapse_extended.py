#!/usr/bin/env python3
"""
plot_spdp_collapse_extended.py

Generates SPDP collapse plots from CSV outputs of batch_collapse_test.py and batch_large_n_test.py.
Used in paper to visualize rank collapse behavior across families and scales.
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load and combine CSVs
df_small = pd.read_csv("collapse_summary.csv")
df_large = pd.read_csv("large_n_summary.csv")
df = pd.concat([df_small, df_large])

# Compute collapse ratio
df["ratio"] = df["collapsed"] / df["total"]

# Plot: Collapse ratio vs n for each family
families = df["family"].unique()
for family in families:
    sub = df[df["family"] == family].sort_values("n")
    plt.plot(sub["n"], sub["ratio"], marker='o', label=family)

plt.axhline(1.0, color='gray', linestyle='--', linewidth=1)
plt.title("SPDP Collapse Ratio by Circuit Family")
plt.xlabel("Input size n")
plt.ylabel("Collapse ratio (rank ≤ √n)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("spdp_collapse_summary.png", dpi=300)
plt.show()
