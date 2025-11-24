
# generate_spdp_rank_plot.py
import matplotlib.pyplot as plt

# Data for plot
data = {
    "Circuit Type": [
        "RandDeg3-1k", "RandDeg3-2k", "RandDeg3-6k", "RandDeg3-40k",
        "RandDeg3-100k", "RandDeg3-1M", "CRVW-4k", "Goldreich-4k", "Diagonal-d4-2k"
    ],
    "n": [1024, 2048, 6096, 40096, 100096, 1000096, 4095, 4096, 2048],
    "Live Vars": [27, 44, 95, 196, 293, 1002, 69, 83, 49],
    "SPDP Rank": [0, 0, 0, 0, 0, 0, 0, 0, 48],
    "Threshold": [32, 46, 79, 201, 317, 1001, 64, 64, 46]
}

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(data["n"], data["Threshold"], label="Collapse Threshold (⌈√n⌉)", linestyle='--', marker='o')
plt.plot(data["n"], data["Live Vars"], label="Live Variables After Pruning", linestyle='-', marker='s')
plt.plot(data["n"], data["SPDP Rank"], label="SPDP Rank", linestyle='-', marker='^')

# Annotate circuit types
for i, label in enumerate(data["Circuit Type"]):
    plt.annotate(label, (data["n"][i], data["Live Vars"][i] + 5), fontsize=8, rotation=15)

plt.xlabel("Circuit Size n (log scale)")
plt.ylabel("Value")
plt.title("SPDP Rank Collapse Across Circuit Types")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xscale("log")
plt.yscale("linear")
plt.savefig("spdp_rank_collapse_final_plot.png", dpi=300)
plt.show()
