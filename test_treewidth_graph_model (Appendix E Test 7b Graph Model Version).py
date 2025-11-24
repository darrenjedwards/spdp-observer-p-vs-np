import networkx as nx
import numpy as np
import csv

def generate_random_circuit_graph(n_vars, n_gates):
    """Creates a random undirected graph where each gate connects two random variables."""
    G = nx.Graph()
    G.add_nodes_from(range(1, n_vars + 1))
    for _ in range(n_gates):
        u, v = np.random.choice(range(1, n_vars + 1), 2, replace=False)
        G.add_edge(u, v)
    return G

def apply_pruning(G, p):
    """Simulates variable survival via independent pruning with probability p."""
    surviving_vars = [v for v in G.nodes if np.random.rand() < p]
    return G.subgraph(surviving_vars).copy()

def estimate_treewidth(G):
    """Estimate treewidth using min-fill heuristic."""
    if G.number_of_nodes() == 0:
        return -1
    try:
        tw, _ = nx.approximation.treewidth_min_fill_in(G)
        return tw
    except:
        return -1

def run_graph_pruning_test(output_csv):
    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Gates", "Vars", "p", "LiveVars", "TW_Before", "TW_After"])
        test_cases = [
            (50, 75, 0.125),
            (100, 150, 0.007812)
        ]
        for g, v, p in test_cases:
            for _ in range(5):  # 5 trials each
                G = generate_random_circuit_graph(v, g)
                tw_before = estimate_treewidth(G)
                pruned_G = apply_pruning(G, p)
                tw_after = estimate_treewidth(pruned_G)
                writer.writerow([g, v, p, pruned_G.number_of_nodes(), tw_before, tw_after])

run_graph_pruning_test("/mnt/data/test7b_treewidth_graph_model.csv")
