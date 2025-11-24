spdp-observer-p-vs-np experimental python tests

The Python scripts are provided largely in template form.
Please configure the parameters according to your specific use case before execution.

This repository contains the code and data behind the SPDP (Shifted Partial Derivative + Projection) framework, the holographic compiler configuration (radius = 1, diagonal basis, Π⁺ = A), and the Evolutionary Algorithm (EA) runs that together support the Global God-Move separation picture. 

SPDP core: build SPDP matrices and compute (pruned) ranks
Workload families: easy vs hard instances (permanent, ROABP, monotone DNF, majority, mod-3, sparse, hybrid, etc.)
Empirical suites: collapse frequency, seeded pruning, phase transition, verifier checks
EA runs: template dominance and best-per-workload summaries
Plots: figures used in the manuscript
📦 Requirements
Python 3.10+
CPU works; CUDA GPU is optional but speeds up large runs
Install deps:
pip install -r requirements.txt
Typical packages: numpy, scipy, pandas, networkx, sympy, numba (CPU JIT), matplotlib, tqdm, optional GPU: cupy-cuda11x (match your CUDA).

🚀 Quickstart

Environment
bash Copy code python -m venv .venv source .venv/bin/activate # Windows: .venv\Scripts\activate pip install -r requirements.txt 2) Small sanity run (CPU)

bash Copy code python experiments/test_selective_collapse.py Produces a small CSV in data/ and a figure in plots/figures/ (if enabled).

Batch collapse test (n=64)
bash Copy code python experiments/batch_collapse_test.py --n 64 --trials 50 --save data/collapse_n64.csv 4) Phase transition plot

bash Copy code python experiments/test_spdp_phase_transition.py --n 64 --save data/phase_transition_n64.csv python plots/generate_spdp_rank_plot.py --in data/phase_transition_n64.csv --out plots/figures/phase_transition_n64.png 5) EA digest (uses EA CSVs in ea/)

bash Copy code python ea/EA.txt Writes ea/ea_summary_digest.csv and ea/ea_findings.txt

🔁 Script → Output mapping (repro guide) Script Primary outputs Notes experiments/test_selective_collapse.py data/selective_collapse.csv, plots/figures/selective_collapse.png Easy vs hard (collapse vs resist) experiments/test_seed_collapse_with_pruning_refined.py data/seeded_pruning_stats.csv Collapse frequency across seeds experiments/test_spdp_phase_transition.py data/phase_transition_n{N}.csv, optional figure Rank vs projection dimension 𝑟 r experiments/test_rank_verifier.py data/verifier_passfail.csv Diagonal rank-certificate checks experiments/test_verifier_runtime_scaling.py data/verifier_timing.csv Empirical scaling experiments/depth4_spdp_demo.py data/depth4_demo.csv, table figure ΣΠΣΠ demo experiments/batch_collapse_test.py data/collapse_n{N}.csv Collapse ratios by family experiments/spdp_prune_and_rank.py data/prune_and_rank.csv Two-phase pruning then rank plots/generate_spdp_rank_plot.py plots/figures/.png Reads one CSV → figure plots/plot_spdp_collapse_extended.py plots/figures/.png Merges multiple CSVs → figure ea/EA.txt ea/ea_summary_digest.csv, ea/ea_findings.txt Dominance, CEW↔rank summaries

⚙️ GPU (optional) To enable GPU acceleration (if you have CUDA):

bash Copy code pip install cupy-cuda11x # choose correct wheel for your CUDA Most experiments auto-detect GPU; otherwise they fall back to CPU.

🔍 Data availability All code and datasets are in this repository. Raw/intermediate outputs (CSV) are written to data/ and ea/ by the scripts above; figures go to plots/figures/. See the paper’s Data Availability section for the exact filenames referenced.

📣 Citation If this code or data is useful in your research, please cite:

@article{Edwards2025SPDPObserver, title = {Toward P̸ = NP: An Observer-Theoretic Separation via SPDP Rank and Equivalence to ZFC Foundations within N-Frame}, author = {Edwards, Darren J.}, year = {2025}, note = {Code and data: https://github.com/DarrenEdwards111/spdp-observer-p-vs-np} } (Replace with arXiv/journal citation when available.)

📄 License Released under the MIT License (or your chosen license). See LICENSE.
