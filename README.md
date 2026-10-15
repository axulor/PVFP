# PVFP

Code and data for *Payoff visibility and a friendship paradox in imitation shape the conflict between cooperation and equality* (Y. Wang, A. K. Srivastav, J. Tanimoto).

- `code/`: Python model, analysis and figure scripts (run inside `code/`; `figN_*.py` draws Fig. N into `figs/`).
- `hpc/`: C++/MPI program for the critical-benefit estimates, with task lists, job scripts, raw outputs and logs.
- `results/final/`: data behind all figures.
- `data/`: the 40 networks (processed inputs) and the network table with sources, construction and statistics (`data/README.md`).

Requirements: Python 3 with numpy, scipy, pandas, networkx, numba and matplotlib; a C++17 compiler with MPI and OpenMP for `hpc/`.

License: code under the MIT License (`LICENSE`); data in `results/` under CC BY 4.0. Third-party network data in `data/` remain under the terms of their original sources.
