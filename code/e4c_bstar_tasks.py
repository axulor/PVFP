# -*- coding: utf-8 -*-
"""e4c_bstar_tasks.py -- task file for the critical benefits of the composition series (figure 12) at (1, 1), both
rules, precision 0.2 %, run locally with the serial build of hpc/src (same program as on the HPC):
    <bstar_serial> results/final/e4c_tasks.txt results/final/nets_comp results/final/e4c_bstar
Initial guesses: E1 values at eps = 1 of the main-set networks with the closest CV."""
import os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIN = os.path.join(ROOT, "results", "final")
if __name__ == "__main__":
    comp = pd.read_csv(os.path.join(FIN, "comp_series.csv"))
    st = pd.read_csv(os.path.join(FIN, "networks40_stats.csv")).set_index("net")
    e1 = pd.read_csv(os.path.join(FIN, "e1_hpc.csv"))
    e1 = e1[np.isclose(e1.eps1, 1.0) & e1.net.isin([n for n in st.index if n.startswith(("ER", "BA", "CL", "HK", "RHG"))])]
    rows = ["# id net rule eps1 eps2 theta delta seed target guess cost\n"]
    i = 0
    for r in comp.itertuples():
        near = st.loc[e1.net.unique()].CV.sub(r.CV).abs().sort_values().index[:3]
        for rule in ("U", "Sh"):
            g = float(e1[e1.net.isin(near) & (e1.rule == rule)].bstar.mean())
            rows.append(f"{i} {r.net} {rule} 1 1 1 1 1 0.002 {g!r} {1000.0 * r.CV:.1f}\n"); i += 1
    with open(os.path.join(FIN, "e4c_tasks.txt"), "w", newline="\n") as f:
        f.writelines(rows)
    print(f"{i} tasks -> results/final/e4c_tasks.txt")
