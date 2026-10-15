# -*- coding: utf-8 -*-
"""small_bstar.py -- figures 11-12: precise b* (0.3%) at (0,1) and (1,1) for the small networks of edge_map.py and
their rewired versions, giving xi = 1 - Lambda(1,1)/Lambda(0,1) for every column.
Output: results/final/small_nets/small_bstar.csv.   Run:  python small_bstar.py hk6 s7"""
import os, sys, time
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kernel import Complex
from exp_common import triangles
from bstar import Estimator
import edge_map as EM

if __name__ == "__main__":
    out = os.path.join(EM.EX, "small_bstar.csv")
    done = set()
    if os.path.exists(out):
        d0 = pd.read_csv(out).dropna(subset=["bstar"]); done = set(zip(d0.name, d0.rule, d0.eps1, d0.eps2))
    for tag in sys.argv[1:] or ["hk6", "s7"]:
        fam, th = EM.family(**EM.CONFIGS[tag])
        for nm, G in fam.items():
            cx = Complex(G, triangles(G))
            for e1, e2 in ((0.0, 1.0), (1.0, 1.0)):
                for rule in ("U", "Sh"):
                    if (f"{tag}_{nm}", rule, e1, e2) in done:
                        continue
                    t0 = time.time()
                    r = Estimator(cx, rule, e1, e2, 1.0, seed=7).estimate(target=0.003)
                    if not r["bstar"] == r["bstar"]:                         # unconverged: once more with another seed
                        r = Estimator(cx, rule, e1, e2, 1.0, seed=8).estimate(target=0.003)
                    pd.DataFrame([dict(name=f"{tag}_{nm}", rule=rule, eps1=e1, eps2=e2, bstar=r["bstar"], se=r.get("se"))]).to_csv(
                        out, mode="a", header=not os.path.exists(out), index=False)
                    print(f"{tag}_{nm} {rule} ({e1:.0f},{e2:.0f}) b* {r['bstar']:.4f} se {r.get('se', 0):.4f} ({time.time() - t0:.0f}s)", flush=True)
