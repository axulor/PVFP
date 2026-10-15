# -*- coding: utf-8 -*-
"""structure_essence.py -- section 3.3, mechanism item 1 (figure 8): what structural feature of the two classes of
third parties makes observing them favour Uniform (non-common) or Shapley (common)?  Pure structure, no simulation.
Numbers cited in the design doc: in games with a non-common neighbour the candidate is the larger player in 71 % of
the comparisons (range 34-93 % over the 40 networks) and the smaller one in 24 %; with a common neighbour 46 % each
(columns non_lt, non_gt, com_lt, com_gt).

For every ordered imitator-candidate pair (i, j), weighted as in death-birth updating (1/N * 1/k_i), and every third
party l of j (common: l in N(i); non-common: otherwise), record
  g   = log(k_l / k_j)                          degree gap (l larger: g > 0)
  w   = (k_j - k_l) / (2 (k_j + k_l))           Uniform - Shapley share of the pool that goes to j (> 0: to j)
  c   = 1 / k_l                                 concentration of l's input into this one pool
  h   = w * c                                   Uniform - Shapley harvest of l's input by j, if l cooperates
Prints, per network and class: mean g, share of third parties with k_l < k_j, = , >, the share of leaves (k_l = 1)
and low-degree partners (k_l <= 2), and the mean harvest h (the term that dominates the cooperators' lead when
cooperators cluster).  Output: results/final/mechanism/structure_essence.csv
Run:  python structure_essence.py"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import networks40 as N40

OUT = os.path.join(N40.ROOT, "results", "final", "mechanism"); os.makedirs(OUT, exist_ok=True)


def essence(G):
    k = dict(G.degree()); N = G.number_of_nodes(); nb = {i: set(G[i]) for i in G}
    acc = {c: dict(W=0.0, g=0.0, lt=0.0, eq=0.0, gt=0.0, leaf=0.0, low=0.0, h=0.0, hpos=0.0, hneg=0.0) for c in ("com", "non")}
    for i in G:
        wi = 1.0 / (N * k[i])
        for j in nb[i]:
            com = nb[i] & nb[j]; non = nb[j] - com - {i}
            for cls, S_ in (("com", com), ("non", non)):
                if not S_:
                    continue
                a = acc[cls]; ww = wi / len(S_) * len(S_)        # every third party counts once per pair
                for l in S_:
                    w = (k[j] - k[l]) / (2.0 * (k[j] + k[l])); h = w / k[l]
                    a["W"] += wi; a["g"] += wi * np.log(k[l] / k[j])
                    a["lt"] += wi * (k[l] < k[j]); a["eq"] += wi * (k[l] == k[j]); a["gt"] += wi * (k[l] > k[j])
                    a["leaf"] += wi * (k[l] == 1); a["low"] += wi * (k[l] <= 2)
                    a["h"] += wi * h; a["hpos"] += wi * max(h, 0); a["hneg"] += wi * min(h, 0)
    out = {}
    for cls, a in acc.items():
        W = a["W"] or 1
        for x in ("g", "lt", "eq", "gt", "leaf", "low"):
            out[f"{cls}_{x}"] = a[x] / W
        out[f"{cls}_h_per_pair"] = a["h"]                         # summed over third parties, per pair
        out[f"{cls}_hpos_per_pair"] = a["hpos"]; out[f"{cls}_hneg_per_pair"] = a["hneg"]
        out[f"{cls}_n_per_pair"] = a["W"]
    return out


if __name__ == "__main__":
    rows = []
    for nm in N40.SYN + N40.REAL:
        G = N40.build(nm)
        rows.append(dict(net=nm, **essence(G)))
    d = pd.DataFrame(rows).set_index("net")
    d.to_csv(os.path.join(OUT, "structure_essence.csv"))
    cols = ["com_g", "non_g", "com_lt", "com_gt", "non_lt", "non_gt", "non_leaf", "non_low", "com_low", "com_h_per_pair", "non_h_per_pair"]
    print(d[cols].round(3).to_string())
    print("\nmeans over 40:"); print(d[cols].mean().round(3).to_string())
    print(f"\ncandidate is the larger player (non-common): mean {d.non_lt.mean():.1%}, range {d.non_lt.min():.1%}-{d.non_lt.max():.1%}; "
          f"smaller: {d.non_gt.mean():.1%};  common: {d.com_lt.mean():.1%} / {d.com_gt.mean():.1%}")
