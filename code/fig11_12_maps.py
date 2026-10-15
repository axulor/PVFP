# -*- coding: utf-8 -*-
"""Figures 11-12 draft v2 (2026-10-09): where in a small network the conflict is produced and where it is weakened.
One small network per figure (2026-10-09): figure 11 a Holme-Kim network (N = 32), figure 12 an S1 network (N = 30),
each with rewired versions of the same degree sequence (code/edge_map.py, families "hk6" and "s7").
  figure 11 (how many common neighbors): all triangles removed / half of the triangles removed / original
  figure 12 (who the common neighbors are): homogeneous / original / mixed triangles (same triangle count and eta)
Layout: ForceAtlas2 (as in Gephi; strong gravity) for every version, rotated / reflected onto the original version;
edges drawn as curves.  Node size: degree.
Edges: game-by-game attribution of the cooperators' lead (two kinds of third party observed, both rules' processes
averaged); line width proportional to the value.
  row "non-common neighbors": Uniform - Shapley (red: favors Uniform)
  row "common neighbors": output shared by two cooperators (green: favors neither rule; drawn 3 times wider) and
      Uniform - Shapley of cooperator-defector games (blue: favors Shapley; drawn 6 times wider)
Header of each column (per network): eta, dbar and xi = 1 - Lambda(1,1)/Lambda(0,1), the share of the conflict
removed by also observing common neighbors.  Text: 8.5 pt, column titles and network name 9.5 pt bold (2026-10-14).
Output: figs/fig11.*, figs/fig12.*"""
import os, sys
import numpy as np, pandas as pd, networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import style as S

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edge_map as EM
import networks_s1 as NS

FS = 7                                  # ticks/base size (no ticks are drawn)
# 2026-10-14: text sizes for the display width 164.6 mm (PDF ~157 mm wide, scale ~1.05): column headers, net-effect
# values and row labels 8.5 pt (-> ~8.9 pt), column titles and the network name 9.5 pt bold (-> ~10 pt)
FS_TXT, FS_HEAD = 8.5, 9.5
S.setup(FS)
EX = os.path.join(S.RES, "final", "small_nets")
OUT = S.FIG
NETS = [("s7", "S$^1$ network"), ("hk6", "Holme–Kim network")]
C_ONE, C_CC, C_CD, C_BASE, C_NODE = "#D62728", "#2CA02C", "#002FA7", "#ECEEF1", "#7A828C"   # vivid red / green / Klein blue (2026-10-09)
W, K_CC, K_CD = 30.0, 3.0, 6.0                         # line width (pt) per unit of lead; row 2: C-C x3, C-D x6
bs = pd.read_csv(os.path.join(EX, "small_bstar.csv")).dropna(subset=["bstar"]).groupby(["name", "eps1", "rule"]).bstar.last()


def lam(name, e1):
    return bs[(name, e1, "Sh")] / bs[(name, e1, "U")] - 1


def layout(G, seed=1):
    p = nx.forceatlas2_layout(G, max_iter=3000, seed=seed, scaling_ratio=2.0, strong_gravity=True, gravity=0.5)
    return {n: np.array(v, float) for n, v in p.items()}


def align(pos, ref):
    """rotate / reflect / scale pos onto ref (orthogonal Procrustes on the common nodes)."""
    nodes = sorted(ref)
    A = np.array([pos[n] for n in nodes]); B = np.array([ref[n] for n in nodes])
    A0, B0 = A - A.mean(0), B - B.mean(0)
    U, s_, Vt = np.linalg.svd(A0.T @ B0)
    R = U @ Vt
    sc = np.sqrt((B0 ** 2).sum() / (A0 ** 2).sum())
    return {n: (pos[n] - A.mean(0)) @ R * sc + B.mean(0) for n in pos}


def curve(p, q, c=0.16, n=24):
    v = q - p; nrm = np.array([v[1], -v[0]])
    m = 0.5 * (p + q) + c * nrm
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p + 2 * (1 - t) * t * m + t ** 2 * q


def draw(ax, G, pos, d, row):
    k = dict(G.degree())
    segs = {(int(r.u), int(r.v)): curve(pos[int(r.u)], pos[int(r.v)]) for r in d.itertuples()}
    ax.add_collection(LineCollection(list(segs.values()), colors=C_BASE, linewidths=0.4, zorder=1))
    if row == 0:
        x = d[d.one > 0].sort_values("one")
        ax.add_collection(LineCollection([segs[(int(r.u), int(r.v))] for r in x.itertuples()], colors=C_ONE,
                                         linewidths=W * x.one.values, capstyle="round", zorder=2))
    else:
        x = d[d.cc > 0].sort_values("cc")
        ax.add_collection(LineCollection([segs[(int(r.u), int(r.v))] for r in x.itertuples()], colors=C_CC,
                                         linewidths=K_CC * W * x.cc.values, capstyle="round", zorder=2))
        x = d[d.two < 0].sort_values("two", ascending=False)
        ax.add_collection(LineCollection([segs[(int(r.u), int(r.v))] for r in x.itertuples()], colors=C_CD,
                                         linewidths=K_CD * W * (-x.two.values), capstyle="round", zorder=3))
    ax.scatter([pos[n][0] for n in G], [pos[n][1] for n in G], s=[3 + 2.2 * k[n] for n in G], color=C_NODE,
               edgecolors="white", linewidths=0.4, zorder=4)
    val, col = (d.one.sum(), C_ONE) if row == 0 else (d.two.sum(), C_CD)
    ax.text(0.5, -0.03, f"net effect ${val:+.2f}$", transform=ax.transAxes, ha="center", va="top", color=col, fontsize=FS_TXT)
    P = np.array(list(pos.values())); lo, hi = P.min(0), P.max(0); pad = 0.06 * (hi - lo).max()
    ax.set_xlim(lo[0] - pad, hi[0] + pad); ax.set_ylim(lo[1] - pad, hi[1] + pad); ax.set_aspect("equal", adjustable="datalim"); ax.axis("off")   # same box in every panel


def figure(cols, name, nets=NETS, c_cc=None):
    global C_CC
    if c_cc:
        C_CC = c_cc
    nb = len(nets)
    fig = plt.figure(figsize=(180 * S.MM, (128 if nb == 1 else 245) * S.MM))
    hr = [1, 1] if nb == 1 else [1, 1, 0.16, 1, 1]
    gs = fig.add_gridspec(len(hr), 3, height_ratios=hr, wspace=0.04, hspace=0.36)
    rows = (0, 1) if nb == 1 else (0, 1, 3, 4)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in rows])
    for c, (key, title) in enumerate(cols):
        axes[0, c].text(0.5, 1.30, title, transform=axes[0, c].transAxes, ha="center", va="bottom", fontsize=FS_HEAD,
                        fontweight="bold", color=S.INK)
    for b, (tag, label) in enumerate(nets):
        fam, _ = EM.family(**EM.CONFIGS[tag])
        ref = layout(fam["gen"])
        for c, (key, title) in enumerate(cols):
            G = fam[key]
            pos = ref if key == "gen" else align(layout(G), ref)
            d = pd.read_csv(os.path.join(EX, f"edge_map_{tag}_{key}.csv"))
            for r in range(2):
                draw(axes[2 * b + r, c], G, pos, d, r)
            T = NS.n_triangles(G)
            L01, L11 = lam(f"{tag}_{key}", 0.0), lam(f"{tag}_{key}", 1.0)
            xi = 1 - L11 / L01
            head = (f"$\\eta={NS.eta(G):.2f}$,  $\\bar d={NS.dbar(G):.1f}$" if T else "$\\eta=0$ (no triangles)") + \
                   f"\nconflict reduction $\\xi={max(xi, 0):.2f}$"
            axes[2 * b, c].text(0.5, 1.02, head, transform=axes[2 * b, c].transAxes, ha="center", va="bottom", fontsize=FS_TXT,
                                color=S.INK, linespacing=1.3)
        for r, lab in enumerate(("non-common neighbors", "common neighbors")):
            axes[2 * b + r, 0].text(-0.05, 0.5, lab, transform=axes[2 * b + r, 0].transAxes, rotation=90, ha="right", va="center",
                                    fontsize=FS_TXT, color=S.INK)
        axes[2 * b, 0].text(-0.20, -0.15, label, transform=axes[2 * b, 0].transAxes, rotation=90, ha="right", va="center",
                            fontsize=FS_HEAD, fontweight="bold", color=S.INK)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=1000 if ext == "png" else None, bbox_inches="tight")
    print("saved", name)


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    cc = sys.argv[2] if len(sys.argv) > 2 else None
    figure([("low", "all triangles removed"), ("mid", "half triangles removed"), ("gen", "original")], "fig11" + tag,
           nets=[("hk6", "Holme–Kim network")], c_cc=cc)
    figure([("homog", "homogeneous triangles"), ("gen", "original"), ("mixed", "mixed triangles")], "fig12" + tag,
           nets=[("s7", "S$^1$ network")], c_cc=cc)
