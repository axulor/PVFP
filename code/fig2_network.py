# -*- coding: utf-8 -*-
"""fig2_network.py -- Figure 2 of the paper: who loses at the cooperation threshold.

Zachary karate club under uniform allocation, drawn as a 2 x 2 factorial design of the two visibility weights:
columns eps1 in {0, 1} (common neighbors), rows eps2 in {0, 1} (non-common neighbors).
Node area = payoff u_i under full cooperation at b = b*_Un; node colour marks u_i >= c or u_i < c (c = 1).
Panel header: b*_Un, b*_Sh, the rule gap Lambda and nu (share of individuals with u_i < c).
Shapley allocation is not drawn: with linear production every individual receives exactly b, so nu = 0 and all
nodes would look the same in every panel; its threshold enters through b*_Sh and Lambda.

Simulation (theta = 1, delta = 1, c = 1, death-birth updating, Omega_ij = all common neighbors):
critical benefits estimated as in section 3.B.2 with target precision 0.2 %; results/final/fig2_karate.csv.
Run:  python fig2_network.py          (computes missing b*, then draws figs/fig2.pdf/.png)
"""
import os
import numpy as np, pandas as pd, networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import style as S
from networks import load
from exp_common import cx_of
from bstar import Estimator

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(S.RES, "final"); os.makedirs(OUTD, exist_ok=True)
CSV = os.path.join(OUTD, "fig2_karate.csv")
SETTINGS = [(0, 0), (1, 0), (0, 1), (1, 1)]
RULES = ("U", "Sh")
C_GAIN, C_LOSS = "#3D5A80", "#EE6C4D"
C_EDGE = "#9FB1C3"
# 2026-10-14: the canvas is drawn at its printed width (0.85 textwidth = 139.9 mm), so font sizes are true points.
# K is the old print scale (139.9 / 155.8 mm): the bold column/row headers keep their printed size (10 pt x K = 9 pt).
# KG shrinks every graphic size of the old 180 mm canvas (panel height, node areas, line widths) together; it is
# 4 % below K so that the larger value lines of the two columns keep a clear gap within the 139.9 mm width.
K = 139.9 / 155.8
KG = 0.96 * K
SMAX = 260.0 * KG ** 2               # marker area (pt^2) of the largest payoff in the figure
WIDTH = 180 * K + 2.5                   # canvas width (mm) before bbox_inches="tight" trims the empty margins
FS_VAL, FS_LEG = 9.0, 9.0                # value line under each panel, legend
FS_HEAD = 10.0 * K * 141.0 / 139.9    # bold eps headers: printed at 8.98 pt as before (the PDF is 141.0 mm wide)


# ------------------------------------------------------------------------------------------------ simulation
def compute():
    done = pd.read_csv(CSV) if os.path.exists(CSV) else pd.DataFrame(columns=["rule", "eps1", "eps2"])
    cx = cx_of("KAR", "fill", 1.0)
    for e1, e2 in SETTINGS:
        for rule in RULES:
            if ((done.rule == rule) & (done.eps1 == e1) & (done.eps2 == e2)).any():
                continue
            out = Estimator(cx, rule, e1, e2, 1.0, delta=1.0, seed=20261007).estimate(guess=1.7, target=0.002)
            row = dict(rule=rule, eps1=e1, eps2=e2, bstar=out["bstar"], se=out.get("se", np.nan),
                       runs=out.get("runs_per_point", 0), flag=out.get("flag", ""))
            done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
            done.to_csv(CSV, index=False)
            print(rule, e1, e2, f"b* = {row['bstar']:.4f} +- {row['se']:.4f}", flush=True)
    return done


# ------------------------------------------------------------------------------------------------ figure
def payoff_factor(G):
    """u_i = b * f_i under uniform allocation and full cooperation (linear production, c = 1)."""
    k = dict(G.degree())
    return {i: 0.5 * (1 + sum(1 / k[j] for j in G[i])) for i in G}


def gini(x):
    x = np.asarray(x, float)
    return np.abs(x[:, None] - x[None, :]).sum() / (2 * len(x) * x.sum())


def layout(G, radius, pt_per_unit, iters=400, gap_pt=2.0 * KG):
    """Kamada-Kawai, leaders 0 and 33 on a horizontal line, then push apart overlapping discs."""
    pos = nx.kamada_kawai_layout(G)
    P = np.array([pos[i] for i in G]); P -= P.mean(0)
    d = P[33] - P[0]; t = -np.arctan2(d[1], d[0])
    P = P @ np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]).T
    P /= np.abs(P).max()
    r = np.array([radius[i] for i in G]) / pt_per_unit + gap_pt / pt_per_unit
    for _ in range(iters):
        moved = False
        for a in range(len(P)):
            for b in range(a + 1, len(P)):
                v = P[b] - P[a]; dist = np.hypot(*v); need = r[a] + r[b]
                if dist < need:
                    v = v / dist if dist > 1e-9 else np.array([1.0, 0.0])
                    shift = 0.5 * (need - dist) * v
                    P[a] -= shift; P[b] += shift; moved = True
        if not moved:
            break
    return {i: P[n] for n, i in enumerate(G)}


def draw(res, out="fig2", fs=FS_VAL):
    S.setup(fs)
    G = load("KAR")[0]
    b = {(r.rule, int(r.eps1), int(r.eps2)): r.bstar for r in res.itertuples()}
    f = payoff_factor(G)
    umax = max(b[("U", *st)] * max(f.values()) for st in SETTINGS)
    size = {st: {i: SMAX * b[("U", *st)] * f[i] / umax for i in G} for st in SETTINGS}       # area proportional to payoff
    W, H = WIDTH * S.MM, 128 * KG * S.MM
    pt_per_unit = (180 * KG * S.MM / 2 * 72) / 2.6                                                        # ~ panel width / data width
    rad = {i: np.sqrt(max(size[st][i] for st in SETTINGS)) / 2 for i in G}
    pos = layout(G, rad, pt_per_unit)
    P = np.array([pos[i] for i in G])
    fig, axes = plt.subplots(2, 2, figsize=(W, H), gridspec_kw=dict(wspace=0.06, hspace=0.16))
    bold = [pe.withStroke(linewidth=0.6 * K, foreground=S.INK)]                                 # bold look for mathtext
    for (e1, e2) in SETTINGS:
        ax = axes[e2, e1]
        bu, bs = b[("U", e1, e2)], b[("Sh", e1, e2)]; lam = (bs - bu) / bu
        for u, v in G.edges():
            ax.plot(*zip(pos[u], pos[v]), color=C_EDGE, lw=0.7 * KG, zorder=1, solid_capstyle="round")
        val = np.array([bu * f[i] for i in G]); loss = val < 1
        s = np.array([size[(e1, e2)][i] for i in G]); o = np.argsort(-s)
        ax.scatter(P[o, 0], P[o, 1], s=s[o], c=np.where(loss, C_LOSS, C_GAIN)[o], edgecolors="white", linewidths=0.8 * KG, zorder=3)
        ax.text(0.5, -0.02, f"$b^*_{{\\rm Un}}={bu:.2f}$,  $b^*_{{\\rm Sh}}={bs:.2f}$,  $\\Lambda={lam:+.2f}$,  $\\nu={loss.mean():.2f}$",
                transform=ax.transAxes, ha="center", va="top", fontsize=FS_VAL)
        ax.set_xlim(P[:, 0].min() - 0.12, P[:, 0].max() + 0.12); ax.set_ylim(P[:, 1].min() - 0.12, P[:, 1].max() + 0.12)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    for e1 in (0, 1):
        axes[0, e1].text(0.5, 1.04, f"$\\varepsilon_1={e1}$", transform=axes[0, e1].transAxes, ha="center", va="bottom",
                         fontsize=FS_HEAD, path_effects=bold)
    for e2 in (0, 1):
        axes[e2, 0].text(-0.03, 0.5, f"$\\varepsilon_2={e2}$", transform=axes[e2, 0].transAxes, ha="right", va="center",
                         fontsize=FS_HEAD, path_effects=bold)
    I_un = gini([f[i] for i in G])
    h = [Line2D([], [], ls="", marker="o", color=C_GAIN, ms=6, label="$u_i\\geq c$"),
         Line2D([], [], ls="", marker="o", color=C_LOSS, ms=6, label="$u_i<c$"),
         Line2D([], [], ls="", label=f"$I_{{\\rm Un}}={I_un:.2f}$,  $I_{{\\rm Sh}}=0$")]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.035), frameon=False, handletextpad=0.2, columnspacing=2.5,
               fontsize=FS_LEG)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(S.FIG, f"{out}.{ext}"), dpi=1000 if ext == "png" else None, bbox_inches="tight")
    print("saved", out)
    # nu margins: distance of b*_Un from the nearest break-even benefit 1 / f_i, in standard errors
    crit = 1 / np.array([f[i] for i in G])
    for st in SETTINGS:
        bb = b[("U", *st)]; se = res[(res.rule == "U") & (res.eps1 == st[0]) & (res.eps2 == st[1])].se.iloc[0]
        gap = np.min(np.abs(crit - bb))
        print(f"U {st}: b* = {bb:.4f} (se {se:.4f}); nearest break-even b = {gap:.4f} away ({gap / se:.1f} se)")


if __name__ == "__main__":
    res = compute()
    draw(res)
