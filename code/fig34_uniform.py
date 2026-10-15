# -*- coding: utf-8 -*-
"""fig34_uniform.py -- Figures 3 and 4 of the paper from experiment E1 (results/final/e1.csv).

Figure 3: rule gap Lambda against the inequality cost I_Un - I_Sh, one point per network and visibility level.
Figure 4: (A) b*_Un and (B) b*_Sh against the inequality cost, same colours.
Also prints the checks that the design doc expects (section 3.D, figures 3 and 4).
Run:  python fig34_uniform.py   (works on partial results; missing networks are skipped)
"""
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
import style as S
import networks40 as N40

E1 = os.path.join(N40.ROOT, "results", "final", "e1.csv")
EPS = [0.0, 0.1, 0.2, 0.3, 1.0]
CM = LinearSegmentedColormap.from_list("blue_wine", ["#8FB8DE", "#A9A6CC", "#C88AA3", "#A84860", "#6E1A2E"])
COL = {e: CM(k / (len(EPS) - 1)) for k, e in enumerate(EPS)}
# 2026-10-14: both figures are drawn at their printed width (fig3 0.6 textwidth = 98.8 mm, fig4 textwidth = 164.6 mm),
# so the sizes below are the sizes the reader sees (as figure 5 at its print scale).
FS = 9                                                                   # tick labels, legend
FS_LAB, FS_PANEL = 11, 12                                                # axis labels, panel letters (bold)
FIG3_W, FIG3_H = 103, 78                                                 # canvas (mm) before bbox_inches="tight"
FIG4_W, FIG4_H = 190.8, 72
PANEL_X = -0.16                                                          # panel letters of figure 4 (axes fraction)
GRID = "#E4E7EB"                                                         # reference lines inside figure 4


E1_HPC = os.path.join(N40.ROOT, "results", "final", "e1_hpc.csv")      # written by hpc/tools/collect.py e1


def inequality_uniform(G):
    k = dict(G.degree())
    f = np.array([0.5 * (1 + sum(1 / k[j] for j in G[i])) for i in G])
    return np.abs(f[:, None] - f[None, :]).sum() / (2 * len(f) * f.sum())


def load():
    if os.path.exists(E1_HPC):                                            # formal results from the HPC
        d = pd.read_csv(E1_HPC)
        fix = os.path.join(N40.ROOT, "results", "final", "fix_hpc.csv")   # reruns with the revised estimator
        if os.path.exists(fix):
            k = ["net", "rule", "eps1", "eps2"]
            r = pd.read_csv(fix).merge(d[k], on=k)
            d = pd.concat([d.merge(r[k], on=k, how="left", indicator=True).query("_merge == 'left_only'")
                           .drop(columns="_merge"), r], ignore_index=True)
        d = d.rename(columns={"eps1": "eps"})
        d["I_un"] = d.net.map({n: inequality_uniform(N40.build(n)) for n in d.net.unique()})
    else:                                                                 # local trial runs
        d = pd.read_csv(E1)
    d["eps"] = d.eps.astype(float)
    w = d.pivot_table(index=["net", "eps"], columns="rule", values="bstar").reset_index()
    w["I"] = w.net.map(d.groupby("net").I_un.first())
    w = w.dropna(subset=["U", "Sh"])
    w["Lam"] = w.Sh / w.U - 1
    w["group"] = np.where(w.net.isin(N40.SYN), "synthetic", "real")
    return w


def mticks(ax, xs=None, ys=None, ylab=None):
    if xs is not None:
        ax.set_xticks(xs); ax.set_xticklabels([f"${v:g}$" for v in xs])
    if ys is not None:
        ax.set_yticks(ys); ax.set_yticklabels(ylab or [f"${v:g}$" for v in ys])


def legend_eps(ax, w, corner="upper", counts=True, fs=FS, ms=32, pitch=1.4, box=None):
    """legend of the visibility levels at the upper-left or lower-left corner of the axes.
    All distances are in em of the font size, so the legend keeps its shape whatever the axes size.
    box: (width, height) in em of a white patch under the legend (keeps the reference lines off it)."""
    n = w.net.nunique()
    bb = ax.get_position(); fw, fh = ax.figure.get_size_inches()
    ex, ey = fs / 72 / (bb.width * fw), fs / 72 / (bb.height * fh)       # one em in axes fraction
    X = lambda x: x * ex
    Y = (lambda y: 1 - y * ey) if corner == "upper" else (lambda y: y * ey)  # y = em from the corner
    if box is not None:
        ax.add_patch(plt.Rectangle((X(0.25), Y(0.25)), X(box[0]), box[1] * ey, transform=ax.transAxes, fc="white", ec="none", zorder=2))
    if counts:
        ax.text(X(0.85), Y(0.8), "networks in conflict", transform=ax.transAxes, va="top", fontsize=fs)
    first = 2.55 if corner == "upper" else 1.2 + (len(EPS) - 1) * pitch
    for k, e in enumerate(EPS):
        y = Y(first + k * pitch) if corner == "upper" else Y(first - k * pitch)
        ax.scatter([X(1.3)], [y], s=ms, color=COL[e], edgecolors="white", linewidths=0.55, transform=ax.transAxes, zorder=10, clip_on=False)
        ax.text(X(2.05), y, f"$\\varepsilon={e:g}$", transform=ax.transAxes, va="center", ha="left", fontsize=fs)
        if counts:
            c = int((w[np.isclose(w.eps, e)].Lam > 0).sum())
            ax.text(X(7.3), y, f"${c}$", transform=ax.transAxes, va="center", ha="right", fontsize=fs)
            ax.text(X(7.3), y, f"$/{n}$", transform=ax.transAxes, va="center", ha="left", fontsize=fs)


def fig3(w, out="fig3"):
    S.setup(FS)
    lo = min(-0.42, w.Lam.min() * 1.15); hi = max(2.0, w.Lam.max() * 1.3)
    xmax = max(0.32, w.I.max() * 1.08)
    fig, ax = plt.subplots(figsize=(FIG3_W * S.MM, FIG3_H * S.MM))
    ax.axhspan(0, hi, color=S.BG_CONFLICT, lw=0, zorder=0); ax.axhspan(lo, 0, color=S.BG_NOCONFLICT, lw=0, zorder=0)
    ax.axhline(0, color="#8A8F98", lw=0.65, ls=(0, (3, 2)), zorder=1)     # Lambda = 0: thin grey dashes (as figures 5, 6)
    for k, e in enumerate(EPS):
        s = w[np.isclose(w.eps, e)]
        ax.scatter(s.I, s.Lam, s=30, color=COL[e], edgecolors="white", linewidths=0.55, zorder=3 + k)
    ax.set_yscale("asinh", linear_width=0.1); ax.set_ylim(lo, hi); ax.set_xlim(0, xmax)
    yt = [v for v in (-0.3, -0.1, 0, 0.1, 0.3, 1, 2, 5) if lo <= v <= hi]
    mticks(ax, xs=[v for v in (0, 0.1, 0.2, 0.3, 0.4) if v <= xmax], ys=yt, ylab=[f"${v:g}$".replace("-", "-") for v in yt])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("$I_{\\rm Un}-I_{\\rm Sh}$", fontsize=FS_LAB); ax.set_ylabel("$\\Lambda$", fontsize=FS_LAB)
    legend_eps(ax, w)
    save(fig, out)


def save(fig, out):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(S.FIG, f"{out}.{ext}"), dpi=1000 if ext == "png" else None, bbox_inches="tight")
    print("saved", out)


def fig4(w, out="fig4"):
    S.setup(FS)
    xmax = max(0.32, w.I.max() * 1.08)
    fig, axes = plt.subplots(1, 2, figsize=(FIG4_W * S.MM, FIG4_H * S.MM), gridspec_kw=dict(wspace=0.34))
    for ax, rule, sub in ((axes[0], "U", "Un"), (axes[1], "Sh", "Sh")):
        for k, e in enumerate(EPS):
            s = w[np.isclose(w.eps, e)]
            ax.scatter(s.I, s[rule], s=28, color=COL[e], edgecolors="white", linewidths=0.55, zorder=3 + k)
        top = 2.1
        lo = np.floor((w[rule].min() - 0.05) * 20) / 20
        # place the tick 2 at the same height in both panels: top margin proportional to the panel range
        frac = (2 - 0.45) / (2.1 - 0.45)
        lo = min(lo, 2 - frac * (top - lo)) if rule == "U" else lo
        topr = 2 + (2 - lo) * (1 - frac) / frac
        ax.set_ylim(lo, topr); ax.set_xlim(0, xmax)
        step = 0.5 if rule == "U" else 0.2
        ys = [round(v, 2) for v in np.arange(0, 2.01, step) if v >= lo - 1e-9]
        mticks(ax, xs=[round(v, 2) for v in np.arange(0, xmax + 1e-9, 0.05)], ys=ys)
        # light reference lines at the main ticks: x = 0.1, 0.2, 0.3 and every labelled y
        for v in (0.1, 0.2, 0.3):
            ax.axvline(v, color=GRID, lw=0.5, zorder=0.5)
        for v in ys:
            ax.axhline(v, color=GRID, lw=0.5, zorder=0.5)
        ax.set_xlabel("$I_{\\rm Un}-I_{\\rm Sh}$", fontsize=FS_LAB); ax.set_ylabel(f"$b^*_{{\\rm {sub}}}$", fontsize=FS_LAB)
    for a, l in zip(axes, "AB"):
        S.panel(a, l, x=PANEL_X, y=1.03, fs=FS_PANEL)
    for ax in axes:                                               # the same legend in the lower-left corner of both panels
        legend_eps(ax, w, corner="lower", counts=False, pitch=1.3, box=(5.6, 6.8))   # white box keeps the reference lines off the legend
    save(fig, out)


def checks(w):
    print(f"networks with all 5 levels: {w.groupby('net').size().eq(5).sum()} / {w.net.nunique()}")
    for e in EPS:
        s = w[np.isclose(w.eps, e)]
        if len(s) < 5:
            continue
        cv = s.net.map(lambda n: N40_stats.loc[n, "CV"])
        print(f"eps={e:<4g} n={len(s):2d} conflict={int((s.Lam > 0).sum()):2d}  rho(Lam,I)={stats.spearmanr(s.Lam, s.I).correlation:+.2f}"
              f"  rho(Lam,CV)={stats.spearmanr(s.Lam, cv).correlation:+.2f}  bUn {s.U.min():.3f}-{s.U.max():.3f}"
              f"  rho(bUn,I)={stats.spearmanr(s.U, s.I).correlation:+.2f}  rho(bSh,I)={stats.spearmanr(s.Sh, s.I).correlation:+.2f}")
    s0 = w[w.eps == 0]
    if len(s0) > 3:
        p = np.polyfit(s0.I, s0.Sh, 1); r = np.corrcoef(s0.I, s0.Sh)[0, 1]
        print(f"b*_Sh(0) = {p[1]:.3f} {p[0]:+.3f} I   (r = {r:.3f})")


N40_stats = pd.read_csv(os.path.join(N40.ROOT, "results", "final", "networks40_stats.csv")).set_index("net")

if __name__ == "__main__":
    w = load()
    checks(w)
    fig3(w); fig4(w)
