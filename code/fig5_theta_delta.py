# -*- coding: utf-8 -*-
"""fig5_theta_delta.py -- Figure 5 (end of section 3.1): does the result of section 3.1 hold, network by network, for
other selection strengths theta and production functions delta?  (2026-10-08: plan B of the review, chosen by the user.)

Figure 5 (figs/fig5.*): 1 x 2 panels, both axes = the cooperation advantage Lambda on the same asinh scale
(linear for |Lambda| < 0.1, as in figure 3).  x = Lambda at another parameter value (colour; light = smaller, dark =
larger; the same quantity on the x axis of both panels), y = Lambda of the same network and the same visibility at
the main value (theta = 1, delta = 1); below the diagonal = stronger at the other value.  Grid lines 0.3 pt except
the axes at 0.  One point per
network and eps in {0, 0.1, 0.2, 0.3, 0.5, 1}: 40 x 6 points per parameter value.  Dashed: y = x (unchanged).  First
quadrant (light purple) = conflict at both values, third (light orange) = no conflict at both; points in the white
quadrants change their conflict status.  Legend in the empty upper-left quadrant of each panel.
Also prints the numbers for the text and the caption (report()).  The former curve version (p and median chi over eps,
2 x 2) was dropped (2026-10-09).

Data (precision 0.2 %; run cap 4.096e7 for theta, delta != 1):
  E8/E9      results/final/e89_hpc.csv, e89w_hpc.csv (hpc/tools/collect.py e89 / e89w)
  theta = delta = 1: E1 (results/final/e1_hpc.csv, with the fix reruns) and, for eps = 0.5, (0.5,0.5) of E3a / E2
  results of resumed tasks whose rank files have not been downloaded are read from hpc/logs/e89r*_*.log (the wide
  mode prints every result line in full); they are used only for tasks missing from the CSV files.
  theta = 2 tasks with long-lived coexistence (hpc/tasks/e89_excluded_coexistence.txt) are excluded; the network
  counts per cell are printed (figure caption).
Run:  python fig5_theta_delta.py
"""
import os, glob, re
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
import style as S
import networks40 as N40
from kernel import Complex
from fig6_phase import override

FIN = os.path.join(N40.ROOT, "results", "final")
HPC = os.path.join(N40.ROOT, "hpc")
FS = 7
EPS = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
THETA = [0.1, 0.2, 0.5, 1.0, 2.0]
DELTA = [0.5, 0.7, 1.0, 1.4, 2.0]
SEQ_T = LinearSegmentedColormap.from_list("theta", ["#AEBBCB", "#8399B0", "#5C7899", "#3D5A80", "#1B2A3A"])
SEQ_D = LinearSegmentedColormap.from_list("delta", ["#D7B3BA", "#BC8B96", "#A26B79", "#7A3B4D", "#4E1A2B"])
C_REF = "#8A8F98"
GRID = "#E4E7EB"                                          # reference lines, as in figure 4
COLS = ["id", "net", "rule", "eps1", "eps2", "theta", "delta", "seed", "bstar", "se", "runs_per_point", "calls", "flag", "seconds"]
KEY = ["net", "rule", "eps1", "eps2", "theta", "delta"]


def from_logs():
    """result lines printed by the wide-mode resume jobs (identical to their rank-file lines)."""
    rows = []
    for fn in glob.glob(os.path.join(HPC, "logs", "e89r*_*.log")):
        for ln in open(fn, encoding="utf-8", errors="replace"):
            t = ln.split()
            if len(t) >= 14 and t[0].isdigit() and t[12] == "ok":
                rows.append(t[:14])
    d = pd.DataFrame(rows, columns=COLS)
    for c in ("eps1", "eps2", "theta", "delta", "bstar", "se", "seconds"):
        d[c] = d[c].astype(float)
    return d


def load():
    parts = [override(pd.read_csv(os.path.join(FIN, "e1_hpc.csv")))]
    for fn in ("e3a_hpc.csv", "e2a_hpc.csv"):
        x = pd.read_csv(os.path.join(FIN, fn))
        parts.append(x[np.isclose(x.eps1, 0.5) & np.isclose(x.eps2, 0.5)])
    e89 = pd.concat([pd.read_csv(os.path.join(FIN, f)) for f in ("e89_hpc.csv", "e89w_hpc.csv")], ignore_index=True)
    lg = from_logs()
    lg = lg[~lg.id.astype(int).isin(pd.read_csv(os.path.join(FIN, "e89_hpc.csv")).id)]
    print(f"results taken from the resume logs (rank files not downloaded yet): {len(lg)}")
    d = pd.concat(parts + [e89, lg], ignore_index=True)
    for c in ("eps1", "eps2", "theta", "delta"):
        d[c] = d[c].astype(float).round(3)
    d = d.drop_duplicates(KEY, keep="first")
    d = d[np.isclose(d.eps1, d.eps2)]
    rse = d.se.astype(float) / d.bstar.astype(float)                      # precision (figure caption)
    new = ~(np.isclose(d.theta, 1) & np.isclose(d.delta, 1))
    print(f"b* used: {len(d)}, of which E8/E9 {new.sum()}; relative se above 0.2 %: E8/E9 {(rse[new] > 0.002).sum()} "
          f"(up to {rse[new].max():.4f}), main values {(rse[~new] > 0.002).sum()} (up to {rse[~new].max():.4f})")
    w = d.pivot_table(index=["net", "theta", "delta", "eps1"], columns="rule", values="bstar").reset_index()
    w = w.rename(columns={"eps1": "eps"})
    w["L"] = w.Sh / w.U - 1
    cost = {}
    for nm in w.net.unique():
        cx = Complex(N40.build(nm), [])
        for de in DELTA:
            cost[(nm, de)] = cx.gini("U", de) - cx.gini("Sh", de)
    w["dI"] = [cost[(r.net, r.delta)] for r in w.itertuples()]
    w["chi"] = w.L * w.dI
    return w.dropna(subset=["chi"])


def grid(w, col, vals, fixed):
    """per parameter value and eps: p, median Lambda, median chi, network count, rank correlation of Lambda with
    I_Un - I_Sh (the reversal, reported in the text)."""
    s = w[np.isclose(w[fixed], 1.0)]
    out = {}
    for name, v, f in (("p", "L", lambda x: (x > 0).mean()), ("lam", "L", "median"), ("chi", "chi", "median"),
                       ("n", "L", "count")):
        out[name] = s.pivot_table(index=col, columns="eps", values=v, aggfunc=f).reindex(index=vals, columns=EPS)
    out["rho"] = s.groupby([col, "eps"])[["L", "dI"]].apply(lambda g: spearmanr(g.L, g.dI)[0]).unstack().reindex(index=vals, columns=EPS)
    return out


def report(w):
    """numbers for the text and the caption: per parameter value and eps, the share of networks with conflict, median
    Lambda, median chi, network count and the rank correlation of Lambda with I_Un - I_Sh (the reversal).  The
    precision of the b* is printed by load()."""
    for col, vals, fixed in (("theta", THETA, "delta"), ("delta", DELTA, "theta")):
        g = grid(w, col, vals, fixed)
        print(f"\n=== rows {col} ({fixed} = 1): share in conflict / median Lambda / median chi / networks / rank corr(Lambda, I_Un - I_Sh)")
        for k in ("p", "lam", "chi", "n", "rho"):
            print(g[k].round(4).to_string())


DRAW_ORDER = {"theta": [0.5, 2.0, 0.2, 0.1], "delta": [1.4, 0.7, 0.5, 2.0]}   # values far from 1 on top
LIM = (-0.45, 2.0)
TICKS = [-0.3, -0.1, 0, 0.1, 0.3, 1]


def main():
    S.setup(FS)
    w = load()
    report(w)
    rows = [("theta", THETA, "delta", "\\theta", SEQ_T), ("delta", DELTA, "theta", "\\delta", SEQ_D)]
    # 2026-10-08 review: the same quantity on the x axis of both panels (Lambda at the other parameter value), the
    # main value on the y axis; thinner grid lines except at 0; more space between the panels
    fig, axes = plt.subplots(1, 2, figsize=(136 * S.MM, 62 * S.MM), gridspec_kw=dict(wspace=0.5))
    print("\n=== figure 5: every network and eps against the main value")
    for c, (col, vals, fixed, sym, cmap) in enumerate(rows):
        ax = axes[c]
        t = w[np.isclose(w[fixed], 1)].pivot_table(index=["net", "eps"], columns=col, values="L")
        colour = {v: cmap(k / (len(vals) - 1)) for k, v in enumerate(vals)}
        print(f"Lambda range: {np.nanmin(t.values):.3f} to {np.nanmax(t.values):.3f}")
        ax.fill_between([0, LIM[1]], 0, LIM[1], color=S.BG_CONFLICT, lw=0, zorder=0)
        ax.fill_between([LIM[0], 0], LIM[0], 0, color=S.BG_NOCONFLICT, lw=0, zorder=0)
        for v in TICKS:                                             # light reference lines, thinner than in figure 4
            if v != 0:
                ax.axvline(v, color=GRID, lw=0.3, zorder=0.5); ax.axhline(v, color=GRID, lw=0.3, zorder=0.5)
        ax.axhline(0, color="#C9CED4", lw=0.6, zorder=0.6); ax.axvline(0, color="#C9CED4", lw=0.6, zorder=0.6)
        ax.plot(LIM, LIM, color=C_REF, lw=0.6, ls=(0, (4, 2.5)), zorder=1)
        for v in DRAW_ORDER[col]:
            ok = t[[1.0, v]].dropna()
            ax.scatter(ok[v], ok[1.0], s=6, color=colour[v], edgecolors="white", linewidths=0.2, zorder=3)
            flip = ok[np.sign(ok[1.0]) != np.sign(ok[v])]
            pos = ok[(ok[1.0] > 0.05)]
            print(f"  {col} = {v:g}: {len(ok)} points; conflict status changed in {len(flip)} "
                  f"(eps: {flip.index.get_level_values('eps').value_counts().sort_index().to_dict()}, max |Lambda| {flip.abs().values.max() if len(flip) else 0:.3f}); "
                  f"median ratio to the main value where Lambda(main) > 0.05: {(pos[v] / pos[1.0]).median():.2f}")
        for a in ("x", "y"):
            getattr(ax, f"set_{a}scale")("asinh", linear_width=0.1)
        ax.set_xlim(LIM); ax.set_ylim(LIM)
        ax.set_xticks(TICKS); ax.set_xticklabels([f"${v:g}$" for v in TICKS])
        ax.set_yticks(TICKS); ax.set_yticklabels([f"${v:g}$" for v in TICKS])
        ax.xaxis.set_minor_locator(plt.NullLocator()); ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlabel("$\\Lambda$", fontsize=FS + 2)
        ax.set_ylabel(f"$\\Lambda\\;({sym}=1)$", fontsize=FS + 1)
        ax.set_box_aspect(1)
        hs = [Line2D([], [], ls="none", marker="o", ms=3.4, color=colour[v], mec="white", mew=0.3, label=f"${sym}={v:g}$")
              for v in vals if v != 1.0]
        ax.legend(handles=hs, loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False, handletextpad=0.2,
                  labelspacing=0.35, borderaxespad=0.5, fontsize=FS - 0.5)
        S.panel(ax, "ab"[c], x=-0.27, y=1.04, fs=FS + 2)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(S.FIG, f"fig5.{ext}"), dpi=1000 if ext == "png" else None, bbox_inches="tight")
    print("saved figs/fig5.*")


if __name__ == "__main__":
    main()
