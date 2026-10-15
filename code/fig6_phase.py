# -*- coding: utf-8 -*-
"""fig6_phase.py -- Figure 6: phase diagrams of the conflict intensity chi = Lambda (I_Un - I_Sh) over (eps2, eps1).

Data: results/final/e2a_hpc.csv (16 x 16 grid, step 0.1), the E1 diagonal points (results/final/e1_hpc.csv) and, when
present, the stage-2 refinement near the critical line (results/final/e2b_hpc.csv).
Colour: bicubic interpolation of the grid, asinh-compressed, one colour scale for all panels.
Critical line Lambda = 0: with stage 2, on every row eps1 the root in eps2 by linear interpolation between the
refined points; without stage 2, the zero contour of the interpolated surface.
Run:  python fig6_phase.py [NET1 NET2 NET3 NET4]   (default: the four networks of the design doc)
Missing grid cells (unfinished tasks) are filled from the nearest finished cell and the output is marked DRAFT.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import RectBivariateSpline
import style as S
import networks40 as N40
from fig34_uniform import inequality_uniform

FIN = os.path.join(N40.ROOT, "results", "final")
NETS = sys.argv[1:5] if len(sys.argv) >= 5 else ["BA_m4", "RHG_g3.0_T0.1", "THS", "CPH"]   # two patterns x two <e>
NAME = {"BA_m4": "BA $m=4$", "HK_m4_p0.9": "HK $m=4$, $p=0.9$", "RHG_g3.0_T0.1": "RHG $\\gamma=3$, $T=0.1$",
        "CPH": "CPH", "FBF": "FBF", "THS": "THS"}
G = np.round(np.arange(16) / 10, 1)
F = np.linspace(0, 1.5, 301)
DIV = LinearSegmentedColormap.from_list("blue_white_wine", ["#3E6A96", "#8FB8DE", "#F7F7F7", "#C88AA3", "#6E1A2E"])
LW = 0.01                                            # asinh width of the colour scale
# 2026-10-14 typography: the canvas is the display width (164.6 mm, \textwidth), so sizes are true points
W_MM = 164.6
FS = 9                                               # tick labels, colour-bar ticks
FS_TITLE = 9                                         # panel titles (network, <e>, CV)
FS_LAB = 10.5                                        # axis labels eps1, eps2 and chi
FS_LETTER = 12                                       # panel letters


def tr(x):
    return np.arcsinh(np.asarray(x) / LW)


def override(d, add=False, names=("e2r_hpc.csv", "fix_hpc.csv")):
    """replace results by later reruns of the same task (net, rule, eps1, eps2), in this order: e2r (the 12 stage-1
    tasks left unfinished; its other values came from the superseded 'polish' step and are replaced again by fix),
    then fix (revised estimator, 2026-10-07).  add=True also adds rerun tasks absent from d (unfinished ones)."""
    k = ["net", "rule", "eps1", "eps2"]
    for nm in names:
        fn = os.path.join(FIN, nm)
        if not os.path.exists(fn):
            continue
        r = pd.read_csv(fn)
        if nm == "e2r_hpc.csv":                       # only its unfinished stage-1 tasks; the rest is superseded by fix
            r = r[r.flag == "ok"]
        r = r[r.net.isin(d.net.unique())]
        if not add:
            r = r.merge(d[k], on=k)
        if not len(r):
            continue
        d = pd.concat([d.merge(r[k], on=k, how="left", indicator=True).query("_merge == 'left_only'").drop(columns="_merge"), r],
                      ignore_index=True)
    return d


def load():
    a = pd.read_csv(os.path.join(FIN, "e2a_hpc.csv"))
    e1 = override(pd.read_csv(os.path.join(FIN, "e1_hpc.csv")))
    a = override(a, add=True)
    d = pd.concat([a, e1[e1.net.isin(a.net.unique())]], ignore_index=True)
    b = pd.read_csv(os.path.join(FIN, "e2b_hpc.csv")) if os.path.exists(os.path.join(FIN, "e2b_hpc.csv")) else None
    return d, b


def wide(d):
    w = d.pivot_table(index=["net", "eps1", "eps2"], columns="rule", values="bstar").reset_index()
    w["L"] = w.Sh / w.U - 1
    return w


def grid(w, nm):
    Z = np.full((16, 16), np.nan)
    s = w[w.net == nm]
    for r in s.itertuples():
        i, j = int(round(r.eps1 * 10)), int(round(r.eps2 * 10))
        if i < 16 and j < 16 and abs(r.eps1 * 10 - i) < 1e-6 and abs(r.eps2 * 10 - j) < 1e-6:
            Z[i, j] = r.L
    miss = np.argwhere(~np.isfinite(Z))
    ok = np.argwhere(np.isfinite(Z))
    for i, j in miss:                                 # draft only: nearest finished cell
        k = np.argmin(((ok - [i, j]) ** 2).sum(1))
        Z[i, j] = Z[tuple(ok[k])]
    return Z, len(miss)


def critical_line(w_all, nm, sp):
    """(eps2*, eps1) points of Lambda = 0, one per row eps1."""
    pts = []
    for a in G:
        r = w_all[(w_all.net == nm) & np.isclose(w_all.eps1, a)].sort_values("eps2")
        x, y = r.eps2.values, r.L.values
        k = np.where(np.sign(y[:-1]) != np.sign(y[1:]))[0]
        if len(k):
            k = k[0]
            pts.append((x[k] - y[k] * (x[k + 1] - x[k]) / (y[k + 1] - y[k]), a))
    return np.array(pts)


def main():
    d, b = load()
    w = wide(d)
    w_all = wide(pd.concat([d, b], ignore_index=True)) if b is not None else w
    st = pd.read_csv(os.path.join(FIN, "networks40_stats.csv")).set_index("net")
    S.setup(FS)
    # layout in mm (2026-10-14): four square panels in one row on a canvas as wide as the text block
    L_MM, R_MM, GAP = 11.5, 2.7, 5.8                  # left margin (eps1 label + ticks), right margin, gap between panels
    PW = (W_MM - L_MM - R_MM - 3 * GAP) / 4           # panel side
    B_MM, T_MM = 21.7, 15.5                           # below the panels (ticks, eps2, colour bar), above (titles, letters)
    H_MM = B_MM + PW + T_MM
    fig = plt.figure(figsize=(W_MM * S.MM, H_MM * S.MM))
    axes = []
    for i in range(4):
        axes.append(fig.add_axes([(L_MM + i * (PW + GAP)) / W_MM, B_MM / H_MM, PW / W_MM, PW / H_MM],
                                 sharey=axes[0] if axes else None))
        if i:
            axes[i].tick_params(labelleft=False)
    chis, n_miss = {}, {}
    for nm in NETS:
        Z, n_miss[nm] = grid(w, nm)
        dI = inequality_uniform(N40.build(nm))
        chis[nm] = (Z, dI)
    vmax = max(np.nanmax(np.abs(Z * dI)) for Z, dI in chis.values())
    VM = tr(vmax)
    for ax, nm, L in zip(axes, NETS, "ABCD"):
        Z, dI = chis[nm]
        sp = RectBivariateSpline(G, G, Z, kx=3, ky=3)
        Zf = sp(F, F)
        cs = ax.contourf(F, F, tr(Zf * dI), levels=np.linspace(-VM, VM, 41), cmap=DIV, extend="both")
        cs.set_edgecolor("face")      # 2026-10-14: no light seams between color levels in PDF viewers
        cs.set_linewidth(0.15)
        if b is not None:
            pts = critical_line(w_all, nm, sp)
            ax.plot(pts[:, 0], pts[:, 1], color="#8A8F98", lw=0.7, ls=(0, (3, 2)))
        else:
            ax.contour(F, F, Zf, levels=[0], colors="#8A8F98", linewidths=0.7, linestyles=[(0, (3, 2))])
        ax.set_xlim(0, 1.5); ax.set_ylim(0, 1.5); ax.set_aspect("equal")
        for setter in (ax.set_xticks, ax.set_yticks):
            setter([0, 0.5, 1, 1.5])
        ax.set_xticklabels(["$0$", "$0.5$", "$1$", "$1.5$"]); ax.set_yticklabels(["$0$", "$0.5$", "$1$", "$1.5$"])
        ax.set_title(NAME.get(nm, nm) + "\n" + f"$\\langle e\\rangle={st.loc[nm, 'e_mean']:.2f}$,  "
                     f"$\\mathrm{{CV}}={st.loc[nm, 'CV']:.2f}$", pad=4, fontsize=FS_TITLE)
        ax.set_xlabel("$\\varepsilon_2$", fontsize=FS_LAB, labelpad=2)
    axes[0].set_ylabel("$\\varepsilon_1$", fontsize=FS_LAB, labelpad=3)
    # panel letters: top-left of every panel, just above the two-line title
    fig.canvas.draw()
    R = fig.canvas.get_renderer()
    y_let = max(ax.title.get_window_extent(R).y1 for ax in axes) / fig.bbox.height + 0.8 / H_MM
    for ax, L in zip(axes, "ABCD"):
        fig.text(ax.get_position().x0 - 1.5 / W_MM, y_let, L, fontsize=FS_LETTER, fontweight="bold",
                 ha="left", va="bottom", color=S.INK)
    # horizontal colour bar under the panels (review): symmetric about 0, and its 0 sits at the centre of the panel row
    x0, x1 = axes[0].get_position().x0, axes[3].get_position().x1
    xc, wbar = 0.5 * (x0 + x1), 0.78 * (x1 - x0)     # wider than before so the asinh tick labels keep apart
    y_bar, h_bar = 5.3 / H_MM, 2.8 / H_MM
    cax = fig.add_axes([xc - wbar / 2, y_bar, wbar, h_bar])
    cb = fig.colorbar(cs, cax=cax, orientation="horizontal")
    ticks = [-0.1, -0.05, -0.02, 0, 0.02, 0.05, 0.1]
    cb.set_ticks(tr(ticks)); cb.set_ticklabels([f"${t:g}$" for t in ticks]); cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(length=2, pad=1.5, labelsize=FS)
    fig.text(xc, y_bar + h_bar + 0.8 / H_MM, "$\\chi$", fontsize=FS_LAB, ha="center", va="bottom")   # above the 0 tick (review)
    no_fix = not os.path.exists(os.path.join(FIN, "fix_hpc.csv"))
    draft = any(n_miss.values()) or b is None or no_fix
    out = "fig6" + ("_draft" if draft else "")
    if draft:
        note = "DRAFT - stage 1 only" if b is None else "DRAFT"
        if no_fix:
            note += "; reruns with the revised estimator pending"
        if any(n_miss.values()):
            note += "; missing cells filled from nearest: " + ", ".join(f"{k} {v}" for k, v in n_miss.items() if v)
        fig.text(0.995, -0.04, note, ha="right", va="top", fontsize=6, color="#9AA3AD")
    # the final figure keeps its exact canvas (width = display width); check that nothing falls outside it
    fig.canvas.draw()
    tb, fb = fig.get_tightbbox(fig.canvas.get_renderer()), fig.bbox_inches
    print(f"canvas {W_MM:.1f} x {H_MM:.1f} mm; content {(tb.x0 - fb.x0) / S.MM:+.1f} {(tb.x1 - fb.x1) / S.MM:+.1f} "
          f"(x), {(tb.y0 - fb.y0) / S.MM:+.1f} {(tb.y1 - fb.y1) / S.MM:+.1f} (y) mm from the edges")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(S.FIG, f"{out}.{ext}"), dpi=1000 if ext == "png" else None,
                    bbox_inches="tight" if draft else None)
    print("saved", out, "| networks:", NETS, "| chi range", round(min(np.nanmin(Z * dI) for Z, dI in chis.values()), 3),
          round(vmax, 3))


if __name__ == "__main__":
    main()
