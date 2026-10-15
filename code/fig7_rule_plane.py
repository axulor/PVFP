# -*- coding: utf-8 -*-
"""fig7_rule_plane.py -- Figure 7: rule-plane small multiples on a two-way grid of 20 networks (design doc, figure 7).

Columns: degree heterogeneity CV increases to the right (< 0.3, 0.3-0.5, 0.5-0.75, 0.75-1, > 1);
rows: mean embeddedness <e> increases downwards (< 1, 1-3, 3-5.5, > 5.5).  Each panel is one network in the plane
(b*_Un, b*_Sh); the diagonal is Lambda = 0, conflict above it.  From the direct-only point (0,0) (small dark dot)
two thin arrows give the same visibility weight to common neighbors only ((1,0), teal triangle) or to non-common
neighbors only ((0,1), brick-red square).  All panels are square and share one scale.

Data (theta = delta = 1, precision 0.2 %):  (0,0) from E1 (results/final/e1_hpc.csv, with the fix reruns);
(1,0), (0,1) from E3a (results/final/e3a_hpc.csv) and, for the five E2 networks, from the E2 grid (e2a + reruns).
Panels without data are left empty and the output is marked DRAFT.  Also prints the checks of the design doc
(counts over all 40 networks, rank correlations, the eta > 0.5 subset).
Run:  python fig7_rule_plane.py
"""
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe
from scipy.stats import spearmanr
import style as S
import networks40 as N40
from fig6_phase import override

FIN = os.path.join(N40.ROOT, "results", "final")
# 2026-10-14 typography: the canvas is the display width (164.6 mm, \textwidth), so sizes are true points
W_MM = 164.6
FS = 8                     # tick labels
FS_NAME = 8                # network names (panel titles)
FS_IN = 7.5                # CV and <e> values inside the panels
FS_HEAD = 10               # column (CV) and row (<e>) bin headers
FS_AX = 11                 # shared axis labels b*_Un, b*_Sh and the gradient labels CV, <e>
FS_LEG = 9                 # legend
C_COM, C_NON, C_DOT, C_DIAG, C_TXT = "#3A7D7C", "#C8553D", "#3C4048", "#8A8F98", "#5A6270"
SETS = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
END = {(1.0, 0.0): ("^", C_COM, 13), (0.0, 1.0): ("s", C_NON, 9)}   # end markers: marker, color, area (pt^2)
S_DOT = 4                                                           # 2026-10-08: markers a little smaller
ARROW = dict(arrowstyle="->,head_length=0.35,head_width=0.17", lw=0.4, mutation_scale=4.5)
HEAD = 0.35 * 4.5
GRID = [["ER_k16", "ER_k8", "CL_g3.2", "BA_m4", "BA_m2"],
        ["WS_p0.2", "UTM", "DOL", "KAR", "UGH"],
        ["WS_p0.05", "RGG_k6", "THS", "HK_m8_p0.9", "RHG_g3.0_T0.5"],
        ["UTE", "ESH", "CPH", "RHG_g3.0_T0.1", "RHG_g2.7_T0.1"]]
SHORT = {"ER_k16": "ER $k=16$", "ER_k8": "ER $k=8$", "CL_g3.2": "CL $\\gamma=3.2$", "BA_m4": "BA $m=4$",
         "BA_m2": "BA $m=2$", "WS_p0.2": "WS $p=0.2$", "WS_p0.05": "WS $p=0.05$", "RGG_k6": "RGG $k=6$",
         "HK_m8_p0.9": "HK 8/0.9", "RHG_g3.0_T0.5": "RHG 3.0/0.5", "RHG_g3.0_T0.1": "RHG 3.0/0.1",
         "RHG_g2.7_T0.1": "RHG 2.7/0.1"}
COLS = ["$<0.3$", "$0.3$–$0.5$", "$0.5$–$0.75$", "$0.75$–$1$", "$>1$"]
ROWS = ["$<1$", "$1$–$3$", "$3$–$5.5$", "$>5.5$"]
KEY = ["net", "rule", "eps1", "eps2"]


def load():
    """b* of every (net, rule, eps1, eps2) available at theta = delta = 1; later sources override earlier ones."""
    parts = [override(pd.read_csv(os.path.join(FIN, "e1_hpc.csv")))]
    a = os.path.join(FIN, "e2a_hpc.csv")
    if os.path.exists(a):
        parts.append(override(pd.read_csv(a), add=True))
    e3 = os.path.join(FIN, "e3a_hpc.csv")
    if os.path.exists(e3):
        parts.append(pd.read_csv(e3))
    d = pd.concat(parts, ignore_index=True)
    d = d[np.isclose(d.theta, 1) & np.isclose(d.delta, 1)]
    d["eps1"], d["eps2"] = d.eps1.round(3), d.eps2.round(3)
    d = d.drop_duplicates(KEY, keep="last")
    w = d.pivot_table(index=["net", "eps1", "eps2"], columns="rule", values="bstar").reset_index()
    w["L"] = w.Sh / w.U - 1
    return w


def point(w, net, st):
    r = w[(w.net == net) & np.isclose(w.eps1, st[0]) & np.isclose(w.eps2, st[1])]
    return (float(r.U.iloc[0]), float(r.Sh.iloc[0])) if len(r) and np.isfinite(r[["U", "Sh"]].values).all() else None


def checks(w, stats):
    """the statements of the design doc (figure 7, 'expected'), over all 40 main networks."""
    s = stats.set_index("net")
    print("\n--- checks over the 40 main networks ---")
    for st, want in (((1.0, 0.0), "no conflict"), ((0.0, 1.0), "conflict"), ((0.5, 0.5), "SI; figure 5 at eps = 0.5"),
                     ((1.0, 0.5), "SI")):
        r = w[np.isclose(w.eps1, st[0]) & np.isclose(w.eps2, st[1]) & w.net.isin(s.index)].dropna(subset=["L"])
        print(f"{st}: {len(r)} networks; conflict (Lambda > 0) in {(r.L > 0).sum()}; "
              f"b*_Un {r.U.min():.3f}-{r.U.max():.3f}; Lambda {r.L.min():+.3f} to {r.L.max():+.3f}   (expected: {want})")
    piv = {st: w[np.isclose(w.eps1, st[0]) & np.isclose(w.eps2, st[1])].set_index("net") for st in SETS}
    nets = sorted(set(piv[SETS[0]].index) & set(piv[SETS[1]].index) & set(piv[SETS[2]].index) & set(s.index))
    if len(nets) < 5:
        print("too few networks with all three settings for the correlations"); return
    dSh_com = np.array([piv[(1.0, 0.0)].Sh[n] - piv[(0.0, 0.0)].Sh[n] for n in nets])
    dSh_non = np.array([piv[(0.0, 1.0)].Sh[n] - piv[(0.0, 0.0)].Sh[n] for n in nets])
    cv, em = s.loc[nets, "CV"].values, s.loc[nets, "e_mean"].values
    print(f"rank correlations over {len(nets)} networks:  b*_Sh(0,1) - b*_Sh(0,0) vs CV {spearmanr(dSh_non, cv)[0]:+.2f} "
          f"(exploration +0.87);  b*_Sh(1,0) - b*_Sh(0,0) vs <e> {spearmanr(dSh_com, em)[0]:+.2f} (exploration +0.72), "
          f"vs CV {spearmanr(dSh_com, cv)[0]:+.2f} (exploration -0.16)")
    print(f"brick-red arrow, b*_Sh(0,1) - b*_Sh(0,0) < 0 for CV < 0.4: "
          f"{sum(dSh_non[cv < 0.4] < 0)}/{sum(cv < 0.4)};  > 0 for CV > 0.6: {sum(dSh_non[cv > 0.6] > 0)}/{sum(cv > 0.6)}")
    ratio = pd.Series({n: piv[(1.0, 0.0)].Sh[n] / piv[(0.0, 0.0)].Sh[n] for n in nets}).sort_values()
    print("teal arrow, b*_Sh(1,0) / b*_Sh(0,0):  " + ", ".join(f"{n} {v:.3f}" for n, v in ratio.items()))
    sh = pd.read_csv(os.path.join(FIN, "networks40_share.csv")).set_index("net").share_pair
    hi = [n for n in nets if sh.get(n, 0) > 0.5]
    print(f"eta > 0.5 (more common than non-common third parties per pair): {len(hi)} networks "
          f"({', '.join(hi)}); conflict at (1,0) in {sum(piv[(1.0, 0.0)].L[n] > 0 for n in hi)}")


def main():
    S.setup(FS)
    w = load()
    stats = pd.read_csv(os.path.join(FIN, "networks40_stats.csv"))
    st_of = stats.set_index("net")
    data = {}
    for r in range(4):
        for c in range(5):
            P = {st: point(w, GRID[r][c], st) for st in SETS}
            if all(v is not None for v in P.values()):
                data[(r, c)] = P
    draft = len(data) < 20
    allp = np.array([p for P in data.values() for p in P.values()]) if data else np.array([[0.4, 1.3], [2.1, 2.1]])
    X0, X1 = np.floor(allp[:, 0].min() * 10 - 0.5) / 10, np.ceil(allp[:, 0].max() * 10 + 0.5) / 10
    Y0, Y1 = np.floor(allp[:, 1].min() * 10 - 0.5) / 10, np.ceil(allp[:, 1].max() * 10 + 0.5) / 10
    print(f"panels with data: {len(data)} of 20;  axis range b*_Un {X0:g}-{X1:g}, b*_Sh {Y0:g}-{Y1:g}")
    XT = [t for t in np.round(np.arange(0, 3.01, 0.5), 1) if X0 <= t <= X1]
    YT = [t for t in np.round(np.arange(0, 3.01, 0.2), 1) if Y0 <= t <= Y1]

    # layout in mm (2026-10-14): 4 x 5 square panels on a canvas as wide as the text block
    L_MM, R_MM = 14.2, 14.8          # left: b*_Sh label + tick labels; right: <e> arrow + row headers
    HGAP, VGAP = 2.3, 7.6            # between panels (VGAP holds the network name of the panel below)
    PW = (W_MM - L_MM - R_MM - 4 * HGAP) / 5
    B_MM, T_MM = 19.0, 13.0          # below: tick labels, b*_Un, legend; above: CV arrow and column headers
    H_MM = B_MM + 4 * PW + 3 * VGAP + T_MM
    fig = plt.figure(figsize=(W_MM * S.MM, H_MM * S.MM))
    axes = [[fig.add_axes([(L_MM + c * (PW + HGAP)) / W_MM, (B_MM + (3 - r) * (PW + VGAP)) / H_MM, PW / W_MM, PW / H_MM])
             for c in range(5)] for r in range(4)]
    for r in range(4):
        for c in range(5):
            ax, n = axes[r][c], GRID[r][c]
            ax.set_box_aspect(1)
            ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1)
            ax.set_xticks(XT); ax.set_yticks(YT)
            ax.set_xticklabels([f"${t:g}$" for t in XT] if r == 3 else [])
            ax.set_yticklabels([f"${t:g}$" for t in YT] if c == 0 else [])
            ax.set_title(SHORT.get(n, n), pad=1.5, fontsize=FS_NAME)
            ax.fill_between([X0, X1], [X0, X1], [Y1, Y1], color=S.BG_CONFLICT, lw=0, zorder=0)
            ax.fill_between([X0, X1], [Y0, Y0], [X0, X1], color=S.BG_NOCONFLICT, lw=0, zorder=0)
            ax.plot([X0, X1], [X0, X1], color=C_DIAG, lw=0.6, ls=(0, (3, 2)), zorder=1)
            m = st_of.loc[n]
            # 2026-10-14: the two lines at fixed baselines (1.1 em apart) so the larger text keeps clear of the
            # diagonal (UTE, <e> = 11.9) and of the brick-red arrow (BA m = 2)
            for y, t in ((0.199, f"$\\mathrm{{CV}}={m.CV:.2f}$"), (0.083, f"$\\langle e\\rangle={m.e_mean:.1f}$")):
                ax.text(0.035, y, t, transform=ax.transAxes, ha="left", va="baseline", fontsize=FS_IN, color=C_TXT)
            if (r, c) not in data:
                ax.text(0.5, 0.55, "(no data yet)", transform=ax.transAxes, ha="center", va="center", color="#9AA3AD", fontsize=6)

    fig.canvas.draw()
    PT = fig.dpi / 72.0
    r_dot = np.sqrt(S_DOT) / 2
    for (r, c), P in data.items():
        ax = axes[r][c]
        p0 = ax.transData.transform(P[(0.0, 0.0)])
        for st, (mk, col, area) in END.items():
            p1 = ax.transData.transform(P[st])
            r_end = np.sqrt(area) / 2
            avail = np.hypot(*(p1 - p0)) / PT - r_dot - r_end                 # free length between the markers (pt)
            # 2026-10-08 review (ESH lost its teal arrow): the arrow runs from edge to edge, and a short arrow gets
            # a proportionally smaller head; only when the markers touch is the arrow left out
            if avail >= 0.6:
                hl = min(HEAD, 0.85 * avail)
                arrow = dict(ARROW, arrowstyle=f"->,head_length={hl / 4.5:.3f},head_width={0.17 * hl / HEAD:.3f}")
                ax.annotate("", P[st], P[(0.0, 0.0)], arrowprops=dict(color=col, shrinkA=r_dot, shrinkB=r_end, **arrow), zorder=3)
            if avail < HEAD:
                print(f"  {GRID[r][c]} {st}: free length {avail:.2f} pt -> {'short head' if avail >= 0.6 else 'no arrow'}")
            ax.scatter(*P[st], marker=mk, s=area, color=col, edgecolors="white", linewidths=0.3, zorder=4)
        ax.scatter(*P[(0.0, 0.0)], s=S_DOT, color=C_DOT, edgecolors="white", linewidths=0.25, zorder=5)

    # axis labels (2026-10-08 review: further from the panels, bold) and the two structural gradients (larger, bold)
    # 2026-10-14: positions in mm on the display-width canvas
    x_l, x_r = axes[0][0].get_position().x0, axes[0][4].get_position().x1
    y_t0, y_b0 = axes[0][0].get_position().y1, axes[3][0].get_position().y0
    fig.text(0.5 * (x_l + x_r), 9.3 / H_MM, "$\\mathbfit{b}^{*}_{\\mathbf{Un}}$", fontsize=FS_AX, ha="center", va="center")
    fig.text(3.4 / W_MM, 0.5 * (y_t0 + y_b0), "$\\mathbfit{b}^{*}_{\\mathbf{Sh}}$", fontsize=FS_AX, ha="center", va="center",
             rotation=90)
    BOLD = [pe.withStroke(linewidth=0.45, foreground=C_TXT)]          # thickens text, including math
    ARR = dict(arrowstyle="->,head_length=4.5,head_width=2.2", lw=1.0, color=C_TXT)
    y_t = y_t0 + 6.3 / H_MM
    fig.add_artist(FancyArrowPatch((x_l, y_t), (x_r, y_t), transform=fig.transFigure, **ARR))
    fig.text(x_r, y_t + 1.5 / H_MM, "$\\mathrm{CV}$", ha="right", va="bottom", fontsize=FS_AX, color=C_TXT, path_effects=BOLD)
    for c, t in enumerate(COLS):
        pos = axes[0][c].get_position()
        fig.text(0.5 * (pos.x0 + pos.x1), y_t + 1.5 / H_MM, t, ha="center", va="bottom", fontsize=FS_HEAD, color=C_TXT,
                 path_effects=BOLD)
    x_a = x_r + 3.0 / W_MM
    fig.add_artist(FancyArrowPatch((x_a, y_t0), (x_a, y_b0), transform=fig.transFigure, **ARR))
    fig.text(x_a + 1.6 / W_MM, y_b0, "$\\langle e\\rangle$", ha="left", va="bottom", fontsize=FS_AX, color=C_TXT, path_effects=BOLD)
    for r, t in enumerate(ROWS):
        pos = axes[r][4].get_position()
        fig.text(x_a + 1.6 / W_MM, 0.5 * (pos.y0 + pos.y1), t, ha="left", va="center", fontsize=FS_HEAD, color=C_TXT,
                 path_effects=BOLD)

    # legend: three entries in the notation of the text, centred under the panel block (spacings in mm)
    LW_MM = (x_r - x_l) * W_MM
    lax = fig.add_axes([x_l, 1.0 / H_MM, x_r - x_l, 4.0 / H_MM]); lax.set_xlim(0, LW_MM); lax.set_ylim(0, 1); lax.axis("off")
    fig.canvas.draw(); R = fig.canvas.get_renderer(); PX_MM = lax.get_window_extent(R).width / LW_MM

    def legend(x):
        arts = []

        def put_text(x, t):
            h = lax.text(x, 0.5, t, va="center", fontsize=FS_LEG)        # 2026-10-08 review: larger, not bold
            arts.append(h)
            return x + h.get_window_extent(R).width / PX_MM

        arts.append(lax.scatter([x], [0.5], s=S_DOT + 2, color=C_DOT, edgecolors="white", linewidths=0.25))
        x = put_text(x + 1.7, "$(\\varepsilon_1,\\varepsilon_2)=(0,0)$") + 7.5
        for (mk, col, area), t in ((END[(1.0, 0.0)], "$(\\varepsilon_1,\\varepsilon_2)=(1,0)$"),
                                   (END[(0.0, 1.0)], "$(\\varepsilon_1,\\varepsilon_2)=(0,1)$")):
            arts.append(lax.annotate("", (x + 4.3, 0.5), (x, 0.5), arrowprops=dict(color=col, shrinkA=0, shrinkB=0, **ARROW)))
            arts.append(lax.scatter([x + 5.3], [0.5], marker=mk, s=area, color=col, edgecolors="white", linewidths=0.3))
            x = put_text(x + 7.2, t) + 7.5
        return arts, x - 7.5

    arts, x_end = legend(0.0)
    for a in arts:
        a.remove()
    legend(0.5 * (LW_MM - x_end))
    if draft:
        fig.text(0.5, -0.02, f"DRAFT - {20 - len(data)} of 20 panels without data (E3a not collected yet)",
                 ha="center", fontsize=6, color="#9AA3AD")
    out = "fig7" + ("_draft" if draft else "")
    # the final figure keeps its exact canvas (width = display width); check that nothing falls outside it
    fig.canvas.draw()
    tb, fb = fig.get_tightbbox(fig.canvas.get_renderer()), fig.bbox_inches
    print(f"canvas {W_MM:.1f} x {H_MM:.1f} mm, panel side {PW:.1f} mm; content {(tb.x0 - fb.x0) / S.MM:+.1f} "
          f"{(tb.x1 - fb.x1) / S.MM:+.1f} (x), {(tb.y0 - fb.y0) / S.MM:+.1f} {(tb.y1 - fb.y1) / S.MM:+.1f} (y) mm from the edges")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(S.FIG, f"{out}.{ext}"), dpi=1000 if ext == "png" else None,
                    bbox_inches="tight" if draft else None)
    print("saved", out)
    checks(w, stats)


if __name__ == "__main__":
    main()
