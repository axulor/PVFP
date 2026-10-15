# -*- coding: utf-8 -*-
"""fig9_10_split.py -- figures 9 and 10 (section 3.3): where the cooperators' lead in visible payoff
comes from, split by transfer direction (design doc, decision 2026-10-08: plan ii).

Every bar = Uniform minus Shapley of the cooperators' lead from one class of third parties (left half: common
neighbors, right half: non-common neighbors), in the process where both classes are observed, (eps1, eps2) = (1, 1),
at the same b for both rules (mean of the two critical benefits).  Two stacked segments:
  k_j > k_l   pools in which the candidate j is the larger-degree member: Uniform moves output to j  (favors Uniform)
  k_j < k_l   pools in which j is the smaller-degree member: Uniform moves output away from j        (favors Shapley)
(pools with k_j = k_l are identical under both rules); diamond = net effect (sum), with its standard error.
The states are sampled from both rules' invasion processes and accounted under both rules; the two processes are
averaged with equal weight (results per process: results/final/e4b_comp.csv, e4c_comp.csv).

Figure 9 (figs/fig9.*): ten networks of the main set (rows 1 and 4 of the figure-7 grid: <e> < 1 and <e> > 5.5), by CV.
Figure 10 (figs/fig10.*): the composition series (degree sequence, triangle count and eta = 0.3 fixed; most homogeneous, as generated,
most mixed triangles; tick labels homog. / S$^1$ / mixed), three degree sequences (CV ~ 0.4, 0.65, 0.9), mean over three independent networks (error
bar: +-1 SD over the networks).
Run:  python fig9_10_split.py
Both figures have two panels (common / non-common neighbors), each with its own y scale; the background marks which
rule is favored (as in figure 7).
"""
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import style as S

FIN = os.path.join(S.ROOT, "results", "final")
FS = 9                                                   # 2026-10-14: sizes in true points; the PDF is ~164.6 mm wide,
FS_TITLE, FS_YLAB, FS_NOTE, FS_LETTER = 10.5, 11, 9, 12  # the manuscript display width (scale ~1)
ROT = 50                                                  # rotation of the bar names (40 -> 50: room for 9 pt names)
C_HI, C_LO, C_NET = "#C9A35E", "#6B6889", "#2B2D42"      # muted ochre / grey violet (figure-10 palette), net effect
HALF = [("common", "Common neighbors", S.C_IN), ("noncommon", "Non-common neighbors", S.C_OUT)]
TEN = [["ER_k16", "ER_k8", "CL_g3.2", "BA_m4", "BA_m2"], ["UTE", "ESH", "CPH", "RHG_g3.0_T0.1", "RHG_g2.7_T0.1"]]
SHORT = {"ER_k16": "ER $k=16$", "ER_k8": "ER $k=8$", "CL_g3.2": "CL $\\gamma=3.2$", "BA_m4": "BA $m=4$", "BA_m2": "BA $m=2$",
         "UTE": "UTE", "ESH": "ESH", "CPH": "CPH", "RHG_g3.0_T0.1": "RHG 3.0/0.1", "RHG_g2.7_T0.1": "RHG 2.7/0.1"}
GRID = "#ECEEF1"
TXT_U, TXT_SH = "#8A76A6", "#C8894A"                   # text in the hues of the backgrounds (light purple / light orange)


def averaged(fn):
    """process-averaged components: value = mean over the two processes, se = combined standard error."""
    d = pd.read_csv(os.path.join(FIN, fn))
    g = d.groupby(["net", "channel", "component"])
    out = g.value.mean().to_frame("value")
    out["se"] = g.se.apply(lambda x: np.sqrt((x ** 2).sum()) / len(x))
    return out.reset_index()


def comp_of(a, net, chn):
    s = a[(a.net == net) & (a.channel == chn)].set_index("component")
    if not len(s):                                     # network not computed yet
        return 0.0, 0.0, np.nan, 0.0
    return s.loc["hi", "value"], s.loc["lo", "value"], s.loc["net_effect", "value"], s.loc["net_effect", "se"]


def bar(ax, x, hi, lo, tot, err):
    ax.bar(x, hi, width=0.64, color=C_HI, lw=0, zorder=3)
    ax.bar(x, lo, width=0.64, color=C_LO, lw=0, zorder=3)
    ax.errorbar([x], [tot], yerr=[err], fmt="none", ecolor=C_NET, elinewidth=0.9, capsize=2.2, capthick=0.9, zorder=4)
    ax.scatter([x], [tot], marker="D", s=24, color=C_NET, edgecolors="white", linewidths=0.55, zorder=5)


def legend(fig, y):
    fig.legend(handles=[Patch(fc=C_HI, label="$k_j>k_l$"), Patch(fc=C_LO, label="$k_j<k_l$"),
                        Line2D([], [], ls="", marker="D", color=C_NET, mec="white", mew=0.55, ms=6, label="net effect")],
               loc="lower center", bbox_to_anchor=(0.5, y), ncol=3, frameon=False, fontsize=FS, handlelength=1.4,
               handleheight=0.9, columnspacing=2.2, borderpad=0.2)


COMP_SERIES = [("cv035", 0.4), ("cv065", 0.65), ("cv100", 0.9)]       # labels: measured CV 0.37-0.41, 0.62-0.67, 0.91-0.94
COMP = [("homog", "homog."), ("asgen", "S$^1$"), ("mixed", "mixed")]     # S1: the network as produced by the S1 model


# ------------------------------------------------------------------------------------------------ the two figures
def two_panel(groups, values, out, group_labels, xnote=None):
    """the two classes of third parties in two panels side by side, each with its own y scale (the common-neighbor
    bars are about ten times smaller).  groups: list of lists of
    (key, tick label); values[(key, cls)] = (hi, lo, net, err)."""
    # layout in mm (2026-10-14): left margin (y label + ticks) | panel A | gap (ticks of B) | panel B | side notes
    W, H, L, AW, GAP, BOT, AH = 168.0, 96.0, 16.0, 60.75, 12.5, 24.0, 58.0
    fig = plt.figure(figsize=(W * S.MM, H * S.MM))
    axes = [fig.add_axes([(L + i * (AW + GAP)) / W, BOT / H, AW / W, AH / H]) for i in range(2)]
    for ax, (cls, name, col) in zip(axes, HALF):
        x, ticks, labels, vals, gx = 0.0, [], [], [], []
        for gi, grp in enumerate(groups):
            g0 = x
            for key, lab in grp:
                hi, lo, tot, err = values[(key, cls)]
                bar(ax, x, hi, lo, tot, err); vals += [v for v in (hi, lo, tot + err, tot - err) if np.isfinite(v)]
                ticks.append(x); labels.append(lab); x += 1
            gx.append((g0, x - 1)); x += 0.6
        lo_, hi_ = min(vals + [0]), max(vals + [0])
        span = hi_ - lo_
        step = [s_ for s_ in (0.01, 0.02, 0.05, 0.1, 0.2) if span / s_ <= 7][0]
        yt = [round(v, 3) for v in np.arange(np.floor(lo_ / step) * step, hi_ + step * 0.999, step)]
        ylim = (yt[0] - 0.18 * step, yt[-1] + 0.6 * step)
        ax.axhspan(0, ylim[1], color=S.BG_CONFLICT, lw=0, zorder=0)        # favors Uniform (as in figure 7)
        ax.axhspan(ylim[0], 0, color=S.BG_NOCONFLICT, lw=0, zorder=0)      # favors Shapley
        for v in yt:
            if v != 0:
                ax.axhline(v, color=GRID, lw=0.4, zorder=0.5)
        ax.axhline(0, color="#8A8F98", lw=0.8, zorder=2)
        ax.set_xlim(-0.7, x - 0.9); ax.set_ylim(*ylim)
        ax.set_yticks(yt); ax.set_yticklabels([f"${t:g}$" for t in yt])
        ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=ROT, ha="right", rotation_mode="anchor")
        ax.set_title(name, color=col, fontsize=FS_TITLE, pad=4)
        if xnote:
            ax.text(0.5, -0.255, xnote, transform=ax.transAxes, ha="center", va="top", fontsize=FS_NOTE, color="#4A505A")
        for (g0, g1), gl in zip(gx, group_labels):
            ax.text(0.5 * (g0 + g1), ylim[0] + 0.022 * (ylim[1] - ylim[0]), gl, ha="center", va="bottom", fontsize=FS_NOTE,
                    color="#4A505A")
    axes[0].set_ylabel("Uniform $-$ Shapley", fontsize=FS_YLAB, labelpad=3)
    axes[1].text(1.025, 0.85, "favors\nUniform", transform=axes[1].transAxes, va="center", ha="left", fontsize=FS_NOTE, color=TXT_U)
    axes[1].text(1.025, 0.12, "favors\nShapley", transform=axes[1].transAxes, va="center", ha="left", fontsize=FS_NOTE, color=TXT_SH)
    # panel letters on the baseline of the panel titles, left-aligned with the leftmost text of the panel
    # (A: the y label, B: its tick labels)
    yl = 1 + (4 / 72 / S.MM) / AH                                       # title pad (4 pt) in axes fraction
    fig.canvas.draw(); rd = fig.canvas.get_renderer()                  # label positions are set at draw time
    for ax, letter, items in ((axes[0], "A", [axes[0].yaxis.label]), (axes[1], "B", axes[1].get_yticklabels())):
        x0 = min(t.get_window_extent(rd).x0 for t in items)
        xa = ax.transAxes.inverted().transform((x0, 0))[0]
        ax.text(xa, yl, letter, transform=ax.transAxes, fontsize=FS_LETTER, fontweight="bold", va="baseline", ha="left",
                color=S.INK)
    legend(fig, (BOT + AH + 7.2) / H)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(S.FIG, f"{out}.{ext}"), dpi=1000 if ext == "png" else None, bbox_inches="tight", pad_inches=0.03)
    print(f"saved figs/{out}.*")


def figures():
    a = averaged("e4b_comp.csv")
    vals = {}
    for net in TEN[0] + TEN[1]:
        for cls, _, _ in HALF:
            vals[(net, cls)] = comp_of(a, net, cls)
    two_panel([[(n, SHORT[n]) for n in TEN[0]], [(n, SHORT[n]) for n in TEN[1]]], vals, "fig9",
              ["$\\langle e\\rangle<1$", "$\\langle e\\rangle>5.5$"])
    fn = os.path.join(FIN, "e4c_comp.csv")
    if os.path.exists(fn):
        c = averaged("e4c_comp.csv")
        c["series"] = c.net.str.extract(r"S1_(cv\d+)_")[0]; c["comp"] = c.net.str.extract(r"_(homog|asgen|mixed)_")[0]
        g = c[c.component.isin(["hi", "lo", "net_effect"])].groupby(["series", "comp", "channel", "component"]).value
        m, sd = g.mean(), g.std()
        vals = {}
        for ser, _ in COMP_SERIES:
            for comp, _ in COMP:
                for cls, _, _ in HALF:
                    vals[((ser, comp), cls)] = (m[(ser, comp, cls, "hi")], m[(ser, comp, cls, "lo")], m[(ser, comp, cls, "net_effect")],
                                                sd[(ser, comp, cls, "net_effect")])
        two_panel([[((ser, comp), lab) for comp, lab in COMP] for ser, _ in COMP_SERIES], vals, "fig10",
                  [f"$\\mathrm{{CV}}\\approx{cv:g}$" for _, cv in COMP_SERIES])


if __name__ == "__main__":
    S.setup(FS)
    plt.rcParams.update({"axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
                         "xtick.major.size": 3.0, "ytick.major.size": 3.0, "xtick.major.pad": 2.5, "ytick.major.pad": 2.5})
    figures()
