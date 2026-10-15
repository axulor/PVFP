# -*- coding: utf-8 -*-
"""figB1_omega_eta.py -- Fig. B1 (appendix B.1): the share omega of the Shapley rule coefficient a_Sh that comes from
pools between two candidates (classes 2 and 3), against the relative embeddedness eta, on the 40 networks.
Networks sorted by eta; bar = eta (value printed at right), circle = omega.
Data: results/final/mechanism/lead_predict.csv (D_p11 = omega; from lead_decomp.py / lead_predict.py).
Run:  python figB1_omega_eta.py"""
import os, re
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import style as S

FIN = os.path.join(S.ROOT, "results", "final")
C_BAR, C_BAR_E, C_VAL = "#DCE6EE", "#8FA3B8", "#5A6270"
W_MM = 88                                               # canvas width (mm) before bbox_inches="tight"


def label(n):
    for pat, fmt in ((r"(ER|RGG)_k(\d+)$", "{0} $k={1}$"), (r"BA_m(\d+)$", "BA $m={0}$"), (r"CL_g([\d.]+)$", "CL $\\gamma={0}$"),
                     (r"WS_p([\d.]+)$", "WS $p={0}$"), (r"HK_m(\d+)_p([\d.]+)$", "HK {0}/{1}"), (r"RHG_g([\d.]+)_T([\d.]+)$", "RHG {0}/{1}")):
        m = re.match(pat, n)
        if m:
            return fmt.format(*m.groups())
    return n


def main():
    d = pd.read_csv(os.path.join(FIN, "mechanism", "lead_predict.csv")).sort_values("eta").reset_index(drop=True)
    # 2026-10-14: drawn at its printed width (0.6 textwidth = 98.8 mm), so the sizes below are what the reader sees.
    # The eta values (Computer Modern digits) are set 0.5 pt above the Arial names so that both look the same size;
    # the single Greek letters of the legend and of the eta header are set larger for the same reason.
    FS_NAME, FS_VAL, FS_TICK, FS_LAB, FS_LEG, FS_HEAD = 8.5, 9.0, 9.5, 10.5, 11.5, 11.5
    PITCH = 3.4                                         # row pitch (mm)
    S.setup(FS_NAME)
    fig, ax = plt.subplots(figsize=(W_MM * S.MM, (len(d) + 0.4) * PITCH / 0.77 * S.MM))   # axes = 77 % of the height
    y = np.arange(len(d))
    for t in (0.2, 0.4, 0.6):
        ax.axvline(t, color="#ECEEF1", lw=0.5, zorder=0)
    ax.barh(y, d.eta, height=0.62, color=C_BAR, edgecolor=C_BAR_E, lw=0.5, zorder=2)
    ax.scatter(d.D_p11, y, marker="o", s=26, color=S.C_IN, edgecolors=S.INK, linewidths=0.45, zorder=4)
    ax.set_yticks(y); ax.set_yticklabels([label(n) for n in d.net]); ax.tick_params(axis="y", length=0, labelsize=FS_NAME)
    ax.set_xlim(0, 0.7); ax.set_ylim(-0.7, len(d) - 0.3)
    ax.set_xticks([0, 0.2, 0.4, 0.6]); ax.set_xticklabels(["$0$", "$0.2$", "$0.4$", "$0.6$"]); ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.set_xlabel("proportion", fontsize=FS_LAB)
    tr = ax.get_yaxis_transform()                       # x in axes fraction, y in data
    for yi, e in enumerate(d.eta):
        ax.text(1.03, yi, f"${e:.2f}$", transform=tr, va="center", ha="left", fontsize=FS_VAL, color=C_VAL)
    ax.text(1.03, len(d) - 0.1, "$\\eta$", transform=tr, va="bottom", ha="left", fontsize=FS_HEAD, color=C_VAL)

    ax.legend(handles=[Patch(fc=C_BAR, ec=C_BAR_E, lw=0.5, label="$\\eta$"),
                       Line2D([], [], ls="", marker="o", mfc=S.C_IN, mec=S.INK, mew=0.45, ms=5.1, label="$\\omega$")],
              loc="lower right", bbox_to_anchor=(0.97, 0.04), ncol=1, frameon=False, fontsize=FS_LEG,
              handlelength=1.3, handleheight=0.55, labelspacing=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(S.FIG, f"figB1.{ext}"), dpi=1000 if ext == "png" else None, bbox_inches="tight")
    print("saved figB1")


if __name__ == "__main__":
    main()
