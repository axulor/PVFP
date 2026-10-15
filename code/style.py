# -*- coding: utf-8 -*-
"""style.py -- house style for every figure of the paper (see 3-仿真设计稿.md, 3.4).

Text: Arial (sans-serif), one size per figure; mathematics: LaTeX Computer Modern (mathtext 'cm').
Markers: synthetic networks o, real higher-order networks *, real pairwise networks D, classic (karate) ^.
Conflict shading: Lambda > 0 light purple, Lambda < 0 light orange.
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "figs")
os.makedirs(FIG, exist_ok=True)
RES = os.path.join(ROOT, "results")
MM = 1 / 25.4

INK = "#2B2D42"
C_U, C_SH = "#E07A5F", "#3D5A80"                 # uniform / Shapley allocation (2026-10-07: terracotta / navy)
C_IN, FILL_IN, C_OUT = "#3A7D7C", "#DDEFEC", "#C8553D"   # 2026-10-08: common neighbors green (no conflict), non-common brick red (conflict)
BG_CONFLICT, BG_NOCONFLICT = "#F7F3FA", "#FEF8F1"   # 2026-10-07: the original light purple / light orange, made lighter
GREY = "#A9B1BA"
CMAP_SEQ = LinearSegmentedColormap.from_list("inkseq", ["#DCE3EA", "#8FA3B8", "#3D5A80", "#1B2A3A"])
CMAP_ALT = LinearSegmentedColormap.from_list("amber", ["#F6E7C8", "#E3A857", "#B5651D", "#5C2E0E"])
CMAP_TEAL = LinearSegmentedColormap.from_list("teal", ["#DDEFEC", "#8CC5BE", "#3A7D7C", "#173F3E"])


def inset_cbar(fig, ax, cmap, norm, rect, label, ticks=None):
    """small horizontal colour bar inside an axes (rect in axes fraction)."""
    import matplotlib.pyplot as plt
    cax = ax.inset_axes(rect)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=ticks)
    cb.set_label(label, labelpad=1, fontsize=6.5); cb.ax.tick_params(labelsize=6, length=1.5, pad=1)
    cb.ax.xaxis.set_label_position("top"); cb.outline.set_linewidth(0.4)
    return cb
MARK = {"synthetic": "o", "real-HO": "*", "real-pair": "D", "classic": "^"}
MSIZE = {"synthetic": 14, "real-HO": 70, "real-pair": 26, "classic": 30}
LABEL = {"CPH": "CPH", "CPH-FB": "CPH-FB", "UTM": "UTM", "UTE": "UTE", "SFHH": "SFHH", "EEU": "EEU",
         "URV": "URV", "ESH": "ESH", "KAR": "Karate"}


def setup(fs=7.0):
    mpl.rcParams.update({
        "font.family": "Arial", "font.size": fs, "axes.titlesize": fs, "axes.labelsize": fs,
        "xtick.labelsize": fs, "ytick.labelsize": fs, "legend.fontsize": fs,
        "mathtext.fontset": "cm", "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "axes.spines.top": True, "axes.spines.right": True,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "lines.linewidth": 1.1, "legend.frameon": False, "savefig.dpi": 600})


def panel(ax, letter, x=-0.16, y=1.04, fs=9):
    ax.text(x, y, letter.upper(), transform=ax.transAxes, fontsize=fs, fontweight="bold", va="bottom", ha="left", color=INK)


def conflict_bg(ax, horizontal=True):
    """shade Lambda > 0 (conflict) and Lambda < 0 (no conflict) halves of an axis whose y is Lambda."""
    lo, hi = ax.get_ylim()
    ax.axhspan(0, max(hi, 0), color=BG_CONFLICT, zorder=0, lw=0)
    ax.axhspan(min(lo, 0), 0, color=BG_NOCONFLICT, zorder=0, lw=0)
    ax.axhline(0, color=INK, lw=0.6, zorder=1)
    ax.set_ylim(lo, hi)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=600 if ext == "png" else None, bbox_inches="tight")
    print("saved", name)
