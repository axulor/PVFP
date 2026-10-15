# -*- coding: utf-8 -*-
"""Figure 1: the model in two stages.

(A) Stage 1, on every edge: each individual splits its endowment c equally over its k edges; a cooperator puts its
    share into the edge's pool, the pool produces b g(t), and the output is split between the two by the allocation
    rule.  Example: k_i = 3, k_j = 6 (linear production): uniform 1/2 : 1/2, Shapley 2/3 : 1/3.  The allocation bars
    are coloured by recipient (i dark, j blue), like the nodes.
(B) Stage 2, imitation: the imitator i compares candidates by the payoffs it can see; for candidate j these are the
    direct payoff (weight 1), the payoffs from common neighbors l1, l2 (weight eps_1) and from non-common neighbors
    l3-l5 (weight eps_2).
House style: English text, sans-serif (Arial), one text size (node labels and panel letters excepted), LaTeX
(Computer Modern) mathematics.
2026-10-14: drawn at its display size (full text width, 164.6 mm), so font sizes are the sizes the reader sees:
text 9.5 pt, node labels 10.5 pt, panel letters 12 pt bold.  The geometry is in data units of U mm each.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs")
FS, FS_NODE, FS_PANEL = 9.5, 10.5, 12
plt.rcParams.update({"font.family": "Arial", "font.size": FS, "mathtext.fontset": "cm", "pdf.fonttype": 42})
LW = 1.4                                          # edges
LW_RING = 0.9                                     # node rings and the pool box (bar and pie outlines 0.8)

INK = "#2B2D42"
C_I, C_J = "#2B2D42", "#3D5A80"                   # imitator / candidate
C_IN, C_OUT = "#3A7D7C", "#C8553D"                # common (teal-green) / non-common (brick red) neighbors, 2026-10-08
C_CTX, C_SLICE, C_NOTE = "#CED4DA", "#E9ECEF", "#6C757D"
NODE_FILL, NODE_EDGE = "#F1F3F5", "#ADB5BD"
DASH = (0, (4.5, 2.5))
R = 0.062

# layout in millimetres: two equal-aspect panels side by side, U mm per data unit; the canvas is the display width
MM = 1 / 25.4
W = 164.6
XA, XB, YL = (-0.01, 1.25), (0.0, 1.225), (-0.05, 1.16)
ML, GAP, MB = 0.5, 2.0, 0.5
U = (W - 2 * ML - GAP) / ((XA[1] - XA[0]) + (XB[1] - XB[0]))
wA, wB, hP = (XA[1] - XA[0]) * U, (XB[1] - XB[0]) * U, (YL[1] - YL[0]) * U
H = MB + hP + MB
fig = plt.figure(figsize=(W * MM, H * MM))
axa = fig.add_axes([ML / W, MB / H, wA / W, hP / H])
axb = fig.add_axes([(ML + wA + GAP) / W, MB / H, wB / W, hP / H])
for ax, xl in ((axa, XA), (axb, XB)):
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(*xl); ax.set_ylim(*YL)


def node(ax, xy, label, face, ring, tc, r=R):
    ax.add_patch(Circle(xy, r, facecolor=face, edgecolor=ring, linewidth=LW_RING, zorder=5))
    ax.text(xy[0], xy[1], label, ha="center", va="center", color=tc, fontsize=FS_NODE, zorder=6)


def arrow(ax, p, q, color, rad=0.0, lw=1.0, ms=9, z=4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=1.5, shrinkB=1.5))


# ------------------------------------------------------------------------------------------------ (A) stage 1
ax = axa
I, J = np.array([0.16, 0.60]), np.array([1.06, 0.60])
POOL = np.array([0.61, 0.60])
BW, BH = 0.42, 0.19                               # pool box
ax.add_line(Line2D([I[0], J[0]], [I[1], I[1]], color=C_CTX, lw=LW, zorder=1))
ax.add_patch(FancyBboxPatch((POOL[0] - BW / 2, POOL[1] - BH / 2), BW, BH, boxstyle="round,pad=0.01,rounding_size=0.04",
                            facecolor="#EEF2F6", edgecolor=INK, lw=LW_RING, zorder=3))
ax.text(POOL[0], POOL[1] + 0.042, "Public pool", ha="center", va="center", color=INK, zorder=4)
ax.text(POOL[0], POOL[1] - 0.042, r"$t=p_{ij}s_i+p_{ji}s_j$", ha="center", va="center", color=INK, zorder=4)
node(ax, I, r"$i$", C_I, "white", "white")
node(ax, J, r"$j$", C_J, "white", "white")
ax.text(I[0], I[1] - 0.125, r"$k_i=3$", ha="center", va="center")
ax.text(J[0], J[1] - 0.125, r"$k_j=6$", ha="center", va="center")

# endowment pies (same size as the nodes): c split equally over the k edges; the slice for edge {i,j} is coloured
PI, PJ = np.array([0.16, 0.90]), np.array([1.06, 0.90])
for centre, k, col in ((PI, 3, C_I), (PJ, 6, C_J)):
    for s in range(k):
        a0 = 90 - 360 * (s + 1) / k; a1 = 90 - 360 * s / k
        ax.add_patch(Wedge(centre, R, a0, a1, facecolor=col if s == 0 else C_SLICE, edgecolor="white", lw=0.8, zorder=3))
    ax.text(centre[0], centre[1] + R + 0.05, r"Endowment $c$", ha="center", va="center")
arrow(ax, (PI[0] + R + 0.005, PI[1] + 0.01), (POOL[0] - 0.15, POOL[1] + BH / 2 + 0.01), C_I, rad=-0.25)
arrow(ax, (PJ[0] - R - 0.005, PJ[1] + 0.01), (POOL[0] + 0.15, POOL[1] + BH / 2 + 0.01), C_J, rad=0.25)
ax.text(0.34, 0.95, r"$p_{ij}c=c/3$", ha="left", va="center", color=C_I)
ax.text(0.88, 0.95, r"$p_{ji}c=c/6$", ha="right", va="center", color=C_J)
ax.text(POOL[0], 0.80, "if cooperating", ha="center", va="center", color=C_NOTE, style="italic")

# output and allocation (bars coloured by recipient)
arrow(ax, (POOL[0], POOL[1] - BH / 2 - 0.01), (POOL[0], 0.39), INK, ms=8.5)
ax.text(POOL[0] + 0.02, 0.445, r"Output $b\,g(t)$", ha="left", va="center")
X0, X1, HB = 0.40, 0.82, 0.075
for y, lab, m, txt in ((0.31, "Uniform", 0.5, (r"$1/2$", r"$1/2$")), (0.195, "Shapley", 2 / 3, (r"$2/3$", r"$1/3$"))):
    xm = X0 + m * (X1 - X0)
    ax.add_patch(Rectangle((X0, y - HB / 2), xm - X0, HB, facecolor=C_I, edgecolor="white", lw=0.8, zorder=3))
    ax.add_patch(Rectangle((xm, y - HB / 2), X1 - xm, HB, facecolor=C_J, edgecolor="white", lw=0.8, zorder=3))
    ax.text((X0 + xm) / 2, y, txt[0], ha="center", va="center", color="white", zorder=4)
    ax.text((xm + X1) / 2, y, txt[1], ha="center", va="center", color="white", zorder=4)
    ax.text(X0 - 0.025, y, lab, ha="right", va="center", color=INK)
ax.text(X0, 0.105, r"to $i$", ha="left", va="center", color=C_I)
ax.text(X1, 0.105, r"to $j$", ha="right", va="center", color=C_J)
ax.text(POOL[0], 0.0, r"Shapley ($\delta=1$): $m_{ij}=k_j/(k_i+k_j)$", ha="center", va="center", color=INK)
ax.text(XA[0], YL[1], "A", fontsize=FS_PANEL, fontweight="bold", va="top", ha="left", color=INK)

# ------------------------------------------------------------------------------------------------ (B) stage 2
ax = axb
off = np.array([0.08, 0.10])
I, J = np.array([0.15, 0.50]) + off, np.array([0.55, 0.50]) + off
L = {1: np.array([0.35, 0.86]) + off, 2: np.array([0.35, 0.14]) + off, 3: np.array([0.90, 0.83]) + off,
     4: np.array([0.97, 0.50]) + off, 5: np.array([0.90, 0.17]) + off}


def edge(p, q, color, ls="-", z=2):
    ax.add_line(Line2D([p[0], q[0]], [p[1], q[1]], color=color, lw=LW, ls=ls, zorder=z, solid_capstyle="round"))


def edge_label(p, q, text, color, side=1, d=0.042, t=0.5):
    v = q - p; n = np.array([-v[1], v[0]]) / np.hypot(*v)
    x, y = p + t * v + side * d * n
    ax.text(x, y, text, ha="center", va="center", color=color, zorder=7)


for q in (L[1], L[2]):
    edge(I, q, C_CTX)
edge(I, J, INK, z=3)
for k in (1, 2):
    edge(J, L[k], C_IN, z=3)
for k in (3, 4, 5):
    edge(J, L[k], C_OUT, ls=DASH, z=3)
ax.add_patch(Circle(I, R + 0.022, facecolor="none", edgecolor=INK, lw=0.7, ls=(0, (2, 2)), zorder=5))
node(ax, I, r"$i$", C_I, "white", "white")
node(ax, J, r"$j$", C_J, "white", "white")
for k, xy in L.items():
    node(ax, xy, rf"$l_{k}$", NODE_FILL, NODE_EDGE, INK)
ax.text(I[0], I[1] - 0.125, "Imitator", ha="center", va="center", color=C_NOTE, bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none"), zorder=6)
ax.text(J[0] + 0.02, J[1] - 0.125, "Candidate", ha="center", va="center", color=C_NOTE, bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none"), zorder=6)
edge_label(I, J, r"$u_{ji}$", INK, side=1, d=0.05)
edge_label(J, L[1], r"$u_{jl_1}$", C_IN, side=-1, d=0.065, t=0.7)
edge_label(J, L[2], r"$u_{jl_2}$", C_IN, side=1, d=0.065, t=0.7)
edge_label(J, L[3], r"$u_{jl_3}$", C_OUT, side=1, d=0.06, t=0.7)
edge_label(J, L[4], r"$u_{jl_4}$", C_OUT, side=1, d=0.05, t=0.45)
edge_label(J, L[5], r"$u_{jl_5}$", C_OUT, side=-1, d=0.06, t=0.7)
ax.text(L[1][0] - 0.10, L[1][1], "Common\nneighbors", ha="right", va="center", color=C_IN)
ax.text(L[3][0] - 0.05, L[3][1] + 0.115, "Non-common neighbors", ha="center", va="center", color=C_OUT)
ax.text(XB[0], YL[1], "B", fontsize=FS_PANEL, fontweight="bold", va="top", ha="left", color=INK)

# formulas below panel B, each term coloured like its edges
fig.canvas.draw()
rend = fig.canvas.get_renderer()


def centred_pieces(pieces, y, xc):
    texts = [ax.text(0, y, s, color=c, ha="left", va="center") for s, c in pieces]
    fig.canvas.draw()
    widths = [t.get_window_extent(renderer=rend).width for t in texts]
    xd0, yd = ax.transData.transform((xc, y))
    gap = 0.28 * FS * fig.dpi / 72
    x = xd0 - (sum(widths) + gap * (len(widths) - 1)) / 2
    inv = ax.transData.inverted()
    for t, w in zip(texts, widths):
        t.set_x(inv.transform((x, yd))[0]); x += w + gap


centred_pieces([(r"$v_{j\to i}=u_{ji}$", INK), (r"$+\,\varepsilon_1\,(u_{jl_1}+u_{jl_2})$", C_IN),
                (r"$+\,\varepsilon_2\,(u_{jl_3}+u_{jl_4}+u_{jl_5})$", C_OUT)], 0.09, 0.61)
centred_pieces([(r"$i$ copies $j$ with probability $P_{i\leftarrow j}\propto\exp(\theta\,v_{j\to i})$", INK)], 0.0, 0.61)

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig1.{ext}"), dpi=1000)
from matplotlib.transforms import Bbox
bb = Bbox.union([t.get_window_extent(rend) for a_ in (axa, axb) for t in a_.texts + a_.patches])   # all ink
print("saved fig1: %.1f x %.1f mm, U = %.2f mm; ink extends %.1f..%.1f x %.1f..%.1f mm"
      % (W, H, U, *(np.array([bb.x0, bb.x1, bb.y0, bb.y1]) / fig.dpi * 25.4)))
