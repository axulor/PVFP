# -*- coding: utf-8 -*-
"""Figure 8 (final, 2026-10-09): the three kinds of games an imitator can observe, drawn in the house style of
figure 1 (code/fig1_model.py): English text, Arial, one text size (node labels and panel letters excepted), LaTeX
(Computer Modern) mathematics.

(A) Imitator i compares its neighbors (candidates) j1, j2, j3; l is a neighbor of j1 but not of i.  The other
    neighbors of j1, j3 and l are not drawn; each of their edges is a short dashed stub, so that every labelled
    degree equals the number of lines at the node.
    1  game j1-l:  only j1 is compared (l is a non-common neighbor)
    2  game j1-j2: both players compared, both cooperate
    3  game j2-j3: both players compared, a cooperator and a defector
(B) How each game's output is split under the two rules (section 2: 1/2, or k_l/(k_j + k_l) under Shapley with
    linear production).  Degrees k_j1 = 6, k_j2 = 3, k_j3 = 6, k_l = 2 give the shares 1/4, 1/3, 2/3.
Colours: games as in figures 11-12 (terracotta / teal / navy); players by strategy (cooperator blue, defector red).
Output: figs/fig8.png / .pdf
2026-10-14: drawn at its display size (full text width, 164.6 mm), so font sizes are the sizes the reader sees:
text 9.5 pt, node labels 10.5 pt, game numbers 8.5 pt bold, panel letters 12 pt bold.  The geometry is in data units
of U mm each."""
import os
from fractions import Fraction
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figs")
FS, FS_NODE, FS_PANEL, FS_TAG = 9.5, 10.5, 12, 8.5
plt.rcParams.update({"font.family": "Arial", "font.size": FS, "mathtext.fontset": "cm", "pdf.fonttype": 42})
LW = 1.4                                          # edges (game edges LW + 0.7)
LW_RING = 0.9                                     # node rings (bar outlines 0.8)
INK, C_NOTE, C_CTX, C_STUB = "#2B2D42", "#6C757D", "#CED4DA", "#ADB5BD"
C_C, C_D, C_OUTSIDE = "#5A7FA8", "#B9605A", "#E9ECEF"
G1, G2, G3 = "#E07A5F", "#3A7D7C", "#3D5A80"              # game kinds 1-3 (as in figures 11-12)
K = dict(j1=6, j2=3, j3=6, l=2)
STUB = (0, (2.2, 2.0))

# layout in millimetres: two equal-aspect panels side by side, U mm per data unit; the canvas is the display width
MM = 1 / 25.4
W = 164.6
XA, XB, YL = (-0.01, 1.22), (0.0, 1.11), (-0.085, 1.12)
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


def rad(k):
    return 0.050 + 0.0065 * k


def node(ax, xy, label, face, r):
    ax.add_patch(Circle(xy, r, facecolor=face, edgecolor="white", linewidth=LW_RING, zorder=5))
    ax.text(xy[0], xy[1], label, ha="center", va="center", color="white", fontsize=FS_NODE, zorder=6)


RT = 0.033                                        # radius of the game-number discs


def tag(ax, xy, n, color):
    """game number in a small white disc with a coloured ring (placed beside its edge, not on it)."""
    ax.add_patch(Circle(xy, RT, facecolor="white", edgecolor=color, linewidth=0.8, zorder=7))
    ax.text(xy[0], xy[1] - 0.001, str(n), ha="center", va="center", color=color, fontsize=FS_TAG, fontweight="bold", zorder=8)


def beside(p, q, t, side, d=0.05):
    """point at fraction t along p -> q, moved a distance d to the left (side = +1) or right (side = -1)."""
    v = q - p; nrm = np.array([-v[1], v[0]]) / np.hypot(*v)
    return p + t * v + side * d * nrm


def stubs(ax, xy, r, angles, length=0.075):
    """short dashed edges to neighbors that are not drawn."""
    for a in np.radians(angles):
        u = np.array([np.cos(a), np.sin(a)])
        p, q = xy + (r + 0.004) * u, xy + (r + length) * u
        ax.add_line(Line2D(*zip(p, q), color=C_STUB, lw=1.0, ls=STUB, zorder=2))


# ------------------------------------------------------------------------------------------------ (A)
ax = axa
P = dict(i=np.array([0.13, 0.50]), j1=np.array([0.58, 0.80]), j2=np.array([0.66, 0.44]), j3=np.array([0.52, 0.17]),
         l=np.array([1.03, 0.96]))
Rn = {n: rad(K[n]) for n in K}
ax.add_patch(FancyBboxPatch((0.36, -0.01), 0.48, 0.95, boxstyle="round,pad=0.0,rounding_size=0.08",
                            facecolor="#F4F6F8", edgecolor="none", zorder=0))
ax.text(0.60, -0.05, "Candidates (neighbors of $i$)", ha="center", va="center", color=C_NOTE)
for j in ("j1", "j2", "j3"):
    ax.add_line(Line2D(*zip(P["i"], P[j]), color=C_CTX, lw=LW, zorder=1, solid_capstyle="round"))
for a, b, col, n, t, side in (("j1", "l", G1, 1, 0.55, -1), ("j1", "j2", G2, 2, 0.5, -1), ("j2", "j3", G3, 3, 0.5, 1)):
    ax.add_line(Line2D(*zip(P[a], P[b]), color=col, lw=LW + 0.7, zorder=2, solid_capstyle="round"))
    tag(ax, beside(P[a], P[b], t, side), n, col)
# undrawn neighbors: k minus the drawn edges (j1: i, j2, l; j2: i, j1, j3; j3: i, j2; l: j1)
stubs(ax, P["j1"], Rn["j1"], (65, 105, 145))
stubs(ax, P["j3"], Rn["j3"], (200, 240, 280, 320), length=0.06)
stubs(ax, P["l"], Rn["l"], (60,))
ax.add_patch(Circle(P["i"], 0.062 + 0.022, facecolor="none", edgecolor=INK, lw=0.7, ls=(0, (2, 2)), zorder=5))
node(ax, P["i"], r"$i$", INK, 0.062)
for n in ("j1", "j2", "j3"):
    node(ax, P[n], rf"$j_{n[1]}$", C_C if n != "j3" else C_D, Rn[n])
node(ax, P["l"], r"$l$", C_C, Rn["l"])
ax.text(P["i"][0], P["i"][1] - 0.13, "Imitator", ha="center", va="center", color=C_NOTE)
ax.text(P["j1"][0] + 0.72 * Rn["j1"] + 0.025, P["j1"][1] - 0.72 * Rn["j1"] - 0.005, rf"$k={K['j1']}$", ha="left", va="top", color=INK)
ax.text(P["j2"][0] - 0.055, P["j2"][1] + Rn["j2"] - 0.012, rf"$k={K['j2']}$", ha="right", va="bottom", color=INK)
ax.text(P["j3"][0] + Rn["j3"] + 0.035, P["j3"][1] + 0.005, rf"$k={K['j3']}$", ha="left", va="center", color=INK)
ax.text(P["l"][0], P["l"][1] - Rn["l"] - 0.055, rf"$k={K['l']}$", ha="center", va="center", color=INK)
# strategy key
for y, col, lab in ((0.30, C_C, "Cooperator"), (0.22, C_D, "Defector")):
    ax.add_patch(Circle((0.89, y), 0.025, facecolor=col, edgecolor="white", lw=0.6, zorder=5))
    ax.text(0.935, y, lab, ha="left", va="center", color=INK)
ax.text(XA[0], YL[1], "A", fontsize=FS_PANEL, fontweight="bold", va="top", ha="left", color=INK)

# ------------------------------------------------------------------------------------------------ (B)
ax = axb
X0, X1, HB = 0.30, 0.86, 0.07
XC = 0.5 * (X0 + X1)


def shapley(kj, kl):
    return Fraction(kl, kj + kl)                      # share of the first player


def frac(f):
    return rf"${f.numerator}/{f.denominator}$"


games = [  # (colour, number, left player, right player, verdict)
    (G1, 1, "j1", "l", "favors\nUniform"),
    (G2, 2, "j1", "j2", "favors\nneither"),
    (G3, 3, "j2", "j3", "favors\nShapley"),
]
COL = dict(j1=C_C, j2=C_C, j3=C_D, l=C_OUTSIDE)
NAME = dict(j1=r"$j_1$", j2=r"$j_2$", j3=r"$j_3$", l=r"$l$")
fig.canvas.draw(); rend = fig.canvas.get_renderer()
for r_, (col, n, pl, pr, verdict) in enumerate(games):
    y0 = 1.04 - 0.41 * r_          # 2026-10-14: wider gap between games, tighter within a game
    head = ax.text(XC, y0, rf"Game {NAME[pl]}–{NAME[pr]}", ha="center", va="center", color=col)
    bb = head.get_window_extent(renderer=rend).transformed(ax.transData.inverted())
    tag(ax, (bb.x0 - 0.05, y0), n, col)
    ax.text(X0, y0 - 0.07, rf"to {NAME[pl]}", ha="left", va="center", color=INK)
    ax.text(X1, y0 - 0.07, rf"to {NAME[pr]}", ha="right", va="center", color=C_NOTE if pr == "l" else INK)
    m_sh = shapley(K[pl], K[pr])
    for q, (rule, m) in enumerate((("Uniform", Fraction(1, 2)), ("Shapley", m_sh))):
        y = y0 - 0.145 - 0.09 * q
        xm = X0 + float(m) * (X1 - X0)
        ax.add_patch(Rectangle((X0, y - HB / 2), xm - X0, HB, facecolor=COL[pl], edgecolor="white", lw=0.8, zorder=3))
        ax.add_patch(Rectangle((xm, y - HB / 2), X1 - xm, HB, facecolor=COL[pr], edgecolor="white", lw=0.8, zorder=3))
        ax.text((X0 + xm) / 2, y, frac(m), ha="center", va="center", color="white", zorder=4)
        ax.text((xm + X1) / 2, y, frac(1 - m), ha="center", va="center", color=INK if pr == "l" else "white", zorder=4)
        ax.text(X0 - 0.025, y, rule, ha="right", va="center", color=INK)
    ax.text(X1 + 0.05, y0 - 0.19, verdict, ha="left", va="center", color=col, linespacing=1.1)
ax.text(XB[0], YL[1], "B", fontsize=FS_PANEL, fontweight="bold", va="top", ha="left", color=INK)

os.makedirs(OUT, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig8.{ext}"), dpi=1000)
from matplotlib.transforms import Bbox
bb = Bbox.union([t.get_window_extent(rend) for a_ in (axa, axb) for t in a_.texts + a_.patches + a_.lines])   # all ink
print("saved fig8: %.1f x %.1f mm, U = %.2f mm; ink extends %.1f..%.1f x %.1f..%.1f mm"
      % (W, H, U, *(np.array([bb.x0, bb.x1, bb.y0, bb.y1]) / fig.dpi * 25.4)))
