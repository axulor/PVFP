# -*- coding: utf-8 -*-
"""edge_map.py -- figures 11-12: where in a small network the rule difference is made and where it is weakened.
Networks: CONFIGS (an S1 network for figure 12, a Holme-Kim network for figure 11) and their rewired versions
(family(): all / half of the triangles removed, most homogeneous / most mixed triangles; pendant edges never move).

Cooperator invasions under rule `proc`, visibility (1, 1), theta = 1, b = mean of the two critical benefits of the
network (estimated here).  At every moment (imitator i with both C and D neighbours) the Uniform - Shapley lead
    sum over candidates j of w_j * sum over pools (j, l) of t(j, l),   w_j = 1/n_C (j cooperates) or -1/n_D,
    t(j, l) = (1/2 - m_jl) b (s_j/k_j + s_l/k_l)  (Uniform - Shapley share of the pool output to j)
is attributed to the pool (edge) {j, l}, separately for
    one-sided views (l not a neighbour of i: only j is compared),
    two-sided views (l a neighbour of i: both members of the pool are compared),
    direct pools (l = i).
Also "cc": the output of two-sided pools between two cooperating candidates, counted for the cooperators
(O / n_C per pool; the same under both rules -- it dilutes the rule difference).
Summed over edges this is the E4b net effect (common = two-sided, non-common = one-sided).
Per edge: average over moments (equal weight), both rules' processes averaged.
Output: results/final/small_nets/edge_map_<name>.csv; b* from small_bstar.py (results/final/small_nets/small_bstar.csv).
Run:  python small_bstar.py hk6 s7  then  python edge_map.py hk6 ; python edge_map.py s7 ; python fig11_12_maps.py"""
import os, sys, time
import numpy as np, pandas as pd, networkx as nx
import numba as nb
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kernel import _seed, _unif, Complex
from exp_common import triangles
from bstar import Estimator
import networks_s1 as NS
import networks40 as N40

EX = os.path.join(N40.ROOT, "results", "final", "small_nets"); os.makedirs(EX, exist_ok=True)


@nb.njit(parallel=True)
def _edge_runs(indptr, nbr, rev, U, kd, b, runs, seed, nch):
    N = len(indptr) - 1; M = len(nbr)
    acc = np.zeros((nch, 4, M)); mom = np.zeros(nch)
    per = (runs + nch - 1) // nch
    kmax = 0
    for x in range(N):
        kmax = max(kmax, indptr[x + 1] - indptr[x])
    for ch in nb.prange(nch):
        s = np.empty(N, np.int64); u = np.empty(N); dc = np.empty(N, np.int64)
        act = np.empty(N, np.int64); pos = np.empty(N, np.int64); w = np.empty(kmax)
        mark = np.zeros(N, np.int64)
        rstate = np.zeros(1, np.uint64)
        for run in range(ch * per, min(runs, (ch + 1) * per)):
            _seed(rstate, seed, run)
            for x in range(N):
                s[x] = 0
            s[run % N] = 1
            for x in range(N):
                uu = 0.0
                for a in range(indptr[x], indptr[x + 1]):
                    uu += U[a, 2 * s[x] + s[nbr[a]]]
                u[x] = uu
            nact = 0
            for x in range(N):
                c = 0
                for a in range(indptr[x], indptr[x + 1]):
                    if s[nbr[a]] != s[x]:
                        c += 1
                dc[x] = c
                if c > 0:
                    pos[x] = nact; act[nact] = x; nact += 1
                else:
                    pos[x] = -1
            while nact > 0:
                i = act[int(_unif(rstate) * nact)]
                si = s[i]; a0, a1 = indptr[i], indptr[i + 1]; ki = a1 - a0
                nc = 0
                for a in range(a0, a1):
                    nc += s[nbr[a]]
                if 0 < nc < ki:
                    nd = ki - nc
                    for a in range(a0, a1):
                        mark[nbr[a]] = 1
                    for a in range(a0, a1):
                        j = nbr[a]; kj = kd[j]; sj = s[j]
                        wj = 1.0 / nc if sj == 1 else -1.0 / nd
                        for e in range(indptr[j], indptr[j + 1]):
                            l = nbr[e]; kl = kd[l]
                            t = (0.5 - kl / (kj + kl)) * b * (sj / kj + s[l] / kl)
                            if l == i:
                                acc[ch, 2, e] += wj * t
                            elif mark[l] == 1:
                                acc[ch, 1, e] += wj * t
                                if sj == 1 and s[l] == 1:                # two cooperators: j's half of the output
                                    acc[ch, 3, e] += wj * 0.5 * b * (1.0 / kj + 1.0 / kl)
                            else:
                                acc[ch, 0, e] += wj * t
                    for a in range(a0, a1):
                        mark[nbr[a]] = 0
                    mom[ch] += 1.0
                vmax = -1e300
                for a in range(a0, a1):
                    w[a - a0] = u[nbr[a]]                              # (1, 1)
                    if w[a - a0] > vmax:
                        vmax = w[a - a0]
                tot = 0.0
                for t_ in range(ki):
                    w[t_] = np.exp(w[t_] - vmax); tot += w[t_]
                r = _unif(rstate) * tot; accw = 0.0; pick = a1 - 1
                for t_ in range(ki):
                    accw += w[t_]
                    if r < accw:
                        pick = a0 + t_; break
                ns = s[nbr[pick]]
                if ns == si:
                    continue
                s[i] = ns
                ui = 0.0
                for a in range(a0, a1):
                    y = nbr[a]; sy = s[y]
                    ui += U[a, 2 * ns + sy]
                    e = rev[a]; u[y] += U[e, 2 * sy + ns] - U[e, 2 * sy + si]
                    if sy == ns:
                        dc[y] -= 1
                    else:
                        dc[y] += 1
                    if dc[y] > 0 and pos[y] < 0:
                        pos[y] = nact; act[nact] = y; nact += 1
                    elif dc[y] == 0 and pos[y] >= 0:
                        pp = pos[y]; last = act[nact - 1]; act[pp] = last; pos[last] = pp; pos[y] = -1; nact -= 1
                u[i] = ui
                dc[i] = ki - dc[i]
                if dc[i] > 0 and pos[i] < 0:
                    pos[i] = nact; act[nact] = i; nact += 1
                elif dc[i] == 0 and pos[i] >= 0:
                    pp = pos[i]; last = act[nact - 1]; act[pp] = last; pos[last] = pp; pos[i] = -1; nact -= 1
    return acc.sum(0), mom.sum()


def bstar_pair(G, name, target=0.01):
    """b* of both rules at (1,1): the precise values of small_bstar.py if present, else a 1% estimate cached by name."""
    f0 = os.path.join(EX, "small_bstar.csv")
    if os.path.exists(f0):
        d = pd.read_csv(f0)
        d = d[(d.name == name) & (d.eps1 == 1.0) & (d.eps2 == 1.0)].dropna(subset=["bstar"]).groupby("rule").bstar.first()
        if {"U", "Sh"} <= set(d.index):
            return float(d.U), float(d.Sh)
    f = os.path.join(EX, "edge_map_bstar.csv")
    d = pd.read_csv(f) if os.path.exists(f) else pd.DataFrame(columns=["name", "U", "Sh"])
    if name in set(d.name):
        r = d[d.name == name].iloc[0]; return float(r.U), float(r.Sh)
    cx = Complex(G, triangles(G))
    out = [Estimator(cx, rule, 1.0, 1.0, 1.0, seed=3).estimate(target=target)["bstar"] for rule in ("U", "Sh")]
    pd.DataFrame([dict(name=name, U=out[0], Sh=out[1])]).to_csv(f, mode="a", header=not os.path.exists(f), index=False)
    return tuple(out)


def edge_map(G, name, runs=200000, b=None):
    """per undirected edge: one-sided, two-sided and direct parts of the Uniform - Shapley lead (both processes averaged)."""
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    if b is None:
        bU, bS = bstar_pair(G, name); b = 0.5 * (bU + bS)
    cx = Complex(G, triangles(G))
    tot = np.zeros((4, len(cx.nbr)))
    for proc in ("U", "Sh"):
        a, m = _edge_runs(cx.indptr, cx.nbr, cx.rev, cx.table(b, proc, 1.0), cx.k.copy(), b, runs, 17, nb.get_num_threads() * 4)
        tot += a / m / 2
    src = np.repeat(np.arange(cx.N), np.diff(cx.indptr))
    rows = {}
    for e in range(len(cx.nbr)):
        u, v = int(src[e]), int(cx.nbr[e]); key = (min(u, v), max(u, v))
        r = rows.setdefault(key, dict(u=key[0], v=key[1], one=0.0, two=0.0, direct=0.0, cc=0.0))
        r["one"] += tot[0, e]; r["two"] += tot[1, e]; r["direct"] += tot[2, e]; r["cc"] += tot[3, e]
    d = pd.DataFrame(list(rows.values()))
    k = dict(G.degree()); nbs = {x: set(G[x]) for x in G}
    d["ku"] = d.u.map(k); d["kv"] = d.v.map(k)
    d["tri"] = [len(nbs[a] & nbs[c]) for a, c in zip(d.u, d.v)]
    d["b"] = b
    d.to_csv(os.path.join(EX, f"edge_map_{name}.csv"), index=False)
    return d, b


def _leafless_propose(C):
    """one proposal of C (a Composition) that does not touch pendant edges."""
    while True:
        st = C.rng.bit_generator.state
        n = len(C.edges)
        i1, i2 = C.rng.integers(n), C.rng.integers(n)
        a, b = C.edges[i1]; c, d = C.edges[i2]
        if min(C.k[a], C.k[b], C.k[c], C.k[d]) > 1:
            C.rng.bit_generator.state = st
            return NS.Composition.propose(C)


def detriangle(G, seed=0, steps=40000, check_every=20, keep_leaves=True):
    """degree-preserving swaps that never increase the triangle count (greedy, connectivity kept)."""
    C = NS.Composition(G, seed=seed)
    if keep_leaves:
        C.propose = lambda: _leafless_propose(C)
    good = (list(C.edges), {x: set(y) for x, y in C.nb.items()}, C.T, C.S, C.Q); acc = 0
    for it in range(steps):
        p = C.propose()
        if p is None:
            continue
        dd, rec = p
        if dd[0] > 0 or (dd[0] == 0 and C.rng.random() < 0.5):
            C.undo(rec); continue
        C.commit(rec, dd); acc += 1
        if acc % check_every == 0:
            if not nx.is_connected(C.graph()):
                C.edges, C.nb, C.T, C.S, C.Q = (list(good[0]), {x: set(y) for x, y in good[1].items()}, good[2], good[3], good[4])
                continue
            good = (list(C.edges), {x: set(y) for x, y in C.nb.items()}, C.T, C.S, C.Q)
        if C.T == 0:
            break
    H = C.graph()
    if not nx.is_connected(H):
        C.edges = list(good[0]); H = C.graph()
    return H


def reduce_eta(G, target, seed=0, keep_leaves=True):
    """greedy triangle-removing swaps (as detriangle) in small blocks until eta <= target."""
    H = G.copy(); it = 0
    while NS.eta(H) > target and it < 4000:
        H = detriangle(H, seed=seed + it, steps=8, check_every=4, keep_leaves=keep_leaves); it += 1
    return H


def variants(G0, seed=5, comp_steps=60000, keep_leaves=True, mid_eta=None):
    """rewired versions of G0 with the same degree sequence: triangles removed (low), down to mid_eta (mid), and the
    two composition extremes at the eta of G0 (triangle count within 3%, eta within 0.01); pendant edges stay put."""
    out = {"gen": G0}
    out["low"] = detriangle(G0, seed=seed, keep_leaves=keep_leaves)
    for extra in range(1, 20):                                       # insist on no triangle at all
        if NS.n_triangles(out["low"]) == 0:
            break
        out["low"] = detriangle(out["low"], seed=seed + 100 * extra, steps=200000, keep_leaves=keep_leaves)
    if mid_eta is not None:
        out["mid"] = reduce_eta(G0, mid_eta, seed=seed, keep_leaves=keep_leaves)
    T0, e0 = NS.n_triangles(G0), NS.eta(G0)
    for nm, dr in (("homog", -1), ("mixed", +1)):
        C = NS.Composition(G0, seed=seed)
        if keep_leaves:
            C.propose = (lambda C_: (lambda: _leafless_propose(C_)))(C)
        C.run(dr, comp_steps, T0, e0, tol=0.03, eta_tol=0.01, check_every=20)
        out[nm] = C.graph()
    return out


def family(cvh=0.9, beta=4.0, seed=5, N=42, km=5.0, comp_steps=60000, keep_leaves=True, mid_eta=None, graph=None):
    """one small network and its variants: an S1 network (hidden angles returned for the layout), or a named
    network of the main set (graph="KAR"; no angles)."""
    if graph is not None and graph.startswith("HK:"):                  # Holme-Kim: "HK:n:m:p:seed"
        n_, m_, p_, s_ = graph.split(":")[1:]
        G0 = nx.powerlaw_cluster_graph(int(n_), int(m_), float(p_), seed=int(s_)); th = None
    elif graph is not None:
        G0 = nx.convert_node_labels_to_integers(N40.build(graph), ordering="sorted"); th = None
    else:
        G0, th = NS.build_s1(cvh, beta, seed, N=N, km=km, angles=True)
    return variants(G0, seed=seed, comp_steps=comp_steps, keep_leaves=keep_leaves, mid_eta=mid_eta), th


CONFIGS = {"s7": dict(cvh=0.9, beta=3.0, seed=7, N=32, km=4.5, mid_eta=0.19),   # S1: N 30, E 68, CV 0.81, eta 0.37, one hub (figure 12)
           "hk6": dict(graph="HK:32:2:0.8:6", seed=6, mid_eta=0.2)}               # Holme-Kim: N 32, E 60, CV 0.84, two hubs (figure 11)

if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "s7"
    fam, th = family(**CONFIGS[tag])
    for nm, G in fam.items():
        t0 = time.time()
        d, b = edge_map(G, f"{tag}_{nm}")
        print(f"{nm:6s} N {G.number_of_nodes()} E {G.number_of_edges()} CV {NS.cv(G):.2f} eta {NS.eta(G):.3f} T {NS.n_triangles(G)} "
              f"dbar {NS.dbar(G):.2f} b {b:.3f}: one-sided {d.one.sum():+.4f}  two-sided {d.two.sum():+.4f}  direct {d.direct.sum():+.4f} "
              f"two-sided C-C output {d.cc.sum():+.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
