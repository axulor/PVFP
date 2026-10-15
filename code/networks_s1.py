# -*- coding: utf-8 -*-
"""networks_s1.py -- artificial networks with a fixed degree sequence (design doc 3.3; figures 9 and 12).

  build_s1(kh_cv, beta, seed)   S1 model (Serrano, Krioukov, Boguna 2008): N nodes on a circle with gamma-distributed
                                hidden degrees (coefficient of variation kh_cv); connection probability
                                1 / (1 + (d / (mu kappa kappa'))^beta), mu calibrated so that <k> = 8; giant component.
  eta(G)                        relative embeddedness of the design doc (3.A.2): (1/N) sum_i (1/k_i) sum_{j in N(i)}
                                e_ij / (k_j - 1), the share of common neighbors among the candidate's third parties,
                                averaged as in death-birth updating
  dbar(G)                       triangle degree disparity: mean over triangles of max(k) / min(k)
  rewire_to_eta(G, target)      random degree-preserving swaps (connected) until eta reaches the target from above
  Composition                   degree- and triangle-preserving targeted swaps that move dbar towards homogeneous
                                (small) or mixed (large) triangles; the triangle count T stays within +-tol of its start
                                value and eta within +-eta_tol; dbar and T are updated incrementally.
Run as a script to build the composition series of figure 10 (results/final/nets_comp/ + comp_series.csv); since
2026-10-09 edges of degree-1 nodes are never moved (both the random swaps down to eta = 0.3 and the targeted ones).
"""
import os, sys, time
import numpy as np, networkx as nx, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
N0, KM = 500, 8.0


def build_s1(kh_cv, beta, seed, N=N0, km=KM, angles=False):
    """angles=True also returns the hidden angle of every node of the returned graph (for circle layouts)."""
    rng = np.random.default_rng(seed)
    if kh_cv > 0.01:
        sh = 1 / kh_cv ** 2
        kap = rng.gamma(sh, km / sh, N)
    else:
        kap = np.full(N, km)
    kap = np.maximum(kap, 1.5); kap *= km / kap.mean()
    th = rng.uniform(0, 2 * np.pi, N)
    d = (N / (2 * np.pi)) * (np.pi - np.abs(np.pi - np.abs(th[:, None] - th[None, :])))
    np.fill_diagonal(d, np.inf)
    KK = np.outer(kap, kap)
    U = rng.uniform(size=(N, N))

    def prob(mu):
        with np.errstate(divide="ignore", over="ignore"):
            return 1.0 / (1.0 + (d / (mu * KK)) ** beta)
    lo, hi = 1e-8, 1e3
    for _ in range(80):
        mu = np.sqrt(lo * hi)
        if prob(mu).sum() / N > km:
            hi = mu
        else:
            lo = mu
    A = np.triu(U < prob(mu), 1)
    G = nx.Graph(); G.add_nodes_from(range(N)); G.add_edges_from(zip(*np.nonzero(A)))
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    H = nx.convert_node_labels_to_integers(G, label_attribute="orig")
    if angles:
        return H, {n: float(th[H.nodes[n]["orig"]]) for n in H}
    return H


def eta(G):
    k = dict(G.degree()); nb = {i: set(G[i]) for i in G}
    tot = 0.0
    for i in G:
        s = 0.0
        for j in nb[i]:
            if k[j] > 1:
                s += len(nb[i] & nb[j]) / (k[j] - 1)
        tot += s / k[i]
    return tot / G.number_of_nodes()


def dbar(G):
    k = dict(G.degree()); nb = {i: set(G[i]) for i in G}
    r = []
    for u, v in G.edges():
        for w in nb[u] & nb[v]:
            if w > max(u, v):
                t = (k[u], k[v], k[w]); r.append(max(t) / min(t))
    return float(np.mean(r)) if r else float("nan")


def n_triangles(G):
    return sum(nx.triangles(G).values()) // 3


def cv(G):
    k = np.array([d for _, d in G.degree()]); return float(k.std() / k.mean())


def rewire_to_eta(G, target, seed=0, tol=0.004):
    """random connected double-edge swaps in small batches until eta <= target + tol (eta falls as triangles are
    destroyed); a batch that overshoots below target - tol is undone and retried with fewer swaps."""
    H = G.copy(); E = H.number_of_edges(); rng = np.random.default_rng(seed)
    cur = eta(H)
    if cur <= target + tol:
        return H, cur
    step = max(1, E // 40)
    while cur > target + tol:
        T = H.copy()
        nx.connected_double_edge_swap(T, nswap=step, seed=int(rng.integers(1 << 30)))
        e = eta(T)
        if e < target - tol and step > 1:
            step = max(1, step // 2); continue
        H, cur = T, e
    return H, cur


def random_down(G, target, seed=0, block=50, keep_leaves=True):
    """random degree-preserving swaps in blocks (connectivity checked per block) until eta <= target; with
    keep_leaves, edges of degree-1 nodes are never moved (2026-10-09: the first composition series let them
    migrate onto hubs)."""
    C = Composition(G, seed=seed, keep_leaves=keep_leaves)
    good = list(C.edges)
    while C.Q / C.N > target:
        n = 0
        while n < block:
            p = C.propose()
            if p is None:
                continue
            dd, rec = p
            C.commit(rec, dd); n += 1
        if nx.is_connected(C.graph()):
            good = list(C.edges)
        else:
            C = Composition(nx.Graph(good), seed=seed + 1, keep_leaves=keep_leaves)
    return C.graph(), C.Q / C.N


class Composition:
    """degree-preserving swaps that move dbar (= S / T, S = sum over triangles of max/min degree ratio) while the
    triangle count T and the relative embeddedness eta stay in narrow bands.  T, S and eta are updated incrementally:
    eta = (1/N) sum over ordered edges (i, j) of e_ij / (k_i (k_j - 1)), and a swap only changes the e of the
    removed / added edges and of the edges of the triangles they close or open."""

    def __init__(self, G, seed=0, keep_leaves=False):
        self.keep_leaves = keep_leaves                        # never move an edge of a node with degree 1
        self.k = dict(G.degree()); self.nb = {i: set(G[i]) for i in G}; self.N = G.number_of_nodes()
        self.edges = [tuple(e) for e in G.edges()]
        self.rng = np.random.default_rng(seed)
        self.T, self.S = 0, 0.0
        for u, v in self.edges:
            for w in self.nb[u] & self.nb[v]:
                if w > max(u, v):
                    self.T += 1; self.S += self.r(u, v, w)
        self.Q = eta(G) * self.N                              # N * eta

    def r(self, a, b, c):
        t = (self.k[a], self.k[b], self.k[c]); return max(t) / min(t)

    def q(self, i, j):                                        # weight of one unit of e_ij in N * eta, both directions
        x = 1.0 / (self.k[i] * (self.k[j] - 1)) if self.k[j] > 1 else 0.0
        y = 1.0 / (self.k[j] * (self.k[i] - 1)) if self.k[i] > 1 else 0.0
        return x + y

    def _remove(self, a, b, d):
        cw = self.nb[a] & self.nb[b]
        d[2] -= len(cw) * self.q(a, b)                       # the pair (a, b) itself
        self.nb[a].discard(b); self.nb[b].discard(a)
        for w in cw:                                          # triangles a-b-w are opened
            d[0] -= 1; d[1] -= self.r(a, b, w); d[2] -= self.q(a, w) + self.q(b, w)

    def _add(self, a, b, d):
        cw = self.nb[a] & self.nb[b]
        for w in cw:
            d[0] += 1; d[1] += self.r(a, b, w); d[2] += self.q(a, w) + self.q(b, w)
        d[2] += len(cw) * self.q(a, b)
        self.nb[a].add(b); self.nb[b].add(a)

    def propose(self):
        """a random swap (a-b, c-d) -> (a-d, c-b); returns the changes [dT, dS, dQ] and an undo record, or None."""
        n = len(self.edges)
        i1, i2 = self.rng.integers(n), self.rng.integers(n)
        if i1 == i2:
            return None
        a, b = self.edges[i1]; c, d = self.edges[i2]
        if self.rng.random() < 0.5:
            c, d = d, c
        if len({a, b, c, d}) < 4 or d in self.nb[a] or b in self.nb[c]:
            return None
        if self.keep_leaves and min(self.k[a], self.k[b], self.k[c], self.k[d]) == 1:
            return None
        dd = [0, 0.0, 0.0]
        self._remove(a, b, dd); self._remove(c, d, dd); self._add(a, d, dd); self._add(c, b, dd)
        return dd, (i1, i2, a, b, c, d)

    def undo(self, rec):
        i1, i2, a, b, c, d = rec
        tmp = [0, 0.0, 0.0]
        self._remove(a, d, tmp); self._remove(c, b, tmp); self._add(a, b, tmp); self._add(c, d, tmp)

    def commit(self, rec, dd):
        i1, i2, a, b, c, d = rec
        self.edges[i1] = (a, d); self.edges[i2] = (c, b)
        self.T += dd[0]; self.S += dd[1]; self.Q += dd[2]

    def graph(self):
        H = nx.Graph(); H.add_nodes_from(self.k); H.add_edges_from(self.edges); return H

    def run(self, direction, steps, T0, eta0, tol=0.01, eta_tol=0.005, check_every=5000, log=None):
        """greedy: accept a swap if T stays within T0 (1 +- tol), eta within eta0 +- eta_tol, and dbar moves in
        `direction` (-1 homogeneous, +1 mixed).  Connectivity is checked every `check_every` accepted swaps; a block
        that disconnects the graph is reverted to the last connected state."""
        good = (list(self.edges), {x: set(y) for x, y in self.nb.items()}, self.T, self.S, self.Q)
        acc = 0
        for it in range(steps):
            p = self.propose()
            if p is None:
                continue
            dd, rec = p
            Tn, Sn, Qn = self.T + dd[0], self.S + dd[1], self.Q + dd[2]
            ok = (abs(Tn - T0) <= tol * T0 and Tn > 0 and abs(Qn / self.N - eta0) <= eta_tol
                  and direction * (Sn / Tn - self.S / self.T) > 1e-12)
            if not ok:
                self.undo(rec); continue
            self.commit(rec, dd); acc += 1
            if acc % check_every == 0 or it == steps - 1:
                H = self.graph()
                if not nx.is_connected(H):
                    self.edges, self.nb, self.T, self.S, self.Q = (list(good[0]), {x: set(y) for x, y in good[1].items()},
                                                                   good[2], good[3], good[4])
                    if log:
                        log(f"    block reverted at accepted {acc} (disconnected)")
                    continue
                good = (list(self.edges), {x: set(y) for x, y in self.nb.items()}, self.T, self.S, self.Q)
                if log:
                    log(f"    accepted {acc:6d} of {it + 1}: T {self.T}  dbar {self.S / self.T:.3f}  eta {self.Q / self.N:.4f}")
        H = self.graph()
        if not nx.is_connected(H):                            # last block
            self.edges = list(good[0]); H = self.graph()
        return acc


def describe(G):
    return dict(N=G.number_of_nodes(), E=G.number_of_edges(), CV=cv(G), T=n_triangles(G), eta=eta(G), dbar=dbar(G))


def write_edges(G, path, name):
    E = sorted((min(u, v), max(u, v)) for u, v in G.edges())
    with open(path, "w", newline="\n") as f:
        f.write(f"# {name}: code/networks_s1.py\n{G.number_of_nodes()} {len(E)}\n")
        f.writelines(f"{u} {v}\n" for u, v in E)


SERIES = [("cv035", 0.2, 4.5), ("cv065", 0.55, 5.0), ("cv100", 0.95, 5.0)]   # (name, hidden CV, beta) -> CV ~ 0.35, 0.65, 1.0
ETA_TARGET = 0.3

if __name__ == "__main__":
    out = os.path.join(ROOT, "results", "final", "nets_comp"); os.makedirs(out, exist_ok=True)
    seeds = [int(x) for x in (sys.argv[1:] or ["1"])]
    steps = int(os.environ.get("STEPS", "300000"))
    rows = []
    for sd in seeds:
        for name, kh, beta in SERIES:
            t0 = time.time()
            G = build_s1(kh, beta, sd)
            base, e = random_down(G, ETA_TARGET, seed=sd)      # pendant edges frozen (2026-10-09 rebuild)
            d0 = describe(base); print(f"{name} seed {sd}: generated {d0}", flush=True)
            for lab, direction in (("asgen", 0), ("homog", -1), ("mixed", +1)):
                if direction == 0:
                    H = base
                else:
                    C = Composition(base, seed=sd * 10 + (direction > 0), keep_leaves=True)
                    C.run(direction, steps, d0["T"], d0["eta"], tol=0.01, eta_tol=0.005, check_every=50,
                          log=lambda m: print(m, flush=True))
                    H = C.graph()
                dd = describe(H)
                nm = f"S1_{name}_{lab}_s{sd}"
                write_edges(H, os.path.join(out, nm + ".txt"), nm)
                rows.append(dict(net=nm, series=name, comp=lab, seed=sd, **dd))
                print(f"  {lab}: {dd}   ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "final", "comp_series.csv"), mode="a",
                              header=not os.path.exists(os.path.join(ROOT, "results", "final", "comp_series.csv")), index=False)
