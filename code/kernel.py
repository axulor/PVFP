# -*- coding: utf-8 -*-
"""
kernel.py -- Monte Carlo fixation probabilities for the model of section 2 (2-模型.md).

Model (all notation as in section 2):
  * simplicial complex K = (V, E, Delta); every edge {i,j} hosts Sheng et al.'s two-player public goods game:
      u_ij = m_ij * b * g(p_ij s_i + p_ji s_j) + p_ij * c * (1 - s_i)          (eq. 1), p_ij = 1/k_i,
      g(t) = (delta^t - 1)/(delta - 1)  (g(t) = t for delta = 1),
      m_ij = 1/2 (uniform)  or  1/2 + [g(p_ij) - g(p_ji)] / [2 g(p_ij + p_ji)] (Shapley);
  * u_i = sum_j u_ij (eq. 2);
  * death-birth: a uniformly chosen i copies neighbour j with probability ~ exp(theta * v_{j->i}) (eq. 3),
      v_{j->i} = u_ji + eps1 * sum_{l in Omega_ij} u_jl + eps2 * sum_{l in Psi_ij} u_jl   (eq. 4),
      Omega_ij = {l : {i,j,l} in Delta},  Psi_ij = N(j) \\ (Omega_ij U {i}).
  * rho_C: start all-D, one cooperator at node r mod N; rho_D: start all-C, one defector at node r mod N.

Speed-ups that leave fixation probabilities unchanged:
  * the imitator is drawn uniformly among "active" individuals (with at least one neighbour of the other strategy);
    choosing an inactive individual can never change the state, so skipping those steps only changes time;
  * payoffs are updated incrementally (only the flipped individual and its neighbours).
Each run has its own xorshift64* stream seeded from (seed, run index), so the two allocation rules and different
visibility settings can be run with common random numbers.
"""
import numpy as np
import numba as nb
import networkx as nx


# ------------------------------------------------------------------ model construction (python side)
def g_fun(t, delta):
    t = np.asarray(t, float)
    return t if delta == 1.0 else (delta ** t - 1.0) / (delta - 1.0)


def shares(pij, pji, rule, delta):
    """share m_ij of the pool on edge {i,j} that goes to i."""
    if rule == "U":
        return np.full_like(pij, 0.5)
    if rule == "E":                                                     # Sheng's equitable rule (for comparison)
        return pij / (pij + pji)
    if rule == "Sh":
        return 0.5 + (g_fun(pij, delta) - g_fun(pji, delta)) / (2.0 * g_fun(pij + pji, delta))
    raise ValueError(rule)


class Complex:
    """CSR form of a simplicial complex: graph G (nodes 0..N-1) plus a set of filled triangles."""

    def __init__(self, G, simplices):
        G = nx.convert_node_labels_to_integers(G, ordering="sorted")
        N = G.number_of_nodes()
        nbrs = [sorted(G[i]) for i in range(N)]
        indptr = np.zeros(N + 1, np.int64)
        for i in range(N):
            indptr[i + 1] = indptr[i] + len(nbrs[i])
        nbr = np.array([j for i in range(N) for j in nbrs[i]], np.int64)
        pos = {}
        for i in range(N):
            for a in range(indptr[i], indptr[i + 1]):
                pos[(i, int(nbr[a]))] = a
        rev = np.array([pos[(int(nbr[a]), i)] for i in range(N) for a in range(indptr[i], indptr[i + 1])], np.int64)
        tri = {tuple(sorted(t)) for t in simplices}
        # Omega lists: for directed edge a = (i -> j), the directed edges (j -> l) with {i,j,l} in Delta
        optr = np.zeros(len(nbr) + 1, np.int64)
        oe = []
        for i in range(N):
            for a in range(indptr[i], indptr[i + 1]):
                j = int(nbr[a])
                for l in nbrs[j]:
                    if l != i and tuple(sorted((i, j, l))) in tri:
                        oe.append(pos[(j, l)])
                optr[a + 1] = len(oe)
        self.G, self.N, self.indptr, self.nbr, self.rev = G, N, indptr, nbr, rev
        self.optr, self.oe = optr, np.array(oe, np.int64)
        self.k = np.diff(indptr).astype(float)
        self.n_simplices = len(tri)

    def table(self, b, rule, delta=1.0, c=1.0):
        """U[a, 2*s_i + s_j] = payoff to i from edge a = (i -> j)."""
        src = np.repeat(np.arange(self.N), np.diff(self.indptr))
        pij = 1.0 / self.k[src]
        pji = 1.0 / self.k[self.nbr]
        m = shares(pij, pji, rule, delta)
        U = np.empty((len(self.nbr), 4))
        U[:, 0] = pij * c                                                 # i D, j D
        U[:, 1] = m * b * g_fun(pji, delta) + pij * c                     # i D, j C
        U[:, 2] = m * b * g_fun(pij, delta)                               # i C, j D
        U[:, 3] = m * b * g_fun(pij + pji, delta)                         # i C, j C
        return U

    def allc_payoffs(self, b, rule, delta=1.0):
        U = self.table(b, rule, delta)
        return np.bincount(np.repeat(np.arange(self.N), np.diff(self.indptr)), weights=U[:, 3], minlength=self.N)

    def gini(self, rule, delta=1.0):
        u = self.allc_payoffs(1.0, rule, delta)
        return float(np.abs(u[:, None] - u[None, :]).sum() / (2 * len(u) * u.sum()))


# ------------------------------------------------------------------ numba kernel
@nb.njit(inline="always")
def _next(state):
    x = state[0]
    x ^= x >> np.uint64(12)
    x ^= x << np.uint64(25)
    x ^= x >> np.uint64(27)
    state[0] = x
    return (x * np.uint64(2685821657736338717)) >> np.uint64(11)


@nb.njit(inline="always")
def _unif(state):
    return _next(state) * (1.0 / 9007199254740992.0)


@nb.njit
def _seed(state, seed, run):
    z = np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(run) * np.uint64(0xBF58476D1CE4E5B9) + np.uint64(1)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    if z == np.uint64(0):
        z = np.uint64(88172645463325252)
    state[0] = z


@nb.njit
def _one_run(indptr, nbr, rev, U, optr, oe, eps1, eps2, theta, s, u, dc, act, pos, w, rstate):
    """run until absorption from the state already in s, u, dc, act/pos; return final number of cooperators."""
    N = len(indptr) - 1
    nC = 0
    for x in range(N):
        nC += s[x]
    nact = 0
    for x in range(N):
        if dc[x] > 0:
            pos[x] = nact
            act[nact] = x
            nact += 1
        else:
            pos[x] = -1
    while nact > 0:
        i = act[int(_unif(rstate) * nact)]
        si = s[i]
        a0, a1 = indptr[i], indptr[i + 1]
        vmax = -1e300
        for a in range(a0, a1):
            j = nbr[a]
            sj = s[j]
            uji = U[rev[a], 2 * sj + si]
            h = 0.0
            for q in range(optr[a], optr[a + 1]):
                e = oe[q]
                h += U[e, 2 * sj + s[nbr[e]]]
            v = uji + eps1 * h + eps2 * (u[j] - uji - h)
            w[a - a0] = v
            if v > vmax:
                vmax = v
        tot = 0.0
        for t in range(a1 - a0):
            w[t] = np.exp(theta * (w[t] - vmax))
            tot += w[t]
        r = _unif(rstate) * tot
        acc = 0.0
        pick = a1 - 1
        for t in range(a1 - a0):
            acc += w[t]
            if r < acc:
                pick = a0 + t
                break
        ns = s[nbr[pick]]
        if ns == si:
            continue
        # i switches from si to ns
        s[i] = ns
        nC += 1 if ns == 1 else -1
        ui = 0.0
        for a in range(a0, a1):
            y = nbr[a]
            sy = s[y]
            ui += U[a, 2 * ns + sy]
            eyi = rev[a]
            u[y] += U[eyi, 2 * sy + ns] - U[eyi, 2 * sy + si]
            if sy == ns:
                dc[y] -= 1
            else:
                dc[y] += 1
            # active-set membership of y
            if dc[y] > 0 and pos[y] < 0:
                pos[y] = nact
                act[nact] = y
                nact += 1
            elif dc[y] == 0 and pos[y] >= 0:
                p = pos[y]
                last = act[nact - 1]
                act[p] = last
                pos[last] = p
                pos[y] = -1
                nact -= 1
        u[i] = ui
        dc[i] = (a1 - a0) - dc[i]
        if dc[i] > 0 and pos[i] < 0:
            pos[i] = nact
            act[nact] = i
            nact += 1
        elif dc[i] == 0 and pos[i] >= 0:
            p = pos[i]
            last = act[nact - 1]
            act[p] = last
            pos[last] = p
            pos[i] = -1
            nact -= 1
    return nC


@nb.njit(parallel=True)
def fixation_counts(indptr, nbr, rev, U, optr, oe, eps1, eps2, theta, resident, runs, seed, nchunks):
    """number of runs (out of `runs`) in which a single mutant of type 1-resident takes over a resident population.
    The mutant sits at node (run mod N).  Returns the count of fixations."""
    N = len(indptr) - 1
    kmax = 0
    for x in range(N):
        if indptr[x + 1] - indptr[x] > kmax:
            kmax = indptr[x + 1] - indptr[x]
    # baseline payoffs of the resident population
    u0 = np.zeros(N)
    for x in range(N):
        for a in range(indptr[x], indptr[x + 1]):
            u0[x] += U[a, 3 * resident]
    counts = np.zeros(nchunks, np.int64)
    per = (runs + nchunks - 1) // nchunks
    for ch in nb.prange(nchunks):
        s = np.empty(N, np.int64)
        u = np.empty(N)
        dc = np.empty(N, np.int64)
        act = np.empty(N, np.int64)
        pos = np.empty(N, np.int64)
        w = np.empty(kmax)
        rstate = np.zeros(1, np.uint64)
        lo = ch * per
        hi = min(runs, lo + per)
        cnt = 0
        for run in range(lo, hi):
            _seed(rstate, seed, run)
            for x in range(N):
                s[x] = resident
                u[x] = u0[x]
                dc[x] = 0
            m = run % N
            mut = 1 - resident
            # place the mutant: update its payoff, its neighbours' payoffs and discordance counts
            s[m] = mut
            um = 0.0
            for a in range(indptr[m], indptr[m + 1]):
                y = nbr[a]
                um += U[a, 2 * mut + resident]
                e = rev[a]
                u[y] += U[e, 2 * resident + mut] - U[e, 3 * resident]
                dc[y] = 1
            u[m] = um
            dc[m] = indptr[m + 1] - indptr[m]
            nC = _one_run(indptr, nbr, rev, U, optr, oe, eps1, eps2, theta, s, u, dc, act, pos, w, rstate)
            if (mut == 1 and nC == N) or (mut == 0 and nC == 0):
                cnt += 1
        counts[ch] = cnt
    return counts.sum()


def fixation(cx, b, rule, eps1, eps2, theta, runs, seed=1, delta=1.0, nchunks=None):
    """(rho_C, rho_D, D, SE(D)) for complex cx at benefit b."""
    U = cx.table(b, rule, delta)
    nchunks = nchunks or max(1, nb.get_num_threads() * 4)
    fc = fixation_counts(cx.indptr, cx.nbr, cx.rev, U, cx.optr, cx.oe, eps1, eps2, theta, 0, runs, seed, nchunks)
    fd = fixation_counts(cx.indptr, cx.nbr, cx.rev, U, cx.optr, cx.oe, eps1, eps2, theta, 1, runs, seed + 7919, nchunks)
    rc, rd = fc / runs, fd / runs
    se = cx.N * np.sqrt((rc * (1 - rc) + rd * (1 - rd)) / runs)
    return rc, rd, cx.N * (rc - rd), se
