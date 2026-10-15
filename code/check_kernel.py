# -*- coding: utf-8 -*-
"""
check_kernel.py -- validate kernel.py against an exact Markov-chain solution on a small graph (section 3, methods).

The exact solver is an independent implementation written directly from eqs. (1)-(4) of section 2:
all 2^N strategy states, death-birth with a uniformly chosen imitator, absorption probabilities by a linear solve.
"""
import itertools, time
import numpy as np, networkx as nx
from kernel import Complex, fixation, g_fun, shares
from exp_common import triangles


def exact(G, simplices, b, rule, eps1, eps2, theta, delta, c=1.0):
    N = G.number_of_nodes()
    tri = {tuple(sorted(t)) for t in simplices}
    k = {i: G.degree(i) for i in G}

    def u_pair(i, j, s):
        pij, pji = 1.0 / k[i], 1.0 / k[j]
        m = float(shares(np.array([pij]), np.array([pji]), rule, delta)[0])
        return m * b * float(g_fun(pij * s[i] + pji * s[j], delta)) + pij * c * (1 - s[i])

    def visible(j, i, s):                                  # v_{j -> i}, eq. (4)
        omega = [l for l in G[j] if l != i and tuple(sorted((i, j, l))) in tri]
        psi = [l for l in G[j] if l != i and l not in omega]
        return u_pair(j, i, s) + eps1 * sum(u_pair(j, l, s) for l in omega) + eps2 * sum(u_pair(j, l, s) for l in psi)

    states = list(itertools.product((0, 1), repeat=N))
    index = {st: n for n, st in enumerate(states)}
    S = len(states)
    P = np.zeros((S, S))
    for st in states:
        n = index[st]
        if sum(st) in (0, N):
            P[n, n] = 1.0
            continue
        for i in range(N):
            nb_ = list(G[i])
            v = np.array([visible(j, i, st) for j in nb_])
            w = np.exp(theta * (v - v.max())); w /= w.sum()
            for j, pj in zip(nb_, w):
                new = list(st); new[i] = st[j]
                P[n, index[tuple(new)]] += pj / N
    allc = index[tuple([1] * N)]
    trans = [n for n, st in enumerate(states) if sum(st) not in (0, N)]
    A = np.eye(len(trans)) - P[np.ix_(trans, trans)]
    rhs = P[np.ix_(trans, [allc])][:, 0]
    x = np.linalg.solve(A, rhs)
    absorb_c = dict(zip(trans, x))
    rhoC = np.mean([absorb_c[index[tuple(1 if q == m else 0 for q in range(N))]] for m in range(N)])
    rhoD = np.mean([1 - absorb_c[index[tuple(0 if q == m else 1 for q in range(N))]] for m in range(N)])
    return rhoC, rhoD


if __name__ == "__main__":
    # 6 nodes with three triangles {0,1,2}, {1,2,3}, {2,3,4} and a 5-cycle through node 5; common neighbours = all triangles
    G = nx.Graph([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 5), (0, 5)])
    simplices = triangles(G)
    cx = Complex(G, simplices)
    runs = int(__import__("sys").argv[1]) if len(__import__("sys").argv) > 1 else 2_000_000
    print("setting                                   exact rhoC  MC rhoC   z  |  exact rhoD  MC rhoD   z")
    for rule, eps1, eps2, theta, delta, b in [("U", 1, 1, 1.0, 1.0, 1.5), ("Sh", 1, 1, 1.0, 1.0, 1.5),
                                              ("U", 3, 0.5, 1.0, 1.0, 2.0), ("Sh", 3, 0.5, 2.0, 3.0, 2.0),
                                              ("Sh", 0, 1, 1.5, 0.5, 1.2), ("U", 2, 0, 1.0, 1.0, 1.8),
                                              ("Sh", 1, 1, 0.0, 1.0, 1.5)]:
        ec, ed = exact(G, simplices, b, rule, eps1, eps2, theta, delta)
        t0 = time.time()
        rc, rd, D, se = fixation(cx, b, rule, float(eps1), float(eps2), theta, runs, seed=3, delta=delta)
        zc = (rc - ec) / np.sqrt(ec * (1 - ec) / runs)
        zd = (rd - ed) / np.sqrt(ed * (1 - ed) / runs)
        print(f"{rule:2s} eps=({eps1},{eps2}) theta={theta} delta={delta} b={b}:  {ec:.5f}  {rc:.5f} {zc:+5.2f} |"
              f"  {ed:.5f}  {rd:.5f} {zd:+5.2f}   ({time.time() - t0:.1f}s)", flush=True)
    print("neutral expectation 1/N =", 1 / 6)
