# -*- coding: utf-8 -*-
"""identity_check.py -- appendix B: exact identity for the common-neighbour part of the cooperators'
lead (Uniform minus Shapley), checked moment by moment in a plain-Python death-birth invasion on a small network.

Uniform and Shapley differ only by a zero-sum transfer inside each pool: in pool (j, l) Uniform gives j
t(j, l) = (1/2 - m_jl) * O_jl more than Shapley, and l gets t(l, j) = -t(j, l).  A common neighbour l of (i, j) is
itself a neighbour of i, i.e. another candidate.  In the lead = mean over cooperating candidates - mean over
defecting candidates, transfers between two cooperators (or two defectors) cancel, so
    lead_common(U - Sh) = sum over pools (j, l) inside N(i) with s_j = 1, s_l = 0 of t(j, l) * (1/n_C + 1/n_D).
The script compares the brute-force value with this formula at every moment (imitator with both C and D neighbours)."""
import os, sys
import numpy as np, networkx as nx
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import networks40 as N40


def m_share(kj, kl, rule):
    return 0.5 if rule == "U" else (1 / kj) / (1 / kj + 1 / kl)       # Shapley, linear production


def pool_payoff(j, l, s, k, b, rule):
    O = b * (s[j] / k[j] + s[l] / k[l])
    return m_share(k[j], k[l], rule) * O + (1 - s[j]) / k[j]


def lead_common(i, s, G, k, b):
    Ni = set(G[i]); C = [j for j in Ni if s[j] == 1]; D = [j for j in Ni if s[j] == 0]
    def tot(j, rule):
        return sum(pool_payoff(j, l, s, k, b, rule) for l in Ni & set(G[j]))
    out = {}
    for rule in ("U", "Sh"):
        out[rule] = np.mean([tot(j, rule) for j in C]) - np.mean([tot(j, rule) for j in D])
    brute = out["U"] - out["Sh"]
    nC, nD = len(C), len(D)
    formula = 0.0
    for j in C:
        for l in Ni & set(G[j]):
            if s[l] == 0:
                O = b * (1 / k[j])                                           # l defects: only j's input
                formula += (0.5 - m_share(k[j], k[l], "Sh")) * O * (1 / nC + 1 / nD)
    return brute, formula


if __name__ == "__main__":
    G = N40.build("KAR"); k = dict(G.degree()); N = G.number_of_nodes(); b = 1.7
    rng = np.random.default_rng(1)
    diffs, n = [], 0
    for run in range(300):
        s = {x: 0 for x in G}; s[run % N] = 1
        for step in range(2000):
            act = [x for x in G if any(s[y] != s[x] for y in G[x])]
            if not act:
                break
            i = act[rng.integers(len(act))]
            Ni = list(G[i])
            if 0 < sum(s[j] for j in Ni) < len(Ni):
                bf, fm = lead_common(i, s, G, k, b); diffs.append(abs(bf - fm)); n += 1
            # Uniform process, full visibility, theta = 1
            v = [sum(pool_payoff(j, l, s, k, b, "U") for l in G[j]) for j in Ni]
            w = np.exp(np.array(v) - max(v)); j = Ni[rng.choice(len(Ni), p=w / w.sum())]
            s[i] = s[j]
    print(f"moments checked: {n}; max |brute - formula| = {max(diffs):.2e}")
