# -*- coding: utf-8 -*-
"""
exp_common.py -- shared helpers for the exploratory experiments of sections 3.2-3.4 (x2_*.py ... x7_*.py).

* complexes:  cx_of(name, mode, f, order, seed)
      mode "real"  -> the stored 2-simplices of a real higher-order network
      mode "pair"  -> no 2-simplices
      mode "fill"  -> a fraction f of all triangles of G, chosen by `order`:
                      "random" (nested random order), "mixed" (largest degree disparity first),
                      "homog" (smallest degree disparity first); disparity = max degree / min degree of the triangle
* b* runner:  Runner(csv).bstar(key, cx, rule, e1, e2, theta, target, guess) appends one row per estimate and skips
  keys already in the csv (so every experiment script can be stopped and resumed).
Threads: NTHREADS (default 14 of 20 logical cores, ~70% load).
"""
import os, time, json
import numpy as np, pandas as pd
import numba as nb
import networkx as nx

NTH = int(os.environ.get("NTHREADS", "14"))
nb.set_num_threads(NTH)

from networks import load
from kernel import Complex
from bstar import Estimator

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES2 = os.path.join(ROOT, "results", "x2")
os.makedirs(RES2, exist_ok=True)


def triangles(G):
    out = []
    for u in G:
        nu = G[u]
        for v in nu:
            if v <= u:
                continue
            for w in nu:
                if w <= v or w not in G[v]:
                    continue
                out.append((u, v, w))
    return out


def disparity(G, tris):
    k = dict(G.degree())
    return np.array([max(k[a], k[b], k[c]) / min(k[a], k[b], k[c]) for a, b, c in tris])


_cache = {}


def cx_of(name, mode="pair", f=1.0, order="random", seed=0):
    key = (name, mode, round(float(f), 6), order, seed)
    if key in _cache:
        return _cache[key]
    G, real = load(name)
    if mode == "real":
        cx = Complex(G, real)
    elif mode == "pair":
        cx = Complex(G, [])
    elif mode == "fill":
        T = triangles(G)
        if order == "random":
            idx = np.random.default_rng(1000 + seed).permutation(len(T))
        else:
            d = disparity(G, T)
            jitter = np.random.default_rng(2000 + seed).random(len(T)) * 1e-9
            idx = np.argsort(-(d + jitter)) if order == "mixed" else np.argsort(d + jitter)
        n = int(f) if f > 1 else int(np.floor(f * len(T) + 1e-9))       # f > 1 means a count of 2-simplices
        cx = Complex(G, [T[t] for t in idx[:n]])
    else:
        raise ValueError(mode)
    _cache[key] = cx
    return cx


def partial_overlap(cx, q, seed=0):
    """control complex: in every in-group list, a fraction q of the entries (rounded at random) is replaced by
    partners of j who are strangers to the imitator i; q = 0 is the real complex, q = 1 the pseudo-group control."""
    import copy
    rng = np.random.default_rng(4000 + seed)
    N, indptr, nbr = cx.N, cx.indptr, cx.nbr
    nb_sets = [set(nbr[indptr[i]:indptr[i + 1]].tolist()) for i in range(N)]
    pos = {}
    for i in range(N):
        for a in range(indptr[i], indptr[i + 1]):
            pos[(i, int(nbr[a]))] = a
    optr = np.zeros(len(nbr) + 1, np.int64)
    oe = []
    for i in range(N):
        for a in range(indptr[i], indptr[i + 1]):
            j = int(nbr[a])
            cur = cx.oe[cx.optr[a]:cx.optr[a + 1]].tolist()
            m = len(cur)
            if m:
                nrep = int(np.floor(q * m + rng.random()))
                keep = [cur[t] for t in rng.permutation(m)[:m - nrep]]
                cand = [l for l in nbr[indptr[j]:indptr[j + 1]].tolist() if l != i and l not in nb_sets[i]]
                rep = [pos[(j, cand[t])] for t in rng.permutation(len(cand))[:nrep]] if cand else []
                oe.extend(keep + rep)
            optr[a + 1] = len(oe)
    new = copy.copy(cx)
    new.optr, new.oe = optr, np.array(oe, np.int64)
    return new


def pseudo_groups(cx, seed=0):
    """control complex: for every directed edge (i -> j) the 'in-group' list Omega_ij is replaced by the same number of
    neighbours of j that are NOT neighbours of i (strangers to the imitator), drawn at random; if too few strangers
    exist, all of them are used.  Returns a shallow copy of cx with new optr / oe."""
    import copy
    rng = np.random.default_rng(3000 + seed)
    N, indptr, nbr = cx.N, cx.indptr, cx.nbr
    nb_sets = [set(nbr[indptr[i]:indptr[i + 1]].tolist()) for i in range(N)]
    pos = {}
    for i in range(N):
        for a in range(indptr[i], indptr[i + 1]):
            pos[(i, int(nbr[a]))] = a
    optr = np.zeros(len(nbr) + 1, np.int64)
    oe = []
    for i in range(N):
        for a in range(indptr[i], indptr[i + 1]):
            j = int(nbr[a])
            m = cx.optr[a + 1] - cx.optr[a]
            if m:
                cand = [l for l in nbr[indptr[j]:indptr[j + 1]].tolist() if l != i and l not in nb_sets[i]]
                pick = rng.permutation(len(cand))[:m] if cand else []
                oe.extend(pos[(j, cand[t])] for t in pick)
            optr[a + 1] = len(oe)
    new = copy.copy(cx)
    new.optr, new.oe = optr, np.array(oe, np.int64)
    return new


def omega_mean(cx):
    return len(cx.oe) / len(cx.nbr)            # mean |Omega_ij| over directed edges = 3|Delta|/|E|


class Runner:
    def __init__(self, csv, log=None):
        self.csv = os.path.join(RES2, csv)
        self.log = os.path.join(RES2, log or csv.replace(".csv", ".log"))
        self.done = {}
        if os.path.exists(self.csv):
            d = pd.read_csv(self.csv)
            for _, r in d.iterrows():
                self.done[r["key"]] = r.to_dict()

    def say(self, msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        with open(self.log, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def bstar(self, key, cx, rule, e1, e2, theta=1.0, target=0.005, guess=None, delta=1.0, seed=1, **extra):
        k = f"{key}|{rule}|{e1:g}|{e2:g}|{theta:g}|{delta:g}"
        if k in self.done:
            return self.done[k]["bstar"]
        t0 = time.time()
        est = Estimator(cx, rule, e1, e2, theta, delta=delta, seed=seed)
        out = est.estimate(guess=guess, target=target)
        row = dict(key=k, tag=key, rule=rule, eps1=e1, eps2=e2, theta=theta, delta=delta, bstar=out["bstar"],
                   se=out.get("se", np.nan), runs=out.get("runs_per_point", 0), flag=out.get("flag", ""),
                   secs=round(time.time() - t0, 1), N=cx.N, n_simplex=cx.n_simplices, omega=omega_mean(cx), **extra)
        pd.DataFrame([row]).to_csv(self.csv, mode="a", header=not os.path.exists(self.csv), index=False)
        self.done[k] = row
        self.say(f"{key} {rule} e1={e1:g} e2={e2:g} th={theta:g}: b*={out['bstar']:.4f} se={row['se']:.4f} ({row['secs']}s)")
        return out["bstar"]


def lam(bu, bs):
    return (bs - bu) / bu
