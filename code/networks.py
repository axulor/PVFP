# -*- coding: utf-8 -*-
"""
networks.py -- the network catalogue of section 3.

Synthetic set (N = 1000, three random samples each):
  BA      Barabasi-Albert, m in {2, 3, 5}                       (mean degree 4, 6, 10)
  CL      Chung-Lu graphs with power-law expected degrees, exponent in {2.2, 2.6, 3.0}, mean degree 6
  ER      Erdos-Renyi G(N, M), mean degree in {4, 6, 10}
  WS      Watts-Strogatz, k in {4, 6, 10} with p = 0.1, and k = 6 with p in {0.05, 0.3}
  HK      Holme-Kim, m = 3, triad-formation probability p in {0, 0.3, 0.6, 0.9}
Real networks (built by code/build_nets.py from the raw downloads in data/raw/, see data/README.md):
  higher-order: CPH, CPH-FB, UTM, UTE, SFHH, EEU;  pairwise: URV, ESH (Spanish high school t6);  classic: KAR.
Every network is stored as data/nets/<name>.npz (edges, real 2-simplices if any), largest component, nodes 0..N-1.
"""
import os, sys, importlib.util
import numpy as np, networkx as nx, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NET_DIR = os.path.join(ROOT, "data", "nets")
os.makedirs(NET_DIR, exist_ok=True)


def _save(name, G, tris=None):
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    mp = {u: t for t, u in enumerate(sorted(G.nodes()))}
    E = np.array([(mp[u], mp[v]) for u, v in G.edges() if u != v], np.int64)
    T = np.zeros((0, 3), np.int64)
    if tris is not None and len(tris):
        T = np.array([sorted(mp[x] for x in t) for t in tris if all(x in mp for x in t)
                      and G.has_edge(t[0], t[1]) and G.has_edge(t[0], t[2]) and G.has_edge(t[1], t[2])],
                     np.int64).reshape(-1, 3)
        T = np.unique(T, axis=0)
    np.savez(os.path.join(NET_DIR, name + ".npz"), edges=E, tris=T, N=len(mp))


def chung_lu(N, gamma, kmean, seed):
    i = np.arange(1, N + 1)
    w = (i + 10.0) ** (-1.0 / (gamma - 1.0))
    w *= kmean * N / w.sum()
    w = np.minimum(w, np.sqrt(kmean * N))                 # structural cut-off
    w *= kmean * N / w.sum()
    return nx.expected_degree_graph(w, seed=seed, selfloops=False)


def synthetic_specs():
    specs = []
    for s in (1, 2, 3):
        for m in (2, 3, 5):
            specs.append((f"BA_m{m}_s{s}", "BA", lambda m=m, s=s: nx.barabasi_albert_graph(1000, m, seed=100 * s + m)))
        for g in (2.2, 2.6, 3.0):
            specs.append((f"CL_g{g}_s{s}", "CL", lambda g=g, s=s: chung_lu(1000, g, 6, 200 * s + int(10 * g))))
        for k in (4, 6, 10):
            specs.append((f"ER_k{k}_s{s}", "ER", lambda k=k, s=s: nx.gnm_random_graph(1000, 500 * k, seed=300 * s + k)))
        for k, p in ((4, 0.1), (6, 0.1), (10, 0.1), (6, 0.05), (6, 0.3)):
            specs.append((f"WS_k{k}_p{p}_s{s}", "WS",
                          lambda k=k, p=p, s=s: nx.connected_watts_strogatz_graph(1000, k, p, seed=400 * s + k + int(100 * p))))
        for p in (0.0, 0.3, 0.6, 0.9):
            specs.append((f"HK_p{p}_s{s}", "HK", lambda p=p, s=s: nx.powerlaw_cluster_graph(1000, 3, p, seed=500 * s + int(10 * p))))
    return specs


REAL_HO = ["CPH", "CPH-FB", "UTM", "UTE", "SFHH", "EEU"]
REAL_PAIR = ["URV", "ESH"]
CLASSIC = ["KAR"]


def build_all():
    for name, fam, make in synthetic_specs():
        _save(name, make())
    # real networks: reuse the exploration builder, redirecting its output folder
    spec = importlib.util.spec_from_file_location("build_nets", os.path.join(HERE, "build_nets.py"))
    bn = importlib.util.module_from_spec(spec); spec.loader.exec_module(bn)
    bn.OUT = NET_DIR
    bn.real()
    os.replace(os.path.join(NET_DIR, "ES-t6.npz"), os.path.join(NET_DIR, "ESH.npz"))
    _save("KAR", nx.karate_club_graph())


def catalogue():
    rows = [(n, f, "synthetic") for n, f, _ in synthetic_specs()]
    rows += [(n, n, "real-HO") for n in REAL_HO] + [(n, n, "real-pair") for n in REAL_PAIR] + [("KAR", "KAR", "classic")]
    return pd.DataFrame(rows, columns=["net", "family", "kind"])


def load(name):
    d = np.load(os.path.join(NET_DIR, name + ".npz"))
    N = int(d["N"])
    G = nx.Graph(); G.add_nodes_from(range(N)); G.add_edges_from(map(tuple, d["edges"]))
    return G, [tuple(t) for t in d["tris"]]


def stats(G, tris=()):
    k = np.array([d for _, d in G.degree()], float)
    E = G.number_of_edges()
    return dict(N=G.number_of_nodes(), E=E, kmean=k.mean(), kcv=k.std() / k.mean(), kmax=int(k.max()),
                C=nx.transitivity(G), r=nx.degree_assortativity_coefficient(G),
                n_tri=sum(nx.triangles(G).values()) // 3, n_simplex=len(tris), omega_mean=3 * len(tris) / E)


if __name__ == "__main__":
    build_all()
    cat = catalogue()
    out = []
    for name in cat.net:
        G, tris = load(name)
        out.append(dict(net=name, **stats(G, tris)))
    df = cat.merge(pd.DataFrame(out), on="net")
    df.to_csv(os.path.join(ROOT, "results", "networks.csv"), index=False)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))
