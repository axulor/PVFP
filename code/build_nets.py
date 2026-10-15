# -*- coding: utf-8 -*-
"""
build_nets.py -- build the real networks from the raw downloads in data/raw/ and cache them as npz (edges, 2-simplices) under data/nets/
(called by networks.build_all; the 2-simplices are not used by the current model, which takes all common neighbors).  Thresholds as listed in data/README.md
(LCC of edges with w >= W_MIN; 2-simplices = triads with n >= 1 whose three edges are in the LCC).
"""
import os, sys, io, gzip, zipfile, importlib.util
import numpy as np, pandas as pd, networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "data", "nets"); os.makedirs(OUT, exist_ok=True)
spec = importlib.util.spec_from_file_location("stat", os.path.join(HERE, "contact_counts.py"))
st = importlib.util.module_from_spec(spec); spec.loader.exec_module(st)


def save(name, G, tris=None):
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    mp = {u: t for t, u in enumerate(sorted(G.nodes()))}
    E = np.array([(mp[u], mp[v]) for u, v in G.edges()], np.int64)
    T = np.zeros((0, 3), np.int64)
    if tris is not None and len(tris):
        T = np.array([sorted(mp[x] for x in t) for t in tris if all(x in mp for x in t)
                      and G.has_edge(t[0], t[1]) and G.has_edge(t[0], t[2]) and G.has_edge(t[1], t[2])], np.int64).reshape(-1, 3)
        T = np.unique(T, axis=0)
    np.savez(os.path.join(OUT, name + ".npz"), edges=E, tris=T, N=len(mp))
    print(f"{name}: N={len(mp)} E={len(E)} real 2-simplices={len(T)}", flush=True)


def from_counts(c, W_MIN, T_MIN=1, edge_override=None):
    src = edge_override if edge_override is not None else c
    sel = src.ekeys[src.ew >= W_MIN]
    base = src.base
    G = nx.Graph(); G.add_edges_from(zip((sel // base).tolist(), (sel % base).tolist()))
    G.remove_edges_from(nx.selfloop_edges(G))
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    n = c.base
    tk = c.tkeys[c.tn >= T_MIN]
    x, y, z = tk // (n * n), (tk // n) % n, tk % n
    nodes = set(G.nodes())
    tris = [(a, b, d) for a, b, d in zip(x.tolist(), y.tolist(), z.tolist())
            if a in nodes and b in nodes and d in nodes and G.has_edge(a, b) and G.has_edge(a, d) and G.has_edge(b, d)]
    return G, tris


def synthetic():
    N = 1000
    save("ER_k6", nx.gnm_random_graph(N, 3 * N, seed=1))
    save("ER_k10", nx.gnm_random_graph(N, 5 * N, seed=2))
    for m in (2, 3, 5):
        save(f"BA_m{m}", nx.barabasi_albert_graph(N, m, seed=10 + m))
    for p in (0.0, 0.3, 0.6, 0.9):
        save(f"HK_m3_p{p}", nx.powerlaw_cluster_graph(N, 3, p, seed=20))
    save("HK_m5_p0.9", nx.powerlaw_cluster_graph(N, 5, 0.9, seed=21))
    for p in (0.05, 0.1, 0.3):
        save(f"WS_k6_p{p}", nx.connected_watts_strogatz_graph(N, 6, p, seed=30))
    save("WS_k10_p0.1", nx.connected_watts_strogatz_graph(N, 10, 0.1, seed=31))


def real():
    # CPH (Copenhagen BT, RSSI >= -75 dBm, W >= 15) and CPH-FB (FB edges + same BT co-presence 2-simplices)
    bt = pd.read_csv(st.p("copenhagen", "bt_symmetric.csv")); bt.columns = ["t", "a", "b", "rssi"]
    bt = bt[(bt.b >= 0) & (bt.rssi >= -75)]
    fb = pd.read_csv(st.p("copenhagen", "fb_friends.csv")); fb.columns = ["a", "b"]
    c = st.window_counts(bt.t.values // 300, bt.a.values, bt.b.values, extra_nodes=np.concatenate([fb.a.values, fb.b.values]))
    save("CPH", *from_counts(c, 15))
    save("CPH-FB", *from_counts(c, 1, edge_override=st._reindex_pairs(fb.a.values, fb.b.values, c)))
    # UTM (Utah middle school single day, W >= 15)
    d = pd.read_csv(st.p("utah_middle_school", "pone.0153690.s002.csv")); d.columns = ["i", "j", "t", "dur"]
    save("UTM", *from_counts(st.window_counts(*st._expand(d)), 15))
    # UTE (Utah elementary Elem1, 2 m, two days, W >= 15)
    parts = [st._expand(st._read_toth(k), offset=day * 100000) for day, k in enumerate((4, 5))]
    t, a, b = (np.concatenate(x) for x in zip(*parts))
    save("UTE", *from_counts(st.window_counts(t, a, b), 15))
    # EEU (email-Eu, >= 15 common emails)
    import tarfile
    tf = tarfile.open(st.p("benson_email-Eu", "email-Eu.tar.gz"))
    nv = np.loadtxt(tf.extractfile("email-Eu/email-Eu-nverts.txt"), dtype=int)
    sx = np.loadtxt(tf.extractfile("email-Eu/email-Eu-simplices.txt"), dtype=int)
    groups = np.split(sx, np.cumsum(nv)[:-1])
    save("EEU", *from_counts(st.window_counts(*st.groups_to_contacts(groups)), 15))
    # SFHH (W >= 3)
    with gzip.open(st.p("sociopatterns_SFHH", "SFHH_tij.dat.gz"), "rt") as fh:
        d = pd.read_csv(fh, sep=r"\s+", header=None, names=["t", "i", "j"])
    save("SFHH", *from_counts(st.window_counts(d.t.values // 20, d.i.values, d.j.values), 3))
    # pairwise: URV email, Spanish t6 mutual
    with zipfile.ZipFile(st.p("urv_email", "email.zip")) as z:
        d = pd.read_csv(io.BytesIO(z.read("email.txt")), sep=r"\s+", header=None, names=["a", "b", "w"])
    G = nx.Graph(); G.add_edges_from(zip(d.a, d.b)); G.remove_edges_from(nx.selfloop_edges(G)); save("URV", G)
    d = pd.read_csv(st.p("spanish_highschools", "Edges_t6.csv"))
    pos = d[d.weight > 0]; s = set(zip(pos["from"], pos["to"]))
    G = nx.Graph(); G.add_edges_from((x, y) for x, y in s if (y, x) in s and x != y); save("ES-t6", G)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "syn"):
        synthetic()
    if which in ("all", "real"):
        real()
