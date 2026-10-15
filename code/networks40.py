# -*- coding: utf-8 -*-
"""networks40.py -- the main network set of the paper (design doc v5, section 3.B.1): 20 synthetic + 20 real networks.

Every network is one fixed instance (synthetic: fixed seed); all are simple, undirected, largest connected component.
    build(name)  -> networkx.Graph (nodes relabelled 0..N-1)
    SYN, REAL    -> ordered lists of names
    python networks40.py   -> results/final/networks40_stats.csv (measured statistics of every instance)

Construction rules
* directed survey ties (RH, UGF, UGH): an edge if either side names the other; ESH uses mutual nominations because the
  either-side version is too dense (<k> = 30.1).
* time-resolved contact data: thresholds as listed in data/README.md.
* projection networks (an edge whenever two nodes share a group): JAZZ, NS, EEU -- flagged in `PROJECTION`.
"""
import os, io, zipfile
import numpy as np, pandas as pd, networkx as nx
from networks import load as load_old, chung_lu

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
N_SYN = 500

# ------------------------------------------------------------------------------------------------ synthetic
def rhg(N, kmean, gamma, T, seed):
    """random hyperbolic graph (Krioukov et al. 2010, quasi-uniform node density); disc radius tuned for <k>."""
    rng = np.random.default_rng(seed)
    alpha = (gamma - 1) / 2
    theta = rng.uniform(0, 2 * np.pi, N); u = rng.uniform(0, 1, N); coin = rng.uniform(0, 1, (N, N))
    dth = np.pi - np.abs(np.pi - np.abs(theta[:, None] - theta[None, :]))

    def build(R):
        r = np.arccosh(1 + (np.cosh(alpha * R) - 1) * u) / alpha
        ch = np.cosh(r[:, None]) * np.cosh(r[None, :]) - np.sinh(r[:, None]) * np.sinh(r[None, :]) * np.cos(dth)
        x = np.arccosh(np.maximum(ch, 1.0))
        P = (x < R).astype(float) if T == 0 else 1.0 / (1.0 + np.exp(np.clip((x - R) / (2 * T), -50, 50)))
        A = np.triu(coin < P, 1)
        G = nx.Graph(); G.add_nodes_from(range(N)); G.add_edges_from(zip(*np.nonzero(A)))
        return G
    lo, hi = 2.0, 40.0
    for _ in range(40):
        R = 0.5 * (lo + hi); G = build(R)
        if 2 * G.number_of_edges() / N > kmean:
            lo = R
        else:
            hi = R
    return G


def rgg_torus(N, kmean, seed):
    """random geometric graph on the unit torus (no boundary effects)."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (N, 2)); r = np.sqrt(kmean / (np.pi * (N - 1)))
    D = np.abs(X[:, None, :] - X[None, :, :]); D = np.minimum(D, 1 - D); d = np.sqrt((D ** 2).sum(-1))
    A = np.triu(d < r, 1)
    G = nx.Graph(); G.add_nodes_from(range(N)); G.add_edges_from(zip(*np.nonzero(A)))
    return G


SYN_SPEC = {
    "ER_k4": lambda: nx.gnm_random_graph(N_SYN, N_SYN * 4 // 2, seed=101),
    "ER_k8": lambda: nx.gnm_random_graph(N_SYN, N_SYN * 8 // 2, seed=102),
    "ER_k16": lambda: nx.gnm_random_graph(N_SYN, N_SYN * 16 // 2, seed=103),
    "BA_m2": lambda: nx.barabasi_albert_graph(N_SYN, 2, seed=201),
    "BA_m4": lambda: nx.barabasi_albert_graph(N_SYN, 4, seed=202),
    "BA_m8": lambda: nx.barabasi_albert_graph(N_SYN, 8, seed=203),
    "CL_g2.2": lambda: chung_lu(N_SYN, 2.2, 8, 301),
    "CL_g2.6": lambda: chung_lu(N_SYN, 2.6, 8, 302),
    "CL_g3.2": lambda: chung_lu(N_SYN, 3.2, 8, 303),
    "WS_p0.05": lambda: nx.connected_watts_strogatz_graph(N_SYN, 8, 0.05, seed=401),
    "WS_p0.2": lambda: nx.connected_watts_strogatz_graph(N_SYN, 8, 0.2, seed=402),
    "HK_m4_p0.3": lambda: nx.powerlaw_cluster_graph(N_SYN, 4, 0.3, seed=501),
    "HK_m4_p0.6": lambda: nx.powerlaw_cluster_graph(N_SYN, 4, 0.6, seed=502),
    "HK_m4_p0.9": lambda: nx.powerlaw_cluster_graph(N_SYN, 4, 0.9, seed=503),
    "HK_m8_p0.9": lambda: nx.powerlaw_cluster_graph(N_SYN, 8, 0.9, seed=504),
    "RGG_k6": lambda: rgg_torus(N_SYN, 6, 601),
    "RGG_k12": lambda: rgg_torus(N_SYN, 12, 602),
    "RHG_g3.0_T0.1": lambda: rhg(N_SYN, 8, 3.0, 0.1, 701),
    "RHG_g3.0_T0.5": lambda: rhg(N_SYN, 8, 3.0, 0.5, 702),
    "RHG_g2.7_T0.1": lambda: rhg(N_SYN, 8, 2.7, 0.1, 703),
}
SYN = list(SYN_SPEC)

# ------------------------------------------------------------------------------------------------ real
def _socio(fn, W=15):
    z = np.load(os.path.join(ROOT, "data", "socio", fn))
    G = nx.Graph(); G.add_edges_from(map(tuple, z["edges"][z["w"] >= W])); return G


def _zip_edges(path):
    with zipfile.ZipFile(path) as zf:
        d = pd.read_csv(io.BytesIO(zf.read("edges.csv")))
    d.columns = [c.strip().lstrip("#").strip() for c in d.columns]
    G = nx.Graph(); G.add_edges_from(zip(d["source"], d["target"])); return G


NETZ = os.path.join(ROOT, "data", "netz")
REAL_SPEC = {
    # face-to-face contact
    "CPH": lambda: load_old("CPH")[0], "UTM": lambda: load_old("UTM")[0], "UTE": lambda: load_old("UTE")[0],
    "SFHH": lambda: load_old("SFHH")[0],
    "LPS": lambda: _socio("sp_primary_school__sp_primary_school.npz"),
    "THS": lambda: _socio("sp_high_school__proximity.npz"),
    "LH10": lambda: _socio("sp_hospital__sp_hospital.npz"),
    # survey friendship
    "KAR": lambda: load_old("KAR")[0], "ESH": lambda: load_old("ESH")[0],
    "RH": lambda: _zip_edges(os.path.join(NETZ, "residence_hall.csv.zip")),
    "UGF": lambda: _zip_edges(os.path.join(ROOT, "data", "netz", "ugandan_village__friendship-8.csv.zip")),
    # online friendship
    "CPH-FB": lambda: load_old("CPH-FB")[0],
    "FBF": lambda: _zip_edges(os.path.join(NETZ, "facebook_friends.csv.zip")),
    # email
    "EEU": lambda: load_old("EEU")[0], "URV": lambda: load_old("URV")[0],
    # collaboration
    "JAZZ": lambda: _zip_edges(os.path.join(NETZ, "jazz_collab.csv.zip")),
    "NS": lambda: _zip_edges(os.path.join(NETZ, "netscience.csv.zip")),
    # animal, advice, covert
    "DOL": lambda: _zip_edges(os.path.join(NETZ, "dolphins.csv.zip")),
    "UGH": lambda: _zip_edges(os.path.join(ROOT, "data", "netz", "ugandan_village__health-advice_9.csv.zip")),
    "TER": lambda: _zip_edges(os.path.join(NETZ, "train_terrorists.csv.zip")),
}
REAL = list(REAL_SPEC)
TYPE = {"CPH": "contact", "UTM": "contact", "UTE": "contact", "SFHH": "contact", "LPS": "contact", "THS": "contact",
        "LH10": "contact", "KAR": "friendship", "ESH": "friendship", "RH": "friendship", "UGF": "friendship",
        "CPH-FB": "online", "FBF": "online", "EEU": "email", "URV": "email", "JAZZ": "collaboration", "NS": "collaboration",
        "DOL": "animal", "UGH": "advice", "TER": "covert"}
PROJECTION = {"JAZZ", "NS", "EEU"}


def _clean(G):
    G = nx.Graph(G); G.remove_edges_from(nx.selfloop_edges(G))
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return nx.convert_node_labels_to_integers(G, ordering="sorted")


def build(name):
    G = (SYN_SPEC.get(name) or REAL_SPEC[name])()
    return _clean(G)


def stats(G):
    deg = dict(G.degree()); k = np.array(list(deg.values()), float); E = G.number_of_edges()
    nb = {u: set(G[u]) for u in G}
    e = np.array([len(nb[u] & nb[v]) for u, v in G.edges()])
    ds = [max(deg[u], deg[v], deg[w]) / min(deg[u], deg[v], deg[w])
          for u in G for v in nb[u] if v > u for w in nb[u] & nb[v] if w > v]
    return dict(N=G.number_of_nodes(), E=E, kmean=k.mean(), kmax=int(k.max()), CV=k.std() / k.mean(),
                C=nx.transitivity(G), e_mean=e.mean(), T=len(ds), dbar=np.mean(ds) if ds else np.nan,
                r=nx.degree_assortativity_coefficient(G))


if __name__ == "__main__":
    rows = []
    for nm in SYN + REAL:
        st = stats(build(nm)); st.update(net=nm, group="synthetic" if nm in SYN else "real",
                                          type=TYPE.get(nm, nm.split("_")[0]), projection=nm in PROJECTION)
        rows.append(st); print(nm, {k: round(v, 3) if isinstance(v, float) else v for k, v in st.items() if k in ("N", "kmean", "CV", "C", "e_mean", "T", "dbar", "r")}, flush=True)
    out = os.path.join(ROOT, "results", "final"); os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "networks40_stats.csv"), index=False)
