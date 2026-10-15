# -*- coding: utf-8 -*-
"""e4c_run.py -- E4c (figure 12): the payoff split of e4b_split.py on the composition series (results/final/nets_comp,
comp_series.csv), at (1, 1), b = mean of the two critical benefits computed locally (results/final/e4c_bstar/,
see e4c_bstar_tasks.py).  Output: results/final/e4c_split.csv, e4c_coopshare.csv, e4c_comp.csv (appends; networks
already in e4c_comp.csv are skipped).
Run:  python e4c_run.py        (env RUNS = invasions per process, default 40000)"""
import os, glob, time
import numpy as np, pandas as pd, networkx as nx
import e4b_split as E

FIN = E.FIN
COLS = ["id", "net", "rule", "eps1", "eps2", "theta", "delta", "seed", "bstar", "se", "runs_per_point", "calls", "flag", "seconds"]


def bstars():
    rows = [l.split() for f in glob.glob(os.path.join(FIN, "e4c_bstar", "rank_*.txt")) for l in open(f) if len(l.split()) == 14]
    d = pd.DataFrame(rows, columns=COLS)
    d["bstar"] = d.bstar.astype(float); d["se"] = d.se.astype(float)
    return d.pivot_table(index="net", columns="rule", values="bstar")


def read_net(name):
    with open(os.path.join(FIN, "nets_comp", name + ".txt")) as f:
        lines = [l for l in f if not l.startswith("#")]
    N = int(lines[0].split()[0])
    G = nx.Graph(); G.add_nodes_from(range(N)); G.add_edges_from(tuple(map(int, l.split())) for l in lines[1:])
    return G


if __name__ == "__main__":
    runs = int(os.environ.get("RUNS", "40000"))
    b = bstars()
    comp = pd.read_csv(os.path.join(FIN, "comp_series.csv"))
    cout = os.path.join(FIN, "e4c_comp.csv")
    done = set(pd.read_csv(cout).net) if os.path.exists(cout) else set()
    for r in comp.itertuples():
        if r.net in done or r.net not in b.index or b.loc[r.net].isna().any():
            continue
        G = read_net(r.net)
        for proc in E.RULES:
            t0 = time.time()
            rows, frac, cp = E.measure(r.net, proc, runs=runs, G=G, bU=float(b.loc[r.net, "U"]), bS=float(b.loc[r.net, "Sh"]))
            for tag, data in (("split", rows), ("coopshare", [frac]), ("comp", cp)):
                fn = os.path.join(FIN, f"e4c_{tag}.csv")
                pd.DataFrame(data).to_csv(fn, mode="a", header=not os.path.exists(fn), index=False)
            c = pd.DataFrame(cp).pivot_table(index="channel", columns="component", values="value")
            print(time.strftime("%H:%M:%S"), r.net, proc, f"{time.time() - t0:.0f}s  dbar {r.dbar:.2f}  eta {r.eta:.3f}  "
                  f"common net {c.loc['common', 'net_effect']:+.4f}  noncommon net {c.loc['noncommon', 'net_effect']:+.4f}", flush=True)
