# -*- coding: utf-8 -*-
"""prepare.py -- build the files to upload to the HPC (run locally, from any folder).

    python prepare.py            networks, E1 and the validation tasks (as before):
                                   ../nets/<name>.txt        edge lists of the 40 main networks (code/networks40.py)
                                   ../tasks/validate.txt     17 tasks with known local results
                                   ../tasks/e1.txt           E1: 40 networks x eps in {0, 0.1, 0.2, 0.3, 1} x {Un, Sh}
                                   ../reference/validate_expected.csv
    python prepare.py e2a [K]    E2 stage 1 (figure 5): every network of E2_NETS x 16 x 16 grid (step 0.1) x {Un, Sh},
                                 minus the diagonal points already computed in E1:  ../tasks/e2a_all.txt + part files
    python prepare.py e2b [K]    E2 stage 2: needs results/final/e2a_hpc.csv (python collect.py e2a).  On every row
                                 eps1 of every network, the grid cell in eps2 where Lambda changes sign gets 4 extra
                                 points (spacing 0.02):  ../tasks/e2b_all.txt + part files
    python prepare.py e3a [K]    E3a (figure 7): the 35 main networks not in E2 x {(1,0), (0,1), (1,0.5), (0.5,0.5)}
                                 x {Un, Sh}:  ../tasks/e3a_all.txt + part files
  All are APPEND-ONLY: tasks already in <exp>_all.txt are skipped, new tasks get new ids and new part files numbered
  after the existing ones (a part that may already have run is never changed).  K = number of new parts; by default
  enough parts for ~12 min each on the standard layout.  The submit command is printed.
"""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HPC = os.path.dirname(HERE)
ROOT = os.path.dirname(HPC)
sys.path.insert(0, os.path.join(ROOT, "code"))
import networks40 as N40                                     # noqa: E402

NETS, TASKS, REF = (os.path.join(HPC, d) for d in ("nets", "tasks", "reference"))
for d in (NETS, TASKS, REF):
    os.makedirs(d, exist_ok=True)
HEAD = "# id net rule eps1 eps2 theta delta seed target guess cost\n"
E1_CSV = os.path.join(ROOT, "results", "final", "e1_hpc.csv")
E2_NETS = ["BA_m4", "HK_m4_p0.9", "RHG_g3.0_T0.1", "CPH", "THS"]  # figure 5 = BA, RHG, THS, CPH (2026-10-07); HK to the SI
E2_GRID = [round(k / 10, 1) for k in range(16)]                 # 0, 0.1, ..., 1.5  (16 x 16, decided in review)
E1_EPS = [0.0, 0.1, 0.2, 0.3, 1.0]
WORKERS = 14                                                    # per node: 15 MPI ranks, rank 0 only dispatches


def export(name):
    G = N40.build(name)
    E = sorted((min(u, v), max(u, v)) for u, v in G.edges())
    with open(os.path.join(NETS, f"{name}.txt"), "w", newline="\n") as f:
        f.write(f"# {name}: largest connected component, nodes 0..N-1 (code/networks40.py)\n")
        f.write(f"{G.number_of_nodes()} {len(E)}\n")
        f.writelines(f"{u} {v}\n" for u, v in E)
    return G


def inequality_uniform(G):
    k = dict(G.degree())
    f = np.array([0.5 * (1 + sum(1 / k[j] for j in G[i])) for i in G])
    return np.abs(f[:, None] - f[None, :]).sum() / (2 * len(f) * f.sum())


def line(i, net, rule, e1, e2, seed, target, guess, cost, theta=1.0, delta=1.0):
    return f"{i} {net} {rule} {e1:g} {e2:g} {theta:g} {delta:g} {seed} {target:g} {float(guess)!r} {float(cost):.1f}\n"


def write_tasks(fn, rows):
    with open(os.path.join(TASKS, fn), "w", newline="\n") as f:
        f.write(HEAD); f.writelines(rows)


def check_files(names):
    for fn in names:                                         # every field must be a plain number or a name
        with open(os.path.join(TASKS, fn)) as f:
            for ln in f:
                if ln.startswith("#"):
                    continue
                tok = ln.split()
                assert len(tok) == 11 and all(float(x) == float(x) for x in tok[3:]), (fn, ln)
    print(f"task files checked ({len(names)}): 11 plain fields per line")


# ------------------------------------------------------------------------------------------------ E1 and validation
def make_e1_validate():
    names = N40.SYN + N40.REAL
    info = {}
    for nm in names:
        G = export(nm)
        info[nm] = (G.number_of_edges(), inequality_uniform(G))
    print("exported", len(names), "networks")

    # dispatch order: measured single-core time of one counts call (R = 1e4) per (network, eps), from the efficiency
    # review (local MSVC build); only the ranking matters -- expensive tasks are handed out first
    unit = pd.read_csv(os.path.join(REF, "unit_cost_by_net_eps.csv")).set_index(["net", "eps"]).secs
    rows, i = [], 0
    for nm in names:
        E, I = info[nm]
        for eps in E1_EPS:
            for rule in ("U", "Sh"):
                guess = (2.0 if eps == 0 else 1.5) if rule == "U" else max(1.05, 2.09 - 2.48 * I)
                cost = float(unit.get((nm, eps), E * 1e-3)) * 1000.0  # ms per call; edge count as a fallback
                rows.append(line(i, nm, rule, eps, eps, 1, 0.002, guess, cost)); i += 1
    write_tasks("e1.txt", rows)
    print("e1 tasks:", len(rows))

    # validation: figure 2 (karate, seed 20261007) + the 9 E1 points run locally (seed 1)
    rows, exp = [], []
    k2 = pd.read_csv(os.path.join(ROOT, "results", "final", "fig2_karate.csv"), float_precision="round_trip")
    for _, r in k2.iterrows():
        rows.append(line(len(rows), "KAR", r.rule, r.eps1, r.eps2, 20261007, 0.002, 1.7, 1.0))
        exp.append(dict(id=len(rows) - 1, net="KAR", rule=r.rule, eps=r.eps1, eps2=r.eps2, bstar_py=r.bstar, se_py=r.se))
    e1 = pd.read_csv(os.path.join(ROOT, "results", "final", "e1.csv"), float_precision="round_trip")
    for _, r in e1.iterrows():
        guess = 2.0 if r.rule == "U" else max(1.05, 2.09 - 2.48 * info[r.net][1])  # exactly as in code/e1_uniform.py
        rows.append(line(len(rows), r.net, r.rule, r.eps, r.eps, 1, 0.002, guess, 1.0))
        exp.append(dict(id=len(rows) - 1, net=r.net, rule=r.rule, eps=r.eps, eps2=r.eps, bstar_py=r.bstar, se_py=r.se))
    write_tasks("validate.txt", rows)
    pd.DataFrame(exp).to_csv(os.path.join(REF, "validate_expected.csv"), index=False)
    print("validation tasks:", len(rows))
    check_files(["e1.txt", "validate.txt"])


# ------------------------------------------------------------------------------------------------ E2 helpers
TCOLS = ["id", "net", "rule", "eps1", "eps2", "theta", "delta", "seed", "target", "guess", "cost"]
SPEEDUP_120 = 4.2           # measured (e2r, 2026-10-07): one task on 120 threads vs 8 threads, per counts call


def e1_table():
    """E1 results (formal HPC run): b* and wall seconds per (net, rule, eps)."""
    d = pd.read_csv(E1_CSV)
    return d.set_index(["net", "rule", "eps1"])


def e1_curve(e1, net, rule):
    """b*(eps) of E1 on the uniform diagonal, as arrays (eps, b*)."""
    return np.array(E1_EPS), np.array([e1.loc[(net, rule, e), "bstar"] for e in E1_EPS])


def guess_from_e1(e1, net, rule, eps1, eps2):
    """initial guess for the bracket: E1's uniform curve read at eps2 (uniform rule: b*_Un is set mainly by the
    non-common neighbours) or at the mean weight (Shapley); linear extrapolation beyond eps = 1."""
    x, y = e1_curve(e1, net, rule)
    t = eps2 if rule == "U" else 0.5 * (eps1 + eps2)
    if t <= 1:
        g = float(np.interp(t, x, y))
    else:
        slope = (y[4] - y[3]) / (x[4] - x[3])
        g = float(y[4] + slope * (t - 1)) if rule == "U" else float(y[4])
    return max(0.3, g)


def cost_from_e1(e1, net, rule, eps1, eps2):
    """expected wall seconds of one task (8 threads): the E1 task of the same network and rule at eps = 1, times a
    factor for the total weight eps1 + eps2 measured in E2 stage 1 (2026-10-07: up to ~3x the eps = 1 time at total
    weight 3 in the median, up to ~7x in the worst case; RHG Un was the extreme)."""
    W = eps1 + eps2
    f = 1.0 if W <= 1.5 else 1.3 if W <= 2.0 else 1.8 if W <= 2.5 else 3.0
    return float(e1.loc[(net, rule, 1.0), "seconds"]) * f


def split_parts(items, K):
    """greedy longest-processing-time partition of items (cost first) into K parts of similar total cost."""
    parts = [[] for _ in range(K)]
    load = np.zeros(K)
    for it in sorted(items, key=lambda t: -t[0]):
        k = int(np.argmin(load))
        parts[k].append(it); load[k] += it[0]
    return parts, load


def read_tasks(fn):
    path = os.path.join(TASKS, fn)
    if not os.path.exists(path):
        return pd.DataFrame(columns=TCOLS)
    return pd.read_csv(path, sep=r"\s+", comment="#", header=None, names=TCOLS)


def key(net, rule, e1, e2, theta=1.0, delta=1.0):
    return (net, rule, round(float(e1), 3), round(float(e2), 3), round(float(theta), 3), round(float(delta), 3))


def item_key(it):
    """items are (cost, net, rule, eps1, eps2, guess) or (..., guess, theta, delta)."""
    return key(*it[1:5], *(it[6:8] if len(it) > 6 else ()))


def write_parts(exp, items, K=None, target_min=12.0, speed=WORKERS, job=None):
    """items: (cost, net, rule, eps1, eps2, guess[, theta, delta]).  APPEND-ONLY: tasks already listed in
    tasks/<exp>_all.txt (same net, rule, eps1, eps2, theta, delta) are skipped; new tasks get new ids and new part
    files numbered after the existing ones, so a part file that may already have run is never changed.  K = number of
    new parts (default: enough parts for about `target_min` minutes each).  `speed` = 8-thread task-seconds done per
    second on one node: 14 on the standard layout (14 workers x 8 threads), SPEEDUP_120 on the one-process 120-thread
    layout.  `job` = job script name."""
    old = read_tasks(f"{exp}_all.txt")
    have = {key(r.net, r.rule, r.eps1, r.eps2, r.theta, r.delta) for r in old.itertuples()}
    new = [it for it in items if item_key(it) not in have]
    if not new:
        print(f"{exp}: nothing new to add ({len(old)} tasks already listed)")
        return
    next_id = int(old.id.max()) + 1 if len(old) else 0
    n_old_parts = len([f for f in os.listdir(TASKS) if f.startswith(f"{exp}_part")])
    total = sum(it[0] for it in new)
    if K is None:
        K = max(1, int(np.ceil(total / speed / 60 / target_min)))
    rows = {}
    for it in new:
        c, nm, rule, a, b, g = it[:6]
        th, de = it[6:8] if len(it) > 6 else (1.0, 1.0)
        rows[id(it)] = line(next_id, nm, rule, a, b, 1, 0.002, g, c, theta=th, delta=de); next_id += 1
    with open(os.path.join(TASKS, f"{exp}_all.txt"), "a" if len(old) else "w", newline="\n") as f:
        if not len(old):
            f.write(HEAD)
        f.writelines(rows[id(it)] for it in new)
    parts, load = split_parts(new, K)
    names = [f"{exp}_all.txt"]
    for k, p in enumerate(parts, n_old_parts + 1):
        write_tasks(f"{exp}_part{k:02d}.txt", [rows[id(it)] for it in p])
        names.append(f"{exp}_part{k:02d}.txt")
    longest = max(it[0] for it in new)
    print(f"{exp}: {len(new)} new tasks (ids from {next_id - len(new)}) in {K} new parts "
          f"{n_old_parts + 1:02d}-{n_old_parts + K:02d}; estimated {total / 3600:.1f} task-hours (8 threads); "
          f"longest single task ~{longest / 60:.1f} min")
    for k, p in enumerate(parts, n_old_parts + 1):
        print(f"  part {k:02d}: {len(p):4d} tasks, estimated wall ~{load[k - n_old_parts - 1] / speed / 60:.1f} min + tail")
    print(f"  submit:  pjsub --bulk --sparam {n_old_parts + 1}-{n_old_parts + K} jobs/{job or exp}.sh")
    check_files(names)


# ------------------------------------------------------------------------------------------------ E2 stage 1
def make_e2a(K=None):
    """stage-1 grid for every network in E2_NETS (append-only: networks already listed are skipped)."""
    for nm in E2_NETS:
        export(nm)
    e1 = e1_table()
    items = []
    for nm in E2_NETS:
        for a in E2_GRID:                                   # eps1 (rows of the phase diagram)
            for c in E2_GRID:                               # eps2 (columns)
                if a == c and a in E1_EPS:                  # diagonal points of E1: reused, not recomputed
                    continue
                for rule in ("U", "Sh"):
                    items.append((cost_from_e1(e1, nm, rule, a, c), nm, rule, a, c, guess_from_e1(e1, nm, rule, a, c)))
    write_parts("e2a", items, K)


# ------------------------------------------------------------------------------------------------ E2 stage 2
def make_e2b(K=None):
    """refinement of the critical line: on every row eps1 of every network with stage-1 results, the grid cell in eps2
    where Lambda changes sign gets 4 points (spacing 0.02).  Cells with a missing end are skipped (none of the 12
    unfinished stage-1 tasks lies next to the critical line).  Append-only, like stage 1."""
    a_res = pd.read_csv(os.path.join(ROOT, "results", "final", "e2a_hpc.csv"))
    e1 = pd.read_csv(E1_CSV)
    e1 = e1[e1.net.isin(a_res.net.unique())]
    d = pd.concat([a_res, e1], ignore_index=True)           # full grid = stage 1 + E1 diagonal
    w = d.pivot_table(index=["net", "eps1", "eps2"], columns="rule", values="bstar").reset_index()
    t = d.pivot_table(index=["net", "eps1", "eps2"], columns="rule", values="seconds").reset_index()
    w = w.merge(t, on=["net", "eps1", "eps2"], suffixes=("", "_sec"))
    w["L"] = w.Sh / w.U - 1
    items, cells = [], {}
    for nm in a_res.net.unique():
        for a in E2_GRID:
            r = w[(w.net == nm) & np.isclose(w.eps1, a)].set_index("eps2").sort_index()
            for c0, c1 in zip(E2_GRID[:-1], E2_GRID[1:]):
                p = r[np.isclose(r.index, c0)]; q = r[np.isclose(r.index, c1)]
                if not len(p) or not len(q) or not (np.isfinite(p.L.iloc[0]) and np.isfinite(q.L.iloc[0])):
                    continue
                p, q = p.iloc[0], q.iloc[0]
                if np.sign(p.L) == np.sign(q.L):
                    continue
                cells[nm] = cells.get(nm, 0) + 1
                for k in range(1, 5):                       # 4 interior points, spacing 0.02
                    c = round(c0 + 0.02 * k, 2)
                    f = (c - c0) / (c1 - c0)
                    for rule in ("U", "Sh"):
                        g = (1 - f) * p[rule] + f * q[rule]
                        cost = float(max(p[rule + "_sec"], q[rule + "_sec"]))     # the slower end of the cell
                        items.append((cost, nm, rule, a, c, g))
    print("e2b: sign-change cells per network:", cells)
    write_parts("e2b", items, K)


# ------------------------------------------------------------------------------------------------ E2 reruns
def make_e2r(K=None):
    """2026-10-07: rerun, with the polished estimator (src/model.hpp), the stage-1 tasks whose refine window had been
    widened (old estimator: up to 25 % off on a few of them) and the stage-1 tasks left unfinished, for the networks
    that already have stage-1 results.  Same task definitions (seed, guess) as in e2a; results override e2a by
    (net, rule, eps1, eps2).  One process with 120 threads per node (jobs/e2r.sh), because some tasks are long."""
    d = pd.read_csv(os.path.join(ROOT, "results", "final", "e2a_hpc.csv"))
    t = read_tasks("e2a_all.txt").set_index("id")
    rounds = np.log2(d.runs_per_point / 40000).round().astype(int) + 1
    wid = d[d.calls - (2 + 20 * rounds) >= 18]               # an extra refine attempt = 20 extra calls
    miss = t[t.net.isin(set(d.net)) & ~t.index.isin(d.id)]
    items = [(float(r.seconds) * 1.6, r.net, r.rule, r.eps1, r.eps2, float(t.loc[r.id, "guess"])) for r in wid.itertuples()]
    items += [(2400.0, r.net, r.rule, r.eps1, r.eps2, float(r.guess)) for r in miss.itertuples()]
    print(f"e2r: {len(wid)} widened + {len(miss)} unfinished stage-1 tasks")
    write_parts("e2r", items, K, target_min=15.0, speed=SPEEDUP_120)


# ------------------------------------------------------------------------------------------------ estimator fix
FIX_BORDERLINE = [("HK_m4_p0.9", "U", 1.2, 0.7), ("RHG_g3.0_T0.1", "U", 0.2, 1.4), ("RHG_g3.0_T0.1", "U", 0.6, 0.9),
                  ("RHG_g3.0_T0.1", "U", 0.8, 1.2)]       # 3-4 % off their neighbours with a 0.2 % SE


def widened(d):
    """tasks whose refine needed an extra attempt with the old estimator (20 extra counts calls per attempt)."""
    rounds = np.log2(d.runs_per_point / 40000).round().astype(int) + 1
    return d[d.calls - (2 + 20 * rounds) >= 18]


def make_fix(K=None):
    """2026-10-07: rerun with the revised refine (src/model.hpp, code/bstar.py: never widen the window, never accept a
    root extrapolated from a one-sided window, keep the start inside the bracket) every task on which the old refine
    widened its window -- 48 in E2 stage 1 (their e2r values came from the superseded 'polish' step), 2 in E1 -- and 4
    stage-1 points 3-4 % off their neighbours.  Tasks with an ordinary first window give identical results under both
    versions, so all other results stay valid.  Same task definitions; results override E1/E2 by (net, rule, eps1,
    eps2).  One process with 120 threads per node (jobs/fix.sh)."""
    a = pd.read_csv(os.path.join(ROOT, "results", "final", "e2a_hpc.csv"))
    r = pd.read_csv(os.path.join(ROOT, "results", "final", "e2r_hpc.csv"))
    e1 = pd.read_csv(E1_CSV)
    ta = read_tasks("e2a_all.txt"); t1 = read_tasks("e1.txt")
    k = ["net", "rule", "eps1", "eps2"]

    def guess_of(t, net, rule, x, y):
        m = t[(t.net == net) & (t.rule == rule) & np.isclose(t.eps1, x) & np.isclose(t.eps2, y)]
        return float(m.guess.iloc[0])

    items = []
    for x in r[r.flag != "ok"].itertuples():                # the 48 widened stage-1 tasks
        old = a[(a.net == x.net) & (a.rule == x.rule) & np.isclose(a.eps1, x.eps1) & np.isclose(a.eps2, x.eps2)]
        items.append((float(old.seconds.iloc[0]) * 1.5, x.net, x.rule, x.eps1, x.eps2, guess_of(ta, x.net, x.rule, x.eps1, x.eps2)))
    for x in widened(e1).itertuples():                      # E1
        items.append((float(x.seconds) * 1.5, x.net, x.rule, x.eps1, x.eps2, guess_of(t1, x.net, x.rule, x.eps1, x.eps2)))
    for net, rule, x1, x2 in FIX_BORDERLINE:
        old = a[(a.net == net) & (a.rule == rule) & np.isclose(a.eps1, x1) & np.isclose(a.eps2, x2)]
        items.append((float(old.seconds.iloc[0]) * 1.5, net, rule, x1, x2, guess_of(ta, net, rule, x1, x2)))
    print(f"fix: {len(r[r.flag != 'ok'])} widened E2 + {len(widened(e1))} widened E1 + {len(FIX_BORDERLINE)} borderline tasks")
    write_parts("fix", items, K, target_min=10.0, speed=SPEEDUP_120, job="fix")


# ------------------------------------------------------------------------------------------------ E3a
E3A_SETTINGS = [(1.0, 0.0), (0.0, 1.0), (1.0, 0.5), (0.5, 0.5)]  # figure 7 (first two); SI kappa, figure 5 eps = 0.5


def make_e3a(K=None):
    """2026-10-08, figure 7: every main network x {(1,0), (0,1), (1,0.5), (0.5,0.5)} x {Un, Sh}, theta = delta = 1,
    precision 0.2 %.  The five E2 networks already have these four points on their 16 x 16 grid and are skipped, so
    35 x 4 x 2 = 280 new tasks.  Guesses and costs from E1, as in E2.  Standard layout (15 ranks x 8 threads), bulk job
    jobs/e3a.sh.  (0.5,0.5) is also the eps = 0.5 column of figure 5 at theta = delta = 1."""
    e1 = e1_table()
    items = []
    for nm in N40.SYN + N40.REAL:
        if nm in E2_NETS:
            continue
        for a, c in E3A_SETTINGS:
            for rule in ("U", "Sh"):
                items.append((cost_from_e1(e1, nm, rule, a, c), nm, rule, a, c, guess_from_e1(e1, nm, rule, a, c)))
    write_parts("e3a", items, K, target_min=15.0)


# ------------------------------------------------------------------------------------------------ E8, E9 (figure 5)
E89_EPS = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
E8_THETA = [0.1, 0.2, 0.5, 2.0]   # 2026-10-08: theta = 5 dropped (local pilot: 10 of 24 tasks over 10 min, hours each)
E9_DELTA = [0.5, 0.7, 1.4, 2.0]
RCAP_E89 = 40960000         # BSTAR_RCAP of jobs/e89.sh and jobs/e89w.sh (2026-10-08: 16 x the old effective cap)
LONG_SEC = 1200.0           # tasks predicted longer than this on 8 threads go to the wide jobs
WIDE_NODES = 4              # nodes per wide sub-job
WIDE_SPEED_NODE = 12.0      # 8-thread task-seconds per second per node in wide mode (assumed; measured in round 1)


def base_table():
    """theta = delta = 1 results per (net, rule, eps) on the uniform diagonal: E1 and, for eps = 0.5, E3a (0.5,0.5).
    Columns: seconds (8 threads), rpp, calls, se, bstar, trun (seconds per simulated run), c1 (relative SE x sqrt(R))."""
    e1 = pd.read_csv(E1_CSV)
    half = []
    for fn in ("e3a_hpc.csv", "e2a_hpc.csv"):                               # (0.5,0.5): E3a; the E2 networks: E2
        x = pd.read_csv(os.path.join(ROOT, "results", "final", fn))
        half.append(x[np.isclose(x.eps1, 0.5) & np.isclose(x.eps2, 0.5)])
    d = pd.concat([e1] + half, ignore_index=True).drop_duplicates(["net", "rule", "eps1", "eps2"])
    d["eps"] = d.eps1.round(3)
    rounds = np.log2(d.runs_per_point / 40000).round() + 1
    d["trun"] = d.seconds / (d.runs_per_point * 10 + (d.calls - 20 * rounds) * 20000)
    d["c1"] = d.se / d.bstar * np.sqrt(d.runs_per_point)
    return d.set_index(["net", "rule", "eps"])


def runs_needed(c1, theta, target=0.002):
    """total runs per point for the target precision: the relative SE scales as 1/theta at weak selection
    (exploration: about 70x the runs at theta = 0.1); unchanged for theta >= 1."""
    return (c1 / min(theta, 1.0) / target) ** 2


def cost_e89(base, net, rule, eps, theta, delta, cap=RCAP_E89):
    """predicted seconds of one task on 8 threads."""
    r = base.loc[(net, rule, round(eps, 3))]
    if theta < 1:
        R = 40000
        while R < runs_needed(r.c1, theta) and R < cap:
            R *= 2
        # round 1 (2026-10-08): theta = 0.5 took 0.95 x, the theta = 0.1 calibration tasks about 0.3-1 x this estimate
        # in total, but single tasks differ by up to 10x either way -> x 0.6; tasks that overrun resume in wide mode
        return float(r.trun * (R * 10 + 4 * 20000) * 1.2 * 0.6)
    if delta != 1:
        return float(r.seconds * 1.0)                                       # round 1: 0.6-0.8 x the old factor 1.3
    # theta = 2, local pilot (2026-10-08): eps = 0 about 1 x, eps 0.3 1-3 x, eps = 1 3-13 x the theta = 1 time
    f = 1.2 if eps == 0 else 2.0 if eps <= 0.3 else 8.0
    return float(r.seconds * f)


def guess_e89(base, net, rule, eps, theta, delta):
    """theta = 1 value; for delta != 1 scaled by (delta - 1)/ln(delta), the slope of g at small inputs."""
    g = float(base.loc[(net, rule, round(eps, 3)), "bstar"])
    return g * ((delta - 1) / np.log(delta) if delta != 1 else 1.0)


def e89_items(thetas=(), deltas=(), nets=None, eps=E89_EPS):
    base = base_table()
    items = []
    for nm in (nets or N40.SYN + N40.REAL):
        for e in eps:
            for rule in ("U", "Sh"):
                for th in thetas:
                    if th == 1.0 and e == 0.5:
                        continue                                            # from E3a
                    items.append((cost_e89(base, nm, rule, e, th, 1.0), nm, rule, e, e, guess_e89(base, nm, rule, e, th, 1.0), th, 1.0))
                for de in deltas:
                    items.append((cost_e89(base, nm, rule, e, 1.0, de), nm, rule, e, e, guess_e89(base, nm, rule, e, 1.0, de), 1.0, de))
    return items


def make_e89(round_no):
    """figure 5, in rounds (append-only; experiment name e89 for E8 and E9 together).
    Round 1: tasks predicted to exceed LONG_SEC on 8 threads went to wide parts (tasks/e89w_partKK.txt, jobs/e89w.sh).
    Round 2 on (2026-10-08, after round 1 showed that single theta < 1 tasks cannot be predicted to better than 10x):
    TWO STAGES -- every task goes to a standard part (jobs/e89.sh); the few that are still running at the 30-min limit
    leave a checkpoint and are finished in wide mode on one node (jobs/e89r.sh, same part number)."""
    if round_no == 1:
        # E9 (delta), theta = 0.5, and 24 calibration tasks at theta = 0.1 (six networks of different cost, eps 0 and
        # 1): they test the cost model of theta < 1 before the large rounds
        items = e89_items(deltas=E9_DELTA) + e89_items(thetas=[0.5])
        cal = ["ER_k8", "WS_p0.2", "BA_m4", "HK_m4_p0.9", "THS", "KAR"]
        items += e89_items(thetas=[0.1], nets=cal, eps=[0.0, 1.0])
    elif round_no == 2:
        items = e89_items(thetas=[0.2, 0.1, 2.0])
        long_ = [it for it in items if it[0] > 1500.0]                      # predicted > 25 min: wide from the start
        short = [it for it in items if it[0] <= 1500.0]
        print(f"e89 round 2: {len(items)} tasks; {len(short)} standard ({sum(i[0] for i in short) / 3600:.1f} task-hours), "
              f"{len(long_)} wide ({sum(i[0] for i in long_) / 3600:.1f} task-hours)")
        write_parts("e89", short, target_min=15.0, job="e89")
        # wide speed measured in round 1 (widecheck): 10.8x per node on one node, 22x on four (short tasks); 6 per
        # node is kept on the safe side because single theta = 0.1 tasks are predicted only to within ~10x
        write_parts("e89w", long_, target_min=18.0, speed=6.0 * WIDE_NODES, job="e89w")
        return
    else:
        raise SystemExit("no round 3: theta = 5 was dropped (2026-10-08)")
    short = [it for it in items if it[0] <= LONG_SEC]
    long_ = [it for it in items if it[0] > LONG_SEC]
    print(f"e89 round {round_no}: {len(items)} tasks; {len(short)} standard ({sum(i[0] for i in short) / 3600:.1f} task-hours), "
          f"{len(long_)} wide ({sum(i[0] for i in long_) / 3600:.1f} task-hours, longest {max([i[0] for i in long_], default=0) / 3600:.1f} h)")
    write_parts("e89", short, target_min=15.0, job="e89")
    if long_:
        write_parts("e89w", long_, target_min=18.0, speed=WIDE_SPEED_NODE * WIDE_NODES, job="e89w")


def make_wide_check():
    """round 1 test of the wide mode: seven E1/E3a tasks with their original definitions (same seeds, default cap) on
    1 node and on 4 nodes (jobs/widecheck.sh): results must be identical to E1/E3a; the wall time per task gives the
    speed-up of the wide mode over the 8-thread farm."""
    t1 = read_tasks("e1.txt"); t3 = read_tasks("e3a_all.txt")
    pick1 = [("URV", "Sh", 1.0), ("RHG_g3.0_T0.1", "U", 1.0), ("CPH", "U", 0.0), ("EEU", "Sh", 0.1)]
    pick3 = [("UGF", "U", 0.0, 1.0), ("HK_m4_p0.3", "U", 0.0, 1.0), ("CPH-FB", "Sh", 0.5, 0.5)]
    rows = []
    for nm, rule, e in pick1:
        r = t1[(t1.net == nm) & (t1.rule == rule) & np.isclose(t1.eps1, e)].iloc[0]
        rows.append(line(len(rows), nm, rule, r.eps1, r.eps2, r.seed, r.target, r.guess, r.cost))
    for nm, rule, a, b in pick3:
        r = t3[(t3.net == nm) & (t3.rule == rule) & np.isclose(t3.eps1, a) & np.isclose(t3.eps2, b)].iloc[0]
        rows.append(line(len(rows), nm, rule, r.eps1, r.eps2, r.seed, r.target, r.guess, r.cost))
    write_tasks("widecheck.txt", rows)
    check_files(["widecheck.txt"])
    print("widecheck tasks:", len(rows))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "e1"
    K = int(sys.argv[2]) if len(sys.argv) > 2 else None
    if what == "e1":
        make_e1_validate()
    elif what == "e2a":
        make_e2a(K)
    elif what == "e2b":
        make_e2b(K)
    elif what == "e2r":
        make_e2r(K)
    elif what == "fix":
        make_fix(K)
    elif what == "e3a":
        make_e3a(K)
    elif what == "e89":
        make_e89(K or 1)
    elif what == "widecheck":
        make_wide_check()
    else:
        raise SystemExit(__doc__)
