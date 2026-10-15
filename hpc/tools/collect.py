# -*- coding: utf-8 -*-
"""collect.py -- read the downloaded rank files of one experiment and write one CSV (run locally).

    python collect.py validate_omp  # hpc/data/validate_omp/rank_*.txt  -> compare with reference/validate_expected.csv
    python collect.py e1            # hpc/data/e1/rank_*.txt        -> results/final/e1_hpc.csv
    python collect.py e2a           # hpc/data/e2a/partNN/rank_*.txt (bulk job) -> results/final/e2a_hpc.csv
    python collect.py e89           # E8/E9 standard parts; e89w = the wide parts (separate ids)
    python collect.py widecheck1    # wide-mode test: identical to E1/E3a? speed-up per task
Bulk-job experiments keep one sub-folder per part; their task list is tasks/<exp>_all.txt.
"""
import os, sys, glob
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HPC = os.path.dirname(HERE)
ROOT = os.path.dirname(HPC)
COLS = ["id", "net", "rule", "eps1", "eps2", "theta", "delta", "seed", "bstar", "se", "runs_per_point", "calls", "flag", "seconds"]


def read(exp):
    rows = []
    files = glob.glob(os.path.join(HPC, "data", exp, "rank_*.txt")) + glob.glob(os.path.join(HPC, "data", exp, "part*", "rank_*.txt"))
    for fn in sorted(files):
        with open(fn) as f:
            for line in f:
                tok = line.split()
                if len(tok) == len(COLS):
                    rows.append(tok)
    d = pd.DataFrame(rows, columns=COLS)
    for c in ("id", "seed", "runs_per_point", "calls"):
        d[c] = d[c].astype(np.int64)
    for c in ("eps1", "eps2", "theta", "delta", "bstar", "se", "seconds"):
        d[c] = d[c].astype(float)
    return d.drop_duplicates("id", keep="last").sort_values("id").reset_index(drop=True)


def tasks(exp):
    fn = os.path.join(HPC, "tasks", f"{exp}_all.txt")
    if not os.path.exists(fn):
        fn = os.path.join(HPC, "tasks", f"{exp}.txt")
    return pd.read_csv(fn, sep=r"\s+", comment="#", header=None,
                       names=["id", "net", "rule", "eps1", "eps2", "theta", "delta", "seed", "target", "guess", "cost"])


if __name__ == "__main__":
    exp = sys.argv[1] if len(sys.argv) > 1 else "e1"
    d = read(exp)
    # data/validate_* use tasks/validate.txt, data/widecheck* use tasks/widecheck.txt
    t = tasks("validate" if exp.startswith("validate") else "widecheck" if exp.startswith("widecheck") else exp)
    # every result must belong to the task with the same id (guards against an old data folder or a changed task list)
    chk = d.merge(t, on="id", how="left", suffixes=("", "_task"))
    bad = chk[(chk.net != chk.net_task) | (chk.rule != chk.rule_task) | ~np.isclose(chk.eps1, chk.eps1_task)
              | ~np.isclose(chk.eps2, chk.eps2_task) | ~np.isclose(chk.theta, chk.theta_task)
              | ~np.isclose(chk.delta, chk.delta_task) | (chk.seed != chk.seed_task)]
    if len(bad):
        raise SystemExit(f"{len(bad)} result lines do not match the task list (ids {list(bad.id)[:10]}); "
                         f"was data/{exp}/ produced with a different task file?")
    missing = sorted(set(t.id) - set(d.id))
    print(f"{exp}: {len(d)} of {len(t)} tasks finished; missing: {missing[:20]}{' ...' if len(missing) > 20 else ''}")
    print(f"flags: {d.flag.value_counts().to_dict()}")
    print(f"wall seconds per task: median {d.seconds.median():.0f}, max {d.seconds.max():.0f}, total {d.seconds.sum() / 3600:.1f} task-hours")
    # the estimator stops at the run cap even when the target precision is not reached, with flag 'ok'
    prec = d.merge(t[["id", "target"]], on="id")
    over = prec[prec.se > prec.target * prec.bstar * 1.05]
    print(f"precision: {len(prec) - len(over)} of {len(prec)} within the target; above it (run cap reached): "
          + (", ".join(f"{r.net} {r.rule} ({r.eps1:g},{r.eps2:g}) {100 * r.se / r.bstar:.2f}%" for r in over.itertuples()) or "none"))
    if exp.startswith("validate"):
        e = pd.read_csv(os.path.join(HPC, "reference", "validate_expected.csv"))
        m = e.merge(d[["id", "bstar", "se"]], on="id", how="left")
        m["rel_diff"] = (m.bstar - m.bstar_py) / m.bstar_py
        m["z"] = (m.bstar - m.bstar_py) / np.sqrt(m.se ** 2 + m.se_py ** 2)
        print(m[["id", "net", "rule", "eps", "eps2", "bstar_py", "bstar", "rel_diff", "z"]].to_string(index=False))
        exact = (m.rel_diff.abs() < 1e-8).sum()
        print(f"\n{exact} of {len(m)} identical to the Python results (|rel diff| < 1e-8); max |z| = {m.z.abs().max():.2f}")
        print("PASS" if (m.z.abs() < 3).all() and m.bstar.notna().all() else "CHECK THE DIFFERENCES ABOVE")
    elif exp.startswith("widecheck"):
        # wide mode (2026-10-08): the same tasks as in E1/E3a must give the same numbers; time ratio = speed-up
        fin = os.path.join(ROOT, "results", "final")
        ref = pd.concat([pd.read_csv(os.path.join(fin, f)) for f in ("e1_hpc.csv", "e3a_hpc.csv")], ignore_index=True)
        k = ["net", "rule", "eps1", "eps2"]
        m = d.merge(ref[k + ["bstar", "se", "runs_per_point", "calls", "seconds"]], on=k, suffixes=("", "_ref"))
        m["same"] = (m.bstar == m.bstar_ref) & (m.se == m.se_ref) & (m.runs_per_point == m.runs_per_point_ref) & (m.calls == m.calls_ref)
        m["speedup_vs_8threads"] = m.seconds_ref / m.seconds
        print(m[k + ["bstar", "bstar_ref", "same", "seconds", "seconds_ref", "speedup_vs_8threads"]].to_string(index=False))
        print(f"\n{int(m.same.sum())} of {len(m)} identical to E1/E3a; total speed-up {m.seconds_ref.sum() / m.seconds.sum():.1f}x "
              f"over one 8-thread worker")
        print("PASS" if m.same.all() and len(m) == len(t) else "CHECK THE DIFFERENCES ABOVE")
    else:
        out = os.path.join(ROOT, "results", "final", f"{exp}_hpc.csv")
        d.to_csv(out, index=False)
        print("written", out)
