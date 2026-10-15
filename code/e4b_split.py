# -*- coding: utf-8 -*-
"""
e4b_split.py -- E4b (figures 10-12): where the cooperators' lead in visible payoff comes from (design doc, 3.3 second
half; 2026-10-08 version, replaces the exploration program th1_split.py).

Process: cooperator invasion (one cooperator in all-D) under rule `proc` (U or Sh), visibility (eps1, eps2), theta,
benefit b (the same b for both rules: the mean of b*_Un and b*_Sh of the setting).  Common neighbors = all common
neighbors of imitator and candidate (no 2-simplices).

At every update whose imitator i has both cooperating and defecting neighbors (a "moment"; all moments weigh the
same), the visible payoff of every candidate j is split by channel (direct u_ji; common neighbors l in Omega_ij;
non-common neighbors l in Psi_ij) and by part, and the part is accounted under BOTH rules on the same state:
    own_hi / own_lo / own_eq   m b p_jl s_j      (own return) on pools where k_j > k_l / k_j < k_l / k_j = k_l
    par_hi / par_lo / par_eq   m b p_lj s_l      (partner share), same three kinds of pool
    endow                      p_jl c (1 - s_j)  (kept endowment; identical under both rules)
For every part, rule of accounting and channel the program records the mean over the cooperating candidates and the
mean over the defecting candidates; the lead is their difference.  The two decompositions of the figure follow by
summing parts:  (i) by input  own = own_hi + own_lo + own_eq, partner = par_hi + par_lo + par_eq;
                (ii) by transfer direction  hi = own_hi + par_hi, lo = own_lo + par_lo  (eq parts: Un - Sh = 0).
Uniform - Shapley of the lead = lead accounted under U minus lead accounted under Sh; the endowment cancels exactly.
Also recorded: the share of cooperators among the candidate's common / non-common neighbors (excluding i).

Run:  python e4b_split.py NET [NET ...]     (env RUNS = invasions per process, default 8000; EPS = "1,1")
Output (appends): results/final/<TAG>_split.csv (one row per net x proc x part x acc x channel), <TAG>_coopshare.csv,
<TAG>_comp.csv (components hi, lo, own, partner, net_effect per channel with batch SEs).  TAG defaults to e4b.
"""
import os, sys, time
import numpy as np, pandas as pd
import numba as nb
from kernel import _seed, _unif, Complex, shares
from exp_common import triangles
import networks40 as N40

FIN = os.path.join(N40.ROOT, "results", "final")
PARTS = ["own_hi", "own_lo", "own_eq", "par_hi", "par_lo", "par_eq", "endow"]
RULES = ["U", "Sh"]
CHANNELS = ["direct", "common", "noncommon"]
NB = 8                                                  # batches for the standard error


def part_tables(cx, b, c=1.0):
    """T[r * 7 + p, a, 2 s_src + s_dst]: part p of the payoff to src from edge a = (src -> dst), accounted under rule r."""
    src = np.repeat(np.arange(cx.N), np.diff(cx.indptr))
    pij, pji = 1.0 / cx.k[src], 1.0 / cx.k[cx.nbr]
    kind = np.sign(cx.k[src] - cx.k[cx.nbr]).astype(int)          # +1: src has the larger degree
    col = {1: 0, -1: 1, 0: 2}
    T = np.zeros((len(RULES) * len(PARTS), len(pij), 4))
    for r, rule in enumerate(RULES):
        m = shares(pij, pji, rule, 1.0)
        for a in range(len(pij)):
            q = col[int(kind[a])]
            T[r * 7 + q, a, 2] = T[r * 7 + q, a, 3] = m[a] * b * pij[a]             # own return (src cooperates)
            T[r * 7 + 3 + q, a, 1] = T[r * 7 + 3 + q, a, 3] = m[a] * b * pji[a]     # partner share (dst cooperates)
        T[r * 7 + 6, :, 0] = T[r * 7 + 6, :, 1] = pij * c                           # kept endowment (src defects)
        assert np.allclose(T[r * 7:(r + 1) * 7].sum(0), cx.table(b, rule, 1.0)), "parts must add up to the payoff table"
    return np.ascontiguousarray(T)


@nb.njit(parallel=True)
def _split_runs(indptr, nbr, rev, U, optr, oe, eps1, eps2, theta, T, runs, seed, nchunks):
    N = len(indptr) - 1
    Q = T.shape[0]
    Z = Q * 3
    kmax = 0
    for x in range(N):
        if indptr[x + 1] - indptr[x] > kmax:
            kmax = indptr[x + 1] - indptr[x]
    accC = np.zeros((nchunks, Z)); accD = np.zeros((nchunks, Z)); nmom = np.zeros(nchunks)
    fr = np.zeros((nchunks, 4)); nfr = np.zeros((nchunks, 4))      # coop share: C-cand common, C-cand non, D-cand common, D-cand non
    per = (runs + nchunks - 1) // nchunks
    for ch in nb.prange(nchunks):
        s = np.empty(N, np.int64); u = np.empty(N); dc = np.empty(N, np.int64)
        act = np.empty(N, np.int64); pos = np.empty(N, np.int64); w = np.empty(kmax)
        ut = np.empty((Q, N)); comp = np.empty((kmax, Z)); fra = np.empty((kmax, 2))
        rstate = np.zeros(1, np.uint64)
        lo = ch * per
        hi = min(runs, lo + per)
        for run in range(lo, hi):
            _seed(rstate, seed, run)
            m = run % N
            for x in range(N):
                s[x] = 0
            s[m] = 1
            for x in range(N):
                uu = 0.0
                for a in range(indptr[x], indptr[x + 1]):
                    uu += U[a, 2 * s[x] + s[nbr[a]]]
                u[x] = uu
                for p in range(Q):
                    vv = 0.0
                    for a in range(indptr[x], indptr[x + 1]):
                        vv += T[p, a, 2 * s[x] + s[nbr[a]]]
                    ut[p, x] = vv
            nact = 0
            for x in range(N):
                c = 0
                for a in range(indptr[x], indptr[x + 1]):
                    if s[nbr[a]] != s[x]:
                        c += 1
                dc[x] = c
                if c > 0:
                    pos[x] = nact; act[nact] = x; nact += 1
                else:
                    pos[x] = -1
            while nact > 0:
                i = act[int(_unif(rstate) * nact)]
                si = s[i]
                a0, a1 = indptr[i], indptr[i + 1]
                ki = a1 - a0
                vmax = -1e300
                nc = 0
                for a in range(a0, a1):
                    nc += s[nbr[a]]
                moment = 0 < nc < ki
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
                    if moment:
                        for p in range(Q):
                            d = T[p, rev[a], 2 * sj + si]
                            hh = 0.0
                            for q in range(optr[a], optr[a + 1]):
                                e = oe[q]
                                hh += T[p, e, 2 * sj + s[nbr[e]]]
                            comp[a - a0, p * 3 + 0] = d
                            comp[a - a0, p * 3 + 1] = eps1 * hh
                            comp[a - a0, p * 3 + 2] = eps2 * (ut[p, j] - d - hh)
                        ncom = optr[a + 1] - optr[a]
                        ccom = 0
                        for q in range(optr[a], optr[a + 1]):
                            ccom += s[nbr[oe[q]]]
                        kj = indptr[j + 1] - indptr[j]
                        cnb = dc[j] if sj == 0 else kj - dc[j]                 # cooperating neighbors of j
                        cnon = cnb - si - ccom
                        nnon = kj - 1 - ncom
                        fra[a - a0, 0] = ccom / ncom if ncom > 0 else -1.0
                        fra[a - a0, 1] = cnon / nnon if nnon > 0 else -1.0
                if moment:
                    for z in range(Z):
                        mc_ = 0.0; md_ = 0.0
                        for a in range(a0, a1):
                            if s[nbr[a]] == 1:
                                mc_ += comp[a - a0, z]
                            else:
                                md_ += comp[a - a0, z]
                        accC[ch, z] += mc_ / nc
                        accD[ch, z] += md_ / (ki - nc)
                    nmom[ch] += 1.0
                    for cls in range(2):
                        sc = 0.0; nC_ = 0; sd = 0.0; nD_ = 0
                        for a in range(a0, a1):
                            f = fra[a - a0, cls]
                            if f >= 0.0:
                                if s[nbr[a]] == 1:
                                    sc += f; nC_ += 1
                                else:
                                    sd += f; nD_ += 1
                        if nC_ > 0:
                            fr[ch, cls] += sc / nC_; nfr[ch, cls] += 1.0
                        if nD_ > 0:
                            fr[ch, 2 + cls] += sd / nD_; nfr[ch, 2 + cls] += 1.0
                tot = 0.0
                for t in range(ki):
                    w[t] = np.exp(theta * (w[t] - vmax))
                    tot += w[t]
                r = _unif(rstate) * tot
                accw = 0.0
                pick = a1 - 1
                for t in range(ki):
                    accw += w[t]
                    if r < accw:
                        pick = a0 + t
                        break
                ns = s[nbr[pick]]
                if ns == si:
                    continue
                s[i] = ns
                ui = 0.0
                for a in range(a0, a1):
                    y = nbr[a]
                    sy = s[y]
                    ui += U[a, 2 * ns + sy]
                    e = rev[a]
                    u[y] += U[e, 2 * sy + ns] - U[e, 2 * sy + si]
                u[i] = ui
                for p in range(Q):
                    vi = 0.0
                    for a in range(a0, a1):
                        y = nbr[a]
                        sy = s[y]
                        vi += T[p, a, 2 * ns + sy]
                        e = rev[a]
                        ut[p, y] += T[p, e, 2 * sy + ns] - T[p, e, 2 * sy + si]
                    ut[p, i] = vi
                for a in range(a0, a1):
                    y = nbr[a]
                    if s[y] == ns:
                        dc[y] -= 1
                    else:
                        dc[y] += 1
                    if dc[y] > 0 and pos[y] < 0:
                        pos[y] = nact; act[nact] = y; nact += 1
                    elif dc[y] == 0 and pos[y] >= 0:
                        pp = pos[y]; last = act[nact - 1]; act[pp] = last; pos[last] = pp; pos[y] = -1; nact -= 1
                dc[i] = ki - dc[i]
                if dc[i] > 0 and pos[i] < 0:
                    pos[i] = nact; act[nact] = i; nact += 1
                elif dc[i] == 0 and pos[i] >= 0:
                    pp = pos[i]; last = act[nact - 1]; act[pp] = last; pos[last] = pp; pos[i] = -1; nact -= 1
    return accC, accD, nmom, fr, nfr


def bstar_11(net, e1, e2):
    """b* of both rules at the setting, from the formal results (E1 diagonal, E2 grid, E3a)."""
    import fig7_rule_plane as F7
    w = F7.load()
    r = w[(w.net == net) & np.isclose(w.eps1, e1) & np.isclose(w.eps2, e2)]
    return float(r.U.iloc[0]), float(r.Sh.iloc[0])


COMPONENTS = {                                          # Uniform - Shapley of the lead = sum of (part, sign) over both rules
    "hi": ["own_hi", "par_hi"], "lo": ["own_lo", "par_lo"], "eq": ["own_eq", "par_eq"],
    "own": ["own_hi", "own_lo", "own_eq"], "partner": ["par_hi", "par_lo", "par_eq"], "endow": ["endow"],
    "net_effect": ["own_hi", "own_lo", "own_eq", "par_hi", "par_lo", "par_eq", "endow"]}


def measure(net, proc, e1=1.0, e2=1.0, theta=1.0, runs=8000, seed=11, G=None, bU=None, bS=None):
    """one process (proc) on network `net` (built from the main set unless G is given); b = mean of the two critical
    benefits of the setting (from the formal results unless bU, bS are given).  Returns the per-part rows, the
    cooperator shares, and the component rows (value and batch SE of hi, lo, own, partner, net_effect per channel)."""
    G = G if G is not None else N40.build(net)
    cx = Complex(G, triangles(G))
    if bU is None:
        bU, bS = bstar_11(net, e1, e2)
    b = 0.5 * (bU + bS)
    U = cx.table(b, proc, 1.0)
    T = part_tables(cx, b)
    nch = nb.get_num_threads() * 4
    nch -= nch % NB
    accC, accD, nmom, fr, nfr = _split_runs(cx.indptr, cx.nbr, cx.rev, U, cx.optr, cx.oe, float(e1), float(e2),
                                            float(theta), T, runs, seed + (0 if proc == "U" else 500), nch)
    # batches of chunks -> per-batch ratio estimates -> SE
    g = np.arange(nch) % NB
    bC = np.array([accC[g == k].sum(0) / nmom[g == k].sum() for k in range(NB)])
    bD = np.array([accD[g == k].sum(0) / nmom[g == k].sum() for k in range(NB)])
    mC, mD = accC.sum(0) / nmom.sum(), accD.sum(0) / nmom.sum()
    rows = []
    for r_, rule in enumerate(RULES):
        for p_, part in enumerate(PARTS):
            for x_, chn in enumerate(CHANNELS):
                z = (r_ * 7 + p_) * 3 + x_
                rows.append(dict(net=net, proc=proc, eps1=e1, eps2=e2, theta=theta, b=b, bU=bU, bSh=bS, acc=rule, part=part,
                                 channel=chn, meanC=mC[z], meanD=mD[z], lead=mC[z] - mD[z],
                                 lead_se=np.std(bC[:, z] - bD[:, z], ddof=1) / np.sqrt(NB), moments=nmom.sum(), runs=runs))
    f = fr.sum(0) / np.maximum(nfr.sum(0), 1)
    frac = dict(net=net, proc=proc, C_common=f[0], C_noncommon=f[1], D_common=f[2], D_noncommon=f[3])
    lead_all, lead_b = mC - mD, bC - bD                   # (Z,) and (NB, Z)
    comp = []
    for x_, chn in enumerate(CHANNELS):
        for name, parts in COMPONENTS.items():
            zs_u = [(0 * 7 + PARTS.index(p)) * 3 + x_ for p in parts]
            zs_s = [(1 * 7 + PARTS.index(p)) * 3 + x_ for p in parts]
            val = lead_all[zs_u].sum() - lead_all[zs_s].sum()
            vb = lead_b[:, zs_u].sum(1) - lead_b[:, zs_s].sum(1)
            comp.append(dict(net=net, proc=proc, eps1=e1, eps2=e2, b=b, channel=chn, component=name, value=val,
                             se=np.std(vb, ddof=1) / np.sqrt(NB), moments=nmom.sum(), runs=runs))
    return rows, frac, comp


def summary(d):
    """Uniform - Shapley of the lead per channel, for both decompositions (rows: net, proc)."""
    p = d.pivot_table(index=["net", "proc", "channel", "part"], columns="acc", values="lead").reset_index()
    p["diff"] = p.U - p.Sh
    out = []
    for (net, proc, chn), s in p.groupby(["net", "proc", "channel"]):
        v = s.set_index("part")["diff"]
        out.append(dict(net=net, proc=proc, channel=chn, own=v[["own_hi", "own_lo", "own_eq"]].sum(),
                        partner=v[["par_hi", "par_lo", "par_eq"]].sum(), hi=v.own_hi + v.par_hi, lo=v.own_lo + v.par_lo,
                        eq=v.own_eq + v.par_eq, endow=v.endow, net_effect=v.sum()))
    return pd.DataFrame(out)


if __name__ == "__main__":
    nets = sys.argv[1:] or ["CPH", "BA_m4", "HK_m4_p0.9"]
    runs = int(os.environ.get("RUNS", "8000"))
    e1, e2 = (float(x) for x in os.environ.get("EPS", "1,1").split(","))
    tag = os.environ.get("TAG", "e4b")                  # output: results/final/<tag>_split.csv, _coopshare.csv, _comp.csv
    out, fout, cout = (os.path.join(FIN, f"{tag}_{k}.csv") for k in ("split", "coopshare", "comp"))
    for net in nets:
        for proc in RULES:
            t0 = time.time()
            rows, frac, comp = measure(net, proc, e1, e2, runs=runs)
            d = pd.DataFrame(rows)
            d.to_csv(out, mode="a", header=not os.path.exists(out), index=False)
            pd.DataFrame([frac]).to_csv(fout, mode="a", header=not os.path.exists(fout), index=False)
            pd.DataFrame(comp).to_csv(cout, mode="a", header=not os.path.exists(cout), index=False)
            c = pd.DataFrame(comp).pivot_table(index="channel", columns="component", values="value")
            print(time.strftime("%H:%M:%S"), net, proc, f"{time.time() - t0:.0f}s, {d.moments.iloc[0]:.0f} moments", flush=True)
            print(c[["hi", "lo", "own", "partner", "net_effect"]].round(4).to_string(), flush=True)
