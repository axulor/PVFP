# -*- coding: utf-8 -*-
"""lead_decomp.py -- section 3.3 ("dilution or offset"): the pool-by-pool account of the cooperators' lead on the 40
networks; lead_predict.py turns it into the prediction of xi and its split into dilution and offset.

Cooperator invasions under rule `proc`, visibility (eps1, eps2), theta = 1, b = mean of the two formal critical benefits
of the setting.  At every moment (imitator i with both C and D neighbours) the payoff of every candidate j is split
pool by pool:
    direct      pool (j, i)
    internal    pools (j, l) with l another candidate (l in N(i): a common neighbour of i and j)
                split by the strategies of the pool: CC (both cooperate), CD (one cooperates)
    external    pools (j, l) with l outside N(i) (a non-common neighbour)
and each pool payoff into output share m_jl b (s_j/k_j + s_l/k_l) (accounted under both rules) and kept endowment
(1 - s_j)/k_j (the same under both rules).  Lead = mean over cooperating candidates - mean over defecting candidates.
Two weightings of the moments: equal (as E4b) and the weak-selection drift weight n_C n_D / k_i^2.

Linear prediction (exploratory, not for the paper): cooperators' visible advantage under rule r at visibility
(e1, e2) is  b a_r + kappa  with  a_r = [dir + e1 (CC + CD) + e2 ext]_r / b  and  kappa = endowment lead (negative), so
b*_r = -kappa / a_r  and  Lambda_pred = a_U / a_Sh - 1  (kappa cancels).
Also: U - Sh of the internal C-D pools split by whether the cooperator is the larger (res_hi) or smaller (res_lo) member.
Output: results/final/mechanism/lead_decomp.csv (3000 invasions per process; resumes from the existing file)
Run:  python lead_decomp.py   then   python lead_predict.py"""
import os, sys, time
import numpy as np, pandas as pd
import numba as nb
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kernel import _seed, _unif, Complex
from exp_common import triangles
import networks40 as N40

OUT = os.path.join(N40.ROOT, "results", "final", "mechanism"); os.makedirs(OUT, exist_ok=True)
Q = ["dir_U", "dir_Sh", "CC_U", "CC_Sh", "CD_U", "CD_Sh", "ext_U", "ext_Sh", "end_dir", "end_int", "end_ext", "res_hi", "res_lo"]
NQ = len(Q)


@nb.njit(parallel=True)
def _decomp(indptr, nbr, rev, U, optr, oe, eps1, eps2, kd, b, runs, seed, nch):
    N = len(indptr) - 1
    acc = np.zeros((nch, 2, NQ)); wsum = np.zeros((nch, 2))
    per = (runs + nch - 1) // nch
    kmax = 0
    for x in range(N):
        kmax = max(kmax, indptr[x + 1] - indptr[x])
    for ch in nb.prange(nch):
        s = np.empty(N, np.int64); u = np.empty(N); dc = np.empty(N, np.int64)
        act = np.empty(N, np.int64); pos = np.empty(N, np.int64); w = np.empty(kmax)
        mark = np.zeros(N, np.int64); q = np.empty(NQ); lead = np.empty(NQ)
        rstate = np.zeros(1, np.uint64)
        for run in range(ch * per, min(runs, (ch + 1) * per)):
            _seed(rstate, seed, run)
            for x in range(N):
                s[x] = 0
            s[run % N] = 1
            for x in range(N):
                uu = 0.0
                for a in range(indptr[x], indptr[x + 1]):
                    uu += U[a, 2 * s[x] + s[nbr[a]]]
                u[x] = uu
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
                si = s[i]; a0, a1 = indptr[i], indptr[i + 1]; ki = a1 - a0
                nc = 0
                for a in range(a0, a1):
                    nc += s[nbr[a]]
                if 0 < nc < ki:
                    nd = ki - nc
                    for a in range(a0, a1):
                        mark[nbr[a]] = 1
                    for t in range(NQ):
                        lead[t] = 0.0
                    for a in range(a0, a1):
                        j = nbr[a]; kj = kd[j]; sj = s[j]
                        for t in range(NQ):
                            q[t] = 0.0
                        for e in range(indptr[j], indptr[j + 1]):
                            l = nbr[e]; kl = kd[l]; sl = s[l]
                            O = b * (sj / kj + sl / kl)
                            mS = kl / (kj + kl)
                            en = (1 - sj) / kj
                            if l == i:
                                q[0] += 0.5 * O; q[1] += mS * O; q[8] += en
                            elif mark[l] == 1:
                                if sj == 1 and sl == 1:
                                    q[2] += 0.5 * O; q[3] += mS * O
                                elif sj != sl:
                                    q[4] += 0.5 * O; q[5] += mS * O
                                    if sj == 1:                         # U - Sh of the C-D pool, cooperator's side
                                        if kj > kl:
                                            q[11] += (0.5 - mS) * O
                                        elif kj < kl:
                                            q[12] += (0.5 - mS) * O
                                    else:                               # defector's side: attribute to the cooperator l's size
                                        if kl > kj:
                                            q[11] += (0.5 - mS) * O
                                        elif kl < kj:
                                            q[12] += (0.5 - mS) * O
                                q[9] += en
                            else:
                                q[6] += 0.5 * O; q[7] += mS * O; q[10] += en
                        if sj == 1:
                            for t in range(NQ):
                                lead[t] += q[t] / nc
                        else:
                            for t in range(NQ):
                                lead[t] -= q[t] / nd
                    for a in range(a0, a1):
                        mark[nbr[a]] = 0
                    wt = nc * nd / (ki * ki)
                    for t in range(NQ):
                        acc[ch, 0, t] += lead[t]; acc[ch, 1, t] += wt * lead[t]
                    wsum[ch, 0] += 1.0; wsum[ch, 1] += wt
                vmax = -1e300
                for a in range(a0, a1):
                    j = nbr[a]; sj = s[j]
                    uji = U[rev[a], 2 * sj + si]
                    h = 0.0
                    for qq in range(optr[a], optr[a + 1]):
                        e = oe[qq]; h += U[e, 2 * sj + s[nbr[e]]]
                    v = uji + eps1 * h + eps2 * (u[j] - uji - h)
                    w[a - a0] = v
                    if v > vmax:
                        vmax = v
                tot = 0.0
                for t in range(ki):
                    w[t] = np.exp(w[t] - vmax); tot += w[t]
                r = _unif(rstate) * tot; accw = 0.0; pick = a1 - 1
                for t in range(ki):
                    accw += w[t]
                    if r < accw:
                        pick = a0 + t; break
                ns = s[nbr[pick]]
                if ns == si:
                    continue
                s[i] = ns
                ui = 0.0
                for a in range(a0, a1):
                    y = nbr[a]; sy = s[y]
                    ui += U[a, 2 * ns + sy]
                    e = rev[a]; u[y] += U[e, 2 * sy + ns] - U[e, 2 * sy + si]
                    if sy == ns:
                        dc[y] -= 1
                    else:
                        dc[y] += 1
                    if dc[y] > 0 and pos[y] < 0:
                        pos[y] = nact; act[nact] = y; nact += 1
                    elif dc[y] == 0 and pos[y] >= 0:
                        pp = pos[y]; last = act[nact - 1]; act[pp] = last; pos[last] = pp; pos[y] = -1; nact -= 1
                u[i] = ui
                dc[i] = ki - dc[i]
                if dc[i] > 0 and pos[i] < 0:
                    pos[i] = nact; act[nact] = i; nact += 1
                elif dc[i] == 0 and pos[i] >= 0:
                    pp = pos[i]; last = act[nact - 1]; act[pp] = last; pos[last] = pp; pos[i] = -1; nact -= 1
    return acc.sum(0), wsum.sum(0)


def bstars(net, e1, e2):
    import fig7_rule_plane as F7
    w = F7.load()
    r = w[(w.net == net) & np.isclose(w.eps1, e1) & np.isclose(w.eps2, e2)]
    return (float(r.U.iloc[0]), float(r.Sh.iloc[0])) if len(r) else (np.nan, np.nan)


def run(net, proc, e1, e2, runs=3000, G=None, b=None, seed=5):
    G = G if G is not None else N40.build(net)
    cx = Complex(G, triangles(G))
    if b is None:
        bU, bS = bstars(net, e1, e2); b = 0.5 * (bU + bS)
    U = cx.table(b, proc, 1.0)
    a, ws = _decomp(cx.indptr, cx.nbr, cx.rev, U, cx.optr, cx.oe, float(e1), float(e2), cx.k.copy(), b, runs,
                    seed, nb.get_num_threads() * 4)
    rows = []
    for wi, wname in enumerate(("equal", "drift")):
        rows.append(dict(net=net, proc=proc, eps1=e1, eps2=e2, b=b, weight=wname, **{x: a[wi, t] / ws[wi] for t, x in enumerate(Q)}))
    return rows


def predict(r, e1, e2):
    """Lambda_pred and b*_pred of both rules from one row of leads (output parts proportional to b)."""
    aU = (r.dir_U + e1 * (r.CC_U + r.CD_U) + e2 * r.ext_U) / r.b
    aS = (r.dir_Sh + e1 * (r.CC_Sh + r.CD_Sh) + e2 * r.ext_Sh) / r.b
    kap = r.end_dir + e1 * r.end_int + e2 * r.end_ext
    return aU / aS - 1, -kap / aU, -kap / aS


if __name__ == "__main__":
    nets = N40.SYN + N40.REAL
    settings = [(0.0, 1.0), (1.0, 1.0)]
    out = os.path.join(OUT, "lead_decomp.csv")
    done = set()
    if os.path.exists(out):
        d0 = pd.read_csv(out); done = set(zip(d0.net, d0.proc, d0.eps1, d0.eps2))
    for net in nets:
        G = N40.build(net)
        for e1, e2 in settings:
            for proc in ("U", "Sh"):
                if (net, proc, e1, e2) in done:
                    continue
                t0 = time.time()
                rows = run(net, proc, e1, e2, G=G)
                pd.DataFrame(rows).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
                print(f"{net:16s} ({e1:.0f},{e2:.0f}) {proc:2s} {time.time() - t0:5.1f}s", flush=True)
