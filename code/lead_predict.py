# -*- coding: utf-8 -*-
"""lead_predict.py -- section 3.3 ("dilution or offset"): compare the pool-by-pool linear prediction of Lambda with the
formal results, and split xi = 1 - Lambda(1,1)/Lambda(0,1) into its two sources (reads
results/final/mechanism/lead_decomp.csv from lead_decomp.py).  Cited in the design doc: prediction vs formal xi
(Pearson, median ratio); dilution D vs eta (Spearman); the share of xi due to dilution, D / xi_pred, median and
quartiles over the networks with formal xi > 0.1 (68 %, 57-77 %, 32 networks).

With the leads measured in the (1,1) process (both rules' processes averaged):
    a_r(0,1) b = dir_r + ext_r                    a_r(1,1) b = dir_r + ext_r + CC + CD_r
    Delta(0,1) = a_U - a_Sh = dir_U - dir_Sh + ext_U - ext_Sh,   Delta(1,1) = Delta(0,1) + CD_U - CD_Sh
    Lambda = Delta / a_Sh
so  1 - xi = [Delta(1,1) / Delta(0,1)] * [a_Sh(0,1) / a_Sh(1,1)] = (1 + rho) (1 - D)  with
    D   = (CC + CD_Sh) / (dir_Sh + ext_Sh + CC + CD_Sh)   share of the cooperators' visible advantage (Shapley) that
                                                         comes from pools between two candidates ("dilution")
    rho = (CD_U - CD_Sh) / Delta(0,1)                     the C-D residue relative to the conflict ("residue", < 0)"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import networks40 as N40
import fig7_rule_plane as F7

EX = os.path.join(N40.ROOT, "results", "final", "mechanism")


def table(weight="equal"):
    d = pd.read_csv(os.path.join(EX, "lead_decomp.csv"))
    d = d[d.weight == weight].groupby(["net", "eps1", "eps2"]).mean(numeric_only=True).reset_index()
    w = F7.load()
    sh = pd.read_csv(os.path.join(N40.ROOT, "results", "final", "networks40_share.csv")).set_index("net")
    st = pd.read_csv(os.path.join(N40.ROOT, "results", "final", "networks40_stats.csv")).set_index("net")
    rows = []
    for net, g in d.groupby("net"):
        r01 = g[(g.eps1 == 0) & (g.eps2 == 1)].iloc[0]; r11 = g[(g.eps1 == 1) & (g.eps2 == 1)].iloc[0]
        L = {e: float(w[(w.net == net) & np.isclose(w.eps1, e[0]) & np.isclose(w.eps2, e[1])].L.iloc[0]) for e in [(0, 1), (1, 1)]}
        out = dict(net=net, CV=st.CV[net], eta=sh.share_pair[net], dbar=st.dbar[net], L01=L[(0, 1)], L11=L[(1, 1)])
        for tag, r in (("p01", r01), ("p11", r11)):                   # leads measured in the (0,1) / (1,1) process
            aU01 = r.dir_U + r.ext_U; aS01 = r.dir_Sh + r.ext_Sh
            aU11 = aU01 + r.CC_U + r.CD_U; aS11 = aS01 + r.CC_Sh + r.CD_Sh
            out[f"L01_{tag}"] = aU01 / aS01 - 1; out[f"L11_{tag}"] = aU11 / aS11 - 1
            D = (r.CC_Sh + r.CD_Sh) / aS11; rho = (r.CD_U - r.CD_Sh) / (aU01 - aS01)
            out[f"D_{tag}"] = D; out[f"rho_{tag}"] = rho; out[f"xi_{tag}"] = 1 - (1 + rho) * (1 - D)
            out[f"CCshare_{tag}"] = r.CC_Sh / aS11; out[f"res_hi_{tag}"] = r.res_hi; out[f"res_lo_{tag}"] = r.res_lo
        # mixed: Lambda(0,1) from the (0,1) process, Lambda(1,1) from the (1,1) process
        out["xi_mixed"] = 1 - out["L11_p11"] / out["L01_p01"]
        out["xi"] = 1 - L[(1, 1)] / L[(0, 1)]
        rows.append(out)
    return pd.DataFrame(rows).set_index("net")


if __name__ == "__main__":
    for weight in ("equal", "drift"):
        t = table(weight)
        print(f"\n===== weight: {weight} =====")
        for a, b in [("L01", "L01_p01"), ("L01", "L01_p11"), ("L11", "L11_p11"), ("xi", "xi_p11"), ("xi", "xi_mixed")]:
            x, y = t[a], t[b]
            print(f"{a:4s} vs {b:8s}: Pearson {pearsonr(x, y)[0]:+.3f}  Spearman {spearmanr(x, y)[0]:+.3f}  "
                  f"median ratio {np.median(y / x):.2f}")
        print(f"D (dilution) vs eta: Spearman {spearmanr(t.D_p11, t.eta)[0]:+.3f}, Pearson {pearsonr(t.D_p11, t.eta)[0]:+.3f}")
        print(f"xi vs eta: Spearman {spearmanr(t.xi, t.eta)[0]:+.3f};  xi vs D: {spearmanr(t.xi, t.D_p11)[0]:+.3f}")
        s = t[t.xi > 0.1]; f = s.D_p11 / s.xi_p11                  # share of xi due to dilution (C-C games)
        print(f"share of xi due to dilution (networks with xi > 0.1: {len(s)}): median {f.median():.1%}, "
              f"quartiles {f.quantile(0.25):.1%}-{f.quantile(0.75):.1%}")
        if weight == "equal":
            cols = ["CV", "eta", "dbar", "L01", "L01_p11", "L11", "L11_p11", "xi", "xi_p11", "D_p11", "rho_p11", "CCshare_p11"]
            print(t.sort_values("eta")[cols].round(3).to_string())
            t.to_csv(os.path.join(EX, "lead_predict.csv"))
