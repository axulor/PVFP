# -*- coding: utf-8 -*-
"""
contact_counts.py -- read the raw downloads in data/raw/<dataset>/ (kept as downloaded) and count, for each
time-resolved contact data set, the number of time windows with an i-j contact (edge weight w_ij).  Used by
build_nets.py; run as a script it also writes summary statistics of the threshold variants to data/raw/survey.csv.
Edges are kept when w_ij >= W_MIN, and the largest connected component is taken (thresholds in data/README.md).
"""
import os, io, gzip, tarfile, zipfile, itertools, collections, time
import numpy as np, pandas as pd, networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.path.dirname(HERE), "data", "raw")
LOCAL_SOCIO = os.path.join(os.path.dirname(HERE), "data", "socio")   # 已有的 SocioPatterns 计数（只读，可选）
OUT = os.path.join(RAW, "survey.csv")
W_LIST, T_LIST = (1, 3, 15), (1, 3)


# ----------------------------------------------------------------------------------------------------------------
# 计数：与 survey_socio.counts() 相同的逻辑，但用整数编码以节省内存
# ----------------------------------------------------------------------------------------------------------------
class Counts:
    """边计数 (ekey -> w) 与三元组计数 (tkey -> n)，节点已重编号为 0..n-1；key 用 base 进制整数编码。"""

    def __init__(self, ekeys, ew, tkeys, tn, base, labels):
        self.ekeys, self.ew, self.tkeys, self.tn, self.base, self.labels = ekeys, ew, tkeys, tn, base, labels


def _merge(keys_list, cnt_list):
    k = np.concatenate(keys_list)
    c = np.concatenate(cnt_list)
    u, inv = np.unique(k, return_inverse=True)
    return u, np.bincount(inv, weights=c).astype(np.int64)


def window_counts(t, a, b, extra_nodes=None):
    """t, a, b: 每行一个 (时间窗, 节点, 节点) 接触记录。返回 Counts。三角形枚举与 survey_socio 完全相同。
    extra_nodes: 额外纳入编号的节点（用于与另一边集合共用编号）。"""
    t = np.asarray(t, dtype=np.int64); a = np.asarray(a); b = np.asarray(b)
    m = a != b
    t, a, b = t[m], a[m], b[m]
    allv = np.concatenate([a, b] + ([np.asarray(extra_nodes, dtype=a.dtype)] if extra_nodes is not None else []))
    labels, inv = np.unique(allv, return_inverse=True)
    n = len(labels); base = np.int64(n)
    a, b = inv[:len(a)].astype(np.int64), inv[len(a):2 * len(a)].astype(np.int64)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    order = np.lexsort((hi, lo, t))
    t, lo, hi = t[order], lo[order], hi[order]
    keep = np.ones(len(t), bool)
    keep[1:] = (t[1:] != t[:-1]) | (lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1])   # 同一窗内重复记录只算一次
    t, lo, hi = t[keep], lo[keep], hi[keep]
    ek = lo * base + hi
    ekeys, ew = np.unique(ek, return_counts=True)

    bounds = np.flatnonzero(np.diff(t)) + 1
    starts, ends = np.r_[0, bounds], np.r_[bounds, len(t)]
    buf, run_k, run_c, nbuf = [], [], [], 0
    lo_l, hi_l = lo.tolist(), hi.tolist()
    for s, e in zip(starts.tolist(), ends.tolist()):
        if e - s < 3:
            continue
        adj = collections.defaultdict(set)
        L, H = lo_l[s:e], hi_l[s:e]
        for x, y in zip(L, H):
            adj[x].add(y); adj[y].add(x)
        out = []
        for x, y in zip(L, H):                       # x < y
            for z in adj[x] & adj[y]:
                if z > y:                            # 每个三角形 x<y<z 只记一次
                    out.append((x * n + y) * n + z)
        if out:
            buf.append(np.array(out, dtype=np.int64)); nbuf += len(out)
        if nbuf > 5_000_000:                         # 分块合并，控制内存
            u, c = np.unique(np.concatenate(buf), return_counts=True)
            run_k.append(u); run_c.append(c); buf, nbuf = [], 0
            if len(run_k) > 4:
                u, c = _merge(run_k, run_c); run_k, run_c = [u], [c]
    if buf:
        u, c = np.unique(np.concatenate(buf), return_counts=True)
        run_k.append(u); run_c.append(c)
    if run_k:
        tkeys, tn = _merge(run_k, run_c)
    else:
        tkeys, tn = np.zeros(0, np.int64), np.zeros(0, np.int64)
    return Counts(ekeys, ew, tkeys, tn, base, labels)


def groups_to_contacts(groups):
    """群组（单纯形）列表 -> (窗口id, i, j) 接触记录：同一组内所有人两两接触。"""
    T, A, B = [], [], []
    for g_id, g in enumerate(groups):
        g = sorted(set(g))
        for x, y in itertools.combinations(g, 2):
            T.append(g_id); A.append(x); B.append(y)
    return np.array(T), np.array(A), np.array(B)


def pairwise_counts(a, b):
    """纯成对网络：每条边计数 1，无三元组信息。"""
    a = np.asarray(a); b = np.asarray(b)
    m = a != b
    c = window_counts(np.arange(m.sum(), dtype=np.int64), a[m], b[m])   # 每条边单独一窗 -> 不产生三元组
    c.tkeys = None; c.tn = None                      # 标记：无高阶信息
    c.ew = np.ones_like(c.ew)
    return c


# ----------------------------------------------------------------------------------------------------------------
# 统计量
# ----------------------------------------------------------------------------------------------------------------
def lcc_stats(c, W_MIN=1, T_MIN=1, edge_override=None):
    """edge_override: 另给一个边集合 (Counts，同一节点编号)，用其边建网络、用 c 的三元组算 2-单纯形。"""
    src = edge_override if edge_override is not None else c
    sel = src.ekeys[src.ew >= W_MIN]
    base = src.base
    G = nx.Graph()
    G.add_edges_from(zip((sel // base).tolist(), (sel % base).tolist()))
    if G.number_of_nodes() == 0:
        return None
    n_raw = G.number_of_nodes()
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    deg = np.array([d for _, d in G.degree()], float)
    N, E = G.number_of_nodes(), G.number_of_edges()
    TG = sum(nx.triangles(G).values()) // 3
    row = dict(N_raw=n_raw, N=N, E=E, kmean=deg.mean(), k_cv=deg.std() / deg.mean(), kmax=int(deg.max()),
               C=nx.transitivity(G), T_G=TG, S=np.nan, f=np.nan)
    if c.tkeys is not None:
        tk = c.tkeys[c.tn >= T_MIN]
        n = c.base
        x, y, z = tk // (n * n), (tk // n) % n, tk % n
        ek = np.array(sorted(min(u, v) * base + max(u, v) for u, v in G.edges()), dtype=np.int64)
        ok = np.isin(x * n + y, ek) & np.isin(x * n + z, ek) & np.isin(y * n + z, ek)
        S = int(ok.sum())
        row.update(S=S, f=S / TG if TG else np.nan)
    return row


ROWS = []


def report(dataset, variant, kind, window, c, W_list=(None,), T_list=(None,), note="", edge_override=None):
    for W in W_list:
        for T in T_list:
            r = lcc_stats(c, W or 1, T or 1, edge_override=edge_override)
            if r is None:
                continue
            ROWS.append(dict(dataset=dataset, variant=variant, kind=kind, window=window,
                             W_MIN=W if W is not None else np.nan,
                             T_MIN=T if (T is not None and not np.isnan(r["S"])) else np.nan, **r, note=note))
            print(f"  {dataset:22s} {variant:24s} W={W} T={T}: N={r['N']} E={r['E']} <k>={r['kmean']:.1f} "
                  f"cv={r['k_cv']:.2f} C={r['C']:.3f} T_G={r['T_G']} S={r['S']} f={r['f']:.3f}", flush=True)


def p(*parts):
    return os.path.join(RAW, *parts)


# ----------------------------------------------------------------------------------------------------------------
# 1. Copenhagen Networks Study（蓝牙 5 分钟共现 + Facebook 好友）
# ----------------------------------------------------------------------------------------------------------------
def do_copenhagen():
    print("Copenhagen ...", flush=True)
    bt = pd.read_csv(p("copenhagen", "bt_symmetric.csv"))
    bt.columns = ["t", "a", "b", "rssi"]
    bt = bt[bt.b >= 0]                                   # -1 = 空扫描, -2 = 实验外设备
    fb = pd.read_csv(p("copenhagen", "fb_friends.csv"))
    fb.columns = ["a", "b"]
    cfb = pairwise_counts(fb.a.values, fb.b.values)
    report("copenhagen_fb", "facebook好友", "pairwise", "", cfb)
    for r_min in (-100, -80, -75):
        d = bt[bt.rssi >= r_min]
        c = window_counts(d.t.values // 300, d.a.values, d.b.values,
                          extra_nodes=np.concatenate([fb.a.values, fb.b.values]))
        var = "BT全部信号" if r_min == -100 else f"BT RSSI>={r_min}dBm"
        report("copenhagen_bt", var, "higher-order", "300s", c, W_LIST, T_LIST)
        # 变体：边 = Facebook 好友；2-单纯形 = 三个好友在同一 5 分钟窗内两两蓝牙共现
        fb_same = _reindex_pairs(fb.a.values, fb.b.values, c)
        report("copenhagen_fb+bt", f"FB边+{var}共现单纯形", "higher-order", "300s", c, (None,), T_LIST,
               edge_override=fb_same)


def _reindex_pairs(a, b, c):
    """把另一组成对边编码到 Counts c 的节点编号里（不在 c 中的节点追加编号）。"""
    lab = {int(v): i for i, v in enumerate(c.labels)}
    nxt = len(lab)
    keys = set()
    for x, y in zip(np.asarray(a).tolist(), np.asarray(b).tolist()):
        if x == y:
            continue
        for v in (x, y):
            if v not in lab:
                lab[v] = nxt; nxt += 1
        u, v = sorted((lab[x], lab[y]))
        keys.add(u * int(c.base) + v)
    if nxt > c.base:
        raise ValueError("节点编号超出 base")
    ek = np.array(sorted(keys), dtype=np.int64)
    return Counts(ek, np.ones(len(ek), np.int64), None, None, c.base, None)


# ----------------------------------------------------------------------------------------------------------------
# 2. SocioPatterns：SFHH 会议面对面接触；Infectious 展览（取人数最多的一天）
# ----------------------------------------------------------------------------------------------------------------
def do_sociopatterns():
    print("SocioPatterns SFHH ...", flush=True)
    with gzip.open(p("sociopatterns_SFHH", "SFHH_tij.dat.gz"), "rt") as fh:
        d = pd.read_csv(fh, sep=r"\s+", header=None, names=["t", "i", "j"])
    c = window_counts(d.t.values // 20, d.i.values, d.j.values)
    report("sociopatterns_SFHH", "面对面20s", "higher-order", "20s", c, W_LIST, T_LIST)

    print("SocioPatterns Infectious ...", flush=True)
    tf = tarfile.open(p("sociopatterns_infectious", "sciencegallery_infectious_contacts.tgz"))
    days = []
    for m in tf.getmembers():
        if m.isfile():
            d = pd.read_csv(tf.extractfile(m), sep="\t", header=None, names=["t", "i", "j"])
            days.append((len(set(d.i) | set(d.j)), os.path.basename(m.name), d))
    days.sort(key=lambda x: -x[0])
    nn, name, d = days[0]                                 # 各天访客 ID 互不相同，取人数最多的一天
    c = window_counts(d.t.values // 20, d.i.values, d.j.values)
    report("sociopatterns_infectious", f"最大单日 {name[13:23]}", "higher-order", "20s", c, W_LIST, T_LIST,
           note=f"共 {len(days)} 天，单日人数 {days[-1][0]}–{nn}")


# ----------------------------------------------------------------------------------------------------------------
# 3. Utah 学校（Toth et al. 2015 JRSI；Leecaster et al. 2016 PLoS ONE），WREN 传感器 ~19.53 s 一次
#    每条记录 (id1, id2, startTime, duration)，时间单位 19.53125 s；覆盖窗 startTime ... startTime+duration-1
# ----------------------------------------------------------------------------------------------------------------
def _expand(d, offset=0):
    rep = d.dur.values.astype(np.int64)
    within = np.arange(rep.sum()) - np.repeat(np.cumsum(rep) - rep, rep)       # 0,1,...,dur-1 for each record
    t = np.repeat(d.t.values.astype(np.int64), rep) + within + offset
    return t, np.repeat(d.i.values, rep), np.repeat(d.j.values, rep)


def _read_toth(k):
    f = p("utah_schools_toth2015", f"rsif20150279supp{k}.txt")
    d = pd.read_csv(f, sep=r"\s+", header=0 if k <= 5 else None)
    d.columns = ["i", "j", "t", "dur"]
    return d


def do_utah():
    print("Utah schools ...", flush=True)
    sets = [("utah_mid1", "2m标准 两天(D1+D2)", (2, 3)), ("utah_mid1", "1m严格标准 两天(D7+D8)", (8, 9)),
            ("utah_elem1", "2m标准 两天(D3+D4)", (4, 5)), ("utah_elem1", "1m严格标准 两天(D9+D10)", (10, 11))]
    for name, var, ks in sets:
        parts = [_expand(_read_toth(k), offset=day * 100000) for day, k in enumerate(ks)]
        t, a, b = (np.concatenate(x) for x in zip(*parts))
        c = window_counts(t, a, b)
        report(name, var, "higher-order", "19.53s", c, W_LIST, T_LIST)
    d = pd.read_csv(p("utah_middle_school", "pone.0153690.s002.csv"))
    d.columns = ["i", "j", "t", "dur"]
    c = window_counts(*_expand(d))
    report("utah_mid_plos2016", "单日(2012-11-28)", "higher-order", "19.53s", c, W_LIST, T_LIST)


# ----------------------------------------------------------------------------------------------------------------
# 4. Haslemere（BBC Pandemic；手机 GPS 两两距离，5 分钟一次，<=50 m）
# ----------------------------------------------------------------------------------------------------------------
def do_haslemere():
    print("Haslemere ...", flush=True)
    h = pd.read_csv(p("haslemere", "Kissler_DataS1.csv"), header=None, names=["t", "i", "j", "d"])
    for dmax in (5, 20):
        x = h[h.d <= dmax]
        c = window_counts(x.t.values, x.i.values, x.j.values)
        report("haslemere", f"距离<={dmax}m", "higher-order", "300s", c, W_LIST, T_LIST)


# ----------------------------------------------------------------------------------------------------------------
# 5. 邮件：Benson email-Eu（单纯形 = 发件人+收件人，<=25 人）；SNAP email-Eu-core；URV email
# ----------------------------------------------------------------------------------------------------------------
def do_email():
    print("email-Eu (Benson) ...", flush=True)
    tf = tarfile.open(p("benson_email-Eu", "email-Eu.tar.gz"))
    nv = np.loadtxt(tf.extractfile("email-Eu/email-Eu-nverts.txt"), dtype=int)
    sx = np.loadtxt(tf.extractfile("email-Eu/email-Eu-simplices.txt"), dtype=int)
    groups = np.split(sx, np.cumsum(nv)[:-1])
    c = window_counts(*groups_to_contacts(groups))
    report("benson_email-Eu", "单纯形<=25人", "higher-order", "每封邮件", c, W_LIST, T_LIST)

    print("email-Eu-core (SNAP) ...", flush=True)
    with gzip.open(p("snap_email-Eu-core", "email-Eu-core.txt.gz"), "rt") as fh:
        d = pd.read_csv(fh, sep=r"\s+", header=None, names=["a", "b"])
    report("snap_email-Eu-core", "无向化", "pairwise", "", pairwise_counts(d.a.values, d.b.values))

    print("URV email ...", flush=True)
    with zipfile.ZipFile(p("urv_email", "email.zip")) as z:
        d = pd.read_csv(io.BytesIO(z.read("email.txt")), sep=r"\s+", header=None, names=["a", "b", "w"])
    report("urv_email", "无向", "pairwise", "", pairwise_counts(d.a.values, d.b.values))


# ----------------------------------------------------------------------------------------------------------------
# 6. 西班牙中学友谊问卷（Ruiz-García et al. 2023 PNAS；13 所学校，边权 -2..+2）
# ----------------------------------------------------------------------------------------------------------------
def do_spanish():
    print("Spanish high schools ...", flush=True)
    files = sorted(f for f in os.listdir(p("spanish_highschools")) if f.startswith("Edges_"))
    for f in files:
        school = f[len("Edges_"):-4]
        d = pd.read_csv(p("spanish_highschools", f))
        d = d[["from", "to", "weight"]]
        pos = d[d.weight > 0]
        report("spanish_highschools", f"{school} 正向边(任一方提名)", "pairwise", "",
               pairwise_counts(pos["from"].values, pos["to"].values))
        s = set(zip(pos["from"], pos["to"]))
        mut = [(x, y) for x, y in s if (y, x) in s and x < y]
        if mut:
            a, b = zip(*mut)
            report("spanish_highschools", f"{school} 正向互选边", "pairwise", "", pairwise_counts(a, b))


# ----------------------------------------------------------------------------------------------------------------
# 7. 参照：本地已有 SocioPatterns 计数（data/socio/*.npz，由 survey_socio.py 等生成；只读，不存在则跳过）
# ----------------------------------------------------------------------------------------------------------------
def do_local():
    if not os.path.isdir(LOCAL_SOCIO):
        return
    print("本地 SocioPatterns npz ...", flush=True)
    for f in sorted(os.listdir(LOCAL_SOCIO)):
        if not f.endswith(".npz"):
            continue
        z = np.load(os.path.join(LOCAL_SOCIO, f))
        E, w, Tr, n = z["edges"], z["w"], z["tris"], z["n"]
        labels, inv = np.unique(np.concatenate([E.ravel(), Tr.ravel()]), return_inverse=True)
        base = np.int64(len(labels))
        e = inv[:E.size].reshape(-1, 2).astype(np.int64)
        tr = np.sort(inv[E.size:].reshape(-1, 3).astype(np.int64), axis=1)
        e = np.sort(e, axis=1)
        c = Counts(e[:, 0] * base + e[:, 1], w.astype(np.int64),
                   (tr[:, 0] * base + tr[:, 1]) * base + tr[:, 2], n.astype(np.int64), base, labels)
        kind = "higher-order(本地,同房间共现)" if "colocation" in f else "higher-order(本地)"
        report("local:" + f[:-4], "已有数据", kind, "20s", c, W_LIST, T_LIST)


if __name__ == "__main__":
    t0 = time.time()
    for fn in (do_sociopatterns, do_utah, do_haslemere, do_email, do_spanish, do_copenhagen, do_local):
        fn()
        pd.DataFrame(ROWS).to_csv(OUT, index=False, encoding="utf-8-sig")
    out = pd.DataFrame(ROWS)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    print(out.round(3).to_string(index=False))
    print(f"写出 {OUT}，用时 {time.time() - t0:.0f} s")
