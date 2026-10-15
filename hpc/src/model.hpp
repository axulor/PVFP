// model.hpp -- C++ port of code/kernel.py and code/bstar.py (design doc v5, section 2 model).
//
// Model: on every edge {i,j} a two-player public goods game; p_ij = 1/k_i; production g(t) (delta = 1: g(t) = t);
// share m_ij = 1/2 (uniform) or 1/2 + [g(p_ij) - g(p_ji)] / [2 g(p_ij + p_ji)] (Shapley); endowment c = 1.
// Death-birth: a uniformly chosen active individual i copies neighbour j with probability ~ exp(theta * v_{j->i}),
//   v_{j->i} = u_ji + eps1 * sum_{l in Omega_ij} u_jl + eps2 * (u_j - u_ji - sum_{l in Omega_ij} u_jl),
// Omega_ij = common neighbours of i and j.
// The code mirrors the numba kernel statement by statement (same xorshift64* streams, same seeds, same order of
// operations), so that a task gives the same counts and the same b* as the Python version.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

// Parallelism: the runs inside one call of fixation_counts are distributed over OpenMP threads.  Every run has its
// own random stream (seed, run index) and the result is an integer count, so the outcome does not depend on the
// number of threads or on the schedule: results stay bit-identical to the serial and Python versions.

namespace model {

// ------------------------------------------------------------------------------------------------ network
struct Net {
    int64_t N = 0;
    std::vector<int64_t> indptr, nbr, rev, optr, oe;   // CSR; rev[a] = index of the reverse edge; Omega lists
    std::vector<double> k;
    int64_t kmax = 0;
};

// edge file: lines starting with '#' are comments; first data line "N E"; then E lines "u v" (0-based, u != v)
inline Net load_net(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open network file " + path);
    std::string line;
    int64_t N = -1, E = -1;
    std::vector<std::vector<int64_t>> adj;
    int64_t read = 0;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream is(line);
        if (N < 0) {
            is >> N >> E;
            adj.assign(N, {});
            continue;
        }
        int64_t u, v;
        if (!(is >> u >> v)) continue;
        if (u < 0 || v < 0 || u >= N || v >= N || u == v) throw std::runtime_error("bad edge in " + path);
        adj[u].push_back(v);
        adj[v].push_back(u);
        ++read;
    }
    if (read != E) throw std::runtime_error("edge count mismatch in " + path);
    Net g;
    g.N = N;
    for (auto& a : adj) std::sort(a.begin(), a.end());
    g.indptr.assign(N + 1, 0);
    for (int64_t i = 0; i < N; ++i) g.indptr[i + 1] = g.indptr[i] + (int64_t)adj[i].size();
    g.nbr.reserve(g.indptr[N]);
    for (int64_t i = 0; i < N; ++i)
        for (auto j : adj[i]) g.nbr.push_back(j);
    g.rev.assign(g.nbr.size(), -1);
    for (int64_t i = 0; i < N; ++i)
        for (int64_t a = g.indptr[i]; a < g.indptr[i + 1]; ++a) {
            int64_t j = g.nbr[a];
            auto it = std::lower_bound(adj[j].begin(), adj[j].end(), i);
            g.rev[a] = g.indptr[j] + (it - adj[j].begin());
        }
    // Omega lists: for directed edge a = (i -> j), the directed edges (j -> l) with l a common neighbour of i and j
    g.optr.assign(g.nbr.size() + 1, 0);
    for (int64_t i = 0; i < N; ++i)
        for (int64_t a = g.indptr[i]; a < g.indptr[i + 1]; ++a) {
            int64_t j = g.nbr[a];
            for (size_t t = 0; t < adj[j].size(); ++t) {
                int64_t l = adj[j][t];
                if (l != i && std::binary_search(adj[i].begin(), adj[i].end(), l)) g.oe.push_back(g.indptr[j] + (int64_t)t);
            }
            g.optr[a + 1] = (int64_t)g.oe.size();
        }
    g.k.resize(N);
    for (int64_t i = 0; i < N; ++i) {
        g.k[i] = (double)adj[i].size();
        g.kmax = std::max<int64_t>(g.kmax, (int64_t)adj[i].size());
    }
    return g;
}

// ------------------------------------------------------------------------------------------------ payoff table
inline double gfun(double t, double delta) { return delta == 1.0 ? t : (std::pow(delta, t) - 1.0) / (delta - 1.0); }

// U[4*a + 2*s_i + s_j] = payoff to i from edge a = (i -> j)
inline std::vector<double> table(const Net& g, double b, const std::string& rule, double delta, double c = 1.0) {
    std::vector<double> U(4 * g.nbr.size());
    for (int64_t i = 0; i < g.N; ++i)
        for (int64_t a = g.indptr[i]; a < g.indptr[i + 1]; ++a) {
            double pij = 1.0 / g.k[i], pji = 1.0 / g.k[g.nbr[a]];
            double m;
            if (rule == "U") m = 0.5;
            else if (rule == "Sh") m = 0.5 + (gfun(pij, delta) - gfun(pji, delta)) / (2.0 * gfun(pij + pji, delta));
            else if (rule == "E") m = pij / (pij + pji);
            else throw std::runtime_error("unknown rule " + rule);
            U[4 * a + 0] = pij * c;
            U[4 * a + 1] = m * b * gfun(pji, delta) + pij * c;
            U[4 * a + 2] = m * b * gfun(pij, delta);
            U[4 * a + 3] = m * b * gfun(pij + pji, delta);
        }
    return U;
}

// ------------------------------------------------------------------------------------------------ random numbers
inline uint64_t rnext(uint64_t& x) {
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    return (x * 2685821657736338717ULL) >> 11;
}
inline double runif(uint64_t& x) { return (double)rnext(x) * (1.0 / 9007199254740992.0); }
inline uint64_t rseed(uint64_t seed, uint64_t run) {
    uint64_t z = seed * 0x9E3779B97F4A7C15ULL + run * 0xBF58476D1CE4E5B9ULL + 1ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    if (z == 0ULL) z = 88172645463325252ULL;
    return z;
}

// ------------------------------------------------------------------------------------------------ one run
struct Work {
    std::vector<int64_t> s, dc, act, pos;
    std::vector<double> u, w;
    explicit Work(const Net& g) : s(g.N), dc(g.N), act(g.N), pos(g.N), u(g.N), w(std::max<int64_t>(g.kmax, 1)) {}
};

// run until absorption from the state in W; return the final number of cooperators
inline int64_t one_run(const Net& g, const std::vector<double>& U, double eps1, double eps2, double theta, Work& W, uint64_t& rs) {
    const int64_t N = g.N;
    auto& s = W.s; auto& u = W.u; auto& dc = W.dc; auto& act = W.act; auto& pos = W.pos; auto& w = W.w;
    // with eps1 = eps2 = 0 the third-party term is exactly +0 (0*h = +0, 0*x = +-0), so v = uji bit for bit:
    // the common-neighbour sum h can be skipped without changing any result
    const bool need_h = !(eps1 == 0.0 && eps2 == 0.0);
    int64_t nC = 0;
    for (int64_t x = 0; x < N; ++x) nC += s[x];
    int64_t nact = 0;
    for (int64_t x = 0; x < N; ++x) {
        if (dc[x] > 0) { pos[x] = nact; act[nact] = x; ++nact; }
        else pos[x] = -1;
    }
    while (nact > 0) {
        int64_t i = act[(int64_t)(runif(rs) * (double)nact)];
        int64_t si = s[i];
        int64_t a0 = g.indptr[i], a1 = g.indptr[i + 1];
        double vmax = -1e300;
        for (int64_t a = a0; a < a1; ++a) {
            int64_t j = g.nbr[a];
            int64_t sj = s[j];
            double uji = U[4 * g.rev[a] + 2 * sj + si];
            double h = 0.0;
            if (need_h)
                for (int64_t q = g.optr[a]; q < g.optr[a + 1]; ++q) {
                    int64_t e = g.oe[q];
                    h += U[4 * e + 2 * sj + s[g.nbr[e]]];
                }
            double v = uji + eps1 * h + eps2 * (u[j] - uji - h);
            w[a - a0] = v;
            if (v > vmax) vmax = v;
        }
        double tot = 0.0;
        for (int64_t t = 0; t < a1 - a0; ++t) {
            w[t] = std::exp(theta * (w[t] - vmax));
            tot += w[t];
        }
        double r = runif(rs) * tot;
        double acc = 0.0;
        int64_t pick = a1 - 1;
        for (int64_t t = 0; t < a1 - a0; ++t) {
            acc += w[t];
            if (r < acc) { pick = a0 + t; break; }
        }
        int64_t ns = s[g.nbr[pick]];
        if (ns == si) continue;
        s[i] = ns;
        nC += (ns == 1) ? 1 : -1;
        double ui = 0.0;
        for (int64_t a = a0; a < a1; ++a) {
            int64_t y = g.nbr[a];
            int64_t sy = s[y];
            ui += U[4 * a + 2 * ns + sy];
            int64_t eyi = g.rev[a];
            u[y] += U[4 * eyi + 2 * sy + ns] - U[4 * eyi + 2 * sy + si];
            if (sy == ns) dc[y] -= 1;
            else dc[y] += 1;
            if (dc[y] > 0 && pos[y] < 0) { pos[y] = nact; act[nact] = y; ++nact; }
            else if (dc[y] == 0 && pos[y] >= 0) {
                int64_t p = pos[y], last = act[nact - 1];
                act[p] = last; pos[last] = p; pos[y] = -1; --nact;
            }
        }
        u[i] = ui;
        dc[i] = (a1 - a0) - dc[i];
        if (dc[i] > 0 && pos[i] < 0) { pos[i] = nact; act[nact] = i; ++nact; }
        else if (dc[i] == 0 && pos[i] >= 0) {
            int64_t p = pos[i], last = act[nact - 1];
            act[p] = last; pos[last] = p; pos[i] = -1; --nact;
        }
    }
    return nC;
}

// One count job: `runs` runs in which a single mutant of type 1-resident, placed at node (run mod N), invades a
// population of residents; the job counts the runs in which the mutant fixes.  Run `run` of a job uses the random
// stream rseed(seed, run).
struct CountJob {
    const std::vector<double>* U;
    int64_t resident;
    uint64_t seed;
};

// 2026-10-08: all jobs of one call share ONE parallel region (one loop over job x run).  Before, every job was its
// own region, and a refine round (5 points x 4 batches x 2 residents = 40 jobs) waited 40 times for its slowest run;
// with 120 threads this made one task only 4.2x faster than with 8.  Every run still uses its own stream and the
// counts are integer sums, so the results are bit-identical to the one-region-per-job version (and to Python).
// Ws: one work buffer per OpenMP thread (Ws.size() >= number of threads).
// ishare / nshare (wide mode, 2026-10-08): this process does only the runs t = ishare, ishare + nshare, ... of the
// job x run loop; the counts of all processes are then summed (MPI_Allreduce), which again gives the same integers.
inline std::vector<int64_t> fixation_counts(const Net& g, const std::vector<CountJob>& jobs, double eps1, double eps2,
                                            double theta, int64_t runs, std::vector<Work>& Ws, int64_t ishare = 0,
                                            int64_t nshare = 1) {
    const int64_t N = g.N, J = (int64_t)jobs.size();
    std::vector<std::vector<double>> u0(J, std::vector<double>(N, 0.0));
    for (int64_t q = 0; q < J; ++q) {
        const std::vector<double>& U = *jobs[q].U;
        for (int64_t x = 0; x < N; ++x)
            for (int64_t a = g.indptr[x]; a < g.indptr[x + 1]; ++a) u0[q][x] += U[4 * a + 3 * jobs[q].resident];
    }
    std::vector<int64_t> cnt(J, 0);
    const int64_t total = J * runs;
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#ifdef _OPENMP
        Work& W = Ws[omp_get_thread_num()];
#else
        Work& W = Ws[0];
#endif
        std::vector<int64_t> loc(J, 0);
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 8) nowait
#endif
        for (int64_t t = ishare; t < total; t += nshare) {
            const int64_t q = t / runs, run = t % runs;
            const std::vector<double>& U = *jobs[q].U;
            const int64_t resident = jobs[q].resident;
            uint64_t rs = rseed(jobs[q].seed, (uint64_t)run);
            for (int64_t x = 0; x < N; ++x) { W.s[x] = resident; W.u[x] = u0[q][x]; W.dc[x] = 0; }
            int64_t m = run % N, mut = 1 - resident;
            W.s[m] = mut;
            double um = 0.0;
            for (int64_t a = g.indptr[m]; a < g.indptr[m + 1]; ++a) {
                int64_t y = g.nbr[a];
                um += U[4 * a + 2 * mut + resident];
                int64_t e = g.rev[a];
                W.u[y] += U[4 * e + 2 * resident + mut] - U[4 * e + 3 * resident];
                W.dc[y] = 1;
            }
            W.u[m] = um;
            W.dc[m] = g.indptr[m + 1] - g.indptr[m];
            int64_t nC = one_run(g, U, eps1, eps2, theta, W, rs);
            if ((mut == 1 && nC == N) || (mut == 0 && nC == 0)) ++loc[q];
        }
#ifdef _OPENMP
#pragma omp critical
#endif
        for (int64_t q = 0; q < J; ++q) cnt[q] += loc[q];
    }
    return cnt;
}

// ------------------------------------------------------------------------------------------------ b* estimator
struct Result {
    double bstar = std::numeric_limits<double>::quiet_NaN();
    double se = std::numeric_limits<double>::infinity();
    int64_t runs_per_point = 0, calls = 0;
    int attempt = 0;                    // refine attempt (window widenings) at which the estimate was accepted
    std::string flag = "";
};

inline void linfit(const std::vector<double>& x, const std::vector<double>& y, double& slope, double& icpt) {
    double n = (double)x.size(), mx = 0, my = 0;
    for (size_t t = 0; t < x.size(); ++t) { mx += x[t]; my += y[t]; }
    mx /= n; my /= n;
    double sxy = 0, sxx = 0;
    for (size_t t = 0; t < x.size(); ++t) { sxy += (x[t] - mx) * (y[t] - my); sxx += (x[t] - mx) * (x[t] - mx); }
    slope = sxy / sxx;
    icpt = my - slope * mx;
}

// state of refine at the start of a round (checkpoint, 2026-10-08).  Restoring it and continuing gives exactly the
// result of an uninterrupted run: every later seed depends only on (attempt, batch, point, runs) and the counts add up.
struct Snapshot {
    bool valid = false;
    int64_t calls = 0;
    int attempt = 0;
    double b0 = 0, h = 0;
    int64_t runs = 0, R = 0;
    std::vector<double> fc, fd;          // nbatch x npoints, row-major (integer-valued)
};

class Estimator {
public:
    static constexpr double B_MAX = 30.0;
    Estimator(const Net& g, const std::string& rule, double e1, double e2, double theta, double delta, int64_t seed)
        : g_(g), rule_(rule), e1_(e1), e2_(e2), th_(theta), de_(delta), seed_(seed) {
        int nt = 1;
#ifdef _OPENMP
        nt = omp_get_max_threads();
#endif
        for (int t = 0; t < nt; ++t) W_.emplace_back(g);
    }

    // run cap of refine (total runs per point over the batches).  The estimate is accepted at the first doubling with
    // at least rcap runs, so the default 1.6e6 stops at 2.56e6 (E1-E3a); set with BSTAR_RCAP in bstar_mpi.
    void set_rcap(int64_t r) { rcap_ = r; }
    // wide mode: this process does the share ishare of nshare of every call; `reduce` sums the counts of all processes
    void set_share(int64_t ishare, int64_t nshare, std::function<void(std::vector<int64_t>&)> reduce) {
        ish_ = ishare; nsh_ = nshare; reduce_ = std::move(reduce);
    }
    // checkpoint: `save` is called with the state at the start of every refine round (may be empty); `resume`, if
    // valid, is a state saved earlier for this task: bracket and the first two refine calls are skipped
    void set_checkpoint(std::function<void(const Snapshot&)> save, const Snapshot* resume) {
        save_ = std::move(save); resume_ = (resume && resume->valid) ? resume : nullptr;
    }

    std::vector<int64_t> count_jobs(const std::vector<CountJob>& jobs, int64_t runs) {
        if (nsh_ > 1 && !reduce_) {                           // local test of the wide mode: all shares here, summed
            std::vector<int64_t> c(jobs.size(), 0);
            for (int64_t q = 0; q < nsh_; ++q) {
                std::vector<int64_t> cq = fixation_counts(g_, jobs, e1_, e2_, th_, runs, W_, q, nsh_);
                for (size_t z = 0; z < c.size(); ++z) c[z] += cq[z];
            }
            return c;
        }
        std::vector<int64_t> c = fixation_counts(g_, jobs, e1_, e2_, th_, runs, W_, ish_, nsh_);
        if (reduce_) reduce_(c);
        return c;
    }

    // one "call": cooperator invasion (resident 0) and defector invasion (resident 1) at benefit b
    void counts(double b, int64_t runs, int64_t batch_seed, int64_t& fc, int64_t& fd) {
        std::vector<double> U = table(g_, b, rule_, de_);
        std::vector<CountJob> jobs = {{&U, 0, (uint64_t)batch_seed}, {&U, 1, (uint64_t)(batch_seed + 7919)}};
        std::vector<int64_t> c = count_jobs(jobs, runs);
        fc = c[0]; fd = c[1];
        ++calls_;
    }
    double D(double b, int64_t runs, int64_t salt) {
        int64_t fc, fd;
        counts(b, runs, seed_ * 1000003 + salt, fc, fd);
        return (double)g_.N * ((double)fc / runs - (double)fd / runs);
    }

    // returns false in lo_ok / hi_ok when the bound is missing (Python None)
    void bracket(double guess, double& lo, bool& lo_ok, double& hi, bool& hi_ok, int64_t R0 = 10000) {
        lo_ok = hi_ok = true;
        if (std::isfinite(guess) && guess > 0) {
            lo = hi = guess;
            double d0 = D(guess, R0, 1);
            if (d0 < 0) {
                hi = guess;
                for (int it = 0; it < 12; ++it) {
                    lo = hi; hi = hi * 1.12;
                    if (hi > B_MAX) break;
                    if (D(hi, R0, 2) > 0) return;
                }
            } else {
                lo = guess;
                for (int it = 0; it < 12; ++it) {
                    hi = lo; lo = lo / 1.12;
                    if (D(lo, R0, 3) < 0) return;
                }
            }
        }
        const int NG = 25;
        double l0 = std::log10(0.3), l1 = std::log10(B_MAX), step = (l1 - l0) / (NG - 1);
        bool have_prev = false;
        double pb = 0, pd = 0;
        for (int t = 0; t < NG; ++t) {
            double b = (t == 0) ? 0.3 : (t == NG - 1) ? B_MAX : std::pow(10.0, l0 + t * step);
            double d = D(b, R0, 10 + t);
            if (d > 0 && have_prev && pd < 0) { lo = pb; hi = b; return; }
            if (d > 0 && !have_prev) { lo_ok = false; hi = b; return; }
            have_prev = true; pb = b; pd = d;
        }
        lo = pb; hi_ok = false;
    }

    Result refine(double lo, double hi, double target, int64_t R1 = 10000, int64_t Rcap = 1600000, int nbatch = 4,
                  int64_t salt0 = 0, int max_shift = 12) {
        const double N = (double)g_.N;
        const double NaN = std::numeric_limits<double>::quiet_NaN(), INF = std::numeric_limits<double>::infinity();
        double b0, h;
        const Snapshot* rs = resume_;                          // resume a checkpointed task at the saved round
        if (rs) { b0 = rs->b0; h = rs->h; }
        else {
            double dlo = D(lo, R1, 21 + salt0), dhi = D(hi, R1, 22 + salt0);
            b0 = (dhi != dlo) ? lo + (hi - lo) * (-dlo) / (dhi - dlo) : 0.5 * (lo + hi);
            // 2026-10-07: the two moderate-precision values disagreed with the bracket (both of one sign) and the
            // interpolation threw the window far outside it: start from the middle of the bracket instead
            if (!(lo - (hi - lo) <= b0 && b0 <= hi + (hi - lo))) b0 = 0.5 * (lo + hi);
            h = std::max(0.015 * b0, 0.25 * (hi - lo));
        }
        // 2026-10-07 (E2): the window is never widened.  A root extrapolated from a window whose five points are all
        // significantly on one side of zero is never accepted; such a window is moved by its own width.  If the root
        // falls outside the window otherwise, more runs are added to the same window.  (Widening let the straight-
        // line fit span the bend of D(b) -- steep below the root, nearly flat above it at large visibility weights --
        // and biased the root while the SE stayed small.)  Mirrors code/bstar.py; tasks whose root fell inside the
        // first window are unaffected.
        double root = NaN, se = INF, sl = 0, ic = 0;
        int64_t runs = 0;
        const double mult[5] = {-2, -1, 0, 1, 2};
        int attempt = rs ? rs->attempt : 0;
        for (; attempt <= max_shift; ++attempt) {
            std::vector<double> pts;
            for (double mm : mult) { double p = b0 + h * mm / 2.0; if (p > 0) pts.push_back(p); }
            size_t np = pts.size();
            std::vector<std::vector<double>> fc(nbatch, std::vector<double>(np, 0.0)), fd = fc;
            std::vector<std::vector<double>> Ut(np);                  // payoff tables of the window (fixed per window)
            for (size_t j = 0; j < np; ++j) Ut[j] = table(g_, pts[j], rule_, de_);
            runs = 0;
            int64_t R = R1;
            if (rs) {                                                 // restore the saved round (first window only)
                if (rs->fc.size() != (size_t)nbatch * np || rs->fd.size() != (size_t)nbatch * np)
                    throw std::runtime_error("checkpoint does not match the window");
                runs = rs->runs; R = rs->R;
                for (int kk = 0; kk < nbatch; ++kk)
                    for (size_t j = 0; j < np; ++j) { fc[kk][j] = rs->fc[kk * np + j]; fd[kk][j] = rs->fd[kk * np + j]; }
                rs = nullptr;
            }
            bool moved = false;
            while (true) {
                if (save_) {                                          // checkpoint: state at the start of the round
                    Snapshot sn; sn.valid = true; sn.calls = calls_; sn.attempt = attempt; sn.b0 = b0; sn.h = h;
                    sn.runs = runs; sn.R = R;
                    for (int kk = 0; kk < nbatch; ++kk)
                        for (size_t j = 0; j < np; ++j) { sn.fc.push_back(fc[kk][j]); sn.fd.push_back(fd[kk][j]); }
                    save_(sn);
                }
                // one round: the same calls (seeds) as before, all in one parallel region (see fixation_counts)
                std::vector<CountJob> jobs;
                for (int kk = 0; kk < nbatch; ++kk)
                    for (size_t j = 0; j < np; ++j) {
                        int64_t bs = seed_ * 7777 + 1000 * attempt + 97 * kk + (int64_t)j + 13 * runs + salt0;
                        jobs.push_back({&Ut[j], 0, (uint64_t)bs});
                        jobs.push_back({&Ut[j], 1, (uint64_t)(bs + 7919)});
                    }
                std::vector<int64_t> c = count_jobs(jobs, R);
                for (int kk = 0; kk < nbatch; ++kk)
                    for (size_t j = 0; j < np; ++j) {
                        size_t q = 2 * ((size_t)kk * np + j);
                        fc[kk][j] += (double)c[q]; fd[kk][j] += (double)c[q + 1];
                    }
                calls_ += (int64_t)nbatch * (int64_t)np;
                runs += R;
                std::vector<std::vector<double>> Dk(nbatch, std::vector<double>(np));
                for (int kk = 0; kk < nbatch; ++kk)
                    for (size_t j = 0; j < np; ++j) Dk[kk][j] = N * (fc[kk][j] - fd[kk][j]) / (double)runs;
                std::vector<double> roots;
                for (int kk = 0; kk < nbatch; ++kk) {
                    double s1, i1; linfit(pts, Dk[kk], s1, i1);
                    roots.push_back(s1 > 0 ? -i1 / s1 : NaN);
                }
                std::vector<double> Dall(np, 0.0);
                for (size_t j = 0; j < np; ++j) {
                    for (int kk = 0; kk < nbatch; ++kk) Dall[j] += Dk[kk][j];
                    Dall[j] /= nbatch;
                }
                linfit(pts, Dall, sl, ic);
                root = sl > 0 ? -ic / sl : NaN;
                std::vector<double> fin;
                for (double r : roots) if (std::isfinite(r)) fin.push_back(r);
                if (fin.size() > 1) {
                    double m = 0; for (double r : fin) m += r; m /= fin.size();
                    double ss = 0; for (double r : fin) ss += (r - m) * (r - m);
                    se = std::sqrt(ss / (fin.size() - 1)) / std::sqrt((double)fin.size());
                } else se = INF;
                bool inside = std::isfinite(root) && pts.front() - h <= root && root <= pts.back() + h;
                const int64_t tot = runs * nbatch;
                bool below = true, above = true;              // whole window significantly below / above the root
                for (size_t j = 0; j < np; ++j) {
                    double sc = 0, sd = 0;
                    for (int kk = 0; kk < nbatch; ++kk) { sc += fc[kk][j]; sd += fd[kk][j]; }
                    double pc = sc / (double)tot, pd = sd / (double)tot;
                    double seD = N * std::sqrt((pc * (1 - pc) + pd * (1 - pd)) / (double)tot);
                    if (!(Dall[j] + 2 * seD < 0)) below = false;
                    if (!(Dall[j] - 2 * seD > 0)) above = false;
                }
                above = above && (b0 - 2 * h > 0);
                if (inside && !(below || above) && (se <= target * root || tot >= Rcap)) {
                    Result out; out.bstar = root; out.se = se; out.runs_per_point = tot; out.attempt = attempt;
                    return out;
                }
                if (below) { b0 = b0 + 2 * h; moved = true; break; }
                if (above) { b0 = b0 - 2 * h; moved = true; break; }
                if (!inside && tot >= Rcap) {
                    Result out; out.bstar = NaN; out.se = INF; out.runs_per_point = tot; out.attempt = attempt;
                    out.flag = "unconverged";
                    return out;
                }
                R = runs;                                     // double the runs, same window
            }
            (void)moved;
        }
        Result out; out.bstar = NaN; out.se = INF; out.runs_per_point = runs * nbatch; out.attempt = attempt;
        out.flag = "unconverged";
        return out;
    }

    Result estimate(double guess, double target) {
        if (resume_) {                                        // checkpointed task: continue refine where it stopped
            calls_ = resume_->calls;
            Result out = refine(0.0, 0.0, target, 10000, rcap_);
            out.calls = calls_;
            return out;
        }
        double lo, hi; bool lo_ok, hi_ok;
        bracket(guess, lo, lo_ok, hi, hi_ok);
        if (!hi_ok) {
            double d = D(B_MAX, 200000, 99);
            if (d <= 0) {
                Result out; out.bstar = std::numeric_limits<double>::infinity(); out.se = 0; out.flag = "never_favoured";
                out.calls = calls_; return out;
            }
            hi = B_MAX;
        }
        if (!lo_ok) {
            lo = hi / 1.5;
            while (D(lo, 20000, 98) > 0 && lo > 1e-3) lo /= 1.5;
        }
        Result out = refine(lo, hi, target, 10000, rcap_);
        out.calls = calls_;
        return out;
    }

private:
    const Net& g_;
    std::string rule_;
    double e1_, e2_, th_, de_;
    int64_t seed_;
    std::vector<Work> W_;
    int64_t calls_ = 0;
    int64_t rcap_ = 1600000;
    int64_t ish_ = 0, nsh_ = 1;
    std::function<void(std::vector<int64_t>&)> reduce_;
    std::function<void(const Snapshot&)> save_;
    const Snapshot* resume_ = nullptr;
};

}  // namespace model
