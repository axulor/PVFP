# -*- coding: utf-8 -*-
"""
bstar.py -- Monte Carlo estimate of the critical benefit b* (section 3, "仿真方法").

D(b) = N (rho_C - rho_D) is estimated by kernel.fixation_counts.  The estimate of b* proceeds in two steps:
  1. bracket: starting from a guess (or a coarse log-grid), find b_lo < b_hi with D(b_lo) < 0 < D(b_hi) using
     R0 runs per point (C and D each);
  2. refine: 5 equally spaced points around the interpolated root, 4 independent batches; in each batch a straight
     line is fitted to D(b) and its root taken; b* = root of the pooled fit, SE = sd(batch roots)/2.
     Runs are added until SE/b* <= target or the run cap is reached.
If D <= 0 for every b up to B_MAX (checked with 10x the runs), b* = inf.
"""
import numpy as np
import numba as nb
from kernel import fixation_counts

B_MAX = 30.0


class Estimator:
    def __init__(self, cx, rule, eps1, eps2, theta, delta=1.0, seed=1, nchunks=None):
        self.cx, self.rule, self.e1, self.e2, self.theta, self.delta = cx, rule, float(eps1), float(eps2), float(theta), delta
        self.seed = seed
        self.nchunks = nchunks or max(1, nb.get_num_threads() * 4)
        self.cache = {}                                      # b -> [fc, fd, runs] accumulated over batches
        self.calls = 0

    def counts(self, b, runs, batch_seed):
        cx = self.cx
        U = cx.table(b, self.rule, self.delta)
        fc = fixation_counts(cx.indptr, cx.nbr, cx.rev, U, cx.optr, cx.oe, self.e1, self.e2, self.theta, 0, runs,
                             batch_seed, self.nchunks)
        fd = fixation_counts(cx.indptr, cx.nbr, cx.rev, U, cx.optr, cx.oe, self.e1, self.e2, self.theta, 1, runs,
                             batch_seed + 7919, self.nchunks)
        self.calls += 1
        return fc, fd

    def D(self, b, runs, salt=0):
        fc, fd = self.counts(b, runs, self.seed * 1000003 + salt)
        N = self.cx.N
        rc, rd = fc / runs, fd / runs
        return N * (rc - rd), N * np.sqrt((rc * (1 - rc) + rd * (1 - rd)) / runs)

    # ---------------------------------------------------------------- step 1
    def bracket(self, guess=None, R0=10000):
        if guess is not None and np.isfinite(guess) and guess > 0:
            lo = hi = guess
            d0, _ = self.D(guess, R0, 1)
            if d0 < 0:
                hi = guess
                for _ in range(12):
                    lo, hi = hi, hi * 1.12
                    if hi > B_MAX:
                        break
                    if self.D(hi, R0, 2)[0] > 0:
                        return lo, hi
            else:
                lo = guess
                for _ in range(12):
                    hi, lo = lo, lo / 1.12
                    if self.D(lo, R0, 3)[0] < 0:
                        return lo, hi
        grid = np.geomspace(0.3, B_MAX, 25)
        prev = None
        for t, b in enumerate(grid):
            d, _ = self.D(b, R0, 10 + t)
            if d > 0 and prev is not None and prev[1] < 0:
                return prev[0], b
            if d > 0 and prev is None:
                return None, b                                    # positive already at the lowest b
            prev = (b, d)
        return prev[0], None                                      # never positive

    # ---------------------------------------------------------------- step 2
    def refine(self, lo, hi, target=0.002, R1=10000, Rcap=1_600_000, nb_=4, salt0=0, max_shift=12):
        # initial root by interpolation at moderate precision
        dlo, _ = self.D(lo, R1, 21 + salt0)
        dhi, _ = self.D(hi, R1, 22 + salt0)
        b0 = lo + (hi - lo) * (-dlo) / (dhi - dlo) if dhi != dlo else 0.5 * (lo + hi)
        if not (lo - (hi - lo) <= b0 <= hi + (hi - lo)):
            # 2026-10-07: the two moderate-precision values disagreed with the bracket (both of one sign), and the
            # interpolation threw the window far outside it; start from the middle of the bracket instead
            b0 = 0.5 * (lo + hi)
        h = max(0.015 * b0, 0.25 * (hi - lo))
        # 2026-10-07 (E2): the window is never widened.  If the fitted root falls outside the window, the window is
        # moved by its own width only when all five points lie significantly on one side of zero; otherwise more
        # runs are added to the same window.  (Widening let the straight-line fit span the bend of D(b) -- steep
        # below the root, nearly flat above it at large visibility weights -- and biased the root while the SE stayed
        # small.)  Tasks whose root fell inside the first window are unaffected (identical results).
        N = self.cx.N
        attempt = 0
        while attempt <= max_shift:
            pts = b0 + h * np.array([-2, -1, 0, 1, 2]) / 2.0
            pts = pts[pts > 0]
            fc = np.zeros((nb_, len(pts))); fd = np.zeros((nb_, len(pts)))
            runs = 0
            R = R1
            while True:
                for k in range(nb_):
                    for j, b in enumerate(pts):
                        c1, c2 = self.counts(b, R, self.seed * 7777 + 1000 * attempt + 97 * k + j + 13 * runs + salt0)
                        fc[k, j] += c1; fd[k, j] += c2
                runs += R
                Dk = N * (fc - fd) / runs                            # batch x points
                roots = []
                for k in range(nb_):
                    s, a = np.polyfit(pts, Dk[k], 1)
                    roots.append(-a / s if s > 0 else np.nan)
                Dall = Dk.mean(0)
                s, a = np.polyfit(pts, Dall, 1)
                root = -a / s if s > 0 else np.nan
                roots = np.array(roots)
                se = np.nanstd(roots, ddof=1) / np.sqrt(np.sum(np.isfinite(roots))) if np.sum(np.isfinite(roots)) > 1 else np.inf
                inside = np.isfinite(root) and pts[0] - h <= root <= pts[-1] + h
                tot = runs * nb_
                pc = fc.sum(0) / tot; pd_ = fd.sum(0) / tot
                seD = N * np.sqrt((pc * (1 - pc) + pd_ * (1 - pd_)) / tot)
                below = np.all(Dall + 2 * seD < 0)                    # whole window significantly below the root
                above = np.all(Dall - 2 * seD > 0) and b0 - 2 * h > 0 # ... significantly above it
                if inside and not (below or above) and (se <= target * root or tot >= Rcap):
                    return dict(bstar=root, se=se, runs_per_point=tot, h=h, slope=s, points=len(pts), attempt=attempt)
                if below:                                             # never accept a root extrapolated from a
                    b0 = b0 + 2 * h                                   # one-sided window: move it by its width
                    break
                if above:
                    b0 = b0 - 2 * h
                    break
                if not inside:
                    if tot >= Rcap:
                        return dict(bstar=np.nan, se=np.inf, runs_per_point=tot, h=h, slope=s, points=len(pts),
                                    attempt=attempt, flag="unconverged")
                R = runs                                              # double the runs, same window
            attempt += 1
        return dict(bstar=np.nan, se=np.inf, runs_per_point=runs * nb_, h=h, slope=s, points=len(pts), attempt=attempt,
                    flag="unconverged")

    def estimate(self, guess=None, target=0.002):
        lo, hi = self.bracket(guess)
        if hi is None:
            d, se = self.D(B_MAX, 200000, 99)
            if d <= 0:
                return dict(bstar=np.inf, se=0.0, runs_per_point=0, flag="never favoured")
            lo, hi = lo, B_MAX
        if lo is None:
            lo = hi / 1.5
            while self.D(lo, 20000, 98)[0] > 0 and lo > 1e-3:
                lo /= 1.5
        out = self.refine(lo, hi, target)
        out["calls"] = self.calls
        return out
