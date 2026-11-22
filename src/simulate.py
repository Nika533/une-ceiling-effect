"""Parallel-test (retest) simulation under a binomial error model with an empirical-Bayes true-score prior.

For each (year, subject, version):
  * true proportion-correct p has an unknown distribution G; G is estimated by nonparametric maximum
    likelihood (Kiefer-Wolfowitz NPMLE, EM on a grid) from the observed raw-score counts, assuming
    raw | p ~ Binomial(n_max, p). The fitted marginal reproduces the observed distribution.
  * For a student with observed raw x, p is drawn from the posterior G(p | x); a retest raw score is
    drawn as Binomial(L * n_max, p) / L (rounded). L = 1 is the actual test length; L = 2 or 4 mimic a test
    with 2x or 4x as many items, i.e. 1/sqrt(L) of the measurement error.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

GRID = np.linspace(0.001, 0.999, 500)


def npmle(counts: np.ndarray, iters: int = 3000) -> np.ndarray:
    """Mixing weights on GRID for a binomial mixture fitted to counts[x], x = 0..n."""
    n = len(counts) - 1
    x = np.arange(n + 1)
    lik = stats.binom.pmf(x[:, None], n, GRID[None, :])  # (n+1, K)
    w = np.full(len(GRID), 1 / len(GRID))
    N = counts.sum()
    for _ in range(iters):
        px = lik @ w
        w = w * (lik.T @ (counts / np.maximum(px, 1e-300))) / N
    return w


def posterior_cdf(counts: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Cumulative posterior over GRID for each observed raw score x: array (n+1, K)."""
    n = len(counts) - 1
    lik = stats.binom.pmf(np.arange(n + 1)[:, None], n, GRID[None, :])
    post = lik * w[None, :]
    post /= post.sum(axis=1, keepdims=True)
    return np.cumsum(post, axis=1)


def fit_check(counts: np.ndarray, w: np.ndarray) -> dict:
    n = len(counts) - 1
    fitted = counts.sum() * (stats.binom.pmf(np.arange(n + 1)[:, None], n, GRID[None, :]) @ w)
    return {"max_abs_diff": float(np.abs(fitted - counts).max()),
            "top3_obs": float(counts[-3:].sum()), "top3_fit": float(fitted[-3:].sum())}


def draw_retest(raw: np.ndarray, cdf: np.ndarray, n: int, L: float, rng: np.random.Generator) -> np.ndarray:
    """One retest raw score per student (raw: observed integer raw scores; cdf from posterior_cdf).

    The retest is Binomial(m, p) rescaled to n points, with m = round(L * n) scoring units:
    L = 1 is the real test, L > 1 a longer test (less error), L < 1 fewer, larger scoring units (more error).
    """
    p = np.empty(len(raw))
    u = rng.random(len(raw))
    for x in np.unique(raw):
        sel = raw == x
        k = np.searchsorted(cdf[x], u[sel])
        p[sel] = GRID[np.minimum(k, len(GRID) - 1)]
    m = max(1, int(round(L * n)))
    return np.clip(np.rint(rng.binomial(m, p) * n / m), 0, n).astype(int)


def equipercentile(n_new: np.ndarray, n_ref: np.ndarray) -> np.ndarray:
    """NAEC-style equipercentile equating of version 'new' onto the reference version's raw scale
    (mid-point percentile ranks, maximum fixed at 100, linear interpolation)."""
    def pr(n):
        p = 100 * (np.cumsum(n) - n / 2) / n.sum()
        p[-1] = 100.0
        return p
    return np.interp(pr(n_new), pr(n_ref), np.arange(len(n_ref)))


def subject_model(eq_ys: pd.DataFrame, min_n: int = 500) -> dict:
    """Per exam version of one subject-year: prior (NPMLE), posterior cdf, equating table, counts."""
    S = {}
    for v, e in eq_ys.groupby("version"):
        e = e.sort_values("raw")
        c = e.n.values.astype(float)
        if c.sum() < min_n:
            continue
        w = npmle(c)
        S[int(v)] = {"counts": c, "N": c.sum(), "n": len(c) - 1, "w": w, "F": np.cumsum(w),
                     "cdf_post": posterior_cdf(c, w), "equated": e.equated.values}
    return S


def _raw_equating_delta(S: dict, rng: np.random.Generator) -> dict:
    vs = sorted(S)
    ref = S[vs[0]]["counts"]
    bref = rng.multinomial(int(ref.sum()), ref / ref.sum())
    d = {vs[0]: np.zeros(len(ref))}
    for v in vs[1:]:
        c = S[v]["counts"]
        b = rng.multinomial(int(c.sum()), c / c.sum())
        d[v] = equipercentile(b, bref) - equipercentile(c, ref)
    return d


def equating_delta(S: dict, rng: np.random.Generator, n_center: int = 400) -> dict:
    """Sampling error of the equating: re-equate multinomially resampled score distributions
    (reference = lowest-numbered version) and return the change of each version's equating function.
    The resampling mean of the change is subtracted (computed once per subject model and cached), so the
    simulated equating error has mean zero and does not shift scores systematically."""
    if "_center" not in S.get("__cache__", {}):
        c_rng = np.random.default_rng(12345)
        draws = [_raw_equating_delta({k: v for k, v in S.items() if k != "__cache__"}, c_rng) for _ in range(n_center)]
        S["__cache__"] = {"_center": {v: np.mean([d[v] for d in draws], axis=0) for v in draws[0]}}
    center = S["__cache__"]["_center"]
    d = _raw_equating_delta({k: v for k, v in S.items() if k != "__cache__"}, rng)
    return {v: d[v] - center[v] for v in d}


def retest_equated(raw: np.ndarray, ver: np.ndarray, S: dict, L: float, switch: bool,
                   eq_delta: dict | None, rng: np.random.Generator) -> np.ndarray:
    """Equated score on a simulated parallel test for each student (NaN where the version is not modelled).

    switch=True: the retest is taken on a random version (probability proportional to version size); the
    student keeps the same percentile of true ability (true-score equipercentile mapping between the
    versions' estimated true-score distributions)."""
    out = np.full(len(raw), np.nan)
    vs = sorted(k for k in S if k != "__cache__")
    probs = np.array([S[v]["N"] for v in vs])
    probs = probs / probs.sum()
    for v in vs:
        sel = np.where(ver == v)[0]
        if not len(sel):
            continue
        s = S[v]
        x = raw[sel].astype(int)
        u = rng.random(len(sel))
        k = np.empty(len(sel), int)
        for xv in np.unique(x):
            m = x == xv
            k[m] = np.minimum(np.searchsorted(s["cdf_post"][xv], u[m]), len(GRID) - 1)
        nv = rng.choice(vs, size=len(sel), p=probs) if switch else np.full(len(sel), v)
        n = s["n"]
        mtr = max(1, int(round(L * n)))
        for v2 in vs:
            m2 = nv == v2
            if not m2.any():
                continue
            kk = k[m2]
            if v2 != v:
                Flo = np.concatenate([[0.0], s["F"]])[kk]
                kk = np.minimum(np.searchsorted(S[v2]["F"], Flo + rng.random(len(kk)) * s["w"][kk]), len(GRID) - 1)
            rr = np.clip(np.rint(rng.binomial(mtr, GRID[kk]) * n / mtr), 0, n).astype(int)
            e = S[v2]["equated"][rr]
            if eq_delta is not None:
                e = e + eq_delta[v2][rr]
            out[sel[m2]] = e
    return out
