"""Recover raw scores and exam versions from scaled scores.

NAEC pipeline: raw score (per exam version) -> equated score (equating table) -> scaled score.
The scaled score is one monotone function g(equated) per (year, subject), shared by all versions.
g is reconstructed from admitted students, for whom both raw and scaled scores are published.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from load import load_admissions, load_equating

SUBJ_COLS = {"GEO": ("geo_raw", "geo_sc", None), "FL": ("fl_raw", "fl_sc", "fl_code"),
             "EL1": ("el1_raw", "el1_sc", "el1_code"), "EL2": ("el2_raw", "el2_sc", "el2_code")}


def admitted_pairs(adm: pd.DataFrame) -> pd.DataFrame:
    """Unique (subject, raw, scaled) pairs observed among admitted students."""
    parts = []
    for slot, (rc, sc, cc) in SUBJ_COLS.items():
        p = adm[[rc, sc] + ([cc] if cc else [])].dropna(subset=[rc, sc]).copy()
        p["subject"] = p[cc] if cc else "GEO"
        parts.append(p.rename(columns={rc: "raw", sc: "scaled"})[["subject", "raw", "scaled"]])
    return pd.concat(parts).drop_duplicates().reset_index(drop=True)


def scaling_function(eq_ys: pd.DataFrame, pairs_s: pd.DataFrame) -> pd.DataFrame:
    """Knots (equated, scaled) of g for one year-subject.

    For each raw score, the distinct scaled values seen among admits are matched to the
    versions' distinct equated values in sorted order (g is increasing).
    """
    knots = []
    piv = eq_ys.pivot(index="raw", columns="version", values="equated")
    for raw, g in pairs_s.groupby("raw"):
        if raw not in piv.index:
            continue
        eqs = np.unique(piv.loc[raw].dropna().round(4).values)
        scs = np.sort(g["scaled"].unique())
        if len(eqs) == len(scs):
            knots += list(zip(eqs, scs))
    k = pd.DataFrame(knots, columns=["equated", "scaled"]).drop_duplicates().sort_values("equated")
    k = k.groupby("equated", as_index=False)["scaled"].mean()
    return k


def score_table(year: int, eq: pd.DataFrame | None = None, adm: pd.DataFrame | None = None) -> pd.DataFrame:
    """For every (subject, version, raw): equated score and predicted scaled score g(equated)."""
    eq = load_equating([year]) if eq is None else eq[eq.year == year]
    adm = load_admissions(year) if adm is None else adm
    pairs = admitted_pairs(adm)
    out = []
    for subj, e in eq.groupby("subject"):
        p = pairs[pairs.subject == subj]
        if p.empty:
            continue
        k = scaling_function(e, p)
        t = e[["subject", "version", "raw", "equated", "n", "max_raw"]].copy()
        # g is piecewise linear between knots; linear extrapolation beyond the observed range
        t["scaled"] = np.interp(t["equated"], k["equated"], k["scaled"])
        lo, hi = k.iloc[:2], k.iloc[-2:]
        for sel, kk in [(t["equated"] < k["equated"].min(), lo), (t["equated"] > k["equated"].max(), hi)]:
            if len(kk) == 2:
                slope = np.diff(kk["scaled"].values)[0] / np.diff(kk["equated"].values)[0]
                t.loc[sel, "scaled"] = kk["scaled"].values[0] + slope * (t.loc[sel, "equated"] - kk["equated"].values[0])
        t["scaled"] = t["scaled"].round(1)
        t["n_knots"] = len(k)
        out.append(t)
    return pd.concat(out, ignore_index=True)


def recover_raw(rating: pd.DataFrame, eq: pd.DataFrame) -> pd.DataFrame:
    """Attach raw score, version and equated score to each rating-list row by rank matching.

    scaled = g(equated) is increasing, so the k-th highest scaled score in the rating list
    belongs to the k-th highest equated score in the equating table (cells expanded by n).
    Matching starts from the top; if the rating list has a few more students than the
    equating table (tiny extra versions), the surplus is left unmatched at the bottom.
    Within equal scaled scores the cell order is by equated score (true assignment unknowable).
    """
    res = []
    for (year, subj), r in rating.groupby(["year", "subject"]):
        e = eq[(eq.year == year) & (eq.subject == subj) & (eq.n > 0)]
        if e.empty:
            continue
        e = e.sort_values(["equated", "version"], ascending=[False, True])
        cells = e.loc[e.index.repeat(e["n"])].reset_index(drop=True)
        r = r.sort_values("scaled", ascending=False, kind="stable").reset_index(drop=True)
        k = min(len(r), len(cells))
        for c in ["raw", "version", "equated", "max_raw"]:
            r[c] = np.nan
            r.loc[: k - 1, c] = cells.loc[: k - 1, c].values
        res.append(r)
    return pd.concat(res, ignore_index=True)


def check_recovery(rec: pd.DataFrame, adm: pd.DataFrame) -> pd.DataFrame:
    """Share of admitted students whose recovered raw score equals the published raw score,
    and the number of equating cells that received more than one distinct scaled score."""
    rows = []
    pub = []
    for slot, (rc, sc, cc) in SUBJ_COLS.items():
        p = adm[["exam_id", rc] + ([cc] if cc else [])].dropna(subset=[rc]).copy()
        p["subject"] = p[cc] if cc else "GEO"
        pub.append(p.rename(columns={rc: "raw_pub"})[["exam_id", "subject", "raw_pub"]])
    pub = pd.concat(pub)
    m = pub.merge(rec[["exam_id", "subject", "raw"]], on=["exam_id", "subject"])
    for subj, g in m.groupby("subject"):
        cells = rec[rec.subject == subj].groupby(["version", "raw"])["scaled"].nunique()
        rows.append({"subject": subj, "n_admitted": len(g), "raw_exact": (g.raw == g.raw_pub).mean(),
                     "raw_within1": ((g.raw - g.raw_pub).abs() <= 1).mean(),
                     "cells_multi_scaled": int((cells > 1).sum()), "cells": len(cells)})
    return pd.DataFrame(rows)


def top_linear_g(scaled: np.ndarray, eq_ys: pd.DataFrame, n_top_pts: int = 6) -> tuple[float, float, float]:
    """Fit scaled = a + b * equated on the top of the scale from scaled scores alone.

    Used when raw scores are not published (admitted lists 2020-2024). The highest observed scaled value is
    taken as g(max); b is chosen on a grid so that every distinct observed scaled value within the top
    n_top_pts raw points lies within 0.06 of a predicted cell value. Returns (a, b, share_matched).
    """
    vals = np.unique(np.round(scaled, 1))
    e = eq_ys[eq_ys.n > 0]
    M = e.max_raw.iloc[0]
    cells = np.unique(e[e.raw >= M - n_top_pts].equated.round(4))
    smax = vals.max()
    best = (None, None, -1, 1e9)
    for b in np.arange(0.8, 2.5, 0.0005):
        a = smax - b * M
        pred = np.round(a + b * cells, 1)
        sel = vals >= a + b * (M - n_top_pts)
        d = np.abs(vals[sel][:, None] - pred[None, :]).min(axis=1)
        score = (d <= 0.06).mean()
        if score > best[2] or (score == best[2] and d.sum() < best[3]):
            best = (a, b, score, d.sum())
    return best[0], best[1], best[2]


def scaled_to_raw_top(scaled: np.ndarray, eq_ys: pd.DataFrame, a: float, b: float) -> pd.DataFrame:
    """Nearest (version, raw) cell for each scaled score under scaled = a + b * equated."""
    e = eq_ys[eq_ys.n > 0].copy()
    e["pred"] = a + b * e.equated
    e = e.sort_values("pred").reset_index(drop=True)
    idx = np.abs(np.asarray(scaled)[:, None] - e.pred.values[None, :]).argmin(axis=1)
    out = e.loc[idx, ["version", "raw", "equated", "pred"]].reset_index(drop=True)
    out["err"] = np.abs(np.asarray(scaled) - out.pred.values)
    return out
