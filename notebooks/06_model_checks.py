# %% [markdown]
# # 06 - Model and data checks (for the methods appendix)
#
# 1. Data validation: rating lists vs equating tables, admitted students found in rating lists,
#    raw-score recovery accuracy.
# 2. True-score model: fit at the top, implied test-retest reliability per subject-year-version.
# 3. Retest simulation preserves the score distribution (same version, and with version switching +
#    equating error).

# %%
from common import *
from load import load_equating, load_rating, load_admissions
from scores import recover_raw, check_recovery
from simulate import subject_model, retest_equated, equating_delta

eq = load_equating()

# %% [markdown]
# ## 1. Data validation

# %%
rows, rec_rows = [], []
for y in [2024, 2025, 2026]:
    r = load_rating(y)
    tot = eq[eq.year == y].groupby("subject").n.sum()
    cnt = r.subject.value_counts()
    for s in ["MATH", "ENG", "GEO", "HIST"]:
        rows.append({"year": y, "subject": s, "rating_list": int(cnt.get(s, 0)), "equating_tables": int(tot.get(s, 0))})
    if y >= 2025:
        a = load_admissions(y)
        rec = recover_raw(r, eq)
        c = check_recovery(rec, a)
        c.insert(0, "year", y)
        c["admitted_found_in_rating_list"] = a.exam_id.isin(set(r.exam_id)).mean()
        rec_rows.append(c)
counts = pd.DataFrame(rows)
counts["diff"] = counts.rating_list - counts.equating_tables
counts.to_csv(OUT / "t06_counts_check.csv", index=False)
recovery = pd.concat(rec_rows)
recovery.to_csv(OUT / "t06_raw_recovery_check.csv", index=False)
counts

# %%
recovery.round(4)

# %% [markdown]
# ## 2. True-score model: fit and implied reliability
# Reliability = correlation between the observed raw score and a simulated parallel-test raw score
# (same version, baseline error), i.e. the test-retest reliability implied by the binomial error model.

# %%
rng = np.random.default_rng(7)
rows = []
for (y, s), e in eq[eq.subject.isin(["MATH", "ENG", "GEO"]) & (eq.year >= 2024)].groupby(["year", "subject"]):
    S = subject_model(e)
    for v, m in S.items():
        c = m["counts"].astype(int)
        raw = np.repeat(np.arange(len(c)), c)
        new = retest_equated(raw, np.full(len(raw), v), S, 1.0, False, None, rng)
        rows.append({"year": y, "subject": s, "version": v, "N": int(c.sum()),
                     "implied_reliability": np.corrcoef(m["equated"][raw], new)[0, 1],
                     "top3_observed": int(c[-3:].sum())})
rel = pd.DataFrame(rows)
fit = pd.read_csv(OUT / "t04_npmle_fit_checks.csv")
rel = rel.merge(fit[["year", "subject", "version", "top3_fit"]], on=["year", "subject", "version"], how="left")
rel.to_csv(OUT / "t06_model_reliability.csv", index=False)
rel.groupby(["subject"]).implied_reliability.agg(["min", "max"]).round(3)

# %%
rel.round(3)

# %% [markdown]
# ## 3. Does the simulated retest preserve the population score distribution? (2026)
# Compared: number at or above max - 2.5 and max - 5.5 (equated scale) and the mean equated score of the top 2%.

# %%
rows = []
for s in ["MATH", "ENG", "GEO"]:
    S = subject_model(eq[(eq.year == 2026) & (eq.subject == s)])
    raws, vers, e0 = [], [], []
    for v, m in S.items():
        if not isinstance(v, (int, np.integer)):
            continue
        c = m["counts"].astype(int)
        r_ = np.repeat(np.arange(len(c)), c)
        raws.append(r_), vers.append(np.full(len(r_), v)), e0.append(m["equated"][r_])
    raw, ver, e0 = map(np.concatenate, (raws, vers, e0))
    M = S[1]["n"]

    def q(e):
        return [(e >= M - 2.5).sum(), (e >= M - 5.5).sum(), np.sort(e)[-int(0.02 * len(e)):].mean()]

    rows.append({"subject": s, "case": "observed", **dict(zip(["n_ge_max-2.5", "n_ge_max-5.5", "mean_top2pct"], q(e0)))})
    for name, sw, ee in [("retest, same version", False, False), ("retest, new version + equating error", True, True)]:
        sims = [q(retest_equated(raw, ver, S, 1.0, sw, equating_delta(S, rng) if ee else None, rng)) for _ in range(20)]
        rows.append({"subject": s, "case": name, **dict(zip(["n_ge_max-2.5", "n_ge_max-5.5", "mean_top2pct"], np.mean(sims, 0)))})
pres = pd.DataFrame(rows)
pres.to_csv(OUT / "t06_distribution_preservation.csv", index=False)
pres.round(2)
