# %% [markdown]
# # 04 - How stable are admission lines at the top? Parallel-test simulation
#
# Question: if every student sat a parallel version of the same exams (same ability, new questions of the
# same difficulty), how many of the students above a selective program's cutoff would still be above it?
#
# Model (src/simulate.py), per year x subject x exam version:
# 1. Binomial error model: raw | p ~ Binomial(n_max, p), p = the student's true proportion of points.
# 2. The distribution of p in the population is estimated from the observed raw-score counts
#    (nonparametric maximum likelihood, no shape assumption).
# 3. For each student, a true p is drawn from its posterior given the observed raw score, then a retest
#    raw score is drawn; it is equated with the NAEC equating table and converted to a scaled score with
#    that year's scaled-score function (reconstructed from the data).
# 4. The program's competitive score is recomputed; the top K students are compared with the original top K,
#    where K = number of students at/above the program's actual cutoff ("program-level places").
#
# Scenarios:
# - error multipliers on the error variance of every subject, same exam version as the student actually took:
#   "more error (x1.7)" (open items all-or-nothing, 87/51), "baseline (x1)", "half error (x0.5)",
#   "quarter error (x0.25)" (as if the tests had 2x / 4x as many items);
# - "realistic: new version + equating error": baseline error, plus the retest is taken on a randomly assigned
#   version (the student keeps the same percentile of true ability) and NAEC's equating is re-estimated from a
#   resampled score distribution (sampling error of the equating), as happens in a real exam year.
# Students whose foreign language is not English keep their foreign-language score (no equating table).

# %%
from common import *
from load import load_equating, load_rating, load_admissions
from scores import recover_raw, top_linear_g
from simulate import subject_model, equating_delta, retest_equated, fit_check

eq = load_equating()
old036 = pd.read_csv(ROOT / "data/processed/admitted_036_all.csv", dtype={"prog_code": str, "exam_id": str})
iset_path = ROOT / "data/processed/admitted_iset_all.csv"
oldiset = pd.read_csv(iset_path, dtype={"prog_code": str, "exam_id": str}) if iset_path.exists() else None

PROGRAMS = [  # (label, year, program code)
    ("MACS", 2024, "03601044"), ("MACS", 2025, "03601054"), ("MACS", 2026, "03601044"),
    ("Free Uni ECE", 2024, "03601054"), ("Free Uni ECE", 2025, "03601034"), ("Free Uni ECE", 2026, "03601054"),
    ("ISET (TSU)", 2025, "00101394"), ("ISET (TSU)", 2026, "00101534"),
]
if oldiset is not None and (oldiset.year == 2024).any():
    PROGRAMS.append(("ISET (TSU)", 2024, oldiset[oldiset.year == 2024].prog_code.mode()[0]))
SCEN = {"more error (x1.7)": (51 / 87, False, False), "baseline (x1)": (1.0, False, False),
        "half error (x0.5)": (2.0, False, False), "quarter error (x0.25)": (4.0, False, False),
        "realistic: new version + equating error": (1.0, True, True)}
FULL_SCEN_FOR = {"MACS"}  # other programs: baseline + realistic only
R = 1000


def admits(label, year, code):
    if year <= 2024:
        d = old036 if code.startswith("036") else oldiset
        d = d[(d.year == year) & (d.prog_code == code)]
        return d[d.el1.astype(str).str.contains("მათემ")]
    a = load_admissions(year)
    return a[(a.prog_code == code) & (a.el1_code == "MATH")]


YEAR = {}


def year_data(year):
    if year in YEAR:
        return YEAR[year]
    r = load_rating(year)
    rec = recover_raw(r, eq)
    g = {s: d.groupby("equated").scaled.median() for s, d in rec.dropna(subset=["equated"]).groupby("subject")}
    models, checks = {}, []
    for s in ["MATH", "ENG", "GEO"]:
        models[s] = subject_model(eq[(eq.year == year) & (eq.subject == s)])
        for v, m in models[s].items():
            from simulate import npmle  # noqa
            checks.append({"year": year, "subject": s, "version": v, **fit_check(m["counts"], m["w"])})
    wide = r.pivot(index="exam_id", columns="subject", values="scaled")
    fl = wide.reindex(columns=["ENG", "RUS", "GER", "FRE"]).bfill(axis=1).iloc[:, 0]
    pop = pd.DataFrame({"GEO": wide["GEO"], "FL": fl, "MATH": wide["MATH"]}).dropna()
    info = rec.set_index(["exam_id", "subject"])[["raw", "version"]]
    for s in ["MATH", "ENG", "GEO"]:
        sub = info.xs(s, level="subject").reindex(pop.index)
        pop[f"{s}_raw"], pop[f"{s}_ver"] = sub.raw, sub.version
    pop["is_eng"] = wide["ENG"].reindex(pop.index).notna()
    b = top_linear_g(pop.MATH.values, eq[(eq.year == year) & (eq.subject == "MATH")])[1]
    YEAR[year] = (pop, models, g, b, pd.DataFrame(checks))
    return YEAR[year]


def run(label, year, code, rng):
    pop_all, models, g, b, _ = year_data(year)
    a = admits(label, year, code)
    X = a[["geo_sc", "fl_sc", "el1_sc"]].values
    w, *_ = np.linalg.lstsq(X, a.comp.values, rcond=None)
    w = np.round(w, 6)
    assert np.abs(X @ w - a.comp.values).max() < 0.06, (label, year, w)
    cut = a.comp.min()
    pt = w[2] * b  # one raw math point at the top, in competitive-score points
    pop = pop_all.copy()
    pop["comp"] = (w[0] * pop.GEO + w[1] * pop.FL + w[2] * pop.MATH).round(1)
    pop = pop[pop.comp >= cut - 20 * pt].copy()
    K = int((pop.comp >= cut).sum())
    orig_top = (pop.comp >= cut).values
    rows, per = [], []
    for name, (L, switch, eqerr) in SCEN.items():
        if label not in FULL_SCEN_FOR and name not in ("baseline (x1)", "realistic: new version + equating error"):
            continue
        stay = np.zeros(len(pop))
        changed = []
        for _ in range(R):
            new = {}
            for s, col in [("MATH", "MATH"), ("ENG", "FL"), ("GEO", "GEO")]:
                sc = pop[col].values.copy()
                ok = pop[f"{s}_raw"].notna().values & pop[f"{s}_ver"].isin(list(models[s])).values
                if s == "ENG":
                    ok &= pop.is_eng.values
                d = equating_delta(models[s], rng) if eqerr else None
                e = retest_equated(pop[f"{s}_raw"].values[ok], pop[f"{s}_ver"].values[ok], models[s], L, switch, d, rng)
                sc[ok] = np.interp(e, g[s].index.values, g[s].values)
                new[col] = sc
            comp = w[0] * new["GEO"] + w[1] * new["FL"] + w[2] * new["MATH"] + rng.random(len(pop)) * 1e-6
            top = np.zeros(len(pop), bool)
            top[np.argsort(-comp)[:K]] = True
            stay += top
            changed.append((orig_top & ~top).sum())
        changed = np.array(changed)
        p_top = stay / R
        rows.append({"program": label, "year": year, "scenario": name, "weights": "/".join(f"{x:g}" for x in w),
                     "admitted_math_route": len(a), "cutoff": cut, "K_places": K, "one_math_question": pt,
                     "mean_places_changing_hands": changed.mean(), "pct_places_changing_hands": 100 * changed.mean() / K,
                     "p05": np.percentile(changed, 5), "p95": np.percentile(changed, 95),
                     "n_students_simulated": len(pop),
                     "max_p_top_bottom_decile": p_top[pop.comp.values <= np.percentile(pop.comp, 10)].max()})
        d = pop[["comp", "MATH_raw"]].copy()
        d["orig_top"], d["p_top"], d["program"], d["year"], d["scenario"] = orig_top, p_top, label, year, name
        d["dist_q"] = (d.comp - cut) / pt
        per.append(d)
    return rows, per


# %%
rng = np.random.default_rng(2026)
results, per_student = [], []
for label, year, code in PROGRAMS:
    rws, per = run(label, year, code, rng)
    results += rws
    per_student += per
res = pd.DataFrame(results)
res.to_csv(OUT / "t04_stability_summary.csv", index=False)
ps = pd.concat(per_student)
fits = pd.concat([YEAR[y][4] for y in sorted(YEAR)])
fits.to_csv(OUT / "t04_npmle_fit_checks.csv", index=False)
res[["program", "year", "scenario", "weights", "K_places", "mean_places_changing_hands", "pct_places_changing_hands",
     "p05", "p95", "max_p_top_bottom_decile"]].round(2)

# %% [markdown]
# Check: `max_p_top_bottom_decile` ~ 0 means students at the bottom of the simulated window never reach the
# top K, so restricting the simulation to the window does not change the result.
#
# ## Probability of being above the line on a parallel test, by distance from the cutoff
# Distance measured in math questions (raw math points at the top, converted with the program's weight).

# %%
bins = [-12, -4, -3, -2, -1, 0, 1, 2, 3, 10]
ps["band"] = pd.cut(ps.dist_q, bins, right=False)
band = (ps.groupby(["program", "year", "scenario", "band"], observed=True)
        .agg(n=("p_top", "size"), mean_p_above_line=("p_top", "mean")).reset_index())
band.to_csv(OUT / "t04_stability_by_distance.csv", index=False)
band[(band.program == "MACS") & band.scenario.isin(["baseline (x1)", "realistic: new version + equating error"])].pivot(
    index="band", columns=["year", "scenario"], values="mean_p_above_line").round(3)

# %% [markdown]
# ## Who would replace them? (MACS, expected entrants from below the line, by original math raw score)

# %%
ent = (ps[(ps.program == "MACS") & (~ps.orig_top) & ps.scenario.isin(["baseline (x1)", "realistic: new version + equating error"])]
       .groupby(["year", "scenario", "MATH_raw"]).p_top.sum().rename("expected_entrants").reset_index())
ent = ent[ent.expected_entrants > 0.05]
ent.to_csv(OUT / "t04_entrants_by_math_raw.csv", index=False)
ent.pivot(index="MATH_raw", columns=["year", "scenario"], values="expected_entrants").round(2)

# %% [markdown]
# ## Figure 6 - Probability of staying above the MACS line on a parallel test (2026)

# %%
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
ax = axes[0]
colors = dict(zip(SCEN, ["tab:red", "black", "tab:blue", "tab:green", "tab:purple"]))
for name in SCEN:
    d = ps[(ps.program == "MACS") & (ps.year == 2026) & (ps.scenario == name)].sort_values("comp")
    sm = d.p_top.rolling(25, center=True, min_periods=5).mean()
    ax.plot(d.dist_q, sm, color=colors[name], label=name, lw=1.8 if name.startswith(("baseline", "realistic")) else 1.1,
            ls="--" if name.startswith("realistic") else "-")
ax.axvline(0, color="grey", ls=":")
ax.set_xlim(-6, 3)
ax.set_xlabel("Distance from the 2026 MACS cutoff (in math questions)")
ax.set_ylabel("P(above the line on a parallel test)")
ax.set_title("Above or below the line: mostly luck within ~2 questions")
ax.legend(fontsize=7, frameon=False)
ax = axes[1]
macs = res[res.program == "MACS"]
xs = np.arange(len(SCEN))
yrs = sorted(macs.year.unique())
wbar = 0.8 / len(yrs)
for i, (y, c) in enumerate(zip(yrs, ["tab:olive", "tab:gray", "tab:purple"])):
    dd = macs[macs.year == y].set_index("scenario").reindex(list(SCEN))
    ax.bar(xs + (i - (len(yrs) - 1) / 2) * wbar, dd.pct_places_changing_hands, width=wbar, color=c, label=str(y),
           yerr=[100 * (dd.mean_places_changing_hands - dd.p05) / dd.K_places, 100 * (dd.p95 - dd.mean_places_changing_hands) / dd.K_places],
           capsize=2)
ax.set_xticks(xs, ["more error\n(x1.7)", "baseline\n(x1)", "half error\n(x0.5)", "quarter error\n(x0.25)", "realistic:\nnew version\n+ equating"], fontsize=7.5)
ax.set_ylabel("% of MACS-level places changing hands")
ax.set_title("MACS: places that change hands on a parallel test")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
savefig(fig, "fig06_stability")

# %% [markdown]
# ## Figure 8 - Other selective math programs

# %%
fig, ax = plt.subplots(figsize=(8.5, 3.8))
progs = [p for p in ["MACS", "Free Uni ECE", "ISET (TSU)"] if p in res.program.unique()]
labels, base, real, lo, hi = [], [], [], [], []
for p in progs:
    for y in sorted(res[res.program == p].year.unique()):
        b_ = res[(res.program == p) & (res.year == y) & (res.scenario == "baseline (x1)")].iloc[0]
        r_ = res[(res.program == p) & (res.year == y) & (res.scenario.str.startswith("realistic"))].iloc[0]
        labels.append(f"{p}\n{y} (K={b_.K_places})")
        base.append(b_.pct_places_changing_hands)
        real.append(r_.pct_places_changing_hands)
x = np.arange(len(labels))
ax.bar(x - 0.2, base, 0.4, color="black", alpha=0.75, label="baseline")
ax.bar(x + 0.2, real, 0.4, color="tab:purple", alpha=0.85, label="realistic: new version + equating error")
ax.set_xticks(x, labels, fontsize=7.5)
ax.set_ylabel("% of program-level places\nchanging hands")
ax.set_title("Places that change hands on a parallel test, by program and year")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
savefig(fig, "fig08_programs_stability")
