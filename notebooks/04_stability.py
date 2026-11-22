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
# Mathematics, English and Georgian are all re-drawn on every simulated retake, except in the extra
# "realistic, mathematics only" scenario, where only mathematics is redrawn.
#
# Uncertainty in the true-score distribution: every prior is refitted to B = 200 multinomially resampled
# score distributions (bootstrap); a second pass where retake i uses refit i % B gives the boot_* columns.
# Tail sensitivity (MACS): the math prior is replaced by the true-score distributions with the smallest / largest
# error variance of the top scorers that still fit the data (95% profile likelihood, simulate.npmle_bound).

# %%
from common import *
from load import load_equating, load_rating, load_admissions
from scores import recover_raw, top_linear_g
from simulate import (subject_model, equating_delta, retest_equated, fit_check, refit, npmle_bound,
                      top_functional, loglik, GRID, _lik)

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
SCEN = {"more error (x1.7)": (51 / 87, False, False, "npmle"), "baseline (x1)": (1.0, False, False, "npmle"),
        "half error (x0.5)": (2.0, False, False, "npmle"), "quarter error (x0.25)": (4.0, False, False, "npmle"),
        "realistic: new version + equating error": (1.0, True, True, "npmle"),
        "realistic, alternative math tail: true scores closer to the maximum": (1.0, True, True, "fav"),
        "realistic, alternative math tail: true scores further below the maximum": (1.0, True, True, "unf"),
        "realistic, mathematics only": (1.0, True, True, "npmle")}
MATH_ONLY = {"realistic, mathematics only"}  # only MATH is redrawn; GEO and FL keep their scaled scores
MAIN_SCEN = list(SCEN)[:5]
FULL_SCEN_FOR = {"MACS"}  # other programs: baseline + realistic only
R = 1000  # simulated retakes per scenario
B = 200  # bootstrap refits of every true-score prior; retake i uses refit i % B


def prior_bounds(S, year):
    """Alternative math true-score priors at the profile-likelihood bounds of the top scorers' binomial error
    variance (simulate.npmle_bound): lowering it moves the fitted true scores toward the maximum, raising it
    moves them further below. Also reports each prior's fit at the top."""
    fav, unf, rows = {}, {}, []
    for v, m in S.items():
        if v == "__cache__":
            continue
        c, w, n = m["counts"], m["w"], m["n"]
        h = top_functional(n, w, "error")
        for name, sign, store in [("NPMLE", 0, None), ("true scores closer to the maximum", 1, fav), ("true scores further below the maximum", -1, unf)]:
            wb, drop = (w, 0.0) if sign == 0 else (lambda r: (r[0], r[1]["loglik_drop"]))(npmle_bound(c, w, h, sign))
            if store is not None:
                store[v] = wb
            post = _lik(n)[n - 2:] * wb
            post /= post.sum(axis=1, keepdims=True)
            fitted = c.sum() * (_lik(n) @ wb)
            rows.append({"year": year, "version": v, "prior": name, "loglik_drop": drop,
                         **{f"obs_{x}": c[x] for x in (n - 2, n - 1, n)},
                         **{f"fit_{x}": fitted[x] for x in (n - 2, n - 1, n)},
                         **{f"sem_at_{x}": s for x, s in zip((n - 2, n - 1, n), np.sqrt(post @ (n * GRID * (1 - GRID))))},
                         **{f"true_mean_at_{x}": t for x, t in zip((n - 2, n - 1, n), n * post @ GRID)}})
    return refit(S, w_override=fav), refit(S, w_override=unf), rows


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
    models, checks, boot = {}, [], {}
    brng = np.random.default_rng(year)
    for s in ["MATH", "ENG", "GEO"]:
        models[s] = subject_model(eq[(eq.year == year) & (eq.subject == s)])
        for v, m in models[s].items():
            checks.append({"year": year, "subject": s, "version": v, **fit_check(m["counts"], m["w"])})
        boot[s] = [refit(models[s], rng=brng) for _ in range(B)]
    fav, unf, bound_rows = prior_bounds(models["MATH"], year)
    wide = r.pivot(index="exam_id", columns="subject", values="scaled")
    fl = wide.reindex(columns=["ENG", "RUS", "GER", "FRE"]).bfill(axis=1).iloc[:, 0]
    pop = pd.DataFrame({"GEO": wide["GEO"], "FL": fl, "MATH": wide["MATH"]}).dropna()
    info = rec.set_index(["exam_id", "subject"])[["raw", "version"]]
    for s in ["MATH", "ENG", "GEO"]:
        sub = info.xs(s, level="subject").reindex(pop.index)
        pop[f"{s}_raw"], pop[f"{s}_ver"] = sub.raw, sub.version
    pop["is_eng"] = wide["ENG"].reindex(pop.index).notna()
    b = top_linear_g(pop.MATH.values, eq[(eq.year == year) & (eq.subject == "MATH")])[1]
    YEAR[year] = (pop, models, g, b, pd.DataFrame(checks), boot, {"fav": fav, "unf": unf}, bound_rows)
    return YEAR[year]


def run(label, year, code, rng, rng_x):
    pop_all, models, g, b, _, boot, bounds, _ = year_data(year)
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

    def sim(L, switch, eqerr, prior, rng, math_only=False):
        stay = np.zeros(len(pop))
        changed = []
        for i in range(R):
            new = {}
            for s, col in [("MATH", "MATH"), ("ENG", "FL"), ("GEO", "GEO")]:
                sc = pop[col].values.copy()
                if not (math_only and s != "MATH"):
                    ok = pop[f"{s}_raw"].notna().values & pop[f"{s}_ver"].isin(list(models[s])).values
                    if s == "ENG":
                        ok &= pop.is_eng.values
                    d = equating_delta(models[s], rng) if eqerr else None
                    M = {"npmle": models[s], "boot": boot[s][i % B]}.get(prior, bounds.get(prior) if s == "MATH" else models[s])
                    e = retest_equated(pop[f"{s}_raw"].values[ok], pop[f"{s}_ver"].values[ok], M, L, switch, d, rng)
                    sc[ok] = np.interp(e, g[s].index.values, g[s].values)
                new[col] = sc
            comp = w[0] * new["GEO"] + w[1] * new["FL"] + w[2] * new["MATH"] + rng.random(len(pop)) * 1e-6
            top = np.zeros(len(pop), bool)
            top[np.argsort(-comp)[:K]] = True
            stay += top
            changed.append((orig_top & ~top).sum())
        return np.array(changed), stay / R

    for name, (L, switch, eqerr, prior) in SCEN.items():
        if label not in FULL_SCEN_FOR and name not in ("baseline (x1)", "realistic: new version + equating error"):
            continue
        # main scenarios: point estimates with the fitted prior (rng, as before); the bootstrap pass (rng_x)
        # adds the uncertainty of the estimated true-score distribution to the 90% range. The math-only
        # scenario uses rng_m for both passes so it cannot perturb any existing result.
        math_only = name in MATH_ONLY
        changed, p_top = sim(L, switch, eqerr, prior, rng_m if math_only else (rng if prior == "npmle" else rng_x), math_only)
        cb = sim(L, switch, eqerr, "boot", rng_m if math_only else rng_x, math_only)[0] if prior == "npmle" else changed
        rows.append({"program": label, "year": year, "scenario": name, "weights": "/".join(f"{x:g}" for x in w),
                     "admitted_math_route": len(a), "cutoff": cut, "K_places": K, "one_math_question": pt,
                     "mean_places_changing_hands": changed.mean(), "pct_places_changing_hands": 100 * changed.mean() / K,
                     "p05": np.percentile(changed, 5), "p95": np.percentile(changed, 95),
                     "boot_mean": cb.mean(), "boot_p05": np.percentile(cb, 5), "boot_p95": np.percentile(cb, 95),
                     "n_students_simulated": len(pop),
                     "max_p_top_bottom_decile": p_top[pop.comp.values <= np.percentile(pop.comp, 10)].max()})
        d = pop[["comp", "MATH_raw"]].copy()
        d["orig_top"], d["p_top"], d["program"], d["year"], d["scenario"] = orig_top, p_top, label, year, name
        d["dist_q"] = (d.comp - cut) / pt
        per.append(d)
    return rows, per


# %%
rng = np.random.default_rng(2026)
rng_x = np.random.default_rng(7)  # bootstrap and tail passes; keeps rng identical to the main runs
rng_m = np.random.default_rng(11)  # math-only scenario only; no shared draws with rng / rng_x
results, per_student = [], []
for label, year, code in PROGRAMS:
    rws, per = run(label, year, code, rng, rng_x)
    results += rws
    per_student += per
res = pd.DataFrame(results)
res.to_csv(OUT / "t04_stability_summary.csv", index=False)
ps = pd.concat(per_student)
fits = pd.concat([YEAR[y][4] for y in sorted(YEAR)])
fits.to_csv(OUT / "t04_npmle_fit_checks.csv", index=False)
tail = pd.DataFrame([r for y in sorted(YEAR) for r in YEAR[y][7]])
tail.to_csv(OUT / "t04_tail_bounds.csv", index=False)
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
fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.9))
ax = axes[0]
colors = dict(zip(MAIN_SCEN, ["tab:red", "black", "tab:blue", "tab:green", "tab:purple"]))
for name in MAIN_SCEN:
    d = ps[(ps.program == "MACS") & (ps.year == 2026) & (ps.scenario == name)].sort_values("comp")
    sm = d.p_top.rolling(25, center=True, min_periods=5).mean()
    ax.plot(d.dist_q, sm, color=colors[name], label=name, lw=1.8 if name.startswith(("baseline", "realistic")) else 1.1,
            ls="--" if name.startswith("realistic") else "-")
ax.axvline(0, color="grey", ls=":")
ax.set_xlim(-6, 3)
ax.set_xlabel("Distance from the 2026 MACS cutoff (in math questions)")
ax.set_ylabel("P(above the line\non a parallel test)")
ax.set_title("Above or below the line: mostly luck\nwithin ~2 questions")
ax.legend(framealpha=0.85, edgecolor="none")
ax = axes[1]
macs = res[res.program == "MACS"]
xs = np.arange(len(MAIN_SCEN))
yrs = sorted(macs.year.unique())
wbar = 0.8 / len(yrs)
for i, (y, c) in enumerate(zip(yrs, ["tab:olive", "tab:gray", "tab:purple"])):
    dd = macs[macs.year == y].set_index("scenario").reindex(MAIN_SCEN)
    ax.bar(xs + (i - (len(yrs) - 1) / 2) * wbar, dd.pct_places_changing_hands, width=wbar, color=c, label=str(y),
           yerr=[100 * (dd.mean_places_changing_hands - dd.boot_p05).clip(lower=0) / dd.K_places, 100 * (dd.boot_p95 - dd.mean_places_changing_hands) / dd.K_places],
           capsize=2)
ax.set_xticks(xs, ["more\nx1.7", "baseline\nx1", "half\nx0.5", "quarter\nx0.25", "realistic\nnew ver.\n+ equating"])
ax.set_ylabel("% of MACS-level places\nchanging hands")
ax.set_title("MACS: places that change hands\non a parallel test")
ax.set_ylim(0, 62)
ax.legend(ncol=3, loc="upper center", framealpha=0.85, edgecolor="none")
fig.tight_layout()
savefig(fig, "fig06_stability")

# %% [markdown]
# ## Figure 8 - Other selective math programs

# %%
fig, ax = plt.subplots(figsize=(7.5, 3.35))
progs = [p for p in ["MACS", "Free Uni ECE", "ISET (TSU)"] if p in res.program.unique()]
labels, base, real, lo, hi = [], [], [], [], []
short = {"MACS": "MACS", "Free Uni ECE": "ECE", "ISET (TSU)": "ISET"}
for p in progs:
    for y in sorted(res[res.program == p].year.unique()):
        b_ = res[(res.program == p) & (res.year == y) & (res.scenario == "baseline (x1)")].iloc[0]
        r_ = res[(res.program == p) & (res.year == y) & (res.scenario.str.startswith("realistic"))].iloc[0]
        labels.append(f"{short[p]}\n{y}\nK={b_.K_places}")
        base.append(b_.pct_places_changing_hands)
        real.append(r_.pct_places_changing_hands)
x = np.arange(len(labels))
ax.bar(x - 0.2, base, 0.4, color="black", alpha=0.75, label="baseline")
ax.bar(x + 0.2, real, 0.4, color="tab:purple", alpha=0.85, label="realistic: new version + equating error")
ax.set_xticks(x, labels)
ax.set_ylabel("% of program-level places\nchanging hands")
ax.set_ylim(0, max(base + real) * 1.32)
ax.set_title("Places that change hands on a parallel test,\nby program and year")
ax.legend(framealpha=0.85, edgecolor="none")
fig.tight_layout()
savefig(fig, "fig08_programs_stability")
