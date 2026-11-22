# %% [markdown]
# # 02 - Resolution at the top: test structure, measurement error, version equating
#
# Inputs: equating tables (2020-2026), math scoring schemes (`data/raw/math_tests`),
# admitted students 2026.

# %%
from common import *
import re
from pypdf import PdfReader
from load import load_equating, load_admissions

eq = load_equating()
math = eq[eq.subject == "MATH"].copy()

# %% [markdown]
# ## 1. Structure of the mathematics test (from the official scoring schemes)
# Open-ended items are marked "(k) item." in the scheme, k = maximum points. All earlier items are
# 4-option multiple-choice items worth 1 point.

# %%
rows = []
for f in sorted((ROOT / "data/raw/math_tests").glob("math_*_scheme_v[12].pdf")):
    text = "\n".join(p.extract_text() for p in PdfReader(f).pages)
    opens = [(int(n), int(k)) for k, n in re.findall(r"\((\d)\)\s*(\d{1,2})\s*\.", text)]
    first = min(n for n, _ in opens)
    rows.append({"year": int(f.stem.split("_")[1]), "version": f.stem.split("_")[-1], "n_mc": first - 1,
                 "n_open": len(opens), "open_points": "+".join(str(k) for _, k in opens),
                 "open_total": sum(k for _, k in opens), "max": first - 1 + sum(k for _, k in opens),
                 "sum_k2": first - 1 + sum(k * k for _, k in opens)})
struct = pd.DataFrame(rows)
assert (struct["max"] == 51).all()
struct.to_csv(OUT / "t02_math_test_structure.csv", index=False)
struct

# %% [markdown]
# ## 2. Conditional standard error of measurement (SEM) near the top
# Binomial error model (Lord 1965): for a student whose true proportion of points is p, the raw score
# is the sum of item scores, each scored independently. Two bounds for the open items:
# - lower: an open item worth k points behaves like k independent 1-point items -> Var = 51 p(1-p)
# - upper: an open item is all-or-nothing (k points or 0) -> Var = (n_mc + sum k^2) p(1-p)
# Items of unequal difficulty make the true SEM somewhat smaller than the equal-difficulty value,
# so these are approximations; the simulation in notebook 04 therefore also uses 0.5x and 0.75x.

# %%
sk2 = struct.drop_duplicates("year").set_index("year")["sum_k2"]
true_scores = np.arange(40, 51.5, 1.0)
sem = pd.DataFrame({"true_raw": true_scores})
sem["sem_lower"] = np.sqrt(51 * (true_scores / 51) * (1 - true_scores / 51))
for y, k2 in [(2022, sk2[2022]), (2026, sk2[2026])]:
    sem[f"sem_upper_{y}fmt"] = np.sqrt(k2 * (true_scores / 51) * (1 - true_scores / 51))
sem.to_csv(OUT / "t02_math_sem.csv", index=False)
sem.round(2)

# %% [markdown]
# ## 3. Version equating at the top
# NAEC equates the versions by equipercentile equating: percentile rank (mid-point definition, fixed to
# 100 at the maximum) of a raw score on version B is mapped to the reference version's raw-score scale by
# linear interpolation. My reconstruction reproduces NAEC's 2026 math table to within 0.005 points.
# Equating is estimated from a sample, so its error is largest where few students are, i.e. at the top.
# Bootstrap: resample both versions' score distributions (multinomial, same N) and re-equate.

# %%
def pr(n):
    N = n.sum()
    p = 100 * (np.cumsum(n) - n / 2) / N
    p[-1] = 100.0
    return p


def equate(n_new, n_ref):
    return np.interp(pr(n_new), pr(n_ref), np.arange(len(n_ref)))


g = math[math.year == 2026].pivot(index="raw", columns="version", values="n").fillna(0)
E = math[math.year == 2026].pivot(index="raw", columns="version", values="equated")
rec = equate(g[2].values, g[1].values)
print("max |reconstruction - NAEC| for raw >= 20:", np.abs(rec - E[2].values)[20:].max().round(4))

rng = np.random.default_rng(1)
boot = []
for _ in range(2000):
    b1 = rng.multinomial(int(g[1].sum()), g[1].values / g[1].sum())
    b2 = rng.multinomial(int(g[2].sum()), g[2].values / g[2].sum())
    boot.append(equate(b2, b1))
boot = np.array(boot)
eqse = pd.DataFrame({"raw_v2": g.index, "naec_equated": E[2].values, "boot_mean": boot.mean(0),
                     "boot_se": boot.std(0), "ci95_lo": np.percentile(boot, 2.5, 0), "ci95_hi": np.percentile(boot, 97.5, 0)})
eqse = eqse[eqse.raw_v2 >= 40]
eqse.to_csv(OUT / "t02_math_equating_bootstrap_2026.csv", index=False)
eqse.round(2)

# %% [markdown]
# ## 4. The maximum cannot be equated
# Both versions' maximum (51) is mapped to 51, even when one version's perfect score is much rarer.
# Percentile position of a perfect score, per version and year.

# %%
mx = (math[math.raw == math.max_raw].merge(math.groupby(["year", "version"]).n.sum().rename("N"), on=["year", "version"])
      .assign(pct_at_max=lambda d: 100 * d.n / d.N, top_pct_of_perfect=lambda d: 100 * d.n / d.N))
mx = mx[mx.N > 500][["year", "version", "N", "n", "pct_at_max"]].rename(columns={"n": "n_at_max"})
ratio = mx.pivot(index="year", columns="version", values="pct_at_max")
ratio["ratio_max_min"] = ratio.max(axis=1) / ratio.min(axis=1)
mx.to_csv(OUT / "t02_math_perfect_by_version.csv", index=False)
ratio.round(2)

# %% [markdown]
# ## 5. Raw score -> scaled score at the top, 2026 (from admitted students)
# One raw point of mathematics at the top is ~1.3 scaled points; versions differ by up to ~1.5 scaled points
# at the same raw score.

# %%
adm = load_admissions(2026)
top = (adm[(adm.el1_code == "MATH") & (adm.el1_raw >= 44)].groupby(["el1_raw", "el1_sc"]).size()
       .rename("n_admitted").reset_index())
top.to_csv(OUT / "t02_math_raw_to_scaled_2026.csv", index=False)
top

# %% [markdown]
# ## Figure 3 - Measurement error vs. the score gaps that decide admission

# %%
fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.9))
ax = axes[0]
ax.fill_between(sem.true_raw, sem.sem_lower, sem.sem_upper_2026fmt, alpha=0.3, color="tab:red",
                label="SEM range, 2023-26 format")
ax.plot(sem.true_raw, sem.sem_lower, color="tab:red")
ax.plot(sem.true_raw, sem.sem_upper_2026fmt, color="tab:red", ls="--")
ax.axhline(1, color="grey", ls=":", lw=1)
ax.text(40.2, 1.15, "1 raw point", fontsize=9.5, color="grey")
ax.set_xlabel("True raw score (mathematics, max 51)")
ax.set_ylabel("Standard error (raw points)")
ax.set_title("Conditional measurement error\nnear the top")
ax.legend(framealpha=0.85, edgecolor="none")
ax = axes[1]
ax.errorbar(eqse.raw_v2, eqse.boot_mean - eqse.raw_v2, yerr=[eqse.boot_mean - eqse.ci95_lo, eqse.ci95_hi - eqse.boot_mean],
            fmt="o", ms=3, color="tab:purple", capsize=2)
ax.axhline(0, color="grey", lw=0.8)
ax.set_xlabel("Raw score on version II (2026)")
ax.set_ylabel("Equated score minus raw score")
ax.set_title("Version adjustment at the top,\nwith 95% bootstrap CI")
fig.tight_layout()
savefig(fig, "fig03_measurement_error")
