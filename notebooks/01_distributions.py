# %% [markdown]
# # 01 - Score distributions and ceiling indicators (2020-2026)
#
# Source: NAEC equating tables (`Stats_YYYY.xlsx`): number of test-takers at each raw score,
# per subject, year and exam version.
#
# Ceiling indicators (computed per exam version, then averaged across versions weighted by N):
# - `pct_max`: % of test-takers at the maximum raw score.
# - `pct_top3`: % of test-takers in the top 3 raw-score points (max, max-1, max-2).
# - `pts_top1pct` / `pts_top5pct`: how many raw-score points the top 1% / top 5% of test-takers
#   are spread over (resolution at the top; 1 = everybody in the top 1% has the same score).
# - `tail_flatness`: mean count at the top 3 points divided by the mean count at the 5 points
#   just below them. A test with headroom has a tail that thins out before the maximum
#   (ratio well below 1); a ratio near or above 1 means test-takers are pressed against the maximum.

# %%
from common import *
from load import load_equating
import matplotlib.ticker as mticker

eq = load_equating()
eq = eq[eq.subject.isin(["MATH", "ENG", "GEO", "HIST", "RUS"])]


def version_stats(g):
    N = g.n.sum()
    M = g.max_raw.iloc[0]
    x, n = g.raw.values, g.n.values
    mean = (x * n).sum() / N
    sd = np.sqrt(((x - mean) ** 2 * n).sum() / N)
    cum = np.cumsum(n[::-1])[::-1]  # number scoring >= x

    def pts(q):
        return M - x[cum / N >= q].max() + 1

    return pd.Series({
        "N": N, "max_raw": M, "mean": mean, "mean_pct": 100 * mean / M, "sd": sd,
        "skew": ((x - mean) ** 3 * n).sum() / N / sd ** 3,
        "n_max": n[x == M].sum(), "pct_max": 100 * n[x == M].sum() / N,
        "n_top3": n[x >= M - 2].sum(), "pct_top3": 100 * n[x >= M - 2].sum() / N,
        "pts_top1pct": pts(0.01), "pts_top5pct": pts(0.05),
        "tail_flatness": n[x >= M - 2].mean() / n[(x >= M - 7) & (x <= M - 3)].mean(),
    })


vs = (eq.sort_values("raw").groupby(["subject", "year", "version"]).apply(version_stats, include_groups=False)
      .reset_index())
vs = vs[vs.N >= 500]  # drop tiny special versions
vs.to_csv(OUT / "t01_version_stats.csv", index=False)
vs.round(2)

# %% [markdown]
# Sanity check against NAEC's published statistical analysis for 2025 math
# (version I: N=6525, mean 19.17, 25 at max; version II: N=6453, mean 18.33, 22 at max).

# %%
vs[(vs.subject == "MATH") & (vs.year == 2025)][["version", "N", "mean", "n_max"]]

# %%
def wavg(g):
    w = g.N
    out = {c: np.average(g[c], weights=w) for c in ["mean_pct", "sd", "skew", "pct_max", "pct_top3",
                                                     "pts_top1pct", "pts_top5pct", "tail_flatness"]}
    out.update(N=w.sum(), versions=len(g), max_raw=g.max_raw.iloc[0], n_max=g.n_max.sum(), n_top3=g.n_top3.sum(),
               pts_top1pct_min=g.pts_top1pct.min(), pts_top1pct_max=g.pts_top1pct.max())
    return pd.Series(out)


ys = vs.groupby(["subject", "year"]).apply(wavg, include_groups=False).reset_index()
ys.to_csv(OUT / "t01_subject_year.csv", index=False)
ys.round(2)

# %% [markdown]
# ## Figure 1 - Mathematics: the top of the distribution, by year
# Share of test-takers at each raw score (both versions pooled), log scale, from raw 30 upward.
# 2020 had a 59-point test; from 2021 the maximum is 51.

# %%
m = eq[eq.subject == "MATH"].groupby(["year", "raw"]).n.sum().reset_index()
m["share"] = 100 * m.n / m.groupby("year").n.transform("sum")
fig, ax = plt.subplots(figsize=(6.5, 3.3))
colors = {2020: "black", 2021: "#6baed6", 2022: "#2171b5",
          2023: "#fdae6b", 2024: "#fd8d3c", 2025: "#e6550d", 2026: "#a63603"}
labels = {2020: "2020 (max 59)", 2021: "2021 (old format)", 2022: "2022 (old format)",
          2023: "2023 (new format)", 2024: "2024 (new format)", 2025: "2025 (new format)", 2026: "2026 (new format)"}
for y in sorted(m.year.unique()):
    d = m[(m.year == y) & (m.raw >= 30)]
    ax.plot(d.raw, d.share, marker="o", ms=2.5, lw=2.4 if y in (2020, 2021, 2026) else 1.2,
            color=colors[y], label=labels[y])
ax.axvline(51, color="dimgray", ls="--", lw=1)
ax.set_yscale("log")
ax.yaxis.set_major_locator(mticker.FixedLocator([0.1, 0.2, 0.5, 1, 2]))
ax.yaxis.set_minor_locator(mticker.NullLocator())
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}%"))
ax.set_ylim(0.08, 3)
ax.text(51.3, 1.6, "max = 51\n(2021-2026)", color="dimgray", fontsize=9.5)
ax.set_xlabel("Raw score (mathematics)")
ax.set_ylabel("Share of test-takers at each score (log scale)")
ax.set_title("Mathematics: upper tail of the raw-score distribution")
ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.16))
savefig(fig, "fig01_math_upper_tail")

# %% [markdown]
# ## Figure 2 - Ceiling indicators by subject and year

# %%
fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.5))
colors = {"MATH": "tab:red", "ENG": "tab:blue", "GEO": "tab:green", "HIST": "tab:orange"}
for s, c in colors.items():
    d = ys[ys.subject == s]
    for ax, col in zip(axes, ["pct_top3", "pts_top1pct", "tail_flatness"]):
        ax.plot(d.year, d[col], marker="o", color=c, label=SUBJ_NAME[s])
axes[0].set_title("% of test-takers in\nthe top 3 points")
axes[1].set_title("Score points occupied\nby the top 1%")
axes[2].set_title("Tail flatness\nnear the maximum")
axes[2].axhline(1, color="grey", lw=0.8, ls=":")
for ax in axes:
    ax.set_xlabel("Year")
handles, labels_ = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_, loc="lower center", ncol=4,
           framealpha=0.85, edgecolor="none")
fig.tight_layout(rect=[0, 0.11, 1, 1])
savefig(fig, "fig02_ceiling_indicators")

# %% [markdown]
# ## Effect of shortening the tests
# Maximum raw score changes: Mathematics 59 -> 51 (2021), Georgian 70 -> 60 (2022), English 80 -> 70 (2023).
# Compare the indicators in the year before vs the year of the change.

# %%
changes = [("MATH", 2020, 2021), ("GEO", 2021, 2022), ("ENG", 2022, 2023)]
cmp = pd.concat([ys[(ys.subject == s) & ys.year.isin([a, b])] for s, a, b in changes])
cmp = cmp[["subject", "year", "max_raw", "N", "mean_pct", "pct_max", "pct_top3", "pts_top1pct", "pts_top5pct", "tail_flatness"]]
cmp.to_csv(OUT / "t01_test_length_changes.csv", index=False)
cmp.round(2)

# %% [markdown]
# ## Mathematics: how many students share each top score (both versions pooled)

# %%
top = (eq[(eq.subject == "MATH") & (eq.raw >= eq.max_raw - 4)]
       .assign(pts_below_max=lambda d: d.max_raw - d.raw)
       .groupby(["year", "pts_below_max"]).n.sum().unstack())
top.columns = [f"max-{c}" if c else "max" for c in top.columns]
top["top3_total"] = top[["max", "max-1", "max-2"]].sum(axis=1)
top.to_csv(OUT / "t01_math_top_counts.csv")
top
