"""Step 4: movement-related gamma activity (Results; Table 3; Fig. 3).

 - 2 (group) x 2 (hemisphere) mixed ANOVA for gamma power and for peak gamma frequency
 - post hoc comparisons in two families, Holm-corrected within each family:
     between-group comparisons within each hemisphere (independent t-tests, Welch if Levene P <= 0.05)
     within-group comparisons of contralateral and ipsilateral M1 (paired t-tests)
 - laterality index: independent t-test
 - model checks: Shapiro-Wilk tests of the residuals, Levene's tests, Mann-Whitney U tests
 - preprocessing: ICA components removed, epochs rejected and epochs averaged per group (Methods)
Output: OUTPUT_DIR/Table3_gamma.xlsx"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

from common import OUT, describe, holm, load, mixed_2x2, ttest_levene

df = load()
ft, pt = df[df.Group == "FT"], df[df.Group == "PT"]
sheets = {}

table3, anova, posthoc = [], [], []
for label, C, I in [("Gamma power (%)", "MEG_Contrapower", "MEG_Ipsipower"),
                    ("Peak gamma frequency (Hz)", "MEG_Contrapeakfreq", "MEG_Ipsipeakfreq")]:
    a, d = mixed_2x2(df, C, I)
    a.insert(0, "Measure", label)
    a["n FT/PT"] = f"{(d.Group == 'FT').sum()}/{(d.Group == 'PT').sum()}"
    anova.append(a)
    # between-group comparisons (all available values in each hemisphere)
    between = []
    for hemi, col in [("Contralateral", C), ("Ipsilateral", I)]:
        r = ttest_levene(ft[col], pt[col])
        between.append(r)
        (fm, nf), (pm, npt) = describe(ft[col]), describe(pt[col])
        table3.append({"Measure": f"{hemi} {label}", "Full-term mean (SD)": fm, "n FT": nf, "Preterm mean (SD)": pm,
                       "n PT": npt, "t": round(r["t"], 2), "df": round(r["df"], 1), "P (uncorrected)": r["p"],
                       "Cohen's d": round(r["d"], 2), "Welch": r["welch"]})
    for row, p_adj in zip(table3[-2:], holm([r["p"] for r in between])):
        row["P (Holm)"] = p_adj
    # within-group comparisons (children with values in both hemispheres)
    within = []
    for grp, g in [("Full-term", d[d.Group == "FT"]), ("Preterm", d[d.Group == "PT"])]:
        r = stats.ttest_rel(g[C], g[I])
        within.append(dict(Measure=label, Comparison=f"{grp}: contralateral vs ipsilateral",
                           t=r.statistic, df=len(g) - 1, P=r.pvalue))
    for row, p_adj in zip(within, holm([w["P"] for w in within])):
        row["P (Holm)"] = p_adj
    posthoc += within

r = ttest_levene(ft.MEG_laterality_index, pt.MEG_laterality_index)
(fm, nf), (pm, npt) = describe(ft.MEG_laterality_index), describe(pt.MEG_laterality_index)
table3.append({"Measure": "Laterality index", "Full-term mean (SD)": fm, "n FT": nf, "Preterm mean (SD)": pm, "n PT": npt,
               "t": round(r["t"], 2), "df": round(r["df"], 1), "P (uncorrected)": r["p"], "P (Holm)": np.nan,
               "Cohen's d": round(r["d"], 2), "Welch": r["welch"]})
sheets["Table3"] = pd.DataFrame(table3)
sheets["ANOVA"] = pd.concat(anova)
sheets["Within_group_posthoc"] = pd.DataFrame(posthoc)

# ---- model checks and non-parametric tests (gamma power) ----
_, d = mixed_2x2(df, "MEG_Contrapower", "MEG_Ipsipower")
checks = []
for lab, col in [("Hemispheric difference (interaction)", "D"), ("Hemisphere average (group effect)", "M"),
                 ("Laterality index", "MEG_laterality_index")]:
    sw = stats.shapiro(smf.ols(f"{col} ~ g", d).fit().resid)
    checks.append(dict(Measure=lab, test="Shapiro-Wilk (residuals)", statistic=sw.statistic, P=sw.pvalue))
    lev = stats.levene(d.loc[d.Group == "FT", col], d.loc[d.Group == "PT", col])
    checks.append(dict(Measure=lab, test="Levene", statistic=lev.statistic, P=lev.pvalue))
for grp, g in [("Full-term", ft), ("Preterm", pt)]:
    for col in ["MEG_Contrapower", "MEG_Ipsipower"]:
        sw = stats.shapiro(g[col])
        checks.append(dict(Measure=f"{col}, {grp}", test="Shapiro-Wilk", statistic=sw.statistic, P=sw.pvalue))
for lab, col in [("Hemispheric difference (interaction)", "D"), ("Contralateral gamma power", "MEG_Contrapower"),
                 ("Ipsilateral gamma power", "MEG_Ipsipower"), ("Laterality index", "MEG_laterality_index")]:
    u = stats.mannwhitneyu(d.loc[d.Group == "FT", col], d.loc[d.Group == "PT", col], alternative="two-sided")
    checks.append(dict(Measure=lab, test="Mann-Whitney U", statistic=u.statistic, P=u.pvalue))
sheets["Model_checks"] = pd.DataFrame(checks)

# ---- preprocessing details by group (Methods) ----
prep = []
for lab, col in [("ICA components removed", "ica_removed"), ("Epochs rejected for artefacts", "n_rejected_epochs"),
                 ("Epochs averaged", "nAvg_contra")]:
    r = ttest_levene(ft[col], pt[col])
    prep.append(dict(Measure=lab, FT=describe(ft[col])[0], PT=describe(pt[col])[0], t=r["t"], df=r["df"], P=r["p"],
                     P_MannWhitney=stats.mannwhitneyu(ft[col].dropna(), pt[col].dropna()).pvalue))
sheets["Preprocessing"] = pd.DataFrame(prep)

with pd.ExcelWriter(OUT / "Table3_gamma.xlsx") as xw:
    for k, v in sheets.items():
        v.to_excel(xw, sheet_name=k, index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
for k, v in sheets.items():
    print(f"\n== {k}\n{v.round(4).to_string(index=False)}")
