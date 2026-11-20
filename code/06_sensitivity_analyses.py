"""Step 6: sensitivity and additional analyses reported in the response to the reviewers.

 A  covariate-adjusted group effects (age + sex; + K-ABC Mental Processing Composite; number of averaged epochs;
    ADHD Rating Scale) on contralateral and ipsilateral gamma power, the hemispheric difference
    (Group x Hemisphere interaction) and the laterality index
 B  exclusion analyses (autism spectrum disorder; retinopathy of prematurity; any visual concern;
    glasses not worn during MEG)
 C  exploratory Group x Sex analyses
 D  head position: left-right asymmetry in sensor-to-scalp distance
 E  alternative laterality measures (hemispheric difference in dB; absolute-denominator index; difference in %)
 F  hemispheric asymmetry versus overall gamma level as predictors of age at independent walking
    (OLS with standardised variables; Freedman-Lane permutation test and bootstrap confidence interval)
 G  ADHD Rating Scale and task performance of preterm children with visual concerns
Output: OUTPUT_DIR/Sensitivity_analyses.xlsx"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

from common import OUT, describe, load, partial_spearman, spearman, ttest_levene

df = load()
df["PT"] = (df.Group == "PT").astype(int)
df["g"] = df.PT - 0.5
df["s"] = df.Female - 0.5
df["D"] = df.MEG_Contrapower - df.MEG_Ipsipower
has_clinical = "ROP_status" in df
outcomes = [("Contralateral gamma power", "MEG_Contrapower"), ("Ipsilateral gamma power", "MEG_Ipsipower"),
            ("Hemispheric difference (interaction)", "D"), ("Laterality index", "MEG_laterality_index")]
sheets = {}

# ---- A: covariate-adjusted group effects ----
models = {"age + sex": "Age_month + Female", "age + sex + K-ABC MPC": "Age_month + Female + KABC_MPC",
          "number of averaged epochs": "nAvg_contra"}
if has_clinical:
    models.update({"ADHD-RS": "ADHD_RS", "age + sex + ADHD-RS": "Age_month + Female + ADHD_RS"})
rows = []
for oname, ocol in outcomes:
    for mname, cov in models.items():
        m = smf.ols(f"{ocol} ~ PT + {cov}", df).fit()
        lo, hi = m.conf_int().loc["PT"]
        rows.append(dict(Outcome=oname, Covariates=mname, n=int(m.nobs), B_preterm=m.params["PT"], CI_lo=lo, CI_hi=hi,
                         t=m.tvalues["PT"], df=int(m.df_resid), P=m.pvalues["PT"]))
sheets["A_covariates"] = pd.DataFrame(rows)


# ---- B: exclusion analyses ----
def main_tests(d):
    f_, p_ = d[d.Group == "FT"], d[d.Group == "PT"]
    inter = smf.ols("D ~ g", d).fit()
    c = ttest_levene(f_.MEG_Contrapower, p_.MEG_Contrapower)
    li = ttest_levene(f_.MEG_laterality_index, p_.MEG_laterality_index)
    rw, rp = spearman(d.MEG_laterality_index, d.Walk_alone_ca), spearman(p_.MEG_laterality_index, p_.Walk_alone_ca)
    return {"n FT/PT": f"{len(f_)}/{len(p_)}", "Interaction F": inter.tvalues["g"] ** 2, "Interaction P": inter.pvalues["g"],
            "Contra t": c["t"], "Contra P": c["p"], "Contra d": c["d"], "LI t": li["t"], "LI P": li["p"], "LI d": li["d"],
            "LI-walking rho (whole)": rw[0], "P (whole)": rw[1], "LI-walking rho (preterm)": rp[0], "P (preterm)": rp[1]}


excl = {"All participants": pd.Series(False, index=df.index), "Excluding autism spectrum disorder": df.ASD_dx == 1}
if has_clinical:
    rop = df.ROP_status == "yes"
    vis = rop | (df.Reduced_acuity == 1) | (df.Glasses_owned == 1) | (df.Strabismus == 1)
    excl.update({"Excluding confirmed ROP": rop,
                 "Excluding ROP, visual acuity < 0.8, glasses or strabismus": vis,
                 "Excluding glasses not worn during MEG": (df.Glasses_owned == 1) & (df.Glasses_during_MEG == 0)})
sheets["B_exclusions"] = pd.DataFrame([dict(Analysis=k, **main_tests(df[~mask])) for k, mask in excl.items()])

# ---- C: exploratory Group x Sex ----
rows = []
for oname, ocol in outcomes:
    m = smf.ols(f"{ocol} ~ g * s", df).fit()
    for term, lab in [("g", "Group"), ("s", "Sex"), ("g:s", "Group x Sex")]:
        rows.append(dict(Outcome=oname, Effect=lab, F=m.tvalues[term] ** 2, df=f"1, {int(m.df_resid)}", P=m.pvalues[term]))
sheets["C_sex"] = pd.DataFrame(rows)

# ---- D: head position (sensor-to-scalp distance asymmetry; > 0: left hemisphere farther from the sensors) ----
df["dist_asym_mm"] = df.sensor_scalp_left_upper_mm - df.sensor_scalp_right_upper_mm
ft, pt = df[df.Group == "FT"], df[df.Group == "PT"]
r = ttest_levene(ft.dist_asym_mm, pt.dist_asym_mm)
rows = [dict(Analysis="Distance asymmetry (mm), group comparison", FT=describe(ft.dist_asym_mm)[0],
             PT=describe(pt.dist_asym_mm)[0], statistic=f"t({r['df']:.1f}) = {r['t']:.2f}", P=r["p"])]
m = smf.ols("MEG_laterality_index ~ PT + dist_asym_mm", df).fit()
rows.append(dict(Analysis="Laterality index ~ group + distance asymmetry: group effect",
                 statistic=f"B = {m.params['PT']:.3f}, t({int(m.df_resid)}) = {m.tvalues['PT']:.2f}", P=m.pvalues["PT"]))
for sname, d in [("Whole sample", df), ("Preterm", pt)]:
    rho, p, n = partial_spearman(d, "MEG_laterality_index", "Walk_alone_ca", ["dist_asym_mm"])
    rows.append(dict(Analysis=f"Laterality index x age at independent walking | distance asymmetry ({sname})",
                     statistic=f"partial rho = {rho:.2f} (n = {n})", P=p))
sheets["D_head_position"] = pd.DataFrame(rows)

# ---- E: alternative laterality measures ----
Rc, Ri = 1 + df.MEG_Contrapower / 100, 1 + df.MEG_Ipsipower / 100
df["LI_dB_difference"] = 10 * np.log10(Rc) - 10 * np.log10(Ri)
df["LI_absolute_denominator"] = df.D / (df.MEG_Contrapower.abs() + df.MEG_Ipsipower.abs())
ft, pt = df[df.Group == "FT"], df[df.Group == "PT"]
rows = []
for lab, col in [("Laterality index (power ratios; main analysis)", "MEG_laterality_index"),
                 ("Hemispheric difference in dB", "LI_dB_difference"),
                 ("Absolute-denominator index, (C - I) / (|C| + |I|)", "LI_absolute_denominator"),
                 ("Hemispheric difference, contra - ipsi (%)", "D")]:
    r = ttest_levene(ft[col], pt[col])
    rw, rp = spearman(df[col], df.Walk_alone_ca), spearman(pt[col], pt.Walk_alone_ca)
    rows.append(dict(Measure=lab, FT=describe(ft[col])[0], PT=describe(pt[col])[0], t=r["t"], df=r["df"], P=r["p"],
                     d=r["d"], P_MannWhitney=stats.mannwhitneyu(ft[col], pt[col]).pvalue,
                     walking_rho_whole=rw[0], walking_P_whole=rw[1], walking_rho_preterm=rp[0], walking_P_preterm=rp[1]))
sheets["E_laterality_variants"] = pd.DataFrame(rows)

# ---- F: asymmetry versus overall level as predictors of age at independent walking ----
df["A_asymmetry"] = np.log(Rc / Ri)              # hemispheric asymmetry of the log power ratios
df["M_level"] = (np.log(Rc) + np.log(Ri)) / 2     # overall gamma level across hemispheres


def asym_level(d, n_perm=10000, n_boot=10000, seed=1):
    d = d[["Walk_alone_ca", "A_asymmetry", "M_level"]].dropna()
    z = (d - d.mean()) / d.std(ddof=1)
    full = smf.ols("Walk_alone_ca ~ A_asymmetry + M_level", z).fit()
    reduced = smf.ols("Walk_alone_ca ~ M_level", z).fit()
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(len(z)), z.A_asymmetry, z.M_level])
    XtX_inv = np.linalg.inv(X.T @ X)

    def t_asym(y):
        b = XtX_inv @ X.T @ y
        res = y - X @ b
        se = np.sqrt(res @ res / (len(y) - 3) * XtX_inv[1, 1])
        return b[1] / se

    t_obs = t_asym(z.Walk_alone_ca.values)
    fit, res = reduced.fittedvalues.values, reduced.resid.values
    count = sum(abs(t_asym(fit + rng.permutation(res))) >= abs(t_obs) for _ in range(n_perm))   # Freedman-Lane
    raw = smf.ols("Walk_alone_ca ~ A_asymmetry + M_level", d).fit()
    boot = []
    for _ in range(n_boot):                     # case-resampling bootstrap of the unstandardised coefficient
        b = d.iloc[rng.integers(0, len(d), len(d))]
        Xb = np.column_stack([np.ones(len(b)), b.A_asymmetry, b.M_level])
        boot.append(np.linalg.lstsq(Xb, b.Walk_alone_ca.values, rcond=None)[0][1])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(n=len(d), beta_std_asymmetry=full.params["A_asymmetry"], P_asymmetry=full.pvalues["A_asymmetry"],
                P_permutation=(count + 1) / (n_perm + 1), B_asymmetry=raw.params["A_asymmetry"], boot_CI_lo=lo,
                boot_CI_hi=hi, beta_std_level=full.params["M_level"], P_level=full.pvalues["M_level"], R2=full.rsquared,
                r_asymmetry_level=d.A_asymmetry.corr(d.M_level))


sheets["F_asymmetry_vs_level"] = pd.DataFrame([dict(Sample=s, **asym_level(d))
                                               for s, d in [("Preterm", df[df.Group == "PT"]), ("Whole sample", df)]])

# ---- G: ADHD Rating Scale; task performance of preterm children with visual concerns ----
if has_clinical:
    ft, pt = df[df.Group == "FT"], df[df.Group == "PT"]
    r = ttest_levene(ft.ADHD_RS, pt.ADHD_RS)
    rows = [dict(Analysis="ADHD-RS total, group comparison", FT=describe(ft.ADHD_RS)[0], PT=describe(pt.ADHD_RS)[0],
                 statistic=f"t({r['df']:.1f}) = {r['t']:.2f}", P=r["p"],
                 P_MannWhitney=stats.mannwhitneyu(ft.ADHD_RS, pt.ADHD_RS).pvalue)]
    for sname, d in [("Whole sample", df), ("Full-term", ft), ("Preterm", pt)]:
        for lab, col in [outcomes[0], outcomes[1], outcomes[3]]:
            rho, p, n = spearman(d.ADHD_RS, d[col])
            rows.append(dict(Analysis=f"ADHD-RS x {lab} ({sname})", statistic=f"rho = {rho:.2f} (n = {n})", P=p))
    pt_vis = pt.assign(vis=vis[pt.index])
    for lab, col in [("Mean response time (ms)", "RT_mean_ms"), ("Epochs averaged", "nAvg_contra")]:
        a, b = pt_vis.loc[pt_vis.vis, col], pt_vis.loc[~pt_vis.vis, col]
        rows.append(dict(Analysis=f"Preterm with vs without visual concern: {lab}",
                         statistic=f"median {a.median():.0f} (n = {len(a)}) vs {b.median():.0f} (n = {len(b)})",
                         P_MannWhitney=stats.mannwhitneyu(a, b).pvalue))
    rows.append(dict(Analysis="ROP status (preterm)", statistic=str(pt.ROP_status.value_counts().to_dict())))
    sheets["G_ADHD_vision"] = pd.DataFrame(rows)

with pd.ExcelWriter(OUT / "Sensitivity_analyses.xlsx") as xw:
    for k, v in sheets.items():
        v.to_excel(xw, sheet_name=k, index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 70)
for k, v in sheets.items():
    print(f"\n== {k}\n{v.round(4).to_string(index=False)}")
