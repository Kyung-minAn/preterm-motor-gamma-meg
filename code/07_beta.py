"""Step 7: movement-related beta activity (response to the reviewers).

Beta power (15-25 Hz) in the windows of An et al. (2021; same task and MEG system): pre-movement beta decrease
(-200 to 0 ms) and movement-related beta decrease (0 to 500 ms); mean baseline-normalised power in each window.
Analysed with the same 2 x 2 mixed ANOVA and Holm-corrected post hoc comparisons as gamma power.
Output: OUTPUT_DIR/Beta.xlsx"""
import pandas as pd

from common import OUT, describe, holm, load, mixed_2x2, ttest_levene

df = load()
ft, pt = df[df.Group == "FT"], df[df.Group == "PT"]
rows = []
for label, C, I in [("Pre-movement beta decrease, -200 to 0 ms (%)", "beta_pre_contra", "beta_pre_ipsi"),
                    ("Movement-related beta decrease, 0 to 500 ms (%)", "beta_move_contra", "beta_move_ipsi")]:
    a, _ = mixed_2x2(df, C, I)
    for _, e in a.iterrows():
        rows.append(dict(Measure=label, Test=f"ANOVA: {e.Effect}", Statistic=f"F({e.df1}, {e.df2}) = {e.F:.2f}",
                         P=e.P, Effect_size=f"partial eta2 = {e.partial_eta2:.3f}"))
    res = [ttest_levene(ft[col], pt[col]) for col in (C, I)]
    for hemi, col, r, p_holm in zip(["contralateral", "ipsilateral"], (C, I), res, holm([r["p"] for r in res])):
        rows.append(dict(Measure=label, Test=f"Full-term vs preterm, {hemi}",
                         Statistic=f"FT {describe(ft[col])[0]}, PT {describe(pt[col])[0]}; t({r['df']:.1f}) = {r['t']:.2f}",
                         P=r["p"], P_Holm=p_holm, Effect_size=f"d = {r['d']:.2f}"))
out = pd.DataFrame(rows)
out.to_excel(OUT / "Beta.xlsx", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
print(out.round(4).to_string(index=False))
