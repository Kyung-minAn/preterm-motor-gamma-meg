"""Step 3: participant characteristics (Table 1), behavioural group comparisons (Fig. 1B) and correlations of
response-time measures with perinatal and motor measures (Table 2).
Output: OUTPUT_DIR/Table1_Table2_behaviour.xlsx"""
import pandas as pd
from scipy import stats

from common import OUT, fdr_bh, load, spearman, ttest_levene

df = load()
ft, pt = df[df.Group == "FT"], df[df.Group == "PT"]

# ---- Table 1: continuous variables (mean, SD, median, IQR, range; t-test with Levene/Welch) ----
cont = [
    ("Age at experiment (months)", "Age_month"),
    ("K-ABC Achievement", "KABC_Achievement"),
    ("Gestational age (weeks)", "Gest_Age_Week"),
    ("Birth weight (g)", "Gest_Weight"),
    ("Birth length (cm)", "Birth_Height"),
    ("Birth head circumference (cm)", "Birth_HeadCircumference"),
    ("Age at head control (months)", "Head_controll_ca"),
    ("Age at sitting unsupported (months)", "Sitdown_ca"),
    ("Age at independent walking (months)", "Walk_alone_ca"),
    ("K-ABC Hand Movements", "KABC_HandMovement"),
    ("VABS-II Gross Motor", "VABS_Motor_Gross"),
    ("VABS-II Fine Motor", "VABS_Motor_Fine"),
    ("Mean response time (ms)", "RT_mean_ms"),            # Fig. 1B
    ("Response-time variability, SD (ms)", "RT_sd_ms"),   # Fig. 1B
    # additional characteristics reported in the response to the reviewers
    ("Handedness (Edinburgh laterality quotient)", "Handedness_LQ"),
    ("Head circumference at 5 years (cm)", "5y_HS"),
    ("Height at 5 years (SD score)", "5y_Ht_SD"),
    ("Weight at 5 years (SD score)", "5y_BW_SD"),
    ("K-ABC Mental Processing Composite", "KABC_MPC"),
    ("SRS total (T score)", "SRS_total"),
]


def desc(s):
    s = s.dropna().astype(float)
    q1, q3 = s.quantile([0.25, 0.75])
    return dict(n=len(s), mean=s.mean(), sd=s.std(ddof=1), median=s.median(), q1=q1, q3=q3, min=s.min(), max=s.max())


rows = []
for label, col in cont:
    if col not in df or ft[col].notna().sum() < 2 or pt[col].notna().sum() < 2:
        continue   # skipped when a group has fewer than two values
    a, b, t = desc(ft[col]), desc(pt[col]), ttest_levene(ft[col], pt[col])
    row = {"Variable": label}
    for g, s in [("FT", a), ("PT", b)]:
        row.update({f"{g} n": s["n"], f"{g} mean (SD)": f"{s['mean']:.2f} ({s['sd']:.2f})",
                    f"{g} median [IQR]": f"{s['median']:.2f} [{s['q1']:.2f}-{s['q3']:.2f}]",
                    f"{g} range": f"{s['min']:.2f}-{s['max']:.2f}"})
    row.update({"t": round(t["t"], 2), "df": round(t["df"], 1), "P": t["p"], "Cohen's d": round(t["d"], 2),
                "Welch": t["welch"]})
    rows.append(row)
table1 = pd.DataFrame(rows)

cat = []
for label, col in [("Female sex", "Female"), ("Autism spectrum disorder diagnosis", "ASD_dx")]:
    tbl = pd.crosstab(df.Group, df[col]).reindex(index=["FT", "PT"], columns=[0, 1], fill_value=0)
    cat.append({"Variable": label, "FT": f"{tbl.loc['FT', 1]}/{tbl.loc['FT'].sum()}",
                "PT": f"{tbl.loc['PT', 1]}/{tbl.loc['PT'].sum()}", "Fisher P": stats.fisher_exact(tbl.values)[1]})
table1_cat = pd.DataFrame(cat)
ga_cat = pt["GA_category"].value_counts().rename_axis("Gestational age category").reset_index(name="n preterm")

# ---- Table 2: response-time measures x perinatal and motor measures (Spearman; BH-FDR over 7 per RT measure) ----
ys = [("Gestational age", "Gest_Age_Week"), ("Birth weight", "Gest_Weight"), ("Birth length", "Birth_Height"),
      ("Birth head circumference", "Birth_HeadCircumference"), ("K-ABC Hand Movements", "KABC_HandMovement"),
      ("VABS-II Gross Motor", "VABS_Motor_Gross"), ("VABS-II Fine Motor", "VABS_Motor_Fine")]
t2 = []
for xl, xc in [("Mean response time", "RT_mean_ms"), ("Response-time variability", "RT_sd_ms")]:
    res = [spearman(df[xc], df[yc]) for _, yc in ys]
    q = fdr_bh([r[1] for r in res])
    for (yl, _), (rho, p, n), qq in zip(ys, res, q):
        t2.append(dict(X=xl, Y=yl, rho=round(rho, 2), P=p, q=qq, n=n))
table2 = pd.DataFrame(t2)

with pd.ExcelWriter(OUT / "Table1_Table2_behaviour.xlsx") as xw:
    table1.to_excel(xw, sheet_name="Table1_continuous", index=False)
    table1_cat.to_excel(xw, sheet_name="Table1_categorical", index=False)
    ga_cat.to_excel(xw, sheet_name="Preterm_GA_categories", index=False)
    table2.to_excel(xw, sheet_name="Table2_RT_correlations", index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 45)
print(table1[["Variable", "FT n", "FT mean (SD)", "PT n", "PT mean (SD)", "t", "df", "P", "Cohen's d"]].to_string(index=False))
print(table1_cat.to_string(index=False))
print(ga_cat.to_string(index=False))
print(table2.round(3).to_string(index=False))
