"""Step 5: correlation analyses (Results; Fig. 4).

 - laterality index x early motor milestones and motor scores (6 measures), whole sample and preterm group;
   Spearman, Benjamini-Hochberg FDR over the 6 correlations within each sample
 - the same as partial Spearman correlations controlling for gestational age and age at testing
 - exploratory: contralateral and ipsilateral gamma power, laterality index and contralateral peak gamma frequency
   x mean response time and response-time variability (8 correlations), whole sample and each group;
   FDR over the 8 correlations within each sample
Output: OUTPUT_DIR/Correlations.xlsx"""
import pandas as pd

from common import OUT, fdr_bh, load, partial_spearman, spearman

df = load()
samples = {"Whole sample": df, "Preterm": df[df.Group == "PT"], "Full-term": df[df.Group == "FT"]}
dev = [("Age at head control", "Head_controll_ca"), ("Age at sitting unsupported", "Sitdown_ca"),
       ("Age at independent walking", "Walk_alone_ca"), ("K-ABC Hand Movements", "KABC_HandMovement"),
       ("VABS-II Gross Motor", "VABS_Motor_Gross"), ("VABS-II Fine Motor", "VABS_Motor_Fine")]
gamma = [("Contralateral gamma power", "MEG_Contrapower"), ("Ipsilateral gamma power", "MEG_Ipsipower"),
         ("Laterality index", "MEG_laterality_index"), ("Contralateral peak gamma frequency", "MEG_Contrapeakfreq")]
rt = [("Mean response time", "RT_mean_ms"), ("Response-time variability", "RT_sd_ms")]


def family(pairs, d, covars=None):
    res = [partial_spearman(d, x, y, covars) if covars else spearman(d[x], d[y]) for (_, x), (_, y) in pairs]
    q = fdr_bh([r[1] for r in res])
    return [dict(X=xl, Y=yl, rho=r[0], P=r[1], q=qq, n=r[2]) for ((xl, _), (yl, _)), r, qq in zip(pairs, res, q)]


sheets = {}
rows = []
li = ("Laterality index", "MEG_laterality_index")
for sname in ["Whole sample", "Preterm"]:
    for model, cov in [("zero-order", None), ("partial: gestational age + age at testing", ["Gest_Age_Week", "Age_month"])]:
        for r in family([(li, y) for y in dev], samples[sname], cov):
            rows.append(dict(Sample=sname, Model=model, **r))
sheets["LI_development"] = pd.DataFrame(rows)

rows = []
for sname, d in samples.items():
    for r in family([(g, y) for g in gamma for y in rt], d):
        rows.append(dict(Sample=sname, **r))
sheets["Gamma_RT_exploratory"] = pd.DataFrame(rows)

with pd.ExcelWriter(OUT / "Correlations.xlsx") as xw:
    for k, v in sheets.items():
        v.to_excel(xw, sheet_name=k, index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
for k, v in sheets.items():
    print(f"\n== {k}\n{v.round(4).to_string(index=False)}")
