"""Step 8: structural MRI measures from FreeSurfer (response to the reviewers).

Reads aseg.stats (intracranial volume, brain volumes, corpus callosum segments) and lh/rh.aparc.stats
(precentral cortical thickness) from each participant's FreeSurfer subject folder.
Group comparisons, and exploratory associations of corpus callosum volume and precentral thickness with
the gamma measures (Spearman; FDR within each sample).
Output: OUTPUT_DIR/FreeSurfer.xlsx"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import CFG, OUT, fdr_bh, load, partial_spearman, spearman, ttest_levene

FS = Path(CFG.FREESURFER_DIR)
CC = ["CC_Posterior", "CC_Mid_Posterior", "CC_Central", "CC_Mid_Anterior", "CC_Anterior"]


def subject_dir(sid, grp):
    pattern = CFG.FREESURFER_SUBJECT_PATTERN[grp].format(id=sid)
    hits = sorted(d for d in FS.glob(pattern) if (d / "stats" / "aseg.stats").exists())
    return hits[0] if hits else None


def aseg(path):
    txt = (path / "stats" / "aseg.stats").read_text(errors="ignore")
    meas = {m.group(1): float(m.group(2)) for m in re.finditer(r"^# Measure [^,]+, (\w[\w-]*), [^,]+, ([\d.]+)", txt, re.M)}
    vol = {}
    for line in txt.splitlines():
        if line.strip() and not line.startswith("#"):
            p = line.split()
            vol[p[4]] = float(p[3])
    return meas, vol


def precentral(path, hemi):
    for line in (path / "stats" / f"{hemi}.aparc.stats").read_text(errors="ignore").splitlines():
        p = line.split()
        if p and p[0] == "precentral":
            return float(p[4])                     # mean cortical thickness (mm)
    return np.nan


df = load()
rows = []
for _, r in df.iterrows():
    d = subject_dir(int(r.ID), r.Group)
    rec = dict(ID=int(r.ID))
    if d is not None:
        meas, vol = aseg(d)
        rec.update({k: vol.get(k, np.nan) for k in CC})
        rec.update(CC_total=sum(vol.get(k, np.nan) for k in CC), eTIV=meas.get("eTIV"),
                   BrainSegVol=meas.get("BrainSegVol"), TotalGrayVol=meas.get("TotalGrayVol"),
                   CerebralWhiteMatterVol=meas.get("CerebralWhiteMatterVol"),
                   precentral_thick_L=precentral(d, "lh"), precentral_thick_R=precentral(d, "rh"))
    rows.append(rec)
df = df.merge(pd.DataFrame(rows), on="ID")
print("FreeSurfer subjects found:", int(df.eTIV.notna().sum()), "of", len(df))
df["CC_total_rel"] = df.CC_total / df.eTIV * 1000          # mm3 per litre of eTIV
df["precentral_thick_asym"] = df.precentral_thick_L - df.precentral_thick_R
ft, pt = df[df.Group == "FT"], df[df.Group == "PT"]

measures = ([("Estimated total intracranial volume (cm3)", "eTIV", 1e-3), ("Brain segmentation volume (cm3)", "BrainSegVol", 1e-3),
             ("Total grey matter volume (cm3)", "TotalGrayVol", 1e-3),
             ("Cerebral white matter volume (cm3)", "CerebralWhiteMatterVol", 1e-3),
             ("Corpus callosum, total (mm3)", "CC_total", 1), ("Corpus callosum per litre eTIV (mm3/L)", "CC_total_rel", 1)]
            + [(f"{k.replace('_', ' ')} (mm3)", k, 1) for k in CC]
            + [("Precentral thickness, left (mm)", "precentral_thick_L", 1),
               ("Precentral thickness, right (mm)", "precentral_thick_R", 1)])
g = []
for lab, col, sc in measures:
    t = ttest_levene(ft[col] * sc, pt[col] * sc)
    g.append(dict(Measure=lab, FT_mean=ft[col].mean() * sc, FT_sd=ft[col].std() * sc, PT_mean=pt[col].mean() * sc,
                  PT_sd=pt[col].std() * sc, t=t["t"], df=t["df"], P=t["p"], d=t["d"]))
grp = pd.DataFrame(g)
seg = grp.Measure.str.startswith("CC ")
grp.loc[seg, "q (callosal segments)"] = fdr_bh(grp.loc[seg, "P"])
m = smf.ols("CC_total ~ C(Group, Treatment('FT')) + eTIV", df).fit()
grp.loc[len(grp)] = dict(Measure="Corpus callosum, total, adjusted for eTIV (preterm effect, mm3)",
                         PT_mean=m.params.iloc[1], t=m.tvalues.iloc[1], df=m.df_resid, P=m.pvalues.iloc[1])

ex = []
for sname, d in [("Whole sample", df), ("Preterm", pt)]:
    for x, y, cov in [("CC_total_rel", "MEG_laterality_index", []), ("CC_Central", "MEG_laterality_index", ["eTIV"]),
                      ("CC_Mid_Posterior", "MEG_laterality_index", ["eTIV"]), ("precentral_thick_L", "MEG_Contrapower", []),
                      ("precentral_thick_asym", "MEG_laterality_index", []), ("CC_total_rel", "Walk_alone_ca", [])]:
        rho, p, n = partial_spearman(d, x, y, cov) if cov else spearman(d[x], d[y])
        ex.append(dict(Sample=sname, X=x, Y=y, adjusted_for=", ".join(cov), rho=rho, P=p, n=n))
assoc = pd.DataFrame(ex)
for s in assoc.Sample.unique():
    k = assoc.Sample == s
    assoc.loc[k, "q"] = fdr_bh(assoc.loc[k, "P"])

with pd.ExcelWriter(OUT / "FreeSurfer.xlsx") as xw:
    grp.to_excel(xw, sheet_name="Group_comparison", index=False)
    assoc.to_excel(xw, sheet_name="Associations", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(grp.round(3).to_string(index=False))
print(assoc.round(3).to_string(index=False))
