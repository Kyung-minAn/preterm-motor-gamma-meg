"""Step 2: build the participant-level analysis dataset.

Inputs (paths in the configuration file):
  BEHAVIOUR_XLSX   one row per child: group (Groups: 0 = full-term, 1 = preterm), age, perinatal measures,
                   motor milestones (corrected age for preterm children), behavioural response times,
                   K-ABC and VABS-II scores
  BACKGROUND_XLSX  sex, handedness, autism diagnosis, K-ABC scales, visual acuity, growth at 5 years
  CLINICAL_XLSX    retinopathy of prematurity, glasses, strabismus, ADHD Rating Scale (optional)
  OUTPUT_DIR/meg_measures.csv from step 1
Output: OUTPUT_DIR/analysis_dataset.csv

Laterality index: LI = (Rc - Ri) / (Rc + Ri), where R = 1 + gamma/100 is gamma power expressed as a ratio to the
baseline power (always positive, so the index stays within -1 to +1 even when power falls below baseline).
"""
import numpy as np
import pandas as pd

from common import CFG, DATASET, MEG_MEASURES

# ---- behavioural and perinatal data ----
main = pd.read_excel(CFG.BEHAVIOUR_XLSX)
main = main.drop(columns=[c for c in main.columns if str(c).startswith("MEG_")])   # MEG measures come from step 1
for c in ["Walk_alone_ca", "VABS_Motor_Fine", "VABS_Motor_Gross", "Sitdown_ca", "Head_controll_ca", "SRS_total"]:
    main[c] = pd.to_numeric(main[c], errors="coerce")
main["Group"] = main["Groups"].map({0: "FT", 1: "PT"})
main["RT_mean_ms"] = main["Responsetime_Mean"] * 1000
main["RT_sd_ms"] = main["Responsetime_SD"] * 1000

# ---- background variables ----
bg = pd.read_excel(CFG.BACKGROUND_XLSX, header=1)
bg.columns = [str(c) for c in bg.columns]
bg = bg.rename(columns={"index": "Handedness_LQ", "sex": "Sex", "hand": "Hand",
                        "K-ABC Sequential Processing Scale": "KABC_Sequential",
                        "K-ABC Simultaneous Processing Scale": "KABC_Simultaneous",
                        "K-ABC Mental Processing Scale": "KABC_MPC"})
bg["ID"] = pd.to_numeric(bg["ID"], errors="coerce")
bg = bg.dropna(subset=["ID"]).drop_duplicates("ID")
keep = ["ID", "Sex", "Hand", "Handedness_LQ", "ASD_judge", "5y_HS", "5y_Ht_SD", "5y_BW_SD",
        "VABS_cd_total", "VABS_dls_total", "VABS_msd_total", "Rt_eye", "Lt_eye",
        "KABC_Sequential", "KABC_Simultaneous", "KABC_MPC"]
df = main.merge(bg[keep], on="ID", how="left", validate="one_to_one")
for c in ["Handedness_LQ", "5y_HS", "5y_Ht_SD", "5y_BW_SD", "VABS_cd_total", "VABS_dls_total", "VABS_msd_total",
          "KABC_Sequential", "KABC_Simultaneous", "KABC_MPC"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["Female"] = (df["Sex"].astype(str).str.strip().str.lower() == "f").astype(int)
df["ASD_dx"] = (df["ASD_judge"].astype(str).str.strip() == "A").astype(int)
df["Reduced_acuity"] = (df["Rt_eye"].astype(str).str.strip().str.startswith("<")
                        | df["Lt_eye"].astype(str).str.strip().str.startswith("<")).astype(int)
ga = df["Gest_Age_Week"]
df["GA_category"] = np.select([ga < 28, ga < 32, ga < 37], ["<28 wk", "28-31 wk", "32-36 wk"], "term")

# ---- clinical variables (retinopathy of prematurity, glasses, strabismus, ADHD Rating Scale) ----
if getattr(CFG, "CLINICAL_XLSX", None):
    cl = pd.read_excel(CFG.CLINICAL_XLSX, header=None).iloc[2:, :10]
    cl.columns = ["grp", "ID", "ROP", "Note", "Glasses_during_MEG", "ADHD_RS", "sex", "Rt_eye_new", "Lt_eye_new",
                  "Correction_new"]
    cl = cl.dropna(subset=["ID"])
    cl["ID"] = cl["ID"].astype(int)
    cl["ROP_status"] = cl["ROP"].map(lambda v: "yes" if v == 1 else "no" if v == 0 else "pending")
    cl["ADHD_RS"] = pd.to_numeric(cl["ADHD_RS"], errors="coerce")
    cl["Glasses_owned"] = cl["Correction_new"].astype(str).str.contains("eyeglass").astype(int)
    cl["Strabismus"] = cl["Note"].astype(str).str.contains("Strabismus").astype(int)
    df = df.merge(cl[["ID", "ROP_status", "Glasses_owned", "Glasses_during_MEG", "Strabismus", "ADHD_RS"]],
                  on="ID", how="left", validate="one_to_one")

# ---- MEG measures ----
meg = pd.read_csv(MEG_MEASURES).drop(columns="Group")
df = df.merge(meg, on="ID", how="left", validate="one_to_one")
df["MEG_Contrapower"] = df["gamma_contra"]
df["MEG_Ipsipower"] = df["gamma_ipsi"]
Rc, Ri = 1 + df["gamma_contra"] / 100, 1 + df["gamma_ipsi"] / 100
df["MEG_laterality_index"] = (Rc - Ri) / (Rc + Ri)
df["MEG_Contrapeakfreq"] = df["peakfreq_contra"]
df["MEG_Ipsipeakfreq"] = df["peakfreq_ipsi"]

assert df["MEG_Contrapower"].notna().all(), "MEG measures missing for some participants"
df.to_csv(DATASET, index=False)
print(f"Saved {DATASET}: {df.shape[0]} children ({(df.Group == 'FT').sum()} full-term, {(df.Group == 'PT').sum()} preterm)")
