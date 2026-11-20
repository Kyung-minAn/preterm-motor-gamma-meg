"""Step 1: extract participant-level MEG measures from the Brainstorm database.

Input (paths in the configuration file):
  TFR_FILES        CSV with one row per participant and hemisphere: ID, Group (FT/PT), side (contra/ipsi), file.
                   'file' is the Brainstorm time-frequency file (timefreq_morlet_*_ersd_time.mat) relative to
                   BRAINSTORM_ROOT. Each file contains the baseline-normalised power (% change relative to
                   -2000 to -1500 ms) of the first principal component of the left and right precentral scouts.
  BRAINSTORM_ROOT  root folder of the Brainstorm database.
Output: OUTPUT_DIR/meg_measures.csv

Measures (right index-finger task: contralateral = left precentral gyrus, ipsilateral = right precentral gyrus):
  gamma power            mean baseline-normalised power, 70-90 Hz x 0-100 ms after the button press (An et al., 2018)
  peak gamma frequency   frequency of maximal power within 60-100 Hz x -100 to 200 ms (An et al., 2018);
                         peaks above 95 Hz are set to missing
  beta power             mean power, 15-25 Hz, -200 to 0 ms (pre-movement) and 0 to 500 ms (movement-related)
                         (An et al., 2021)
  preprocessing details  number of averaged epochs, epochs rejected for artefacts, ICA components removed,
                         sensor-to-scalp distance over the left and right upper head
"""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

from common import CFG, MEG_MEASURES

ROOT = Path(CFG.BRAINSTORM_ROOT)
SCOUT = {"contra": "precentral L", "ipsi": "precentral R"}   # right index-finger movement

GAMMA = (70, 90, 0.0, 0.1)          # Hz, Hz, s, s
PEAK = (60, 100, -0.1, 0.2)
PEAK_MAX_HZ = 95
BETA = {"beta_pre": (15, 25, -0.2, 0.0), "beta_move": (15, 25, 0.0, 0.5)}


def loadmat(f, **kw):
    return sio.loadmat(str(f), squeeze_me=True, struct_as_record=False, **kw)


_cache = {}


def read_tf(f):
    if f not in _cache:
        m = loadmat(ROOT / f, variable_names=["TF", "Time", "Freqs", "RowNames", "nAvg"])
        TF = np.asarray(m["TF"])
        TF = TF[None] if TF.ndim == 2 else TF                      # rows x time x frequency
        _cache[f] = (TF, np.asarray(m["Time"], float), np.asarray(m["Freqs"], float),
                     list(np.atleast_1d(m["RowNames"])), int(m["nAvg"]))
    return _cache[f]


def window(tf_row, t, fr, f0, f1, t0, t1):
    ti, fi = (t >= t0) & (t <= t1), (fr >= f0) & (fr <= f1)
    return tf_row[np.ix_(ti, fi)], t[ti], fr[fi]


files = pd.read_csv(CFG.TFR_FILES)
rows = []
for sid, sub in files.groupby("ID", sort=True):
    rec = dict(ID=int(sid), Group=sub["Group"].iloc[0])
    for side in ["contra", "ipsi"]:
        hit = sub[sub["side"] == side]
        f = (hit if len(hit) else sub)["file"].iloc[0]     # both scouts are stored in the same file
        TF, t, fr, names, navg = read_tf(f)
        row = TF[names.index(SCOUT[side])]
        rec[f"tf_file_{side}"] = f
        rec[f"nAvg_{side}"] = navg
        rec[f"gamma_{side}"] = window(row, t, fr, *GAMMA)[0].mean()
        W, tw, fw = window(row, t, fr, *PEAK)
        it, jf = np.unravel_index(np.argmax(W), W.shape)
        rec[f"peakfreq_{side}"] = fw[jf] if fw[jf] <= PEAK_MAX_HZ else np.nan
        for lab, w in BETA.items():
            rec[f"{lab}_{side}"] = window(row, t, fr, *w)[0].mean()

    # epochs: the time-frequency file sits in the folder of the participant's epoched trials
    ep_dir = ROOT / Path(rec["tf_file_contra"]).parent
    bad = loadmat(ep_dir / "brainstormstudy.mat").get("BadTrials", [])
    bad = [bad] if isinstance(bad, str) else list(np.atleast_1d(bad))
    rec["n_epochs"] = len(list(ep_dir.glob("data_*trial*.mat")))
    rec["n_rejected_epochs"] = len(bad)

    # ICA components removed and sensor-to-scalp distance, from the channel file of the continuous recording
    raw = sorted(d for d in ep_dir.parent.glob("@raw*notch") if d.is_dir())
    if raw:
        ch = loadmat(sorted(raw[0].glob("channel*.mat"))[0])
        proj = [q for q in np.atleast_1d(ch.get("Projector", [])) if "ICA" in str(getattr(q, "Comment", ""))]
        if proj:
            mask = np.atleast_1d(proj[-1].CompMask)
            rec.update(ica_components=len(mask), ica_removed=int(mask.sum()))
        hp = np.asarray(ch["HeadPoints"].Loc, float)
        meg = np.array([np.asarray(c.Loc, float).reshape(3, -1)[:, 0]
                        for c in np.atleast_1d(ch["Channel"]) if str(c.Type) == "MEG"])
        if hp.size and len(meg):
            dist = np.sqrt(((meg[:, None, :] - hp.T[None, :, :]) ** 2).sum(-1)).min(axis=1)
            upper = meg[:, 2] > np.median(hp[2])
            rec.update(sensor_scalp_median_mm=np.median(dist) * 1000,
                       sensor_scalp_left_upper_mm=dist[upper & (meg[:, 1] > 0.02)].mean() * 1000,
                       sensor_scalp_right_upper_mm=dist[upper & (meg[:, 1] < -0.02)].mean() * 1000)
    rows.append(rec)

out = pd.DataFrame(rows)
out.drop(columns=["tf_file_contra", "tf_file_ipsi"]).to_csv(MEG_MEASURES, index=False)
print(f"Saved {MEG_MEASURES} ({len(out)} participants)")
print(out.groupby("Group")[["gamma_contra", "gamma_ipsi", "peakfreq_contra", "peakfreq_ipsi", "nAvg_contra"]]
      .agg(["mean", "std", "count"]).round(2).T.to_string())
