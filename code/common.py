"""Shared configuration loading and statistical helpers used by all analysis scripts."""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]


def _load_config():
    """Load the user configuration (paths to the data, which are not part of this repository).

    The configuration file is read from the path in the environment variable PRETERM_MEG_CONFIG or,
    if that is not set, from config.py in the repository root (copy config_example.py and edit it)."""
    path = Path(os.environ.get("PRETERM_MEG_CONFIG", REPO / "config.py"))
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}. Copy config_example.py to config.py "
                                "(or set PRETERM_MEG_CONFIG) and edit the paths to the data.")
    spec = importlib.util.spec_from_file_location("preterm_meg_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CFG = _load_config()
OUT = Path(CFG.OUTPUT_DIR)
OUT.mkdir(parents=True, exist_ok=True)
MEG_MEASURES = OUT / "meg_measures.csv"      # written by 01_extract_meg_measures.py
DATASET = OUT / "analysis_dataset.csv"       # written by 02_build_dataset.py


def load():
    """Participant-level analysis dataset (one row per child)."""
    return pd.read_csv(DATASET)


def ttest_levene(a, b):
    """Two-tailed independent-samples t-test; Welch's test when Levene's test is significant (P <= 0.05).
    Cohen's d uses the pooled SD; its 95% CI uses the Hedges & Olkin large-sample approximation."""
    a, b = pd.Series(a).dropna().astype(float), pd.Series(b).dropna().astype(float)
    welch = stats.levene(a, b).pvalue <= 0.05
    t, p = stats.ttest_ind(a, b, equal_var=not welch)
    n1, n2 = len(a), len(b)
    s1, s2 = a.var(ddof=1), b.var(ddof=1)
    if welch:
        dof = (s1 / n1 + s2 / n2) ** 2 / ((s1 / n1) ** 2 / (n1 - 1) + (s2 / n2) ** 2 / (n2 - 1))
    else:
        dof = n1 + n2 - 2
    sp = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    d = (a.mean() - b.mean()) / sp
    se_d = np.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2)))
    return dict(t=t, df=dof, p=p, d=d, d_lo=d - 1.96 * se_d, d_hi=d + 1.96 * se_d, welch=welch,
                n1=n1, n2=n2, diff=a.mean() - b.mean())


def holm(pvals):
    """Holm step-down adjusted P-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def fdr_bh(pvals):
    """Benjamini-Hochberg false discovery rate adjusted P-values (q-values)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


def spearman(x, y):
    """Spearman correlation on complete pairs; returns (rho, P, n)."""
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    r, p = stats.spearmanr(d["x"], d["y"])
    return r, p, len(d)


def partial_spearman(df, x, y, covars):
    """Partial Spearman correlation: Pearson correlation between the residuals of the rank-transformed
    variables after regression on the rank-transformed covariates; returns (rho, P, n)."""
    d = df[[x, y] + covars].dropna()
    rk = d.rank()
    Z = np.column_stack([np.ones(len(rk))] + [rk[c].values for c in covars])

    def resid(v):
        beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ beta

    rx, ry = resid(rk[x].values), resid(rk[y].values)
    r = np.corrcoef(rx, ry)[0, 1]
    dof = len(d) - 2 - len(covars)
    t = r * np.sqrt(dof / (1 - r ** 2))
    return r, 2 * stats.t.sf(abs(t), dof), len(d)


def mixed_2x2(df, contra, ipsi):
    """2 (group, between) x 2 (hemisphere, within) mixed ANOVA for two groups.

    With two levels of each factor, the Group x Hemisphere interaction is the group effect on the hemispheric
    difference D = contra - ipsi, the hemisphere main effect is the intercept of D (effect-coded group), and the
    group main effect is the group effect on the hemisphere average M. Each test is F(1, n - 2) = t^2.
    Returns (table of effects, data used)."""
    import statsmodels.formula.api as smf
    d = df.dropna(subset=[contra, ipsi]).copy()
    d["g"] = np.where(d["Group"] == "PT", 0.5, -0.5)
    d["D"] = d[contra] - d[ipsi]
    d["M"] = (d[contra] + d[ipsi]) / 2
    mD, mM = smf.ols("D ~ g", d).fit(), smf.ols("M ~ g", d).fit()
    e = int(mD.df_resid)
    rows = []
    for eff, t, p in [("Group", mM.tvalues["g"], mM.pvalues["g"]),
                      ("Hemisphere", mD.tvalues["Intercept"], mD.pvalues["Intercept"]),
                      ("Group x Hemisphere", mD.tvalues["g"], mD.pvalues["g"])]:
        F = t ** 2
        rows.append(dict(Effect=eff, F=F, df1=1, df2=e, P=p, partial_eta2=F / (F + e)))
    return pd.DataFrame(rows), d


def fmt_p(p):
    return f"{p:.1e}" if p < 0.001 else f"{p:.3f}"


def describe(s):
    """'mean (SD)' string and n."""
    s = pd.Series(s).dropna().astype(float)
    return f"{s.mean():.2f} ({s.std(ddof=1):.2f})", len(s)
