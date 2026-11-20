"""Step 9: data figures (statistics for Fig. 1B are computed in 03_participants_behaviour.py).
  Figure 3   (A) gamma power by group and hemisphere with post hoc comparisons; (B) gamma laterality index
  Figure 4   associations of the laterality index with K-ABC Hand Movements and age at independent walking
Outputs in OUTPUT_DIR/figures: PDF, PNG preview and TIFF (170 mm wide, 600 dpi)."""
import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from scipy import stats

from common import OUT, fdr_bh, holm, load, spearman, ttest_levene

FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
C_FT, C_PT = "#2E86AB", "#E63946"
C_FT_D, C_PT_D = "#1F6E90", "#C92A3A"
MINUS = "−"

df = load()
ft, pt = df[df["Groups"] == 0], df[df["Groups"] == 1]


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def ptxt(p, sym="P"):
    return f"${sym}$ < 0.001" if p < 0.001 else f"${sym}$ = {p:.3f}"


def num(x, nd=2):
    return f"{x:.{nd}f}".replace("-", MINUS)


def save(fig, stem, width_mm=170, dpi_out=600):
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", dpi=150, bbox_inches="tight")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=500, bbox_inches="tight")
    buf.seek(0)
    im = Image.open(buf).convert("RGB")
    w = round(width_mm / 25.4 * dpi_out)
    im.resize((w, round(im.height * w / im.width)), Image.LANCZOS).save(
        FIG / f"{stem}.tif", compression="tiff_lzw", dpi=(dpi_out, dpi_out))
    print(f"  -> {FIG / stem}.pdf/.png/.tif")


# =============================================================================
# Figure 3: statistics as in Table 3 (2 x 2 mixed ANOVA; Holm-corrected post hoc tests)
# =============================================================================
bc = ttest_levene(ft["MEG_Contrapower"], pt["MEG_Contrapower"])
bi = ttest_levene(ft["MEG_Ipsipower"], pt["MEG_Ipsipower"])
hb = holm([bc["p"], bi["p"]])
wf = stats.ttest_rel(ft["MEG_Contrapower"], ft["MEG_Ipsipower"])
wp = stats.ttest_rel(pt["MEG_Contrapower"], pt["MEG_Ipsipower"])
hw = holm([wf.pvalue, wp.pvalue])
li = ttest_levene(ft["MEG_laterality_index"], pt["MEG_laterality_index"])

fig, axes = plt.subplots(1, 2, figsize=(9.8, 5.0))
palette = {"Full-term": C_FT, "Pre-term": C_PT}
ax = axes[0]
np.random.seed(42)
for sub, off, c, a in [(ft, 0, C_FT, 0.6), (pt, 2, C_PT, 0.5)]:
    for _, row in sub.iterrows():
        xj = np.random.uniform(-0.15, 0.15) + off
        ax.plot([xj, 1 + xj], [row["MEG_Contrapower"], row["MEG_Ipsipower"]], color=c, alpha=0.4, linewidth=0.8)
        ax.scatter([xj, 1 + xj], [row["MEG_Contrapower"], row["MEG_Ipsipower"]], color=c, s=15, alpha=a, zorder=3)
for sub, off, c in [(ft, 0, C_FT_D), (pt, 2, C_PT_D)]:
    m = [sub["MEG_Contrapower"].mean(), sub["MEG_Ipsipower"].mean()]
    se = [sub["MEG_Contrapower"].sem(), sub["MEG_Ipsipower"].sem()]
    ax.errorbar([off, off + 1], m, yerr=se, fmt="none", ecolor=c, elinewidth=1.5, capsize=4, capthick=1.5, zorder=4)
    ax.plot([off, off + 1], m, color=c, linewidth=3, marker="D", markersize=8, markeredgecolor="white",
            markeredgewidth=0.8, zorder=5)
ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(["Contra", "Ipsi", "Contra", "Ipsi"], fontsize=12.5)
ax.tick_params(axis="y", labelsize=12.5)
ax.text(0.25, -0.11, "Full-term", transform=ax.transAxes, ha="center", va="top", fontsize=13.5, fontweight="bold")
ax.text(0.75, -0.11, "Preterm", transform=ax.transAxes, ha="center", va="top", fontsize=13.5, fontweight="bold")
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(-30, 190)
ax.set_yticks(np.arange(-30, 151, 30))
ax.set_ylabel("Gamma Power Change (%)", fontsize=13.5, fontweight="bold")
ax.set_title("A. Group × Hemisphere Interaction", fontsize=14.5, fontweight="bold")
ax.text(0.5, 125, f"Contra > Ipsi\n{ptxt(hw[0])}", ha="center", va="top", fontsize=11.5, color=C_FT)
ax.text(2.5, 125, f"Contra vs Ipsi\n{ptxt(hw[1])}", ha="center", va="top", fontsize=11.5, color=C_PT)
for x0, x1, y, p in [(0, 2, 142, hb[0]), (1, 3, 168, hb[1])]:
    ax.plot([x0, x0, x1, x1], [y - 5, y, y, y - 5], color="black", linewidth=1)
    s = stars(p)
    if s == "n.s.":
        ax.text((x0 + x1) / 2, y + 5, s, ha="center", va="center", fontsize=12.5, style="italic")
    else:
        ax.text((x0 + x1) / 2, y + 1.8, s, ha="center", va="center", fontsize=16.5, fontweight="bold")

ax = axes[1]
df_lat = df[["Groups", "MEG_laterality_index"]].dropna().copy()
df_lat["Group_Label"] = df_lat["Groups"].map({0: "Full-term", 1: "Pre-term"})
np.random.seed(42)
sns.boxplot(data=df_lat, x="Group_Label", y="MEG_laterality_index", order=["Full-term", "Pre-term"],
            hue="Group_Label", palette=palette, legend=False, width=0.4, ax=ax, showfliers=False,
            medianprops=dict(color="#E5804B", linewidth=1.5), whiskerprops=dict(color="gray", linewidth=1),
            capprops=dict(color="black", linewidth=1))
for patch, color in zip(ax.patches, [C_FT, C_PT]):
    patch.set_facecolor(color)
    patch.set_alpha(0.25)
    patch.set_edgecolor("none")
sns.stripplot(data=df_lat, x="Group_Label", y="MEG_laterality_index", order=["Full-term", "Pre-term"],
              hue="Group_Label", palette=palette, legend=False, alpha=0.7, jitter=0.1, size=4.5, edgecolor="none", ax=ax)
ax.scatter([0, 1], [ft["MEG_laterality_index"].mean(), pt["MEG_laterality_index"].mean()],
           color=[C_FT_D, C_PT_D], marker="D", s=45, edgecolor="white", linewidth=0.8, zorder=4)
ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.30, 0.40)
ax.set_yticks(np.arange(-0.3, 0.31, 0.1))
ax.set_xlabel("")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Full-term", "Preterm"], fontsize=12.5, fontweight="bold")
ax.tick_params(axis="y", labelsize=12.5)
ax.set_ylabel("Gamma Laterality Index", fontsize=13.5, fontweight="bold")
ax.set_title("B. Laterality Index", fontsize=14.5, fontweight="bold")
ax.plot([0, 0, 1, 1], [0.293, 0.31, 0.31, 0.293], color="black", linewidth=1)
ax.text(0.5, 0.323, stars(li["p"]), ha="center", va="center", fontsize=16.5, fontweight="bold")
plt.tight_layout()
plt.subplots_adjust(top=0.87, bottom=0.18, wspace=0.36)
save(fig, "Figure3")
plt.close(fig)

# =============================================================================
# Figure 4: laterality index x motor measures (Spearman; FDR over 6 measures within each sample)
# =============================================================================
dev = ["Head_controll_ca", "Sitdown_ca", "Walk_alone_ca", "KABC_HandMovement", "VABS_Motor_Gross", "VABS_Motor_Fine"]


def fam_q(d):
    res = [spearman(d["MEG_laterality_index"], d[c]) for c in dev]
    q = fdr_bh([r[1] for r in res])
    return {c: dict(rho=r[0], p=r[1], q=qq, n=r[2]) for c, r, qq in zip(dev, res, q)}


Qw, Qp = fam_q(df), fam_q(pt)


def regression_ci(ax, x, y, color="gray", linestyle="-", alpha_line=0.5, alpha_shading=0.15):
    coeffs = np.polyfit(x, y, 1)
    xl = np.linspace(x.min() - (x.max() - x.min()) * 0.1, x.max() + (x.max() - x.min()) * 0.1, 100)
    yl = np.polyval(coeffs, xl)
    ax.plot(xl, yl, color=color, linewidth=2, alpha=alpha_line, linestyle=linestyle)
    n = len(x)
    s_err = np.sqrt(np.sum((y - np.polyval(coeffs, x)) ** 2) / (n - 2))
    ci = stats.t.ppf(0.975, n - 2) * s_err * np.sqrt(1 / n + (xl - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    ax.fill_between(xl, yl - ci, yl + ci, color=color, alpha=alpha_shading, zorder=1)


FS_STAT, FS_LAB, FS_TICK, FS_TITLE = 12, 13, 13.5, 13.5
box = dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75)
fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.8))
panels = [
    (axes[0], df, "KABC_HandMovement", "K-ABC Hand Movements", Qw, "Whole sample",
     "A. Laterality Index and\nK-ABC Hand Movements", "lower right", (0.05, "left")),
    (axes[1], df, "Walk_alone_ca", "Age at Independent Walking (months)", Qw, "Whole sample",
     "B. Laterality Index and Age at\nIndependent Walking (Whole Sample)", "lower left", (0.95, "right")),
    (axes[2], pt, "Walk_alone_ca", "Age at Independent Walking (months)", Qp, "Preterm only",
     "C. Laterality Index and Age at\nIndependent Walking (Preterm Group)", "lower left", (0.95, "right")),
]
for ax, d0, xcol, xlab, Q, lab, title, legloc, (tx, tha) in panels:
    d = d0[[xcol, "MEG_laterality_index", "Groups"]].dropna()
    x, y, g = d[xcol], d["MEG_laterality_index"], d["Groups"]
    if lab == "Preterm only":
        ax.scatter(x, y, color=C_PT, s=65, edgecolors="white", linewidth=0.8, zorder=3, label="Preterm")
        regression_ci(ax, x, y, color=C_PT, alpha_line=0.7, alpha_shading=0.12)
    else:
        ax.scatter(x[g == 0], y[g == 0], color=C_FT, s=60, edgecolors="white", linewidth=0.8, zorder=3, label="Full-term")
        ax.scatter(x[g == 1], y[g == 1], color=C_PT, s=60, edgecolors="white", linewidth=0.8, zorder=3, label="Preterm")
        regression_ci(ax, x, y, color="gray", linestyle="--", alpha_line=0.5, alpha_shading=0.1)
    r = Q[xcol]
    ax.text(tx, 0.98, f"{lab} ($n$ = {len(d)})\nρ = {num(r['rho'])}, {ptxt(r['q'], 'q')}", transform=ax.transAxes,
            va="top", ha=tha, fontsize=FS_STAT, bbox=box)
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.set_xlabel(xlab, fontsize=FS_LAB, fontweight="bold", labelpad=8)
    ax.set_ylabel("Gamma Laterality Index", fontsize=FS_LAB, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=FS_TICK)
    ax.set_xlim(x.min() - 1.5, x.max() + 1.5)
    ax.set_ylim(-0.32, 0.42)
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", pad=15)
    ax.legend(fontsize=FS_STAT, loc=legloc, framealpha=0.8, handletextpad=0.2)
plt.tight_layout()
plt.subplots_adjust(top=0.82, wspace=0.45)
save(fig, "Figure4")
plt.close(fig)
