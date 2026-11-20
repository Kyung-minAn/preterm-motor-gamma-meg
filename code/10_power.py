"""Step 10: statistical power of the between-group comparison (response to the reviewers).

Needs no participant data. Two-tailed independent-samples t-test, alpha = 0.05, noncentral t distribution
(as in G*Power).
 - sensitivity: smallest effect (Cohen's d) detectable with 80% power for 19 full-term and 18 preterm children
 - power for the contralateral gamma power difference reported with the same task by An et al. (2018;
   J Neurosci 38:7878-7886, Table 3: typically developing 37.44 +/- 27.56%, n = 15; autism 19.48 +/- 14.73%, n = 14)
   and the sample size per group needed for 80% and 90% power"""
import numpy as np
from statsmodels.stats.power import TTestIndPower

N_FT, N_PT, ALPHA = 19, 18, 0.05
power = TTestIndPower()


def cohens_d(m1, s1, n1, m2, s2, n2):
    return (m1 - m2) / np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))


mde = power.solve_power(effect_size=None, nobs1=N_FT, alpha=ALPHA, power=0.80, ratio=N_PT / N_FT)
print(f"Sensitivity: with n = {N_FT}/{N_PT}, 80% power for d >= {mde:.2f}")

d_prior = cohens_d(37.44, 27.56, 15, 19.48, 14.73, 14)
pw = power.power(effect_size=d_prior, nobs1=N_FT, alpha=ALPHA, ratio=N_PT / N_FT)
n80 = power.solve_power(effect_size=d_prior, power=0.80, alpha=ALPHA, ratio=1)
n90 = power.solve_power(effect_size=d_prior, power=0.90, alpha=ALPHA, ratio=1)
print(f"An et al. (2018), contralateral gamma power: d = {d_prior:.2f}; power with n = {N_FT}/{N_PT}: {pw:.2f}; "
      f"n per group for 80% power: {int(np.ceil(n80))}, for 90% power: {int(np.ceil(n90))}")
