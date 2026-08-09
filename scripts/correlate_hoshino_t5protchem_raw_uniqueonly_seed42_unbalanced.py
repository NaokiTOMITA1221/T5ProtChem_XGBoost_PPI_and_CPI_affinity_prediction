import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr

NEUTRALIZATION = {
    "A2T0": 9.4, "A2T1": 3.6, "A2T2": 16.0, "A2T3": 26.4, "A2T4": 25.2, "A2T5": 31.8,
    "A3T0": -4.7, "A3T1": 32.4, "A3T2": 19.4, "A3T3": 6.8, "A3T4": 21.1,
    "A4T0": 20.0, "A4T1": 21.3, "A4T2": 30.9, "A4T3": 62.3,
}
# Values are from Figure S24's BOTTOM panel (ligand concentration 0.1 mM),
# pixel-extracted from the source PDF (bar-color detection + y-axis
# box-border calibration).

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO, "results/predicted_pKa_KanM_T5ProtChem_raw_uniqueonly_seed42_unbalanced.csv")
PRED_COL = "predicted_pKd_T5ProtChem_raw_uniqueonly_seed42_unbalanced"
N_RESAMPLES = 99999
PERM_SEED = 42

df = pd.read_csv(CSV)
df["label"] = "A" + df["n"].astype(str) + "T" + df["m"].astype(str)
df["neutralization"] = df["label"].map(NEUTRALIZATION)
sub = df.dropna(subset=["neutralization"])
print(f"Matched {len(sub)} / {len(df)} rows")


def pearsonr_perm(x, y):
    """Pearson r with a permutation-test p-value: shuffles one variable
    relative to the other to build the null distribution of r under
    independence, with no distributional (bivariate-normality) assumption --
    the preferred method for small-n correlation significance testing over
    both the parametric t-distribution approximation and a naive bootstrap
    percentile p-value (see conversation: bootstrap resampling estimates the
    sampling distribution of r under the OBSERVED data, not a properly
    enforced null, and is unstable at n=15)."""
    method = stats.PermutationMethod(n_resamples=N_RESAMPLES, random_state=np.random.default_rng(PERM_SEED))
    res = stats.pearsonr(x, y, method=method)
    return res.statistic, res.pvalue


def spearmanr_perm(x, y):
    """Spearman rho with a permutation-test p-value, computed manually since
    scipy.stats.spearmanr has no `method=PermutationMethod` option (unlike
    pearsonr) -- same null (independence, via shuffling) and same rigor as
    pearsonr_perm above, for consistency."""
    rng = np.random.default_rng(PERM_SEED)
    rho_obs = spearmanr(x, y).statistic
    y_arr = np.asarray(y)
    count = 0
    for _ in range(N_RESAMPLES):
        y_perm = rng.permutation(y_arr)
        rho_perm = spearmanr(x, y_perm).statistic
        if abs(rho_perm) >= abs(rho_obs):
            count += 1
    p = (count + 1) / (N_RESAMPLES + 1)
    return rho_obs, p


r_pred, p_pred = pearsonr_perm(sub[PRED_COL], sub["neutralization"])
rho_pred, p_rho_pred = spearmanr_perm(sub[PRED_COL], sub["neutralization"])

print(f"predicted_pKd vs neutralization: pearson r={r_pred:.4f} (permutation p={p_pred:.5f}), "
     f"spearman rho={rho_pred:.4f} (permutation p={p_rho_pred:.5f})")
print(sub[["label", "pKd", PRED_COL, "neutralization"]].to_string(index=False))
