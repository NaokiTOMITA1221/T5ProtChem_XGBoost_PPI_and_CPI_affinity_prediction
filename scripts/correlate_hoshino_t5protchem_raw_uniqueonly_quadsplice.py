import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr

NEUTRALIZATION = {
    "A2T0": 19.9, "A2T1": 38.8, "A2T2": 63.3, "A2T3": 56.6, "A2T4": 86.9, "A2T5": 51.7,
    "A3T0": 24.8, "A3T1": 90.6, "A3T2": 51.5, "A3T3": 67.8, "A3T4": 58.7,
    "A4T0": 84.3, "A4T1": 83.3, "A4T2": 77.5, "A4T3": 85.0,
}

CSV = "/home/tomita/Hoshino_polymer/predicted_pKa_KanM_T5ProtChem_raw_uniqueonly_quadsplice.csv"
PRED_COL = "predicted_pKd_T5ProtChem_raw_uniqueonly_quadsplice"
N_RESAMPLES = 99999
PERM_SEED = 42

df = pd.read_csv(CSV)
df["label"] = "A" + df["n"].astype(str) + "T" + df["m"].astype(str)
df["neutralization"] = df["label"].map(NEUTRALIZATION)
sub = df.dropna(subset=["neutralization"])
print(f"Matched {len(sub)} / {len(df)} rows")


def pearsonr_perm(x, y):
    """Pearson r with a permutation-test p-value (scipy's pearsonr default
    p-value assumes bivariate normality, which is a poor fit for n=15 --
    permutation testing makes no distributional assumption)."""
    method = stats.PermutationMethod(n_resamples=N_RESAMPLES, random_state=np.random.default_rng(PERM_SEED))
    res = stats.pearsonr(x, y, method=method)
    return res.statistic, res.pvalue


r_pred, p_pred = pearsonr_perm(sub[PRED_COL], sub["neutralization"])
rho_pred, p_rho_pred = spearmanr(sub[PRED_COL], sub["neutralization"])
r_ref, p_ref = pearsonr_perm(sub["pKd"], sub["neutralization"])

print(f"predicted_pKd vs neutralization: pearson r={r_pred:.4f} (permutation p={p_pred:.5f}), "
     f"spearman rho={rho_pred:.4f} (p={p_rho_pred:.4f}, asymptotic -- scipy has no permutation option for spearmanr)")
print(f"reference pKd vs neutralization: pearson r={r_ref:.4f} (permutation p={p_ref:.5f})")
print(sub[["label", "pKd", PRED_COL, "neutralization"]].to_string(index=False))
