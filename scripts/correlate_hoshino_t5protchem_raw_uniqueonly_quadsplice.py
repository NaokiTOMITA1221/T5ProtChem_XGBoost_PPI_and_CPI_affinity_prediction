import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

NEUTRALIZATION = {
    "A2T0": 19.9, "A2T1": 38.8, "A2T2": 63.3, "A2T3": 56.6, "A2T4": 86.9, "A2T5": 51.7,
    "A3T0": 24.8, "A3T1": 90.6, "A3T2": 51.5, "A3T3": 67.8, "A3T4": 58.7,
    "A4T0": 84.3, "A4T1": 83.3, "A4T2": 77.5, "A4T3": 85.0,
}

CSV = "/home/tomita/Hoshino_polymer/predicted_pKa_KanM_T5ProtChem_raw_uniqueonly_quadsplice.csv"
PRED_COL = "predicted_pKd_T5ProtChem_raw_uniqueonly_quadsplice"
N_BOOT = 99999
BOOT_SEED = 42

df = pd.read_csv(CSV)
df["label"] = "A" + df["n"].astype(str) + "T" + df["m"].astype(str)
df["neutralization"] = df["label"].map(NEUTRALIZATION)
sub = df.dropna(subset=["neutralization"])
print(f"Matched {len(sub)} / {len(df)} rows")


def pearsonr_bootstrap(x, y, n_boot=N_BOOT, seed=BOOT_SEED):
    """Pearson r with a BOOTSTRAP p-value: resample (x_i, y_i) pairs WITH
    replacement n_boot times, compute r for each resample, then take
    p = 2 * min(P(r_boot <= 0), P(r_boot >= 0)) from the empirical bootstrap
    distribution -- no bivariate-normality assumption (unlike scipy's
    default parametric p-value), and no assumption that r=0 under any null
    resampling scheme (unlike a permutation test)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    r_obs = float(np.corrcoef(x, y)[0, 1])

    rng = np.random.default_rng(seed)
    rs = np.empty(n_boot)
    idx = np.arange(n)
    for b in range(n_boot):
        samp = rng.choice(idx, size=n, replace=True)
        xs, ys = x[samp], y[samp]
        if np.std(xs) == 0 or np.std(ys) == 0:
            rs[b] = np.nan
        else:
            rs[b] = np.corrcoef(xs, ys)[0, 1]

    valid = rs[~np.isnan(rs)]
    frac_le0 = float(np.mean(valid <= 0))
    frac_ge0 = float(np.mean(valid >= 0))
    p_boot = min(2 * min(frac_le0, frac_ge0), 1.0)
    ci_lo, ci_hi = np.percentile(valid, [2.5, 97.5])
    return r_obs, p_boot, (float(ci_lo), float(ci_hi))


r_pred, p_pred, ci_pred = pearsonr_bootstrap(sub[PRED_COL], sub["neutralization"])
rho_pred, p_rho_pred = spearmanr(sub[PRED_COL], sub["neutralization"])
r_ref, p_ref, ci_ref = pearsonr_bootstrap(sub["pKd"], sub["neutralization"])

print(f"predicted_pKd vs neutralization: pearson r={r_pred:.4f} (bootstrap p={p_pred:.5f}, "
     f"95% CI=[{ci_pred[0]:.3f}, {ci_pred[1]:.3f}]), "
     f"spearman rho={rho_pred:.4f} (p={p_rho_pred:.4f}, asymptotic -- no bootstrap variant computed)")
print(f"reference pKd vs neutralization: pearson r={r_ref:.4f} (bootstrap p={p_ref:.5f}, "
     f"95% CI=[{ci_ref[0]:.3f}, {ci_ref[1]:.3f}])")
print(sub[["label", "pKd", PRED_COL, "neutralization"]].to_string(index=False))
