"""
Scatter plot of this repo's model's predicted pKd vs. the Figure S24 bottom
panel (ligand concentration 0.1 mM) neutralization ratio, for the 15
Hoshino_polymer pairs with a matching Figure S24 value. Annotated with the
permutation-test Pearson r and Spearman rho (see correlate_hoshino_
t5protchem_raw_uniqueonly_quadsplice.py) -- blue markers for T0 (m=0)
samples, red for the rest, matching Figure S24's own color coding.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr

NEUTRALIZATION = {
    "A2T0": 9.4, "A2T1": 3.6, "A2T2": 16.0, "A2T3": 26.4, "A2T4": 25.2, "A2T5": 31.8,
    "A3T0": -4.7, "A3T1": 32.4, "A3T2": 19.4, "A3T3": 6.8, "A3T4": 21.1,
    "A4T0": 20.0, "A4T1": 21.3, "A4T2": 30.9, "A4T3": 62.3,
}

CSV = "/mnt/hdd/tomita/PPI_CPI_prediction/T5ProtChem_XGBoost_PPI_and_CPI_affinity_prediction/results/predicted_pKa_KanM_T5ProtChem_raw_uniqueonly_quadsplice.csv"
PRED_COL = "predicted_pKd_T5ProtChem_raw_uniqueonly_quadsplice"
OUT_PNG = "/mnt/hdd/tomita/PPI_CPI_prediction/T5ProtChem_XGBoost_PPI_and_CPI_affinity_prediction/results/hoshino_correlation_scatter.png"
N_RESAMPLES = 99999
PERM_SEED = 42

df = pd.read_csv(CSV)
df["label"] = "A" + df["n"].astype(str) + "T" + df["m"].astype(str)
df["neutralization"] = df["label"].map(NEUTRALIZATION)
sub = df.dropna(subset=["neutralization"]).reset_index(drop=True)


def pearsonr_perm(x, y):
    method = stats.PermutationMethod(n_resamples=N_RESAMPLES, random_state=np.random.default_rng(PERM_SEED))
    res = stats.pearsonr(x, y, method=method)
    return res.statistic, res.pvalue


def spearmanr_perm(x, y):
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


r, p = pearsonr_perm(sub[PRED_COL], sub["neutralization"])
rho, p_rho = spearmanr_perm(sub[PRED_COL], sub["neutralization"])

colors = ["tab:blue" if label.endswith("T0") else "tab:red" for label in sub["label"]]

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.scatter(sub[PRED_COL], sub["neutralization"], c=colors, s=60, edgecolor="black", linewidth=0.5)

slope, intercept = np.polyfit(sub[PRED_COL], sub["neutralization"], 1)
x_line = np.array([sub[PRED_COL].min(), sub[PRED_COL].max()])
ax.plot(x_line, slope * x_line + intercept, linestyle=":", color="black", linewidth=1.5,
       label=f"least-squares fit (y={slope:.2f}x+{intercept:.2f})")
ax.legend(fontsize=8, loc="upper left")

for _, row in sub.iterrows():
    ax.annotate(row["label"], (row[PRED_COL], row["neutralization"]),
               textcoords="offset points", xytext=(5, 3), fontsize=8)
ax.set_xlabel("Predicted pKd (this model)")
ax.set_ylabel("Neutralization Ratio (%) -- Figure S24 bottom panel (0.1 mM)")
r2 = r ** 2
ax.set_title(f"n=15: Pearson r={r:.3f} (p={p:.4f}), R²={r2:.3f}, Spearman rho={rho:.3f} (p={p_rho:.4f})",
            fontsize=10.5)
fig.subplots_adjust(top=0.90)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150)
print(f"Saved {OUT_PNG}")
print(f"pearson r={r:.4f} (permutation p={p:.5f}), r^2={r2:.4f}, "
     f"spearman rho={rho:.4f} (permutation p={p_rho:.5f})")
