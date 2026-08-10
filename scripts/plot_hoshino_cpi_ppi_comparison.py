"""
Bar plot comparing the Hoshino_polymer external-validation correlation
(vs. Figure S24 bottom panel, 0.1 mM neutralization ratio, n=15) across
three models trained on the same split_pure_random_uniqueonly data (same
encoder/architecture): CPI-only, PPI-only, and this repo's featured
CPI+PPI combined (balanced) model. Both Pearson r and Spearman rho are
shown, grouped side by side per metric, with all three models' bars
adjacent within each group. Combined is solid black; CPI-only/PPI-only use
hatch patterns (dots / diagonal lines) instead of color so the figure reads
in grayscale too. All p-values are from permutation tests (99999
resamples) for consistency with the rest of this repo's methodology.
"""
import os

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
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMBINED_CSV = f"{REPO}/results/predicted_pKa_KanM_balanced.csv"
CPI_ONLY_CSV = f"{REPO}/results/predicted_pKa_KanM_cpi_only.csv"
PPI_ONLY_CSV = f"{REPO}/results/predicted_pKa_KanM_ppi_only.csv"
OUT_PNG = f"{REPO}/results/hoshino_cpi_ppi_comparison_bar.png"
N_RESAMPLES = 99999
PERM_SEED = 42


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


def load_pred(csv_path, pred_col="predicted_pKd"):
    df = pd.read_csv(csv_path)
    df["label"] = "A" + df["n"].astype(str) + "T" + df["m"].astype(str)
    df["neutralization"] = df["label"].map(NEUTRALIZATION)
    sub = df.dropna(subset=["neutralization"])
    r, p_r = pearsonr_perm(sub[pred_col], sub["neutralization"])
    rho, p_rho = spearmanr_perm(sub[pred_col], sub["neutralization"])
    return r, p_r, rho, p_rho


models = ["CPI-only", "PPI-only", "Combined\n(featured model)"]
csvs = [CPI_ONLY_CSV, PPI_ONLY_CSV, COMBINED_CSV]
hatches = [".", "/", None]  # dots / diagonal lines / solid
facecolors = ["white", "white", "black"]

results = [load_pred(csv) for csv in csvs]
for name, (r, p_r, rho, p_rho) in zip(models, results):
    print(f"{name.replace(chr(10), ' ')}: pearson r={r:.4f} (p={p_r:.5f}), spearman rho={rho:.4f} (p={p_rho:.5f})")

metrics = ["Pearson r", "Spearman rho"]
group_centers = np.arange(len(metrics))
n_models = len(models)
bar_width = 0.8 / n_models

fig, ax = plt.subplots(figsize=(8, 6))
for i, (name, hatch, facecolor) in enumerate(zip(models, hatches, facecolors)):
    r, p_r, rho, p_rho = results[i]
    values = [r, rho]
    pvals = [p_r, p_rho]
    offsets = group_centers + (i - (n_models - 1) / 2) * bar_width
    bars = ax.bar(offsets, values, width=bar_width, facecolor=facecolor, edgecolor="black",
                 linewidth=0.9, hatch=hatch, label=name.replace("\n", " "))
    for x, v, p in zip(offsets, values, pvals):
        sig = "*" if p < 0.05 else ""
        y = v + (0.03 if v >= 0 else -0.03)
        va = "bottom" if v >= 0 else "top"
        ax.annotate(f"{v:.3f}{sig}", (x, y), ha="center", va=va, fontsize=8)

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(group_centers)
ax.set_xticklabels(metrics)
ax.set_ylabel("Correlation vs. neutralization ratio\n(Figure S24 bottom panel, 0.1 mM, n=15)")
ax.set_title("Hoshino_polymer correlation: CPI-only vs. PPI-only vs. combined")
ax.set_ylim(-0.5, 0.9)
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150)
print(f"Saved {OUT_PNG}")
