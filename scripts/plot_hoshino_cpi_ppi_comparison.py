"""
Bar plot comparing the Hoshino_polymer external-validation Pearson
correlation (vs. Figure S24 bottom panel, 0.1 mM neutralization ratio,
n=15) across three models trained on the same split_pure_random_uniqueonly
data: CPI-only, PPI-only, and this repo's featured CPI+PPI combined model.
CPI-only/PPI-only predictions are from results/hoshino_predictions_cpi_
ppi_only.csv (same repo split, same raw T5ProtChem encoder, PPI-only run
with matching augmentation -- see conversation); the combined model's
predictions are from results/predicted_pKa_KanM_T5ProtChem_raw_uniqueonly_
quadsplice.csv. All three use permutation-test Pearson p-values (99999
resamples) for consistency with the rest of this repo's methodology.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

NEUTRALIZATION = {
    "A2T0": 9.4, "A2T1": 3.6, "A2T2": 16.0, "A2T3": 26.4, "A2T4": 25.2, "A2T5": 31.8,
    "A3T0": -4.7, "A3T1": 32.4, "A3T2": 19.4, "A3T3": 6.8, "A3T4": 21.1,
    "A4T0": 20.0, "A4T1": 21.3, "A4T2": 30.9, "A4T3": 62.3,
}
REPO = "/mnt/hdd/tomita/PPI_CPI_prediction/T5ProtChem_XGBoost_PPI_and_CPI_affinity_prediction"
CPI_PPI_ONLY_CSV = f"{REPO}/results/hoshino_predictions_cpi_ppi_only.csv"
COMBINED_CSV = f"{REPO}/results/predicted_pKa_KanM_T5ProtChem_raw_uniqueonly_quadsplice.csv"
COMBINED_COL = "predicted_pKd_T5ProtChem_raw_uniqueonly_quadsplice"
OUT_PNG = f"{REPO}/results/hoshino_cpi_ppi_comparison_bar.png"
N_RESAMPLES = 99999
PERM_SEED = 42


def pearsonr_perm(x, y):
    method = stats.PermutationMethod(n_resamples=N_RESAMPLES, random_state=np.random.default_rng(PERM_SEED))
    res = stats.pearsonr(x, y, method=method)
    return res.statistic, res.pvalue


cpi_ppi_df = pd.read_csv(CPI_PPI_ONLY_CSV)
cpi_ppi_df["neutralization"] = cpi_ppi_df["label"].map(NEUTRALIZATION)
sub_cp = cpi_ppi_df.dropna(subset=["neutralization"])

combined_df = pd.read_csv(COMBINED_CSV)
combined_df["label"] = "A" + combined_df["n"].astype(str) + "T" + combined_df["m"].astype(str)
combined_df["neutralization"] = combined_df["label"].map(NEUTRALIZATION)
sub_combined = combined_df.dropna(subset=["neutralization"])

r_cpi, p_cpi = pearsonr_perm(sub_cp["predicted_pKd_cpi_only"], sub_cp["neutralization"])
r_ppi, p_ppi = pearsonr_perm(sub_cp["predicted_pKd_ppi_only"], sub_cp["neutralization"])
r_combined, p_combined = pearsonr_perm(sub_combined[COMBINED_COL], sub_combined["neutralization"])

print(f"CPI-only:  r={r_cpi:.4f} (permutation p={p_cpi:.5f})")
print(f"PPI-only:  r={r_ppi:.4f} (permutation p={p_ppi:.5f})")
print(f"Combined (featured): r={r_combined:.4f} (permutation p={p_combined:.5f})")

labels = ["CPI-only", "PPI-only", "Combined\n(featured model)"]
rs = [r_cpi, r_ppi, r_combined]
ps = [p_cpi, p_ppi, p_combined]
colors = ["tab:orange", "tab:green", "tab:red"]

fig, ax = plt.subplots(figsize=(6.5, 5.5))
bars = ax.bar(labels, rs, color=colors, edgecolor="black", linewidth=0.7)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Pearson r vs. neutralization ratio\n(Figure S24 bottom panel, 0.1 mM, n=15)")
ax.set_title("Hoshino_polymer correlation: CPI-only vs. PPI-only vs. combined")

for bar, r, p in zip(bars, rs, ps):
    sig = "*" if p < 0.05 else ""
    y = r + (0.03 if r >= 0 else -0.03)
    va = "bottom" if r >= 0 else "top"
    ax.annotate(f"r={r:.3f}{sig}\n(p={p:.3f})", (bar.get_x() + bar.get_width() / 2, y),
               ha="center", va=va, fontsize=9)

ax.set_ylim(-0.5, 0.9)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150)
print(f"Saved {OUT_PNG}")
