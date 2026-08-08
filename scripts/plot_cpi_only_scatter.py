"""
Scatter plot of the CPI-only single-task model's predicted pKd vs. the
Figure S24 neutralization ratio (results/hoshino_predictions_cpi_ppi_only.csv).
This model -- trained on ONLY CPI-origin rows from split_pure_random_
uniqueonly, raw/no-LoRA T5ProtChem encoder -- showed the strongest single-
split Hoshino correlation found in this project (Pearson r=0.683, p=0.0050).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr

CSV = "/mnt/hdd/tomita/PPI_CPI_prediction/T5ProtChem_XGBoost_PPI_and_CPI_affinity_prediction/results/hoshino_predictions_cpi_ppi_only.csv"
OUT_PNG = "/mnt/hdd/tomita/PPI_CPI_prediction/T5ProtChem_XGBoost_PPI_and_CPI_affinity_prediction/results/cpi_only_hoshino_scatter.png"

df = pd.read_csv(CSV)
sub = df.dropna(subset=["neutralization"])
r, p = pearsonr(sub["predicted_pKd_cpi_only"], sub["neutralization"])

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(sub["predicted_pKd_cpi_only"], sub["neutralization"], s=40, color="tab:green")
for _, row in sub.iterrows():
    ax.annotate(row["label"], (row["predicted_pKd_cpi_only"], row["neutralization"]),
               fontsize=7, xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("Predicted pKd (CPI-only model)")
ax.set_ylabel("Neutralization Ratio (%) [Figure S24]")
ax.set_title(f"CPI-only model: predicted pKd vs. neutralization ratio (n={len(sub)})\nPearson r={r:.3f}, p={p:.4f}")
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150)
print(f"Saved {OUT_PNG}, r={r:.4f}, p={p:.4f}")
