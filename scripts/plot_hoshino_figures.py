"""
Two figures:
  1. A reproduction of Figure S24's top histogram (hemolysis neutralization
     ratio per A{n}T{m} sample, pixel-extracted from the source PDF earlier
     in this project -- see conversation) -- blue bars for the T0 (m=0)
     samples, red for the rest, matching the source figure's color coding.
  2. A histogram of this repo's model's predicted pKd values themselves
     (results/predicted_pKa_KanM_T5ProtChem_raw_uniqueonly_quadsplice.csv)
     across all 18 Hoshino_polymer pairs.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

NEUTRALIZATION = {
    "A2T0": 19.9, "A2T1": 38.8, "A2T2": 63.3, "A2T3": 56.6, "A2T4": 86.9, "A2T5": 51.7,
    "A3T0": 24.8, "A3T1": 90.6, "A3T2": 51.5, "A3T3": 67.8, "A3T4": 58.7,
    "A4T0": 84.3, "A4T1": 83.3, "A4T2": 77.5, "A4T3": 85.0,
}
LABEL_ORDER = ["A2T0", "A2T1", "A2T2", "A2T3", "A2T4", "A2T5",
              "A3T0", "A3T1", "A3T2", "A3T3", "A3T4",
              "A4T0", "A4T1", "A4T2", "A4T3"]

PRED_CSV = "/mnt/hdd/tomita/PPI_CPI_prediction/T5ProtChem_XGBoost_PPI_and_CPI_affinity_prediction/results/predicted_pKa_KanM_T5ProtChem_raw_uniqueonly_quadsplice.csv"
PRED_COL = "predicted_pKd_T5ProtChem_raw_uniqueonly_quadsplice"
OUT_BAR = "/mnt/hdd/tomita/PPI_CPI_prediction/T5ProtChem_XGBoost_PPI_and_CPI_affinity_prediction/results/figureS24_reproduction.png"
OUT_HIST = "/mnt/hdd/tomita/PPI_CPI_prediction/T5ProtChem_XGBoost_PPI_and_CPI_affinity_prediction/results/predicted_pkd_histogram.png"

# --- Figure 1: Figure S24 top histogram reproduction ---
values = [NEUTRALIZATION[label] for label in LABEL_ORDER]
colors = ["tab:blue" if label.endswith("T0") else "tab:red" for label in LABEL_ORDER]

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(LABEL_ORDER, values, color=colors)
ax.set_ylabel("Neutralization Ratio (%)")
ax.set_ylim(0, 100)
ax.set_title("Figure S24 (top histogram) reproduction")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
fig.savefig(OUT_BAR, dpi=150)
print(f"Saved {OUT_BAR}")

# --- Figure 2: histogram of predicted pKd values ---
df = pd.read_csv(PRED_CSV)
print(f"n={len(df)}, predicted pKd: mean={df[PRED_COL].mean():.3f}, "
     f"std={df[PRED_COL].std():.3f}, min={df[PRED_COL].min():.3f}, max={df[PRED_COL].max():.3f}")

fig2, ax2 = plt.subplots(figsize=(6, 4.5))
ax2.hist(df[PRED_COL], bins=10, color="tab:purple", edgecolor="black")
ax2.set_xlabel("Predicted pKd (this model)")
ax2.set_ylabel("Count")
ax2.set_title(f"Distribution of predicted pKd across Hoshino_polymer pairs (n={len(df)})")
fig2.tight_layout()
fig2.savefig(OUT_HIST, dpi=150)
print(f"Saved {OUT_HIST}")
