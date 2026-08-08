"""
Three figures, sharing the same x-axis (A{n}T{m} sample labels, in the same
order) so they can be visually compared side by side:
  1. A reproduction of Figure S24's top histogram (hemolysis neutralization
     ratio per A{n}T{m} sample, pixel-extracted from the source PDF earlier
     in this project -- see conversation) -- blue bars for the T0 (m=0)
     samples, red for the rest, matching the source figure's color coding.
  2. A bar plot of this repo's model's predicted pKd values
     (results/predicted_pKa_KanM_T5ProtChem_raw_uniqueonly_quadsplice.csv)
     for the SAME samples/order (the 15 with a matching neutralization
     value; 3 of the 18 Hoshino_polymer pairs have no Figure S24 value and
     are excluded here for a like-for-like x-axis).
  3. A bar plot of the reference pKd column already present in the
     Hoshino_polymer input CSV (a pKd value from another, external source
     -- NOT this repo's model), same samples/order, for comparison.
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
OUT_PRED_BAR = "/mnt/hdd/tomita/PPI_CPI_prediction/T5ProtChem_XGBoost_PPI_and_CPI_affinity_prediction/results/predicted_pkd_barplot.png"
OUT_REF_BAR = "/mnt/hdd/tomita/PPI_CPI_prediction/T5ProtChem_XGBoost_PPI_and_CPI_affinity_prediction/results/reference_pkd_barplot.png"

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

# --- Figure 2: predicted pKd bar plot, same x-axis/order as Figure 1 ---
df = pd.read_csv(PRED_CSV)
df["label"] = "A" + df["n"].astype(str) + "T" + df["m"].astype(str)
pred_by_label = df.set_index("label")[PRED_COL]
pred_values = [pred_by_label[label] for label in LABEL_ORDER]

fig2, ax2 = plt.subplots(figsize=(9, 4.5))
ax2.bar(LABEL_ORDER, pred_values, color=colors)
ax2.set_ylabel("Predicted pKd (this model)")
ax2.set_title("Predicted pKd, same samples/order as Figure S24 above")
ax2.tick_params(axis="x", rotation=45)
fig2.tight_layout()
fig2.savefig(OUT_PRED_BAR, dpi=150)
print(f"Saved {OUT_PRED_BAR}")

# --- Figure 3: reference pKd bar plot, same x-axis/order ---
ref_by_label = df.set_index("label")["pKd"]
ref_values = [ref_by_label[label] for label in LABEL_ORDER]

fig3, ax3 = plt.subplots(figsize=(9, 4.5))
ax3.bar(LABEL_ORDER, ref_values, color=colors)
ax3.set_ylabel("Reference pKd")
ax3.set_title("Reference pKd (external source), same samples/order as Figure S24 above")
ax3.tick_params(axis="x", rotation=45)
fig3.tight_layout()
fig3.savefig(OUT_REF_BAR, dpi=150)
print(f"Saved {OUT_REF_BAR}")
