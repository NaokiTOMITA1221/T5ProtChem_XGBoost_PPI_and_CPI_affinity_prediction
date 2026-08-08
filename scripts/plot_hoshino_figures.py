"""
Two side-by-side bar plot panels (same figure), sharing the same x-axis
(A{n}T{m} sample labels, in the same order) so they can be visually compared:
  1. The Figure S24 hemolysis neutralization ratio (pixel-extracted from the
     source PDF earlier in this project -- see conversation).
  2. This repo's model's predicted pKd
     (results/predicted_pKa_KanM_T5ProtChem_raw_uniqueonly_quadsplice.csv).

Each panel uses its own natural y-scale (percent vs. pKd units) -- blue bars
for the T0 (m=0) samples, red for the rest, matching Figure S24's own color
coding. Only the 15 of 18 Hoshino_polymer pairs with a matching Figure S24
value are included, for a like-for-like x-axis.
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
OUT_PNG = "/mnt/hdd/tomita/PPI_CPI_prediction/T5ProtChem_XGBoost_PPI_and_CPI_affinity_prediction/results/neutralization_vs_predicted.png"

colors = ["tab:blue" if label.endswith("T0") else "tab:red" for label in LABEL_ORDER]

neut_values = [NEUTRALIZATION[label] for label in LABEL_ORDER]

df = pd.read_csv(PRED_CSV)
df["label"] = "A" + df["n"].astype(str) + "T" + df["m"].astype(str)
pred_by_label = df.set_index("label")[PRED_COL]
pred_values = [pred_by_label[label] for label in LABEL_ORDER]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4.5))

ax1.bar(LABEL_ORDER, neut_values, color=colors)
ax1.set_ylabel("Neutralization Ratio (%)")
ax1.set_ylim(0, 100)
ax1.set_title("Figure S24 (top histogram) reproduction")
ax1.tick_params(axis="x", rotation=45)

ax2.bar(LABEL_ORDER, pred_values, color=colors)
ax2.set_ylabel("Predicted pKd (this model)")
ax2.set_title("Predicted pKd, same samples/order")
ax2.tick_params(axis="x", rotation=45)

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150)
print(f"Saved {OUT_PNG}")
