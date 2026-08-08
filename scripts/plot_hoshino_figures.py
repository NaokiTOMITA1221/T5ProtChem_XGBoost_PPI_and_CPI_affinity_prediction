"""
One grouped bar plot comparing, per A{n}T{m} sample (the 15 with a matching
Figure S24 neutralization value; 3 of the 18 Hoshino_polymer pairs have no
Figure S24 value and are excluded here for a like-for-like x-axis):
  - the Figure S24 hemolysis neutralization ratio (pixel-extracted from the
    source PDF earlier in this project -- see conversation)
  - this repo's model's predicted pKd
    (results/predicted_pKa_KanM_T5ProtChem_raw_uniqueonly_quadsplice.csv)

Both series are normalized by their OWN max value (so each ranges 0-1) so
their shapes can be compared directly despite being on very different
natural scales (percent vs. pKd units). Two bars per x-axis label.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
OUT_COMBINED = "/mnt/hdd/tomita/PPI_CPI_prediction/T5ProtChem_XGBoost_PPI_and_CPI_affinity_prediction/results/neutralization_vs_predicted_normalized.png"

neut_values = np.array([NEUTRALIZATION[label] for label in LABEL_ORDER])
neut_norm = neut_values / neut_values.max()

df = pd.read_csv(PRED_CSV)
df["label"] = "A" + df["n"].astype(str) + "T" + df["m"].astype(str)
pred_by_label = df.set_index("label")[PRED_COL]
pred_values = np.array([pred_by_label[label] for label in LABEL_ORDER])
pred_norm = pred_values / pred_values.max()

x = np.arange(len(LABEL_ORDER))
width = 0.4

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width / 2, neut_norm, width, label="Neutralization ratio (norm.)", color="tab:red")
ax.bar(x + width / 2, pred_norm, width, label="Predicted pKd (norm.)", color="tab:purple")
ax.set_xticks(x)
ax.set_xticklabels(LABEL_ORDER, rotation=45)
ax.set_ylabel("Value / max(value)")
ax.set_title("Neutralization ratio vs. predicted pKd (each normalized by its own max)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_COMBINED, dpi=150)
print(f"Saved {OUT_COMBINED}")
