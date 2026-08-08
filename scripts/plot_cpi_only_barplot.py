"""
Bar plot of the CPI-only single-task model's predicted pKd across
Hoshino_polymer samples (results/hoshino_predictions_cpi_ppi_only.csv),
same x-axis/order/color-coding as the Figure S24 reproduction, for
comparison. This model was trained on ONLY CPI-origin rows from
split_pure_random_uniqueonly (raw/no-LoRA T5ProtChem encoder) and showed the
strongest Hoshino correlation found so far (Pearson r=0.683, p=0.0050).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LABEL_ORDER = ["A2T0", "A2T1", "A2T2", "A2T3", "A2T4", "A2T5",
              "A3T0", "A3T1", "A3T2", "A3T3", "A3T4",
              "A4T0", "A4T1", "A4T2", "A4T3"]

PRED_CSV = "/mnt/hdd/tomita/PPI_CPI_prediction/T5ProtChem_XGBoost_PPI_and_CPI_affinity_prediction/results/hoshino_predictions_cpi_ppi_only.csv"
PRED_COL = "predicted_pKd_cpi_only"
OUT_PNG = "/mnt/hdd/tomita/PPI_CPI_prediction/T5ProtChem_XGBoost_PPI_and_CPI_affinity_prediction/results/cpi_only_predicted_pkd_barplot.png"

colors = ["tab:blue" if label.endswith("T0") else "tab:red" for label in LABEL_ORDER]

df = pd.read_csv(PRED_CSV)
pred_by_label = df.set_index("label")[PRED_COL]
pred_values = [pred_by_label[label] for label in LABEL_ORDER]

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(LABEL_ORDER, pred_values, color=colors)
ax.set_ylabel("Predicted pKd (CPI-only model)")
ax.set_ylim(3, max(pred_values) * 1.02)
ax.set_title("CPI-only model: predicted pKd, same samples/order as Figure S24")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150)
print(f"Saved {OUT_PNG}")
