"""
Two side-by-side bar plot panels (same figure), sharing the same x-axis
(A{n}T{m} sample labels, in the same order) so they can be visually compared:
  1. The Figure S24 hemolysis neutralization ratio -- BOTTOM panel (ligand
     concentration 0.1 mM), pixel-extracted from the source PDF
     (bar-color pixel detection + y-axis box-border calibration).
  2. This repo's featured model's predicted pKd (results/predicted_pKa_
     KanM_balanced.csv).

Each panel uses its own natural y-scale (percent vs. pKd units) -- blue
bars for the T0 (m=0) samples, red for the rest, matching Figure S24's own
color coding. Only the 15 of 18 Hoshino_polymer pairs with a matching
Figure S24 value are included, for a like-for-like x-axis.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

NEUTRALIZATION = {
    "A2T0": 9.4, "A2T1": 3.6, "A2T2": 16.0, "A2T3": 26.4, "A2T4": 25.2, "A2T5": 31.8,
    "A3T0": -4.7, "A3T1": 32.4, "A3T2": 19.4, "A3T3": 6.8, "A3T4": 21.1,
    "A4T0": 20.0, "A4T1": 21.3, "A4T2": 30.9, "A4T3": 62.3,
}
LABEL_ORDER = ["A2T0", "A2T1", "A2T2", "A2T3", "A2T4", "A2T5",
              "A3T0", "A3T1", "A3T2", "A3T3", "A3T4",
              "A4T0", "A4T1", "A4T2", "A4T3"]

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_CSV = os.path.join(REPO, "results", "predicted_pKa_KanM_balanced.csv")
PRED_COL = "predicted_pKd"
OUT_PNG = os.path.join(REPO, "results", "neutralization_vs_predicted.png")

colors = ["tab:blue" if label.endswith("T0") else "tab:red" for label in LABEL_ORDER]

neut_values = [NEUTRALIZATION[label] for label in LABEL_ORDER]

df = pd.read_csv(PRED_CSV)
df["label"] = "A" + df["n"].astype(str) + "T" + df["m"].astype(str)
pred_by_label = df.set_index("label")[PRED_COL]
pred_values = [pred_by_label[label] for label in LABEL_ORDER]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4.5))

ax1.bar(LABEL_ORDER, neut_values, color=colors)
ax1.set_ylabel("Neutralization Ratio (%)")
ax1.set_ylim(-10, 100)
ax1.axhline(0, color="black", linewidth=0.8)
ax1.set_title("Figure S24 (bottom histogram, 0.1 mM) reproduction")
ax1.tick_params(axis="x", rotation=45)

ax2.bar(LABEL_ORDER, pred_values, color=colors)
ax2.set_ylabel("Predicted pKd (featured model)")
ax2.set_title("Predicted pKd, same samples/order")
ax2.tick_params(axis="x", rotation=45)

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150)
print(f"Saved {OUT_PNG}")
