"""
Reproduction of Figure S24's top histogram (hemolysis neutralization ratio
per A{n}T{m} sample, pixel-extracted from the source PDF) -- blue bars for
the T0 (m=0) samples, red for the rest, matching the source figure's color
coding. Independent of any trained model; only needs the neutralization
values themselves.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NEUTRALIZATION = {
    "A2T0": 19.9, "A2T1": 38.8, "A2T2": 63.3, "A2T3": 56.6, "A2T4": 86.9, "A2T5": 51.7,
    "A3T0": 24.8, "A3T1": 90.6, "A3T2": 51.5, "A3T3": 67.8, "A3T4": 58.7,
    "A4T0": 84.3, "A4T1": 83.3, "A4T2": 77.5, "A4T3": 85.0,
}
LABEL_ORDER = ["A2T0", "A2T1", "A2T2", "A2T3", "A2T4", "A2T5",
              "A3T0", "A3T1", "A3T2", "A3T3", "A3T4",
              "A4T0", "A4T1", "A4T2", "A4T3"]
OUT_BAR = "/mnt/hdd/tomita/PPI_CPI_prediction/T5ProtChem_XGBoost_PPI_and_CPI_affinity_prediction/results/figureS24_reproduction.png"

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
