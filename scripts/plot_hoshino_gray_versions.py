"""
Gray-colored (no T0-vs-rest blue/red split) versions of the repo's existing
Hoshino scatter plots (0.1/0.3/1.0mM) and the neutralization-vs-predicted
bar plot. Saved to results/ with a "_gray" suffix, not referenced from
README.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO, "results", "predicted_pKa_KanM_balanced.csv")
PRED_COL = "predicted_pKd"
N_RESAMPLES = 99999
PERM_SEED = 42
GRAY = "gray"

NEUTRALIZATION_0P1 = {
    "A2T0": 9.4, "A2T1": 3.6, "A2T2": 16.0, "A2T3": 26.4, "A2T4": 25.2, "A2T5": 31.8,
    "A3T0": -4.7, "A3T1": 32.4, "A3T2": 19.4, "A3T3": 6.8, "A3T4": 21.1,
    "A4T0": 20.0, "A4T1": 21.3, "A4T2": 30.9, "A4T3": 62.3,
}
LABEL_ORDER = ["A2T0", "A2T1", "A2T2", "A2T3", "A2T4", "A2T5",
              "A3T0", "A3T1", "A3T2", "A3T3", "A3T4",
              "A4T0", "A4T1", "A4T2", "A4T3"]


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


df = pd.read_csv(CSV)
df["label"] = "A" + df["n"].astype(str) + "T" + df["m"].astype(str)

# --- scatter plots (0.1mM has no p-value in title, matching the repo's
# just-updated version; 0.3mM/1.0mM keep p-values, matching their current
# un-edited scripts) ---
scatter_specs = [
    {
        "neut_col": None, "neut_map": NEUTRALIZATION_0P1,
        "out": "hoshino_correlation_scatter_gray.png",
        "ylabel": "Neutralization Ratio (%) -- Figure S24 bottom panel (0.1 mM)",
        "show_p": False,
    },
    {
        "neut_col": "neutralization_0.3mM", "neut_map": None,
        "out": "hoshino_correlation_scatter_0p3mM_gray.png",
        "ylabel": "Neutralization Ratio (%) -- Figure S24 middle panel (0.3 mM)",
        "show_p": True,
    },
    {
        "neut_col": "neutralization_1.0mM", "neut_map": None,
        "out": "hoshino_correlation_scatter_1p0mM_gray.png",
        "ylabel": "Neutralization Ratio (%) -- Figure S24 top panel (1.0 mM)",
        "show_p": True,
    },
]

for spec in scatter_specs:
    if spec["neut_map"] is not None:
        sub = df.copy()
        sub["neutralization"] = sub["label"].map(spec["neut_map"])
        sub = sub.dropna(subset=["neutralization"]).reset_index(drop=True)
        neut_col = "neutralization"
    else:
        sub = df.dropna(subset=[spec["neut_col"]]).reset_index(drop=True)
        neut_col = spec["neut_col"]

    r, p = pearsonr_perm(sub[PRED_COL], sub[neut_col])
    rho, p_rho = spearmanr_perm(sub[PRED_COL], sub[neut_col])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(sub[PRED_COL], sub[neut_col], c=GRAY, s=60, edgecolor="black", linewidth=0.5)

    slope, intercept = np.polyfit(sub[PRED_COL], sub[neut_col], 1)
    x_line = np.array([sub[PRED_COL].min(), sub[PRED_COL].max()])
    ax.plot(x_line, slope * x_line + intercept, linestyle=":", color="black", linewidth=1.5,
           label=f"least-squares fit (y={slope:.2f}x+{intercept:.2f})")
    ax.legend(fontsize=8, loc="upper left")

    for _, row in sub.iterrows():
        ax.annotate(row["label"], (row[PRED_COL], row[neut_col]),
                   textcoords="offset points", xytext=(5, 3), fontsize=8)
    ax.set_xlabel("Predicted pKd (combined model)")
    ax.set_ylabel(spec["ylabel"])
    if spec["show_p"]:
        ax.set_title(f"n=15: Pearson r={r:.3f} (p={p:.4f}), Spearman rho={rho:.3f} (p={p_rho:.4f})", fontsize=11)
    else:
        ax.set_title(f"n=15: Pearson r={r:.3f}, Spearman rho={rho:.3f}", fontsize=11)
    fig.subplots_adjust(top=0.90)
    fig.tight_layout()
    out_path = os.path.join(REPO, "results", spec["out"])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

# --- bar plot (neutralization vs predicted, 0.1mM) ---
neut_values = [NEUTRALIZATION_0P1[label] for label in LABEL_ORDER]
pred_by_label = df.set_index("label")[PRED_COL]
pred_values = [pred_by_label[label] for label in LABEL_ORDER]


def padded_range(values, frac=0.05):
    lo, hi = min(values), max(values)
    pad = (hi - lo) * frac
    return lo - pad, hi + pad


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4.5))

ax1.bar(LABEL_ORDER, neut_values, color=GRAY)
ax1.set_ylabel("Neutralization Ratio (%)")
ax1.set_ylim(*padded_range(neut_values))
ax1.axhline(0, color="black", linewidth=0.8)
ax1.set_title("Figure S24 (bottom histogram, 0.1 mM) reproduction")
ax1.tick_params(axis="x", rotation=45)

ax2.bar(LABEL_ORDER, pred_values, color=GRAY)
ax2.set_ylabel("Predicted pKd (combined model)")
ax2.set_ylim(*padded_range(pred_values))
ax2.set_title("Predicted pKd, same samples/order")
ax2.tick_params(axis="x", rotation=45)

fig.tight_layout()
out_path = os.path.join(REPO, "results", "neutralization_vs_predicted_gray.png")
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"Saved {out_path}")
