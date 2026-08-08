import pandas as pd
from scipy.stats import pearsonr, spearmanr

NEUTRALIZATION = {
    "A2T0": 19.9, "A2T1": 38.8, "A2T2": 63.3, "A2T3": 56.6, "A2T4": 86.9, "A2T5": 51.7,
    "A3T0": 24.8, "A3T1": 90.6, "A3T2": 51.5, "A3T3": 67.8, "A3T4": 58.7,
    "A4T0": 84.3, "A4T1": 83.3, "A4T2": 77.5, "A4T3": 85.0,
}

CSV = "/home/tomita/Hoshino_polymer/predicted_pKa_KanM_T5ProtChem_raw_uniqueonly_quadsplice.csv"
PRED_COL = "predicted_pKd_T5ProtChem_raw_uniqueonly_quadsplice"

df = pd.read_csv(CSV)
df["label"] = "A" + df["n"].astype(str) + "T" + df["m"].astype(str)
df["neutralization"] = df["label"].map(NEUTRALIZATION)
sub = df.dropna(subset=["neutralization"])
print(f"Matched {len(sub)} / {len(df)} rows")

r_pred, p_pred = pearsonr(sub[PRED_COL], sub["neutralization"])
rho_pred, p_rho_pred = spearmanr(sub[PRED_COL], sub["neutralization"])
r_ref, p_ref = pearsonr(sub["pKd"], sub["neutralization"])

print(f"predicted_pKd vs neutralization: pearson r={r_pred:.4f} (p={p_pred:.4f}), spearman rho={rho_pred:.4f} (p={p_rho_pred:.4f})")
print(f"reference pKd vs neutralization: pearson r={r_ref:.4f} (p={p_ref:.4f})")
print(sub[["label", "pKd", PRED_COL, "neutralization"]].to_string(index=False))
