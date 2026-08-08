"""
Rebuilds the exact "seed=51, unbalanced" model from the 10-seed robustness
check (robustness_t5protchem_raw_uniqueonly_quadsplice_10splits.py):
RAW/no-LoRA T5ProtChem-native encoder, uniqueonly pool (swap-pair duplicates
removed before splitting) reshuffled with seed=51 into a fresh 80/10/10
partition, NO oversampling / NO ProtSMILES splice augmentation ("unbalanced"
condition) -- test overall pearson 0.783, Hoshino r=0.618 (p=0.014).

Saves the seed=51 split CSVs, trains the model, evaluates, and predicts on
Hoshino_polymer -- everything needed to replace the repo's current
(fixed-split, balanced) model with this one.
"""
import importlib.util
import json
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

sys.path.insert(0, "/mnt/hdd/tomita/PPI_CPI_prediction/T5VQBoost/src")
import unified_dataset as ud

_spec = importlib.util.spec_from_file_location(
    "t5contactpretrain_unified_model", "/mnt/hdd/tomita/PPI_CPI_prediction/T5ContactPretrain/src/unified_model.py")
um_cp = importlib.util.module_from_spec(_spec)
sys.path.insert(0, "/mnt/hdd/tomita/PPI_CPI_prediction/T5ContactPretrain/src")
sys.modules["t5contactpretrain_unified_model"] = um_cp
_spec.loader.exec_module(um_cp)

T5_CHECKPOINT_PATH = "/mnt/hdd/tomita/PPI_CPI_prediction/T5ProtChem/weights/Lightning_weights/Pretrained/T5ProtChem/model.pt"
VOCAB_FILE = "/mnt/hdd/tomita/PPI_CPI_prediction/T5ProtChem/src/vocab/style2.json"
SPLIT_CSVS = [
    "/mnt/hdd/tomita/PPI_CPI_prediction/data/Affinity_data_culation/integrated_data_csv/split_pure_random_uniqueonly_train.csv",
    "/mnt/hdd/tomita/PPI_CPI_prediction/data/Affinity_data_culation/integrated_data_csv/split_pure_random_uniqueonly_val.csv",
    "/mnt/hdd/tomita/PPI_CPI_prediction/data/Affinity_data_culation/integrated_data_csv/split_pure_random_uniqueonly_test.csv",
]
HOSHINO_CSV = "/home/tomita/Hoshino_polymer/predicted_pKa_KanM.csv"
OUT_DATA_DIR = "/mnt/hdd/tomita/PPI_CPI_prediction/data/Affinity_data_culation/integrated_data_csv"
OUTPUT_DIR = "/mnt/hdd/tomita/PPI_CPI_prediction/T5VQBoost/runs/boost_baseline/version_t5protchem_raw_uniqueonly_seed51_unbalanced"
PROTEIN_MAX_LENGTH = 768
DRUG_MAX_LENGTH = 768
VAL_FRAC = 0.1
TEST_FRAC = 0.1
SEED = 51

NEUTRALIZATION = {
    "A2T0": 19.9, "A2T1": 38.8, "A2T2": 63.3, "A2T3": 56.6, "A2T4": 86.9, "A2T5": 51.7,
    "A3T0": 24.8, "A3T1": 90.6, "A3T2": 51.5, "A3T3": 67.8, "A3T4": 58.7,
    "A4T0": 84.3, "A4T1": 83.3, "A4T2": 77.5, "A4T3": 85.0,
}

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Constructing RAW (use_lora=False, no fine-tuning) T5ProtChem-native encoder from {T5_CHECKPOINT_PATH} ...")
model = um_cp.ContactPretrainModel(t5_checkpoint_path=T5_CHECKPOINT_PATH, use_lora=False)
model.to(device)
model.eval()

full_df = pd.concat([pd.read_csv(p) for p in SPLIT_CSVS], ignore_index=True)
print(f"Reconstructed full uniqueonly pool: {len(full_df)} rows")

dm = ud.ContactPretrainDataModule(
    train_csv_path=SPLIT_CSVS[0], val_csv_path=SPLIT_CSVS[1], test_csv_path=SPLIT_CSVS[2],
    vocab_file=VOCAB_FILE)
protein_tokenizer, drug_tokenizer, text_encoder = dm.protein_tokenizer, dm.drug_tokenizer, dm.text_encoder

origins_set = ud.CPI_ORIGINS | ud.PPI_ORIGINS
relevant = full_df[full_df["data_origin"].isin(origins_set) & full_df["pKd"].notna()]
single_chain = ~(relevant["molA"].str.contains(ud.CHAIN_SEP, regex=False)
                 | relevant["molB"].str.contains(ud.CHAIN_SEP, regex=False))
relevant = relevant[single_chain].reset_index(drop=True)
N = len(relevant)
print(f"[pool] {N} eligible single-chain rows")

rng = random.Random(SEED)
idx = list(range(N))
rng.shuffle(idx)
n_val = int(N * VAL_FRAC)
n_test = int(N * TEST_FRAC)
val_idx = idx[:n_val]
test_idx = idx[n_val:n_val + n_test]
train_idx = idx[n_val + n_test:]
print(f"seed={SEED}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

train_df = relevant.iloc[train_idx].reset_index(drop=True)
val_df = relevant.iloc[val_idx].reset_index(drop=True)
test_df = relevant.iloc[test_idx].reset_index(drop=True)

os.makedirs(OUT_DATA_DIR, exist_ok=True)
train_df.to_csv(os.path.join(OUT_DATA_DIR, "split_pure_random_uniqueonly_seed51_train.csv"), index=False)
val_df.to_csv(os.path.join(OUT_DATA_DIR, "split_pure_random_uniqueonly_seed51_val.csv"), index=False)
test_df.to_csv(os.path.join(OUT_DATA_DIR, "split_pure_random_uniqueonly_seed51_test.csv"), index=False)
print("Saved seed=51 split CSVs")


def encode_side(raw, fmt):
    return text_encoder.encode_aa(raw) if fmt == "AA" else text_encoder.encode_smiles(raw)


@torch.no_grad()
def mean_pool_feature(molA_text, molB_text):
    a_enc = protein_tokenizer(molA_text, truncation=True, max_length=PROTEIN_MAX_LENGTH, return_tensors="pt")
    b_enc = drug_tokenizer(molB_text, truncation=True, max_length=DRUG_MAX_LENGTH, return_tensors="pt")
    a_ids, a_mask = a_enc["input_ids"].to(device), a_enc["attention_mask"].to(device)
    b_ids, b_mask = b_enc["input_ids"].to(device), b_enc["attention_mask"].to(device)
    h_a = model.encoder(a_ids, a_mask)
    h_b = model.encoder(b_ids, b_mask)
    mean_a = h_a[0].mean(dim=0)
    mean_b = h_b[0].mean(dim=0)
    return torch.cat([mean_a, mean_b]).cpu().numpy()


def extract_clean(df):
    X = np.zeros((len(df), 1280), dtype=np.float32)
    for i, (molA, molB, fa, fb) in enumerate(zip(df["molA"], df["molB"], df["molA_format"], df["molB_format"])):
        molA_text = encode_side(molA, fa)
        molB_text = encode_side(molB, fb)
        X[i] = mean_pool_feature(molA_text, molB_text)
        if (i + 1) % 5000 == 0:
            print(f"    ... {i + 1}/{len(df)}")
    y = df["pKd"].astype("float32").to_numpy()
    origins = df["data_origin"].to_numpy()
    return X, y, origins


print("Extracting train features (no augmentation -- unbalanced) ...")
X_train, y_train, origins_train = extract_clean(train_df)
print("Extracting val features ...")
X_val, y_val, origins_val = extract_clean(val_df)
print("Extracting test features ...")
X_test, y_test, origins_test = extract_clean(test_df)


def evaluate(y_true, y_pred, origins):
    groups = {"overall": None, "cpi": ud.CPI_ORIGINS, "ppi": ud.PPI_ORIGINS}
    metrics = {}
    for name, origin_set in groups.items():
        mask = np.ones(len(y_true), dtype=bool) if origin_set is None else np.isin(origins, list(origin_set))
        if mask.sum() < 2:
            continue
        yt, yp = y_true[mask], y_pred[mask]
        metrics[f"{name}/pearson"] = float(pearsonr(yt, yp)[0])
        metrics[f"{name}/spearman"] = float(spearmanr(yt, yp)[0])
        metrics[f"{name}/rmse"] = float(mean_squared_error(yt, yp) ** 0.5)
    return metrics


print(f"Fitting XGBoost (train={X_train.shape}, val={X_val.shape}, test={X_test.shape}) ...")
reg = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=30, eval_metric="rmse")
reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
print(f"Best iteration: {reg.best_iteration}")

val_pred = reg.predict(X_val)
test_pred = reg.predict(X_test)
val_metrics = evaluate(y_val, val_pred, origins_val)
test_metrics = evaluate(y_test, test_pred, origins_test)

print("=== val ===")
for k, v in val_metrics.items():
    print(f"  {k}: {v:.4f}")
print("=== test ===")
for k, v in test_metrics.items():
    print(f"  {k}: {v:.4f}")

model_path = os.path.join(OUTPUT_DIR, "xgb_model.json")
reg.save_model(model_path)
with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump({"val": val_metrics, "test": test_metrics, "best_iteration": reg.best_iteration,
              "seed": SEED, "unbalanced": True}, f, indent=2)
print(f"Saved model to {model_path}")

# --- test scatter plot ---
r_test, p_test = pearsonr(y_test, test_pred)
rmse_test = mean_squared_error(y_test, test_pred) ** 0.5
is_cpi = np.isin(origins_test, list(ud.CPI_ORIGINS))
is_ppi = np.isin(origins_test, list(ud.PPI_ORIGINS))
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test[is_cpi], test_pred[is_cpi], s=10, alpha=0.4, label=f"CPI (n={is_cpi.sum()})", color="tab:blue")
ax.scatter(y_test[is_ppi], test_pred[is_ppi], s=10, alpha=0.4, label=f"PPI (n={is_ppi.sum()})", color="tab:orange")
lo = min(y_test.min(), test_pred.min())
hi = max(y_test.max(), test_pred.max())
ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
ax.set_xlabel("True pKd")
ax.set_ylabel("Predicted pKd")
ax.set_title(f"Test set (n={len(y_test)}): Pearson r={r_test:.3f}, RMSE={rmse_test:.3f}")
ax.legend(loc="upper left", fontsize=9)
ax.set_aspect("equal", adjustable="box")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "test_scatter.png"), dpi=150)
print("Saved test_scatter.png")

# --- Hoshino prediction + correlation ---
hoshino_df = pd.read_csv(HOSHINO_CSV)
print(f"Loaded {len(hoshino_df)} Hoshino rows")


@torch.no_grad()
def extract_hoshino(seq, smi):
    return mean_pool_feature(text_encoder.encode_aa(seq), text_encoder.encode_smiles(smi))


X_hoshino = np.stack([extract_hoshino(seq, smi)
                      for seq, smi in zip(hoshino_df["target_seq"], hoshino_df["drug_smiles"])])
hoshino_pred = reg.predict(X_hoshino)
hoshino_df["predicted_pKd_T5ProtChem_raw_uniqueonly_seed51_unbalanced"] = hoshino_pred
hoshino_out_csv = "/home/tomita/Hoshino_polymer/predicted_pKa_KanM_T5ProtChem_raw_uniqueonly_seed51_unbalanced.csv"
hoshino_df.to_csv(hoshino_out_csv, index=False)
print(f"Saved Hoshino predictions to {hoshino_out_csv}")

hoshino_df["label"] = "A" + hoshino_df["n"].astype(str) + "T" + hoshino_df["m"].astype(str)
hoshino_df["neutralization"] = hoshino_df["label"].map(NEUTRALIZATION)
sub = hoshino_df.dropna(subset=["neutralization"])
r_hosh, p_hosh = pearsonr(sub["predicted_pKd_T5ProtChem_raw_uniqueonly_seed51_unbalanced"], sub["neutralization"])
print(f"Hoshino correlation (n={len(sub)}): pearson r={r_hosh:.4f} (p={p_hosh:.4f})")

fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.scatter(sub["predicted_pKd_T5ProtChem_raw_uniqueonly_seed51_unbalanced"], sub["neutralization"],
           s=40, color="tab:purple")
for _, row in sub.iterrows():
    ax2.annotate(row["label"], (row["predicted_pKd_T5ProtChem_raw_uniqueonly_seed51_unbalanced"],
                               row["neutralization"]),
                fontsize=7, xytext=(3, 3), textcoords="offset points")
ax2.set_xlabel("Predicted pKd (this model)")
ax2.set_ylabel("Neutralization Ratio (%) [Figure S24]")
ax2.set_title(f"Predicted pKd vs. neutralization ratio (n={len(sub)})\nPearson r={r_hosh:.3f}, p={p_hosh:.4f}")
fig2.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, "hoshino_correlation_scatter.png"), dpi=150)
print("Saved hoshino_correlation_scatter.png")

with open(os.path.join(OUTPUT_DIR, "hoshino_correlation.json"), "w") as f:
    json.dump({"n": len(sub), "pearson_r": float(r_hosh), "pearson_p": float(p_hosh)}, f, indent=2)
print("Done.")
