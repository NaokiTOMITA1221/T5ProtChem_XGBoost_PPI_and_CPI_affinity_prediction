"""
This repo's featured model: RAW/vanilla T5ProtChem-native encoder
(use_lora=False, never fine-tuned) + XGBoost, trained on the seed=42
uniqueonly resplit (see resplit_seed42_unbalanced.py), with NO PPI
oversampling and NO ProtSMILES splice augmentation ("unbalanced" -- train
set used exactly as split, no synthetic copies). val/test are the same
clean split rows (never augmented, matching train here since there's no
augmentation in this configuration).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

sys.path.insert(0, "/mnt/hdd/tomita/PPI_CPI_prediction/T5VQBoost/src")
import unified_dataset as ud

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "t5contactpretrain_unified_model", "/mnt/hdd/tomita/PPI_CPI_prediction/T5ContactPretrain/src/unified_model.py")
um_cp = importlib.util.module_from_spec(_spec)
sys.path.insert(0, "/mnt/hdd/tomita/PPI_CPI_prediction/T5ContactPretrain/src")
sys.modules["t5contactpretrain_unified_model"] = um_cp
_spec.loader.exec_module(um_cp)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T5_CHECKPOINT_PATH = "/mnt/hdd/tomita/PPI_CPI_prediction/T5ProtChem/weights/Lightning_weights/Pretrained/T5ProtChem/model.pt"
VOCAB_FILE = "/mnt/hdd/tomita/PPI_CPI_prediction/T5ProtChem/src/vocab/style2.json"
TRAIN_CSV = os.path.join(REPO, "data/split_pure_random_uniqueonly_seed42_train.csv")
VAL_CSV = os.path.join(REPO, "data/split_pure_random_uniqueonly_seed42_val.csv")
TEST_CSV = os.path.join(REPO, "data/split_pure_random_uniqueonly_seed42_test.csv")
OUTPUT_DIR = os.path.join(REPO, "results")
PROTEIN_MAX_LENGTH = 768
DRUG_MAX_LENGTH = 768

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Constructing RAW (use_lora=False, no fine-tuning) T5ProtChem-native encoder from {T5_CHECKPOINT_PATH} ...")
model = um_cp.ContactPretrainModel(t5_checkpoint_path=T5_CHECKPOINT_PATH, use_lora=False)
model.to(device)
model.eval()
print(f"use_lora={model.encoder.use_lora}")

dm = ud.ContactPretrainDataModule(
    train_csv_path=TRAIN_CSV, val_csv_path=VAL_CSV, test_csv_path=TEST_CSV,
    vocab_file=VOCAB_FILE)
protein_tokenizer, drug_tokenizer, text_encoder = dm.protein_tokenizer, dm.drug_tokenizer, dm.text_encoder


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


def extract(df):
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


print("Loading seed=42 split ...")
train_df, val_df, test_df = pd.read_csv(TRAIN_CSV), pd.read_csv(VAL_CSV), pd.read_csv(TEST_CSV)
print(f"  train={len(train_df)} val={len(val_df)} test={len(test_df)}")

print("Extracting train features (no augmentation) ...")
X_train, y_train, origins_train = extract(train_df)
print("Extracting val features ...")
X_val, y_val, origins_val = extract(val_df)
print("Extracting test features ...")
X_test, y_test, origins_test = extract(test_df)


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

os.makedirs(OUTPUT_DIR, exist_ok=True)
model_path = os.path.join(OUTPUT_DIR, "xgb_model.json")
reg.save_model(model_path)
with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump({"val": val_metrics, "test": test_metrics, "best_iteration": reg.best_iteration,
              "seed": 42, "unbalanced": True}, f, indent=2)
print(f"Saved model to {model_path} and metrics.json to {OUTPUT_DIR}")
