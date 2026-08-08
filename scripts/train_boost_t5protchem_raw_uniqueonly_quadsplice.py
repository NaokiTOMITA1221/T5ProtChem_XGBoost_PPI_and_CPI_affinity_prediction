"""
Same data setup as train_boost_t5protchem_contactpretrain_uniqueonly_
quadsplice.py (split_pure_random_uniqueonly -- swap-pair duplicates removed
BEFORE splitting -- + CPI:PPI ~1:1 oversampling with mutants included +
ProtSMILES splice range 12-32 residues, quadruple the original 3-8 default),
but using the RAW/vanilla T5ProtChem-native encoder (use_lora=False, freshly
constructed from the raw pretrained checkpoint, NOT contact-pretrained, NOT
pKd-fine-tuned at all) instead of the contact-pretrain-only checkpoint.
val/test stay CLEAN (unaugmented).
"""
import json
import os
import random
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

T5_CHECKPOINT_PATH = "/mnt/hdd/tomita/PPI_CPI_prediction/T5ProtChem/weights/Lightning_weights/Pretrained/T5ProtChem/model.pt"
VOCAB_FILE = "/mnt/hdd/tomita/PPI_CPI_prediction/T5ProtChem/src/vocab/style2.json"
TRAIN_CSV = "/mnt/hdd/tomita/PPI_CPI_prediction/data/Affinity_data_culation/integrated_data_csv/split_pure_random_uniqueonly_train.csv"
VAL_CSV = "/mnt/hdd/tomita/PPI_CPI_prediction/data/Affinity_data_culation/integrated_data_csv/split_pure_random_uniqueonly_val.csv"
TEST_CSV = "/mnt/hdd/tomita/PPI_CPI_prediction/data/Affinity_data_culation/integrated_data_csv/split_pure_random_uniqueonly_test.csv"
OUTPUT_DIR = "/mnt/hdd/tomita/PPI_CPI_prediction/T5VQBoost/runs/boost_baseline/version_t5protchem_raw_uniqueonly_quadsplice"
PROTEIN_MAX_LENGTH = 768
DRUG_MAX_LENGTH = 768
MAX_RAW_LENGTH = float("inf")
TARGET_RATIO = 1.0
MAX_FACTOR = 20
SMILES_SPLICE_MIN, SMILES_SPLICE_MAX = 12, 32
PKD_NOISE_FRAC = 0.05
AUG_SEED = 12345

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Constructing RAW (use_lora=False, no fine-tuning) T5ProtChem-native encoder from {T5_CHECKPOINT_PATH} ...")
model = um_cp.ContactPretrainModel(t5_checkpoint_path=T5_CHECKPOINT_PATH, use_lora=False)
model.to(device)
model.eval()
print(f"use_lora={model.encoder.use_lora}")

dm = ud.ContactPretrainDataModule(
    train_csv_path=TRAIN_CSV, val_csv_path=VAL_CSV, test_csv_path=TEST_CSV,
    vocab_file=VOCAB_FILE)
protein_tokenizer, drug_tokenizer, text_encoder = dm.protein_tokenizer, dm.drug_tokenizer, dm.text_encoder


def load_rows(csv_path):
    """Same eligibility filter as unified_dataset.load_pkd_rows (single-chain
    only, CPI/PPI origins with real pKd) -- mutants INCLUDED, matching #6's
    original (non-WT-restricted) data."""
    df = pd.read_csv(csv_path)
    origins_set = ud.CPI_ORIGINS | ud.PPI_ORIGINS
    relevant = df[df["data_origin"].isin(origins_set) & df["pKd"].notna()]
    single_chain = ~(relevant["molA"].str.contains(ud.CHAIN_SEP, regex=False)
                     | relevant["molB"].str.contains(ud.CHAIN_SEP, regex=False))
    n_multichain = int((~single_chain).sum())
    relevant = relevant[single_chain].reset_index(drop=True)
    print(f"  {os.path.basename(csv_path)}: {len(relevant)} rows ({n_multichain} multi-chain dropped)")
    return relevant


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


print("Loading splits (mutants included) ...")
train_df = load_rows(TRAIN_CSV)
val_df = load_rows(VAL_CSV)
test_df = load_rows(TEST_CSV)


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


print("Extracting CLEAN train features ...")
X_train_clean, y_train_clean, origins_train_clean = extract_clean(train_df)
print("Extracting CLEAN val features ...")
X_val, y_val, origins_val = extract_clean(val_df)
print("Extracting CLEAN test features ...")
X_test, y_test, origins_test = extract_clean(test_df)

cpi_count = int(np.isin(origins_train_clean, list(ud.CPI_ORIGINS)).sum())
ppi_count = int(np.isin(origins_train_clean, list(ud.PPI_ORIGINS)).sum())
extra_needed = TARGET_RATIO * cpi_count - ppi_count
factor = max(round(extra_needed / ppi_count), 0) if ppi_count > 0 else 0
factor = min(factor, MAX_FACTOR)
print(f"Train CPI={cpi_count} PPI={ppi_count} -> oversample_factor={factor}")

ppi_train_idx = [i for i in range(len(train_df)) if origins_train_clean[i] in ud.PPI_ORIGINS]
raw_molA = train_df["molA"].tolist()
raw_molB = train_df["molB"].tolist()
fmt_molA = train_df["molA_format"].tolist()
fmt_molB = train_df["molB_format"].tolist()
pKd_list = train_df["pKd"].astype("float32").tolist()
origin_list = train_df["data_origin"].tolist()

aug_rng = random.Random(AUG_SEED)
X_aug_list, y_aug_list, origin_aug_list = [], [], []
print(f"Generating {factor} augmented copies for {len(ppi_train_idx)} PPI train rows ...")
for count, i in enumerate(ppi_train_idx):
    for c in range(factor):
        def splice_side(raw, fmt):
            if fmt != "AA":
                return text_encoder.encode_smiles(raw)
            random.seed(aug_rng.random())
            return text_encoder.splice_residues_with_smiles(raw, SMILES_SPLICE_MIN, SMILES_SPLICE_MAX)

        molA_text = splice_side(raw_molA[i], fmt_molA[i])
        molB_text = splice_side(raw_molB[i], fmt_molB[i])
        X_aug_list.append(mean_pool_feature(molA_text, molB_text))
        jitter = 1.0 + aug_rng.uniform(-PKD_NOISE_FRAC, PKD_NOISE_FRAC)
        y_aug_list.append(pKd_list[i] * jitter)
        origin_aug_list.append(origin_list[i])
    if (count + 1) % 500 == 0:
        print(f"  ... {count + 1}/{len(ppi_train_idx)} PPI rows augmented")

X_aug = np.stack(X_aug_list) if X_aug_list else np.zeros((0, 1280), dtype=np.float32)
y_aug = np.array(y_aug_list, dtype=np.float32)

X_train = np.concatenate([X_train_clean, X_aug], axis=0)
y_train = np.concatenate([y_train_clean, y_aug], axis=0)
print(f"Final train set: {X_train.shape} (clean={X_train_clean.shape[0]}, augmented={X_aug.shape[0]})")


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
              "oversample_factor": factor}, f, indent=2)
print(f"Saved model to {model_path} and metrics.json to {OUTPUT_DIR}")
