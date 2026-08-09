"""
Re-extracts test-set features (raw T5ProtChem encoder, same as
train_boost_t5protchem_raw_uniqueonly_seed42_unbalanced.py) and plots
predicted vs. true pKd for the test split as two SEPARATE panels, one for
CPI rows and one for PPI rows, each with its own y=x reference line and
Pearson r / RMSE.
"""
import importlib.util
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from scipy.stats import pearsonr
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
REPO = "/mnt/hdd/tomita/PPI_CPI_prediction/T5ProtChem_XGBoost_PPI_and_CPI_affinity_prediction"
TEST_CSV = f"{REPO}/data/split_pure_random_uniqueonly_seed42_test.csv"
XGB_MODEL_PATH = f"{REPO}/results/xgb_model.json"
OUTPUT_PNG = f"{REPO}/results/test_scatter.png"
PROTEIN_MAX_LENGTH = 768
DRUG_MAX_LENGTH = 768

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Constructing RAW (use_lora=False) T5ProtChem-native encoder from {T5_CHECKPOINT_PATH} ...")
model = um_cp.ContactPretrainModel(t5_checkpoint_path=T5_CHECKPOINT_PATH, use_lora=False)
model.to(device)
model.eval()

dm = ud.ContactPretrainDataModule(
    train_csv_path=TEST_CSV, val_csv_path=TEST_CSV, test_csv_path=TEST_CSV,
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


df = pd.read_csv(TEST_CSV)
origins_set = ud.CPI_ORIGINS | ud.PPI_ORIGINS
relevant = df[df["data_origin"].isin(origins_set) & df["pKd"].notna()]
single_chain = ~(relevant["molA"].str.contains(ud.CHAIN_SEP, regex=False)
                 | relevant["molB"].str.contains(ud.CHAIN_SEP, regex=False))
relevant = relevant[single_chain].reset_index(drop=True)
print(f"Test rows: {len(relevant)}")

X = np.zeros((len(relevant), 1280), dtype=np.float32)
for i, (molA, molB, fa, fb) in enumerate(zip(relevant["molA"], relevant["molB"],
                                             relevant["molA_format"], relevant["molB_format"])):
    molA_text = encode_side(molA, fa)
    molB_text = encode_side(molB, fb)
    X[i] = mean_pool_feature(molA_text, molB_text)
    if (i + 1) % 1000 == 0:
        print(f"  ... {i + 1}/{len(relevant)}")

y_true = relevant["pKd"].astype("float32").to_numpy()
origins = relevant["data_origin"].to_numpy()

reg = xgb.XGBRegressor()
reg.load_model(XGB_MODEL_PATH)
y_pred = reg.predict(X)

r, p = pearsonr(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred) ** 0.5
print(f"overall: pearson r={r:.4f} (p={p:.2e}), rmse={rmse:.4f}")

is_cpi = np.isin(origins, list(ud.CPI_ORIGINS))
is_ppi = np.isin(origins, list(ud.PPI_ORIGINS))

lo = min(y_true.min(), y_pred.min())
hi = max(y_true.max(), y_pred.max())

fig, (ax_cpi, ax_ppi) = plt.subplots(1, 2, figsize=(12, 6))

for ax, mask, name, color in [
    (ax_cpi, is_cpi, "CPI", "tab:blue"),
    (ax_ppi, is_ppi, "PPI", "tab:orange"),
]:
    yt, yp = y_true[mask], y_pred[mask]
    r_sub, _ = pearsonr(yt, yp)
    rmse_sub = mean_squared_error(yt, yp) ** 0.5
    ax.scatter(yt, yp, s=10, alpha=0.4, color=color)
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
    ax.set_xlabel("True pKd")
    ax.set_ylabel("Predicted pKd")
    ax.set_title(f"{name} (n={mask.sum()}): Pearson r={r_sub:.3f}, RMSE={rmse_sub:.3f}")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_aspect("equal", adjustable="box")

fig.tight_layout()
fig.savefig(OUTPUT_PNG, dpi=150)
print(f"Saved scatter plot to {OUTPUT_PNG}")
