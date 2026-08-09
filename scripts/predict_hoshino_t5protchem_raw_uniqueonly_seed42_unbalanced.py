"""
Applies this repo's featured model
(train_boost_t5protchem_raw_uniqueonly_seed42_unbalanced.py) to every
protein-compound pair in the external Hoshino_polymer validation dataset.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

sys.path.insert(0, "/mnt/hdd/tomita/PPI_CPI_prediction/T5VQBoost/src")
import unified_dataset as ud

_spec = importlib.util.spec_from_file_location(
    "t5contactpretrain_unified_model", "/mnt/hdd/tomita/PPI_CPI_prediction/T5ContactPretrain/src/unified_model.py")
um_cp = importlib.util.module_from_spec(_spec)
sys.path.insert(0, "/mnt/hdd/tomita/PPI_CPI_prediction/T5ContactPretrain/src")
sys.modules["t5contactpretrain_unified_model"] = um_cp
_spec.loader.exec_module(um_cp)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T5_CHECKPOINT_PATH = "/mnt/hdd/tomita/PPI_CPI_prediction/T5ProtChem/weights/Lightning_weights/Pretrained/T5ProtChem/model.pt"
VOCAB_FILE = "/mnt/hdd/tomita/PPI_CPI_prediction/T5ProtChem/src/vocab/style2.json"
XGB_MODEL_PATH = os.path.join(REPO, "results/xgb_model.json")
INPUT_CSV = "/home/tomita/Hoshino_polymer/predicted_pKa_KanM.csv"
OUTPUT_CSV = os.path.join(REPO, "results/predicted_pKa_KanM_T5ProtChem_raw_uniqueonly_seed42_unbalanced.csv")
PRED_COL = "predicted_pKd_T5ProtChem_raw_uniqueonly_seed42_unbalanced"
PROTEIN_MAX_LENGTH = 768
DRUG_MAX_LENGTH = 768

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Constructing RAW (use_lora=False) T5ProtChem-native encoder from {T5_CHECKPOINT_PATH} ...")
model = um_cp.ContactPretrainModel(t5_checkpoint_path=T5_CHECKPOINT_PATH, use_lora=False)
model.to(device)
model.eval()

dm = ud.ContactPretrainDataModule(
    train_csv_path=INPUT_CSV, val_csv_path=INPUT_CSV, test_csv_path=INPUT_CSV,
    vocab_file=VOCAB_FILE)
protein_tokenizer, drug_tokenizer, text_encoder = dm.protein_tokenizer, dm.drug_tokenizer, dm.text_encoder

reg = xgb.XGBRegressor()
reg.load_model(XGB_MODEL_PATH)

df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} rows from {INPUT_CSV}")


@torch.no_grad()
def extract_feature(protein_seq, smiles):
    a_enc = protein_tokenizer(text_encoder.encode_aa(protein_seq), truncation=True,
                              max_length=PROTEIN_MAX_LENGTH, return_tensors="pt")
    b_enc = drug_tokenizer(text_encoder.encode_smiles(smiles), truncation=True,
                           max_length=DRUG_MAX_LENGTH, return_tensors="pt")
    a_ids, a_mask = a_enc["input_ids"].to(device), a_enc["attention_mask"].to(device)
    b_ids, b_mask = b_enc["input_ids"].to(device), b_enc["attention_mask"].to(device)
    h_a = model.encoder(a_ids, a_mask)
    h_b = model.encoder(b_ids, b_mask)
    mean_a = h_a[0].mean(dim=0)
    mean_b = h_b[0].mean(dim=0)
    return torch.cat([mean_a, mean_b]).cpu().numpy()


X = np.stack([extract_feature(seq, smi) for seq, smi in zip(df["target_seq"], df["drug_smiles"])])
df[PRED_COL] = reg.predict(X)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved predictions to {OUTPUT_CSV}")
