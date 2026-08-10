"""
Step 1 of the current (Transformer-residual + per-token-MLP-sum) pipeline:
encodes every row of the repo's fixed pure-random-uniqueonly split with the
RAW, frozen T5ProtChem encoder (use_lora=False -- no fine-tuning at all),
concatenating molA's and molB's token embeddings into one per-row sequence,
and caches the result to disk. This cache is reused, unmodified, by every
downstream script (contact-mask extraction, PPI splice augmentation,
training) -- since the encoder is frozen, a row's embedding never changes
regardless of which script consumes it later.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

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
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO, "outputs")
CACHE_PT = os.path.join(OUTPUT_DIR, "features_raw_concat_tokens_cache.pt")
PROTEIN_MAX_LENGTH = 2000
DRUG_MAX_LENGTH = 768

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Constructing RAW (use_lora=False, frozen) T5ProtChem encoder from {T5_CHECKPOINT_PATH} ...")
model = um_cp.ContactPretrainModel(t5_checkpoint_path=T5_CHECKPOINT_PATH, use_lora=False)
model.to(device)
model.eval()

dm = ud.ContactPretrainDataModule(
    train_csv_path=f"{REPO}/data/split_pure_random_uniqueonly_train.csv",
    val_csv_path=f"{REPO}/data/split_pure_random_uniqueonly_val.csv",
    test_csv_path=f"{REPO}/data/split_pure_random_uniqueonly_test.csv",
    vocab_file=VOCAB_FILE)
protein_tokenizer, drug_tokenizer, text_encoder = dm.protein_tokenizer, dm.drug_tokenizer, dm.text_encoder


@torch.no_grad()
def encode_concat(molA, molB, fa, fb):
    textA = text_encoder.encode_aa(molA) if fa == "AA" else text_encoder.encode_smiles(molA)
    textB = text_encoder.encode_aa(molB) if fb == "AA" else text_encoder.encode_smiles(molB)
    tokA = protein_tokenizer if fa == "AA" else drug_tokenizer
    tokB = protein_tokenizer if fb == "AA" else drug_tokenizer
    maxA = PROTEIN_MAX_LENGTH if fa == "AA" else DRUG_MAX_LENGTH
    maxB = PROTEIN_MAX_LENGTH if fb == "AA" else DRUG_MAX_LENGTH
    a_enc = tokA(textA, truncation=True, max_length=maxA, return_tensors="pt")
    b_enc = tokB(textB, truncation=True, max_length=maxB, return_tensors="pt")
    a_ids, a_mask = a_enc["input_ids"].to(device), a_enc["attention_mask"].to(device)
    b_ids, b_mask = b_enc["input_ids"].to(device), b_enc["attention_mask"].to(device)
    h_a = model.encoder(a_ids, a_mask)[0]
    h_b = model.encoder(b_ids, b_mask)[0]
    La = int(a_mask[0].sum().item())
    Lb = int(b_mask[0].sum().item())
    combined = torch.cat([h_a[:La], h_b[:Lb]], dim=0)
    return combined.half().cpu()


def load_rows(csv_path):
    df = pd.read_csv(csv_path)
    relevant = df[df["pKd"].notna()]
    single_chain = ~(relevant["molA"].str.contains(ud.CHAIN_SEP, regex=False)
                     | relevant["molB"].str.contains(ud.CHAIN_SEP, regex=False))
    return relevant[single_chain].reset_index(drop=True)


def extract(df, name):
    feats, y_list, origins_list = [], [], []
    for i, row in df.iterrows():
        molA, molB, fa, fb, origin = row["molA"], row["molB"], row["molA_format"], row["molB_format"], row["data_origin"]
        combined = encode_concat(molA, molB, fa, fb)
        feats.append(combined)
        y_list.append(float(row["pKd"]))
        origins_list.append(origin)
        if len(feats) % 2000 == 0:
            print(f"    ... {name}: {len(feats)}/{len(df)}")
    y = np.array(y_list, dtype=np.float32)
    origins = np.array(origins_list)
    return feats, y, origins


print("Loading splits (full pool, all origins) ...")
train_df = load_rows(f"{REPO}/data/split_pure_random_uniqueonly_train.csv")
val_df = load_rows(f"{REPO}/data/split_pure_random_uniqueonly_val.csv")
test_df = load_rows(f"{REPO}/data/split_pure_random_uniqueonly_test.csv")
print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

print("Extracting raw-concat frozen-encoder token features (train) ...")
train_feats, y_train, origins_train = extract(train_df, "train")
print("Extracting raw-concat frozen-encoder token features (val) ...")
val_feats, y_val, origins_val = extract(val_df, "val")
print("Extracting raw-concat frozen-encoder token features (test) ...")
test_feats, y_test, origins_test = extract(test_df, "test")

print(f"Saving cache to {CACHE_PT} ...")
torch.save({"train_feats": train_feats, "y_train": y_train, "origins_train": origins_train,
            "val_feats": val_feats, "y_val": y_val, "origins_val": origins_val,
            "test_feats": test_feats, "y_test": y_test, "origins_test": origins_test}, CACHE_PT)
print("Done.")
