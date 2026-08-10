"""
Companion to extract_ppi_contact_splice_augmented.py: augments the PPI-
origin train rows that have NO usable contact reference (PPB_affinity, plus
the handful of PDB_bind_PP rows whose contact lookup failed) using the
project's ORIGINAL random-position ProtSMILES splice (uniform residue
selection, since there's no contact info to bias toward), at the SAME
oversample factor as the contact-biased augmentation, so overall PPI and
CPI train totals end up roughly balanced.

Augmented rows have NO contact mask (no contact info exists for this
subset) -- they contribute to the FULL-SUM loss only, same as the project's
established convention for random-splice augmentation.
"""
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

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
DATA_DIR = "/mnt/hdd/tomita/PPI_CPI_prediction/data/Affinity_data_culation"
OUTPUT_DIR = os.path.join(REPO, "outputs")
OUT_PT = f"{OUTPUT_DIR}/features_ppi_random_splice_augmented_nocontact_cache.pt"
PROTEIN_MAX_LENGTH = 2000
PPI_ORIGINS = {"PPB_affinity", "PDB_bind_PP"}
FACTOR = 6  # same multiplier as the contact-biased PPI splice augmentation
SPLICE_MIN, SPLICE_MAX = 3, 8
PKD_NOISE_FRAC = 0.05
AUG_SEED = 12345

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)
rng = random.Random(AUG_SEED)

print(f"Constructing RAW (use_lora=False, frozen) T5ProtChem encoder from {T5_CHECKPOINT_PATH} ...")
model = um_cp.ContactPretrainModel(t5_checkpoint_path=T5_CHECKPOINT_PATH, use_lora=False)
model.to(device)
model.eval()

dm = ud.ContactPretrainDataModule(
    train_csv_path=f"{REPO}/data/split_pure_random_uniqueonly_val.csv",
    val_csv_path=f"{REPO}/data/split_pure_random_uniqueonly_val.csv",
    test_csv_path=f"{REPO}/data/split_pure_random_uniqueonly_test.csv",
    vocab_file=VOCAB_FILE)
protein_tokenizer, text_encoder = dm.protein_tokenizer, dm.text_encoder

print("Loading PP contact CSV (to identify which PDB_bind_PP rows already have contact splice) ...")
pp_contacts = pd.read_csv(f"{DATA_DIR}/PDB_bind/csv/pdb_bind_PP_contacts.csv")
pp_keys = set()
for a, b in zip(pp_contacts["sequence_a"], pp_contacts["sequence_b"]):
    pp_keys.add((a, b))
    pp_keys.add((b, a))

_AA_SMILES_CACHE = {}


def aa_to_smiles(residue):
    if residue not in _AA_SMILES_CACHE:
        mol = Chem.MolFromSequence(residue)
        _AA_SMILES_CACHE[residue] = Chem.MolToSmiles(mol) if mol is not None else None
    return _AA_SMILES_CACHE[residue]


def splice_random(seq, min_residues, max_residues):
    n = len(seq)
    if n == 0:
        return ""
    k = min(rng.randint(min_residues, max_residues), n)
    chosen = set(rng.sample(range(n), k))
    parts = []
    for i, residue in enumerate(seq):
        if i in chosen:
            smi = aa_to_smiles(residue)
            parts.append(smi if smi is not None else f"<P>{residue}")
        else:
            parts.append(f"<P>{residue}")
    return "".join(parts)


@torch.no_grad()
def encode_spliced_concat(splicedA, splicedB):
    a_enc = protein_tokenizer(splicedA, truncation=True, max_length=PROTEIN_MAX_LENGTH, return_tensors="pt")
    b_enc = protein_tokenizer(splicedB, truncation=True, max_length=PROTEIN_MAX_LENGTH, return_tensors="pt")
    a_ids, a_mask = a_enc["input_ids"].to(device), a_enc["attention_mask"].to(device)
    b_ids, b_mask = b_enc["input_ids"].to(device), b_enc["attention_mask"].to(device)
    h_a = model.encoder(a_ids, a_mask)[0]
    h_b = model.encoder(b_ids, b_mask)[0]
    La = int(a_mask[0].sum().item())
    Lb = int(b_mask[0].sum().item())
    return torch.cat([h_a[:La], h_b[:Lb]], dim=0).half().cpu()


def load_rows(csv_path):
    df = pd.read_csv(csv_path)
    relevant = df[df["pKd"].notna()]
    single_chain = ~(relevant["molA"].str.contains(ud.CHAIN_SEP, regex=False)
                     | relevant["molB"].str.contains(ud.CHAIN_SEP, regex=False))
    return relevant[single_chain].reset_index(drop=True)


print("Loading train split (full pool) ...")
train_df = load_rows(f"{REPO}/data/split_pure_random_uniqueonly_train.csv")
ppi_df = train_df[train_df["data_origin"].isin(PPI_ORIGINS)].reset_index(drop=True)


def has_contact(row):
    if row["data_origin"] != "PDB_bind_PP":
        return False
    return (row["molA"], row["molB"]) in pp_keys


ppi_df["_has_contact"] = ppi_df.apply(has_contact, axis=1)
nocontact_df = ppi_df[~ppi_df["_has_contact"]].reset_index(drop=True)
print(f"PPI train rows: {len(ppi_df)} total ({ppi_df['_has_contact'].sum()} with contact -- already augmented "
     f"separately), {len(nocontact_df)} WITHOUT contact -- these get random-splice augmentation, "
     f"factor={FACTOR} copies each -> {len(nocontact_df) * FACTOR} augmented rows")
print(nocontact_df["data_origin"].value_counts().to_dict())

feats, y_list, origins_list = [], [], []
for copy_idx in range(FACTOR):
    for i, row in nocontact_df.iterrows():
        splicedA = splice_random(row["molA"], SPLICE_MIN, SPLICE_MAX)
        splicedB = splice_random(row["molB"], SPLICE_MIN, SPLICE_MAX)
        combined = encode_spliced_concat(splicedA, splicedB)
        feats.append(combined)
        jittered = float(row["pKd"]) * (1.0 + rng.uniform(-PKD_NOISE_FRAC, PKD_NOISE_FRAC))
        y_list.append(jittered)
        origins_list.append(row["data_origin"])
        if len(feats) % 1000 == 0:
            print(f"    ... {len(feats)}/{len(nocontact_df) * FACTOR} augmented rows extracted "
                 f"(copy {copy_idx + 1}/{FACTOR})")

y = np.array(y_list, dtype=np.float32)
origins = np.array(origins_list)
print(f"Total augmented rows: {len(feats)}")

print(f"Saving to {OUT_PT} ...")
torch.save({"feats": feats, "y": y, "origins": origins, "masks": [None] * len(feats), "factor": FACTOR,
            "n_source_rows": len(nocontact_df)}, OUT_PT)
print("Done.")
