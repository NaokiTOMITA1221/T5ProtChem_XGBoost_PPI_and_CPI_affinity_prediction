"""
Step 2: builds a per-row contact-token boolean mask aligned to the SAME
full-token concatenated sequence cached by extract_base_features.py
(train_feats/val_feats/test_feats). Only tokenization (offsets) is needed
here -- NOT the encoder -- since the embeddings already exist; this is
CPU-only and fast.

For rows whose origin is PDB_bind_PL/PDB_bind_PP AND has a usable 4.5A
contact lookup (from pdb_bind_PL/PP_contacts.csv, produced upstream by
extract_pl_contacts.py / extract_pp_contacts.py -- not included in this
repo), a boolean mask (True at contact-marked token positions) is stored;
all other rows get mask=None. These masks are what let the contact-
consistency term (see train_balanced.py) restrict its second pKd estimate
to only the contact-involved tokens.
"""
import os
import sys

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

VOCAB_FILE = "/mnt/hdd/tomita/PPI_CPI_prediction/T5ProtChem/src/vocab/style2.json"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = "/mnt/hdd/tomita/PPI_CPI_prediction/data/Affinity_data_culation"
OUTPUT_DIR = os.path.join(REPO, "outputs")
FULL_CACHE_PT = os.path.join(OUTPUT_DIR, "features_raw_concat_tokens_cache.pt")
OUT_PT = os.path.join(OUTPUT_DIR, "contact_masks_for_full_tokens.pt")
PROTEIN_MAX_LENGTH = 2000
DRUG_MAX_LENGTH = 768

print(f"Loading full-token cache (for length verification) from {FULL_CACHE_PT} ...")
full_cache = torch.load(FULL_CACHE_PT, map_location="cpu", weights_only=False)

dm = ud.ContactPretrainDataModule(
    train_csv_path=f"{REPO}/data/split_pure_random_uniqueonly_val.csv",
    val_csv_path=f"{REPO}/data/split_pure_random_uniqueonly_val.csv",
    test_csv_path=f"{REPO}/data/split_pure_random_uniqueonly_test.csv",
    vocab_file=VOCAB_FILE)
protein_tokenizer, drug_tokenizer, text_encoder = dm.protein_tokenizer, dm.drug_tokenizer, dm.text_encoder

print("Loading contact CSVs ...")
pl_contacts = pd.read_csv(f"{DATA_DIR}/PDB_bind/csv/pdb_bind_PL_contacts.csv")
pp_contacts = pd.read_csv(f"{DATA_DIR}/PDB_bind/csv/pdb_bind_PP_contacts.csv")


def parse_idx_list(s):
    if pd.isna(s) or s == "":
        return []
    return [int(x) for x in str(s).split(";") if x != ""]


def parse_span_list(s):
    if pd.isna(s) or s == "":
        return []
    out = []
    for tok in str(s).split(";"):
        if not tok:
            continue
        a, b = tok.split("-")
        out.append((int(a), int(b)))
    return out


pl_contacts["_res_idx"] = pl_contacts["contact_residue_indices"].apply(parse_idx_list)
pl_contacts["_span_idx"] = pl_contacts["contact_atom_char_spans"].apply(parse_span_list)
pl_lookup = {}
for _, row in pl_contacts.iterrows():
    pl_lookup[(row["sequence"], row["smiles"])] = (row["_res_idx"], row["_span_idx"])

pp_contacts["_res_idx_a"] = pp_contacts["contact_residue_indices_a"].apply(parse_idx_list)
pp_contacts["_res_idx_b"] = pp_contacts["contact_residue_indices_b"].apply(parse_idx_list)
pp_lookup = {}
for _, row in pp_contacts.iterrows():
    pp_lookup[(row["sequence_a"], row["sequence_b"])] = (row["_res_idx_a"], row["_res_idx_b"])
    pp_lookup[(row["sequence_b"], row["sequence_a"])] = (row["_res_idx_b"], row["_res_idx_a"])


def contact_mask_side(offsets, is_protein, contact_info):
    if is_protein:
        contact_set = set(contact_info)
        return [off[1] - off[0] > 0 and (off[0] // 4) in contact_set for off in offsets]
    else:
        spans = contact_info
        return [any(s <= off[0] < e for s, e in spans) for off in offsets]


def get_offsets(text, fmt):
    tok = protein_tokenizer if fmt == "AA" else drug_tokenizer
    maxlen = PROTEIN_MAX_LENGTH if fmt == "AA" else DRUG_MAX_LENGTH
    enc = tok(text, truncation=True, max_length=maxlen, return_tensors="pt", return_offsets_mapping=True)
    L = int(enc["attention_mask"][0].sum().item())
    return enc["offset_mapping"][0].tolist()[:L], L


def load_rows(csv_path):
    df = pd.read_csv(csv_path)
    relevant = df[df["pKd"].notna()]
    single_chain = ~(relevant["molA"].str.contains(ud.CHAIN_SEP, regex=False)
                     | relevant["molB"].str.contains(ud.CHAIN_SEP, regex=False))
    return relevant[single_chain].reset_index(drop=True)


def build_masks(df, feats, name):
    assert len(df) == len(feats), f"{name}: row count mismatch {len(df)} vs {len(feats)}"
    masks = []
    n_with_mask = 0
    for i, row in df.iterrows():
        molA, molB, fa, fb, origin = row["molA"], row["molB"], row["molA_format"], row["molB_format"], row["data_origin"]
        mask = None
        if origin == "PDB_bind_PL":
            seq = molB if fa == "SMILES" else molA
            smi = molA if fa == "SMILES" else molB
            key = (seq, smi)
            if key in pl_lookup:
                res_idx, span_idx = pl_lookup[key]
                textA = text_encoder.encode_aa(molA) if fa == "AA" else text_encoder.encode_smiles(molA)
                textB = text_encoder.encode_aa(molB) if fb == "AA" else text_encoder.encode_smiles(molB)
                offA, La = get_offsets(textA, fa)
                offB, Lb = get_offsets(textB, fb)
                contact_a = res_idx if fa == "AA" else span_idx
                contact_b = res_idx if fb == "AA" else span_idx
                mask_a = contact_mask_side(offA, fa == "AA", contact_a)
                mask_b = contact_mask_side(offB, fb == "AA", contact_b)
                combined_mask = mask_a + mask_b
                if len(combined_mask) == feats[i].shape[0] and any(combined_mask):
                    mask = torch.tensor(combined_mask, dtype=torch.bool)
        elif origin == "PDB_bind_PP":
            key = (molA, molB)
            if key in pp_lookup:
                res_idx_a, res_idx_b = pp_lookup[key]
                textA = text_encoder.encode_aa(molA)
                textB = text_encoder.encode_aa(molB)
                offA, La = get_offsets(textA, "AA")
                offB, Lb = get_offsets(textB, "AA")
                mask_a = contact_mask_side(offA, True, res_idx_a)
                mask_b = contact_mask_side(offB, True, res_idx_b)
                combined_mask = mask_a + mask_b
                if len(combined_mask) == feats[i].shape[0] and any(combined_mask):
                    mask = torch.tensor(combined_mask, dtype=torch.bool)
        masks.append(mask)
        if mask is not None:
            n_with_mask += 1
        if (i + 1) % 5000 == 0:
            print(f"    ... {name}: {i + 1}/{len(df)} scanned, {n_with_mask} with usable contact mask so far")
    print(f"  {name}: {n_with_mask}/{len(df)} rows have a usable contact mask")
    return masks


print("Loading splits (full pool, all origins, same filter as feature cache) ...")
train_df = load_rows(f"{REPO}/data/split_pure_random_uniqueonly_train.csv")
val_df = load_rows(f"{REPO}/data/split_pure_random_uniqueonly_val.csv")
test_df = load_rows(f"{REPO}/data/split_pure_random_uniqueonly_test.csv")

print("Building contact masks (train) ...")
train_masks = build_masks(train_df, full_cache["train_feats"], "train")
print("Building contact masks (val) ...")
val_masks = build_masks(val_df, full_cache["val_feats"], "val")
print("Building contact masks (test) ...")
test_masks = build_masks(test_df, full_cache["test_feats"], "test")

print(f"Saving to {OUT_PT} ...")
torch.save({"train_masks": train_masks, "val_masks": val_masks, "test_masks": test_masks}, OUT_PT)
print("Done.")
