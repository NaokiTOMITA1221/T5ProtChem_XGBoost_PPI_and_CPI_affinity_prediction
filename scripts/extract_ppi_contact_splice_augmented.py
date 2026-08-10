"""
PPI data augmentation (oversampling) for PDB_bind_PP train rows, using the
project's established "ProtSMILES splice" mechanism (replace a subset of
residues' <P>X token with that residue's own free-amino-acid SMILES
fragment) -- but with residue SELECTION biased toward the row's own 4.5A
CONTACT residues (from pdb_bind_PP_contacts.csv) instead of uniformly random
residues, per user's request that "the splice into SMILES should occur at
the contact location."

Only PDB_bind_PP rows have real contact data (PPB_affinity does not), so
only that origin is augmented here. UNLIKE the project's existing random-
splice augmentation (which never carries contact labels, since contact
positions can't be realigned onto arbitrarily-spliced text), here the
spliced residues themselves ARE contact residues by construction -- so we
track their character spans in the spliced text and, after tokenization,
mark every token overlapping those spans as a contact token. Each augmented
row therefore DOES carry a usable contact mask, letting it contribute to
the CONTACT-SUM loss too.

Oversample factor uses the same formula as the project's other "quadsplice"
augmentation scripts: factor = round(target_ratio * cpi_count / ppi_count),
capped at MAX_FACTOR.
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
OUT_PT = f"{OUTPUT_DIR}/features_ppi_contact_splice_augmented_cache.pt"
PROTEIN_MAX_LENGTH = 2000
CPI_ORIGINS = {"BindingDB", "PDB_bind_PL"}
PPI_ORIGINS = {"PPB_affinity", "PDB_bind_PP"}
TARGET_RATIO = 1.0
MAX_FACTOR = 20
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

print("Loading PP contact CSV ...")
pp_contacts = pd.read_csv(f"{DATA_DIR}/PDB_bind/csv/pdb_bind_PP_contacts.csv")


def parse_idx_list(s):
    if pd.isna(s) or s == "":
        return []
    return [int(x) for x in str(s).split(";") if x != ""]


pp_contacts["_res_idx_a"] = pp_contacts["contact_residue_indices_a"].apply(parse_idx_list)
pp_contacts["_res_idx_b"] = pp_contacts["contact_residue_indices_b"].apply(parse_idx_list)
pp_lookup = {}
for _, row in pp_contacts.iterrows():
    pp_lookup[(row["sequence_a"], row["sequence_b"])] = (row["_res_idx_a"], row["_res_idx_b"])
    pp_lookup[(row["sequence_b"], row["sequence_a"])] = (row["_res_idx_b"], row["_res_idx_a"])

_AA_SMILES_CACHE = {}


def aa_to_smiles(residue):
    if residue not in _AA_SMILES_CACHE:
        mol = Chem.MolFromSequence(residue)
        _AA_SMILES_CACHE[residue] = Chem.MolToSmiles(mol) if mol is not None else None
    return _AA_SMILES_CACHE[residue]


def splice_contact_biased_with_spans(seq, contact_idx, min_residues, max_residues):
    """Splices a subset of contact residues into SMILES form (for input
    diversity), but returns the char span of EVERY contact residue --
    spliced or not -- in the assembled text, so the contact mask can mark
    ALL contact-involved tokens, not just the ones that got spliced. Once
    any residue is spliced, subsequent residues no longer sit at fixed
    4-char offsets, so spans must be tracked incrementally rather than
    computed via off[0]//4."""
    n = len(seq)
    if n == 0:
        return "", []
    contact_set = set(contact_idx)
    pool = list(contact_idx) if contact_idx else list(range(n))
    k = min(rng.randint(min_residues, max_residues), len(pool))
    chosen = set(rng.sample(pool, k)) if k > 0 else set()
    parts = []
    residue_spans = []
    pos = 0
    for i, residue in enumerate(seq):
        if i in chosen:
            smi = aa_to_smiles(residue)
            frag = smi if smi is not None else f"<P>{residue}"
        else:
            frag = f"<P>{residue}"
        parts.append(frag)
        residue_spans.append((pos, pos + len(frag)))
        pos += len(frag)
    contact_spans = [residue_spans[i] for i in contact_set if i < len(residue_spans)]
    return "".join(parts), contact_spans


@torch.no_grad()
def encode_spliced_concat_with_mask(splicedA, spansA, splicedB, spansB):
    a_enc = protein_tokenizer(splicedA, truncation=True, max_length=PROTEIN_MAX_LENGTH,
                              return_tensors="pt", return_offsets_mapping=True)
    b_enc = protein_tokenizer(splicedB, truncation=True, max_length=PROTEIN_MAX_LENGTH,
                              return_tensors="pt", return_offsets_mapping=True)
    a_ids, a_mask = a_enc["input_ids"].to(device), a_enc["attention_mask"].to(device)
    b_ids, b_mask = b_enc["input_ids"].to(device), b_enc["attention_mask"].to(device)
    h_a = model.encoder(a_ids, a_mask)[0]
    h_b = model.encoder(b_ids, b_mask)[0]
    La = int(a_mask[0].sum().item())
    Lb = int(b_mask[0].sum().item())
    combined = torch.cat([h_a[:La], h_b[:Lb]], dim=0).half().cpu()

    off_a = a_enc["offset_mapping"][0].tolist()[:La]
    off_b = b_enc["offset_mapping"][0].tolist()[:Lb]
    mask_a = [any(s <= off[0] < e for s, e in spansA) for off in off_a]
    mask_b = [any(s <= off[0] < e for s, e in spansB) for off in off_b]
    combined_mask = torch.tensor(mask_a + mask_b, dtype=torch.bool)
    return combined, combined_mask


def load_rows(csv_path):
    df = pd.read_csv(csv_path)
    relevant = df[df["pKd"].notna()]
    single_chain = ~(relevant["molA"].str.contains(ud.CHAIN_SEP, regex=False)
                     | relevant["molB"].str.contains(ud.CHAIN_SEP, regex=False))
    return relevant[single_chain].reset_index(drop=True)


print("Loading train split (full pool) ...")
train_df = load_rows(f"{REPO}/data/split_pure_random_uniqueonly_train.csv")
cpi_count = int(train_df["data_origin"].isin(CPI_ORIGINS).sum())
ppi_count = int(train_df["data_origin"].isin(PPI_ORIGINS).sum())
pp_df = train_df[train_df["data_origin"] == "PDB_bind_PP"].reset_index(drop=True)
pp_with_contact = pp_df[pp_df.apply(lambda r: (r["molA"], r["molB"]) in pp_lookup, axis=1)].reset_index(drop=True)

extra_needed = TARGET_RATIO * cpi_count - ppi_count
factor = max(round(extra_needed / ppi_count), 0) if ppi_count > 0 else 0
factor = min(factor, MAX_FACTOR)
print(f"train CPI={cpi_count} PPI={ppi_count} (all PPI origins) -> factor={factor} (TARGET_RATIO={TARGET_RATIO}, MAX_FACTOR={MAX_FACTOR})")
print(f"PDB_bind_PP train rows: {len(pp_df)} total, {len(pp_with_contact)} with usable contact lookup "
     f"-- only these get contact-biased splice augmentation, factor={factor} copies each "
     f"-> {len(pp_with_contact) * factor} augmented rows")

feats, y_list, masks = [], [], []
n_empty_mask = 0
for copy_idx in range(factor):
    for i, row in pp_with_contact.iterrows():
        res_idx_a, res_idx_b = pp_lookup[(row["molA"], row["molB"])]
        splicedA, spansA = splice_contact_biased_with_spans(row["molA"], res_idx_a, SPLICE_MIN, SPLICE_MAX)
        splicedB, spansB = splice_contact_biased_with_spans(row["molB"], res_idx_b, SPLICE_MIN, SPLICE_MAX)
        combined, mask = encode_spliced_concat_with_mask(splicedA, spansA, splicedB, spansB)
        feats.append(combined)
        jittered = float(row["pKd"]) * (1.0 + rng.uniform(-PKD_NOISE_FRAC, PKD_NOISE_FRAC))
        y_list.append(jittered)
        if mask.any():
            masks.append(mask)
        else:
            masks.append(None)
            n_empty_mask += 1
        if len(feats) % 1000 == 0:
            print(f"    ... {len(feats)}/{len(pp_with_contact) * factor} augmented rows extracted "
                 f"(copy {copy_idx + 1}/{factor})")

y = np.array(y_list, dtype=np.float32)
origins = np.array(["PDB_bind_PP"] * len(feats))
print(f"Total augmented rows: {len(feats)} ({len(feats) - n_empty_mask} with usable contact mask, "
     f"{n_empty_mask} without)")

print(f"Saving to {OUT_PT} ...")
torch.save({"feats": feats, "y": y, "origins": origins, "masks": masks, "factor": factor,
            "n_source_rows": len(pp_with_contact)}, OUT_PT)
print("Done.")
