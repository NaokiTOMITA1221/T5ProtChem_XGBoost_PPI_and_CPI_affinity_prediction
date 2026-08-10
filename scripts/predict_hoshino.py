"""
Applies the combined model (train_balanced.py's checkpoint, outputs/
mlp_balanced.pt) to the external, out-of-domain Hoshino_polymer
polymer-peptide dataset (INPUT_CSV, not included in this repo -- see
README), using the FULL-SUM prediction (Transformer-residual + per-token-MLP
score summed over the ENTIRE token sequence; no contact restriction). This
is the number the README's External validation section reports.

Correlated against the Figure S24 neutralization-ratio readout at all three
ligand concentrations tested in the source paper's supplementary figure
(0.1/0.3/1.0 mM), pixel-extracted from the supplementary PDF (bar-color
detection + y-axis box-border calibration). p-values are from PERMUTATION
tests (99999 resamples) rather than parametric approximations, since n=15
is too small for those assumptions to be reliable.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from scipy import stats
from scipy.stats import spearmanr

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
CKPT_PATH = os.path.join(OUTPUT_DIR, "mlp_balanced.pt")
INPUT_CSV = "/home/tomita/Hoshino_polymer/predicted_pKa_KanM.csv"
OUTPUT_CSV = os.path.join(REPO, "results", "predicted_pKa_KanM_balanced.csv")
PROTEIN_MAX_LENGTH = 2000
DRUG_MAX_LENGTH = 768
N_RESAMPLES = 99999
PERM_SEED = 42

# Figure S24 neutralization ratios (%), pixel-extracted from the source
# paper's supplementary PDF, one dict per ligand concentration panel.
NEUTRALIZATION_PANELS = {
    "0.1mM": {
        "A2T0": 9.4, "A2T1": 3.6, "A2T2": 16.0, "A2T3": 26.4, "A2T4": 25.2, "A2T5": 31.8,
        "A3T0": -4.7, "A3T1": 32.4, "A3T2": 19.4, "A3T3": 6.8, "A3T4": 21.1,
        "A4T0": 20.0, "A4T1": 21.3, "A4T2": 30.9, "A4T3": 62.3,
    },
    "0.3mM": {
        "A2T0": -9.4, "A2T1": 30.2, "A2T2": 26.6, "A2T3": -9.4, "A2T4": 44.1, "A2T5": 20.1,
        "A3T0": 3.9, "A3T1": 31.9, "A3T2": 42.4, "A3T3": 40.1, "A3T4": 38.8,
        "A4T0": 22.9, "A4T1": 69.0, "A4T2": 58.3, "A4T3": 75.9,
    },
    "1.0mM": {
        "A2T0": 26.1, "A2T1": 43.9, "A2T2": 72.6, "A2T3": 63.4, "A2T4": 90.2, "A2T5": 59.6,
        "A3T0": 40.0, "A3T1": 92.8, "A3T2": 59.6, "A3T3": 71.6, "A3T4": 61.3,
        "A4T0": 86.0, "A4T1": 87.7, "A4T2": 83.1, "A4T3": 87.5,
    },
}

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Constructing RAW (use_lora=False, frozen) T5ProtChem encoder from {T5_CHECKPOINT_PATH} ...")
model = um_cp.ContactPretrainModel(t5_checkpoint_path=T5_CHECKPOINT_PATH, use_lora=False)
model.to(device)
model.eval()

dm = ud.ContactPretrainDataModule(
    train_csv_path=f"{REPO}/data/split_pure_random_uniqueonly_val.csv",
    val_csv_path=f"{REPO}/data/split_pure_random_uniqueonly_val.csv",
    test_csv_path=f"{REPO}/data/split_pure_random_uniqueonly_test.csv",
    vocab_file=VOCAB_FILE)
protein_tokenizer, drug_tokenizer, text_encoder = dm.protein_tokenizer, dm.drug_tokenizer, dm.text_encoder


class TransformerResidualSumMLP(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, transformer_dropout,
                mlp_hidden_dims, mlp_dropout):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=transformer_dropout, batch_first=True, activation="relu")
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        layers = []
        d = d_model
        for h in mlp_hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(mlp_dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, tokens):
        x = tokens.unsqueeze(0)
        trans_out = self.transformer(x)
        h_res = x + trans_out
        per_token = self.mlp(h_res.squeeze(0)).squeeze(-1)
        return per_token.sum()


ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
net = TransformerResidualSumMLP(ckpt["d_model"], ckpt["nhead"], ckpt["num_layers"], ckpt["dim_feedforward"],
                                ckpt["transformer_dropout"], ckpt["mlp_hidden_dims"], ckpt["mlp_dropout"]).to(device)
net.load_state_dict(ckpt["state_dict"])
net.eval()
print(f"Loaded checkpoint (best_epoch={ckpt['best_epoch']})")


@torch.no_grad()
def encode_concat(seq, smi):
    textA = text_encoder.encode_aa(seq)
    textB = text_encoder.encode_smiles(smi)
    a_enc = protein_tokenizer(textA, truncation=True, max_length=PROTEIN_MAX_LENGTH, return_tensors="pt")
    b_enc = drug_tokenizer(textB, truncation=True, max_length=DRUG_MAX_LENGTH, return_tensors="pt")
    a_ids, a_mask = a_enc["input_ids"].to(device), a_enc["attention_mask"].to(device)
    b_ids, b_mask = b_enc["input_ids"].to(device), b_enc["attention_mask"].to(device)
    h_a = model.encoder(a_ids, a_mask)[0]
    h_b = model.encoder(b_ids, b_mask)[0]
    La = int(a_mask[0].sum().item())
    Lb = int(b_mask[0].sum().item())
    return torch.cat([h_a[:La], h_b[:Lb]], dim=0)


df = pd.read_csv(INPUT_CSV)
df["label"] = "A" + df["n"].astype(str) + "T" + df["m"].astype(str)
print(f"Loaded {len(df)} rows from {INPUT_CSV}")

preds = []
with torch.no_grad():
    for seq, raw_smi in zip(df["target_seq"], df["drug_smiles"]):
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(raw_smi))  # canonicalize -- the char-level tokenizer is sensitive to SMILES string form
        tokens = encode_concat(seq, smi)
        preds.append(net(tokens).item())
df["predicted_pKd"] = preds


def pearsonr_perm(x, y):
    method = stats.PermutationMethod(n_resamples=N_RESAMPLES, random_state=np.random.default_rng(PERM_SEED))
    res = stats.pearsonr(x, y, method=method)
    return res.statistic, res.pvalue


def spearmanr_perm(x, y):
    rng = np.random.default_rng(PERM_SEED)
    rho_obs = spearmanr(x, y).statistic
    y_arr = np.asarray(y)
    count = 0
    for _ in range(N_RESAMPLES):
        y_perm = rng.permutation(y_arr)
        rho_perm = spearmanr(x, y_perm).statistic
        if abs(rho_perm) >= abs(rho_obs):
            count += 1
    p = (count + 1) / (N_RESAMPLES + 1)
    return rho_obs, p


results = {}
for panel_name, neut in NEUTRALIZATION_PANELS.items():
    df[f"neutralization_{panel_name}"] = df["label"].map(neut)
    sub = df.dropna(subset=[f"neutralization_{panel_name}"])
    r, p = pearsonr_perm(sub["predicted_pKd"], sub[f"neutralization_{panel_name}"])
    rho, p_rho = spearmanr_perm(sub["predicted_pKd"], sub[f"neutralization_{panel_name}"])
    print(f"[{panel_name}] pearson r={r:.4f} (permutation p={p:.4f}), "
         f"spearman rho={rho:.4f} (permutation p={p_rho:.4f}), n={len(sub)}")
    results[panel_name] = {"pearson": float(r), "pearson_p": float(p), "spearman": float(rho), "spearman_p": float(p_rho)}

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)
with open(os.path.join(REPO, "results", "hoshino_correlation.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved predictions to {OUTPUT_CSV}")
