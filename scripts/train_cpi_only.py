"""
Same architecture/hyperparameters as train_mlp_frozen_transformer_scratch_balanced.py
(frozen T5ProtChem + 1-layer-Transformer-residual + per-token-MLP-summed,
RANDOM INIT), but trained on CPI-ORIGIN rows ONLY (BindingDB + PDB_bind_PL,
24241 rows from the base pool -- CPI has no augmentation in this pipeline,
so this is exactly the CPI subset of the base full-token pool). Contact
masks kept for the PDB_bind_PL structural subset (part of the 9178 base
masked rows).

Evaluated on the FULL (both-domain) val/test as usual, so CPI in-domain vs
PPI cross-domain performance is directly visible via the standard
evaluate() breakdown.
"""
import json

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

import os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO, "outputs")
FULL_CACHE_PT = f"{OUTPUT_DIR}/features_raw_concat_tokens_cache.pt"
MASKS_PT = f"{OUTPUT_DIR}/contact_masks_for_full_tokens.pt"
SEED = 42
D_MODEL = 640
N_HEAD = 8
NUM_LAYERS = 1
DIM_FEEDFORWARD = 1024
TRANSFORMER_DROPOUT = 0.1
MLP_HIDDEN_DIMS = (256, 64)
MLP_DROPOUT = 0.1
LR = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_ROWS = 32
MAX_EPOCHS = 200
PATIENCE = 20

device = "cuda" if torch.cuda.is_available() else "cpu"

CPI_ORIGINS = {"BindingDB", "PDB_bind_PL"}
PPI_ORIGINS = {"PDB_bind_PP", "PPB_affinity"}

print(f"Loading full-token cache from {FULL_CACHE_PT} ...")
full_cache = torch.load(FULL_CACHE_PT, map_location="cpu", weights_only=False)
train_feats_all = full_cache["train_feats"]
y_train_all = full_cache["y_train"]
origins_train_all = full_cache["origins_train"]
val_feats = full_cache["val_feats"]
y_val = full_cache["y_val"]
origins_val = full_cache["origins_val"]
test_feats = full_cache["test_feats"]
y_test = full_cache["y_test"]
origins_test = full_cache["origins_test"]

print(f"Loading contact masks from {MASKS_PT} ...")
masks_cache = torch.load(MASKS_PT, map_location="cpu", weights_only=False)
train_masks_all = masks_cache["train_masks"]
val_masks = masks_cache["val_masks"]
test_masks = masks_cache["test_masks"]

cpi_idx = [i for i, o in enumerate(origins_train_all) if o in CPI_ORIGINS]
train_feats = [train_feats_all[i] for i in cpi_idx]
y_train = y_train_all[cpi_idx]
origins_train = origins_train_all[cpi_idx]
train_masks = [train_masks_all[i] for i in cpi_idx]
n_with_mask = sum(m is not None for m in train_masks)
print(f"CPI-ONLY train pool: {len(train_feats)} rows, {n_with_mask} with usable contact mask (PDB_bind_PL)")

val_cpi_idx = [i for i, o in enumerate(origins_val) if o in CPI_ORIGINS]
val_feats_domain = [val_feats[i] for i in val_cpi_idx]
y_val_domain = y_val[val_cpi_idx]
print(f"Early-stopping val subset restricted to CPI origins: {len(val_feats_domain)}/{len(val_feats)} rows")


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

    def forward(self, tokens, mask=None):
        x = tokens.unsqueeze(0)
        trans_out = self.transformer(x)
        h_res = x + trans_out
        per_token = self.mlp(h_res.squeeze(0)).squeeze(-1)
        full_sum = per_token.sum()
        contact_sum = per_token[mask].sum() if mask is not None else None
        return full_sum, contact_sum


def evaluate(y_true, y_pred, origins):
    groups = {"overall": None, "CPI": CPI_ORIGINS, "PPI": PPI_ORIGINS}
    metrics = {}
    for name, origin_set in groups.items():
        m = np.ones(len(y_true), dtype=bool) if origin_set is None else np.isin(origins, list(origin_set))
        if m.sum() < 2:
            continue
        yt, yp = y_true[m], y_pred[m]
        metrics[f"{name}/pearson"] = float(pearsonr(yt, yp)[0])
        metrics[f"{name}/spearman"] = float(spearmanr(yt, yp)[0])
        metrics[f"{name}/rmse"] = float(mean_squared_error(yt, yp) ** 0.5)
    return metrics


@torch.no_grad()
def predict_full(net, feats):
    return np.array([net(t.to(device=device, dtype=torch.float32))[0].item() for t in feats])


@torch.no_grad()
def predict_contact_subset(net, feats, masks, y, origins):
    idx = [i for i, m in enumerate(masks) if m is not None]
    preds = []
    for i in idx:
        t = feats[i].to(device=device, dtype=torch.float32)
        m = masks[i].to(device=device)
        _, contact_sum = net(t, m)
        preds.append(contact_sum.item())
    preds = np.array(preds)
    return preds, y[idx], origins[idx]


torch.manual_seed(SEED)
np.random.seed(SEED)
net = TransformerResidualSumMLP(D_MODEL, N_HEAD, NUM_LAYERS, DIM_FEEDFORWARD, TRANSFORMER_DROPOUT,
                                MLP_HIDDEN_DIMS, MLP_DROPOUT).to(device)
n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"Trainable params (transformer + per-token MLP, RANDOM INIT): {n_params:,}")

opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loss_fn = nn.MSELoss()
y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)
y_val_domain_t = torch.tensor(y_val_domain, dtype=torch.float32, device=device)

n_train = len(train_feats)
best_val_rmse = float("inf")
best_state = None
best_epoch = 0
patience_ctr = 0

print(f"Training from scratch, CPI-ONLY (max_epochs={MAX_EPOCHS}, patience={PATIENCE}, "
     f"batch_rows={BATCH_ROWS}, lr={LR}) ...")
for epoch in range(MAX_EPOCHS):
    net.train()
    perm = np.random.permutation(n_train)
    opt.zero_grad()
    for step, i in enumerate(perm):
        tokens = train_feats[i].to(device=device, dtype=torch.float32)
        mask = train_masks[i].to(device=device) if train_masks[i] is not None else None
        full_sum, contact_sum = net(tokens, mask)
        loss = loss_fn(full_sum, y_train_t[i])
        if contact_sum is not None:
            loss = loss + loss_fn(contact_sum, y_train_t[i])
        loss = loss / BATCH_ROWS
        loss.backward()
        if (step + 1) % BATCH_ROWS == 0 or step == n_train - 1:
            opt.step()
            opt.zero_grad()

    net.eval()
    with torch.no_grad():
        val_preds = torch.stack([net(t.to(device=device, dtype=torch.float32))[0] for t in val_feats_domain])
        val_rmse = torch.sqrt(loss_fn(val_preds, y_val_domain_t)).item()

    print(f"  epoch {epoch + 1}: val_rmse(CPI-only)={val_rmse:.4f} (best={best_val_rmse:.4f} @ {best_epoch + 1})")

    if val_rmse < best_val_rmse - 1e-5:
        best_val_rmse = val_rmse
        best_state = {k: v.clone() for k, v in net.state_dict().items()}
        best_epoch = epoch
        patience_ctr = 0
    else:
        patience_ctr += 1
    if patience_ctr >= PATIENCE:
        print(f"  early stop at epoch {epoch + 1} (best={best_epoch + 1}, val_rmse={best_val_rmse:.4f})")
        break

net.load_state_dict(best_state)
net.eval()

ckpt_path = f"{OUTPUT_DIR}/mlp_cpi_only.pt"
torch.save({"state_dict": net.state_dict(), "d_model": D_MODEL, "nhead": N_HEAD, "num_layers": NUM_LAYERS,
            "dim_feedforward": DIM_FEEDFORWARD, "transformer_dropout": TRANSFORMER_DROPOUT,
            "mlp_hidden_dims": MLP_HIDDEN_DIMS, "mlp_dropout": MLP_DROPOUT,
            "seed": SEED, "best_epoch": best_epoch + 1}, ckpt_path)

val_pred_full = predict_full(net, val_feats)
test_pred_full = predict_full(net, test_feats)
val_metrics_full = evaluate(y_val, val_pred_full, origins_val)
test_metrics_full = evaluate(y_test, test_pred_full, origins_test)

val_pred_ct, y_val_ct, origins_val_ct = predict_contact_subset(net, val_feats, val_masks, y_val, origins_val)
test_pred_ct, y_test_ct, origins_test_ct = predict_contact_subset(net, test_feats, test_masks, y_test, origins_test)
val_metrics_ct = evaluate(y_val_ct, val_pred_ct, origins_val_ct)
test_metrics_ct = evaluate(y_test_ct, test_pred_ct, origins_test_ct)

print(f"\n[seed={SEED}] FROM SCRATCH, CPI-ONLY, best_epoch={best_epoch + 1}")
print(f"FULL-SUM     val: overall={val_metrics_full['overall/pearson']:.4f}  "
     f"test: overall={test_metrics_full['overall/pearson']:.4f} "
     f"CPI={test_metrics_full['CPI/pearson']:.4f} PPI={test_metrics_full.get('PPI/pearson', float('nan')):.4f}")
print(f"CONTACT-SUM  val: overall={val_metrics_ct['overall/pearson']:.4f}  "
     f"test: overall={test_metrics_ct['overall/pearson']:.4f} "
     f"CPI={test_metrics_ct.get('CPI/pearson', float('nan')):.4f} PPI={test_metrics_ct.get('PPI/pearson', float('nan')):.4f}")

with open(f"{OUTPUT_DIR}/cpi_only_metrics.json", "w") as f:
    json.dump({"best_epoch": best_epoch + 1, "full_val": val_metrics_full, "full_test": test_metrics_full,
               "contact_val": val_metrics_ct, "contact_test": test_metrics_ct}, f, indent=2)
print(f"\nSaved checkpoint and metrics to {OUTPUT_DIR}")
