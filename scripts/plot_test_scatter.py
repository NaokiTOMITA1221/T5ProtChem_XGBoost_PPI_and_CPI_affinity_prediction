"""
Predicted vs. true pKd for the combined model's test
split, as two SEPARATE panels -- one for CPI rows, one for PPI rows -- each
with its own y=x reference line and Pearson r / RMSE. Uses only the
FULL-SUM prediction (no contact restriction). Reuses the frozen-encoder
test features cached by extract_base_features.py.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO, "outputs")
CACHE_PT = os.path.join(OUTPUT_DIR, "features_raw_concat_tokens_cache.pt")
CKPT_PATH = os.path.join(OUTPUT_DIR, "mlp_balanced.pt")
OUT_PNG = os.path.join(REPO, "results", "test_scatter.png")
CPI_ORIGINS = {"BindingDB", "PDB_bind_PL"}
PPI_ORIGINS = {"PDB_bind_PP", "PPB_affinity"}

device = "cuda" if torch.cuda.is_available() else "cpu"


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


print(f"Loading test features from {CACHE_PT} ...")
cache = torch.load(CACHE_PT, map_location="cpu", weights_only=False)
test_feats, y_true, origins = cache["test_feats"], cache["y_test"], cache["origins_test"]
print(f"Test rows: {len(test_feats)}")

ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
net = TransformerResidualSumMLP(ckpt["d_model"], ckpt["nhead"], ckpt["num_layers"], ckpt["dim_feedforward"],
                                ckpt["transformer_dropout"], ckpt["mlp_hidden_dims"], ckpt["mlp_dropout"]).to(device)
net.load_state_dict(ckpt["state_dict"])
net.eval()

with torch.no_grad():
    y_pred = np.array([net(t.to(device=device, dtype=torch.float32)).item() for t in test_feats])

r, p = pearsonr(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred) ** 0.5
print(f"overall: pearson r={r:.4f} (p={p:.2e}), rmse={rmse:.4f}")

is_cpi = np.isin(origins, list(CPI_ORIGINS))
is_ppi = np.isin(origins, list(PPI_ORIGINS))

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

fig.suptitle("Combined model -- test set")
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150)
print(f"Saved scatter plot to {OUT_PNG}")
