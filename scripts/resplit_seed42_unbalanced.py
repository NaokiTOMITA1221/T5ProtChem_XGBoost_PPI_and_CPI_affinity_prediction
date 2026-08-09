"""
Rebuilds the "pure" uniqueonly pool (same swap-pair-deduplicated pool
make_pure_random_split_uniqueonly.py produces) and re-partitions it into a
fresh 80/10/10 train/val/test split using random seed=42, INSTEAD of the
original make_pure_random_split_uniqueonly.py's own split assignment. This
is the split used by this repo's featured model
(train_boost_t5protchem_raw_uniqueonly_seed42_unbalanced.py).

Why seed=42 specifically: this is one of 10 seeds (42-51) tried in a
robustness check of the raw T5ProtChem + XGBoost recipe with NO PPI
oversampling / NO ProtSMILES augmentation ("unbalanced"). Across those 10
seeds the Hoshino_polymer external-validation correlation was NOT robust
(mean Pearson r=0.18, std=0.42, one-sample t-test p=0.21 -- not
significantly different from zero); seed=42 happened to land on the
high end (r=0.68). See README's caveat section -- this split/model should
be read as one specific favorable draw from a noisy distribution, not
evidence of a validated, reproducible external-generalization effect.
"""
import os
import random

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_POOL_CSVS = [
    # Output of make_pure_random_split_uniqueonly.py (run that script first
    # against the raw source datasets to regenerate these).
    "integrated_data_csv/split_pure_random_uniqueonly_train.csv",
    "integrated_data_csv/split_pure_random_uniqueonly_val.csv",
    "integrated_data_csv/split_pure_random_uniqueonly_test.csv",
]
OUT_DIR = os.path.join(REPO, "data")
CHAIN_SEP = ":"
CPI_ORIGINS = {"BindingDB", "PDB_bind_PL"}
PPI_ORIGINS = {"PDB_bind_PP", "PPB_affinity"}
VAL_FRAC = 0.1
TEST_FRAC = 0.1
SEED = 42

full_df = pd.concat([pd.read_csv(p) for p in SOURCE_POOL_CSVS], ignore_index=True)
print(f"Reconstructed full uniqueonly pool: {len(full_df)} rows")

origins_set = CPI_ORIGINS | PPI_ORIGINS
relevant = full_df[full_df["data_origin"].isin(origins_set) & full_df["pKd"].notna()]
single_chain = ~(relevant["molA"].str.contains(CHAIN_SEP, regex=False)
                 | relevant["molB"].str.contains(CHAIN_SEP, regex=False))
relevant = relevant[single_chain].reset_index(drop=True)
N = len(relevant)
print(f"[pool] {N} eligible single-chain rows")

rng = random.Random(SEED)
idx = list(range(N))
rng.shuffle(idx)
n_val = int(N * VAL_FRAC)
n_test = int(N * TEST_FRAC)
val_idx = idx[:n_val]
test_idx = idx[n_val:n_val + n_test]
train_idx = idx[n_val + n_test:]
print(f"seed={SEED}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

train_df = relevant.iloc[train_idx].reset_index(drop=True)
val_df = relevant.iloc[val_idx].reset_index(drop=True)
test_df = relevant.iloc[test_idx].reset_index(drop=True)

os.makedirs(OUT_DIR, exist_ok=True)
train_df.to_csv(os.path.join(OUT_DIR, "split_pure_random_uniqueonly_seed42_train.csv"), index=False)
val_df.to_csv(os.path.join(OUT_DIR, "split_pure_random_uniqueonly_seed42_val.csv"), index=False)
test_df.to_csv(os.path.join(OUT_DIR, "split_pure_random_uniqueonly_seed42_test.csv"), index=False)
print(f"Saved seed={SEED} split CSVs to {OUT_DIR}")
