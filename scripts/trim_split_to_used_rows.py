"""
Trims the FULL split_pure_random_uniqueonly_{train,val,test}.csv (the direct
output of make_pure_random_split_uniqueonly.py, which still contains
non-CPI/PPI-origin rows and multi-chain rows) down to exactly the rows
actually used for training/evaluation by this repo's combined model:
single-chain, CPI/PPI-origin rows with a real (non-null) pKd.

This is the exact eligibility filter used throughout this repo's training/
prediction/plotting scripts (see e.g. train_boost_t5protchem_raw_uniqueonly_
quadsplice.py's load_rows()). Run this AFTER make_pure_random_split_
uniqueonly.py to (re)produce the CSVs checked into data/.
"""
import os

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = os.path.join(os.path.dirname(BASE), "data", "Affinity_data_culation", "integrated_data_csv")
OUT_DIR = os.path.join(BASE, "data")

CPI_ORIGINS = {"BindingDB", "PDB_bind_PL"}
PPI_ORIGINS = {"PDB_bind_PP", "PPB_affinity"}
CHAIN_SEP = ":"


def trim(csv_path):
    df = pd.read_csv(csv_path)
    origins_set = CPI_ORIGINS | PPI_ORIGINS
    relevant = df[df["data_origin"].isin(origins_set) & df["pKd"].notna()]
    single_chain = ~(relevant["molA"].str.contains(CHAIN_SEP, regex=False)
                     | relevant["molB"].str.contains(CHAIN_SEP, regex=False))
    n_dropped_origin = len(df) - len(relevant)
    n_dropped_multichain = int((~single_chain).sum())
    trimmed = relevant[single_chain].reset_index(drop=True)
    print(f"  {os.path.basename(csv_path)}: {len(df)} -> {len(trimmed)} rows "
         f"({n_dropped_origin} non-CPI/PPI-origin or no-pKd dropped, "
         f"{n_dropped_multichain} multi-chain dropped)")
    return trimmed


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for partition in ("train", "val", "test"):
        src = os.path.join(SOURCE_DIR, f"split_pure_random_uniqueonly_{partition}.csv")
        trimmed = trim(src)
        out_path = os.path.join(OUT_DIR, f"split_pure_random_uniqueonly_{partition}.csv")
        trimmed.to_csv(out_path, index=False)
        print(f"    -> saved to {out_path}")


if __name__ == "__main__":
    main()
