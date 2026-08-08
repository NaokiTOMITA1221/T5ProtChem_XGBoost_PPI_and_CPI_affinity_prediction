"""
Same "pure" random train/val/test split as make_pure_random_split.py (plain
row-level random shuffle, NO protein-sequence-identity grouping, NO CPI:PPI
balancing), but with a stronger fix than make_pure_random_split_noswapleak.py:
instead of keeping swap-pair duplicates (a row whose (molA,molB) matches
another row's (molB,molA) -- see conversation, 113 such groups found, all
PPB_affinity) together in the same partition, this DEDUPLICATES them BEFORE
splitting -- so the eligible pool going into the split is guaranteed to
contain each underlying interaction exactly once. There is then nothing left
to leak across partitions via this mechanism, by construction.

Dedup mechanic: canonicalize each row's (molA, molB) key by treating
(molA,molB) and (molB,molA) as the same interaction; for each such group,
keep ONE row (first occurrence) with pKd averaged across the group's
member(s), label recomputed -- same averaging convention as cluster_pairs.py's
dedup_within (which only catches EXACT (molA,molB)-order duplicates; this
extends that to catch swap-order duplicates too).

Outputs (integrated_data_csv/): split_pure_random_uniqueonly_{train,val,test}.csv
"""

import os
import random

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "integrated_data_csv")

DATASET_CSVS = {
    "BindingDB":      os.path.join(BASE, "BindingDB_BALM_bench/csv/bindingdb.csv"),
    "Human_Celegans": os.path.join(BASE, "Human_Celegans/csv/human_celegans.csv"),
    "Negatome":       os.path.join(BASE, "Negatome/csv/negatome.csv"),
    "PDB_bind_PL":    os.path.join(BASE, "PDB_bind/csv/pdb_bind_PL.csv"),
    "PDB_bind_PP":    os.path.join(BASE, "PDB_bind/csv/pdb_bind_PP.csv"),
    "PDB_bind_NL":    os.path.join(BASE, "PDB_bind/csv/pdb_bind_NL.csv"),
    "PDB_bind_PN":    os.path.join(BASE, "PDB_bind/csv/pdb_bind_PN.csv"),
    "PPB_affinity":   os.path.join(BASE, "PPB_affinity/csv/ppb_affinity.csv"),
}

CPI_ORIGINS = {"BindingDB", "PDB_bind_PL"}
PPI_ORIGINS = {"PDB_bind_PP", "PPB_affinity"}

SEED = 42
VAL_FRAC = 0.1
TEST_FRAC = 0.1


def has_pkd(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").notna()


def load_all() -> pd.DataFrame:
    """Identical to make_pure_random_split.py's load_all()."""
    df_map = {}
    for origin, path in DATASET_CSVS.items():
        print(f"  Reading {os.path.basename(path)} ...", end=" ", flush=True)
        df = pd.read_csv(path, dtype={"pair_cluster_id": "Int64"})
        df["data_origin"] = origin
        df_map[origin] = df
        print(f"{len(df):,} rows")

    pp_df = df_map.get("PDB_bind_PP")
    ppb_df = df_map.get("PPB_affinity")
    if pp_df is not None and ppb_df is not None:
        pp_pairs = (set(zip(pp_df["molA"], pp_df["molB"]))
                    | {(b, a) for a, b in zip(pp_df["molA"], pp_df["molB"])})
        keep = ppb_df.apply(lambda r: (r["molA"], r["molB"]) not in pp_pairs, axis=1)
        n_removed = (~keep).sum()
        df_map["PPB_affinity"] = ppb_df[keep].reset_index(drop=True)
        print(f"  PPB_affinity: {n_removed:,} rows removed (overlap with PDB_bind_PP) "
              f"-> {len(df_map['PPB_affinity']):,} rows remain")

    return pd.concat(list(df_map.values()), ignore_index=True)


def dedup_swap_pairs(df: pd.DataFrame, eligible_idx: list) -> pd.DataFrame:
    """Returns a NEW dataframe (only the eligible rows, deduplicated) where
    each underlying interaction -- treating (molA,molB) and (molB,molA) as
    the same -- appears exactly once, pKd averaged across duplicates
    (including the row's own original value if it had no duplicate)."""
    sub = df.loc[eligible_idx].copy()
    sub["_canon"] = [tuple(sorted((a, b))) for a, b in zip(sub["molA"], sub["molB"])]

    mean_pkd = sub.groupby("_canon")["pKd"].mean()
    n_groups_with_dup = int((sub.groupby("_canon").size() > 1).sum())

    out = sub.drop_duplicates(subset="_canon", keep="first").copy()
    out["pKd"] = out["_canon"].map(mean_pkd).round(4)
    if "label" in out.columns:
        out["label"] = 0
        out.loc[out["pKd"] >= 6.0, "label"] = 1
    out = out.drop(columns=["_canon"]).reset_index(drop=True)

    print(f"    swap-aware dedup: {len(sub):,} eligible rows -> {len(out):,} unique interactions "
         f"({n_groups_with_dup:,} groups had a swap/exact duplicate merged)")
    return out


def make_split(df: pd.DataFrame) -> pd.DataFrame:
    """Returns (non_eligible_df, eligible_unique_df_with_partition_col)."""
    rng = random.Random(SEED)

    eligible_mask = df["data_origin"].isin(CPI_ORIGINS | PPI_ORIGINS) & has_pkd(df["pKd"])
    eligible_idx = list(df.index[eligible_mask])
    non_eligible = df.loc[~df.index.isin(eligible_idx)].copy()

    unique_df = dedup_swap_pairs(df, eligible_idx)

    idx = list(range(len(unique_df)))
    rng.shuffle(idx)
    n = len(idx)
    n_val = int(n * VAL_FRAC)
    n_test = int(n * TEST_FRAC)
    val_idx = set(idx[:n_val])
    test_idx = set(idx[n_val:n_val + n_test])

    partition = ["train"] * n
    for i in val_idx:
        partition[i] = "val"
    for i in test_idx:
        partition[i] = "test"
    unique_df["partition"] = partition

    non_eligible["partition"] = "train"
    return non_eligible, unique_df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading all datasets ...")
    df = load_all()
    print(f"Total rows: {len(df):,}\n")

    print(f"=== Pure random split, unique-only (dedup BEFORE split, seed={SEED}) ===")
    non_eligible, unique_df = make_split(df)

    for partition in ("train", "val", "test"):
        subset = pd.concat([
            non_eligible[non_eligible["partition"] == partition],
            unique_df[unique_df["partition"] == partition],
        ], ignore_index=True).drop(columns=["partition"])
        pkd_count = has_pkd(subset["pKd"]).sum()
        cpi_pkd = has_pkd(subset.loc[subset["data_origin"].isin(CPI_ORIGINS), "pKd"]).sum()
        ppi_pkd = has_pkd(subset.loc[subset["data_origin"].isin(PPI_ORIGINS), "pKd"]).sum()
        print(f"    {partition}: {len(subset):,} rows | pKd total={pkd_count:,} | "
             f"CPI pKd={cpi_pkd:,} | PPI pKd={ppi_pkd:,}")
        out_path = os.path.join(OUT_DIR, f"split_pure_random_uniqueonly_{partition}.csv")
        subset.to_csv(out_path, index=False)

    print("\nDone.")


if __name__ == "__main__":
    main()
