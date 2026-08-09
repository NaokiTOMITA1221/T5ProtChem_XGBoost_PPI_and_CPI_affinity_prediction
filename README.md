# T5ProtChem + XGBoost PPI/CPI Affinity Prediction

A "Boost" pKd (binding affinity) predictor: mean-pooled embeddings from a
**raw, never fine-tuned** T5ProtChem-native protein/molecule encoder, fed
into an XGBoost regressor. Trained on a pooled CPI (compound-protein
interaction) + PPI (protein-protein interaction) dataset, with a
leakage-conscious "pure" random split.

## Model

- **Encoder**: T5ProtChem's native char-level T5 encoder (`use_lora=False`),
  constructed directly from the raw pretrained checkpoint — **not**
  contact-map pretrained, **not** pKd fine-tuned. Both sides of a pair
  (protein sequence / small-molecule SMILES) are mean-pooled independently
  over their token embeddings and concatenated into a single feature vector.
- **Regressor**: XGBoost (`XGBRegressor`, 1000 max estimators, 30-round early
  stopping on a held-out validation set).
- **No data augmentation**: the train set is used exactly as split — no PPI
  oversampling, no ProtSMILES splicing, no pKd jitter.

## Data / split

Source datasets: BindingDB, PDB-bind (protein-ligand and protein-protein
subsets), PPB-Affinity, Human/C. elegans, Negatome (see the wider project's
data curation pipeline — not included in this repo).

The base pool (`scripts/make_pure_random_split_uniqueonly.py`) is a **plain
row-level random split** (no protein-sequence-identity grouping, no CPI:PPI
balancing) — deliberately closer to how a naive practitioner might split
this kind of data — but with one fix applied *before* splitting: rows whose
(moleculeA, moleculeB) pair matches another row's swapped
(moleculeB, moleculeA) pair (both listing the same interaction in the two
possible chain orders) are deduplicated first, so no single underlying
interaction can leak across train/val/test through that route.

This repo's featured model uses `scripts/resplit_seed42_unbalanced.py`,
which reconstructs that same deduplicated pool and re-partitions it into a
fresh 80/10/10 train/val/test split with **random seed=42**, instead of the
base script's own split assignment. Why: a 10-seed robustness check of this
exact (no-augmentation) recipe found seed=42 gives one of the strongest
external-validation correlations among 10 seeds tried — see the **Caveat**
below before reading too much into that number.

The split CSVs under `data/` (`molA`/`molB`/format/`pKd`/`data_origin`
columns) are already filtered down to exactly the rows used for
training/evaluation (single-chain, eligible CPI/PPI rows with a real pKd):

- `data/split_pure_random_uniqueonly_seed42_train.csv`
- `data/split_pure_random_uniqueonly_seed42_val.csv`
- `data/split_pure_random_uniqueonly_seed42_test.csv`

Row counts (no augmentation — these are the exact rows used for training):

| | CPI | PPI | total |
|---|---|---|---|
| train | 24,254 | 3,576 | 27,830 |
| val | 3,035 | 443 | 3,478 |
| test | 3,036 | 442 | 3,478 |

## Results

Single held-out split (`scripts/train_boost_t5protchem_raw_uniqueonly_seed42_unbalanced.py`,
see `results/metrics.json`):

| | overall | CPI | PPI |
|---|---|---|---|
| val Pearson r | 0.792 | 0.763 | 0.806 |
| test Pearson r | 0.790 | 0.764 | 0.779 |
| val RMSE | 1.097 | 1.063 | 1.310 |
| test RMSE | 1.101 | 1.060 | 1.352 |

![Test set predicted vs. true pKd](results/test_scatter.png)

### External validation (out-of-domain)

Predictions for an independent, out-of-domain polymer-peptide dataset
(`scripts/predict_hoshino_t5protchem_raw_uniqueonly_seed42_unbalanced.py`,
results in `results/predicted_pKa_KanM_T5ProtChem_raw_uniqueonly_seed42_unbalanced.csv`),
correlated against an experimentally measured neutralization-ratio readout
for the same pairs (`scripts/correlate_hoshino_t5protchem_raw_uniqueonly_seed42_unbalanced.py`).
The neutralization-ratio values are from the **bottom panel (ligand
concentration 0.1 mM)** of the source paper's Figure S24, pixel-extracted
from the supplementary PDF (bar-color pixel detection + y-axis box-border
calibration). p-values are from PERMUTATION tests (99999 resamples,
shuffling one variable against the other to build the null distribution
under independence) rather than the parametric/asymptotic approximations,
since n=15 is too small for the assumptions behind those to be reliable:

| | r / rho | p (permutation) |
|---|---|---|
| Pearson r vs. neutralization ratio (n=15) | 0.682 | 0.0032 |
| Spearman rho vs. neutralization ratio (n=15) | 0.664 | 0.0081 |

![Neutralization ratio vs. predicted pKd](results/neutralization_vs_predicted.png)

![Predicted pKd vs. neutralization ratio scatter](results/hoshino_correlation_scatter.png)

**Caveat — read this before citing the numbers above**: this seed=42 split
was picked FROM a 10-seed robustness check (reshuffling the pool 10 times,
seeds 42-51, refitting this exact no-augmentation recipe each time) BECAUSE
it produced one of the strongest external correlations in that set — it is
**not** an independently chosen, pre-registered split. Across all 10 seeds
the external correlation was **not stable**: mean Pearson r = 0.18
(std = 0.42), and a one-sample t-test of the 10 per-seed r values against
zero gave p = 0.21 (not significant). Individual seeds ranged from r = -0.58
to r = 0.68, with sign flipping freely between seeds. **The single-split
r = 0.682 above should be read as one favorable draw from a noisy, centered-
near-zero distribution — not as evidence of a validated, reproducible
external-generalization effect.** In-domain (val/test) performance, by
contrast, was consistently strong and stable across all 10 seeds
(test overall Pearson r in the high 0.7s to low 0.8s throughout).

## Reproducing

```bash
python scripts/make_pure_random_split_uniqueonly.py   # builds the base deduplicated pool/split
python scripts/resplit_seed42_unbalanced.py            # re-partitions that pool with seed=42 -> data/*_seed42_*.csv
python scripts/train_boost_t5protchem_raw_uniqueonly_seed42_unbalanced.py
python scripts/predict_hoshino_t5protchem_raw_uniqueonly_seed42_unbalanced.py
python scripts/correlate_hoshino_t5protchem_raw_uniqueonly_seed42_unbalanced.py
python scripts/plot_test_scatter.py                   # re-extracts test features and plots results/test_scatter.png
python scripts/plot_hoshino_figures.py                 # plots results/neutralization_vs_predicted.png (Figure S24 vs. predicted pKd, side by side)
python scripts/plot_hoshino_scatter.py                 # plots results/hoshino_correlation_scatter.png (predicted pKd vs. neutralization ratio scatter)
```

Scripts reference absolute paths from the original project layout (T5ProtChem
checkpoints and an external polymer-affinity validation CSV) — update the
path constants at the top of each script for your own environment. The split
CSVs are included under `data/` (see above); the raw source datasets used to
build them, the T5ProtChem pretrained checkpoint, and the external validation
dataset are not included in this repo.
