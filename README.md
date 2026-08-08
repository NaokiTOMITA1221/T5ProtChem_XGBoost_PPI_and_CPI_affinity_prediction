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
- **Data augmentation** (train set only): PPI-origin rows are oversampled
  toward a ~1:1 CPI:PPI ratio; each augmented copy replaces 12-32 residues
  per protein chain with their free-amino-acid SMILES representation
  ("ProtSMILES splicing") and jitters pKd by ±5%.

## Data / split

Source datasets: BindingDB, PDB-bind (protein-ligand and protein-protein
subsets), PPB-Affinity, Human/C. elegans, Negatome (see the wider project's
data curation pipeline — not included in this repo).

The split (`scripts/make_pure_random_split_uniqueonly.py`) is a **plain
row-level random split** (no protein-sequence-identity grouping, no CPI:PPI
balancing) — deliberately closer to how a naive practitioner might split
this kind of data — but with one fix applied *before* splitting: rows whose
(moleculeA, moleculeB) pair matches another row's swapped
(moleculeB, moleculeA) pair (both listing the same interaction in the two
possible chain orders) are deduplicated first, so no single underlying
interaction can leak across train/val/test through that route.

The split CSVs under `data/` (`molA`/`molB`/format/`pKd`/`data_origin`
columns) are already filtered down to exactly the rows actually used for
training/evaluation (single-chain, eligible CPI/PPI rows with a real pKd --
non-CPI/PPI-origin rows and multi-chain rows, which `make_pure_random_
split_uniqueonly.py` itself leaves in place, have been removed here):

- `data/split_pure_random_uniqueonly_train.csv`
- `data/split_pure_random_uniqueonly_val.csv`
- `data/split_pure_random_uniqueonly_test.csv`

Row counts (augmented copies are generated on the fly at train time, not
stored in these CSVs):

| | original CPI | original PPI | augmented PPI (factor=6) | total used |
|---|---|---|---|---|
| train | 24,241 | 3,587 | 21,522 | 49,350 |
| val | 3,019 | 448 | — (clean) | 3,467 |
| test | 3,065 | 426 | — (clean) | 3,491 |

Augmentation (ProtSMILES splice + ±5% pKd jitter, see Model section above)
is applied ONLY to PPI-origin train rows -- CPI-origin rows are never
augmented, and val/test always stay clean/unaugmented for evaluation.

## Results

Single held-out split (`scripts/train_boost_t5protchem_raw_uniqueonly_quadsplice.py`,
see `results/metrics.json`):

| | overall | CPI | PPI |
|---|---|---|---|
| val Pearson r | 0.797 | 0.769 | 0.839 |
| test Pearson r | 0.810 | 0.771 | 0.829 |
| val RMSE | 1.069 | 1.056 | 1.153 |
| test RMSE | 1.070 | 1.045 | 1.233 |

![Test set predicted vs. true pKd](results/test_scatter.png)

### External validation (out-of-domain)

Predictions for an independent, out-of-domain polymer-peptide dataset
(`scripts/predict_hoshino_t5protchem_raw_uniqueonly_quadsplice.py`, results
in `results/predicted_pKa_KanM_T5ProtChem_raw_uniqueonly_quadsplice.csv`),
correlated against an experimentally measured neutralization-ratio readout
for the same pairs (`scripts/correlate_hoshino_t5protchem_raw_uniqueonly_quadsplice.py`).
p-value is from a PERMUTATION test (99999 resamples, shuffling one variable
against the other to build the null distribution of r under independence)
rather than the parametric t-distribution approximation, since n=15 is too
small for the bivariate-normality assumption behind the parametric p-value
to be reliable. Permutation testing was chosen over a naive bootstrap
percentile p-value, which doesn't properly enforce the null hypothesis and
is unstable at this sample size (see conversation):

| | Pearson r | p (permutation) |
|---|---|---|
| vs. neutralization ratio (n=15) | 0.643 | 0.0158 |

![Neutralization ratio vs. predicted pKd](results/neutralization_vs_predicted.png)

**Caveat**: a 10-seed robustness check (reshuffling the train/val/test
partition 10 times and refitting) found this external correlation is **not
stable** — across seeds the mean Pearson r was 0.18 (unbalanced) / -0.06
(balanced), both not significantly different from zero. The single-split
result above should be read as one favorable draw, not a validated,
reproducible effect. In-domain (val/test) performance was consistently
strong and stable across all seeds tested.

## Reproducing

```bash
python scripts/make_pure_random_split_uniqueonly.py   # builds the split CSVs
python scripts/train_boost_t5protchem_raw_uniqueonly_quadsplice.py
python scripts/predict_hoshino_t5protchem_raw_uniqueonly_quadsplice.py
python scripts/correlate_hoshino_t5protchem_raw_uniqueonly_quadsplice.py
python scripts/plot_test_scatter.py                   # re-extracts test features and plots results/test_scatter.png
python scripts/plot_hoshino_figures.py                 # plots results/neutralization_vs_predicted.png (Figure S24 vs. predicted pKd, side by side)
```

Scripts reference absolute paths from the original project layout (T5ProtChem
checkpoints and an external polymer-affinity validation CSV) — update the
path constants at the top of each script for your own environment. The split
CSVs are included under `data/` (see above); the raw source datasets used to
build them, the T5ProtChem pretrained checkpoint, and the external validation
dataset are not included in this repo.
