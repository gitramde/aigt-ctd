# Reproducible baseline preparation

This implements the cleaning, chronological cohort design and fitted preprocessing stage for a binary CSE-CIC-IDS2018 baseline. `Benign` maps to 0; observed attacks map to 1. Original attack labels remain available for per-attack and unseen-attack diagnostics. It does not train a classifier, construct graphs, or implement GATv2/AIGT-CTD.

## Run

The execution environment is recorded in `results/baseline_v1/environment.json`. Pinned Python dependencies are in `requirements-baseline.txt`. The verified interpreter for this run is Python 3.13.5 with NumPy 2.1.3, pandas 2.2.3, PyArrow 19.0.0 and scikit-learn 1.6.1. The older project `.venv` may not contain these dependencies.

From the repository root, with those dependencies installed:

```powershell
python -m unittest src.baseline.test_baseline -v
python -m src.baseline.pipeline --config configs/baseline.json
```

Or run the stages individually:

```powershell
python -m src.baseline.pipeline --config configs/baseline.json --stage clean
python -m src.baseline.pipeline --config configs/baseline.json --stage split
python -m src.baseline.pipeline --config configs/baseline.json --stage preprocess
python -m src.baseline.pipeline --config configs/baseline.json --stage verify
```

Cleaning, split selection and preprocessing refuse to overwrite existing outputs. To reproduce independently, copy the configuration and choose new `data_dir` and `report_dir` values. Verification is read-only except for updating its own verification report. An interrupted cleaning run is incomplete; use a new version directory rather than treating partial files as finished.

## Cleaning and provenance

Raw files are opened read-only. Cleaning computes their SHA-256 hashes and checks file size/modification time during the read. Final verification recomputes every original hash. Source files and the previous Phase 2 reports are not overwritten.

Processing order is explicit:

1. Remove complete repeated-header records, comparing every field to the initial header.
2. Remove exact repeated records within each file, retaining the earliest original data-row number. A 128-bit fingerprint locates candidates; the original full field tuple is read back from its byte offset to verify equality. Numeric representations, whitespace and labels are not canonicalized before comparison. Hash collisions cannot remove unequal records.
3. Exclude year-1970 and any other invalid/wrong-date timestamps from temporal data. Retain these records separately in `*.non_temporal.parquet`; raw files still preserve their original representation.
4. Convert positive and negative infinity in `Flow Byts/s` and `Flow Pkts/s` to missing values. Infinity in any other feature is an error. Original NaN tokens become missing values without imputation at this stage. Parquet nulls represent missing values; they become NumPy NaN when preprocessing reads them.
5. Sort eligible records by recorded timestamp, with original source row breaking ties. Hourly staging files bound sorting memory and are removed only after their derived output is written. Only this run's generated staging files are deleted.

The 410,763 Phase 2 duplicates include 56 extra repeated-header occurrences. Removing all 59 repeated headers first therefore leaves 410,707 non-header duplicates, not another 410,763 removals. Expected temporal population:

```text
16,233,002 raw - 59 headers - 410,707 duplicates - 14 year-1970 exclusions
= 15,822,222 temporal rows
```

`cleaning_audit.csv` has one row for every removal or temporal exclusion: filename, original data-row number (1-based after the initial header), physical source line, byte offset, label, original timestamp, reason, disposition, and the retained first occurrence for duplicates. `cleaning_summary.csv` reconciles counts; `cleaning_cell_audit.csv` reports infinity conversions and remaining missing cells by file/feature.

`source_manifest.json` binds raw and cleaned files to hashes and counts. Each cleaned row retains `source_file`, `source_row`, `timestamp`, `file_date`, `attack_label` and `target_binary` as metadata, never as model features. The scanner requires one physical line per CSV record, as established for these source files; malformed or multiline input fails explicitly.

## Class/date report and chronological split

Cleaning writes `pre_split_class_distribution.csv` **before** split selection. It contains each attack class, source filename/date and row count at the raw, deduplicated, and temporally eligible stages. The split plan embeds this report's SHA-256 hash.

The default selection protocol considers contiguous whole-date blocks. Each partition must contain at least 100 benign and 100 malicious records. At least three dates go to training, one to validation and two to test. Among feasible candidates, maximize training attack-class coverage, then minimize absolute deviation from the configured 80%/10%/10% row proportions; chronological cutoffs resolve ties. All feasible candidates and their ranks are saved in `split_candidates.csv`. This is cohort design using the requested label/date counts, not model-performance-driven selection.

The pipeline requires one source file per date, matching the ten current files. It never shuffles or moves later observations into training to force class balance. Whole-date boundaries avoid splitting equal timestamps across partitions. Dates are parsed day-first. Source clock timezone and AM/PM information are unavailable, so this preserves **recorded chronology**, without claiming to reconstruct physical capture chronology.

`split_plan.json` specifies each partition's dates, exact source-file hashes, cleaned-file hashes, inclusive/exclusive timestamp bounds, row rule, order and expected counts. `data/baseline_v1/membership/{train,validation,test}.parquet` enumerates every original source row assigned to each partition. Final verification proves that these membership rows equal the chronological feature rows and that every raw row is accounted for exactly once as eligible, removed, or quarantined.

Attack classes occur on different dates. Strict chronology cannot provide every attack type in all three partitions. `partition_class_distribution.csv` includes zero-count classes and identifies attacks unseen in training. The binary target retains these attacks as positives; this is temporal generalization to potentially unseen attacks, not a closed-set multiclass evaluation. No resampling, balancing or train/test window overlap is introduced.

## Feature leakage and preprocessing

Only fields common to all ten source schemas are eligible. The target and target-named fields are excluded explicitly. Timestamp, flow ID, IP addresses, source port and generated provenance fields are metadata/schema identifiers and are excluded from baseline features. Destination port remains an observed numeric flow feature; protocol is categorical. The distinct reasons are recorded in `schema_exclusions.csv` and `feature_exclusion_audit.csv`.

A second check uses **training rows only** to find low-cardinality fields that are one-to-one encodings of the binary target or the attack label. Detected equivalents are explicitly excluded and their observed mappings are recorded. This is a conservative check for direct target encodings, not a proof that every remaining correlated feature is free of every possible causal leakage mechanism. Validation/test values do not decide feature exclusions.

Fitting is hard-wired to the frozen training partition:

- Numeric missing values use the full-training finite-value arithmetic mean, accumulated in float64 batches. A feature entirely missing in training uses the configured constant 0 and is flagged explicitly.
- Protocol missing values use the training mode, with deterministic lexical tie-breaking. One-hot categories come only from training. Unknown validation/test categories produce an all-zero one-hot block and are counted. A category entirely missing in training gets an explicit placeholder.
- `StandardScaler.partial_fit` sees only training numeric features after applying the already fitted imputation values. Constant features have scale 1. One-hot columns are not standardized.
- Feature order, selected/excluded fields, imputation values, categories, scaler mean/variance/scale, training count, dependency versions and split/source hashes are persisted in `preprocessing.json`.

The fitted JSON is reloaded and the same transform-only object is applied to **every row** of training, validation and test. `preprocessing_application_audit.csv` records row counts, missing cells imputed, unknown categories, finite-output checks, the unchanged preprocessing hash and a checksum of each partition's transformed float64 values. Full transformed feature matrices are generated on demand rather than stored as a second multi-gigabyte copy.

For later model code:

```python
from src.baseline.common import load_config
from src.baseline.preprocess import iter_transformed

config = load_config('configs/baseline.json')
for X, y, provenance in iter_transformed(config, 'train', scaled=True):
    # X: numeric imputation + scaling + fixed protocol encoding
    # y: binary target; provenance retains timestamps, source rows and attack labels
    pass

# Use the identical saved preprocessing for 'validation' and 'test'.
# scaled=False retains fitted imputation/encoding but omits numeric scaling
# for a future model that does not require it. Neither path performs fitting.
```

The fitting/transform separation follows the [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage). Incremental scaling uses the installed [StandardScaler API](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html), with versions recorded for reproducibility.

## Verification

`verification.json` records source-hash preservation, row conservation, chronological ordering, unique row identities, disjoint partitions, membership/feature alignment, class counts, target/metadata exclusion and application of one unchanged fitted preprocessor. Synthetic tests cover collisions, duplicate/header overlap, invalid timestamps, train-only means, test-set perturbation, unknown categories, all-missing/constant features, impossible split designs, and rejection of changed raw files.

No model metrics are reported because no classifier has been trained or evaluated in this preparation step. No GATv2 or AIGT-CTD components are implemented.
