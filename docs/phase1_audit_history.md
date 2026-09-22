> Historical Phase 1 documentation. Its stage-specific statements describe the initial audit, not current project status. See the [current README](../README.md) for Phases 1-4A and the frozen baseline results.

# AIGT-CTD

**AI/ML-Based Dynamic Graph-Transformer Framework for Explainable Threat Detection**

An academic cybersecurity experiment using CSE-CIC-IDS2018 as its primary dataset. The completed stage documented here is a full audit of the ten locally downloaded processed traffic CSVs. Graph construction and model training are not part of this process.

The audit reads every record. It does not clean, impute, normalize, sample, split, deduplicate, or export transformed source data.

## Audit findings

The ten files contain **16,233,002 records**, occupying **6,886,649,507 bytes**. Nine files have 80 columns; February 20 has 84. Counts exclude each initial CSV header but retain repeated headers inside the data.

| Measure | Count |
| --- | ---: |
| Benign records | 13,484,708 |
| Malicious records | 2,748,235 |
| Unclassified repeated-header records | 59 |
| Empty or whitespace-only cells | 0 |
| Explicit NaN cells | 59,721 |
| Null-marker cells | 0 |
| Infinite cells | 131,799 |
| Extra exact duplicate records within files | 410,763 |

The 59 unclassified records have the literal label `Label`. They are retained and reported separately from attacks. Duplicate counts include extra occurrences of repeated headers, so these measures overlap. Source data was not modified. These findings describe the previously completed full audit, whose synthetic correctness tests and independent reconciliation passed for all ten files.

**Saved-report discrepancy found while writing this README:** the CSVs currently on disk use the older auditor's schema. They lack fields such as `unclassified_count` and quality `scope`, classify `Label` as malicious, and use different timestamp/type inference. The verified findings below remain the audit snapshot; the current saved CSVs should not be treated as that verified report set. Run a fresh audit with `audit_full_dataset.py` **without `--resume`**, then run the validator, to regenerate the documented format. This documentation update does not replace those CSVs.

**Graph limitation:** only `Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv` preserves source IP, destination IP, and source port. The other nine files cannot independently support host-level source-to-destination interaction graphs. See [graph feasibility](#temporal-interaction-graph-feasibility).

## Repository layout

```text
aigt-ctd/
  README.md
  data/raw/Processed Traffic Data for ML Algorithms/
    [10 original CSV files]
  src/data/
    audit_full_dataset.py
    test_audit_full_dataset.py
    validate_dataset_audit.py
  results/
    dataset_summary.csv
    class_by_file.csv
    column_inventory.csv
    data_quality_report.csv
    dataset_audit_notes.md
```

Use `src/data/audit_full_dataset.py` for this workflow. The older `audit_processed_dataset.py` is not the implementation used to generate these reports.

## How to run the process

### 1. Prepare Python

Open PowerShell in the repository root. Use Python 3.11 or newer and NumPy. The audit does not require pandas, a GPU, or a training framework.

Create a virtual environment if `.venv` does not already exist:

```powershell
py -3 -m venv .venv
```

Install the project dependencies using that environment's interpreter. NumPy is used by the full audit; pandas supports the retained legacy script:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import sys, numpy; print(sys.version); print('NumPy:', numpy.__version__)"
```

Direct interpreter calls avoid needing to activate the environment. If an existing environment already has NumPy, replace `.\.venv\Scripts\python.exe` with its `python` command throughout.

The Git repository includes source code, documentation, and small audit reports. The original dataset, virtual environment, caches, and local credential files are excluded by `.gitignore`. After cloning, place your downloaded CSVs in the input folder below and recreate the Python environment locally.

### 2. Check the input files

Put all ten CSVs directly inside `data/raw/Processed Traffic Data for ML Algorithms/`. The audit does not search subdirectories beneath `--input-dir`.

```powershell
Get-ChildItem -LiteralPath 'data/raw/Processed Traffic Data for ML Algorithms' -Filter '*.csv' -File |
    Select-Object Name, Length
(Get-ChildItem -LiteralPath 'data/raw/Processed Traffic Data for ML Algorithms' -Filter '*.csv' -File).Count
```

The count must be `10`. Preserve the original names, including the supplied spelling `Thuesday-20-02-2018`.

### 3. Run the full audit

```powershell
.\.venv\Scripts\python.exe src/data/audit_full_dataset.py --input-dir "data/raw/Processed Traffic Data for ML Algorithms" --output-dir results --workers 2 --chunk-size 10000
```

This scans every record and writes four CSV reports. A fresh run replaces the reports in the selected output directory without modifying the input files. Run only one audit against a given output directory at a time.

Files are scheduled largest first across the worker processes. Progress shows record counts, duplicate verification, and completion of individual files. Successful completion ends with:

```text
All 10 files audited; four reports written.
```

The February 20 file is about 4.05 GB and contains nearly half the records. Runtime depends on CPU, disk speed, available memory, and competing workloads. Duplicate candidates trigger a second full read; progress can be quiet during that verification. Reports can contain only completed files while the audit is running. Wait for the final completion message before treating them as complete.

### 4. Verify the results

Run the small synthetic correctness test:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s src/data -p test_audit_full_dataset.py
```

It covers duplicates across batches, forced fingerprint collisions, day-first timestamps, repeated headers, mixed types, blanks, NaNs, null markers, infinities, and agreement between accelerated and scalar inspection. Expected result: `OK`.

Reconcile the reports against all original files:

```powershell
.\.venv\Scripts\python.exe src/data/validate_dataset_audit.py
```

Expected result: ten `PASS` lines followed by JSON containing aggregate counts and labels. The validator checks file sizes, independent binary line counts, source headers, column inventories, class totals, quality totals, constants, and timestamp accounting.

The validator reads the repository's `results/` directory and searches recursively for CSVs under `data/`. It expects exactly the original ten CSVs and has no custom-path arguments. Its physical-line check applies to these files, whose records have no embedded line breaks. Use normal Python, not `python -O`, because verification uses assertions.

### 5. Inspect the outputs

```powershell
Import-Csv results/dataset_summary.csv |
    Select-Object filename, row_count, column_count, benign_count, malicious_count, unclassified_count

Import-Csv results/data_quality_report.csv |
    Where-Object { $_.scope -eq 'file' } |
    Select-Object filename, missing_values, nan_values, infinite_values, duplicate_records, repeated_header_records

Import-Csv results/column_inventory.csv |
    Where-Object { $_.is_source_identifier -eq 'True' -or $_.is_destination_identifier -eq 'True' }
```

### Options and resource use

| Option | Default | Meaning |
| --- | --- | --- |
| `--input-dir` | Required | Directory containing exactly ten CSVs. |
| `--output-dir` | Required | Report directory, created if absent. |
| `--chunk-size` | `10000` | Records per inspection batch; every batch is scanned. |
| `--workers` | `2` | Concurrent file-audit processes; use a positive integer. |
| `--resume` | Off | Reuse completed file entries saved in the four reports. |
| `--help` | — | Display command usage without running the audit. |

For lower available memory, reduce concurrency and batch size:

```powershell
.\.venv\Scripts\python.exe src/data/audit_full_dataset.py --input-dir "data/raw/Processed Traffic Data for ML Algorithms" --output-dir results --workers 1 --chunk-size 5000
```

Chunk size controls batch memory, but fingerprints are retained across the full file. Exact verification also retains duplicate candidates. Total memory therefore grows with file size and is not capped by `--chunk-size`. Even `--workers 1` uses a worker process.

### Resume an interrupted audit

Reports are checkpointed after completed files, not after individual batches. If the four reports are consistent and the inputs are unchanged:

```powershell
.\.venv\Scripts\python.exe src/data/audit_full_dataset.py --input-dir "data/raw/Processed Traffic Data for ML Algorithms" --output-dir results --workers 2 --chunk-size 10000 --resume
```

Completed files are reused; unfinished files restart from the beginning. Resuming an already completed run reuses its results rather than rescanning the sources.

Resume checks file sizes, not content hashes. It cannot detect changes that preserve file size. Reports are written sequentially rather than as an atomic set. If any report is missing or inconsistent, interruption occurred during report writing, or source data changed, run a fresh audit without `--resume`. Always run the validator afterward.

### Troubleshooting

| Symptom | Action |
| --- | --- |
| `Expected 10 files, found ...` | Check the quoted path and the CSV count directly inside it. |
| `ModuleNotFoundError: numpy` | Install NumPy with the interpreter used for the audit. |
| Python command not recognized | Use the full path to an installed Python interpreter. |
| Worker-pipe permission error | Use a normal local terminal with process-creation access. Restricted agent sandboxes may require execution approval. |
| Temporary-file permission error in tests | Ensure the interpreter can create and remove files in its temporary directory. |
| High memory use | Stop the current run before restarting with fewer workers. Resume only from consistent checkpoints. |
| Incomplete reports | Wait for all ten files to finish, then run the validator. |
| Validator assertion failure | Inspect the failed check, source-file set, and report consistency. Do not change raw data to force a pass. |

## Report contents

| Report | Contents |
| --- | --- |
| [dataset_summary.csv](../results/dataset_summary.csv) | One row per file: filename, sizes in bytes and MiB, row/column counts, names and inferred types, timestamp range, labels, class totals, quality totals, constants, graph-field presence and schema sufficiency. |
| [class_by_file.csv](../results/class_by_file.csv) | Exact observed labels and counts by file, with benign/malicious/unclassified grouping. |
| [column_inventory.csv](../results/column_inventory.csv) | Column order, names, inferred types, CSV text storage, constant flags/values and graph-field roles for every file. |
| [data_quality_report.csv](../results/data_quality_report.csv) | File and column quality counts distinguished by `scope`; duplicates and repeated headers appear at file scope. |

The documented auditor generates 10 summary rows, 31 file/label rows, 804 inventory rows, and 814 quality rows for these inputs, excluding report headers. Summary cells containing names, type mappings, labels or constant lists use JSON. These definitions apply after regenerating the newer format, not to the legacy reports noted above.

Do not sum `scope=file` and `scope=column` rows together: file rows already aggregate column counts. Blank, NaN, null-marker and infinity counts measure **cells**, not affected records. See also [dataset_audit_notes.md](../results/dataset_audit_notes.md).

## Audit definitions

- **Rows:** all CSV records after the initial header, including repeated headers. Field counts must match the header.
- **Missing values:** empty or whitespace-only fields.
- **NaNs:** explicit `NaN`, `+NaN`, or `-NaN` tokens, matched without regard to case or surrounding whitespace. Counted separately from blanks.
- **Null markers:** `null`, `none`, `na`, `n/a`, `#n/a`, and `<na>`, matched without regard to case or surrounding whitespace.
- **Infinities:** values parsed as positive or negative infinity, including `inf` and `Infinity` spellings.
- **Duplicates:** extra occurrences after the first identical complete record within a file. Equality compares every original parsed field without numeric coercion or whitespace normalization. Cross-file duplicates are not measured.
- **Duplicate verification:** 128-bit BLAKE2 fingerprints locate candidates; a second read compares full field tuples to avoid counting hash collisions as duplicates.
- **Strict constants:** every original value in a column is identical, including missing tokens and repeated headers. The additional `constant_non_missing_excluding_header_tokens` flag is an analytical diagnostic, not a modification of the data.
- **Timestamps:** parsed strictly as `DD/MM/YYYY HH:MM:SS`. Ranges cover parseable values. Nonmissing parse failures are counted separately. No timezone or missing clock information is reconstructed.
- **Labels:** `Benign` is benign; blank/missing-marker labels and the token `Label` are unclassified; other observed nonempty labels are malicious. Original strings are preserved.
- **Types:** inferred across all records. CSV storage is text. Missing tokens are disregarded for type inference, but repeated headers are retained, causing mixed types. `integer` describes integral finite values, not a guaranteed storage width. Exact distinct-value counts are not reported.
- **Source checks:** size and modification time are checked before and after each file. These are metadata checks, not cryptographic provenance verification.

```text
13,484,708 benign + 2,748,235 malicious + 59 unclassified = 16,233,002 records
```

## Detailed results

The following tables document the verified audit findings, rather than endorsing the legacy reports currently on disk. Short file IDs omit only the common suffix `_TrafficForML_CICFlowMeter.csv`. Original filenames are preserved in the CSV reports.

### File sizes and record counts

| File ID | Size (bytes) | Rows | Columns | Benign | Malicious | Unclassified |
| --- | --- | --- | --- | --- | --- | --- |
| Friday-02-03-2018 | 352,368,373 | 1,048,575 | 80 | 762,384 | 286,191 | 0 |
| Friday-16-02-2018 | 333,723,605 | 1,048,575 | 80 | 446,772 | 601,802 | 1 |
| Friday-23-02-2018 | 382,840,456 | 1,048,575 | 80 | 1,048,009 | 566 | 0 |
| Thuesday-20-02-2018 | 4,054,925,350 | 7,948,748 | 84 | 7,372,557 | 576,191 | 0 |
| Thursday-01-03-2018 | 107,842,858 | 331,125 | 80 | 238,037 | 93,063 | 25 |
| Thursday-15-02-2018 | 375,945,899 | 1,048,575 | 80 | 996,077 | 52,498 | 0 |
| Thursday-22-02-2018 | 382,636,202 | 1,048,575 | 80 | 1,048,213 | 362 | 0 |
| Wednesday-14-02-2018 | 358,223,333 | 1,048,575 | 80 | 667,626 | 380,949 | 0 |
| Wednesday-21-02-2018 | 328,893,673 | 1,048,575 | 80 | 360,833 | 687,742 | 0 |
| Wednesday-28-02-2018 | 209,249,758 | 613,104 | 80 | 544,200 | 68,871 | 33 |

### Timestamp ranges

| File ID | Earliest timestamp | Latest timestamp | Unparseable nonmissing values |
| --- | --- | --- | --- |
| Friday-02-03-2018 | 2018-03-02T01:00:00 | 2018-03-02T12:59:59 | 0 |
| Friday-16-02-2018 | 2018-02-16T01:00:32 | 2018-02-16T12:58:24 | 1 |
| Friday-23-02-2018 | 2018-02-23T01:00:00 | 2018-02-23T12:59:59 | 0 |
| Thuesday-20-02-2018 | 2018-02-20T01:00:00 | 2018-02-20T12:59:59 | 0 |
| Thursday-01-03-2018 | 2018-03-01T01:00:00 | 2018-03-01T12:59:59 | 25 |
| Thursday-15-02-2018 | 2018-02-15T01:00:00 | 2018-02-15T12:59:59 | 0 |
| Thursday-22-02-2018 | 1970-01-10T03:04:26 | 2018-02-22T12:59:59 | 0 |
| Wednesday-14-02-2018 | 1970-01-05T03:01:17 | 2018-02-14T12:59:59 | 0 |
| Wednesday-21-02-2018 | 2018-02-21T01:55:46 | 2018-02-21T10:43:21 | 0 |
| Wednesday-28-02-2018 | 2018-02-28T01:00:00 | 2018-02-28T12:59:59 | 33 |

### Quality by file

| File ID | Blank cells | NaN cells | Null markers | Infinite cells | Extra duplicates | Repeated headers | Strict constants |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Friday-02-03-2018 | 0 | 2,558 | 0 | 5,542 | 5,459 | 0 | 10 |
| Friday-16-02-2018 | 0 | 0 | 0 | 0 | 147,586 | 1 | 0 |
| Friday-23-02-2018 | 0 | 3,754 | 0 | 7,662 | 2,614 | 0 | 10 |
| Thuesday-20-02-2018 | 0 | 36,767 | 0 | 82,139 | 2 | 0 | 10 |
| Thursday-01-03-2018 | 0 | 1,834 | 0 | 4,004 | 97 | 25 | 0 |
| Thursday-15-02-2018 | 0 | 4,921 | 0 | 11,133 | 2,421 | 0 | 10 |
| Thursday-22-02-2018 | 0 | 3,569 | 0 | 7,651 | 3,278 | 0 | 10 |
| Wednesday-14-02-2018 | 0 | 2,277 | 0 | 5,371 | 225,628 | 0 | 10 |
| Wednesday-21-02-2018 | 0 | 0 | 0 | 0 | 17,557 | 0 | 10 |
| Wednesday-28-02-2018 | 0 | 4,041 | 0 | 8,297 | 6,121 | 33 | 0 |

### Every observed label by file

| File ID | Observed label | Count | Group |
| --- | --- | --- | --- |
| Friday-02-03-2018 | Benign | 762,384 | benign |
| Friday-02-03-2018 | Bot | 286,191 | malicious |
| Friday-16-02-2018 | Benign | 446,772 | benign |
| Friday-16-02-2018 | DoS attacks-Hulk | 461,912 | malicious |
| Friday-16-02-2018 | DoS attacks-SlowHTTPTest | 139,890 | malicious |
| Friday-16-02-2018 | Label | 1 | unclassified |
| Friday-23-02-2018 | Benign | 1,048,009 | benign |
| Friday-23-02-2018 | Brute Force -Web | 362 | malicious |
| Friday-23-02-2018 | Brute Force -XSS | 151 | malicious |
| Friday-23-02-2018 | SQL Injection | 53 | malicious |
| Thuesday-20-02-2018 | Benign | 7,372,557 | benign |
| Thuesday-20-02-2018 | DDoS attacks-LOIC-HTTP | 576,191 | malicious |
| Thursday-01-03-2018 | Benign | 238,037 | benign |
| Thursday-01-03-2018 | Infilteration | 93,063 | malicious |
| Thursday-01-03-2018 | Label | 25 | unclassified |
| Thursday-15-02-2018 | Benign | 996,077 | benign |
| Thursday-15-02-2018 | DoS attacks-GoldenEye | 41,508 | malicious |
| Thursday-15-02-2018 | DoS attacks-Slowloris | 10,990 | malicious |
| Thursday-22-02-2018 | Benign | 1,048,213 | benign |
| Thursday-22-02-2018 | Brute Force -Web | 249 | malicious |
| Thursday-22-02-2018 | Brute Force -XSS | 79 | malicious |
| Thursday-22-02-2018 | SQL Injection | 34 | malicious |
| Wednesday-14-02-2018 | Benign | 667,626 | benign |
| Wednesday-14-02-2018 | FTP-BruteForce | 193,360 | malicious |
| Wednesday-14-02-2018 | SSH-Bruteforce | 187,589 | malicious |
| Wednesday-21-02-2018 | Benign | 360,833 | benign |
| Wednesday-21-02-2018 | DDOS attack-HOIC | 686,012 | malicious |
| Wednesday-21-02-2018 | DDOS attack-LOIC-UDP | 1,730 | malicious |
| Wednesday-28-02-2018 | Benign | 544,200 | benign |
| Wednesday-28-02-2018 | Infilteration | 68,871 | malicious |
| Wednesday-28-02-2018 | Label | 33 | unclassified |

### All class totals

| Observed label | Count | Group |
| --- | --- | --- |
| Benign | 13,484,708 | benign |
| Bot | 286,191 | malicious |
| Brute Force -Web | 611 | malicious |
| Brute Force -XSS | 230 | malicious |
| DDOS attack-HOIC | 686,012 | malicious |
| DDOS attack-LOIC-UDP | 1,730 | malicious |
| DDoS attacks-LOIC-HTTP | 576,191 | malicious |
| DoS attacks-GoldenEye | 41,508 | malicious |
| DoS attacks-Hulk | 461,912 | malicious |
| DoS attacks-SlowHTTPTest | 139,890 | malicious |
| DoS attacks-Slowloris | 10,990 | malicious |
| FTP-BruteForce | 193,360 | malicious |
| Infilteration | 161,934 | malicious |
| Label | 59 | unclassified |
| SQL Injection | 87 | malicious |
| SSH-Bruteforce | 187,589 | malicious |

Original label spelling and capitalization are retained, including Infilteration.

The following inventory lists every source column and its content category. Exact inferred types by file are produced by the documented auditor in column_inventory.csv and the summary data_types mapping. Regenerate the current legacy reports before relying on those type fields.

### Complete column inventory

| Column name | Files containing it | Content category |
| --- | --- | --- |
| Dst Port | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Protocol | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Timestamp | 10 | Datetime; mixed datetime/string in files with repeated headers |
| Flow Duration | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Tot Fwd Pkts | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Tot Bwd Pkts | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| TotLen Fwd Pkts | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| TotLen Bwd Pkts | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Fwd Pkt Len Max | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Fwd Pkt Len Min | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Fwd Pkt Len Mean | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Fwd Pkt Len Std | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Bwd Pkt Len Max | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Bwd Pkt Len Min | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Bwd Pkt Len Mean | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Bwd Pkt Len Std | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Flow Byts/s | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Flow Pkts/s | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Flow IAT Mean | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Flow IAT Std | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Flow IAT Max | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Flow IAT Min | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Fwd IAT Tot | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Fwd IAT Mean | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Fwd IAT Std | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Fwd IAT Max | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Fwd IAT Min | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Bwd IAT Tot | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Bwd IAT Mean | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Bwd IAT Std | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Bwd IAT Max | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Bwd IAT Min | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Fwd PSH Flags | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Bwd PSH Flags | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Fwd URG Flags | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Bwd URG Flags | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Fwd Header Len | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Bwd Header Len | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Fwd Pkts/s | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Bwd Pkts/s | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Pkt Len Min | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Pkt Len Max | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Pkt Len Mean | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Pkt Len Std | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Pkt Len Var | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| FIN Flag Cnt | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| SYN Flag Cnt | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| RST Flag Cnt | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| PSH Flag Cnt | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| ACK Flag Cnt | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| URG Flag Cnt | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| CWE Flag Count | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| ECE Flag Cnt | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Down/Up Ratio | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Pkt Size Avg | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Fwd Seg Size Avg | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Bwd Seg Size Avg | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Fwd Byts/b Avg | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Fwd Pkts/b Avg | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Fwd Blk Rate Avg | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Bwd Byts/b Avg | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Bwd Pkts/b Avg | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Bwd Blk Rate Avg | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Subflow Fwd Pkts | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Subflow Fwd Byts | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Subflow Bwd Pkts | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Subflow Bwd Byts | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Init Fwd Win Byts | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Init Bwd Win Byts | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Fwd Act Data Pkts | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Fwd Seg Size Min | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Active Mean | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Active Std | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Active Max | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Active Min | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Idle Mean | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Idle Std | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Idle Max | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Idle Min | 10 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Label | 10 | String |
| Flow ID | 1 | String |
| Src IP | 1 | String |
| Src Port | 1 | Numeric (integer or float by file); mixed numeric/string where headers occur |
| Dst IP | 1 | String |


## Timestamp and constant-column findings

The February 14 source contains `05/01/1970 03:01:17` at line 410958. February 22 contains `10/01/1970 03:04:26` at line 246435. Both were checked directly in the source text. They remain in the timestamp ranges and are not parse failures. A successful parse does not establish chronological correctness.

These ten columns are strictly constant in the seven files without repeated headers:

- `Bwd PSH Flags`
- `Fwd URG Flags`
- `Bwd URG Flags`
- `CWE Flag Count`
- `Fwd Byts/b Avg`
- `Fwd Pkts/b Avg`
- `Fwd Blk Rate Avg`
- `Bwd Byts/b Avg`
- `Bwd Pkts/b Avg`
- `Bwd Blk Rate Avg`

February 16, February 28, and March 1 have no strictly constant columns because embedded headers introduce an additional value into every column. The inventory's additional diagnostic identifies otherwise constant nonmissing values without removing those records.

## Temporal interaction graph feasibility

| Required field | Availability | Source columns |
| --- | --- | --- |
| Source IP or equivalent source identifier | February 20 only | `Src IP` |
| Destination IP or equivalent destination identifier | February 20 only | `Dst IP` |
| Source port | February 20 only | `Src Port` |
| Destination port | All ten files | `Dst Port` |
| Timestamp | All ten files | `Timestamp` |
| Protocol | All ten files | `Protocol` |
| Flow duration | All ten files | `Flow Duration` |
| Packet counts | All ten files | `Tot Fwd Pkts`, `Tot Bwd Pkts`, and subflow packet counts |
| Byte counts | All ten files | `TotLen Fwd Pkts`, `TotLen Bwd Pkts`, `Subflow Fwd Byts`, `Subflow Bwd Byts` |
| Attack label | All ten files | `Label` |

February 20 also contains `Flow ID`. Rates, average packet lengths and TCP window sizes are not substituted for byte totals. Ports and flow statistics alone do not identify network hosts.

**The available columns are insufficient for temporal source-to-destination interaction graphs across all ten files.** Only `Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv` is structurally sufficient because it preserves both endpoints and timestamps. It contains Benign and DDoS attacks-LOIC-HTTP records, so using it alone narrows class coverage.

The other nine files require additional endpoint-preserving flow data or corresponding captures from which endpoints can be obtained. Schema sufficiency does not establish timestamp correctness or replace a future data-quality policy. This audit has not constructed graphs, prepared training data, or trained models.


