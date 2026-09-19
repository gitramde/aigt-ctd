# Dataset audit methodology

Scope: every CSV record in the ten files under `data/raw/Processed Traffic Data for ML Algorithms`. The source files are opened read-only. No cleaning, imputation, normalization, sampling, splitting, deduplication, or transformed dataset is performed or exported.

## Report definitions

- `dataset_summary.csv`: one row per file. File sizes are provided in bytes and MiB (1,048,576 bytes). Row counts exclude the initial CSV header but include repeated headers and any other anomalous records. Column names, types, class counts, and constant-column lists use JSON inside CSV cells.
- `class_by_file.csv`: exact observed label strings and counts. `Benign` is benign; empty/missing-marker labels and the repeated-header token `Label` are unclassified. Other observed nonempty labels are malicious. Unclassified records are retained and reported separately so benign + malicious + unclassified equals the full row count.
- `column_inventory.csv`: one row per file/column in source order. CSV has text storage; `data_type` describes the content inferred across the entire file, ignoring blank/NaN/null tokens but retaining repeated header tokens. Consequently, a numeric column contaminated by a repeated header is reported as mixed numeric/string. Integer denotes integral finite numeric values, not a guaranteed native storage width. No exact distinct-value count is claimed.
- `data_quality_report.csv`: `scope=file` rows contain file totals; `scope=column` rows contain column counts. Do not sum both scopes together. Duplicate counts appear only at file scope.

## Quality measures

- Missing values are empty or whitespace-only fields.
- NaN values are explicit `NaN`, `+NaN`, or `-NaN` tokens, matched without regard to case and surrounding whitespace. They are reported separately from empty fields.
- Null markers are `null`, `none`, `na`, `n/a`, `#n/a`, or `<na>`, matched without regard to case and surrounding whitespace. They are reported separately from blanks and NaNs.
- Infinite values are values parsed as positive or negative infinity, including `Infinity`/`inf` spellings. Counts are cells, not rows.
- Duplicate records are extra occurrences after the first identical complete record, **within each file**, including duplicate anomalous records. Equality compares every original parsed CSV field without numeric coercion or whitespace normalization. A 128-bit BLAKE2 fingerprint locates candidates, followed by a second read that compares complete field tuples. This verifies exact equality even in the event of hash collisions. Cross-file duplicates are not included in per-file counts.
- `is_constant`/`constant_column` means all original values in the column are exactly identical, including missing tokens and repeated header tokens. `constant_value` gives that original value. The additional `constant_non_missing_excluding_header_tokens` flag identifies columns with one observed value after analytically disregarding missing tokens and tokens equal to the column name; this is a diagnostic only and does not remove or alter records.
- Repeated headers are records whose entire field sequence equals the initial header. They remain included in all raw counts and duplicate checks.

## Timestamps and graph fields

Timestamps are interpreted strictly as `DD/MM/YYYY HH:MM:SS`, consistent with unambiguous February dates and the file naming convention. Ranges cover parseable timestamps only; nonempty, nonmissing unparseable values are counted as invalid. The CSVs do not specify a timezone. No timezone is inferred or converted. Parsing for audit statistics does not modify source values.

Some syntactically valid dates are anomalous: the February 14 file contains `05/01/1970 03:01:17` (source line 410958), and the February 22 file contains `10/01/1970 03:04:26` (source line 246435). These were checked directly in the source text. Such values remain in the timestamp ranges and are not counted as parsing failures. A successful parse therefore does not establish that a timestamp is chronologically correct.

Graph-presence flags describe columns available in each original file. Packet counts include `Tot Fwd Pkts` and `Tot Bwd Pkts`. Byte totals include `TotLen Fwd Pkts` and `TotLen Bwd Pkts`, plus subflow byte totals. Rates, packet lengths, and TCP window sizes are not substituted for byte totals. Ports and flow statistics alone do not identify network hosts.

The graph-sufficiency flag requires a source identifier, destination identifier, and timestamp column. It addresses schema sufficiency, not completion of graph construction or approval of a data-cleaning policy. Only `Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv` has `Src IP`, `Dst IP`, `Src Port`, and `Flow ID`. The other nine files cannot independently support host-level source-to-destination interaction graphs from their available columns.

## Reproduction and verification

Run from the project root:

```powershell
python src/data/audit_full_dataset.py --input-dir 'data/raw/Processed Traffic Data for ML Algorithms' --output-dir results
python -m unittest discover -s src/data -p test_audit_full_dataset.py
python src/data/validate_dataset_audit.py
```

The audit uses Python and NumPy for full-batch numeric inspection, validates each record's column count, reconciles label totals, and checks source size and modification time before and after each file. Synthetic correctness checks cover duplicates across batches, forced hash collisions, day-first timestamps, repeated headers, mixed types, blanks, NaNs, null markers, and infinities. Independent binary line counts provide a second check of row totals for these files.
