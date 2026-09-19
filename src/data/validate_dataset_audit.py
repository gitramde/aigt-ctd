"""Independent report reconciliation, including source binary line counts."""
import csv
import json
from collections import Counter
from pathlib import Path


def read(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def main():
    root = Path(__file__).resolve().parents[2]
    summary, classes, inventory, quality = [read(root / 'results' / name) for name in (
        'dataset_summary.csv', 'class_by_file.csv', 'column_inventory.csv', 'data_quality_report.csv')]
    sources = {p.name: p for p in (root / 'data').rglob('*.csv')}
    assert len(summary) == len(sources) == 10
    assert {r['filename'] for r in summary} == set(sources)
    for rows in (classes, inventory, quality):
        assert {r['filename'] for r in rows} == set(sources)
    totals = Counter()
    all_labels = Counter()
    for row in summary:
        filename = row['filename']
        path = sources[filename]
        with path.open('rb') as stream:
            physical_lines = 0
            last = b''
            while block := stream.read(8 * 1024 * 1024):
                physical_lines += block.count(b'\n')
                last = block[-1:]
            physical_lines += int(last != b'\n')
        assert physical_lines - 1 == int(row['row_count']), filename
        assert path.stat().st_size == int(row['file_size_bytes'])
        with path.open(newline='', encoding='utf-8-sig') as stream:
            header = next(csv.reader(stream))
        columns = [r for r in inventory if r['filename'] == filename]
        assert [r['column_name'] for r in columns] == header == json.loads(row['column_names'])
        assert len(columns) == int(row['column_count'])
        assert {r['column_name']: r['data_type'] for r in columns} == json.loads(row['data_types'])
        label_rows = [r for r in classes if r['filename'] == filename]
        assert {r['label']: int(r['count']) for r in label_rows} == json.loads(row['class_labels'])
        assert sum(int(r['count']) for r in label_rows) == int(row['row_count'])
        for group in ('benign', 'malicious', 'unclassified'):
            assert sum(int(r['count']) for r in label_rows if r['class_group'] == group) == int(row[group + '_count'])
        file_quality = [r for r in quality if r['filename'] == filename and r['scope'] == 'file']
        col_quality = [r for r in quality if r['filename'] == filename and r['scope'] == 'column']
        assert len(file_quality) == 1 and len(col_quality) == len(header)
        for measure in ('missing_values', 'nan_values', 'null_marker_values', 'infinite_values'):
            assert sum(int(r[measure]) for r in col_quality) == int(row[measure]) == int(file_quality[0][measure])
        timestamp_quality = next(r for r in col_quality if r['column_name'] == 'Timestamp')
        assert int(row['valid_timestamp_values']) + int(row['invalid_timestamp_values']) + sum(int(timestamp_quality[k]) for k in ('missing_values', 'nan_values', 'null_marker_values')) == int(row['row_count'])
        assert json.loads(row['constant_columns']) == [r['column_name'] for r in columns if r['is_constant'] == 'True']
        assert int(row['duplicate_records']) == int(file_quality[0]['duplicate_records'])
        for key in ('row_count', 'benign_count', 'malicious_count', 'unclassified_count', 'missing_values', 'nan_values', 'null_marker_values', 'infinite_values', 'duplicate_records', 'repeated_header_records'):
            totals[key] += int(row[key])
        all_labels.update({r['label']: int(r['count']) for r in label_rows})
        print(f'PASS {filename}: {row["row_count"]} rows', flush=True)
    print(json.dumps({'totals': totals, 'labels': all_labels}, indent=2))


if __name__ == '__main__':
    main()
