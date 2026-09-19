"""Read-only full CSV audit. NumPy accelerates whole-batch numeric inspection."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import islice
from pathlib import Path
import numpy as np

NAN = {'nan', '+nan', '-nan'}
NULL = {'null', 'none', 'na', 'n/a', '#n/a', '<na>'}
ROLES = {
    'source_identifier': {'src ip', 'source ip', 'src_ip', 'source_ip'},
    'destination_identifier': {'dst ip', 'destination ip', 'dst_ip', 'destination_ip'},
    'source_port': {'src port', 'source port'},
    'destination_port': {'dst port', 'destination port'},
    'timestamp': {'timestamp'}, 'protocol': {'protocol'},
    'flow_duration': {'flow duration'},
    'packet_counts': {'tot fwd pkts', 'tot bwd pkts', 'subflow fwd pkts', 'subflow bwd pkts'},
    'byte_counts': {'totlen fwd pkts', 'totlen bwd pkts', 'subflow fwd byts', 'subflow bwd byts'},
    'attack_label': {'label'},
}


def fingerprint(row):
    if any('\x1f' in value for value in row):
        raise ValueError('Unexpected fingerprint separator in CSV field')
    return hashlib.blake2b('\x1f'.join(row).encode('utf-8'), digest_size=16).digest()


def label_group(label):
    if label.strip().casefold() == 'benign':
        return 'benign'
    if label.strip().casefold() in {'', 'label'} | NAN | NULL:
        return 'unclassified'
    return 'malicious'


def audit_file(path, chunk_size):
    before = path.stat()
    seen, candidates = set(), set()
    states, labels = [], Counter()
    count = headers = 0
    tmin = tmax = None
    invalid_time = valid_time = 0
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.reader(stream)
        columns = next(reader)
        for _ in columns:
            states.append(dict(missing=0, nan=0, null=0, infinite=0, finite=0,
                               text=0, fractional=False, values=set(), constant_values=set()))
        label_index = columns.index('Label')
        while batch := list(islice(reader, chunk_size)):
            for row in batch:
                if len(row) != len(columns):
                    raise ValueError(f'{path.name}: record {count + 1} has {len(row)} fields')
                count += 1
                headers += row == columns
                digest = fingerprint(row)
                if digest in seen:
                    candidates.add(digest)
                else:
                    seen.add(digest)
                labels[row[label_index]] += 1
            for column, state, values in zip(columns, states, zip(*batch)):
                frequencies = Counter(values)
                # Inspect all numeric values with native array operations. This
                # changes only computation speed, not source values or counts.
                try:
                    numbers = np.asarray(list(frequencies), dtype=np.float64)
                except ValueError:
                    numbers = None
                if numbers is not None:
                    weights = np.asarray(list(frequencies.values()), dtype=np.int64)
                    nan = np.isnan(numbers)
                    infinite = np.isinf(numbers)
                    finite = np.isfinite(numbers)
                    state['nan'] += int(weights[nan].sum())
                    state['infinite'] += int(weights[infinite].sum())
                    state['finite'] += int(weights[finite].sum())
                    if not state['fractional']:
                        state['fractional'] = bool(np.any(numbers[finite] % 1 != 0))
                    for raw in frequencies:
                        if len(state['values']) >= 2:
                            break
                        state['values'].add(raw)
                    if len(state['constant_values']) < 2:
                        for raw, is_nan in zip(frequencies, nan):
                            if not is_nan:
                                state['constant_values'].add(raw)
                            if len(state['constant_values']) >= 2:
                                break
                    continue
                for raw, n in frequencies.items():
                    value = raw.strip()
                    lower = value.casefold()
                    if len(state['values']) < 2:
                        state['values'].add(raw)
                    if value == '':
                        state['missing'] += n
                        continue
                    if lower in NAN:
                        state['nan'] += n
                        continue
                    if lower in NULL:
                        state['null'] += n
                        continue
                    if len(state['constant_values']) < 2 and raw != column:
                        state['constant_values'].add(raw)
                    try:
                        number = float(value)
                    except ValueError:
                        state['text'] += n
                    else:
                        if math.isinf(number):
                            state['infinite'] += n
                        else:
                            state['finite'] += n
                            state['fractional'] |= not number.is_integer()
                    if column == 'Timestamp':
                        try:
                            parsed = datetime.strptime(value, '%d/%m/%Y %H:%M:%S')
                        except ValueError:
                            invalid_time += n
                        else:
                            valid_time += n
                            tmin = parsed if tmin is None else min(tmin, parsed)
                            tmax = parsed if tmax is None else max(tmax, parsed)
            if count % 500_000 == 0:
                print(f'  {path.name}: {count:,} records scanned', flush=True)
    del seen
    # Hashes identify candidates only. Complete original field tuples establish
    # exact duplicate counts, so hash collisions cannot inflate the result.
    repeated = Counter()
    if candidates:
        print('  Verifying duplicate candidates against full records', flush=True)
        with path.open(newline='', encoding='utf-8-sig') as stream:
            reader = csv.reader(stream)
            next(reader)
            for row in reader:
                if fingerprint(row) in candidates:
                    repeated[tuple(row)] += 1
    duplicates = sum(n - 1 for n in repeated.values())
    duplicate_groups = sum(n > 1 for n in repeated.values())
    inventory, quality = [], []
    for position, (column, state) in enumerate(zip(columns, states), 1):
        if column == 'Timestamp':
            dtype = 'datetime (DD/MM/YYYY HH:MM:SS)' if invalid_time == 0 else 'mixed datetime/string'
        elif state['text']:
            dtype = 'mixed numeric/string' if state['finite'] or state['infinite'] else 'string'
        elif state['finite'] or state['infinite']:
            dtype = 'float64' if state['fractional'] or state['infinite'] or state['nan'] else 'integer'
        else:
            dtype = 'missing-only'
        roles = {f'is_{role}': column.strip().casefold() in names for role, names in ROLES.items()}
        inventory.append(dict(filename=path.name, column_position=position, column_name=column,
                              data_type=dtype, storage_type='CSV text',
                              is_constant=len(state['values']) == 1,
                              constant_value=next(iter(state['values'])) if len(state['values']) == 1 else '',
                              constant_non_missing_excluding_header_tokens=len(state['constant_values']) == 1,
                              **roles))
        quality.append(dict(filename=path.name, scope='column', column_name=column,
                            missing_values=state['missing'], nan_values=state['nan'],
                            null_marker_values=state['null'], infinite_values=state['infinite'],
                            constant_column=len(state['values']) == 1,
                            invalid_timestamp_values=invalid_time if column == 'Timestamp' else '',
                            duplicate_records='', repeated_header_records=''))
    presence = {f'has_{role}': any(row[f'is_{role}'] for row in inventory) for role in ROLES}
    groups = Counter()
    for label, n in labels.items():
        groups[label_group(label)] += n
    summary = dict(filename=path.name, file_size_bytes=before.st_size,
                   file_size_mib=round(before.st_size / 1024**2, 3), row_count=count,
                   column_count=len(columns), column_names=json.dumps(columns),
                   data_types=json.dumps({row['column_name']: row['data_type'] for row in inventory}),
                   timestamp_min=tmin.isoformat() if tmin else '', timestamp_max=tmax.isoformat() if tmax else '',
                   timestamp_format='%d/%m/%Y %H:%M:%S', timestamp_timezone='unspecified in source',
                   valid_timestamp_values=valid_time, invalid_timestamp_values=invalid_time,
                   class_labels=json.dumps(dict(sorted(labels.items()))), benign_count=groups['benign'],
                   malicious_count=groups['malicious'], unclassified_count=groups['unclassified'],
                   missing_values=sum(s['missing'] for s in states), nan_values=sum(s['nan'] for s in states),
                   null_marker_values=sum(s['null'] for s in states), infinite_values=sum(s['infinite'] for s in states),
                   duplicate_records=duplicates, duplicate_groups=duplicate_groups, repeated_header_records=headers,
                   constant_columns=json.dumps([r['column_name'] for r in inventory if r['is_constant']]),
                   **presence,
                   temporal_endpoint_graph_columns_sufficient=presence['has_source_identifier'] and presence['has_destination_identifier'] and presence['has_timestamp'])
    quality.insert(0, dict(filename=path.name, scope='file', column_name='',
                           missing_values=summary['missing_values'], nan_values=summary['nan_values'],
                           null_marker_values=summary['null_marker_values'], infinite_values=summary['infinite_values'],
                           constant_column='', invalid_timestamp_values=invalid_time,
                           duplicate_records=duplicates, repeated_header_records=headers))
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'Source changed during audit'
    assert sum(labels.values()) == count
    assert sum(groups.values()) == count
    print(f'  DONE: {count:,} rows; {duplicates:,} exact duplicates; {headers} repeated headers', flush=True)
    return summary, [dict(filename=path.name, label=label, count=n, class_group=label_group(label)) for label, n in sorted(labels.items())], inventory, quality


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--chunk-size', type=int, default=10_000)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--resume', action='store_true', help='Reuse completed reports from an interrupted audit')
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob('*.csv'))
    if len(paths) != 10:
        raise ValueError(f'Expected 10 files, found {len(paths)}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = [[], [], [], []]
    names = ['dataset_summary.csv', 'class_by_file.csv', 'column_inventory.csv', 'data_quality_report.csv']
    if args.resume:
        for report, name in zip(reports, names):
            with (args.output_dir / name).open(newline='', encoding='utf-8') as stream:
                report.extend(csv.DictReader(stream))
        completed = {r['filename'] for r in reports[0]}
        for path in paths:
            if path.name in completed:
                saved = next(r for r in reports[0] if r['filename'] == path.name)
                assert path.stat().st_size == int(saved['file_size_bytes'])
        print(f'Reusing {len(completed)} completed file audits', flush=True)
        paths = [p for p in paths if p.name not in completed]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(audit_file, path, args.chunk_size): path for path in sorted(paths, key=lambda p: -p.stat().st_size)}
        for future in as_completed(pending):
            print(f'Completed: {pending[future].name}', flush=True)
            for report, rows in zip(reports, future.result()):
                report.extend(rows if isinstance(rows, list) else [rows])
            for name, rows in zip(names, reports):
                rows.sort(key=lambda row: row['filename'])
                write_csv(args.output_dir / name, rows)
    print('All 10 files audited; four reports written.', flush=True)


if __name__ == '__main__':
    main()
