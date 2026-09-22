"""Exact, source-preserving cleaning with an individual audit of every exclusion."""
import argparse
import csv
import gc
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .common import ROOT, csv_write, file_date, json_write, load_config, require, sha256, target_alias, versions

META = ['source_file', 'source_row', 'timestamp', 'file_date', 'attack_label', 'target_binary']


def digest_record(row):
    # Digest collisions (including delimiter collisions) are resolved by full tuple equality.
    return hashlib.blake2b('\x1f'.join(row).encode('utf-8'), digest_size=16).digest()


def decode_record(line):
    rows = list(csv.reader([line.decode('utf-8').rstrip('\r\n')], strict=True))
    require(len(rows) == 1, 'Expected one physical line per CSV record.')
    return tuple(rows[0])


def make_table(rows, source_rows, timestamps, header, features, config, filename, quality):
    positions = {c: i for i, c in enumerate(header)}
    arrays, names = [], []
    for name in features:
        values = [row[positions[name]] for row in rows]
        arr = pa.array(values, type=pa.string())
        null = pc.is_in(pc.utf8_lower(pc.utf8_trim_whitespace(arr)),
                        value_set=pa.array(['', 'nan', '+nan', '-nan', 'null', 'none', 'na', 'n/a', '#n/a', '<na>']))
        arr = pc.if_else(null, pa.scalar(None, type=pa.string()), arr)
        if name not in config['categorical_columns']:
            try:
                arr = pc.cast(arr, pa.float64())
            except pa.ArrowInvalid as exc:
                raise ValueError(f'Unexpected nonnumeric feature {filename}/{name}; refusing silent coercion') from exc
            positive = pc.equal(arr, float('inf'))
            negative = pc.equal(arr, -float('inf'))
            pos, neg = (pc.sum(x).as_py() or 0 for x in (positive, negative))
            if pos or neg:
                require(name in config['infinity_to_nan'], f'Unexpected infinity outside configured columns: {name}')
                quality[(name, 'positive_infinity_to_nan')] += pos
                quality[(name, 'negative_infinity_to_nan')] += neg
                arr = pc.if_else(pc.or_(positive, negative), pa.scalar(None, type=pa.float64()), arr)
            arr = pc.if_else(pc.is_nan(arr), pa.scalar(None, type=pa.float64()), arr)
        quality[(name, 'missing_after_cleaning')] += arr.null_count
        arrays.append(arr)
        names.append(name)
    labels = [row[positions['Label']] for row in rows]
    arrays += [pa.array([filename]*len(rows)), pa.array(source_rows, type=pa.int64()),
               pa.array(timestamps, type=pa.timestamp('us')), pa.array([file_date(filename)]*len(rows), type=pa.date32()),
               pa.array(labels), pa.array([int(label != config['benign_label']) for label in labels], type=pa.int8())]
    return pa.Table.from_arrays(arrays, names=names + META)


def clean_file(path, config, features, audit_writer, source_inventory):
    before = path.stat()
    filename, date = path.name, file_date(path)
    folder = Path(config['data_dir']) / 'cleaned'
    staging = Path(config['data_dir']) / 'staging' / path.stem
    staging.mkdir(parents=True, exist_ok=False)
    seen, cache, hourly_writers = {}, {}, {}
    rows, row_ids, datetimes = [], [], []
    counts, quality, distributions = Counter(), Counter(), Counter()
    quarantined_writer = None
    source_hash = hashlib.sha256()
    quarantine_path = folder / (path.stem + '.non_temporal.parquet')

    def flush():
        nonlocal quarantined_writer
        if not rows:
            return
        table = make_table(rows, row_ids, datetimes, header, features, config, filename, quality)
        eligible = pa.array([dt is not None and dt.date() == date for dt in datetimes])
        quarantine = table.filter(pc.invert(eligible))
        if len(quarantine):
            if quarantined_writer is None:
                quarantined_writer = pq.ParquetWriter(quarantine_path, quarantine.schema, compression='zstd')
            quarantined_writer.write_table(quarantine)
        table = table.filter(eligible)
        hours = pc.hour(table['timestamp'])
        for hour in sorted(pc.unique(hours).to_pylist()):
            part = table.filter(pc.equal(hours, hour))
            if hour not in hourly_writers:
                hourly_writers[hour] = pq.ParquetWriter(staging / f'{hour:02d}.parquet', part.schema, compression='zstd')
            hourly_writers[hour].write_table(part)
        rows.clear(); row_ids.clear(); datetimes.clear()

    with path.open('rb') as stream, path.open('rb') as lookup:
        first_line = stream.readline()
        source_hash.update(first_line)
        header = tuple(next(csv.reader([first_line.decode('utf-8-sig').rstrip('\r\n')], strict=True)))
        label_index, time_index = header.index('Label'), header.index('Timestamp')
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            source_hash.update(line)
            row = decode_record(line)
            counts['raw_rows'] += 1
            row_number = counts['raw_rows']
            require(len(row) == len(header), f'{filename}:{row_number}: unexpected field count')
            label, raw_time = row[label_index], row[time_index]
            distributions[(label, 'raw')] += 1
            event = dict(source_file=filename, source_row=row_number, source_line=row_number+1,
                         byte_offset=offset, file_date=str(date), attack_label=label, raw_timestamp=raw_time,
                         reason='', disposition='', retained_source_row='')
            if row == header:
                counts['repeated_headers'] += 1
                audit_writer.writerow(dict(event, reason='repeated_header', disposition='remove'))
                continue
            require(label and label != 'Label', f'Unexpected target token {filename}:{row_number}')
            fingerprint = digest_record(row)
            bucket = seen.get(fingerprint)
            duplicate_of = None
            if bucket is not None:
                offsets = bucket if isinstance(bucket, list) else [bucket]
                for first_offset, first_row in offsets:
                    lookup.seek(first_offset)
                    if decode_record(lookup.readline()) == row:
                        duplicate_of = first_row
                        break
                if duplicate_of is None:
                    seen[fingerprint] = offsets + [(offset, row_number)]
            else:
                seen[fingerprint] = (offset, row_number)
            if duplicate_of is not None:
                counts['exact_nonheader_duplicates'] += 1
                audit_writer.writerow(dict(event, reason='exact_duplicate', disposition='remove', retained_source_row=duplicate_of))
                continue
            if raw_time not in cache:
                try:
                    cache[raw_time] = datetime.strptime(raw_time, '%d/%m/%Y %H:%M:%S')
                except ValueError:
                    cache[raw_time] = None
            dt = cache[raw_time]
            counts['cleaned_rows'] += 1
            distributions[(label, 'cleaned_before_temporal_exclusion')] += 1
            if dt is None or dt.date() != date:
                reason = 'year_1970' if dt is not None and dt.year == 1970 else 'invalid_timestamp' if dt is None else 'date_mismatch'
                counts['temporal_exclusions'] += 1
                counts[reason + '_temporal_exclusions'] += 1
                audit_writer.writerow(dict(event, reason=reason, disposition='retain_in_non_temporal_quarantine'))
            else:
                counts['temporal_rows'] += 1
                distributions[(label, 'temporal_eligible')] += 1
            rows.append(row); row_ids.append(row_number); datetimes.append(dt)
            if len(rows) >= config['batch_size']:
                flush()
            if row_number % 500000 == 0:
                print(f'Clean {filename}: {row_number:,} source rows', flush=True)
        flush()
    for writer in hourly_writers.values():
        writer.close()
    if quarantined_writer:
        quarantined_writer.close()
    del seen, cache
    gc.collect()
    output = folder / (path.stem + '.parquet')
    final_writer = None
    last_time = None
    for hour in sorted(hourly_writers):
        fragment = staging / f'{hour:02d}.parquet'
        table = pq.read_table(fragment).sort_by([('timestamp', 'ascending'), ('source_row', 'ascending')])
        first = table['timestamp'][0].as_py()
        require(last_time is None or last_time <= first, 'Chronological sort failed')
        last_time = table['timestamp'][-1].as_py()
        if final_writer is None:
            final_writer = pq.ParquetWriter(output, table.schema, compression='zstd')
        final_writer.write_table(table, row_group_size=config['batch_size'])
        del table
        # Only delete this run's explicitly created intermediate, never source files.
        require(fragment.resolve().is_relative_to(Path(config['data_dir']).resolve() / 'staging'), 'Unsafe staging path')
        fragment.unlink()
        pa.default_memory_pool().release_unused()
    require(final_writer is not None, f'No temporal records in {filename}')
    final_writer.close()
    staging.rmdir()
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), f'Source changed: {filename}')
    require(counts['raw_rows'] == counts['repeated_headers'] + counts['exact_nonheader_duplicates'] + counts['cleaned_rows'], 'Cleaning count mismatch')
    require(counts['cleaned_rows'] == counts['temporal_rows'] + counts['temporal_exclusions'], 'Temporal count mismatch')
    inventory = dict(filename=filename, date=str(date), raw_path=str(path), raw_bytes=before.st_size,
                     raw_mtime_ns=before.st_mtime_ns, raw_sha256=source_hash.hexdigest(),
                     cleaned_path=str(output), cleaned_sha256=sha256(output), counts=dict(counts))
    if quarantined_writer:
        inventory.update(non_temporal_path=str(quarantine_path), non_temporal_sha256=sha256(quarantine_path))
    source_inventory.append(inventory)
    print(f'DONE {filename}: {counts["temporal_rows"]:,} temporal rows; {counts["exact_nonheader_duplicates"]:,} duplicates removed', flush=True)
    return counts, quality, distributions


def run(config):
    raw, data, reports = (Path(config[k]) for k in ('raw_dir', 'data_dir', 'report_dir'))
    require(not data.exists(), f'Derived output already exists: {data}. Use a new output version; no automatic overwrite.')
    require(not reports.exists(), f'Report output already exists: {reports}. Use a new output version.')
    paths = sorted(raw.glob('*.csv'), key=lambda p: (file_date(p), p.name))
    require(len(paths) == config['expected_raw_files'], 'Unexpected raw file count')
    require(len({file_date(p) for p in paths}) == len(paths), 'This whole-date protocol requires one source file per date.')
    headers = {}
    for path in paths:
        with path.open(newline='', encoding='utf-8-sig') as f:
            headers[path.name] = next(csv.reader(f))
    common = set.intersection(*(set(header) for header in headers.values()))
    features = [c for c in headers[paths[0].name] if c in common and c not in config['metadata_only_columns'] and not target_alias(c)]
    require(bool(features), 'No eligible features')
    exclusions = []
    for name in sorted(set.union(*(set(header) for header in headers.values()))):
        if name not in features:
            reason = 'direct_target_or_target_named_field' if target_alias(name) else 'metadata_or_identifier' if name in config['metadata_only_columns'] else 'not_in_all_source_schemas'
            exclusions.append(dict(column=name, stage='schema_before_split', decision='exclude', reason=reason))
    exclusions.extend(dict(column=name, stage='generated_metadata', decision='exclude',
                           reason='derived_target' if name in ('attack_label','target_binary') else 'generated_provenance')
                      for name in META)
    (data / 'cleaned').mkdir(parents=True)
    reports.mkdir(parents=True)
    json_write(reports / 'run_config.json', config)
    json_write(reports / 'environment.json', versions())
    csv_write(reports / 'schema_exclusions.csv', exclusions)
    inventory, file_counts, cell_quality, distributions = [], [], [], []
    fields = ['source_file', 'source_row', 'source_line', 'byte_offset', 'file_date', 'attack_label', 'raw_timestamp', 'reason', 'disposition', 'retained_source_row']
    with (reports / 'cleaning_audit.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for path in paths:
            counts, quality, classes = clean_file(path, config, features, writer, inventory)
            file_counts.append(dict(filename=path.name, date=str(file_date(path)), **counts))
            cell_quality.extend(dict(filename=path.name, column=col, operation=op, cells=count) for (col, op), count in sorted(quality.items()))
            distributions.extend(dict(attack_class=label, filename=path.name, file_date=str(file_date(path)), stage=stage, records=count)
                                 for (label, stage), count in sorted(classes.items()))
            json_write(reports / 'source_manifest.in_progress.json', inventory)
    totals = Counter()
    for row in file_counts:
        totals.update({k:v for k,v in row.items() if isinstance(v, int)})
    for key, expected in config.get('expected_totals', {}).items():
        require(totals[key] == expected, f'Expected {key}={expected}; observed {totals[key]}')
    normalized_counts = [dict(filename=row['filename'], date=row['date'],
                              **{key:row.get(key, 0) for key in totals}) for row in file_counts]
    csv_write(reports / 'cleaning_summary.csv', normalized_counts + [dict(filename='ALL', date='', **totals)])
    csv_write(reports / 'cleaning_cell_audit.csv', cell_quality)
    # Written before any split is selected; its hash is later embedded in the split rules.
    csv_write(reports / 'pre_split_class_distribution.csv', distributions)
    manifest = dict(status='cleaning_complete', files=inventory, features=features, totals=dict(totals),
                    config_sha256=sha256(reports / 'run_config.json'),
                    cleaning_audit_sha256=sha256(reports / 'cleaning_audit.csv'),
                    pre_split_report_sha256=sha256(reports / 'pre_split_class_distribution.csv'))
    json_write(reports / 'source_manifest.json', manifest)
    (reports / 'source_manifest.in_progress.json').unlink()
    print(json.dumps(dict(totals)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/baseline.json')
    run(load_config(parser.parse_args().config))
