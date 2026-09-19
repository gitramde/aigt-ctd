from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_REPORTS = (
    "dataset_summary.csv",
    "class_by_file.csv",
    "column_inventory.csv",
    "data_quality_report.csv",
)


def classify_dtype(values: pd.Series, column: str) -> str:
    nonempty = values[values.str.strip() != ""]
    if nonempty.empty:
        return "empty"
    lowered = nonempty.str.strip().str.casefold()
    if column.casefold() == "timestamp":
        parsed = pd.to_datetime(nonempty, errors="coerce", format="mixed")
        if parsed.notna().all():
            return "datetime64[ns]"
    numeric = pd.to_numeric(nonempty, errors="coerce")
    if numeric.notna().all():
        if np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
            return "float64"
        if (numeric % 1 == 0).all():
            return "int64"
        return "float64"
    if lowered.isin({"nan", "+nan", "-nan"}).all():
        return "float64"
    return "string"


def add_hash_counts(seen_hashes: set[int], frame: pd.DataFrame) -> int:
    hashes = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64)
    unique_hashes, counts = np.unique(hashes, return_counts=True)
    if seen_hashes:
        previously_seen = np.isin(unique_hashes, np.fromiter(seen_hashes, dtype=np.uint64))
    else:
        previously_seen = np.zeros(len(unique_hashes), dtype=bool)
    duplicate_count = int(counts[previously_seen].sum())
    duplicate_count += int(np.maximum(counts[~previously_seen] - 1, 0).sum())
    seen_hashes.update(int(value) for value in unique_hashes)
    return duplicate_count


def audit_file(path: Path, chunk_size: int) -> tuple[dict, list[dict], list[dict], dict]:
    file_size = path.stat().st_size
    row_count = 0
    columns: list[str] = []
    stats: dict[str, dict] = {}
    labels: Counter[str] = Counter()
    timestamp_min = None
    timestamp_max = None

    seen_hashes: set[int] = set()
    duplicate_records = 0
    reader = pd.read_csv(
        path,
        chunksize=chunk_size,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        low_memory=False,
    )
    for chunk in reader:
        if not columns:
            columns = [str(column) for column in chunk.columns]
            for column in columns:
                stats[column] = {
                        "dtype": None,
                        "missing": 0,
                        "nan": 0,
                        "infinite": 0,
                        "nonempty": 0,
                        "sample_values": set(),
                        "more_than_two_values": False,
                        "numeric_only": True,
                        "numeric_seen": False,
                        "integer_only": True,
                    }
        row_count += len(chunk)
        duplicate_records += add_hash_counts(seen_hashes, chunk)
        for column in columns:
                values = chunk[column].astype(str)
                stripped = values.str.strip()
                missing = stripped.eq("")
                nan_tokens = stripped.str.casefold().isin({"nan", "+nan", "-nan"})
                infinite_tokens = stripped.str.casefold().isin(
                    {"inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
                )
                numeric = pd.to_numeric(stripped.where(~missing & ~nan_tokens), errors="coerce")
                numeric_infinite = pd.Series(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)), index=values.index)
                state = stats[column]
                state["missing"] += int(missing.sum())
                state["nan"] += int(nan_tokens.sum())
                state["infinite"] += int((infinite_tokens | numeric_infinite).sum())
                state["nonempty"] += int((~missing).sum())
                if not state["more_than_two_values"]:
                    for value in stripped[~missing].unique():
                        if value not in state["sample_values"]:
                            if len(state["sample_values"]) < 2:
                                state["sample_values"].add(value)
                            else:
                                state["more_than_two_values"] = True
                                break
                numeric_values = numeric[~numeric.isna() & ~np.isinf(numeric)]
                if not numeric_values.empty:
                    state["numeric_seen"] = True
                    state["integer_only"] &= bool((numeric_values % 1 == 0).all())
                state["numeric_only"] &= bool(numeric[~missing & ~nan_tokens].notna().all())
                if column.casefold() == "label":
                    labels.update(stripped[~missing].tolist())
                if column.casefold() == "timestamp":
                    parsed = pd.to_datetime(stripped[~missing], errors="coerce", format="mixed")
                    parsed = parsed.dropna()
                    if not parsed.empty:
                        current_min = parsed.min().isoformat()
                        current_max = parsed.max().isoformat()
                        timestamp_min = current_min if timestamp_min is None else min(timestamp_min, current_min)
                        timestamp_max = current_max if timestamp_max is None else max(timestamp_max, current_max)

    inventory = []
    quality = []
    constant_columns = []
    for column in columns:
        state = stats[column]
        if state["numeric_seen"] and state["numeric_only"]:
            dtype = "int64" if state["integer_only"] and state["nan"] == 0 and state["infinite"] == 0 else "float64"
        else:
            dtype = classify_dtype(pd.Series(list(state["sample_values"]), dtype=str), column)
        unique_count = 2 if state["more_than_two_values"] else len(state["sample_values"])
        is_constant = unique_count <= 1
        if is_constant:
            constant_columns.append(column)
        inventory.append(
            {
                "filename": path.name,
                "column_name": column,
                "data_type": dtype,
                "non_missing_values": state["nonempty"] - state["nan"],
                "unique_values": unique_count,
                "is_constant": is_constant,
                "is_source_identifier": column.casefold() in {"src ip", "source ip", "src_ip", "source_ip"},
                "is_destination_identifier": column.casefold() in {"dst ip", "destination ip", "dst_ip", "destination_ip"},
                "is_source_port": column.casefold() in {"src port", "source port", "src_port", "source_port"},
                "is_destination_port": column.casefold() in {"dst port", "destination port", "dst_port", "destination_port"},
                "is_timestamp": column.casefold() == "timestamp",
                "is_protocol": column.casefold() == "protocol",
                "is_flow_duration": column.casefold() == "flow duration",
                "is_packet_count": "pkt" in column.casefold() and ("tot" in column.casefold() or "subflow" in column.casefold()),
                "is_byte_count": "byt" in column.casefold() and ("tot" in column.casefold() or "subflow" in column.casefold()),
                "is_attack_label": column.casefold() == "label",
            }
        )
        quality.append(
            {
                "filename": path.name,
                "column_name": column,
                "missing_values": state["missing"],
                "nan_values": state["nan"],
                "infinite_values": state["infinite"],
                "duplicate_records": duplicate_records,
                "constant_column": is_constant,
            }
        )

    labels_json = json.dumps(dict(sorted(labels.items())), ensure_ascii=True, sort_keys=True)
    benign_count = sum(count for label, count in labels.items() if label.casefold() == "benign")
    malicious_count = sum(count for label, count in labels.items() if label.casefold() != "benign")
    required_names = {
        "source_identifier": {"src ip", "source ip", "src_ip", "source_ip"},
        "destination_identifier": {"dst ip", "destination ip", "dst_ip", "destination_ip"},
        "source_port": {"src port", "source port", "src_port", "source_port"},
        "destination_port": {"dst port", "destination port", "dst_port", "destination_port"},
        "timestamp": {"timestamp"},
        "protocol": {"protocol"},
        "flow_duration": {"flow duration"},
        "packet_counts": set(),
        "byte_counts": set(),
        "attack_label": {"label"},
    }
    lowered_columns = {column.casefold() for column in columns}
    presence = {
        "has_" + name: any(item["is_" + name] for item in inventory) if name not in {"packet_counts", "byte_counts"} else any(
            item["is_packet_count" if name == "packet_counts" else "is_byte_count"] for item in inventory
        )
        for name in required_names
    }
    summary = {
        "filename": path.name,
        "file_size_bytes": file_size,
        "row_count": row_count,
        "column_count": len(columns),
        "column_names": json.dumps(columns, ensure_ascii=True),
        "data_types": json.dumps({row["column_name"]: row["data_type"] for row in inventory}, ensure_ascii=True, sort_keys=True),
        "timestamp_min": timestamp_min or "",
        "timestamp_max": timestamp_max or "",
        "class_labels": labels_json,
        "benign_count": benign_count,
        "malicious_count": malicious_count,
        "duplicate_records": duplicate_records,
        "constant_columns": json.dumps(constant_columns, ensure_ascii=True),
        **presence,
    }
    class_rows = [
        {
            "filename": path.name,
            "label": label,
            "count": count,
            "class_group": "benign" if label.casefold() == "benign" else "malicious",
        }
        for label, count in sorted(labels.items())
    ]
    return summary, class_rows, inventory, quality


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob("*.csv"))
    if len(paths) != 10:
        raise RuntimeError(f"Expected 10 CSV files, found {len(paths)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries, class_rows, inventories, quality_rows = [], [], [], []
    for index, path in enumerate(paths, start=1):
        print(f"Auditing {index}/10: {path.name}", flush=True)
        summary, classes, inventory, quality = audit_file(path, args.chunk_size)
        summaries.append(summary)
        class_rows.extend(classes)
        inventories.extend(inventory)
        quality_rows.extend(quality)
    pd.DataFrame(summaries).to_csv(args.output_dir / "dataset_summary.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame(class_rows).to_csv(args.output_dir / "class_by_file.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame(inventories).to_csv(args.output_dir / "column_inventory.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame(quality_rows).to_csv(args.output_dir / "data_quality_report.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote {', '.join(REQUIRED_REPORTS)}", flush=True)


if __name__ == "__main__":
    main()
