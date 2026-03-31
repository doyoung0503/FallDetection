#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import csv
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "dataset" / "raw"
DEFAULT_DEST_ROOT = PROJECT_ROOT / "dataset" / "preprocessed_raw"

EXPECTED_RAW_LEN = 384
HT_LTF_START_PAIR = 64

# For this dataset format (HT, 40 MHz, secondary channel below, non-STBC),
# the 128 HT-LTF complex entries contain 114 valid subcarriers.
# The remaining bins are fixed null/guard/DC positions and are masked out.
VALID_HT_LTF_LOCAL_PAIR_INDICES = list(range(2, 59)) + list(range(70, 127))
VALID_HT_LTF_GLOBAL_PAIR_INDICES = [
    HT_LTF_START_PAIR + idx for idx in VALID_HT_LTF_LOCAL_PAIR_INDICES
]
FILTERED_LEN = len(VALID_HT_LTF_GLOBAL_PAIR_INDICES) * 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the valid HT-LTF subcarriers from ESP32 CSI CSV files "
            "for datasets stored in class-wise folders."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Root directory containing class subfolders of raw CSI CSV files.",
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=DEFAULT_DEST_ROOT,
        help="Root directory where preprocessed CSV files will be written.",
    )
    return parser.parse_args()


def parse_data_field(data_field: str) -> list[int]:
    parsed = ast.literal_eval(data_field)
    if not isinstance(parsed, list):
        raise ValueError("data field is not a list")
    return [int(value) for value in parsed]


def filter_ht_ltf_pairs(raw_values: list[int]) -> list[int]:
    filtered: list[int] = []
    for pair_index in VALID_HT_LTF_GLOBAL_PAIR_INDICES:
        base_index = pair_index * 2
        filtered.extend(raw_values[base_index : base_index + 2])
    return filtered


def serialize_data_field(values: list[int]) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


def preprocess_csv(source_path: Path, dest_path: Path, stats: dict[str, Counter]) -> None:
    with source_path.open(newline="") as src_file:
        reader = csv.DictReader(src_file)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with dest_path.open("w", newline="") as dest_file:
        writer = csv.DictWriter(dest_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            class_name = source_path.parent.name
            stats[class_name]["rows_seen"] += 1

            try:
                raw_values = parse_data_field(row["data"])
            except Exception:
                raw_values = []

            if len(raw_values) != EXPECTED_RAW_LEN:
                row["len"] = "0"
                row["data"] = "[]"
                stats[class_name]["rows_invalid_length"] += 1
            else:
                filtered_values = filter_ht_ltf_pairs(raw_values)
                row["len"] = str(FILTERED_LEN)
                row["data"] = serialize_data_field(filtered_values)
                stats[class_name]["rows_written"] += 1

            writer.writerow(row)


def main() -> None:
    args = parse_args()
    source_root = args.source_root
    dest_root = args.dest_root

    if not source_root.exists():
        raise SystemExit(f"Source directory does not exist: {source_root}")

    stats: dict[str, Counter] = defaultdict(Counter)

    for class_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        for source_path in sorted(class_dir.glob("*.csv")):
            relative_path = source_path.relative_to(source_root)
            dest_path = dest_root / relative_path
            preprocess_csv(source_path, dest_path, stats)
            stats[class_dir.name]["files_written"] += 1

    print(f"source={source_root}")
    print(f"dest={dest_root}")
    print(f"expected_raw_len={EXPECTED_RAW_LEN}")
    print(f"filtered_len={FILTERED_LEN}")
    print(f"valid_ht_ltf_pair_count={len(VALID_HT_LTF_GLOBAL_PAIR_INDICES)}")
    for class_name in sorted(stats):
        class_stats = stats[class_name]
        print(
            f"{class_name}: files={class_stats['files_written']} "
            f"rows_seen={class_stats['rows_seen']} "
            f"rows_written={class_stats['rows_written']} "
            f"rows_invalid_length={class_stats['rows_invalid_length']}"
        )


if __name__ == "__main__":
    main()
