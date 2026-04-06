#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "dataset" / "esp32"
DEFAULT_DEST_ROOT = PROJECT_ROOT / "dataset" / "esp32_raw_csi_variants"

EXPECTED_RAW_LEN = 384

LLTF_START_PAIR = 0
HT_LTF_START_PAIR = 64

VALID_LLTF_LOCAL_PAIR_INDICES = list(range(6, 32)) + list(range(33, 59))
VALID_HT_LTF_LOCAL_PAIR_INDICES = list(range(2, 59)) + list(range(70, 127))

VALID_LLTF_GLOBAL_PAIR_INDICES = [
    LLTF_START_PAIR + idx for idx in VALID_LLTF_LOCAL_PAIR_INDICES
]
VALID_HT_LTF_GLOBAL_PAIR_INDICES = [
    HT_LTF_START_PAIR + idx for idx in VALID_HT_LTF_LOCAL_PAIR_INDICES
]

VARIANT_SPECS = {
    "lltf_only": VALID_LLTF_GLOBAL_PAIR_INDICES,
    "htltf_only": VALID_HT_LTF_GLOBAL_PAIR_INDICES,
    "lltf_htltf": VALID_LLTF_GLOBAL_PAIR_INDICES + VALID_HT_LTF_GLOBAL_PAIR_INDICES,
}

FILENAME_PATTERN = re.compile(
    r"^csi_\d{6}_\d{6}_(?P<person>[A-Za-z]+)_(?P<label>big|small|smal)\.csv$",
    re.IGNORECASE,
)
PERSON_NORMALIZATION = {
    "cheawon": "chaewon",
}
LABEL_NORMALIZATION = {
    "smal": "small",
}


@dataclass(frozen=True)
class FileRecord:
    source_path: Path
    filename: str
    person_raw: str
    person: str
    label_raw: str
    label: str
    row_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract LLTF-only, HT-LTF-only, and LLTF+HT-LTF raw CSI variants "
            "from the esp32 dataset, then split the files by person into "
            "train/validation sets with a row-count ratio as close to 8:2 as possible."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Directory containing esp32 CSV files.",
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=DEFAULT_DEST_ROOT,
        help="Directory where extracted datasets will be written.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of worker processes to use for file-level extraction.",
    )
    return parser.parse_args()


def count_csv_rows(csv_path: Path) -> int:
    with csv_path.open(newline="") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def parse_filename(csv_path: Path) -> tuple[str, str, str, str] | None:
    match = FILENAME_PATTERN.match(csv_path.name)
    if match is None:
        return None

    person_raw = match.group("person").lower()
    label_raw = match.group("label").lower()
    person = PERSON_NORMALIZATION.get(person_raw, person_raw)
    label = LABEL_NORMALIZATION.get(label_raw, label_raw)
    return person_raw, person, label_raw, label


def scan_records(source_root: Path) -> tuple[list[FileRecord], list[str]]:
    records: list[FileRecord] = []
    skipped_filenames: list[str] = []

    for source_path in sorted(source_root.glob("*.csv")):
        parsed = parse_filename(source_path)
        if parsed is None:
            skipped_filenames.append(source_path.name)
            continue

        person_raw, person, label_raw, label = parsed
        records.append(
            FileRecord(
                source_path=source_path,
                filename=source_path.name,
                person_raw=person_raw,
                person=person,
                label_raw=label_raw,
                label=label,
                row_count=count_csv_rows(source_path),
            )
        )

    return records, skipped_filenames


def choose_person_split(records: list[FileRecord]) -> tuple[set[str], set[str], dict[str, object]]:
    person_totals: dict[str, int] = defaultdict(int)
    person_label_rows: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        person_totals[record.person] += record.row_count
        person_label_rows[record.person][record.label] += record.row_count

    people = sorted(person_totals)
    total_rows = sum(person_totals.values())
    target_train_rows = total_rows * 0.8

    best_combo: tuple[str, ...] | None = None
    best_diff: float | None = None
    best_train_rows = 0

    for combo_size in range(1, len(people)):
        for combo in combinations(people, combo_size):
            train_rows = sum(person_totals[person] for person in combo)
            diff = abs(train_rows - target_train_rows)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_combo = combo
                best_train_rows = train_rows

    if best_combo is None:
        raise RuntimeError("Could not compute a non-trivial person split.")

    train_people = set(best_combo)
    validation_people = set(people) - train_people

    train_counts = Counter()
    validation_counts = Counter()
    for person in train_people:
        train_counts.update(person_label_rows[person])
    for person in validation_people:
        validation_counts.update(person_label_rows[person])

    split_summary = {
        "total_rows": total_rows,
        "target_train_rows": target_train_rows,
        "train_rows": best_train_rows,
        "validation_rows": total_rows - best_train_rows,
        "train_ratio": best_train_rows / total_rows if total_rows else 0.0,
        "validation_ratio": (total_rows - best_train_rows) / total_rows if total_rows else 0.0,
        "train_people": sorted(train_people),
        "validation_people": sorted(validation_people),
        "train_label_rows": dict(train_counts),
        "validation_label_rows": dict(validation_counts),
    }
    return train_people, validation_people, split_summary


def parse_data_field(data_field: str) -> list[int]:
    parsed = ast.literal_eval(data_field)
    if not isinstance(parsed, list):
        raise ValueError("data field is not a list")
    return [int(value) for value in parsed]


def serialize_data_field(values: list[int]) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


def extract_pairs(raw_values: list[int], pair_indices: list[int]) -> list[int]:
    extracted: list[int] = []
    for pair_index in pair_indices:
        base_index = pair_index * 2
        extracted.extend(raw_values[base_index : base_index + 2])
    return extracted


def write_variant_csvs(
    source_path: Path,
    variant_dest_paths: dict[str, Path],
    variant_stats: dict[str, Counter],
) -> dict[str, Counter]:
    with source_path.open(newline="") as src_file:
        reader = csv.DictReader(src_file)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    writers: dict[str, csv.DictWriter] = {}
    handles = []
    try:
        for variant_name, dest_path in variant_dest_paths.items():
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            handle = dest_path.open("w", newline="")
            handles.append(handle)
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writers[variant_name] = writer

        for source_row in rows:
            try:
                raw_values = parse_data_field(source_row["data"])
            except Exception:
                raw_values = []

            valid_raw_len = len(raw_values) == EXPECTED_RAW_LEN

            for variant_name, pair_indices in VARIANT_SPECS.items():
                row = dict(source_row)
                stats = variant_stats[variant_name]
                stats["rows_seen"] += 1

                if not valid_raw_len:
                    row["len"] = "0"
                    row["data"] = "[]"
                    stats["rows_invalid_length"] += 1
                else:
                    extracted_values = extract_pairs(raw_values, pair_indices)
                    row["len"] = str(len(extracted_values))
                    row["data"] = serialize_data_field(extracted_values)
                    stats["rows_written"] += 1

                writers[variant_name].writerow(row)
    finally:
        for handle in handles:
            handle.close()

    return variant_stats


def process_record(task: tuple[Path, Path, str, str, str]) -> dict[str, object]:
    source_path, dest_root, split, label, filename = task
    variant_dest_paths = {
        variant_name: dest_root / variant_name / split / label / filename
        for variant_name in VARIANT_SPECS
    }
    local_stats: dict[str, Counter] = defaultdict(Counter)
    stats = write_variant_csvs(
        source_path=source_path,
        variant_dest_paths=variant_dest_paths,
        variant_stats=local_stats,
    )
    return {
        "source_path": str(source_path),
        "variant_stats": {variant_name: dict(counter) for variant_name, counter in stats.items()},
    }


def write_manifest(
    dest_root: Path,
    records: list[FileRecord],
    train_people: set[str],
    validation_people: set[str],
) -> None:
    manifest_path = dest_root / "split_manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        fieldnames = [
            "filename",
            "split",
            "label",
            "person_raw",
            "person_normalized",
            "row_count",
            "source_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            split = "train" if record.person in train_people else "validation"
            writer.writerow(
                {
                    "filename": record.filename,
                    "split": split,
                    "label": record.label,
                    "person_raw": record.person_raw,
                    "person_normalized": record.person,
                    "row_count": record.row_count,
                    "source_path": str(record.source_path),
                }
            )


def main() -> None:
    args = parse_args()
    source_root = args.source_root
    dest_root = args.dest_root

    if not source_root.exists():
        raise SystemExit(f"Source directory does not exist: {source_root}")

    records, skipped_filenames = scan_records(source_root)
    if not records:
        raise SystemExit(f"No labeled CSV files found in: {source_root}")

    train_people, validation_people, split_summary = choose_person_split(records)

    variant_stats: dict[str, Counter] = defaultdict(Counter)
    split_file_counts: dict[str, Counter] = defaultdict(Counter)
    split_row_counts: dict[str, Counter] = defaultdict(Counter)

    tasks: list[tuple[Path, Path, str, str, str]] = []
    for record in records:
        split = "train" if record.person in train_people else "validation"
        split_file_counts[split][record.label] += 1
        split_row_counts[split][record.label] += record.row_count
        tasks.append((record.source_path, dest_root, split, record.label, record.filename))

    with ProcessPoolExecutor(max_workers=max(args.workers, 1)) as executor:
        for result in executor.map(process_record, tasks, chunksize=4):
            per_variant = result["variant_stats"]
            for variant_name in VARIANT_SPECS:
                variant_stats[variant_name].update(per_variant.get(variant_name, {}))
                variant_stats[variant_name]["files_written"] += 1

    write_manifest(dest_root, records, train_people, validation_people)

    summary = {
        "source_root": str(source_root),
        "dest_root": str(dest_root),
        "expected_raw_len": EXPECTED_RAW_LEN,
        "person_normalization": PERSON_NORMALIZATION,
        "label_normalization": LABEL_NORMALIZATION,
        "skipped_filenames": skipped_filenames,
        "split_summary": split_summary,
        "split_file_counts": {split: dict(counts) for split, counts in split_file_counts.items()},
        "split_row_counts": {split: dict(counts) for split, counts in split_row_counts.items()},
        "variants": {
            variant_name: {
                "pair_count": len(pair_indices),
                "serialized_len": len(pair_indices) * 2,
                "stats": dict(variant_stats[variant_name]),
            }
            for variant_name, pair_indices in VARIANT_SPECS.items()
        },
    }

    summary_path = dest_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print(f"source_root={source_root}")
    print(f"dest_root={dest_root}")
    print(f"train_people={sorted(train_people)}")
    print(f"validation_people={sorted(validation_people)}")
    print(
        "train_rows="
        f"{split_summary['train_rows']} ({split_summary['train_ratio']:.6f}), "
        "validation_rows="
        f"{split_summary['validation_rows']} ({split_summary['validation_ratio']:.6f})"
    )
    print(f"skipped_filenames={len(skipped_filenames)}")
    for variant_name in VARIANT_SPECS:
        stats = variant_stats[variant_name]
        print(
            f"{variant_name}: files={stats['files_written']} "
            f"rows_seen={stats['rows_seen']} "
            f"rows_written={stats['rows_written']} "
            f"rows_invalid_length={stats['rows_invalid_length']}"
        )


if __name__ == "__main__":
    main()
