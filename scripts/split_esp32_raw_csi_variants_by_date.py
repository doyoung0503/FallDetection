#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "dataset" / "esp32_raw_csi_variants"
DEFAULT_DEST_ROOT = PROJECT_ROOT / "dataset" / "esp32_raw_csi_variants_by_date"

VARIANTS = ["lltf_only", "htltf_only", "lltf_htltf"]
SPLITS = ["train", "validation"]
CLASS_NAMES = ["big", "small"]

FILENAME_PATTERN = re.compile(
    r"^csi_(?P<date>\d{6})_\d{6}_(?P<person>[A-Za-z]+)_(?P<label>big|small|smal)\.csv$",
    re.IGNORECASE,
)
LABEL_NORMALIZATION = {"smal": "small"}


@dataclass(frozen=True)
class VariantFile:
    variant: str
    source_path: Path
    filename: str
    date: str
    label: str
    row_count: int


def os_cpu_count() -> int | None:
    try:
        return os.cpu_count()
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a date-based train/validation split for esp32 raw CSI variants. "
            "Files from the same date are always assigned to the same split."
        )
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--dest-root", type=Path, default=DEFAULT_DEST_ROOT)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os_cpu_count() or 1))),
        help="Worker processes used when counting/creating links.",
    )
    return parser.parse_args()


def count_csv_rows(csv_path: Path) -> int:
    with csv_path.open(newline="") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def scan_variant_files(source_root: Path) -> list[VariantFile]:
    records: list[VariantFile] = []
    seen: set[tuple[str, str]] = set()

    for variant in VARIANTS:
        variant_root = source_root / variant
        for path in sorted(variant_root.rglob("*.csv")):
            key = (variant, path.name)
            if key in seen:
                continue
            seen.add(key)

            match = FILENAME_PATTERN.match(path.name)
            if match is None:
                continue

            date = match.group("date")
            label = LABEL_NORMALIZATION.get(match.group("label").lower(), match.group("label").lower())
            if label not in CLASS_NAMES:
                continue

            records.append(
                VariantFile(
                    variant=variant,
                    source_path=path,
                    filename=path.name,
                    date=date,
                    label=label,
                    row_count=count_csv_rows(path),
                )
            )

    return records


def choose_date_split(records: list[VariantFile]) -> tuple[set[str], set[str], dict[str, object]]:
    date_rows: Counter[str] = Counter()
    date_label_rows: dict[str, Counter[str]] = defaultdict(Counter)
    date_file_counts: Counter[str] = Counter()

    # Use a single variant to estimate row totals because all variants share the same rows.
    for record in records:
        if record.variant != "htltf_only":
            continue
        date_rows[record.date] += record.row_count
        date_label_rows[record.date][record.label] += record.row_count
        date_file_counts[record.date] += 1

    dates = sorted(date_rows)
    total_rows = sum(date_rows.values())
    target_train_rows = total_rows * 0.8

    best: tuple[float, tuple[str, ...], int] | None = None
    from itertools import combinations

    for r in range(1, len(dates)):
        for combo in combinations(dates, r):
            train_rows = sum(date_rows[d] for d in combo)
            diff = abs(train_rows - target_train_rows)
            candidate = (diff, combo, train_rows)
            if best is None or candidate < best:
                best = candidate

    if best is None:
        raise RuntimeError("Failed to choose a non-trivial date split.")

    train_dates = set(best[1])
    validation_dates = set(dates) - train_dates

    train_label_rows: Counter[str] = Counter()
    validation_label_rows: Counter[str] = Counter()
    train_file_count = 0
    validation_file_count = 0
    for date in train_dates:
        train_label_rows.update(date_label_rows[date])
        train_file_count += date_file_counts[date]
    for date in validation_dates:
        validation_label_rows.update(date_label_rows[date])
        validation_file_count += date_file_counts[date]

    summary = {
        "dates": dates,
        "date_rows": dict(date_rows),
        "date_file_counts": dict(date_file_counts),
        "total_rows": total_rows,
        "target_train_rows": target_train_rows,
        "train_dates": sorted(train_dates),
        "validation_dates": sorted(validation_dates),
        "train_rows": sum(date_rows[d] for d in train_dates),
        "validation_rows": sum(date_rows[d] for d in validation_dates),
        "train_ratio": sum(date_rows[d] for d in train_dates) / total_rows if total_rows else 0.0,
        "validation_ratio": sum(date_rows[d] for d in validation_dates) / total_rows if total_rows else 0.0,
        "train_label_rows": dict(train_label_rows),
        "validation_label_rows": dict(validation_label_rows),
        "train_file_count": train_file_count,
        "validation_file_count": validation_file_count,
    }
    return train_dates, validation_dates, summary


def ensure_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.link(src, dst)


def link_task(task: tuple[str, str]) -> None:
    src, dst = task
    ensure_link(Path(src), Path(dst))


def write_manifest(dest_root: Path, records: list[VariantFile], train_dates: set[str]) -> None:
    manifest_path = dest_root / "split_manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["variant", "filename", "date", "label", "split", "source_path", "dest_path", "row_count"],
        )
        writer.writeheader()
        for record in sorted(records, key=lambda r: (r.variant, r.date, r.filename)):
            split = "train" if record.date in train_dates else "validation"
            dest_path = dest_root / record.variant / split / record.label / record.filename
            writer.writerow(
                {
                    "variant": record.variant,
                    "filename": record.filename,
                    "date": record.date,
                    "label": record.label,
                    "split": split,
                    "source_path": str(record.source_path),
                    "dest_path": str(dest_path),
                    "row_count": record.row_count,
                }
            )


def main() -> None:
    args = parse_args()
    args.dest_root.mkdir(parents=True, exist_ok=True)

    records = scan_variant_files(args.source_root)
    train_dates, validation_dates, split_summary = choose_date_split(records)

    link_jobs: list[tuple[str, str]] = []
    variant_stats: dict[str, Counter[str]] = {
        variant: Counter(
            {
                "files": 0,
                "rows": 0,
                "train_big_files": 0,
                "train_small_files": 0,
                "validation_big_files": 0,
                "validation_small_files": 0,
                "train_big_rows": 0,
                "train_small_rows": 0,
                "validation_big_rows": 0,
                "validation_small_rows": 0,
            }
        )
        for variant in VARIANTS
    }

    for record in records:
        split = "train" if record.date in train_dates else "validation"
        dest_path = args.dest_root / record.variant / split / record.label / record.filename
        link_jobs.append((str(record.source_path), str(dest_path)))

        stats = variant_stats[record.variant]
        stats["files"] += 1
        stats["rows"] += record.row_count
        stats[f"{split}_{record.label}_files"] += 1
        stats[f"{split}_{record.label}_rows"] += record.row_count

    max_workers = max(1, args.workers)
    if max_workers == 1:
        for job in link_jobs:
            link_task(job)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(link_task, link_jobs, chunksize=max(1, len(link_jobs) // max_workers)))

    write_manifest(args.dest_root, records, train_dates)

    summary = {
        "source_root": str(args.source_root),
        "dest_root": str(args.dest_root),
        "date_split": split_summary,
        "variant_stats": {variant: dict(stats) for variant, stats in variant_stats.items()},
    }
    summary_path = args.dest_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
