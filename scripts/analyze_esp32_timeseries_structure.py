#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "dataset" / "esp32"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "analysis" / "esp32_timeseries_stats"

FILENAME_PATTERN = re.compile(
    r"^csi_\d{6}_\d{6}_(?P<person>[A-Za-z]+)_(?P<label>big|small|smal)\.csv$",
    re.IGNORECASE,
)
PERSON_NORMALIZATION = {"cheawon": "chaewon"}
LABEL_NORMALIZATION = {"smal": "small"}
CLASS_NAMES = ["big", "small"]


@dataclass
class FileStats:
    filename: str
    person: str
    label: str
    row_count: int
    duration_ms: float | None
    median_gap_ms: float | None
    mean_gap_ms: float | None
    p90_gap_ms: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze row-count and sampling-gap distributions for the esp32 dataset."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Directory containing the esp32 CSV files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where plots and summaries will be written.",
    )
    return parser.parse_args()


def parse_filename(csv_path: Path) -> tuple[str, str] | None:
    match = FILENAME_PATTERN.match(csv_path.name)
    if match is None:
        return None
    person = PERSON_NORMALIZATION.get(match.group("person").lower(), match.group("person").lower())
    label = LABEL_NORMALIZATION.get(match.group("label").lower(), match.group("label").lower())
    return person, label


def compute_file_stats(csv_path: Path, person: str, label: str) -> tuple[FileStats, np.ndarray]:
    timestamps: list[int] = []
    row_count = 0

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_count += 1
            try:
                timestamps.append(int(row["local_timestamp"]))
            except Exception:
                continue

    if len(timestamps) >= 2:
        gaps_ms = np.diff(np.asarray(timestamps, dtype=np.int64)).astype(np.float64) / 1000.0
        duration_ms = float(timestamps[-1] - timestamps[0]) / 1000.0
        median_gap_ms = float(np.median(gaps_ms))
        mean_gap_ms = float(np.mean(gaps_ms))
        p90_gap_ms = float(np.quantile(gaps_ms, 0.9))
    else:
        gaps_ms = np.asarray([], dtype=np.float64)
        duration_ms = None
        median_gap_ms = None
        mean_gap_ms = None
        p90_gap_ms = None

    return (
        FileStats(
            filename=csv_path.name,
            person=person,
            label=label,
            row_count=row_count,
            duration_ms=duration_ms,
            median_gap_ms=median_gap_ms,
            mean_gap_ms=mean_gap_ms,
            p90_gap_ms=p90_gap_ms,
        ),
        gaps_ms,
    )


def grouped_boxplot(
    ax: plt.Axes,
    groups: list[str],
    values_by_group_class: dict[tuple[str, str], list[float]],
    class_names: list[str],
    ylabel: str,
    title: str,
) -> None:
    width = 0.34
    positions = np.arange(len(groups))
    colors = {"big": "#1f77b4", "small": "#ff7f0e"}

    for class_index, class_name in enumerate(class_names):
        data = [values_by_group_class.get((group, class_name), []) for group in groups]
        offset = (class_index - 0.5) * width
        box = ax.boxplot(
            data,
            positions=positions + offset,
            widths=width * 0.92,
            patch_artist=True,
            showfliers=False,
        )
        for patch in box["boxes"]:
            patch.set_facecolor(colors[class_name])
            patch.set_alpha(0.6)
        for median in box["medians"]:
            median.set_color("black")

    ax.set_xticks(positions)
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    handles = [
        plt.Line2D([0], [0], color=colors[class_name], lw=8, alpha=0.6)
        for class_name in class_names
    ]
    ax.legend(handles, class_names, loc="upper right")


def save_plots(
    output_root: Path,
    file_stats: list[FileStats],
    all_gaps_by_class: dict[str, np.ndarray],
) -> None:
    persons = sorted({stats.person for stats in file_stats})

    row_counts_by_person_class: dict[tuple[str, str], list[float]] = defaultdict(list)
    median_gap_by_person_class: dict[tuple[str, str], list[float]] = defaultdict(list)
    duration_by_person_class: dict[tuple[str, str], list[float]] = defaultdict(list)

    for stats in file_stats:
        row_counts_by_person_class[(stats.person, stats.label)].append(float(stats.row_count))
        if stats.median_gap_ms is not None:
            median_gap_by_person_class[(stats.person, stats.label)].append(stats.median_gap_ms)
        if stats.duration_ms is not None:
            duration_by_person_class[(stats.person, stats.label)].append(stats.duration_ms / 1000.0)

    fig, axes = plt.subplots(3, 1, figsize=(16, 15), constrained_layout=True)
    grouped_boxplot(
        axes[0],
        persons,
        row_counts_by_person_class,
        CLASS_NAMES,
        ylabel="Rows per file",
        title="File Length Distribution by Person and Class",
    )
    grouped_boxplot(
        axes[1],
        persons,
        duration_by_person_class,
        CLASS_NAMES,
        ylabel="File duration (s)",
        title="Recording Duration by Person and Class",
    )
    grouped_boxplot(
        axes[2],
        persons,
        median_gap_by_person_class,
        CLASS_NAMES,
        ylabel="Median inter-row gap (ms)",
        title="Per-file Median Sampling Gap by Person and Class",
    )
    fig.savefig(output_root / "file_level_distributions.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)
    colors = {"big": "#1f77b4", "small": "#ff7f0e"}
    for class_name in CLASS_NAMES:
        gaps = all_gaps_by_class[class_name]
        clipped = gaps[(gaps >= 0.0) & (gaps <= 100.0)]
        axes[0].hist(
            clipped,
            bins=100,
            alpha=0.55,
            density=True,
            label=f"{class_name} (<=100 ms)",
            color=colors[class_name],
        )
        axes[1].hist(
            np.log10(np.clip(gaps, 1e-3, None)),
            bins=120,
            alpha=0.55,
            density=True,
            label=class_name,
            color=colors[class_name],
        )

    axes[0].set_title("Inter-row Gap Distribution (clipped to 0-100 ms)")
    axes[0].set_xlabel("Gap (ms)")
    axes[0].set_ylabel("Density")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    axes[1].set_title("Inter-row Gap Distribution (log10 scale)")
    axes[1].set_xlabel("log10(gap in ms)")
    axes[1].set_ylabel("Density")
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    fig.savefig(output_root / "gap_distributions.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    file_stats: list[FileStats] = []
    skipped_files: list[str] = []
    row_count_by_class = Counter()
    file_count_by_class = Counter()
    all_gaps: list[float] = []
    gaps_by_class: dict[str, list[np.ndarray]] = defaultdict(list)

    for csv_path in sorted(args.source_root.glob("*.csv")):
        parsed = parse_filename(csv_path)
        if parsed is None:
            skipped_files.append(csv_path.name)
            continue

        person, label = parsed
        stats, gaps_ms = compute_file_stats(csv_path, person, label)
        file_stats.append(stats)
        row_count_by_class[label] += stats.row_count
        file_count_by_class[label] += 1
        if gaps_ms.size:
            all_gaps.extend(gaps_ms.tolist())
            gaps_by_class[label].append(gaps_ms)

    all_gaps_array = np.asarray(all_gaps, dtype=np.float64)
    all_gaps_by_class = {
        class_name: np.concatenate(gaps_by_class[class_name]) if gaps_by_class[class_name] else np.asarray([], dtype=np.float64)
        for class_name in CLASS_NAMES
    }

    by_person_class: dict[str, dict[str, dict[str, float | int]]] = defaultdict(dict)
    for person in sorted({stats.person for stats in file_stats}):
        for class_name in CLASS_NAMES:
            subset = [
                stats for stats in file_stats if stats.person == person and stats.label == class_name
            ]
            if not subset:
                continue
            row_counts = np.asarray([stats.row_count for stats in subset], dtype=np.float64)
            durations = np.asarray(
                [stats.duration_ms for stats in subset if stats.duration_ms is not None],
                dtype=np.float64,
            )
            med_gaps = np.asarray(
                [stats.median_gap_ms for stats in subset if stats.median_gap_ms is not None],
                dtype=np.float64,
            )
            by_person_class[person][class_name] = {
                "file_count": len(subset),
                "row_count_mean": float(np.mean(row_counts)),
                "row_count_median": float(np.median(row_counts)),
                "row_count_p90": float(np.quantile(row_counts, 0.9)),
                "duration_sec_median": float(np.median(durations) / 1000.0) if durations.size else None,
                "median_gap_ms_median": float(np.median(med_gaps)) if med_gaps.size else None,
            }

    summary = {
        "source_root": str(args.source_root),
        "output_root": str(output_root),
        "skipped_files": skipped_files,
        "total_labeled_files": len(file_stats),
        "file_count_by_class": dict(file_count_by_class),
        "row_count_by_class": dict(row_count_by_class),
        "overall": {
            "row_count_mean": float(np.mean([stats.row_count for stats in file_stats])),
            "row_count_median": float(np.median([stats.row_count for stats in file_stats])),
            "row_count_p90": float(np.quantile([stats.row_count for stats in file_stats], 0.9)),
            "gap_ms_mean": float(np.mean(all_gaps_array)),
            "gap_ms_median": float(np.median(all_gaps_array)),
            "gap_ms_std": float(np.std(all_gaps_array)),
            "gap_ms_p90": float(np.quantile(all_gaps_array, 0.9)),
            "gap_ms_p99": float(np.quantile(all_gaps_array, 0.99)),
        },
        "class_gap_stats": {
            class_name: {
                "gap_ms_mean": float(np.mean(all_gaps_by_class[class_name])),
                "gap_ms_median": float(np.median(all_gaps_by_class[class_name])),
                "gap_ms_std": float(np.std(all_gaps_by_class[class_name])),
                "gap_ms_p90": float(np.quantile(all_gaps_by_class[class_name], 0.9)),
                "gap_ms_p99": float(np.quantile(all_gaps_by_class[class_name], 0.99)),
            }
            for class_name in CLASS_NAMES
        },
        "by_person_class": by_person_class,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )

    save_plots(output_root, file_stats, all_gaps_by_class)

    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    print(f"plots={output_root / 'file_level_distributions.png'}")
    print(f"plots={output_root / 'gap_distributions.png'}")


if __name__ == "__main__":
    main()
