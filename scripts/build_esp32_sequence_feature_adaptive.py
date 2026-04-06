#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from build_esp32_sequence_ma10_dataset import (
    CLASS_NAMES,
    SPLITS,
    build_windows,
    transform_segment,
    variant_name_for,
    write_manifest,
)
from build_esp32_sequence_variants_adaptive import (
    DEFAULT_INPUT_ROOT,
    build_segments_adaptive,
    load_observed_samples,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "dataset" / "esp32_sequence_htltf_firstdiff_w64_s10_adaptive_ema50_k3_rel018_abs4000"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build amplitude-derived esp32 sequence datasets on top of adaptive gap "
            "reconstruction (initial median + EMA + k-multiple loss detection)."
        )
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--feature-mode",
        choices=("ma_residual", "first_difference", "rolling_std"),
        default="first_difference",
    )
    parser.add_argument("--moving-average-steps", type=int, default=10)
    parser.add_argument("--window-length", type=int, default=64)
    parser.add_argument("--window-stride", type=int, default=10)
    parser.add_argument("--max-interp-gap-steps", type=int, default=3)
    parser.add_argument("--ema-span", type=int, default=50)
    parser.add_argument("--initial-median-gaps", type=int, default=50)
    parser.add_argument("--relative-tolerance", type=float, default=0.18)
    parser.add_argument("--absolute-tolerance-us", type=int, default=4000)
    parser.add_argument("--long-gap-factor", type=float, default=3.5)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 4))),
    )
    return parser.parse_args()


def process_csv(
    csv_path_str: str,
    split: str,
    label: str,
    output_root_str: str,
    feature_mode: str,
    moving_average_steps: int,
    window_length: int,
    window_stride: int,
    max_interp_gap_steps: int,
    ema_span: int,
    initial_median_gaps: int,
    relative_tolerance: float,
    absolute_tolerance_us: int,
    long_gap_factor: float,
) -> dict[str, float]:
    csv_path = Path(csv_path_str)
    output_root = Path(output_root_str)
    variant_name = variant_name_for(feature_mode, moving_average_steps)

    samples = load_observed_samples(csv_path)
    stats: dict[str, float] = {
        "files_seen": 1.0,
        "files_with_samples": 1.0 if samples else 0.0,
        "rows_seen": float(len(samples)),
        "segments_before_feature": 0.0,
        "segments_after_feature": 0.0,
        "windows": 0.0,
        "train_big_windows": 0.0,
        "train_small_windows": 0.0,
        "validation_big_windows": 0.0,
        "validation_small_windows": 0.0,
        "classified_k1": 0.0,
        "classified_k2": 0.0,
        "classified_k3": 0.0,
        "interpolated_points": 0.0,
        "ema_updates": 0.0,
        "long_gap_splits": 0.0,
        "unmatched_gap_splits": 0.0,
        "nonpositive_gap_splits": 0.0,
        "initial_base_gap_us_sum": 0.0,
        "initial_base_gap_count": 0.0,
    }

    if not samples:
        return stats

    segments, adaptive_stats = build_segments_adaptive(
        csv_path=csv_path,
        split=split,
        label=label,
        samples=samples,
        max_interp_gap_steps=max_interp_gap_steps,
        ema_span=ema_span,
        initial_median_gaps=initial_median_gaps,
        relative_tolerance=relative_tolerance,
        absolute_tolerance_us=absolute_tolerance_us,
        long_gap_factor=long_gap_factor,
    )
    stats["segments_before_feature"] = float(len(segments))
    for key, value in adaptive_stats.items():
        stats[key] = stats.get(key, 0.0) + float(value)

    for segment in segments:
        feature_segment = transform_segment(
            segment=segment,
            feature_mode=feature_mode,
            moving_average_steps=moving_average_steps,
        )
        if feature_segment is None:
            continue

        stats["segments_after_feature"] += 1.0
        window_stats = build_windows(
            segment=feature_segment,
            output_root=output_root,
            window_length=window_length,
            window_stride=window_stride,
            variant_name=variant_name,
        )
        for key, value in window_stats.items():
            stats[key] = stats.get(key, 0.0) + float(value)

    return stats


def process_csv_task(
    task: tuple[str, str, str, str, str, int, int, int, int, int, int, float, int, float]
) -> dict[str, float]:
    return process_csv(*task)


def merge_stats(total: dict[str, float], partial: dict[str, float]) -> None:
    for key, value in partial.items():
        total[key] = total.get(key, 0.0) + float(value)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    variant_name = variant_name_for(args.feature_mode, args.moving_average_steps)

    tasks: list[tuple[str, str, str, str, str, int, int, int, int, int, int, float, int, float]] = []
    for split in SPLITS:
        for label in CLASS_NAMES:
            class_dir = args.input_root / split / label
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing directory: {class_dir}")
            for csv_path in sorted(class_dir.glob("*.csv")):
                tasks.append(
                    (
                        str(csv_path),
                        split,
                        label,
                        str(args.output_root),
                        args.feature_mode,
                        args.moving_average_steps,
                        args.window_length,
                        args.window_stride,
                        args.max_interp_gap_steps,
                        args.ema_span,
                        args.initial_median_gaps,
                        args.relative_tolerance,
                        args.absolute_tolerance_us,
                        args.long_gap_factor,
                    )
                )

    merged_stats: dict[str, float] = {
        "files_seen": 0.0,
        "files_with_samples": 0.0,
        "rows_seen": 0.0,
        "segments_before_feature": 0.0,
        "segments_after_feature": 0.0,
        "windows": 0.0,
        "train_big_windows": 0.0,
        "train_small_windows": 0.0,
        "validation_big_windows": 0.0,
        "validation_small_windows": 0.0,
        "classified_k1": 0.0,
        "classified_k2": 0.0,
        "classified_k3": 0.0,
        "interpolated_points": 0.0,
        "ema_updates": 0.0,
        "long_gap_splits": 0.0,
        "unmatched_gap_splits": 0.0,
        "nonpositive_gap_splits": 0.0,
        "initial_base_gap_us_sum": 0.0,
        "initial_base_gap_count": 0.0,
    }

    max_workers = max(1, args.num_workers)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for partial in executor.map(
            process_csv_task,
            tasks,
            chunksize=max(1, math.ceil(len(tasks) / max_workers)),
        ):
            merge_stats(merged_stats, partial)

    write_manifest(args.output_root, args.window_length, variant_name)

    initial_gap_mean_us = None
    if merged_stats["initial_base_gap_count"] > 0:
        initial_gap_mean_us = (
            merged_stats["initial_base_gap_us_sum"] / merged_stats["initial_base_gap_count"]
        )

    summary = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "feature_mode": args.feature_mode,
        "moving_average_steps": args.moving_average_steps,
        "window_length": args.window_length,
        "window_stride": args.window_stride,
        "num_workers": max_workers,
        "variant": variant_name,
        "adaptive_gap_model": {
            "initial_median_gaps": args.initial_median_gaps,
            "ema_span": args.ema_span,
            "ema_alpha": 2.0 / (float(args.ema_span) + 1.0),
            "max_interp_gap_steps": args.max_interp_gap_steps,
            "relative_tolerance": args.relative_tolerance,
            "absolute_tolerance_us": args.absolute_tolerance_us,
            "long_gap_factor": args.long_gap_factor,
        },
        "source_stats": {
            "files_seen": int(merged_stats["files_seen"]),
            "files_with_samples": int(merged_stats["files_with_samples"]),
            "rows_seen": int(merged_stats["rows_seen"]),
            "segments_before_feature": int(merged_stats["segments_before_feature"]),
            "segments_after_feature": int(merged_stats["segments_after_feature"]),
            "classified_k1": int(merged_stats["classified_k1"]),
            "classified_k2": int(merged_stats["classified_k2"]),
            "classified_k3": int(merged_stats["classified_k3"]),
            "interpolated_points": int(merged_stats["interpolated_points"]),
            "ema_updates": int(merged_stats["ema_updates"]),
            "long_gap_splits": int(merged_stats["long_gap_splits"]),
            "unmatched_gap_splits": int(merged_stats["unmatched_gap_splits"]),
            "nonpositive_gap_splits": int(merged_stats["nonpositive_gap_splits"]),
            "initial_base_gap_us_mean": initial_gap_mean_us,
        },
        "variant_stats": {
            "windows": int(merged_stats["windows"]),
            "train_big_windows": int(merged_stats["train_big_windows"]),
            "train_small_windows": int(merged_stats["train_small_windows"]),
            "validation_big_windows": int(merged_stats["validation_big_windows"]),
            "validation_small_windows": int(merged_stats["validation_small_windows"]),
        },
    }

    summary_path = args.output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
