#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "dataset" / "esp32_raw_csi_variants" / "htltf_only"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "esp32_sequence_htltf_variants_w64_s10_adaptive_ema50_k3_rel018_abs4000"
)

INPUT_DIM = 228
NUM_SUBCARRIERS = 114
CLASS_NAMES = ["big", "small"]
SPLITS = ["train", "validation"]
VARIANT_NAMES = ["interp_only", "interp_mask", "interp_mask_deltat"]


@dataclass
class ObservedSample:
    local_timestamp: int
    source_row_number: int
    amplitude: np.ndarray


@dataclass
class SequenceSegment:
    source_csv: str
    split: str
    label: str
    segment_id: int
    amplitudes: np.ndarray
    interp_mask: np.ndarray
    delta_t_ms: np.ndarray
    grid_timestamps: np.ndarray
    actual_timestamps: np.ndarray
    source_row_numbers: np.ndarray


@dataclass
class GapClassification:
    k_steps: int
    step_gap_us: float
    error_us: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build esp32 HT-LTF sequence datasets using adaptive local gap estimation: "
            "initial median, EMA updates on accepted 1-step gaps, k-multiple loss "
            "judgment, and long-gap segment splitting."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root directory containing train/validation and big/small CSV folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for the generated sequence datasets.",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=64,
        help="Sliding-window length in resampled steps.",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=10,
        help="Sliding-window stride in resampled steps.",
    )
    parser.add_argument(
        "--max-interp-gap-steps",
        type=int,
        default=3,
        help="Only gaps classified as 1..K steps are kept/interpolated.",
    )
    parser.add_argument(
        "--ema-span",
        type=int,
        default=50,
        help="EMA span used to track the local normal packet interval.",
    )
    parser.add_argument(
        "--initial-median-gaps",
        type=int,
        default=50,
        help="Number of early positive gaps used for the initial median base gap.",
    )
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=0.18,
        help="Relative tolerance for k-multiple gap classification.",
    )
    parser.add_argument(
        "--absolute-tolerance-us",
        type=int,
        default=4000,
        help="Absolute floor tolerance for k-multiple gap classification in microseconds.",
    )
    parser.add_argument(
        "--long-gap-factor",
        type=float,
        default=3.5,
        help=(
            "If an unclassified positive gap exceeds this factor times the current "
            "base gap, count it as a long-gap split."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
        help="Number of worker processes.",
    )
    return parser.parse_args()


def parse_amplitude(data_str: str) -> np.ndarray:
    values = ast.literal_eval(data_str)
    if len(values) != INPUT_DIM:
        raise ValueError(f"Expected {INPUT_DIM} values, got {len(values)}")

    arr = np.asarray(values, dtype=np.float32).reshape(NUM_SUBCARRIERS, 2)
    csi = arr[:, 1] + 1j * arr[:, 0]
    return np.abs(csi).astype(np.float32)


def load_observed_samples(csv_path: Path) -> list[ObservedSample]:
    samples: list[ObservedSample] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            if row.get("len") != str(INPUT_DIM):
                continue
            try:
                amplitude = parse_amplitude(row["data"])
                local_timestamp = int(row["local_timestamp"])
            except Exception:
                continue

            samples.append(
                ObservedSample(
                    local_timestamp=local_timestamp,
                    source_row_number=row_number,
                    amplitude=amplitude,
                )
            )
    return samples


def compute_initial_base_gap_us(
    samples: list[ObservedSample],
    initial_median_gaps: int,
) -> float | None:
    positive_gaps: list[int] = []
    for prev_sample, sample in zip(samples[:-1], samples[1:]):
        gap_us = sample.local_timestamp - prev_sample.local_timestamp
        if gap_us > 0:
            positive_gaps.append(gap_us)
            if len(positive_gaps) >= initial_median_gaps:
                break

    if not positive_gaps:
        return None
    return float(np.median(np.asarray(positive_gaps, dtype=np.float64)))


def classify_gap(
    gap_us: int,
    base_gap_us: float,
    max_interp_gap_steps: int,
    relative_tolerance: float,
    absolute_tolerance_us: int,
) -> GapClassification | None:
    if gap_us <= 0 or base_gap_us <= 0:
        return None

    best: GapClassification | None = None
    best_relative_error = float("inf")
    for k_steps in range(1, max_interp_gap_steps + 1):
        expected_gap_us = base_gap_us * k_steps
        tolerance_us = max(
            float(absolute_tolerance_us),
            relative_tolerance * expected_gap_us,
        )
        error_us = abs(gap_us - expected_gap_us)
        if error_us > tolerance_us:
            continue

        relative_error = error_us / expected_gap_us
        if relative_error < best_relative_error:
            best_relative_error = relative_error
            best = GapClassification(
                k_steps=k_steps,
                step_gap_us=float(gap_us) / float(k_steps),
                error_us=float(error_us),
            )

    return best


def finalize_segment(
    source_csv: str,
    split: str,
    label: str,
    segment_id: int,
    amplitudes: list[np.ndarray],
    interp_mask: list[int],
    delta_t_ms: list[float],
    grid_timestamps: list[int],
    actual_timestamps: list[int],
    source_row_numbers: list[int],
) -> SequenceSegment | None:
    if not amplitudes:
        return None

    return SequenceSegment(
        source_csv=source_csv,
        split=split,
        label=label,
        segment_id=segment_id,
        amplitudes=np.stack(amplitudes, axis=0).astype(np.float32),
        interp_mask=np.asarray(interp_mask, dtype=np.uint8),
        delta_t_ms=np.asarray(delta_t_ms, dtype=np.float32),
        grid_timestamps=np.asarray(grid_timestamps, dtype=np.int64),
        actual_timestamps=np.asarray(actual_timestamps, dtype=np.int64),
        source_row_numbers=np.asarray(source_row_numbers, dtype=np.int32),
    )


def build_segments_adaptive(
    csv_path: Path,
    split: str,
    label: str,
    samples: list[ObservedSample],
    max_interp_gap_steps: int,
    ema_span: int,
    initial_median_gaps: int,
    relative_tolerance: float,
    absolute_tolerance_us: int,
    long_gap_factor: float,
) -> tuple[list[SequenceSegment], dict[str, float]]:
    if not samples:
        return [], {
            "segments": 0,
            "classified_k1": 0,
            "classified_k2": 0,
            "classified_k3": 0,
            "interpolated_points": 0,
            "ema_updates": 0,
            "long_gap_splits": 0,
            "unmatched_gap_splits": 0,
            "nonpositive_gap_splits": 0,
            "initial_base_gap_us_sum": 0.0,
            "initial_base_gap_count": 0,
        }

    initial_base_gap_us = compute_initial_base_gap_us(samples, initial_median_gaps)
    base_gap_us = initial_base_gap_us if initial_base_gap_us is not None else 0.0
    ema_alpha = 2.0 / (float(ema_span) + 1.0)

    stats: dict[str, float] = {
        "segments": 0,
        "classified_k1": 0,
        "classified_k2": 0,
        "classified_k3": 0,
        "interpolated_points": 0,
        "ema_updates": 0,
        "long_gap_splits": 0,
        "unmatched_gap_splits": 0,
        "nonpositive_gap_splits": 0,
        "initial_base_gap_us_sum": float(initial_base_gap_us or 0.0),
        "initial_base_gap_count": 1.0 if initial_base_gap_us is not None else 0.0,
    }

    segments: list[SequenceSegment] = []
    segment_id = 0

    amplitudes: list[np.ndarray] = [samples[0].amplitude]
    interp_mask: list[int] = [0]
    delta_t_ms: list[float] = [0.0]
    grid_timestamps: list[int] = [samples[0].local_timestamp]
    actual_timestamps: list[int] = [samples[0].local_timestamp]
    source_row_numbers: list[int] = [samples[0].source_row_number]

    prev_sample = samples[0]
    prev_grid_timestamp = float(samples[0].local_timestamp)

    for sample in samples[1:]:
        gap_us = sample.local_timestamp - prev_sample.local_timestamp
        classification = classify_gap(
            gap_us=gap_us,
            base_gap_us=base_gap_us,
            max_interp_gap_steps=max_interp_gap_steps,
            relative_tolerance=relative_tolerance,
            absolute_tolerance_us=absolute_tolerance_us,
        )

        if classification is not None:
            stats.setdefault(f"classified_k{classification.k_steps}", 0.0)
            stats[f"classified_k{classification.k_steps}"] += 1

            for step in range(1, classification.k_steps):
                alpha = step / classification.k_steps
                interpolated = (
                    (1.0 - alpha) * prev_sample.amplitude + alpha * sample.amplitude
                ).astype(np.float32)
                prev_grid_timestamp += classification.step_gap_us
                amplitudes.append(interpolated)
                interp_mask.append(1)
                delta_t_ms.append(classification.step_gap_us / 1000.0)
                grid_timestamps.append(int(round(prev_grid_timestamp)))
                actual_timestamps.append(-1)
                source_row_numbers.append(-1)
                stats["interpolated_points"] += 1

            prev_grid_timestamp += classification.step_gap_us
            amplitudes.append(sample.amplitude)
            interp_mask.append(0)
            delta_t_ms.append(classification.step_gap_us / 1000.0)
            grid_timestamps.append(int(round(prev_grid_timestamp)))
            actual_timestamps.append(sample.local_timestamp)
            source_row_numbers.append(sample.source_row_number)

            if classification.k_steps == 1:
                if base_gap_us <= 0:
                    base_gap_us = float(gap_us)
                else:
                    base_gap_us = ema_alpha * float(gap_us) + (1.0 - ema_alpha) * base_gap_us
                stats["ema_updates"] += 1
        else:
            if gap_us <= 0:
                stats["nonpositive_gap_splits"] += 1
            elif base_gap_us > 0 and gap_us > long_gap_factor * base_gap_us:
                stats["long_gap_splits"] += 1
            else:
                stats["unmatched_gap_splits"] += 1

            segment = finalize_segment(
                source_csv=str(csv_path),
                split=split,
                label=label,
                segment_id=segment_id,
                amplitudes=amplitudes,
                interp_mask=interp_mask,
                delta_t_ms=delta_t_ms,
                grid_timestamps=grid_timestamps,
                actual_timestamps=actual_timestamps,
                source_row_numbers=source_row_numbers,
            )
            if segment is not None:
                segments.append(segment)

            segment_id += 1
            amplitudes = [sample.amplitude]
            interp_mask = [0]
            delta_t_ms = [0.0]
            grid_timestamps = [sample.local_timestamp]
            actual_timestamps = [sample.local_timestamp]
            source_row_numbers = [sample.source_row_number]
            prev_grid_timestamp = float(sample.local_timestamp)

        prev_sample = sample

    segment = finalize_segment(
        source_csv=str(csv_path),
        split=split,
        label=label,
        segment_id=segment_id,
        amplitudes=amplitudes,
        interp_mask=interp_mask,
        delta_t_ms=delta_t_ms,
        grid_timestamps=grid_timestamps,
        actual_timestamps=actual_timestamps,
        source_row_numbers=source_row_numbers,
    )
    if segment is not None:
        segments.append(segment)

    stats["segments"] = len(segments)
    return segments, stats


def save_window(
    output_path: Path,
    segment: SequenceSegment,
    start_index: int,
    end_index: int,
    variant_name: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, np.ndarray | str | int] = {
        "amplitude": segment.amplitudes[start_index:end_index].astype(np.float16),
        "source_csv": segment.source_csv,
        "split": segment.split,
        "label_name": segment.label,
        "segment_id": np.asarray(segment.segment_id, dtype=np.int32),
        "start_index": np.asarray(start_index, dtype=np.int32),
        "grid_timestamps": segment.grid_timestamps[start_index:end_index],
        "actual_timestamps": segment.actual_timestamps[start_index:end_index],
        "source_row_numbers": segment.source_row_numbers[start_index:end_index],
    }
    if variant_name in {"interp_mask", "interp_mask_deltat"}:
        payload["interp_mask"] = segment.interp_mask[start_index:end_index]
    if variant_name == "interp_mask_deltat":
        payload["delta_t_ms"] = segment.delta_t_ms[start_index:end_index]

    np.savez_compressed(output_path, **payload)


def build_windows(
    segment: SequenceSegment,
    output_root: Path,
    window_length: int,
    window_stride: int,
    stats: dict[str, int],
) -> None:
    total_steps = len(segment.amplitudes)
    if total_steps < window_length:
        return

    csv_stem = Path(segment.source_csv).stem
    for start_index in range(0, total_steps - window_length + 1, window_stride):
        end_index = start_index + window_length
        for variant_name in VARIANT_NAMES:
            output_path = (
                output_root
                / variant_name
                / f"windows_{window_length}"
                / segment.split
                / segment.label
                / f"{csv_stem}_seg{segment.segment_id:02d}_start{start_index:04d}.npz"
            )
            save_window(output_path, segment, start_index, end_index, variant_name)
            stats[variant_name]["windows"] += 1
            stats[variant_name][f"{segment.split}_{segment.label}_windows"] += 1


def process_csv(
    csv_path_str: str,
    split: str,
    label: str,
    output_root_str: str,
    window_length: int,
    window_stride: int,
    max_interp_gap_steps: int,
    ema_span: int,
    initial_median_gaps: int,
    relative_tolerance: float,
    absolute_tolerance_us: int,
    long_gap_factor: float,
) -> dict[str, object]:
    csv_path = Path(csv_path_str)
    output_root = Path(output_root_str)
    samples = load_observed_samples(csv_path)

    source_stats: dict[str, float] = {
        "files_seen": 1,
        "files_with_samples": 1 if samples else 0,
        "rows_seen": len(samples),
        "segments": 0,
        "classified_k1": 0,
        "classified_k2": 0,
        "classified_k3": 0,
        "interpolated_points": 0,
        "ema_updates": 0,
        "long_gap_splits": 0,
        "unmatched_gap_splits": 0,
        "nonpositive_gap_splits": 0,
        "initial_base_gap_us_sum": 0.0,
        "initial_base_gap_count": 0.0,
    }
    variant_stats: dict[str, dict[str, int]] = {
        variant_name: {
            "windows": 0,
            "train_big_windows": 0,
            "train_small_windows": 0,
            "validation_big_windows": 0,
            "validation_small_windows": 0,
        }
        for variant_name in VARIANT_NAMES
    }

    if not samples:
        return {"source_stats": source_stats, "variant_stats": variant_stats}

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
    for key, value in adaptive_stats.items():
        source_stats[key] = source_stats.get(key, 0.0) + float(value)

    for segment in segments:
        build_windows(
            segment=segment,
            output_root=output_root,
            window_length=window_length,
            window_stride=window_stride,
            stats=variant_stats,
        )

    return {"source_stats": source_stats, "variant_stats": variant_stats}


def process_csv_task(
    task: tuple[str, str, str, str, int, int, int, int, int, float, int, float]
) -> dict[str, object]:
    return process_csv(*task)


def merge_nested_stats(
    total_source_stats: dict[str, float],
    total_variant_stats: dict[str, dict[str, int]],
    partial: dict[str, object],
) -> None:
    partial_source = partial["source_stats"]
    partial_variant = partial["variant_stats"]

    for key, value in partial_source.items():
        total_source_stats[key] = total_source_stats.get(key, 0.0) + float(value)

    for variant_name, variant_payload in partial_variant.items():
        for key, value in variant_payload.items():
            total_variant_stats[variant_name][key] = (
                total_variant_stats[variant_name].get(key, 0) + int(value)
            )


def write_manifest(output_root: Path, window_length: int) -> None:
    for variant_name in VARIANT_NAMES:
        manifest_path = output_root / variant_name / f"windows_{window_length}" / "manifest.csv"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "window_path",
                    "split",
                    "label",
                    "source_csv",
                    "segment_id",
                    "start_index",
                ],
            )
            writer.writeheader()

            for split in SPLITS:
                for label in CLASS_NAMES:
                    for window_path in sorted(
                        (output_root / variant_name / f"windows_{window_length}" / split / label).glob("*.npz")
                    ):
                        stem = window_path.stem
                        parts = stem.split("_seg")
                        source_csv_stem = parts[0]
                        seg_part, start_part = parts[1].split("_start")
                        writer.writerow(
                            {
                                "window_path": str(window_path),
                                "split": split,
                                "label": label,
                                "source_csv": source_csv_stem + ".csv",
                                "segment_id": int(seg_part),
                                "start_index": int(start_part),
                            }
                        )


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[str, str, str, str, int, int, int, int, int, float, int, float]] = []
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

    merged_source_stats: dict[str, float] = {
        "files_seen": 0.0,
        "files_with_samples": 0.0,
        "rows_seen": 0.0,
        "segments": 0.0,
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
    merged_variant_stats: dict[str, dict[str, int]] = {
        variant_name: {
            "windows": 0,
            "train_big_windows": 0,
            "train_small_windows": 0,
            "validation_big_windows": 0,
            "validation_small_windows": 0,
        }
        for variant_name in VARIANT_NAMES
    }

    max_workers = max(1, args.num_workers)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for partial in executor.map(
            process_csv_task,
            tasks,
            chunksize=max(1, math.ceil(len(tasks) / max_workers)),
        ):
            merge_nested_stats(merged_source_stats, merged_variant_stats, partial)

    write_manifest(args.output_root, args.window_length)

    initial_gap_mean_us = None
    if merged_source_stats["initial_base_gap_count"] > 0:
        initial_gap_mean_us = (
            merged_source_stats["initial_base_gap_us_sum"]
            / merged_source_stats["initial_base_gap_count"]
        )

    summary = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "window_length": args.window_length,
        "window_stride": args.window_stride,
        "adaptive_gap_model": {
            "initial_median_gaps": args.initial_median_gaps,
            "ema_span": args.ema_span,
            "ema_alpha": 2.0 / (float(args.ema_span) + 1.0),
            "max_interp_gap_steps": args.max_interp_gap_steps,
            "relative_tolerance": args.relative_tolerance,
            "absolute_tolerance_us": args.absolute_tolerance_us,
            "long_gap_factor": args.long_gap_factor,
        },
        "num_workers": max_workers,
        "source_stats": {
            "files_seen": int(merged_source_stats["files_seen"]),
            "files_with_samples": int(merged_source_stats["files_with_samples"]),
            "rows_seen": int(merged_source_stats["rows_seen"]),
            "segments": int(merged_source_stats["segments"]),
            "classified_k1": int(merged_source_stats["classified_k1"]),
            "classified_k2": int(merged_source_stats["classified_k2"]),
            "classified_k3": int(merged_source_stats["classified_k3"]),
            "interpolated_points": int(merged_source_stats["interpolated_points"]),
            "ema_updates": int(merged_source_stats["ema_updates"]),
            "long_gap_splits": int(merged_source_stats["long_gap_splits"]),
            "unmatched_gap_splits": int(merged_source_stats["unmatched_gap_splits"]),
            "nonpositive_gap_splits": int(merged_source_stats["nonpositive_gap_splits"]),
            "initial_base_gap_us_mean": initial_gap_mean_us,
        },
        "variants": merged_variant_stats,
    }

    summary_path = args.output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
