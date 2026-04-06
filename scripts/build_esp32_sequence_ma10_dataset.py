#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "dataset" / "esp32_raw_csi_variants" / "htltf_only"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "dataset" / "esp32_sequence_htltf_ma10diff_w64_s10_tol4000"
)

INPUT_DIM = 228
NUM_SUBCARRIERS = 114
CLASS_NAMES = ["big", "small"]
SPLITS = ["train", "validation"]
DEFAULT_FEATURE_MODE = "ma_residual"


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
    grid_timestamps: np.ndarray
    actual_timestamps: np.ndarray
    source_row_numbers: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build esp32 HT-LTF sequence datasets from amplitude-derived features "
            "such as moving-average residual, first difference, and rolling std."
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
        help="Root directory for the generated dataset.",
    )
    parser.add_argument(
        "--grid-us",
        type=int,
        default=11_000,
        help="Target resampling interval in microseconds.",
    )
    parser.add_argument(
        "--grid-tolerance-us",
        type=int,
        default=4_000,
        help="Tolerance when snapping packet gaps to multiples of the target grid.",
    )
    parser.add_argument(
        "--max-interp-gap-steps",
        type=int,
        default=3,
        help="Only gaps up to this many grid steps are interpolated.",
    )
    parser.add_argument(
        "--interp-mode",
        choices=("linear", "forward_fill", "nearest"),
        default="linear",
        help="Interpolation method used when filling short missing gaps.",
    )
    parser.add_argument(
        "--packet-filter",
        choices=("none", "amp_mad3", "amp_mad4", "amp_mad5"),
        default="none",
        help=(
            "Optional per-file packet filtering before time-axis reconstruction. "
            "amp_madK removes rows whose packet mean amplitude is more than K robust "
            "z-score units away from the file median."
        ),
    )
    parser.add_argument(
        "--moving-average-steps",
        type=int,
        default=10,
        help="History length used by ma_residual and rolling_std features.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=("raw_amplitude", "ma_residual", "first_difference", "rolling_std"),
        default=DEFAULT_FEATURE_MODE,
        help="Derived feature to compute from the resampled amplitude sequence.",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=64,
        help="Sliding-window length in residual steps.",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=10,
        help="Sliding-window stride in residual steps.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, min(8, (os_cpu_count() or 1))),
        help="Parallel workers used for per-file preprocessing.",
    )
    return parser.parse_args()


def os_cpu_count() -> int | None:
    try:
        import os

        return os.cpu_count()
    except Exception:
        return None


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


def filter_observed_samples(
    samples: list[ObservedSample],
    packet_filter: str,
) -> tuple[list[ObservedSample], int]:
    if packet_filter == "none" or not samples:
        return samples, 0

    k_map = {
        "amp_mad3": 3.0,
        "amp_mad4": 4.0,
        "amp_mad5": 5.0,
    }
    if packet_filter not in k_map:
        raise ValueError(f"Unsupported packet_filter: {packet_filter}")

    k = k_map[packet_filter]
    packet_means = np.asarray(
        [float(sample.amplitude.mean()) for sample in samples],
        dtype=np.float32,
    )
    median = float(np.median(packet_means))
    mad = float(np.median(np.abs(packet_means - median)))
    scale = max(mad * 1.4826, 1e-6)
    robust_z = np.abs((packet_means - median) / scale)
    keep_mask = robust_z <= k

    filtered = [sample for sample, keep in zip(samples, keep_mask) if keep]
    removed = int((~keep_mask).sum())
    return filtered, removed


def finalize_segment(
    source_csv: str,
    split: str,
    label: str,
    segment_id: int,
    amplitudes: list[np.ndarray],
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
        grid_timestamps=np.asarray(grid_timestamps, dtype=np.int64),
        actual_timestamps=np.asarray(actual_timestamps, dtype=np.int64),
        source_row_numbers=np.asarray(source_row_numbers, dtype=np.int32),
    )


def build_segments(
    csv_path: Path,
    split: str,
    label: str,
    samples: list[ObservedSample],
    grid_us: int,
    grid_tolerance_us: int,
    max_interp_gap_steps: int,
    interp_mode: str,
) -> list[SequenceSegment]:
    if not samples:
        return []

    segments: list[SequenceSegment] = []
    segment_id = 0

    amplitudes: list[np.ndarray] = [samples[0].amplitude]
    grid_timestamps: list[int] = [samples[0].local_timestamp]
    actual_timestamps: list[int] = [samples[0].local_timestamp]
    source_row_numbers: list[int] = [samples[0].source_row_number]

    prev_sample = samples[0]
    prev_grid_timestamp = samples[0].local_timestamp

    for sample in samples[1:]:
        gap_us = sample.local_timestamp - prev_sample.local_timestamp
        nearest_steps = round(gap_us / grid_us) if gap_us > 0 else -1
        can_snap = (
            gap_us > 0
            and 1 <= nearest_steps <= max_interp_gap_steps
            and abs(gap_us - nearest_steps * grid_us) <= grid_tolerance_us
        )

        if can_snap:
            for step in range(1, nearest_steps):
                alpha = step / nearest_steps
                if interp_mode == "linear":
                    interpolated = (
                        (1.0 - alpha) * prev_sample.amplitude + alpha * sample.amplitude
                    ).astype(np.float32)
                elif interp_mode == "forward_fill":
                    interpolated = prev_sample.amplitude.astype(np.float32)
                elif interp_mode == "nearest":
                    interpolated = (
                        prev_sample.amplitude
                        if alpha < 0.5
                        else sample.amplitude
                    ).astype(np.float32)
                else:
                    raise ValueError(f"Unsupported interp_mode: {interp_mode}")
                prev_grid_timestamp += grid_us
                amplitudes.append(interpolated)
                grid_timestamps.append(prev_grid_timestamp)
                actual_timestamps.append(-1)
                source_row_numbers.append(-1)

            prev_grid_timestamp += grid_us
            amplitudes.append(sample.amplitude)
            grid_timestamps.append(prev_grid_timestamp)
            actual_timestamps.append(sample.local_timestamp)
            source_row_numbers.append(sample.source_row_number)
        else:
            segment = finalize_segment(
                source_csv=str(csv_path),
                split=split,
                label=label,
                segment_id=segment_id,
                amplitudes=amplitudes,
                grid_timestamps=grid_timestamps,
                actual_timestamps=actual_timestamps,
                source_row_numbers=source_row_numbers,
            )
            if segment is not None:
                segments.append(segment)

            segment_id += 1
            amplitudes = [sample.amplitude]
            grid_timestamps = [sample.local_timestamp]
            actual_timestamps = [sample.local_timestamp]
            source_row_numbers = [sample.source_row_number]
            prev_grid_timestamp = sample.local_timestamp

        prev_sample = sample

    segment = finalize_segment(
        source_csv=str(csv_path),
        split=split,
        label=label,
        segment_id=segment_id,
        amplitudes=amplitudes,
        grid_timestamps=grid_timestamps,
        actual_timestamps=actual_timestamps,
        source_row_numbers=source_row_numbers,
    )
    if segment is not None:
        segments.append(segment)

    return segments


def compute_ma_residual_segment(
    segment: SequenceSegment,
    moving_average_steps: int,
) -> SequenceSegment | None:
    total_steps = len(segment.amplitudes)
    if total_steps <= moving_average_steps:
        return None

    amplitudes = segment.amplitudes
    cumsum = np.concatenate(
        [np.zeros((1, amplitudes.shape[1]), dtype=np.float32), amplitudes.cumsum(axis=0)],
        axis=0,
    )
    history_mean = (
        cumsum[moving_average_steps:total_steps] - cumsum[: total_steps - moving_average_steps]
    ) / float(moving_average_steps)
    residual = amplitudes[moving_average_steps:total_steps] - history_mean

    return SequenceSegment(
        source_csv=segment.source_csv,
        split=segment.split,
        label=segment.label,
        segment_id=segment.segment_id,
        amplitudes=residual.astype(np.float32),
        grid_timestamps=segment.grid_timestamps[moving_average_steps:total_steps],
        actual_timestamps=segment.actual_timestamps[moving_average_steps:total_steps],
        source_row_numbers=segment.source_row_numbers[moving_average_steps:total_steps],
    )


def compute_raw_amplitude_segment(segment: SequenceSegment) -> SequenceSegment | None:
    if len(segment.amplitudes) == 0:
        return None

    return SequenceSegment(
        source_csv=segment.source_csv,
        split=segment.split,
        label=segment.label,
        segment_id=segment.segment_id,
        amplitudes=segment.amplitudes.astype(np.float32),
        grid_timestamps=segment.grid_timestamps,
        actual_timestamps=segment.actual_timestamps,
        source_row_numbers=segment.source_row_numbers,
    )


def compute_first_difference_segment(segment: SequenceSegment) -> SequenceSegment | None:
    total_steps = len(segment.amplitudes)
    if total_steps <= 1:
        return None

    diff = segment.amplitudes[1:total_steps] - segment.amplitudes[: total_steps - 1]
    return SequenceSegment(
        source_csv=segment.source_csv,
        split=segment.split,
        label=segment.label,
        segment_id=segment.segment_id,
        amplitudes=diff.astype(np.float32),
        grid_timestamps=segment.grid_timestamps[1:total_steps],
        actual_timestamps=segment.actual_timestamps[1:total_steps],
        source_row_numbers=segment.source_row_numbers[1:total_steps],
    )


def compute_rolling_std_segment(
    segment: SequenceSegment,
    moving_average_steps: int,
) -> SequenceSegment | None:
    total_steps = len(segment.amplitudes)
    if total_steps <= moving_average_steps:
        return None

    amplitudes = segment.amplitudes
    cumsum = np.concatenate(
        [np.zeros((1, amplitudes.shape[1]), dtype=np.float32), amplitudes.cumsum(axis=0)],
        axis=0,
    )
    cumsum_sq = np.concatenate(
        [np.zeros((1, amplitudes.shape[1]), dtype=np.float32), np.square(amplitudes).cumsum(axis=0)],
        axis=0,
    )

    history_sum = cumsum[moving_average_steps:total_steps] - cumsum[: total_steps - moving_average_steps]
    history_sq_sum = (
        cumsum_sq[moving_average_steps:total_steps]
        - cumsum_sq[: total_steps - moving_average_steps]
    )
    history_mean = history_sum / float(moving_average_steps)
    history_var = history_sq_sum / float(moving_average_steps) - np.square(history_mean)
    rolling_std = np.sqrt(np.maximum(history_var, 1e-6))

    return SequenceSegment(
        source_csv=segment.source_csv,
        split=segment.split,
        label=segment.label,
        segment_id=segment.segment_id,
        amplitudes=rolling_std.astype(np.float32),
        grid_timestamps=segment.grid_timestamps[moving_average_steps:total_steps],
        actual_timestamps=segment.actual_timestamps[moving_average_steps:total_steps],
        source_row_numbers=segment.source_row_numbers[moving_average_steps:total_steps],
    )


def transform_segment(
    segment: SequenceSegment,
    feature_mode: str,
    moving_average_steps: int,
) -> SequenceSegment | None:
    if feature_mode == "raw_amplitude":
        return compute_raw_amplitude_segment(segment)
    if feature_mode == "ma_residual":
        return compute_ma_residual_segment(
            segment=segment,
            moving_average_steps=moving_average_steps,
        )
    if feature_mode == "first_difference":
        return compute_first_difference_segment(segment)
    if feature_mode == "rolling_std":
        return compute_rolling_std_segment(
            segment=segment,
            moving_average_steps=moving_average_steps,
        )
    raise ValueError(f"Unsupported feature mode: {feature_mode}")


def variant_name_for(feature_mode: str, moving_average_steps: int) -> str:
    if feature_mode == "raw_amplitude":
        return "raw_amplitude"
    if feature_mode == "ma_residual":
        return f"ma{moving_average_steps}_diff"
    if feature_mode == "first_difference":
        return "first_difference"
    if feature_mode == "rolling_std":
        return f"rolling_std{moving_average_steps}"
    raise ValueError(f"Unsupported feature mode: {feature_mode}")


def save_window(
    output_path: Path,
    segment: SequenceSegment,
    start_index: int,
    end_index: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        amplitude=segment.amplitudes[start_index:end_index].astype(np.float16),
        source_csv=segment.source_csv,
        split=segment.split,
        label_name=segment.label,
        segment_id=np.asarray(segment.segment_id, dtype=np.int32),
        start_index=np.asarray(start_index, dtype=np.int32),
        grid_timestamps=segment.grid_timestamps[start_index:end_index],
        actual_timestamps=segment.actual_timestamps[start_index:end_index],
        source_row_numbers=segment.source_row_numbers[start_index:end_index],
    )


def build_windows(
    segment: SequenceSegment,
    output_root: Path,
    window_length: int,
    window_stride: int,
    variant_name: str,
) -> dict[str, int]:
    total_steps = len(segment.amplitudes)
    if total_steps < window_length:
        return {
            "windows": 0,
            f"{segment.split}_{segment.label}_windows": 0,
        }

    csv_stem = Path(segment.source_csv).stem
    count = 0
    for start_index in range(0, total_steps - window_length + 1, window_stride):
        end_index = start_index + window_length
        output_path = (
            output_root
            / variant_name
            / f"windows_{window_length}"
            / segment.split
            / segment.label
            / f"{csv_stem}_seg{segment.segment_id:02d}_start{start_index:04d}.npz"
        )
        save_window(output_path, segment, start_index, end_index)
        count += 1

    return {
        "windows": count,
        f"{segment.split}_{segment.label}_windows": count,
    }


def process_csv(
    csv_path_str: str,
    split: str,
    label: str,
    output_root_str: str,
    grid_us: int,
    grid_tolerance_us: int,
    max_interp_gap_steps: int,
    interp_mode: str,
    packet_filter: str,
    moving_average_steps: int,
    feature_mode: str,
    window_length: int,
    window_stride: int,
) -> dict[str, int]:
    csv_path = Path(csv_path_str)
    output_root = Path(output_root_str)
    variant_name = variant_name_for(feature_mode, moving_average_steps)
    samples = load_observed_samples(csv_path)

    stats = {
        "files_seen": 1,
        "files_with_samples": 1 if samples else 0,
        "rows_seen": len(samples),
        "rows_filtered_out": 0,
        "segments_before_ma": 0,
        "segments_after_ma": 0,
        "windows": 0,
        "train_big_windows": 0,
        "train_small_windows": 0,
        "validation_big_windows": 0,
        "validation_small_windows": 0,
    }

    if not samples:
        return stats

    samples, removed = filter_observed_samples(samples, packet_filter)
    stats["rows_filtered_out"] = removed
    if not samples:
        return stats

    segments = build_segments(
        csv_path=csv_path,
        split=split,
        label=label,
        samples=samples,
        grid_us=grid_us,
        grid_tolerance_us=grid_tolerance_us,
        max_interp_gap_steps=max_interp_gap_steps,
        interp_mode=interp_mode,
    )
    stats["segments_before_ma"] = len(segments)

    for segment in segments:
        feature_segment = transform_segment(
            segment=segment,
            feature_mode=feature_mode,
            moving_average_steps=moving_average_steps,
        )
        if feature_segment is None:
            continue

        stats["segments_after_ma"] += 1
        window_stats = build_windows(
            segment=feature_segment,
            output_root=output_root,
            window_length=window_length,
            window_stride=window_stride,
            variant_name=variant_name,
        )
        for key, value in window_stats.items():
            stats[key] += value

    return stats


def process_csv_task(
    task: tuple[str, str, str, str, int, int, int, str, str, int, str, int, int]
) -> dict[str, int]:
    return process_csv(*task)


def merge_stats(total: dict[str, int], partial: dict[str, int]) -> None:
    for key, value in partial.items():
        total[key] = total.get(key, 0) + int(value)


def write_manifest(output_root: Path, window_length: int, variant_name: str) -> None:
    manifest_path = output_root / variant_name / f"windows_{window_length}" / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["window_path", "split", "label", "source_csv", "segment_id", "start_index"],
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

    variant_name = variant_name_for(args.feature_mode, args.moving_average_steps)

    tasks: list[tuple[str, str, str, str, int, int, int, str, str, int, str, int, int]] = []
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
                        args.grid_us,
                        args.grid_tolerance_us,
                        args.max_interp_gap_steps,
                        args.interp_mode,
                        args.packet_filter,
                        args.moving_average_steps,
                        args.feature_mode,
                        args.window_length,
                        args.window_stride,
                    )
                )

    merged_stats = {
        "files_seen": 0,
        "files_with_samples": 0,
        "rows_seen": 0,
        "segments_before_ma": 0,
        "segments_after_ma": 0,
        "windows": 0,
        "train_big_windows": 0,
        "train_small_windows": 0,
        "validation_big_windows": 0,
        "validation_small_windows": 0,
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

    summary = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "grid_us": args.grid_us,
        "grid_tolerance_us": args.grid_tolerance_us,
        "max_interp_gap_steps": args.max_interp_gap_steps,
        "interp_mode": args.interp_mode,
        "packet_filter": args.packet_filter,
        "moving_average_steps": args.moving_average_steps,
        "feature_mode": args.feature_mode,
        "window_length": args.window_length,
        "window_stride": args.window_stride,
        "num_workers": max_workers,
        "variant": variant_name,
        "source_stats": {
            "files_seen": merged_stats["files_seen"],
            "files_with_samples": merged_stats["files_with_samples"],
            "rows_seen": merged_stats["rows_seen"],
            "rows_filtered_out": merged_stats["rows_filtered_out"],
            "segments_before_ma": merged_stats["segments_before_ma"],
            "segments_after_ma": merged_stats["segments_after_ma"],
        },
        "variant_stats": {
            "windows": merged_stats["windows"],
            "train_big_windows": merged_stats["train_big_windows"],
            "train_small_windows": merged_stats["train_small_windows"],
            "validation_big_windows": merged_stats["validation_big_windows"],
            "validation_small_windows": merged_stats["validation_small_windows"],
        },
    }

    summary_path = args.output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
