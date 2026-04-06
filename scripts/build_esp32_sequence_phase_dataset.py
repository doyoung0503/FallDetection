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
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "dataset" / "esp32_sequence_phase_sincos_w64_s10_tol4000"

INPUT_DIM = 228
NUM_SUBCARRIERS = 114
CLASS_NAMES = ["big", "small"]
SPLITS = ["train", "validation"]


@dataclass
class ObservedSample:
    local_timestamp: int
    source_row_number: int
    csi: np.ndarray


@dataclass
class SequenceSegment:
    source_csv: str
    split: str
    label: str
    segment_id: int
    csi: np.ndarray
    grid_timestamps: np.ndarray
    actual_timestamps: np.ndarray
    source_row_numbers: np.ndarray


def os_cpu_count() -> int | None:
    try:
        import os

        return os.cpu_count()
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build esp32 HT-LTF phase-derived sequence datasets."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--grid-us", type=int, default=11_000)
    parser.add_argument("--grid-tolerance-us", type=int, default=4_000)
    parser.add_argument("--max-interp-gap-steps", type=int, default=3)
    parser.add_argument(
        "--feature-mode",
        choices=("phase_sin_cos", "phase_temporal_diff", "phase_rolling_std"),
        required=True,
    )
    parser.add_argument(
        "--rolling-steps",
        type=int,
        default=5,
        help="History length used by phase_rolling_std.",
    )
    parser.add_argument("--window-length", type=int, default=64)
    parser.add_argument("--window-stride", type=int, default=10)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, min(8, (os_cpu_count() or 1))),
    )
    return parser.parse_args()


def parse_complex_csi(data_str: str) -> np.ndarray:
    values = ast.literal_eval(data_str)
    if len(values) != INPUT_DIM:
        raise ValueError(f"Expected {INPUT_DIM} values, got {len(values)}")

    arr = np.asarray(values, dtype=np.float32).reshape(NUM_SUBCARRIERS, 2)
    return (arr[:, 1] + 1j * arr[:, 0]).astype(np.complex64)


def load_observed_samples(csv_path: Path) -> list[ObservedSample]:
    samples: list[ObservedSample] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            if row.get("len") != str(INPUT_DIM):
                continue
            try:
                csi = parse_complex_csi(row["data"])
                local_timestamp = int(row["local_timestamp"])
            except Exception:
                continue
            samples.append(
                ObservedSample(
                    local_timestamp=local_timestamp,
                    source_row_number=row_number,
                    csi=csi,
                )
            )
    return samples


def finalize_segment(
    source_csv: str,
    split: str,
    label: str,
    segment_id: int,
    csi_values: list[np.ndarray],
    grid_timestamps: list[int],
    actual_timestamps: list[int],
    source_row_numbers: list[int],
) -> SequenceSegment | None:
    if not csi_values:
        return None
    return SequenceSegment(
        source_csv=source_csv,
        split=split,
        label=label,
        segment_id=segment_id,
        csi=np.stack(csi_values, axis=0).astype(np.complex64),
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
) -> list[SequenceSegment]:
    if not samples:
        return []

    segments: list[SequenceSegment] = []
    segment_id = 0
    csi_values: list[np.ndarray] = [samples[0].csi]
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
                interpolated = ((1.0 - alpha) * prev_sample.csi + alpha * sample.csi).astype(
                    np.complex64
                )
                prev_grid_timestamp += grid_us
                csi_values.append(interpolated)
                grid_timestamps.append(prev_grid_timestamp)
                actual_timestamps.append(-1)
                source_row_numbers.append(-1)

            prev_grid_timestamp += grid_us
            csi_values.append(sample.csi)
            grid_timestamps.append(prev_grid_timestamp)
            actual_timestamps.append(sample.local_timestamp)
            source_row_numbers.append(sample.source_row_number)
        else:
            segment = finalize_segment(
                source_csv=str(csv_path),
                split=split,
                label=label,
                segment_id=segment_id,
                csi_values=csi_values,
                grid_timestamps=grid_timestamps,
                actual_timestamps=actual_timestamps,
                source_row_numbers=source_row_numbers,
            )
            if segment is not None:
                segments.append(segment)
            segment_id += 1
            csi_values = [sample.csi]
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
        csi_values=csi_values,
        grid_timestamps=grid_timestamps,
        actual_timestamps=actual_timestamps,
        source_row_numbers=source_row_numbers,
    )
    if segment is not None:
        segments.append(segment)
    return segments


def circular_std_from_phase(history_phase: np.ndarray) -> np.ndarray:
    mean_cos = np.cos(history_phase).mean(axis=0)
    mean_sin = np.sin(history_phase).mean(axis=0)
    resultant = np.sqrt(np.square(mean_cos) + np.square(mean_sin))
    resultant = np.clip(resultant, 1e-6, 1.0)
    return np.sqrt(np.maximum(-2.0 * np.log(resultant), 0.0)).astype(np.float32)


def transform_segment(
    segment: SequenceSegment,
    feature_mode: str,
    rolling_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    csi = segment.csi
    phase = np.angle(csi).astype(np.float32)
    total_steps = phase.shape[0]

    if feature_mode == "phase_sin_cos":
        features = np.concatenate([np.sin(phase), np.cos(phase)], axis=1).astype(np.float32)
        return (
            features,
            segment.grid_timestamps,
            segment.actual_timestamps,
            segment.source_row_numbers,
        )

    if feature_mode == "phase_temporal_diff":
        if total_steps <= 1:
            return None
        phase_diff = np.angle(csi[1:] * np.conj(csi[:-1])).astype(np.float32)
        return (
            phase_diff,
            segment.grid_timestamps[1:],
            segment.actual_timestamps[1:],
            segment.source_row_numbers[1:],
        )

    if feature_mode == "phase_rolling_std":
        if total_steps <= rolling_steps:
            return None
        rows: list[np.ndarray] = []
        for idx in range(rolling_steps, total_steps):
            history = phase[idx - rolling_steps : idx]
            rows.append(circular_std_from_phase(history))
        features = np.stack(rows, axis=0).astype(np.float32)
        return (
            features,
            segment.grid_timestamps[rolling_steps:],
            segment.actual_timestamps[rolling_steps:],
            segment.source_row_numbers[rolling_steps:],
        )

    raise ValueError(f"Unsupported feature mode: {feature_mode}")


def variant_name_for(feature_mode: str, rolling_steps: int) -> str:
    if feature_mode == "phase_sin_cos":
        return "phase_sin_cos"
    if feature_mode == "phase_temporal_diff":
        return "phase_temporal_diff"
    if feature_mode == "phase_rolling_std":
        return f"phase_rolling_std{rolling_steps}"
    raise ValueError(f"Unsupported feature mode: {feature_mode}")


def save_window(
    output_path: Path,
    features: np.ndarray,
    source_csv: str,
    split: str,
    label: str,
    segment_id: int,
    start_index: int,
    end_index: int,
    grid_timestamps: np.ndarray,
    actual_timestamps: np.ndarray,
    source_row_numbers: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        amplitude=features[start_index:end_index].astype(np.float16),
        source_csv=source_csv,
        split=split,
        label_name=label,
        segment_id=np.asarray(segment_id, dtype=np.int32),
        start_index=np.asarray(start_index, dtype=np.int32),
        grid_timestamps=grid_timestamps[start_index:end_index],
        actual_timestamps=actual_timestamps[start_index:end_index],
        source_row_numbers=source_row_numbers[start_index:end_index],
    )


def build_windows(
    output_root: Path,
    variant_name: str,
    source_csv: str,
    split: str,
    label: str,
    segment_id: int,
    features: np.ndarray,
    grid_timestamps: np.ndarray,
    actual_timestamps: np.ndarray,
    source_row_numbers: np.ndarray,
    window_length: int,
    window_stride: int,
) -> dict[str, int]:
    total_steps = features.shape[0]
    if total_steps < window_length:
        return {"windows": 0, f"{split}_{label}_windows": 0}

    csv_stem = Path(source_csv).stem
    count = 0
    for start_index in range(0, total_steps - window_length + 1, window_stride):
        end_index = start_index + window_length
        output_path = (
            output_root
            / variant_name
            / f"windows_{window_length}"
            / split
            / label
            / f"{csv_stem}_seg{segment_id:02d}_start{start_index:04d}.npz"
        )
        save_window(
            output_path=output_path,
            features=features,
            source_csv=source_csv,
            split=split,
            label=label,
            segment_id=segment_id,
            start_index=start_index,
            end_index=end_index,
            grid_timestamps=grid_timestamps,
            actual_timestamps=actual_timestamps,
            source_row_numbers=source_row_numbers,
        )
        count += 1
    return {"windows": count, f"{split}_{label}_windows": count}


def process_csv(
    csv_path_str: str,
    split: str,
    label: str,
    output_root_str: str,
    grid_us: int,
    grid_tolerance_us: int,
    max_interp_gap_steps: int,
    feature_mode: str,
    rolling_steps: int,
    window_length: int,
    window_stride: int,
) -> dict[str, int]:
    csv_path = Path(csv_path_str)
    output_root = Path(output_root_str)
    variant_name = variant_name_for(feature_mode, rolling_steps)
    samples = load_observed_samples(csv_path)

    stats = {
        "files_seen": 1,
        "files_with_samples": 1 if samples else 0,
        "rows_seen": len(samples),
        "segments_before_feature": 0,
        "segments_after_feature": 0,
        "windows": 0,
        "train_big_windows": 0,
        "train_small_windows": 0,
        "validation_big_windows": 0,
        "validation_small_windows": 0,
    }

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
    )
    stats["segments_before_feature"] = len(segments)

    for segment in segments:
        transformed = transform_segment(
            segment=segment,
            feature_mode=feature_mode,
            rolling_steps=rolling_steps,
        )
        if transformed is None:
            continue

        features, grid_timestamps, actual_timestamps, source_row_numbers = transformed
        stats["segments_after_feature"] += 1
        partial = build_windows(
            output_root=output_root,
            variant_name=variant_name,
            source_csv=segment.source_csv,
            split=segment.split,
            label=segment.label,
            segment_id=segment.segment_id,
            features=features,
            grid_timestamps=grid_timestamps,
            actual_timestamps=actual_timestamps,
            source_row_numbers=source_row_numbers,
            window_length=window_length,
            window_stride=window_stride,
        )
        for key, value in partial.items():
            stats[key] += int(value)
    return stats


def process_csv_task(
    task: tuple[str, str, str, str, int, int, int, str, int, int, int]
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
    variant_name = variant_name_for(args.feature_mode, args.rolling_steps)

    tasks: list[tuple[str, str, str, str, int, int, int, str, int, int, int]] = []
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
                        args.feature_mode,
                        args.rolling_steps,
                        args.window_length,
                        args.window_stride,
                    )
                )

    merged_stats = {
        "files_seen": 0,
        "files_with_samples": 0,
        "rows_seen": 0,
        "segments_before_feature": 0,
        "segments_after_feature": 0,
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
        "feature_mode": args.feature_mode,
        "rolling_steps": args.rolling_steps,
        "window_length": args.window_length,
        "window_stride": args.window_stride,
        "num_workers": max_workers,
        "variant": variant_name,
        "source_stats": {
            "files_seen": merged_stats["files_seen"],
            "files_with_samples": merged_stats["files_with_samples"],
            "rows_seen": merged_stats["rows_seen"],
            "segments_before_feature": merged_stats["segments_before_feature"],
            "segments_after_feature": merged_stats["segments_after_feature"],
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
