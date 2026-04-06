#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "dataset" / "esp32_raw_csi_variants" / "htltf_only"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "dataset" / "esp32_sequence_htltf_variants_w64_s20"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build esp32 HT-LTF sequence datasets with three feature variants: "
            "interpolation only, interpolation+mask, interpolation+mask+delta_t."
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
        "--grid-us",
        type=int,
        default=11_000,
        help="Target resampling interval in microseconds.",
    )
    parser.add_argument(
        "--grid-tolerance-us",
        type=int,
        default=3_000,
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
        "--window-length",
        type=int,
        default=64,
        help="Sliding-window length in resampled steps.",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=20,
        help="Sliding-window stride in resampled steps.",
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

    grid_ms = grid_us / 1000.0
    segments: list[SequenceSegment] = []
    segment_id = 0

    amplitudes: list[np.ndarray] = [samples[0].amplitude]
    interp_mask: list[int] = [0]
    delta_t_ms: list[float] = [0.0]
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
                interp_mask.append(1)
                delta_t_ms.append(grid_ms)
                grid_timestamps.append(prev_grid_timestamp)
                actual_timestamps.append(-1)
                source_row_numbers.append(-1)

            prev_grid_timestamp += grid_us
            amplitudes.append(sample.amplitude)
            interp_mask.append(0)
            delta_t_ms.append(gap_us / 1000.0)
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
            prev_grid_timestamp = sample.local_timestamp

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

    return segments


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
    stats: dict[str, dict[str, int]],
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


def write_manifest(output_root: Path, window_length: int) -> None:
    for variant_name in VARIANT_NAMES:
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
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "input_root": str(args.input_root),
        "output_root": str(output_root),
        "grid_us": args.grid_us,
        "grid_tolerance_us": args.grid_tolerance_us,
        "max_interp_gap_steps": args.max_interp_gap_steps,
        "interp_mode": args.interp_mode,
        "window_length": args.window_length,
        "window_stride": args.window_stride,
        "variants": {variant_name: {} for variant_name in VARIANT_NAMES},
        "source_stats": {
            "files_seen": 0,
            "files_with_samples": 0,
            "rows_seen": 0,
            "segments": 0,
        },
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

    for split in SPLITS:
        for label in CLASS_NAMES:
            class_dir = args.input_root / split / label
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing directory: {class_dir}")

            for csv_path in sorted(class_dir.glob("*.csv")):
                summary["source_stats"]["files_seen"] += 1
                samples = load_observed_samples(csv_path)
                summary["source_stats"]["rows_seen"] += len(samples)
                if not samples:
                    continue

                summary["source_stats"]["files_with_samples"] += 1
                segments = build_segments(
                    csv_path=csv_path,
                    split=split,
                    label=label,
                    samples=samples,
                    grid_us=args.grid_us,
                    grid_tolerance_us=args.grid_tolerance_us,
                    max_interp_gap_steps=args.max_interp_gap_steps,
                    interp_mode=args.interp_mode,
                )
                summary["source_stats"]["segments"] += len(segments)

                for segment in segments:
                    build_windows(
                        segment=segment,
                        output_root=output_root,
                        window_length=args.window_length,
                        window_stride=args.window_stride,
                        stats=variant_stats,
                    )

    write_manifest(output_root, args.window_length)

    for variant_name in VARIANT_NAMES:
        summary["variants"][variant_name] = variant_stats[variant_name]

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
