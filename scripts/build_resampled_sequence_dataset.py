#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


INPUT_DIM = 228
NUM_SUBCARRIERS = 114


@dataclass
class ObservedSample:
    local_timestamp: int
    source_row_number: int
    amplitude: np.ndarray


@dataclass
class SequenceSegment:
    source_csv: str
    segment_id: int
    amplitudes: np.ndarray
    interp_mask: np.ndarray
    grid_timestamps: np.ndarray
    actual_timestamps: np.ndarray
    source_row_numbers: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build 10ms-grid CSI sequence datasets with limited linear interpolation "
            "and sliding windows."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset/preprocessed_raw"),
        help="Root directory containing class-wise preprocessed CSI CSV folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset/sequence_10ms_amp_mask"),
        help="Root directory for the generated sequence datasets.",
    )
    parser.add_argument(
        "--grid-us",
        type=int,
        default=10_000,
        help="Target resampling interval in microseconds.",
    )
    parser.add_argument(
        "--grid-tolerance-us",
        type=int,
        default=2_500,
        help="Tolerance when snapping packet gaps to multiples of the target grid.",
    )
    parser.add_argument(
        "--max-interp-gap-steps",
        type=int,
        default=3,
        help="Only gaps up to this many grid steps are interpolated.",
    )
    parser.add_argument(
        "--window-lengths",
        type=int,
        nargs="+",
        default=[64, 50],
        help="Sliding-window lengths to export as separate datasets.",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=1,
        help="Sliding-window stride in resampled steps.",
    )
    return parser.parse_args()


def discover_class_names(dataset_root: Path) -> list[str]:
    return sorted(path.name for path in dataset_root.iterdir() if path.is_dir())


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
            except Exception:
                continue

            samples.append(
                ObservedSample(
                    local_timestamp=int(row["local_timestamp"]),
                    source_row_number=row_number,
                    amplitude=amplitude,
                )
            )
    return samples


def finalize_segment(
    source_csv: str,
    segment_id: int,
    amplitudes: list[np.ndarray],
    interp_mask: list[int],
    grid_timestamps: list[int],
    actual_timestamps: list[int],
    source_row_numbers: list[int],
) -> SequenceSegment | None:
    if not amplitudes:
        return None

    return SequenceSegment(
        source_csv=source_csv,
        segment_id=segment_id,
        amplitudes=np.stack(amplitudes, axis=0).astype(np.float32),
        interp_mask=np.asarray(interp_mask, dtype=np.uint8),
        grid_timestamps=np.asarray(grid_timestamps, dtype=np.int64),
        actual_timestamps=np.asarray(actual_timestamps, dtype=np.int64),
        source_row_numbers=np.asarray(source_row_numbers, dtype=np.int32),
    )


def build_segments(
    csv_path: Path,
    samples: list[ObservedSample],
    grid_us: int,
    grid_tolerance_us: int,
    max_interp_gap_steps: int,
) -> list[SequenceSegment]:
    if not samples:
        return []

    segments: list[SequenceSegment] = []
    segment_id = 0

    amplitudes: list[np.ndarray] = [samples[0].amplitude]
    interp_mask: list[int] = [0]
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
                interpolated = (
                    (1.0 - alpha) * prev_sample.amplitude + alpha * sample.amplitude
                ).astype(np.float32)
                prev_grid_timestamp += grid_us
                amplitudes.append(interpolated)
                interp_mask.append(1)
                grid_timestamps.append(prev_grid_timestamp)
                actual_timestamps.append(-1)
                source_row_numbers.append(-1)

            prev_grid_timestamp += grid_us
            amplitudes.append(sample.amplitude)
            interp_mask.append(0)
            grid_timestamps.append(prev_grid_timestamp)
            actual_timestamps.append(sample.local_timestamp)
            source_row_numbers.append(sample.source_row_number)
        else:
            segment = finalize_segment(
                source_csv=str(csv_path),
                segment_id=segment_id,
                amplitudes=amplitudes,
                interp_mask=interp_mask,
                grid_timestamps=grid_timestamps,
                actual_timestamps=actual_timestamps,
                source_row_numbers=source_row_numbers,
            )
            if segment is not None:
                segments.append(segment)

            segment_id += 1
            amplitudes = [sample.amplitude]
            interp_mask = [0]
            grid_timestamps = [sample.local_timestamp]
            actual_timestamps = [sample.local_timestamp]
            source_row_numbers = [sample.source_row_number]
            prev_grid_timestamp = sample.local_timestamp

        prev_sample = sample

    segment = finalize_segment(
        source_csv=str(csv_path),
        segment_id=segment_id,
        amplitudes=amplitudes,
        interp_mask=interp_mask,
        grid_timestamps=grid_timestamps,
        actual_timestamps=actual_timestamps,
        source_row_numbers=source_row_numbers,
    )
    if segment is not None:
        segments.append(segment)

    return segments


def write_resampled_csv(
    output_path: Path,
    segment: SequenceSegment,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_csv",
        "segment_id",
        "step_index",
        "relative_time_us",
        "grid_timestamp",
        "actual_timestamp",
        "source_row_number",
        "interp_mask",
    ] + [f"sc_{index:03d}" for index in range(NUM_SUBCARRIERS)]

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        segment_start = int(segment.grid_timestamps[0])

        for step_index in range(len(segment.amplitudes)):
            row = {
                "source_csv": segment.source_csv,
                "segment_id": segment.segment_id,
                "step_index": step_index,
                "relative_time_us": int(segment.grid_timestamps[step_index] - segment_start),
                "grid_timestamp": int(segment.grid_timestamps[step_index]),
                "actual_timestamp": (
                    "" if int(segment.actual_timestamps[step_index]) < 0
                    else int(segment.actual_timestamps[step_index])
                ),
                "source_row_number": (
                    "" if int(segment.source_row_numbers[step_index]) < 0
                    else int(segment.source_row_numbers[step_index])
                ),
                "interp_mask": int(segment.interp_mask[step_index]),
            }
            for subcarrier_idx, value in enumerate(segment.amplitudes[step_index]):
                row[f"sc_{subcarrier_idx:03d}"] = float(value)
            writer.writerow(row)


def save_window_npz(output_path: Path, segment: SequenceSegment, start: int, window_length: int) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    end = start + window_length

    amplitude = segment.amplitudes[start:end]
    interp_mask = segment.interp_mask[start:end]
    grid_timestamps = segment.grid_timestamps[start:end]
    actual_timestamps = segment.actual_timestamps[start:end]
    source_row_numbers = segment.source_row_numbers[start:end]

    np.savez_compressed(
        output_path,
        amplitude=amplitude,
        interp_mask=interp_mask,
        grid_timestamps=grid_timestamps,
        actual_timestamps=actual_timestamps,
        source_row_numbers=source_row_numbers,
        source_csv=np.asarray(segment.source_csv),
        segment_id=np.asarray(segment.segment_id),
        start_step=np.asarray(start),
    )

    return {
        "window_path": str(output_path),
        "source_csv": segment.source_csv,
        "segment_id": segment.segment_id,
        "start_step": start,
        "end_step": end - 1,
        "window_length": window_length,
        "interpolated_steps": int(np.sum(interp_mask)),
    }


def main() -> None:
    args = parse_args()
    class_names = discover_class_names(args.dataset_root)
    if not class_names:
        raise SystemExit(f"No class directories found under: {args.dataset_root}")

    output_root = args.output_root
    resampled_root = output_root / "resampled"
    window_roots = {
        window_length: output_root / f"windows_{window_length}"
        for window_length in args.window_lengths
    }

    summary: dict[str, object] = {
        "dataset_root": str(args.dataset_root),
        "output_root": str(output_root),
        "grid_us": args.grid_us,
        "grid_tolerance_us": args.grid_tolerance_us,
        "max_interp_gap_steps": args.max_interp_gap_steps,
        "window_lengths": args.window_lengths,
        "window_stride": args.window_stride,
        "class_names": class_names,
        "classes": {},
    }

    for window_root in window_roots.values():
        window_root.mkdir(parents=True, exist_ok=True)

    for class_name in class_names:
        class_summary = {
            "source_files": 0,
            "segments": 0,
            "resampled_steps": 0,
            "interpolated_steps": 0,
            "observed_steps": 0,
            "windows": {str(window_length): 0 for window_length in args.window_lengths},
        }
        manifests = {
            window_length: []
            for window_length in args.window_lengths
        }

        for csv_path in sorted((args.dataset_root / class_name).glob("*.csv")):
            class_summary["source_files"] += 1
            samples = load_observed_samples(csv_path)
            segments = build_segments(
                csv_path=csv_path,
                samples=samples,
                grid_us=args.grid_us,
                grid_tolerance_us=args.grid_tolerance_us,
                max_interp_gap_steps=args.max_interp_gap_steps,
            )

            for segment in segments:
                class_summary["segments"] += 1
                class_summary["resampled_steps"] += int(len(segment.amplitudes))
                class_summary["interpolated_steps"] += int(np.sum(segment.interp_mask))
                class_summary["observed_steps"] += int(len(segment.interp_mask) - np.sum(segment.interp_mask))

                resampled_path = (
                    resampled_root
                    / class_name
                    / f"{Path(segment.source_csv).stem}_seg{segment.segment_id:02d}.csv"
                )
                write_resampled_csv(resampled_path, segment)

                for window_length in args.window_lengths:
                    if len(segment.amplitudes) < window_length:
                        continue

                    for start in range(
                        0,
                        len(segment.amplitudes) - window_length + 1,
                        args.window_stride,
                    ):
                        window_path = (
                            window_roots[window_length]
                            / class_name
                            / (
                                f"{Path(segment.source_csv).stem}"
                                f"_seg{segment.segment_id:02d}_start{start:04d}.npz"
                            )
                        )
                        manifest_row = save_window_npz(
                            output_path=window_path,
                            segment=segment,
                            start=start,
                            window_length=window_length,
                        )
                        manifests[window_length].append(manifest_row)
                        class_summary["windows"][str(window_length)] += 1

        summary["classes"][class_name] = class_summary

        for window_length, rows in manifests.items():
            manifest_path = window_roots[window_length] / f"{class_name}_manifest.csv"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with manifest_path.open("w", newline="") as handle:
                fieldnames = [
                    "window_path",
                    "source_csv",
                    "segment_id",
                    "start_step",
                    "end_step",
                    "window_length",
                    "interpolated_steps",
                ]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"saved_summary={summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
