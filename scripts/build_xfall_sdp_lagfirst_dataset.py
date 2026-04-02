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
    csi: np.ndarray


@dataclass
class ComplexSegment:
    source_csv: str
    segment_id: int
    csi: np.ndarray
    interp_mask: np.ndarray
    grid_timestamps: np.ndarray
    actual_timestamps: np.ndarray
    source_row_numbers: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build lag-first XFall-style datasets from preprocessed HT-LTF CSI. "
            "For each resampled time step, compute a lag-correlation vector against "
            "the previous lag_steps CSI samples, then slice those lag vectors into "
            "fixed-length windows."
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
        default=Path("dataset/xfall_sdp_lagfirst_10ms_real_shift"),
        help="Root directory where lag-first SDP windows and manifests are written.",
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
        "--window-lengths",
        type=int,
        nargs="+",
        default=[50],
        help="Number of consecutive lag-profile columns kept in each output window.",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=1,
        help="Sliding-window stride over the lag-profile sequence.",
    )
    parser.add_argument(
        "--lag-steps",
        type=int,
        default=50,
        help="Number of lag steps used to build each lag-profile column.",
    )
    parser.add_argument(
        "--rho-mode",
        choices=["real", "abs-real", "magnitude"],
        default="real",
        help="How to convert the normalized complex lag product into a real tensor.",
    )
    parser.add_argument(
        "--column-normalization",
        choices=["shift", "clamp", "none"],
        default="shift",
        help="How to make each lag-profile column non-negative before normalization.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        help="Small constant to avoid divide-by-zero.",
    )
    return parser.parse_args()


def discover_class_names(dataset_root: Path) -> list[str]:
    return sorted(path.name for path in dataset_root.iterdir() if path.is_dir())


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
            except Exception:
                continue

            samples.append(
                ObservedSample(
                    local_timestamp=int(row["local_timestamp"]),
                    source_row_number=row_number,
                    csi=csi,
                )
            )
    return samples


def finalize_segment(
    source_csv: str,
    segment_id: int,
    csi_rows: list[np.ndarray],
    interp_mask: list[int],
    grid_timestamps: list[int],
    actual_timestamps: list[int],
    source_row_numbers: list[int],
) -> ComplexSegment | None:
    if not csi_rows:
        return None

    return ComplexSegment(
        source_csv=str(source_csv),
        segment_id=segment_id,
        csi=np.stack(csi_rows, axis=0).astype(np.complex64),
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
) -> list[ComplexSegment]:
    if not samples:
        return []

    segments: list[ComplexSegment] = []
    segment_id = 0

    csi_rows: list[np.ndarray] = [samples[0].csi]
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
                    (1.0 - alpha) * prev_sample.csi + alpha * sample.csi
                ).astype(np.complex64)
                prev_grid_timestamp += grid_us
                csi_rows.append(interpolated)
                interp_mask.append(1)
                grid_timestamps.append(prev_grid_timestamp)
                actual_timestamps.append(-1)
                source_row_numbers.append(-1)

            prev_grid_timestamp += grid_us
            csi_rows.append(sample.csi)
            interp_mask.append(0)
            grid_timestamps.append(prev_grid_timestamp)
            actual_timestamps.append(sample.local_timestamp)
            source_row_numbers.append(sample.source_row_number)
        else:
            segment = finalize_segment(
                source_csv=str(csv_path),
                segment_id=segment_id,
                csi_rows=csi_rows,
                interp_mask=interp_mask,
                grid_timestamps=grid_timestamps,
                actual_timestamps=actual_timestamps,
                source_row_numbers=source_row_numbers,
            )
            if segment is not None:
                segments.append(segment)

            segment_id += 1
            csi_rows = [sample.csi]
            interp_mask = [0]
            grid_timestamps = [sample.local_timestamp]
            actual_timestamps = [sample.local_timestamp]
            source_row_numbers = [sample.source_row_number]
            prev_grid_timestamp = sample.local_timestamp

        prev_sample = sample

    segment = finalize_segment(
        source_csv=str(csv_path),
        segment_id=segment_id,
        csi_rows=csi_rows,
        interp_mask=interp_mask,
        grid_timestamps=grid_timestamps,
        actual_timestamps=actual_timestamps,
        source_row_numbers=source_row_numbers,
    )
    if segment is not None:
        segments.append(segment)

    return segments


def realify_rho(values: np.ndarray, rho_mode: str) -> np.ndarray:
    if rho_mode == "real":
        return np.real(values).astype(np.float32)
    if rho_mode == "abs-real":
        return np.abs(np.real(values)).astype(np.float32)
    if rho_mode == "magnitude":
        return np.abs(values).astype(np.float32)
    raise ValueError(f"Unsupported rho mode: {rho_mode}")


def compute_lag_profile_sequence(
    csi: np.ndarray,
    lag_steps: int,
    rho_mode: str,
    column_normalization: str,
    epsilon: float,
) -> np.ndarray:
    segment_length, num_subcarriers = csi.shape
    if lag_steps <= 0:
        raise ValueError("lag_steps must be positive")
    if segment_length <= lag_steps:
        raise ValueError("segment_length must be greater than lag_steps")

    valid_columns = segment_length - lag_steps
    current = csi[lag_steps:, :]
    current_abs = np.abs(current)
    rho = np.empty((lag_steps, valid_columns, num_subcarriers), dtype=np.float32)

    for lag in range(1, lag_steps + 1):
        past = csi[lag_steps - lag : segment_length - lag, :]
        denom = current_abs * np.abs(past) + epsilon
        normalized = current * np.conj(past) / denom
        rho[lag - 1] = realify_rho(normalized, rho_mode)

    merged = rho.mean(axis=2)
    sdp_sequence = np.empty_like(merged)

    for column_index in range(valid_columns):
        column = merged[:, column_index].copy()

        if column_normalization == "shift":
            min_value = float(column.min())
            if min_value < 0.0:
                column -= min_value
        elif column_normalization == "clamp":
            column = np.clip(column, 0.0, None)
        elif column_normalization != "none":
            raise ValueError(f"Unsupported normalization mode: {column_normalization}")

        column_sum = float(column.sum())
        if column_sum > epsilon:
            column /= column_sum
        else:
            column[:] = 0.0

        sdp_sequence[:, column_index] = column

    return sdp_sequence.astype(np.float32)


def write_window_npz(
    output_path: Path,
    *,
    segment: ComplexSegment,
    start_column: int,
    window_length: int,
    lag_steps: int,
    sdp_window: np.ndarray,
    rho_mode: str,
    column_normalization: str,
    grid_us: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    current_start = lag_steps + start_column
    current_end = current_start + window_length
    context_start = start_column
    context_end = current_end

    np.savez_compressed(
        output_path,
        sdp=sdp_window,
        interp_mask=segment.interp_mask[current_start:current_end],
        current_grid_timestamps=segment.grid_timestamps[current_start:current_end],
        current_actual_timestamps=segment.actual_timestamps[current_start:current_end],
        current_source_row_numbers=segment.source_row_numbers[current_start:current_end],
        context_interp_mask=segment.interp_mask[context_start:context_end],
        context_grid_timestamps=segment.grid_timestamps[context_start:context_end],
        context_actual_timestamps=segment.actual_timestamps[context_start:context_end],
        context_source_row_numbers=segment.source_row_numbers[context_start:context_end],
        source_csv=np.asarray(segment.source_csv),
        segment_id=np.asarray(segment.segment_id, dtype=np.int32),
        start_step=np.asarray(start_column, dtype=np.int32),
        window_length=np.asarray(window_length, dtype=np.int32),
        lag_steps=np.asarray(lag_steps, dtype=np.int32),
        wt=np.asarray(window_length, dtype=np.int32),
        raw_context_length=np.asarray(lag_steps + window_length, dtype=np.int32),
        num_subcarriers=np.asarray(NUM_SUBCARRIERS, dtype=np.int32),
        rho_mode=np.asarray(rho_mode),
        column_normalization=np.asarray(column_normalization),
        grid_us=np.asarray(grid_us, dtype=np.int32),
        lag_first=np.asarray(True),
    )


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    output_root = args.output_root
    class_names = discover_class_names(dataset_root)

    if not class_names:
        raise SystemExit(f"No class folders found in {dataset_root}")

    summary: dict[str, object] = {
        "dataset_root": str(dataset_root.resolve()),
        "output_root": str(output_root.resolve()),
        "grid_us": args.grid_us,
        "grid_tolerance_us": args.grid_tolerance_us,
        "max_interp_gap_steps": args.max_interp_gap_steps,
        "window_lengths": list(args.window_lengths),
        "window_stride": args.window_stride,
        "lag_steps": args.lag_steps,
        "rho_mode": args.rho_mode,
        "column_normalization": args.column_normalization,
        "lag_first": True,
        "num_subcarriers": NUM_SUBCARRIERS,
        "class_names": class_names,
        "classes": {},
    }

    manifest_root = output_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)

    for class_name in class_names:
        class_dir = dataset_root / class_name
        class_summary = {
            "source_files": 0,
            "segments": 0,
            "resampled_steps": 0,
            "interpolated_steps": 0,
            "observed_steps": 0,
            "lag_profile_columns": 0,
            "windows": {str(window_length): 0 for window_length in args.window_lengths},
        }

        manifest_path = manifest_root / f"{class_name}_manifest.csv"
        with manifest_path.open("w", newline="") as manifest_handle:
            fieldnames = [
                "window_path",
                "source_csv",
                "class_name",
                "segment_id",
                "start_step",
                "window_length",
                "lag_steps",
                "wt",
                "raw_context_length",
                "window_interp_count",
                "window_interp_ratio",
                "grid_start",
                "grid_end",
                "rho_mode",
                "column_normalization",
                "lag_first",
            ]
            writer = csv.DictWriter(manifest_handle, fieldnames=fieldnames)
            writer.writeheader()

            for csv_path in sorted(class_dir.glob("*.csv")):
                samples = load_observed_samples(csv_path)
                class_summary["source_files"] += 1
                if not samples:
                    continue

                segments = build_segments(
                    csv_path=csv_path,
                    samples=samples,
                    grid_us=args.grid_us,
                    grid_tolerance_us=args.grid_tolerance_us,
                    max_interp_gap_steps=args.max_interp_gap_steps,
                )

                for segment in segments:
                    segment_length = len(segment.csi)
                    class_summary["segments"] += 1
                    class_summary["resampled_steps"] += segment_length
                    class_summary["interpolated_steps"] += int(segment.interp_mask.sum())
                    class_summary["observed_steps"] += int(
                        segment_length - segment.interp_mask.sum()
                    )

                    if segment_length <= args.lag_steps:
                        continue

                    lag_profile_sequence = compute_lag_profile_sequence(
                        csi=segment.csi,
                        lag_steps=args.lag_steps,
                        rho_mode=args.rho_mode,
                        column_normalization=args.column_normalization,
                        epsilon=args.epsilon,
                    )
                    num_columns = lag_profile_sequence.shape[1]
                    class_summary["lag_profile_columns"] += int(num_columns)

                    for window_length in args.window_lengths:
                        if num_columns < window_length:
                            continue

                        for start_column in range(
                            0,
                            num_columns - window_length + 1,
                            args.window_stride,
                        ):
                            sdp_window = lag_profile_sequence[
                                :, start_column : start_column + window_length
                            ]

                            relative_source = Path(segment.source_csv).name.replace(
                                ".csv", ""
                            )
                            output_name = (
                                f"{relative_source}_seg{segment.segment_id:02d}_"
                                f"start{start_column:04d}.npz"
                            )
                            output_path = (
                                output_root
                                / f"windows_{window_length}"
                                / class_name
                                / output_name
                            )

                            write_window_npz(
                                output_path=output_path,
                                segment=segment,
                                start_column=start_column,
                                window_length=window_length,
                                lag_steps=args.lag_steps,
                                sdp_window=sdp_window,
                                rho_mode=args.rho_mode,
                                column_normalization=args.column_normalization,
                                grid_us=args.grid_us,
                            )

                            current_start = args.lag_steps + start_column
                            current_end = current_start + window_length
                            window_interp = segment.interp_mask[current_start:current_end]
                            writer.writerow(
                                {
                                    "window_path": str(output_path),
                                    "source_csv": segment.source_csv,
                                    "class_name": class_name,
                                    "segment_id": segment.segment_id,
                                    "start_step": start_column,
                                    "window_length": window_length,
                                    "lag_steps": args.lag_steps,
                                    "wt": window_length,
                                    "raw_context_length": args.lag_steps + window_length,
                                    "window_interp_count": int(window_interp.sum()),
                                    "window_interp_ratio": float(window_interp.mean()),
                                    "grid_start": int(
                                        segment.grid_timestamps[start_column]
                                    ),
                                    "grid_end": int(
                                        segment.grid_timestamps[
                                            start_column + args.lag_steps + window_length - 1
                                        ]
                                    ),
                                    "rho_mode": args.rho_mode,
                                    "column_normalization": args.column_normalization,
                                    "lag_first": True,
                                }
                            )
                            class_summary["windows"][str(window_length)] += 1

        summary["classes"][class_name] = class_summary

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
