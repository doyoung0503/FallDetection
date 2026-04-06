#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from csi_amplitude_normalization import normalize_htltf_complex_with_lltf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "dataset" / "esp32_raw_csi_variants" / "htltf_only"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "dataset" / "esp32_xfall_sdp_htltf_w100_l20_s10_tol4000"

HTLTF_INPUT_DIM = 228
LLTF_HTLTF_INPUT_DIM = 332
NUM_LLTF = 52
NUM_SUBCARRIERS = 114
CLASS_NAMES = ["big", "small"]
SPLITS = ["train", "validation"]


@dataclass
class ObservedSample:
    local_timestamp: int
    source_row_number: int
    csi: np.ndarray


@dataclass
class ComplexSegment:
    source_csv: str
    split: str
    label: str
    segment_id: int
    csi: np.ndarray
    interp_mask: np.ndarray
    grid_timestamps: np.ndarray
    actual_timestamps: np.ndarray
    source_row_numbers: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build paper-style XFall SDP windows for the esp32 HT-LTF-only dataset "
            "while preserving the existing person-based train/validation split."
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
        help="Root directory where SDP windows and summary files will be written.",
    )
    parser.add_argument(
        "--input-format",
        choices=["htltf_only", "lltf_htltf_norm"],
        default="htltf_only",
        help=(
            "Input CSV payload format. 'htltf_only' uses 114 HT-LTF complex bins. "
            "'lltf_htltf_norm' uses 52 LLTF + 114 HT-LTF bins and scales HT-LTF "
            "with an LLTF-derived scalar before SDP construction."
        ),
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
        "--window-length",
        type=int,
        default=100,
        help="Window length in resampled steps used to form each SDP sample.",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=10,
        help="Sliding-window stride in resampled steps.",
    )
    parser.add_argument(
        "--lag-steps",
        type=int,
        default=20,
        help="Number of lag steps N_delta used to build the SDP.",
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
        help="How to make SDP columns non-negative before column-wise normalization.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        help="Small constant to avoid divide-by-zero.",
    )
    return parser.parse_args()


def parse_complex_csi(data_str: str, input_format: str) -> np.ndarray:
    values = ast.literal_eval(data_str)
    if input_format == "htltf_only":
        if len(values) != HTLTF_INPUT_DIM:
            raise ValueError(f"Expected {HTLTF_INPUT_DIM} values, got {len(values)}")
        arr = np.asarray(values, dtype=np.float32).reshape(NUM_SUBCARRIERS, 2)
        return (arr[:, 1] + 1j * arr[:, 0]).astype(np.complex64)

    if input_format == "lltf_htltf_norm":
        if len(values) != LLTF_HTLTF_INPUT_DIM:
            raise ValueError(f"Expected {LLTF_HTLTF_INPUT_DIM} values, got {len(values)}")
        arr = np.asarray(values, dtype=np.float32).reshape(NUM_LLTF + NUM_SUBCARRIERS, 2)
        csi = (arr[:, 1] + 1j * arr[:, 0]).astype(np.complex64)
        h_l = csi[:NUM_LLTF]
        h_ht = csi[NUM_LLTF:]
        return normalize_htltf_complex_with_lltf(h_l, h_ht)

    raise ValueError(f"Unsupported input format: {input_format}")


def expected_len_for(input_format: str) -> int:
    if input_format == "htltf_only":
        return HTLTF_INPUT_DIM
    if input_format == "lltf_htltf_norm":
        return LLTF_HTLTF_INPUT_DIM
    raise ValueError(f"Unsupported input format: {input_format}")


def load_observed_samples(csv_path: Path, input_format: str) -> list[ObservedSample]:
    samples: list[ObservedSample] = []
    expected_len = expected_len_for(input_format)
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            if row.get("len") != str(expected_len):
                continue
            try:
                csi = parse_complex_csi(row["data"], input_format=input_format)
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
    csi_rows: list[np.ndarray],
    interp_mask: list[int],
    grid_timestamps: list[int],
    actual_timestamps: list[int],
    source_row_numbers: list[int],
) -> ComplexSegment | None:
    if not csi_rows:
        return None

    return ComplexSegment(
        source_csv=source_csv,
        split=split,
        label=label,
        segment_id=segment_id,
        csi=np.stack(csi_rows, axis=0).astype(np.complex64),
        interp_mask=np.asarray(interp_mask, dtype=np.uint8),
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
                split=split,
                label=label,
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
        split=split,
        label=label,
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


def compute_sdp(
    window_csi: np.ndarray,
    lag_steps: int,
    rho_mode: str,
    column_normalization: str,
    epsilon: float,
) -> np.ndarray:
    window_length, num_subcarriers = window_csi.shape
    if lag_steps <= 0:
        raise ValueError("lag_steps must be positive")
    if window_length <= lag_steps:
        raise ValueError("window_length must be greater than lag_steps")

    wt = window_length - lag_steps
    current = window_csi[lag_steps:, :]
    current_abs = np.abs(current)
    rho = np.empty((lag_steps, wt, num_subcarriers), dtype=np.float32)

    for lag in range(1, lag_steps + 1):
        past = window_csi[lag_steps - lag : window_length - lag, :]
        denom = current_abs * np.abs(past) + epsilon
        normalized = current * np.conj(past) / denom
        rho[lag - 1] = realify_rho(normalized, rho_mode)

    merged = rho.mean(axis=2)
    sdp = np.empty_like(merged)
    for time_index in range(wt):
        column = merged[:, time_index].copy()
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
        sdp[:, time_index] = column

    return sdp.astype(np.float32)


def save_window(
    output_path: Path,
    segment: ComplexSegment,
    start_index: int,
    end_index: int,
    sdp: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        sdp=sdp.astype(np.float16),
        interp_mask=segment.interp_mask[start_index:end_index],
        grid_timestamps=segment.grid_timestamps[start_index:end_index],
        actual_timestamps=segment.actual_timestamps[start_index:end_index],
        source_row_numbers=segment.source_row_numbers[start_index:end_index],
        source_csv=segment.source_csv,
        split=segment.split,
        label_name=segment.label,
        segment_id=np.asarray(segment.segment_id, dtype=np.int32),
        start_index=np.asarray(start_index, dtype=np.int32),
    )


def build_windows_for_segment(
    segment: ComplexSegment,
    output_root: Path,
    window_length: int,
    window_stride: int,
    lag_steps: int,
    rho_mode: str,
    column_normalization: str,
    epsilon: float,
    stats: dict[str, int],
) -> None:
    total_steps = len(segment.csi)
    if total_steps < window_length:
        return

    csv_stem = Path(segment.source_csv).stem
    for start_index in range(0, total_steps - window_length + 1, window_stride):
        end_index = start_index + window_length
        window_csi = segment.csi[start_index:end_index]
        sdp = compute_sdp(
            window_csi=window_csi,
            lag_steps=lag_steps,
            rho_mode=rho_mode,
            column_normalization=column_normalization,
            epsilon=epsilon,
        )
        output_path = (
            output_root
            / f"windows_{window_length}"
            / segment.split
            / segment.label
            / f"{csv_stem}_seg{segment.segment_id:02d}_start{start_index:04d}.npz"
        )
        save_window(output_path, segment, start_index, end_index, sdp)
        stats["windows"] += 1
        stats[f"{segment.split}_{segment.label}_windows"] += 1


def write_manifest(output_root: Path, window_length: int) -> None:
    manifest_path = output_root / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["window_path", "split", "label", "source_csv", "segment_id", "start_index"],
        )
        writer.writeheader()
        for split in SPLITS:
            for label in CLASS_NAMES:
                class_dir = output_root / f"windows_{window_length}" / split / label
                for window_path in sorted(class_dir.glob("*.npz")):
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

    stats = {
        "files_seen": 0,
        "files_with_samples": 0,
        "rows_seen": 0,
        "segments": 0,
        "windows": 0,
        "train_big_windows": 0,
        "train_small_windows": 0,
        "validation_big_windows": 0,
        "validation_small_windows": 0,
    }

    for split in SPLITS:
        for label in CLASS_NAMES:
            class_dir = args.input_root / split / label
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing directory: {class_dir}")

            for csv_path in sorted(class_dir.glob("*.csv")):
                stats["files_seen"] += 1
                samples = load_observed_samples(csv_path, input_format=args.input_format)
                stats["rows_seen"] += len(samples)
                if not samples:
                    continue
                stats["files_with_samples"] += 1

                segments = build_segments(
                    csv_path=csv_path,
                    split=split,
                    label=label,
                    samples=samples,
                    grid_us=args.grid_us,
                    grid_tolerance_us=args.grid_tolerance_us,
                    max_interp_gap_steps=args.max_interp_gap_steps,
                )
                stats["segments"] += len(segments)
                for segment in segments:
                    build_windows_for_segment(
                        segment=segment,
                        output_root=output_root,
                        window_length=args.window_length,
                        window_stride=args.window_stride,
                        lag_steps=args.lag_steps,
                        rho_mode=args.rho_mode,
                        column_normalization=args.column_normalization,
                        epsilon=args.epsilon,
                        stats=stats,
                    )

    write_manifest(output_root, args.window_length)

    summary = {
        "input_root": str(args.input_root),
        "output_root": str(output_root),
        "input_format": args.input_format,
        "grid_us": args.grid_us,
        "grid_tolerance_us": args.grid_tolerance_us,
        "max_interp_gap_steps": args.max_interp_gap_steps,
        "window_length": args.window_length,
        "window_stride": args.window_stride,
        "lag_steps": args.lag_steps,
        "rho_mode": args.rho_mode,
        "column_normalization": args.column_normalization,
        "stats": stats,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
