#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIG_CSV = (
    PROJECT_ROOT
    / "dataset"
    / "esp32_raw_csi_variants"
    / "htltf_only"
    / "train"
    / "big"
    / "csi_260331_012032_minhyeok_big.csv"
)
DEFAULT_SMALL_CSV = (
    PROJECT_ROOT
    / "dataset"
    / "esp32_raw_csi_variants"
    / "htltf_only"
    / "train"
    / "small"
    / "csi_260331_012014_minhyeok_small.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis" / "esp32_amp_phase_visuals"
INPUT_DIM = 228
NUM_SUBCARRIERS = 114


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize HT-LTF amplitude and phase for representative esp32 files."
    )
    parser.add_argument("--big-csv", type=Path, default=DEFAULT_BIG_CSV)
    parser.add_argument("--small-csv", type=Path, default=DEFAULT_SMALL_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=300,
        help="Maximum number of rows to visualize per file.",
    )
    return parser.parse_args()


def load_complex_csi(csv_path: Path, max_rows: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("len") != str(INPUT_DIM):
                continue
            values = ast.literal_eval(row["data"])
            if len(values) != INPUT_DIM:
                continue
            arr = np.asarray(values, dtype=np.float32).reshape(NUM_SUBCARRIERS, 2)
            csi = arr[:, 1] + 1j * arr[:, 0]
            rows.append(csi.astype(np.complex64))
            if len(rows) >= max_rows:
                break
    if not rows:
        raise ValueError(f"No valid CSI rows found in {csv_path}")
    return np.stack(rows, axis=0)


def plot_heatmaps(
    big_csi: np.ndarray,
    small_csi: np.ndarray,
    big_name: str,
    small_name: str,
    output_path: Path,
) -> None:
    amp_big = np.abs(big_csi)
    amp_small = np.abs(small_csi)
    phase_big = np.angle(big_csi)
    phase_small = np.angle(small_csi)

    amp_vmin = min(float(amp_big.min()), float(amp_small.min()))
    amp_vmax = max(float(amp_big.max()), float(amp_small.max()))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

    im0 = axes[0, 0].imshow(
        amp_big.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=amp_vmin,
        vmax=amp_vmax,
    )
    axes[0, 0].set_title(f"Big Amplitude\n{big_name}")
    axes[0, 0].set_xlabel("Time Step")
    axes[0, 0].set_ylabel("Subcarrier")

    im1 = axes[0, 1].imshow(
        phase_big.T,
        aspect="auto",
        origin="lower",
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    axes[0, 1].set_title(f"Big Phase\n{big_name}")
    axes[0, 1].set_xlabel("Time Step")
    axes[0, 1].set_ylabel("Subcarrier")

    im2 = axes[1, 0].imshow(
        amp_small.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=amp_vmin,
        vmax=amp_vmax,
    )
    axes[1, 0].set_title(f"Small Amplitude\n{small_name}")
    axes[1, 0].set_xlabel("Time Step")
    axes[1, 0].set_ylabel("Subcarrier")

    im3 = axes[1, 1].imshow(
        phase_small.T,
        aspect="auto",
        origin="lower",
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    axes[1, 1].set_title(f"Small Phase\n{small_name}")
    axes[1, 1].set_xlabel("Time Step")
    axes[1, 1].set_ylabel("Subcarrier")

    cbar_amp = fig.colorbar(im0, ax=[axes[0, 0], axes[1, 0]], shrink=0.9)
    cbar_amp.set_label("Amplitude")
    cbar_phase = fig.colorbar(im1, ax=[axes[0, 1], axes[1, 1]], shrink=0.9)
    cbar_phase.set_label("Wrapped Phase (rad)")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_subcarrier_summaries(
    big_csi: np.ndarray,
    small_csi: np.ndarray,
    big_name: str,
    small_name: str,
    output_path: Path,
) -> None:
    amp_big = np.abs(big_csi)
    amp_small = np.abs(small_csi)
    phase_big = np.angle(big_csi)
    phase_small = np.angle(small_csi)

    subcarrier_axis = np.arange(NUM_SUBCARRIERS)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

    axes[0, 0].plot(subcarrier_axis, amp_big.mean(axis=0), label="big", linewidth=2)
    axes[0, 0].plot(subcarrier_axis, amp_small.mean(axis=0), label="small", linewidth=2)
    axes[0, 0].set_title("Mean Amplitude per Subcarrier")
    axes[0, 0].set_xlabel("Subcarrier")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].legend()

    axes[0, 1].plot(subcarrier_axis, amp_big.std(axis=0), label="big", linewidth=2)
    axes[0, 1].plot(subcarrier_axis, amp_small.std(axis=0), label="small", linewidth=2)
    axes[0, 1].set_title("Temporal Std of Amplitude per Subcarrier")
    axes[0, 1].set_xlabel("Subcarrier")
    axes[0, 1].set_ylabel("Std")
    axes[0, 1].legend()

    axes[1, 0].plot(subcarrier_axis, phase_big.std(axis=0), label=f"big ({big_name})", linewidth=2)
    axes[1, 0].plot(subcarrier_axis, phase_small.std(axis=0), label=f"small ({small_name})", linewidth=2)
    axes[1, 0].set_title("Wrapped Phase Std per Subcarrier")
    axes[1, 0].set_xlabel("Subcarrier")
    axes[1, 0].set_ylabel("Std (rad)")
    axes[1, 0].legend()

    rep_subcarriers = [10, 40, 70, 100]
    for idx in rep_subcarriers:
        axes[1, 1].plot(amp_big[:, idx], alpha=0.85, label=f"big sc{idx}")
        axes[1, 1].plot(amp_small[:, idx], alpha=0.65, linestyle="--", label=f"small sc{idx}")
    axes[1, 1].set_title("Representative Subcarrier Amplitude Traces")
    axes[1, 1].set_xlabel("Time Step")
    axes[1, 1].set_ylabel("Amplitude")
    axes[1, 1].legend(ncol=2, fontsize=8)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    big_csi = load_complex_csi(args.big_csv, args.max_rows)
    small_csi = load_complex_csi(args.small_csv, args.max_rows)

    heatmap_path = args.output_dir / "amp_phase_heatmaps.png"
    summary_path = args.output_dir / "amp_phase_subcarrier_summary.png"
    json_path = args.output_dir / "summary.json"

    plot_heatmaps(
        big_csi=big_csi,
        small_csi=small_csi,
        big_name=args.big_csv.name,
        small_name=args.small_csv.name,
        output_path=heatmap_path,
    )
    plot_subcarrier_summaries(
        big_csi=big_csi,
        small_csi=small_csi,
        big_name=args.big_csv.name,
        small_name=args.small_csv.name,
        output_path=summary_path,
    )

    summary = {
        "big_csv": str(args.big_csv),
        "small_csv": str(args.small_csv),
        "max_rows": args.max_rows,
        "big_rows_used": int(big_csi.shape[0]),
        "small_rows_used": int(small_csi.shape[0]),
        "big_amplitude_mean": float(np.abs(big_csi).mean()),
        "small_amplitude_mean": float(np.abs(small_csi).mean()),
        "big_amplitude_std": float(np.abs(big_csi).std()),
        "small_amplitude_std": float(np.abs(small_csi).std()),
        "big_phase_std": float(np.angle(big_csi).std()),
        "small_phase_std": float(np.angle(small_csi).std()),
        "heatmap_path": str(heatmap_path),
        "summary_plot_path": str(summary_path),
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
