#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch import nn


CLASS_NAMES = ["none", "occupy", "walk"]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
COLORS = {
    "none": "#33658A",
    "occupy": "#758E4F",
    "walk": "#D1495B",
}


@dataclass
class WindowSample:
    window_path: str
    source_csv: str
    class_name: str
    label: int


class SequenceCNN(nn.Module):
    def __init__(
        self,
        input_channels: int = 115,
        hidden_channels_1: int = 64,
        hidden_channels_2: int = 128,
        kernel_size_1: int = 5,
        kernel_size_2: int = 3,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(
                input_channels,
                hidden_channels_1,
                kernel_size_1,
                padding=kernel_size_1 // 2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                hidden_channels_1,
                hidden_channels_2,
                kernel_size_2,
                padding=kernel_size_2 // 2,
            ),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_channels_2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return x.mean(dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create class-separation plots for CSI sequence windows."
    )
    parser.add_argument(
        "--windows-root",
        type=Path,
        default=Path("dataset/sequence_10ms_amp_mask_stride20/windows_50"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "artifacts/sequence_cnn_torch_windows50_stride20_timefile_mps_seed42/best_model.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/sequence_stride20_windows50_visuals"),
    )
    parser.add_argument(
        "--split-mode",
        choices=("time_file", "random_file", "time_block"),
        default="time_file",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--time-block-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--embedding-sample-limit",
        type=int,
        default=None,
        help="Optional cap on validation samples used for embedding plots.",
    )
    return parser.parse_args()


def load_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def parse_source_timestamp(source_csv: str) -> datetime:
    stem = Path(source_csv).stem
    parts = stem.split("_")
    return datetime.strptime(f"{parts[-2]}_{parts[-1]}", "%Y%m%d_%H%M%S")


def split_source_files(
    rows_by_class: dict[str, list[dict[str, str]]],
    split_mode: str,
    val_ratio: float,
    rng: np.random.Generator,
    time_block_count: int,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    train_files: dict[str, set[str]] = {}
    val_files: dict[str, set[str]] = {}

    for class_name, rows in rows_by_class.items():
        unique_files = sorted({row["source_csv"] for row in rows})
        if split_mode in {"time_file", "time_block"}:
            unique_files = sorted(unique_files, key=parse_source_timestamp)
        else:
            unique_files = list(unique_files)
            rng.shuffle(unique_files)

        if split_mode == "time_block":
            block_count = max(2, min(time_block_count, len(unique_files)))
            split_arrays = np.array_split(np.asarray(unique_files, dtype=object), block_count)
            blocks = [[str(item) for item in block.tolist()] for block in split_arrays if len(block) > 0]
            val_split = set(blocks[-1])
            train_split = {
                file_path
                for block in blocks[:-1]
                for file_path in block
            }
        else:
            val_count = max(1, int(round(len(unique_files) * val_ratio)))
            val_split = set(unique_files[-val_count:] if split_mode == "time_file" else unique_files[:val_count])
            train_split = set(unique_files[:-val_count] if split_mode == "time_file" else unique_files[val_count:])

        train_files[class_name] = train_split
        val_files[class_name] = val_split

    return train_files, val_files


def build_samples(
    rows_by_class: dict[str, list[dict[str, str]]],
    selected_files_by_class: dict[str, set[str]],
) -> list[WindowSample]:
    samples: list[WindowSample] = []
    for class_name in CLASS_NAMES:
        for row in rows_by_class[class_name]:
            if row["source_csv"] not in selected_files_by_class[class_name]:
                continue
            samples.append(
                WindowSample(
                    window_path=row["window_path"],
                    source_csv=row["source_csv"],
                    class_name=class_name,
                    label=CLASS_TO_INDEX[class_name],
                )
            )
    return samples


def scatter_by_class(ax: plt.Axes, coords: np.ndarray, class_names: list[str], title: str) -> None:
    for class_name in CLASS_NAMES:
        mask = np.asarray([name == class_name for name in class_names])
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=42,
            alpha=0.9,
            label=f"{class_name} (n={int(mask.sum())})",
            color=COLORS[class_name],
            edgecolors="white",
            linewidths=0.4,
        )
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.grid(alpha=0.2, linewidth=0.5)


def save_embedding_plot(
    features: np.ndarray,
    class_names: list[str],
    output_path: Path,
    title_prefix: str,
    random_state: int,
) -> None:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    pca_coords = PCA(n_components=2, random_state=random_state).fit_transform(scaled)
    perplexity = max(5, min(20, (len(features) - 1) // 3))
    tsne_coords = TSNE(
        n_components=2,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
    ).fit_transform(scaled)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    scatter_by_class(axes[0], pca_coords, class_names, f"{title_prefix} PCA")
    scatter_by_class(axes[1], tsne_coords, class_names, f"{title_prefix} t-SNE")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(title_prefix, y=1.14, fontsize=14)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_heatmaps(
    amplitude_by_class: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    mean_maps = {
        class_name: amplitude_by_class[class_name].mean(axis=0)
        for class_name in CLASS_NAMES
    }
    std_maps = {
        class_name: amplitude_by_class[class_name].std(axis=0)
        for class_name in CLASS_NAMES
    }
    mean_vmin = min(float(mean_maps[class_name].min()) for class_name in CLASS_NAMES)
    mean_vmax = max(float(mean_maps[class_name].max()) for class_name in CLASS_NAMES)
    std_vmin = min(float(std_maps[class_name].min()) for class_name in CLASS_NAMES)
    std_vmax = max(float(std_maps[class_name].max()) for class_name in CLASS_NAMES)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    mean_images = []
    std_images = []
    for col, class_name in enumerate(CLASS_NAMES):
        mean_img = axes[0, col].imshow(
            mean_maps[class_name].T,
            aspect="auto",
            origin="lower",
            vmin=mean_vmin,
            vmax=mean_vmax,
            cmap="viridis",
        )
        std_img = axes[1, col].imshow(
            std_maps[class_name].T,
            aspect="auto",
            origin="lower",
            vmin=std_vmin,
            vmax=std_vmax,
            cmap="magma",
        )
        mean_images.append(mean_img)
        std_images.append(std_img)
        axes[0, col].set_title(f"{class_name} mean")
        axes[1, col].set_title(f"{class_name} std")
        axes[0, col].set_xlabel("time step")
        axes[1, col].set_xlabel("time step")
        axes[0, col].set_ylabel("subcarrier")
        axes[1, col].set_ylabel("subcarrier")

    fig.colorbar(mean_images[-1], ax=axes[0, :], shrink=0.8, label="amplitude")
    fig.colorbar(std_images[-1], ax=axes[1, :], shrink=0.8, label="amplitude std")
    fig.suptitle("Validation Class Heatmaps", y=1.02, fontsize=14)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_temporal_variation_plot(
    class_names: list[str],
    mean_abs_deltas: np.ndarray,
    temporal_stds: np.ndarray,
    output_path: Path,
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    delta_groups = []
    std_groups = []
    for class_name in CLASS_NAMES:
        mask = np.asarray([name == class_name for name in class_names])
        delta_values = mean_abs_deltas[mask]
        std_values = temporal_stds[mask]
        delta_groups.append(delta_values)
        std_groups.append(std_values)
        summary[class_name] = {
            "mean_abs_delta_mean": float(delta_values.mean()),
            "mean_abs_delta_std": float(delta_values.std()),
            "temporal_std_mean": float(std_values.mean()),
            "temporal_std_std": float(std_values.std()),
        }

    delta_box = axes[0].boxplot(delta_groups, tick_labels=CLASS_NAMES, patch_artist=True)
    std_box = axes[1].boxplot(std_groups, tick_labels=CLASS_NAMES, patch_artist=True)
    for box_dict in (delta_box, std_box):
        for patch, class_name in zip(box_dict["boxes"], CLASS_NAMES):
            patch.set_facecolor(COLORS[class_name])
            patch.set_alpha(0.7)
    for ax in axes:
        ax.grid(alpha=0.2, linewidth=0.5)
    axes[0].set_title("Mean |delta amplitude|")
    axes[1].set_title("Mean temporal std")
    axes[0].set_ylabel("value")
    axes[1].set_ylabel("value")
    fig.suptitle("Validation Temporal Variation", y=1.02, fontsize=14)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return summary


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    rows_by_class = {
        class_name: load_manifest_rows(args.windows_root / f"{class_name}_manifest.csv")
        for class_name in CLASS_NAMES
    }
    _, val_files = split_source_files(
        rows_by_class=rows_by_class,
        split_mode=args.split_mode,
        val_ratio=args.val_ratio,
        rng=rng,
        time_block_count=args.time_block_count,
    )
    val_samples = build_samples(rows_by_class, val_files)
    if args.embedding_sample_limit is not None:
        val_samples = val_samples[: args.embedding_sample_limit]

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mean = checkpoint["mean"].astype(np.float32)
    std = checkpoint["std"].astype(np.float32)
    model = SequenceCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    raw_features = []
    cnn_features = []
    class_names = []
    amplitude_by_class: dict[str, list[np.ndarray]] = {name: [] for name in CLASS_NAMES}
    mean_abs_deltas = []
    temporal_stds = []

    with torch.no_grad():
        for sample in val_samples:
            data = np.load(sample.window_path)
            amplitude = data["amplitude"].astype(np.float32)
            interp_mask = data["interp_mask"].astype(np.float32)[:, None]

            normalized = (amplitude - mean) / std
            stacked = np.concatenate([normalized, interp_mask], axis=1).transpose(1, 0)
            tensor = torch.from_numpy(stacked[None, ...]).float()
            feature = model.extract_features(tensor).squeeze(0).cpu().numpy()

            raw_features.append(amplitude.reshape(-1))
            cnn_features.append(feature)
            class_names.append(sample.class_name)
            amplitude_by_class[sample.class_name].append(amplitude)
            mean_abs_deltas.append(float(np.abs(np.diff(amplitude, axis=0)).mean()))
            temporal_stds.append(float(amplitude.std(axis=0).mean()))

    raw_features_arr = np.asarray(raw_features, dtype=np.float32)
    cnn_features_arr = np.asarray(cnn_features, dtype=np.float32)
    mean_abs_deltas_arr = np.asarray(mean_abs_deltas, dtype=np.float32)
    temporal_stds_arr = np.asarray(temporal_stds, dtype=np.float32)
    amplitude_stack_by_class = {
        class_name: np.stack(amplitude_by_class[class_name], axis=0).astype(np.float32)
        for class_name in CLASS_NAMES
    }

    save_embedding_plot(
        features=raw_features_arr,
        class_names=class_names,
        output_path=output_dir / "raw_embeddings.png",
        title_prefix="Raw Validation Windows",
        random_state=args.seed,
    )
    save_embedding_plot(
        features=cnn_features_arr,
        class_names=class_names,
        output_path=output_dir / "cnn_embeddings.png",
        title_prefix="CNN Penultimate Features",
        random_state=args.seed,
    )
    save_heatmaps(
        amplitude_by_class=amplitude_stack_by_class,
        output_path=output_dir / "class_heatmaps.png",
    )
    temporal_summary = save_temporal_variation_plot(
        class_names=class_names,
        mean_abs_deltas=mean_abs_deltas_arr,
        temporal_stds=temporal_stds_arr,
        output_path=output_dir / "temporal_variation.png",
    )

    summary = {
        "windows_root": str(args.windows_root),
        "checkpoint": str(args.checkpoint),
        "split_mode": args.split_mode,
        "seed": args.seed,
        "validation_sample_count": len(val_samples),
        "validation_class_counts": {
            class_name: int(sum(name == class_name for name in class_names))
            for class_name in CLASS_NAMES
        },
        "temporal_summary": temporal_summary,
        "outputs": {
            "raw_embeddings": str(output_dir / "raw_embeddings.png"),
            "cnn_embeddings": str(output_dir / "cnn_embeddings.png"),
            "class_heatmaps": str(output_dir / "class_heatmaps.png"),
            "temporal_variation": str(output_dir / "temporal_variation.png"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
