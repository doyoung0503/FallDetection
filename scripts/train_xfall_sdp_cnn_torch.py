#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class SDPSample:
    window_path: str
    source_csv: str
    label: int


class SDPDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        samples: list[SDPSample],
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        self.samples = samples
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        data = np.load(sample.window_path)
        sdp = data["sdp"].astype(np.float32)
        sdp = (sdp - self.mean) / self.std
        features = torch.from_numpy(sdp[None, :, :]).float()
        label = torch.tensor(sample.label, dtype=torch.long)
        return features, label


class SDPCNN(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        hidden_channels_1: int = 32,
        hidden_channels_2: int = 64,
        kernel_size_1: int = 3,
        kernel_size_2: int = 3,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                input_channels,
                hidden_channels_1,
                kernel_size_1,
                padding=kernel_size_1 // 2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                hidden_channels_1,
                hidden_channels_2,
                kernel_size_2,
                padding=kernel_size_2 // 2,
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(hidden_channels_2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PyTorch 2D CNN baseline on XFall-style SDP windows."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset/xfall_sdp_10ms_real_shift_w50_w64_l20_tol4000"),
        help="Root directory containing manifests/ and windows_<length>/ folders.",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        choices=(50, 64),
        default=50,
        help="Window length to train on.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/xfall_sdp_cnn_windows50_time"),
        help="Directory for checkpoints and training summary.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-channels-1", type=int, default=32)
    parser.add_argument("--hidden-channels-2", type=int, default=64)
    parser.add_argument("--kernel-size-1", type=int, default=3)
    parser.add_argument("--kernel-size-2", type=int, default=3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-mode",
        choices=("time_file", "random_file"),
        default="time_file",
        help="How to split source CSV files into train/validation.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cpu"),
        default="auto",
        help="Preferred compute device.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=8,
        help="Torch CPU thread count when running on CPU.",
    )
    parser.add_argument(
        "--num-interop-threads",
        type=int,
        default=2,
        help="Torch CPU interop thread count when running on CPU.",
    )
    return parser.parse_args()


def load_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def discover_class_names(manifests_root: Path) -> list[str]:
    return sorted(
        manifest_path.name[: -len("_manifest.csv")]
        for manifest_path in manifests_root.glob("*_manifest.csv")
    )


def parse_source_timestamp(source_csv: str) -> datetime:
    stem = Path(source_csv).stem
    parts = stem.split("_")
    date_part = parts[-2]
    time_part = parts[-1]
    return datetime.strptime(f"{date_part}_{time_part}", "%Y%m%d_%H%M%S")


def split_source_files(
    rows_by_class: dict[str, list[dict[str, str]]],
    split_mode: str,
    val_ratio: float,
    rng: np.random.Generator,
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, object]]:
    train_files: dict[str, set[str]] = {}
    val_files: dict[str, set[str]] = {}
    split_details: dict[str, object] = {}

    for class_name, rows in rows_by_class.items():
        unique_files = sorted({row["source_csv"] for row in rows})
        if split_mode == "time_file":
            unique_files = sorted(unique_files, key=parse_source_timestamp)
        else:
            rng.shuffle(unique_files)

        val_count = max(1, int(round(len(unique_files) * val_ratio)))
        if split_mode == "time_file":
            val_split = set(unique_files[-val_count:])
            train_split = set(unique_files[:-val_count])
        else:
            val_split = set(unique_files[:val_count])
            train_split = set(unique_files[val_count:])

        train_files[class_name] = train_split
        val_files[class_name] = val_split
        split_details[class_name] = {
            "file_count": len(unique_files),
            "validation_file_count": len(val_split),
        }

    return train_files, val_files, split_details


def build_samples(
    rows_by_class: dict[str, list[dict[str, str]]],
    selected_files_by_class: dict[str, set[str]],
    class_names: list[str],
) -> list[SDPSample]:
    samples: list[SDPSample] = []
    for label, class_name in enumerate(class_names):
        samples.extend(
            SDPSample(
                window_path=row["window_path"],
                source_csv=row["source_csv"],
                label=label,
            )
            for row in rows_by_class[class_name]
            if row["source_csv"] in selected_files_by_class[class_name]
        )
    return samples


def compute_train_stats(samples: list[SDPSample]) -> tuple[np.ndarray, np.ndarray]:
    sum_map: np.ndarray | None = None
    sumsq_map: np.ndarray | None = None
    count = 0

    for sample in samples:
        data = np.load(sample.window_path)
        sdp = data["sdp"].astype(np.float32)
        if sum_map is None:
            sum_map = np.zeros_like(sdp, dtype=np.float64)
            sumsq_map = np.zeros_like(sdp, dtype=np.float64)
        sum_map += sdp
        sumsq_map += np.square(sdp)
        count += 1

    assert sum_map is not None and sumsq_map is not None
    mean = sum_map / count
    var = sumsq_map / count - np.square(mean)
    std = np.sqrt(np.maximum(var, 1e-6))
    return mean.astype(np.float32), std.astype(np.float32)


def select_device(requested: str) -> tuple[torch.device, str]:
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available in this environment.")
        return torch.device("mps"), "mps"
    if requested == "cpu":
        return torch.device("cpu"), "cpu"
    if torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


def make_loader(
    samples: list[SDPSample],
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = SDPDataset(samples=samples, mean=mean, std=std)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    num_classes: int,
) -> tuple[float, float, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)
            preds = logits.argmax(dim=1)

            total_loss += float(loss.item()) * inputs.size(0)
            total_correct += int((preds == targets).sum().item())
            total_count += int(inputs.size(0))

            preds_np = preds.cpu().numpy()
            targets_np = targets.cpu().numpy()
            for target, pred in zip(targets_np, preds_np):
                confusion[target, pred] += 1

    return total_loss / total_count, total_correct / total_count, confusion


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    manifests_root = args.dataset_root / "manifests"
    class_names = discover_class_names(manifests_root)
    if not class_names:
        raise SystemExit(f"No *_manifest.csv files found under: {manifests_root}")

    rows_by_class = {
        class_name: [
            row
            for row in load_manifest_rows(manifests_root / f"{class_name}_manifest.csv")
            if int(row["window_length"]) == args.window_length
        ]
        for class_name in class_names
    }

    train_files, val_files, split_details = split_source_files(
        rows_by_class=rows_by_class,
        split_mode=args.split_mode,
        val_ratio=args.val_ratio,
        rng=rng,
    )
    train_samples = build_samples(rows_by_class, train_files, class_names)
    val_samples = build_samples(rows_by_class, val_files, class_names)

    mean, std = compute_train_stats(train_samples)
    device, resolved_device_name = select_device(args.device)
    if resolved_device_name == "cpu":
        torch.set_num_threads(args.num_threads)
        torch.set_num_interop_threads(args.num_interop_threads)

    train_loader = make_loader(
        train_samples, mean, std, args.batch_size, True, args.num_workers
    )
    val_loader = make_loader(
        val_samples, mean, std, args.batch_size, False, args.num_workers
    )

    model = SDPCNN(
        hidden_channels_1=args.hidden_channels_1,
        hidden_channels_2=args.hidden_channels_2,
        kernel_size_1=args.kernel_size_1,
        kernel_size_2=args.kernel_size_2,
        num_classes=len(class_names),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_val_accuracy = -1.0
    best_confusion: np.ndarray | None = None
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            total_loss += float(loss.item()) * inputs.size(0)
            total_correct += int((preds == targets).sum().item())
            total_count += int(inputs.size(0))

        train_loss = total_loss / total_count
        train_accuracy = total_correct / total_count
        val_loss, val_accuracy, confusion = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            criterion=criterion,
            num_classes=len(class_names),
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )

        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_confusion = confusion.copy()
            torch.save(model.state_dict(), args.output_dir / "best_model.pt")

    summary = {
        "dataset_root": str(args.dataset_root.resolve()),
        "window_length": args.window_length,
        "output_dir": str(args.output_dir.resolve()),
        "class_names": class_names,
        "split_mode": args.split_mode,
        "device_requested": args.device,
        "device_resolved": resolved_device_name,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "hidden_channels": [args.hidden_channels_1, args.hidden_channels_2],
        "kernel_sizes": [args.kernel_size_1, args.kernel_size_2],
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "num_threads": args.num_threads if resolved_device_name == "cpu" else None,
        "num_interop_threads": (
            args.num_interop_threads if resolved_device_name == "cpu" else None
        ),
        "samples_train": len(train_samples),
        "samples_val": len(val_samples),
        "train_source_file_counts": {
            class_name: len(train_files[class_name]) for class_name in class_names
        },
        "val_source_file_counts": {
            class_name: len(val_files[class_name]) for class_name in class_names
        },
        "split_details": split_details,
        "best_val_accuracy": best_val_accuracy,
        "validation_confusion_matrix": (
            best_confusion.tolist() if best_confusion is not None else None
        ),
        "history": history,
    }

    with (args.output_dir / "training_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
