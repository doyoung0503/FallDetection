#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sam_torch import SAM

try:
    import wandb
except ImportError:
    wandb = None


AMPLITUDE_DIM = 114
INPUT_CHANNELS = 115


@dataclass
class WindowSample:
    window_path: str
    source_csv: str
    label: int


class SequenceWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        samples: list[WindowSample],
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
        amplitude = data["amplitude"].astype(np.float32)
        interp_mask = data["interp_mask"].astype(np.float32)[:, None]

        amplitude = (amplitude - self.mean) / self.std
        stacked = np.concatenate([amplitude, interp_mask], axis=1).transpose(1, 0)

        features = torch.from_numpy(stacked).float()
        label = torch.tensor(sample.label, dtype=torch.long)
        return features, label


class SequenceCNN(nn.Module):
    def __init__(
        self,
        input_channels: int = INPUT_CHANNELS,
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
        x = self.features(x)
        x = x.mean(dim=-1)
        return self.classifier(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PyTorch 1D CNN on CSI sequence windows."
    )
    parser.add_argument(
        "--windows-root",
        type=Path,
        default=Path("dataset/sequence_10ms_amp_mask/windows_64"),
        help="Root directory containing class-wise sequence windows.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/sequence_cnn_torch_windows64_time"),
        help="Directory for checkpoints and training summary.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--optimizer",
        choices=("adam", "sam_sgd"),
        default="adam",
        help="Optimizer choice.",
    )
    parser.add_argument(
        "--sam-rho",
        type=float,
        default=0.05,
        help="SAM neighborhood size when optimizer=sam_sgd.",
    )
    parser.add_argument(
        "--sgd-momentum",
        type=float,
        default=0.9,
        help="Momentum for SGD when optimizer=sam_sgd.",
    )
    parser.add_argument("--hidden-channels-1", type=int, default=64)
    parser.add_argument("--hidden-channels-2", type=int, default=128)
    parser.add_argument("--kernel-size-1", type=int, default=5)
    parser.add_argument("--kernel-size-2", type=int, default=3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-mode",
        choices=("time_file", "random_file", "time_block"),
        default="time_file",
        help="How to split source CSV files into train/validation.",
    )
    parser.add_argument(
        "--time-block-count",
        type=int,
        default=3,
        help=(
            "Number of contiguous chronological file blocks per class when "
            "using split-mode=time_block. The last block becomes validation."
        ),
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
        default=4,
        help="DataLoader workers for CPU-side parallel loading.",
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
    parser.add_argument(
        "--smoke-train-limit",
        type=int,
        default=None,
        help="Optional cap on the number of training windows for smoke tests.",
    )
    parser.add_argument(
        "--smoke-val-limit",
        type=int,
        default=None,
        help="Optional cap on the number of validation windows for smoke tests.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Optional Weights & Biases project name for online/offline logging.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Optional Weights & Biases entity/team.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional Weights & Biases run name.",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="Optional Weights & Biases run group, useful for dataset-wise comparisons.",
    )
    parser.add_argument(
        "--wandb-job-type",
        type=str,
        default="train",
        help="Optional Weights & Biases job type.",
    )
    parser.add_argument(
        "--wandb-tags",
        nargs="*",
        default=None,
        help="Optional Weights & Biases tags.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
        help="Weights & Biases logging mode.",
    )
    parser.add_argument(
        "--wandb-notes",
        type=str,
        default=None,
        help="Optional Weights & Biases notes.",
    )
    return parser.parse_args()


def load_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def discover_class_names(windows_root: Path) -> list[str]:
    manifest_names = []
    for manifest_path in sorted(windows_root.glob("*_manifest.csv")):
        if manifest_path.name.endswith("_manifest.csv"):
            manifest_names.append(manifest_path.name[: -len("_manifest.csv")])
    return manifest_names


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
    time_block_count: int,
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, object]]:
    train_files: dict[str, set[str]] = {}
    val_files: dict[str, set[str]] = {}
    split_details: dict[str, object] = {}

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
            split_details[class_name] = {
                "block_count": len(blocks),
                "block_sizes": [len(block) for block in blocks],
                "validation_block_index": len(blocks) - 1,
                "validation_block_files": blocks[-1],
            }
        else:
            val_count = max(1, int(round(len(unique_files) * val_ratio)))
            val_split = set(unique_files[-val_count:] if split_mode == "time_file" else unique_files[:val_count])
            train_split = set(unique_files[:-val_count] if split_mode == "time_file" else unique_files[val_count:])

        train_files[class_name] = train_split
        val_files[class_name] = val_split
        if split_mode != "time_block":
            split_details[class_name] = {
                "file_count": len(unique_files),
                "validation_file_count": len(val_split),
            }

    return train_files, val_files, split_details


def build_samples(
    rows_by_class: dict[str, list[dict[str, str]]],
    selected_files_by_class: dict[str, set[str]],
    limit: int | None,
    class_names: list[str],
) -> list[WindowSample]:
    samples: list[WindowSample] = []

    for label, class_name in enumerate(class_names):
        class_samples = [
            WindowSample(
                window_path=row["window_path"],
                source_csv=row["source_csv"],
                label=label,
            )
            for row in rows_by_class[class_name]
            if row["source_csv"] in selected_files_by_class[class_name]
        ]
        if limit is not None:
            class_samples = class_samples[:limit]
        samples.extend(class_samples)

    return samples


def compute_train_stats(samples: list[WindowSample]) -> tuple[np.ndarray, np.ndarray]:
    sum_channels = np.zeros(AMPLITUDE_DIM, dtype=np.float64)
    sumsq_channels = np.zeros(AMPLITUDE_DIM, dtype=np.float64)
    count = 0

    for sample in samples:
        data = np.load(sample.window_path)
        amplitude = data["amplitude"].astype(np.float32)
        sum_channels += amplitude.sum(axis=0)
        sumsq_channels += np.square(amplitude).sum(axis=0)
        count += amplitude.shape[0]

    mean = sum_channels / count
    var = sumsq_channels / count - np.square(mean)
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
    samples: list[WindowSample],
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = SequenceWindowDataset(samples=samples, mean=mean, std=std)
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


def build_wandb_config(
    args: argparse.Namespace,
    *,
    class_names: list[str],
    resolved_device_name: str,
    train_samples: list[WindowSample],
    val_samples: list[WindowSample],
    train_files: dict[str, set[str]],
    val_files: dict[str, set[str]],
) -> dict[str, Any]:
    return {
        "windows_root": str(args.windows_root),
        "output_dir": str(args.output_dir),
        "class_names": class_names,
        "split_mode": args.split_mode,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "device_requested": args.device,
        "device_resolved": resolved_device_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
        "sam_rho": args.sam_rho if args.optimizer == "sam_sgd" else None,
        "sgd_momentum": args.sgd_momentum if args.optimizer == "sam_sgd" else None,
        "hidden_channels_1": args.hidden_channels_1,
        "hidden_channels_2": args.hidden_channels_2,
        "kernel_size_1": args.kernel_size_1,
        "kernel_size_2": args.kernel_size_2,
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
    }


def maybe_init_wandb(
    args: argparse.Namespace,
    *,
    config: dict[str, Any],
) -> Any | None:
    if args.wandb_project is None or args.wandb_mode == "disabled":
        return None
    if wandb is None:
        raise RuntimeError(
            "wandb logging was requested but the wandb package is not installed."
        )

    wandb_root = args.output_dir / "wandb"
    wandb_cache = wandb_root / ".cache"
    wandb_config = wandb_root / ".config"
    wandb_data = wandb_root / ".data"
    for path in (wandb_root, wandb_cache, wandb_config, wandb_data):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["WANDB_DIR"] = str(wandb_root)
    os.environ["WANDB_CACHE_DIR"] = str(wandb_cache)
    os.environ["WANDB_CONFIG_DIR"] = str(wandb_config)
    os.environ["WANDB_DATA_DIR"] = str(wandb_data)

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or None,
        group=args.wandb_group or None,
        job_type=args.wandb_job_type,
        tags=args.wandb_tags or None,
        notes=args.wandb_notes,
        mode=args.wandb_mode,
        config=config,
        dir=str(wandb_root),
    )


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    class_names = discover_class_names(args.windows_root)
    if not class_names:
        raise SystemExit(f"No *_manifest.csv files found under: {args.windows_root}")

    rows_by_class = {
        class_name: load_manifest_rows(args.windows_root / f"{class_name}_manifest.csv")
        for class_name in class_names
    }
    train_files, val_files, split_details = split_source_files(
        rows_by_class=rows_by_class,
        split_mode=args.split_mode,
        val_ratio=args.val_ratio,
        rng=rng,
        time_block_count=args.time_block_count,
    )

    train_samples = build_samples(
        rows_by_class, train_files, args.smoke_train_limit, class_names
    )
    val_samples = build_samples(
        rows_by_class, val_files, args.smoke_val_limit, class_names
    )

    mean, std = compute_train_stats(train_samples)

    device, resolved_device_name = select_device(args.device)
    if resolved_device_name == "cpu":
        torch.set_num_threads(args.num_threads)
        torch.set_num_interop_threads(args.num_interop_threads)

    wandb_run = maybe_init_wandb(
        args,
        config=build_wandb_config(
            args,
            class_names=class_names,
            resolved_device_name=resolved_device_name,
            train_samples=train_samples,
            val_samples=val_samples,
            train_files=train_files,
            val_files=val_files,
        ),
    )

    train_loader = make_loader(
        samples=train_samples,
        mean=mean,
        std=std,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = make_loader(
        samples=val_samples,
        mean=mean,
        std=std,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = SequenceCNN(
        hidden_channels_1=args.hidden_channels_1,
        hidden_channels_2=args.hidden_channels_2,
        kernel_size_1=args.kernel_size_1,
        kernel_size_2=args.kernel_size_2,
        num_classes=len(class_names),
    ).to(device)
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = SAM(
            model.parameters(),
            base_optimizer_cls=torch.optim.SGD,
            lr=args.learning_rate,
            momentum=args.sgd_momentum,
            weight_decay=args.weight_decay,
            rho=args.sam_rho,
        )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state_dict = None
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            if args.optimizer == "sam_sgd":
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.first_step(zero_grad=True)

                second_logits = model(inputs)
                second_loss = criterion(second_logits, targets)
                second_loss.backward()
                optimizer.second_step(zero_grad=True)
                preds = second_logits.argmax(dim=1)
            else:
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                preds = logits.argmax(dim=1)

            train_loss_sum += float(loss.item()) * inputs.size(0)
            train_correct += int((preds == targets).sum().item())
            train_count += int(inputs.size(0))

        train_loss = train_loss_sum / train_count
        train_acc = train_correct / train_count
        val_loss, val_acc, _ = evaluate(
            model, val_loader, device, criterion, len(class_names)
        )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc),
        }
        history.append(epoch_metrics)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if wandb_run is not None:
            wandb.log(
                {
                    **epoch_metrics,
                    "best_val_accuracy_so_far": float(max(item["val_accuracy"] for item in history)),
                },
                step=epoch,
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    _, best_val_acc_eval, confusion = evaluate(
        model, val_loader, device, criterion, len(class_names)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "mean": mean,
        "std": std,
        "class_names": class_names,
        "resolved_device": resolved_device_name,
    }
    torch.save(checkpoint, args.output_dir / "best_model.pt")

    metadata = {
        "windows_root": str(args.windows_root),
        "output_dir": str(args.output_dir),
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
        "optimizer": args.optimizer,
        "sam_rho": args.sam_rho if args.optimizer == "sam_sgd" else None,
        "sgd_momentum": args.sgd_momentum if args.optimizer == "sam_sgd" else None,
        "hidden_channels": [args.hidden_channels_1, args.hidden_channels_2],
        "kernel_sizes": [args.kernel_size_1, args.kernel_size_2],
        "val_ratio": args.val_ratio,
        "time_block_count": args.time_block_count if args.split_mode == "time_block" else None,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "num_threads": args.num_threads if resolved_device_name == "cpu" else None,
        "num_interop_threads": (
            args.num_interop_threads if resolved_device_name == "cpu" else None
        ),
        "smoke_train_limit": args.smoke_train_limit,
        "smoke_val_limit": args.smoke_val_limit,
        "samples_train": len(train_samples),
        "samples_val": len(val_samples),
        "train_source_file_counts": {
            class_name: len(train_files[class_name]) for class_name in class_names
        },
        "val_source_file_counts": {
            class_name: len(val_files[class_name]) for class_name in class_names
        },
        "split_details": split_details,
        "best_val_accuracy": float(best_val_acc_eval),
        "validation_confusion_matrix": confusion.tolist(),
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_run_name": args.wandb_run_name,
        "wandb_group": args.wandb_group,
        "wandb_mode": args.wandb_mode,
        "wandb_url": (
            wandb_run.url
            if wandb_run is not None and getattr(wandb_run, "url", None)
            else None
        ),
        "wandb_run_id": (
            wandb_run.id
            if wandb_run is not None and getattr(wandb_run, "id", None)
            else None
        ),
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(metadata | {"history": history}, indent=2)
    )

    if wandb_run is not None:
        confusion_data = []
        for true_index, true_name in enumerate(class_names):
            for pred_index, pred_name in enumerate(class_names):
                confusion_data.append(
                    [true_name, pred_name, int(confusion[true_index, pred_index])]
                )
        wandb.log(
            {
                "best_val_accuracy": float(best_val_acc_eval),
                "confusion_matrix_table": wandb.Table(
                    columns=["true_label", "pred_label", "count"],
                    data=confusion_data,
                ),
            }
        )
        wandb.summary["best_val_accuracy"] = float(best_val_acc_eval)
        wandb.summary["validation_confusion_matrix"] = confusion.tolist()
        wandb.summary["samples_train"] = len(train_samples)
        wandb.summary["samples_val"] = len(val_samples)
        wandb.finish()

    print(f"resolved_device={resolved_device_name}")
    print(f"saved_checkpoint={args.output_dir / 'best_model.pt'}")
    print(f"saved_summary={args.output_dir / 'training_summary.json'}")
    print("validation_confusion_matrix=")
    print(confusion)


if __name__ == "__main__":
    main()
