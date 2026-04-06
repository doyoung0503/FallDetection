#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES = ["big", "small"]
LABEL_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}


@dataclass
class SDPSample:
    path: str
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
        data = np.load(sample.path)
        sdp = data["sdp"].astype(np.float32)
        sdp = (sdp - self.mean) / self.std
        features = torch.from_numpy(sdp[None, :, :]).float()
        label = torch.tensor(sample.label, dtype=torch.long)
        return features, label


class SDPCNN(nn.Module):
    def __init__(
        self,
        hidden_channels_1: int = 32,
        hidden_channels_2: int = 64,
        kernel_size_1: int = 3,
        kernel_size_2: int = 3,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, hidden_channels_1, kernel_size_1, padding=kernel_size_1 // 2),
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
        self.classifier = nn.Linear(hidden_channels_2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a 2D CNN on esp32 XFall-style SDP windows."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root directory containing windows_100/train|validation/big|small.",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=100,
        help="Window length folder to read, e.g. windows_100.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
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
    parser.add_argument(
        "--class-weight-mode",
        choices=("none", "balanced"),
        default="none",
        help=(
            "Loss weighting mode. 'balanced' uses inverse-frequency weights "
            "computed from the training windows."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cpu"),
        default="auto",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--num-interop-threads", type=int, default=2)
    return parser.parse_args()


def load_samples(root: Path) -> list[SDPSample]:
    samples: list[SDPSample] = []
    for class_name in CLASS_NAMES:
        class_dir = root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for path in sorted(class_dir.glob("*.npz")):
            samples.append(SDPSample(path=str(path), label=LABEL_TO_ID[class_name]))
    return samples


def compute_class_weights(samples: list[SDPSample]) -> np.ndarray:
    counts = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    for sample in samples:
        counts[sample.label] += 1.0

    total = float(counts.sum())
    weights = total / (len(CLASS_NAMES) * np.maximum(counts, 1.0))
    return weights.astype(np.float32)


def compute_train_stats(samples: list[SDPSample]) -> tuple[np.ndarray, np.ndarray]:
    sum_map: np.ndarray | None = None
    sumsq_map: np.ndarray | None = None
    count = 0

    for sample in samples:
        sdp = np.load(sample.path)["sdp"].astype(np.float32)
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
    seed: int,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    dataset = SDPDataset(samples=samples, mean=mean, std=std)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    confusion = np.zeros((2, 2), dtype=np.int64)

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


def precision_recall_f1(confusion: np.ndarray) -> dict[str, object]:
    per_class: dict[str, dict[str, float]] = {}
    f1_values = []
    for idx, class_name in enumerate(CLASS_NAMES):
        tp = float(confusion[idx, idx])
        fp = float(confusion[:, idx].sum() - tp)
        fn = float(confusion[idx, :].sum() - tp)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        f1_values.append(f1)

    return {
        "per_class": per_class,
        "macro_f1": float(np.mean(f1_values)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cpu":
        torch.set_num_threads(args.num_threads)
        torch.set_num_interop_threads(args.num_interop_threads)

    window_root = args.dataset_root / f"windows_{args.window_length}"
    train_samples = load_samples(window_root / "train")
    val_samples = load_samples(window_root / "validation")

    mean, std = compute_train_stats(train_samples)
    device, device_name = select_device(args.device)

    train_loader = make_loader(
        train_samples,
        mean,
        std,
        args.batch_size,
        True,
        args.num_workers,
        args.seed,
    )
    val_loader = make_loader(
        val_samples,
        mean,
        std,
        args.batch_size,
        False,
        args.num_workers,
        args.seed,
    )

    model = SDPCNN(
        hidden_channels_1=args.hidden_channels_1,
        hidden_channels_2=args.hidden_channels_2,
        kernel_size_1=args.kernel_size_1,
        kernel_size_2=args.kernel_size_2,
    ).to(device)
    class_weights = None
    if args.class_weight_mode == "balanced":
        class_weights = compute_class_weights(train_samples)
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_confusion = np.zeros((2, 2), dtype=np.int64)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_samples = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            running_loss += float(loss.item()) * batch_size
            running_correct += int((logits.argmax(dim=1) == targets).sum().item())
            running_samples += batch_size

        train_loss = running_loss / running_samples
        train_acc = running_correct / running_samples
        val_loss, val_acc, confusion = evaluate(model, val_loader, criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "validation_loss": val_loss,
                "validation_accuracy": val_acc,
            }
        )

        if (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            best_confusion = confusion.copy()
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    assert best_state is not None
    torch.save(
        {
            "state_dict": best_state,
            "hidden_channels_1": args.hidden_channels_1,
            "hidden_channels_2": args.hidden_channels_2,
            "kernel_size_1": args.kernel_size_1,
            "kernel_size_2": args.kernel_size_2,
            "mean": mean,
            "std": std,
            "class_names": CLASS_NAMES,
        },
        args.output_dir / "best_model.pt",
    )

    metrics = precision_recall_f1(best_confusion)
    summary = {
        "dataset_root": str(args.dataset_root),
        "window_length": args.window_length,
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "hidden_channels_1": args.hidden_channels_1,
        "hidden_channels_2": args.hidden_channels_2,
        "kernel_size_1": args.kernel_size_1,
        "kernel_size_2": args.kernel_size_2,
        "class_weight_mode": args.class_weight_mode,
        "class_weights": class_weights.tolist() if class_weights is not None else None,
        "device": device_name,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "train_windows": len(train_samples),
        "validation_windows": len(val_samples),
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_val_acc,
        "best_validation_loss": best_val_loss,
        "best_confusion_matrix": best_confusion.tolist(),
        "metrics": metrics,
        "history": history,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(
        {
            "dataset_root": str(args.dataset_root),
            "seed": args.seed,
            "best_epoch": best_epoch,
            "best_validation_accuracy": best_val_acc,
            "best_validation_loss": best_val_loss,
            "macro_f1": metrics["macro_f1"],
        },
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
