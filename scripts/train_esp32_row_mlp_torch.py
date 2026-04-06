#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


CLASS_NAMES = ["big", "small"]
LABEL_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}


@dataclass
class DatasetBundle:
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    input_dim: int
    skipped_rows: int
    train_counts: dict[str, int]
    val_counts: dict[str, int]


class RowMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim_1: int, hidden_dim_2: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a 3-layer PyTorch MLP on esp32 raw CSI rows."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="One variant root under dataset/esp32_raw_csi_variants.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for checkpoints and training summary.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim-1", type=int, default=256)
    parser.add_argument("--hidden-dim-2", type=int, default=128)
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


def parse_row_values(row: dict[str, str]) -> list[int]:
    values = ast.literal_eval(row["data"])
    if not isinstance(values, list):
        raise ValueError("data is not a list")
    return [int(value) for value in values]


def load_split_rows(split_dir: Path) -> tuple[np.ndarray, np.ndarray, int, dict[str, int], int]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    skipped_rows = 0
    input_dim: int | None = None
    class_counts = {class_name: 0 for class_name in CLASS_NAMES}

    for class_name in CLASS_NAMES:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")

        for csv_path in sorted(class_dir.glob("*.csv")):
            with csv_path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    try:
                        values = parse_row_values(row)
                    except Exception:
                        skipped_rows += 1
                        continue

                    if input_dim is None:
                        input_dim = len(values)

                    if len(values) != input_dim or row.get("len") != str(input_dim):
                        skipped_rows += 1
                        continue

                    features.append(np.asarray(values, dtype=np.float32))
                    labels.append(LABEL_TO_ID[class_name])
                    class_counts[class_name] += 1

    if input_dim is None:
        raise RuntimeError(f"No valid rows found under {split_dir}")

    return (
        np.stack(features, axis=0),
        np.asarray(labels, dtype=np.int64),
        input_dim,
        class_counts,
        skipped_rows,
    )


def load_dataset(dataset_root: Path) -> DatasetBundle:
    train_x, train_y, train_dim, train_counts, train_skipped = load_split_rows(dataset_root / "train")
    val_x, val_y, val_dim, val_counts, val_skipped = load_split_rows(dataset_root / "validation")
    if train_dim != val_dim:
        raise RuntimeError(f"Input dimension mismatch: train={train_dim}, validation={val_dim}")

    return DatasetBundle(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        input_dim=train_dim,
        skipped_rows=train_skipped + val_skipped,
        train_counts=train_counts,
        val_counts=val_counts,
    )


def standardize(bundle: DatasetBundle) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = bundle.train_x.mean(axis=0, dtype=np.float64)
    std = bundle.train_x.std(axis=0, dtype=np.float64)
    std = np.where(std < 1e-6, 1.0, std)
    train_x = ((bundle.train_x - mean) / std).astype(np.float32)
    val_x = ((bundle.val_x - mean) / std).astype(np.float32)
    return train_x, val_x, mean.astype(np.float32), std.astype(np.float32)


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
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    dataset = TensorDataset(
        torch.from_numpy(features).float(),
        torch.from_numpy(labels).long(),
    )
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
    total_samples = 0
    total_correct = 0
    confusion = np.zeros((2, 2), dtype=np.int64)

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            batch_size = batch_y.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            preds = logits.argmax(dim=1)
            total_correct += int((preds == batch_y).sum().item())

            preds_np = preds.cpu().numpy()
            targets_np = batch_y.cpu().numpy()
            for target, pred in zip(targets_np, preds_np):
                confusion[target, pred] += 1

    return total_loss / total_samples, total_correct / total_samples, confusion


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

    bundle = load_dataset(args.dataset_root)
    train_x, val_x, mean, std = standardize(bundle)

    device, device_name = select_device(args.device)

    train_loader = make_loader(
        train_x,
        bundle.train_y,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    val_loader = make_loader(
        val_x,
        bundle.val_y,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = RowMLP(
        input_dim=bundle.input_dim,
        hidden_dim_1=args.hidden_dim_1,
        hidden_dim_2=args.hidden_dim_2,
    ).to(device)
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

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            batch_size = batch_y.shape[0]
            running_loss += float(loss.item()) * batch_size
            running_samples += batch_size
            running_correct += int((logits.argmax(dim=1) == batch_y).sum().item())

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
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            best_confusion = confusion.copy()

    assert best_state is not None
    torch.save(
        {
            "state_dict": best_state,
            "input_dim": bundle.input_dim,
            "hidden_dim_1": args.hidden_dim_1,
            "hidden_dim_2": args.hidden_dim_2,
            "mean": mean,
            "std": std,
            "class_names": CLASS_NAMES,
        },
        args.output_dir / "best_model.pt",
    )

    metrics = precision_recall_f1(best_confusion)
    summary = {
        "dataset_root": str(args.dataset_root),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "hidden_dim_1": args.hidden_dim_1,
        "hidden_dim_2": args.hidden_dim_2,
        "input_dim": bundle.input_dim,
        "device": device_name,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "skipped_rows": bundle.skipped_rows,
        "train_counts": bundle.train_counts,
        "validation_counts": bundle.val_counts,
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
