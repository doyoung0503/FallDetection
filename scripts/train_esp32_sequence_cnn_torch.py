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
class WindowSample:
    path: str
    label: int


class SequenceVariantDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        samples: list[WindowSample],
        amplitude_mean: np.ndarray,
        amplitude_std: np.ndarray,
        use_mask: bool,
        use_delta_t: bool,
        delta_t_mean: float = 0.0,
        delta_t_std: float = 1.0,
    ) -> None:
        self.samples = samples
        self.amplitude_mean = amplitude_mean.astype(np.float32)
        self.amplitude_std = amplitude_std.astype(np.float32)
        self.use_mask = use_mask
        self.use_delta_t = use_delta_t
        self.delta_t_mean = float(delta_t_mean)
        self.delta_t_std = float(delta_t_std)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        data = np.load(sample.path)
        amplitude = data["amplitude"].astype(np.float32)
        amplitude = (amplitude - self.amplitude_mean) / self.amplitude_std

        channels = [amplitude]

        if self.use_mask:
            interp_mask = data["interp_mask"].astype(np.float32)[:, None]
            channels.append(interp_mask)

        if self.use_delta_t:
            delta_t = data["delta_t_ms"].astype(np.float32)
            delta_t = ((delta_t - self.delta_t_mean) / self.delta_t_std)[:, None]
            channels.append(delta_t)

        stacked = np.concatenate(channels, axis=1).transpose(1, 0)
        features = torch.from_numpy(stacked).float()
        label = torch.tensor(sample.label, dtype=torch.long)
        return features, label


class SequenceCNN(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels_1: int = 64,
        hidden_channels_2: int = 128,
        kernel_size_1: int = 5,
        kernel_size_2: int = 3,
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
        self.classifier = nn.Linear(hidden_channels_2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.mean(dim=-1)
        return self.classifier(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a 1D CNN on esp32 HT-LTF sequence dataset variants."
    )
    parser.add_argument(
        "--variant-root",
        type=Path,
        required=True,
        help="Variant root such as .../interp_only/windows_64.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for checkpoints and training summary.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-channels-1", type=int, default=64)
    parser.add_argument("--hidden-channels-2", type=int, default=128)
    parser.add_argument("--kernel-size-1", type=int, default=5)
    parser.add_argument("--kernel-size-2", type=int, default=3)
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


def discover_feature_mode(variant_root: Path) -> tuple[bool, bool]:
    sample_path = next(variant_root.glob("train/*/*.npz"))
    data = np.load(sample_path)
    return "interp_mask" in data, "delta_t_ms" in data


def discover_feature_dim(variant_root: Path) -> int:
    sample_path = next(variant_root.glob("train/*/*.npz"))
    data = np.load(sample_path)
    return int(data["amplitude"].shape[1])


def load_samples(split_root: Path) -> list[WindowSample]:
    samples: list[WindowSample] = []
    for class_name in CLASS_NAMES:
        class_dir = split_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for path in sorted(class_dir.glob("*.npz")):
            samples.append(WindowSample(path=str(path), label=LABEL_TO_ID[class_name]))
    return samples


def compute_train_stats(
    samples: list[WindowSample],
    feature_dim: int,
    use_delta_t: bool,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    amp_sum = np.zeros(feature_dim, dtype=np.float64)
    amp_sumsq = np.zeros(feature_dim, dtype=np.float64)
    amp_count = 0

    dt_sum = 0.0
    dt_sumsq = 0.0
    dt_count = 0

    for sample in samples:
        data = np.load(sample.path)
        amplitude = data["amplitude"].astype(np.float32)
        amp_sum += amplitude.sum(axis=0)
        amp_sumsq += np.square(amplitude).sum(axis=0)
        amp_count += amplitude.shape[0]

        if use_delta_t:
            delta_t = data["delta_t_ms"].astype(np.float32)
            dt_sum += float(delta_t.sum())
            dt_sumsq += float(np.square(delta_t).sum())
            dt_count += int(delta_t.size)

    amp_mean = amp_sum / amp_count
    amp_var = amp_sumsq / amp_count - np.square(amp_mean)
    amp_std = np.sqrt(np.maximum(amp_var, 1e-6))

    if use_delta_t and dt_count > 0:
        dt_mean = dt_sum / dt_count
        dt_var = dt_sumsq / dt_count - dt_mean * dt_mean
        dt_std = float(np.sqrt(max(dt_var, 1e-6)))
    else:
        dt_mean = 0.0
        dt_std = 1.0

    return amp_mean.astype(np.float32), amp_std.astype(np.float32), dt_mean, dt_std


def select_device(requested: str) -> tuple[torch.device, str]:
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps"), "mps"
    if requested == "cpu":
        return torch.device("cpu"), "cpu"
    if torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


def make_loader(
    samples: list[WindowSample],
    amplitude_mean: np.ndarray,
    amplitude_std: np.ndarray,
    use_mask: bool,
    use_delta_t: bool,
    delta_t_mean: float,
    delta_t_std: float,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    dataset = SequenceVariantDataset(
        samples=samples,
        amplitude_mean=amplitude_mean,
        amplitude_std=amplitude_std,
        use_mask=use_mask,
        use_delta_t=use_delta_t,
        delta_t_mean=delta_t_mean,
        delta_t_std=delta_t_std,
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
        per_class[class_name] = {"precision": precision, "recall": recall, "f1": f1}
        f1_values.append(f1)

    return {"per_class": per_class, "macro_f1": float(np.mean(f1_values))}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    use_mask, use_delta_t = discover_feature_mode(args.variant_root)
    feature_dim = discover_feature_dim(args.variant_root)

    if args.device == "cpu":
        torch.set_num_threads(args.num_threads)
        torch.set_num_interop_threads(args.num_interop_threads)

    train_samples = load_samples(args.variant_root / "train")
    val_samples = load_samples(args.variant_root / "validation")

    amplitude_mean, amplitude_std, delta_t_mean, delta_t_std = compute_train_stats(
        train_samples, feature_dim, use_delta_t
    )

    device, device_name = select_device(args.device)
    train_loader = make_loader(
        train_samples,
        amplitude_mean,
        amplitude_std,
        use_mask,
        use_delta_t,
        delta_t_mean,
        delta_t_std,
        args.batch_size,
        True,
        args.num_workers,
        args.seed,
    )
    val_loader = make_loader(
        val_samples,
        amplitude_mean,
        amplitude_std,
        use_mask,
        use_delta_t,
        delta_t_mean,
        delta_t_std,
        args.batch_size,
        False,
        args.num_workers,
        args.seed,
    )

    input_channels = feature_dim + int(use_mask) + int(use_delta_t)
    model = SequenceCNN(
        input_channels=input_channels,
        hidden_channels_1=args.hidden_channels_1,
        hidden_channels_2=args.hidden_channels_2,
        kernel_size_1=args.kernel_size_1,
        kernel_size_2=args.kernel_size_2,
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
            best_confusion = confusion.copy()
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    assert best_state is not None
    torch.save(
        {
            "state_dict": best_state,
            "input_channels": input_channels,
            "hidden_channels_1": args.hidden_channels_1,
            "hidden_channels_2": args.hidden_channels_2,
            "kernel_size_1": args.kernel_size_1,
            "kernel_size_2": args.kernel_size_2,
            "amplitude_mean": amplitude_mean,
            "amplitude_std": amplitude_std,
            "delta_t_mean": delta_t_mean,
            "delta_t_std": delta_t_std,
            "use_mask": use_mask,
            "use_delta_t": use_delta_t,
            "class_names": CLASS_NAMES,
        },
        args.output_dir / "best_model.pt",
    )

    metrics = precision_recall_f1(best_confusion)
    summary = {
        "variant_root": str(args.variant_root),
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
        "device": device_name,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "use_mask": use_mask,
        "use_delta_t": use_delta_t,
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
            "variant_root": str(args.variant_root),
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
